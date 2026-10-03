#!/usr/bin/env python3
# /// script
# dependencies = ["pyyaml"]
# ///
"""Which workflows does a crate-gate branch wake up?

The per-crate gate (crate-gate.yml) runs on draft PRs from experiment/crate/<package>.
Fifteen of those PRs must not also fire every shared workflow (quality, portability,
shannon-verify, mutants, riscv-reduction, ...) fifteen times: the queue backs up, and the
per-PR concurrency groups cancel each other's first round on every rebase. GitHub filters
`pull_request` on the BASE branch and offers no head-branch filter, so crate PRs are based
on experiment/crate-gate, and the push triggers on experiment/** exclude experiment/crate/**.

This models GitHub's trigger rules (branch globs with `!`, branches-ignore, paths,
paths-ignore, pull_request types) well enough to assert that, from the workflow files alone.

Run: uv run scripts/test_workflow_triggers.py      (or: python3 -m unittest, with pyyaml)
"""

import re
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
GATE_BASE = "experiment/crate-gate"
CRATE_BRANCH = "experiment/crate/larql-core"
CRATE_FILE = "crates/larql-core/src/lib.rs"


def load():
    out = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        out[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out


def glob(pattern):
    """GitHub filter pattern -> regex: `**` crosses `/`, `*` and `?` do not."""
    rx, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            rx, i = rx + ".*", i + 2
        elif pattern[i] == "*":
            rx, i = rx + "[^/]*", i + 1
        elif pattern[i] == "?":
            rx, i = rx + "[^/]", i + 1
        else:
            rx, i = rx + re.escape(pattern[i]), i + 1
    return re.compile(rx + r"\Z")


def selected(patterns, value):
    """Ordered include/`!`exclude evaluation, as GitHub documents it."""
    hit = False
    for p in patterns:
        if p.startswith("!"):
            if glob(p[1:]).match(value):
                hit = False
        elif glob(p).match(value):
            hit = True
    return hit


def triggers(workflow, event, ref, changed):
    """Does `event` on branch `ref` (the base branch, for pull_request) start this workflow?"""
    on = workflow.get(True, workflow.get("on"))
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = {e: None for e in on}
    if event not in on:
        return False
    cfg = on[event] or {}
    if event == "pull_request" and "types" in cfg and "opened" not in cfg["types"]:
        return False
    if "branches" in cfg and not selected(cfg["branches"], ref):
        return False
    if "branches" not in cfg and "tags" in cfg and "branches-ignore" not in cfg:
        return False
    if any(glob(p).match(ref) for p in cfg.get("branches-ignore", [])):
        return False
    if changed is None:  # ignore path filters: "could this ever fire on that branch?"
        return True
    if "paths" in cfg:
        return any(selected(cfg["paths"], f) for f in changed)
    if "paths-ignore" in cfg:
        return any(not any(glob(p).match(f) for p in cfg["paths-ignore"]) for f in changed)
    return True


def woken(event, ref, changed):
    return {n for n, w in load().items() if triggers(w, event, ref, changed)}


class CrateBranchTriggers(unittest.TestCase):
    def test_pushing_a_crate_branch_wakes_nothing(self):
        for changed in ([], [CRATE_FILE], ["Cargo.lock", ".github/workflows/crate-gate.yml"]):
            with self.subTest(changed=changed):
                self.assertEqual(woken("push", CRATE_BRANCH, changed), set())

    def test_crate_pr_into_the_gate_base_wakes_only_the_gate(self):
        for changed in ([], [CRATE_FILE], ["Cargo.lock", ".github/workflows/crate-gate.yml"]):
            with self.subTest(changed=changed):
                self.assertEqual(woken("pull_request", GATE_BASE, changed), {"crate-gate"})

    def test_ordinary_pr_into_main_still_wakes_the_usual_workflows(self):
        got = woken("pull_request", "main", [CRATE_FILE, "Cargo.lock"])
        for expected in ("commit-messages", "quality", "larql-core", "crate-gate"):
            self.assertIn(expected, got)

    def test_the_gate_itself_is_pull_request_only(self):
        # A push trigger as well would start the heavy strict matrix twice per commit.
        self.assertFalse(triggers(load()["crate-gate"], "push", CRATE_BRANCH, [CRATE_FILE]))


if __name__ == "__main__":
    unittest.main()
