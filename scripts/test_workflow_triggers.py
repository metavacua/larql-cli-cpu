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

import unittest

from workflow_lib import glob, load, triggers, woken

GATE_BASE = "experiment/crate-gate"
CRATE_BRANCH = "experiment/crate/larql-core"
CRATE_FILE = "crates/larql-core/src/lib.rs"
# One gate workflow: fmt once per crate, then every target's chain in the same run.
GATE = "crate-gate"


class CrateBranchTriggers(unittest.TestCase):
    def test_pushing_a_crate_branch_wakes_nothing(self):
        for changed in ([], [CRATE_FILE], ["Cargo.lock", ".github/workflows/crate-gate.yml"]):
            with self.subTest(changed=changed):
                self.assertEqual(woken("push", CRATE_BRANCH, changed), set())

    def test_crate_pr_into_the_gate_base_wakes_only_the_gates(self):
        for changed in ([], [CRATE_FILE], ["Cargo.lock", ".github/workflows/crate-gate.yml"]):
            with self.subTest(changed=changed):
                self.assertEqual(woken("pull_request", GATE_BASE, changed), {GATE})

    def test_ordinary_pr_into_main_still_wakes_the_usual_workflows(self):
        got = woken("pull_request", "main", [CRATE_FILE, "Cargo.lock"])
        for expected in ("commit-messages", "quality", "larql-core", GATE):
            self.assertIn(expected, got)

    def test_the_gates_are_pull_request_only(self):
        # A push trigger as well would start the heavy strict matrix twice per commit.
        self.assertFalse(triggers(load()[GATE], "push", CRATE_BRANCH, [CRATE_FILE]))

    def test_a_filter_character_the_model_does_not_implement_fails_loudly(self):
        for pattern in ("a?b", "a+b", "[ab]", "a\\b"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    glob(pattern)


if __name__ == "__main__":
    unittest.main()
