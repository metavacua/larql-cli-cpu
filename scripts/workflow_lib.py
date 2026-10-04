"""Shared helpers for the workflow tests (test_workflow_*.py, test_crate_gate_chain.py).

Loads .github/workflows/*.yml and models GitHub's trigger rules (branch globs with `!`,
branches-ignore, paths, paths-ignore, pull_request types) well enough to ask, from the
workflow files alone, which workflows an event wakes. Nothing here mutates what load()
returns: the parsed workflows are cached and shared.
"""

import functools
import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Filter-pattern characters GitHub gives a meaning that glob() does not model. A filter that
# uses one must fail loudly here rather than be silently mis-modelled.
UNMODELLED = "?+[]\\"


@functools.cache
def load():
    out = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        out[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out


@functools.cache
def glob(pattern):
    """GitHub filter pattern -> regex: `**` crosses `/`, `*` does not."""
    bad = sorted(set(pattern) & set(UNMODELLED))
    if bad:
        raise ValueError(f"glob() does not implement {bad} in filter pattern {pattern!r}")
    rx, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            rx, i = rx + ".*", i + 2
        elif pattern[i] == "*":
            rx, i = rx + "[^/]*", i + 1
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


def needs(job):
    n = job.get("needs", [])
    return {n} if isinstance(n, str) else set(n)


def steps(workflow):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])
