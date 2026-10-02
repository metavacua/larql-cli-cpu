#!/usr/bin/env python3
"""Prove CI actually exercises what the workspace claims to build.

Every defect in the 2026-09-30 batch of closed issues (#3 to #7) sat undetected
until the first control able to see it existed, not until the bug was written:

* #5  quality.yml filtered on `branches: [main]` while the default branch was
      another name, so `cargo-audit` never ran.
* #6  larql-router's tests ran on Linux only (through the temporary
      extraction-check) and never on Windows, where the test raced.
* #13 the riscv64 gate runs only by dispatch.

This script checks the *existence* of controls, statically and, optionally,
against the Actions API:

1. Every workspace member has a full (unfiltered) `cargo test` per-change run
   (push or pull_request) on each required platform, or a written exemption.
2. No workflow is dispatch-only unless a written exemption says why.
3. Workflows that filter on `branches:` must include the default branch.
4. A job in a pull-request workflow must be able to run on a pull request.
   A control whose first run is after the change merges cannot guard it, and
   cannot be checked before merge. Exemptions name `workflow::job`.
5. With --runs: scheduled workflows are enabled and have actually fired
   recently. GitHub silently disables schedules after 60 days of repository
   inactivity, and a cron on a non-default branch never fires.

Exemptions live in scripts/ci_liveness_policy.json. An exemption that is no
longer needed is itself an error, so the list can only shrink.
"""

import argparse
import fnmatch
import glob
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY = "scripts/ci_liveness_policy.json"
PER_CHANGE = {"push", "pull_request"}
LIVE_TRIGGERS = PER_CHANGE | {"schedule", "workflow_call", "workflow_run"}
REQUIRED_PLATFORMS = ("linux", "windows")
PLATFORM_PREFIX = {"ubuntu": "linux", "windows": "windows", "macos": "macos"}
# A test command restricted to a subset proves nothing about the crate as a whole.
PARTIAL_FLAGS = {"--lib", "--test", "--bins", "--bin", "--doc", "--no-run", "--benches"}
MAKE_TARGET = re.compile(
    r"make\s+(larql-[a-z0-9-]+)-(?:ci|test|test-fast|coverage-summary|coverage)\b"
)
EVENT_ATOM = re.compile(r"github\.event_name\s*(==|!=)\s*'([a-z_]+)'")
DEFAULT_MAX_AGE_DAYS = 45  # the slowest cron here is monthly


def members(root: Path) -> set[str]:
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    names = set()
    for pattern in manifest["workspace"]["members"]:
        for directory in glob.glob(str(root / pattern)) or [str(root / pattern)]:
            cargo = Path(directory) / "Cargo.toml"
            if cargo.exists():
                names.add(tomllib.loads(cargo.read_text())["package"]["name"])
    return names


def strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def on_block(text: str) -> str:
    match = re.search(r"^on:[ \t]*(\S.*)?\n((?:(?:[ \t].*)?\n)*)", strip_comments(text) + "\n", re.M)
    return "" if not match else (match.group(1) or "") + "\n" + match.group(2)


def triggers(text: str) -> set[str]:
    block = on_block(text)
    found = set(re.findall(r"^  ([a-z_]+):", block, re.M))
    first = block.split("\n", 1)[0]  # `on: [push, pull_request]` or `on: push`
    return found | set(re.findall(r"[a-z_]+", first))


def branch_filters(text: str) -> list[list[str]]:
    block = on_block(text)
    lists = [re.findall(r"[\w./*-]+", m) for m in re.findall(r"^\s+branches:\s*\[([^\]]*)\]", block, re.M)]
    for m in re.finditer(r"^\s+branches:\s*\n((?:\s+- .*\n)+)", block, re.M):
        lists.append(re.findall(r"-\s+['\"]?([\w./*-]+)", m.group(1)))
    return lists


def unwrap_quotes(value: str) -> str:
    """Drop one pair of quotes that wrap the whole value, never one that ends an inner string."""
    value = value.strip()
    return value[1:-1] if len(value) > 1 and value[0] == value[-1] and value[0] in "'\"" else value


def job_conditions(text: str) -> dict[str, str | None]:
    """Each job in the workflow with its job-level `if`, or None."""
    match = re.search(r"^jobs:[ \t]*\n((?:(?:[ \t].*)?\n)*)", strip_comments(text) + "\n", re.M)
    found: dict[str, str | None] = {}
    current = None
    for line in (match.group(1) if match else "").splitlines():
        head = re.match(r"^  ([\w-]+):\s*$", line)
        if head:
            current = head.group(1)
            found[current] = None
        elif current and (cond := re.match(r"^    if:\s*(.+?)\s*$", line)):
            found[current] = unwrap_quotes(re.sub(r"\s+#.*$", "", cond.group(1)))
    return found


def can_run_on(condition: str, event: str) -> bool:
    """Whether a job-level `if` can be true for `event`.

    Only `github.event_name` comparisons are evaluated. Any other term is assumed
    true, so a condition this does not understand can never hide a job.
    """
    expr = re.sub(r"^\$\{\{\s*|\s*\}\}$", "", condition.strip())

    def atom(text: str) -> bool:
        match = EVENT_ATOM.fullmatch(text.strip())
        return True if not match else (match.group(2) == event) == (match.group(1) == "==")

    return any(all(atom(a) for a in clause.split("&&")) for clause in expr.split("||"))


def platforms(text: str) -> set[str]:
    body = strip_comments(text)
    return {PLATFORM_PREFIX[p] for p in re.findall(r"\b(ubuntu|windows|macos)-[\w.]+", body)}


def test_commands(text: str) -> list[dict]:
    """Each `cargo test|llvm-cov` or `make <crate>-ci` command, with what it names."""
    logical = re.sub(r"\\\n\s*", " ", strip_comments(text))
    found = []
    for line in logical.splitlines():
        for make in MAKE_TARGET.finditer(line):
            found.append({"packages": {make.group(1)}, "workspace": False, "partial": False})
        if not re.search(r"cargo\s+(test|llvm-cov|nextest)", line):
            continue
        tokens = line.split()
        packages = set()
        for i, token in enumerate(tokens):
            if token in ("-p", "--package") and i + 1 < len(tokens):
                packages.add(tokens[i + 1])
            elif token.startswith("--package="):
                packages.add(token.split("=", 1)[1])
        found.append(
            {
                "packages": {p for p in packages if "${{" not in p},
                "workspace": "--workspace" in tokens or "--all" in tokens,
                "partial": bool(PARTIAL_FLAGS & set(tokens)),
            }
        )
    return found


def load_workflows(root: Path) -> dict[str, str]:
    return {Path(p).name: Path(p).read_text() for p in sorted(glob.glob(str(root / ".github/workflows/*.yml")))}


def coverage_matrix(workflows: dict[str, str], all_members: set[str]) -> dict[str, set[str]]:
    covered: dict[str, set[str]] = {name: set() for name in all_members}
    for text in workflows.values():
        if not triggers(text) & PER_CHANGE:
            continue
        where = platforms(text)
        for cmd in test_commands(text):
            if cmd["partial"]:
                continue
            for name in all_members if cmd["workspace"] else cmd["packages"] & all_members:
                covered[name] |= where
    return covered


def static_errors(root: Path, default_branch: str | None) -> list[str]:
    policy = json.loads((root / POLICY).read_text())
    exempt_members = policy.get("untested_members", {})
    exempt_dispatch = policy.get("dispatch_only_workflows", {})
    all_members = members(root)
    workflows = load_workflows(root)
    errors = []

    covered = coverage_matrix(workflows, all_members)
    for name in sorted(all_members):
        missing = [p for p in REQUIRED_PLATFORMS if p not in covered[name]]
        if name in exempt_members:
            if not missing:
                errors.append(f"{POLICY}: exemption for {name} is stale; it is now tested everywhere required")
        elif missing:
            errors.append(f"{name}: no full per-change `cargo test` on {', '.join(missing)}")
    for name in sorted(set(exempt_members) - all_members):
        errors.append(f"{POLICY}: exemption names {name}, which is not a workspace member")

    dispatch_only = {f for f, t in workflows.items() if not triggers(t) & LIVE_TRIGGERS}
    for name in sorted(dispatch_only - set(exempt_dispatch)):
        errors.append(f"{name}: runs only by manual dispatch; wire a trigger or add a reasoned exemption")
    for name in sorted(set(exempt_dispatch) - dispatch_only):
        errors.append(f"{POLICY}: dispatch-only exemption for {name} is stale or names no workflow")

    exempt_pr_skip = policy.get("pr_skipped_jobs", {})
    pr_skipped = {
        f"{name}::{job}"
        for name, text in workflows.items()
        if "pull_request" in triggers(text)
        for job, cond in job_conditions(text).items()
        if cond and not can_run_on(cond, "pull_request")
    }
    for key in sorted(pr_skipped - set(exempt_pr_skip)):
        errors.append(
            f"{key}: job cannot run on pull_request, so its first run is after the change merges; "
            "remove the condition or exempt it with a reason"
        )
    for key in sorted(set(exempt_pr_skip) - pr_skipped):
        errors.append(f"{POLICY}: pull_request exemption for {key} is stale or names no skipped job")

    if default_branch:
        for name, text in workflows.items():
            filters = branch_filters(text)
            if filters and not any(
                fnmatch.fnmatch(default_branch, glob_) for branches in filters for glob_ in branches
            ):
                errors.append(
                    f"{name}: branch filters {filters} never match the default branch '{default_branch}'"
                )
    return errors


def run_errors(root: Path, fetch, now: datetime, max_age_days: int) -> list[str]:
    policy = json.loads((root / POLICY).read_text())
    skip = set(policy.get("skip_run_check", {}))
    errors = []
    for name, text in load_workflows(root).items():
        if "schedule" not in triggers(text) or name in skip:
            continue
        info = fetch(name)
        if info["state"] != "active":
            errors.append(f"{name}: workflow state is '{info['state']}', its schedule will not fire")
            continue
        limit = now - timedelta(days=max_age_days)
        if parse_time(info["created_at"]) > limit:
            continue  # too young to have been due yet
        last = info["last_schedule_run"]
        if last is None or parse_time(last) < limit:
            errors.append(f"{name}: no scheduled run in the last {max_age_days} days (last: {last})")
    return errors


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def gh_fetcher(repo: str):
    def api(path):
        out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)

    def fetch(filename):
        workflow = api(f"repos/{repo}/actions/workflows/{filename}")
        runs = api(f"repos/{repo}/actions/workflows/{filename}/runs?event=schedule&per_page=1")["workflow_runs"]
        return {
            "state": workflow["state"],
            "created_at": workflow["created_at"],
            "last_schedule_run": runs[0]["created_at"] if runs else None,
        }

    return fetch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--default-branch", help="fail if a workflow's branch filters exclude it")
    parser.add_argument("--runs", metavar="OWNER/REPO", help="also check scheduled runs via the Actions API")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = parser.parse_args()

    errors = static_errors(args.root, args.default_branch)
    if args.runs:
        errors += run_errors(args.root, gh_fetcher(args.runs), datetime.now(timezone.utc), args.max_age_days)
    for error in errors:
        print(f"::error::{error}")
    print(f"ci-liveness: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
