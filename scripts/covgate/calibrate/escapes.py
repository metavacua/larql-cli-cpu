"""Escaped defects from git history: fix commits blamed back to the
changes that introduced the lines they repaired.

Known error sources, reported rather than hidden:
- measurement: a fix not labelled as one is not counted (undercount);
- theoretical: blame names the LAST change to a line, not necessarily
  the change that made it wrong (misattribution, biased toward recent
  commits);
- selection: only defects that were found and fixed are visible
  (survivorship; latent defects are absent by construction).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

FIX_SUBJECT = re.compile(r"^(fix|revert)(\(.+?\))?!?:|^revert\b", re.IGNORECASE)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Escape:
    fix: str
    introduced: str
    latency_hours: float


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def commits(root: Path, ref: str) -> list[tuple[str, int, str]]:
    """(sha, commit time, subject) for every non-merge commit reachable
    from `ref`. Not first-parent: on a merge-based history that would
    skip every change and fix that arrived through a merge."""
    out = _git(root, "log", "--no-merges", "--format=%H%x09%ct%x09%s", ref)
    rows = []
    for line in out.splitlines():
        sha, ts, subject = line.split("\t", 2)
        rows.append((sha, int(ts), subject))
    return rows


def removed_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Old-side line ranges a `--unified=0` diff removes or rewrites."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            source = line[4:].strip()
            path = None if source == "/dev/null" else source.removeprefix("a/")
            continue
        match = _HUNK.match(line)
        if match and path is not None:
            start, count = int(match.group(1)), int(match.group(2) or "1")
            if count > 0:
                ranges.setdefault(path, []).append((start, start + count - 1))
    return ranges


def blamed_commits(blame_porcelain: str) -> set[str]:
    return {
        line.split()[0]
        for line in blame_porcelain.splitlines()
        if re.match(r"^[0-9a-f]{40} \d+ \d+", line)
    }


def escapes(root: Path, ref: str = "main", only: str = "") -> tuple[list[Escape], int]:
    """Every (fix, introducing change) pair on `ref`, and the number of
    non-fix changes examined (the denominator for the defect rate).
    `only` restricts blame to paths under it (e.g. `crates/larql-cli`)."""
    history = commits(root, ref)
    times = {sha: ts for sha, ts, _ in history}
    found: list[Escape] = []
    changes = sum(1 for _, _, s in history if not FIX_SUBJECT.search(s))
    for sha, ts, subject in history:
        if not FIX_SUBJECT.search(subject):
            continue
        diff = _git(root, "diff", "--unified=0", "--no-color", "-M", f"{sha}^", sha, "--", only or ".")
        culprits: set[str] = set()
        for path, spans in removed_ranges(diff).items():
            args = ["blame", "--porcelain", "-M", "-C"]
            for start, end in spans:
                args += ["-L", f"{start},{end}"]
            try:
                culprits |= blamed_commits(_git(root, *args, f"{sha}^", "--", path))
            except subprocess.CalledProcessError:
                continue  # path absent at the parent (renamed beyond -M): unattributable
        for culprit in culprits:
            if culprit in times and culprit != sha:
                found.append(Escape(sha, culprit, (ts - times[culprit]) / SECONDS_PER_HOUR))
    return found, changes
