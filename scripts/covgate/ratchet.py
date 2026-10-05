"""No denominator games: what a change may not loosen, compared with the
base branch.

- no file of the crate may leave the measured scope (judged on the
  crate's actual files, so swapping a narrow glob for a wider one passes)
- per-file debt entries may only rise or disappear; no new ones
- no gate threshold may be lowered (the target spread may not be raised)
- the count of coverage-suppressing source markers may not increase
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from pathlib import Path

from .checks import Finding

# Raising any of these loosens the gate.
_LOOSER_IF_LOWER = {
    "default_line_min_percent": None,
    "total_line_min_percent": None,
    "function_min_percent": "gate",
    "trait_line_min_percent": "gate",
    "diff_line_min_percent": "gate",
    "mutation_min_kill_percent": "gate",
}
_LOOSER_IF_HIGHER: dict[str, str | None] = {}

# Source markers that take code out of coverage measurement.
SUPPRESSION = re.compile(r"coverage\s*\(\s*off|cfg\s*\(\s*not\s*\(\s*coverage|cfg_attr\s*\(\s*coverage")


def _value(policy: dict, key: str, section: str | None) -> float | None:
    holder = policy.get(section, {}) if section else policy
    value = holder.get(key) if isinstance(holder, dict) else None
    return None if value is None else float(value)


def _in_scope(policy: dict, path: str) -> bool:
    include = policy.get("include_globs") or []
    if include and not any(fnmatch.fnmatch(path, g) for g in include):
        return False
    return not any(fnmatch.fnmatch(path, g) for g in policy.get("exclude_globs") or [])


def compare_policies(base: dict, head: dict, files: list[str]) -> list[Finding]:
    out: list[Finding] = []

    def flag(subject: str, message: str) -> None:
        out.append(Finding("ratchet", "-", subject, message))

    if base:
        for path in sorted(files):
            if _in_scope(base, path) and not _in_scope(head, path):
                flag(path, "left the measured scope")
    base_debt = base.get("per_file_line_min_percent", {})
    for path, value in sorted(head.get("per_file_line_min_percent", {}).items()):
        if path not in base_debt:
            flag(path, "new per-file debt entry")
        elif float(value) < float(base_debt[path]):
            flag(path, f"debt lowered {base_debt[path]} -> {value}")
    for key, section in _LOOSER_IF_LOWER.items():
        old, new = _value(base, key, section), _value(head, key, section)
        if old is not None and (new is None or new < old):
            flag(key, f"lowered or removed: {old} -> {new}")
    for key, section in _LOOSER_IF_HIGHER.items():
        old, new = _value(base, key, section), _value(head, key, section)
        if old is not None and (new is None or new > old):
            flag(key, f"raised or removed: {old} -> {new}")
    return out


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)


def base_policy(base: str, policy_path: str, repo_root: Path) -> dict:
    shown = _git(repo_root, "show", f"{base}:{policy_path}")
    return json.loads(shown.stdout) if shown.returncode == 0 else {}


def count_markers(text_by_path: dict[str, str]) -> int:
    return sum(len(SUPPRESSION.findall(text)) for text in text_by_path.values())


def suppression(base: str, crate_dir: str, repo_root: Path) -> list[Finding]:
    def at(ref: str | None) -> int:
        args = ["grep", "-c", "-E", SUPPRESSION.pattern]
        args += [ref] if ref else []
        listed = _git(repo_root, *args, "--", f"{crate_dir}/*.rs")
        # `git grep -c` prints path:count (ref:path:count for a ref); exit 1 = none.
        return sum(int(line.rsplit(":", 1)[1]) for line in listed.stdout.splitlines() if line)

    before, after = at(base), at(None)
    if after > before:
        return [Finding("ratchet", "-", crate_dir, f"coverage-suppression markers {before} -> {after}")]
    return []


def crate_files(crate_dir: str, repo_root: Path) -> list[str]:
    listed = _git(repo_root, "ls-files", "--", crate_dir)
    return listed.stdout.split()
