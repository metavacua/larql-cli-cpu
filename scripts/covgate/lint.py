"""The test-must-assert lint as a ratchet against the base branch.

The rule (lint/ast-grep/rules/test-must-assert.yml) flags tests that check
nothing. Existing violations are a reviewed backlog, and some are false
positives (a success-path test paired with a negative one), so the gate
fails on NEW violations only, keyed by (file, test name) so one bad test
cannot be traded for another.
"""

from __future__ import annotations

import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from .checks import Finding
from .paths import repo_relative

RULE = Path("lint/ast-grep/rules/test-must-assert.yml")
_FN_NAME = re.compile(r"\bfn\s+(\w+)")
# ast-grep exits 1 when error-severity matches exist; that is a result.
_SCAN_OK = (0, 1)


def parse(stream: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for line in stream.splitlines():
        if not line.strip():
            continue
        match = json.loads(line)
        path = repo_relative(match["file"])
        name = _FN_NAME.search(match.get("lines", ""))
        if path is not None and name is not None:
            found.add((path, name.group(1)))
    return found


def scan(root: Path, crate_dir: str, rule: Path) -> set[tuple[str, str]]:
    result = subprocess.run(
        ["ast-grep", "scan", "--rule", str(rule), crate_dir, "--json=stream"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode not in _SCAN_OK:
        raise RuntimeError(f"ast-grep failed ({result.returncode}): {result.stderr.strip()}")
    return parse(result.stdout)


def at_base(base: str, crate_dir: str, repo_root: Path) -> set[tuple[str, str]]:
    rule = (repo_root / RULE).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "base.tar"
        subprocess.run(
            ["git", "archive", "--output", str(archive), base, "--", crate_dir],
            cwd=repo_root, check=True, capture_output=True,
        )
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        return scan(Path(tmp), crate_dir, rule)


def ratchet(base: str | None, crate_dir: str, repo_root: Path) -> tuple[list[Finding], int]:
    head = scan(repo_root, crate_dir, (repo_root / RULE).resolve())
    if base is None:
        return [], len(head)
    new = sorted(head - at_base(base, crate_dir, repo_root))
    return (
        [Finding("assert-lint", "-", f"{p}::{n}", "new test checks nothing") for p, n in new],
        len(head),
    )
