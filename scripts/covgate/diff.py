"""Lines a change adds, from `git diff --unified=0`."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .paths import repo_relative

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

Added = dict[str, set[int]]


def parse(text: str) -> Added:
    added: Added = {}
    current: set[int] | None = None
    for line in text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else repo_relative(target.removeprefix("b/"))
            current = None if path is None else added.setdefault(path, set())
            continue
        match = _HUNK.match(line)
        if match and current is not None:
            start, count = int(match.group(1)), int(match.group(2) or "1")
            current.update(range(start, start + count))
    return {p: lines for p, lines in added.items() if lines}


def added_lines(base: str, crate_dir: str, repo_root: Path) -> Added:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", f"{base}...HEAD", "--", crate_dir],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return parse(result.stdout)
