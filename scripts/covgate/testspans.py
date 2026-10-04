"""Line spans of items compiled only under `cfg(test)`, found by ast-grep.

Whole test-only FILES are found exactly from dep-info (see `depinfo.py`);
this covers the other half: `#[cfg(test)] mod tests { .. }` blocks and
test-only items inside a production file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .paths import repo_relative

QUERY = Path("lint/ast-grep/queries/cfg-test-item.yml")

# path -> list of inclusive 1-based (start, end) line spans
Spans = dict[str, list[tuple[int, int]]]


def parse(stream: str) -> Spans:
    """Parse `ast-grep scan --json=stream` output."""
    spans: Spans = {}
    for line in stream.splitlines():
        if not line.strip():
            continue
        match = json.loads(line)
        path = repo_relative(match["file"])
        if path is None:
            continue
        rng = match["range"]
        # ast-grep lines are 0-based.
        span = (rng["start"]["line"] + 1, rng["end"]["line"] + 1)
        spans.setdefault(path, []).append(span)
    return spans


def find(crate_dir: str, repo_root: Path) -> Spans:
    result = subprocess.run(
        ["ast-grep", "scan", "--rule", str(QUERY), crate_dir, "--json=stream"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    # Matches at severity `hint` exit 0; anything else is a broken scan.
    if result.returncode != 0:
        raise RuntimeError(f"ast-grep failed ({result.returncode}): {result.stderr.strip()}")
    return parse(result.stdout)


def contains(spans: Spans, path: str, line: int) -> bool:
    return any(start <= line <= end for start, end in spans.get(path, ()))
