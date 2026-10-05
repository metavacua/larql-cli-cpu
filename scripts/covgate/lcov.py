"""LCOV line hits: the one line-level measurement every rule is built on."""

from __future__ import annotations

from pathlib import Path

from .paths import repo_relative

# path -> line -> hit count
LineHits = dict[str, dict[int, int]]


def parse(text: str) -> LineHits:
    """Parse `SF:`/`DA:` records. A line recorded twice (two instantiations
    of a generic, two test binaries) keeps the larger hit count: it is
    covered if any execution covered it."""
    hits: LineHits = {}
    current: dict[int, int] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            path = repo_relative(line[3:])
            current = None if path is None else hits.setdefault(path, {})
        elif line.startswith("DA:") and current is not None:
            fields = line[3:].split(",")
            if len(fields) < 2:
                raise ValueError(f"malformed DA record: {raw!r}")
            number, count = int(fields[0]), int(float(fields[1]))
            current[number] = max(current.get(number, 0), count)
        elif line == "end_of_record":
            current = None
    return hits


def load(path: Path) -> LineHits:
    return parse(path.read_text(encoding="utf-8"))
