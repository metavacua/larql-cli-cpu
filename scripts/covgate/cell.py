"""One measured cell, with test code already removed from its denominator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import depinfo, functions, lcov
from .functions import Function
from .lcov import LineHits
from .paths import in_crate
from .testspans import Spans, contains

# Files a cell directory must contain; written by the coverage workflow.
CELL_META, CELL_LCOV, CELL_JSON, CELL_DEPINFO = "cell.json", "lcov.info", "cov.json", "depinfo"


@dataclass
class Cell:
    name: str
    target: str
    feature: str
    hits: LineHits
    functions: list[Function] = field(default_factory=list)
    # GitHub's run_attempt: > 1 means the job was re-run. A pass on a
    # re-run is reported as one, never as a plain pass.
    attempt: int = 1


def restrict(
    hits: LineHits,
    funcs: list[Function],
    crate_dir: str,
    production_files: set[str],
    spans: Spans,
) -> tuple[LineHits, list[Function]]:
    """Keep the crate's production code only: files a non-test build
    compiles, minus `cfg(test)` item spans inside them."""
    kept: LineHits = {}
    for path, lines in hits.items():
        if not in_crate(path, crate_dir) or path not in production_files:
            continue
        kept[path] = {n: c for n, c in lines.items() if not contains(spans, path, n)}
    kept_funcs = [
        f for f in funcs
        if f.path in kept and not contains(spans, f.path, f.start)
    ]
    return kept, kept_funcs


def load(directory: Path, crate_dir: str, spans: Spans) -> Cell:
    meta = json.loads((directory / CELL_META).read_text(encoding="utf-8"))
    production = depinfo.load(sorted((directory / CELL_DEPINFO).glob("*.d")))
    if not any(in_crate(p, crate_dir) for p in production):
        # Without dep-info every file would read as test code and the cell
        # would pass vacuously on an empty denominator.
        raise ValueError(f"{directory}: dep-info names no file of {crate_dir}")
    hits, funcs = restrict(
        lcov.load(directory / CELL_LCOV),
        functions.load(directory / CELL_JSON),
        crate_dir,
        production,
        spans,
    )
    return Cell(meta["name"], meta["target"], meta["feature"], hits, funcs, int(meta["attempt"]))
