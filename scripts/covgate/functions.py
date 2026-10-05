"""Function regions from an llvm-cov JSON export.

LCOV gives line hits but not which function a line belongs to; the JSON
export gives each function's code regions and its execution count. The
two together attribute lines to the trait a function implements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .demangle import demangle_all, implemented_trait
from .paths import repo_relative

# llvm-cov region tuple: [line_start, col_start, line_end, col_end,
# execution_count, file_id, expanded_file_id, kind]; kind 0 is code.
_LINE_START, _LINE_END, _COUNT, _FILE_ID, _KIND = 0, 2, 4, 5, 7
_CODE_REGION = 0


@dataclass(frozen=True)
class Function:
    path: str
    name: str
    trait: str | None
    start: int
    end: int
    executed: bool


def parse(export: dict) -> list[Function]:
    """One `Function` per distinct (file, start line, name). Generic
    instantiations share a source span; the function counts as executed
    if any instantiation ran. Names are demangled in one batch."""
    rows = []
    for data in export.get("data", []):
        for record in data.get("functions", []):
            filenames = record.get("filenames") or []
            path = repo_relative(filenames[0]) if filenames else None
            if path is None:
                continue
            regions = [
                r for r in record.get("regions", [])
                if r[_FILE_ID] == 0 and r[_KIND] == _CODE_REGION
            ]
            if regions:
                rows.append((path, record.get("name", ""), regions, int(record.get("count", 0)) > 0))
    names = demangle_all([raw for _, raw, _, _ in rows])
    merged: dict[tuple[str, int, str], Function] = {}
    for path, raw, regions, executed in rows:
        name = names.get(raw, raw)
        start = min(r[_LINE_START] for r in regions)
        end = max(r[_LINE_END] for r in regions)
        key = (path, start, name)
        previous = merged.get(key)
        if previous is not None:
            executed = executed or previous.executed
            end = max(end, previous.end)
        merged[key] = Function(path, name, implemented_trait(name), start, end, executed)
    return list(merged.values())


def load(path: Path) -> list[Function]:
    with path.open("r", encoding="utf-8") as handle:
        return parse(json.load(handle))
