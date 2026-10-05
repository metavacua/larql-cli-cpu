"""Ratios computed from a `Cell`. Every one is (covered, coverable)."""

from __future__ import annotations

from collections import defaultdict

from .cell import Cell

Ratio = tuple[int, int]


def percent(ratio: Ratio) -> float:
    covered, total = ratio
    # An empty denominator is not 100%: callers must not gate on it.
    return 100.0 * covered / total if total else 0.0


def per_file(cell: Cell) -> dict[str, Ratio]:
    return {
        path: (sum(1 for c in lines.values() if c > 0), len(lines))
        for path, lines in cell.hits.items()
        if lines
    }


def total(cell: Cell) -> Ratio:
    covered = coverable = 0
    for c, t in per_file(cell).values():
        covered, coverable = covered + c, coverable + t
    return covered, coverable


def function_ratio(cell: Cell) -> Ratio:
    return sum(1 for f in cell.functions if f.executed), len(cell.functions)


def per_trait(cell: Cell) -> dict[str, Ratio]:
    """Line coverage of each trait's implementations in this crate: the
    LCOV lines inside the spans of functions that implement it. A line is
    counted once per trait even if two impls share it."""
    lines: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for function in cell.functions:
        if function.trait is None:
            continue
        hits = cell.hits.get(function.path, {})
        for number in range(function.start, function.end + 1):
            if number in hits:
                lines[function.trait].add((function.path, number))
    return {
        trait: (sum(1 for p, n in keys if cell.hits[p][n] > 0), len(keys))
        for trait, keys in lines.items()
    }


def covered_lines(cell: Cell) -> dict[str, dict[int, bool]]:
    return {p: {n: c > 0 for n, c in lines.items()} for p, lines in cell.hits.items()}


def union_and_intersection(cells: list[Cell]) -> tuple[Ratio, Ratio]:
    """Across cells: (union, intersection) line coverage.

    Union: lines covered in ANY cell over lines coverable in any cell —
    what the whole matrix exercises, an upper bound no single build meets.
    Intersection: lines covered in EVERY cell that can compile them, over
    lines coverable in every cell — what is tested whatever is shipped.
    Per-cell gating sits between the two; both are reported, neither gated.
    """
    if not cells:
        return (0, 0), (0, 0)
    coverable: dict[tuple[str, int], int] = {}
    covered_any: set[tuple[str, int]] = set()
    covered_all: dict[tuple[str, int], bool] = {}
    for cell in cells:
        for path, lines in cell.hits.items():
            for number, count in lines.items():
                key = (path, number)
                coverable[key] = coverable.get(key, 0) + 1
                if count > 0:
                    covered_any.add(key)
                covered_all[key] = covered_all.get(key, True) and count > 0
    everywhere = [k for k, seen in coverable.items() if seen == len(cells)]
    union = (len(covered_any), len(coverable))
    intersection = (sum(1 for k in everywhere if covered_all[k]), len(everywhere))
    return union, intersection
