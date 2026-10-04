"""Markdown and JSON renderings of a gate run."""

from __future__ import annotations

from . import metrics
from .cell import Cell
from .checks import Finding

# Rows shown per table; the JSON report carries everything.
WORST_FILES = 15


def _pct(ratio: metrics.Ratio) -> str:
    return f"{metrics.percent(ratio):.2f}% ({ratio[0]}/{ratio[1]})" if ratio[1] else "n/a"


def markdown(cells: list[Cell], findings: list[Finding]) -> str:
    lines = ["## Coverage gate", ""]
    lines += ["| cell | target | feature | lines | functions | traits below |", "|---|---|---|---|---|---|"]
    for cell in sorted(cells, key=lambda c: c.name):
        traits_below = sum(1 for f in findings if f.rule == "trait" and f.cell == cell.name)
        lines.append(
            f"| {cell.name} | {cell.target} | {cell.feature} | {_pct(metrics.total(cell))} "
            f"| {_pct(metrics.function_ratio(cell))} | {traits_below} |"
        )
    union, intersection = metrics.union_and_intersection(cells)
    lines += [
        "",
        f"Across cells (reported, not gated): union {_pct(union)}, intersection {_pct(intersection)}.",
    ]
    retried = sorted(c.name for c in cells if c.attempt > 1)
    if retried:
        lines.append(f"**Retried cells** (a pass here is a pass on a re-run): {', '.join(retried)}.")
    lines += ["", f"**{len(findings)} finding(s)**" if findings else "**All rules pass.**", ""]
    by_rule: dict[str, list[Finding]] = {}
    for finding in findings:
        by_rule.setdefault(finding.rule, []).append(finding)
    for rule, group in sorted(by_rule.items()):
        lines += [f"### {rule} ({len(group)})", "", "| cell | subject | detail |", "|---|---|---|"]
        for finding in group[:WORST_FILES]:
            lines.append(f"| {finding.cell} | `{finding.subject}` | {finding.message} |")
        if len(group) > WORST_FILES:
            lines.append(f"| … | {len(group) - WORST_FILES} more in covgate-report.json | |")
        lines.append("")
    return "\n".join(lines)


def as_json(cells: list[Cell], findings: list[Finding]) -> dict:
    return {
        "cells": {
            cell.name: {
                "target": cell.target,
                "feature": cell.feature,
                "total": metrics.total(cell),
                "functions": metrics.function_ratio(cell),
                "files": metrics.per_file(cell),
                "traits": metrics.per_trait(cell),
                "attempt": cell.attempt,
            }
            for cell in cells
        },
        "union_intersection": metrics.union_and_intersection(cells),
        "findings": [f.__dict__ for f in findings],
    }
