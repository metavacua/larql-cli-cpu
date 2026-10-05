"""The gate's rules. Each returns `Finding`s; an empty list is a pass.

Every rule is decided per cell, so a number can never be carried by a
different target or feature set than the one it describes:

- demangle:  every function name was demangled (otherwise the trait
             rule would pass vacuously)
- cells:     every cell the policy requires reported (a cell that did
             not run is a failure, not an absent row)
- total:     a cell's line coverage
- file:      each file's line coverage, in every cell that compiles it
- stale:     a per-file debt entry that matches no measured file
- functions: share of the crate's functions executed
- trait:     line coverage of each trait's implementations
- spread:    for one feature set, a line two targets both compile has
             the same verdict on both (cfg-gated code cannot hide behind
             the best target; census data, so no tolerance)
- diff:      lines the change adds, where coverable
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import metrics
from .cell import Cell
from .demangle import still_mangled
from .diff import Added
from .policy import Policy

# Measurement noise between llvm-cov builds; see check_coverage_policy.py.
EPSILON = 0.1


@dataclass(frozen=True)
class Finding:
    rule: str
    cell: str
    subject: str
    message: str


def _below(rule: str, cell: str, subject: str, ratio: metrics.Ratio, minimum: float) -> list[Finding]:
    value = metrics.percent(ratio)
    if ratio[1] == 0 or value + EPSILON >= minimum:
        return []
    return [Finding(rule, cell, subject, f"{value:.2f}% ({ratio[0]}/{ratio[1]}) < {minimum:.2f}%")]


def required_cells(cells: list[Cell], policy: Policy) -> list[Finding]:
    present = {c.name for c in cells}
    return [
        Finding("cells", name, name, "required cell did not report")
        for name in policy.required_cells
        if name not in present
    ]


def thresholds(cell: Cell, policy: Policy) -> list[Finding]:
    out = demangling(cell)
    return out + _threshold_rules(cell, policy)


def demangling(cell: Cell) -> list[Finding]:
    """Names left mangled cannot be attributed to traits; reporting the
    trait rule as passing on them would be a vacuous pass."""
    left = [f.name for f in cell.functions if still_mangled(f.name)]
    if not left:
        return []
    return [Finding("demangle", cell.name, f"{len(left)} of {len(cell.functions)} functions",
                    f"names still mangled, e.g. {left[0][:80]}")]


def _threshold_rules(cell: Cell, policy: Policy) -> list[Finding]:
    files = {p: r for p, r in metrics.per_file(cell).items() if policy.in_scope(p)}
    covered = sum(c for c, _ in files.values())
    coverable = sum(t for _, t in files.values())
    out = _below("total", cell.name, policy.crate_dir, (covered, coverable), policy.total_min)
    if coverable == 0:
        out.append(Finding("total", cell.name, policy.crate_dir, "no coverable lines measured"))
    for path, ratio in sorted(files.items()):
        out += _below("file", cell.name, path, ratio, policy.file_minimum(path))
    out += _below("functions", cell.name, policy.crate_dir, metrics.function_ratio(cell), policy.function_min)
    for trait, ratio in sorted(metrics.per_trait(cell).items()):
        out += _below("trait", cell.name, trait, ratio, policy.trait_min)
    return out


def stale_debt(cells: list[Cell], policy: Policy) -> list[Finding]:
    measured = {p for c in cells for p in c.hits}
    return [
        Finding("stale", "-", path, "per-file debt entry matches no measured file")
        for path in sorted(set(policy.per_file_min) - measured)
    ]


def spread(cells: list[Cell], policy: Policy) -> list[Finding]:
    """Cross-target divergence within one feature set.

    Line hits are a census, not a sample: every line is measured, so two
    targets that disagree about a line they both compile disagree in
    fact, not by noise. The rule therefore has no tolerance parameter:
    every line coverable on two targets must have the same verdict on
    both. (A points-based tolerance would only license hiding cfg-gated
    code behind the best target.)"""
    by_feature: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        by_feature[cell.feature].append(cell)
    out: list[Finding] = []
    for feature, group in sorted(by_feature.items()):
        paths = sorted({p for c in group for p in c.hits if policy.in_scope(p)})
        for path in paths:
            present = [c for c in group if c.hits.get(path)]
            if len(present) >= 2:
                out += _line_divergence(feature, path, present)
    return out


def _line_divergence(feature: str, path: str, present: list[Cell]) -> list[Finding]:
    shared = set.intersection(*(set(c.hits[path]) for c in present))
    split = sorted(
        n for n in shared
        if len({c.hits[path][n] > 0 for c in present}) > 1
    )
    if not split:
        return []
    covered_on = {c.target for c in present if all(c.hits[path][n] > 0 for n in split)}
    return [Finding(
        "spread", feature, path,
        f"lines {split[:8]}{'…' if len(split) > 8 else ''} covered on "
        f"{sorted(covered_on) or 'some targets'} but not on every target",
    )]


def diff_coverage(cell: Cell, added: Added, policy: Policy) -> list[Finding]:
    covered = coverable = 0
    for path, lines in added.items():
        hits = cell.hits.get(path)
        if hits is None or not policy.in_scope(path):
            continue
        for number in lines & hits.keys():
            coverable += 1
            covered += hits[number] > 0
    return _below("diff", cell.name, "added lines", (covered, coverable), policy.diff_min)


def run_all(cells: list[Cell], policy: Policy, added: Added | None) -> list[Finding]:
    findings = required_cells(cells, policy) + stale_debt(cells, policy) + spread(cells, policy)
    for cell in sorted(cells, key=lambda c: c.name):
        findings += thresholds(cell, policy)
        if added is not None:
            findings += diff_coverage(cell, added, policy)
    return findings


def split_gating(findings: list[Finding], policy: Policy) -> tuple[list[Finding], list[Finding]]:
    """(gating, reported-only) by the policy's `gating_rules`."""
    gating = [f for f in findings if f.rule in policy.gating_rules]
    return gating, [f for f in findings if f.rule not in policy.gating_rules]
