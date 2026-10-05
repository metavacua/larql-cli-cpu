"""The coverage policy a crate declares in its `coverage-policy.json`.

The top-level keys keep `scripts/check_coverage_policy.py`'s schema; the
`gate` object adds what that script cannot see. Every threshold is read
explicitly: a missing key is an error, never a silent default.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

# Every rule a finding can carry. A policy names which of them gate; the
# rest are reported. Rules decided against a threshold someone picked
# (file, total, functions, trait, diff) report until a measured
# derivation replaces the number; rules with no free parameter gate.
RULES = (
    "demangle", "cells", "total", "file", "stale", "functions", "trait", "spread", "diff",
    "ratchet", "assert-lint", "mutation",
)

GATE_KEYS = (
    "function_min_percent",
    "trait_line_min_percent",
    "diff_line_min_percent",
    "mutation_min_kill_percent",
    "mutation_confidence",
)


@dataclass(frozen=True)
class Policy:
    crate_dir: str
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    file_min: float
    total_min: float
    per_file_min: dict[str, float]
    function_min: float
    trait_min: float
    diff_min: float
    mutation_min: float
    mutation_confidence: float
    required_cells: tuple[str, ...]
    gating_rules: frozenset[str]
    raw: dict = field(compare=False, default_factory=dict)

    def in_scope(self, path: str) -> bool:
        if self.include_globs and not any(fnmatch.fnmatch(path, g) for g in self.include_globs):
            return False
        return not any(fnmatch.fnmatch(path, g) for g in self.exclude_globs)

    def file_minimum(self, path: str) -> float:
        return self.per_file_min.get(path, self.file_min)


def _rules(names: list[str]) -> frozenset[str]:
    unknown = sorted(set(names) - set(RULES))
    if unknown:
        raise ValueError(f"unknown gating rule(s): {', '.join(unknown)}")
    return frozenset(names)


def parse(raw: dict, crate_dir: str) -> Policy:
    gate = raw.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("coverage policy has no `gate` object")
    missing = [k for k in GATE_KEYS + ("required_cells", "gating_rules") if k not in gate]
    if missing:
        raise ValueError(f"coverage policy `gate` is missing {', '.join(missing)}")
    return Policy(
        crate_dir=crate_dir,
        include_globs=tuple(raw.get("include_globs", [])),
        exclude_globs=tuple(raw.get("exclude_globs", [])),
        file_min=float(raw["default_line_min_percent"]),
        total_min=float(raw["total_line_min_percent"]),
        per_file_min={k: float(v) for k, v in raw.get("per_file_line_min_percent", {}).items()},
        function_min=float(gate["function_min_percent"]),
        trait_min=float(gate["trait_line_min_percent"]),
        diff_min=float(gate["diff_line_min_percent"]),
        mutation_min=float(gate["mutation_min_kill_percent"]),
        mutation_confidence=float(gate["mutation_confidence"]),
        required_cells=tuple(gate["required_cells"]),
        gating_rules=_rules(gate["gating_rules"]),
        raw=raw,
    )
