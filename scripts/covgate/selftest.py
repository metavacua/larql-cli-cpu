"""End-to-end controls for the gate itself.

Unit tests check each rule in memory. This runs the real pipeline —
files on disk, dep-info, ast-grep spans, policy parsing — on two
synthetic crates: a known-clean one that must pass (specificity) and a
planted-defect one that must trip every rule (sensitivity). A rule that
does not fire on its planted defect is a gate that cannot fail.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import metrics, mutation
from .cell import CELL_DEPINFO, CELL_JSON, CELL_LCOV, CELL_META
from .testspans import QUERY

CRATE = "crates/demo"
SRC = f"{CRATE}/src/lib.rs"
TEST_FILE = f"{CRATE}/src/lib_tests.rs"
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")
# 40 production lines, then `#[cfg(test)]` on 42 and its `mod tests` on 43..44.
SOURCE = "\n".join(["pub fn f() {}"] * 40 + ["", "#[cfg(test)]", "mod tests {", "}", ""])
TRAIT_SYMBOL = "_ZN" + "".join(
    f"{len(s)}{s}" for s in ("_$LT$demo..S$u20$as$u20$demo..Tr$GT$", "f", "h0123456789abcdef")
) + "E"
EXPECTED_PLANTED = {"cells", "total", "file", "functions", "trait", "spread"}


def _policy() -> dict:
    return {
        "include_globs": [f"{CRATE}/*"], "exclude_globs": [],
        "default_line_min_percent": 90.0, "total_line_min_percent": 90.0,
        "per_file_line_min_percent": {},
        "gate": {
            "function_min_percent": 90.0, "trait_line_min_percent": 90.0,
            "diff_line_min_percent": 90.0,
            "mutation_min_kill_percent": 90.0, "mutation_confidence": 0.95,
            "required_cells": [f"{t}.release" for t in TARGETS] + ["missing.release"],
            # The self-test gates every rule: it proves each one CAN fire.
            "gating_rules": ["cells", "total", "file", "stale", "functions", "trait", "spread", "diff"],
        },
    }


def _cell(root: Path, target: str, covered: int, trait_ran: bool) -> None:
    d = root / "cells" / f"{target}.release"
    (d / CELL_DEPINFO).mkdir(parents=True)
    (d / CELL_META).write_text(json.dumps(
        {"name": f"{target}.release", "target": target, "feature": "release", "attempt": 1}))
    # Production lines 1..40, plus the test file and the cfg(test) span,
    # both fully covered: if test code leaked into the denominator, a
    # planted shortfall would be masked.
    da = "".join(f"DA:{n},{1 if n <= covered else 0}\n" for n in range(1, 41))
    da += "".join(f"DA:{n},1\n" for n in range(43, 45))
    (d / CELL_LCOV).write_text(
        f"SF:/w/{SRC}\n{da}end_of_record\nSF:/w/{TEST_FILE}\nDA:1,9\nend_of_record\n")
    (d / CELL_DEPINFO / "demo.d").write_text(f"/t/demo: /w/{SRC}\n")
    region = [1, 1, 40, 2, 1 if trait_ran else 0, 0, 0, 0]
    (d / CELL_JSON).write_text(json.dumps({"data": [{"functions": [
        {"name": TRAIT_SYMBOL, "count": int(trait_ran), "filenames": [f"/w/{SRC}"], "regions": [region]},
    ]}]}))


def _crate(root: Path, repo_root: Path, clean: bool) -> None:
    (root / CRATE / "src").mkdir(parents=True)
    (root / SRC).write_text(SOURCE)
    policy = _policy()
    if clean:
        policy["gate"]["required_cells"] = [f"{t}.release" for t in TARGETS]
    (root / CRATE / "coverage-policy.json").write_text(json.dumps(policy))
    (root / QUERY).parent.mkdir(parents=True)
    shutil.copy(repo_root / QUERY, root / QUERY)
    if clean:
        for target in TARGETS:
            _cell(root, target, covered=40, trait_ran=True)
    else:
        # 30/40 and 39/40: below every floor, and lines 31..39 diverge across targets.
        _cell(root, TARGETS[0], covered=30, trait_ran=False)
        _cell(root, TARGETS[1], covered=39, trait_ran=True)


def _mutation_controls() -> list[str]:
    failures = []
    for label, caught, missed, expect_fail in (("clean", 100, 0, False), ("planted", 80, 20, True)):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.json"
            rows = [{"scenario": "Baseline", "summary": "Success"}]
            rows += [{"scenario": {"Mutant": {"file": SRC}}, "summary": "CaughtMutant"}] * caught
            rows += [{"scenario": {"Mutant": {"file": SRC}}, "summary": "MissedMutant"}] * missed
            path.write_text(json.dumps({"outcomes": rows}))
            found = mutation.check(path, CRATE, 90.0, 0.95)[0]
        if bool(found) != expect_fail:
            failures.append(f"mutation {label}: expected {'a finding' if expect_fail else 'none'}, got {found}")
    return failures


def run(repo_root: Path) -> list[str]:
    from .__main__ import run_check

    failures: list[str] = []
    for clean in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _crate(root, repo_root, clean)
            cells, findings, _ = run_check(root, root / "cells", CRATE, None)
        rules = {f.rule for f in findings}
        measured = {metrics.total(c)[1] for c in cells}
        if measured != {40}:
            failures.append(f"test code leaked into the denominator: {measured} lines, expected 40")
        if clean and findings:
            failures.append(f"known-clean crate failed: {findings}")
        if not clean and rules != EXPECTED_PLANTED:
            failures.append(
                f"planted defects: missed {sorted(EXPECTED_PLANTED - rules)}, "
                f"unexpected {sorted(rules - EXPECTED_PLANTED)}")
    return failures + _mutation_controls()
