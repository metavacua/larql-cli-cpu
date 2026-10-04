"""Mutation kill rate per file, from cargo-mutants' `outcomes.json`.

Line coverage says a line ran; a caught mutant says a test would notice
if it were wrong. Covered lines whose mutants survive are padding.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import stats
from .checks import Finding
from .metrics import Ratio, percent
from .paths import in_crate, repo_relative


def parse(outcomes: dict) -> tuple[str | None, dict[str, dict[str, int]]]:
    """(baseline summary, path -> {outcome: count})."""
    baseline: str | None = None
    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in outcomes.get("outcomes", []):
        scenario = outcome.get("scenario")
        if scenario == "Baseline":
            baseline = outcome.get("summary")
            continue
        mutant = scenario.get("Mutant") if isinstance(scenario, dict) else None
        if not mutant:
            continue
        path = repo_relative(mutant.get("file", ""))
        if path is not None:
            tally[path][outcome.get("summary", "?")] += 1
    return baseline, {p: dict(c) for p, c in tally.items()}


def kill_ratio(counts: dict[str, int]) -> Ratio:
    """Caught over caught+missed. Timeouts and unviable mutants are
    reported but sit outside the ratio: neither is evidence either way."""
    caught, missed = counts.get("CaughtMutant", 0), counts.get("MissedMutant", 0)
    return caught, caught + missed


def judge(ratio: Ratio, minimum: float, confidence: float) -> str | None:
    """`None` when the file's kill rate is certified >= minimum, else why not.

    Below the derived exact threshold every mutant must be caught; at or
    above it, the one-sided Wilson lower bound must reach the minimum.
    A failure says whether the rate is confidently below the minimum or
    merely not demonstrated, because the remedies differ (fix the tests
    vs. gather more evidence)."""
    caught, n = ratio
    if n == 0:
        return None
    rate = minimum / 100.0
    z = stats.z_one_sided(confidence)
    exact_n = stats.exact_threshold(rate, z)
    if n < exact_n:
        missed = n - caught
        if missed == 0:
            return None
        return (
            f"{missed} of {n} mutants survived; below {exact_n} mutants a sample "
            f"cannot certify {minimum:.0f}%, so every mutant must be caught"
        )
    low, high = stats.wilson(caught, n, z)
    if low >= rate:
        return None
    verdict = "confidently below" if high < rate else "not demonstrated"
    return (
        f"kill rate {percent(ratio):.2f}% ({caught}/{n}), {confidence:.0%} one-sided interval "
        f"[{100 * low:.1f}%, {100 * high:.1f}%]: {verdict} {minimum:.0f}%"
    )


def check(
    outcomes_path: Path, crate_dir: str, minimum: float, confidence: float
) -> tuple[list[Finding], dict[str, dict[str, int]]]:
    baseline, tally = parse(json.loads(outcomes_path.read_text(encoding="utf-8")))
    findings: list[Finding] = []
    if baseline not in (None, "Success"):
        findings.append(Finding("mutation", "-", "baseline", f"baseline {baseline}: no mutant was tested"))
    for path, counts in sorted(tally.items()):
        if not in_crate(path, crate_dir):
            continue
        reason = judge(kill_ratio(counts), minimum, confidence)
        if reason is not None:
            findings.append(Finding("mutation", "-", path, reason))
    return findings, tally
