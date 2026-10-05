"""Calibration estimates with their uncertainty, one stratum at a time.

Strata (repository x crate for escapes; repository x workflow for CI
verdicts) are never pooled here: pooling across strata with different
rates can reverse a comparison (Simpson's paradox). Rates carry Wilson
intervals; delays carry distribution-free order-statistic intervals for
their median, because they are heavy-tailed and no normal approximation
is justified.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

from .. import stats
from .false_reds import Verdict, flips

CONFIDENCE = 0.95  # two-sided, for reported intervals


def _binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, j) for j in range(k + 1)) / 2**n if k >= 0 else 0.0


def _point_quantile(xs: list[float], q: float) -> float:
    position = q * (len(xs) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return xs[lo] + (xs[hi] - xs[lo]) * (position - lo)


def quantile_interval(values: list[float], q: float, confidence: float):
    """(lower, point, upper) for the q-quantile; bounds are order
    statistics x(l), x(u) with P(l <= rank < u) >= confidence under
    Binomial(n, q). Only q = 1/2 is supported (exact symmetric ranks).
    Bounds are None when n is too small for any such pair."""
    if q != 0.5:
        raise ValueError("only the median is supported")
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return (None, None, None)
    point = _point_quantile(xs, q)
    tail = (1 - confidence) / 2
    lower_rank = None
    for r in range(n // 2, -1, -1):
        if _binom_cdf(r - 1, n) <= tail:
            lower_rank = r
            break
    if lower_rank is None or lower_rank < 1:
        return (None, point, None)
    upper_rank = n + 1 - lower_rank
    return (xs[lower_rank - 1], point, xs[upper_rank - 1])


def _rate(k: int, n: int) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return (None, None, None)
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    low, high = stats.wilson(k, n, z)
    return (low, k / n, high)


def summarize(escape_reports: list[dict], verdicts: list[Verdict]) -> dict:
    escapes = {}
    for report in escape_reports:
        key = f"{report['repo']} {report.get('only') or '.'}"
        intro = {e["introduced"] for e in report["escapes"]}
        latencies = [e["latency_hours"] for e in report["escapes"]]
        escapes[key] = {
            "changes": report["changes"],
            "introducing_changes": len(intro),
            "defect_rate": _rate(len(intro), report["changes"]),
            "latency_hours_median": quantile_interval(latencies, 0.5, CONFIDENCE),
        }
    by_workflow: dict[str, list[Verdict]] = defaultdict(list)
    for verdict in verdicts:
        by_workflow[verdict.workflow].append(verdict)
    false_reds = {}
    for workflow, group in sorted(by_workflow.items()):
        result = flips(group)
        false_reds[workflow] = {
            "verdicts": result.verdicts,
            "reds": result.reds,
            "false_reds": result.false_reds,
            "nondeterministic": result.nondeterministic,
            "false_red_share_of_reds": _rate(result.false_reds, result.reds),
            "delay_hours_median": quantile_interval(result.delays_hours, 0.5, CONFIDENCE),
        }
    return {"confidence": CONFIDENCE, "escapes": escapes, "false_reds": false_reds}
