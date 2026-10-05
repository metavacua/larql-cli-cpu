"""Interval estimates and the sample sizes they imply.

A mutation kill rate is a SAMPLE: k of n planted defects caught. The
point estimate k/n has a margin of error that is large when n is small,
so the gate certifies a file only when the evidence supports it:

- the Wilson score interval bounds the true kill rate at a stated
  one-sided confidence;
- below `exact_threshold(rate, z)` mutants even a perfect score cannot
  push the lower bound to `rate`, so there the rule is exact instead:
  every mutant must be caught. The cut-over is derived, not chosen.

Line coverage is a CENSUS (every line is measured), so it has no sampling
error and needs none of this; its only margin is instrument noise
(`checks.EPSILON`).
"""

from __future__ import annotations

import math
from statistics import NormalDist


def z_one_sided(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1): {confidence}")
    return NormalDist().inv_cdf(confidence)


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, as fractions."""
    if n <= 0:
        raise ValueError("an interval needs at least one trial")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside 0..{n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_threshold(rate: float, z: float) -> int:
    """Smallest n at which n/n has a Wilson lower bound >= rate.

    For k = n the lower bound is n / (n + z^2), so n >= z^2 * rate / (1 - rate).
    """
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0, 1): {rate}")
    n = math.ceil(z * z * rate / (1 - rate))
    # Guard the float boundary: the bound itself is the definition.
    while wilson(n, n, z)[0] < rate:
        n += 1
    return n
