"""Exact one-sided McNemar test on paired mutant outcomes, and its exact
operating characteristics.

Per sampled mutant, base and head each catch or miss it. Only discordant
mutants carry information: b = caught on base, missed on head (a loss of
detection); c = the reverse. Model: each mutant is discordant with
probability `psi`, independently; a discordant one is a loss with
probability (psi + drop) / (2 psi), so `drop` = P(b) - P(c) is the fall in
D. Under H0 (drop = 0) a discordant mutant is a fair coin.

Reject when b is improbably large given m = b + c: the smallest k with
P(Bin(m, 1/2) >= k) <= alpha. The test is exact, so its true size is at
most alpha; `size` and `power` compute both exactly by summing over m.
"""

from __future__ import annotations

import math
from functools import lru_cache


def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def _upper_tail(m: int, p: float, k: int) -> float:
    if k > m:
        return 0.0
    return sum(math.exp(_log_pmf(j, m, p)) for j in range(max(k, 0), m + 1))


@lru_cache(maxsize=None)
def critical(m: int, alpha: float) -> int:
    """Smallest k with P(Bin(m, 1/2) >= k) <= alpha; m + 1 if none (the
    test cannot reject on so few discordant mutants)."""
    for k in range(m + 1):
        if _upper_tail(m, 0.5, k) <= alpha:
            return k
    return m + 1


def _rejection(n: int, psi: float, loss_share: float, alpha: float) -> float:
    return sum(
        math.exp(_log_pmf(m, n, psi)) * _upper_tail(m, loss_share, critical(m, alpha))
        for m in range(n + 1)
    )


def size(n: int, psi: float, alpha: float) -> float:
    """True false-red rate of the test on n paired mutants."""
    return _rejection(n, psi, 0.5, alpha)


def power(n: int, psi: float, drop: float, alpha: float) -> float:
    """Probability of rejecting when D truly fell by `drop`."""
    if not 0.0 < psi <= 1.0:
        raise ValueError(f"discordance must be in (0, 1]: {psi}")
    if not 0.0 <= drop <= psi:
        raise ValueError(f"a drop of {drop} needs at least that much discordance (psi={psi})")
    return _rejection(n, psi, (psi + drop) / (2 * psi), alpha)
