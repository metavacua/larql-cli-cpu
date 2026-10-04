"""Picking ONE design from the frontier needs an assumption the data
cannot supply. Each rule here returns its assumption as a label, so a
report never shows a choice without the special case it rests on."""

from __future__ import annotations

from .pareto import Vector


def _score(v: Vector, weights: Vector) -> float:
    return sum(w * x for w, x in zip(weights, v))


def weighted_sum(points: dict, weights: Vector) -> tuple[str, object]:
    label = (f"weighted sum {weights}: assumes transferable utility — an hour of one "
             "stakeholder's cost trades for the stated weight of another's")
    return label, min(points, key=lambda k: _score(points[k], weights))


def gamma_minimax(scenarios: list[dict], weights: Vector) -> tuple[str, object]:
    label = ("Γ-minimax over the measured intervals: assumes nature is adversarial "
             "(a zero-sum game against the parameters), and the stated weights")
    keys = scenarios[0].keys()
    return label, min(keys, key=lambda k: max(_score(s[k], weights) for s in scenarios))


def minimax_regret(scenarios: list[dict], weights: Vector) -> tuple[str, object]:
    label = ("minimax regret: minimises the worst shortfall against the best design for "
             "each scenario; assumes the stated weights")
    keys = scenarios[0].keys()
    best = [min(_score(v, weights) for v in s.values()) for s in scenarios]
    return label, min(keys, key=lambda k: max(_score(s[k], weights) - b for s, b in zip(scenarios, best)))
