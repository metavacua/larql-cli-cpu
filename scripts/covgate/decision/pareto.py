"""Designs scored as stakeholder cost vectors, and the Pareto frontier."""

from __future__ import annotations

from dataclasses import dataclass

from . import mcnemar

Vector = tuple[float, ...]


@dataclass(frozen=True)
class Design:
    n: int        # mutants sampled per PR (each run on base AND head)
    alpha: float  # nominal size of the exact test


@dataclass(frozen=True)
class Inputs:
    """Measured quantities; each comes from calibration or the replay
    experiment, never from a default."""
    p_degrade: float                 # share of changes that lower D
    false_red_hours: float           # author delay per false red
    escape_hours: float              # user exposure per escaped degradation
    runner_hours_per_mutant: float   # CI time per mutant, per side
    discordance: float               # psi: share of mutants whose verdict differs base vs head
    drop: float                      # size of a typical degradation in D


AXES = ("author delay (h/PR)", "user exposure (h/PR)", "runner time (h/PR)")


def design_vector(design: Design, inputs: Inputs) -> Vector:
    size = mcnemar.size(design.n, inputs.discordance, design.alpha)
    miss = 1.0 - mcnemar.power(design.n, inputs.discordance, inputs.drop, design.alpha)
    return (
        (1.0 - inputs.p_degrade) * size * inputs.false_red_hours,
        inputs.p_degrade * miss * inputs.escape_hours,
        2 * design.n * inputs.runner_hours_per_mutant,
    )


def dominates(a: Vector, b: Vector) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def frontier(points: dict) -> list:
    """Keys whose vectors no other vector dominates (all axes minimised).
    Equal vectors do not dominate each other, so ties stay."""
    return [k for k, v in points.items() if not any(dominates(w, v) for w in points.values())]
