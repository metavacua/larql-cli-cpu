"""The replay plan and its resource budget.

The plan declares how many PRs, how many mutants each, and a runner-time
budget. The cost is estimated from a per-mutant prior BEFORE any runner
starts, and a plan over budget is refused. The prior is replaced by the
pilot's measured cost for every later plan.
"""

from __future__ import annotations

REQUIRED = (
    "crate_dir", "strata", "seed", "mutants_per_pr",
    "prior_minutes_per_mutant", "setup_minutes", "runner_minutes_budget",
)
SIDES = 2  # base and head


def _require(config: dict) -> None:
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ValueError(f"replay plan is missing {', '.join(missing)}")


def estimate_runner_minutes(config: dict, prs: int) -> float:
    _require(config)
    per_job = float(config["setup_minutes"]) + int(config["mutants_per_pr"]) * float(config["prior_minutes_per_mutant"])
    return prs * SIDES * per_job


def check_budget(config: dict, prs: int) -> float:
    estimate = estimate_runner_minutes(config, prs)
    budget = float(config["runner_minutes_budget"])
    if estimate > budget:
        raise ValueError(f"plan needs {estimate:.0f} runner-minutes, over the declared budget {budget:.0f}")
    return estimate
