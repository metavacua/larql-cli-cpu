"""Paired outcomes of one replayed PR, and the pilot's aggregate."""

from __future__ import annotations

from statistics import NormalDist, median

from .. import stats

CONFIDENCE = 0.95  # two-sided, for reported intervals

VERDICTS = {"CaughtMutant": True, "MissedMutant": False}


def _by_name(outcomes: dict) -> tuple[dict[str, str], list[float]]:
    summaries, seconds = {}, []
    for o in outcomes.get("outcomes", []):
        scenario = o.get("scenario")
        if not isinstance(scenario, dict) or "Mutant" not in scenario:
            continue
        summaries[scenario["Mutant"]["name"]] = o.get("summary", "")
        seconds.append(sum(p.get("duration", 0.0) for p in o.get("phase_results", [])))
    return summaries, seconds


def paired(selection: dict, base_outcomes: dict, head_outcomes: dict) -> dict:
    """`selection` maps a pairing key to (base name, head name). A pair
    counts only when both sides gave a verdict (caught or missed); a
    timeout or unviable mutant on either side is excluded and counted."""
    base, base_s = _by_name(base_outcomes)
    head, head_s = _by_name(head_outcomes)
    n = b = c = excluded = 0
    for base_name, head_name in selection.values():
        x, y = base.get(base_name), head.get(head_name)
        if x not in VERDICTS or y not in VERDICTS:
            excluded += 1
            continue
        n += 1
        b += VERDICTS[x] and not VERDICTS[y]
        c += VERDICTS[y] and not VERDICTS[x]
    durations = base_s + head_s
    return {
        "n": n, "b": b, "c": c, "excluded": excluded,
        "discordance": (b + c) / n if n else None,
        "drop": (b - c) / n if n else None,
        "seconds_per_mutant": sum(durations) / len(durations) if durations else None,
    }


def aggregate(results: dict[str, dict]) -> dict:
    """Pilot estimates across replayed PRs. Discordance is pooled with a
    Wilson interval (mutants are the sampling unit); each PR's drop is
    kept separately, because the drop distribution ACROSS PRs is what the
    decision needs. A PR whose commits no longer build is counted, never
    silently dropped (it is a selection effect on the frame)."""
    ok = {pr: r for pr, r in results.items() if r.get("status") == "ok"}
    n = sum(r["n"] for r in ok.values())
    discordant = sum(r["b"] + r["c"] for r in ok.values())
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    interval = stats.wilson(discordant, n, z) if n else (None, None)
    costs = [r["seconds_per_mutant"] for r in ok.values() if r.get("seconds_per_mutant") is not None]
    return {
        "prs": {"replayed": len(ok), "unbuildable": len(results) - len(ok)},
        "discordance": (interval[0], discordant / n if n else None, interval[1]),
        "drops": {pr: (r["b"] - r["c"]) / r["n"] for pr, r in ok.items() if r["n"]},
        "seconds_per_mutant_median": median(costs) if costs else None,
    }
