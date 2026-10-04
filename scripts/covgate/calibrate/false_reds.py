"""False reds from CI history: verdicts that changed on identical code.

For one workflow and one commit SHA the code under test is fixed, so a
change of verdict is the check's own error, not the code's:

- failure, later success on the same SHA: a FALSE RED. Its cost is the
  delay from the red verdict to the green one.
- success, later failure on the same SHA: the verdicts disagree but which
  is wrong is unknown, so it is counted as nondeterminism only.

Known error sources: a red that nobody re-ran on the same SHA is never
observed to flip (undercount); a re-run can pass because an external
service recovered, which is still a false red of the check as run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

VERDICTS = {"success", "failure"}
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Verdict:
    workflow: str
    sha: str
    conclusion: str
    completed_at: float  # unix seconds
    run_id: int = 0
    attempt: int = 1


@dataclass
class Flips:
    verdicts: int = 0
    reds: int = 0
    false_reds: int = 0
    nondeterministic: int = 0
    delays_hours: list[float] = field(default_factory=list)


def flips(records: list[Verdict]) -> Flips:
    result = Flips()
    groups: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
    for record in records:
        if record.conclusion in VERDICTS:
            groups[(record.workflow, record.sha)].append(record)
    for group in groups.values():
        group.sort(key=lambda r: r.completed_at)
        result.verdicts += len(group)
        reds = [r for r in group if r.conclusion == "failure"]
        result.reds += len(reds)
        for red in reds:
            later_green = next(
                (g for g in group if g.conclusion == "success" and g.completed_at > red.completed_at),
                None,
            )
            if later_green is not None:
                result.false_reds += 1
                result.delays_hours.append((later_green.completed_at - red.completed_at) / SECONDS_PER_HOUR)
        greens = [r for r in group if r.conclusion == "success"]
        if any(g.completed_at < r.completed_at for g in greens for r in reds):
            result.nondeterministic += 1
    return result
