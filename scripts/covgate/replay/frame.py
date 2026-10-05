"""The sampling frame: merged PRs that changed the crate."""

from __future__ import annotations

import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_MERGE = re.compile(r"^Merge pull request #(\d+)\b")


@dataclass(frozen=True)
class Pr:
    number: int
    merge: str
    base: str     # main before the merge (first parent)
    head: str     # the PR's tip (second parent)
    merged_at: float
    # Where base and head can be BUILT. This repo's imported history was
    # rewritten by git filter-repo: its old commits name workspace members
    # whose files were never imported, so they cannot build here. The PR is
    # replayed in the repository that merged it, at its original SHAs.
    repo: str = ""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def merged_prs(root: Path, ref: str, crate_dir: str) -> list[Pr]:
    """Merge commits on `ref`'s first-parent line whose change to `ref`
    touched `crate_dir`, oldest first."""
    out = _git(root, "log", "--merges", "--first-parent", "--format=%H%x09%P%x09%ct%x09%s", ref)
    prs = []
    for line in out.splitlines():
        merge, parents, ts, subject = line.split("\t", 3)
        match = _MERGE.match(subject)
        parent = parents.split()
        if match is None or len(parent) != 2:
            continue
        changed = _git(root, "diff", "--name-only", parent[0], merge, "--", crate_dir).split()
        if changed:
            prs.append(Pr(int(match.group(1)), merge, parent[0], parent[1], float(ts)))
    return sorted(prs, key=lambda p: p.merged_at)


def sample(prs: list[Pr], strata: int, seed: int) -> list[Pr]:
    """One PR from each of `strata` equal-count eras (quantiles of merge
    time), chosen by a seeded generator so the sample is reproducible and
    recorded rather than hand-picked."""
    ordered = sorted(prs, key=lambda p: p.merged_at)
    if strata <= 0 or len(ordered) < strata:
        raise ValueError(f"cannot draw {strata} strata from {len(ordered)} PRs")
    rng = random.Random(seed)
    chosen = []
    for i in range(strata):
        era = ordered[i * len(ordered) // strata : (i + 1) * len(ordered) // strata]
        chosen.append(rng.choice(era))
    return chosen


def resolve(get, repo: str, number: int, merged_at: float = 0.0) -> Pr:
    """The PR as merged in `repo`: base = the merge's first parent, head =
    its second (or the merge commit itself for a squash or rebase merge).
    `get(path)` returns a GitHub REST response body."""
    pull = get(f"repos/{repo}/pulls/{number}")
    if not pull.get("merged") or not pull.get("merge_commit_sha"):
        raise ValueError(f"{repo}#{number} is not merged; nothing to replay")
    merge = pull["merge_commit_sha"]
    parents = [p["sha"] for p in get(f"repos/{repo}/commits/{merge}")["parents"]]
    head = parents[1] if len(parents) > 1 else merge
    return Pr(number, merge, parents[0], head, merged_at, repo)
