"""Fetch CI verdicts — every attempt of every completed workflow run —
from the GitHub API.

Two API properties shape this:
- the runs listing returns at most 1000 results per query and truncates
  silently, so a date window whose `total_count` exceeds that is split
  until every window fits (otherwise the sample is silently biased toward
  whichever runs the API happened to return);
- a listed run is its LATEST attempt; earlier attempts of a re-run (where
  false reds live) need one call each.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .client import Client
from .false_reds import Verdict

RESULT_CAP = 1000
PER_PAGE = 100
ONE_SECOND = timedelta(seconds=1)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _verdict(repo: str, run: dict, attempt: int) -> Verdict:
    return Verdict(
        workflow=f"{repo}:{run['workflow_id']}",
        sha=run["head_sha"],
        conclusion=run.get("conclusion") or "",
        completed_at=_ts(run["updated_at"]),
        run_id=int(run["id"]),
        attempt=attempt,
    )


def _windows(client: Client, repo: str, lo: datetime, hi: datetime) -> list[tuple[datetime, datetime, int]]:
    """Split [lo, hi] (inclusive) into windows of at most RESULT_CAP runs,
    by bisection on the measured `total_count` (one probe per window)."""
    probe = client.get(f"repos/{repo}/actions/runs?per_page=1&status=completed&created={_iso(lo)}..{_iso(hi)}").body
    total = int(probe["total_count"])
    if total <= RESULT_CAP:
        return [(lo, hi, total)] if total else []
    if hi - lo < 2 * ONE_SECOND:
        raise RuntimeError(f"{repo}: more than {RESULT_CAP} runs created within one second at {_iso(lo)}")
    mid = lo + (hi - lo) / 2
    mid = mid.replace(microsecond=0)
    return _windows(client, repo, lo, mid) + _windows(client, repo, mid + ONE_SECOND, hi)


def plan(client: Client, repo: str, start: datetime, end: datetime) -> dict:
    """What a fetch of [start, end] would cost, spending only the window
    probes. Attempt calls for re-runs are not knowable until the listing
    is read; they are reported by `fetch_verdicts` as they are reserved."""
    before = client.counted_requests
    windows = _windows(client, repo, start, end)
    return {
        "repo": repo,
        "runs": sum(total for _, _, total in windows),
        "windows": len(windows),
        "probe_requests": client.counted_requests - before,
        "page_requests": sum(-(-total // PER_PAGE) for _, _, total in windows),
    }


def fetch_verdicts(client: Client, repo: str, start: datetime, end: datetime) -> list[Verdict]:
    """Every attempt of every completed run created in [start, end].

    The windows partition the range, so the distinct runs fetched must
    equal the sum of the windows' `total_count`s; a mismatch (truncation,
    a page lost to an error, an overlap) raises instead of returning a
    silently smaller — and biased — sample."""
    verdicts: dict[tuple[int, int], Verdict] = {}
    windows = _windows(client, repo, start, end)
    # The whole page plan is known from the probes: commit to it, or refuse
    # it, before spending a single page request.
    client.reserve(sum(-(-total // PER_PAGE) for _, _, total in windows))
    for lo, hi, _ in windows:
        first = (
            f"repos/{repo}/actions/runs?per_page={PER_PAGE}&page=1"
            f"&status=completed&created={_iso(lo)}..{_iso(hi)}"
        )
        for response in client.paginate(first):
            for run in response.body.get("workflow_runs", []):
                latest = int(run.get("run_attempt") or 1)
                verdicts[(int(run["id"]), latest)] = _verdict(repo, run, latest)
                if latest > 1:
                    client.reserve(latest - 1)
                for attempt in range(1, latest):
                    earlier = client.get(f"repos/{repo}/actions/runs/{run['id']}/attempts/{attempt}").body
                    verdicts[(int(run["id"]), attempt)] = _verdict(repo, earlier, attempt)
    expected = sum(total for _, _, total in windows)
    fetched = len({run_id for run_id, _ in verdicts})
    if fetched != expected:
        raise RuntimeError(f"{repo}: fetched {fetched} of {expected} runs in {_iso(start)}..{_iso(end)}")
    return list(verdicts.values())
