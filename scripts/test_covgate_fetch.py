"""Tests for covgate.calibrate.runs against a fake GitHub API that
enforces the real API's 1000-result cap per query."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate.calibrate.client import BudgetExceeded, Client, Response  # noqa: E402
from covgate.calibrate.runs import fetch_verdicts, plan  # noqa: E402

RESULT_CAP = 1000
DAY0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeApi:
    """`runs` per day; each listed run is its latest attempt. Run ids
    divisible by 7 were re-run once: attempt 1 failed, attempt 2 passed."""

    def __init__(self, runs_per_day: dict[int, int]):
        self.runs = []
        rid = 1
        for day, count in sorted(runs_per_day.items()):
            for i in range(count):
                t = DAY0 + timedelta(days=day, minutes=i)
                rerun = rid % 7 == 0
                self.runs.append({
                    "id": rid, "workflow_id": 10 + rid % 2, "head_sha": f"sha{rid}",
                    "conclusion": "success", "run_attempt": 2 if rerun else 1,
                    "created_at": iso(t), "updated_at": iso(t + timedelta(hours=2 if rerun else 0, minutes=5)),
                })
                rid += 1
        self.calls: list[str] = []

    def __call__(self, path: str, headers: dict) -> Response:
        body = self.body(path)
        link = ""
        url = urlparse(path)
        q = parse_qs(url.query)
        if "workflow_runs" in body and "page" in q and int(q["per_page"][0]) > 1:
            page = int(q["page"][0])
            total = min(body["total_count"], RESULT_CAP)
            if page * int(q["per_page"][0]) < total:
                nxt = path.replace(f"&page={page}&", f"&page={page + 1}&")
                link = f'<{nxt}>; rel="next"'
        return Response(200, {"x-ratelimit-remaining": "4000", "link": link}, body)

    def body(self, path: str) -> dict:
        self.calls.append(path)
        url = urlparse(path)
        if "/attempts/" in url.path:
            run_id = int(url.path.split("/runs/")[1].split("/")[0])
            run = next(r for r in self.runs if r["id"] == run_id)
            return {**run, "run_attempt": 1, "conclusion": "failure",
                    "updated_at": iso(datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) + timedelta(minutes=5))}
        q = parse_qs(url.query)
        lo, hi = q["created"][0].split("..")
        lo_t = datetime.fromisoformat(lo.replace("Z", "+00:00"))
        hi_t = datetime.fromisoformat(hi.replace("Z", "+00:00"))
        match = [r for r in self.runs
                 if lo_t <= datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) <= hi_t]
        page, per_page = int(q.get("page", ["1"])[0]), int(q["per_page"][0])
        start = (page - 1) * per_page
        visible = match[:RESULT_CAP]  # the real API silently truncates here
        return {"total_count": len(match), "workflow_runs": visible[start:start + per_page]}


def fetch(api, start, end, budget=None):
    return fetch_verdicts(Client(api, budget=budget, sleep=lambda s: None), "o/r", start, end)


class Fetch(unittest.TestCase):
    def test_every_run_is_fetched_even_when_a_window_exceeds_the_cap(self):
        api = FakeApi({0: 700, 1: 900, 3: 1500})  # day 3 alone exceeds the cap
        verdicts = fetch(api, DAY0, DAY0 + timedelta(days=5))
        self.assertEqual(len({v.run_id for v in verdicts}), 3100)

    def test_a_rerun_contributes_its_earlier_attempt_as_a_separate_verdict(self):
        api = FakeApi({0: 14})  # ids 7 and 14 were re-run
        verdicts = fetch(api, DAY0, DAY0 + timedelta(days=1))
        rerun = sorted((v.run_id, v.attempt, v.conclusion) for v in verdicts if v.run_id in (7, 14))
        self.assertEqual(rerun, [(7, 1, "failure"), (7, 2, "success"), (14, 1, "failure"), (14, 2, "success")])
        self.assertEqual(sum("/attempts/" in c for c in api.calls), 2)

    def test_workflows_are_keyed_by_repo_so_two_repos_never_merge(self):
        api = FakeApi({0: 3})
        verdicts = fetch_verdicts(Client(api), "chrishayuk/larql", DAY0, DAY0 + timedelta(days=1))
        self.assertTrue(all(v.workflow.startswith("chrishayuk/larql:") for v in verdicts))

    def test_a_fetch_that_loses_runs_is_an_error_not_a_smaller_sample(self):
        class Lossy(FakeApi):
            def __call__(self, path, headers):
                response = super().__call__(path, headers)
                body = response.body
                if "per_page=100" in path and body["workflow_runs"]:
                    body = {**body, "workflow_runs": body["workflow_runs"][1:]}  # drop one per page
                return Response(response.status, response.headers, body)
        with self.assertRaisesRegex(RuntimeError, "fetched 3 of 4"):
            fetch(Lossy({0: 4}), DAY0, DAY0 + timedelta(days=1))

    def test_the_page_plan_is_reserved_before_any_page_is_fetched(self):
        api = FakeApi({0: 900, 1: 900})  # 2 probes + 1 split... then 18 pages
        with self.assertRaises(BudgetExceeded):
            fetch(api, DAY0, DAY0 + timedelta(days=2), budget=10)
        self.assertFalse(any("per_page=100" in c for c in api.calls), "no page fetched past the budget")

    def test_the_plan_costs_only_probes_and_states_the_full_request_count(self):
        api = FakeApi({0: 700, 1: 900, 3: 1500})
        c = Client(api, sleep=lambda s: None)
        result = plan(c, "o/r", DAY0, DAY0 + timedelta(days=5))
        self.assertEqual(result["runs"], 3100)
        self.assertEqual(result["page_requests"], 33)  # windows of 700+900 split, 1500 split
        self.assertEqual(result["probe_requests"], c.counted_requests)
        self.assertFalse(any("per_page=100" in call for call in api.calls))


if __name__ == "__main__":
    unittest.main()
