"""Tests for covgate.calibrate.client: GitHub's published REST API rules,
each enforced against a fake transport."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate.calibrate.client import BudgetExceeded, Client, RateLimited, Response  # noqa: E402

NOW = 1_800_000_000.0


class Scripted:
    """Returns queued responses in order and records each request."""

    def __init__(self, *responses: Response):
        self.queue = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, path: str, headers: dict) -> Response:
        self.requests.append((path, dict(headers)))
        return self.queue.pop(0)


def ok(body=None, **headers) -> Response:
    return Response(200, {"x-ratelimit-remaining": "4000", **headers}, body if body is not None else {})


class Clock:
    def __init__(self):
        self.t, self.slept = NOW, []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds

    def now(self) -> float:
        return self.t


def client(transport, **kw) -> tuple[Client, Clock]:
    clock = Clock()
    return Client(transport, sleep=clock.sleep, now=clock.now, **kw), clock


class RateLimits(unittest.TestCase):
    def test_primary_limit_waits_until_reset(self):
        limited = Response(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW + 90))}, {})
        c, clock = client(Scripted(limited, ok({"a": 1})))
        self.assertEqual(c.get("p").body, {"a": 1})
        self.assertEqual(clock.slept, [90.0])

    def test_retry_after_is_honoured(self):
        c, clock = client(Scripted(Response(429, {"retry-after": "30"}, {}), ok()))
        c.get("p")
        self.assertEqual(clock.slept, [30.0])

    def test_secondary_limit_without_headers_backs_off_exponentially_from_a_minute(self):
        secondary = Response(403, {}, {"message": "You have exceeded a secondary rate limit"})
        c, clock = client(Scripted(secondary, secondary, ok()))
        c.get("p")
        self.assertEqual(clock.slept, [60.0, 120.0])

    def test_persistent_limiting_stops_instead_of_hammering(self):
        secondary = Response(403, {}, {"message": "secondary rate limit"})
        transport = Scripted(*[secondary] * 10)
        c, _ = client(transport, max_retries=3)
        with self.assertRaises(RateLimited):
            c.get("p")
        self.assertEqual(len(transport.requests), 4)  # first try + 3 retries, then stop

    def test_other_client_errors_are_raised_not_retried(self):
        transport = Scripted(Response(404, {}, {"message": "Not Found"}))
        c, _ = client(transport)
        with self.assertRaises(RuntimeError):
            c.get("p")
        self.assertEqual(len(transport.requests), 1)

    def test_a_wait_longer_than_the_declared_maximum_stops_instead(self):
        limited = Response(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW + 3000))}, {})
        c, clock = client(Scripted(limited, ok()), max_wait=600)
        with self.assertRaisesRegex(RateLimited, "3000"):
            c.get("p")
        self.assertEqual(clock.slept, [])


class Conditional(unittest.TestCase):
    def test_a_304_reuses_the_cached_body_and_sends_the_etag(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Scripted(ok({"v": 1}, etag='"abc"'))
            client(first, cache_dir=Path(tmp))[0].get("p")
            second = Scripted(Response(304, {"x-ratelimit-remaining": "4000"}, None))
            c, _ = client(second, cache_dir=Path(tmp))
            self.assertEqual(c.get("p").body, {"v": 1})
            self.assertEqual(second.requests[0][1].get("If-None-Match"), '"abc"')
            self.assertEqual(c.counted_requests, 0)  # 304s do not count against the limit


class Pagination(unittest.TestCase):
    def test_pages_are_followed_by_link_header_never_constructed(self):
        transport = Scripted(
            ok([1], link='<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=3>; rel="last"'),
            ok([2], link='<https://api.github.com/x?page=3>; rel="next"'),
            ok([3]),
        )
        c, _ = client(transport)
        self.assertEqual([r.body for r in c.paginate("x")], [[1], [2], [3]])
        self.assertEqual([p for p, _ in transport.requests],
                         ["x", "https://api.github.com/x?page=2", "https://api.github.com/x?page=3"])


class Budget(unittest.TestCase):
    def test_spending_past_the_declared_budget_is_refused(self):
        c, _ = client(Scripted(ok(), ok(), ok()), budget=2)
        c.get("a")
        c.get("b")
        with self.assertRaises(BudgetExceeded):
            c.get("c")

    def test_a_plan_is_checked_against_the_budget_before_any_spending(self):
        c, _ = client(Scripted(), budget=100)
        with self.assertRaises(BudgetExceeded):
            c.reserve(150)


if __name__ == "__main__":
    unittest.main()
