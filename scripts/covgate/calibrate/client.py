"""A GitHub REST client that follows GitHub's published rules for API use
(docs.github.com: "Best practices for using the REST API" and "Rate
limits for the REST API"; Acceptable Use Policies section 7).

- requests are made serially, one client, never concurrently;
- a 403/429 with x-ratelimit-remaining: 0 waits until x-ratelimit-reset;
  a `retry-after` is honoured; a secondary limit without either waits a
  minute, doubling each time; limiting that persists past `max_retries`
  STOPS the fetch (continuing while limited risks an integration ban);
- other 4xx/5xx raise immediately rather than being retried;
- GETs are conditional: ETags are cached on disk and a 304 reuses the
  cached body (304s do not count against the primary limit);
- pages are followed through the `Link` header, never constructed;
- a request budget is declared up front and enforced: `reserve` refuses a
  plan that would exceed it before any request is spent.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIRST_BACKOFF_SECONDS = 60.0
_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: Any


Transport = Callable[[str, dict[str, str]], Response]


class RateLimited(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def gh_transport(path: str, headers: dict[str, str]) -> Response:
    """`gh api -i`: gh supplies authentication; status and headers are parsed
    from its output (gh exits non-zero on 4xx/5xx, which is still a response)."""
    args = ["gh", "api", "-i", "--method", "GET", path]
    for name, value in headers.items():
        args += ["-H", f"{name}: {value}"]
    out = subprocess.run(args, capture_output=True, text=True, check=False).stdout
    head, _, body = out.replace("\r\n", "\n").partition("\n\n")
    lines = head.splitlines()
    if not lines or not lines[0].startswith("HTTP/"):
        raise RuntimeError(f"unparseable response for {path}: {out[:200]!r}")
    status = int(lines[0].split()[1])
    parsed = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:]) if k}
    return Response(status, parsed, json.loads(body) if body.strip() else None)


class Client:
    def __init__(
        self,
        transport: Transport = gh_transport,
        *,
        cache_dir: Path | None = None,
        budget: int | None = None,
        max_retries: int = 6,
        max_wait: float = 3600.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ):
        self.transport, self.cache_dir, self.budget = transport, cache_dir, budget
        self.max_retries, self.max_wait, self.sleep, self.now = max_retries, max_wait, sleep, now
        self.counted_requests = 0
        self.reserved = 0
        self.remaining: int | None = None

    # -- budget -------------------------------------------------------------
    def reserve(self, requests: int) -> None:
        """Commit to spending `requests` more; refused if over budget."""
        if self.budget is not None and self.counted_requests + self.reserved + requests > self.budget:
            raise BudgetExceeded(
                f"plan needs {requests} more request(s); spent {self.counted_requests}, "
                f"reserved {self.reserved}, budget {self.budget}"
            )
        self.reserved += requests

    def _spend(self) -> None:
        if self.budget is not None and self.counted_requests >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} request(s) spent")
        self.counted_requests += 1
        self.reserved = max(0, self.reserved - 1)

    # -- conditional-request cache -------------------------------------------
    def _cache_file(self, path: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha256(path.encode()).hexdigest() + ".json")

    def _cached(self, path: str) -> dict | None:
        file = self._cache_file(path)
        return json.loads(file.read_text()) if file is not None and file.exists() else None

    def _store(self, path: str, response: Response) -> None:
        file = self._cache_file(path)
        etag = response.headers.get("etag")
        if file is not None and etag:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps({"etag": etag, "headers": response.headers, "body": response.body}))

    # -- requests -------------------------------------------------------------
    def get(self, path: str) -> Response:
        cached = self._cached(path)
        headers = {"If-None-Match": cached["etag"]} if cached else {}
        backoff = FIRST_BACKOFF_SECONDS
        for _ in range(self.max_retries + 1):
            response = self.transport(path, headers)
            if "x-ratelimit-remaining" in response.headers:
                self.remaining = int(response.headers["x-ratelimit-remaining"])
            if response.status == 304 and cached:
                return Response(200, cached["headers"], cached["body"])
            self._spend()
            if 200 <= response.status < 300:
                self._store(path, response)
                return response
            wait = self._rate_limit_wait(response, backoff)
            if wait is None:
                raise RuntimeError(f"GET {path}: HTTP {response.status} {response.body!r:.200}")
            if wait > self.max_wait:
                raise RateLimited(f"GET {path}: limit asks for a {wait:.0f}s wait, over the "
                                  f"declared maximum of {self.max_wait:.0f}s; stopping")
            if wait == backoff:
                backoff *= 2
            self.sleep(wait)
        raise RateLimited(f"GET {path}: still rate limited after {self.max_retries} retries; stopping")

    def _rate_limit_wait(self, response: Response, backoff: float) -> float | None:
        if response.status not in (403, 429):
            return None
        h = response.headers
        if h.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in h:
            return max(0.0, float(h["x-ratelimit-reset"]) - self.now())
        if "retry-after" in h:
            return float(h["retry-after"])
        message = json.dumps(response.body or {}).lower()
        if response.status == 429 or "rate limit" in message:
            return backoff
        return None  # a plain 403 (permissions) is not retried

    def paginate(self, path: str) -> Iterator[Response]:
        next_path: str | None = path
        while next_path is not None:
            response = self.get(next_path)
            yield response
            match = _LINK_NEXT.search(response.headers.get("link", ""))
            next_path = match.group(1) if match else None
