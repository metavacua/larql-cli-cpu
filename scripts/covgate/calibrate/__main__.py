"""`python3 -m covgate.calibrate runs|escapes|summary` — run from `scripts/`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import escapes as escapes_mod
from . import false_reds, runs
from .client import Client


def _date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def cmd_runs(args: argparse.Namespace) -> int:
    # One client for every repo: requests stay serial and share one budget.
    client = Client(cache_dir=args.cache, budget=args.budget, max_wait=args.max_wait)
    if args.plan_only:
        for repo in args.repo:
            print(json.dumps(runs.plan(client, repo, _date(args.since), _date(args.until))))
        print(f"probes spent: {client.counted_requests}; rate-limit remaining {client.remaining}", file=sys.stderr)
        return 0
    with args.out.open("w", encoding="utf-8") as out:
        for repo in args.repo:
            for verdict in runs.fetch_verdicts(client, repo, _date(args.since), _date(args.until)):
                out.write(json.dumps(dataclasses.asdict(verdict)) + "\n")
            print(f"{repo}: {client.counted_requests} request(s) spent so far, "
                  f"rate-limit remaining {client.remaining}", file=sys.stderr)
    return 0


def cmd_escapes(args: argparse.Namespace) -> int:
    found, changes = escapes_mod.escapes(args.repo_path.resolve(), args.ref, args.only)
    args.out.write_text(json.dumps({
        "repo": args.label, "ref": args.ref, "only": args.only, "changes": changes,
        "escapes": [dataclasses.asdict(e) for e in found],
    }), encoding="utf-8")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from . import summary

    report = summary.summarize(
        [json.loads(p.read_text(encoding="utf-8")) for p in args.escapes],
        [false_reds.Verdict(**json.loads(line)) for p in args.runs for line in p.read_text(encoding="utf-8").splitlines() if line],
    )
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="covgate.calibrate")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("runs", help="fetch every attempt of every completed CI run")
    r.add_argument("--repo", action="append", required=True, help="owner/name; repeatable")
    r.add_argument("--since", required=True, help="YYYY-MM-DD")
    r.add_argument("--until", required=True, help="YYYY-MM-DD")
    r.add_argument("--out", type=Path, required=True, help="JSON lines")
    r.add_argument("--cache", type=Path, required=True, help="ETag cache directory (re-runs cost ~nothing)")
    r.add_argument("--budget", type=int, required=True, help="maximum requests this run may spend")
    r.add_argument("--max-wait", type=float, required=True, help="longest rate-limit wait, seconds, before stopping")
    r.add_argument("--plan-only", action="store_true", help="spend only window probes; print the cost")
    r.set_defaults(run=cmd_runs)

    e = sub.add_parser("escapes", help="fix commits blamed back to the changes they repaired")
    e.add_argument("--repo-path", type=Path, required=True)
    e.add_argument("--label", required=True, help="owner/name the clone came from")
    e.add_argument("--ref", default="main")
    e.add_argument("--only", default="", help="restrict to paths under this prefix")
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(run=cmd_escapes)

    s = sub.add_parser("summary", help="cost vectors with intervals, per stakeholder")
    s.add_argument("--escapes", type=Path, action="append", default=[])
    s.add_argument("--runs", type=Path, action="append", default=[])
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(run=cmd_summary)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
