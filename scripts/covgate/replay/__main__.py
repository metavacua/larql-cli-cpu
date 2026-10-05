"""`python3 -m covgate.replay plan|select|analyze` — run from `scripts/`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..calibrate.client import Client
from . import analyze, frame, pairing, plan


def cmd_plan(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prs = frame.merged_prs(args.repo_root.resolve(), args.ref, config["crate_dir"])
    sampled = frame.sample(prs, int(config["strata"]), int(config["seed"]))
    minutes = plan.check_budget(config, len(sampled))
    # Two requests per PR, under a budget: serial, ETag-cached, rule-abiding.
    client = Client(budget=2 * len(sampled))
    chosen = [frame.resolve(lambda p: client.get(p).body, config["source_repo"], p.number, p.merged_at)
              for p in sampled]
    out = {
        "config": config,
        "frame_size": len(prs),
        "estimated_runner_minutes": minutes,
        "prs": [p.__dict__ for p in chosen],
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"include": [{"pr": p.number, "side": s, "sha": getattr(p, s), "repo": p.repo}
                                  for p in chosen for s in ("base", "head")]}))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    base = json.loads(args.base_list.read_text(encoding="utf-8"))
    head = json.loads(args.head_list.read_text(encoding="utf-8"))
    pairs = pairing.pair(base, head)
    keys = pairing.sample(pairs, args.n, args.seed)
    selection = {json.dumps(k): (pairs[k][0]["name"], pairs[k][1]["name"]) for k in keys}
    args.out.write_text(json.dumps({
        "paired": len(pairs), "base_only": len(base) - len(pairs), "head_only": len(head) - len(pairs),
        "selection": selection,
    }, indent=1), encoding="utf-8")
    args.base_re.write_text("\n".join(pairing.rust_regex(b) for b, _ in selection.values()) + "\n")
    args.head_re.write_text("\n".join(pairing.rust_regex(h) for _, h in selection.values()) + "\n")
    for side, path in (("base", args.base_re), ("head", args.head_re)):
        index = 0 if side == "base" else 1
        path.with_suffix(".files").write_text(
            "\n".join(pairing.files([pair[index] for pair in selection.values()])) + "\n")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    planned = json.loads(args.plan.read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    for pr in planned["prs"]:
        number = str(pr["number"])
        sel_file = args.artifacts / f"select-{number}" / "selection.json"
        sides = {s: args.artifacts / f"run-{number}-{s}" for s in ("base", "head")}
        statuses = {s: json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"exit": None}
                    for s, d in sides.items()}
        if not sel_file.exists() or any(st["exit"] not in (0, 2, 3) for st in statuses.values()):
            # cargo-mutants exit 4 = baseline failed: the commit does not build or test here.
            results[number] = {"status": "baseline-failed" if any(st["exit"] == 4 for st in statuses.values()) else "incomplete",
                               "exits": {s: st["exit"] for s, st in statuses.items()}}
            continue
        selection = json.loads(sel_file.read_text())["selection"]
        outcomes = {s: json.loads((d / "outcomes.json").read_text()) for s, d in sides.items()}
        results[number] = {"status": "ok", **analyze.paired(selection, outcomes["base"], outcomes["head"]),
                           "rustc": {s: st.get("rustc") for s, st in statuses.items()}}
    report = {"plan": planned, "per_pr": results, "pilot": analyze.aggregate(results)}
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    text = "## Replay pilot\n\n```json\n" + json.dumps(report["pilot"], indent=1) + "\n```\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="covgate.replay")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="sample PRs from the frame and check the runner budget")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(".."))
    p.add_argument("--ref", default="HEAD")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_plan)

    s = sub.add_parser("select", help="pair base/head mutants and draw the sample")
    s.add_argument("--base-list", type=Path, required=True)
    s.add_argument("--head-list", type=Path, required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--base-re", type=Path, required=True)
    s.add_argument("--head-re", type=Path, required=True)
    s.set_defaults(run=cmd_select)

    a = sub.add_parser("analyze", help="paired results per PR and the pilot aggregate")
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--artifacts", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--summary", type=Path)
    a.set_defaults(run=cmd_analyze)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
