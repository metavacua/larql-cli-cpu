"""`python3 -m covgate check|mutation|lint|selftest` — run from `scripts/`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cell as cell_mod
from . import checks, diff, lint, mutation, ratchet, report, testspans
from .checks import Finding
from .policy import parse as parse_policy


def emit(text: str, summary: Path | None) -> None:
    print(text)
    if summary is not None:
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def run_check(root: Path, cells_dir: Path, crate_dir: str, base: str | None) -> tuple[list, list[Finding]]:
    policy_rel = f"{crate_dir}/coverage-policy.json"
    raw = json.loads((root / policy_rel).read_text(encoding="utf-8"))
    policy = parse_policy(raw, crate_dir)
    spans = testspans.find(crate_dir, root)
    cells = [cell_mod.load(d, crate_dir, spans) for d in sorted(cells_dir.iterdir()) if d.is_dir()]
    added = diff.added_lines(base, crate_dir, root) if base else None
    findings = checks.run_all(cells, policy, added)
    if base:
        findings += ratchet.compare_policies(ratchet.base_policy(base, policy_rel, root), raw)
        findings += ratchet.suppression(base, crate_dir, root)
    return cells, findings


def cmd_check(args: argparse.Namespace) -> int:
    cells, findings = run_check(args.repo_root.resolve(), args.cells, args.crate_dir, args.base)
    emit(report.markdown(cells, findings), args.summary)
    args.json.write_text(json.dumps(report.as_json(cells, findings), indent=1), encoding="utf-8")
    return 1 if findings else 0


def cmd_mutation(args: argparse.Namespace) -> int:
    raw = json.loads((args.repo_root / args.crate_dir / "coverage-policy.json").read_text(encoding="utf-8"))
    pol = parse_policy(raw, args.crate_dir)
    findings, tally = mutation.check(args.outcomes, args.crate_dir, pol.mutation_min, pol.mutation_confidence)
    lines = ["## Mutation kill rate", "", "| file | caught | missed | timeout | unviable |", "|---|---|---|---|---|"]
    for path, counts in sorted(tally.items()):
        lines.append(
            f"| `{path}` | {counts.get('CaughtMutant', 0)} | {counts.get('MissedMutant', 0)} "
            f"| {counts.get('Timeout', 0)} | {counts.get('Unviable', 0)} |"
        )
    lines += ["", *(f"- **{f.subject}**: {f.message}" for f in findings)]
    emit("\n".join(lines), args.summary)
    return 1 if findings else 0


def cmd_lint(args: argparse.Namespace) -> int:
    findings, total = lint.ratchet(args.base, args.crate_dir, args.repo_root.resolve())
    lines = [f"## Test assertion lint", "", f"{total} test(s) in {args.crate_dir} check nothing (backlog)."]
    lines += [f"- new: `{f.subject}`" for f in findings]
    emit("\n".join(lines), args.summary)
    return 1 if findings else 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from . import selftest

    failures = selftest.run(args.repo_root.resolve())
    emit("## covgate self-test\n\n" + ("\n".join(f"- {f}" for f in failures) or "Known-clean passes; every planted defect is caught."), args.summary)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="covgate")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, crate: bool = True) -> None:
        if crate:
            p.add_argument("--crate-dir", required=True, help="e.g. crates/larql-cli")
        p.add_argument("--repo-root", type=Path, default=Path.cwd())
        p.add_argument("--summary", type=Path, help="append the Markdown report here")

    check = sub.add_parser("check", help="decide the coverage gate over every cell")
    common(check)
    check.add_argument("--cells", type=Path, required=True, help="one sub-directory per cell")
    check.add_argument("--base", help="git ref for the diff and ratchet rules (omit on main)")
    check.add_argument("--json", type=Path, default=Path("covgate-report.json"))
    check.set_defaults(run=cmd_check)

    mut = sub.add_parser("mutation", help="certify cargo-mutants kill rate per file")
    common(mut)
    mut.add_argument("outcomes", type=Path, help="mutants.out/outcomes.json")
    mut.set_defaults(run=cmd_mutation)

    lin = sub.add_parser("lint", help="no new test that checks nothing")
    common(lin)
    lin.add_argument("--base", help="git ref to ratchet against")
    lin.set_defaults(run=cmd_lint)

    st = sub.add_parser("selftest", help="known-clean must pass, planted defects must fail")
    common(st, crate=False)
    st.set_defaults(run=cmd_selftest)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
