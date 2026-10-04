"""Cross-target structural comparison for the lql-strategy-matrix.

The same crate, built for different targets and run through the same legs, must agree on
the mechanically decidable structural facts below. A disagreement means something
target-specific in larql, or in the workflow, broke -- not that a model said something odd.
This script does NOT judge whether any model's answer is right.

An instance is one target tag (artifact directory results-<TAG>-<leg>/).

COMPARED, per leg, across instances:
  (a) the set of legs present
  (b) produce outcome class, ok vs failed (conformance._produce_failed)
  (c) descriptor fields: family, observed_quant, generation, produced, has_model_weights
  (d) feature_count, the integer from the STATS banner
  (e) per cell id: presence, bucket, exit_code, and err_signal (the in-band 'Error: ...' flag;
      'larql lql' exits 0 on in-band errors, so this is the dominant failure shape) -- but
      err_signal ONLY for cells whose 'cat' is not in FP_DEPENDENT_CATS
  (f) completion: a cell or produce that timed out / was hard-killed on one instance while a
      peer finished it normally (see the slow-tag rule)

NOT COMPARED, because a correct larql may legitimately make it differ: stdout/stderr text,
err_line text, any float, duration_ms, and err_signal of FP_DEPENDENT_CATS cells (their
verdict can flip on float rounding or convergence). Which model answer is right is out of scope.

SPEED-DEPENDENT OUTCOMES. A cell or produce whose exit_code is 124 (timeout) or 137
(SIGKILL after 'timeout --kill-after') or whose bucket is 'timeout' depends on speed or
memory, not on larql's logic, so it is never compared as a crash/bucket/exit_code fact and
its output (feature_count, descriptor, cells of that leg) is not used. inv_no_crash in
conformance.py still reports crashes per instance, so nothing is lost.

EXPECTED TAGS (--expect-tag TAG, repeatable). Instances are otherwise inferred from the artifacts
that exist, so a target whose whole lql run uploaded nothing would be invisible. An expected tag
with no artifacts is a conclusive disagreement (workflow breakage on that target), excused only
when it is also a slow tag. The CI job passes every target that runs lql.

SLOW-TAG RULE (--slow-tag TAG, repeatable, always explicit; nothing is inferred). A slow tag is
an emulated target (CI: riscv64gc-unknown-linux-gnu under QEMU user-mode). Only there is a
speed-dependent outcome, or a missing leg or cell, excused as 'inconclusive'. On a non-slow
(native) tag a timeout/137 where a peer instance finished the same cell or produce normally is
a CONCLUSIVE DISAGREEMENT (a genuine hang is a classic target-specific bug), and a leg or cell
missing from a non-slow instance that is present elsewhere is a conclusive disagreement. If
every instance timed out on the same cell they agree.

FEATURE_COUNT/DESCRIPTOR. An instance with a timed-out/hard-killed cell anywhere in that leg
has an inconclusive feature_count (the STATS banner is scanned over every cell); an instance
whose produce timed out/was hard-killed is inconclusive for descriptor, feature_count and cells.
The remaining instances of the leg are still compared with each other.

Fewer than two instances -> 'nothing to compare', exit 0 even under --strict. Otherwise
--strict exits 1 iff there is at least one conclusive disagreement. A results directory whose
name is not results-<TAG>-<leg> for a leg its rows declare, or two directories/files giving the
same (tag, leg), is an error: exit 2 with a message naming it.

Input: artifact directories results-<TAG>-<leg>/ (CI naming); TAG may contain hyphens and
the leg may contain dots. Reporting is the exit code, a markdown file (for the step summary)
and a JSON file; nothing here needs a token or any write permission.

Usage: cross_target.py "<glob of results jsonl>" [--strict] [--slow-tag TAG ...]
                       [--out-md FILE] [--out-json FILE]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conformance as C  # noqa: E402

DESCRIPTOR_FIELDS = ("family", "observed_quant", "generation", "produced", "has_model_weights")
PREFIX = "results-"

# exit codes of 'timeout --kill-after=10 ...' (run_matrix.py): 124 = timed out, 137 = still
# alive 10s later and SIGKILLed (bucketed 'crash'). Both depend on speed/memory, not on larql.
SPEED_EXIT_CODES = (124, 137)

# Corpus categories (commands.jsonl 'cat') whose in-band-error verdict can legitimately flip on
# float rounding, convergence or tolerance, so err_signal is NOT a valid cross-target fact:
#   inference  INFER/EXPLAIN/TRACE: forward-pass logits and confidence thresholds
#   roundtrip  INFER, then INSERT/COMPILE, then INFER again
#   mutation   INSERT captures residuals by forward pass; REBALANCE FLOOR/CEILING/CONVERGED
#              iterate on float scores
#   compact    COMPACT MAJOR WITH LAMBDA: float regularisation
#   compile    COMPILE bakes an INSERT (forward-pass residuals) into the weights
#   patch      patch cells include INSERT/SAVE of a forward-pass edge
# Every other category (browse, lifecycle, negative) is compared. This is deliberately
# conservative: adding to the compared set is an explicit edit here.
FP_DEPENDENT_CATS = frozenset(
    {"inference", "roundtrip", "mutation", "compact", "compile", "patch"})
# The categories whose err_signal IS compared. Together the two sets must cover every category in
# commands.jsonl (a test enforces it), so a new category forces a deliberate choice instead of
# being silently treated as deterministic.
DETERMINISTIC_CATS = frozenset({"browse", "lifecycle", "negative"})


class TagError(ValueError):
    """A results directory name that is not results-<TAG>-<leg> for a declared leg."""


class DuplicateError(ValueError):
    """Two sources give the same (tag, leg)."""


def parse_tag(dirname, leg_names):
    """results-<TAG>-<leg> -> TAG. The longest declared leg that is a suffix wins, so a leg
    that is itself a suffix of another does not eat part of the tag. Anything else raises."""
    if not dirname.startswith(PREFIX):
        raise TagError(f"results directory {dirname!r} does not start with {PREFIX!r}")
    rest = dirname[len(PREFIX):]
    for leg in sorted(leg_names, key=len, reverse=True):
        if leg and rest.endswith("-" + leg) and len(rest) > len(leg) + 1:
            return rest[: -(len(leg) + 1)]
    raise TagError(
        f"results directory {dirname!r} does not end with '-<leg>' for any leg its rows declare "
        f"({', '.join(sorted(leg_names)) or 'none'}); expected results-<TAG>-<leg>")


def load_instances(results_glob):
    """{tag: {leg name: conformance.Leg}}, each results file loaded on its own.
    Raises TagError on a bad directory name, DuplicateError on a repeated (tag, leg)."""
    instances, origin = {}, {}
    for rf in sorted(glob.glob(results_glob)):
        legs = C.load(glob.escape(rf))
        if not legs:
            continue
        dirname = Path(rf).parent.name
        tag = parse_tag(dirname, list(legs))
        for name, leg in legs.items():
            if name in instances.get(tag, {}):
                raise DuplicateError(
                    f"duplicate (tag, leg) = ({tag!r}, {name!r}): {origin[(tag, name)]} and {rf}")
            instances.setdefault(tag, {})[name] = leg
            origin[(tag, name)] = rf
    return instances


def _speed_dependent(row):
    """A timed-out or hard-killed cell/produce: its outcome depends on speed or memory."""
    row = row or {}
    return row.get("bucket") == "timeout" or row.get("exit_code") in SPEED_EXIT_CODES


_ABSENT = object()


def _show(v):
    return "(absent)" if v is _ABSENT else v


def _finding(leg, fact, status, values):
    return {"leg": leg, "fact": fact, "status": status,
            "values": {t: _show(v) for t, v in values.items()}}


def _distinct(values):
    return {json.dumps(_show(v), sort_keys=True, default=str) for v in values.values()}


def _decide(leg, fact, values, speed=(), slow=(), excused=()):
    """values: tag -> value. speed: tags whose observation timed out/was killed (shown as
    'timeout'); a non-slow one next to a tag that finished is a hang -> disagree, only slow
    ones -> inconclusive, all of them -> agree. excused: tags whose value is unusable
    (shown 'inconclusive'); they never disagree, but leave the fact inconclusive."""
    shown = dict(values)
    for t in speed:
        shown[t] = "timeout"
    for t in excused:
        shown[t] = "inconclusive"
    finished = {t: v for t, v in values.items() if t not in speed and t not in excused}
    if len(_distinct(finished)) > 1:
        status = "disagree"
    elif finished and [t for t in speed if t not in slow]:
        status = "disagree"
    elif excused or (speed and finished):
        status = "inconclusive"
    else:
        status = "agree"
    return _finding(leg, fact, status, shown)


def _decide_presence(leg, fact, present, slow):
    """A tag lacking something present elsewhere disagrees unless the tag is slow."""
    if len(set(present.values())) == 1:
        return _finding(leg, fact, "agree", present)
    missing = [t for t, v in present.items() if not v]
    status = "disagree" if [t for t in missing if t not in slow] else "inconclusive"
    return _finding(leg, fact, status, present)


def _compare_leg(name, legs, slow, out):
    """legs: tag -> Leg, at least two. Appends findings."""
    produce_speed = {t for t, lg in legs.items() if _speed_dependent(lg.produce)}
    outcome = {t: ("failed" if C._produce_failed(lg) else "ok") for t, lg in legs.items()}
    out.append(_decide(name, "produce outcome", outcome, speed=produce_speed, slow=slow))
    legs = {t: lg for t, lg in legs.items() if t not in produce_speed}
    if len(legs) < 2:
        out.append(_finding(name, "descriptor, feature_count and cells", "inconclusive",
                            {t: ("produce timed out" if t in produce_speed else "-")
                             for t in sorted(produce_speed | set(legs))}))
        return
    for field in DESCRIPTOR_FIELDS:
        out.append(_decide(name, f"descriptor {field}",
                           {t: lg.descriptor.get(field) for t, lg in legs.items()},
                           excused=produce_speed))
    cell_speed = {t for t, lg in legs.items()
                  if any(_speed_dependent(r) for r in lg.cells.values())}
    out.append(_decide(name, "feature_count",
                       {t: C.feature_count(lg) for t, lg in legs.items()},
                       excused=cell_speed | produce_speed))
    for cid in sorted({c for lg in legs.values() for c in lg.cells}):
        rows = {t: lg.cells.get(cid) for t, lg in legs.items()}
        out.append(_decide_presence(name, f"cell {cid} present",
                                    {t: r is not None for t, r in rows.items()}, slow))
        have = {t: r for t, r in rows.items() if r is not None}
        speed = {t for t, r in have.items() if _speed_dependent(r)}
        if speed:
            out.append(_decide(name, f"cell {cid}",
                               {t: "finished" for t in have}, speed=speed, slow=slow))
        done = {t: r for t, r in have.items() if t not in speed}
        if len(done) < 2:
            continue
        keys = ["bucket", "exit_code"]
        if not any(r.get("cat") in FP_DEPENDENT_CATS for r in done.values()):
            keys.append("err_signal")
        for key in keys:
            out.append(_decide(name, f"cell {cid} {key}", {t: r.get(key) for t, r in done.items()}))


def compare(instances, slow_tags=(), expect_tags=()):
    tags = sorted(instances)
    slow = frozenset(slow_tags)
    report = {"instances": tags, "slow_tags": sorted(slow), "expected_tags": sorted(set(expect_tags)),
              "findings": [], "disagreements": 0, "inconclusive": 0, "compared": len(tags) >= 2}
    out = report["findings"]
    # Instances are otherwise inferred from the artifacts that exist, so a target whose whole lql
    # run uploaded nothing would be invisible. An expected tag with no artifacts is workflow
    # breakage on that target, excused only when the target is slow (emulated).
    for t in sorted(set(expect_tags) - set(tags)):
        out.append(_finding("(instance)", "instance present", "inconclusive" if t in slow else "disagree",
                            {x: (x in instances or x != t) for x in sorted(set(expect_tags) | set(tags))}))
    if len(tags) < 2:
        report["disagreements"] = sum(f["status"] == "disagree" for f in out)
        report["inconclusive"] = sum(f["status"] == "inconclusive" for f in out)
        return report
    for name in sorted({n for legs in instances.values() for n in legs}):
        present = {t: (name in instances[t]) for t in tags}
        out.append(_decide_presence(name, "leg present", present, slow))
        legs = {t: instances[t][name] for t in tags if present[t]}
        if len(legs) >= 2:
            _compare_leg(name, legs, slow, out)
    report["disagreements"] = sum(f["status"] == "disagree" for f in out)
    report["inconclusive"] = sum(f["status"] == "inconclusive" for f in out)
    return report


def render(report):
    L = ["# LQL Matrix -- Cross-target structural comparison", ""]
    cols = sorted(set(report["instances"]) | set(report.get("expected_tags", [])))
    if not report["compared"] and not report["findings"]:
        n = len(report["instances"])
        L += [f"Nothing to compare: {n} instance(s) found "
              f"({', '.join(f'`{t}`' for t in report['instances']) or 'none'}); need at least two.", ""]
        return "\n".join(L)
    agree = sum(f["status"] == "agree" for f in report["findings"])
    L += [f"Instances: {', '.join(f'`{t}`' for t in report['instances'])}"
          + (f" (expected: {', '.join(f'`{t}`' for t in report['expected_tags'])})" if report.get("expected_tags") else ""), "",
          f"Agree: {agree} · Disagree: {report['disagreements']} · "
          f"Inconclusive (speed-dependent): {report['inconclusive']}", "",
          "Compared: legs present, produce outcome, descriptor family/quant/generation/produced/"
          "has_model_weights, feature_count, per-cell presence/bucket/exit_code, and err_signal "
          f"for cells outside {', '.join(sorted(FP_DEPENDENT_CATS))}. Not compared: output text, "
          "err_line, floats, durations, or anything that timed out (124/137). Slow tags "
          f"(excused when slow or missing): {', '.join(f'`{t}`' for t in report['slow_tags']) or 'none'}.",
          ""]
    for status, title in (("disagree", "Disagreements"), ("inconclusive", "Inconclusive")):
        rows = [f for f in report["findings"] if f["status"] == status]
        if not rows:
            continue
        L += [f"## {title}", "", "| leg | fact | " + " | ".join(cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for f in rows:
            vals = " | ".join(f"`{f['values'].get(t, '(absent)')}`" for t in cols)
            L.append(f"| `{f['leg']}` | {f['fact']} | {vals} |")
        L.append("")
    if not report["disagreements"]:
        L += ["No conclusive disagreement between targets.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_glob")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--slow-tag", action="append", default=[], metavar="TAG",
                    help="emulated (slow) target tag; repeatable. Only these may be excused "
                         "for timeouts, 124/137 and missing legs/cells.")
    ap.add_argument("--expect-tag", action="append", default=[], metavar="TAG",
                    help="a target that must have produced artifacts; repeatable. Without it an "
                         "instance whose whole run uploaded nothing is invisible. A missing expected "
                         "tag disagrees unless it is also a --slow-tag.")
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    args = ap.parse_args(argv)
    try:
        instances = load_instances(args.results_glob)
    except (TagError, DuplicateError) as e:
        print(f"cross_target: error: {e}", file=sys.stderr)
        return 2
    report = compare(instances, slow_tags=args.slow_tag, expect_tags=args.expect_tag)
    md = render(report)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(md)
    return 1 if (args.strict and report["disagreements"]) else 0


if __name__ == "__main__":
    sys.exit(main())
