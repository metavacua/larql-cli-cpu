"""Conformance oracle for the lql-strategy-matrix. Evaluates principled,
no-expected-output invariants over already-captured artifacts and emits a
report. Default exit 0 (discovery preserved); non-zero only under --strict."""
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Leg:
    name: str
    meta: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)       # id -> row
    descriptor: dict = field(default_factory=dict)
    produce: dict = field(default_factory=dict)


@dataclass
class Violation:
    invariant: str
    leg: str
    cell: str
    detail: str


def _read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load(results_glob):
    legs = {}
    for rf in sorted(glob.glob(results_glob)):
        d = Path(rf).parent
        names_here = []
        try:
            text = Path(rf).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                continue
            name = r.get("level")
            if not name:
                continue
            lg = legs.setdefault(name, Leg(name=name))
            if name not in names_here:
                names_here.append(name)
            if r.get("type") == "meta":
                lg.meta = r
            else:
                lg.cells[r.get("id", "?")] = r
        # sidecars: only for legs introduced by THIS results file, from THIS dir
        for name in names_here:
            lg = legs[name]
            dp = d / f"descriptor-{name}.json"
            pp = d / f"produce-{name}.json"
            ep = d / f"produce-{name}.err"
            if dp.exists():
                lg.descriptor = _read_json(dp)
            if pp.exists():
                lg.produce = _read_json(pp)
            if ep.exists():
                try:
                    lg.produce["stderr_head"] = ep.read_text(
                        encoding="utf-8", errors="replace")[:800]
                except OSError:
                    pass
    return legs


_FEAT = re.compile(r"\(\s*\d+\s+layers?,\s*([\d.]+)\s*([KM]?)\s+features", re.I)


def feature_count(leg):
    for row in leg.cells.values():
        m = _FEAT.search(row.get("stdout_head", "") or "")
        if m:
            try:
                n = float(m.group(1))
            except ValueError:
                continue
            n *= {"K": 1e3, "M": 1e6, "": 1}[m.group(2).upper()]
            return int(round(n))
    return None


def inv_completeness(legs):
    out = []
    for name, lg in legs.items():
        if _produce_failed(lg):
            continue  # vindex was never produced → inv_produce, not a hollow vindex
        if lg.descriptor.get("generation") == "v3":
            # V2's STATS "N layers, X features" banner has no V3 analog (a V3
            # catalogue has no single feature count); completeness for V3 is
            # read from the descriptor itself, which already ran.
            if not lg.descriptor.get("has_model_weights"):
                out.append(Violation("completeness", name, "",
                                     "hollow v3 container: 0 representations (cf #183, v3 analog)"))
            continue
        fc = feature_count(lg)
        if fc is None:
            out.append(Violation("completeness", name, "",
                                 "feature_count unknown (no STATS/banner — produce may have failed)"))
        elif fc == 0:
            out.append(Violation("completeness", name, "",
                                 "hollow vindex: 0 features (silent-incomplete extraction, cf #183)"))
    return out


_LAYER_ROW = re.compile(r"^L\d+\s+([\d.]+[KM]?)\s+[\d.]+[KM]?", re.M)

_CRASH_CODES = {101, 134, 137, 139}

_WARN = re.compile(r"warn|overrid|ignor", re.I)


def _is_crash(row):
    return row.get("bucket") == "crash" or row.get("exit_code") in _CRASH_CODES


def _produce_failed(lg):
    """True iff this leg's vindex was not successfully produced — the definitive
    signal (descriptor.produced=False, or a non-zero/err/timeout produce). Distinct
    from a produced-but-hollow vindex (feat=0), which is a completeness violation."""
    if lg.descriptor.get("produced") is False:
        return True
    p = lg.produce or {}
    ec = p.get("exit_code")
    if isinstance(ec, int) and ec != 0:
        return True
    return p.get("bucket") in {"err", "crash", "timeout"}


def inv_no_crash(legs):
    out = []
    for name, lg in legs.items():
        if lg.produce and _is_crash(lg.produce):
            detail = f"produce crashed (exit {lg.produce.get('exit_code')})"
            stderr_head = lg.produce.get("stderr_head")
            if stderr_head:
                detail += f": {stderr_head[:200]!r}"
            out.append(Violation("no-crash", name, "produce", detail))
        for cid, row in lg.cells.items():
            if _is_crash(row):
                detail = f"panic/crash (exit {row.get('exit_code')})"
                el = row.get("err_line")
                if el:
                    detail += f": {el}"
                out.append(Violation("no-crash", name, cid, detail))
    return out


def inv_produce(legs):
    """Definitive produce-failure violation: the vindex was never created. Produce
    *crashes* (SIGKILL/panic) are already reported by inv_no_crash, so skip them here
    to avoid double-counting; this catches the non-crash failures (exit!=0 / err /
    timeout / descriptor.produced=False) that would otherwise be mislabeled as a
    hedged completeness-unknown."""
    out = []
    for name, lg in legs.items():
        if _is_crash(lg.produce):
            continue
        if _produce_failed(lg):
            p = lg.produce or {}
            detail = (f"produce failed: op={p.get('op')} exit={p.get('exit_code')} "
                     f"bucket={p.get('bucket')} (vindex not created; "
                     f"descriptor.produced={lg.descriptor.get('produced')})")
            stderr_head = p.get("stderr_head")
            if stderr_head:
                detail += f" stderr: {stderr_head[:200]!r}"
            out.append(Violation("produce", name, "produce", detail))
    return out


def inv_descriptor_match(legs):
    out = []
    for name, lg in legs.items():
        d = lg.descriptor
        if not d or not d.get("produced", True) or d.get("error"):
            continue  # produce-side failure — covered by completeness/no-crash
        if not d.get("quant_match", True):
            out.append(Violation("descriptor-match", name, "",
                                 f"quant mismatch: produced {d.get('observed_quant')} != "
                                 f"expected {d.get('expect_quant')}"))
        fam = (d.get("family") or "").lower()
        if fam in ("", "generic"):
            out.append(Violation("descriptor-match", name, "",
                                 f"unrecognized/generic arch fallback (family={d.get('family')!r}, cf #154)"))
    return out


def _parse_num(tok):
    m = re.fullmatch(r"([\d.]+)([KM]?)", tok, re.I)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    return n * {"K": 1e3, "M": 1e6, "": 1}[m.group(2).upper()]


def show_layers_total(leg):
    row = leg.cells.get("show.layers")
    if not row:
        return None
    toks = _LAYER_ROW.findall(row.get("stdout_head", "") or "")
    if not toks:
        return None
    total, seen = 0.0, False
    for t in toks:
        v = _parse_num(t)
        if v is not None:
            total += v
            seen = True
    return int(round(total)) if seen else None


def inv_cross_check(legs):
    out = []
    for name, lg in legs.items():
        sl = show_layers_total(lg)
        fc = feature_count(lg)
        if sl is not None and fc and sl == 0:
            out.append(Violation("cross-check", name, "show.layers",
                                 f"SHOW LAYERS reports 0 features but STATS reports {fc} "
                                 "(mmap heap-only accessor, cf SHOW LAYERS bug)"))
    return out


def inv_diagnostic(legs):
    out = []
    for name, lg in legs.items():
        p = lg.produce
        # R-2: --quant q4k silently overrides a non-all --level
        if (p.get("op") == "extract" and "--quant q4k" in (p.get("flags") or "")
                and p.get("level") in {"browse", "attention", "inference"}):
            txt = (p.get("stdout_head", "") or "") + (p.get("stderr_head", "") or "")
            if not _WARN.search(txt):
                out.append(Violation("diagnostic", name, "produce",
                                     f"--level {p.get('level')} silently ignored under --quant q4k "
                                     "(no warn/override text; cf #208)"))
        # R-3: error text blames --compact but the recipe never passed it
        flags = (p.get("flags") or "")
        if "compact" not in flags:
            for cid, row in lg.cells.items():
                if "--compact" in (row.get("err_line", "") or ""):
                    out.append(Violation("diagnostic", name, cid,
                                         "error text misattributes to `--compact` (not in recipe flags)"))
                    break
    return out


INVARIANTS = [inv_completeness, inv_produce, inv_no_crash, inv_descriptor_match, inv_cross_check,
              inv_diagnostic]


# Per-process / per-leg noise stripped from a violation's detail to form its
# root-cause key. Exit codes, file and line numbers are kept - they are what
# distinguishes one root cause from another.
#  - thread id: `thread 'larql-main' (3135) panicked` and the older id-less banner
#    are the same fault. Inside repr(stderr_head) holding both quote kinds the
#    quotes are escaped (`thread \'larql-main\' (3135)`), so tolerate backslashes.
#  - the leg's own vindex path (out/<leg>.vindex), mktemp-style /tmp/<random> roots,
#    and the leg name itself, all of which differ per leg for one and the same fault.
_THREAD_ID = re.compile(r"(thread \\*'[^'\\]*\\*') \(\d+\)")
_VINDEX_PATH = re.compile(r"out/[^\s'\"\\/]+\.vindex")
_TMP_PATH = re.compile(r"/tmp/[^\s'\"\\/:]+")
# A detail that is only the exit-code stub carries no message or location, so it
# cannot identify a cause; the cell id has to.
_EXIT_STUB = re.compile(r"(?:panic/crash|produce crashed) \(exit \d+\)")

ROOT_CAUSE_LEG_NAMES = 3
_EXAMPLE_CLIP = 240


def normalise_detail(detail, leg=""):
    d = _THREAD_ID.sub(r"\1", detail)
    d = _VINDEX_PATH.sub("out/<leg>.vindex", d)
    d = _TMP_PATH.sub("/tmp/<tmp>", d)
    if leg:
        d = re.sub(r"(?<![\w.\-])" + re.escape(leg) + r"(?![\w\-]|\.\w)", "<leg>", d)
    return d


def _cause_key(v):
    d = normalise_detail(v.detail, v.leg)
    if v.cell and _EXIT_STUB.fullmatch(d):
        d = f"{d} [cell {v.cell}]"
    return v.invariant, d


def root_causes(violations):
    """Group violations by (invariant, normalised detail). Leg is not part of the
    key; the cell is only for bare exit-code stubs. Reporting only: the violations
    list itself is untouched. `cells` counts distinct non-empty cell ids
    (leg-level violations have none)."""
    groups = {}
    for v in violations:
        groups.setdefault(_cause_key(v), []).append(v)
    out = []
    for (inv, detail), vs in groups.items():
        legs = list(dict.fromkeys(v.leg for v in vs))
        out.append({
            "invariant": inv,
            "detail": detail,
            "violations": len(vs),
            "legs": len(legs),
            "cells": len({v.cell for v in vs if v.cell}),
            "example": vs[0].detail[:_EXAMPLE_CLIP],
            "leg_names": legs[:ROOT_CAUSE_LEG_NAMES],
        })
    out.sort(key=lambda r: (-r["violations"], r["invariant"], r["detail"]))
    return out


def run(results_glob, out_md, out_json, strict):
    legs = load(results_glob)
    violations = [v for inv in INVARIANTS for v in inv(legs)]
    Path(out_json).write_text(json.dumps(
        {"violations": [v.__dict__ for v in violations],
         "root_causes": root_causes(violations)}, indent=2), encoding="utf-8")
    Path(out_md).write_text(render(legs, violations), encoding="utf-8")
    print(f"conformance: {len(violations)} violation(s) across {len(legs)} legs "
          f"(strict={strict})")
    return 1 if (strict and violations) else 0


def render(legs, violations):
    L = ["# LQL Matrix — Conformance", "",
         f"Legs: {len(legs)} · Violations: {len(violations)}", ""]
    if violations:
        L += ["| invariant | leg | cell | detail |", "|---|---|---|---|"]
        for v in violations:
            L.append(f"| {v.invariant} | `{v.leg}` | {v.cell or '-'} | {v.detail} |")
    else:
        L.append("No invariant violations.")
    L += ["", "## Root causes", ""]
    rcs = root_causes(violations)
    if rcs:
        L += ["| invariant | violations | legs | cells | example | first legs |",
              "|---|---|---|---|---|---|"]
        for r in rcs:
            ex = r["example"].replace("|", "\\|")
            names = ", ".join(f"`{n}`" for n in r["leg_names"])
            L.append(f"| {r['invariant']} | {r['violations']} | {r['legs']} | "
                     f"{r['cells']} | {ex} | {names} |")
    else:
        L.append("No root causes.")
    return "\n".join(L) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/results-*.jsonl"
    out_md = args[1] if len(args) > 1 else "conformance.md"
    out_json = args[2] if len(args) > 2 else "conformance.json"
    sys.exit(run(results_glob, out_md, out_json, strict))


if __name__ == "__main__":
    main()
