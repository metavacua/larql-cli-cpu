#!/usr/bin/env python3
"""Build the stratification matrices from per-target probe results + AST facts.

Inputs : DIR containing probe-*/result.json (stratify_probe.py), ast.json
         (stratify_ast.py), and a readable workspace (for cargo metadata).
Outputs: DIR/matrices/*.md and DIR/matrices/summary.json.

Matrices (all square ones are n x n over the same ordered set):
  1 crate x crate        workspace dependency adjacency (D direct, t transitive),
                         ordered by stratum, with lib/bin kind and AST facts
  2 target x target      agreement   - units with the same check outcome on both
  3 target x target      implication - units that pass on ROW but fail on COLUMN
                         (0 means "if it builds on ROW it builds on COLUMN")
  4 unit x target        cargo check outcome + cause class
  5 unit x target        cargo clippy outcome + warning / restriction-lint counts
  6 package x target     which package refused the target first, by cause class,
                         and the lowest-stratum unit that pulls it in
  7 unit x target        dependency-tree delta vs the baseline target
  8 stratification       per target: the first stratum that breaks; lib-xor-bin
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BASELINE = "x86_64-unknown-linux-gnu"
PASS, FAIL = "ok", "FAIL"
CLASS_ABBREV = {
    "std-absent": "std", "sysroot-absent": "sysroot", "native-lib-cross": "nativelib",
    "cc-missing": "cc", "platform-unsupported": "platform", "build-script": "build.rs",
    "source-error": "src", "<unattributed>": "?",
}


def col(r: dict) -> str:
    return r["target"] if r["mode"] == "strict" else f"{r['target']}~sysroot"


def short(t: str) -> str:
    return (t.replace("-unknown-", "-").replace("-none-elf", "-none").replace("-linux-gnu", "-gnu")
             .replace("riscv", "rv").replace("wasm32-", "w32-"))


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def workspace_deps() -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    r = subprocess.run(["cargo", "metadata", "--no-deps", "--locked", "--format-version", "1"],
                       capture_output=True, text=True, check=True)
    pk = {p["name"]: p for p in json.loads(r.stdout)["packages"]}
    direct = {n: {d["name"] for d in p["dependencies"] if d["name"] in pk and d.get("kind") != "dev"} for n, p in pk.items()}
    kinds = {n: sorted({k for t in p["targets"] for k in t["kind"]}) for n, p in pk.items()}
    return direct, kinds


def closure_of(direct: dict[str, set[str]], n: str) -> set[str]:
    seen: set[str] = set()
    stack = list(direct[n])
    while stack:
        d = stack.pop()
        if d not in seen:
            seen.add(d)
            stack.extend(direct[d])
    return seen


def passed(cell: dict) -> bool:
    return cell["check"]["exit"] == 0


def main() -> int:
    root = Path(sys.argv[1])
    results = [json.loads(p.read_text()) for p in sorted(root.glob("probe-*/result.json"))]
    if not results:
        print("::error::no probe results found - refusing to emit an empty matrix", file=sys.stderr)
        return 1
    ast = json.loads((root / "ast.json").read_text()) if (root / "ast.json").exists() else {}
    by_col = {col(r): r for r in results}
    cols = sorted(by_col, key=lambda c: (c != BASELINE, c))
    units = sorted(results[0]["units"], key=lambda k: (results[0]["units"][k]["stratum"], k))
    stratum = {k: results[0]["units"][k]["stratum"] for k in units}
    direct, pkg_kinds = workspace_deps()
    md: dict[str, str] = {}

    # 1 ── crate x crate
    crates = sorted({k.split(":")[0] for k in units}, key=lambda c: (min(stratum[k] for k in units if k.startswith(c + ":")), c))
    clos = {c: closure_of(direct, c) for c in crates}
    rows = []
    for c in crates:
        # A crate with no AST hits at all is absent from ast.json: that is 0 facts, not unknown.
        a = ast.get(c, {"os_bound_total": 0, "cfg_target": 0, "unsafe": 0} if ast else {})
        rows.append([f"**{c}**", ",".join(sorted(set(pkg_kinds[c]) & {"lib", "bin", "cdylib", "proc-macro"})),
                     str(min(stratum[k] for k in units if k.startswith(c + ":"))),
                     *["D" if d in direct[c] else "t" if d in clos[c] else ("·" if d != c else "■") for d in crates],
                     "yes" if a.get("no_std_declared") else "no", str(a.get("os_bound_total", "?")),
                     str(a.get("cfg_target", "?")), str(a.get("unsafe", "?"))])
    md["1-crate-by-crate"] = ("Row depends on column: D direct, t transitive, ■ self. Dev-dependencies excluded.\n\n" +
        table(["crate", "kind", "stratum", *[c.replace("larql-", "") for c in crates], "no_std", "OS-bound std paths", "cfg(target)", "unsafe"], rows))

    # 2/3 ── target x target
    agree, impl = [], []
    for a in cols:
        ra, rb = [], []
        for b in cols:
            same = sum(passed(by_col[a]["units"][k]) == passed(by_col[b]["units"][k]) for k in units)
            pa_fb = sum(passed(by_col[a]["units"][k]) and not passed(by_col[b]["units"][k]) for k in units)
            ra.append(f"{same}/{len(units)}")
            rb.append("·" if a == b else str(pa_fb))
        agree.append([f"**{short(a)}**", *ra]); impl.append([f"**{short(a)}**", *rb])
    hdr = ["", *[short(c) for c in cols]]
    md["2-target-agreement"] = "Units with the same `cargo check` outcome on both targets.\n\n" + table(hdr, agree)
    md["3-target-implication"] = ("Cell = units that PASS on the row target but FAIL on the column target. "
                                  "0 means building on the row implies building on the column.\n\n" + table(hdr, impl))

    # 4/5 ── unit x target
    def check_cell(cell):
        if passed(cell):
            return PASS
        classes = sorted({CLASS_ABBREV.get(f["cls"], f["cls"]) for f in cell["check"]["failed"]})
        return "FAIL " + "/".join(classes)
    md["4-unit-by-target-check"] = table(["unit", "s", *[short(c) for c in cols]],
        [[f"{k}", str(stratum[k]), *[check_cell(by_col[c]["units"][k]) for c in cols]] for k in units])

    def clippy_cell(cell):
        cp = cell["clippy"]
        if "skipped" in cp:
            return "–"
        if cp["exit"]:
            return "FAIL"
        total = sum(cp["lints"].values())
        restr = sum(v for k, v in cp["lints"].items() if k in ("clippy::std_instead_of_core", "clippy::std_instead_of_alloc", "clippy::alloc_instead_of_core"))
        return f"ok w{total} r{restr}"
    md["5-unit-by-target-clippy"] = ("`–` = clippy not run because check failed. `wN` warnings, `rN` core/alloc/std restriction-lint hits.\n\n" +
        table(["unit", *[short(c) for c in cols]], [[k, *[clippy_cell(by_col[c]["units"][k]) for c in cols]] for k in units]))

    # 6 ── package x target (who refuses the target, and who pulls it in)
    refusers: dict[str, dict[str, str]] = defaultdict(dict)
    firsterr: dict[str, str] = {}
    for c in cols:
        for k in units:
            for f in by_col[c]["units"][k]["check"]["failed"]:
                refusers[f["pkg"]].setdefault(c, CLASS_ABBREV.get(f["cls"], f["cls"]))
                firsterr.setdefault(f["pkg"], f["first"].splitlines()[0] if f["first"] else "")
    def introduced_by(pkg: str, c: str) -> str:
        name = pkg.split("@")[0]
        for k in units:  # units are stratum-ordered
            if any(p.split("@")[0] == name for p in by_col[c]["units"][k]["closure"]["pkgs"]):
                return k
        return "?"
    prow = []
    for pkg in sorted(refusers, key=lambda p: (-len(refusers[p]), p)):
        first_t = next(iter(refusers[pkg]))
        prow.append([f"`{pkg}`", str(len(refusers[pkg])), introduced_by(pkg, first_t),
                     *[refusers[pkg].get(c, "") for c in cols], firsterr[pkg][:70].replace("|", "/")])
    md["6-package-by-target"] = ("Packages that failed to compile, by target. `introduced by` = lowest-stratum unit whose "
        "dependency closure contains the package.\n\n" + table(["package", "#targets", "introduced by", *[short(c) for c in cols], "first error"], prow))

    # 7 ── dependency-tree delta vs baseline
    base = by_col.get(BASELINE)
    if base:
        drow = []
        for k in units:
            bset = set(base["units"][k]["closure"]["pkgs"])
            cells = []
            for c in cols:
                s = set(by_col[c]["units"][k]["closure"]["pkgs"])
                cells.append("=" if s == bset else f"-{len(bset - s)}/+{len(s - bset)}")
            drow.append([k, str(len(bset)), *cells])
        md["7-dependency-tree-delta"] = (f"Per unit, packages in the cfg-resolved closure vs `{BASELINE}`: "
            "`-n` packages not needed on that target, `+n` needed only there.\n\n" + table(["unit", "baseline pkgs", *[short(c) for c in cols]], drow))

    # 8 ── stratification
    srows, strata_ids = [], sorted(set(stratum.values()))
    for c in cols:
        cells, frontier = [], None
        for s in strata_ids:
            ks = [k for k in units if stratum[k] == s]
            ok = sum(passed(by_col[c]["units"][k]) for k in ks)
            cells.append(f"{ok}/{len(ks)}")
            if frontier is None and ok < len(ks):
                frontier = s
        srows.append([short(c), *cells, "all pass" if frontier is None else f"breaks at s{frontier}"])
    xor = [n for n in crates if {"lib", "bin"} <= {"lib" if set(pkg_kinds[n]) & {"lib", "rlib", "cdylib", "dylib", "staticlib", "proc-macro"} else "", "bin" if "bin" in pkg_kinds[n] else ""}]
    md["8-stratification"] = ("Units passing `cargo check` per stratum (s0 = no workspace dependencies).\n\n" +
        table(["target", *[f"s{s}" for s in strata_ids], "frontier"], srows) +
        f"\n**lib XOR bin violations (package has both a lib and a bin target):** {', '.join(xor) or 'none'}\n")

    out = root / "matrices"
    out.mkdir(exist_ok=True)
    for name, body in md.items():
        (out / f"{name}.md").write_text(f"# {name}\n\n{body}")
    (out / "summary.json").write_text(json.dumps({"columns": cols, "units": units, "stratum": stratum}, indent=1))
    summary = "\n\n".join(f"## {n}\n\n{b}" for n, b in md.items())
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
