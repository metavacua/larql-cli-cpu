#!/usr/bin/env python3
"""Assemble the stratification matrices from probe, AST and LSIF data.

Index sets (every axis label below is a member of one of these):
  C crates   U units (crate:lib|bin)   T columns (target|mode)
  Q cause classes   P failing packages   Z strata   F AST facts

Everything is a matrix and every relation is computed from other matrices:

  D   [C,C] bool   declared dependency, D[i,j]=1 iff i depends on j
  R   [C,C] bool   reachability = D + D^2 + ... (boolean semiring)
  s   [C,1] int    stratum = longest path from i; D nilpotent <=> acyclic
  DU  [U,U] bool   unit-level dependency (bin -> its own lib, crate edges -> libs)
  O   [U,T] bool   cargo check passes
  Ag  [T,T] int    O^T O + (1-O)^T (1-O)   units with the same outcome on a,b
  Imp [T,T] int    O^T (1-O)               units passing on a, failing on b
  V   [U,T] int    O * (DU (1-O))          passing units with a failing dep (must be 0)
  Bk  [P,T] bool   package p failed on target t
  Ex  [U,T] bool   failure is explained by a failing package in the unit's
                   closure (K_t Bk_t > 0) or by the unit's own crate
  Dead[C,C] bool   D * (N == 0): declared dependency never used (LSIF)

Output: <root>/matrices/{matrices.json,csv/,mtx/,matrices.npz,invariants.json}
Exit 1 if inputs are missing or the output fails structural validation;
data invariants (e.g. V == 0) are recorded in invariants.json, not hidden.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import stratify_matrix as sm

BASELINE_TARGET = "x86_64-unknown-linux-gnu"
RESTRICTION_LINTS = ("clippy::std_instead_of_core", "clippy::std_instead_of_alloc", "clippy::alloc_instead_of_core")
KIND_COLS = ("lib", "bin", "cdylib", "proc-macro")
AST_COLS = ("no_std_declared", "no_std_conditional", "cfg_target", "unsafe", "extern_c", "os_bound_total")
SCHEMA_PATH = Path(__file__).parent / "schemas" / "stratification-matrices.schema.json"
BASE_CLASSES = ("sysroot-absent", "std-absent", "native-lib-cross", "cc-missing", "platform-unsupported",
                "build-script", "source-error", "<unattributed>")


def col_label(r: dict) -> str:
    return f"{r['target']}|{r['mode']}"


def workspace() -> tuple[list[str], np.ndarray, dict[str, set[str]]]:
    r = subprocess.run(["cargo", "metadata", "--no-deps", "--locked", "--format-version", "1"],
                       capture_output=True, text=True, check=True)
    pk = {p["name"]: p for p in json.loads(r.stdout)["packages"]}
    names = sorted(pk)
    ix = {n: i for i, n in enumerate(names)}
    d = np.zeros((len(names), len(names)), dtype=np.int64)
    for n, p in pk.items():
        for dep in p["dependencies"]:
            if dep["name"] in ix and dep.get("kind") != "dev":
                d[ix[n], ix[dep["name"]]] = 1
    kinds = {n: {k for t in p["targets"] for k in t["kind"]} for n, p in pk.items()}
    return names, d, kinds


def pkg_name(p: str) -> str:
    return p.split("@")[0]


def main() -> int:
    root = Path(sys.argv[1])
    results = [json.loads(p.read_text()) for p in sorted(root.glob("probe-*/result.json"))]
    if not results:
        print(json.dumps({"error": "no probe results found"}), file=sys.stderr)
        return 1
    cols = sorted((col_label(r) for r in results), key=lambda c: (c != f"{BASELINE_TARGET}|strict", c))
    by_col = {col_label(r): r for r in results}
    first = results[0]["units"]
    units = sorted(first, key=lambda k: (first[k]["stratum"], k))
    nu, nt = len(units), len(cols)
    crate_of = [u.split(":")[0] for u in units]
    kind_of = [u.split(":")[1] for u in units]
    out: list[sm.Matrix] = []
    inv: list[dict] = []

    def add(name, rows, cs, data, semiring="counting", definition="", **meta):
        out.append(sm.Matrix(name, list(rows), list(cs), data, semiring, definition, meta))
        return out[-1].data

    def check(name: str, ok: bool, detail=None) -> None:
        inv.append({"name": name, "ok": bool(ok), "detail": detail})

    # ── crate level ────────────────────────────────────────────────────────
    crates, D, kinds = workspace()
    R, nilpotent = sm.closure(D)
    check("D_nilpotent_acyclic", nilpotent)
    s = sm.strata(D) if nilpotent else np.zeros(len(crates), dtype=np.int64)
    add("D", crates, crates, D, "boolean", "D[i,j]=1 iff crate i depends on crate j (normal+build)")
    add("R", crates, crates, R, "boolean", "R = D + D^2 + ... (boolean semiring)")
    add("stratum", crates, ["s"], s.reshape(-1, 1), "none", "s_i = longest dependency path from i")
    kinc = np.array([[int(k in kinds[c]) for k in KIND_COLS] for c in crates])
    add("kind_incidence", crates, KIND_COLS, kinc, "boolean", "cargo target kinds present in the package")
    add("lib_xor_bin_violation", crates, ["violates"], (kinc[:, 0] * kinc[:, 1]).reshape(-1, 1), "boolean",
        "lib * bin: package has both a library and a binary target")

    ast = json.loads((root / "ast.json").read_text()) if (root / "ast.json").exists() else None
    if ast is not None:
        from_ast = lambda c, k: int(ast.get(c, {}).get(k, 0))  # absent crate = 0 hits, not unknown
        mods = sorted({m for f in ast.values() for m in f["std_paths"]})
        fcols = [*AST_COLS, *[f"std::{m}" for m in mods]]
        fa = [[from_ast(c, k) for k in AST_COLS] + [int(ast.get(c, {}).get("std_paths", {}).get(m, 0)) for m in mods] for c in crates]
        add("ast_facts", crates, fcols, fa, "counting", "tree-sitter counts per crate, production code only")
        net = {m: n for f in ast.values() for m, n in f.get("net_modules", {}).items()}
        if net:
            roots_ = sorted({r for n in net.values() for r in n})
            mods_ = sorted(net)
            add("net_use", mods_, roots_, [[net[m].get(r, 0) for r in roots_] for m in mods_], "counting",
                "net_use[m,r] = paths rooted at network crate r in module m (production source, ast-grep)")
    use_path = root / "actual-use.json"
    if use_path.exists():
        pairs = json.loads(use_path.read_text())
        N = np.zeros_like(D)
        for key, n in pairs.items():
            u, d = key.split("->")
            if u in crates and d in crates:
                N[crates.index(u), crates.index(d)] = n
        add("N", crates, crates, N, "counting", "N[i,j] = reference ranges in crate i resolving to a definition in crate j (LSIF)")
        add("Dead", crates, crates, D * (N == 0), "boolean", "D * (N == 0): declared, never referenced")
        add("Undeclared", crates, crates, (N > 0) * (1 - R), "boolean", "(N > 0) * (1 - R): used without a declared path")

    # ── module level: network and IO, from the type-resolved reference graph ──
    # MB[i,j]=1 iff module i references a symbol defined in module j (rust-analyzer
    # LSIF). Seeds come from the AST: modules whose own code names a network (IO)
    # crate or std item. mockito alone is test scaffolding, not network behaviour.
    # A module is tainted if it is a seed or reaches one: taint = seed + R seed.
    mod_path = root / "module-use.json"
    if ast is not None and mod_path.exists():
        pairs = json.loads(mod_path.read_text())
        net_seed = {m for f in ast.values() for m, n in f.get("net_modules", {}).items() if set(n) - {"mockito"}}
        io_seed = {m for f in ast.values() for m, n in f.get("io_modules", {}).items() if n}
        mods = sorted({m for k in pairs for m in k.split("->")} | net_seed | io_seed)
        mi = {m: i for i, m in enumerate(mods)}
        MU = np.zeros((len(mods), len(mods)), dtype=np.int64)
        for k, n in pairs.items():
            u, d = k.split("->")
            MU[mi[u], mi[d]] = n
        MB = (MU > 0).astype(np.int64)
        MR, mod_acyclic = sm.closure(MB)
        vec = lambda s: np.array([[int(m in s)] for m in mods])
        ns, ios = vec(net_seed), vec(io_seed)
        net_taint = ((ns + sm.bool_matmul(MR, ns)) > 0).astype(np.int64)
        io_taint = ((ios + sm.bool_matmul(MR, ios)) > 0).astype(np.int64)
        add("module_ref", mods, mods, MU, "counting", "MU[i,j] = references from module i to symbols defined in module j (LSIF, production source)")
        add("module_net_seed", mods, ["net"], ns, "boolean", "module's own code uses a network crate or std::net")
        add("module_io_seed", mods, ["io"], ios, "boolean", "module's own code uses std fs/io/env/process/path or an IO crate")
        add("module_net_taint", mods, ["net"], net_taint, "boolean", "seed + R seed: the module is, or transitively depends on, network code")
        add("module_io_taint", mods, ["io"], io_taint, "boolean", "seed + R seed: the module is, or transitively depends on, IO code")
        add("module_net_callers", mods, ["net"], ((sm.bool_matmul(MB, ns) > 0) * (1 - ns)), "boolean",
            "modules that reference a network module directly (the edges to put behind a trait)")
        add("module_free", mods, ["free"], (1 - net_taint) * (1 - io_taint), "boolean", "neither network- nor IO-dependent")
        check("module_graph_acyclic", mod_acyclic, f"{len(mods)} modules")

    # ── unit level ─────────────────────────────────────────────────────────
    ci = {c: i for i, c in enumerate(crates)}
    DU = np.zeros((nu, nu), dtype=np.int64)
    for i in range(nu):
        for j in range(nu):
            if kind_of[j] == "lib" and (D[ci[crate_of[i]], ci[crate_of[j]]] or (kind_of[i] == "bin" and crate_of[i] == crate_of[j])):
                DU[i, j] = 1
    add("DU", units, units, DU, "boolean", "unit i depends on lib unit j (crate edge, or bin -> own lib)")
    su = np.array([first[u]["stratum"] for u in units])
    add("unit_stratum", units, ["s"], su.reshape(-1, 1), "none", "stratum of the unit (a bin sits above its own lib)")
    zs = sorted(set(su.tolist()))
    SZ = np.array([[int(su[i] == z) for i in range(nu)] for z in zs])
    add("SZ", [f"s{z}" for z in zs], units, SZ, "boolean", "stratum membership: SZ[z,u]=1 iff unit u is in stratum z")

    O = np.zeros((nu, nt), dtype=np.int64)
    ran = np.zeros_like(O); cp = np.zeros_like(O); warn = np.zeros_like(O); restr = np.zeros_like(O)
    classes = sorted(set(BASE_CLASSES) | {f["cls"] for r in results for c in r["units"].values() for f in c["check"]["failed"]})
    cause = {q: np.zeros_like(O) for q in classes}
    own = np.zeros_like(O)
    failed_pkgs: dict[str, set[int]] = {}
    for j, c in enumerate(cols):
        for i, u in enumerate(units):
            cell = by_col[c]["units"][u]
            O[i, j] = int(cell["check"]["exit"] == 0)
            cl = cell["clippy"]
            if "skipped" not in cl:
                ran[i, j] = 1; cp[i, j] = int(cl["exit"] == 0)
                warn[i, j] = sum(cl["lints"].values())
                restr[i, j] = sum(v for k, v in cl["lints"].items() if k in RESTRICTION_LINTS)
            for f in cell["check"]["failed"]:
                cause[f["cls"]][i, j] = 1
                failed_pkgs.setdefault(f["pkg"], set()).add(j)
                own[i, j] |= int(pkg_name(f["pkg"]) == crate_of[i])
    add("O", units, cols, O, "boolean", "O[u,t]=1 iff cargo check of unit u passes on column t")
    add("clippy_ran", units, cols, ran, "boolean", "clippy ran (check passed)")
    add("clippy_pass", units, cols, cp, "boolean", "clippy ran and exited 0")
    add("clippy_warnings", units, cols, warn, "counting", "clippy warnings emitted")
    add("clippy_restriction_hits", units, cols, restr, "counting", "std_instead_of_core + std_instead_of_alloc + alloc_instead_of_core hits")
    for q in classes:
        add(f"cause[{q}]", units, cols, cause[q], "boolean", f"unit fails with a package of class {q}")
    add("cause_counts", classes, cols, np.array([cause[q].sum(axis=0) for q in classes]), "counting", "units failing with class q on column t")

    # ── target x target ────────────────────────────────────────────────────
    notO = 1 - O
    Ag = O.T @ O + notO.T @ notO
    Imp = O.T @ notO
    add("Ag", cols, cols, Ag, "counting", "O^T O + (1-O)^T (1-O)")
    add("Imp", cols, cols, Imp, "counting", "O^T (1-O): units passing on row, failing on column")
    add("Compat", cols, cols, (Imp == 0), "boolean", "Imp == 0: passing on row implies passing on column, for every unit")
    check("Ag_symmetric_diag_full", (Ag == Ag.T).all() and (np.diag(Ag) == nu).all())

    # ── dependency tree, per column ────────────────────────────────────────
    closures = [[set(by_col[c]["units"][u]["closure"]["pkgs"]) for u in units] for c in cols]
    base = cols.index(f"{BASELINE_TARGET}|strict") if f"{BASELINE_TARGET}|strict" in cols else 0
    csize = np.array([[len(closures[j][i]) for j in range(nt)] for i in range(nu)])
    add("closure_size", units, cols, csize, "counting", "|cfg-resolved dependency closure of u on t|")
    add("closure_dropped", units, cols, [[len(closures[base][i] - closures[j][i]) for j in range(nt)] for i in range(nu)],
        "counting", "|closure(u, baseline) \\ closure(u, t)|", baseline=cols[base])
    add("closure_added", units, cols, [[len(closures[j][i] - closures[base][i]) for j in range(nt)] for i in range(nu)],
        "counting", "|closure(u, t) \\ closure(u, baseline)|", baseline=cols[base])

    P = sorted(failed_pkgs, key=lambda p: (-len(failed_pkgs[p]), p))
    if P:
        Bk = np.zeros((len(P), nt), dtype=np.int64)
        for a, p in enumerate(P):
            Bk[a, list(failed_pkgs[p])] = 1
        add("Bk", P, cols, Bk, "boolean", "package p failed to compile on column t")
        Ex = np.zeros_like(O)
        Intro = np.zeros((len(P), nu), dtype=np.int64)
        pnames = [pkg_name(p) for p in P]
        for j in range(nt):
            K = np.array([[int(any(pkg_name(x) == n for x in closures[j][i])) for n in pnames] for i in range(nu)])
            Ex[:, j] = ((K @ Bk[:, j]) > 0).astype(np.int64) | own[:, j]
            Intro |= K.T
        Ex = Ex * notO
        add("Ex", units, cols, Ex, "boolean", "failing unit explained by a failing package in its closure, or its own crate")
        add("Unexplained", units, cols, notO * (1 - Ex), "boolean", "(1-O)*(1-Ex): failure with no failing package in the closure")
        add("Intro", P, units, Intro, "boolean", "failing package p lies in the closure of unit u on some column")
        check("no_unexplained_failure", not (notO * (1 - Ex)).any(), int((notO * (1 - Ex)).sum()))
    # ── slices: which direct dependencies drag a root blocker into which crate ──
    # Exact per-target edges (probe: cargo tree --prefix depth). `entry[t][c,x]=1`
    # iff workspace crate c has direct external dependency x and x's resolved
    # subtree contains a root blocker on t - the edges to cut or feature-gate.
    # `drags[t][x,b]=1` iff x's subtree contains blocker package b on t.
    if all("edges" in by_col[c]["units"][u]["closure"] for c in cols for u in units):
        blockers = sorted({pkg_name(p) for p in failed_pkgs if pkg_name(p) not in crates})
        direct_ext = sorted({pkg_name(by_col[c]["units"][u]["closure"]["pkgs"][b])
                             for c in cols for u in units
                             for a, b in by_col[c]["units"][u]["closure"]["edges"]
                             if a == 0 and pkg_name(by_col[c]["units"][u]["closure"]["pkgs"][b]) not in crates})
        for j, col in enumerate(cols):
            roots = {p for p, cs in failed_pkgs.items() if j in cs and pkg_name(p) not in crates}
            if not roots:
                continue
            entry = np.zeros((len(crates), len(direct_ext)), dtype=np.int64)
            drags = np.zeros((len(direct_ext), len(blockers)), dtype=np.int64)
            for i, u in enumerate(units):
                g = by_col[col]["units"][u]["closure"]
                pk, adj = g["pkgs"], {}
                for a, b in g["edges"]:
                    adj.setdefault(a, []).append(b)
                for child in adj.get(0, []):
                    x = pkg_name(pk[child])
                    if x in crates:
                        continue
                    seen, todo = {child}, [child]
                    while todo:  # forward closure of x's subtree in this unit's resolved graph
                        for nxt in adj.get(todo.pop(), []):
                            if nxt not in seen:
                                seen.add(nxt)
                                todo.append(nxt)
                    hit = {pkg_name(pk[n]) for n in seen if pk[n] in roots}
                    if hit:
                        entry[crates.index(crate_of[i]), direct_ext.index(x)] = 1
                        for b in hit:
                            drags[direct_ext.index(x), blockers.index(b)] = 1
            add(f"entry|{col}", crates, direct_ext, entry, "boolean",
                "entry[c,x]=1: crate c directly depends on x and x's resolved subtree contains a root blocker on this column")
            add(f"drags|{col}", direct_ext, blockers, drags, "boolean",
                "drags[x,b]=1: x's resolved subtree on this column contains root-blocker package b")
            # reach = entry (x) drags: blockers a crate hits through its own external deps; the
            # slice adds those reached through workspace dependencies, R (x) reach.
            reach = sm.bool_matmul(entry, drags)
            add(f"slice|{col}", crates, blockers, ((reach + sm.bool_matmul(R, reach)) > 0), "boolean",
                "slice[c,b]=1: crate c depends, directly or through workspace crates, on blocker b: (entry drags) + R (entry drags)")

    # monotonicity: a unit cannot pass while one of its dependencies fails
    V = O * (DU @ notO)
    add("V", units, cols, V, "counting", "O * (DU (1-O)): passing units with a failing dependency (must be all 0)")
    check("monotone_check", not V.any(), int(V.sum()))
    # per-stratum pass counts and frontier, both from SZ and O
    add("stratum_pass", [f"s{z}" for z in zs], cols, SZ @ O, "counting", "SZ O: units passing in stratum z on column t")
    add("stratum_size", [f"s{z}" for z in zs], ["n"], SZ.sum(axis=1).reshape(-1, 1), "counting", "SZ 1")
    full = (SZ @ O) == SZ.sum(axis=1, keepdims=True)
    frontier = np.array([next((zs[z] for z in range(len(zs)) if not full[z, j]), -1) for j in range(nt)])
    add("frontier", cols, ["first_failing_stratum"], frontier.reshape(-1, 1), "none", "min stratum with a failing unit; -1 = none")

    # ── write + validate ───────────────────────────────────────────────────
    mdir = root / "matrices"
    sm.write_all(out, mdir, {"tool": "stratify_aggregate.py", "baseline": cols[base], "rustc": first and results[0]["rustc"]})
    doc = json.loads((mdir / "matrices.json").read_text())
    try:
        import jsonschema
        jsonschema.validate(doc, json.loads(SCHEMA_PATH.read_text()))
        check("json_schema_valid", True)
    except ImportError:
        check("json_schema_valid", False, "jsonschema not installed; shape/label checks in Matrix still ran")
    except jsonschema.ValidationError as e:
        check("json_schema_valid", False, e.message[:200])
    (mdir / "invariants.json").write_text(json.dumps(inv, indent=1))
    print(json.dumps({"matrices": {m.name: list(m.data.shape) for m in out}, "invariants": inv}))
    return 0 if all(i["ok"] for i in inv if i["name"] == "json_schema_valid") else 1


if __name__ == "__main__":
    sys.exit(main())
