#!/usr/bin/env python3
"""Fact extraction for binding.dl, and Lean certificates from the facts.

  lqlfacts.py astgrep SRC_DIR OUT_DIR       structural facts via ast-grep (rules.yml)
  lqlfacts.py ir [--demangler CMD] [--crate NAME] IR_FILE... OUT_DIR
                                            call edges from rustc's LLVM IR
  lqlfacts.py lean FACTS_DIR OUT.lean [--claim HANDLER]...
                                            a Lean certificate that each claimed
                                            handler reaches no backend read

Functions are identified by name; same-named functions are merged, which
over-approximates reachability. The IR pass sees calls inside macros and
monomorphised generics that the syntactic pass misses; `dyn` calls appear
only as counts of indirect calls (indirect.facts).
"""
import argparse, collections, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def write(out, name, rows):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".facts"), "w") as fh:
        for r in sorted(rows):
            fh.write("\t".join(r if isinstance(r, tuple) else (r,)) + "\n")


def is_test_file(path):
    return "/tests/" in path or path.endswith("tests.rs")


def astgrep(src, out):
    proc = subprocess.run(
        ["ast-grep", "scan", "-r", os.path.join(HERE, "rules.yml"), "--json=stream", "."],
        cwd=src, capture_output=True, text=True, check=True)
    matches = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    span = lambda m: (m["range"]["byteOffset"]["start"], m["range"]["byteOffset"]["end"])
    test_mods = collections.defaultdict(list)
    for m in matches:
        if m["ruleId"] == "testmod":
            test_mods[m["file"]].append(span(m))
    in_test = lambda f, s: is_test_file(f) or any(a <= s < b for a, b in test_mods[f])
    fns = collections.defaultdict(list)
    for m in matches:
        s, e = span(m)
        if m["ruleId"] == "fn" and not in_test(m["file"], s):
            fns[m["file"]].append((s, e, re.search(r"fn\s+(\w+)", m["text"]).group(1)))
    names = {n for v in fns.values() for _, _, n in v}

    def owner(f, s):
        c = [(e - a, n) for a, e, n in fns.get(f, []) if a <= s < e]
        return min(c)[1] if c else None

    calls, reads, writes, raises, dispatch = set(), set(), set(), set(), set()
    for m in matches:
        f, (s, _) = m["file"], span(m)
        if in_test(f, s):
            continue
        o = owner(f, s)
        if not o:
            continue
        rid = m["ruleId"]
        if rid == "call":
            callee = re.split(r"[.:]", m["text"].split("(")[0].split("<")[0].strip())[-1].strip()
            if callee in names and callee != o:
                calls.add((o, callee))
        elif rid == "backend_read":
            reads.add(o)
        elif rid == "backend_write":
            writes.add(o)
        elif rid == "nobackend":
            raises.add(o)
        elif rid == "arm":
            variant = re.match(r"Statement::(\w+)", m["text"]).group(1)
            for h in re.findall(r"self\.(exec_\w+)\(", m["text"].split("=>", 1)[1]):
                dispatch.add((variant, h))
    write(out, "fn", names)
    write(out, "calls", calls)
    write(out, "reads_backend", reads)
    write(out, "writes_backend", writes)
    write(out, "raises_nobackend", raises)
    write(out, "dispatch", dispatch)
    if not os.path.exists(os.path.join(out, "ir_calls.facts")):
        write(out, "ir_calls", [])
    print(f"astgrep: {len(names)} fns, {len(calls)} call edges, {len(reads)} backend readers, "
          f"{len(writes)} writers, {len(raises)} NoBackend sites, {len(dispatch)} dispatch pairs")


DEFINE = re.compile(r'^define\b.*?@("(?:[^"\\]|\\.)*"|[\w.$]+)\(')
CALL = re.compile(r'\b(?:call|invoke)\b.*?@("(?:[^"\\]|\\.)*"|[\w.$]+)\(')
INDIRECT = re.compile(r'\b(?:call|invoke)\b[^@]*?%[\w.]+\(')


def short_name(path):
    """`a::b::Type::method::{closure#0}` -> method; `<X as T>::fmt` -> fmt."""
    path = re.sub(r"\[[0-9a-f]+\]", "", path)          # v0 crate disambiguators
    path = re.sub(r"::h[0-9a-f]{16}$", "", path)        # legacy hash
    if ">::" in path:
        path = path.rsplit(">::", 1)[1]
    depth, segs, cur = 0, [], ""
    for ch in path:                                     # split on :: outside <>
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if depth == 0 and cur.endswith(":") and ch == ":":
            segs.append(cur[:-1]); cur = ""; continue
        cur += ch
    segs.append(cur)
    segs = [re.sub(r"<.*>$", "", s) for s in segs if s and not s.startswith("{")]
    return segs[-1] if segs else path


def ir(files, out, demangler, crate):
    raw = []
    for path in files:
        with open(path, errors="replace") as fh:
            raw.extend(fh)
    syms = set()
    for line in raw:
        for rx in (DEFINE, CALL):
            for s in rx.findall(line):
                syms.add(s.strip('"'))
    syms = sorted(syms)
    dem = subprocess.run(demangler, shell=True, input="\n".join(syms) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    if len(dem) != len(syms):
        sys.exit(f"demangler returned {len(dem)} lines for {len(syms)} symbols")
    full = dict(zip(syms, dem))
    ours = lambda s: crate in full[s]
    edges, indirect, defined = set(), collections.Counter(), set()
    cur = None
    for line in raw:
        d = DEFINE.match(line)
        if d:
            s = d.group(1).strip('"')
            cur = short_name(full[s]) if ours(s) else None
            if cur:
                defined.add(cur)
            continue
        if line.startswith("}"):
            cur = None
            continue
        if cur is None:
            continue
        for s in CALL.findall(line):
            s = s.strip('"')
            if ours(s):
                callee = short_name(full[s])
                if callee != cur:
                    edges.add((cur, callee))
        if INDIRECT.search(line):
            indirect[cur] += 1
    write(out, "calls", edges)
    write(out, "indirect", {(k, str(v)) for k, v in indirect.items()})
    write(out, "ir_fn", defined)
    print(f"ir: {len(defined)} {crate} fns, {len(edges)} call edges, "
          f"{sum(indirect.values())} indirect calls in {len(indirect)} fns")


def read(d, name):
    p = os.path.join(d, name + ".facts")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [tuple(l.rstrip("\n").split("\t")) for l in fh if l.strip()]


def lean(facts, out, claims):
    names = sorted(r[0] for r in read(facts, "fn"))
    idx = {n: i for i, n in enumerate(names)}
    edges = [(a, b) for a, b in read(facts, "calls")]
    edges += [(a, b) for a, b in read(facts, "ir_calls") if a in idx and b in idx]
    edges = sorted(set(edges))
    readers = {r[0] for r in read(facts, "reads_backend")}
    if not claims:
        handlers = sorted({h for _, h in read(facts, "dispatch") if h in idx})
        claims = [h for h in handlers if not (closure(h, edges) & readers)]
    lines = [
        "set_option maxRecDepth 100000",
        "/- Generated by lqlfacts.py. Node ids index fn.facts in sorted order. -/",
        "inductive Reach (E : List (Nat × Nat)) : Nat → Nat → Prop",
        "  | refl (a) : Reach E a a",
        "  | step {a b c} : Reach E a b → (b, c) ∈ E → Reach E a c",
        "",
        "theorem closed_reach (E : List (Nat × Nat)) (S : List Nat)",
        "    (hc : ∀ e ∈ E, e.1 ∈ S → e.2 ∈ S) {a b : Nat} (ha : a ∈ S) (h : Reach E a b) : b ∈ S := by",
        "  induction h with",
        "  | refl => exact ha",
        "  | step _ he ih => exact hc _ he ih",
        "",
        "theorem no_reach (E : List (Nat × Nat)) (S Rd : List Nat)",
        "    (hc : ∀ e ∈ E, e.1 ∈ S → e.2 ∈ S) (hd : ∀ x ∈ S, x ∉ Rd) {a b : Nat}",
        "    (ha : a ∈ S) (h : Reach E a b) : b ∉ Rd :=",
        "  hd b (closed_reach E S hc ha h)",
        "",
        "def E : List (Nat × Nat) := [" + ", ".join(f"({idx[a]}, {idx[b]})" for a, b in edges) + "]",
        "def Rd : List Nat := [" + ", ".join(str(idx[r]) for r in sorted(readers) if r in idx) + "]",
    ]
    for h in claims:
        if h not in idx:
            sys.exit(f"claim {h}: not a known function")
        s = sorted(closure(h, edges))
        lines += [
            "",
            f"-- {h} = {idx[h]}; closed set: {', '.join(s)}",
            f"def S_{h} : List Nat := [" + ", ".join(str(idx[x]) for x in s) + "]",
            f"theorem {h}_reaches_no_backend_read : ∀ b, Reach E {idx[h]} b → b ∉ Rd :=",
            f"  fun _ h => no_reach E S_{h} Rd (by decide) (by decide) (by decide) h",
        ]
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"lean: {len(claims)} claims, {len(edges)} edges -> {out}")


def closure(h, edges):
    succ = collections.defaultdict(set)
    for a, b in edges:
        succ[a].add(b)
    seen, stack = {h}, [h]
    while stack:
        for y in succ[stack.pop()]:
            if y not in seen:
                seen.add(y); stack.append(y)
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("astgrep"); a.add_argument("src"); a.add_argument("out")
    i = sub.add_parser("ir"); i.add_argument("--demangler", default="c++filt")
    i.add_argument("--crate", default="larql_lql"); i.add_argument("paths", nargs="+")
    l = sub.add_parser("lean"); l.add_argument("facts"); l.add_argument("out")
    l.add_argument("--claim", action="append", default=[])
    args = ap.parse_args()
    if args.cmd == "astgrep":
        astgrep(args.src, args.out)
    elif args.cmd == "ir":
        ir(args.paths[:-1], args.paths[-1], args.demangler, args.crate)
    else:
        lean(args.facts, args.out, args.claim)


if __name__ == "__main__":
    main()
