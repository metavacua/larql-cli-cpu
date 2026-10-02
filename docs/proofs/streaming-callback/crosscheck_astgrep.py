#!/usr/bin/env python3
"""Cross-check the tree-sitter facts against an independent ast-grep extraction.

usage: crosscheck_astgrep.py REPO FACTS_DIR      (needs `ast-grep` on PATH)

Structural, not line-pinned, so it keeps meaning after the code changes.
"""
import collections, json, subprocess, sys

REPO, FACTS = sys.argv[1], sys.argv[2]
def rows(name): return [l.rstrip("\n").split("\t") for l in open(f"{FACTS}/{name}.facts")]
fn_file = {r[0]: r[1] for r in rows("fn")}

def ag(pattern, path):
    r = subprocess.run(["ast-grep", "run", "--lang", "rust", "--pattern", pattern,
                        "--json=compact", f"{REPO}/{path}"], capture_output=True, text=True)
    # ast-grep exits 1 when nothing matches; anything else non-zero is a real error.
    if r.returncode not in (0, 1) or (r.returncode == 1 and r.stderr.strip()):
        sys.exit(f"ast-grep failed on {path}: {r.stderr.strip()}")
    return sorted({m["range"]["start"]["line"] + 1 for m in json.loads(r.stdout or "[]")})

failed = False
def check(label, got, want):
    global failed
    ok = got == want
    failed |= not ok
    print(("OK   " if ok else "DIFF ") + label)
    if not ok: print(f"  ast-grep   : {got}\n  tree-sitter: {want}")

files = sorted({fn_file[f] for f in fn_file})

# (a) every `tokens.push(..)`: the same lines from both extractors, per file
for path in files:
    ts = sorted({int(r[2]) for r in rows("push") if r[1] == "tokens" and fn_file.get(r[0]) == path})
    check(f"tokens.push sites in {path}", ag("tokens.push($$$A)", path), ts)

# (b) callers of the CPU fallback entry point
for path in files:
    ts = sorted({int(r[2]) for r in rows("call") if r[1] == "generate_via_cpu_q4k" and fn_file.get(r[0]) == path})
    check(f"calls to generate_via_cpu_q4k in {path}", ag("generate_via_cpu_q4k($$$A)", path), ts)

# (c) every argument identifier `on_token` sits at a line where ast-grep also finds `on_token`
by_file = collections.defaultdict(set)
for r in rows("arg_line"):
    if r[3] == "on_token" and r[0] in fn_file: by_file[fn_file[r[0]]].add(int(r[4]))
for path, lines in sorted(by_file.items()):
    check(f"on_token argument lines in {path}", sorted(lines - set(ag("on_token", path))), [])

sys.exit(1 if failed else 0)
