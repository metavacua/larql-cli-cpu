#!/usr/bin/env python3
"""Source-level (tree-sitter AST) facts per workspace crate, via ast-grep.

The target probe answers "does it compile on T". This answers the question
a compiler cannot cheaply: WHERE in the source the coupling to an operating
system lives, independent of any target - so a failing cell can be traced to
a concrete construct and a no_std port can be sized before it is attempted.

Facts per crate (all counted from the AST, never from text grep, so strings
and comments never match):

  inner_attrs      crate-root `#![...]` attributes - is `no_std` declared?
  std_paths        `std::<module>` scoped paths, by module
  cfg_target       `#[cfg(...)]` / `cfg!(...)` / `cfg_attr(...)` predicates that
                   name target_os / target_arch / target_family / unix / windows
  unsafe_blocks    `unsafe` blocks and `unsafe fn`
  extern_c         `extern "C"` blocks and fns (FFI surface)

Usage: stratify_ast.py --ast-grep PATH --crates crates --out facts.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

STD_MODULES = (
    "fs", "net", "env", "process", "thread", "path", "time", "os", "io",
    "sync", "collections", "alloc", "ffi", "mem", "fmt", "string", "vec",
    "boxed", "rc", "cell", "iter", "ops", "cmp", "str", "slice", "num",
)
# std modules that are OS services (no core/alloc counterpart exists).
OS_BOUND = {"fs", "net", "env", "process", "thread", "path", "time", "os", "io", "sync"}
# Roots whose use means network (or async-network) behaviour. `tokio` is the
# async runtime the network stack sits on; `larql_router*` are this repo's own
# network crates. A scoped path rooted here is one use.
NET_ROOTS = ("reqwest", "hf_hub", "tonic", "axum", "tokio", "hyper", "hyper_util", "quinn", "h3", "h3_quinn",
             "h3_axum", "rustls", "tungstenite", "tower", "mio", "socket2", "mockito", "larql_router",
             "larql_router_protocol", "std::net")
# IO = the process talking to its environment other than over a network: the
# filesystem, standard streams, environment, child processes, terminals.
IO_STD = ("fs", "io", "env", "process", "path")
IO_CRATES = ("memmap2", "rustyline", "indicatif", "console", "dirs", "walkdir", "tempfile")
NET_MODULE_DEPTH = 2  # directories below src/ that name a module: crate/dir/subdir
CFG_TARGET = re.compile(r"target_(os|arch|family|env|pointer_width|endian|has_atomic|vendor|feature)|\bunix\b|\bwindows\b|\bwasm")

RULES = {
    "std_path": {
        "kind": "scoped_identifier",
        "regex": rf"^(::)?std::({'|'.join(STD_MODULES)})\b",
    },
    "net_path": {
        "kind": "scoped_identifier",
        "regex": rf"^(::)?({'|'.join(NET_ROOTS)})(::|$)",
    },
    "io_path": {
        "kind": "scoped_identifier",
        "regex": rf"^(::)?({'|'.join(IO_CRATES)})(::|$)",
    },
    "unsafe": {"any": [
        {"kind": "unsafe_block"},
        {"kind": "function_item", "has": {"kind": "function_modifiers", "regex": "unsafe"}},
    ]},
    "extern_c": {"any": [
        {"kind": "foreign_mod_item"},
        {"kind": "function_item", "has": {"kind": "function_modifiers", "regex": "extern"}},
    ]},
    "attr": {"kind": "attribute_item"},
    "inner_attr": {"kind": "inner_attribute_item"},
    "cfg_macro": {"pattern": "cfg!($$$)"},
}


def scan(ast_grep: str, crates: Path, rule_id: str, rule: dict) -> list[dict]:
    inline = json.dumps({"id": rule_id, "language": "rust", "rule": rule})
    r = subprocess.run([ast_grep, "scan", "--inline-rules", inline, "--json=compact", str(crates)],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1) or (r.stdout.strip() and not r.stdout.lstrip().startswith("[")):
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ast-grep failed on rule {rule_id}")
    return json.loads(r.stdout) if r.stdout.strip() else []


def crate_of(path: str) -> str | None:
    parts = Path(path).parts
    return parts[parts.index("crates") + 1] if "crates" in parts else None


def module_of(path: str) -> str:
    """`crate/dir/subdir` for a file under src/ (a bare file counts as its stem)."""
    parts = Path(path).parts
    i = parts.index("crates")
    crate, rest = parts[i + 1], list(parts[i + 2:])
    rest = rest[1:] if rest and rest[0] == "src" else rest
    names = [p[:-3] if p.endswith(".rs") else p for p in rest]
    return "/".join([crate, *names[:NET_MODULE_DEPTH]])


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or "benches" in parts or "examples" in parts or Path(path).name.endswith("_tests.rs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-grep", required=True)
    ap.add_argument("--crates", type=Path, default=Path("crates"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    facts: dict[str, dict] = defaultdict(lambda: {
        "inner_attrs": [], "std_paths": Counter(), "cfg_target": 0, "cfg_target_files": Counter(),
        "unsafe": 0, "extern_c": 0, "no_std_declared": False, "no_std_conditional": False,
    })
    io_modules: dict[str, Counter] = defaultdict(Counter)
    for m in scan(a.ast_grep, a.crates, "std_path", RULES["std_path"]):
        c = crate_of(m["file"])
        if c and not is_test_path(m["file"]):  # production code decides portability, tests do not
            mod = re.match(r"(?:::)?std::(\w+)", m["text"]).group(1)
            facts[c]["std_paths"][mod] += 1
            if mod in IO_STD:
                io_modules[module_of(m["file"])][f"std::{mod}"] += 1
    for m in scan(a.ast_grep, a.crates, "io_path", RULES["io_path"]):
        c = crate_of(m["file"])
        if c and not is_test_path(m["file"]):
            io_modules[module_of(m["file"])][re.match(rf"(?:::)?({'|'.join(IO_CRATES)})", m["text"]).group(1)] += 1
            facts[c]
    net_modules: dict[str, Counter] = defaultdict(Counter)
    for m in scan(a.ast_grep, a.crates, "net_path", RULES["net_path"]):
        c = crate_of(m["file"])
        if c and not is_test_path(m["file"]):
            root = re.match(rf"(?:::)?({'|'.join(NET_ROOTS)})(?:::|$)", m["text"]).group(1)
            net_modules[module_of(m["file"])][root] += 1
            facts[c]  # make sure the crate has an entry even if it has no other facts
    for rid in ("unsafe", "extern_c"):
        for m in scan(a.ast_grep, a.crates, rid, RULES[rid]):
            c = crate_of(m["file"])
            if c and not is_test_path(m["file"]):
                facts[c][rid] += 1
    for m in scan(a.ast_grep, a.crates, "inner_attr", RULES["inner_attr"]):
        c = crate_of(m["file"])
        if c and Path(m["file"]).name in ("lib.rs", "main.rs"):
            facts[c]["inner_attrs"].append(m["text"][:120])
            if "no_std" in m["text"]:
                facts[c]["no_std_declared"] = True
                facts[c]["no_std_conditional"] = "cfg_attr" in m["text"]
    for rid in ("attr", "cfg_macro"):
        for m in scan(a.ast_grep, a.crates, rid, RULES[rid]):
            c = crate_of(m["file"])
            if c and not is_test_path(m["file"]) and re.search(r"\bcfg(_attr)?\s*\(", m["text"]) and CFG_TARGET.search(m["text"]):
                facts[c]["cfg_target"] += 1
                facts[c]["cfg_target_files"][m["file"]] += 1

    out = {}
    for c, f in sorted(facts.items()):
        os_bound = {k: v for k, v in f["std_paths"].items() if k in OS_BOUND}
        out[c] = {
            "no_std_declared": f["no_std_declared"], "no_std_conditional": f["no_std_conditional"],
            "inner_attrs": f["inner_attrs"], "std_paths": dict(f["std_paths"]),
            "os_bound_std_paths": os_bound, "os_bound_total": sum(os_bound.values()),
            "net_modules": {m: dict(n) for m, n in sorted(net_modules.items()) if m.split("/")[0] == c},
            "io_modules": {m: dict(n) for m, n in sorted(io_modules.items()) if m.split("/")[0] == c},
            "cfg_target": f["cfg_target"], "cfg_target_top_files": f["cfg_target_files"].most_common(5),
            "unsafe": f["unsafe"], "extern_c": f["extern_c"],
        }
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{'crate':28} no_std  os-bound std  cfg(target)  unsafe  extern-C")
    for c, f in out.items():
        print(f"{c:28} {str(f['no_std_declared']):6}  {f['os_bound_total']:11}  {f['cfg_target']:11}  {f['unsafe']:6}  {f['extern_c']:8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
