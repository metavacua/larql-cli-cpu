#!/usr/bin/env python3
"""Probe every workspace (package, lib|bin) unit against ONE target triple.

One job per target runs this. Units are compiled in dependency order inside
one target dir, so shared third-party dependencies compile once per target
rather than once per cell. For each unit it records, as data and never as a
pass/fail colour:

  * `cargo check --keep-going`   - exit code, every failing package, the
                                   first error text, a cause class
  * `cargo clippy --keep-going`  - same, plus warning counts by lint,
                                   including the core/alloc/std restriction
                                   lints (skipped, visibly, when check failed)
  * `cargo tree -e normal,build` - the cfg-resolved dependency closure for
                                   this target (what Cargo itself believes
                                   the unit needs here)

`--keep-going` is deliberate: stopping at the first failing package would
show one blocker per cell; continuing shows every package the target
refuses, which is what the dependency-stratum analysis needs.

Usage: stratify_probe.py --target T --mode strict|sysroot-assumed --out DIR
Exit status is 0 whenever the probe itself ran: a failing cell is a result,
not a probe failure. It is non-zero only if cargo metadata cannot be read.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LIB_KINDS = {"lib", "rlib", "cdylib", "dylib", "staticlib", "proc-macro"}
RESTRICTION_LINTS = (
    "clippy::std_instead_of_core",
    "clippy::std_instead_of_alloc",
    "clippy::alloc_instead_of_core",
)
# Cap on error text kept per failing package; the full text is in the log.
FIRST_ERROR_CHARS = 400

# Cause classes, in precedence order. The first pattern that matches a
# failing package's text decides its class.
CAUSE_PATTERNS = (
    ("sysroot-absent", re.compile(r"can't find crate for `(core|alloc|compiler_builtins)`")),
    ("std-absent", re.compile(r"can't find crate for `std`")),
    ("native-lib-cross", re.compile(r"pkg-config has not been configured to support cross-compilation|PKG_CONFIG_ALLOW_CROSS|cross.?compil\w* .*pkg-config", re.I)),
    ("cc-missing", re.compile(r"failed to find tool|is `.*-gcc` not installed|ToolNotFound|couldn't find .*(compiler|cc)", re.I)),
    ("platform-unsupported", re.compile(r"target is not supported|not supported on this (platform|target)|unsupported (platform|target|operating system)|compile_error!|does not support this target|unresolved imports? `(libc|std::os|mio)", re.I)),
    ("build-script", re.compile(r"failed to run custom build command")),
)
COMPILE_FAIL = re.compile(r"could not compile `([^`]+)`")
BUILD_SCRIPT_FAIL = re.compile(r"failed to run custom build command for `([^` ]+) v?([^` ]+)")
PKG_ID_NAME = re.compile(r"#(?:[^#]*/)?([^/#@]+)@([^@#]+)$")
PKG_ID_PATHVER = re.compile(r"/([^/#]+)#([0-9][^#]*)$")


def sh(args: list[str], cwd: Path | None = None, env: dict | None = None):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")


def pkg_of(package_id: str) -> str:
    """`name@version` from either package-id spelling cargo has used."""
    m = PKG_ID_NAME.search(package_id)
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    m = PKG_ID_PATHVER.search(package_id)
    return f"{m.group(1)}@{m.group(2)}" if m else package_id


def workspace_units() -> list[dict]:
    """Units (name, kind) in dependency order, with stratum numbers."""
    r = sh(["cargo", "metadata", "--no-deps", "--locked", "--format-version", "1"])
    if r.returncode:
        sys.stderr.write(r.stderr)
        sys.exit(2)
    meta = json.loads(r.stdout)
    pkgs = {p["name"]: p for p in meta["packages"]}
    ws = set(pkgs)

    def deps_of(name: str) -> set[str]:
        return {d["name"] for d in pkgs[name]["dependencies"] if d["name"] in ws and d.get("kind") != "dev"}

    stratum: dict[str, int] = {}

    def depth(name: str, trail=()) -> int:
        if name in stratum:
            return stratum[name]
        if name in trail:
            raise SystemExit(f"workspace dependency cycle through {name}")
        stratum[name] = 1 + max((depth(d, trail + (name,)) for d in deps_of(name)), default=-1)
        return stratum[name]

    units = []
    for name, p in pkgs.items():
        kinds = {k for t in p["targets"] for k in t["kind"]}
        if kinds & LIB_KINDS:
            units.append({"name": name, "kind": "lib", "stratum": depth(name), "kinds": sorted(kinds)})
        if "bin" in kinds:
            # a bin sits one stratum above its own lib, if it has one
            units.append({"name": name, "kind": "bin", "stratum": depth(name) + (1 if kinds & LIB_KINDS else 0), "kinds": sorted(kinds)})
    units.sort(key=lambda u: (u["stratum"], u["name"], u["kind"]))
    return units


def classify(text: str) -> str:
    for cls, pat in CAUSE_PATTERNS:
        if pat.search(text):
            return cls
    return "source-error"


def run_cargo(verb: str, unit: dict, target: str, env: dict, extra: list[str], logdir: Path) -> dict:
    sel = ["--lib"] if unit["kind"] == "lib" else ["--bins"]
    cmd = ["cargo", verb, "--locked", "--keep-going", "--target", target, "-p", unit["name"], *sel,
           "--message-format", "json", *extra]
    t0 = time.monotonic()
    r = sh(cmd, env=env)
    elapsed = round(time.monotonic() - t0, 1)
    per_pkg: dict[str, list[str]] = {}
    lints: dict[str, int] = {}
    rendered_errors: list[str] = []
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        code = (msg.get("code") or {}).get("code")
        if msg["level"] == "error":
            per_pkg.setdefault(pkg_of(m["package_id"]), []).append(msg["rendered"])
            rendered_errors.append(msg["rendered"])
        elif msg["level"] == "warning" and code:
            lints[code] = lints.get(code, 0) + 1
    # Failures cargo reports only on stderr (build scripts, linker-less aborts).
    for pkg, ver in BUILD_SCRIPT_FAIL.findall(r.stderr):
        per_pkg.setdefault(f"{pkg}@{ver}", []).append(r.stderr)
    for pkg in COMPILE_FAIL.findall(r.stderr):
        if not any(k.startswith(pkg + "@") for k in per_pkg):
            per_pkg.setdefault(f"{pkg}@?", []).append(r.stderr)
    if r.returncode and not per_pkg:
        per_pkg["<unattributed>"] = [r.stderr]
    failed = [
        {"pkg": pkg, "cls": classify("\n".join(texts)), "first": ANSI.sub("", texts[0])[:FIRST_ERROR_CHARS]}
        for pkg, texts in per_pkg.items()
    ]
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{unit['name']}.{unit['kind']}.{verb}.log").write_text(
        f"$ {' '.join(cmd)}\nexit {r.returncode}\n--- stderr ---\n{r.stderr}\n--- rendered errors ---\n" + "\n".join(rendered_errors))
    # Unfiltered text goes straight to the job log (house style: never summarise a failure away).
    print(f"::group::{verb} {unit['name']}:{unit['kind']} -> exit {r.returncode}")
    print(r.stderr[-6000:])
    for e in rendered_errors[:40]:
        print(e)
    print("::endgroup::", flush=True)
    return {"exit": r.returncode, "secs": elapsed, "failed": failed, "lints": lints}


ANSI = re.compile(r"\x1b\[[0-9;]*m")


TREE_LINE = re.compile(r"^(\d+)\|(\S+) v(\S+)")


def closure(unit: dict, target: str, env: dict) -> dict:
    """Cfg- and feature-resolved dependency tree of one unit on one target.

    `--prefix depth` with a `|` separator gives parent -> child edges exactly
    as Cargo resolved them for this unit (normal + build, never dev), which a
    flat package list - or the target-agnostic Cargo.lock - cannot: lock edges
    include dev-dependencies and edges another target would not resolve.
    `pkgs` is the closure; `edges` are index pairs into `pkgs`.
    """
    r = sh(["cargo", "tree", "--locked", "--target", target, "-p", unit["name"], "-e", "normal,build",
            "--prefix", "depth", "-f", "|{p}"], env=env)
    if r.returncode:
        return {"exit": r.returncode, "error": r.stderr[:FIRST_ERROR_CHARS], "pkgs": [], "edges": []}
    names: dict[str, int] = {}
    edges: set[tuple[int, int]] = set()
    stack: list[int] = []
    for line in r.stdout.splitlines():
        m = TREE_LINE.match(line)
        if not m:
            continue  # section headings such as `[build-dependencies]`
        depth, pkg = int(m.group(1)), f"{m.group(2)}@{m.group(3)}"
        i = names.setdefault(pkg, len(names))
        del stack[depth:]
        if stack:
            edges.add((stack[-1], i))
        stack.append(i)
    order = sorted(names, key=names.get)
    return {"exit": 0, "pkgs": order, "edges": sorted(edges)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--mode", choices=("strict", "sysroot-assumed"), default="strict")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import os
    env = dict(os.environ)
    if a.mode == "sysroot-assumed":
        # Pretend a target-arch OpenBLAS exists, so a native-library gap does
        # not hide the Rust-level incompatibilities that sit above it. check
        # never links, so this only changes what build scripts are told.
        env["PKG_CONFIG_ALLOW_CROSS"] = "1"

    rustc = sh(["rustc", "--version", "--verbose"]).stdout
    units = workspace_units()
    result = {"target": a.target, "mode": a.mode, "rustc": rustc.splitlines()[0] if rustc else "?", "units": {}}
    logdir = a.out / "logs"
    for u in units:
        key = f"{u['name']}:{u['kind']}"
        cell = {"stratum": u["stratum"], "kinds": u["kinds"]}
        cell["check"] = run_cargo("check", u, a.target, env, [], logdir)
        if cell["check"]["exit"] == 0:
            cell["clippy"] = run_cargo("clippy", u, a.target, env, ["--no-deps", "--", *[f"-W{l}" for l in RESTRICTION_LINTS]], logdir)
        else:
            cell["clippy"] = {"skipped": "check failed"}
        cell["closure"] = closure(u, a.target, env)
        result["units"][key] = cell
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    ok = sum(1 for c in result["units"].values() if c["check"]["exit"] == 0)
    print(f"probe {a.target} [{a.mode}]: {ok}/{len(units)} units pass cargo check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
