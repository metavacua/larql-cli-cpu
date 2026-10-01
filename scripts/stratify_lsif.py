#!/usr/bin/env python3
"""Crate x crate matrix of ACTUAL symbol use, from a rust-analyzer LSIF dump.

Cargo.toml says which crates a crate MAY use. The semantic index says which
it DOES: every reference range whose definition lives in another workspace
crate is one use. Comparing the two gives the edges that can be cut without
changing behaviour (declared, never used) - the cheapest way to shrink a
dependency stratum before any no_std port.

Reads a stream of LSIF JSON objects (one per line, rust-analyzer's `lsif`
subcommand). Only three shapes matter:
  vertex  document            {id, label:"document", uri}
  edge    textDocument/references  resultSet -> referenceResult
  edge    item (property definitions|references)  referenceResult -> ranges,
          carrying the `document` the ranges live in

Usage: stratify_lsif.py --lsif FILE --root WORKSPACE_ROOT --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

CRATE_DIR = "crates"
PROP_DEFS, PROP_REFS = "definitions", "references"


def crate_of(uri: str, root: str) -> str | None:
    path = unquote(urlparse(uri).path)
    if not path.startswith(root):
        return None
    parts = Path(path[len(root):].lstrip("/")).parts
    return parts[1] if len(parts) > 1 and parts[0] == CRATE_DIR else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsif", required=True, type=Path)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    root = str(Path(a.root).resolve()).rstrip("/") + "/"

    doc_crate: dict[int, str | None] = {}
    # referenceResult id -> {"defs": Counter(doc), "refs": Counter(doc)}
    results: dict[int, dict[str, Counter]] = defaultdict(lambda: {"defs": Counter(), "refs": Counter()})
    labels: Counter = Counter()
    for line in a.lsif.open():
        try:
            o = json.loads(line)
        except ValueError:
            continue
        labels[o.get("label", "?")] += 1
        if o.get("label") == "document":
            doc_crate[o["id"]] = crate_of(o["uri"], root)
        elif o.get("label") == "item" and o.get("property") in (PROP_DEFS, PROP_REFS):
            side = "defs" if o["property"] == PROP_DEFS else "refs"
            results[o["outV"]][side][o["document"]] += len(o["inVs"])

    uses: Counter = Counter()  # (user crate, defining crate) -> reference count
    for res in results.values():
        def_crates = {doc_crate.get(d) for d in res["defs"]} - {None}
        for doc, n in res["refs"].items():
            user = doc_crate.get(doc)
            for dc in def_crates:
                if user and dc != user:
                    uses[(user, dc)] += n
    if not uses:
        # Never emit an empty matrix as if it meant "no coupling".
        sys.stderr.write(f"no cross-crate references found; LSIF labels seen: {dict(labels.most_common(12))}\n")
        return 1

    crates = sorted({c for pair in uses for c in pair})
    a.out.mkdir(parents=True, exist_ok=True)
    header = "| user \\ defines | " + " | ".join(c.replace("larql-", "") for c in crates) + " |"
    rows = [header, "|" + "---|" * (len(crates) + 1)]
    for u in crates:
        rows.append(f"| **{u}** | " + " | ".join(str(uses[(u, d)] or "·") if u != d else "■" for d in crates) + " |")
    (a.out / "crate-by-crate-actual-use.md").write_text(
        "# crate x crate - actual symbol references (rust-analyzer LSIF)\n\n"
        "Row uses symbols defined in column. Compare with the declared-dependency matrix: "
        "a declared edge with `·` here is dead weight.\n\n" + "\n".join(rows) + "\n")
    (a.out / "actual-use.json").write_text(json.dumps({f"{u}->{d}": n for (u, d), n in sorted(uses.items())}, indent=1))
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
