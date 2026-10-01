#!/usr/bin/env python3
"""Round-trip every serialisation of the stratification matrices through an
independent parser and require agreement with the canonical JSON.

  matrices.json  parsed with `json`, validated with `jsonschema`
  csv/*.csv      parsed with the `csv` module
  mtx/*.mtx      parsed with scipy.io.mmread (the reference Matrix Market reader)
  matrices.npz   parsed with numpy.load

Prints one JSON object per format: {"format", "checked", "mismatches", "skipped"}.
A format that cannot be checked (library absent) says so in `skipped`; it is
never reported as passing. Exit 1 on any mismatch or any skipped format when
--require-all is given (CI), so the check cannot be quietly weakened.

Usage: stratify_verify.py DIR [--require-all]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = Path(__file__).parent / "schemas" / "stratification-matrices.schema.json"


def main() -> int:
    root = Path(sys.argv[1])
    require_all = "--require-all" in sys.argv
    doc = json.loads((root / "matrices.json").read_text())
    ms = doc["matrices"]
    report = []

    def rec(fmt, checked, bad, skipped=None):
        report.append({"format": fmt, "checked": checked, "mismatches": bad, "skipped": skipped})

    try:
        import jsonschema
        jsonschema.validate(doc, json.loads(SCHEMA.read_text()))
        rec("json-schema", len(ms), [])
    except ImportError:
        rec("json-schema", 0, [], "jsonschema not installed")
    except jsonschema.ValidationError as e:
        rec("json-schema", len(ms), [e.message[:200]])

    bad = []
    for n, m in ms.items():
        with (root / "csv" / f"{n}.csv").open(newline="") as f:
            rows = list(csv.reader(f))
        data = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=np.int64).reshape(m["shape"])
        if rows[0][1:] != m["cols"] or [r[0] for r in rows[1:]] != m["rows"] or not (data == np.array(m["data"]).reshape(m["shape"])).all():
            bad.append(n)
    rec("csv", len(ms), bad)

    z = np.load(root / "matrices.npz")
    bad = [n for n, m in ms.items()
           if not (z[n] == np.array(m["data"]).reshape(m["shape"])).all()
           or list(z[f"{n}__rows"]) != m["rows"] or list(z[f"{n}__cols"]) != m["cols"]]
    rec("npz", len(ms), bad)

    try:
        from scipy.io import mmread
    except ImportError:
        rec("mtx", 0, [], "scipy not installed")
    else:
        bad = [n for n, m in ms.items()
               if not (mmread(root / "mtx" / f"{n}.mtx").toarray() == np.array(m["data"]).reshape(m["shape"])).all()]
        rec("mtx", len(ms), bad)

    for r in report:
        print(json.dumps(r))
    failed = any(r["mismatches"] for r in report) or (require_all and any(r["skipped"] for r in report))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
