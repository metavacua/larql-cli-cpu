#!/usr/bin/env python3
"""Labelled matrices as data: the one representation every stratification
script reads and writes.

A Matrix is a dense integer array with a label per row and per column and the
formula that defines it. It serialises to four formats, so any consumer can
pick the one its parser already speaks:

  matrices.json   canonical; validated against schemas/stratification-matrices.schema.json
  csv/<name>.csv  RFC 4180, first row and first column are the labels
  mtx/<name>.mtx  Matrix Market coordinate/integer/general (scipy.io.mmread,
                  Octave, MATLAB, Julia); labels in `%` comment lines
  matrices.npz    NumPy archive: <name>, <name>__rows, <name>__cols

Booleans are stored as 0/1 integers so the same array is a boolean-semiring
operand (`bool_matmul`) and a counting-semiring operand (`@`).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_ID = "larql.stratification.matrices/1"
SEMIRINGS = ("boolean", "counting", "none")


@dataclass
class Matrix:
    name: str
    rows: list[str]
    cols: list[str]
    data: np.ndarray
    semiring: str = "counting"
    definition: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.int64)
        if self.data.ndim != 2:
            raise ValueError(f"{self.name}: not 2-D (shape {self.data.shape})")
        if self.data.shape != (len(self.rows), len(self.cols)):
            raise ValueError(f"{self.name}: shape {self.data.shape} != labels ({len(self.rows)}, {len(self.cols)})")
        if self.semiring not in SEMIRINGS:
            raise ValueError(f"{self.name}: unknown semiring {self.semiring!r}")
        if self.semiring == "boolean" and not np.isin(self.data, (0, 1)).all():
            raise ValueError(f"{self.name}: boolean matrix holds a value other than 0/1")
        for axis, labels in (("rows", self.rows), ("cols", self.cols)):
            if len(set(labels)) != len(labels):
                raise ValueError(f"{self.name}: duplicate {axis} labels")

    def to_json(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "shape": list(self.data.shape), "semiring": self.semiring,
                "definition": self.definition, "meta": self.meta, "data": self.data.tolist()}


# ── linear algebra over the boolean semiring ───────────────────────────────

def bool_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a @ b) > 0).astype(np.int64)


def closure(d: np.ndarray) -> tuple[np.ndarray, bool]:
    """Transitive closure R = D + D^2 + ... and whether D is nilpotent.

    D is nilpotent exactly when the dependency graph is acyclic: some power
    D^k (k <= n) is the zero matrix.
    """
    n = d.shape[0]
    reach = np.zeros_like(d)
    power = d.copy()
    nilpotent = False
    for _ in range(n + 1):
        if not power.any():
            nilpotent = True
            break
        reach = ((reach + power) > 0).astype(np.int64)
        power = bool_matmul(power, d)
    return reach, nilpotent


def strata(d: np.ndarray) -> np.ndarray:
    """s_i = length of the longest dependency path starting at i (a DAG's rank).

    s_i = 1 + max_j { s_j : D[i,j] = 1 }, 0 for a leaf; equals the largest k
    with row i of D^k non-zero.
    """
    n = d.shape[0]
    s = np.zeros(n, dtype=np.int64)
    for _ in range(n + 1):
        nxt = np.array([1 + max((s[j] for j in np.flatnonzero(d[i])), default=-1) for i in range(n)], dtype=np.int64)
        if (nxt == s).all():
            return s
        s = nxt
    raise ValueError("dependency matrix is not nilpotent: workspace graph has a cycle")


# ── writers ────────────────────────────────────────────────────────────────

def write_all(matrices: list[Matrix], out: Path, producer: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = [m.name for m in matrices]
    if len(set(names)) != len(names):
        raise ValueError("duplicate matrix names")
    doc = {"schema": SCHEMA_ID, "producer": producer, "matrices": {m.name: m.to_json() for m in matrices}}
    (out / "matrices.json").write_text(json.dumps(doc, indent=1))
    (out / "csv").mkdir(exist_ok=True)
    (out / "mtx").mkdir(exist_ok=True)
    arrays = {}
    for m in matrices:
        with (out / "csv" / f"{m.name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([m.name, *m.cols])
            for label, row in zip(m.rows, m.data.tolist()):
                w.writerow([label, *row])
        r, c = np.nonzero(m.data)
        lines = ["%%MatrixMarket matrix coordinate integer general", f"% name {m.name}", f"% definition {m.definition}"]
        lines += [f"% row {i + 1} {label}" for i, label in enumerate(m.rows)]
        lines += [f"% col {j + 1} {label}" for j, label in enumerate(m.cols)]
        lines.append(f"{m.data.shape[0]} {m.data.shape[1]} {len(r)}")
        lines += [f"{i + 1} {j + 1} {int(m.data[i, j])}" for i, j in zip(r, c)]
        (out / "mtx" / f"{m.name}.mtx").write_text("\n".join(lines) + "\n")
        arrays[m.name] = m.data
        arrays[f"{m.name}__rows"] = np.array(m.rows, dtype=str)
        arrays[f"{m.name}__cols"] = np.array(m.cols, dtype=str)
    np.savez_compressed(out / "matrices.npz", **arrays)


def read_json(path: Path) -> dict[str, Matrix]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA_ID:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA_ID!r}")
    return {n: Matrix(n, m["rows"], m["cols"], np.array(m["data"], dtype=np.int64).reshape(m["shape"]),
                      m["semiring"], m["definition"], m["meta"]) for n, m in doc["matrices"].items()}
