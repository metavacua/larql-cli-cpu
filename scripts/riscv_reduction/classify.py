#!/usr/bin/env python3
"""Classify per-command behaviour between a reference run and a reduced run.

  identical   same exit code and stdout
  similar     same exit code, stdout differs only in whitespace/digits
  refused     reference succeeded (rc 0), reduced did not -> reduced is a
              strict subset of the reference (inclusion holds)
  harness-error  the runner (QEMU) failed to start; no evidence either way, FAILS
  divergent   anything else (reduced succeeds where the reference fails,
              or both fail with different codes) -> inclusion VIOLATED

Usage: classify.py <reference-dir> <reduced-dir> <label>
Exit 1 if any command is divergent. Emits a markdown table on stdout.
"""
import pathlib, re, sys

def load(d, i):
    p = pathlib.Path(d)
    return (int((p / f"{i}.rc").read_text()), (p / f"{i}.out").read_text())

def norm(s):
    return re.sub(r"\d+", "#", " ".join(s.split()))

ref, red, label = sys.argv[1:4]
ids = sorted(p.stem for p in pathlib.Path(ref).glob("*.cmd"))
if not ids:
    sys.exit("no commands captured in reference dir")
counts, bad = {}, 0
print(f"### {label}\n\n| command | reference rc | reduced rc | class |\n|---|---|---|---|")
for i in ids:
    cmd = (pathlib.Path(ref) / f"{i}.cmd").read_text().strip()
    (rrc, rout), (drc, dout) = load(ref, i), load(red, i)
    if (pathlib.Path(red) / f"{i}.harness").exists() or (pathlib.Path(ref) / f"{i}.harness").exists():
        k = "harness-error"; bad += 1
    elif rrc == drc and rout == dout: k = "identical"
    elif rrc == drc and norm(rout) == norm(dout): k = "similar"
    elif rrc == 0 and drc != 0: k = "refused"
    else: k = "divergent"; bad += 1
    counts[k] = counts.get(k, 0) + 1
    print(f"| `{cmd}` | {rrc} | {drc} | {k} |")
print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
sys.exit(1 if bad else 0)
