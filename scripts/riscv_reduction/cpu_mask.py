#!/usr/bin/env python3
"""Find a QEMU rv64 -cpu string with F and D removed that QEMU accepts.

QEMU refuses CPU definitions with dangling dependencies ("Zfa extension requires
F extension"). Disabling F/D by hand therefore fails before any guest code runs,
which would be mistaken for a lossy result. Start from f=false,d=false and keep
disabling whatever QEMU names until it starts the probe binary, then print the
accepted -cpu string. Exit 1 if it never converges.

Usage: cpu_mask.py <sysroot> <probe-binary>
"""
import re, subprocess, sys

sysroot, probe = sys.argv[1:3]
disabled = ["f", "d"]
for _ in range(40):
    cpu = "rv64," + ",".join(f"{e}=false" for e in disabled)
    r = subprocess.run(["qemu-riscv64-static", "-cpu", cpu, "-L", sysroot, probe, "--version"],
                       capture_output=True, text=True)
    err = r.stderr.strip().splitlines()[0] if r.stderr.strip() else ""
    if not err.startswith("qemu-"):
        print(cpu)
        sys.exit(0)
    m = re.search(r"(\w+) extension requires", err)
    p = re.search(r"Property '\.?(\w+)' not found", err)
    if m and m.group(1).lower() not in disabled:
        disabled.append(m.group(1).lower())
    elif p and p.group(1) in disabled:
        disabled.remove(p.group(1))
    else:
        print(f"unhandled qemu error: {err}", file=sys.stderr)
        sys.exit(1)
print("did not converge", file=sys.stderr)
sys.exit(1)
