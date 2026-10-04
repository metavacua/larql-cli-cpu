#!/usr/bin/env bash
# Front bin/larql with an emulator, so every caller that runs ./bin/larql, in any working
# directory, runs it under the emulator without knowing.
#
#   scripts/ci/wrap_emulator.sh qemu-riscv64-static -cpu rv64 -L /usr/riscv64-linux-gnu
#
# Run from the repository root, after the binary is downloaded to bin/larql. The real binary
# moves to bin/larql.real and is named by absolute path: the matrix cells run in other
# directories.
set -euo pipefail
[ "$#" -gt 0 ] || { echo "usage: $0 <emulator command...>" >&2; exit 2; }
real="$(pwd)/bin/larql.real"
mv bin/larql "$real"
chmod +x "$real"
printf '#!/bin/sh\nexec %s %s "$@"\n' "$*" "$real" > bin/larql
chmod +x bin/larql
