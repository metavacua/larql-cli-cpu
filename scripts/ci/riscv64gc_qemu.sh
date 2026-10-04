#!/usr/bin/env bash
# riscv64gc-unknown-linux-gnu on an x86_64 Ubuntu runner: the cross C toolchain, riscv64
# OpenBLAS, and QEMU user-mode, so the target can be linted, built, linked and RUN.
# The recipe is target-matrix.yml's binary-gnu-riscv64 job, which passes. Idempotent.
#
# Exports to $GITHUB_ENV what cargo and the loader need:
#   CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER   cross linker (and cc-rs finds its gcc)
#   OPENBLAS_LIB_DIR_riscv64gc_unknown_linux_gnu      where larql-blas-link finds -lopenblas
#   QEMU_LD_PREFIX                                    sysroot for qemu-riscv64-static
#
# Run it AFTER any plain `apt-get update`: it adds the riscv64 foreign architecture, and the
# runner's own archive carries no riscv64 index, so a later plain update would 404.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  gcc-riscv64-linux-gnu g++-riscv64-linux-gnu libc6-dev-riscv64-cross \
  qemu-user-static pkg-config

# riscv64 packages live on ports.ubuntu.com; update only that list.
sudo dpkg --add-architecture riscv64
CODENAME=$(lsb_release -cs)
{
  echo "deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports $CODENAME main universe"
  echo "deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports $CODENAME-updates main universe"
  echo "deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports $CODENAME-security main universe"
} | sudo tee /etc/apt/sources.list.d/ports-riscv64.list > /dev/null
sudo apt-get update \
  -o Dir::Etc::sourcelist="sources.list.d/ports-riscv64.list" \
  -o Dir::Etc::sourceparts="-" \
  -o APT::Get::List-Cleanup="0"
sudo apt-get install -y libopenblas-dev:riscv64 pkg-config

{
  echo "OPENBLAS_LIB_DIR_riscv64gc_unknown_linux_gnu=/usr/lib/riscv64-linux-gnu"
  echo "CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc"
  echo "QEMU_LD_PREFIX=/usr/riscv64-linux-gnu"
} >> "${GITHUB_ENV:-/dev/null}"
