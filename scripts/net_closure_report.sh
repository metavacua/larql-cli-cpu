#!/usr/bin/env bash
# Network-closure gate for larql-cli. Metadata only: no compile, no
# OpenBLAS, every cargo call is `cargo tree --locked`.
#
# larql-cli's `net` feature (default on) owns every network dependency the
# CLI itself names: `reqwest` and `larql-router` are optional dependencies
# enabled only by `net`. This script proves that wiring from the resolved
# dependency graph.
#
# HARD gates (exit 1):
#   (a) with --no-default-features, `larql-router` is not in the graph;
#   (b) with --no-default-features, `larql-cli` is not a direct dependent
#       of `reqwest`;
#   (c) positive controls with --no-default-features --features net:
#       `larql-router` IS present and `larql-cli` IS a direct dependent of
#       `reqwest`, so (a) and (b) cannot pass vacuously (a typo'd package
#       name or a failing cargo call would otherwise look like success).
#
# INFORMATIONAL (always exit 0): under --no-default-features, which crates
# still pull reqwest / tokio / tonic at depth 1, and the closure size.
# Those lines must match the out-of-scope leak list in the PR description
# (the library crates larql-inference, larql-vindex, larql-lql,
# larql-factory, larql-kv, larql-router-protocol have no `net` feature of
# their own yet). When a library crate gains `net`, update that list, never
# the hard gates.

set -u

fail=0
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# dependents <feature flags> <package>: depth-1 reverse dependencies of
# <package> in larql-cli's normal-edge graph. A package absent from the
# graph makes cargo exit non-zero; that is expected output here, not an
# error, so the status is swallowed and the text is matched instead.
dependents() {
  # shellcheck disable=SC2086 # $1 is a deliberate word-split flag list
  cargo tree --locked -p larql-cli $1 -e normal -i "$2" --depth 1 --prefix none 2>&1 || true
}

NONET="--no-default-features"
WITHNET="--no-default-features --features net"

# --- (a) larql-router absent without net -------------------------------
out=$(dependents "$NONET" larql-router)
if printf '%s\n' "$out" | grep -Eq '^larql-router v'; then
  echo "::error::larql-router is in larql-cli's graph under --no-default-features"
  printf '%s\n' "$out"
  fail=1
else
  echo "ok (a): larql-router absent under $NONET"
fi

# --- (b) larql-cli not a direct reqwest dependent without net ----------
out=$(dependents "$NONET" reqwest)
if printf '%s\n' "$out" | grep -Eq '^larql-cli v'; then
  echo "::error::larql-cli depends on reqwest directly under --no-default-features"
  printf '%s\n' "$out"
  fail=1
else
  echo "ok (b): larql-cli is not a direct reqwest dependent under $NONET"
fi

# --- (c) positive controls with net ------------------------------------
out=$(dependents "$WITHNET" larql-router)
if printf '%s\n' "$out" | grep -Eq '^larql-router v'; then
  echo "ok (c1): larql-router present under $WITHNET"
else
  echo "::error::positive control failed: larql-router absent under $WITHNET (is the net feature wired?)"
  printf '%s\n' "$out"
  fail=1
fi

out=$(dependents "$WITHNET" reqwest)
if printf '%s\n' "$out" | grep -Eq '^larql-cli v'; then
  echo "ok (c2): larql-cli is a direct reqwest dependent under $WITHNET"
else
  echo "::error::positive control failed: larql-cli is not a direct reqwest dependent under $WITHNET"
  printf '%s\n' "$out"
  fail=1
fi

# --- informational: what the library crates still pull -----------------
{
  echo "## larql-cli network closure (\`$NONET\`)"
  echo
  echo "Hard gates: $([ "$fail" -eq 0 ] && echo passed || echo FAILED)"
  echo
  for pkg in reqwest tokio tonic; do
    echo "### \`$pkg\`: still pulled by library crates (expected until they grow a \`net\` feature)"
    echo
    echo '```'
    dependents "$NONET" "$pkg"
    echo '```'
    echo
  done
  size=$(cargo tree --locked -p larql-cli $NONET -e normal --prefix none --format '{p}' 2>/dev/null | sort -u | wc -l)
  echo "Closure size under \`$NONET\`: $size packages"
} | tee -a "$SUMMARY"

exit "$fail"
