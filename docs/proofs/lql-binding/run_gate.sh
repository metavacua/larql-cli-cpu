#!/usr/bin/env bash
# Which LQL statements depend on the session binding (USE)?  Issues #44, #46.
#   run_gate.sh SRC_DIR WORK_DIR [IR_FACTS_DIR]
# SRC_DIR is larql-lql's src/. With IR_FACTS_DIR (from `lqlfacts.py ir`), the
# compiler's call edges are added and compared with the syntactic ones.
#   needs: python3, ast-grep, souffle; lean (required when REQUIRE_LEAN=1)
# exit 0: facts derived (and certificates checked); 2: tooling failure or
# vacuous extraction. The derived tables are a measurement, not a verdict.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$1; W=$2; IR=${3:-}
rm -rf "$W"; mkdir -p "$W/out" "$W/lean"
python3 "$HERE/lqlfacts.py" astgrep "$SRC" "$W/facts" || exit 2
for f in dispatch reads_backend; do
  [ -s "$W/facts/$f.facts" ] || { echo "NOT CHECKED: no $f facts extracted (rule or code shape changed?)"; exit 2; }
done
souffle -F "$W/facts" -D "$W/out" "$HERE/binding.dl" || exit 2
# Partition of the code by statement class (decisions/statement_class.facts is the
# recorded decision; CLASS_FACTS overrides it, e.g. for the skeleton tests).
cp "${CLASS_FACTS:-$HERE/decisions/statement_class.facts}" "$W/facts/statement_class.facts"
cp "$HERE/decisions/noise.facts" "$W/facts/noise.facts"
souffle -F "$W/facts" -D "$W/out" "$HERE/partition.dl" || exit 2
if [ -n "$IR" ]; then
  [ -s "$IR/calls.facts" ] || { echo "NOT CHECKED: IR facts are empty"; exit 2; }
  missing=$(cut -f2 "$W/facts/dispatch.facts" | sort -u | comm -23 - <(sort -u "$IR/ir_fn.facts"))
  [ -z "$missing" ] || { echo "NOT CHECKED: handlers absent from the IR (demangling or codegen changed?):"; echo "$missing"; exit 2; }
  mkdir -p "$W/facts_ir" "$W/out_ir"; cp "$W/facts/"*.facts "$W/facts_ir/"; cp "$IR/calls.facts" "$W/facts_ir/ir_calls.facts"
  souffle -F "$W/facts_ir" -D "$W/out_ir" "$HERE/binding.dl" || exit 2
  cp "$W/out_ir/no_read.csv" "$W/out/no_read_ir.csv"; cp "$W/out_ir/astgrep_missed.csv" "$W/out/astgrep_missed.csv"
  cp "$IR/indirect.facts" "$W/out/indirect.csv" 2>/dev/null || true
  LEANFACTS="$W/facts_ir"
else LEANFACTS="$W/facts"; fi
python3 "$HERE/lqlfacts.py" lean "$LEANFACTS" "$W/lean/Binding.lean" || exit 2
if command -v lean >/dev/null; then
  lean -j 1 "$W/lean/Binding.lean" || { echo "LEAN REJECTED the certificates"; exit 2; }
  echo "lean: certificates checked"
elif [ "${REQUIRE_LEAN:-0}" = 1 ]; then echo "NOT CHECKED: lean missing"; exit 2; fi
show() { echo "-- $1"; sed 's/\t/  /' "$W/out/$1.csv"; }
show side_total; show unclassified_statement; show recursive; show no_read; show binds; show reads_never_raises; show reaches_use; show may_raise
if [ -n "$IR" ]; then show no_read_ir; show astgrep_missed; show indirect; fi
exit 0
