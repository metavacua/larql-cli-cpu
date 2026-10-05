#!/usr/bin/env bash
# Regression suite for the binding-facts tooling: mutate a skeleton and assert
# what run_gate.sh derives. A measurement that cannot change, or passes
# vacuously, is worse than none.   needs: python3, ast-grep, souffle; lean optional
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mktemp -d); fails=0
python3 - "$HERE/tests/skeleton/executor.rs" "$W" <<'PY'
import sys, os
src = open(sys.argv[1]).read(); out = sys.argv[2]
def mutant(name, *edits):
    s = src
    for old, new in edits:
        assert s.count(old) == 1, (name, old)
        s = s.replace(old, new)
    os.makedirs(f"{out}/{name}/src", exist_ok=True); open(f"{out}/{name}/src/executor.rs", "w").write(s)
mutant("base")
# a binding-free handler starts reading the binding through a helper
mutant("show_models_reads", ("        Ok(list_dir())", "        self.require_vindex()?;\n        Ok(list_dir())"))
# vacuity: the dispatch function is renamed, so no Statement arm is found
mutant("dispatch_unrecognised", ("pub fn execute(", "pub fn run("))
PY
expect() { # name, file, needle, present(1)/absent(0)
  if grep -qP "$3" "$W/$1/work/out/$2.csv"; then got=1; else got=0; fi
  want=$([ "$4" = 1 ] && echo has || echo lacks)
  if [ "$got" = "$4" ]; then echo "PASS $1: $2 $want '$3'"
  else echo "FAIL $1: $2 should be $want '$3'"; fails=$((fails+1)); fi
}
run() { "$HERE/run_gate.sh" "$W/$1/src" "$W/$1/work" ${2:+"$2"} > "$W/$1/log" 2>&1; echo $?; }

[ "$(run base)" = 0 ] || { echo "FAIL base: gate did not exit 0"; cat "$W/base/log"; fails=$((fails+1)); }
expect base no_read '^ShowModels\t' 1
expect base no_read '^BeginPatch\t' 1
expect base no_read '^Hidden\t' 1          # known blind spot: call inside format!()
expect base no_read '^Stats\t' 0
expect base no_read '^Walk\t' 0
expect base may_raise '^Stats\t' 1
expect base may_raise '^Walk\t' 1
expect base binds '^Use\t' 1

[ "$(run show_models_reads)" = 0 ] || { echo "FAIL show_models_reads: exit"; fails=$((fails+1)); }
expect show_models_reads no_read '^ShowModels\t' 0
expect show_models_reads may_raise '^ShowModels\t' 1

rc=$(run dispatch_unrecognised)
if [ "$rc" = 2 ]; then echo "PASS dispatch_unrecognised: exit 2 (not checked)"; else echo "FAIL dispatch_unrecognised: exit $rc, want 2"; fails=$((fails+1)); fi

# compiler-backed edges close the macro blind spot
mkdir -p "$W/ir"; python3 "$HERE/lqlfacts.py" ir --demangler cat --crate larql_lql "$HERE/tests/fixture.ll" "$W/ir"
mkdir -p "$W/withir_case/src" && cp "$W/base/src/executor.rs" "$W/withir_case/src/"
[ "$(run withir_case "$W/ir")" = 0 ] || { echo "FAIL withir_case: exit"; cat "$W/withir_case/log"; fails=$((fails+1)); }
expect withir_case no_read_ir '^Hidden\t' 0
expect withir_case no_read_ir '^ShowModels\t' 1
expect withir_case astgrep_missed '^exec_hidden\tpeek' 1

# vacuity: IR that lacks a dispatched handler (e.g. demangling broke) is not a result
mkdir -p "$W/ir_partial" && cp "$W/ir/"*.facts "$W/ir_partial/" && sed -i '/^exec_use$/d' "$W/ir_partial/ir_fn.facts"
mkdir -p "$W/partial_case/src" && cp "$W/base/src/executor.rs" "$W/partial_case/src/"
rc=$(run partial_case "$W/ir_partial")
if [ "$rc" = 2 ]; then echo "PASS partial_case: exit 2 (handler missing from IR)"; else echo "FAIL partial_case: exit $rc, want 2"; fails=$((fails+1)); fi

# Lean: the certificates for derived claims check, and a forged claim is rejected
if command -v lean >/dev/null; then
  if [ -f "$W/base/work/lean/Binding.lean" ] && lean -j 1 "$W/base/work/lean/Binding.lean" >/dev/null 2>&1; then echo "PASS lean: derived certificates check"
  else echo "FAIL lean: derived certificates do not check"; fails=$((fails+1)); fi
  python3 "$HERE/lqlfacts.py" lean "$W/base/work/facts" "$W/forged.lean" --claim exec_stats >/dev/null
  if [ ! -s "$W/forged.lean" ]; then echo "FAIL lean: forged certificate was not generated"; fails=$((fails+1))
  elif lean -j 1 "$W/forged.lean" >/dev/null 2>&1; then echo "FAIL lean: forged claim for exec_stats was accepted"; fails=$((fails+1))
  else echo "PASS lean: forged claim rejected"; fi
elif [ "${REQUIRE_LEAN:-0}" = 1 ]; then echo "FAIL lean: not installed and REQUIRE_LEAN=1"; fails=$((fails+1))
else echo "SKIP lean: not installed"; fi

python3 -m unittest discover -s "$HERE/tests" -p 'test_*.py' -q 2>&1 | tail -3 || true
python3 -m unittest discover -s "$HERE/tests" -p 'test_*.py' -q >/dev/null 2>&1 || { echo "FAIL unit tests"; fails=$((fails+1)); }
echo "failures: $fails"; [ "$fails" = 0 ]
