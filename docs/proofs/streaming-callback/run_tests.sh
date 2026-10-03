#!/usr/bin/env bash
# Regression suite for the proof tooling itself: mutate a skeleton and assert
# what run_gate.sh says. A gate that cannot fail, or that passes vacuously, is
# worse than no gate.  needs: python3 (tree-sitter, tree-sitter-rust), souffle, ast-grep
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mktemp -d); fails=0
python3 - "$HERE/tests/skeleton.rs" "$W" <<'PY'
import sys, os
src = open(sys.argv[1]).read(); out = sys.argv[2]
def mutant(name, *edits):
    s = src
    for old, new in edits:
        assert s.count(old) == 1, (name, old)
        s = s.replace(old, new)
    os.makedirs(f"{out}/{name}", exist_ok=True); open(f"{out}/{name}/skel.rs", "w").write(s)
mutant("fixed")
# the original bug: the CPU path takes no callback at all
mutant("bug",
  ("return via_cpu(&mut on_token);", "return via_cpu();"),
  ("fn via_cpu(on_token: &mut impl FnMut(u32, &str, f64)) -> R {", "fn via_cpu() -> R {"),
  ("emit(&mut tokens, on_token, 1, (\"a\".into(), 0.5));", "tokens.push((\"a\".into(), 0.5));"))
# hole 1 (vacuity): same behaviour, but the guard is behind a helper the extractor cannot see through
mutant("guard_shape_unrecognised", ("if !backend_supports_fused(backend) {", "if lacks_fused(backend) {"))
# hole 2 (sufficiency): the callback reaches the emitter, which never calls it
mutant("callback_never_invoked", ("    on_token(id, &p.0, p.1);\n", "    let _ = (&on_token, id);\n"))
# partial fix: a throwaway closure is passed instead of the callback
mutant("callback_replaced", ("emit(&mut tokens, on_token, 1,", "emit(&mut tokens, &mut |_, _, _| {}, 1,"))
# aliasing: the callback is forwarded under another name (a known, conservative false positive)
mutant("callback_aliased", ("emit(&mut tokens, on_token, 1,", "let cb = on_token; emit(&mut tokens, cb, 1,"))
PY
expect() {  # name expected-exit expected-message-regex
  rm -rf "$W/w-$1"
  STREAMING_GATE_FILES="skel.rs" "$HERE/run_gate.sh" "$W/$1" "$W/w-$1" >"$W/$1.out" 2>&1; got=$?
  # the exit code alone is not enough: a crash also exits 2, and must not pass as "vacuity detected"
  if [ "$got" = "$2" ] && grep -Eq "$3" "$W/$1.out"; then echo "OK   $1: exit $got, \"$(grep -Eo "$3" "$W/$1.out" | head -1)\""
  else echo "FAIL $1: expected exit $2 with /$3/, got exit $got"; sed 's/^/     | /' "$W/$1.out" | tail -12; fails=$((fails+1)); fi
}
expect fixed 0 'OK: on_token reaches'                              # property holds
expect bug 1 'VIOLATION'                                           # violation reported
expect guard_shape_unrecognised 2 'NOT CHECKED: no guarded'        # vacuity is an error, not a pass
expect callback_never_invoked 1 'generate_streaming.emit.[0-9]+'  # the emitter is named: passing the callback is not calling it
expect callback_replaced 1 'VIOLATION'                             # dropped_call
expect callback_aliased 1 'VIOLATION'                              # conservative: aliasing is flagged for review
exit $((fails > 0))
