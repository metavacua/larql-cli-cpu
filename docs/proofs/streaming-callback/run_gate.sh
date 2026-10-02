#!/usr/bin/env bash
# Static gate for issue #15: every token-emitting path reachable from the CPU
# fallback of generate_streaming must receive `on_token`.
#   needs: python3 with tree-sitter + tree-sitter-rust, souffle, ast-grep
# exit 0: property holds; 1: violation (lists it); 2: tooling/cross-check failure.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=${1:-$(git -C "$HERE" rev-parse --show-toplevel)}
W=${2:-$(mktemp -d)}; rm -rf "$W/facts" "$W/out"; mkdir -p "$W/out"
python3 "$HERE/extract_facts.py" "$REPO" "$W/facts" || exit 2
souffle -F "$W/facts" -D "$W/out" "$HERE/streaming_callback_extracted.dl" || exit 2
python3 "$HERE/crosscheck_astgrep.py" "$REPO" "$W/facts" || { echo "cross-check failed: extraction is not trustworthy"; exit 2; }
echo "takes:";        cat "$W/out/takes.csv"
n=$(cat "$W/out/silent_emit.csv" "$W/out/dropped_call.csv" | wc -l)
if [ "$n" -ne 0 ]; then
  echo "VIOLATION: on_token cannot reach these token-emitting sites"
  echo "-- silent_emit (entry, fn, line)";  cat "$W/out/silent_emit.csv"
  echo "-- dropped_call (caller, callee, line)"; cat "$W/out/dropped_call.csv"
  exit 1
fi
echo "OK: on_token reaches every token-emitting path from the CPU fallback"
