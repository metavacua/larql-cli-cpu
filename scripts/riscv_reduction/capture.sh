#!/usr/bin/env bash
# Run every line of commands.txt through a runner prefix and record
# stdout, stderr and exit status per command.
# Usage: capture.sh <out-dir> <binary> [runner words...]
set -u
out=$1; bin=$2; shift 2
mkdir -p "$out"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i + 1)); id=$(printf '%03d' "$i")
  printf '%s\n' "$line" > "$out/$id.cmd"
  # shellcheck disable=SC2086
  "$@" "$bin" $line > "$out/$id.out" 2> "$out/$id.err"
  echo $? > "$out/$id.rc"
done < "$(dirname "$0")/commands.txt"
