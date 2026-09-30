#!/usr/bin/env python3
"""Check a no_std/freestanding-target result against a per-crate policy file.

Companion to check_coverage_policy.py, same convention: a checked-in JSON
policy declares the CURRENTLY EXPECTED state, a script asserts the real
result matches it, and improving the real result requires updating the
policy file as part of that same change -- never the other way around.

Unlike coverage (a number to compare against a floor), a no_std/freestanding
check has exactly two states per (crate, target): does `cargo check --lib`
(or a `cargo clippy --fix` + clean-diff round trip) succeed against that
target today. The policy file's `targets` map declares which state is
CURRENTLY EXPECTED for each target this crate is checked against:

  "expected-fail" -- no `#![no_std]` port has landed yet for this crate.
                     A failing result here is the honest, unsurprising
                     baseline, not a defect in this workflow.
  "expected-pass" -- the crate is a proven, gated no_std/freestanding
                     build for this target. A failing result here IS a
                     regression and must fail the calling step.

A crate passing on a target the policy still marks "expected-fail" is
NOT failed -- it is good news, printed loudly (`::warning::`) so it is
impossible to miss in a job log, with an instruction to promote that
target's entry to "expected-pass" as part of landing the change that
made it pass. That promotion is the mechanism: the policy file only ever
tightens, one target at a time, exactly like coverage-policy.json's
per-file floors only ever ratchet up.

Usage:
    check_no_std_policy.py <crate> <target> <exit-code> <check-name> <policy-file>

Exit status: 0 if the result is consistent with (or better than) the
policy; 1 on a genuine regression (declared expected-pass, but failed)
or a target the policy file does not mention at all (a policy file must
be exhaustive over what it is asked about, the same way DOC-GATE-1
treats an empty corpus as a failure rather than a vacuous pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_FAIL = "expected-fail"
EXPECTED_PASS = "expected-pass"
VALID_STATES = (EXPECTED_FAIL, EXPECTED_PASS)


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: check_no_std_policy.py <crate> <target> <exit-code> "
            "<check-name> <policy-file>",
            file=sys.stderr,
        )
        return 2

    crate, target, exit_code_str, check_name, policy_path_str = sys.argv[1:]
    try:
        exit_code = int(exit_code_str)
    except ValueError:
        print(f"::error::exit-code argument {exit_code_str!r} is not an integer", file=sys.stderr)
        return 2

    policy_path = Path(policy_path_str)
    if not policy_path.is_file():
        print(f"::error::no policy file at {policy_path} -- every crate this workflow checks needs one", file=sys.stderr)
        return 1

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    targets = policy.get("targets", {})
    expected = targets.get(target)

    if expected is None:
        print(
            f"::error::{policy_path} has no entry for target {target!r} -- "
            "a policy file must declare every target it is checked against, "
            "the same way an empty DOC-GATE-1 corpus is a failure rather "
            "than a vacuous pass",
            file=sys.stderr,
        )
        return 1
    if expected not in VALID_STATES:
        print(
            f"::error::{policy_path} target {target!r} has invalid state "
            f"{expected!r} -- must be one of {VALID_STATES}",
            file=sys.stderr,
        )
        return 1

    passed = exit_code == 0

    if expected == EXPECTED_PASS and not passed:
        print(
            f"::error::{crate} regressed on {target} ({check_name}): "
            f"policy declares expected-pass, but this run exited {exit_code}. "
            "This is a real regression, not an unported-crate baseline -- "
            "the no_std port for this (crate, target) pair was already proven "
            "and something broke it.",
            file=sys.stderr,
        )
        return 1

    if expected == EXPECTED_FAIL and passed:
        print(
            f"::warning::{crate} NOW PASSES {check_name} on {target}, but "
            f"{policy_path} still declares expected-fail for it. This is "
            "good news, not a failure -- promote this target's entry to "
            '"expected-pass" in that file as part of the change that made '
            "it pass, so a future regression here is caught rather than "
            "silently re-absorbed as \"still unported\"."
        )
        return 0

    # expected-fail + failed (the honest, unsurprising baseline), or
    # expected-pass + passed (a proven port holding steady): both fine.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
