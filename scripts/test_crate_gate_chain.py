#!/usr/bin/env python3
# /// script
# dependencies = ["pyyaml"]
# ///
"""The order a crate is gated in, per target.

  fmt                                  once per run; a failure ends the run
  clippy                               per target; a failure ends that target
  build (cargo build --lib)            per target, only after its clippy; a failure ends it
  lql-strategy-matrix                  per target, only after its build, and only where the
                                       CLI can run (the freestanding targets have no OS)

Each target family is its own wrapper workflow (so its own run: artifact names and
concurrency groups cannot collide between two lql instances), sharing crate-gate-run.yml
(resolve + fmt + a matrix over targets) and crate-gate-chain.yml (clippy -> build -> lql).

Run: uv run scripts/test_crate_gate_chain.py
"""

import json
import unittest

from test_workflow_triggers import GATES, load

FREESTANDING = {
    "wasm32v1-none", "riscv32i-unknown-none-elf", "riscv32im-unknown-none-elf",
    "riscv32imac-unknown-none-elf", "riscv32imafc-unknown-none-elf",
    "riscv32imc-unknown-none-elf", "riscv64gc-unknown-none-elf",
    "riscv64imac-unknown-none-elf", "aarch64-unknown-none", "x86_64-unknown-none",
    "aarch64-unknown-uefi", "x86_64-unknown-uefi",
}
GNU = {
    "crate-gate-x86_64-linux-gnu": "x86_64-unknown-linux-gnu",
    "crate-gate-aarch64-linux-gnu": "aarch64-unknown-linux-gnu",
    "crate-gate-x86_64-windows-gnu": "x86_64-pc-windows-gnu",
    "crate-gate-riscv64gc-linux-gnu": "riscv64gc-unknown-linux-gnu",
}
# The gnu targets whose CLI the lql matrix can run today: native runners.
LQL_RUNNERS = {"x86_64-unknown-linux-gnu": "ubuntu-latest", "aarch64-unknown-linux-gnu": "ubuntu-24.04-arm"}


def needs(job):
    n = job.get("needs", [])
    return {n} if isinstance(n, str) else set(n)


def targets_of(wrapper):
    return json.loads(load()[wrapper]["jobs"]["gate"]["with"]["targets"])


class Order(unittest.TestCase):
    def test_chain_is_clippy_then_build_then_lql(self):
        jobs = load()["crate-gate-chain"]["jobs"]
        self.assertEqual(needs(jobs["clippy"]), set())
        self.assertEqual(needs(jobs["build"]), {"clippy"})
        self.assertEqual(needs(jobs["lql"]), {"build"})

    def test_lql_runs_only_where_the_cli_can_run(self):
        lql = load()["crate-gate-chain"]["jobs"]["lql"]
        self.assertIn("inputs.lql", lql["if"])
        self.assertEqual(lql["uses"], "./.github/workflows/lql-strategy-matrix.yml")
        self.assertIs(lql["with"]["strict"], True)

    def test_run_is_fmt_first_and_every_target_waits_for_it(self):
        jobs = load()["crate-gate-run"]["jobs"]
        self.assertEqual(needs(jobs["fmt"]), {"resolve"})
        self.assertEqual(needs(jobs["chain"]), {"resolve", "fmt"})
        self.assertEqual(jobs["chain"]["uses"], "./.github/workflows/crate-gate-chain.yml")
        self.assertIs(jobs["chain"]["strategy"]["fail-fast"], False)  # one red target hides no other


class Targets(unittest.TestCase):
    def test_every_gate_is_a_thin_wrapper_over_the_shared_run(self):
        for name in GATES:
            with self.subTest(gate=name):
                job = load()[name]["jobs"]["gate"]
                self.assertEqual(job["uses"], "./.github/workflows/crate-gate-run.yml")

    def test_freestanding_targets_are_gated_without_lql(self):
        got = targets_of("crate-gate")
        self.assertEqual({t["target"] for t in got}, FREESTANDING)
        for t in got:
            with self.subTest(target=t["target"]):
                self.assertFalse(t["lql"], "no OS to run the CLI on")
                self.assertEqual(t["features"], "--no-default-features")
                self.assertIn("std_instead_of_core", t["clippy_args"])

    def test_gnu_targets_run_default_features_and_lql_where_runnable(self):
        for wrapper, target in GNU.items():
            with self.subTest(target=target):
                (t,) = targets_of(wrapper)
                self.assertEqual(t["target"], target)
                self.assertEqual(t["features"], "")
                self.assertEqual(t["lql"], target in LQL_RUNNERS)
                if t["lql"]:
                    self.assertEqual(t["runs_on"], LQL_RUNNERS[target])

    def test_lql_matrix_takes_its_runner_as_an_input(self):
        wf = load()["lql-strategy-matrix"]
        self.assertIn("runs_on", wf[True]["workflow_call"]["inputs"])
        for name, job in wf["jobs"].items():
            if "runs-on" in job:
                with self.subTest(job=name):
                    self.assertIn("inputs.runs_on", job["runs-on"])


if __name__ == "__main__":
    unittest.main()
