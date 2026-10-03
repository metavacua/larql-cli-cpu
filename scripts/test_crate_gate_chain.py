#!/usr/bin/env python3
# /// script
# dependencies = ["pyyaml"]
# ///
"""The order a crate is gated in, and what is shared between targets.

One workflow run per crate (crate-gate.yml):

  fmt                                  ONCE per crate, not per target; a failure ends the run
  per target (16, in parallel):
    clippy                             target-specific lints, so per target; a failure ends it
    build (cargo build --lib)          only after that target's clippy
    lql-strategy-matrix                only after that target's build, and only where the CLI
                                       can run (the freestanding targets have no OS)

Two lql instances can run in one workflow run (x86_64 and aarch64), so everything an
instance shares with the run -- artifact names, download globs, the concurrency group --
carries a per-instance tag, or the second instance would read or cancel the first.

Run: uv run scripts/test_crate_gate_chain.py
"""

import unittest
from pathlib import Path

from test_workflow_triggers import GATES, WORKFLOWS, load

FREESTANDING = {
    "wasm32v1-none", "riscv32i-unknown-none-elf", "riscv32im-unknown-none-elf",
    "riscv32imac-unknown-none-elf", "riscv32imafc-unknown-none-elf",
    "riscv32imc-unknown-none-elf", "riscv64gc-unknown-none-elf",
    "riscv64imac-unknown-none-elf", "aarch64-unknown-none", "x86_64-unknown-none",
    "aarch64-unknown-uefi", "x86_64-unknown-uefi",
}
# The gnu targets, and the runner the lql matrix can use for them today (None: not wired).
GNU = {
    "x86_64-unknown-linux-gnu": "ubuntu-latest",
    "aarch64-unknown-linux-gnu": "ubuntu-24.04-arm",
    "x86_64-pc-windows-gnu": None,
    "riscv64gc-unknown-linux-gnu": None,
}


def needs(job):
    n = job.get("needs", [])
    return {n} if isinstance(n, str) else set(n)


def steps(workflow):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


def targets():
    return load()["crate-gate"]["jobs"]["chain"]["strategy"]["matrix"]["t"]


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
        self.assertIn("inputs.target", lql["with"]["tag"])

    def test_fmt_runs_once_per_crate_and_every_target_waits_for_it(self):
        jobs = load()["crate-gate"]["jobs"]
        self.assertEqual(needs(jobs["fmt"]), {"resolve"})
        self.assertEqual(needs(jobs["chain"]), {"resolve", "fmt"})
        self.assertEqual(jobs["chain"]["uses"], "./.github/workflows/crate-gate-chain.yml")
        self.assertIs(jobs["chain"]["strategy"]["fail-fast"], False)  # a red target hides no other

    def test_fmt_appears_exactly_once_in_the_whole_gate(self):
        hits = [
            (path.name, step.get("run"))
            for path in sorted(WORKFLOWS.glob("crate-gate*.yml"))
            for step in steps(load()[path.stem])
            if "cargo fmt" in str(step.get("run", ""))
        ]
        self.assertEqual(len(hits), 1, hits)
        self.assertEqual(hits[0][0], "crate-gate.yml")


class Targets(unittest.TestCase):
    def test_one_gate_workflow_and_no_per_family_wrappers(self):
        self.assertEqual({p.stem for p in WORKFLOWS.glob("crate-gate*.yml")}, GATES | {"crate-gate-chain"})

    def test_every_selected_target_is_gated(self):
        self.assertEqual({t["target"] for t in targets()}, FREESTANDING | set(GNU))

    def test_freestanding_targets_have_no_lql_and_the_no_std_lints(self):
        for t in targets():
            if t["target"] not in FREESTANDING:
                continue
            with self.subTest(target=t["target"]):
                self.assertFalse(t["lql"], "no OS to run the CLI on")
                self.assertEqual(t["features"], "--no-default-features")
                self.assertIn("std_instead_of_core", t["clippy_args"])

    def test_gnu_targets_use_default_features_and_lql_where_runnable(self):
        got = {t["target"]: t for t in targets()}
        for target, runner in GNU.items():
            with self.subTest(target=target):
                t = got[target]
                self.assertEqual(t["features"], "")
                self.assertEqual(t["lql"], runner is not None)
                if runner:
                    self.assertEqual(t["runs_on"], runner)


class LqlInstancesDoNotCollide(unittest.TestCase):
    def artifact_steps(self):
        for step in steps(load()["lql-strategy-matrix"]):
            if str(step.get("uses", "")).startswith("actions/") and "artifact" in step["uses"]:
                yield step

    def test_takes_its_runner_and_a_tag_as_inputs(self):
        wf = load()["lql-strategy-matrix"]
        for name in ("runs_on", "tag"):
            self.assertIn(name, wf[True]["workflow_call"]["inputs"])
        for name, job in wf["jobs"].items():
            if "runs-on" in job:
                with self.subTest(job=name):
                    self.assertIn("inputs.runs_on", job["runs-on"])

    def test_every_artifact_name_carries_the_tag(self):
        for step in self.artifact_steps():
            with self.subTest(step=step.get("name", step["with"])):
                self.assertIn("env.TAG", step["with"].get("name", step["with"].get("pattern", "")))

    def test_downloading_everything_is_filtered_to_this_instance(self):
        for step in self.artifact_steps():
            if "download" in step["uses"] and "name" not in step["with"]:
                with self.subTest(step=step.get("name", step["with"])):
                    self.assertIn("env.TAG", step["with"].get("pattern", ""))

    def test_concurrency_group_is_per_instance(self):
        self.assertIn("inputs.tag", load()["lql-strategy-matrix"]["concurrency"]["group"])


if __name__ == "__main__":
    unittest.main()
