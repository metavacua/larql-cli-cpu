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

import re
import unittest
from pathlib import Path

from test_workflow_triggers import GATES, WORKFLOWS, load

# Tier 2 without-host-tools freestanding targets.
#
# The 32-bit riscv ladder is a controlled elimination, not a performance choice. Each rung adds
# capability; a pass-to-fail step names what the code actually depends on, so float that
# compiles on a soft-float rung is float that is used but not required:
#     riscv32i  ->  riscv32im  ->  riscv32imac  ->  riscv32imafc
#                      +M           +A +C            +F
# riscv32imac is the control for riscv32imafc: they differ by F alone. Without it, the step
# from im to imafc adds A, C and F together. No riscv target has F without C, so F can only
# be isolated against imac. imc (C without A) and riscv64imac are trimmed for now.
FREESTANDING = {
    "wasm32v1-none", "riscv32i-unknown-none-elf", "riscv32im-unknown-none-elf",
    "riscv32imac-unknown-none-elf", "riscv32imafc-unknown-none-elf",
    "riscv64gc-unknown-none-elf", "aarch64-unknown-none", "x86_64-unknown-none",
    "aarch64-unknown-uefi", "x86_64-unknown-uefi",
}
TRIMMED = {
    "riscv32imc-unknown-none-elf", "riscv64imac-unknown-none-elf", "x86_64-pc-windows-gnu",
}
LADDER = [
    "riscv32i-unknown-none-elf", "riscv32im-unknown-none-elf",
    "riscv32imac-unknown-none-elf", "riscv32imafc-unknown-none-elf",
]
# The gnu targets, and where the lql matrix runs the CLI: natively, or under QEMU user-mode.
GNU = {
    "x86_64-unknown-linux-gnu": "ubuntu-latest",
    "aarch64-unknown-linux-gnu": "ubuntu-24.04-arm",
    "riscv64gc-unknown-linux-gnu": "ubuntu-latest",
}
EMULATED = {"riscv64gc-unknown-linux-gnu"}
RISCV_SETUP = "scripts/ci/riscv64gc_qemu.sh"


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

    def test_the_riscv32_ladder_is_complete_and_in_order(self):
        # i -> im -> imac -> imafc, so imac stays as the one-extension control for F.
        got = [t["target"] for t in targets() if t["target"] in LADDER]
        self.assertEqual(got, LADDER)

    def test_trimmed_targets_are_not_gated(self):
        self.assertEqual({t["target"] for t in targets()} & TRIMMED, set())

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
                self.assertIs(t["lql"], True)
                self.assertEqual(t["runs_on"], runner)
                if target in EMULATED:
                    self.assertEqual(t["setup"], RISCV_SETUP)
                    self.assertIn("qemu-riscv64", t["emulator"])
                else:
                    self.assertEqual((t["setup"], t["emulator"]), ("", ""))


class Emulation(unittest.TestCase):
    """riscv64gc-unknown-linux-gnu runs the lql matrix under QEMU user-mode."""

    def test_the_chain_passes_the_cross_build_and_emulator_to_lql(self):
        chain = load()["crate-gate-chain"]
        for name in ("setup", "emulator"):
            self.assertIn(name, chain[True]["workflow_call"]["inputs"])
        lql = chain["jobs"]["lql"]["with"]
        self.assertIn("inputs.emulator", lql["target"])  # cross-build only when emulated
        self.assertEqual(lql["setup"], "${{ inputs.setup }}")
        self.assertEqual(lql["emulator"], "${{ inputs.emulator }}")

    def test_clippy_and_build_get_the_cross_toolchain_when_there_is_a_setup(self):
        jobs = load()["crate-gate-chain"]["jobs"]
        for name in ("clippy", "build"):
            with self.subTest(job=name):
                setup = [s for s in jobs[name]["steps"] if "inputs.setup" in s.get("if", "")]
                self.assertEqual(len(setup), 1)
                self.assertEqual(setup[0]["env"]["SETUP"], "${{ inputs.setup }}")

    def test_the_matrix_workflow_cross_builds_and_wraps_the_binary(self):
        wf = load()["lql-strategy-matrix"]
        for name in ("target", "setup", "emulator"):
            self.assertIn(name, wf[True]["workflow_call"]["inputs"])
        build = wf["jobs"]["build"]["steps"]
        self.assertTrue(any("inputs.setup" in s.get("if", "") for s in build))
        # The target reaches the shell through the environment, never interpolated into the script.
        cross = [s for s in build if "--target" in str(s.get("run", ""))]
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["env"]["TARGET"], "${{ inputs.target }}")
        self.assertNotIn("inputs.", cross[0]["run"])
        up = [s for s in build if "upload-artifact" in s.get("uses", "")][0]
        self.assertIn("inputs.target", up["with"]["path"])

    def test_every_job_that_runs_the_cli_runs_it_under_the_emulator(self):
        # The jobs that download the binary and execute it: matrix legs, model lifecycle, mistral.
        for name, job in load()["lql-strategy-matrix"]["jobs"].items():
            steps_ = job.get("steps", [])
            if not any(str(s.get("uses", "")).startswith("actions/download-artifact") and
                       str(s["with"].get("name", "")).startswith("larql-bin") for s in steps_):
                continue
            with self.subTest(job=name):
                wrap = [s for s in steps_ if "wrap_emulator.sh" in str(s.get("run", ""))]
                self.assertEqual(len(wrap), 1)
                self.assertEqual(wrap[0]["if"], "inputs.emulator != ''")
                self.assertEqual(wrap[0]["env"]["EMULATOR"], "${{ inputs.emulator }}")

    def test_the_setup_script_installs_the_recipe_target_matrix_proved(self):
        text = Path("scripts/ci/riscv64gc_qemu.sh").read_text()
        for needle in ("gcc-riscv64-linux-gnu", "libc6-dev-riscv64-cross", "qemu-user-static",
                       "libopenblas-dev:riscv64", "OPENBLAS_LIB_DIR_riscv64gc_unknown_linux_gnu",
                       "CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER", "QEMU_LD_PREFIX"):
            self.assertIn(needle, text)

    def test_the_wrapper_execs_the_emulator_on_an_absolute_path(self):
        text = Path("scripts/ci/wrap_emulator.sh").read_text()
        self.assertIn("larql.real", text)
        self.assertIn('$(pwd)', text)  # cells run in other directories


class CrossTarget(unittest.TestCase):
    """The structural comparison across targets: read-only, so it needs no write permission."""

    def job(self):
        return load()["crate-gate"]["jobs"]["cross-target"]

    def test_runs_after_resolve_and_chain_even_when_a_chain_failed(self):
        job = self.job()
        self.assertEqual(needs(job), {"resolve", "chain"})
        self.assertIn("!cancelled()", job["if"])
        self.assertIn("needs.resolve.outputs.crate != ''", job["if"])
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertLessEqual(job["timeout-minutes"], 10)

    def test_downloads_every_results_artifact_and_runs_the_comparison_strictly(self):
        job_steps = self.job()["steps"]
        dl = [s for s in job_steps if str(s.get("uses", "")).startswith("actions/download-artifact@v4")]
        self.assertEqual(len(dl), 1)
        self.assertEqual(dl[0]["with"]["pattern"], "results-*")
        self.assertEqual(dl[0]["with"]["path"], "artifacts")
        run = "\n".join(str(s.get("run", "")) for s in job_steps)
        self.assertIn("scripts/lql_matrix/cross_target.py", run)
        self.assertIn("--strict", run)
        self.assertIn("artifacts/results-*/results-*.jsonl", run)
        self.assertIn("GITHUB_STEP_SUMMARY", run)

    def test_the_qemu_target_is_the_only_explicit_slow_tag(self):
        wf = load()["crate-gate"]
        run = "\n".join(str(s.get("run", "")) for s in self.job()["steps"])
        self.assertEqual(re.findall(r"--slow-tag\s+(\S+)", run), ["riscv64gc-unknown-linux-gnu"])
        # the slow tag is exactly the target the matrix runs under an emulator, nothing implicit
        emulated = [t["target"] for t in wf["jobs"]["chain"]["strategy"]["matrix"]["t"]
                    if t.get("emulator")]
        self.assertEqual(emulated, ["riscv64gc-unknown-linux-gnu"])
        text = Path(".github/workflows/crate-gate.yml").read_text()
        self.assertIn("QEMU", text[text.index("cross-target:") - 2500:text.index("cross-target:")])

    def test_every_target_that_runs_lql_is_expected(self):
        # An instance is otherwise inferred from the artifacts that exist, so a target whose whole
        # lql run uploaded nothing would be invisible. The expected tags are exactly the lql targets.
        wf = load()["crate-gate"]
        run = "\n".join(str(s.get("run", "")) for s in self.job()["steps"])
        lql_targets = [t["target"] for t in wf["jobs"]["chain"]["strategy"]["matrix"]["t"] if t["lql"]]
        self.assertEqual(sorted(re.findall(r"--expect-tag\s+(\S+)", run)), sorted(lql_targets))

    def test_the_summary_is_appended_even_when_the_comparison_fails(self):
        run = "\n".join(str(s.get("run", "")) for s in self.job()["steps"])
        self.assertLess(run.index("cross_target.py"), run.index("GITHUB_STEP_SUMMARY"))
        self.assertIn("exit $status", run)  # the script's own status is what the job reports

    def test_no_write_permission_anywhere(self):
        wf = load()["crate-gate"]
        self.assertEqual(wf["permissions"], {"contents": "read"})
        self.assertNotIn("permissions", self.job())
        for name, job in wf["jobs"].items():
            with self.subTest(job=name):
                self.assertNotIn("permissions", job)
        for step in self.job()["steps"]:
            self.assertNotIn("github.token", str(step))
            self.assertNotIn("GITHUB_TOKEN", str(step.get("env", "")))


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

    def test_the_gate_trims_the_quality_job_and_main_keeps_it(self):
        # The gate already does fmt and clippy per crate; quality repeats them and adds the
        # LQL stack's cargo test. Skipped only when a caller passes quality=false: a push to
        # main or a dispatch has no such input and must still run it.
        wf = load()["lql-strategy-matrix"]
        self.assertIs(wf[True]["workflow_call"]["inputs"]["quality"]["default"], True)
        self.assertEqual(wf["jobs"]["quality"]["if"], "format('{0}', inputs.quality) != 'false'")
        self.assertIs(load()["crate-gate-chain"]["jobs"]["lql"]["with"]["quality"], False)

    def test_nothing_waits_on_quality(self):
        # It runs alongside the matrix, never as a dependency of it, so trimming it skips nothing.
        for name, job in load()["lql-strategy-matrix"]["jobs"].items():
            with self.subTest(job=name):
                self.assertNotIn("quality", needs(job))

    def test_concurrency_group_is_per_instance(self):
        self.assertIn("inputs.tag", load()["lql-strategy-matrix"]["concurrency"]["group"])


if __name__ == "__main__":
    unittest.main()
