"""Negative controls for check_ci_liveness.py: each real gap must be reported, a clean tree must pass."""

import atexit
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ci_liveness as live  # noqa: E402

PER_CHANGE_TEST = """name: {name}
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps:
      - run: {command}
"""


class Tree:
    """A throwaway workspace with crates, workflows and a policy file."""

    def __init__(self, crates, policy=None):
        self.root = Path(tempfile.mkdtemp())
        atexit.register(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "scripts").mkdir()
        (self.root / ".github/workflows").mkdir(parents=True)
        members = ", ".join(f'"crates/{c}"' for c in crates)
        (self.root / "Cargo.toml").write_text(f"[workspace]\nmembers = [{members}]\n")
        for crate in crates:
            (self.root / "crates" / crate).mkdir(parents=True)
            (self.root / "crates" / crate / "Cargo.toml").write_text(f'[package]\nname = "{crate}"\n')
        self.policy(policy or {})

    def policy(self, policy):
        (self.root / live.POLICY).write_text(json.dumps(policy))

    def workflow(self, filename, text):
        (self.root / ".github/workflows" / filename).write_text(text)

    def tested(self, filename, command):
        self.workflow(filename, PER_CHANGE_TEST.format(name=filename, command=command))

    def errors(self, default_branch=None):
        return live.static_errors(self.root, default_branch)


class StaticChecks(unittest.TestCase):
    def test_clean_tree_passes(self):
        t = Tree(["alpha", "beta"])
        t.tested("a.yml", "cargo test -p alpha -p beta")
        self.assertEqual(t.errors("main"), [])

    def test_member_with_no_workflow_is_reported(self):
        t = Tree(["alpha", "beta"])
        t.tested("a.yml", "cargo test -p alpha")
        errors = t.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("beta", errors[0])

    def test_member_only_tested_on_linux_is_reported(self):
        """Issue #6: the router ran on Linux only, so a Windows race was invisible."""
        t = Tree(["alpha"])
        t.workflow(
            "a.yml",
            "name: a\non: [push]\njobs:\n  t:\n    runs-on: ubuntu-latest\n    steps:\n      - run: cargo test -p alpha\n",
        )
        self.assertIn("windows", t.errors()[0])

    def test_filtered_test_command_does_not_count_as_coverage(self):
        t = Tree(["alpha"])
        t.tested("a.yml", "cargo test -p alpha --test one_file")
        self.assertEqual(len(t.errors()), 1)

    def test_make_target_counts_as_coverage(self):
        t = Tree(["larql-alpha"])
        t.tested("a.yml", "make larql-alpha-ci")
        self.assertEqual(t.errors(), [])

    def test_workspace_flag_covers_every_member_on_its_platforms(self):
        t = Tree(["alpha", "beta"])
        t.tested("a.yml", "cargo test --workspace")
        self.assertEqual(t.errors(), [])

    def test_workflow_that_never_triggers_per_change_gives_no_coverage(self):
        t = Tree(["alpha"])
        t.workflow(
            "a.yml",
            "name: a\non:\n  workflow_dispatch: {}\njobs:\n  t:\n    runs-on: [ubuntu-latest, windows-latest]\n    steps:\n      - run: cargo test -p alpha\n",
        )
        self.assertTrue(any("alpha" in e for e in t.errors()))

    def test_commented_out_test_command_does_not_count(self):
        t = Tree(["alpha"])
        t.workflow(
            "a.yml",
            "name: a\non: [push]\njobs:\n  t:\n    runs-on: windows-latest\n    steps:\n      # - run: cargo test -p alpha\n      - run: echo hi\n",
        )
        self.assertTrue(any("alpha" in e for e in t.errors()))

    def test_exemption_silences_a_gap(self):
        t = Tree(["alpha"], {"untested_members": {"alpha": "reason"}})
        self.assertEqual(t.errors(), [])

    def test_stale_exemption_is_an_error(self):
        t = Tree(["alpha"], {"untested_members": {"alpha": "reason"}})
        t.tested("a.yml", "cargo test -p alpha")
        self.assertIn("stale", t.errors()[0])

    def test_exemption_for_unknown_member_is_an_error(self):
        t = Tree(["alpha"], {"untested_members": {"ghost": "reason"}})
        t.tested("a.yml", "cargo test -p alpha")
        self.assertIn("not a workspace member", t.errors()[0])

    def test_dispatch_only_workflow_is_reported(self):
        """Issue #13: a gate that only a human can start is not a gate."""
        t = Tree(["alpha"])
        t.tested("a.yml", "cargo test -p alpha")
        t.workflow("manual.yml", "name: m\non:\n  workflow_dispatch: {}\njobs:\n  j:\n    steps: []\n")
        self.assertIn("manual.yml", t.errors()[0])

    def test_scheduled_only_workflow_is_not_dispatch_only(self):
        t = Tree(["alpha"])
        t.tested("a.yml", "cargo test -p alpha")
        t.workflow("cron.yml", "name: c\non:\n  schedule:\n    - cron: '0 0 * * 0'\n  workflow_dispatch: {}\njobs:\n  j:\n    steps: []\n")
        self.assertEqual(t.errors(), [])

    def test_stale_dispatch_exemption_is_an_error(self):
        t = Tree(["alpha"], {"dispatch_only_workflows": {"a.yml": "reason"}})
        t.tested("a.yml", "cargo test -p alpha")
        self.assertIn("stale", t.errors()[0])

    def test_branch_filter_that_misses_the_default_branch_is_reported(self):
        """Issue #5: quality.yml filtered on `main` while the default branch had another name."""
        t = Tree(["alpha"])
        t.tested("a.yml", "cargo test -p alpha")
        self.assertEqual(t.errors("main"), [])
        errors = t.errors("cpu-extraction")
        self.assertEqual(len(errors), 1)
        self.assertIn("cpu-extraction", errors[0])

    def test_block_style_branch_list_is_read(self):
        t = Tree(["alpha"])
        t.workflow(
            "a.yml",
            "name: a\non:\n  push:\n    branches:\n      - main\n      - 'release/*'\njobs:\n  t:\n    runs-on: [ubuntu-latest, windows-latest]\n    steps:\n      - run: cargo test -p alpha\n",
        )
        self.assertEqual(t.errors("release/1"), [])
        self.assertTrue(t.errors("other"))

    def test_inline_trigger_forms_are_read(self):
        self.assertEqual(live.triggers("name: a\non: [push, pull_request]\njobs: {}\n"), {"push", "pull_request"})
        self.assertEqual(live.triggers("name: a\non: push\njobs: {}\n"), {"push"})


class RunHistory(unittest.TestCase):
    NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def tree(self):
        t = Tree(["alpha"])
        t.workflow("cron.yml", "name: c\non:\n  schedule:\n    - cron: '0 0 * * 0'\njobs:\n  j:\n    steps: []\n")
        return t

    def info(self, state="active", created_days=200, last_days=3):
        iso = lambda days: None if days is None else (self.NOW - timedelta(days=days)).isoformat()
        return lambda _name: {
            "state": state,
            "created_at": iso(created_days),
            "last_schedule_run": iso(last_days),
        }

    def errors_for(self, **kwargs):
        return live.run_errors(self.tree().root, self.info(**kwargs), self.NOW, 45)

    def test_recent_scheduled_run_passes(self):
        self.assertEqual(self.errors_for(), [])

    def test_workflow_that_never_fired_is_reported(self):
        self.assertIn("no scheduled run", self.errors_for(last_days=None)[0])

    def test_workflow_that_stopped_firing_is_reported(self):
        self.assertIn("no scheduled run", self.errors_for(last_days=90)[0])

    def test_schedule_disabled_for_inactivity_is_reported(self):
        self.assertIn("disabled_inactivity", self.errors_for(state="disabled_inactivity")[0])

    def test_young_workflow_gets_a_grace_period(self):
        self.assertEqual(self.errors_for(created_days=5, last_days=None), [])


if __name__ == "__main__":
    unittest.main()
