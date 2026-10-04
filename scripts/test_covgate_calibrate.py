"""Tests for covgate.calibrate, against synthetic git histories whose
answers are known exactly. Run: python3 -m unittest discover -s scripts -p test_covgate_calibrate.py"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate.calibrate.escapes import Escape, escapes  # noqa: E402
from covgate.calibrate.false_reds import Verdict, flips  # noqa: E402

HOUR = 3600
T0 = 1_700_000_000


class Repo:
    """A throwaway git repo with explicit commit timestamps."""

    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")

    def git(self, *args: str, at: int = T0) -> str:
        env = dict(os.environ, GIT_AUTHOR_DATE=f"@{at} +0000", GIT_COMMITTER_DATE=f"@{at} +0000")
        return subprocess.run(["git", *args], cwd=self.root, env=env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, path: str, text: str, subject: str, at: int) -> str:
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        self.git("add", path, at=at)
        self.git("commit", "-q", "-m", subject, at=at)
        return self.git("rev-parse", "HEAD")


def lines(*xs: str) -> str:
    return "\n".join(xs) + "\n"


class Escapes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Repo(Path(self.tmp.name))
        self.r = self.repo
        self.base = self.r.commit("crates/x/src/a.rs", lines("fn a() {}", "fn b() {}", "fn c() {}"), "feat: start", T0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_fix_is_blamed_on_the_change_that_wrote_the_line_it_repairs(self):
        bad = self.r.commit("crates/x/src/a.rs", lines("fn a() {}", "fn b() { bug }", "fn c() {}"), "feat: b", T0 + 10 * HOUR)
        fix = self.r.commit("crates/x/src/a.rs", lines("fn a() {}", "fn b() {}", "fn c() {}"), "fix: b", T0 + 34 * HOUR)
        found, changes = escapes(self.r.root, "main")
        self.assertEqual(found, [Escape(fix, bad, 24.0)])
        self.assertEqual(changes, 2)  # the two non-fix changes

    def test_a_fix_that_lands_through_a_merge_is_found(self):
        bad = self.r.commit("crates/x/src/a.rs", lines("fn a() { bug }", "fn b() {}", "fn c() {}"), "feat: a", T0 + 1 * HOUR)
        self.r.git("switch", "-q", "-c", "topic")
        fix = self.r.commit("crates/x/src/a.rs", lines("fn a() {}", "fn b() {}", "fn c() {}"), "fix: a", T0 + 5 * HOUR)
        self.r.git("switch", "-q", "main")
        self.r.commit("crates/x/src/other.rs", lines("x"), "feat: other", T0 + 6 * HOUR)
        self.r.git("merge", "-q", "--no-ff", "-m", "Merge topic", "topic", at=T0 + 7 * HOUR)
        found, _ = escapes(self.r.root, "main")
        self.assertEqual(found, [Escape(fix, bad, 4.0)])

    def test_a_pure_addition_fix_blames_nothing(self):
        # A fix that only ADDS lines repaired no existing line: there is
        # nothing to attribute, and guessing would invent an escape.
        self.r.commit("crates/x/src/a.rs", lines("fn a() {}", "fn b() {}", "fn c() {}", "fn d() {}"), "fix: add d", T0 + HOUR)
        self.assertEqual(escapes(self.r.root, "main")[0], [])

    def test_only_restricts_attribution_to_one_crate(self):
        bad = self.r.commit("crates/y/src/z.rs", lines("bug"), "feat: y", T0 + HOUR)
        self.r.commit("crates/y/src/z.rs", lines("ok"), "fix: y", T0 + 2 * HOUR)
        self.assertEqual(escapes(self.r.root, "main", only="crates/x")[0], [])
        self.assertEqual([e.introduced for e in escapes(self.r.root, "main", only="crates/y")[0]], [bad])


def v(workflow: str, sha: str, conclusion: str, hour: float) -> Verdict:
    return Verdict(workflow, sha, conclusion, T0 + hour * HOUR)


class FalseReds(unittest.TestCase):
    def test_a_failure_then_success_on_the_same_code_is_a_false_red_costing_the_wait(self):
        result = flips([v("lql", "s1", "failure", 1), v("lql", "s1", "success", 3.5)])
        self.assertEqual((result.false_reds, result.reds, result.verdicts), (1, 1, 2))
        self.assertEqual(result.delays_hours, [2.5])

    def test_a_failure_fixed_by_new_code_is_a_true_red(self):
        result = flips([v("lql", "s1", "failure", 1), v("lql", "s2", "success", 2)])
        self.assertEqual((result.false_reds, result.reds), (0, 1))

    def test_success_then_failure_on_the_same_code_is_a_flip_but_not_a_false_red(self):
        # Green then red on identical code: one of the two verdicts is wrong,
        # but which is unknown, so it is counted as nondeterminism, never
        # silently as a false red or a true one.
        result = flips([v("lql", "s1", "success", 1), v("lql", "s1", "failure", 2)])
        self.assertEqual((result.false_reds, result.nondeterministic), (0, 1))

    def test_workflows_are_never_compared_with_each_other(self):
        result = flips([v("lql", "s1", "failure", 1), v("kv", "s1", "success", 2)])
        self.assertEqual(result.false_reds, 0)

    def test_cancelled_and_skipped_runs_are_not_verdicts(self):
        result = flips([v("lql", "s1", "cancelled", 1), v("lql", "s1", "skipped", 2), v("lql", "s1", "success", 3)])
        self.assertEqual((result.verdicts, result.reds), (1, 0))


if __name__ == "__main__":
    unittest.main()
