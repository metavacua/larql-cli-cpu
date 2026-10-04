"""Tests for covgate.replay: frame, mutant pairing and sampling, analysis."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_covgate_calibrate import HOUR, T0, Repo, lines  # noqa: E402

from covgate.replay import analyze, frame, pairing, plan  # noqa: E402

CRATE = "crates/larql-cli"


class Frame(unittest.TestCase):
    def test_only_merged_prs_touching_the_crate_enter_the_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Repo(Path(tmp))
            r.commit(f"{CRATE}/src/a.rs", lines("a"), "init", T0)
            heads = {}
            for i, (path, number) in enumerate([(f"{CRATE}/src/a.rs", 7), ("crates/other/x.rs", 8), (f"{CRATE}/src/b.rs", 9)]):
                r.git("switch", "-q", "-c", f"pr{number}", "main")
                heads[number] = r.commit(path, lines(f"v{i}"), f"feat {number}", T0 + (i + 1) * HOUR)
                r.git("switch", "-q", "main")
                r.git("merge", "-q", "--no-ff", "-m", f"Merge pull request #{number} from x/pr{number}", f"pr{number}",
                      at=T0 + (i + 1) * HOUR + 60)
            found = frame.merged_prs(r.root, "main", CRATE)
            self.assertEqual([(p.number, p.head) for p in found], [(7, heads[7]), (9, heads[9])])
            self.assertTrue(all(p.base and p.merged_at for p in found))

    def test_the_sample_is_stratified_by_era_and_reproducible(self):
        prs = [frame.Pr(n, f"m{n}", f"b{n}", f"h{n}", float(n)) for n in range(1, 41)]
        a = frame.sample(prs, strata=4, seed=11)
        self.assertEqual(a, frame.sample(prs, strata=4, seed=11))
        self.assertEqual(len(a), 4)
        # One PR per era quartile: 1-10, 11-20, 21-30, 31-40.
        self.assertEqual([(p.number - 1) // 10 for p in sorted(a, key=lambda p: p.number)], [0, 1, 2, 3])

    def test_a_pr_resolves_to_its_merge_in_the_source_repo(self):
        api = {
            "repos/up/larql/pulls/83": {"merged": True, "merge_commit_sha": "m83"},
            "repos/up/larql/commits/m83": {"parents": [{"sha": "base83"}, {"sha": "head83"}]},
            "repos/up/larql/pulls/90": {"merged": True, "merge_commit_sha": "m90"},
            "repos/up/larql/commits/m90": {"parents": [{"sha": "base90"}]},  # squash merge
        }
        got = [frame.resolve(api.__getitem__, "up/larql", n) for n in (83, 90)]
        self.assertEqual([(g.repo, g.base, g.head) for g in got],
                         [("up/larql", "base83", "head83"), ("up/larql", "base90", "m90")])

    def test_an_unmerged_pr_is_refused_not_guessed(self):
        api = {"repos/up/larql/pulls/7": {"merged": False, "merge_commit_sha": None}}
        with self.assertRaisesRegex(ValueError, "not merged"):
            frame.resolve(api.__getitem__, "up/larql", 7)



def mutant(file, fn, replacement, line, genre="FnValue"):
    return {"file": file, "function": {"function_name": fn}, "genre": genre, "replacement": replacement,
            "span": {"start": {"line": line, "column": 5}},
            "name": f"{file}:{line}:5: replace {fn} with {replacement}"}


class Pairing(unittest.TestCase):
    def test_mutants_pair_by_what_they_mutate_not_by_line(self):
        base = [mutant("a.rs", "f", "true", 10), mutant("a.rs", "f", "true", 20), mutant("a.rs", "g", "0", 30)]
        head = [mutant("a.rs", "f", "true", 14), mutant("a.rs", "f", "true", 24), mutant("a.rs", "h", "0", 40)]
        pairs = pairing.pair(base, head)
        self.assertEqual(sorted((b["span"]["start"]["line"], h["span"]["start"]["line"]) for b, h in pairs.values()),
                         [(10, 14), (20, 24)])  # g and h exist on one side only

    def test_the_sample_allocates_by_file_and_is_reproducible(self):
        base = [mutant("a.rs", "f", str(i), i) for i in range(60)] + [mutant("b.rs", "f", str(i), i) for i in range(40)]
        pairs = pairing.pair(base, base)
        picked = pairing.sample(pairs, 10, seed=3)
        self.assertEqual(picked, pairing.sample(pairs, 10, seed=3))
        files = sorted(pairs[k][0]["file"] for k in picked)
        self.assertEqual((files.count("a.rs"), files.count("b.rs")), (6, 4))

    def test_names_are_escaped_for_rust_regex_and_anchored(self):
        name = "crates/x/src/a.rs:1:5: replace f -> Result<(), Box<dyn E>> with Ok(())"
        pattern = pairing.rust_regex(name)
        self.assertTrue(pattern.startswith("^") and pattern.endswith("$"))
        self.assertIn(r"\(\)", pattern)
        self.assertNotIn(r"\<", pattern)  # Rust regex rejects escaping '<'


def outcome(name, summary, seconds=60.0):
    return {"scenario": {"Mutant": {"name": name}}, "summary": summary,
            "phase_results": [{"phase": "Build", "duration": seconds / 2}, {"phase": "Test", "duration": seconds / 2}]}


class Analyze(unittest.TestCase):
    def test_discordant_counts_and_cost_per_mutant(self):
        selection = {"k1": ("b1", "h1"), "k2": ("b2", "h2"), "k3": ("b3", "h3"), "k4": ("b4", "h4")}
        base = {"outcomes": [outcome("b1", "CaughtMutant"), outcome("b2", "CaughtMutant"),
                             outcome("b3", "MissedMutant"), outcome("b4", "CaughtMutant", 120.0)]}
        head = {"outcomes": [outcome("h1", "CaughtMutant"), outcome("h2", "MissedMutant"),
                             outcome("h3", "CaughtMutant"), outcome("h4", "Timeout")]}
        result = analyze.paired(selection, base, head)
        self.assertEqual((result["n"], result["b"], result["c"], result["excluded"]), (3, 1, 1, 1))
        self.assertAlmostEqual(result["discordance"], 2 / 3)
        self.assertAlmostEqual(result["drop"], 0.0)
        self.assertAlmostEqual(result["seconds_per_mutant"], 67.5)  # (7 x 60 + 120) / 8 outcomes


class Plan(unittest.TestCase):
    config = {"source_repo": "up/larql", "crate_dir": CRATE, "strata": 2, "seed": 1, "mutants_per_pr": 20,
              "prior_minutes_per_mutant": 3.0, "setup_minutes": 15.0, "runner_minutes_budget": 400.0}

    def test_the_estimate_counts_both_sides_of_every_pr(self):
        # 2 PRs x 2 sides x (15 + 20 x 3) minutes
        self.assertEqual(plan.estimate_runner_minutes(self.config, prs=2), 300.0)

    def test_a_plan_over_budget_is_refused_before_any_runner_starts(self):
        config = {**self.config, "mutants_per_pr": 35}
        expected = 2 * 2 * (config["setup_minutes"] + 35 * config["prior_minutes_per_mutant"])
        with self.assertRaisesRegex(ValueError, f"{expected:.0f} runner-minutes.*budget 400"):
            plan.check_budget(config, prs=2)

    def test_every_config_key_is_required(self):
        broken = dict(self.config)
        del broken["seed"]
        with self.assertRaisesRegex(ValueError, "seed"):
            plan.check_budget(broken, prs=2)


class Aggregate(unittest.TestCase):
    def test_per_pr_results_stay_separate_and_the_pool_carries_an_interval(self):
        results = {
            "83": {"status": "ok", "n": 20, "b": 3, "c": 1, "excluded": 0, "seconds_per_mutant": 100.0},
            "213": {"status": "ok", "n": 20, "b": 0, "c": 0, "excluded": 2, "seconds_per_mutant": 140.0},
            "411": {"status": "baseline-failed"},
        }
        report = analyze.aggregate(results)
        self.assertEqual(report["prs"], {"replayed": 2, "unbuildable": 1})
        low, point, high = report["discordance"]
        self.assertAlmostEqual(point, 4 / 40)
        self.assertTrue(low < point < high)
        self.assertEqual(report["drops"], {"83": 0.1, "213": 0.0})
        self.assertEqual(report["seconds_per_mutant_median"], 120.0)


if __name__ == "__main__":
    unittest.main()
