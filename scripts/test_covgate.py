"""Unit tests for scripts/covgate. Run: python3 -m unittest discover -s scripts -p test_covgate.py"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate import checks, depinfo, diff, functions, lcov, metrics, mutation, ratchet, stats  # noqa: E402
from covgate.cell import Cell, restrict  # noqa: E402
from covgate.demangle import demangle_all, implemented_trait, still_mangled  # noqa: E402
from covgate.functions import Function  # noqa: E402
from covgate.paths import repo_relative  # noqa: E402
from covgate.policy import parse as parse_policy  # noqa: E402

CRATE = "crates/larql-cli"
A = f"{CRATE}/src/a.rs"
B = f"{CRATE}/src/b.rs"


def mangle(*segments: str) -> str:
    return "_ZN" + "".join(f"{len(s)}{s}" for s in segments) + "17h0123456789abcdefE"


def policy(**gate_overrides) -> object:
    gate = {
        "function_min_percent": 90.0,
        "trait_line_min_percent": 90.0,
        "diff_line_min_percent": 90.0,
        "mutation_min_kill_percent": 90.0,
        "mutation_confidence": 0.95,
        "required_cells": [],
        "gating_rules": ["cells", "spread", "ratchet"],
    }
    gate.update(gate_overrides)
    return parse_policy(
        {"default_line_min_percent": 90.0, "total_line_min_percent": 90.0, "gate": gate}, CRATE
    )


def cell(name="x86-release", target="x86", feature="release", hits=None, funcs=None) -> Cell:
    return Cell(name, target, feature, hits or {}, funcs or [])


def full(n: int, covered: int) -> dict[int, int]:
    return {i: (1 if i <= covered else 0) for i in range(1, n + 1)}


class Paths(unittest.TestCase):
    def test_every_runner_layout_normalises_to_the_same_key(self):
        for raw in (
            "/home/runner/work/r/r/crates/larql-cli/src/a.rs",
            "/Users/runner/work/r/r/crates/larql-cli/src/a.rs",
            "D:\\a\\r\\r\\crates\\larql-cli\\src\\a.rs",
            "crates/larql-cli/src/a.rs",
        ):
            self.assertEqual(repo_relative(raw), A)

    def test_registry_sources_are_not_workspace_files(self):
        self.assertIsNone(repo_relative("/home/u/.cargo/registry/src/index.crates.io-1/serde-1.0/src/lib.rs"))


class Lcov(unittest.TestCase):
    def test_a_line_seen_twice_keeps_its_best_hit(self):
        text = f"SF:/w/{A}\nDA:1,0\nDA:2,3\nend_of_record\nSF:/w/{A}\nDA:1,5\nend_of_record\n"
        self.assertEqual(lcov.parse(text), {A: {1: 5, 2: 3}})

    def test_a_malformed_record_is_an_error_not_a_skip(self):
        with self.assertRaises(ValueError):
            lcov.parse(f"SF:{A}\nDA:7\n")


class DepInfo(unittest.TestCase):
    def test_sources_across_continuation_lines(self):
        text = f"/t/larql: /w/{A} \\\n /w/{B} /rustc/lib.rs\n/w/{A}:\n"
        self.assertEqual(depinfo.parse(text), {A, B})


# Real symbols from a larql-cli coverage export (rustc 1.98, v0 mangling is
# the default). Not hand-written: hand-written legacy symbols are how the
# trait rule passed vacuously on real data.
V0_DISPLAY = "_RNvXNtCs93EXaNUS1sC_5larql10trampolineNtB2_19ResearchUnavailableNtNtCsc36rpYXAlPq_4core3fmt7Display3fmt"
V0_CAPTURE = ("_RNvXNtNtNtCs93EXaNUS1sC_5larql8commands10extraction13residuals_cmdNtB2_17ProgressCallbacks"
              "NtNtCsepvdsOKcNLz_15larql_inference7capture16CaptureCallbacks14on_entity_done")


class Demangle(unittest.TestCase):
    def test_real_v0_trait_impls_demangle_to_their_trait_without_disambiguators(self):
        names = demangle_all([V0_DISPLAY, V0_CAPTURE])
        self.assertEqual(implemented_trait(names[V0_DISPLAY]), "core::fmt::Display")
        self.assertEqual(implemented_trait(names[V0_CAPTURE]), "larql_inference::capture::CaptureCallbacks")

    def test_legacy_symbols_still_demangle(self):
        legacy = mangle("_$LT$larql_cli..Foo$u20$as$u20$core..fmt..Display$GT$", "fmt")
        self.assertEqual(implemented_trait(demangle_all([legacy])[legacy]), "core::fmt::Display")

    def test_an_inherent_method_with_a_trait_in_its_generics_names_no_trait(self):
        self.assertIsNone(implemented_trait("<larql[ab12]::X>::f::<<larql[ab12]::Y as core[cd34]::ops::Fn>>"))

    def test_a_name_left_mangled_is_detected(self):
        self.assertTrue(still_mangled("_RNvCs1_5larql4main"))
        self.assertTrue(still_mangled("src/a.rs:_ZN3foo3barE"))
        self.assertFalse(still_mangled("larql::main"))


class Functions(unittest.TestCase):
    def test_instantiations_merge_and_execute_if_any_ran(self):
        record = lambda count: {"name": V0_DISPLAY, "count": count, "filenames": [f"/w/{A}"],
                                "regions": [[3, 1, 7, 2, count, 0, 0, 0], [9, 1, 9, 5, 0, 1, 0, 0]]}
        found = functions.parse({"data": [{"functions": [record(0), record(4)]}]})
        self.assertEqual(len(found), 1)
        f = found[0]
        self.assertEqual((f.path, f.trait, f.start, f.end, f.executed), (A, "core::fmt::Display", 3, 7, True))
        self.assertFalse(still_mangled(f.name))


class Restrict(unittest.TestCase):
    def test_test_files_and_cfg_test_spans_leave_the_denominator(self):
        test_file = f"{CRATE}/src/a_tests.rs"
        hits = {A: full(10, 5), test_file: full(4, 4), "crates/other/src/x.rs": full(2, 2)}
        funcs = [Function(A, "in_tests", None, 7, 9, True), Function(A, "real", None, 1, 3, True)]
        kept, kept_funcs = restrict(hits, funcs, CRATE, {A}, {A: [(6, 10)]})
        self.assertEqual(kept, {A: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}})
        self.assertEqual([f.name for f in kept_funcs], ["real"])


class Metrics(unittest.TestCase):
    def test_trait_lines_are_the_lcov_lines_inside_impl_spans(self):
        c = cell(hits={A: {1: 1, 2: 0, 3: 1, 10: 0}},
                 funcs=[Function(A, "f", "x::Tr", 1, 3, True), Function(A, "g", None, 10, 10, False)])
        self.assertEqual(metrics.per_trait(c), {"x::Tr": (2, 3)})
        self.assertEqual(metrics.function_ratio(c), (1, 2))
        self.assertEqual(metrics.total(c), (2, 4))

    def test_union_and_intersection_bracket_the_cells(self):
        x = cell("x", hits={A: {1: 1, 2: 0, 3: 1}})
        arm = cell("arm", hits={A: {1: 1, 2: 1, 4: 0}})
        union, intersection = metrics.union_and_intersection([x, arm])
        self.assertEqual(union, (3, 4))          # 1,2,3 covered somewhere; 1..4 coverable somewhere
        self.assertEqual(intersection, (1, 2))   # lines 1,2 in both; only 1 covered in both

    def test_an_empty_denominator_is_not_full_coverage(self):
        self.assertEqual(metrics.percent((0, 0)), 0.0)


class Rules(unittest.TestCase):
    def rules(self, findings):
        return sorted({f.rule for f in findings})

    def test_a_cell_whose_names_stayed_mangled_fails_instead_of_passing_the_trait_rule(self):
        mangled = cell(hits={A: full(10, 10)}, funcs=[Function(A, "_RNvCs1_5larql4main", None, 1, 10, True)])
        self.assertEqual(self.rules(checks.thresholds(mangled, policy())), ["demangle"])
        clean = cell(hits={A: full(10, 10)}, funcs=[Function(A, "larql::main", None, 1, 10, True)])
        self.assertEqual(checks.thresholds(clean, policy()), [])

    def test_each_threshold_fires_below_and_not_at_its_minimum(self):
        low = cell(hits={A: full(10, 8)}, funcs=[Function(A, "f", "x::Tr", 1, 10, True),
                                                 Function(A, "g", None, 1, 1, False)])
        self.assertEqual(self.rules(checks.thresholds(low, policy())), ["file", "functions", "total", "trait"])
        ok = cell(hits={A: full(10, 9)}, funcs=[Function(A, "f", "x::Tr", 1, 10, True)])
        self.assertEqual(checks.thresholds(ok, policy()), [])

    def test_a_cell_measuring_nothing_fails(self):
        self.assertEqual(self.rules(checks.thresholds(cell(), policy())), ["total"])

    def test_a_missing_required_cell_fails(self):
        found = checks.required_cells([cell(name="a")], policy(required_cells=["a", "b"]))
        self.assertEqual([f.subject for f in found], ["b"])

    def test_spread_is_exact_per_line_within_one_feature_only(self):
        x = cell("x-rel", "x86", "release", {A: {1: 1, 2: 1, 3: 0}})
        same = cell("arm-rel", "arm", "release", {A: {1: 1, 2: 1, 3: 0, 4: 0}})  # 4 is arm-only
        other_feature = cell("arm-res", "arm", "research", {A: {1: 0, 2: 0, 3: 0}})
        self.assertEqual(checks.spread([x, same, other_feature], policy()), [])
        split = cell("arm-rel", "arm", "release", {A: {1: 1, 2: 0, 3: 0}})
        found = checks.spread([x, split], policy())
        self.assertEqual([(f.rule, f.subject) for f in found], [("spread", A)])
        self.assertIn("[2]", found[0].message)

    def test_diff_counts_only_added_coverable_lines(self):
        c = cell(hits={A: {1: 0, 2: 1, 3: 1}})
        self.assertEqual(self.rules(checks.diff_coverage(c, {A: {1, 2, 50}}, policy())), ["diff"])
        self.assertEqual(checks.diff_coverage(c, {A: {2, 3, 50}}, policy()), [])

    def test_unified_zero_diff_parsing(self):
        text = f"--- a/{A}\n+++ b/{A}\n@@ -1,0 +2,3 @@\n+x\n@@ -9 +12 @@\n+y\n--- a/{B}\n+++ /dev/null\n"
        self.assertEqual(diff.parse(text), {A: {2, 3, 4, 12}})

    def test_only_named_rules_gate_the_rest_are_reported(self):
        found = [checks.Finding("file", "c", "a", "low"), checks.Finding("spread", "c", "a", "split")]
        gating, reported = checks.split_gating(found, policy())
        self.assertEqual([f.rule for f in gating], ["spread"])
        self.assertEqual([f.rule for f in reported], ["file"])

    def test_an_unknown_gating_rule_is_a_policy_error(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            policy(gating_rules=["spread", "vibes"])


class Ratchet(unittest.TestCase):
    base = {"include_globs": ["a/*"], "exclude_globs": ["x"], "default_line_min_percent": 90,
            "total_line_min_percent": 90, "per_file_line_min_percent": {"f": 80},
            "gate": {"trait_line_min_percent": 90, "mutation_confidence": 0.95}}

    files = ["a/1.rs", "a/2.rs", "b/3.rs", "x", "y"]

    def test_every_loosening_is_named(self):
        head = {"include_globs": ["b/*"], "exclude_globs": ["x", "y"], "default_line_min_percent": 89,
                "total_line_min_percent": 90, "per_file_line_min_percent": {"f": 70, "g": 50},
                "gate": {"trait_line_min_percent": 89}}
        subjects = sorted(f.subject for f in ratchet.compare_policies(self.base, head, self.files))
        self.assertEqual(subjects, ["a/1.rs", "a/2.rs", "default_line_min_percent", "f", "g", "trait_line_min_percent"])

    def test_scope_is_judged_on_real_files_not_glob_text(self):
        # Replacing a narrow glob with a wider one is a widening, even
        # though the narrow glob's text disappears.
        head = {**self.base, "include_globs": ["*"], "exclude_globs": ["x"]}
        self.assertEqual(ratchet.compare_policies(self.base, head, self.files), [])

    def test_tightening_and_widening_pass(self):
        head = {"include_globs": [], "exclude_globs": [], "default_line_min_percent": 95,
                "total_line_min_percent": 92, "per_file_line_min_percent": {},
                "gate": {"trait_line_min_percent": 91}}
        self.assertEqual(ratchet.compare_policies(self.base, head, self.files), [])

    def test_suppression_markers_are_counted(self):
        text = "#[coverage(off)]\n#[cfg(not(coverage))]\n#[cfg_attr(coverage, x)]\nlet coverage = 1;"
        self.assertEqual(ratchet.count_markers({"a": text}), 3)


class Mutation(unittest.TestCase):
    def outcomes(self, baseline, *rows):
        out = [{"scenario": "Baseline", "summary": baseline}]
        out += [{"scenario": {"Mutant": {"file": path}}, "summary": s} for path, s in rows]
        return {"outcomes": out}

    def run_check(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.json"
            path.write_text(json.dumps(data))
            return mutation.check(path, CRATE, 90.0, 0.95)[0]

    def test_a_large_sample_must_certify_the_floor_with_its_lower_bound(self):
        # 95/100: lower bound 90.1% certifies; 93/100: point estimate above
        # 90% but the interval reaches below it, so not demonstrated.
        ok = [(A, "CaughtMutant")] * 95 + [(A, "MissedMutant")] * 5 + [(A, "Timeout")] * 9
        self.assertEqual(self.run_check(self.outcomes("Success", *ok)), [])
        weak = [(A, "CaughtMutant")] * 93 + [(A, "MissedMutant")] * 7
        found = self.run_check(self.outcomes("Success", *weak))
        self.assertEqual([f.subject for f in found], [A])
        self.assertIn("not demonstrated", found[0].message)
        bad = [(A, "CaughtMutant")] * 60 + [(A, "MissedMutant")] * 40
        self.assertIn("confidently below", self.run_check(self.outcomes("Success", *bad))[0].message)

    def test_a_small_sample_must_be_perfect(self):
        small = [(A, "CaughtMutant")] * 23 + [(A, "MissedMutant")]
        self.assertIn("every mutant must be caught", self.run_check(self.outcomes("Success", *small))[0].message)
        perfect = [(A, "CaughtMutant")] * 3 + [("crates/other/src/x.rs", "MissedMutant")]
        self.assertEqual(self.run_check(self.outcomes("Success", *perfect)), [])

    def test_a_failed_baseline_is_a_finding(self):
        self.assertEqual([f.subject for f in self.run_check(self.outcomes("Failure"))], ["baseline"])


class Stats(unittest.TestCase):
    def test_wilson_matches_reference_values(self):
        # Reference: z = 1.96, 8/10 -> [0.4902, 0.9433] (Newcombe 1998, method 3).
        low, high = stats.wilson(8, 10, 1.959964)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)

    def test_the_exact_threshold_is_the_first_n_a_perfect_score_certifies(self):
        z = stats.z_one_sided(0.95)
        n = stats.exact_threshold(0.9, z)
        self.assertGreaterEqual(stats.wilson(n, n, z)[0], 0.9)
        self.assertLess(stats.wilson(n - 1, n - 1, z)[0], 0.9)

    def test_out_of_range_inputs_are_errors(self):
        for call in (lambda: stats.z_one_sided(0.4), lambda: stats.wilson(3, 2, 1.0),
                     lambda: stats.wilson(0, 0, 1.0), lambda: stats.exact_threshold(1.0, 1.0)):
            with self.assertRaises(ValueError):
                call()


class SelfTest(unittest.TestCase):
    def test_the_gate_passes_clean_and_catches_every_planted_defect(self):
        from covgate import selftest
        self.assertEqual(selftest.run(Path(__file__).resolve().parents[1]), [])


if __name__ == "__main__":
    unittest.main()
