"""Tests for covgate.calibrate.summary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate.calibrate.false_reds import Verdict  # noqa: E402
from covgate.calibrate.summary import quantile_interval, summarize  # noqa: E402


class QuantileInterval(unittest.TestCase):
    def test_median_interval_uses_order_statistics(self):
        # n = 20 sorted values 1..20: the distribution-free 95% interval for
        # the median is (x(6), x(15)) — ranks from Binomial(20, 1/2).
        xs = list(range(1, 21))
        self.assertEqual(quantile_interval(xs, 0.5, 0.95), (6, 10.5, 15))

    def test_too_few_observations_give_no_interval_rather_than_a_false_one(self):
        self.assertEqual(quantile_interval([1.0, 2.0, 3.0], 0.5, 0.95), (None, 2.0, None))


class Summary(unittest.TestCase):
    def test_strata_are_reported_separately_never_pooled(self):
        escapes = [
            {"repo": "a/x", "only": "crates/c", "changes": 100,
             "escapes": [{"fix": "f", "introduced": "i1", "latency_hours": 10.0},
                         {"fix": "f", "introduced": "i2", "latency_hours": 30.0}]},
            {"repo": "b/y", "only": "crates/c", "changes": 50, "escapes": []},
        ]
        runs = [Verdict("a/x:1", "s", "failure", 0.0, 1, 1), Verdict("a/x:1", "s", "success", 7200.0, 1, 2),
                Verdict("b/y:9", "t", "success", 0.0, 2, 1)]
        report = summarize(escapes, runs)
        self.assertEqual(set(report["escapes"]), {"a/x crates/c", "b/y crates/c"})
        self.assertEqual(report["escapes"]["a/x crates/c"]["introducing_changes"], 2)
        self.assertEqual(report["escapes"]["b/y crates/c"]["introducing_changes"], 0)
        low, point, high = report["escapes"]["a/x crates/c"]["defect_rate"]
        self.assertTrue(low < point == 0.02 < high)
        self.assertEqual(report["false_reds"]["a/x:1"]["false_reds"], 1)
        self.assertEqual(report["false_reds"]["a/x:1"]["delay_hours_median"], (None, 2.0, None))
        self.assertEqual(report["false_reds"]["b/y:9"]["false_reds"], 0)


if __name__ == "__main__":
    unittest.main()
