"""Tests for covgate.decision. Run: python3 -m unittest discover -s scripts -p test_covgate_decision.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from covgate.decision import mcnemar  # noqa: E402
from covgate.decision.pareto import Design, Inputs, design_vector, frontier  # noqa: E402
from covgate.decision.choose import gamma_minimax, minimax_regret, weighted_sum  # noqa: E402


class McNemar(unittest.TestCase):
    def test_the_exact_test_never_exceeds_its_nominal_size(self):
        for n in (10, 40, 120):
            for psi in (0.05, 0.2):
                self.assertLessEqual(mcnemar.size(n, psi, 0.05), 0.05 + 1e-12)

    def test_power_at_no_drop_is_the_size(self):
        # The control: under H0 the "power" is exactly the false-red rate.
        self.assertAlmostEqual(mcnemar.power(60, 0.2, 0.0, 0.05), mcnemar.size(60, 0.2, 0.05), places=12)

    def test_power_grows_with_sample_size_and_with_the_drop(self):
        self.assertLess(mcnemar.power(20, 0.2, 0.05, 0.05), mcnemar.power(200, 0.2, 0.05, 0.05))
        self.assertLess(mcnemar.power(100, 0.2, 0.02, 0.05), mcnemar.power(100, 0.2, 0.08, 0.05))

    def test_a_drop_larger_than_the_discordance_is_rejected_as_impossible(self):
        with self.assertRaises(ValueError):
            mcnemar.power(50, 0.05, 0.06, 0.05)


INPUTS = Inputs(p_degrade=0.1, false_red_hours=2.0, escape_hours=200.0,
                runner_hours_per_mutant=0.02, discordance=0.2, drop=0.05)


class Pareto(unittest.TestCase):
    def test_the_vector_keeps_stakeholders_apart(self):
        d = Design(n=100, alpha=0.05)
        author, users, runners = design_vector(d, INPUTS)
        size = mcnemar.size(100, 0.2, 0.05)
        beta = 1 - mcnemar.power(100, 0.2, 0.05, 0.05)
        self.assertAlmostEqual(author, 0.9 * size * 2.0)
        self.assertAlmostEqual(users, 0.1 * beta * 200.0)
        self.assertAlmostEqual(runners, 2 * 100 * 0.02)  # base and head both run

    def test_dominated_designs_leave_the_frontier_and_ties_stay(self):
        points = {"a": (1, 1, 1), "b": (2, 2, 2), "c": (1, 1, 1), "d": (0, 3, 1)}
        self.assertEqual(sorted(frontier(points)), ["a", "c", "d"])


class Choose(unittest.TestCase):
    points = {"cheap": (5.0, 5.0, 0.0), "balanced": (1.0, 1.0, 2.0), "heavy": (0.1, 0.1, 9.0)}

    def test_weighted_sum_is_labelled_and_picks_the_minimum(self):
        label, pick = weighted_sum(self.points, (1.0, 1.0, 1.0))
        self.assertIn("transferable", label)
        self.assertEqual(pick, "balanced")

    def test_gamma_minimax_minimises_the_worst_scenario(self):
        scenarios = [self.points, {"cheap": (50.0, 50.0, 0.0), "balanced": (3.0, 3.0, 2.0), "heavy": (0.2, 0.2, 9.0)}]
        label, pick = gamma_minimax(scenarios, (1.0, 1.0, 1.0))
        self.assertIn("adversar", label)
        self.assertEqual(pick, "balanced")  # worst cases: cheap 100, balanced 8, heavy 9.4

    def test_minimax_regret_differs_from_minimax_loss_when_it_should(self):
        # Regret is loss minus the best achievable in that scenario.
        s1 = {"x": (0.0, 0.0, 10.0), "y": (0.0, 0.0, 12.0)}
        s2 = {"x": (0.0, 0.0, 30.0), "y": (0.0, 0.0, 21.0)}
        self.assertEqual(gamma_minimax([s1, s2], (1, 1, 1))[1], "y")      # worst: x 30, y 21
        label, pick = minimax_regret([s1, s2], (1, 1, 1))
        self.assertIn("regret", label)
        self.assertEqual(pick, "y")                                        # regret: x 9, y 2
        s3 = {"x": (0.0, 0.0, 1.0), "y": (0.0, 0.0, 6.0)}
        self.assertEqual(gamma_minimax([s1, s3], (1, 1, 1))[1], "x")      # worst: x 10, y 12
        self.assertEqual(minimax_regret([s1, s3], (1, 1, 1))[1], "x")     # regret: x 0, y 5


if __name__ == "__main__":
    unittest.main()
