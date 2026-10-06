"""Numerical and temporal invariants of the availability-aware analysis."""
import unittest

import numpy as np
import pandas as pd

from analysis.data import historical
from analysis.evaluation import paired_bootstrap
from analysis.model import dual_lp, robust_cvar, transport_threshold, weighted_cvar


class MathematicalTests(unittest.TestCase):
    def test_fractional_tail_mass(self):
        self.assertAlmostEqual(weighted_cvar([0, 1, 2], [.90, .08, .02]), 1.4)

    def test_empirical_saturation_cannot_be_removed(self):
        cvar = weighted_cvar([0, 2], [.94, .06])
        self.assertEqual(cvar, 2)
        self.assertEqual(robust_cvar(cvar, 0, 2), 2)
        self.assertEqual(transport_threshold(np.array([0,2]), np.array([.94,.06]), 2), 0)

    def test_closed_form_against_transport_and_lp(self):
        rng = np.random.default_rng(20260914)
        for alpha in (.90, .95):
            for _ in range(20):
                x = np.r_[np.sort(rng.uniform(0, 2160, 9)), 2160.]
                p = rng.dirichlet(np.ones(len(x)))
                cvar = weighted_cvar(x, p, alpha)
                threshold = (1-alpha)*(2160-cvar)
                self.assertAlmostEqual(transport_threshold(x, p, 2160, alpha), threshold, places=7)
                previous = -1
                for epsilon in (0., threshold/2, threshold, threshold*2):
                    exact = robust_cvar(cvar, epsilon, 2160, alpha)
                    lp = dual_lp(x, p, epsilon, 2160, alpha)
                    self.assertAlmostEqual(exact, lp, places=6)
                    self.assertGreaterEqual(exact, previous)
                    self.assertLessEqual(exact, 2160)
                    previous = exact


class TemporalAndPairingTests(unittest.TestCase):
    def test_destination_must_precede_decision(self):
        cutoff = pd.Timestamp("2018-07-01", tz="UTC")
        frame = pd.DataFrame({"origin": pd.to_datetime(["2018-06-01"]*3, utc=True),
                              "destination": [cutoff-pd.Timedelta(seconds=1), cutoff, cutoff+pd.Timedelta(days=1)]})
        result = historical(frame, pd.Timestamp("2018-01-01", tz="UTC"), cutoff, cutoff)
        self.assertEqual(len(result), 1)
        self.assertTrue((result.destination < cutoff).all())

    def test_same_membership_has_exact_zero_paired_difference(self):
        start = pd.Timestamp("2018-07-01", tz="UTC")
        frame = pd.DataFrame({"case": np.arange(40)//2,
                              "origin": [start+pd.Timedelta(days=i) for i in range(40)],
                              "transition": ["A -> B", "C -> D"]*20,
                              "delay": np.arange(40)*3.71})
        ports = {("Empirical Tail", 2): ("A -> B", "C -> D"),
                 ("Original DRO", 2): ("C -> D", "A -> B")}
        for scheme in ("calendar_7d", "calendar_14d", "calendar_28d", "case_cluster"):
            result = paired_bootstrap(frame, ports, ["A -> B", "C -> D"], start,
                                      replicates=30, scheme=scheme)
            same = result[result.identical_membership]
            self.assertTrue((same[["difference_bootstrap_mean_pp", "ci_lower_pp", "ci_upper_pp"]] == 0).all().all())


if __name__ == "__main__":
    unittest.main()
