"""Tests for the per-query cost the Chapter 5 figures plot.

Those panels used to plot wall-clock milliseconds, which cannot be committed to
a tracked figure: the number moves on every run, so the figure churned whenever
it was regenerated, and it told a reader nothing about their own hardware. They
plot counted operations instead.

A count makes falsifiable claims about the implementations, so the claims are
what gets tested here rather than the picture. Two of them are asymmetries a
reader would not guess and which the timing noise had hidden.

Author: Li-Ta Hsu
References: Chapter 5, Sections 5.1.1-5.1.3, Eqs. (5.1)-(5.6)
"""

import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch5_fingerprinting.example_comparison import per_query_operation_counts
from ch5_fingerprinting.example_deterministic import per_query_operations
from core.fingerprinting import load_fingerprint_database

DB_PATH = Path("data/sim/ch5_wifi_fingerprint_grid")


class TestComparisonCostModel(unittest.TestCase):
    """Check what the cost panels of comparison_all_methods assert."""

    @classmethod
    def setUpClass(cls):
        cls.db = load_fingerprint_database(DB_PATH)
        cls.on_floor = per_query_operation_counts(cls.db, floor_id=0)
        cls.all_floors = per_query_operation_counts(cls.db, floor_id=None)

    def test_counts_are_plain_integers(self):
        """A float here would mean a measurement leaked back in."""
        for method, count in self.on_floor.items():
            with self.subTest(method=method):
                self.assertIsInstance(count, int, f"{method} is not an int")

    def test_repeated_calls_agree(self):
        """The whole point: the number cannot move between runs."""
        self.assertEqual(per_query_operation_counts(self.db, floor_id=0), self.on_floor)

    def test_linear_model_cost_ignores_the_database(self):
        """The learned model is O(N_features), not O(N_RPs).

        This is the claim the example makes in prose -- "very fast prediction
        (single matrix multiplication)" -- and it is why linear regression sits
        two orders of magnitude below the matching methods in the figure.
        Constraining the floor cannot change it, because the database never
        enters the prediction.
        """
        self.assertEqual(
            self.on_floor["Linear Regression"],
            self.all_floors["Linear Regression"],
        )

    def test_floor_constraint_helps_distances_but_not_likelihoods(self):
        """The asymmetry that dominates the cost figure.

        nn_localize and knn_localize slice the database by floor and then scan
        what is left, so constraining the floor cuts their work. log_likelihood
        evaluates every RP in the database and masks the other floors to -inf
        afterwards, so constraining the floor saves it nothing. A reader
        comparing k-NN against MAP is seeing this, not an algorithmic gap.
        """
        self.assertLess(
            self.on_floor["NN (Euclidean)"], self.all_floors["NN (Euclidean)"]
        )
        self.assertLess(self.on_floor["k-NN (k=3)"], self.all_floors["k-NN (k=3)"])

        self.assertEqual(self.on_floor["MAP"], self.all_floors["MAP"])
        self.assertEqual(
            self.on_floor["Posterior Mean"], self.all_floors["Posterior Mean"]
        )

    def test_top_k_trims_the_cheap_term_only(self):
        """top_k cannot avoid the likelihood, whatever the docstring implies.

        posterior_mean_localize computes the posterior over every RP first and
        only then truncates the weighted sum, so top-k saves an O(M*d) step
        while the O(M*N) likelihood stands. It is a real saving but a small
        fraction of the total, and calling it "more efficient" oversells it.
        """
        n_rps = self.db.n_reference_points
        n_features = self.db.features.shape[1]
        likelihood_terms = n_rps * n_features

        full = self.on_floor["Posterior Mean"]
        top_k = self.on_floor["Post.Mean (k=10)"]

        self.assertGreater(top_k, likelihood_terms)
        self.assertLess(top_k, full)
        saving = (full - top_k) / full
        self.assertLess(saving, 0.25, f"top-k saves {saving:.0%}, more than expected")


class TestDeterministicCostModel(unittest.TestCase):
    """Check what the cost panel of deterministic_positioning asserts."""

    @classmethod
    def setUpClass(cls):
        cls.db = load_fingerprint_database(DB_PATH)

    def test_k_and_metric_are_free(self):
        """Every variant scans the whole floor, so none of them costs more.

        This is why the panel puts them on one vertical line: accuracy here is
        bought by choosing k and the weighting, and the scan is what you pay
        for either way.
        """
        base = per_query_operations(self.db, floor_id=0)
        euclid = per_query_operations(self.db, floor_id=0, metric="euclidean")
        manhattan = per_query_operations(self.db, floor_id=0, metric="manhattan")

        self.assertEqual(euclid, manhattan)
        self.assertEqual(euclid, base)

        n_dims = self.db.locations.shape[1]
        for k in (1, 3, 5):
            with self.subTest(k=k):
                self.assertEqual(
                    per_query_operations(self.db, floor_id=0, k=k),
                    base + k * n_dims,
                )

    def test_only_the_database_scan_moves_the_cost(self):
        """Dropping the floor constraint widens the scan, and that shows."""
        self.assertLess(
            per_query_operations(self.db, floor_id=0),
            per_query_operations(self.db, floor_id=None),
        )

    def test_scan_dominates_the_neighbour_average(self):
        """The k-way average is a rounding error; the scan is the bill.

        Splitting the total three ways on this database: the distance scan is
        ~88%, the argmin/selection over the same RPs ~11%, and the weighted
        average over k neighbours under 1%. Choosing k is therefore free, which
        is what the panel claims. If this stops holding, the claim needs
        revisiting.
        """
        n_searched = int(np.sum(self.db.get_floor_mask(0)))
        n_features = self.db.features.shape[1]
        n_dims = self.db.locations.shape[1]

        k = 5
        total = per_query_operations(self.db, floor_id=0, k=k)
        scan = n_searched * n_features
        neighbour_average = k * n_dims

        self.assertLess(neighbour_average / total, 0.01)
        self.assertGreater(scan / total, 0.85)
        self.assertGreater(scan, n_searched + neighbour_average)


if __name__ == "__main__":
    unittest.main()
