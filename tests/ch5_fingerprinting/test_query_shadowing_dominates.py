"""Chapter 5's noise labels name only half of the noise.

The scenarios were labelled "sigma=1dBm", "2dBm", "5dBm". Each query also
draws its own shadow-fading term at the database's shadow_fading_std_dBm,
which is 4.0 and is never mentioned. Measured with nearest-neighbour matching
on the grid database: the labelled sweep from 1 to 5 dBm costs 3.49 m of RMSE,
and removing the unmentioned shadowing recovers 2.59 m. Comparable terms, one
of them invisible from the label.

Redrawing shadowing per query is also a modelling choice with consequences.
Shadowing is a property of a location, so an independent draw makes the query
inconsistent with the map at its own position -- the correlation fingerprinting
depends on. A spatially correlated field would be faithful; this is not.

Author: Li-Ta Hsu
References: Chapter 5, Sections 5.1-5.3
"""

import contextlib
import io
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch5_fingerprinting.example_comparison import (
    _generate_queries_pathloss,
    load_fingerprint_database,
)

DATABASE = Path("data/sim/ch5_wifi_fingerprint_grid")
GRID_SPACING_M = 5.0


def _nn_rmse(db, shadow_dbm, extra_noise=0.0, n_queries=200):
    """Nearest-neighbour RMSE with a chosen query shadow-fading level."""
    config = dict(db.meta["path_loss_model"])
    config["shadow_fading_std_dBm"] = shadow_dbm
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        queries, truth, _ = _generate_queries_pathloss(
            db,
            np.asarray(db.meta["ap_positions"]),
            config,
            n_queries=n_queries,
            floor_id=0,
            noise_std=extra_noise,
        )
    features = np.asarray(db.features)
    locations = np.asarray(db.locations)
    estimates = np.array(
        [locations[np.argmin(np.linalg.norm(features - q, axis=1))] for q in queries]
    )
    errors = np.linalg.norm(estimates[:, :2] - np.asarray(truth)[:, :2], axis=1)
    return float(np.sqrt(np.mean(errors**2)))


class TestQueryShadowingDominates(unittest.TestCase):
    """Which knob actually moves the reported accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.db = load_fingerprint_database(DATABASE)
        cls.shadow = cls.db.meta["path_loss_model"]["shadow_fading_std_dBm"]

    def test_the_database_carries_shadow_fading_the_labels_omit(self):
        """The premise: there is a second, larger noise term in play."""
        self.assertGreaterEqual(self.shadow, 2.0)

    def test_the_unlabelled_shadowing_is_comparable_to_the_whole_sweep(self):
        """Both terms matter, and only one of them is named.

        Written after asserting the opposite and measuring it: the labelled
        sweep costs 3.49 m and the unmentioned shadowing 2.59 m, so the claim
        is comparability, not dominance.
        """
        swept = _nn_rmse(self.db, self.shadow, extra_noise=5.0) - _nn_rmse(
            self.db, self.shadow, extra_noise=1.0
        )
        unshadowed = _nn_rmse(self.db, self.shadow, extra_noise=1.0) - _nn_rmse(
            self.db, 0.0, extra_noise=1.0
        )

        self.assertGreater(unshadowed, 0.3 * swept)
        self.assertLess(unshadowed, 3.0 * swept)

    def test_accuracy_stays_far_above_the_grid_floor(self):
        """Even noise-free queries cannot approach what the grid allows.

        A 5 m grid puts a nearest-neighbour floor near 2 m RMS. Noiseless
        queries still score around 7 m, so something beyond the labelled noise
        and beyond quantisation dominates these numbers. Pinned as an open
        question rather than a target: this test documents that the gap exists
        and will fail if it is ever closed, which is when the chapter's
        absolute accuracy figures start meaning what a reader assumes.
        """
        floor = 0.4 * GRID_SPACING_M

        self.assertGreater(_nn_rmse(self.db, 0.0, extra_noise=0.0), 2.0 * floor)


if __name__ == "__main__":
    unittest.main()
