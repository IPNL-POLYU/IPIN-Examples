"""Tests for the Chapter 3 bimodal particle-filter example.

The example exists to demonstrate a claim -- that a particle filter can carry a
posterior a single Gaussian cannot -- so the claim itself is what gets tested.
Rendering alone would pass even if the posterior were unimodal, which is
precisely the failure worth catching.

Author: Li-Ta Hsu
References: Chapter 3, Section 3.3, Eqs. (3.32)-(3.34)
"""

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch3_estimators.example_particle_bimodal import (
    THIRD_ANCHOR_STEP,
    animate_bimodal,
    plot_bimodal_summary,
    run_bimodal_scenario,
)
from core.eval import save_animation


class TestBimodalScenario(unittest.TestCase):
    """The posterior must actually be bimodal, then actually resolve."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = run_bimodal_scenario()
        cls.bimodal = cls.scenario["steps"] < THIRD_ANCHOR_STEP
        cls.resolved = ~cls.bimodal

    def test_both_modes_stay_populated_with_two_anchors(self):
        """Two range circles intersect twice; both hypotheses must survive.

        If this fails the example is demonstrating nothing -- a unimodal cloud
        is exactly what a Kalman filter would give.
        """
        fraction = self.scenario["fraction_above"][self.bimodal]
        both_alive = (fraction > 0.02) & (fraction < 0.98)

        self.assertGreaterEqual(both_alive.sum(), 10)

    def test_third_anchor_kills_the_mirror_mode(self):
        """Once geometry disambiguates, every particle picks the same side."""
        fraction = self.scenario["fraction_above"][self.resolved]

        np.testing.assert_allclose(fraction, 1.0, atol=1e-9)

    def test_cloud_collapses_when_resolved(self):
        """Spread drops by roughly an order of magnitude or more."""
        spread = np.array(
            [np.sqrt(np.linalg.det(np.cov(c.T))) for c in self.scenario["clouds"]]
        )

        self.assertGreater(spread[self.bimodal].mean(),
                           10.0 * spread[self.resolved].mean())

    def test_mean_is_misleading_while_bimodal(self):
        """The headline lesson: the mean sits between the modes, in nothing.

        A Kalman filter can only report a mean and a covariance, so this is the
        precise sense in which it cannot represent the situation.
        """
        mean_error = self.scenario["error_mean"][self.bimodal].mean()
        mode_error = self.scenario["error_best_mode"][self.bimodal].mean()

        self.assertGreater(mean_error, 2.0)
        self.assertLess(mode_error, 1.0)
        self.assertGreater(mean_error, 5.0 * mode_error)

    def test_mean_becomes_trustworthy_once_resolved(self):
        """With one mode the mean is the mode, so both agree."""
        mean_error = self.scenario["error_mean"][self.resolved]
        mode_error = self.scenario["error_best_mode"][self.resolved]

        np.testing.assert_allclose(mean_error, mode_error, atol=1e-9)
        self.assertLess(mean_error.mean(), 1.0)

    def test_scenario_is_reproducible(self):
        """Seeded, and it must not disturb the caller's global RNG."""
        np.random.seed(1234)
        expected_next = np.random.rand()

        np.random.seed(1234)
        first = run_bimodal_scenario()
        actual_next = np.random.rand()
        second = run_bimodal_scenario()

        np.testing.assert_allclose(first["error_mean"], second["error_mean"])
        self.assertAlmostEqual(expected_next, actual_next)


class TestBimodalFigures(unittest.TestCase):
    """Rendering checks."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = run_bimodal_scenario()

    def test_summary_figure_has_three_panels(self):
        fig = plot_bimodal_summary(self.scenario)
        try:
            self.assertEqual(len(fig.axes), 3)
        finally:
            plt.close(fig)

    def test_animation_renders_and_stays_small(self):
        """Committed binaries live in git history forever."""
        fig, update, n_frames = animate_bimodal(self.scenario, n_frames=6)
        try:
            self.assertEqual(n_frames, 6)
            for frame in range(n_frames):
                self.assertEqual(len(update(frame)), 3)
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(fig, update, n_frames, tmp, "pf", fps=4)
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()
