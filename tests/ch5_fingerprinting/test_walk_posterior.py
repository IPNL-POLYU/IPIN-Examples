"""Tests for the Chapter 5 walking-posterior example.

The example's claim is counterintuitive -- the posterior does NOT spread under
noise, it stays sharp and jumps -- so the claim is what gets tested. The most
important guard is that the failure is aliasing (a confident teleport), not a
diffuse posterior, because that is the whole point.

Author: Li-Ta Hsu
References: Chapter 5, Sections 5.1-5.2, Eqs. (5.3)-(5.4)
"""

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch5_fingerprinting.example_walk_posterior import (
    ALIASING_THRESHOLD,
    animate_walk,
    plot_walk_summary,
    run_walk,
)
from core.eval import save_animation


class TestWalkPosterior(unittest.TestCase):
    """Check what the figure claims about the posterior and its failure."""

    @classmethod
    def setUpClass(cls):
        cls.low_noise = run_walk(noise_std=1.0)
        cls.high_noise = run_walk(noise_std=6.0)

    def test_posterior_stays_sharp_not_diffuse(self):
        """The posterior peaks on a single cell -- it does not hedge.

        If it were spreading into many cells the example would be telling the
        opposite (and wrong) story, so this is the load-bearing assertion.
        """
        self.assertLessEqual(int(np.median(self.high_noise["hot_counts"])), 2)

    def test_low_noise_tracks_without_aliasing(self):
        """At 1 dB the MAP follows the walk exactly."""
        errors = self.low_noise["errors"]

        self.assertEqual(int(np.sum(errors > ALIASING_THRESHOLD)), 0)
        self.assertLess(np.median(errors), 1.0)

    def test_high_noise_aliases_as_confident_jumps(self):
        """At 6 dB the failure is teleports, not drift."""
        errors = self.high_noise["errors"]

        # Usually exactly right...
        self.assertLess(np.median(errors), 1.0)
        # ...but several confident jumps far away.
        self.assertGreaterEqual(int(np.sum(errors > ALIASING_THRESHOLD)), 3)

    def test_mean_hides_what_the_median_reveals(self):
        """The headline: a mean error would conceal the aliasing entirely."""
        errors = self.high_noise["errors"]

        self.assertLess(np.median(errors), 1.0)
        self.assertGreater(errors.mean(), 3.0)

    def test_estimate_matches_posterior_argmax(self):
        """The MAP marker sits on the brightest cell, not the runner-up.

        Rendering once put a solid marker over the peak cell and made the
        runner-up look like the answer; this pins that the reported estimate
        really is the argmax of the displayed posterior.
        """
        walk = self.high_noise
        locations = walk["locations"]
        for index in range(len(walk["errors"])):
            argmax_xy = locations[np.argmax(walk["posteriors"][index])]
            np.testing.assert_allclose(walk["estimates"][index], argmax_xy)

    def test_reproducible(self):
        """Seeded; two runs agree."""
        first = run_walk(noise_std=6.0)
        second = run_walk(noise_std=6.0)

        np.testing.assert_allclose(first["errors"], second["errors"])


class TestWalkFigures(unittest.TestCase):
    """Rendering checks."""

    @classmethod
    def setUpClass(cls):
        cls.walk = run_walk(noise_std=6.0)

    def test_summary_has_two_panels(self):
        fig = plot_walk_summary(self.walk)
        try:
            # Two content axes plus one colorbar axis.
            self.assertGreaterEqual(len(fig.axes), 2)
        finally:
            plt.close(fig)

    def test_animation_has_a_single_colorbar(self):
        """Regression: a per-frame colorbar stacked one per step and bloated
        the GIF. Exactly one must exist after several frames are drawn.
        """
        fig, update, n_frames = animate_walk(self.walk)
        try:
            for frame in range(min(4, n_frames)):
                update(frame)
            # subplots(1, 2) gives 2 axes; one colorbar adds a third. More than
            # three means colorbars are stacking again.
            self.assertEqual(len(fig.axes), 3)
        finally:
            plt.close(fig)

    def test_animation_stays_small(self):
        fig, update, n_frames = animate_walk(self.walk)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(fig, update, n_frames, tmp, "walk", fps=3)
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()
