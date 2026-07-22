"""Tests for the Chapter 8 anchor-outage example.

The example makes several quantitative claims in its captions, and every one of
them was measured rather than assumed -- including one that disproved the
original hypothesis. These tests pin the claims so a caption cannot drift away
from the behaviour it describes.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1 (loose vs tight coupling)
"""

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch8_sensor_fusion.example_anchor_outage import (
    ANCHORS_KEPT,
    DEFAULT_DATA,
    OUTAGE_WINDOW,
    animate_anchor_outage,
    apply_anchor_outage,
    run_outage_scenario,
)
from ch8_sensor_fusion.lc_uwb_imu_ekf import load_fusion_dataset
from core.eval import save_animation


class TestAnchorOutageConstruction(unittest.TestCase):
    """The outage must actually be present in the data."""

    def test_outage_reduces_visibility_inside_the_window_only(self):
        """Anchors are blacked out in the window and nowhere else."""
        base = load_fusion_dataset(DEFAULT_DATA)
        outaged = apply_anchor_outage(base)

        times = np.asarray(outaged["uwb"]["t"])
        visible = np.sum(~np.isnan(np.asarray(outaged["uwb"]["ranges"])), axis=1)
        inside = (times >= OUTAGE_WINDOW[0]) & (times <= OUTAGE_WINDOW[1])

        # "At most" ANCHORS_KEPT: the dataset's own dropouts stack on the mask.
        self.assertLessEqual(visible[inside].max(), ANCHORS_KEPT)
        self.assertGreater(visible[~inside].max(), ANCHORS_KEPT)

    def test_original_dataset_is_not_mutated(self):
        """apply_anchor_outage works on a copy."""
        base = load_fusion_dataset(DEFAULT_DATA)
        before = np.isnan(np.asarray(base["uwb"]["ranges"])).sum()

        apply_anchor_outage(base)

        after = np.isnan(np.asarray(base["uwb"]["ranges"])).sum()
        self.assertEqual(before, after)


class TestOutageClaims(unittest.TestCase):
    """Check the numbers the captions quote."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = run_outage_scenario(verbose=False)

    def _peak(self, key_t, key_e, low, high):
        t = self.scenario[key_t]
        e = self.scenario[key_e]
        return e[(t >= low) & (t <= high)].max()

    def test_lc_fixes_fail_during_the_outage(self):
        """LC's front end cannot solve a position from two ranges."""
        self.assertGreater(self.scenario["lc"]["n_uwb_failed"], 50)

    def test_tc_far_better_during_the_outage(self):
        """The headline claim: LC dead-reckons, TC keeps correcting."""
        end = OUTAGE_WINDOW[1] + 3.0
        peak_lc = self._peak("t_lc", "error_lc", OUTAGE_WINDOW[0], end)
        peak_tc = self._peak("t_tc", "error_tc", OUTAGE_WINDOW[0], end)

        self.assertGreater(peak_lc, 3.0)
        self.assertLess(peak_tc, 1.0)
        self.assertGreater(peak_lc / peak_tc, 5.0)

    def test_tc_is_worse_at_the_outlier_events(self):
        """The counterweight, and the reason gating exists.

        Tight coupling is a trade: a corrupted range enters the filter
        directly, where LC's front end can absorb it in the position solve.
        If this ever stops holding, the example's caption must change too.
        """
        for low, high in ((35.0, 40.0), (55.0, 60.0)):
            with self.subTest(window=(low, high)):
                peak_lc = self._peak("t_lc", "error_lc", low, high)
                peak_tc = self._peak("t_tc", "error_tc", low, high)
                self.assertGreater(peak_tc, peak_lc)

    def test_tc_still_wins_overall(self):
        """Despite the outlier exposure, TC's RMSE is lower over the run."""
        rmse_lc = np.sqrt(np.mean(self.scenario["error_lc"] ** 2))
        rmse_tc = np.sqrt(np.mean(self.scenario["error_tc"] ** 2))

        self.assertLess(rmse_tc, rmse_lc)


class TestOutageAnimation(unittest.TestCase):
    """The GIF must render, and stay small."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = run_outage_scenario(verbose=False)

    def test_every_frame_draws(self):
        fig, update, n_frames = animate_anchor_outage(self.scenario, n_frames=5)
        try:
            self.assertEqual(n_frames, 5)
            for frame in range(n_frames):
                self.assertEqual(len(update(frame)), 3)
        finally:
            plt.close(fig)

    def test_animation_stays_small(self):
        """Committed binaries live in git history forever."""
        fig, update, n_frames = animate_anchor_outage(self.scenario, n_frames=8)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(fig, update, n_frames, tmp, "outage", fps=5)
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()
