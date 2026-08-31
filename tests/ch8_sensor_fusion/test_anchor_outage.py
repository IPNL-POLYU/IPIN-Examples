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
    format_peak_ratio,
    run_outage_scenario,
)
from core.eval import save_animation
from core.fusion import load_fusion_dataset


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

    def test_peak_ratio_names_the_larger_error(self):
        """The console summary should not round LC/TC down to a misleading 0x."""
        self.assertEqual(format_peak_ratio(3.25, 43.92), "TC peak is 13.5x LC")
        self.assertEqual(format_peak_ratio(2.0, 0.5), "LC peak is 4.0x TC")


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

    def test_tc_bounds_its_worst_case_lc_does_not(self):
        """LC has no worst-case guarantee during an outage; TC does.

        Was ``test_tc_far_better_during_the_outage``, asserting a fixed
        per-window bound (``peak_lc > 1.5``) at all four placements. That
        stopped holding once ``run_lc_fusion`` was fixed to read the
        dataset's real UWB noise and stopped flooring its WLS covariance
        (see CLAUDE.md's Chapter 8 entries): LC now enters every outage from
        much better baseline tracking, so its peak error during any one
        placement is no longer reliably large -- three of these four windows
        now give LC a peak under 1 m, where all four used to exceed 1.5 m.

        What survives is the asymmetry, not a fixed per-window number: TC can
        always take a partial (1-2 range) update, so its worst case across
        placements stays small everywhere. LC gets zero updates for the whole
        window and has no such guarantee -- at at least one of the same four
        placements its peak is still far larger than TC's worst case at any
        of them.

        Measured away from the mirror-flip window elsewhere in this file,
        because that flip is a separate phenomenon and would otherwise mask
        this one. Checked at four outage placements rather than one, since
        the whole reason the claim needed re-establishing (twice now) is that
        single-window evidence does not generalise.
        """
        peaks_lc = []
        peaks_tc = []
        for window in ((18.0, 26.0), (22.0, 30.0), (24.0, 32.0), (40.0, 48.0)):
            scenario = run_outage_scenario(window=window, verbose=False)
            end = window[1] + 3.0
            t_lc, e_lc = scenario["t_lc"], scenario["error_lc"]
            t_tc, e_tc = scenario["t_tc"], scenario["error_tc"]
            peaks_lc.append(e_lc[(t_lc >= window[0]) & (t_lc <= end)].max())
            peaks_tc.append(e_tc[(t_tc >= window[0]) & (t_tc <= end)].max())

        # TC always gets some correction (2 surviving ranges), so its worst
        # case across placements stays small. Measured max is 0.06 m.
        self.assertLess(max(peaks_tc), 0.5)

        # LC gets none (it needs 3+ ranges and has at most 2), so its peak
        # depends on trajectory dynamics during the blackout and has no such
        # guarantee. Measured max is 1.18 m, at window (18, 26) -- roughly
        # 20x TC's worst. The bound is on the *ratio* rather than on LC's
        # metres, because LC's absolute peak has now moved twice (once when
        # its WLS noise bug was fixed, once when the accelerometer frame was)
        # while the asymmetry survived both.
        self.assertGreater(max(peaks_lc), 5.0 * max(peaks_tc))

    def test_the_default_window_does_not_flip_branches(self):
        """The mirror ambiguity is real; falling into it was a symptom.

        The outage keeps the anchors at (0, 0) and (20, 0) while the platform
        walks x = 20, so two ranges fit the truth and its reflection across
        y = 0 equally well. Nothing in the ranges breaks the tie -- the IMU
        prediction does.

        This demo used to take the wrong branch here, jumping to (30.1, -35.5)
        against a truth of (20.0, 7.2) for about 0.2 s, and two tests in this
        file pinned that excursion as a property of tight coupling. It was a
        property of the *dataset*: the shipped accelerometer was map-frame
        where every Chapter 8 filter integrates it as body-frame, so the
        prediction that should have broken the tie was wrong too.

        Written the other way round now, so it fails if the flip returns.
        """
        window = self.scenario["window"]
        t_tc = np.asarray(self.scenario["t_tc"])
        error_tc = np.asarray(self.scenario["error_tc"])
        truth = self.scenario["dataset"]["truth"]
        est_y = np.asarray(self.scenario["tc"]["x_est"])[:, 1]

        # Scoped to the outage plus recovery: the ambiguity only exists while
        # two anchors are missing. (TC's largest error over the whole run is
        # the initialisation transient at t = 0, where the platform is on the
        # y = 0 leg and "which side of the baseline" is not yet a question.)
        inside = (t_tc >= window[0]) & (t_tc <= window[1] + 3.0)
        true_y = np.interp(t_tc[inside], truth["t"], truth["p_xy"][:, 1])

        # Measured peak is 0.06 m over outage-plus-recovery.
        self.assertLess(float(error_tc[inside].max()), 1.0)

        # The platform is well clear of the anchor baseline throughout, and
        # the estimate stays on the same side of it. A branch flip is exactly
        # a sign disagreement between these two.
        self.assertGreater(float(true_y.min()), 1.0)
        self.assertGreater(float(est_y[inside].min()), 0.0)

    def test_tc_beats_lc_over_the_whole_run(self):
        """The one number where LC used to win, and no longer does.

        With the mirror flip gone there is no sub-second excursion to exclude
        or apologise for: TC measures 0.029 m whole-run against LC's 0.127 m,
        about 0.23x. The previous version of this test asserted the opposite
        inequality (``rmse_tc > rmse_lc``) and carried a paragraph explaining
        why the one adverse number was honest to report. It was honest; it was
        also an artifact.
        """
        rmse_lc = float(np.sqrt(np.mean(self.scenario["error_lc"] ** 2)))
        rmse_tc = float(np.sqrt(np.mean(self.scenario["error_tc"] ** 2)))

        self.assertLess(rmse_tc, rmse_lc)
        self.assertLess(rmse_tc, 0.5 * rmse_lc)


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
