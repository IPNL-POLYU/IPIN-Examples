"""The tuning demo's gating result is a claim, so it is tested as one.

Chi-square gating scores ~18 m RMSE against a ~0.7 m baseline in this demo,
which looks like a bug and is not. The figure and the printed summary now
explain why; this file pins the explanation, link by link, so that a future
change to the tuning either keeps the story true or fails here.

The chain being asserted:

  1. The dataset really is NLOS-corrupted -- about half the ranges carry a bias
     an order of magnitude above the 0.05 m the filter is told to expect.
  2. So the ungated filter is over-confident: its NIS sits far above the
     chi-square(1) values a consistent filter would produce.
  3. So a 95% gate rejects most measurements rather than a few outliers.
  4. So the gated filter starves and diverges, while the robust losses, facing
     exactly the same mis-specified R, do not.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3; Eqs. (8.6), (8.7), (8.9)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset
from ch8_sensor_fusion.tuning_robust_demo import run_fusion_with_strategy
from core.eval import compute_rmse

# The demo's assumed range noise, from run_fusion_with_strategy.
ASSUMED_RANGE_NOISE_STD = 0.05

# Median of the chi-square distribution with one degree of freedom, and the
# 95% critical value the gate uses.
CHI2_1DOF_MEDIAN = 0.4549
CHI2_1DOF_95 = 3.841


def _rmse_against_truth(result, truth):
    """Horizontal RMSE of an estimate against the truth, at truth samples.

    Norm first, then RMS. Handing the (N, 2) error vectors to compute_rmse
    averages over 2N components rather than N positions, giving the per-axis
    RMS -- smaller by exactly sqrt(2), and not what "position RMSE" means.
    """
    est = np.asarray(result["x_est"])[:, :2]
    t_est = np.asarray(result["t"])
    idx = np.searchsorted(truth["t"], t_est).clip(0, len(truth["t"]) - 1)
    return float(compute_rmse(np.linalg.norm(est - truth["p_xy"][idx], axis=1)))


class TestGatingFailureIsExplained(unittest.TestCase):
    """Each link in the published explanation, asserted separately."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_fusion_dataset("data/sim/ch8_fusion_2d_imu_uwb_nlos")
        cls.truth = cls.dataset["truth"]
        cls.baseline = run_fusion_with_strategy(cls.dataset, strategy="baseline")
        cls.gating = run_fusion_with_strategy(
            cls.dataset, strategy="gating", use_gating=True
        )
        cls.huber = run_fusion_with_strategy(cls.dataset, strategy="huber")

    def test_dataset_ranges_are_nlos_corrupted(self):
        """Link 1: the measurements really are far worse than R claims.

        Without this the rest of the story could be blamed on the filter alone.
        """
        uwb = self.dataset["uwb"]
        anchors = np.asarray(self.dataset["uwb_anchors"])
        idx = np.searchsorted(self.truth["t"], uwb["t"]).clip(
            0, len(self.truth["t"]) - 1
        )
        true_range = np.linalg.norm(
            self.truth["p_xy"][idx][:, None, :] - anchors[None, :, :2], axis=2
        )
        error = np.abs(np.asarray(uwb["ranges"]) - true_range)
        error = error[np.isfinite(error)]

        # A substantial minority sit an order of magnitude above the assumed
        # noise -- that is what makes this an NLOS dataset.
        fraction_biased = float(np.mean(error > 10 * ASSUMED_RANGE_NOISE_STD))
        self.assertGreater(fraction_biased, 0.2)

    def test_ungated_filter_is_overconfident(self):
        """Link 2: NIS is far above chi-square(1), with no gate to blame.

        The baseline never rejects anything, so it cannot have been driven off
        course by its own rejections. Whatever its NIS says is a statement
        about the covariance, not about a feedback loop.
        """
        nis = np.asarray(self.baseline["nis"])

        self.assertGreater(np.median(nis), 20 * CHI2_1DOF_MEDIAN)
        # ...and most samples would fail the gate the next test applies.
        self.assertLess(float(np.mean(nis < CHI2_1DOF_95)), 0.5)

    def test_the_gate_rejects_most_measurements_not_a_few_outliers(self):
        """Link 3: a 95% gate keeping under half its data is not filtering."""
        accepted = self.gating["n_uwb_accepted"]
        total = accepted + self.gating["n_uwb_rejected"]

        self.assertLess(accepted / total, 0.5)

    def test_gating_diverges_while_a_robust_loss_survives(self):
        """Link 4: the contrast the figure exists to draw.

        Both face the same mis-specified R, so the difference is the hard
        rejection rather than the noise model.
        """
        rmse_baseline = _rmse_against_truth(self.baseline, self.truth)
        rmse_gating = _rmse_against_truth(self.gating, self.truth)
        rmse_huber = _rmse_against_truth(self.huber, self.truth)

        self.assertGreater(rmse_gating, 10 * rmse_baseline)
        self.assertLess(rmse_huber, 2 * rmse_baseline)

    def test_baseline_still_tracks_despite_being_overconfident(self):
        """A caveat worth pinning: the position is fine, the covariance is not.

        Guards against "fixing" the over-confidence by concluding the whole
        baseline is broken -- it tracks to well under a metre.
        """
        self.assertLess(_rmse_against_truth(self.baseline, self.truth), 2.0)


if __name__ == "__main__":
    unittest.main()
