"""Chapter 8's fusion accuracy must reflect the sensors, not the trajectory.

It did not. Three demos reported a single RMSE about 20x worse than the
ranging feeding them -- 0.739 m against range errors of 0.035 m median -- and
the gap was never the filter. Median position error was already 0.026 m. But
7.6% of samples exceeded 0.5 m and peaked at 4.5 m, and those alone lifted the
RMS tenfold; excluding them it was 0.074 m.

Those excursions were the two seconds after each corner, with all four anchors
visible. The trajectory turned *instantaneously*: yaw stepped 90 degrees inside
one sample, 9000 deg/s, which the IMU forward model rendered as 4501 deg/s and
5.1 g. No estimator follows a step.

generate_rectangular_trajectory now rounds each corner, giving 57.3 deg/s at
1 m/s, and RMSE fell to 0.167 m with the median unchanged -- confirming the
steady state had been right all along and only the artifact was being measured.
These tests hold that in place and keep the diagnosis attached to its cause, so
a regression in the generator surfaces here rather than as a mysteriously worse
fusion result.

Author: Li-Ta Hsu
References: Chapter 8, Sections 8.1-8.3
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import contextlib
import io

import numpy as np

from core.fusion import load_fusion_dataset, run_tc_fusion
from core.eval import compute_position_rmse

CLEAN_DATASET = "data/sim/ch8_fusion_2d_imu_uwb"

# Above this the filter is in a post-corner transient rather than tracking.
TRANSIENT_THRESHOLD_M = 0.5

# A pedestrian or indoor robot turning a corner in a second or so. Anything
# far above this is not a manoeuvre, it is a discontinuity.
PLAUSIBLE_TURN_RATE_DEG_S = 360.0


class TestFusionAccuracyIsTransientDominated(unittest.TestCase):
    """Separate the steady-state result from the artifact inflating it."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_fusion_dataset(CLEAN_DATASET)
        with contextlib.redirect_stdout(io.StringIO()):
            history = run_tc_fusion(cls.dataset)

        truth = cls.dataset["truth"]
        t = np.asarray(history["t"])
        estimate = np.asarray(history["x_est"])[:, :2]
        p_true = np.column_stack(
            [
                np.interp(t, truth["t"], truth["p_xy"][:, 0]),
                np.interp(t, truth["t"], truth["p_xy"][:, 1]),
            ]
        )
        cls.error = np.linalg.norm(estimate - p_true, axis=1)

    def test_ranging_is_centimetre_level(self):
        """The premise: this dataset is not the NLOS one, its ranges are good."""
        uwb = self.dataset["uwb"]
        anchors = np.asarray(self.dataset["uwb_anchors"])
        truth = self.dataset["truth"]
        idx = np.searchsorted(truth["t"], uwb["t"]).clip(0, len(truth["t"]) - 1)
        true_range = np.linalg.norm(
            truth["p_xy"][idx][:, None, :] - anchors[None, :, :2], axis=2
        )
        range_error = np.abs(np.asarray(uwb["ranges"]) - true_range)
        range_error = range_error[np.isfinite(range_error)]

        self.assertLess(np.median(range_error), 0.1)

    def test_steady_state_accuracy_matches_the_ranging(self):
        """Typical error is centimetres, as the sensors imply.

        This is the result the chapter has actually earned, and the headline
        RMSE hides it.
        """
        self.assertLess(np.median(self.error), 0.1)

    def test_the_transients_have_largely_gone(self):
        """What the generator fix bought, held in place.

        Before the corners were rounded, 7.6% of samples exceeded 0.5 m,
        peaking at 4.5 m, and the RMS sat ten times above its own median. The
        residual is ordinary manoeuvre lag: a filter still takes a moment to
        follow a 57 deg/s turn, and that is a tuning question rather than an
        artifact.
        """
        transient = self.error > TRANSIENT_THRESHOLD_M
        rmse_all = compute_position_rmse(self.error)

        self.assertLess(np.mean(transient), 0.05)
        self.assertLess(self.error.max(), 2.0)
        self.assertLess(rmse_all, 0.3)

    def test_the_trajectory_turns_at_a_physical_rate(self):
        """The cause, fixed at its source.

        This assertion used to run the other way, as the signal that the
        generator still stepped yaw by 90 degrees inside one sample: 9000
        deg/s, which the IMU model turned into 5.1 g. Rounding the corners
        brought it to speed / corner_radius, about 57 deg/s at 1 m/s, and the
        transients above went with it.
        """
        truth = self.dataset["truth"]
        yaw_rate = np.abs(np.diff(np.unwrap(truth["yaw"]))) / np.diff(truth["t"])

        self.assertLess(np.degrees(yaw_rate.max()), PLAUSIBLE_TURN_RATE_DEG_S)


if __name__ == "__main__":
    unittest.main()
