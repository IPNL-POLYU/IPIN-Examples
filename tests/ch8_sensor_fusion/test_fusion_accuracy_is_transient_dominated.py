"""Chapter 8's headline fusion RMSE measures a trajectory artifact.

Three demos report a single RMSE for tightly and loosely coupled UWB+IMU
fusion, and the number is about 20x worse than the ranging that feeds it: the
clean dataset's range errors have median 0.035 m, yet TC reports 0.739 m.

The gap is not the filter. Median position error is 0.026 m, which is what
3.5 cm ranging with four anchors should give. But 7.6% of samples exceed
0.5 m, peaking at 4.5 m, and those alone lift the RMS to 0.739 m -- excluding
them it is 0.074 m.

Those excursions are the two seconds after each 90-degree corner, with all
four anchors still visible. The trajectory turns *instantaneously*: yaw steps
from 90 to 180 degrees inside one sample, a rate of 9000 deg/s, and the IMU
forward model faithfully reports 4501 deg/s and 5.1 g. No estimator tracks
that, so the filter lags for a couple of seconds after every corner.

A real indoor platform at 1 m/s turns in a second or two, tens of deg/s. The
underlying fix is a finite turn rate in the trajectory generator, which would
change every committed Chapter 8 dataset, figure and number -- deliberately not
done here. What is done is refusing to let the artifact keep masquerading as a
sensor-fusion accuracy result.

Author: Li-Ta Hsu
References: Chapter 8, Sections 8.1-8.3
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import contextlib
import io

import numpy as np

from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset, run_tc_fusion
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

    def test_rmse_is_dominated_by_a_small_fraction_of_samples(self):
        """A handful of excursions carry the whole headline number.

        If this ever fails because the two converge, the distribution has
        stopped being bimodal and the single-number summary becomes honest --
        at which point the warnings in these demos should go.
        """
        transient = self.error > TRANSIENT_THRESHOLD_M
        rmse_all = compute_position_rmse(self.error)
        rmse_excluding = compute_position_rmse(self.error[~transient])

        self.assertLess(np.mean(transient), 0.15)
        self.assertGreater(rmse_all, 5.0 * rmse_excluding)

    def test_the_trajectory_turns_faster_than_anything_physical(self):
        """The cause, pinned at its source rather than at the symptom.

        Fixing the generator to turn at a plausible rate should make this fail,
        which is the point: it is the signal that the transients -- and the
        inflated RMSE -- can stop being explained away.
        """
        truth = self.dataset["truth"]
        yaw_rate = np.abs(np.diff(np.unwrap(truth["yaw"]))) / np.diff(truth["t"])

        self.assertGreater(np.degrees(yaw_rate.max()), PLAUSIBLE_TURN_RATE_DEG_S)


if __name__ == "__main__":
    unittest.main()
