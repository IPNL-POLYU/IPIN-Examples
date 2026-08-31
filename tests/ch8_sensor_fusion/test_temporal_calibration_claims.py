"""The temporal calibration demo must actually demonstrate temporal calibration.

It did not. With chi-square gating on -- the previous default -- the filter
starved and drifted to 17.8 m RMSE, which had nothing to do with timing and
was 600x the effect the file exists to show. The demo then reported "a -50 ms
offset causes 17.78 m RMSE", attributing an unrelated divergence to the offset,
and "correction improves RMSE by 0.1%", which made a correct TimeSyncModel look
broken. Both statements were false. The offset costs about 0.02 m here, and the
correction recovers essentially all of it -- 8% of the remaining error.

The proof that the divergence was unrelated: the zero-offset dataset produced
the same 17.760 m.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.5 (Temporal Calibration)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.fusion import load_fusion_dataset
from ch8_sensor_fusion.example_temporal_calibration import run_fusion_with_time_sync
from core.eval import compute_position_rmse

OFFSET_DATASET = "data/sim/ch8_fusion_2d_imu_uwb_timeoffset"
CLEAN_DATASET = "data/sim/ch8_fusion_2d_imu_uwb"


def _rmse(dataset, history):
    """Horizontal position RMSE of a run against its truth."""
    truth = dataset["truth"]
    p_true = np.column_stack(
        [
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 0]),
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 1]),
        ]
    )
    return compute_position_rmse(np.asarray(history["x_est"])[:, :2] - p_true)


class TestTemporalCalibrationIsVisible(unittest.TestCase):
    """The correction's effect must not be swamped by an unrelated failure."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_fusion_dataset(OFFSET_DATASET)
        cls.uncorrected = run_fusion_with_time_sync(cls.dataset, apply_correction=False)
        cls.corrected = run_fusion_with_time_sync(cls.dataset, apply_correction=True)

    def test_the_demo_defaults_to_a_filter_that_works(self):
        """Sub-metre on both runs, so the timing effect is not buried.

        This is the guard on the gating default. Turning it back on sends both
        runs to ~17.8 m and this fails.
        """
        self.assertLess(_rmse(self.dataset, self.uncorrected), 1.0)
        self.assertLess(_rmse(self.dataset, self.corrected), 1.0)

    def test_correction_measurably_improves_accuracy(self):
        """The claim the figure makes, as a number rather than a hope."""
        uncorrected = _rmse(self.dataset, self.uncorrected)
        corrected = _rmse(self.dataset, self.corrected)

        self.assertLess(corrected, uncorrected)
        improvement = (uncorrected - corrected) / uncorrected
        self.assertGreater(improvement, 0.03)

    def test_the_offset_costs_what_the_kinematics_predict(self):
        """Sanity-check the magnitude instead of accepting whatever comes out.

        A platform moving at v whose ranges are stamped dt late is fused
        against a position v*dt away, so the cost should sit in that
        neighbourhood -- not orders of magnitude off it, in either direction.
        A wrong-signed correction or a no-op would both break this.
        """
        offset_s = abs(
            self.dataset["config"]["temporal_calibration"]["time_offset_sec"]
        )
        speed = float(np.mean(np.linalg.norm(self.dataset["truth"]["v_xy"], axis=1)))
        predicted = speed * offset_s

        measured = _rmse(self.dataset, self.uncorrected) - _rmse(
            self.dataset, self.corrected
        )

        self.assertGreater(measured, 0.1 * predicted)
        self.assertLess(measured, 10.0 * predicted)

    def test_gating_no_longer_starves_the_filter(self):
        """The re-attribution, pinned so the old diagnosis cannot come back.

        This test used to assert the opposite: that gating on the *clean*,
        zero-offset dataset diverged past 10 m, proving the large error was a
        starvation feedback loop rather than a temporal effect. The second
        half of that reasoning was right -- it was never a timing problem --
        and the first half named the wrong culprit.

        The innovations were heavy-tailed because the shipped accelerometer
        was map-frame where this filter integrates it as body-frame, so a
        Gaussian gate rejected far too much. With the frame corrected the gate
        costs 1.5 mm on this dataset: 0.0214 m gated against 0.0199 m ungated.

        Gating stays off by default here because this demo is about temporal
        alignment and a gate is a second variable, not because it is unsafe.
        """
        clean = load_fusion_dataset(CLEAN_DATASET)
        gated = run_fusion_with_time_sync(
            clean, apply_correction=False, use_gating=True
        )
        ungated = run_fusion_with_time_sync(
            clean, apply_correction=False, use_gating=False
        )

        self.assertLess(_rmse(clean, gated), 0.1)
        self.assertLess(_rmse(clean, ungated), 0.1)
        # A gate can cost a little; it must not cost an order of magnitude.
        self.assertLess(_rmse(clean, gated), 2.0 * _rmse(clean, ungated))


if __name__ == "__main__":
    unittest.main()
