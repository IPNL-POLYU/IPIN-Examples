"""Every method in the Chapter 6 comparison must actually navigate.

The comparison reports RMSE, final error, median and 90th percentile for four
dead reckoning methods, and every one of those numbers can be satisfied by an
estimator that never leaves the start point. The ground truth is a closed loop,
so standing still is exactly right at loop closure; and RMSE for a stationary
estimate is just the mean distance of the truth from the origin. Two of the
four methods were doing precisely that while the example printed "90-95% error
reduction", and no test noticed.

This file is the check that would have noticed. It is deliberately separate
from test_comparison_figures.py so that the in-flight fix to the detectors can
add its own tests without a merge conflict here.

Author: Li-Ta Hsu
References: Chapter 6, Sections 6.1-6.4
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch6_dead_reckoning.example_comparison import (
    DEFAULT_SEED,
    LEVER_ARM_A,
    add_sensor_noise,
    generate_mixed_trajectory,
    run_imu_only,
    run_imu_zupt,
    run_pdr,
    run_wheel_odom,
)
from core.eval import motion_ratio, path_length
from core.sensors import FrameConvention, IMUNoiseParams, NavStateQPVP

# A method tracing less than this fraction of the truth is not navigating, it
# is sitting still. Loose on purpose: this is a smoke alarm, not an accuracy
# bound -- accuracy is what RMSE and the CDF are for.
MIN_MOTION_RATIO = 0.5

# Unaided strapdown is expected to overshoot; it is the one method allowed to.
MAX_MOTION_RATIO = 3.0

# Methods still known to be pinned at the start point. Empty, and it should
# stay that way: "IMU + ZUPT" and "PDR (Mag)" were listed here when this file
# was written, and the entries came out the moment the detector fix landed --
# the ratchet asserted they were *still* frozen, so the fix turned this test
# red and forced the promotion instead of leaving a silent allowance behind.
KNOWN_FROZEN: set = set()


class TestMethodsActuallyMove(unittest.TestCase):
    """Each method's traced path must be comparable to the truth's."""

    @classmethod
    def setUpClass(cls):
        # Half the example's 120 s: long enough to cover several legs of the
        # rectangle and both corners, short enough to keep the suite quick.
        duration, dt = 60.0, 0.01
        frame = FrameConvention.create_enu()
        imu_params = IMUNoiseParams.consumer_grade()

        (t, pos_true, vel_true, accel_body, gyro_body, _, mag_body, _, wheel_true) = (
            generate_mixed_trajectory(duration=duration, dt=dt, frame=frame)
        )
        accel, gyro, mag, wheel = add_sensor_noise(
            accel_body,
            gyro_body,
            mag_body,
            wheel_true,
            dt,
            imu_params,
            seed=DEFAULT_SEED,
        )
        initial = NavStateQPVP(
            q=np.array([1.0, 0.0, 0.0, 0.0]), v=vel_true[0], p=pos_true[0]
        )

        # run_imu_zupt and run_pdr each return (positions, diagnostics). The
        # diagnostics -- the detection mask and the step count -- came with the
        # detector fix and are not what this file is guarding.
        zupt_pos, _ = run_imu_zupt(t, accel, gyro, initial, frame, imu_params)
        pdr_pos, _ = run_pdr(t, accel, mag, 1.75)

        cls.truth_xy = pos_true[:, :2]
        cls.results = {
            "IMU Only": run_imu_only(t, accel, gyro, initial, frame)[:, :2],
            "IMU + ZUPT": zupt_pos[:, :2],
            "Wheel Odom": run_wheel_odom(t, wheel, gyro, initial, LEVER_ARM_A)[:, :2],
            "PDR (Mag)": pdr_pos[:, :2],
        }

    def test_truth_actually_walks(self):
        """Guard the guard: a degenerate truth would make everything pass."""
        self.assertGreater(path_length(self.truth_xy), 10.0)

    def test_every_method_traces_a_comparable_path(self):
        """No method may score well by standing still.

        Methods in KNOWN_FROZEN are asserted to be *still broken*, so that the
        fix flips this test rather than leaving a silent allowance behind.
        """
        for name, est_xy in self.results.items():
            with self.subTest(method=name):
                ratio = motion_ratio(est_xy, self.truth_xy)

                if name in KNOWN_FROZEN:
                    self.assertLess(
                        ratio,
                        MIN_MOTION_RATIO,
                        f"{name} now moves (ratio {ratio:.3f}). The detector "
                        f"fix has landed -- remove it from KNOWN_FROZEN so "
                        f"this test starts guarding it for real.",
                    )
                    continue

                self.assertGreater(
                    ratio,
                    MIN_MOTION_RATIO,
                    f"{name} traced only {ratio:.1%} of the truth's path "
                    f"length; it is not navigating, and its error metrics are "
                    f"an artefact of the truth returning to its start.",
                )

    def test_only_the_unaided_integrator_may_overshoot(self):
        """A corrected method that wanders far has a different bug.

        Splitting the two directions matters: too little motion means a dead
        detector, too much means uncorrected drift, and one threshold either
        side of 1 would conflate them.
        """
        for name, est_xy in self.results.items():
            if name == "IMU Only" or name in KNOWN_FROZEN:
                continue
            with self.subTest(method=name):
                ratio = motion_ratio(est_xy, self.truth_xy)
                self.assertLess(
                    ratio,
                    MAX_MOTION_RATIO,
                    f"{name} traced {ratio:.1f}x the truth's path length, "
                    f"which for a corrected method means drift it should be "
                    f"removing.",
                )

    def test_unaided_imu_does_drift(self):
        """The chapter's premise, pinned: pure integration is unbounded."""
        final_error = np.linalg.norm(self.results["IMU Only"][-1] - self.truth_xy[-1])

        self.assertGreater(final_error, 1.0)


if __name__ == "__main__":
    unittest.main()
