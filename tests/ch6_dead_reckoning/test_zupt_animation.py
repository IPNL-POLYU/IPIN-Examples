"""Tests for the Chapter 6 ZUPT drift animation.

The animation makes a claim -- that ZUPT keeps the drift bounded while pure
strapdown does not -- so the claim is tested, not just the rendering. Figure
code otherwise rots silently: nothing fails until a human opens the GIF.

Author: Li-Ta Hsu
References: Chapter 6, Section 6.5, Eqs. (6.44)-(6.45)
"""

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch6_dead_reckoning.example_zupt import (
    add_imu_noise,
    animate_zupt_drift,
    generate_walking_trajectory,
    run_imu_only,
    run_imu_with_zupt_ekf,
)
from core.eval import save_animation
from core.sensors import FrameConvention, IMUNoiseParams, NavStateQPVP


class TestZuptDriftAnimation(unittest.TestCase):
    """Render the animation and check what it asserts."""

    @classmethod
    def setUpClass(cls):
        # Shorter than the example's 60 s so the suite stays quick, but long
        # enough to contain several walk/stance cycles (7 s per cycle).
        np.random.seed(42)
        frame = FrameConvention.create_enu()
        imu_params = IMUNoiseParams.consumer_grade()

        (cls.t, cls.pos_true, vel_true, quat_true,
         accel_body, gyro_body, cls.stance_mask) = generate_walking_trajectory(
            duration=21.0, dt=0.01, step_freq=2.0, step_length=0.7, frame=frame
        )
        accel_meas, gyro_meas = add_imu_noise(
            accel_body, gyro_body, 0.01, imu_params
        )
        initial_state = NavStateQPVP(
            q=quat_true[0].copy(), v=vel_true[0].copy(), p=cls.pos_true[0].copy()
        )

        cls.pos_imu, _ = run_imu_only(
            cls.t, accel_meas, gyro_meas, initial_state, frame
        )
        cls.pos_zupt, _, cls.detections = run_imu_with_zupt_ekf(
            cls.t, accel_meas, gyro_meas, initial_state, frame, imu_params,
            window_size=10, gamma=1000.0,
        )

    def _errors(self):
        error_imu = np.linalg.norm(
            self.pos_imu[:, :2] - self.pos_true[:, :2], axis=1
        )
        error_zupt = np.linalg.norm(
            self.pos_zupt[:, :2] - self.pos_true[:, :2], axis=1
        )
        return error_imu, error_zupt

    def test_zupt_bounds_the_drift(self):
        """The animation's whole point: ZUPT ends far closer than strapdown."""
        error_imu, error_zupt = self._errors()

        self.assertLess(error_zupt[-1], error_imu[-1])
        self.assertLess(
            np.sqrt(np.mean(error_zupt ** 2)),
            0.5 * np.sqrt(np.mean(error_imu ** 2)),
        )

    def test_trajectory_contains_stance_and_swing(self):
        """Stance bands only mean something if both phases occur."""
        self.assertTrue(self.stance_mask.any())
        self.assertFalse(self.stance_mask.all())

    def test_animation_renders_every_frame(self):
        """Frame count matches the request and each frame draws."""
        fig, update, n_frames = animate_zupt_drift(
            self.t, self.pos_true, self.pos_imu, self.pos_zupt,
            self.stance_mask, self.detections, n_frames=6,
        )
        try:
            self.assertEqual(n_frames, 6)
            for frame in range(n_frames):
                axes = update(frame)
                self.assertEqual(len(axes), 3)
        finally:
            plt.close(fig)

    def test_animation_stays_small(self):
        """Committed binaries live in git history forever."""
        fig, update, n_frames = animate_zupt_drift(
            self.t, self.pos_true, self.pos_imu, self.pos_zupt,
            self.stance_mask, self.detections, n_frames=8,
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(fig, update, n_frames, tmp, "zupt", fps=5)
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()
