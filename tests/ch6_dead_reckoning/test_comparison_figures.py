"""Tests for the Chapter 6 dead-reckoning comparison example.

Two properties of the committed figures are pinned here, both of which used to
fail silently:

  1. Reproducibility. add_sensor_noise drew from the unseeded global RNG, so
     every run picked a different bias vector and the IMU-only track drifted
     off in a different direction. The committed figures could not be
     regenerated, which also made any figure diff meaningless.
  2. The zoom panel's claim. The comparison overlays an unbounded method
     against bounded ones, and a trajectory plot needs equal axes, so a single
     pair of limits collapses everything near the truth into one blob. The
     second panel exists to fix that, and it is only worth having if it really
     does frame the ground-truth path.

Author: Li-Ta Hsu
References: Chapter 6, Sections 6.1-6.4
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch6_dead_reckoning.example_comparison import (
    DEFAULT_SEED,
    add_sensor_noise,
    generate_mixed_trajectory,
)
from core.eval import plot_trajectory_2d
from core.sensors import FrameConvention, IMUNoiseParams


class TestSensorNoiseIsReproducible(unittest.TestCase):
    """The example's only source of randomness must be seed-controlled."""

    @classmethod
    def setUpClass(cls):
        # Short: this exercises the noise draw, not the trajectory shape.
        cls.dt = 0.01
        frame = FrameConvention.create_enu()
        (_, _, _, cls.accel_body, cls.gyro_body, _, cls.mag_body, _, cls.wheel_true) = (
            generate_mixed_trajectory(duration=2.0, dt=cls.dt, frame=frame)
        )
        cls.imu_params = IMUNoiseParams.consumer_grade()

    def _noise(self, seed):
        """Run add_sensor_noise at ``seed``."""
        return add_sensor_noise(
            self.accel_body,
            self.gyro_body,
            self.mag_body,
            self.wheel_true,
            self.dt,
            self.imu_params,
            seed=seed,
        )

    def test_same_seed_gives_identical_measurements(self):
        """Two runs at one seed agree bit for bit, on every sensor.

        Bit-exactness rather than allclose: the point is that regenerating the
        committed figures reproduces them, and save_figure already writes
        byte-reproducible output.
        """
        first = self._noise(DEFAULT_SEED)
        second = self._noise(DEFAULT_SEED)

        for label, a, b in zip(("accel", "gyro", "mag", "wheel"), first, second):
            with self.subTest(sensor=label):
                np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_measurements(self):
        """The seed is actually wired through, not ignored.

        Without this, a function that hardcoded one draw would pass the
        equality test above while silently ignoring its argument.
        """
        accel_a, gyro_a, _, _ = self._noise(DEFAULT_SEED)
        accel_b, gyro_b, _, _ = self._noise(DEFAULT_SEED + 1)

        self.assertFalse(np.array_equal(accel_a, accel_b))
        self.assertFalse(np.array_equal(gyro_a, gyro_b))

    def test_bias_is_seed_controlled_not_just_white_noise(self):
        """The constant offset differs between seeds, not only the jitter.

        The bias is what sets the *direction* the unaided IMU drifts, and it is
        a single 3-vector drawn once -- averaging over the run recovers it,
        while the zero-mean white noise averages away.
        """
        accel_a, _, _, _ = self._noise(DEFAULT_SEED)
        accel_b, _, _, _ = self._noise(DEFAULT_SEED + 1)

        offset_a = np.mean(accel_a - self.accel_body, axis=0)
        offset_b = np.mean(accel_b - self.accel_body, axis=0)

        self.assertGreater(np.linalg.norm(offset_a - offset_b), 1e-6)


class TestComparisonZoomPanel(unittest.TestCase):
    """The zoom panel must frame the truth at Chapter 6's dynamic range.

    Uses the real geometry -- a 30 x 20 m rectangle against a diverging track
    that reaches hundreds of metres -- because that ratio is what breaks the
    single-panel version.
    """

    @classmethod
    def setUpClass(cls):
        frame = FrameConvention.create_enu()
        # dt coarse and duration short: only the path's extent matters here.
        _, pos_true, _, _, _, _, _, _, _ = generate_mixed_trajectory(
            duration=120.0, dt=0.1, frame=frame
        )
        cls.truth_xy = pos_true[:, :2]
        # Stand in for unaided strapdown: a ramp out to ~500 m.
        n = len(cls.truth_xy)
        ramp = np.linspace(0.0, 500.0, n)
        cls.diverging = np.column_stack([ramp, -0.5 * ramp])

    def setUp(self):
        self.fig = plot_trajectory_2d(
            self.truth_xy,
            {"diverging": self.diverging, "bounded": self.truth_xy + 0.5},
            axis_labels=("East [m]", "North [m]"),
            zoom_to_truth=True,
        )
        self.addCleanup(plt.close, self.fig)

    def test_produces_two_panels(self):
        """Full extent plus zoom."""
        self.assertEqual(len(self.fig.axes), 2)

    def test_zoom_panel_contains_the_whole_truth_path(self):
        """Every truth sample falls inside the zoomed limits.

        This is the panel's entire justification; if the truth is clipped, the
        reader is being shown a partial path with no indication of it.
        """
        x_lo, x_hi = self.fig.axes[1].get_xlim()
        y_lo, y_hi = self.fig.axes[1].get_ylim()

        self.assertLessEqual(x_lo, self.truth_xy[:, 0].min())
        self.assertGreaterEqual(x_hi, self.truth_xy[:, 0].max())
        self.assertLessEqual(y_lo, self.truth_xy[:, 1].min())
        self.assertGreaterEqual(y_hi, self.truth_xy[:, 1].max())

    def test_zoom_panel_excludes_the_diverging_track(self):
        """The zoom is a zoom: it must not stretch to the far end of the drift.

        Without this the test above would pass on limits that span the full
        500 m, which is precisely the unreadable figure being fixed.
        """
        x_hi = self.fig.axes[1].get_xlim()[1]

        self.assertLess(x_hi, self.diverging[:, 0].max() / 2)

    def test_full_panel_still_shows_the_whole_drift(self):
        """The unbounded drift is the lesson of Section 6.1; don't crop it."""
        x_lo, x_hi = self.fig.axes[0].get_xlim()
        y_lo, y_hi = self.fig.axes[0].get_ylim()

        self.assertGreaterEqual(x_hi, self.diverging[:, 0].max())
        self.assertLessEqual(y_lo, self.diverging[:, 1].min())
        self.assertLessEqual(x_lo, self.truth_xy[:, 0].min())
        self.assertGreaterEqual(y_hi, self.truth_xy[:, 1].max())

    def test_both_panels_draw_the_same_series(self):
        """A zoom that silently dropped a method would misreport the comparison."""
        full_labels = [line.get_label() for line in self.fig.axes[0].lines]
        zoom_labels = [line.get_label() for line in self.fig.axes[1].lines]

        self.assertEqual(full_labels, zoom_labels)
        self.assertIn("diverging", zoom_labels)


if __name__ == "__main__":
    unittest.main()
