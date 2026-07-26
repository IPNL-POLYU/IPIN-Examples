"""The calibration demo's residual must be checked against the right number.

It printed "Alignment RMSE after calibration: 0.1021 m (Should be close to
measurement noise: ~0.05 m)" -- inviting the reader to conclude the calibration
was twice as bad as it should be. It is not. The residual is a *difference* of
two independently noisy sensor positions, so its per-axis standard deviation is
sigma*sqrt(2), and the reported figure is the RMS of the 2-D magnitude, which
brings another sqrt(2). The expectation is 2*sigma = 0.10 m, and 0.1021 m
agrees with it to 2%.

Same shape as the other Chapter 8 defects found in this sweep: a stated
expectation that did not match the physics of the quantity being measured, so a
correct result looked wrong.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.4 (Spatial and Temporal Calibration)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch8_sensor_fusion.calibration_demo import (
    calibrate_extrinsic_2d_least_squares,
    generate_synthetic_extrinsic_data,
)


class TestExtrinsicCalibrationClaims(unittest.TestCase):
    """The estimate is near-exact, and the residual matches theory."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.data = generate_synthetic_extrinsic_data()
        cls.R_est, cls.t_est = calibrate_extrinsic_2d_least_squares(
            cls.data["p_sensor1"], cls.data["p_sensor2"]
        )

    def test_recovers_the_rotation_and_lever_arm(self):
        """Least squares should be essentially exact at this noise level."""
        # true_rotation_angle is stored in radians; the demo converts only for
        # printing.
        angle = float(np.arctan2(self.R_est[1, 0], self.R_est[0, 0]))

        self.assertAlmostEqual(
            np.degrees(angle),
            np.degrees(self.data["true_rotation_angle"]),
            delta=0.5,
        )
        np.testing.assert_allclose(self.t_est, self.data["true_t"], atol=0.02)

    def test_residual_matches_two_sigma_not_one(self):
        """The corrected expectation, derived rather than quoted.

        Guards both directions. A residual near sigma would mean the fit is
        absorbing noise it should not; one far above 2*sigma would mean the
        calibration is genuinely poor. The old note expected sigma, so a
        correct answer read as a failure.
        """
        residuals = self.data["p_sensor2"] - (
            (self.R_est @ self.data["p_sensor1"].T).T + self.t_est
        )
        rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        expected = 2.0 * self.data["noise_std"]

        self.assertAlmostEqual(rmse / expected, 1.0, delta=0.15)

    def test_the_generator_reports_the_noise_it_used(self):
        """The expectation is derived from the data, not kept in sync by hand.

        This key exists so the printed sanity check cannot drift away from the
        noise actually injected -- which is how the wrong benchmark survived.
        """
        self.assertIn("noise_std", self.data)
        self.assertGreater(self.data["noise_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
