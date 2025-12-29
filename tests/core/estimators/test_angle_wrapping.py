"""
Unit tests for angle wrapping in EKF/IEKF/UKF bearing measurements.

Tests that filters correctly handle the pi <-> -pi discontinuity
in bearing measurements.

Book Reference: Chapter 3, EKF/IEKF update equations where
innovation nu = z - z_pred must be wrapped for angle measurements.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from core.estimators import (
    ExtendedKalmanFilter,
    IteratedExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from core.utils import angle_diff


def create_bearing_only_innovation_func():
    """Create innovation function that wraps all measurements as bearings."""
    def innovation_func(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        return np.array([angle_diff(z[i], z_pred[i]) for i in range(len(z))])
    return innovation_func


class TestEKFAngleWrapping(unittest.TestCase):
    """Test EKF handles bearing crossing pi <-> -pi correctly."""

    def setUp(self):
        """Setup bearing-only tracking scenario."""
        # Process model: constant angular velocity
        def process_model(x, u, dt):
            # State: [angle, angular_velocity]
            return np.array([x[0] + x[1] * dt, x[1]])

        def process_jacobian(x, u, dt):
            return np.array([[1, dt], [0, 1]])

        # Measurement model: direct bearing observation
        def measurement_model(x):
            return np.array([x[0]])

        def measurement_jacobian(x):
            return np.array([[1, 0]])

        def Q_func(dt):
            return 0.01 * np.eye(2)

        def R_func():
            return np.array([[0.05]])

        self.process_model = process_model
        self.process_jacobian = process_jacobian
        self.measurement_model = measurement_model
        self.measurement_jacobian = measurement_jacobian
        self.Q_func = Q_func
        self.R_func = R_func
        self.dt = 0.1

    def test_ekf_without_wrapping_fails_at_pi_crossing(self):
        """Test that EKF WITHOUT angle wrapping produces large error at crossing."""
        # Start just before pi, velocity will carry us past pi
        x0 = np.array([np.pi - 0.3, 0.5])  # Near +pi, moving positive
        P0 = np.diag([0.1, 0.1])

        # EKF without angle wrapping
        ekf_no_wrap = ExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0.copy(), P0.copy(),
            innovation_func=None  # No wrapping!
        )

        # True trajectory crosses pi -> -pi
        true_states = []
        measurements = []
        true_angle = x0[0]

        np.random.seed(42)
        for _ in range(10):
            # Move past pi
            true_angle += 0.5 * self.dt
            # Wrap true angle to [-pi, pi]
            true_angle = np.arctan2(np.sin(true_angle), np.cos(true_angle))
            true_states.append(true_angle)

            # Measurement with small noise
            z = true_angle + np.random.normal(0, 0.1)
            measurements.append(z)

        # Run EKF without wrapping
        errors_no_wrap = []
        for z in measurements:
            ekf_no_wrap.predict(dt=self.dt)
            ekf_no_wrap.update(np.array([z]))
            est, _ = ekf_no_wrap.get_state()
            errors_no_wrap.append(abs(est[0] - true_states[len(errors_no_wrap)]))

        # Should have at least one large error when crossing pi
        max_error = max(errors_no_wrap)
        # Without wrapping, when we cross pi, innovation can be ~2*pi instead of small
        self.assertGreater(max_error, 1.0,
                           f"Expected large error without wrapping, got {max_error}")

    def test_ekf_with_wrapping_handles_pi_crossing(self):
        """Test that EKF WITH angle wrapping correctly handles pi <-> -pi crossing."""
        # Start just before pi, velocity will carry us past pi
        x0 = np.array([np.pi - 0.3, 0.5])  # Near +pi, moving positive
        P0 = np.diag([0.1, 0.1])

        # EKF with angle wrapping
        innovation_func = create_bearing_only_innovation_func()
        ekf_wrap = ExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0.copy(), P0.copy(),
            innovation_func=innovation_func
        )

        # True trajectory crosses pi -> -pi
        true_states = []
        measurements = []
        true_angle = x0[0]

        np.random.seed(42)
        for _ in range(10):
            # Move past pi
            true_angle += 0.5 * self.dt
            # Wrap true angle to [-pi, pi]
            true_angle = np.arctan2(np.sin(true_angle), np.cos(true_angle))
            true_states.append(true_angle)

            # Measurement with small noise
            z = true_angle + np.random.normal(0, 0.1)
            measurements.append(z)

        # Run EKF with wrapping
        errors_wrap = []
        for z in measurements:
            ekf_wrap.predict(dt=self.dt)
            ekf_wrap.update(np.array([z]))
            est, _ = ekf_wrap.get_state()

            # Compute angle error properly (wrapped)
            error = abs(angle_diff(est[0], true_states[len(errors_wrap)]))
            errors_wrap.append(error)

        # With wrapping, errors should remain small throughout
        max_error = max(errors_wrap)
        self.assertLess(max_error, 0.5,
                        f"Expected small error with wrapping, got {max_error}")

    def test_pi_crossing_scenario_detailed(self):
        """Detailed test showing exactly what happens at pi crossing."""
        # Predicted angle just before pi
        z_pred = np.pi - 0.1  # +179.4 degrees

        # True measurement just after pi (wrapped to negative)
        z = -np.pi + 0.2  # -177.1 degrees

        # Without wrapping: innovation = z - z_pred
        innovation_no_wrap = z - z_pred  # = -2*pi + 0.3 ~ -6.0 (WRONG!)

        # With wrapping: innovation = angle_diff(z, z_pred)
        innovation_wrap = angle_diff(z, z_pred)  # ~ +0.3 (CORRECT!)

        # Verify
        self.assertLess(abs(innovation_wrap), 0.5,
                        "Wrapped innovation should be small")
        self.assertGreater(abs(innovation_no_wrap), 5.0,
                           "Unwrapped innovation should be large (~2*pi)")

        print(f"\nPi Crossing Test:")
        print(f"  z_pred (predicted bearing): {np.rad2deg(z_pred):.1f} deg (+179.4 deg)")
        print(f"  z (true bearing):           {np.rad2deg(z):.1f} deg (-177.1 deg)")
        print(f"  Innovation WITHOUT wrap:    {np.rad2deg(innovation_no_wrap):.1f} deg (WRONG!)")
        print(f"  Innovation WITH wrap:       {np.rad2deg(innovation_wrap):.1f} deg (CORRECT)")


class TestIEKFAngleWrapping(unittest.TestCase):
    """Test IEKF handles bearing crossing pi <-> -pi correctly."""

    def test_iekf_with_wrapping_handles_pi_crossing(self):
        """Test IEKF with angle wrapping at pi crossing."""
        # Process model: constant angular velocity
        def process_model(x, u, dt):
            return np.array([x[0] + x[1] * dt, x[1]])

        def process_jacobian(x, u, dt):
            return np.array([[1, dt], [0, 1]])

        def measurement_model(x):
            return np.array([x[0]])

        def measurement_jacobian(x):
            return np.array([[1, 0]])

        def Q_func(dt):
            return 0.01 * np.eye(2)

        def R_func():
            return np.array([[0.05]])

        dt = 0.1

        # Start just before pi
        x0 = np.array([np.pi - 0.3, 0.5])
        P0 = np.diag([0.1, 0.1])

        # IEKF with angle wrapping
        innovation_func = create_bearing_only_innovation_func()
        iekf = IteratedExtendedKalmanFilter(
            process_model, process_jacobian,
            measurement_model, measurement_jacobian,
            Q_func, R_func, x0.copy(), P0.copy(),
            innovation_func=innovation_func,
            max_iterations=3
        )

        # Cross pi
        true_angle = x0[0]
        np.random.seed(42)

        errors = []
        for _ in range(10):
            true_angle += 0.5 * dt
            true_angle = np.arctan2(np.sin(true_angle), np.cos(true_angle))

            z = true_angle + np.random.normal(0, 0.1)

            iekf.predict(dt=dt)
            iekf.update(np.array([z]))
            est, _ = iekf.get_state()

            error = abs(angle_diff(est[0], true_angle))
            errors.append(error)

        # With wrapping, errors should remain small
        max_error = max(errors)
        self.assertLess(max_error, 0.5,
                        f"IEKF with wrapping should have small error, got {max_error}")


class TestUKFAngleWrapping(unittest.TestCase):
    """Test UKF handles bearing crossing pi <-> -pi correctly."""

    def test_ukf_with_wrapping_handles_pi_crossing(self):
        """Test UKF with angle wrapping at pi crossing."""
        # Process model: constant angular velocity
        def process_model(x, u, dt):
            return np.array([x[0] + x[1] * dt, x[1]])

        def measurement_model(x):
            return np.array([x[0]])

        def Q_func(dt):
            return 0.01 * np.eye(2)

        def R_func():
            return np.array([[0.05]])

        dt = 0.1

        # Start just before pi
        x0 = np.array([np.pi - 0.3, 0.5])
        P0 = np.diag([0.1, 0.1])

        # UKF with angle wrapping
        innovation_func = create_bearing_only_innovation_func()
        ukf = UnscentedKalmanFilter(
            process_model, measurement_model,
            Q_func, R_func, x0.copy(), P0.copy(),
            innovation_func=innovation_func
        )

        # Cross pi
        true_angle = x0[0]
        np.random.seed(42)

        errors = []
        for _ in range(10):
            true_angle += 0.5 * dt
            true_angle = np.arctan2(np.sin(true_angle), np.cos(true_angle))

            z = true_angle + np.random.normal(0, 0.1)

            ukf.predict(dt=dt)
            ukf.update(np.array([z]))
            est, _ = ukf.get_state()

            error = abs(angle_diff(est[0], true_angle))
            errors.append(error)

        # With wrapping, errors should remain small
        max_error = max(errors)
        self.assertLess(max_error, 0.5,
                        f"UKF with wrapping should have small error, got {max_error}")


if __name__ == "__main__":
    unittest.main()











