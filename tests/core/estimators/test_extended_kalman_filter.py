"""
Unit tests for Extended Kalman Filter.

Tests cover:
    - Jacobian evaluation point (Eq. 3.22)
    - Range-only tracking
    - Bearing-only tracking
    - State/covariance propagation

Book Reference: Chapter 3, Section 3.2.2 (Extended Kalman Filter)
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from core.estimators.extended_kalman_filter import ExtendedKalmanFilter


class TestEKFJacobianEvaluationPoint(unittest.TestCase):
    """Test that process Jacobian F_{k-1} is evaluated at pre-prediction state.

    Book Reference (Eq. 3.22):
        "F_{k-1} = df/dx|_{x_hat_{k-1}} is the Jacobian of f with respect to
        the state, evaluated at the current estimate."

    This is a regression test to ensure the Jacobian is evaluated at x_{k-1}
    (pre-prediction), NOT at x_k^- (post-prediction).
    """

    def setUp(self):
        """Setup nonlinear process model where Jacobian depends on state."""
        self.dt = 0.1

        # Nonlinear process: x_k = x_{k-1} + v*dt + 0.1*x^2*dt
        def process_model(x, u, dt):
            x_new = np.zeros(2)
            x_new[0] = x[0] + x[1] * dt + 0.1 * x[0] ** 2 * dt
            x_new[1] = x[1]
            return x_new

        def process_jacobian(x, u, dt):
            """Jacobian depends on x[0], so evaluation point matters!"""
            return np.array([
                [1.0 + 0.2 * x[0] * dt, dt],
                [0.0, 1.0]
            ])

        def measurement_model(x):
            return np.array([x[0]])

        def measurement_jacobian(x):
            return np.array([[1.0, 0.0]])

        def Q_func(dt):
            return 0.01 * np.eye(2)

        def R_func():
            return np.array([[0.1]])

        self.process_model = process_model
        self.process_jacobian = process_jacobian
        self.measurement_model = measurement_model
        self.measurement_jacobian = measurement_jacobian
        self.Q_func = Q_func
        self.R_func = R_func

    def test_jacobian_evaluated_at_pre_prediction_state(self):
        """Test F_{k-1} is evaluated at x_{k-1}, not x_k^- (Eq. 3.22)."""
        # Large initial x[0] so Jacobian varies significantly with state
        x0 = np.array([10.0, 2.0])
        P0 = np.diag([1.0, 0.5])

        ekf = ExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0, P0
        )

        # Record pre-prediction state
        x_pre = ekf.state.copy()

        # Perform prediction
        ekf.predict(dt=self.dt)

        # Get post-prediction state
        x_post = ekf.state.copy()

        # Verify states differ (nonlinear model)
        self.assertFalse(np.allclose(x_pre, x_post))

        # Compute CORRECT covariance (Jacobian at pre-state per Eq. 3.22)
        F_correct = self.process_jacobian(x_pre, None, self.dt)
        P_correct = F_correct @ P0 @ F_correct.T + self.Q_func(self.dt)

        # Compute WRONG covariance (if Jacobian was at post-state)
        F_wrong = self.process_jacobian(x_post, None, self.dt)
        P_wrong = F_wrong @ P0 @ F_wrong.T + self.Q_func(self.dt)

        # Jacobians at different points should differ
        self.assertFalse(np.allclose(F_correct, F_wrong),
                         "Jacobians at different states should differ")

        # EKF must use CORRECT (pre-state) Jacobian
        assert_allclose(ekf.covariance, P_correct, atol=1e-10,
                        err_msg="EKF used wrong Jacobian evaluation point!")

    def test_jacobian_difference_is_significant(self):
        """Test that the Jacobian difference is large enough to matter."""
        x_pre = np.array([10.0, 2.0])
        x_post = self.process_model(x_pre, None, self.dt)

        F_pre = self.process_jacobian(x_pre, None, self.dt)
        F_post = self.process_jacobian(x_post, None, self.dt)

        # The F[0,0] element should differ by at least 1%
        diff = abs(F_pre[0, 0] - F_post[0, 0])
        relative_diff = diff / F_pre[0, 0]

        self.assertGreater(relative_diff, 0.01,
                           f"Jacobian difference too small: {relative_diff:.4f}")


class TestEKFRangeOnlyTracking(unittest.TestCase):
    """Test EKF with range-only measurements (nonlinear observation)."""

    def test_range_only_tracking_converges(self):
        """Test 2D range-only tracking produces bounded errors."""
        dt = 0.1
        n_steps = 100

        def process_model(x, u, dt):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return F @ x

        def process_jacobian(x, u, dt):
            return np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        def measurement_model(x):
            return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])

        def measurement_jacobian(x):
            r = np.sqrt(x[0] ** 2 + x[1] ** 2)
            if r < 1e-6:
                return np.array([[0, 0, 0, 0]])
            return np.array([[x[0] / r, x[1] / r, 0, 0]])

        q = 0.1

        def Q_func(dt):
            return q * np.array([
                [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                [dt ** 2 / 2, 0, dt, 0],
                [0, dt ** 2 / 2, 0, dt]
            ])

        def R_func():
            return np.array([[0.5]])

        x0 = np.array([10.0, 5.0, 1.0, 0.5])
        P0 = np.diag([1.0, 1.0, 0.5, 0.5])

        ekf = ExtendedKalmanFilter(
            process_model, process_jacobian,
            measurement_model, measurement_jacobian,
            Q_func, R_func, x0, P0
        )

        # Generate true trajectory
        true_state = x0.copy()
        np.random.seed(42)

        for _ in range(n_steps):
            true_state = process_model(true_state, None, dt)
            true_state += np.random.multivariate_normal(np.zeros(4), Q_func(dt))

            true_range = measurement_model(true_state)[0]
            z = true_range + np.random.normal(0, np.sqrt(R_func()[0, 0]))

            ekf.predict(dt=dt)
            ekf.update(np.array([z]))

        x_est, P_est = ekf.get_state()

        # Check errors are bounded
        position_error = np.linalg.norm(x_est[:2] - true_state[:2])
        velocity_error = np.linalg.norm(x_est[2:] - true_state[2:])

        self.assertLess(position_error, 3.0,
                        f"Position error too large: {position_error}")
        self.assertLess(velocity_error, 2.0,
                        f"Velocity error too large: {velocity_error}")


class TestEKFCovarianceProperties(unittest.TestCase):
    """Test EKF covariance remains valid."""

    def test_covariance_remains_positive_definite(self):
        """Test covariance stays positive definite after prediction."""
        def process_model(x, u, dt):
            return x + np.array([x[1] * dt, 0])

        def process_jacobian(x, u, dt):
            return np.array([[1, dt], [0, 1]])

        def measurement_model(x):
            return np.array([x[0]])

        def measurement_jacobian(x):
            return np.array([[1, 0]])

        def Q_func(dt):
            return 0.01 * np.eye(2)

        def R_func():
            return np.array([[0.1]])

        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)

        ekf = ExtendedKalmanFilter(
            process_model, process_jacobian,
            measurement_model, measurement_jacobian,
            Q_func, R_func, x0, P0
        )

        for _ in range(10):
            ekf.predict(dt=0.1)

            # Check positive definite
            eigvals = np.linalg.eigvalsh(ekf.covariance)
            self.assertTrue(np.all(eigvals > 0),
                            "Covariance not positive definite")


if __name__ == "__main__":
    unittest.main()



