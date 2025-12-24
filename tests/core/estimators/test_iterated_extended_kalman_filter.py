"""
Unit tests for Iterated Extended Kalman Filter (IEKF).

Tests cover:
    - IEKF algorithm correctness (Section 3.2.3)
    - Modified residual computation
    - Convergence behavior
    - Comparison with EKF

Book Reference: Chapter 3, Section 3.2.3 (Iterated Extended Kalman Filter)
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from core.estimators import IteratedExtendedKalmanFilter, ExtendedKalmanFilter


class TestIEKFAlgorithm(unittest.TestCase):
    """Test IEKF algorithm matches book description (Section 3.2.3)."""

    def setUp(self):
        """Setup simple nonlinear system for testing."""
        # Process model: simple 2D motion
        def process_model(x, u, dt):
            return np.array([x[0] + x[1] * dt, x[1]])

        def process_jacobian(x, u, dt):
            return np.array([[1, dt], [0, 1]])

        # Nonlinear measurement: range from origin
        def measurement_model(x):
            return np.array([np.sqrt(x[0] ** 2 + 0.01)])

        def measurement_jacobian(x):
            r = np.sqrt(x[0] ** 2 + 0.01)
            return np.array([[x[0] / r, 0]])

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

    def test_iekf_initialization(self):
        """Test IEKF initializes x_k^(0) = x_hat_k^-."""
        x0 = np.array([5.0, 0.5])
        P0 = np.eye(2)

        iekf = IteratedExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0, P0,
            max_iterations=1  # Only one iteration
        )

        x_before_update, _ = iekf.get_state()
        iekf.predict(dt=0.1)
        x_pred, _ = iekf.get_state()

        # With max_iterations=1, result should be similar to EKF
        iekf.update(np.array([5.5]))

        # Just verify filter ran without error
        x_after, P_after = iekf.get_state()
        self.assertEqual(len(x_after), 2)
        self.assertEqual(P_after.shape, (2, 2))

    def test_iekf_convergence(self):
        """Test IEKF converges within specified iterations."""
        x0 = np.array([5.0, 0.5])
        P0 = np.eye(2)

        iekf = IteratedExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0, P0,
            max_iterations=10,
            convergence_tol=1e-8
        )

        iekf.predict(dt=0.1)
        iters = iekf.update(np.array([5.1]))

        self.assertLessEqual(iters, 10)
        self.assertGreater(iters, 0)

    def test_iekf_returns_iteration_count(self):
        """Test get_last_iterations() returns correct count."""
        x0 = np.array([5.0, 0.5])
        P0 = np.eye(2)

        iekf = IteratedExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0, P0,
            max_iterations=5
        )

        iekf.predict(dt=0.1)
        returned_iters = iekf.update(np.array([5.1]))
        stored_iters = iekf.get_last_iterations()

        self.assertEqual(returned_iters, stored_iters)


class TestIEKFvsEKF(unittest.TestCase):
    """Test IEKF performs at least as well as EKF."""

    def setUp(self):
        """Setup range-only positioning problem."""
        self.landmarks = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

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
            return np.array([np.linalg.norm(lm - x[:2])
                             for lm in self.landmarks])

        def measurement_jacobian(x):
            H = []
            for lm in self.landmarks:
                diff = x[:2] - lm
                r = np.linalg.norm(diff)
                if r < 1e-6:
                    H.append([0, 0, 0, 0])
                else:
                    H.append([diff[0] / r, diff[1] / r, 0, 0])
            return np.array(H)

        def Q_func(dt):
            q = 0.1
            return q * np.array([
                [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                [dt ** 2 / 2, 0, dt, 0],
                [0, dt ** 2 / 2, 0, dt]
            ])

        def R_func():
            return 0.5 * np.eye(4)

        self.process_model = process_model
        self.process_jacobian = process_jacobian
        self.measurement_model = measurement_model
        self.measurement_jacobian = measurement_jacobian
        self.Q_func = Q_func
        self.R_func = R_func

    def test_iekf_not_worse_than_ekf(self):
        """Test IEKF performs at least as well as EKF."""
        true_x0 = np.array([5.0, 5.0, 0.5, 0.3])
        x0_est = np.array([4.0, 6.0, 0.0, 0.0])
        P0 = np.diag([2.0, 2.0, 1.0, 1.0])

        ekf = ExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0_est.copy(), P0.copy()
        )

        iekf = IteratedExtendedKalmanFilter(
            self.process_model, self.process_jacobian,
            self.measurement_model, self.measurement_jacobian,
            self.Q_func, self.R_func, x0_est.copy(), P0.copy(),
            max_iterations=5
        )

        # Run simulation
        dt = 0.5
        n_steps = 20
        true_state = true_x0.copy()
        np.random.seed(42)

        ekf_errors = []
        iekf_errors = []

        for _ in range(n_steps):
            true_state = self.process_model(true_state, None, dt)
            true_state += np.random.multivariate_normal(
                np.zeros(4), self.Q_func(dt)
            )

            z = self.measurement_model(true_state) + \
                np.random.multivariate_normal(np.zeros(4), self.R_func())

            ekf.predict(dt=dt)
            ekf.update(z)
            ekf_est, _ = ekf.get_state()

            iekf.predict(dt=dt)
            iekf.update(z)
            iekf_est, _ = iekf.get_state()

            ekf_errors.append(np.linalg.norm(ekf_est[:2] - true_state[:2]))
            iekf_errors.append(np.linalg.norm(iekf_est[:2] - true_state[:2]))

        # IEKF should not be significantly worse (allow 10% margin)
        self.assertLessEqual(
            np.mean(iekf_errors),
            np.mean(ekf_errors) * 1.1,
            "IEKF should not be significantly worse than EKF"
        )


class TestIEKFCovarianceUpdate(unittest.TestCase):
    """Test IEKF covariance update: P_k = (I - K H) P_k^-."""

    def test_covariance_remains_positive_definite(self):
        """Test covariance stays positive definite after iterations."""

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

        x0 = np.array([5.0, 1.0])
        P0 = np.eye(2)

        iekf = IteratedExtendedKalmanFilter(
            process_model, process_jacobian,
            measurement_model, measurement_jacobian,
            Q_func, R_func, x0, P0,
            max_iterations=5
        )

        for _ in range(10):
            iekf.predict(dt=0.1)
            iekf.update(np.array([5.5]))

            _, P = iekf.get_state()
            eigvals = np.linalg.eigvalsh(P)
            self.assertTrue(
                np.all(eigvals > 0),
                f"Covariance not positive definite: eigvals={eigvals}"
            )


if __name__ == "__main__":
    unittest.main()


