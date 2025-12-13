"""
Unscented Kalman Filter implementation for nonlinear systems.

This module implements the Unscented Kalman Filter (UKF) as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.24): Sigma point generation χ₀, χᵢ, χ_{i+n}
    - Eq. (3.25): Sigma point propagation through process model
    - Eq. (3.30): UKF update with cross-covariances and Kalman gain
"""

from typing import Callable, Optional, Tuple

import numpy as np

from core.estimators.base import StateEstimator


class UnscentedKalmanFilter(StateEstimator):
    """
    Unscented Kalman Filter for nonlinear systems using the Unscented Transform.

    The UKF addresses the limitations of the EKF by using a deterministic sampling
    approach called the Unscented Transform. Instead of linearizing the nonlinear
    functions, it propagates carefully chosen sample points (sigma points) through
    the true nonlinear functions.

    Implements Eqs. (3.24)-(3.30) from Chapter 3.

    Attributes:
        process_model: Function f(x, u, dt) -> x_next for state propagation
        measurement_model: Function h(x) -> z_pred for measurement prediction
        Q: Process noise covariance (n×n) or callable Q(dt) -> np.ndarray
        R: Measurement noise covariance (m×m) or callable R() -> np.ndarray
        alpha: Spread of sigma points (typically 1e-3 <= alpha <= 1)
        beta: Parameter for incorporating prior knowledge (beta=2 optimal for Gaussian)
        kappa: Secondary scaling parameter (typically 0 or 3-n)
        state: Current state estimate x̂_k (n,)
        covariance: Current state covariance P_k (n×n)
    """

    def __init__(
        self,
        process_model: Callable[[np.ndarray, Optional[np.ndarray], float], np.ndarray],
        measurement_model: Callable[[np.ndarray], np.ndarray],
        Q: Callable[[float], np.ndarray],
        R: Callable[[], np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: Optional[float] = None,
    ):
        """
        Initialize Unscented Kalman Filter.

        Args:
            process_model: Nonlinear state transition function f(x, u, dt) -> x_next.
                Implements f in Eq. (3.25).
            measurement_model: Nonlinear measurement function h(x) -> z_pred.
            Q: Process noise covariance function Q(dt) -> (n×n).
            R: Measurement noise covariance function R() -> (m×m).
            x0: Initial state estimate (n,).
            P0: Initial state covariance (n×n).
            alpha: Spread of sigma points, typically 1e-3 <= alpha <= 1.
                Controls the spread of sigma points around the mean.
            beta: Parameter for incorporating prior knowledge of distribution.
                For Gaussian distributions, beta = 2 is optimal.
            kappa: Secondary scaling parameter. If None, uses kappa = 3 - n.

        Raises:
            ValueError: If dimensions are inconsistent.
        """
        state_dim = len(x0)
        super().__init__(state_dim)

        self.process_model = process_model
        self.measurement_model = measurement_model
        self.Q = Q
        self.R = R

        self.state = np.asarray(x0, dtype=float).copy()
        self.covariance = np.asarray(P0, dtype=float).copy()

        if self.covariance.shape != (state_dim, state_dim):
            raise ValueError(
                f"P0 shape {self.covariance.shape} inconsistent with state_dim {state_dim}"
            )

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - state_dim

        # Compute weights for sigma points
        self._compute_weights()

    def _compute_weights(self) -> None:
        """
        Compute weights for mean and covariance of sigma points.

        These weights are used in the Unscented Transform to reconstruct
        the mean and covariance from the propagated sigma points.
        """
        n = self.state_dim
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        # Weights for mean computation
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1.0 / (2 * (n + lambda_))

        # Weights for covariance computation
        self.Wc = self.Wm.copy()
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        self.lambda_ = lambda_

    def _generate_sigma_points(
        self, x: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        """
        Generate sigma points using the Unscented Transform.

        Implements Eq. (3.24): Sigma point generation
            χ₀ = x̂_{k-1}
            χᵢ = x̂_{k-1} + δᵢ    for i = 1, ..., n
            χ_{i+n} = x̂_{k-1} - δᵢ for i = 1, ..., n

        Args:
            x: Mean state vector (n,).
            P: State covariance matrix (n×n).

        Returns:
            Sigma points matrix (2n+1, n) where each row is a sigma point.
        """
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))

        # Eq. (3.24): Center point χ₀ = x̂
        sigma_points[0] = x

        # Compute matrix square root: P = L L^T
        # Using Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eig(P)
            L = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) * np.sqrt(n + self.lambda_)

        # Eq. (3.24): χᵢ = x̂ + δᵢ and χ_{i+n} = x̂ - δᵢ
        for i in range(n):
            sigma_points[i + 1] = x + L[:, i]
            sigma_points[n + i + 1] = x - L[:, i]

        return sigma_points

    def _unscented_transform(
        self, sigma_points: np.ndarray, weights_m: np.ndarray, weights_c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance from sigma points using weighted average.

        Args:
            sigma_points: Matrix of sigma points (N, n).
            weights_m: Weights for mean computation (N,).
            weights_c: Weights for covariance computation (N,).

        Returns:
            Tuple of (mean, covariance).
        """
        # Weighted mean
        mean = np.sum(weights_m[:, np.newaxis] * sigma_points, axis=0)

        # Weighted covariance
        diff = sigma_points - mean
        covariance = (weights_c[:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0)

        return mean, covariance

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 1.0) -> None:
        """
        Perform prediction step using Unscented Transform.

        Implements the UKF prediction:
        - Eq. (3.24): Generate sigma points from current state
        - Eq. (3.25): Propagate sigma points through process model χᵢ⁻ = f(χᵢ, u_k)
        - Compute predicted mean and covariance from propagated sigma points

        Args:
            u: Optional control input vector.
            dt: Time step for integration.

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("State and covariance must be initialized")

        # Eq. (3.24): Generate sigma points from current state
        sigma_points = self._generate_sigma_points(self.state, self.covariance)

        # Eq. (3.25): Propagate sigma points through process model
        sigma_points_pred = np.array([
            self.process_model(sp, u, dt) for sp in sigma_points
        ])

        # Compute predicted state and covariance using Unscented Transform
        self.state, P_pred = self._unscented_transform(
            sigma_points_pred, self.Wm, self.Wc
        )

        # Add process noise
        Q = self.Q(dt)
        self.covariance = P_pred + Q

    def update(self, z: np.ndarray) -> None:
        """
        Perform measurement update using Unscented Transform.

        Implements Eq. (3.30): UKF update with cross-covariances
        - Generate sigma points from predicted state
        - Propagate through measurement model
        - Compute innovation and Kalman gain
        - Update state and covariance

        Args:
            z: Measurement vector (m,).

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before update()")

        z = np.asarray(z, dtype=float)

        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(self.state, self.covariance)

        # Propagate sigma points through measurement model
        sigma_points_meas = np.array([
            self.measurement_model(sp) for sp in sigma_points
        ])

        # Predicted measurement mean and covariance
        z_pred, Pzz = self._unscented_transform(
            sigma_points_meas, self.Wm, self.Wc
        )

        # Add measurement noise
        R = self.R()
        Pzz = Pzz + R

        # Cross-covariance between state and measurement
        diff_x = sigma_points - self.state
        diff_z = sigma_points_meas - z_pred
        Pxz = (self.Wc[:, np.newaxis, np.newaxis] * diff_x[:, :, np.newaxis] * diff_z[:, np.newaxis, :]).sum(axis=0)

        # Eq. (3.30): Kalman gain K_k
        K = Pxz @ np.linalg.inv(Pzz)

        # Innovation
        innovation = z - z_pred

        # State update
        self.state = self.state + K @ innovation

        # Covariance update
        self.covariance = self.covariance - K @ Pzz @ K.T


def test_ukf_range_only_tracking():
    """
    Unit test: 2D range-only tracking with UKF.

    Tests the UKF on a nonlinear range-only measurement problem.
    Expected: UKF should handle nonlinearity better than EKF.
    """
    dt = 0.1
    n_steps = 50

    # Process model: constant velocity in 2D
    def process_model(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    # Measurement model: range from origin
    def measurement_model(x):
        return np.array([np.sqrt(x[0]**2 + x[1]**2)])

    q = 0.1
    def Q_func(dt):
        return q * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

    def R_func():
        return np.array([[0.5]])

    x0 = np.array([10.0, 5.0, 1.0, 0.5])
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])

    # Create UKF
    ukf = UnscentedKalmanFilter(
        process_model, measurement_model,
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

        ukf.predict(dt=dt)
        ukf.update(np.array([z]))

    x_est, P_est = ukf.get_state()

    # Check that filter ran successfully
    assert np.all(np.linalg.eigvals(P_est) > 0), "Covariance not positive definite"

    print("UKF Range-Only Tracking Test:")
    print(f"  Final state: {x_est}")
    print(f"  Covariance trace: {np.trace(P_est):.4f}")
    print("  [PASS] Test passed")


def test_ukf_bearing_only_tracking():
    """
    Unit test: 2D bearing-only tracking with UKF.

    Tests UKF on highly nonlinear bearing-only measurements.
    """
    dt = 0.1
    n_steps = 50

    def process_model(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    def measurement_model(x):
        return np.array([np.arctan2(x[1], x[0])])

    q = 0.1
    def Q_func(dt):
        return q * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

    def R_func():
        return np.array([[0.05]])

    x0 = np.array([10.0, 5.0, 0.5, 0.3])
    P0 = np.diag([2.0, 2.0, 1.0, 1.0])

    ukf = UnscentedKalmanFilter(
        process_model, measurement_model,
        Q_func, R_func, x0, P0
    )

    true_state = x0.copy()
    np.random.seed(42)

    for _ in range(n_steps):
        true_state = process_model(true_state, None, dt)
        true_state += np.random.multivariate_normal(np.zeros(4), Q_func(dt))

        true_bearing = measurement_model(true_state)[0]
        z = true_bearing + np.random.normal(0, np.sqrt(R_func()[0, 0]))

        ukf.predict(dt=dt)
        ukf.update(np.array([z]))

    x_est, P_est = ukf.get_state()

    assert np.all(np.linalg.eigvals(P_est) > 0), "Covariance not positive definite"

    print("UKF Bearing-Only Tracking Test:")
    print(f"  Final state: {x_est}")
    print(f"  Covariance trace: {np.trace(P_est):.4f}")
    print("  [PASS] Test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("UNSCENTED KALMAN FILTER UNIT TESTS")
    print("=" * 70)
    print()

    test_ukf_range_only_tracking()
    print()
    test_ukf_bearing_only_tracking()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


