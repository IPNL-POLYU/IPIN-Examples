"""
Kalman Filter implementation for linear Gaussian systems.

This module implements the linear Kalman filter as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.8): Linear measurement model z_k = H_k x_k + w_{z,k}
    - Eq. (3.9): Likelihood mean/covariance
    - Eq. (3.11): State propagation x_{k|k-1} = F_k x_{k-1} + u_k + w_{x,k-1}
    - Eq. (3.12): Covariance propagation P_{k|k-1} = F_k P_{k-1} F_k^T + Q_k
    - Eq. (3.17): KF update x̂_{k,MAP} = x_{k|k-1} + K_k (z_k - H_k x_{k|k-1})
    - Eq. (3.18): Kalman gain K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}
    - Eq. (3.19): Covariance update P_k = P_{k|k-1} - K_k H_k P_{k|k-1}
    - Eq. (3.20): Summary of five KF equations
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np

from core.estimators.base import StateEstimator


class KalmanFilter(StateEstimator):
    """
    Linear Kalman Filter for optimal state estimation in linear Gaussian systems.

    The Kalman filter provides the optimal (minimum mean squared error) estimate
    for linear systems with Gaussian noise. It operates in two steps:
    1. Prediction: Propagate state and covariance forward in time
    2. Update: Correct prediction using new measurement

    Implements Eqs. (3.8)-(3.20) from Chapter 3.

    Attributes:
        F: State transition matrix (n×n) or callable F(dt) -> np.ndarray
        Q: Process noise covariance (n×n) or callable Q(dt) -> np.ndarray
        H: Measurement matrix (m×n) or callable H() -> np.ndarray
        R: Measurement noise covariance (m×m) or callable R() -> np.ndarray
        state: Current state estimate x̂_k (n,)
        covariance: Current state covariance P_k (n×n)
    """

    def __init__(
        self,
        F: Union[np.ndarray, Callable],
        Q: Union[np.ndarray, Callable],
        H: Union[np.ndarray, Callable],
        R: Union[np.ndarray, Callable],
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        """
        Initialize Kalman Filter.

        Args:
            F: State transition matrix (n×n) or callable F(dt) -> np.ndarray.
               Implements the F_k in Eq. (3.11).
            Q: Process noise covariance (n×n) or callable Q(dt) -> np.ndarray.
               Represents Σ_{w,u,k} in Eq. (3.12).
            H: Measurement matrix (m×n) or callable H() -> np.ndarray.
               Implements H_k in Eq. (3.8).
            R: Measurement noise covariance (m×m) or callable R() -> np.ndarray.
               Represents Σ_{w,z,k} in Eq. (3.9).
            x0: Initial state estimate (n,). If None, must be set before first predict().
            P0: Initial state covariance (n×n). If None, must be set before first predict().

        Raises:
            ValueError: If matrix dimensions are inconsistent.
        """
        # Determine state dimension
        if isinstance(F, np.ndarray):
            state_dim = F.shape[0]
            if F.shape != (state_dim, state_dim):
                raise ValueError(f"F must be square, got shape {F.shape}")
        elif x0 is not None:
            state_dim = len(x0)
        else:
            raise ValueError("Must provide either F as ndarray or x0 to determine state_dim")

        super().__init__(state_dim)

        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        # Initialize state and covariance
        if x0 is not None:
            self.state = np.asarray(x0, dtype=float).copy()
        if P0 is not None:
            self.covariance = np.asarray(P0, dtype=float).copy()
            if self.covariance.shape != (state_dim, state_dim):
                raise ValueError(
                    f"P0 shape {self.covariance.shape} inconsistent with state_dim {state_dim}"
                )

    def _get_matrix(
        self, mat: Union[np.ndarray, Callable], *args
    ) -> np.ndarray:
        """
        Helper to get matrix value (handles both constant and callable).

        Args:
            mat: Matrix or callable that returns matrix.
            *args: Arguments to pass to callable.

        Returns:
            Matrix as numpy array.
        """
        if callable(mat):
            return mat(*args)
        return mat

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 1.0) -> None:
        """
        Perform prediction step (time update).

        Implements the prediction step of the linear Kalman filter:
        - Eq. (3.11): x̂_{k|k-1} = F_k x̂_{k-1} + u_k
        - Eq. (3.12): P_{k|k-1} = F_k P_{k-1} F_k^T + Q_k
        - First two equations of Eq. (3.20)

        Args:
            u: Optional control input vector (n,). If None, assumes zero control.
            dt: Time step (used if F or Q are callable).

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError(
                "State and covariance must be initialized before predict(). "
                "Provide x0 and P0 in constructor or set manually."
            )

        # Get matrices (handle both constant and time-varying)
        F = self._get_matrix(self.F, dt)
        Q = self._get_matrix(self.Q, dt)

        # Eq. (3.11): State propagation
        self.state = F @ self.state
        if u is not None:
            self.state += u

        # Eq. (3.12): Covariance propagation
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        """
        Perform measurement update (correction step).

        Implements the update step of the linear Kalman filter:
        - Eq. (3.17): x̂_{k,MAP} = x̂_{k|k-1} + K_k (z_k - H_k x̂_{k|k-1})
        - Eq. (3.18): K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}
        - Eq. (3.19): P_k = P_{k|k-1} - K_k H_k P_{k|k-1}
        - Last three equations of Eq. (3.20)

        Args:
            z: Measurement vector (m,).

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before update()")

        z = np.asarray(z, dtype=float)

        # Get measurement matrices
        H = self._get_matrix(self.H)
        R = self._get_matrix(self.R)

        # Innovation (measurement residual): ν = z - H x̂_{k|k-1}
        innovation = z - H @ self.state

        # Innovation covariance: S = H P_{k|k-1} H^T + R
        S = H @ self.covariance @ H.T + R

        # Eq. (3.18): Kalman gain K_k = P_{k|k-1} H^T S^{-1}
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Eq. (3.17): State update x̂_k = x̂_{k|k-1} + K ν
        self.state = self.state + K @ innovation

        # Eq. (3.19): Covariance update P_k = P_{k|k-1} - K H P_{k|k-1}
        # Using Joseph form for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

    def get_innovation(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute innovation (measurement residual) and its covariance.

        Useful for consistency checking and outlier detection.

        Args:
            z: Measurement vector (m,).

        Returns:
            Tuple of (innovation, innovation_covariance).
                - innovation: ν = z - H x̂_{k|k-1} (m,)
                - innovation_covariance: S = H P_{k|k-1} H^T + R (m×m)
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before get_innovation()")

        z = np.asarray(z, dtype=float)
        H = self._get_matrix(self.H)
        R = self._get_matrix(self.R)

        innovation = z - H @ self.state
        innovation_cov = H @ self.covariance @ H.T + R

        return innovation, innovation_cov


def test_kalman_filter_1d_constant_velocity():
    """
    Unit test: 1D constant velocity tracking.

    Tests the Kalman filter on a simple 1D constant velocity model.
    State: [position, velocity]
    Measurement: position only

    Expected: Filter should converge to true state with small error.
    """
    dt = 0.1  # Time step
    n_steps = 50

    # State transition matrix (constant velocity model)
    F = np.array([[1.0, dt], [0.0, 1.0]])

    # Process noise (small)
    q = 0.01
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    # Measurement matrix (observe position only)
    H = np.array([[1.0, 0.0]])

    # Measurement noise
    R = np.array([[0.1]])

    # Initial state and covariance
    x0 = np.array([0.0, 1.0])  # Start at position 0, velocity 1
    P0 = np.eye(2) * 1.0

    # Create filter
    kf = KalmanFilter(F, Q, H, R, x0, P0)

    # Generate true trajectory
    true_state = x0.copy()
    true_states = [true_state.copy()]
    measurements = []

    np.random.seed(42)
    for _ in range(n_steps):
        # Propagate true state
        true_state = F @ true_state + np.random.multivariate_normal([0, 0], Q)
        true_states.append(true_state.copy())

        # Generate measurement
        z = H @ true_state + np.random.normal(0, np.sqrt(R[0, 0]))
        measurements.append(z[0])

    # Run filter
    estimates = [x0.copy()]
    for z in measurements:
        kf.predict(dt=dt)
        kf.update(np.array([z]))
        x_est, _ = kf.get_state()
        estimates.append(x_est.copy())

    # Check convergence: final position error should be small
    final_true = true_states[-1]
    final_est = estimates[-1]
    position_error = abs(final_est[0] - final_true[0])
    velocity_error = abs(final_est[1] - final_true[1])

    print(f"1D Constant Velocity Test:")
    print(f"  Final position error: {position_error:.4f} m")
    print(f"  Final velocity error: {velocity_error:.4f} m/s")
    print(f"  Expected: < 0.5 m and < 0.2 m/s")

    assert position_error < 0.5, f"Position error {position_error} too large"
    assert velocity_error < 0.2, f"Velocity error {velocity_error} too large"

    print("  [PASS] Test passed")


def test_kalman_filter_innovation():
    """
    Unit test: Innovation computation.

    Tests that innovation and innovation covariance are computed correctly.
    """
    # Simple 1D system
    F = np.array([[1.0]])
    Q = np.array([[0.1]])
    H = np.array([[1.0]])
    R = np.array([[0.5]])

    x0 = np.array([0.0])
    P0 = np.array([[1.0]])

    kf = KalmanFilter(F, Q, H, R, x0, P0)

    # Predict
    kf.predict()

    # Measurement
    z = np.array([1.5])

    # Get innovation before update
    innov, innov_cov = kf.get_innovation(z)

    # Check innovation
    expected_innov = z - H @ kf.state
    assert np.allclose(innov, expected_innov), "Innovation mismatch"

    # Check innovation covariance
    expected_innov_cov = H @ kf.covariance @ H.T + R
    assert np.allclose(innov_cov, expected_innov_cov), "Innovation covariance mismatch"

    print("Innovation Test:")
    print(f"  Innovation: {innov[0]:.4f}")
    print(f"  Innovation std: {np.sqrt(innov_cov[0, 0]):.4f}")
    print("  [PASS] Test passed")


def test_kalman_filter_callable_matrices():
    """
    Unit test: Time-varying matrices (callable).

    Tests that the filter works with time-varying F, Q, H, R.
    """

    def F_func(dt):
        return np.array([[1.0, dt], [0.0, 1.0]])

    def Q_func(dt):
        q = 0.01
        return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    def H_func():
        return np.array([[1.0, 0.0]])

    def R_func():
        return np.array([[0.1]])

    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)

    kf = KalmanFilter(F_func, Q_func, H_func, R_func, x0, P0)

    # Run a few steps
    for i in range(5):
        kf.predict(dt=0.1)
        kf.update(np.array([i * 0.1 + np.random.normal(0, 0.1)]))

    x_est, P_est = kf.get_state()

    assert x_est.shape == (2,), "State shape mismatch"
    assert P_est.shape == (2, 2), "Covariance shape mismatch"
    assert np.all(np.linalg.eigvals(P_est) > 0), "Covariance not positive definite"

    print("Callable Matrices Test:")
    print(f"  Final state: {x_est}")
    print(f"  Covariance trace: {np.trace(P_est):.4f}")
    print("  [PASS] Test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("KALMAN FILTER UNIT TESTS")
    print("=" * 70)
    print()

    test_kalman_filter_1d_constant_velocity()
    print()
    test_kalman_filter_innovation()
    print()
    test_kalman_filter_callable_matrices()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

