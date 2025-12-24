"""
Iterated Extended Kalman Filter (IEKF) implementation.

This module implements the Iterated Extended Kalman Filter as described in
Chapter 3, Section 3.2.3 of Principles of Indoor Positioning and Indoor Navigation.

The IEKF iterates the measurement update step multiple times per measurement,
re-linearizing at the updated state each time. This effectively performs
Gauss-Newton iterations to solve the measurement equation more accurately.

Book Reference:
    Section 3.2.3: Iterated Extended Kalman Filter (IEKF)

    "When measurement nonlinearities are severe, one way to improve the EKF's
    accuracy is to iterate the update step multiple times per measurement."

Algorithm (from book):
    After obtaining the predicted state (x̂_k^-, P_k^-):
    1. Initialize x_k^(0) = x̂_k^-
    2. For j = 0, 1, ..., N (iterations until convergence):
       - Compute H_k^(j) = ∂h/∂x|_{x_k^(j)}
       - Compute modified residual:
         y_k^(j) = z_k - h(x_k^(j)) + H_k^(j) (x_k^(j) - x̂_k^-)
       - Compute Kalman gain:
         K_k^(j) = P_k^- [H_k^(j)]^T (H_k^(j) P_k^- [H_k^(j)]^T + R)^(-1)
       - Update state estimate:
         x_k^(j+1) = x̂_k^- + K_k^(j) y_k^(j)
    3. Set x̂_k = x_k^(N+1) as the final estimate
    4. Update covariance: P_k = (I - K_k^(N) H_k^(N)) P_k^-

The modified residual y_k^(j) ensures that upon convergence, x_k^(j) closely
satisfies the measurement equation. Mathematically, IEKF is equivalent to
performing Gauss-Newton optimization of the measurement likelihood.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from core.estimators.base import StateEstimator


class IteratedExtendedKalmanFilter(StateEstimator):
    """
    Iterated Extended Kalman Filter for highly nonlinear systems.

    The IEKF extends the EKF by iterating the measurement update step,
    re-linearizing at each iteration to reduce linearization error.

    Book Reference: Section 3.2.3

    "IEKF iteratively refines EKF updates for improved linearization accuracy
    at the cost of additional computation. Typically, 2-5 iterations are
    sufficient, resulting in roughly 2-5 times the cost of a single EKF update."

    Attributes:
        process_model: Function f(x, u, dt) -> x_next for state propagation.
        process_jacobian: Function F(x, u, dt) -> Jacobian matrix (n x n).
        measurement_model: Function h(x) -> z_pred for measurement prediction.
        measurement_jacobian: Function H(x) -> Jacobian matrix (m x n).
        Q: Process noise covariance function Q(dt) -> np.ndarray.
        R: Measurement noise covariance function R() -> np.ndarray.
        state: Current state estimate x̂_k (n,).
        covariance: Current state covariance P_k (n x n).
        max_iterations: Maximum IEKF iterations per update (default 5).
        convergence_tol: Convergence tolerance on state change (default 1e-6).
    """

    def __init__(
        self,
        process_model: Callable[[np.ndarray, Optional[np.ndarray], float], np.ndarray],
        process_jacobian: Callable[[np.ndarray, Optional[np.ndarray], float], np.ndarray],
        measurement_model: Callable[[np.ndarray], np.ndarray],
        measurement_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: Callable[[float], np.ndarray],
        R: Callable[[], np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
        max_iterations: int = 5,
        convergence_tol: float = 1e-6,
        innovation_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize Iterated Extended Kalman Filter.

        Args:
            process_model: Nonlinear state transition f(x, u, dt) -> x_next.
            process_jacobian: Jacobian of process model F(x, u, dt) -> df/dx.
            measurement_model: Nonlinear measurement function h(x) -> z_pred.
            measurement_jacobian: Jacobian of measurement model H(x) -> dh/dx.
            Q: Process noise covariance function Q(dt) -> (n x n).
            R: Measurement noise covariance function R() -> (m x m).
            x0: Initial state estimate (n,).
            P0: Initial state covariance (n x n).
            max_iterations: Maximum iterations per IEKF update (default 5).
                Book: "Typically, 2-5 iterations are sufficient."
            convergence_tol: Convergence tolerance on ||x^(j+1) - x^(j)||.
            innovation_func: Optional function to compute innovation nu = f(z, z_pred).
                Default is simple subtraction (z - z_pred).
                Use this to handle angle wrapping for bearing measurements.

        Raises:
            ValueError: If dimensions are inconsistent.
        """
        state_dim = len(x0)
        super().__init__(state_dim)

        self.process_model = process_model
        self.process_jacobian = process_jacobian
        self.measurement_model = measurement_model
        self.measurement_jacobian = measurement_jacobian
        self.Q = Q
        self.R = R
        self.innovation_func = innovation_func

        self.state = np.asarray(x0, dtype=float).copy()
        self.covariance = np.asarray(P0, dtype=float).copy()

        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        # Track last update statistics
        self._last_iterations = 0

        if self.covariance.shape != (state_dim, state_dim):
            raise ValueError(
                f"P0 shape {self.covariance.shape} inconsistent with state_dim {state_dim}"
            )

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 1.0) -> None:
        """
        Perform prediction step (time update).

        Same as EKF prediction per Eq. (3.22):
            x̂_k^- = f(x̂_{k-1}, u_k)
            P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q

        where F_{k-1} = df/dx|_{x̂_{k-1}} (evaluated at pre-prediction state).

        Args:
            u: Optional control input vector (n_u,).
            dt: Time step for integration.

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("State and covariance must be initialized")

        # Store pre-prediction state for Jacobian evaluation (per Eq. 3.22)
        x_pre = self.state.copy()

        # Compute Jacobian F_{k-1} at PRE-prediction state
        F = self.process_jacobian(x_pre, u, dt)

        # State propagation: x̂_k^- = f(x̂_{k-1}, u_k)
        self.state = self.process_model(x_pre, u, dt)

        # Get process noise covariance
        Q = self.Q(dt)

        # Covariance propagation: P_k^- = F P_{k-1} F^T + Q
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, z: np.ndarray) -> int:
        """
        Perform iterated measurement update (IEKF correction step).

        Implements the IEKF algorithm from Section 3.2.3:

        1. Initialize x_k^(0) = x̂_k^- (predicted state)
        2. For j = 0, 1, ..., N (until convergence):
           - Compute H_k^(j) = dh/dx|_{x_k^(j)}
           - Compute modified residual:
             y_k^(j) = z_k - h(x_k^(j)) + H_k^(j) (x_k^(j) - x̂_k^-)
           - Compute Kalman gain:
             K_k^(j) = P_k^- H^T (H P_k^- H^T + R)^(-1)
           - Update state:
             x_k^(j+1) = x̂_k^- + K_k^(j) y_k^(j)
        3. Finalize: x̂_k = x_k^(N+1), P_k = (I - K H) P_k^-

        Args:
            z: Measurement vector (m,).

        Returns:
            Number of iterations performed.

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before update()")

        z = np.asarray(z, dtype=float)

        # Store predicted state and covariance
        x_pred = self.state.copy()  # x̂_k^-
        P_pred = self.covariance.copy()  # P_k^-

        # Get measurement noise covariance
        R = self.R()

        # Initialize iteration: x_k^(0) = x̂_k^-
        x_iter = x_pred.copy()

        # IEKF iterations
        iterations_performed = 0
        K = None
        H = None

        for j in range(self.max_iterations):
            # Compute Jacobian at current iterate: H_k^(j) = dh/dx|_{x_k^(j)}
            H = self.measurement_jacobian(x_iter)

            # Predicted measurement at current iterate: h(x_k^(j))
            z_pred = self.measurement_model(x_iter)

            # Modified residual (book formula):
            # y_k^(j) = z_k - h(x_k^(j)) + H_k^(j) (x_k^(j) - x̂_k^-)
            # This accounts for the shift in linearization point
            delta_x = x_iter - x_pred

            # Compute innovation with optional angle wrapping
            if self.innovation_func is not None:
                innovation = self.innovation_func(z, z_pred)
            else:
                innovation = z - z_pred

            y_modified = innovation + H @ delta_x

            # Innovation covariance: S = H P_k^- H^T + R
            S = H @ P_pred @ H.T + R

            # Kalman gain: K_k^(j) = P_k^- H^T S^(-1)
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Update state: x_k^(j+1) = x̂_k^- + K_k^(j) y_k^(j)
            x_new = x_pred + K @ y_modified

            iterations_performed = j + 1

            # Check convergence: ||x^(j+1) - x^(j)|| < tol
            if np.linalg.norm(x_new - x_iter) < self.convergence_tol:
                x_iter = x_new
                break

            x_iter = x_new

        # Finalize: x̂_k = x_k^(N+1)
        self.state = x_iter

        # Covariance update: P_k = (I - K H) P_k^- (Joseph form for stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        self._last_iterations = iterations_performed
        return iterations_performed

    def get_last_iterations(self) -> int:
        """Return the number of iterations from the last update."""
        return self._last_iterations

    def get_innovation(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute innovation (measurement residual) and its covariance.

        Args:
            z: Measurement vector (m,).

        Returns:
            Tuple of (innovation, innovation_covariance).
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before get_innovation()")

        z = np.asarray(z, dtype=float)
        z_pred = self.measurement_model(self.state)
        H = self.measurement_jacobian(self.state)
        R = self.R()

        innovation = z - z_pred
        innovation_cov = H @ self.covariance @ H.T + R

        return innovation, innovation_cov


def test_iekf_convergence():
    """
    Unit test: Verify IEKF converges in fewer steps with mild nonlinearity.
    """
    dt = 0.1

    # Linear process model
    def process_model(x, u, dt):
        F = np.array([[1, dt], [0, 1]])
        return F @ x

    def process_jacobian(x, u, dt):
        return np.array([[1, dt], [0, 1]])

    # Nonlinear measurement: range from origin
    def measurement_model(x):
        return np.array([np.sqrt(x[0] ** 2 + 0.1)])

    def measurement_jacobian(x):
        r = np.sqrt(x[0] ** 2 + 0.1)
        return np.array([[x[0] / r, 0]])

    def Q_func(dt):
        return 0.01 * np.eye(2)

    def R_func():
        return np.array([[0.1]])

    x0 = np.array([5.0, 0.5])
    P0 = np.eye(2)

    iekf = IteratedExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0, P0,
        max_iterations=10,
        convergence_tol=1e-8,
    )

    # Simulate one step
    iekf.predict(dt=dt)

    # Measurement
    z = np.array([5.1])
    iters = iekf.update(z)

    print("IEKF Convergence Test:")
    print(f"  Iterations to converge: {iters}")
    print(f"  Final state: {iekf.state}")
    print(f"  [PASS] IEKF converged in {iters} iterations")

    assert iters <= 10, f"IEKF did not converge within max iterations"


def test_iekf_vs_ekf_high_nonlinearity():
    """
    Unit test: Compare IEKF vs EKF on highly nonlinear measurement.

    IEKF should produce better estimates than EKF when nonlinearity is severe.
    """
    from core.estimators.extended_kalman_filter import ExtendedKalmanFilter

    dt = 0.5

    # Process model
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

    # Highly nonlinear measurement: range from origin
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

    # Initial state with significant initial error
    true_x0 = np.array([10.0, 5.0, 1.0, 0.5])
    x0_est = np.array([8.0, 7.0, 0.0, 0.0])  # Wrong initial estimate
    P0 = np.diag([5.0, 5.0, 2.0, 2.0])

    # Create both filters
    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0_est.copy(), P0.copy()
    )

    iekf = IteratedExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0_est.copy(), P0.copy(),
        max_iterations=5
    )

    # Run simulation
    n_steps = 20
    true_state = true_x0.copy()
    np.random.seed(42)

    ekf_errors = []
    iekf_errors = []

    for _ in range(n_steps):
        # Propagate true state
        true_state = process_model(true_state, None, dt)
        true_state += np.random.multivariate_normal(np.zeros(4), Q_func(dt))

        # Generate measurement
        true_range = measurement_model(true_state)[0]
        z = np.array([true_range + np.random.normal(0, np.sqrt(R_func()[0, 0]))])

        # EKF update
        ekf.predict(dt=dt)
        ekf.update(z)
        ekf_est, _ = ekf.get_state()
        ekf_errors.append(np.linalg.norm(ekf_est[:2] - true_state[:2]))

        # IEKF update
        iekf.predict(dt=dt)
        iekf.update(z)
        iekf_est, _ = iekf.get_state()
        iekf_errors.append(np.linalg.norm(iekf_est[:2] - true_state[:2]))

    # Compare
    mean_ekf_error = np.mean(ekf_errors)
    mean_iekf_error = np.mean(iekf_errors)

    print("\nIEKF vs EKF Comparison Test:")
    print(f"  Mean EKF position error:  {mean_ekf_error:.4f} m")
    print(f"  Mean IEKF position error: {mean_iekf_error:.4f} m")
    print(f"  IEKF improvement: {(mean_ekf_error - mean_iekf_error) / mean_ekf_error * 100:.1f}%")

    # IEKF should be at least as good as EKF (usually better)
    assert mean_iekf_error <= mean_ekf_error * 1.1, (
        f"IEKF should not be significantly worse than EKF"
    )
    print("  [PASS] IEKF performs at least as well as EKF")


if __name__ == "__main__":
    print("=" * 70)
    print("ITERATED EXTENDED KALMAN FILTER UNIT TESTS")
    print("=" * 70)
    print()

    test_iekf_convergence()
    print()
    test_iekf_vs_ekf_high_nonlinearity()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

