"""
Extended Kalman Filter implementation for nonlinear systems.

This module implements the Extended Kalman Filter (EKF) as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.21): Nonlinear state and measurement models
      x_k = f(x_{k-1}, u_k) + w_k, z_k = h(x_k) + v_k
    - Eq. (3.22): EKF prediction
      x̂_k^- = f(x̂_{k-1}, u_k)
      P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q
    - EKF update (following Eq. (3.21))
"""

from typing import Callable, Optional, Tuple

import numpy as np

from core.estimators.base import StateEstimator


class ExtendedKalmanFilter(StateEstimator):
    """
    Extended Kalman Filter for nonlinear systems.

    The EKF extends the Kalman filter to nonlinear systems by linearizing
    the process and measurement models around the current state estimate.

    Implements Eqs. (3.21)-(3.22) from Chapter 3.

    Attributes:
        process_model: Function f(x, u, dt) -> x_next for state propagation
        process_jacobian: Function F(x, u, dt) -> Jacobian matrix (n×n)
        measurement_model: Function h(x) -> z_pred for measurement prediction
        measurement_jacobian: Function H(x) -> Jacobian matrix (m×n)
        Q: Process noise covariance (n×n) or callable Q(dt) -> np.ndarray
        R: Measurement noise covariance (m×m) or callable R() -> np.ndarray
        state: Current state estimate x̂_k (n,)
        covariance: Current state covariance P_k (n×n)
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
        innovation_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            process_model: Nonlinear state transition function f(x, u, dt) -> x_next.
                Implements f in Eq. (3.21).
            process_jacobian: Jacobian of process model F(x, u, dt) -> ∂f/∂x (n×n).
                Implements F_k in Eq. (3.22).
            measurement_model: Nonlinear measurement function h(x) -> z_pred.
                Implements h in Eq. (3.21).
            measurement_jacobian: Jacobian of measurement model H(x) -> ∂h/∂x (m×n).
                Implements H_k for EKF update.
            Q: Process noise covariance function Q(dt) -> (n×n).
                Represents process noise in Eq. (3.21).
            R: Measurement noise covariance function R() -> (m×m).
                Represents measurement noise in Eq. (3.21).
            x0: Initial state estimate (n,).
            P0: Initial state covariance (n×n).
            innovation_func: Optional function to compute innovation nu = f(z, z_pred).
                Default is simple subtraction (z - z_pred).
                Use this to handle angle wrapping for bearing measurements:
                    innovation_func=lambda z, z_pred: angle_aware_diff(z, z_pred)

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

        if self.covariance.shape != (state_dim, state_dim):
            raise ValueError(
                f"P0 shape {self.covariance.shape} inconsistent with state_dim {state_dim}"
            )

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 1.0) -> None:
        """
        Perform prediction step (time update) for nonlinear system.

        Implements EKF prediction per Eq. (3.22) from Chapter 3:

            x̂_k^- = f(x̂_{k-1}, u_k)
            P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q

        where F_{k-1} = ∂f/∂x|_{x̂_{k-1}} is the Jacobian evaluated at the
        **pre-prediction** state estimate x̂_{k-1} (NOT at x̂_k^-).

        Book Reference (Eq. 3.22):
            "F_{k-1} = ∂f/∂x|_{x̂_{k-1}} is the Jacobian of f with respect to
            the state, evaluated at the current estimate."

        Args:
            u: Optional control input vector (n_u,). If None, assumes zero control.
            dt: Time step for integration.

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("State and covariance must be initialized")

        # Store pre-prediction state x̂_{k-1} for Jacobian evaluation
        x_pre = self.state.copy()

        # Eq. (3.22): Compute Jacobian F_{k-1} = ∂f/∂x at PRE-prediction state
        # This is CRITICAL: F must be evaluated at x̂_{k-1}, not at x̂_k^-
        F = self.process_jacobian(x_pre, u, dt)

        # Eq. (3.21)/(3.22): Nonlinear state propagation x̂_k^- = f(x̂_{k-1}, u_k)
        self.state = self.process_model(x_pre, u, dt)

        # Get process noise covariance
        Q = self.Q(dt)

        # Eq. (3.22): Covariance propagation P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        """
        Perform measurement update (correction step) for nonlinear system.

        Implements EKF update (following Eq. (3.21)):
        - Measurement prediction: ẑ = h(x̂_k^-)
        - Jacobian computation: H_k = ∂h/∂x|_{x̂_k^-}
        - Innovation: ν = z - ẑ
        - Kalman gain: K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}
        - State update: x̂_k = x̂_k^- + K_k ν
        - Covariance update: P_k = (I - K_k H_k) P_k^-

        Args:
            z: Measurement vector (m,).

        Raises:
            RuntimeError: If state or covariance not initialized.
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Must call predict() before update()")

        z = np.asarray(z, dtype=float)

        # Predicted measurement: ẑ = h(x̂_k^-)
        z_pred = self.measurement_model(self.state)

        # Compute measurement Jacobian H_k = ∂h/∂x at predicted state
        H = self.measurement_jacobian(self.state)

        # Get measurement noise covariance
        R = self.R()

        # Innovation: ν = z - ẑ (with optional angle wrapping)
        if self.innovation_func is not None:
            innovation = self.innovation_func(z, z_pred)
        else:
            innovation = z - z_pred

        # Innovation covariance: S = H P_k^- H^T + R
        S = H @ self.covariance @ H.T + R

        # Kalman gain: K_k = P_k^- H^T S^{-1}
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # State update: x̂_k = x̂_k^- + K ν
        self.state = self.state + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

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


def test_ekf_range_only_tracking():
    """
    Unit test: 2D range-only tracking with EKF.

    State: [x, y, vx, vy] (position and velocity in 2D)
    Measurement: range from origin (nonlinear)

    Expected: EKF should track the target despite nonlinear measurements.
    """
    dt = 0.1
    n_steps = 100

    # Process model: constant velocity in 2D
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

    # Measurement model: range from origin (nonlinear)
    def measurement_model(x):
        return np.array([np.sqrt(x[0]**2 + x[1]**2)])

    def measurement_jacobian(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        if r < 1e-6:
            return np.array([[0, 0, 0, 0]])
        return np.array([[x[0]/r, x[1]/r, 0, 0]])

    # Noise covariances
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

    # Initial state and covariance
    x0 = np.array([10.0, 5.0, 1.0, 0.5])  # Start at (10, 5) with velocity (1, 0.5)
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])

    # Create EKF
    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0, P0
    )

    # Generate true trajectory
    true_state = x0.copy()
    true_states = [true_state.copy()]
    measurements = []

    np.random.seed(42)
    for _ in range(n_steps):
        # Propagate true state
        true_state = process_model(true_state, None, dt)
        true_state += np.random.multivariate_normal(np.zeros(4), Q_func(dt))
        true_states.append(true_state.copy())

        # Generate measurement (range)
        true_range = measurement_model(true_state)[0]
        z = true_range + np.random.normal(0, np.sqrt(R_func()[0, 0]))
        measurements.append(z)

    # Run EKF
    estimates = [x0.copy()]
    for z in measurements:
        ekf.predict(dt=dt)
        ekf.update(np.array([z]))
        x_est, _ = ekf.get_state()
        estimates.append(x_est.copy())

    # Check final position error
    final_true = np.array(true_states[-1])
    final_est = estimates[-1]
    position_error = np.linalg.norm(final_est[:2] - final_true[:2])
    velocity_error = np.linalg.norm(final_est[2:] - final_true[2:])

    print(f"2D Range-Only Tracking Test:")
    print(f"  Final position error: {position_error:.4f} m")
    print(f"  Final velocity error: {velocity_error:.4f} m/s")
    print(f"  Expected: < 2.0 m and < 1.0 m/s")

    assert position_error < 2.0, f"Position error {position_error} too large"
    assert velocity_error < 1.0, f"Velocity error {velocity_error} too large"

    print("  [PASS] Test passed")


def test_ekf_bearing_only_tracking():
    """
    Unit test: 2D bearing-only tracking with EKF.

    State: [x, y, vx, vy]
    Measurement: bearing angle from origin (nonlinear)

    Expected: EKF should estimate state despite bearing-only measurements.
    """
    dt = 0.1
    n_steps = 50

    # Same process model as before
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

    # Measurement model: bearing angle from origin
    def measurement_model(x):
        return np.array([np.arctan2(x[1], x[0])])

    def measurement_jacobian(x):
        r_sq = x[0]**2 + x[1]**2
        if r_sq < 1e-6:
            return np.array([[0, 0, 0, 0]])
        return np.array([[-x[1]/r_sq, x[0]/r_sq, 0, 0]])

    q = 0.1
    def Q_func(dt):
        return q * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

    def R_func():
        return np.array([[0.05]])  # 0.05 rad^2 variance

    x0 = np.array([10.0, 5.0, 0.5, 0.3])
    P0 = np.diag([2.0, 2.0, 1.0, 1.0])

    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0, P0
    )

    # Generate data
    true_state = x0.copy()
    np.random.seed(42)

    for _ in range(n_steps):
        true_state = process_model(true_state, None, dt)
        true_state += np.random.multivariate_normal(np.zeros(4), Q_func(dt))

        true_bearing = measurement_model(true_state)[0]
        z = true_bearing + np.random.normal(0, np.sqrt(R_func()[0, 0]))

        ekf.predict(dt=dt)
        ekf.update(np.array([z]))

    x_est, P_est = ekf.get_state()

    # Just check that filter ran without errors and covariance is positive definite
    assert np.all(np.linalg.eigvals(P_est) > 0), "Covariance not positive definite"

    print("Bearing-Only Tracking Test:")
    print(f"  Final state: {x_est}")
    print(f"  Covariance trace: {np.trace(P_est):.4f}")
    print("  [PASS] Test passed")


def test_ekf_jacobian_evaluation_point():
    """
    Regression test: Verify process Jacobian F_{k-1} is evaluated at x̂_{k-1}.

    This test uses a nonlinear process model where the Jacobian differs
    significantly between x_{k-1} (pre-prediction) and x_k^- (post-prediction).

    Book Reference (Eq. 3.22):
        F_{k-1} = ∂f/∂x|_{x̂_{k-1}}

    The test creates a scenario where using the wrong evaluation point
    would produce measurably different covariance propagation.
    """
    # State: [x, v] where x is position and v is velocity
    # Nonlinear process model: x_k = x_{k-1} + v_{k-1}*dt + 0.1*x_{k-1}^2*dt
    # This has state-dependent dynamics where Jacobian changes with x

    dt = 0.1

    def process_model(x, u, dt):
        """Nonlinear: position update depends on x^2."""
        x_new = np.zeros(2)
        x_new[0] = x[0] + x[1] * dt + 0.1 * x[0] ** 2 * dt  # Nonlinear term
        x_new[1] = x[1]  # Constant velocity
        return x_new

    def process_jacobian(x, u, dt):
        """Jacobian depends on x[0], so evaluation point matters!"""
        F = np.array([
            [1.0 + 0.2 * x[0] * dt, dt],  # ∂f_0/∂x_0 = 1 + 0.2*x*dt
            [0.0, 1.0]
        ])
        return F

    # Simple linear measurement (position only)
    def measurement_model(x):
        return np.array([x[0]])

    def measurement_jacobian(x):
        return np.array([[1.0, 0.0]])

    def Q_func(dt):
        return 0.01 * np.eye(2)

    def R_func():
        return np.array([[0.1]])

    # Initial state with large position (so Jacobian varies significantly)
    x0 = np.array([10.0, 2.0])  # Large x[0] makes Jacobian state-dependent
    P0 = np.diag([1.0, 0.5])

    # Create EKF
    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0, P0
    )

    # Record pre-prediction state
    x_pre = ekf.state.copy()

    # Perform prediction
    ekf.predict(dt=dt)

    # Get post-prediction state
    x_post = ekf.state.copy()

    # Verify the states are different (nonlinear model)
    assert not np.allclose(x_pre, x_post), "Pre and post states should differ"

    # Compute what covariance SHOULD be (using pre-state Jacobian per Eq. 3.22)
    F_correct = process_jacobian(x_pre, None, dt)  # At x_{k-1}
    P_correct = F_correct @ P0 @ F_correct.T + Q_func(dt)

    # Compute what covariance WOULD BE if Jacobian was at wrong point
    F_wrong = process_jacobian(x_post, None, dt)  # At x_k^- (WRONG!)
    P_wrong = F_wrong @ P0 @ F_wrong.T + Q_func(dt)

    # The Jacobians should be different due to state-dependent term
    assert not np.allclose(F_correct, F_wrong), (
        f"Jacobians at different points should differ: "
        f"F(x_pre)={F_correct[0,0]:.4f} vs F(x_post)={F_wrong[0,0]:.4f}"
    )

    # Verify EKF used the CORRECT (pre-state) Jacobian
    assert np.allclose(ekf.covariance, P_correct, atol=1e-10), (
        f"Covariance mismatch: EKF used wrong Jacobian evaluation point!\n"
        f"EKF P = {ekf.covariance}\n"
        f"Correct P (at x_pre) = {P_correct}\n"
        f"Wrong P (at x_post) = {P_wrong}"
    )

    # Verify it's NOT the wrong covariance
    assert not np.allclose(ekf.covariance, P_wrong, atol=1e-10) or \
           np.allclose(P_correct, P_wrong, atol=1e-10), (
        "EKF appears to use post-prediction state for Jacobian (violates Eq. 3.22)"
    )

    print("Jacobian Evaluation Point Test (Eq. 3.22):")
    print(f"  Pre-prediction state x_{{k-1}}:  {x_pre}")
    print(f"  Post-prediction state x_k^-: {x_post}")
    print(f"  F(x_{{k-1}})[0,0] = {F_correct[0, 0]:.6f}")
    print(f"  F(x_k^-)[0,0]   = {F_wrong[0, 0]:.6f}")
    print(f"  Jacobian difference: {abs(F_correct[0,0] - F_wrong[0,0]):.6f}")
    print(f"  Covariance P[0,0]: {ekf.covariance[0, 0]:.6f}")
    print(f"  Expected P[0,0]:   {P_correct[0, 0]:.6f}")
    print("  [PASS] Jacobian correctly evaluated at pre-prediction state")


if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED KALMAN FILTER UNIT TESTS")
    print("=" * 70)
    print()

    test_ekf_jacobian_evaluation_point()
    print()
    test_ekf_range_only_tracking()
    print()
    test_ekf_bearing_only_tracking()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)



