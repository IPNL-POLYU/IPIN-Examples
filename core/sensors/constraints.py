"""
Drift correction constraints for INS (Chapter 6).

This module implements constraint-based drift reduction techniques:
    - ZUPT: Zero velocity update (Eq. (6.44)-(6.45))
    - ZARU: Zero angular rate update (Eq. (6.60))
    - NHC: Nonholonomic constraint (Eq. (6.61))

These constraints provide pseudo-measurements that can be integrated into
an EKF to correct IMU drift when specific motion assumptions hold (e.g.,
foot on ground, vehicle not sliding sideways).

Design Note:
    These are MeasurementModel classes compatible with core/estimators/ExtendedKalmanFilter.
    They provide h(x), H(x), and R(x) methods for EKF updates.

Frame Conventions:
    - B: Body frame (IMU/sensor frame)
    - M: Map frame (navigation frame, typically ENU)
    - Constraints operate on velocity in specific frames

References:
    Chapter 6, Section 6.2: Drift correction constraints
    Eq. (6.44): ZUPT detector (stationary detection)
    Eq. (6.45): ZUPT pseudo-measurement (v = 0)
    Eq. (6.60): ZARU pseudo-measurement (ω = 0)
    Eq. (6.61): NHC pseudo-measurement (lateral/vertical velocity = 0)
"""

from typing import Optional
import numpy as np


def detect_zupt(
    gyro_b: np.ndarray,
    accel_b: np.ndarray,
    delta_omega: float,
    delta_f: float,
) -> bool:
    """
    Detect zero velocity (stationary) condition for ZUPT.

    Implements Eq. (6.44) in Chapter 6:
        ZUPT if: ||ω_B|| < δ_ω  AND  ||f_B - g|| < δ_f

    where:
        ω_B: angular velocity in body frame [rad/s]
        f_B: specific force in body frame [m/s²]
        g: gravity magnitude (≈ 9.81 m/s²)
        δ_ω: angular rate threshold [rad/s]
        δ_f: acceleration threshold [m/s²]

    The detector checks if the sensor is stationary (not rotating and not
    accelerating). This is typically used for foot-mounted INS where stance
    phases can be detected.

    Args:
        gyro_b: Angular velocity in body frame B (corrected).
                Shape: (3,). Units: rad/s.
        accel_b: Specific force in body frame B (corrected).
                 Shape: (3,). Units: m/s².
                 For stationary sensor: ||accel_b|| ≈ g (gravity only).
        delta_omega: Angular rate threshold for stationary detection.
                     Units: rad/s. Typical value: 0.05 rad/s (≈3°/s).
        delta_f: Acceleration threshold for stationary detection.
                 Units: m/s². Typical value: 0.5 m/s².

    Returns:
        True if ZUPT condition is met (sensor is stationary), False otherwise.

    Notes:
        - Thresholds (δ_ω, δ_f) are tuning parameters.
        - Too strict: miss some stance phases; too loose: false positives.
        - For foot-mounted INS: δ_ω ≈ 0.05 rad/s, δ_f ≈ 0.5 m/s² work well.
        - Detector should be applied to a window (e.g., last 5-10 samples).
        - False positives during dynamic motion will degrade performance.
        - Can be extended with variance-based detection (not implemented here).

    Example:
        >>> import numpy as np
        >>> # Stationary sensor
        >>> gyro = np.array([0.01, -0.005, 0.002])  # very small rotation
        >>> accel = np.array([0.0, 0.0, -9.81])  # gravity only
        >>> is_stationary = detect_zupt(gyro, accel, delta_omega=0.05, delta_f=0.5)
        >>> print(is_stationary)  # True
        >>> 
        >>> # Moving sensor
        >>> gyro_moving = np.array([0.5, 0.2, -0.1])  # rotating
        >>> accel_moving = np.array([2.0, 0.5, -9.0])  # accelerating
        >>> is_stationary = detect_zupt(gyro_moving, accel_moving, 0.05, 0.5)
        >>> print(is_stationary)  # False

    Related Equations:
        - Eq. (6.44): ZUPT detector (THIS FUNCTION)
        - Eq. (6.45): ZUPT pseudo-measurement (see ZuptMeasurementModel)
    """
    if gyro_b.shape != (3,):
        raise ValueError(f"gyro_b must have shape (3,), got {gyro_b.shape}")
    if accel_b.shape != (3,):
        raise ValueError(f"accel_b must have shape (3,), got {accel_b.shape}")

    # Check angular rate: ||ω|| < δ_ω
    omega_norm = np.linalg.norm(gyro_b)
    omega_stationary = omega_norm < delta_omega

    # Check specific force deviation from gravity: ||f - g|| < δ_f
    # For stationary sensor: f ≈ [0, 0, ±g] (depends on orientation)
    g_magnitude = 9.81  # m/s²
    f_norm = np.linalg.norm(accel_b)
    f_deviation = np.abs(f_norm - g_magnitude)
    f_stationary = f_deviation < delta_f

    # ZUPT condition (Eq. 6.44): both conditions must hold
    zupt_detected = omega_stationary and f_stationary

    return zupt_detected


class ZuptMeasurementModel:
    """
    Zero velocity update (ZUPT) pseudo-measurement model.

    Implements Eq. (6.45) in Chapter 6:
        z_ZUPT = 0 (expected measurement when stationary)
        h(x) = v (velocity from state)
        Innovation: ν = z_ZUPT - h(x) = 0 - v = -v

    ZUPT provides a velocity measurement of exactly zero when the sensor
    is detected to be stationary. This corrects velocity drift in the EKF.

    The measurement model is:
        z = h(x) + w, where h(x) = v and z = 0
        R = σ_ZUPT² I (measurement noise covariance)

    Usage with EKF:
        1. Detect ZUPT condition using detect_zupt()
        2. If ZUPT detected, call ekf.update(z=np.zeros(3)) with this model
        3. EKF will constrain velocity estimate to zero

    Attributes:
        sigma_zupt: ZUPT measurement noise standard deviation.
                    Units: m/s. Typical value: 0.01 to 0.1 m/s.
                    Smaller σ = stronger constraint (more confident in zero velocity).

    Notes:
        - Only apply ZUPT when detect_zupt() returns True.
        - False ZUPT application during motion will corrupt navigation.
        - ZUPT is most effective for foot-mounted INS (stance phases).
        - Can be applied to individual axes (e.g., only horizontal velocity).

    Example:
        >>> import numpy as np
        >>> from core.sensors.constraints import ZuptMeasurementModel, detect_zupt
        >>> 
        >>> # Assume we have an EKF with state [q, v, p, b_g, b_a]
        >>> zupt_model = ZuptMeasurementModel(sigma_zupt=0.05)
        >>> 
        >>> # Check if stationary
        >>> if detect_zupt(gyro, accel, delta_omega=0.05, delta_f=0.5):
        >>>     z_zupt = np.zeros(3)  # zero velocity measurement
        >>>     # ekf.update(z_zupt, zupt_model)  # would update EKF

    Related Equations:
        - Eq. (6.44): ZUPT detector (see detect_zupt)
        - Eq. (6.45): ZUPT pseudo-measurement (THIS CLASS)
    """

    def __init__(self, sigma_zupt: float = 0.05):
        """
        Initialize ZUPT measurement model.

        Args:
            sigma_zupt: ZUPT measurement noise std dev (m/s). Default: 0.05 m/s.
        """
        if sigma_zupt <= 0:
            raise ValueError(f"sigma_zupt must be positive, got {sigma_zupt}")
        self.sigma_zupt = sigma_zupt

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: h(x) = v (extract velocity from state).

        Assumes state vector x = [q (4), v (3), p (3), ...] where velocity
        is at indices 4:7.

        Args:
            x: State vector. Shape: (n,) where n >= 7.
               Expected: [q0, q1, q2, q3, vx, vy, vz, px, py, pz, ...]

        Returns:
            Predicted measurement (velocity).
            Shape: (3,). Units: m/s.
        """
        if x.shape[0] < 7:
            raise ValueError(
                f"State vector must have at least 7 elements, got {x.shape[0]}"
            )

        # Extract velocity (indices 4:7)
        v = x[4:7]
        return v

    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian: H = ∂h/∂x (ZUPT measurement).

        For ZUPT, h(x) = v, so:
            ∂h/∂q = 0 (3×4)
            ∂h/∂v = I (3×3)
            ∂h/∂p = 0 (3×3)
            ∂h/∂(rest) = 0

        Args:
            x: State vector. Shape: (n,) where n >= 7.

        Returns:
            Measurement Jacobian H.
            Shape: (3, n). Selects velocity components from state.
        """
        n = x.shape[0]
        H = np.zeros((3, n))

        # ∂h/∂v = I at indices 4:7
        H[:, 4:7] = np.eye(3)

        return H

    def R(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Measurement noise covariance R.

        For ZUPT: R = σ_ZUPT² I (isotropic, same uncertainty in all directions).

        Args:
            x: State vector (optional, not used for ZUPT).

        Returns:
            Measurement covariance R.
            Shape: (3, 3). Units: (m/s)².
        """
        R = (self.sigma_zupt**2) * np.eye(3)
        return R


class ZaruMeasurementModel:
    """
    Zero angular rate update (ZARU) pseudo-measurement model.

    Implements Eq. (6.60) in Chapter 6:
        z_ZARU = 0 (expected measurement when not rotating)
        h(x) = ω (angular velocity, if estimated in state)
        Innovation: ν = 0 - ω = -ω

    ZARU provides an angular velocity measurement of exactly zero when the
    sensor is stationary or moving without rotation. This corrects gyro
    bias drift in the EKF.

    The measurement model is:
        z = h(x) + w, where h(x) = ω and z = 0
        R = σ_ZARU² I (measurement noise covariance)

    Note: This implementation assumes gyro bias b_g is part of the state.
    The measurement is effectively on the bias: h(x) = ω_measured - b_g,
    and we expect this to be zero when stationary.

    Attributes:
        sigma_zaru: ZARU measurement noise standard deviation.
                    Units: rad/s. Typical value: 0.01 to 0.1 rad/s.

    Notes:
        - ZARU is often applied together with ZUPT (both stationary conditions).
        - Helps estimate and correct gyro bias online.
        - Less common than ZUPT; requires very still sensor.
        - Can be applied when detect_zupt() returns True (same detector).

    Example:
        >>> zaru_model = ZaruMeasurementModel(sigma_zaru=0.01)
        >>> # Apply when stationary (same condition as ZUPT)
        >>> if detect_zupt(gyro, accel, 0.05, 0.5):
        >>>     z_zaru = np.zeros(3)  # zero angular rate measurement
        >>>     # ekf.update(z_zaru, zaru_model)

    Related Equations:
        - Eq. (6.44): Stationary detector (shared with ZUPT)
        - Eq. (6.60): ZARU pseudo-measurement (THIS CLASS)
    """

    def __init__(self, sigma_zaru: float = 0.01):
        """
        Initialize ZARU measurement model.

        Args:
            sigma_zaru: ZARU measurement noise std dev (rad/s). Default: 0.01 rad/s.
        """
        if sigma_zaru <= 0:
            raise ValueError(f"sigma_zaru must be positive, got {sigma_zaru}")
        self.sigma_zaru = sigma_zaru

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: h(x) = corrected angular velocity.

        For augmented state x = [q, v, p, b_g, ...], the corrected angular
        velocity is ω_corrected = ω_measured - b_g. But we don't have
        ω_measured in the state, so we assume ZARU is applied directly
        to gyro bias correction.

        Simplified implementation: returns zero (pseudo-measurement approach).

        Args:
            x: State vector. Shape: (n,).

        Returns:
            Predicted measurement (zero for ZARU).
            Shape: (3,). Units: rad/s.
        """
        # For ZARU: we expect zero angular velocity
        # The measurement is applied as a correction to gyro bias
        return np.zeros(3)

    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian for ZARU.

        For state x = [q, v, p, b_g, b_a], ZARU affects gyro bias b_g.
        Assuming b_g is at indices 10:13 (after q, v, p):
            ∂h/∂b_g = -I (3×3) [since ω = ω_meas - b_g, and we measure ω ≈ 0]

        Simplified: returns zeros for now (needs full state structure).

        Args:
            x: State vector. Shape: (n,).

        Returns:
            Measurement Jacobian H.
            Shape: (3, n).
        """
        n = x.shape[0]
        H = np.zeros((3, n))

        # If gyro bias is at indices 10:13, set ∂h/∂b_g = -I
        if n >= 13:
            H[:, 10:13] = -np.eye(3)

        return H

    def R(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Measurement noise covariance for ZARU.

        Args:
            x: State vector (optional, not used).

        Returns:
            Measurement covariance R.
            Shape: (3, 3). Units: (rad/s)².
        """
        R = (self.sigma_zaru**2) * np.eye(3)
        return R


class NhcMeasurementModel:
    """
    Nonholonomic constraint (NHC) pseudo-measurement model.

    Implements Eq. (6.61) in Chapter 6:
        z_NHC = 0 (expected lateral and vertical velocity)
        h(x) = [v_lateral, v_vertical] in vehicle frame
        Innovation: ν = 0 - h(x)

    NHC assumes that a vehicle (e.g., car, wheeled robot) cannot move
    sideways or vertically due to wheel/ground contact constraints.
    This provides a pseudo-measurement that lateral (y) and vertical (z)
    velocities in the vehicle body frame are zero.

    The measurement model is:
        z = h(x) + w, where h(x) = [v_y, v_z] in body frame and z = 0
        R = diag(σ_lateral², σ_vertical²)

    Usage:
        - Apply continuously during vehicle motion (not just when stationary).
        - Invalid during: wheel slip, jumping, lateral sliding.
        - Very effective for reducing horizontal drift in wheeled vehicles.

    Attributes:
        sigma_lateral: Lateral velocity measurement noise std dev (m/s).
        sigma_vertical: Vertical velocity measurement noise std dev (m/s).

    Notes:
        - NHC is vehicle-specific; invalid for flying or swimming robots.
        - Requires transforming map frame velocity to body frame.
        - Stronger constraint than ZUPT (applies during motion).
        - Can be violated during aggressive maneuvers or rough terrain.

    Example:
        >>> nhc_model = NhcMeasurementModel(sigma_lateral=0.1, sigma_vertical=0.05)
        >>> # Apply during normal vehicle motion
        >>> z_nhc = np.zeros(2)  # zero lateral and vertical velocity
        >>> # ekf.update(z_nhc, nhc_model)

    Related Equations:
        - Eq. (6.61): NHC pseudo-measurement (THIS CLASS)
    """

    def __init__(self, sigma_lateral: float = 0.1, sigma_vertical: float = 0.05):
        """
        Initialize NHC measurement model.

        Args:
            sigma_lateral: Lateral velocity noise std dev (m/s). Default: 0.1 m/s.
            sigma_vertical: Vertical velocity noise std dev (m/s). Default: 0.05 m/s.
        """
        if sigma_lateral <= 0:
            raise ValueError(f"sigma_lateral must be positive, got {sigma_lateral}")
        if sigma_vertical <= 0:
            raise ValueError(f"sigma_vertical must be positive, got {sigma_vertical}")

        self.sigma_lateral = sigma_lateral
        self.sigma_vertical = sigma_vertical

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: h(x) = [v_y, v_z] in body frame.

        Extracts velocity from state x = [q, v_map, p, ...], then transforms
        to body frame using quaternion q, and returns lateral (y) and
        vertical (z) components.

        Args:
            x: State vector. Shape: (n,) where n >= 7.
               Expected: [q (4), v_map (3), p (3), ...]

        Returns:
            Predicted measurement: [v_y_body, v_z_body].
            Shape: (2,). Units: m/s.
        """
        if x.shape[0] < 7:
            raise ValueError(
                f"State vector must have at least 7 elements, got {x.shape[0]}"
            )

        # Extract quaternion and velocity
        q = x[0:4]
        v_map = x[4:7]

        # Transform velocity from map to body frame: v_body = C_M^B @ v_map
        # C_M^B = C_B^M^T (transpose of rotation matrix)
        from core.sensors.strapdown import quat_to_rotmat

        C_B_M = quat_to_rotmat(q)
        C_M_B = C_B_M.T
        v_body = C_M_B @ v_map

        # NHC: lateral (y) and vertical (z) velocity in body frame should be zero
        h_nhc = np.array([v_body[1], v_body[2]])

        return h_nhc

    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian: H = ∂h/∂x for NHC.

        For NHC, h(x) depends on both q and v:
            ∂h/∂q: derivatives of body frame velocity w.r.t. quaternion
            ∂h/∂v: rotation from map to body frame (C_M^B)

        Simplified implementation (linearized around current state).

        Args:
            x: State vector. Shape: (n,) where n >= 7.

        Returns:
            Measurement Jacobian H.
            Shape: (2, n).
        """
        n = x.shape[0]
        H = np.zeros((2, n))

        # Extract quaternion and velocity
        q = x[0:4]
        v_map = x[4:7]

        # Compute C_M^B (rotation from map to body)
        from core.sensors.strapdown import quat_to_rotmat

        C_B_M = quat_to_rotmat(q)
        C_M_B = C_B_M.T

        # ∂h/∂v: extract rows 1 and 2 of C_M^B (y and z components)
        H[:, 4:7] = C_M_B[1:3, :]

        # ∂h/∂q: more complex (quaternion derivative)
        # Simplified: set to zero (acceptable for small orientation errors)
        # Full implementation would compute ∂(C_M^B @ v)/∂q
        H[:, 0:4] = 0.0

        return H

    def R(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Measurement noise covariance for NHC.

        Args:
            x: State vector (optional, not used).

        Returns:
            Measurement covariance R.
            Shape: (2, 2). Units: (m/s)².
            Diagonal matrix: diag(σ_lateral², σ_vertical²).
        """
        R = np.diag([self.sigma_lateral**2, self.sigma_vertical**2])
        return R


