"""
IMU measurement correction and calibration helpers (Chapter 6).

This module provides functions for correcting raw IMU measurements:
    - Gyroscope bias and noise correction (Eq. (6.6))
    - Accelerometer bias and noise correction (Eq. (6.9))
    - Scale factor and misalignment correction (Eq. (6.59))

These corrections are applied before strapdown integration to obtain
true specific force and angular velocity in the body frame.

Frame Conventions:
    - B: Body frame (sensor frame)
    - All IMU measurements and corrections are in body frame B

References:
    Chapter 6, Section 6.1: IMU error models and strapdown integration
    Eq. (6.5): Gyro error model (ω̃ = ω + b_g + n_g)
    Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
    Eq. (6.9): Accel error model and correction
    Eq. (6.59): Scale factor and misalignment model
"""

from typing import Optional
import numpy as np


def correct_gyro(
    gyro_meas: np.ndarray,
    b_g: np.ndarray,
    n_g: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Correct gyroscope measurement by removing bias and noise.

    Implements Eq. (6.6) in Chapter 6:
        ω = ω̃ - b_g - n_g

    where:
        ω̃ (gyro_meas): raw gyro measurement [rad/s]
        b_g: gyro bias [rad/s]
        n_g: gyro noise [rad/s] (optional, typically zero mean)
        ω: true angular velocity in body frame B [rad/s]

    The gyro error model (Eq. (6.5)) states:
        ω̃ = ω + b_g + n_g

    This function inverts the error model to recover the true angular velocity.

    Args:
        gyro_meas: Raw gyroscope measurement in body frame B.
                   Shape: (3,) or (N, 3). Units: rad/s.
        b_g: Gyroscope bias in body frame B.
             Shape: (3,) or (N, 3) matching gyro_meas. Units: rad/s.
             Typically slow-varying (random walk process in EKF).
        n_g: Gyroscope white noise (optional).
             Shape: (3,) or (N, 3) matching gyro_meas. Units: rad/s.
             If None, assumes zero noise (n_g = 0).

    Returns:
        Corrected angular velocity ω in body frame B.
        Shape matches input gyro_meas. Units: rad/s.

    Notes:
        - Bias b_g is typically estimated online via EKF (NavStateQPVPBias).
        - Noise n_g is usually zero-mean and handled stochastically in filters.
        - For deterministic integration, set n_g=None (default).
        - Broadcast rules apply: b_g and n_g can be scalar, (3,), or (N, 3).

    Example:
        >>> import numpy as np
        >>> gyro_raw = np.array([0.1, 0.05, -0.02])  # rad/s
        >>> bias = np.array([0.001, -0.0005, 0.0002])  # small bias
        >>> gyro_true = correct_gyro(gyro_raw, bias)
        >>> print(gyro_true)  # [0.099, 0.0505, -0.0202]

    Related Equations:
        - Eq. (6.5): Gyro error model (ω̃ = ω + b_g + n_g)
        - Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
        - Eq. (6.2): Quaternion kinematics using corrected ω
    """
    # Apply correction: ω = ω̃ - b_g - n_g (Eq. 6.6)
    omega_corrected = gyro_meas - b_g

    if n_g is not None:
        omega_corrected = omega_corrected - n_g

    return omega_corrected


def correct_accel(
    accel_meas: np.ndarray,
    b_a: np.ndarray,
    n_a: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Correct accelerometer measurement by removing bias and noise.

    Implements the accelerometer correction consistent with Eq. (6.9) in Chapter 6:
        f = f̃ - b_a - n_a

    where:
        f̃ (accel_meas): raw accel measurement [m/s²]
        b_a: accel bias [m/s²]
        n_a: accel noise [m/s²] (optional, typically zero mean)
        f: true specific force in body frame B [m/s²]

    The accelerometer error model (similar to Eq. (6.5) for gyro) states:
        f̃ = f + b_a + n_a

    This function inverts the error model to recover true specific force.
    Note: specific force includes all non-gravitational accelerations measured
    by the accelerometer (gravity is removed later in velocity update via Eq. (6.7)).

    Args:
        accel_meas: Raw accelerometer measurement in body frame B.
                    Shape: (3,) or (N, 3). Units: m/s².
                    Includes gravity component (f̃ = a - g in body frame).
        b_a: Accelerometer bias in body frame B.
             Shape: (3,) or (N, 3) matching accel_meas. Units: m/s².
             Typically slow-varying (random walk in EKF).
        n_a: Accelerometer white noise (optional).
             Shape: (3,) or (N, 3) matching accel_meas. Units: m/s².
             If None, assumes zero noise (n_a = 0).

    Returns:
        Corrected specific force f in body frame B.
        Shape matches input accel_meas. Units: m/s².

    Notes:
        - Bias b_a is typically estimated online via EKF (NavStateQPVPBias).
        - Noise n_a is usually zero-mean and handled stochastically in filters.
        - Specific force f is used directly in velocity update (Eq. 6.7).
        - Gravity is NOT removed here; it's handled in vel_update via C_B^M rotation.

    Example:
        >>> import numpy as np
        >>> accel_raw = np.array([0.05, 0.1, -9.81])  # m/s² (mostly gravity)
        >>> bias = np.array([0.01, -0.005, 0.02])  # small bias
        >>> accel_corrected = correct_accel(accel_raw, bias)
        >>> print(accel_corrected)  # [0.04, 0.105, -9.83]

    Related Equations:
        - Eq. (6.9): Accel error model and correction
        - Eq. (6.7): Velocity update using corrected specific force
        - Eq. (6.8): Gravity vector definition
    """
    # Apply correction: f = f̃ - b_a - n_a (consistent with Eq. 6.9)
    f_corrected = accel_meas - b_a

    if n_a is not None:
        f_corrected = f_corrected - n_a

    return f_corrected


def apply_imu_scale_misalignment(
    u: np.ndarray,
    M: np.ndarray,
    S: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Apply scale factor and misalignment correction to IMU measurements.

    Implements Eq. (6.59) in Chapter 6:
        u_corrected = M @ S @ (u_raw - b)

    where:
        u_raw: raw IMU measurement (gyro or accel)
        b: bias vector
        S: scale factor matrix (diagonal)
        M: misalignment matrix (orthogonal or near-orthogonal)
        u_corrected: calibrated measurement

    This correction accounts for:
        - Non-unity scale factors (sensor sensitivity errors)
        - Non-orthogonality of sensor axes (misalignment)
        - Bias offsets

    Args:
        u: Raw IMU measurement (gyro or accel).
           Shape: (3,) or (N, 3). Units depend on sensor (rad/s or m/s²).
        M: Misalignment correction matrix.
           Shape: (3, 3). Typically close to identity for well-calibrated IMUs.
           Corrects for non-orthogonality of sensor axes.
        S: Scale factor matrix (diagonal).
           Shape: (3, 3). Diagonal elements are scale corrections.
           Example: S = diag([1.02, 0.98, 1.01]) for ±2% scale errors.
        b: Bias vector.
           Shape: (3,). Units match u (rad/s or m/s²).
           Static bias determined during calibration.

    Returns:
        Calibrated IMU measurement u_corrected.
        Shape matches input u.

    Notes:
        - This is a deterministic calibration step, typically applied offline.
        - Calibration parameters (M, S, b) are determined via:
            * Multi-position static tests (for accel)
            * Rate table tests (for gyro)
            * Allan variance analysis (see allan_variance in calibration.py)
        - For online bias estimation, use NavStateQPVPBias with EKF.
        - Misalignment matrix M is often parameterized by 3 small angles.

    Example:
        >>> import numpy as np
        >>> # Raw gyro with 2% scale error and small misalignment
        >>> gyro_raw = np.array([0.1, 0.05, -0.02])
        >>> b = np.array([0.001, -0.0005, 0.0002])  # bias
        >>> S = np.diag([1.02, 0.98, 1.01])  # scale factors
        >>> M = np.eye(3)  # no misalignment (identity)
        >>> gyro_cal = apply_imu_scale_misalignment(gyro_raw, M, S, b)

    Related Equations:
        - Eq. (6.59): Scale and misalignment model
        - Eqs. (6.56)-(6.58): Allan variance for noise characterization
    """
    # Validate shapes
    if M.shape != (3, 3):
        raise ValueError(f"M must be (3, 3), got {M.shape}")
    if S.shape != (3, 3):
        raise ValueError(f"S must be (3, 3), got {S.shape}")
    if b.shape != (3,):
        raise ValueError(f"b must be (3,), got {b.shape}")

    # Handle both single measurement (3,) and batch (N, 3)
    if u.ndim == 1:
        # Single measurement: u_corrected = M @ S @ (u - b)  (Eq. 6.59)
        u_corrected = M @ S @ (u - b)
    elif u.ndim == 2 and u.shape[1] == 3:
        # Batch measurements: apply correction row-wise
        u_corrected = np.zeros_like(u)
        for i in range(u.shape[0]):
            u_corrected[i, :] = M @ S @ (u[i, :] - b)
    else:
        raise ValueError(f"u must have shape (3,) or (N, 3), got {u.shape}")

    return u_corrected


def remove_gravity_component(
    accel_body: np.ndarray,
    gravity_map: np.ndarray,
    C_B_M: np.ndarray,
) -> np.ndarray:
    """
    Remove gravity component from accelerometer measurement.

    Helper function to compute body-frame acceleration excluding gravity:
        a_body = f_body - C_M^B @ g_map

    where:
        f_body: specific force measured by accelerometer (includes gravity)
        g_map: gravity vector in map frame (typically [0, 0, -9.81] for ENU)
        C_M^B: rotation matrix from map to body (inverse of C_B^M)
        a_body: true acceleration in body frame (excludes gravity)

    This is NOT directly used in strapdown integration (which uses Eq. 6.7
    in map frame), but is useful for:
        - Pre-processing accelerometer data
        - Computing body-frame acceleration for sensor fusion
        - Step detection algorithms (PDR, Eq. 6.46)

    Args:
        accel_body: Accelerometer measurement (specific force) in body frame B.
                    Shape: (3,) or (N, 3). Units: m/s².
        gravity_map: Gravity vector in map frame M.
                     Shape: (3,). Units: m/s². Typically [0, 0, -9.81] for ENU.
        C_B_M: Rotation matrix from body to map frame.
               Shape: (3, 3). Orthogonal matrix (C_M^B = C_B^M^T).

    Returns:
        True acceleration in body frame B (excludes gravity).
        Shape matches input accel_body. Units: m/s².

    Notes:
        - For strapdown integration, use vel_update which handles gravity in map frame.
        - This function is useful for PDR step detection (Eq. 6.46).
        - Assumes gravity_map is constant (valid for indoor positioning scales).

    Example:
        >>> import numpy as np
        >>> # Stationary sensor: f_body ≈ -C_M^B @ g
        >>> accel = np.array([0.0, 0.0, -9.81])  # body = map (level)
        >>> g_map = np.array([0.0, 0.0, -9.81])
        >>> C_B_M = np.eye(3)  # identity (body = map)
        >>> a_true = remove_gravity_component(accel, g_map, C_B_M)
        >>> print(a_true)  # ≈ [0, 0, 0] (no motion)

    Related Equations:
        - Eq. (6.7): Velocity update (uses gravity in map frame)
        - Eq. (6.8): Gravity vector definition
        - Eq. (6.46): Acceleration magnitude for step detection
    """
    # Validate shapes
    if C_B_M.shape != (3, 3):
        raise ValueError(f"C_B_M must be (3, 3), got {C_B_M.shape}")
    if gravity_map.shape != (3,):
        raise ValueError(f"gravity_map must be (3,), got {gravity_map.shape}")

    # C_M^B = C_B^M^T (transpose of rotation matrix)
    C_M_B = C_B_M.T

    # Gravity in body frame: g_body = C_M^B @ g_map
    gravity_body = C_M_B @ gravity_map

    # True acceleration: a = f - g (in body frame)
    if accel_body.ndim == 1:
        a_true = accel_body - gravity_body
    elif accel_body.ndim == 2 and accel_body.shape[1] == 3:
        # Broadcast gravity_body across batch dimension
        a_true = accel_body - gravity_body[np.newaxis, :]
    else:
        raise ValueError(f"accel_body must have shape (3,) or (N, 3), got {accel_body.shape}")

    return a_true

