"""
Generate synthetic IMU measurements from ground truth trajectory.

This module implements the correct IMU forward model consistent with
Chapter 6 sensor equations:
    - Eq. (6.7): Velocity update v_k = v_{k-1} + (C_B^M @ f_b + g_M) * dt
    - Eq. (6.9): Accelerometer model a_tilde = a + b_a + n_a

Key insight: Accelerometers measure **specific force** (reaction force),
NOT gravitational acceleration. For a stationary device in ENU:
    - True acceleration: a_M = [0, 0, 0]
    - Specific force: f_b = [0, 0, +9.81] (upward reaction from ground)

The correct forward model is:
    1. Compute true acceleration in map frame: a_M = d²p/dt²
    2. Subtract gravity: a_M - g_M
    3. Rotate to body frame: f_b = C_M^B @ (a_M - g_M)

This is the INVERSE of Eq. (6.7):
    - Eq. (6.7): a_M = C_B^M @ f_b + g_M (measurement → acceleration)
    - Forward:   f_b = C_M^B @ (a_M - g_M) (acceleration → measurement)

Author: Li-Ta Hsu
Date: December 2025
"""

from typing import Optional, Tuple
import numpy as np

from core.sensors.types import FrameConvention
from core.sensors.strapdown import quat_to_rotmat


def compute_specific_force_body(
    accel_map: np.ndarray,
    quat_b_to_m: np.ndarray,
    frame: Optional[FrameConvention] = None,
    g: float = 9.81,
) -> np.ndarray:
    """
    Compute specific force in body frame from true acceleration in map frame.

    This implements the correct IMU forward model for accelerometers.
    Accelerometers measure specific force (reaction force), not acceleration.

    Forward model (inverse of Eq. 6.7):
        f_b = C_M^B @ (a_M - g_M)

    where:
        f_b: specific force in body frame [m/s²] (accelerometer reading)
        a_M: true acceleration in map frame [m/s²]
        g_M: gravity vector in map frame [m/s²]
        C_M^B: rotation matrix from map to body (inverse of C_B^M)

    Args:
        accel_map: True acceleration in map frame.
                   Shape: (N, 3) or (3,). Units: m/s².
                   This is d²p/dt² or dv/dt.
        quat_b_to_m: Quaternion(s) representing body-to-map rotation.
                     Shape: (N, 4) or (4,). Scalar-first [q0, q1, q2, q3].
        frame: Frame convention defining gravity direction.
               Default: None (creates ENU).
        g: Gravitational acceleration magnitude.
           Default: 9.81 m/s².

    Returns:
        Specific force in body frame.
        Shape: (N, 3) or (3,). Units: m/s².
        This is what an ideal accelerometer would measure.

    Notes:
        - For stationary: a_M = 0, f_b = C_M^B @ (-g_M)
        - In ENU with identity rotation: f_b = [0, 0, +9.81] (upward reaction)
        - In NED with identity rotation: f_b = [0, 0, -9.81] (upward reaction)
        - This is the INVERSE of the velocity update equation (Eq. 6.7)

    Example:
        >>> import numpy as np
        >>> from core.sensors import FrameConvention
        >>> # Stationary device in ENU
        >>> frame = FrameConvention.create_enu()
        >>> a_M = np.zeros(3)  # no acceleration
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        >>> f_b = compute_specific_force_body(a_M, q, frame=frame)
        >>> print(f_b)  # [0, 0, +9.81] (upward reaction)

    Related Equations:
        - Eq. (6.7): Velocity update (measurement → acceleration)
        - Eq. (6.9): Accelerometer sensor model
        - This function: Acceleration → measurement (forward model)
    """
    if frame is None:
        frame = FrameConvention.create_enu()

    # Handle single sample vs time series
    single_sample = accel_map.ndim == 1
    if single_sample:
        accel_map = accel_map.reshape(1, -1)
        quat_b_to_m = quat_b_to_m.reshape(1, -1)

    N = accel_map.shape[0]

    # Gravity vector in map frame
    g_M = frame.gravity_vector(g)

    # Compute specific force in body frame for each sample
    f_b = np.zeros((N, 3))
    for i in range(N):
        # Rotation matrix: body to map
        C_B_M = quat_to_rotmat(quat_b_to_m[i])

        # Rotation matrix: map to body (transpose/inverse)
        C_M_B = C_B_M.T

        # Forward model: f_b = C_M^B @ (a_M - g_M)
        # This is the inverse of Eq. (6.7): a_M = C_B^M @ f_b + g_M
        f_b[i] = C_M_B @ (accel_map[i] - g_M)

    if single_sample:
        return f_b[0]
    return f_b


def compute_gyro_body(
    quat_series: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Compute angular velocity in body frame from quaternion time series.

    Derives gyroscope measurements by numerically differentiating
    the quaternion time series. Uses Eq. (6.2) in discrete form:
        ω ≈ 2 * Ω^(-1) @ (q_k - q_{k-1}) / dt

    where Ω is the omega matrix from Eq. (6.3).

    Args:
        quat_series: Quaternion time series (body-to-map rotation).
                     Shape: (N, 4). Scalar-first [q0, q1, q2, q3].
        dt: Time step between samples.
            Units: seconds.

    Returns:
        Angular velocity in body frame.
        Shape: (N, 3). Units: rad/s.
        First sample is zero (no previous sample to differentiate).

    Notes:
        - Uses simple finite difference approximation
        - More accurate methods exist (SLERP-based, etc.) but this is sufficient
        - First sample is set to zero (could also extrapolate)
        - For planar motion (only yaw changes), ω_z = dψ/dt

    Example:
        >>> import numpy as np
        >>> # Constant yaw rate trajectory
        >>> t = np.linspace(0, 10, 100)
        >>> dt = t[1] - t[0]
        >>> yaw_rate = 0.1  # rad/s
        >>> yaw = yaw_rate * t
        >>> # Convert to quaternions (assuming level)
        >>> quat = np.column_stack([
        ...     np.cos(yaw/2), np.zeros_like(yaw),
        ...     np.zeros_like(yaw), np.sin(yaw/2)
        ... ])
        >>> omega = compute_gyro_body(quat, dt)
        >>> print(np.mean(omega[:, 2]))  # Should be ~0.1 rad/s

    Related Equations:
        - Eq. (6.2): Quaternion kinematics dq/dt = 0.5 * Ω(ω) * q
        - Eq. (6.3): Ω(ω) matrix definition
        - This function: Inverse problem (q → ω)
    """
    N = quat_series.shape[0]
    omega = np.zeros((N, 3))

    # Finite difference approximation
    for i in range(1, N):
        q_prev = quat_series[i - 1]
        q_curr = quat_series[i]

        # Quaternion difference
        dq = q_curr - q_prev

        # From Eq. (6.2): dq/dt ≈ 0.5 * Ω(ω) * q
        # Solve for ω using the quaternion product inverse
        # Simplified approach: extract from quaternion derivative
        # ω ≈ 2 * q^* ⊗ dq/dt (quaternion product with conjugate)

        # Conjugate of q_prev
        q_prev_conj = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])

        # Quaternion product: q_prev^* ⊗ dq
        # Result gives ω approximately in vector part
        dq_dt = dq / dt
        omega_quat = _quat_multiply(q_prev_conj, dq_dt)

        # Extract angular velocity from vector part, scaled by 2
        omega[i] = 2.0 * omega_quat[1:4]

    # First sample: assume zero (or copy second)
    omega[0] = omega[1] if N > 1 else np.zeros(3)

    return omega


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q1 ⊗ q2.

    Scalar-first convention: q = [q0, q1, q2, q3].

    Args:
        q1: First quaternion, shape (4,).
        q2: Second quaternion, shape (4,).

    Returns:
        Product quaternion, shape (4,).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def generate_imu_from_trajectory(
    pos_map: np.ndarray,
    vel_map: np.ndarray,
    quat_b_to_m: np.ndarray,
    dt: float,
    frame: Optional[FrameConvention] = None,
    g: float = 9.81,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic IMU measurements from ground truth trajectory.

    This is the complete forward model for IMU synthesis:
        1. Compute map acceleration: a_M = dv/dt (numerical derivative)
        2. Compute specific force: f_b = C_M^B @ (a_M - g_M)
        3. Compute gyro rates: ω_b from quaternion derivative

    Args:
        pos_map: Position trajectory in map frame.
                 Shape: (N, 3). Units: m.
                 Used for validation (can be None if vel_map is accurate).
        vel_map: Velocity trajectory in map frame.
                 Shape: (N, 3). Units: m/s.
        quat_b_to_m: Quaternion trajectory (body-to-map rotation).
                     Shape: (N, 4). Scalar-first [q0, q1, q2, q3].
        dt: Time step between samples.
            Units: seconds.
        frame: Frame convention defining gravity direction.
               Default: None (creates ENU).
        g: Gravitational acceleration magnitude.
           Default: 9.81 m/s².

    Returns:
        Tuple (accel_body, gyro_body):
            accel_body: Specific force in body frame, shape (N, 3), units: m/s²
            gyro_body: Angular velocity in body frame, shape (N, 3), units: rad/s

    Notes:
        - These are **ideal** measurements (no noise, no bias)
        - Add noise/bias using core.sensors.imu_models functions
        - For stationary: accel_body ≈ [0, 0, ±g] (depending on frame)
        - First samples may have artifacts from numerical differentiation

    Example:
        >>> import numpy as np
        >>> from core.sensors import FrameConvention
        >>> # Simple trajectory: constant velocity eastward
        >>> frame = FrameConvention.create_enu()
        >>> t = np.linspace(0, 10, 100)
        >>> dt = t[1] - t[0]
        >>> pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
        >>> vel = np.column_stack([np.ones_like(t), np.zeros_like(t), np.zeros_like(t)])
        >>> quat = np.tile([1, 0, 0, 0], (100, 1))  # identity
        >>> accel_b, gyro_b = generate_imu_from_trajectory(
        ...     pos, vel, quat, dt, frame=frame
        ... )
        >>> # Constant velocity → zero acceleration → upward reaction only
        >>> print(np.mean(accel_b, axis=0))  # ≈ [0, 0, +9.81]

    Related Equations:
        - Eq. (6.7): Velocity update (inverse of this)
        - Eq. (6.9): Accelerometer sensor model
        - Eq. (6.2)-(6.4): Quaternion kinematics (inverse)
    """
    if frame is None:
        frame = FrameConvention.create_enu()

    N = vel_map.shape[0]

    # Step 1: Compute acceleration in map frame (numerical derivative of velocity)
    accel_map = np.zeros((N, 3))
    for i in range(1, N):
        accel_map[i] = (vel_map[i] - vel_map[i - 1]) / dt

    # First sample: use second sample (or could extrapolate)
    accel_map[0] = accel_map[1] if N > 1 else np.zeros(3)

    # Step 2: Compute specific force in body frame
    accel_body = compute_specific_force_body(accel_map, quat_b_to_m, frame, g)

    # Step 3: Compute gyro rates in body frame
    gyro_body = compute_gyro_body(quat_b_to_m, dt)

    return accel_body, gyro_body








