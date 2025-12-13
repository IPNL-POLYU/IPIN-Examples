"""
IMU strapdown integration: quaternion, velocity, and position propagation (Chapter 6).

This module implements the core strapdown inertial navigation equations:
    - Quaternion kinematics and discrete integration (Eqs. (6.2)-(6.4))
    - Velocity update with gravity compensation (Eq. (6.7))
    - Position update (Eq. (6.10))
    - Gravity vector (Eq. (6.8))

The strapdown algorithm integrates IMU measurements (corrected gyro and accel)
to propagate the navigation state (attitude, velocity, position) over time.

Frame Conventions:
    - B: Body frame (IMU/sensor frame)
    - M: Map frame (navigation frame, typically ENU or NED)
    - Quaternion q represents rotation from B to M: v_M = C_B^M(q) @ v_B

Quaternion Convention:
    - Scalar-first: q = [q0, q1, q2, q3] where q0 is scalar, [q1,q2,q3] is vector
    - Unit quaternion: ||q|| = 1
    - Identity quaternion: [1, 0, 0, 0] (body aligned with map)

References:
    Chapter 6, Section 6.1: Strapdown integration
    Eq. (6.2): Quaternion kinematics (dq/dt = 0.5 * Ω(ω) * q)
    Eq. (6.3): Ω matrix definition
    Eq. (6.4): Discrete quaternion update
    Eq. (6.7): Velocity update with gravity
    Eq. (6.8): Gravity vector
    Eq. (6.10): Position update
"""

import numpy as np


def omega_matrix(omega_b: np.ndarray) -> np.ndarray:
    """
    Build the Ω(ω) matrix used in quaternion kinematics.

    Implements Eq. (6.3) in Chapter 6:

        Ω(ω) = [  0    -ωx   -ωy   -ωz ]
               [ ωx     0     ωz   -ωy ]
               [ ωy    -ωz    0     ωx ]
               [ ωz     ωy   -ωx    0  ]

    where ω = [ωx, ωy, ωz]^T is the angular velocity in body frame B.

    The quaternion kinematics equation (Eq. 6.2) is:
        dq/dt = 0.5 * Ω(ω) * q

    This matrix enables the linear differential equation formulation of
    quaternion propagation under constant angular velocity.

    Args:
        omega_b: Angular velocity in body frame B.
                 Shape: (3,). Units: rad/s.
                 Components: [ωx, ωy, ωz] representing rotation rates
                 about body x, y, z axes.

    Returns:
        Ω matrix for quaternion kinematics.
        Shape: (4, 4). Skew-symmetric matrix (Ω^T = -Ω).

    Notes:
        - Ω is skew-symmetric: Ω(ω)^T = -Ω(ω)
        - Trace(Ω) = 0
        - This formulation is standard in aerospace navigation literature
        - Alternative formulations exist (e.g., Hamilton vs. JPL conventions)

    Example:
        >>> import numpy as np
        >>> omega = np.array([0.1, 0.0, 0.0])  # rotation about x-axis
        >>> Omega = omega_matrix(omega)
        >>> print(Omega.shape)  # (4, 4)
        >>> print(np.allclose(Omega.T, -Omega))  # True (skew-symmetric)

    Related Equations:
        - Eq. (6.2): dq/dt = 0.5 * Ω(ω) * q
        - Eq. (6.3): Ω(ω) matrix definition (THIS FUNCTION)
        - Eq. (6.4): Discrete quaternion update using Ω
    """
    if omega_b.shape != (3,):
        raise ValueError(f"omega_b must have shape (3,), got {omega_b.shape}")

    wx, wy, wz = omega_b

    # Eq. (6.3): Ω(ω) matrix
    Omega = np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ]
    )

    return Omega


def quat_integrate(
    q_prev: np.ndarray,
    omega_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Discrete quaternion integration step.

    Implements Eqs. (6.2)-(6.4) in Chapter 6:
        - Continuous: dq/dt = 0.5 * Ω(ω) * q     (Eq. 6.2)
        - Discrete:   q_k = q_{k-1} + 0.5 * Ω(ω) * q_{k-1} * Δt    (Eq. 6.4)

    This performs first-order Euler integration of quaternion kinematics.
    The output quaternion is automatically normalized to maintain unit norm.

    Args:
        q_prev: Previous quaternion (body to map frame rotation).
                Shape: (4,). Scalar-first: [q0, q1, q2, q3].
                Must be unit quaternion (||q|| = 1).
        omega_b: Angular velocity in body frame B (corrected).
                 Shape: (3,). Units: rad/s.
                 Typically obtained from correct_gyro() in imu_models.py.
        dt: Time step.
            Units: seconds. Typically 0.001 to 0.01 s for IMU integration.

    Returns:
        Updated quaternion q_k at time k.
        Shape: (4,). Normalized to unit norm.

    Notes:
        - Input q_prev should be normalized; if not, integration may diverge.
        - Output is always normalized to prevent numerical drift.
        - First-order Euler is sufficient for small dt (< 0.01 s).
        - For higher accuracy, use RK4 or matrix exponential (not implemented here).
        - Assumes constant ω over interval [t_{k-1}, t_k] (zero-order hold).

    Warnings:
        - Large dt (> 0.1 s) or high angular rates (> 10 rad/s) may introduce errors.
        - Numerical drift accumulates; periodic normalization is essential.

    Example:
        >>> import numpy as np
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        >>> omega = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s yaw rate
        >>> dt = 0.01  # 10 ms
        >>> q1 = quat_integrate(q0, omega, dt)
        >>> print(q1)  # small rotation about z-axis

    Related Equations:
        - Eq. (6.2): Quaternion kinematics (continuous)
        - Eq. (6.3): Ω(ω) matrix (see omega_matrix)
        - Eq. (6.4): Discrete quaternion update (THIS FUNCTION)
        - Eq. (6.6): Gyro correction (applied before calling this function)
    """
    if q_prev.shape != (4,):
        raise ValueError(f"q_prev must have shape (4,), got {q_prev.shape}")
    if omega_b.shape != (3,):
        raise ValueError(f"omega_b must have shape (3,), got {omega_b.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Build Ω(ω) matrix (Eq. 6.3)
    Omega = omega_matrix(omega_b)

    # Discrete quaternion update (Eq. 6.4): q_k = q_{k-1} + 0.5 * Ω * q_{k-1} * Δt
    q_dot = 0.5 * Omega @ q_prev
    q_next = q_prev + q_dot * dt

    # Normalize to maintain unit quaternion constraint
    q_next = q_next / np.linalg.norm(q_next)

    return q_next


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix (body to map frame).

    Converts a unit quaternion q (scalar-first) to the corresponding
    3×3 rotation matrix C_B^M, which transforms vectors from body frame B
    to map frame M:
        v_M = C_B^M @ v_B

    This is a standard aerospace convention consistent with Chapter 2
    coordinate transforms and used in Eq. (6.7) for velocity updates.

    Args:
        q: Unit quaternion (body to map frame rotation).
           Shape: (4,). Scalar-first: [q0, q1, q2, q3].
           Must satisfy ||q|| = 1.

    Returns:
        Rotation matrix C_B^M.
        Shape: (3, 3). Orthogonal: C^T @ C = I, det(C) = 1.

    Notes:
        - Assumes scalar-first quaternion convention.
        - For Chapter 6 integration, this converts body-frame specific force
          to map-frame acceleration (Eq. 6.7).
        - If core/coords has a quat_to_rotmat, use that for consistency.
        - This implementation follows standard aerospace formulas.

    Example:
        >>> import numpy as np
        >>> q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        >>> C = quat_to_rotmat(q_identity)
        >>> print(np.allclose(C, np.eye(3)))  # True

    Related Equations:
        - Chapter 2: Attitude representations
        - Eq. (6.13) (if book defines quaternion rotation matrix there)
        - Eq. (6.7): Velocity update using C_B^M(q)
    """
    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), got {q.shape}")

    # Normalize to be safe
    q = q / np.linalg.norm(q)

    # Scalar-first: [q0, q1, q2, q3]
    q0, q1, q2, q3 = q

    # Rotation matrix (standard formula for scalar-first quaternion)
    C = np.array(
        [
            [
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1**2 + q2**2),
            ],
        ]
    )

    return C


def gravity_vector(g: float = 9.81) -> np.ndarray:
    """
    Gravity vector in map frame (downward).

    Implements Eq. (6.8) in Chapter 6:
        g_M = [0, 0, -g]^T    (for ENU frame)
        g_M = [0, 0, +g]^T    (for NED frame)

    This function assumes ENU (East-North-Up) convention, where gravity
    points downward (negative z-direction). For NED (North-East-Down),
    flip the sign.

    Args:
        g: Gravitational acceleration magnitude.
           Default: 9.81 m/s² (standard gravity).
           Can be adjusted for local gravity variations (±0.05 m/s²).

    Returns:
        Gravity vector in map frame M.
        Shape: (3,). Units: m/s².
        Convention: [0, 0, -g] for ENU, [0, 0, +g] for NED.

    Notes:
        - Standard gravity: g = 9.80665 m/s² (exact)
        - Typical approximation: g = 9.81 m/s²
        - Local variations: ±0.05 m/s² depending on latitude and altitude
        - For indoor positioning, constant g is sufficient
        - Gravity acts downward in ENU (negative z) and downward in NED (positive z)

    Example:
        >>> import numpy as np
        >>> g_enu = gravity_vector(g=9.81)
        >>> print(g_enu)  # [0, 0, -9.81]

    Related Equations:
        - Eq. (6.7): Velocity update (v += (C_B^M @ f + g) * dt)
        - Eq. (6.8): Gravity vector definition (THIS FUNCTION)
    """
    # ENU convention: gravity points downward (negative z)
    g_map = np.array([0.0, 0.0, -g])
    return g_map


def vel_update(
    v_prev: np.ndarray,
    q_prev: np.ndarray,
    f_b: np.ndarray,
    dt: float,
    g: float = 9.81,
) -> np.ndarray:
    """
    Velocity update with gravity compensation.

    Implements Eq. (6.7) in Chapter 6:
        v_k^M = v_{k-1}^M + (C_B^M(q) @ f_b + g_M) * Δt

    where:
        v^M: velocity in map frame M [m/s]
        C_B^M(q): rotation matrix from body to map (from quaternion q)
        f_b: specific force in body frame B (corrected accel) [m/s²]
        g_M: gravity vector in map frame [0, 0, -9.81] for ENU [m/s²]
        Δt: time step [s]

    This equation integrates the corrected accelerometer measurement (specific force)
    in the map frame, accounting for gravity, to update velocity.

    Args:
        v_prev: Previous velocity in map frame M.
                Shape: (3,). Units: m/s. Components: [v_E, v_N, v_U] for ENU.
        q_prev: Previous quaternion (body to map frame rotation).
                Shape: (4,). Scalar-first: [q0, q1, q2, q3].
        f_b: Specific force in body frame B (corrected accel).
             Shape: (3,). Units: m/s². Obtained from correct_accel().
             This is the accelerometer measurement after bias/noise removal.
        dt: Time step.
            Units: seconds. Typically 0.001 to 0.01 s.
        g: Gravitational acceleration magnitude (optional).
           Default: 9.81 m/s².

    Returns:
        Updated velocity v_k in map frame M.
        Shape: (3,). Units: m/s.

    Notes:
        - Specific force f_b includes all non-gravitational accelerations.
        - Gravity g_M is added in map frame (Eq. 6.8).
        - Rotation C_B^M converts body-frame f to map frame.
        - First-order Euler integration: adequate for small dt.
        - Assumes constant f_b over interval [t_{k-1}, t_k].

    Example:
        >>> import numpy as np
        >>> v0 = np.zeros(3)  # stationary
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # identity (body = map)
        >>> f_b = np.array([1.0, 0.0, 0.0])  # 1 m/s² accel in x (body)
        >>> dt = 0.01
        >>> v1 = vel_update(v0, q, f_b, dt)
        >>> print(v1)  # ≈ [0.01, 0, -0.0981] (accel + gravity effect)

    Related Equations:
        - Eq. (6.7): Velocity update (THIS FUNCTION)
        - Eq. (6.8): Gravity vector
        - Eq. (6.9): Accel correction (applied before calling this function)
        - Eq. (6.10): Position update (uses output velocity)
    """
    if v_prev.shape != (3,):
        raise ValueError(f"v_prev must have shape (3,), got {v_prev.shape}")
    if q_prev.shape != (4,):
        raise ValueError(f"q_prev must have shape (4,), got {q_prev.shape}")
    if f_b.shape != (3,):
        raise ValueError(f"f_b must have shape (3,), got {f_b.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Convert quaternion to rotation matrix C_B^M
    C_B_M = quat_to_rotmat(q_prev)

    # Gravity vector in map frame (Eq. 6.8)
    g_M = gravity_vector(g)

    # Acceleration in map frame: a_M = C_B^M @ f_b + g_M
    a_M = C_B_M @ f_b + g_M

    # Velocity update (Eq. 6.7): v_k = v_{k-1} + a_M * Δt
    v_next = v_prev + a_M * dt

    return v_next


def pos_update(
    p_prev: np.ndarray,
    v_current: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Position update using current velocity.

    Implements Eq. (6.10) in Chapter 6:
        p_k^M = p_{k-1}^M + v_k^M * Δt

    where:
        p^M: position in map frame M [m]
        v^M: velocity in map frame M [m/s]
        Δt: time step [s]

    This is simple Euler integration of velocity to obtain position.
    First-order accuracy is sufficient for typical IMU rates (100-1000 Hz).

    Args:
        p_prev: Previous position in map frame M.
                Shape: (3,). Units: m. Components: [x, y, z] or [E, N, U] for ENU.
        v_current: Current velocity in map frame M (from vel_update).
                   Shape: (3,). Units: m/s.
        dt: Time step.
            Units: seconds. Typically 0.001 to 0.01 s.

    Returns:
        Updated position p_k in map frame M.
        Shape: (3,). Units: m.

    Notes:
        - Uses current velocity (v_k) not previous velocity (v_{k-1}).
        - This is consistent with standard strapdown integration order:
            1. Update attitude (quat_integrate)
            2. Update velocity (vel_update)
            3. Update position (pos_update)
        - For higher accuracy, use trapezoidal rule: p += 0.5*(v_prev + v_current)*dt
        - Assumes constant velocity over interval [t_{k-1}, t_k].

    Example:
        >>> import numpy as np
        >>> p0 = np.array([0.0, 0.0, 0.0])  # origin
        >>> v = np.array([1.0, 0.0, 0.0])  # 1 m/s eastward
        >>> dt = 0.01
        >>> p1 = pos_update(p0, v, dt)
        >>> print(p1)  # [0.01, 0, 0]

    Related Equations:
        - Eq. (6.10): Position update (THIS FUNCTION)
        - Eq. (6.7): Velocity update (provides v_current)
    """
    if p_prev.shape != (3,):
        raise ValueError(f"p_prev must have shape (3,), got {p_prev.shape}")
    if v_current.shape != (3,):
        raise ValueError(f"v_current must have shape (3,), got {v_current.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Position update (Eq. 6.10): p_k = p_{k-1} + v_k * Δt
    p_next = p_prev + v_current * dt

    return p_next


def strapdown_update(
    q: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    omega_b: np.ndarray,
    f_b: np.ndarray,
    dt: float,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete strapdown integration step (attitude, velocity, position).

    Convenience function that performs one full strapdown update:
        1. Quaternion integration (Eqs. 6.2-6.4)
        2. Velocity update (Eq. 6.7)
        3. Position update (Eq. 6.10)

    This is the core loop of IMU-based dead reckoning.

    Args:
        q: Current quaternion (body to map frame).
           Shape: (4,). Scalar-first: [q0, q1, q2, q3].
        v: Current velocity in map frame M.
           Shape: (3,). Units: m/s.
        p: Current position in map frame M.
           Shape: (3,). Units: m.
        omega_b: Angular velocity in body frame B (corrected).
                 Shape: (3,). Units: rad/s.
        f_b: Specific force in body frame B (corrected accel).
             Shape: (3,). Units: m/s².
        dt: Time step.
            Units: seconds.
        g: Gravitational acceleration magnitude (optional).
           Default: 9.81 m/s².

    Returns:
        Tuple (q_next, v_next, p_next):
            q_next: Updated quaternion, shape (4,)
            v_next: Updated velocity, shape (3,), units: m/s
            p_next: Updated position, shape (3,), units: m

    Notes:
        - Input omega_b and f_b should be corrected (bias removed).
        - Update order matters: q → v → p (standard mechanization).
        - This function encapsulates the core strapdown loop.
        - For EKF-based INS, use this as the process model f(x, u, dt).

    Example:
        >>> import numpy as np
        >>> # Initial state: origin, stationary, level attitude
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])
        >>> v0 = np.zeros(3)
        >>> p0 = np.zeros(3)
        >>> # IMU measurements (corrected)
        >>> omega = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s yaw rate
        >>> f = np.array([1.0, 0.0, 0.0])  # 1 m/s² forward accel
        >>> dt = 0.01
        >>> q1, v1, p1 = strapdown_update(q0, v0, p0, omega, f, dt)

    Related Equations:
        - Eqs. (6.2)-(6.4): Quaternion integration
        - Eq. (6.7): Velocity update
        - Eq. (6.10): Position update
    """
    # Step 1: Update attitude (quaternion)
    q_next = quat_integrate(q, omega_b, dt)

    # Step 2: Update velocity
    v_next = vel_update(v, q, f_b, dt, g)

    # Step 3: Update position
    p_next = pos_update(p, v_next, dt)

    return q_next, v_next, p_next

