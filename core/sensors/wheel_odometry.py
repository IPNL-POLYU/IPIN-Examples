"""
Wheel odometry dead reckoning for vehicles (Chapter 6).

This module implements wheel-based dead reckoning algorithms:
    - Skew-symmetric matrix for cross products (Eq. (6.12))
    - Wheel speed to attitude velocity with lever arm (Eq. (6.11))
    - Attitude to map frame velocity transform (Eq. (6.14))
    - Position update for odometry (Eq. (6.15))

These functions support vehicle-based navigation where wheel speed sensors
provide velocity measurements that must be integrated with IMU data.

Frame Conventions:
    - S: Speed frame (wheel odometry measurement frame)
    - A: Attitude frame (vehicle body frame for velocity)
    - B: Body frame (IMU/sensor frame)
    - M: Map frame (navigation frame, typically ENU)

References:
    Chapter 6, Section 6.2: Wheel odometry dead reckoning
    Eq. (6.11): Lever arm compensation for wheel speed
    Eq. (6.12): Skew-symmetric matrix [v×]
    Eq. (6.14): Attitude to map velocity transform
    Eq. (6.15): Position update from wheel odometry
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    Compute skew-symmetric matrix [v×] for cross product.

    Implements Eq. (6.12) in Chapter 6:

        [v×] = [  0   -vz    vy ]
               [ vz    0   -vx ]
               [-vy   vx    0  ]

    The skew-symmetric matrix allows cross products to be expressed as
    matrix-vector multiplication:
        v × w = [v×] @ w

    This is used in lever arm compensation (Eq. 6.11) to account for the
    offset between wheel speed measurement point and IMU/navigation center.

    Args:
        v: 3D vector.
           Shape: (3,). Any units (typically m or m/s).
           Components: [vx, vy, vz].

    Returns:
        Skew-symmetric matrix [v×].
        Shape: (3, 3). Satisfies [v×]^T = -[v×] and trace([v×]) = 0.

    Notes:
        - Skew-symmetric: S^T = -S
        - Trace(S) = 0
        - Cross product: v × w = skew(v) @ w
        - Useful for: lever arm corrections, angular velocity cross products
        - Standard aerospace/robotics formulation

    Example:
        >>> import numpy as np
        >>> v = np.array([1.0, 2.0, 3.0])
        >>> S = skew(v)
        >>> w = np.array([0.5, -0.3, 0.2])
        >>> cross_product = S @ w
        >>> # Verify: same as np.cross(v, w)
        >>> np.allclose(cross_product, np.cross(v, w))
        True

    Related Equations:
        - Eq. (6.11): Lever arm compensation using [ω×]
        - Eq. (6.12): Skew-symmetric matrix definition (THIS FUNCTION)
    """
    if v.shape != (3,):
        raise ValueError(f"v must have shape (3,), got {v.shape}")

    vx, vy, vz = v

    # Eq. (6.12): skew-symmetric matrix
    S = np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]])

    return S


def wheel_speed_to_attitude_velocity(
    v_s: np.ndarray,
    omega_b: np.ndarray,
    lever_arm_b: np.ndarray,
) -> np.ndarray:
    """
    Convert wheel speed to attitude frame velocity with lever arm compensation.

    Implements Eq. (6.11) in Chapter 6:
        v^A = v^S - [ω_B ×] l^B

    where:
        v^S: velocity in speed frame S (wheel measurement) [m/s]
        ω_B: angular velocity in body frame B [rad/s]
        l^B: lever arm from IMU to wheel center in body frame [m]
        v^A: velocity in attitude frame A [m/s]
        [ω×]: skew-symmetric matrix (Eq. 6.12)

    The lever arm correction accounts for the fact that the wheel speed sensor
    measures velocity at a different point than the IMU/navigation center.
    The rotational motion (ω_B) causes an additional velocity component at
    the wheel location.

    Args:
        v_s: Velocity measurement in speed frame S.
             Shape: (3,). Units: m/s.
             Typically v_s = [v_forward, 0, 0] for forward vehicle motion.
        omega_b: Angular velocity in body frame B (from corrected gyro).
                 Shape: (3,). Units: rad/s.
                 Typically from IMU gyroscope after bias correction.
        lever_arm_b: Lever arm vector from IMU to wheel center in body frame B.
                     Shape: (3,). Units: m.
                     Example: [0.5, 0, -0.2] means wheel is 0.5m forward,
                     0.2m below IMU.

    Returns:
        Velocity in attitude frame A.
        Shape: (3,). Units: m/s.

    Notes:
        - Speed frame S is typically aligned with vehicle forward direction.
        - Lever arm is positive in direction from IMU to wheel center.
        - For zero lever arm (l = 0), v^A = v^S (no correction needed).
        - Sign convention: cross product [ω×] l gives velocity due to rotation.
        - This correction is crucial for accurate vehicle navigation.

    Example:
        >>> import numpy as np
        >>> # Vehicle moving forward at 5 m/s
        >>> v_s = np.array([5.0, 0.0, 0.0])
        >>> # Turning right (positive yaw rate)
        >>> omega_b = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s
        >>> # Wheel is 0.5 m forward of IMU
        >>> lever_arm = np.array([0.5, 0.0, 0.0])
        >>> v_a = wheel_speed_to_attitude_velocity(v_s, omega_b, lever_arm)
        >>> print(v_a)  # Includes correction for rotation

    Related Equations:
        - Eq. (6.11): Lever arm compensation (THIS FUNCTION)
        - Eq. (6.12): Skew-symmetric matrix [ω×]
        - Eq. (6.14): Attitude to map frame transform (next step)
    """
    if v_s.shape != (3,):
        raise ValueError(f"v_s must have shape (3,), got {v_s.shape}")
    if omega_b.shape != (3,):
        raise ValueError(f"omega_b must have shape (3,), got {omega_b.shape}")
    if lever_arm_b.shape != (3,):
        raise ValueError(f"lever_arm_b must have shape (3,), got {lever_arm_b.shape}")

    # Skew-symmetric matrix [ω×] (Eq. 6.12)
    omega_skew = skew(omega_b)

    # Lever arm compensation (Eq. 6.11): v^A = v^S - [ω×] l
    v_a = v_s - omega_skew @ lever_arm_b

    return v_a


def attitude_to_map_velocity(v_a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Transform velocity from attitude frame to map frame.

    Implements Eq. (6.14) in Chapter 6:
        v^M = C_A^M(q) @ v^A

    where:
        v^A: velocity in attitude frame A [m/s]
        C_A^M(q): rotation matrix from attitude to map (from quaternion q)
        v^M: velocity in map frame M [m/s]

    This transformation converts the vehicle velocity (after lever arm correction)
    from the attitude frame to the navigation/map frame for position integration.

    Args:
        v_a: Velocity in attitude frame A.
             Shape: (3,). Units: m/s.
             Typically obtained from wheel_speed_to_attitude_velocity().
        q: Quaternion representing rotation from attitude to map frame.
           Shape: (4,). Scalar-first: [q0, q1, q2, q3].
           Must be unit quaternion (||q|| = 1).

    Returns:
        Velocity in map frame M.
        Shape: (3,). Units: m/s. Components: [v_E, v_N, v_U] for ENU.

    Notes:
        - Assumes attitude frame A and body frame B are aligned (or close).
        - Quaternion q represents vehicle orientation relative to map.
        - For level vehicle (q = identity), v^M = v^A.
        - Map frame M is typically ENU (East-North-Up) or NED.
        - Uses quat_to_rotmat from strapdown module.

    Example:
        >>> import numpy as np
        >>> from core.sensors.strapdown import quat_to_rotmat
        >>> # Forward velocity in attitude frame
        >>> v_a = np.array([5.0, 0.0, 0.0])
        >>> # Identity quaternion (level vehicle, heading east)
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])
        >>> v_m = attitude_to_map_velocity(v_a, q)
        >>> print(v_m)  # [5.0, 0.0, 0.0] (eastward velocity)

    Related Equations:
        - Eq. (6.11): Lever arm compensation (previous step)
        - Eq. (6.14): Attitude to map velocity (THIS FUNCTION)
        - Eq. (6.15): Position update (next step)
    """
    if v_a.shape != (3,):
        raise ValueError(f"v_a must have shape (3,), got {v_a.shape}")
    if q.shape != (4,):
        raise ValueError(f"q must have shape (4,), got {q.shape}")

    # Import here to avoid circular dependency
    from core.sensors.strapdown import quat_to_rotmat

    # Convert quaternion to rotation matrix C_A^M (Eq. 6.14)
    C_A_M = quat_to_rotmat(q)

    # Transform velocity to map frame: v^M = C_A^M @ v^A
    v_m = C_A_M @ v_a

    return v_m


def odom_pos_update(
    p_prev: np.ndarray,
    v_map: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Update position using wheel odometry velocity.

    Implements Eq. (6.15) in Chapter 6:
        p_k^M = p_{k-1}^M + v_k^M * Δt

    where:
        p^M: position in map frame M [m]
        v^M: velocity in map frame M (from wheel odometry) [m/s]
        Δt: time step [s]

    This is identical to pos_update from strapdown.py, but provided here
    for completeness and to emphasize the wheel odometry context.

    Args:
        p_prev: Previous position in map frame M.
                Shape: (3,). Units: m. Components: [x, y, z] or [E, N, U].
        v_map: Current velocity in map frame M (from attitude_to_map_velocity).
               Shape: (3,). Units: m/s.
        dt: Time step.
            Units: seconds. Typically 0.01 to 0.1 s for wheel odometry.

    Returns:
        Updated position p_k in map frame M.
        Shape: (3,). Units: m.

    Notes:
        - Simple Euler integration of velocity.
        - Assumes constant velocity over interval [t_{k-1}, t_k].
        - For higher accuracy, use trapezoidal rule.
        - This is the same as strapdown pos_update but in odometry context.
        - Wheel odometry typically has lower update rate than IMU (10-100 Hz vs 100-1000 Hz).

    Example:
        >>> import numpy as np
        >>> p0 = np.array([0.0, 0.0, 0.0])
        >>> v = np.array([2.0, 0.0, 0.0])  # 2 m/s eastward
        >>> dt = 0.1  # 100 ms
        >>> p1 = odom_pos_update(p0, v, dt)
        >>> print(p1)  # [0.2, 0.0, 0.0] (moved 0.2m east)

    Related Equations:
        - Eq. (6.11): Lever arm compensation
        - Eq. (6.14): Attitude to map velocity
        - Eq. (6.15): Position update (THIS FUNCTION)
        - Eq. (6.10): Equivalent position update in strapdown
    """
    if p_prev.shape != (3,):
        raise ValueError(f"p_prev must have shape (3,), got {p_prev.shape}")
    if v_map.shape != (3,):
        raise ValueError(f"v_map must have shape (3,), got {v_map.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Position update (Eq. 6.15): p_k = p_{k-1} + v^M * Δt
    p_next = p_prev + v_map * dt

    return p_next


def wheel_odom_update(
    p: np.ndarray,
    q: np.ndarray,
    v_s: np.ndarray,
    omega_b: np.ndarray,
    lever_arm_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Complete wheel odometry position update (convenience function).

    Combines Eqs. (6.11), (6.14), and (6.15) into a single function:
        1. Lever arm compensation: v^A = v^S - [ω×] l (Eq. 6.11)
        2. Attitude to map transform: v^M = C_A^M @ v^A (Eq. 6.14)
        3. Position update: p_k = p_{k-1} + v^M * Δt (Eq. 6.15)

    This is the core loop of wheel-based dead reckoning, typically fused
    with IMU in an integrated navigation system (Chapter 6, Section 6.2).

    Args:
        p: Current position in map frame M.
           Shape: (3,). Units: m.
        q: Current quaternion (attitude to map frame rotation).
           Shape: (4,). Scalar-first: [q0, q1, q2, q3].
        v_s: Wheel speed measurement in speed frame S.
             Shape: (3,). Units: m/s.
        omega_b: Angular velocity in body frame B (from IMU gyro).
                 Shape: (3,). Units: rad/s.
        lever_arm_b: Lever arm from IMU to wheel center in body frame B.
                     Shape: (3,). Units: m.
        dt: Time step.
            Units: seconds.

    Returns:
        Updated position p_k in map frame M.
        Shape: (3,). Units: m.

    Notes:
        - Assumes attitude q is already updated (e.g., via strapdown integration).
        - Does not update velocity or attitude (position-only update).
        - For full IMU+wheel EKF, use ins_ekf_models.py (future implementation).
        - Wheel slip or bias causes drift; use constraints (ZUPT/NHC) to correct.

    Example:
        >>> import numpy as np
        >>> p0 = np.zeros(3)
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # level
        >>> v_s = np.array([5.0, 0.0, 0.0])  # 5 m/s forward
        >>> omega = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s yaw
        >>> lever_arm = np.array([0.5, 0.0, 0.0])
        >>> dt = 0.1
        >>> p1 = wheel_odom_update(p0, q, v_s, omega, lever_arm, dt)

    Related Equations:
        - Eq. (6.11): Lever arm compensation
        - Eq. (6.14): Attitude to map velocity
        - Eq. (6.15): Position update
    """
    # Step 1: Lever arm compensation (Eq. 6.11)
    v_a = wheel_speed_to_attitude_velocity(v_s, omega_b, lever_arm_b)

    # Step 2: Attitude to map transform (Eq. 6.14)
    v_m = attitude_to_map_velocity(v_a, q)

    # Step 3: Position update (Eq. 6.15)
    p_next = odom_pos_update(p, v_m, dt)

    return p_next


