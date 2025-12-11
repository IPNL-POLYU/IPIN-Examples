"""Rotation representations and conversions.

This module provides functions to convert between different rotation
representations commonly used in navigation:
- Rotation matrices (3x3 orthogonal matrices, SO(3))
- Quaternions (unit quaternions, q = [qw, qx, qy, qz])
- Euler angles (roll-pitch-yaw, ZYX convention)

Conventions:
- Quaternions: [qw, qx, qy, qz] where qw is the scalar part
- Euler angles: [roll, pitch, yaw] in radians (ZYX/3-2-1 convention)
  - Roll: rotation about x-axis (φ)
  - Pitch: rotation about y-axis (θ)
  - Yaw: rotation about z-axis (ψ)
- Rotation matrices: 3x3 numpy arrays

Reference: Chapter 2, Section 2.4 - Rotation Representations
"""

import numpy as np
from numpy.typing import NDArray


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Convert Euler angles to rotation matrix.

    Converts roll-pitch-yaw Euler angles (ZYX convention) to a 3x3
    rotation matrix that transforms vectors from body frame to
    navigation frame.

    Args:
        roll: Roll angle φ in radians (rotation about x-axis).
        pitch: Pitch angle θ in radians (rotation about y-axis).
        yaw: Yaw angle ψ in radians (rotation about z-axis).

    Returns:
        3x3 rotation matrix R such that v_nav = R @ v_body.

    Example:
        >>> import numpy as np
        >>> R = euler_to_rotation_matrix(0.1, 0.2, 0.3)
        >>> print(f"Rotation matrix shape: {R.shape}")
        >>> print(f"Determinant (should be 1.0): {np.linalg.det(R):.6f}")

    Reference:
        Chapter 2, Eq. (2.5) - Euler to rotation matrix (ZYX convention)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # ZYX (3-2-1) Euler angle rotation matrix
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )

    return R


def rotation_matrix_to_euler(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert rotation matrix to Euler angles.

    Extracts roll-pitch-yaw Euler angles (ZYX convention) from a 3x3
    rotation matrix. Handles gimbal lock when pitch is near ±90°.

    Args:
        R: 3x3 rotation matrix (orthogonal matrix in SO(3)).

    Returns:
        Euler angles as numpy array [roll, pitch, yaw] in radians.

    Raises:
        ValueError: If R is not a 3x3 matrix.

    Example:
        >>> R = np.eye(3)  # Identity rotation
        >>> euler = rotation_matrix_to_euler(R)
        >>> print(f"Euler angles: {euler}")  # Should be [0, 0, 0]

    Reference:
        Chapter 2, Eq. (2.6) - Rotation matrix to Euler angles
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")

    # Extract pitch
    sin_pitch = -R[2, 0]

    # Handle gimbal lock
    if abs(sin_pitch) >= 1.0:
        # Gimbal lock: pitch = ±90°
        pitch = np.copysign(np.pi / 2.0, sin_pitch)
        # In gimbal lock, only yaw - roll (or yaw + roll) is observable
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0  # Set roll to zero by convention
    else:
        pitch = np.arcsin(sin_pitch)
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quat(
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Convert Euler angles to quaternion.

    Converts roll-pitch-yaw Euler angles (ZYX convention) to a unit
    quaternion representation.

    Args:
        roll: Roll angle φ in radians (rotation about x-axis).
        pitch: Pitch angle θ in radians (rotation about y-axis).
        yaw: Yaw angle ψ in radians (rotation about z-axis).

    Returns:
        Unit quaternion as numpy array [qw, qx, qy, qz].

    Example:
        >>> q = euler_to_quat(0.0, 0.0, np.pi/2)  # 90° yaw
        >>> print(f"Quaternion: {q}")
        >>> print(f"Norm (should be 1.0): {np.linalg.norm(q):.6f}")

    Reference:
        Chapter 2, Eq. (2.7) - Euler to quaternion
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    # Quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz], dtype=np.float64)


def quat_to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to Euler angles.

    Extracts roll-pitch-yaw Euler angles (ZYX convention) from a
    unit quaternion. Handles gimbal lock when pitch is near ±90°.

    Args:
        q: Unit quaternion as numpy array [qw, qx, qy, qz].

    Returns:
        Euler angles as numpy array [roll, pitch, yaw] in radians.

    Raises:
        ValueError: If q is not a 4-element array.

    Example:
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        >>> euler = quat_to_euler(q)
        >>> print(f"Euler angles: {euler}")  # Should be [0, 0, 0]

    Reference:
        Chapter 2, Eq. (2.8) - Quaternion to Euler angles
    """
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")

    qw, qx, qy, qz = q

    # Extract roll
    sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
    cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    # Extract pitch
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    # Clamp to avoid numerical issues with arcsin
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)

    # Extract yaw
    sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to rotation matrix.

    Converts a unit quaternion to a 3x3 rotation matrix.

    Args:
        q: Unit quaternion as numpy array [qw, qx, qy, qz].

    Returns:
        3x3 rotation matrix R such that v_nav = R @ v_body.

    Raises:
        ValueError: If q is not a 4-element array.

    Example:
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        >>> R = quat_to_rotation_matrix(q)
        >>> print(f"Rotation matrix:\\n{R}")  # Should be identity

    Reference:
        Chapter 2, Eq. (2.9) - Quaternion to rotation matrix
    """
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")

    qw, qx, qy, qz = q

    # Rotation matrix from quaternion
    R = np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )

    return R


def rotation_matrix_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert rotation matrix to quaternion.

    Extracts a unit quaternion from a 3x3 rotation matrix using
    Shepperd's method for numerical stability.

    Args:
        R: 3x3 rotation matrix (orthogonal matrix in SO(3)).

    Returns:
        Unit quaternion as numpy array [qw, qx, qy, qz].

    Raises:
        ValueError: If R is not a 3x3 matrix.

    Example:
        >>> R = np.eye(3)  # Identity rotation
        >>> q = rotation_matrix_to_quat(R)
        >>> print(f"Quaternion: {q}")  # Should be [1, 0, 0, 0]

    Reference:
        Chapter 2, Eq. (2.10) - Rotation matrix to quaternion (Shepperd)
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")

    # Shepperd's method: choose largest diagonal element for stability
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)

    # Normalize to ensure unit quaternion
    q = q / np.linalg.norm(q)

    return q
