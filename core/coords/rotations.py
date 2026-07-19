"""Rotation representations and conversions (Chapter 2, Section 2.2).

Attitude conversions between the three representations used in the book:
- Rotation matrices (3x3 orthogonal matrices, SO(3))
- Quaternions (unit quaternions, q = [q0, q1, q2, q3], scalar first)
- Euler angles (roll, pitch, yaw)

Conventions (following *Principles of Indoor Positioning and Indoor
Navigation*, Section 2.2, in an X-right / Y-forward / Z-up body frame):

- Euler angles [roll, pitch, yaw] in radians, where
  - roll  = phi about the **Y-axis**   (book Eq. (2.15))
  - pitch = theta about the **X-axis**  (book Eq. (2.16))
  - yaw   = psi about the **Z-axis**    (book Eq. (2.14))
  Note this is the book's assignment (roll->Y, pitch->X); it differs from the
  common aerospace convention (roll->X, pitch->Y). It is deliberate so the code
  reproduces the book's equations exactly.
- Rotation matrix C is the **coordinate transform** from the old frame to the
  new frame: ``x_new = C @ x_old`` (book Eqs. (2.11), (2.17)). This is the
  passive form; the active (vector-rotating) matrix is its transpose.
- Quaternions: [q0, q1, q2, q3] with q0 the scalar part (book Eq. (2.20)).
- Euler composition order: C = Rx(pitch) @ Ry(roll) @ Rz(yaw) (book Eq. (2.17)).

Reference: Chapter 2, Section 2.2 - Attitude: Definition and Representation.
"""

import numpy as np
from numpy.typing import NDArray


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Convert Euler angles to a rotation (coordinate-transform) matrix.

    Builds the book's rotation matrix ``C`` such that ``x_new = C @ x_old``,
    composed as ``C = Rx(pitch) @ Ry(roll) @ Rz(yaw)`` following the yaw ->
    roll -> pitch axis sequence (about Z, Y, X) described in the text.

    Args:
        roll: Roll angle phi in radians (rotation about the Y-axis).
        pitch: Pitch angle theta in radians (rotation about the X-axis).
        yaw: Yaw angle psi in radians (rotation about the Z-axis).

    Returns:
        3x3 coordinate-transform matrix C with ``x_new = C @ x_old``.

    Example:
        >>> import numpy as np
        >>> C = euler_to_rotation_matrix(0.1, 0.2, 0.3)
        >>> float(np.linalg.det(C))  # doctest: +ELLIPSIS
        1.0...

    Reference:
        Chapter 2, Eq. (2.17) - rotation matrix from Euler angles, composed
        from the per-axis rotations of Section 2.2.
    """
    phi, theta, psi = roll, pitch, yaw
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)

    # C = Rx(pitch) @ Ry(roll) @ Rz(yaw), expanded per Eq. (2.17).
    C = np.array(
        [
            [cf * cy, cf * sy, -sf],
            [-ct * sy + st * sf * cy, ct * cy + st * sf * sy, st * cf],
            [st * sy + ct * sf * cy, -st * cy + ct * sf * sy, ct * cf],
        ],
        dtype=np.float64,
    )

    return C


def rotation_matrix_to_euler(C: NDArray[np.float64]) -> NDArray[np.float64]:
    """Recover Euler angles from a rotation (coordinate-transform) matrix.

    Inverts Eq. (2.17). Handles gimbal lock, which for this convention occurs
    when the roll angle phi approaches +/-90 deg (i.e. ``|C[0, 2]| -> 1``),
    where yaw and pitch become coupled and 1 DOF is lost.

    Args:
        C: 3x3 rotation matrix (orthogonal, SO(3)) with ``x_new = C @ x_old``.

    Returns:
        Euler angles as numpy array [roll, pitch, yaw] in radians.

    Raises:
        ValueError: If C is not a 3x3 matrix.

    Reference:
        Chapter 2, Eq. (2.17) - inverse extraction of Euler angles.
    """
    if C.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {C.shape}")

    sin_roll = -C[0, 2]

    if abs(sin_roll) >= 1.0 - 1e-12:
        # Gimbal lock: roll = +/-90 deg. Pitch and yaw are coupled; fix yaw = 0
        # by convention and solve pitch so the recovered angles reconstruct C.
        # The coupling sign depends on which roll branch (+90 vs -90) we are in.
        roll = np.copysign(np.pi / 2.0, sin_roll)
        yaw = 0.0
        pitch = np.arctan2(np.copysign(1.0, sin_roll) * C[1, 0], C[1, 1])
    else:
        roll = np.arcsin(np.clip(sin_roll, -1.0, 1.0))  # about Y
        yaw = np.arctan2(C[0, 1], C[0, 0])  # about Z
        pitch = np.arctan2(C[1, 2], C[2, 2])  # about X

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quat(
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Convert Euler angles to a unit quaternion.

    Args:
        roll: Roll angle phi in radians (rotation about the Y-axis).
        pitch: Pitch angle theta in radians (rotation about the X-axis).
        yaw: Yaw angle psi in radians (rotation about the Z-axis).

    Returns:
        Unit quaternion as numpy array [q0, q1, q2, q3] (scalar first).

    Example:
        >>> import numpy as np
        >>> q = euler_to_quat(0.0, 0.0, np.pi / 2)  # 90 deg yaw
        >>> float(np.linalg.norm(q))
        1.0

    Reference:
        Chapter 2, Eq. (2.23) - Euler angles to quaternion.
    """
    theta, phi, psi = pitch, roll, yaw
    ct, st = np.cos(theta / 2.0), np.sin(theta / 2.0)
    cf, sf = np.cos(phi / 2.0), np.sin(phi / 2.0)
    cy, sy = np.cos(psi / 2.0), np.sin(psi / 2.0)

    q0 = ct * cf * cy + st * sf * sy
    q1 = st * cf * cy - ct * sf * sy
    q2 = ct * sf * cy + st * cf * sy
    q3 = ct * cf * sy - st * sf * cy

    return np.array([q0, q1, q2, q3], dtype=np.float64)


def quat_to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a unit quaternion to Euler angles.

    The roll (about Y) extraction uses ``arcsin`` and is clamped to [-1, 1] for
    numerical safety; it degrades near the roll = +/-90 deg gimbal lock.

    Args:
        q: Unit quaternion as numpy array [q0, q1, q2, q3] (scalar first).

    Returns:
        Euler angles as numpy array [roll, pitch, yaw] in radians.

    Raises:
        ValueError: If q is not a 4-element array.

    Reference:
        Chapter 2, Eq. (2.22) - quaternion to Euler angles.
    """
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")

    q0, q1, q2, q3 = q

    pitch = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    roll = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a unit quaternion to a rotation (coordinate-transform) matrix.

    Returns the passive matrix C with ``x_new = C @ x_old`` consistent with
    :func:`euler_to_rotation_matrix`.

    Args:
        q: Unit quaternion as numpy array [q0, q1, q2, q3] (scalar first).

    Returns:
        3x3 coordinate-transform matrix C with ``x_new = C @ x_old``.

    Raises:
        ValueError: If q is not a 4-element array.

    Reference:
        Chapter 2, Eq. (2.21) - quaternion to rotation matrix.
    """
    if q.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")

    q0, q1, q2, q3 = q

    C = np.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 + q0 * q3),
                2.0 * (q1 * q3 - q0 * q2),
            ],
            [
                2.0 * (q1 * q2 - q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 + q0 * q1),
            ],
            [
                2.0 * (q1 * q3 + q0 * q2),
                2.0 * (q2 * q3 - q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=np.float64,
    )

    return C


def rotation_matrix_to_quat(C: NDArray[np.float64]) -> NDArray[np.float64]:
    """Recover a unit quaternion from a rotation (coordinate-transform) matrix.

    Uses Shepperd's method (selecting the largest pivot for numerical
    stability) to invert Eq. (2.21). The book does not give an explicit
    matrix-to-quaternion equation, so this is the inverse of Eq. (2.21). The
    result is returned in the canonical hemisphere (q0 >= 0) to remove the
    quaternion double-cover sign ambiguity.

    Args:
        C: 3x3 rotation matrix (orthogonal, SO(3)) with ``x_new = C @ x_old``.

    Returns:
        Unit quaternion as numpy array [q0, q1, q2, q3] (scalar first).

    Raises:
        ValueError: If C is not a 3x3 matrix.

    Reference:
        Chapter 2, Eq. (2.21) - inverse (matrix to quaternion, Shepperd).
    """
    if C.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {C.shape}")

    trace = np.trace(C)

    if trace > 0.0:
        q0 = 0.5 * np.sqrt(1.0 + trace)
        f = 0.25 / q0
        q1 = (C[1, 2] - C[2, 1]) * f
        q2 = (C[2, 0] - C[0, 2]) * f
        q3 = (C[0, 1] - C[1, 0]) * f
    elif C[0, 0] >= C[1, 1] and C[0, 0] >= C[2, 2]:
        q1 = 0.5 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        f = 0.25 / q1
        q0 = (C[1, 2] - C[2, 1]) * f
        q2 = (C[0, 1] + C[1, 0]) * f
        q3 = (C[2, 0] + C[0, 2]) * f
    elif C[1, 1] >= C[2, 2]:
        q2 = 0.5 * np.sqrt(1.0 - C[0, 0] + C[1, 1] - C[2, 2])
        f = 0.25 / q2
        q0 = (C[2, 0] - C[0, 2]) * f
        q1 = (C[0, 1] + C[1, 0]) * f
        q3 = (C[1, 2] + C[2, 1]) * f
    else:
        q3 = 0.5 * np.sqrt(1.0 - C[0, 0] - C[1, 1] + C[2, 2])
        f = 0.25 / q3
        q0 = (C[0, 1] - C[1, 0]) * f
        q1 = (C[2, 0] + C[0, 2]) * f
        q2 = (C[1, 2] + C[2, 1]) * f

    q = np.array([q0, q1, q2, q3], dtype=np.float64)
    if q[0] < 0.0:
        q = -q  # canonical hemisphere: q0 >= 0
    return q / np.linalg.norm(q)
