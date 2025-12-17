"""SE(2) operations for 2D SLAM (Special Euclidean Group in 2D).

This module implements fundamental operations for SE(2), the group of
rigid transformations in 2D (rotation + translation). These operations
are used throughout Chapter 7 for scan matching, pose graph optimization,
and trajectory representation.

Key functions:
    - se2_compose: Compose two SE(2) poses (p1 ⊕ p2)
    - se2_inverse: Invert an SE(2) pose (p⁻¹)
    - se2_apply: Transform points by an SE(2) pose
    - wrap_angle: Normalize angle to [-π, π]

SE(2) representation: poses are NumPy arrays [x, y, yaw] of shape (3,).

Author: Navigation Engineer
Date: 2024
"""

from typing import Union

import numpy as np

from .types import Pose2


def wrap_angle(theta: float) -> float:
    """
    Normalize angle to the range [-π, π].

    Wraps an angle (in radians) to the principal value in [-π, π].
    This is essential for SE(2) operations to avoid angle accumulation
    and ensure consistent representations.

    Args:
        theta: Angle in radians (can be any real value).

    Returns:
        Normalized angle in [-π, π].

    Examples:
        >>> wrap_angle(0.0)
        0.0
        >>> wrap_angle(np.pi)
        3.141592653589793
        >>> wrap_angle(3 * np.pi)  # Should wrap to π
        3.141592653589793
        >>> wrap_angle(-3 * np.pi)  # Should wrap to -π
        -3.141592653589793
        >>> wrap_angle(np.pi + 0.1)  # Wraps to negative side
        -3.0415926535897927

    Notes:
        Uses the formula: θ_wrapped = atan2(sin(θ), cos(θ))
        This is numerically stable and handles all edge cases correctly.
    """
    return np.arctan2(np.sin(theta), np.cos(theta))


def se2_compose(
    p1: Union[np.ndarray, Pose2], p2: Union[np.ndarray, Pose2]
) -> np.ndarray:
    """
    Compose two SE(2) poses: p_result = p1 ⊕ p2.

    Computes the composition (chaining) of two SE(2) transformations.
    This is used in:
        - Scan matching: accumulating relative poses into odometry
        - Pose graph optimization: computing predicted relative poses
        - Frame transformations: T_A_to_C = T_A_to_B ⊕ T_B_to_C

    The composition formula for SE(2):
        x_result = x1 + x2*cos(yaw1) - y2*sin(yaw1)
        y_result = y1 + x2*sin(yaw1) + y2*cos(yaw1)
        yaw_result = yaw1 + yaw2  (wrapped to [-π, π])

    Args:
        p1: First pose, array [x1, y1, yaw1] or Pose2 instance.
        p2: Second pose, array [x2, y2, yaw2] or Pose2 instance.

    Returns:
        Composed pose as array [x, y, yaw] of shape (3,).

    Raises:
        ValueError: If poses do not have shape (3,).

    Examples:
        >>> # Identity composition
        >>> p_id = np.array([0, 0, 0])
        >>> p = np.array([1, 2, np.pi/4])
        >>> result = se2_compose(p_id, p)
        >>> np.allclose(result, p)
        True
        >>>
        >>> # Translation only
        >>> p1 = np.array([1, 0, 0])
        >>> p2 = np.array([2, 0, 0])
        >>> result = se2_compose(p1, p2)
        >>> np.allclose(result, [3, 0, 0])
        True
        >>>
        >>> # With rotation
        >>> p1 = np.array([0, 0, np.pi/2])  # 90° rotation
        >>> p2 = np.array([1, 0, 0])  # 1m forward
        >>> result = se2_compose(p1, p2)
        >>> # After 90° rotation, forward becomes left (0, 1)
        >>> np.allclose(result, [0, 1, np.pi/2], atol=1e-10)
        True

    Notes:
        - Uses the SE(2) group multiplication operation.
        - Result yaw is automatically wrapped to [-π, π].
        - Accepts both array and Pose2 dataclass inputs for convenience.
    """
    # Convert Pose2 to array if needed
    if isinstance(p1, Pose2):
        p1 = p1.to_array()
    if isinstance(p2, Pose2):
        p2 = p2.to_array()

    # Validate shapes
    if p1.shape != (3,):
        raise ValueError(f"p1 must have shape (3,), got {p1.shape}")
    if p2.shape != (3,):
        raise ValueError(f"p2 must have shape (3,), got {p2.shape}")

    x1, y1, yaw1 = p1
    x2, y2, yaw2 = p2

    # SE(2) composition formula
    cos_yaw1 = np.cos(yaw1)
    sin_yaw1 = np.sin(yaw1)

    x_result = x1 + x2 * cos_yaw1 - y2 * sin_yaw1
    y_result = y1 + x2 * sin_yaw1 + y2 * cos_yaw1
    yaw_result = wrap_angle(yaw1 + yaw2)

    return np.array([x_result, y_result, yaw_result], dtype=np.float64)


def se2_inverse(p: Union[np.ndarray, Pose2]) -> np.ndarray:
    """
    Compute the inverse of an SE(2) pose: p_inv = p⁻¹.

    Computes the inverse transformation such that p ⊕ p⁻¹ = identity.
    This is used in:
        - Computing relative poses: T_B_to_A = (T_A_to_B)⁻¹
        - Transforming between reference frames
        - Error computation in pose graph optimization

    The inverse formula for SE(2):
        x_inv = -(x*cos(yaw) + y*sin(yaw))
        y_inv = -(-x*sin(yaw) + y*cos(yaw))
        yaw_inv = -yaw  (wrapped to [-π, π])

    Args:
        p: Pose to invert, array [x, y, yaw] or Pose2 instance.

    Returns:
        Inverted pose as array [x, y, yaw] of shape (3,).

    Raises:
        ValueError: If pose does not have shape (3,).

    Examples:
        >>> # Identity inverse
        >>> p_id = np.array([0, 0, 0])
        >>> p_inv = se2_inverse(p_id)
        >>> np.allclose(p_inv, [0, 0, 0])
        True
        >>>
        >>> # Translation only
        >>> p = np.array([1, 2, 0])
        >>> p_inv = se2_inverse(p)
        >>> np.allclose(p_inv, [-1, -2, 0])
        True
        >>>
        >>> # Verify p ⊕ p⁻¹ = identity
        >>> p = np.array([1, 2, np.pi/4])
        >>> p_inv = se2_inverse(p)
        >>> result = se2_compose(p, p_inv)
        >>> np.allclose(result, [0, 0, 0], atol=1e-10)
        True

    Notes:
        - The inverse is computed directly (not via matrix inversion).
        - Result yaw is automatically wrapped to [-π, π].
        - Accepts both array and Pose2 dataclass inputs for convenience.
    """
    # Convert Pose2 to array if needed
    if isinstance(p, Pose2):
        p = p.to_array()

    # Validate shape
    if p.shape != (3,):
        raise ValueError(f"p must have shape (3,), got {p.shape}")

    x, y, yaw = p

    # SE(2) inverse formula
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    x_inv = -(x * cos_yaw + y * sin_yaw)
    y_inv = -(-x * sin_yaw + y * cos_yaw)
    yaw_inv = wrap_angle(-yaw)

    return np.array([x_inv, y_inv, yaw_inv], dtype=np.float64)


def se2_apply(
    p: Union[np.ndarray, Pose2], points: np.ndarray
) -> np.ndarray:
    """
    Transform 2D points by an SE(2) pose.

    Applies the SE(2) transformation to a set of 2D points:
        points_transformed = R(yaw) * points + [x, y]

    This is used in:
        - Scan matching: transforming scan points to reference frame (Eq. 7.10)
        - Visualization: plotting scans in global frame
        - Correspondence building: aligning point clouds (Eq. 7.11)

    Args:
        p: Pose [x, y, yaw] or Pose2 instance defining the transformation.
        points: Points to transform, array of shape (N, 2) where each row
                is [px, py] in meters.

    Returns:
        Transformed points, array of shape (N, 2).

    Raises:
        ValueError: If points does not have shape (N, 2).

    Examples:
        >>> # Identity transformation
        >>> p_id = np.array([0, 0, 0])
        >>> pts = np.array([[1, 0], [0, 1]])
        >>> result = se2_apply(p_id, pts)
        >>> np.allclose(result, pts)
        True
        >>>
        >>> # Pure translation
        >>> p = np.array([10, 5, 0])
        >>> pts = np.array([[1, 0], [0, 1]])
        >>> result = se2_apply(p, pts)
        >>> np.allclose(result, [[11, 5], [10, 6]])
        True
        >>>
        >>> # 90° rotation + translation
        >>> p = np.array([0, 0, np.pi/2])
        >>> pts = np.array([[1, 0], [0, 1]])
        >>> result = se2_apply(p, pts)
        >>> # [1,0] rotates to [0,1], [0,1] rotates to [-1,0]
        >>> np.allclose(result, [[0, 1], [-1, 0]], atol=1e-10)
        True

    Notes:
        - Uses 2D rotation matrix: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        - This is the action of SE(2) on points in the plane.
        - Vectorized for efficiency (operates on all points simultaneously).
    """
    # Convert Pose2 to array if needed
    if isinstance(p, Pose2):
        p = p.to_array()

    # Validate pose shape
    if p.shape != (3,):
        raise ValueError(f"p must have shape (3,), got {p.shape}")

    # Validate points shape
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"points must have shape (N, 2), got {points.shape}"
        )

    x, y, yaw = p

    # Build 2D rotation matrix
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)

    # Translation vector
    t = np.array([x, y], dtype=np.float64)

    # Apply transformation: R * points^T + t
    # points.T has shape (2, N), R @ points.T has shape (2, N)
    points_transformed = (R @ points.T).T + t

    return points_transformed


def se2_relative(
    p_from: Union[np.ndarray, Pose2], p_to: Union[np.ndarray, Pose2]
) -> np.ndarray:
    """
    Compute relative pose between two global poses.

    Given two poses in the same global frame, compute the relative
    transformation from p_from to p_to:
        p_relative = p_from⁻¹ ⊕ p_to

    This is used in:
        - Scan matching: computing the motion between consecutive scans
        - Loop closure: computing the constraint between non-consecutive poses
        - Pose graph factors: computing residuals (Eq. 7.68-7.70 for visual)

    Args:
        p_from: Starting pose [x, y, yaw] or Pose2 instance.
        p_to: Target pose [x, y, yaw] or Pose2 instance.

    Returns:
        Relative pose as array [x, y, yaw] of shape (3,).

    Examples:
        >>> # Same pose → identity
        >>> p = np.array([1, 2, np.pi/4])
        >>> rel = se2_relative(p, p)
        >>> np.allclose(rel, [0, 0, 0], atol=1e-10)
        True
        >>>
        >>> # Pure translation
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([5, 3, 0])
        >>> rel = se2_relative(p1, p2)
        >>> np.allclose(rel, [5, 3, 0])
        True
        >>>
        >>> # With rotation
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([1, 1, np.pi/2])
        >>> rel = se2_relative(p1, p2)
        >>> # Relative motion: 1m forward, 1m left, turn 90° left
        >>> np.allclose(rel, [1, 1, np.pi/2], atol=1e-10)
        True

    Notes:
        - Equivalent to: se2_compose(se2_inverse(p_from), p_to)
        - Used to convert global pose measurements to relative constraints.
    """
    p_from_inv = se2_inverse(p_from)
    return se2_compose(p_from_inv, p_to)


def se2_to_matrix(p: Union[np.ndarray, Pose2]) -> np.ndarray:
    """
    Convert SE(2) pose to 3x3 homogeneous transformation matrix.

    Converts a pose [x, y, yaw] to the matrix representation:
        T = [[cos(yaw), -sin(yaw), x],
             [sin(yaw),  cos(yaw), y],
             [       0,         0, 1]]

    This is useful for:
        - Matrix-based operations (less common in this repo)
        - Visualization / debugging
        - Interfacing with other SLAM libraries

    Args:
        p: Pose [x, y, yaw] or Pose2 instance.

    Returns:
        Homogeneous transformation matrix of shape (3, 3).

    Examples:
        >>> p = np.array([1, 2, np.pi/2])
        >>> T = se2_to_matrix(p)
        >>> print(T.shape)
        (3, 3)
        >>> # Last row should be [0, 0, 1]
        >>> np.allclose(T[2, :], [0, 0, 1])
        True
    """
    # Convert Pose2 to array if needed
    if isinstance(p, Pose2):
        p = p.to_array()

    if p.shape != (3,):
        raise ValueError(f"p must have shape (3,), got {p.shape}")

    x, y, yaw = p
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    T = np.array(
        [[cos_yaw, -sin_yaw, x], [sin_yaw, cos_yaw, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    return T


def se2_from_matrix(T: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 homogeneous transformation matrix to SE(2) pose.

    Extracts the pose [x, y, yaw] from a transformation matrix.

    Args:
        T: Homogeneous transformation matrix of shape (3, 3).

    Returns:
        Pose as array [x, y, yaw] of shape (3,).

    Raises:
        ValueError: If T is not a valid SE(2) matrix.

    Examples:
        >>> # Round-trip conversion
        >>> p = np.array([1, 2, np.pi/4])
        >>> T = se2_to_matrix(p)
        >>> p_recovered = se2_from_matrix(T)
        >>> np.allclose(p, p_recovered)
        True
    """
    if T.shape != (3, 3):
        raise ValueError(f"T must have shape (3, 3), got {T.shape}")

    # Extract translation
    x = T[0, 2]
    y = T[1, 2]

    # Extract rotation (yaw from rotation matrix)
    yaw = np.arctan2(T[1, 0], T[0, 0])

    return np.array([x, y, yaw], dtype=np.float64)


