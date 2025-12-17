"""Camera projection and distortion models for visual SLAM.

This module implements camera models from Chapter 7 of the book, including:
    - Pinhole camera projection
    - Radial and tangential lens distortion
    - Inverse projection (pixel to ray)

These models are used for visual odometry, bundle adjustment, and reprojection
error computation in visual SLAM systems.

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple
from core.slam.types import CameraIntrinsics


def distort_normalized(
    xy_normalized: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
) -> np.ndarray:
    """
    Apply radial and tangential distortion to normalized image coordinates.

    Implements the Brown-Conrady distortion model from Eqs. (7.43)-(7.46)
    in Chapter 7.

    The distortion model:
        x_distorted = x * (1 + k1*r² + k2*r⁴) + 2*p1*x*y + p2*(r² + 2*x²)
        y_distorted = y * (1 + k1*r² + k2*r⁴) + p1*(r² + 2*y²) + 2*p2*x*y

    where r² = x² + y² and (x, y) are normalized image coordinates.

    Args:
        xy_normalized: Normalized image coordinates, shape (N, 2) or (2,).
                      These are 3D points projected onto z=1 plane: (X/Z, Y/Z).
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.

    Returns:
        Distorted normalized coordinates, same shape as input.

    References:
        Implements Eqs. (7.43)-(7.46) from Chapter 7 (Brown-Conrady model).

    Example:
        >>> xy = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> distorted = distort_normalized(xy, k1=-0.1, k2=0.01, p1=0.001, p2=0.001)
    """
    # Handle both single point and array of points
    single_point = False
    if xy_normalized.ndim == 1:
        single_point = True
        xy_normalized = xy_normalized.reshape(1, -1)

    if xy_normalized.shape[1] != 2:
        raise ValueError(f"Input must be (N, 2) or (2,), got {xy_normalized.shape}")

    x = xy_normalized[:, 0]
    y = xy_normalized[:, 1]

    # Compute r² = x² + y²
    r_squared = x**2 + y**2

    # Radial distortion factor: (1 + k1*r² + k2*r⁴)
    # Eq. (7.43): radial component
    radial_distortion = 1.0 + k1 * r_squared + k2 * r_squared**2

    # Tangential distortion components
    # Eq. (7.44)-(7.45): tangential components
    tangential_x = 2.0 * p1 * x * y + p2 * (r_squared + 2.0 * x**2)
    tangential_y = p1 * (r_squared + 2.0 * y**2) + 2.0 * p2 * x * y

    # Apply distortion
    # Eq. (7.46): combined distortion model
    x_distorted = x * radial_distortion + tangential_x
    y_distorted = y * radial_distortion + tangential_y

    result = np.column_stack([x_distorted, y_distorted])

    if single_point:
        result = result.reshape(-1)

    return result


def undistort_normalized(
    xy_distorted: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Remove distortion from normalized image coordinates (inverse of distortion).

    Uses iterative Newton-Raphson method to invert the distortion model.

    Args:
        xy_distorted: Distorted normalized coordinates, shape (N, 2) or (2,).
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.
        max_iterations: Maximum number of Newton-Raphson iterations.
        tolerance: Convergence tolerance.

    Returns:
        Undistorted normalized coordinates, same shape as input.

    Note:
        This is the inverse operation of distort_normalized(), used when
        going from pixel observations to 3D rays.
    """
    single_point = False
    if xy_distorted.ndim == 1:
        single_point = True
        xy_distorted = xy_distorted.reshape(1, -1)

    # Initial guess: distorted coordinates
    xy_undistorted = xy_distorted.copy()

    # Newton-Raphson iteration
    for _ in range(max_iterations):
        # Forward distortion from current guess
        xy_predicted = distort_normalized(xy_undistorted, k1, k2, p1, p2)

        # Residual
        residual = xy_predicted - xy_distorted

        # Check convergence
        if np.max(np.abs(residual)) < tolerance:
            break

        # Update (simple gradient descent with step size)
        xy_undistorted -= 0.9 * residual

    if single_point:
        xy_undistorted = xy_undistorted.reshape(-1)

    return xy_undistorted


def project_point(
    intrinsics: CameraIntrinsics,
    point_camera: np.ndarray,
) -> np.ndarray:
    """
    Project a 3D point in camera frame to pixel coordinates.

    Implements the full camera projection model: 3D point → normalized coords
    → distortion → pixel coordinates.

    The projection follows:
        1. Normalize: (x_n, y_n) = (X/Z, Y/Z)
        2. Distort: (x_d, y_d) = distort(x_n, y_n)  [Eqs. 7.43-7.46]
        3. Scale: (u, v) = (fx*x_d + cx, fy*y_d + cy)  [Eq. 7.40]

    Args:
        intrinsics: Camera intrinsic parameters (fx, fy, cx, cy, distortion).
        point_camera: 3D point(s) in camera frame, shape (3,) or (N, 3).
                     Camera frame: X-right, Y-down, Z-forward.

    Returns:
        Pixel coordinates (u, v), shape (2,) or (N, 2).

    Raises:
        ValueError: If point is behind camera (Z <= 0).

    References:
        Implements Eq. (7.40) (pinhole projection) combined with
        Eqs. (7.43)-(7.46) (distortion model) from Chapter 7.

    Example:
        >>> intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> point_3d = np.array([1.0, 0.5, 5.0])  # 5m in front, 1m right
        >>> pixel = project_point(intrinsics, point_3d)
    """
    single_point = False
    if point_camera.ndim == 1:
        single_point = True
        point_camera = point_camera.reshape(1, -1)

    if point_camera.shape[1] != 3:
        raise ValueError(f"Point must be (3,) or (N, 3), got {point_camera.shape}")

    X = point_camera[:, 0]
    Y = point_camera[:, 1]
    Z = point_camera[:, 2]

    # Check if points are in front of camera
    if np.any(Z <= 0):
        raise ValueError("Cannot project points behind camera (Z <= 0)")

    # 1. Normalize to image plane (z=1)
    # Eq. (7.40): perspective division
    x_normalized = X / Z
    y_normalized = Y / Z
    xy_normalized = np.column_stack([x_normalized, y_normalized])

    # 2. Apply distortion
    # Eqs. (7.43)-(7.46): distortion model
    xy_distorted = distort_normalized(
        xy_normalized,
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.p1,
        intrinsics.p2,
    )

    # 3. Apply intrinsic matrix (scale and offset)
    # Eq. (7.40): pixel coordinates from normalized coords
    u = intrinsics.fx * xy_distorted[:, 0] + intrinsics.cx
    v = intrinsics.fy * xy_distorted[:, 1] + intrinsics.cy

    result = np.column_stack([u, v])

    if single_point:
        result = result.reshape(-1)

    return result


def unproject_pixel(
    intrinsics: CameraIntrinsics,
    pixel: np.ndarray,
    depth: Optional[float] = None,
) -> np.ndarray:
    """
    Unproject pixel coordinates to a 3D ray or point in camera frame.

    This is the inverse of project_point(). If depth is provided, returns
    the 3D point at that depth. Otherwise, returns a unit direction ray.

    Args:
        intrinsics: Camera intrinsic parameters.
        pixel: Pixel coordinates (u, v), shape (2,) or (N, 2).
        depth: Optional depth value(s). If scalar, applies to all pixels.
               If array, must have shape (N,) matching number of pixels.

    Returns:
        If depth provided: 3D point(s) in camera frame, shape (3,) or (N, 3).
        If no depth: Unit direction ray(s), shape (3,) or (N, 3).

    Example:
        >>> intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> pixel = np.array([420, 340])
        >>> ray = unproject_pixel(intrinsics, pixel)  # Unit direction
        >>> point = unproject_pixel(intrinsics, pixel, depth=5.0)  # 5m away
    """
    single_point = False
    if pixel.ndim == 1:
        single_point = True
        pixel = pixel.reshape(1, -1)

    if pixel.shape[1] != 2:
        raise ValueError(f"Pixel must be (2,) or (N, 2), got {pixel.shape}")

    u = pixel[:, 0]
    v = pixel[:, 1]

    # 1. Inverse intrinsic matrix: pixel → normalized distorted coords
    x_distorted = (u - intrinsics.cx) / intrinsics.fx
    y_distorted = (v - intrinsics.cy) / intrinsics.fy
    xy_distorted = np.column_stack([x_distorted, y_distorted])

    # 2. Remove distortion
    xy_normalized = undistort_normalized(
        xy_distorted,
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.p1,
        intrinsics.p2,
    )

    # 3. Create 3D direction (on z=1 plane, then normalize)
    # Direction in camera frame: [x_n, y_n, 1]
    N = xy_normalized.shape[0]
    directions = np.column_stack([xy_normalized, np.ones(N)])

    # Normalize to unit vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms

    # 4. Scale by depth if provided
    if depth is not None:
        if np.isscalar(depth):
            depth_array = np.full(N, depth)
        else:
            depth_array = np.asarray(depth)
            if depth_array.shape != (N,):
                raise ValueError(f"Depth shape {depth_array.shape} doesn't match pixels {N}")

        # Scale directions by depth
        directions = directions * depth_array[:, np.newaxis]

    if single_point:
        directions = directions.reshape(-1)

    return directions


def compute_reprojection_error(
    intrinsics: CameraIntrinsics,
    point_camera: np.ndarray,
    observed_pixel: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection error between observed and projected pixel.

    This is the core residual function for bundle adjustment and visual SLAM.

    Args:
        intrinsics: Camera intrinsic parameters.
        point_camera: 3D point(s) in camera frame, shape (3,) or (N, 3).
        observed_pixel: Observed pixel coordinates, shape (2,) or (N, 2).

    Returns:
        Reprojection error [Δu, Δv] in pixels, shape (2,) or (N, 2).
        Error = projected_pixel - observed_pixel

    Note:
        This residual is minimized in bundle adjustment (Eqs. 7.68-7.70).
        Lower error indicates better alignment between 3D structure and
        image observations.

    Example:
        >>> intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> point_3d = np.array([1.0, 0.5, 5.0])
        >>> observed = np.array([420.5, 340.2])
        >>> error = compute_reprojection_error(intrinsics, point_3d, observed)
    """
    # Project 3D point to pixel
    projected_pixel = project_point(intrinsics, point_camera)

    # Compute error
    error = projected_pixel - observed_pixel

    return error


def essential_matrix_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute essential matrix from relative pose between two cameras.

    The essential matrix encodes the epipolar geometry constraint:
        p2ᵀ E p1 = 0

    where p1 and p2 are corresponding points in normalized image coordinates.

    Args:
        R: Rotation matrix from camera 1 to camera 2, shape (3, 3).
        t: Translation vector from camera 1 to camera 2, shape (3,).

    Returns:
        Essential matrix E, shape (3, 3).

    Note:
        E = [t]ₓ R, where [t]ₓ is the skew-symmetric matrix of t.
        This is used in two-view geometry and visual odometry initialization.
    """
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3, 3), got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"t must be (3,), got {t.shape}")

    # Skew-symmetric matrix of translation
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

    # Essential matrix
    E = t_skew @ R

    return E


def triangulate_point(
    pixel1: np.ndarray,
    pixel2: np.ndarray,
    intrinsics1: CameraIntrinsics,
    intrinsics2: CameraIntrinsics,
    R_1_to_2: np.ndarray,
    t_1_to_2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate a 3D point from two camera views (simplified linear method).

    Given corresponding pixels in two calibrated cameras with known relative
    pose, compute the 3D point position.

    Args:
        pixel1: Pixel in camera 1, shape (2,).
        pixel2: Pixel in camera 2, shape (2,).
        intrinsics1: Camera 1 intrinsics.
        intrinsics2: Camera 2 intrinsics.
        R_1_to_2: Rotation from camera 1 to camera 2, shape (3, 3).
        t_1_to_2: Translation from camera 1 to camera 2, shape (3,).

    Returns:
        3D point in camera 1 frame, shape (3,).

    Note:
        This uses a simplified midpoint method. For production use,
        DLT (Direct Linear Transform) or optimal triangulation is preferred.
    """
    # Get rays in each camera frame
    ray1 = unproject_pixel(intrinsics1, pixel1)
    ray2 = unproject_pixel(intrinsics2, pixel2)

    # Transform ray2 to camera1 frame
    # point_in_cam1 = R_1_to_2.T @ (point_in_cam2) + (-R_1_to_2.T @ t_1_to_2)
    ray2_in_cam1 = R_1_to_2.T @ ray2

    # Solve for depths using least squares
    # We want: depth1 * ray1 ≈ R_1_to_2.T @ (depth2 * ray2) - R_1_to_2.T @ t_1_to_2
    # This is a simplified approach; proper triangulation uses DLT

    # For simplicity, use midpoint of closest approach
    # This is a heuristic but works for teaching purposes
    origin1 = np.zeros(3)
    origin2_in_cam1 = -R_1_to_2.T @ t_1_to_2

    # Find closest point on ray1 to ray from origin2_in_cam1
    # Use midpoint heuristic
    direction_between = origin2_in_cam1 - origin1
    t1 = np.dot(direction_between, ray1)
    t2 = np.dot(direction_between - t1 * ray1, ray2_in_cam1)

    # Average of two closest points
    point1 = origin1 + t1 * ray1
    point2 = origin2_in_cam1 + t2 * ray2_in_cam1
    point_3d = (point1 + point2) / 2.0

    return point_3d


