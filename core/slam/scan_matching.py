"""ICP (Iterative Closest Point) scan matching for 2D LiDAR SLAM.

This module implements point-to-point ICP scan matching as described in
Section 7.2.1 of Chapter 7 (SLAM Technologies) of the book:
Principles of Indoor Positioning and Indoor Navigation.

ICP is the foundational algorithm for LiDAR-based SLAM, aligning two
point clouds by iteratively finding correspondences and computing the
optimal rigid transformation.

Key functions:
    - find_correspondences: Nearest-neighbor matching
    - compute_icp_residual: Point-to-point error (Eq. 7.10)
    - align_svd: Closed-form SVD alignment (Eq. 7.11)
    - icp_point_to_point: Full ICP algorithm

References:
    - Section 7.2.1: Point-to-point ICP
    - Eq. (7.10): ICP residual cost function
    - Eq. (7.11): SVD-based alignment solution

Author: Li-Ta Hsu
Date: 2024
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from .se2 import se2_apply, se2_compose, se2_inverse
from .types import Pose2


def find_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest-neighbor correspondences between source and target point clouds.

    For each point in the source cloud, finds the closest point in the target
    cloud using a KD-tree for efficient nearest-neighbor search. This is the
    data association step in ICP.

    Args:
        source_points: Source point cloud, shape (N, 2) in meters.
        target_points: Target point cloud, shape (M, 2) in meters.
        max_distance: Maximum correspondence distance in meters. Points beyond
                      this distance are rejected. If None, all correspondences
                      are accepted.

    Returns:
        Tuple of (matched_source, matched_target, distances):
            - matched_source: Source points with valid correspondences, shape (K, 2).
            - matched_target: Corresponding target points, shape (K, 2).
            - distances: Distance for each correspondence, shape (K,).
            where K <= N is the number of valid correspondences.

    Examples:
        >>> source = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> target = np.array([[1.1, 0.0], [0.0, 0.9], [5.0, 5.0]])
        >>> src_matched, tgt_matched, dists = find_correspondences(source, target)
        >>> print(src_matched.shape)  # (2, 2) - both source points matched
        >>> print(dists)  # [0.1, 0.1] - small distances

    Notes:
        - Uses scipy.spatial.KDTree for O(M log M + N log M) complexity.
        - Rejecting distant correspondences (max_distance) improves robustness
          to outliers and partial overlap.
        - Correspondences are one-to-many: multiple source points can match
          the same target point.
    """
    # Validate inputs
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError(
            f"source_points must have shape (N, 2), got {source_points.shape}"
        )
    if target_points.ndim != 2 or target_points.shape[1] != 2:
        raise ValueError(
            f"target_points must have shape (M, 2), got {target_points.shape}"
        )

    if source_points.shape[0] == 0:
        # No source points → no correspondences
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty((0,)),
        )

    if target_points.shape[0] == 0:
        # No target points → no correspondences
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty((0,)),
        )

    # Build KD-tree for target points
    tree = KDTree(target_points)

    # Query nearest neighbor for each source point
    distances, indices = tree.query(source_points, k=1)

    # Filter by max_distance if specified
    if max_distance is not None:
        valid_mask = distances <= max_distance
        distances = distances[valid_mask]
        indices = indices[valid_mask]
        matched_source = source_points[valid_mask]
    else:
        matched_source = source_points

    # Get corresponding target points
    matched_target = target_points[indices]

    return matched_source, matched_target, distances


def compute_icp_residual(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """
    Compute point-to-point ICP residual (cost).

    Computes the sum of squared distances between corresponding points,
    as defined in Eq. (7.10) of Chapter 7:

        E = sum_i ||p_i^source - p_i^target||^2

    This is the objective function minimized by ICP.

    Args:
        source_points: Source points, shape (N, 2).
        target_points: Corresponding target points, shape (N, 2).
                       Must have the same number of points as source.

    Returns:
        Total squared error (scalar, non-negative).

    Raises:
        ValueError: If point clouds have different sizes.

    Examples:
        >>> source = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> target = np.array([[1.1, 0.0], [0.0, 0.9]])
        >>> residual = compute_icp_residual(source, target)
        >>> print(f"{residual:.4f}")  # 0.02 = 0.1^2 + 0.1^2

    Notes:
        - Implements Eq. (7.10) from Section 7.2.1.
        - This is the point-to-point metric (not point-to-plane).
        - Used to assess convergence in ICP iterations.
    """
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Point clouds must have same shape. "
            f"Got source={source_points.shape}, target={target_points.shape}"
        )

    if source_points.shape[0] == 0:
        return 0.0

    # Compute squared distances
    diff = source_points - target_points
    squared_distances = np.sum(diff**2, axis=1)
    total_residual = np.sum(squared_distances)

    return float(total_residual)


def align_svd(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal SE(2) alignment using SVD (closed-form solution).

    Given corresponding point sets, computes the rigid transformation
    (rotation + translation) that minimizes the point-to-point error.
    This is the solution to Eq. (7.11) in Chapter 7.

    The algorithm:
        1. Compute centroids of both point clouds.
        2. Center the point clouds (subtract centroids).
        3. Compute cross-covariance matrix H = sum_i (p_i^source) (p_i^target)^T.
        4. SVD: H = U Σ V^T.
        5. Optimal rotation: R = V U^T (with det correction if needed).
        6. Optimal translation: t = centroid_target - R * centroid_source.
        7. Extract yaw from R and build SE(2) pose.

    Args:
        source_points: Source points, shape (N, 2).
        target_points: Corresponding target points, shape (N, 2).

    Returns:
        Optimal pose as array [x, y, yaw] of shape (3,) that transforms
        source points to align with target points.

    Raises:
        ValueError: If point clouds have different sizes or fewer than 2 points.

    Examples:
        >>> # Perfect alignment: source = target
        >>> source = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> target = source.copy()
        >>> pose = align_svd(source, target)
        >>> np.allclose(pose, [0, 0, 0], atol=1e-10)
        True
        >>>
        >>> # Pure translation
        >>> source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> target = source + np.array([2.0, 3.0])
        >>> pose = align_svd(source, target)
        >>> np.allclose(pose, [2, 3, 0], atol=1e-6)
        True

    Notes:
        - Implements the closed-form SVD solution from Eq. (7.11).
        - Requires at least 2 correspondences for a unique solution.
        - Assumes correspondences are already established (no outlier rejection).
        - For degenerate configurations (e.g., collinear points), the solution
          may be numerically unstable.
    """
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Point clouds must have same shape. "
            f"Got source={source_points.shape}, target={target_points.shape}"
        )

    N = source_points.shape[0]
    if N < 2:
        raise ValueError(
            f"Need at least 2 correspondences for SVD alignment, got {N}"
        )

    # Step 1: Compute centroids
    centroid_source = np.mean(source_points, axis=0)  # shape (2,)
    centroid_target = np.mean(target_points, axis=0)  # shape (2,)

    # Step 2: Center the point clouds
    source_centered = source_points - centroid_source  # shape (N, 2)
    target_centered = target_points - centroid_target  # shape (N, 2)

    # Step 3: Compute cross-covariance matrix H = sum_i (source_i)(target_i)^T
    # H has shape (2, 2)
    H = source_centered.T @ target_centered  # (2, N) @ (N, 2) = (2, 2)

    # Step 4: SVD decomposition
    U, _, Vt = np.linalg.svd(H)  # H = U Σ V^T

    # Step 5: Compute optimal rotation R = V U^T
    R = Vt.T @ U.T  # (2, 2) rotation matrix

    # Handle reflection case: if det(R) < 0, we have a reflection, not rotation
    if np.linalg.det(R) < 0:
        # Flip the sign of the last column of V
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Extract yaw angle from rotation matrix
    # R = [[cos(yaw), -sin(yaw)],
    #      [sin(yaw),  cos(yaw)]]
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Step 7: Compute optimal translation
    # t = centroid_target - R @ centroid_source
    t = centroid_target - R @ centroid_source  # shape (2,)

    # Return as SE(2) pose [x, y, yaw]
    return np.array([t[0], t[1], yaw], dtype=np.float64)


def icp_point_to_point(
    source_scan: np.ndarray,
    target_scan: np.ndarray,
    initial_pose: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: Optional[float] = None,
    min_correspondences: int = 3,
) -> Tuple[np.ndarray, int, float, bool]:
    """
    Point-to-point ICP algorithm for 2D scan matching.

    Iteratively aligns source scan to target scan by alternating between:
        1. Finding correspondences (nearest neighbors).
        2. Computing optimal transformation (SVD alignment).

    Converges when the pose change is below tolerance or max iterations reached.

    Args:
        source_scan: Source point cloud (e.g., current scan), shape (N, 2).
        target_scan: Target point cloud (e.g., reference scan), shape (M, 2).
        initial_pose: Initial guess [x, y, yaw], shape (3,). If None, uses identity.
        max_iterations: Maximum number of ICP iterations (default: 50).
        tolerance: Convergence threshold for pose change (default: 1e-6).
                   Measured as ||delta_pose|| in Euclidean sense.
        max_correspondence_distance: Maximum distance for valid correspondences
                                      in meters. If None, all correspondences accepted.
        min_correspondences: Minimum number of correspondences required for
                             alignment (default: 3). If fewer, ICP fails.

    Returns:
        Tuple of (final_pose, num_iterations, final_residual, converged):
            - final_pose: Estimated pose [x, y, yaw], shape (3,).
            - num_iterations: Number of iterations executed.
            - final_residual: Final ICP residual (sum of squared errors).
            - converged: True if convergence criteria met, False if max_iterations reached.

    Raises:
        ValueError: If scans are empty or have invalid shapes.

    Examples:
        >>> # Perfect alignment with identity
        >>> scan = np.array([[1, 0], [0, 1], [1, 1]])
        >>> pose, iters, residual, converged = icp_point_to_point(scan, scan)
        >>> np.allclose(pose, [0, 0, 0], atol=1e-4)
        True
        >>> converged
        True
        >>>
        >>> # Scan with known translation
        >>> source = np.array([[0, 0], [1, 0], [0, 1]])
        >>> target = source + np.array([2, 3])
        >>> pose, _, _, converged = icp_point_to_point(source, target)
        >>> np.allclose(pose[:2], [2, 3], atol=1e-3)
        True

    Notes:
        - Implements the standard point-to-point ICP from Section 7.2.1.
        - Uses KD-tree for efficient correspondence search.
        - SVD provides closed-form solution at each iteration (Eq. 7.11).
        - Convergence is not guaranteed for poor initialization or low overlap.
        - Returns last valid pose even if convergence fails.
    """
    # Validate inputs
    if source_scan.ndim != 2 or source_scan.shape[1] != 2:
        raise ValueError(
            f"source_scan must have shape (N, 2), got {source_scan.shape}"
        )
    if target_scan.ndim != 2 or target_scan.shape[1] != 2:
        raise ValueError(
            f"target_scan must have shape (M, 2), got {target_scan.shape}"
        )

    if source_scan.shape[0] == 0:
        raise ValueError("source_scan is empty")
    if target_scan.shape[0] == 0:
        raise ValueError("target_scan is empty")

    # Initialize pose
    if initial_pose is None:
        current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        current_pose = initial_pose.astype(np.float64).copy()

    # ICP main loop
    converged = False
    final_residual = np.inf

    for iteration in range(max_iterations):
        # Step 1: Transform source scan by current pose
        transformed_source = se2_apply(current_pose, source_scan)

        # Step 2: Find correspondences
        matched_source, matched_target, _ = find_correspondences(
            transformed_source,
            target_scan,
            max_distance=max_correspondence_distance,
        )

        # Check if we have enough correspondences
        if matched_source.shape[0] < min_correspondences:
            # Not enough correspondences → return current pose as failure
            return current_pose, iteration + 1, final_residual, False

        # Step 3: Compute residual
        current_residual = compute_icp_residual(matched_source, matched_target)
        
        # Update final_residual (CRITICAL: do this before convergence check)
        final_residual = current_residual

        # Step 4: Compute incremental pose using SVD
        delta_pose = align_svd(matched_source, matched_target)

        # Step 5: Check convergence
        # Convergence when delta pose is very small
        pose_change = np.linalg.norm(delta_pose)
        
        if pose_change < tolerance:
            # Converged! Apply final delta and return
            converged = True
            current_pose = se2_compose(current_pose, delta_pose)
            return current_pose, iteration + 1, final_residual, True

        # Step 6: Update current pose for next iteration
        # new_pose = current_pose ⊕ delta_pose
        current_pose = se2_compose(current_pose, delta_pose)

    # Max iterations reached without convergence
    return current_pose, max_iterations, final_residual, False


def compute_icp_covariance(
    source_scan: np.ndarray,
    target_scan: np.ndarray,
    final_pose: np.ndarray,
    max_correspondence_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate covariance of ICP-estimated pose (simplified approach).

    Computes a simplified covariance estimate based on the residual error
    and the number of correspondences. This is a heuristic approximation,
    not a rigorous uncertainty estimate.

    Args:
        source_scan: Source point cloud, shape (N, 2).
        target_scan: Target point cloud, shape (M, 2).
        final_pose: Final ICP pose [x, y, yaw], shape (3,).
        max_correspondence_distance: Maximum correspondence distance.

    Returns:
        Covariance matrix of shape (3, 3) representing uncertainty in [x, y, yaw].

    Notes:
        - This is a simplified heuristic for pose uncertainty.
        - For rigorous covariance, use Fisher Information Matrix or bootstrap methods.
        - Used primarily for downstream pose graph optimization weights.
    """
    # Transform source by final pose
    transformed_source = se2_apply(final_pose, source_scan)

    # Find correspondences
    matched_source, matched_target, _ = find_correspondences(
        transformed_source, target_scan, max_distance=max_correspondence_distance
    )

    if matched_source.shape[0] < 10:
        # Too few correspondences → high uncertainty
        return np.diag([1.0, 1.0, 0.1])  # Large uncertainty

    # Compute residual per point
    residual = compute_icp_residual(matched_source, matched_target)
    residual_per_point = residual / matched_source.shape[0]

    # Heuristic: scale covariance by residual and inverse number of points
    # σ_xy² ≈ residual / N
    # σ_yaw² ≈ residual / (N * lever_arm²)
    N = matched_source.shape[0]
    sigma_xy = np.sqrt(residual_per_point / N)
    sigma_yaw = sigma_xy / 1.0  # Assume 1m lever arm for rotation uncertainty

    # Build diagonal covariance (assume independence for simplicity)
    cov = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])

    return cov

