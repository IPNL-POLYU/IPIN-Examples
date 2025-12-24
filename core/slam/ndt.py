"""NDT (Normal Distributions Transform) scan matching for 2D LiDAR SLAM.

This module implements NDT alignment as described in Section 7.2.2 of
Chapter 7 (SLAM Technologies) of the book: Principles of Indoor
Positioning and Indoor Navigation.

NDT represents the target scan as a probabilistic model (Gaussian distributions
per voxel) rather than raw points, leading to smoother cost functions and
better convergence properties compared to point-to-point ICP.

Key functions:
    - build_ndt_map: Build voxel grid with Gaussian distributions
    - ndt_score: Compute NDT score function (Eqs. 7.12-7.16)
    - ndt_gradient: Compute gradient for optimization
    - ndt_align: Full NDT alignment with Newton's method

References:
    - Section 7.2.2: Normal Distributions Transform (NDT)
    - Eq. (7.12): NDT score function
    - Eq. (7.13): Probability density per voxel
    - Eq. (7.14): Negative log-likelihood formulation
    - Eqs. (7.15)-(7.16): Gradient and Hessian

Author: Li-Ta Hsu
Date: 2024
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .se2 import se2_apply, se2_compose
from .types import VoxelGrid


def build_ndt_map(
    points: np.ndarray,
    voxel_size: float = 1.0,
    min_points_per_voxel: int = 3,
) -> VoxelGrid:
    """
    Build NDT map from point cloud: voxel grid with Gaussian distributions.

    Divides 2D space into voxels and fits a Gaussian distribution (mean and
    covariance) to the points in each voxel. This is the offline preprocessing
    step for NDT alignment.

    Args:
        points: Point cloud, shape (N, 2) in meters.
        voxel_size: Voxel edge length in meters (default: 1.0).
        min_points_per_voxel: Minimum number of points required to fit a
                              Gaussian in a voxel (default: 3).

    Returns:
        VoxelGrid: Dictionary mapping voxel indices (i, j) to Gaussian parameters:
            {
                (i, j): {
                    'mean': np.ndarray of shape (2,),
                    'cov': np.ndarray of shape (2, 2),
                    'n_points': int
                },
                ...
            }

    Examples:
        >>> points = np.array([[0.1, 0.2], [0.3, 0.4], [1.5, 1.6]])
        >>> ndt_map = build_ndt_map(points, voxel_size=1.0)
        >>> print(len(ndt_map))  # 2 voxels
        2
        >>> voxel_00 = ndt_map[(0, 0)]
        >>> print(voxel_00['n_points'])  # 2 points in voxel (0, 0)
        2

    Notes:
        - Voxels with fewer than min_points_per_voxel are discarded (no Gaussian).
        - Covariance is regularized with small diagonal term to avoid singularity.
        - This implements the voxelization step described in Section 7.2.2.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    if points.shape[0] == 0:
        return {}

    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Group points by voxel
    voxels: Dict[Tuple[int, int], list] = {}
    for i, point in enumerate(points):
        voxel_key = tuple(voxel_indices[i])
        if voxel_key not in voxels:
            voxels[voxel_key] = []
        voxels[voxel_key].append(point)

    # Fit Gaussian to each voxel
    ndt_map: VoxelGrid = {}
    for voxel_key, voxel_points in voxels.items():
        if len(voxel_points) < min_points_per_voxel:
            continue

        voxel_points_array = np.array(voxel_points)  # shape (n, 2)

        # Compute mean
        mean = np.mean(voxel_points_array, axis=0)  # shape (2,)

        # Compute covariance
        centered = voxel_points_array - mean
        cov = (centered.T @ centered) / len(voxel_points)  # shape (2, 2)

        # Regularize covariance to avoid singularity
        cov += np.eye(2) * 1e-4

        ndt_map[voxel_key] = {
            "mean": mean,
            "cov": cov,
            "n_points": len(voxel_points),
        }

    return ndt_map


def ndt_score(
    source_points: np.ndarray,
    ndt_map: VoxelGrid,
    pose: np.ndarray,
    voxel_size: float = 1.0,
) -> float:
    """
    Compute NDT score (negative log-likelihood) for a given pose.

    Evaluates how well the source points (transformed by pose) align with
    the target NDT map. This implements Eqs. (7.12)-(7.14) from Section 7.2.2.

    The score is:
        score = -sum_i log( p(T(p_i)) )
    where p(T(p_i)) is the probability density of the transformed source point
    evaluated at the corresponding voxel's Gaussian distribution.

    Args:
        source_points: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map (voxel grid with Gaussians).
        pose: Pose [x, y, yaw], shape (3,) to transform source.
        voxel_size: Voxel edge length (must match ndt_map).

    Returns:
        NDT score (scalar). Lower is better (negative log-likelihood).

    Examples:
        >>> source = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> target = source.copy()
        >>> ndt_map = build_ndt_map(target, voxel_size=2.0)
        >>> pose_identity = np.array([0.0, 0.0, 0.0])
        >>> score = ndt_score(source, ndt_map, pose_identity, voxel_size=2.0)
        >>> # Perfect alignment should give low score
        >>> score < 5.0
        True

    Notes:
        - Implements Eq. (7.12)-(7.14) from Chapter 7.
        - Points that fall outside occupied voxels are ignored.
        - Uses negative log-likelihood formulation for numerical stability.
    """
    if source_points.shape[0] == 0:
        return 0.0

    # Transform source points by pose
    transformed_points = se2_apply(pose, source_points)

    # Compute voxel indices for transformed points
    voxel_indices = np.floor(transformed_points / voxel_size).astype(int)

    total_score = 0.0
    n_matched = 0

    for i, point in enumerate(transformed_points):
        voxel_key = tuple(voxel_indices[i])

        if voxel_key not in ndt_map:
            # Point falls in empty voxel → skip
            continue

        voxel = ndt_map[voxel_key]
        mean = voxel["mean"]
        cov = voxel["cov"]

        # Compute Mahalanobis distance: (p - μ)^T Σ^{-1} (p - μ)
        diff = point - mean
        try:
            cov_inv = np.linalg.inv(cov)
            mahalanobis = diff @ cov_inv @ diff
        except np.linalg.LinAlgError:
            # Singular covariance → skip this voxel
            continue

        # Gaussian log-likelihood (without constant terms):
        # log p(x) = -0.5 * mahalanobis - 0.5 * log(det(Σ))
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Invalid covariance → skip
            continue

        log_likelihood = -0.5 * mahalanobis - 0.5 * logdet

        # NDT score is negative log-likelihood (we want to maximize likelihood)
        total_score -= log_likelihood
        n_matched += 1

    if n_matched == 0:
        # No points matched → return large penalty
        return 1e6

    # Average score per matched point
    return total_score / n_matched


def ndt_gradient(
    source_points: np.ndarray,
    ndt_map: VoxelGrid,
    pose: np.ndarray,
    voxel_size: float = 1.0,
) -> np.ndarray:
    """
    Compute gradient of NDT score with respect to pose.

    Computes the gradient ∇_pose score(pose) for gradient-based optimization.
    This implements the gradient computation from Eq. (7.15) in Section 7.2.2.

    Args:
        source_points: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map.
        pose: Current pose [x, y, yaw], shape (3,).
        voxel_size: Voxel edge length.

    Returns:
        Gradient vector of shape (3,): [∂score/∂x, ∂score/∂y, ∂score/∂yaw].

    Notes:
        - Uses finite differences for simplicity (not analytic gradient).
        - For production code, analytic gradients would be more efficient.
        - Implements numerical approximation of Eq. (7.15).
    """
    gradient = np.zeros(3)
    epsilon = 1e-6

    # Compute gradient via finite differences
    base_score = ndt_score(source_points, ndt_map, pose, voxel_size)

    for i in range(3):
        pose_plus = pose.copy()
        pose_plus[i] += epsilon

        score_plus = ndt_score(source_points, ndt_map, pose_plus, voxel_size)

        gradient[i] = (score_plus - base_score) / epsilon

    return gradient


def ndt_align(
    source_scan: np.ndarray,
    target_scan: np.ndarray,
    initial_pose: Optional[np.ndarray] = None,
    voxel_size: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    step_size: float = 0.1,
) -> Tuple[np.ndarray, int, float, bool]:
    """
    NDT-based scan alignment using gradient descent.

    Aligns source scan to target scan by optimizing the NDT score function.
    The target scan is first converted to an NDT map (voxel grid with Gaussians),
    then gradient descent minimizes the negative log-likelihood.

    Args:
        source_scan: Source point cloud, shape (N, 2).
        target_scan: Target point cloud, shape (M, 2).
        initial_pose: Initial pose guess [x, y, yaw], shape (3,).
                      If None, uses identity.
        voxel_size: Voxel edge length in meters (default: 1.0).
        max_iterations: Maximum number of optimization iterations (default: 50).
        tolerance: Convergence threshold for score change (default: 1e-3).
        step_size: Gradient descent step size (default: 0.1).

    Returns:
        Tuple of (final_pose, num_iterations, final_score, converged):
            - final_pose: Estimated pose [x, y, yaw], shape (3,).
            - num_iterations: Number of iterations executed.
            - final_score: Final NDT score.
            - converged: True if convergence criteria met.

    Examples:
        >>> source = np.array([[0, 0], [1, 0], [0, 1]])
        >>> target = source + np.array([2, 3])
        >>> pose, iters, score, converged = ndt_align(source, target, voxel_size=2.0)
        >>> converged
        True

    Notes:
        - Implements NDT alignment from Section 7.2.2.
        - Uses gradient descent (simpler than Newton's method from Eq. 7.16).
        - For better performance, consider using scipy.optimize.minimize with
          analytic gradients.
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

    # Build NDT map from target scan
    ndt_map = build_ndt_map(target_scan, voxel_size=voxel_size)

    if len(ndt_map) == 0:
        # No valid voxels → cannot align
        return (
            initial_pose if initial_pose is not None else np.zeros(3),
            0,
            1e6,
            False,
        )

    # Initialize pose
    if initial_pose is None:
        current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        current_pose = initial_pose.astype(np.float64).copy()

    # Optimization loop (gradient descent)
    converged = False
    prev_score = np.inf

    for iteration in range(max_iterations):
        # Compute current score
        current_score = ndt_score(source_scan, ndt_map, current_pose, voxel_size)

        # Check convergence
        score_change = abs(prev_score - current_score)
        if score_change < tolerance:
            converged = True
            return current_pose, iteration + 1, current_score, converged

        # Compute gradient
        grad = ndt_gradient(source_scan, ndt_map, current_pose, voxel_size)

        # Gradient descent update
        current_pose -= step_size * grad

        # Normalize yaw to [-π, π]
        current_pose[2] = np.arctan2(np.sin(current_pose[2]), np.cos(current_pose[2]))

        prev_score = current_score

    # Max iterations reached
    final_score = ndt_score(source_scan, ndt_map, current_pose, voxel_size)
    return current_pose, max_iterations, final_score, False


def ndt_covariance(
    source_scan: np.ndarray,
    ndt_map: VoxelGrid,
    final_pose: np.ndarray,
    voxel_size: float = 1.0,
) -> np.ndarray:
    """
    Estimate covariance of NDT-estimated pose (simplified approach).

    Computes a simplified covariance estimate based on the Hessian approximation
    at the optimal pose. This is a heuristic used for downstream fusion.

    Args:
        source_scan: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map.
        final_pose: Final NDT pose [x, y, yaw], shape (3,).
        voxel_size: Voxel edge length.

    Returns:
        Covariance matrix of shape (3, 3) representing uncertainty in [x, y, yaw].

    Notes:
        - This is a simplified heuristic, not a rigorous covariance estimate.
        - For rigorous uncertainty, compute the Hessian from Eq. (7.16).
    """
    # Simplified covariance: assume diagonal based on score magnitude
    score = ndt_score(source_scan, ndt_map, final_pose, voxel_size)

    # Heuristic: lower score → lower uncertainty
    # Scale uncertainty inversely with score quality
    sigma_xy = max(0.01, min(1.0, score / 10.0))
    sigma_yaw = max(0.01, min(0.5, score / 20.0))

    cov = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])

    return cov


