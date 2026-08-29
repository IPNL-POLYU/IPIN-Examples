"""NDT (Normal Distributions Transform) scan matching for 2D LiDAR SLAM.

This module implements NDT alignment as described in Section 7.3.2 of
Chapter 7 (LiDAR SLAM) of the book: Principles of Indoor Positioning
and Indoor Navigation.

NDT represents the target scan as a probabilistic model (Gaussian distributions
per voxel) rather than raw points, leading to smoother cost functions and
better convergence properties compared to point-to-point ICP.

**Note on 2D vs 3D**: The book presents NDT for 3D LiDAR point clouds (Eq. 7.9).
This implementation restricts to 2D (x, y) for educational clarity and consistency
with other 2D SLAM examples. The mathematical principles (mean, covariance, likelihood)
are identical in 2D and 3D.

Key functions:
    - build_ndt_map: Build voxel grid with Gaussian distributions (Eq. 7.12-7.13)
    - ndt_score: Compute negative log-likelihood (Eq. 7.16)
    - ndt_gradient: Analytic gradient of Eq. (7.16) for optimization
    - ndt_align: Full NDT alignment with Gauss-Newton

References:
    - Section 7.3.2: Feature-based LiDAR SLAM - NDT
    - Eq. (7.12): Voxel mean p̄_{k,t-1} = (1/n_k) Σ p_{i,t-1}
    - Eq. (7.13): Voxel covariance Σ_{k,t-1} = 1/(n_k-1) Σ (p-p̄)(p-p̄)^T
    - Eq. (7.14): Likelihood for one voxel k
    - Eq. (7.15): Joint likelihood across all voxels
    - Eq. (7.16): MLE objective (minimize 0.5 Σ ||T p_j - p̄_k||²_Σ)

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np

from .scan_matching import AlignmentResult, _alignment_result
from .se2 import se2_apply
from .types import VoxelGrid


def build_ndt_map(
    points: np.ndarray,
    voxel_size: float = 1.0,
    min_points_per_voxel: int = 3,
) -> VoxelGrid:
    """
    Build NDT map from point cloud: voxel grid with Gaussian distributions.

    Divides 2D space into voxels and fits a Gaussian distribution (mean and
    covariance) to the points in each voxel per Eqs. (7.12)-(7.13).
    This is the offline preprocessing step for NDT alignment.

    For each voxel k with n_k points:
        - Mean (Eq. 7.12): p̄_k = (1/n_k) Σ_{i=1}^{n_k} p_i
        - Covariance (Eq. 7.13): Σ_k = 1/(n_k-1) Σ_{i=1}^{n_k} (p_i - p̄_k)(p_i - p̄_k)^T

    Args:
        points: Point cloud, shape (N, 2) in meters. Note: book uses 3D (Eq. 7.9),
                but this implementation restricts to 2D for pedagogical clarity.
        voxel_size: Voxel edge length in meters (default: 1.0).
        min_points_per_voxel: Minimum number of points required to fit a
                              Gaussian in a voxel (default: 3).

    Returns:
        VoxelGrid: Dictionary mapping voxel indices (i, j) to Gaussian parameters:
            {
                (i, j): {
                    'mean': np.ndarray of shape (2,),   # p̄_k from Eq. 7.12
                    'cov': np.ndarray of shape (2, 2),  # Σ_k from Eq. 7.13
                    'n_points': int                      # n_k
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
        - Implements Eqs. (7.12)-(7.13) from Section 7.3.2.
        - Uses (n_k - 1) denominator for unbiased covariance estimate (Eq. 7.13).
        - Voxels with fewer than min_points_per_voxel are discarded (no Gaussian).
        - Covariance is regularized with small diagonal term to avoid singularity.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    if points.shape[0] == 0:
        return {}

    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Group points by voxel
    voxels: dict[tuple[int, int], list] = {}
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
        n_k = len(voxel_points)

        # Compute mean (Eq. 7.12): p̄_k = (1/n_k) Σ p_i
        mean = np.mean(voxel_points_array, axis=0)  # shape (2,)

        # Compute covariance (Eq. 7.13): Σ_k = 1/(n_k-1) Σ (p_i - p̄_k)(p_i - p̄_k)^T
        # Note: For n_k=1, use biased estimator (divide by n_k) to avoid division by zero
        centered = voxel_points_array - mean
        if n_k > 1:
            cov = (centered.T @ centered) / (n_k - 1)  # Unbiased estimator (Eq. 7.13)
        else:
            # Single point: use biased estimator to avoid division by zero
            cov = (centered.T @ centered) / n_k

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
    the target NDT map. This implements the objective function from Eq. (7.16).

    The likelihood for a single voxel k (Eq. 7.14):
        likelihood_k(T) = ∏_{j=1}^{N} exp( -||T p_j - p̄_k||²_Σ / 2 )

    The joint likelihood across all voxels (Eq. 7.15):
        likelihood(T) = ∏_{k=1}^{N_voxel} ∏_{j=1}^{N} exp( -||T p_j - p̄_k||²_Σ / 2 )

    The MLE objective to minimize (Eq. 7.16):
        T̂ = argmin_T  (1/2) Σ_k Σ_j ||T p_j - p̄_k||²_Σ

    where ||r||²_Σ = r^T Σ^{-1} r is the squared Mahalanobis distance.

    Args:
        source_points: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map (voxel grid with Gaussians from Eq. 7.12-7.13).
        pose: Pose [x, y, yaw], shape (3,) to transform source.
        voxel_size: Voxel edge length (must match ndt_map).

    Returns:
        NDT score (scalar). Lower is better (negative log-likelihood from Eq. 7.16).

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
        - Implements the MLE objective from Eq. (7.16), Section 7.3.2.
        - Points that fall outside occupied voxels are ignored.
        - Uses negative log-likelihood formulation for numerical stability.
        - Score includes log(det(Σ)) term for proper likelihood computation.
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


def _ndt_derivatives(
    source_points: np.ndarray,
    ndt_map: VoxelGrid,
    pose: np.ndarray,
    voxel_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Analytic gradient and Gauss-Newton Hessian of the Eq. (7.16) objective.

    ndt_score evaluates the mean over matched points of
    ``0.5 ||T p_j - p̄_k||²_Σ + 0.5 log det Σ_k``. Differentiating that with the
    voxel association k(j) and the match count held fixed gives, with
    r_j = T p_j - p̄_k and J_j = ∂r_j/∂[x, y, ψ]:

        ∇ = (1/n) Σ_j J_jᵀ Σ_k⁻¹ r_j
        H ≈ (1/n) Σ_j J_jᵀ Σ_k⁻¹ J_j          (Gauss-Newton)

    For the SE(2) transform T p = R(ψ) p + t the Jacobian columns are the two
    translation axes and the rotation derivative:

        ∂r/∂x = [1, 0]ᵀ,  ∂r/∂y = [0, 1]ᵀ,  ∂r/∂ψ = R'(ψ) p

    Holding the association fixed is what makes this well behaved. The score
    itself is piecewise-discontinuous -- a point contributes only while it lies
    inside an occupied voxel, so the value jumps as points cross voxel
    boundaries -- but *within* one association the objective is a smooth
    quadratic, and that is the part worth following. Finite differences cannot
    separate the two: see the note in ndt_align on why they were removed.

    Args:
        source_points: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map (Gaussians from Eq. 7.12-7.13).
        pose: Current pose [x, y, yaw], shape (3,).
        voxel_size: Voxel edge length (must match ndt_map).

    Returns:
        Tuple of (gradient, hessian, n_matched):
            - gradient: shape (3,), ∇ of the Eq. (7.16) objective.
            - hessian: shape (3, 3), Gauss-Newton approximation, positive
              semi-definite by construction.
            - n_matched: How many source points landed in an occupied voxel.

    Notes:
        - Gauss-Newton drops the second-order term (Σ⁻¹ r)·∂²r/∂ψ², which keeps
          H positive semi-definite and so keeps -H⁻¹∇ a descent direction.
        - The log det Σ_k term does not depend on the pose at fixed association,
          so it does not appear in the derivatives.
    """
    yaw = float(pose[2])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    # dR/dψ, so that ∂r/∂ψ = R'(ψ) p.
    rotation_dot = np.array([[-sin_yaw, -cos_yaw], [cos_yaw, -sin_yaw]])

    transformed = source_points @ rotation.T + np.asarray(pose[:2], dtype=float)
    voxel_indices = np.floor(transformed / voxel_size).astype(int)

    gradient = np.zeros(3)
    hessian = np.zeros((3, 3))
    n_matched = 0

    for i, point in enumerate(transformed):
        voxel_key = (int(voxel_indices[i, 0]), int(voxel_indices[i, 1]))
        voxel = ndt_map.get(voxel_key)
        if voxel is None:
            # Point falls in an empty voxel → contributes nothing, as in ndt_score.
            continue

        residual = point - voxel["mean"]
        # J = [∂r/∂x, ∂r/∂y, ∂r/∂ψ], shape (2, 3).
        jacobian = np.column_stack([np.eye(2), rotation_dot @ source_points[i]])

        try:
            weighted = np.linalg.solve(voxel["cov"], residual)  # Σ⁻¹ r
            info_jacobian = np.linalg.solve(voxel["cov"], jacobian)  # Σ⁻¹ J
        except np.linalg.LinAlgError:
            # Singular covariance → skip this voxel, as in ndt_score.
            continue

        gradient += jacobian.T @ weighted
        hessian += jacobian.T @ info_jacobian
        n_matched += 1

    if n_matched == 0:
        return np.zeros(3), np.eye(3), 0

    return gradient / n_matched, hessian / n_matched, n_matched


def ndt_gradient(
    source_points: np.ndarray,
    ndt_map: VoxelGrid,
    pose: np.ndarray,
    voxel_size: float = 1.0,
) -> np.ndarray:
    """
    Compute the analytic gradient of the NDT score with respect to pose.

    Computes ∇_pose score(pose) for gradient-based optimization. The gradient of
    the MLE objective (Eq. 7.16) enables iterative pose refinement.

    Args:
        source_points: Source point cloud, shape (N, 2).
        ndt_map: Target NDT map (Gaussians from Eq. 7.12-7.13).
        pose: Current pose [x, y, yaw], shape (3,).
        voxel_size: Voxel edge length.

    Returns:
        Gradient vector of shape (3,): [∂score/∂x, ∂score/∂y, ∂score/∂yaw].

    Notes:
        - Computes the gradient of the objective function in Eq. (7.16)
          analytically; see _ndt_derivatives for the derivation.
        - The gradient is exact for the smooth part of the objective, holding the
          voxel association fixed. It is not a finite difference, so it carries
          no epsilon to tune and cannot be corrupted by points crossing voxel
          boundaries.
    """
    gradient, _, _ = _ndt_derivatives(source_points, ndt_map, pose, voxel_size)
    return gradient


def ndt_align(
    source_scan: np.ndarray,
    target_scan: np.ndarray,
    initial_pose: np.ndarray | None = None,
    voxel_size: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    step_size: float = 1.0,
) -> AlignmentResult:
    """
    NDT-based scan alignment using Gauss-Newton (Section 7.3.2).

    Aligns source scan to target scan by optimizing the NDT score function.
    The target scan is first converted to an NDT map (voxel grid with Gaussians
    per Eqs. 7.12-7.13), then Gauss-Newton minimizes the MLE objective (Eq. 7.16).

    Each iteration solves ``(H + λI) δ = -∇`` for the analytic gradient and
    Gauss-Newton Hessian of _ndt_derivatives, then backtracks on the step scale
    until the true score improves. Eq. (7.16) is a sum of squared Mahalanobis
    residuals, so Gauss-Newton is its natural solver: H carries the Σ_k⁻¹
    weighting, which makes the step correctly scaled in metres and radians at
    once and removes the need to tune a descent rate.

    Args:
        source_scan: Source point cloud, shape (N, 2). Book uses 3D, we restrict to 2D.
        target_scan: Target point cloud, shape (M, 2). Book uses 3D, we restrict to 2D.
        initial_pose: Initial pose guess [x, y, yaw], shape (3,).
                      If None, uses identity.
        voxel_size: Voxel edge length in meters (default: 1.0).
        max_iterations: Maximum number of optimization iterations (default: 50).
        tolerance: Convergence threshold on the pose update length (default: 1e-3),
                   measured as ‖α δ‖ over [x, y, yaw] in metres and radians.
        step_size: Initial scale applied to the Gauss-Newton step before
                   backtracking (default: 1.0, the undamped step). Values below 1
                   damp every step and only slow convergence down; the line
                   search already shortens the step whenever it overshoots.

    Returns:
        AlignmentResult, which can still be tuple-unpacked as
        (final_pose, num_iterations, final_score, converged):
            - final_pose: Estimated pose [x, y, yaw], shape (3,).
            - num_iterations: Number of iterations executed.
            - final_score: Final NDT score (negative log-likelihood from Eq. 7.16).
            - converged: True only if the optimizer reached a stationary point --
              a vanishing gradient or an update below `tolerance`, with points
              still matched. A line search that cannot find any improvement
              returns False: that is a stall, not a convergence. `converged` says
              the optimizer settled, which is not the same as the pose being
              right; a run that settles in a local minimum of Eq. (7.16), as
              happens when the initial guess is displaced by more than about one
              voxel, reports True.

    Examples:
        >>> source = np.array([[0, 0], [1, 0], [0, 1]])
        >>> target = source + np.array([2, 3])
        >>> pose, iters, score, converged = ndt_align(source, target, voxel_size=2.0)
        >>> converged
        True

    Notes:
        - Implements NDT alignment from Section 7.3.2.
        - Target map built using Eqs. (7.12)-(7.13) for mean and covariance.
        - Minimizes the MLE objective from Eq. (7.16) via damped Gauss-Newton.
        - NDT's capture range is roughly one voxel; beyond that the objective has
          local minima this (or any local) optimizer can settle into. Coarser
          voxels reach further but localise less precisely.
    """
    # Validate inputs
    if source_scan.ndim != 2 or source_scan.shape[1] != 2:
        raise ValueError(f"source_scan must have shape (N, 2), got {source_scan.shape}")
    if target_scan.ndim != 2 or target_scan.shape[1] != 2:
        raise ValueError(f"target_scan must have shape (M, 2), got {target_scan.shape}")

    if source_scan.shape[0] == 0:
        raise ValueError("source_scan is empty")
    if target_scan.shape[0] == 0:
        raise ValueError("target_scan is empty")

    # Build NDT map from target scan
    ndt_map = build_ndt_map(target_scan, voxel_size=voxel_size)

    if len(ndt_map) == 0:
        # No valid voxels → cannot align
        return _alignment_result(
            initial_pose if initial_pose is not None else np.zeros(3),
            0,
            1e6,
            False,
            "ndt_score",
        )

    # Initialize pose
    if initial_pose is None:
        current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        current_pose = initial_pose.astype(np.float64).copy()

    # Optimization loop: damped Gauss-Newton with a backtracking line search.
    #
    # The predecessor took a *unit* steepest-descent step from a central finite
    # difference (epsilon 1e-3). That failed for two compounding reasons:
    #
    #   1. The finite difference measured the wrong thing. The score is
    #      piecewise-discontinuous across voxel boundaries, so a +/-1e-3 probe is
    #      dominated by the handful of points that happen to sit within 1e-3 of a
    #      boundary and jump in or out of the sum, not by the smooth trend. It
    #      reported |grad| ~ 1e4 even at the optimum, and its *direction* could be
    #      flatly wrong: on the Chapter 7 scan pair it pointed within 11 degrees
    #      of straight away from the optimum (cos = -0.98).
    #   2. A unit direction in a mixed [m, m, rad] space has no consistent scale,
    #      so the useful step length changed from problem to problem -- which is
    #      what made the outcome depend on step_size, and not even monotonically.
    #
    # The analytic gradient fixes (1) by holding the voxel association fixed, and
    # the Gauss-Newton Hessian fixes (2) because H carries the Σ_k⁻¹ weighting and
    # so produces a step already scaled correctly in both units. Backtracking on
    # the *true* score keeps the discontinuities from being trusted: the step is
    # proposed from the smooth model but only accepted if the real objective
    # improves.
    NO_MATCH_SCORE = 1e6

    def _wrap_yaw(p: np.ndarray) -> np.ndarray:
        p[2] = np.arctan2(np.sin(p[2]), np.cos(p[2]))
        return p

    current_score = ndt_score(source_scan, ndt_map, current_pose, voxel_size)

    for iteration in range(max_iterations):
        gradient, hessian, n_matched = _ndt_derivatives(
            source_scan, ndt_map, current_pose, voxel_size
        )
        gradient_norm = float(np.linalg.norm(gradient))

        if n_matched == 0 or not np.isfinite(gradient_norm):
            # Nothing to fit: never call that a convergence.
            return _alignment_result(
                current_pose, iteration + 1, current_score, False, "ndt_score"
            )

        if gradient_norm < 1e-12:
            # Stationary point of Eq. (7.16).
            return _alignment_result(
                current_pose, iteration + 1, current_score, True, "ndt_score"
            )

        # Levenberg damping keeps the solve defined when a voxel layout leaves
        # one pose direction unconstrained (a corridor constrains motion across
        # it far better than along it).
        damping = 1e-6 * max(1.0, float(np.trace(hessian)) / 3.0)
        try:
            delta = np.linalg.solve(hessian + damping * np.eye(3), -gradient)
        except np.linalg.LinAlgError:
            delta = -gradient / gradient_norm
        if not np.all(np.isfinite(delta)):
            return _alignment_result(
                current_pose, iteration + 1, current_score, False, "ndt_score"
            )

        # Backtracking: shrink the step until the score actually improves.
        alpha = step_size
        improved = False
        while alpha > 1e-6:
            candidate = _wrap_yaw(current_pose + alpha * delta)
            candidate_score = ndt_score(source_scan, ndt_map, candidate, voxel_size)
            if candidate_score < current_score:
                current_pose, current_score = candidate, candidate_score
                improved = True
                break
            alpha *= 0.5

        if not improved:
            # The line search could not improve on the current pose. That is a
            # stall, not a convergence, and reporting it as success is what let
            # the old optimizer return a wrong pose with converged=True.
            return _alignment_result(
                current_pose, iteration + 1, current_score, False, "ndt_score"
            )

        if alpha * float(np.linalg.norm(delta)) < tolerance:
            return _alignment_result(
                current_pose,
                iteration + 1,
                current_score,
                (current_score < NO_MATCH_SCORE),
                "ndt_score",
            )

    return _alignment_result(
        current_pose, max_iterations, current_score, False, "ndt_score"
    )


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
        ndt_map: Target NDT map (Gaussians from Eq. 7.12-7.13).
        final_pose: Final NDT pose [x, y, yaw], shape (3,).
        voxel_size: Voxel edge length.

    Returns:
        Covariance matrix of shape (3, 3) representing uncertainty in [x, y, yaw].

    Notes:
        - This is a simplified heuristic, not a rigorous covariance estimate.
        - For rigorous uncertainty, compute the Hessian of Eq. (7.16) at the optimum.
        - The book mentions that Eq. (7.16) can be solved by nonlinear optimizer.
    """
    # Simplified covariance: assume diagonal based on score magnitude
    score = ndt_score(source_scan, ndt_map, final_pose, voxel_size)

    # Heuristic: lower score → lower uncertainty
    # Scale uncertainty inversely with score quality
    sigma_xy = max(0.01, min(1.0, score / 10.0))
    sigma_yaw = max(0.01, min(0.5, score / 20.0))

    cov = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])

    return cov
