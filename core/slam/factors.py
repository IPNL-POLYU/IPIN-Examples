"""Factor graph factors for SLAM (pose graph optimization).

This module implements SLAM-specific factors for pose graph optimization
as described in Chapter 7 (SLAM Technologies) of the book: Principles of
Indoor Positioning and Indoor Navigation.

Factors connect poses in a graph and encode constraints from:
    - Odometry: consecutive pose relationships from dead-reckoning
    - Loop closures: non-consecutive pose relationships from scan matching
    - Priors: absolute pose measurements (e.g., GPS, initial pose)

These factors work with the Factor Graph framework from core.estimators.

References:
    - Section 7.3: Pose graph optimization
    - Eqs. (7.68)-(7.70): Reprojection factors (visual SLAM)
    - Factor graphs build on Chapter 3 FGO framework

Author: Navigation Engineer
Date: 2024
"""

from typing import Optional

import numpy as np

from ..estimators.factor_graph import Factor
from .se2 import se2_compose, se2_inverse, se2_relative, wrap_angle


def create_odometry_factor(
    pose_id_from: int,
    pose_id_to: int,
    relative_pose: np.ndarray,
    information: Optional[np.ndarray] = None,
) -> Factor:
    """
    Create odometry factor connecting two consecutive poses.

    An odometry factor encodes the relative motion between two consecutive
    robot poses as measured by wheel encoders, IMU, or other dead-reckoning
    sensors. The factor penalizes deviations from the measured relative pose.

    Residual:
        r = relative_measured ⊖ (pose_from⁻¹ ⊕ pose_to)
    where ⊖ and ⊕ are SE(2) operations.

    Args:
        pose_id_from: Variable ID of the starting pose.
        pose_id_to: Variable ID of the ending pose.
        relative_pose: Measured relative pose [Δx, Δy, Δyaw], shape (3,).
                       This is the motion in the frame of pose_from.
        information: Information matrix (3, 3) for the measurement.
                     If None, uses identity (unit covariance).

    Returns:
        Factor instance for the odometry constraint.

    Examples:
        >>> # Robot moved 1m forward, 0.5m left, rotated 30° left
        >>> rel_pose = np.array([1.0, 0.5, np.pi/6])
        >>> factor = create_odometry_factor(0, 1, rel_pose)
        >>>
        >>> # With covariance (lower uncertainty in x than y)
        >>> cov = np.diag([0.01, 0.04, 0.001])  # σ_x²=0.01, σ_y²=0.04, σ_yaw²=0.001
        >>> info = np.linalg.inv(cov)
        >>> factor = create_odometry_factor(0, 1, rel_pose, information=info)

    Notes:
        - Odometry factors form the "backbone" of the pose graph.
        - They connect consecutive poses in chronological order.
        - Information matrix encodes measurement uncertainty.
        - For typical wheel odometry: σ_x < σ_y (forward more accurate than lateral).
    """
    if information is None:
        information = np.eye(3)

    def residual_func(x_vars):
        """Compute SE(2) relative pose residual."""
        pose_from = x_vars[0]  # shape (3,)
        pose_to = x_vars[1]  # shape (3,)

        # Compute actual relative pose: pose_from⁻¹ ⊕ pose_to
        relative_actual = se2_relative(pose_from, pose_to)

        # Residual: difference between measured and actual
        residual = relative_pose - relative_actual

        # Wrap yaw difference to [-π, π]
        residual[2] = wrap_angle(residual[2])

        return residual

    def jacobian_func(x_vars):
        """Compute Jacobians with respect to both poses."""
        pose_from = x_vars[0]
        pose_to = x_vars[1]

        # Numerical Jacobians (finite differences)
        epsilon = 1e-7

        # Jacobian w.r.t. pose_from
        J_from = np.zeros((3, 3))
        for i in range(3):
            pose_from_plus = pose_from.copy()
            pose_from_plus[i] += epsilon
            r_plus = residual_func([pose_from_plus, pose_to])
            r_base = residual_func([pose_from, pose_to])
            J_from[:, i] = (r_plus - r_base) / epsilon

        # Jacobian w.r.t. pose_to
        J_to = np.zeros((3, 3))
        for i in range(3):
            pose_to_plus = pose_to.copy()
            pose_to_plus[i] += epsilon
            r_plus = residual_func([pose_from, pose_to_plus])
            r_base = residual_func([pose_from, pose_to])
            J_to[:, i] = (r_plus - r_base) / epsilon

        return [J_from, J_to]

    return Factor([pose_id_from, pose_id_to], residual_func, jacobian_func, information)


def create_loop_closure_factor(
    pose_id_from: int,
    pose_id_to: int,
    relative_pose: np.ndarray,
    information: Optional[np.ndarray] = None,
) -> Factor:
    """
    Create loop closure factor connecting non-consecutive poses.

    A loop closure factor is similar to an odometry factor but connects
    non-consecutive poses that have been matched via scan matching (ICP/NDT)
    or place recognition. Loop closures are critical for reducing drift in
    long trajectories.

    The factor structure is identical to odometry factors, but they connect
    poses that are far apart in time. The relative pose is typically obtained
    from ICP or NDT alignment.

    Residual:
        r = relative_measured ⊖ (pose_from⁻¹ ⊕ pose_to)

    Args:
        pose_id_from: Variable ID of the first pose (earlier in time).
        pose_id_to: Variable ID of the second pose (later in time, or any other pose).
        relative_pose: Measured relative pose from scan matching [Δx, Δy, Δyaw], shape (3,).
        information: Information matrix (3, 3). If None, uses identity.
                     Should typically come from ICP/NDT covariance estimate.

    Returns:
        Factor instance for the loop closure constraint.

    Examples:
        >>> # Loop closure detected: pose 100 matches pose 10
        >>> # Scan matching gives relative pose and covariance
        >>> rel_pose = np.array([0.2, 0.1, 0.05])  # Small difference (good closure)
        >>> cov = np.diag([0.05, 0.05, 0.01])  # Scan matching uncertainty
        >>> info = np.linalg.inv(cov)
        >>> factor = create_loop_closure_factor(10, 100, rel_pose, information=info)

    Notes:
        - Loop closures "bend" the trajectory to close loops.
        - They significantly reduce accumulated drift.
        - Information matrix should reflect scan matching quality.
        - For ambiguous closures, use lower information (higher uncertainty).
        - This is the same as odometry factor but conceptually different.
    """
    # Loop closure factor is structurally identical to odometry factor
    return create_odometry_factor(pose_id_from, pose_id_to, relative_pose, information)


def create_prior_factor(
    pose_id: int,
    prior_pose: np.ndarray,
    information: Optional[np.ndarray] = None,
) -> Factor:
    """
    Create prior factor anchoring a pose to a known value.

    A prior factor constrains a single pose to a measured or assumed value.
    This is used to:
        - Fix the first pose in the trajectory (anchor the origin)
        - Incorporate GPS or other absolute position measurements
        - Prevent gauge freedom (entire trajectory drifting/rotating)

    Residual:
        r = prior_pose - pose

    Args:
        pose_id: Variable ID of the pose to constrain.
        prior_pose: Prior pose value [x, y, yaw], shape (3,).
        information: Information matrix (3, 3). If None, uses identity.
                     For a strong prior (e.g., fixed origin), use large information.
                     For weak prior (e.g., noisy GPS), use small information.

    Returns:
        Factor instance for the prior constraint.

    Examples:
        >>> # Fix first pose at origin (strong prior)
        >>> prior = np.array([0.0, 0.0, 0.0])
        >>> info = np.diag([1e6, 1e6, 1e6])  # Very strong prior
        >>> factor = create_prior_factor(0, prior, information=info)
        >>>
        >>> # Weak GPS measurement (loose prior)
        >>> gps_pose = np.array([123.4, 567.8, 0.0])  # Unknown heading
        >>> cov = np.diag([4.0, 4.0, 1e6])  # σ_xy=2m, no yaw info
        >>> info = np.linalg.inv(cov)
        >>> factor = create_prior_factor(5, gps_pose, information=info)

    Notes:
        - At least one prior is usually needed to anchor the coordinate frame.
        - Without priors, the graph has gauge freedom (can slide/rotate).
        - For SLAM, typically fix the first pose with strong prior.
        - For GPS-aided SLAM, add weak priors at GPS measurement points.
    """
    if information is None:
        information = np.eye(3)

    def residual_func(x_vars):
        """Compute prior residual."""
        pose = x_vars[0]  # shape (3,)

        # Residual: difference from prior
        residual = prior_pose - pose

        # Wrap yaw difference to [-π, π]
        residual[2] = wrap_angle(residual[2])

        return residual

    def jacobian_func(x_vars):
        """Compute Jacobian: ∂r/∂pose = -I."""
        # Residual = prior - pose, so ∂r/∂pose = -I
        J = -np.eye(3)
        return [J]

    return Factor([pose_id], residual_func, jacobian_func, information)


def create_landmark_factor(
    pose_id: int,
    landmark_id: int,
    observation: np.ndarray,
    information: Optional[np.ndarray] = None,
) -> Factor:
    """
    Create landmark observation factor (for feature-based SLAM).

    A landmark factor encodes the observation of a 2D landmark from a robot
    pose. The observation is in the robot's local frame (e.g., bearing and
    range from a LiDAR scan, or pixel coordinates from a camera).

    This is used in feature-based SLAM where both poses and landmark positions
    are jointly optimized.

    Residual:
        r = observation - h(pose, landmark)
    where h() is the observation model transforming landmark to local frame.

    Args:
        pose_id: Variable ID of the observing pose.
        landmark_id: Variable ID of the landmark position.
        observation: Observed landmark in local frame [x_local, y_local], shape (2,).
        information: Information matrix (2, 2). If None, uses identity.

    Returns:
        Factor instance for the landmark observation.

    Examples:
        >>> # Robot at pose 0 observes landmark 100 at [2m, 1m] in local frame
        >>> obs = np.array([2.0, 1.0])
        >>> cov = np.diag([0.01, 0.01])  # 10cm measurement noise
        >>> info = np.linalg.inv(cov)
        >>> factor = create_landmark_factor(0, 100, obs, information=info)

    Notes:
        - Used in feature-based SLAM (not scan matching SLAM).
        - Landmark variables are also optimized (not just poses).
        - Observation model: transforms global landmark to robot frame.
        - For bearing-only: observation is 1D angle.
        - For range-bearing: observation is [range, bearing].
        - For Cartesian: observation is [x_local, y_local].
    """
    if information is None:
        information = np.eye(2)

    def residual_func(x_vars):
        """Compute landmark observation residual."""
        pose = x_vars[0]  # shape (3,): [x, y, yaw]
        landmark = x_vars[1]  # shape (2,): [lx, ly]

        # Transform landmark from global to robot local frame
        # 1. Translate to robot frame
        diff = landmark - pose[:2]

        # 2. Rotate to robot heading
        cos_yaw = np.cos(pose[2])
        sin_yaw = np.sin(pose[2])
        x_local = cos_yaw * diff[0] + sin_yaw * diff[1]
        y_local = -sin_yaw * diff[0] + cos_yaw * diff[1]

        predicted_obs = np.array([x_local, y_local])

        # Residual
        residual = observation - predicted_obs

        return residual

    def jacobian_func(x_vars):
        """Compute Jacobians with respect to pose and landmark."""
        pose = x_vars[0]
        landmark = x_vars[1]

        # Numerical Jacobians
        epsilon = 1e-7

        # Jacobian w.r.t. pose (3D)
        J_pose = np.zeros((2, 3))
        for i in range(3):
            pose_plus = pose.copy()
            pose_plus[i] += epsilon
            r_plus = residual_func([pose_plus, landmark])
            r_base = residual_func([pose, landmark])
            J_pose[:, i] = (r_plus - r_base) / epsilon

        # Jacobian w.r.t. landmark (2D)
        J_landmark = np.zeros((2, 2))
        for i in range(2):
            landmark_plus = landmark.copy()
            landmark_plus[i] += epsilon
            r_plus = residual_func([pose, landmark_plus])
            r_base = residual_func([pose, landmark])
            J_landmark[:, i] = (r_plus - r_base) / epsilon

        return [J_pose, J_landmark]

    return Factor([pose_id, landmark_id], residual_func, jacobian_func, information)


def create_pose_graph(
    poses: list[np.ndarray],
    odometry_measurements: list[tuple[int, int, np.ndarray]],
    loop_closures: Optional[list[tuple[int, int, np.ndarray]]] = None,
    prior_pose: Optional[np.ndarray] = None,
    odometry_information: Optional[np.ndarray] = None,
    loop_information: Optional[np.ndarray] = None,
    prior_information: Optional[np.ndarray] = None,
):
    """
    Create a complete pose graph from trajectory data.

    Convenience function to build a factor graph for pose graph SLAM with
    odometry, loop closures, and an optional prior on the first pose.

    Args:
        poses: List of initial pose estimates [x, y, yaw] for each timestep.
        odometry_measurements: List of (from_id, to_id, relative_pose) tuples
                               for odometry constraints.
        loop_closures: Optional list of (from_id, to_id, relative_pose) tuples
                       for loop closure constraints.
        prior_pose: Optional prior for first pose. If None, uses identity with strong prior.
        odometry_information: Information matrix for all odometry factors.
                              If None, uses identity.
        loop_information: Information matrix for all loop closure factors.
                          If None, uses identity.
        prior_information: Information matrix for prior factor.
                           If None, uses strong prior (1e6 * I).

    Returns:
        FactorGraph instance ready for optimization.

    Examples:
        >>> # Simple 3-pose trajectory
        >>> poses = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])]
        >>> odom = [(0, 1, np.array([1, 0, 0])), (1, 2, np.array([1, 0, 0]))]
        >>> graph = create_pose_graph(poses, odom)
        >>> optimized, _ = graph.optimize()

    Notes:
        - This is a high-level convenience function.
        - For more control, build FactorGraph manually with individual factors.
        - Typical workflow:
            1. Run SLAM with odometry → initial trajectory
            2. Detect loop closures via scan matching
            3. Build pose graph with closures
            4. Optimize → corrected trajectory
    """
    from ..estimators.factor_graph import FactorGraph

    graph = FactorGraph()

    # Add pose variables
    for i, pose in enumerate(poses):
        graph.add_variable(i, pose)

    # Add prior on first pose
    if prior_pose is None:
        prior_pose = np.array([0.0, 0.0, 0.0])
    if prior_information is None:
        prior_information = np.diag([1e6, 1e6, 1e6])  # Strong prior

    prior_factor = create_prior_factor(0, prior_pose, prior_information)
    graph.add_factor(prior_factor)

    # Add odometry factors
    if odometry_information is None:
        odometry_information = np.eye(3)

    for from_id, to_id, rel_pose in odometry_measurements:
        factor = create_odometry_factor(
            from_id, to_id, rel_pose, information=odometry_information
        )
        graph.add_factor(factor)

    # Add loop closure factors
    if loop_closures is not None:
        if loop_information is None:
            loop_information = np.eye(3)

        for from_id, to_id, rel_pose in loop_closures:
            factor = create_loop_closure_factor(
                from_id, to_id, rel_pose, information=loop_information
            )
            graph.add_factor(factor)

    return graph

