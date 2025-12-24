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

Author: Li-Ta Hsu
Date: 2024
"""

from typing import Optional, TYPE_CHECKING

import numpy as np

from ..estimators.factor_graph import Factor
from .se2 import se2_compose, se2_inverse, se2_relative, wrap_angle

if TYPE_CHECKING:
    from .types import CameraIntrinsics


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


def create_reprojection_factor(
    camera_pose_id: int,
    landmark_id: int,
    observed_pixel: np.ndarray,
    camera_intrinsics: "CameraIntrinsics",
    information: Optional[np.ndarray] = None,
) -> Factor:
    """
    Create a reprojection factor for visual bundle adjustment.

    A reprojection factor connects a camera pose and a 3D landmark through
    an image observation (pixel coordinates). It penalizes the difference
    between the observed pixel and the projection of the landmark into the
    camera.

    This is the core constraint in visual SLAM and bundle adjustment.

    Residual:
        r = h(pose, landmark) - observed_pixel
    where h is the camera projection function (Eqs. 7.43-7.46, 7.40).

    Args:
        camera_pose_id: Variable ID of the camera pose.
                       For 2D: pose = [x, y, yaw] (SE(2))
                       For 3D: pose = [x, y, z, qw, qx, qy, qz] (SE(3))
        landmark_id: Variable ID of the 3D landmark [x, y, z] or [x, y] in map frame.
        observed_pixel: Observed pixel coordinates [u, v], shape (2,).
        camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy, distortion).
        information: Information matrix (inverse covariance) for the measurement,
                    shape (2, 2). If None, uses identity (equal weight to u and v).

    Returns:
        Factor encoding the reprojection constraint.

    References:
        Implements the reprojection residual from Eqs. (7.68)-(7.70) in Chapter 7:
            - Eq. (7.68): Bundle adjustment cost function
            - Eq. (7.69): Reprojection error definition
            - Eq. (7.70): Robust kernel (optional, not implemented here)

    Example:
        >>> from core.slam.types import CameraIntrinsics
        >>> intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        >>> observed = np.array([420.5, 340.2])
        >>> factor = create_reprojection_factor(
        ...     camera_pose_id=0, landmark_id=10,
        ...     observed_pixel=observed, camera_intrinsics=intrinsics
        ... )

    Notes:
        - For 2D SLAM: landmark is [x, y], assumes Z=constant or computes from pose
        - For 3D SLAM: landmark is [x, y, z], full 3D projection
        - Information matrix encodes pixel measurement uncertainty (typically ~1 pixel)
        - This factor is used in visual odometry and bundle adjustment
    """
    # Import here to avoid circular dependency
    from . import camera as cam_module

    if information is None:
        # Default: 1 pixel std dev in both u and v
        pixel_std = 1.0
        information = np.eye(2) / (pixel_std**2)

    if observed_pixel.shape != (2,):
        raise ValueError(f"observed_pixel must be shape (2,), got {observed_pixel.shape}")

    def residual_func(variables: list[np.ndarray]) -> np.ndarray:
        """
        Compute reprojection residual.

        For 2D SLAM with fixed height:
            - pose = [x, y, yaw] (SE(2))
            - landmark = [x, y] (2D map coordinates)
            - Assume constant Z or compute from geometry

        For 3D SLAM:
            - pose = [x, y, z, qw, qx, qy, qz] (SE(3) with quaternion)
            - landmark = [x, y, z] (3D map coordinates)
        """
        pose = variables[0]
        landmark = variables[1]

        # For now, implement 2D SLAM version (simpler for teaching)
        # Support both 2D [x, y] and 3D [x, y, z] landmarks
        if pose.shape[0] == 3:  # SE(2): [x, y, yaw]
            # Transform landmark from map frame to camera frame
            x_cam, y_cam, yaw_cam = pose
            
            # Handle both 2D and 3D landmarks
            if landmark.shape[0] == 2:
                lx, ly = landmark
                lz = 0.0  # Assume at camera height for 2D
            elif landmark.shape[0] == 3:
                lx, ly, lz = landmark
            else:
                raise ValueError(f"Landmark must be 2D or 3D, got shape {landmark.shape}")

            # Relative position in map frame
            dx_map = lx - x_cam
            dy_map = ly - y_cam
            dz_map = lz  # Height difference (assume camera at z=0)

            # Rotate to camera frame (camera X-axis points forward along yaw)
            # Camera frame: X-forward, Y-left, Z-up (standard robotics)
            cos_yaw = np.cos(yaw_cam)
            sin_yaw = np.sin(yaw_cam)

            # Transform to camera frame
            x_in_cam = cos_yaw * dx_map + sin_yaw * dy_map
            y_in_cam = -sin_yaw * dx_map + cos_yaw * dy_map
            z_in_cam = dz_map  # Height difference

            # Create 3D point in camera frame
            # Adjust for camera coordinate convention (Z-forward for projection)
            point_camera = np.array([y_in_cam, z_in_cam, x_in_cam])

        elif pose.shape[0] == 7:  # SE(3): [x, y, z, qw, qx, qy, qz]
            # Full 3D case (for future extension)
            raise NotImplementedError("3D SE(3) poses not yet implemented")
        else:
            raise ValueError(f"Unsupported pose dimension: {pose.shape[0]}")

        # Check if landmark is in front of camera
        if point_camera[2] <= 0.1:  # Small margin for numerical stability
            # Behind camera → moderate penalty (not too large for numerical stability)
            return np.array([100.0, 100.0])

        # Project to pixel
        try:
            projected_pixel = cam_module.project_point(camera_intrinsics, point_camera)
        except ValueError:
            # Projection failed → moderate penalty
            return np.array([100.0, 100.0])

        # Residual: projected - observed
        # Eq. (7.69): reprojection error
        residual = projected_pixel - observed_pixel

        return residual

    def jacobian_func(variables: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute Jacobians of reprojection error w.r.t. pose and landmark.

        Returns [J_pose, J_landmark] where:
            - J_pose: (2, pose_dim) Jacobian w.r.t. camera pose
            - J_landmark: (2, landmark_dim) Jacobian w.r.t. 3D landmark

        For now, use numerical differentiation (finite differences).
        For production code, analytical Jacobians are preferred for speed.
        """
        epsilon = 1e-7

        # Numerical Jacobian for pose
        pose = variables[0]
        landmark = variables[1]

        base_residual = residual_func(variables)

        # Jacobian w.r.t. pose
        J_pose = np.zeros((2, pose.shape[0]))
        for i in range(pose.shape[0]):
            pose_perturbed = pose.copy()
            pose_perturbed[i] += epsilon
            residual_perturbed = residual_func([pose_perturbed, landmark])
            J_pose[:, i] = (residual_perturbed - base_residual) / epsilon

        # Jacobian w.r.t. landmark
        J_landmark = np.zeros((2, landmark.shape[0]))
        for i in range(landmark.shape[0]):
            landmark_perturbed = landmark.copy()
            landmark_perturbed[i] += epsilon
            residual_perturbed = residual_func([pose, landmark_perturbed])
            J_landmark[:, i] = (residual_perturbed - base_residual) / epsilon

        return [J_pose, J_landmark]

    # Create and return factor
    factor = Factor(
        variable_ids=[camera_pose_id, landmark_id],
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        information=information,
    )

    return factor

