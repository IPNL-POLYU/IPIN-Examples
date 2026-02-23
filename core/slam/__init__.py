"""SLAM algorithms and geometry for Chapter 7.

This module implements minimal, reusable building blocks for Chapter 7
(SLAM Technologies) of the book: Principles of Indoor Positioning and
Indoor Navigation.

This is NOT a full SLAM framework. Instead, it provides:
    - Geometry & residual models (ICP, NDT, LOAM, camera)
    - Factor graph factors for pose graph and bundle adjustment
    - SE(2) operations for 2D transformations

The nonlinear solvers live in core/estimators (Gauss-Newton / LM / FGO).
Chapter examples in ch7_slam/ wire datasets + factors + solver + plots.

Main components:
    - Pose2, CameraIntrinsics: Core data structures
    - se2_compose, se2_inverse, se2_apply: SE(2) operations
    - icp_point_to_point: ICP scan matching (Section 7.3.1)
    - ndt_align: NDT alignment (Section 7.3.2)
    - create_pose_graph: Pose graph optimization (Section 7.1.2 GraphSLAM)
    - project_point, distort_normalized: Camera projection (Section 7.4.1)
    - create_reprojection_factor: Visual SLAM bundle adjustment (Eq. 7.70, Section 7.4.2)
    - create_odometry_factor, create_loop_closure_factor, create_prior_factor: Factor constructors

Example usage:
    >>> from core.slam import Pose2, se2_compose, se2_inverse, se2_apply
    >>> import numpy as np
    >>> 
    >>> # Create poses
    >>> p1 = Pose2(x=0.0, y=0.0, yaw=0.0)
    >>> p2 = Pose2(x=1.0, y=0.0, yaw=np.pi/2)
    >>> 
    >>> # Compose poses
    >>> p_composed = se2_compose(p1.to_array(), p2.to_array())
    >>> 
    >>> # Transform points
    >>> points = np.array([[1, 0], [0, 1]])
    >>> points_transformed = se2_apply(p2.to_array(), points)

References:
    Chapter 7: Indoor Simultaneous Localization and Mapping (SLAM)
    - Section 7.1.2: SLAM Frameworks and Evolution (GraphSLAM)
    - Section 7.3: LiDAR SLAM
        - Section 7.3.1: Point-cloud based LiDAR SLAM - ICP
        - Section 7.3.2: Feature-based LiDAR SLAM - NDT
        - Section 7.3.5: Close-loop Constraints
    - Section 7.4: Visual SLAM
        - Section 7.4.1: Monocular Camera (camera model)
        - Section 7.4.2: Monocular SLAM (bundle adjustment)

Author: Li-Ta Hsu
Date: December 2025
"""

from . import camera
from .camera import (
    compute_reprojection_error,
    distort_normalized,
    project_point,
    undistort_normalized,
    unproject_pixel,
)
from .factors import (
    create_landmark_factor,
    create_loop_closure_factor,
    create_odometry_factor,
    create_pose_graph,
    create_prior_factor,
    create_reprojection_factor,
)
from .ndt import (
    build_ndt_map,
    ndt_align,
    ndt_covariance,
    ndt_gradient,
    ndt_score,
)
from .scan_matching import (
    align_svd,
    compute_icp_covariance,
    compute_icp_residual,
    find_correspondences,
    icp_point_to_point,
)
from .scan_generation import (
    generate_scan_with_occlusion,
    generate_dense_wall_scan,
    ray_segment_intersection,
)
from .se2 import (
    se2_apply,
    se2_compose,
    se2_from_matrix,
    se2_inverse,
    se2_relative,
    se2_to_matrix,
    wrap_angle,
)
from .frontend_2d import SlamFrontend2D, MatchQuality
from .loop_closure_2d import LoopClosureDetector2D, LoopClosure, LoopClosureCandidate
from .scan_descriptor_2d import (
    compute_scan_descriptor,
    compute_descriptor_similarity,
    batch_compute_descriptors,
)
from .submap_2d import Submap2D
from .types import CameraIntrinsics, PointCloud2D, PointCloud3D, Pose2, VoxelGrid

__all__ = [
    # Core types
    "Pose2",
    "CameraIntrinsics",
    "PointCloud2D",
    "PointCloud3D",
    "VoxelGrid",
    "Submap2D",
    "SlamFrontend2D",
    "MatchQuality",
    "LoopClosureDetector2D",
    "LoopClosure",
    "LoopClosureCandidate",
    "compute_scan_descriptor",
    "compute_descriptor_similarity",
    "batch_compute_descriptors",
    # SE(2) operations
    "se2_compose",
    "se2_inverse",
    "se2_apply",
    "se2_relative",
    "se2_to_matrix",
    "se2_from_matrix",
    "wrap_angle",
    # ICP scan matching
    "find_correspondences",
    "compute_icp_residual",
    "align_svd",
    "icp_point_to_point",
    "compute_icp_covariance",
    # Scan generation
    "generate_scan_with_occlusion",
    "generate_dense_wall_scan",
    "ray_segment_intersection",
    # NDT alignment
    "build_ndt_map",
    "ndt_score",
    "ndt_gradient",
    "ndt_align",
    "ndt_covariance",
    # Camera model and projection
    "camera",
    "distort_normalized",
    "undistort_normalized",
    "project_point",
    "unproject_pixel",
    "compute_reprojection_error",
    # Pose graph factors
    "create_odometry_factor",
    "create_loop_closure_factor",
    "create_prior_factor",
    "create_landmark_factor",
    "create_reprojection_factor",
    "create_pose_graph",
]

__version__ = "0.1.0"

