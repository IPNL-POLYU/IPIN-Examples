"""SLAM Front-End: Prediction, Scan-to-Map Alignment, and Map Update.

This module implements an online SLAM front-end that explicitly demonstrates
the core loop of SLAM:
    1. Prediction: Apply odometry delta to estimate pose
    2. Correction: Refine pose via scan-to-map matching (ICP)
    3. Map Update: Add scan to local submap with refined pose

This replaces oracle-based odometry with observation-driven pose estimation.

Author: Li-Ta Hsu
Date: December 2025
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from .scan_matching import icp_point_to_point
from .se2 import se2_compose
from .submap_2d import Submap2D


@dataclass
class MatchQuality:
    """Quality metrics for scan-to-map alignment.

    Attributes:
        residual_rmse_m: ICP alignment RMS residual in metres (lower is better).
        converged: Whether ICP converged.
        n_correspondences: Number of point correspondences found.
        iterations: Number of ICP iterations performed.
    """

    residual_rmse_m: float
    converged: bool
    n_correspondences: int
    iterations: int

    @property
    def residual(self) -> float:
        """Backward-compatible alias for ``residual_rmse_m``."""
        return self.residual_rmse_m

    @property
    def iters(self) -> int:
        """Backward-compatible alias for ``iterations``."""
        return self.iterations


@dataclass(frozen=True)
class FrontendStepResult(Mapping[str, object]):
    """Named result for one front-end update with dict-style compatibility.

    Attributes:
        predicted_pose_body_to_map_se2: Odometry-predicted pose ``[x_m, y_m,
            yaw_rad]`` that transforms body-frame points into the map frame.
        estimated_pose_body_to_map_se2: Scan-matched pose with the same shape,
            units, and transform direction.
        match_quality: ICP convergence and residual details.
        correction_magnitude_m: Translation distance between predicted and
            estimated poses, in metres.
    """

    predicted_pose_body_to_map_se2: np.ndarray
    estimated_pose_body_to_map_se2: np.ndarray
    match_quality: MatchQuality
    correction_magnitude_m: float

    _KEYS = ("pose_pred", "pose_est", "match_quality", "correction_magnitude")
    _KEY_TO_ATTRIBUTE = {
        "pose_pred": "predicted_pose_body_to_map_se2",
        "pose_est": "estimated_pose_body_to_map_se2",
        "match_quality": "match_quality",
        "correction_magnitude": "correction_magnitude_m",
    }

    def __getitem__(self, key: str) -> object:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, self._KEY_TO_ATTRIBUTE[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def as_dict(self) -> dict[str, object]:
        """Return a plain dict for callers that need concrete dict semantics."""
        return {key: self[key] for key in self._KEYS}

    @property
    def pose_pred(self) -> np.ndarray:
        """Backward-compatible alias for the predicted body-to-map pose."""
        return self.predicted_pose_body_to_map_se2

    @property
    def pose_est(self) -> np.ndarray:
        """Backward-compatible alias for the estimated body-to-map pose."""
        return self.estimated_pose_body_to_map_se2

    @property
    def correction_magnitude(self) -> float:
        """Backward-compatible alias for ``correction_magnitude_m``."""
        return self.correction_magnitude_m


class SlamFrontend2D:
    """Online SLAM front-end with scan-to-map alignment.

    Implements the core SLAM loop:
        1. Predict pose from odometry delta
        2. Refine pose via scan-to-map ICP alignment
        3. Update local submap with refined pose

    This front-end maintains a local submap (sliding window of recent scans)
    and uses it to correct odometry drift through scan-to-map matching.

    Attributes:
        submap: Local submap (Submap2D) storing accumulated scans.
        pose_est: Current pose estimate in map frame [x, y, yaw].
        initialized: Whether front-end has been initialized.
        voxel_size: Voxel grid size for submap downsampling (meters).
        min_map_points: Minimum map points required for ICP.
        max_icp_residual: Maximum alignment error to accept, as RMS distance
            per correspondence in meters.
        max_correspondence_distance: ICP correspondence gate in meters
            (d_threshold in Eq. 7.11), or None to disable gating.

    Example:
        >>> frontend = SlamFrontend2D(submap_voxel_size=0.1)
        >>>
        >>> # First step (initialization)
        >>> odom_delta = np.array([0.0, 0.0, 0.0])
        >>> scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> result = frontend.step(0, odom_delta, scan)
        >>>
        >>> # Subsequent steps
        >>> odom_delta = np.array([0.1, 0.0, 0.0])  # Move 0.1m forward
        >>> scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> result = frontend.step(1, odom_delta, scan)
        >>>
        >>> print(f"Predicted: {result['pose_pred']}")
        >>> print(f"Estimated: {result['pose_est']}")
        >>> print(f"Converged: {result['match_quality'].converged}")
    """

    def __init__(
        self,
        submap_voxel_size: float = 0.1,
        min_map_points: int = 10,
        max_icp_residual: float = 1.0,
        initial_pose: np.ndarray | None = None,
        max_correspondence_distance: float | None = 1.0,
    ):
        """Initialize SLAM front-end.

        Args:
            submap_voxel_size: Voxel grid size for submap downsampling (meters).
                              Smaller values preserve more detail but increase
                              computation. Typical: 0.05 - 0.2m.
            min_map_points: Minimum number of map points required to run ICP.
                           If submap has fewer points, use prediction only.
            max_icp_residual: Maximum alignment error to accept, as RMS distance
                             per correspondence in meters. Higher residuals are
                             rejected and prediction is used.
            initial_pose: Initial pose [x, y, yaw] for first step. If None, uses
                         origin [0, 0, 0]. Set this to odometry's starting pose
                         for trajectories that don't start at origin.
            max_correspondence_distance: Correspondence gate for ICP in meters
                             (d_threshold in Eq. 7.11). Scan points with no map
                             point within this radius are dropped rather than
                             paired with a distant one. None disables gating,
                             which is what this class used to do implicitly: on
                             the ch7 square loop that let ICP diverge, reaching
                             residuals of 3.9e3 and 2.1e13 on individual steps.
                             Should exceed the odometry drift ICP is being asked
                             to correct, or true correspondences get cut.
        """
        self.submap = Submap2D()
        self.pose_est: np.ndarray | None = None
        self.initialized: bool = False

        # Parameters
        self.voxel_size = submap_voxel_size
        self.min_map_points = min_map_points
        self.max_icp_residual = max_icp_residual
        self._initial_pose = initial_pose
        self.max_correspondence_distance = max_correspondence_distance

    def step(
        self,
        step_index: int,
        odom_delta: np.ndarray,
        scan: np.ndarray,
    ) -> FrontendStepResult:
        """Execute one step of SLAM front-end.

        This is the core SLAM loop:
            1. Predict pose from odometry
            2. Refine pose via scan-to-map ICP
            3. Update submap with refined pose

        Args:
            step_index: Current time step (for logging/debugging).
            odom_delta: Odometry delta (relative pose) [dx, dy, dyaw], shape (3,).
                       This is the motion estimate from wheel encoders or
                       previous scan-to-scan matching.
            scan: LiDAR scan points in robot frame, shape (N, 2).

        Returns:
            FrontendStepResult, which also supports old dict-style keys:
                - 'pose_pred': Predicted pose (odometry only) [x, y, yaw]
                - 'pose_est': Estimated pose (after scan matching) [x, y, yaw]
                - 'match_quality': MatchQuality dataclass with ICP metrics
                - 'correction_magnitude': Euclidean distance between pred and est

        Raises:
            ValueError: If odom_delta or scan have invalid shapes.

        Example:
            >>> frontend = SlamFrontend2D()
            >>> odom = np.array([0.1, 0.0, 0.0])
            >>> scan = np.array([[1.0, 0.0], [2.0, 0.0]])
            >>> result = frontend.step(1, odom, scan)
            >>> print(f"Correction: {result['correction_magnitude']:.3f} m")
        """
        # Validate inputs
        if odom_delta.shape != (3,):
            raise ValueError(f"odom_delta must have shape (3,), got {odom_delta.shape}")
        if scan.ndim != 2 or scan.shape[1] != 2:
            raise ValueError(f"scan must have shape (N, 2), got {scan.shape}")

        # Handle first step (initialization)
        if not self.initialized:
            return self._initialize_first_step(step_index, scan)

        # 1. PREDICTION: Apply odometry delta to previous pose
        pose_pred = se2_compose(self.pose_est, odom_delta)

        # 2. CORRECTION: Scan-to-map alignment via ICP
        pose_est, match_quality = self._scan_to_map_alignment(scan, pose_pred)

        # 3. MAP UPDATE: Add scan to submap with estimated pose
        self.submap.add_scan(pose_est, scan)

        # Update state
        self.pose_est = pose_est

        # Compute correction magnitude
        correction_magnitude = np.linalg.norm(pose_est[:2] - pose_pred[:2])

        return FrontendStepResult(
            predicted_pose_body_to_map_se2=pose_pred,
            estimated_pose_body_to_map_se2=pose_est,
            match_quality=match_quality,
            correction_magnitude_m=float(correction_magnitude),
        )

    def _initialize_first_step(
        self,
        step_index: int,
        scan: np.ndarray,
    ) -> FrontendStepResult:
        """Initialize front-end with first scan.

        Args:
            step_index: Step index (should be 0).
            scan: First scan in robot frame.

        Returns:
            FrontendStepResult with initialization values.
        """
        # Initialize at specified pose or origin
        if self._initial_pose is not None:
            self.pose_est = np.array(self._initial_pose, dtype=float)
        else:
            self.pose_est = np.array([0.0, 0.0, 0.0])

        # Add first scan to submap
        self.submap.add_scan(self.pose_est, scan)

        self.initialized = True

        # Return initialization result
        match_quality = MatchQuality(
            residual_rmse_m=0.0,
            converged=True,
            n_correspondences=len(scan),
            iterations=0,
        )

        return FrontendStepResult(
            predicted_pose_body_to_map_se2=self.pose_est.copy(),
            estimated_pose_body_to_map_se2=self.pose_est.copy(),
            match_quality=match_quality,
            correction_magnitude_m=0.0,
        )

    def _scan_to_map_alignment(
        self,
        scan: np.ndarray,
        pose_pred: np.ndarray,
    ) -> tuple[np.ndarray, MatchQuality]:
        """Align current scan to submap via ICP.

        Args:
            scan: Current scan in robot frame, shape (N, 2).
            pose_pred: Predicted pose in map frame [x, y, yaw].

        Returns:
            Tuple of (pose_est, match_quality):
                - pose_est: Refined pose after ICP [x, y, yaw]
                - match_quality: ICP alignment metrics

        Notes:
            - If submap has too few points, returns prediction (no correction)
            - If ICP fails to converge, returns prediction (fallback)
            - If ICP residual too high, returns prediction (bad match)
        """
        # Get downsampled submap points
        submap_points = self.submap.get_points(voxel_size=self.voxel_size)

        # Check if submap has enough points for ICP
        if len(submap_points) < self.min_map_points:
            # Not enough map points, use prediction
            match_quality = MatchQuality(
                residual_rmse_m=0.0,
                converged=False,
                n_correspondences=0,
                iterations=0,
            )
            return pose_pred, match_quality

        # Check if scan has enough points
        if len(scan) < 5:
            # Too few scan points, use prediction
            match_quality = MatchQuality(
                residual_rmse_m=0.0,
                converged=False,
                n_correspondences=len(scan),
                iterations=0,
            )
            return pose_pred, match_quality

        # Run ICP: align scan (in robot frame) to submap (in map frame)
        # initial_pose is the transformation from robot frame to map frame
        try:
            pose_est, iters, residual, converged = icp_point_to_point(
                source_scan=scan,
                target_scan=submap_points,
                initial_pose=pose_pred,
                max_iterations=50,
                tolerance=1e-4,
                max_correspondence_distance=self.max_correspondence_distance,
            )
        except Exception:
            # ICP failed (e.g., numerical issues)
            match_quality = MatchQuality(
                residual_rmse_m=float("inf"),
                converged=False,
                n_correspondences=0,
                iterations=0,
            )
            return pose_pred, match_quality

        # Check match quality
        if converged and residual < self.max_icp_residual:
            # Good match: use ICP result
            match_quality = MatchQuality(
                residual_rmse_m=residual,
                converged=True,
                n_correspondences=len(scan),  # Approximate
                iterations=iters,
            )
            return pose_est, match_quality
        else:
            # Poor match or didn't converge: fallback to prediction
            match_quality = MatchQuality(
                residual_rmse_m=residual,
                converged=converged,
                n_correspondences=len(scan),
                iterations=iters,
            )
            return pose_pred, match_quality

    def get_current_pose(self) -> np.ndarray | None:
        """Get current pose estimate.

        Returns:
            Current pose [x, y, yaw] or None if not initialized.
        """
        return self.pose_est.copy() if self.initialized else None

    def get_submap_points(self, voxel_size: float | None = None) -> np.ndarray:
        """Get current submap points.

        Args:
            voxel_size: Optional voxel size for downsampling.

        Returns:
            Submap points in map frame, shape (M, 2).
        """
        return self.submap.get_points(voxel_size=voxel_size)

    def reset(self) -> None:
        """Reset front-end state (clear submap and pose estimate)."""
        self.submap.clear()
        self.pose_est = None
        self.initialized = False
