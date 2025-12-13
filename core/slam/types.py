"""Type definitions and data structures for SLAM algorithms (Chapter 7).

This module defines the core data structures used throughout Chapter 7
(SLAM Technologies) of the book: Principles of Indoor Positioning and
Indoor Navigation.

Key types:
    - Pose2: SE(2) pose representation [x, y, yaw]
    - CameraIntrinsics: Camera calibration parameters
    - PointCloud2D: Type alias for 2D point clouds
    - VoxelGrid: Type alias for NDT voxel maps

Author: Navigation Engineer
Date: 2024
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Type aliases for clarity and documentation
PointCloud2D = np.ndarray  # Shape (N, 2), points in 2D space (meters)
PointCloud3D = np.ndarray  # Shape (N, 3), points in 3D space (meters)
VoxelGrid = Dict[Tuple[int, ...], Dict[str, np.ndarray]]  # Voxel key -> stats dict


@dataclass
class Pose2:
    """
    SE(2) pose representation for 2D SLAM.

    Represents a rigid transformation in the plane: position (x, y) and
    orientation (yaw angle). This is the primary pose representation used
    in Chapter 7 for 2D LiDAR SLAM and simplified visual SLAM examples.

    Attributes:
        x: Position in x-axis (meters).
        y: Position in y-axis (meters).
        yaw: Heading angle (radians), measured counter-clockwise from the
             positive x-axis. Should be normalized to [-π, π].

    Notes:
        - The yaw angle follows the right-hand rule in 2D.
        - SE(2) = Special Euclidean group in 2D = rigid motions (rotation + translation).
        - This representation is used in scan matching (Eqs. 7.10-7.11),
          NDT alignment (Eqs. 7.12-7.16), and pose graph optimization.

    Examples:
        >>> # Robot at origin facing east (0 degrees)
        >>> p1 = Pose2(x=0.0, y=0.0, yaw=0.0)
        >>>
        >>> # Robot at (10, 5) facing north (90 degrees)
        >>> p2 = Pose2(x=10.0, y=5.0, yaw=np.pi/2)
        >>>
        >>> # Convert to array
        >>> pose_array = p1.to_array()
        >>> print(pose_array)  # [0. 0. 0.]
        >>>
        >>> # Create from array
        >>> p3 = Pose2.from_array(np.array([1.0, 2.0, np.pi/4]))
    """

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        """Validate pose values after initialization."""
        if not np.isfinite(self.x):
            raise ValueError(f"x must be finite, got {self.x}")
        if not np.isfinite(self.y):
            raise ValueError(f"y must be finite, got {self.y}")
        if not np.isfinite(self.yaw):
            raise ValueError(f"yaw must be finite, got {self.yaw}")

    def to_array(self) -> np.ndarray:
        """
        Convert pose to NumPy array [x, y, yaw].

        Returns:
            Array of shape (3,) containing [x, y, yaw].

        Examples:
            >>> p = Pose2(x=1.0, y=2.0, yaw=np.pi/4)
            >>> arr = p.to_array()
            >>> print(arr)  # [1.0, 2.0, 0.785...]
        """
        return np.array([self.x, self.y, self.yaw], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Pose2":
        """
        Create Pose2 from NumPy array [x, y, yaw].

        Args:
            arr: Array of shape (3,) containing [x, y, yaw].

        Returns:
            Pose2 instance.

        Raises:
            ValueError: If array does not have exactly 3 elements.

        Examples:
            >>> arr = np.array([1.0, 2.0, np.pi/4])
            >>> p = Pose2.from_array(arr)
        """
        if arr.shape != (3,):
            raise ValueError(f"Array must have shape (3,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), yaw=float(arr[2]))

    @classmethod
    def identity(cls) -> "Pose2":
        """
        Create identity pose (origin with zero rotation).

        Returns:
            Pose2 at (0, 0) with yaw=0.

        Examples:
            >>> p_id = Pose2.identity()
            >>> print(p_id)  # Pose2(x=0.0, y=0.0, yaw=0.0)
        """
        return cls(x=0.0, y=0.0, yaw=0.0)

    def __repr__(self) -> str:
        """Readable string representation."""
        return f"Pose2(x={self.x:.4f}, y={self.y:.4f}, yaw={self.yaw:.4f})"


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters for visual SLAM (Chapter 7).

    Represents the pinhole camera model with radial and tangential
    distortion, as described in Section 7.4 of the book. Used for
    bundle adjustment (Eqs. 7.68-7.70) and reprojection factors.

    Attributes:
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x-coordinate (pixels).
        cy: Principal point y-coordinate (pixels).
        k1: 1st radial distortion coefficient (Eq. 7.43).
        k2: 2nd radial distortion coefficient (Eq. 7.44).
        p1: 1st tangential distortion coefficient (Eq. 7.45).
        p2: 2nd tangential distortion coefficient (Eq. 7.46).
        width: Image width in pixels (optional).
        height: Image height in pixels (optional).

    Notes:
        - Distortion model follows Brown-Conrady / OpenCV convention.
        - Eqs. (7.43)-(7.46) define the radial + tangential distortion.
        - For an ideal pinhole camera, set k1=k2=p1=p2=0.

    Examples:
        >>> # Typical camera with moderate distortion
        >>> K = CameraIntrinsics(
        ...     fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        ...     k1=-0.2, k2=0.05, p1=0.001, p2=0.001,
        ...     width=640, height=480
        ... )
        >>>
        >>> # Ideal pinhole (no distortion)
        >>> K_ideal = CameraIntrinsics(
        ...     fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        ...     k1=0.0, k2=0.0, p1=0.0, p2=0.0
        ... )
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0  # Radial distortion (Eq. 7.43)
    k2: float = 0.0  # Radial distortion (Eq. 7.44)
    p1: float = 0.0  # Tangential distortion (Eq. 7.45)
    p2: float = 0.0  # Tangential distortion (Eq. 7.46)
    width: int = 640  # Image width (pixels)
    height: int = 480  # Image height (pixels)

    def __post_init__(self) -> None:
        """Validate camera parameters after initialization."""
        if self.fx <= 0:
            raise ValueError(f"fx must be positive, got {self.fx}")
        if self.fy <= 0:
            raise ValueError(f"fy must be positive, got {self.fy}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if not (0 <= self.cx < self.width):
            raise ValueError(
                f"cx must be in [0, {self.width}), got {self.cx}"
            )
        if not (0 <= self.cy < self.height):
            raise ValueError(
                f"cy must be in [0, {self.height}), got {self.cy}"
            )

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 intrinsic matrix K.

        Returns:
            Intrinsic matrix of shape (3, 3):
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

        Examples:
            >>> K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
            >>> K_matrix = K.to_matrix()
            >>> print(K_matrix.shape)  # (3, 3)
        """
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def has_distortion(self) -> bool:
        """
        Check if camera has non-zero distortion parameters.

        Returns:
            True if any distortion coefficient is non-zero.

        Examples:
            >>> K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, k1=-0.2)
            >>> print(K.has_distortion())  # True
        """
        return any(
            [
                abs(self.k1) > 1e-10,
                abs(self.k2) > 1e-10,
                abs(self.p1) > 1e-10,
                abs(self.p2) > 1e-10,
            ]
        )

    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"distortion=[{self.k1:.4f}, {self.k2:.4f}, {self.p1:.4f}, {self.p2:.4f}])"
        )

