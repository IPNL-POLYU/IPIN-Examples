"""Lightweight 2D Submap for Scan-to-Map Alignment.

This module implements a simple local submap abstraction for storing and
accessing accumulated LiDAR scans in a map frame. Used in SLAM front-end
for scan-to-map matching.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from typing import Optional
from .se2 import se2_apply


class Submap2D:
    """Lightweight 2D submap for storing accumulated scan points.
    
    A submap maintains a collection of 2D points in a map frame, built by
    transforming individual scans from their robot frames into the map frame.
    Used in SLAM front-end for scan-to-map alignment.
    
    Attributes:
        points: Accumulated map points in map frame, shape (N, 2).
        n_scans: Number of scans added to this submap.
    
    Example:
        >>> submap = Submap2D()
        >>> pose = np.array([1.0, 2.0, 0.5])  # [x, y, yaw]
        >>> scan = np.array([[0.5, 0.0], [1.0, 0.0]])  # Local frame
        >>> submap.add_scan(pose, scan)
        >>> map_points = submap.get_points()
        >>> print(map_points.shape)
        (2, 2)
    
    Notes:
        - Points are stored in map frame (global coordinates)
        - No automatic downsampling unless explicitly requested
        - Thread-safety: not thread-safe, use external locking if needed
    """
    
    def __init__(self) -> None:
        """Initialize an empty submap."""
        self.points: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self.n_scans: int = 0
    
    def add_scan(self, pose_se2: np.ndarray, scan_xy: np.ndarray) -> None:
        """Add a scan to the submap by transforming it into map frame.
        
        Args:
            pose_se2: Robot pose in map frame [x, y, yaw], shape (3,).
            scan_xy: Scan points in robot local frame, shape (N, 2).
        
        Raises:
            ValueError: If pose_se2 is not shape (3,) or scan_xy is not shape (N, 2).
        
        Example:
            >>> submap = Submap2D()
            >>> pose = np.array([0.0, 0.0, 0.0])
            >>> scan = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> submap.add_scan(pose, scan)
            >>> submap.n_scans
            1
        """
        # Validate inputs
        if pose_se2.shape != (3,):
            raise ValueError(f"pose_se2 must have shape (3,), got {pose_se2.shape}")
        if scan_xy.ndim != 2 or scan_xy.shape[1] != 2:
            raise ValueError(f"scan_xy must have shape (N, 2), got {scan_xy.shape}")
        
        if len(scan_xy) == 0:
            # Empty scan, nothing to add
            return
        
        # Transform scan points from robot frame to map frame
        map_points = se2_apply(pose_se2, scan_xy)
        
        # Append to existing points
        if self.points.shape[0] == 0:
            self.points = map_points
        else:
            self.points = np.vstack([self.points, map_points])
        
        self.n_scans += 1
    
    def get_points(self, voxel_size: Optional[float] = None) -> np.ndarray:
        """Get all map points, optionally downsampled.
        
        Args:
            voxel_size: If provided, downsample points using voxel grid filter.
                       Points within the same voxel are replaced by their centroid.
        
        Returns:
            Map points in map frame, shape (M, 2). If voxel_size is provided,
            M <= original point count. If no points exist, returns empty array.
        
        Example:
            >>> submap = Submap2D()
            >>> pose = np.array([0.0, 0.0, 0.0])
            >>> scan = np.array([[0.1, 0.0], [0.11, 0.0], [1.0, 0.0]])
            >>> submap.add_scan(pose, scan)
            >>> # No downsampling
            >>> len(submap.get_points())
            3
            >>> # With downsampling (voxel_size=0.1, two points merge)
            >>> len(submap.get_points(voxel_size=0.1))
            2
        """
        if self.points.shape[0] == 0:
            return self.points.copy()
        
        if voxel_size is None:
            return self.points.copy()
        
        # Voxel grid downsampling
        return self._voxel_downsample(self.points, voxel_size)
    
    def downsample(self, voxel_size: float) -> None:
        """Downsample map points in-place using voxel grid filter.
        
        Args:
            voxel_size: Voxel grid size in meters. Points within the same voxel
                       are replaced by their centroid.
        
        Example:
            >>> submap = Submap2D()
            >>> pose = np.array([0.0, 0.0, 0.0])
            >>> scan = np.array([[0.1, 0.0], [0.11, 0.0], [1.0, 0.0]])
            >>> submap.add_scan(pose, scan)
            >>> len(submap.points)
            3
            >>> submap.downsample(voxel_size=0.1)
            >>> len(submap.points)
            2
        """
        if self.points.shape[0] == 0:
            return
        
        self.points = self._voxel_downsample(self.points, voxel_size)
    
    def clear(self) -> None:
        """Clear all points and reset scan count.
        
        Example:
            >>> submap = Submap2D()
            >>> pose = np.array([0.0, 0.0, 0.0])
            >>> scan = np.array([[1.0, 0.0]])
            >>> submap.add_scan(pose, scan)
            >>> submap.clear()
            >>> len(submap.points)
            0
            >>> submap.n_scans
            0
        """
        self.points = np.empty((0, 2), dtype=np.float64)
        self.n_scans = 0
    
    def __len__(self) -> int:
        """Return number of points in the submap.
        
        Returns:
            Number of points currently stored.
        """
        return self.points.shape[0]
    
    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Voxel grid downsampling using quantization and centroid computation.
        
        Args:
            points: Input points, shape (N, 2).
            voxel_size: Voxel grid size in meters.
        
        Returns:
            Downsampled points, shape (M, 2) where M <= N.
        
        Notes:
            - Points are quantized to voxel grid coordinates
            - Points in the same voxel are averaged to compute centroid
            - Uses dictionary for efficient voxel lookup
        """
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        
        if points.shape[0] == 0:
            return points.copy()
        
        # Quantize points to voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Group points by voxel using dictionary
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = (voxel_idx[0], voxel_idx[1])
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # Compute centroid for each voxel
        downsampled = []
        for voxel_points in voxel_dict.values():
            centroid = np.mean(voxel_points, axis=0)
            downsampled.append(centroid)
        
        return np.array(downsampled)
