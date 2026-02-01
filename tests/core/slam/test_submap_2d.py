"""Unit tests for Submap2D (2D local submap for scan-to-map alignment).

Tests cover:
- Basic operations: add_scan, get_points, clear
- Voxel downsampling (in-place and on-demand)
- Edge cases: empty scans, single points, invalid inputs
- Coordinate transformations: verify SE(2) transformations are applied correctly

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np

from core.slam.submap_2d import Submap2D


class TestSubmap2DBasic(unittest.TestCase):
    """Test basic Submap2D functionality."""
    
    def test_initialization(self):
        """Test submap initializes empty."""
        submap = Submap2D()
        
        self.assertEqual(len(submap), 0)
        self.assertEqual(submap.n_scans, 0)
        self.assertEqual(submap.points.shape, (0, 2))
    
    def test_add_single_scan_identity_pose(self):
        """Test adding a single scan at identity pose."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])  # Identity: [x, y, yaw]
        scan = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        submap.add_scan(pose, scan)
        
        self.assertEqual(len(submap), 3)
        self.assertEqual(submap.n_scans, 1)
        np.testing.assert_allclose(submap.points, scan, rtol=1e-6)
    
    def test_add_single_scan_translated_pose(self):
        """Test adding scan with translation (no rotation)."""
        submap = Submap2D()
        pose = np.array([2.0, 3.0, 0.0])  # Translation only
        scan = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        submap.add_scan(pose, scan)
        
        expected = np.array([[3.0, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(submap.points, expected, rtol=1e-6)
    
    def test_add_single_scan_rotated_pose(self):
        """Test adding scan with rotation (90 degrees)."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, np.pi / 2])  # 90-degree rotation
        scan = np.array([[1.0, 0.0]])  # Point at (1, 0) in robot frame
        
        submap.add_scan(pose, scan)
        
        # After 90-deg rotation: (1, 0) -> (0, 1)
        expected = np.array([[0.0, 1.0]])
        np.testing.assert_allclose(submap.points, expected, atol=1e-6)
    
    def test_add_multiple_scans_increases_count(self):
        """Test that adding multiple scans accumulates points."""
        submap = Submap2D()
        
        pose1 = np.array([0.0, 0.0, 0.0])
        scan1 = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        pose2 = np.array([5.0, 0.0, 0.0])
        scan2 = np.array([[1.0, 0.0]])
        
        submap.add_scan(pose1, scan1)
        self.assertEqual(len(submap), 2)
        self.assertEqual(submap.n_scans, 1)
        
        submap.add_scan(pose2, scan2)
        self.assertEqual(len(submap), 3)
        self.assertEqual(submap.n_scans, 2)
    
    def test_add_empty_scan_no_op(self):
        """Test that adding empty scan doesn't change map."""
        submap = Submap2D()
        pose = np.array([1.0, 2.0, 0.5])
        empty_scan = np.empty((0, 2))
        
        submap.add_scan(pose, empty_scan)
        
        self.assertEqual(len(submap), 0)
        self.assertEqual(submap.n_scans, 0)
    
    def test_get_points_returns_copy(self):
        """Test that get_points returns a copy (not view)."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        scan = np.array([[1.0, 0.0]])
        
        submap.add_scan(pose, scan)
        points1 = submap.get_points()
        points2 = submap.get_points()
        
        # Modify returned array
        points1[0, 0] = 999.0
        
        # Internal points should be unchanged
        self.assertAlmostEqual(submap.points[0, 0], 1.0)
        # Second get_points() should also be unchanged
        self.assertAlmostEqual(points2[0, 0], 1.0)
    
    def test_clear_resets_submap(self):
        """Test that clear() removes all points and resets scan count."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        submap.add_scan(pose, scan)
        self.assertEqual(len(submap), 2)
        self.assertEqual(submap.n_scans, 1)
        
        submap.clear()
        self.assertEqual(len(submap), 0)
        self.assertEqual(submap.n_scans, 0)
        self.assertEqual(submap.points.shape, (0, 2))


class TestSubmap2DDownsampling(unittest.TestCase):
    """Test voxel grid downsampling functionality."""
    
    def test_downsample_reduces_point_count(self):
        """Test that downsampling reduces point count for dense points."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        
        # Create dense points: 3 points within 0.1m voxel
        scan = np.array([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0], [1.0, 0.0]])
        submap.add_scan(pose, scan)
        
        original_count = len(submap)
        self.assertEqual(original_count, 4)
        
        # Downsample with 0.1m voxel size (should merge first 3 points)
        submap.downsample(voxel_size=0.1)
        
        downsampled_count = len(submap)
        self.assertLess(downsampled_count, original_count)
        self.assertEqual(downsampled_count, 2)  # 1 merged + 1 distant
    
    def test_get_points_with_voxel_size(self):
        """Test get_points with voxel_size parameter (non-destructive)."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        
        # Dense points
        scan = np.array([[0.01, 0.0], [0.02, 0.0], [1.0, 0.0]])
        submap.add_scan(pose, scan)
        
        original_count = len(submap)
        self.assertEqual(original_count, 3)
        
        # Get downsampled points (non-destructive)
        downsampled = submap.get_points(voxel_size=0.1)
        
        # Original points unchanged
        self.assertEqual(len(submap), original_count)
        # Returned points are downsampled
        self.assertEqual(len(downsampled), 2)
    
    def test_downsample_empty_submap_no_op(self):
        """Test that downsampling empty submap is a no-op."""
        submap = Submap2D()
        
        submap.downsample(voxel_size=0.5)
        
        self.assertEqual(len(submap), 0)
    
    def test_downsample_computes_centroid(self):
        """Test that downsampling computes centroid of voxel points."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        
        # Two points in same voxel: (0.0, 0.0) and (0.05, 0.0)
        scan = np.array([[0.0, 0.0], [0.05, 0.0]])
        submap.add_scan(pose, scan)
        
        submap.downsample(voxel_size=0.1)
        
        # Should have 1 point at centroid (0.025, 0.0)
        self.assertEqual(len(submap), 1)
        np.testing.assert_allclose(submap.points[0], [0.025, 0.0], atol=1e-6)
    
    def test_downsample_separates_distant_points(self):
        """Test that downsampling keeps distant points separate."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        
        # Three points in different voxels
        scan = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        submap.add_scan(pose, scan)
        
        submap.downsample(voxel_size=0.5)
        
        # All points should remain (different voxels)
        self.assertEqual(len(submap), 3)
    
    def test_voxel_downsample_invalid_size_raises_error(self):
        """Test that invalid voxel size raises ValueError."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        scan = np.array([[1.0, 0.0]])
        submap.add_scan(pose, scan)
        
        with self.assertRaises(ValueError):
            submap.downsample(voxel_size=0.0)
        
        with self.assertRaises(ValueError):
            submap.downsample(voxel_size=-0.1)


class TestSubmap2DInputValidation(unittest.TestCase):
    """Test input validation and edge cases."""
    
    def test_add_scan_invalid_pose_shape_raises_error(self):
        """Test that invalid pose shape raises ValueError."""
        submap = Submap2D()
        invalid_pose = np.array([0.0, 0.0])  # Missing yaw
        scan = np.array([[1.0, 0.0]])
        
        with self.assertRaises(ValueError):
            submap.add_scan(invalid_pose, scan)
    
    def test_add_scan_invalid_scan_shape_raises_error(self):
        """Test that invalid scan shape raises ValueError."""
        submap = Submap2D()
        pose = np.array([0.0, 0.0, 0.0])
        
        # 1D array
        with self.assertRaises(ValueError):
            submap.add_scan(pose, np.array([1.0, 2.0, 3.0]))
        
        # Wrong number of columns
        with self.assertRaises(ValueError):
            submap.add_scan(pose, np.array([[1.0, 2.0, 3.0]]))
    
    def test_add_scan_single_point(self):
        """Test adding scan with single point."""
        submap = Submap2D()
        pose = np.array([1.0, 2.0, 0.0])
        scan = np.array([[0.5, 0.0]])
        
        submap.add_scan(pose, scan)
        
        self.assertEqual(len(submap), 1)
        expected = np.array([[1.5, 2.0]])
        np.testing.assert_allclose(submap.points, expected, rtol=1e-6)


class TestSubmap2DIntegration(unittest.TestCase):
    """Integration tests for typical SLAM usage patterns."""
    
    def test_build_submap_from_trajectory(self):
        """Test building submap from multiple robot poses along trajectory."""
        submap = Submap2D()
        
        # Robot moves along x-axis, observes same landmark from different positions
        poses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]
        
        # Each scan observes a landmark at (5, 0) in map frame
        scans = [
            np.array([[5.0, 0.0]]),  # From pose 0
            np.array([[4.0, 0.0]]),  # From pose 1 (4m ahead in robot frame = 5m in map)
            np.array([[3.0, 0.0]]),  # From pose 2
        ]
        
        for pose, scan in zip(poses, scans):
            submap.add_scan(pose, scan)
        
        self.assertEqual(submap.n_scans, 3)
        self.assertEqual(len(submap), 3)
        
        # All observations should be at approximately (5, 0) in map frame
        for point in submap.points:
            self.assertAlmostEqual(point[0], 5.0, places=5)
            self.assertAlmostEqual(point[1], 0.0, places=5)
    
    def test_submap_with_rotation_and_translation(self):
        """Test submap with complex SE(2) transformations."""
        submap = Submap2D()
        
        # Pose 1: At origin, facing east
        pose1 = np.array([0.0, 0.0, 0.0])
        scan1 = np.array([[1.0, 0.0]])  # Point at (1, 0) in robot frame
        submap.add_scan(pose1, scan1)
        
        # Pose 2: At (1, 0), facing north (90 degrees)
        pose2 = np.array([1.0, 0.0, np.pi / 2])
        scan2 = np.array([[1.0, 0.0]])  # Point at (1, 0) in robot frame
        submap.add_scan(pose2, scan2)
        
        self.assertEqual(len(submap), 2)
        
        # First point: (0, 0) + (1, 0) = (1, 0)
        np.testing.assert_allclose(submap.points[0], [1.0, 0.0], atol=1e-6)
        
        # Second point: (1, 0) + rot(90deg)(1, 0) = (1, 0) + (0, 1) = (1, 1)
        np.testing.assert_allclose(submap.points[1], [1.0, 1.0], atol=1e-6)
    
    def test_realistic_lidar_scenario(self):
        """Test realistic LiDAR scenario with multiple scans and downsampling."""
        submap = Submap2D()
        
        # Simulate robot moving along square trajectory
        poses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, np.pi / 2]),
            np.array([0.0, 1.0, np.pi]),
        ]
        
        # Each pose observes some wall points
        for i, pose in enumerate(poses):
            # Simulate wall observations (10 points per scan)
            scan = np.random.rand(10, 2) * 0.5 + np.array([2.0, 0.0])
            submap.add_scan(pose, scan)
        
        self.assertEqual(submap.n_scans, 4)
        self.assertEqual(len(submap), 40)  # 4 scans × 10 points
        
        # Downsample to reduce density
        submap.downsample(voxel_size=0.2)
        
        # Should have fewer points after downsampling
        self.assertLess(len(submap), 40)
        self.assertGreater(len(submap), 0)


if __name__ == "__main__":
    unittest.main()
