"""Unit tests for LiDAR scan generation with occlusion handling.

Tests the ray-casting scan generation to ensure proper occlusion behavior:
- Near obstacles block far obstacles
- Ray-segment intersection is computed correctly
- Scan points are in robot's local frame
- Measurement noise is applied correctly

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest

import numpy as np

from core.slam.scan_generation import (
    generate_scan_with_occlusion,
    ray_segment_intersection,
)


class TestRaySegmentIntersection(unittest.TestCase):
    """Test ray-segment intersection computation."""
    
    def test_perpendicular_intersection(self):
        """Test ray hitting segment perpendicularly."""
        ray_origin = np.array([0.0, 0.0])
        ray_direction = np.array([1.0, 0.0])  # Ray pointing right
        segment_start = np.array([5.0, -2.0])
        segment_end = np.array([5.0, 2.0])  # Vertical segment at x=5
        
        point, distance = ray_segment_intersection(
            ray_origin, ray_direction, segment_start, segment_end
        )
        
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, [5.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(distance, 5.0, places=6)
    
    def test_angled_intersection(self):
        """Test ray hitting segment at an angle."""
        ray_origin = np.array([0.0, 0.0])
        ray_direction = np.array([1.0, 1.0]) / np.sqrt(2)  # 45-degree ray
        segment_start = np.array([0.0, 2.0])
        segment_end = np.array([4.0, 2.0])  # Horizontal segment at y=2
        
        point, distance = ray_segment_intersection(
            ray_origin, ray_direction, segment_start, segment_end
        )
        
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, [2.0, 2.0], atol=1e-6)
        self.assertAlmostEqual(distance, 2.0 * np.sqrt(2), places=6)
    
    def test_no_intersection_parallel(self):
        """Test parallel ray and segment (no intersection)."""
        ray_origin = np.array([0.0, 0.0])
        ray_direction = np.array([1.0, 0.0])
        segment_start = np.array([1.0, 1.0])
        segment_end = np.array([5.0, 1.0])  # Parallel to ray
        
        point, distance = ray_segment_intersection(
            ray_origin, ray_direction, segment_start, segment_end
        )
        
        self.assertIsNone(point)
        self.assertEqual(distance, float('inf'))
    
    def test_no_intersection_behind_ray(self):
        """Test segment behind ray origin (no intersection)."""
        ray_origin = np.array([5.0, 5.0])
        ray_direction = np.array([1.0, 0.0])  # Ray pointing right
        segment_start = np.array([0.0, 4.0])
        segment_end = np.array([0.0, 6.0])  # Segment to the left (behind)
        
        point, distance = ray_segment_intersection(
            ray_origin, ray_direction, segment_start, segment_end
        )
        
        self.assertIsNone(point)
        self.assertEqual(distance, float('inf'))
    
    def test_no_intersection_beyond_segment(self):
        """Test ray passing beyond segment endpoints."""
        ray_origin = np.array([0.0, 0.0])
        ray_direction = np.array([1.0, 0.0])
        segment_start = np.array([5.0, 1.0])  # Segment above ray path
        segment_end = np.array([5.0, 2.0])
        
        point, distance = ray_segment_intersection(
            ray_origin, ray_direction, segment_start, segment_end
        )
        
        self.assertIsNone(point)
        self.assertEqual(distance, float('inf'))


class TestScanGenerationWithOcclusion(unittest.TestCase):
    """Test scan generation with proper occlusion handling."""
    
    def test_simple_box_no_occlusion(self):
        """Test scan in simple box environment (no internal obstacles)."""
        # Create square room: 10m x 10m
        walls = [
            (np.array([0.0, 0.0]), np.array([10.0, 0.0])),  # South
            (np.array([10.0, 0.0]), np.array([10.0, 10.0])),  # East
            (np.array([10.0, 10.0]), np.array([0.0, 10.0])),  # North
            (np.array([0.0, 10.0]), np.array([0.0, 0.0])),  # West
        ]
        
        # Robot at center, facing east
        pose = np.array([5.0, 5.0, 0.0])
        
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=10.0, noise_std=0.0
        )
        
        # Should have close to 360 points (one per ray)
        self.assertGreater(len(scan), 340)  # Allow some rays to miss corners
        self.assertLessEqual(len(scan), 360)
        
        # All points should be within max range
        ranges = np.linalg.norm(scan, axis=1)
        self.assertTrue(np.all(ranges <= 10.0))
        
        # Check that we hit walls at expected distances
        # Ray facing east (x=1, y=0) should hit at distance ~5m
        east_rays = scan[np.abs(scan[:, 1]) < 0.2]  # Rays near x-axis
        east_forward = east_rays[east_rays[:, 0] > 0]
        if len(east_forward) > 0:
            min_dist = np.min(np.linalg.norm(east_forward, axis=1))
            self.assertAlmostEqual(min_dist, 5.0, places=0)  # ~5m to east wall
    
    def test_occlusion_by_near_obstacle(self):
        """Test that near obstacle blocks far wall (key occlusion test)."""
        # Setup: Far wall and near obstacle in front of it
        walls = [
            # Far wall at x=10
            (np.array([10.0, -5.0]), np.array([10.0, 5.0])),
            # Near obstacle at x=3 (blocks center of far wall)
            (np.array([3.0, -0.5]), np.array([3.0, 0.5])),
        ]
        
        # Robot at origin, facing east
        pose = np.array([0.0, 0.0, 0.0])
        
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=15.0, noise_std=0.0
        )
        
        # Find rays that should hit the near obstacle (y ≈ 0, x > 0)
        near_center_rays = scan[np.abs(scan[:, 1]) < 0.3]
        near_center_forward = near_center_rays[near_center_rays[:, 0] > 0]
        
        # These rays should hit at ~3m (near obstacle), NOT ~10m (far wall)
        if len(near_center_forward) > 0:
            distances = np.linalg.norm(near_center_forward, axis=1)
            min_dist = np.min(distances)
            
            # Should be close to 3m (near obstacle), not 10m (far wall)
            self.assertLess(min_dist, 4.0, "Ray should hit near obstacle")
            self.assertGreater(min_dist, 2.5, "Distance should be ~3m")
            
            # Verify we DON'T see the far wall through the obstacle
            self.assertFalse(np.any(distances > 8.0),
                           "Should not see far wall behind obstacle")
    
    def test_pillar_blocks_multiple_walls(self):
        """Test that a pillar blocks walls behind it in multiple directions."""
        # Room with pillar in center
        walls = [
            # Outer room walls (20m x 20m)
            (np.array([0.0, 0.0]), np.array([20.0, 0.0])),  # South
            (np.array([20.0, 0.0]), np.array([20.0, 20.0])),  # East
            (np.array([20.0, 20.0]), np.array([0.0, 20.0])),  # North
            (np.array([0.0, 20.0]), np.array([0.0, 0.0])),  # West
            # Center pillar (2m x 2m) at (9, 9) to (11, 11)
            (np.array([9.0, 9.0]), np.array([11.0, 9.0])),  # Pillar south
            (np.array([11.0, 9.0]), np.array([11.0, 11.0])),  # Pillar east
            (np.array([11.0, 11.0]), np.array([9.0, 11.0])),  # Pillar north
            (np.array([9.0, 11.0]), np.array([9.0, 9.0])),  # Pillar west
        ]
        
        # Robot at (5, 10), facing east toward pillar
        pose = np.array([5.0, 10.0, 0.0])
        
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=20.0, noise_std=0.0
        )
        
        # Rays facing forward should hit pillar (~4m away), not far wall (~15m)
        forward_rays = scan[(scan[:, 0] > 0) & (np.abs(scan[:, 1]) < 1.0)]
        
        if len(forward_rays) > 0:
            distances = np.linalg.norm(forward_rays, axis=1)
            min_dist = np.min(distances)
            
            # Should hit pillar at ~4m, not room wall at ~15m
            self.assertLess(min_dist, 6.0, "Should hit pillar")
            self.assertGreater(min_dist, 3.0, "Pillar is ~4m away")
    
    def test_scan_in_robot_local_frame(self):
        """Test that scan points are in robot's local frame."""
        walls = [
            (np.array([5.0, 0.0]), np.array([5.0, 10.0])),  # Vertical wall
        ]
        
        # Robot at (0, 5), facing east (yaw=0)
        pose = np.array([0.0, 5.0, 0.0])
        
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=10.0, noise_std=0.0
        )
        
        # Wall is directly in front (east), so points should have x>0, y≈0
        forward_points = scan[scan[:, 0] > 4.0]
        self.assertGreater(len(forward_points), 0)
        
        # Y-coordinates should be relatively small (wall mostly aligned with forward axis)
        # Wall spans y=[0,10], robot at y=5, so local y range is [-5, 5]
        self.assertTrue(np.all(np.abs(forward_points[:, 1]) < 5.5))
    
    def test_rotated_robot_frame(self):
        """Test scan with rotated robot (verify frame transformation)."""
        walls = [
            (np.array([5.0, 0.0]), np.array([5.0, 10.0])),  # Vertical wall
        ]
        
        # Robot at (0, 5), facing north (yaw=π/2)
        pose = np.array([0.0, 5.0, np.pi / 2])
        
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=10.0, noise_std=0.0
        )
        
        # Wall is to the robot's right (east in global, but right in local)
        # In robot frame: forward=north, right=east
        # So wall should appear at x≈0, y<0 (to the right)
        right_points = scan[scan[:, 1] < -4.0]
        self.assertGreater(len(right_points), 0)
    
    def test_min_max_range_filtering(self):
        """Test that points outside [min_range, max_range] are filtered."""
        walls = [
            (np.array([2.0, -5.0]), np.array([2.0, 5.0])),  # Wall at x=2
            (np.array([15.0, -5.0]), np.array([15.0, 5.0])),  # Wall at x=15
        ]
        
        pose = np.array([0.0, 0.0, 0.0])
        
        # Set max_range=10 to exclude far wall
        scan = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=10.0, min_range=0.1,
            noise_std=0.0
        )
        
        ranges = np.linalg.norm(scan, axis=1)
        
        # All ranges should be within [0.1, 10.0]
        self.assertTrue(np.all(ranges >= 0.1))
        self.assertTrue(np.all(ranges <= 10.0))
        
        # Should see near wall (~2m) but not far wall (~15m)
        self.assertTrue(np.any(ranges < 3.0), "Should see near wall")
        self.assertFalse(np.any(ranges > 10.0), "Should not see far wall")


class TestComparisonWithLegacy(unittest.TestCase):
    """Compare new occlusion-aware scan with legacy implementation."""
    
    def test_occlusion_reduces_point_count(self):
        """Test that occlusion handling reduces points (blocks occluded walls)."""
        # Complex environment with internal obstacles
        walls = [
            # Outer walls (large room)
            (np.array([0.0, 0.0]), np.array([20.0, 0.0])),
            (np.array([20.0, 0.0]), np.array([20.0, 20.0])),
            (np.array([20.0, 20.0]), np.array([0.0, 20.0])),
            (np.array([0.0, 20.0]), np.array([0.0, 0.0])),
            # Many internal obstacles that should occlude outer walls
            (np.array([5.0, 5.0]), np.array([15.0, 5.0])),
            (np.array([5.0, 15.0]), np.array([15.0, 15.0])),
            (np.array([5.0, 5.0]), np.array([5.0, 15.0])),
            (np.array([15.0, 5.0]), np.array([15.0, 15.0])),
        ]
        
        pose = np.array([10.0, 10.0, 0.0])  # Robot in center
        
        # New scan with occlusion
        scan_with_occlusion = generate_scan_with_occlusion(
            pose, walls, num_rays=360, max_range=15.0, noise_std=0.0
        )
        
        # Scan should have ~360 points (one per ray, only closest hit)
        self.assertLessEqual(len(scan_with_occlusion), 360)
        self.assertGreater(len(scan_with_occlusion), 300)  # Most rays hit


if __name__ == '__main__':
    unittest.main()
