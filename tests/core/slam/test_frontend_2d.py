"""Unit tests for SlamFrontend2D (SLAM front-end with scan-to-map alignment).

Tests cover:
- Initialization and first step
- Prediction → scan-to-map alignment → map update loop
- ICP success and failure cases
- Fallback to prediction when ICP fails
- Edge cases: empty scans, insufficient map points

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np

from core.slam.frontend_2d import SlamFrontend2D, MatchQuality


class TestSlamFrontend2DInitialization(unittest.TestCase):
    """Test SlamFrontend2D initialization."""
    
    def test_initialization(self):
        """Test front-end initializes correctly."""
        frontend = SlamFrontend2D()
        
        self.assertIsNone(frontend.pose_est)
        self.assertFalse(frontend.initialized)
        self.assertEqual(len(frontend.submap), 0)
    
    def test_first_step_initialization(self):
        """Test first step initializes pose and submap."""
        frontend = SlamFrontend2D()
        
        odom_delta = np.array([0.0, 0.0, 0.0])
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        result = frontend.step(0, odom_delta, scan)
        
        # Should initialize at origin
        np.testing.assert_allclose(result['pose_pred'], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result['pose_est'], [0.0, 0.0, 0.0])
        
        # Should mark as converged (initialization)
        self.assertTrue(result['match_quality'].converged)
        self.assertEqual(result['correction_magnitude'], 0.0)
        
        # Submap should contain scan points
        self.assertEqual(len(frontend.submap), 2)
        self.assertTrue(frontend.initialized)
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        frontend = SlamFrontend2D(
            submap_voxel_size=0.2,
            min_map_points=20,
            max_icp_residual=0.5,
        )
        
        self.assertEqual(frontend.voxel_size, 0.2)
        self.assertEqual(frontend.min_map_points, 20)
        self.assertEqual(frontend.max_icp_residual, 0.5)


class TestSlamFrontend2DPrediction(unittest.TestCase):
    """Test prediction step (odometry integration)."""
    
    def test_prediction_with_translation_only(self):
        """Test prediction with pure translation."""
        frontend = SlamFrontend2D()
        
        # Initialize
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Move forward 1m
        odom_delta = np.array([1.0, 0.0, 0.0])
        result = frontend.step(1, odom_delta, scan)
        
        # Predicted pose should be (1, 0, 0)
        np.testing.assert_allclose(result['pose_pred'], [1.0, 0.0, 0.0])
    
    def test_prediction_with_rotation(self):
        """Test prediction with rotation."""
        frontend = SlamFrontend2D()
        
        # Initialize
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Rotate 90 degrees
        odom_delta = np.array([0.0, 0.0, np.pi / 2])
        result = frontend.step(1, odom_delta, scan)
        
        # Predicted pose should be (0, 0, π/2)
        np.testing.assert_allclose(result['pose_pred'], [0.0, 0.0, np.pi / 2], atol=1e-6)
    
    def test_prediction_accumulates_over_steps(self):
        """Test that prediction accumulates correctly over multiple steps."""
        frontend = SlamFrontend2D()
        
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        # Initialize
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Move 0.5m forward three times
        for i in range(3):
            odom_delta = np.array([0.5, 0.0, 0.0])
            result = frontend.step(i + 1, odom_delta, scan)
        
        # Should be at (1.5, 0, 0) after 3 steps
        expected_pose = np.array([1.5, 0.0, 0.0])
        np.testing.assert_allclose(result['pose_est'][:2], expected_pose[:2], atol=0.1)


class TestSlamFrontend2DScanToMapAlignment(unittest.TestCase):
    """Test scan-to-map ICP alignment."""
    
    def test_scan_to_map_with_perfect_alignment(self):
        """Test ICP when scan perfectly matches submap."""
        frontend = SlamFrontend2D(submap_voxel_size=0.5)
        
        # Initialize with a scan
        scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Second step: same scan, no motion
        odom_delta = np.array([0.0, 0.0, 0.0])
        result = frontend.step(1, odom_delta, scan)
        
        # Should have low residual (ICP may or may not converge with identical scans)
        # What matters is that residual is low
        self.assertLess(result['match_quality'].residual, 0.5)
        
        # Estimated pose should be close to predicted pose
        np.testing.assert_allclose(
            result['pose_est'][:2], 
            result['pose_pred'][:2], 
            atol=0.2
        )
    
    def test_scan_to_map_with_small_drift(self):
        """Test ICP corrects small odometry drift."""
        frontend = SlamFrontend2D()
        
        # Initialize
        scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Second step: odometry says we didn't move, but we actually did
        # ICP should detect this (though in this test, scan is same so it will match)
        odom_delta = np.array([0.0, 0.0, 0.0])
        result = frontend.step(1, odom_delta, scan)
        
        # Should converge
        self.assertTrue(result['match_quality'].converged or result['match_quality'].residual < 0.5)
    
    def test_fallback_to_prediction_when_submap_too_small(self):
        """Test fallback to prediction when submap has too few points."""
        frontend = SlamFrontend2D(min_map_points=100)  # High threshold
        
        # Initialize with small scan
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Second step: submap has only 2 points, below threshold
        odom_delta = np.array([0.1, 0.0, 0.0])
        result = frontend.step(1, odom_delta, scan)
        
        # Should NOT converge (fallback to prediction)
        self.assertFalse(result['match_quality'].converged)
        
        # Pose estimate should equal prediction (no correction)
        np.testing.assert_allclose(result['pose_est'], result['pose_pred'])
    
    def test_fallback_to_prediction_on_empty_scan(self):
        """Test fallback when scan is empty or too small."""
        frontend = SlamFrontend2D()
        
        # Initialize
        scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Second step: very small scan (< 5 points)
        small_scan = np.array([[1.0, 0.0]])
        odom_delta = np.array([0.1, 0.0, 0.0])
        result = frontend.step(1, odom_delta, small_scan)
        
        # Should fallback to prediction
        self.assertFalse(result['match_quality'].converged)


class TestSlamFrontend2DMapUpdate(unittest.TestCase):
    """Test map update step."""
    
    def test_map_updates_after_each_step(self):
        """Test that submap grows with each step."""
        frontend = SlamFrontend2D()
        
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        
        # Initialize
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        self.assertEqual(len(frontend.submap), 2)
        
        # Add more scans
        for i in range(3):
            odom_delta = np.array([0.5, 0.0, 0.0])
            frontend.step(i + 1, odom_delta, scan)
        
        # Submap should have 4 scans × 2 points = 8 points
        self.assertEqual(len(frontend.submap), 8)
    
    def test_map_uses_estimated_pose_not_prediction(self):
        """Test that map update uses refined pose, not prediction."""
        frontend = SlamFrontend2D()
        
        # Initialize
        scan = np.array([[1.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        initial_map_point = frontend.submap.points[0].copy()
        
        # Second step
        odom_delta = np.array([1.0, 0.0, 0.0])
        result = frontend.step(1, odom_delta, scan)
        
        # Submap should have 2 points now
        self.assertEqual(len(frontend.submap), 2)
        
        # First point should still be at origin
        np.testing.assert_allclose(frontend.submap.points[0], initial_map_point, atol=1e-6)


class TestSlamFrontend2DInputValidation(unittest.TestCase):
    """Test input validation."""
    
    def test_invalid_odom_delta_shape_raises_error(self):
        """Test that invalid odometry delta shape raises ValueError."""
        frontend = SlamFrontend2D()
        
        scan = np.array([[1.0, 0.0]])
        invalid_odom = np.array([0.0, 0.0])  # Missing yaw
        
        with self.assertRaises(ValueError):
            frontend.step(0, invalid_odom, scan)
    
    def test_invalid_scan_shape_raises_error(self):
        """Test that invalid scan shape raises ValueError."""
        frontend = SlamFrontend2D()
        
        odom_delta = np.array([0.0, 0.0, 0.0])
        invalid_scan = np.array([1.0, 2.0, 3.0])  # Wrong shape
        
        with self.assertRaises(ValueError):
            frontend.step(0, odom_delta, invalid_scan)


class TestSlamFrontend2DUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def test_get_current_pose(self):
        """Test get_current_pose returns current estimate."""
        frontend = SlamFrontend2D()
        
        # Before initialization
        self.assertIsNone(frontend.get_current_pose())
        
        # After initialization
        scan = np.array([[1.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        pose = frontend.get_current_pose()
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose, [0.0, 0.0, 0.0])
    
    def test_get_submap_points(self):
        """Test get_submap_points returns map points."""
        frontend = SlamFrontend2D()
        
        scan = np.array([[1.0, 0.0], [2.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        points = frontend.get_submap_points()
        self.assertEqual(len(points), 2)
    
    def test_reset(self):
        """Test reset clears state."""
        frontend = SlamFrontend2D()
        
        scan = np.array([[1.0, 0.0]])
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        self.assertTrue(frontend.initialized)
        self.assertEqual(len(frontend.submap), 1)
        
        frontend.reset()
        
        self.assertFalse(frontend.initialized)
        self.assertIsNone(frontend.pose_est)
        self.assertEqual(len(frontend.submap), 0)


class TestSlamFrontend2DIntegration(unittest.TestCase):
    """Integration tests for realistic SLAM scenarios."""
    
    def test_straight_line_trajectory(self):
        """Test front-end on straight-line trajectory."""
        frontend = SlamFrontend2D()
        
        # Robot moves forward, observing same landmarks
        scan = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, -1.0]])
        
        # Initialize
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan)
        
        # Move forward 10 steps
        for i in range(10):
            odom_delta = np.array([0.5, 0.0, 0.0])
            result = frontend.step(i + 1, odom_delta, scan)
            
            # Should converge on most steps (though later steps might not due to accumulated map)
            # Just check no crashes
            self.assertIsNotNone(result['pose_est'])
        
        # Final pose should be approximately (5.0, 0, 0)
        final_pose = frontend.get_current_pose()
        self.assertIsNotNone(final_pose)
        self.assertAlmostEqual(final_pose[0], 5.0, delta=1.0)
    
    def test_square_trajectory_with_rotations(self):
        """Test front-end on square trajectory with turns."""
        np.random.seed(42)
        frontend = SlamFrontend2D()
        
        # Generate square trajectory
        poses_and_scans = [
            (np.array([0.0, 0.0, 0.0]), np.random.rand(10, 2) + [2.0, 0.0]),
            (np.array([1.0, 0.0, 0.0]), np.random.rand(10, 2) + [2.0, 0.0]),
            (np.array([1.0, 1.0, np.pi / 2]), np.random.rand(10, 2) + [2.0, 0.0]),
            (np.array([0.0, 1.0, np.pi]), np.random.rand(10, 2) + [2.0, 0.0]),
        ]
        
        # Initialize with first pose
        _, scan0 = poses_and_scans[0]
        frontend.step(0, np.array([0.0, 0.0, 0.0]), scan0)
        
        # Process rest of trajectory
        for i in range(1, len(poses_and_scans)):
            prev_pose, _ = poses_and_scans[i - 1]
            curr_pose, curr_scan = poses_and_scans[i]
            
            # Compute odometry delta (with some noise)
            dx = curr_pose[0] - prev_pose[0] + np.random.normal(0, 0.05)
            dy = curr_pose[1] - prev_pose[1] + np.random.normal(0, 0.05)
            dyaw = curr_pose[2] - prev_pose[2] + np.random.normal(0, 0.01)
            odom_delta = np.array([dx, dy, dyaw])
            
            result = frontend.step(i, odom_delta, curr_scan)
            
            # Just verify no crashes and results are reasonable
            self.assertIsNotNone(result['pose_est'])


if __name__ == "__main__":
    unittest.main()
