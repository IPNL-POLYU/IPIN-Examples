"""Integration test: ICP convergence on fixed synthetic scans.

Validates that ICP reliably recovers known transformations on
synthetic scan pairs. This is a scenario-level regression test.

References:
    - Eqs. (7.10)-(7.11): ICP algorithm
    - Section 7.2.1: ICP scan matching

Author: Li-Ta Hsu
Date: 2024
"""

import numpy as np
import pytest

from core.slam import icp_point_to_point, se2_apply


class TestICPConvergence:
    """Scenario-level tests for ICP convergence."""

    @pytest.fixture
    def simple_scan(self):
        """Generate a simple rectangular scan pattern."""
        np.random.seed(42)
        # Create rectangle with some internal points
        scan = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0, 1], [3, 1],
            [0, 2], [3, 2],
            [0, 3], [1, 3], [2, 3], [3, 3],
        ], dtype=float)
        # Add small noise
        scan += np.random.normal(0, 0.01, scan.shape)
        return scan

    @pytest.fixture
    def dense_scan(self):
        """Generate a denser scan with more points."""
        np.random.seed(123)
        # Create grid pattern
        x = np.linspace(0, 5, 20)
        y = np.linspace(0, 5, 20)
        xx, yy = np.meshgrid(x, y)
        scan = np.column_stack([xx.ravel(), yy.ravel()])
        # Add noise
        scan += np.random.normal(0, 0.02, scan.shape)
        return scan

    def test_icp_pure_translation_small(self, simple_scan):
        """Test ICP recovers small pure translation."""
        true_translation = np.array([0.5, 0.3, 0.0])
        target = se2_apply(true_translation, simple_scan)

        pose, iters, residual, converged = icp_point_to_point(
            simple_scan,
            target,
            max_iterations=50,
            tolerance=1e-6,
        )

        # Assertions
        assert converged, "ICP should converge for simple translation"
        assert iters < 20, f"ICP should converge quickly, took {iters} iterations"
        np.testing.assert_allclose(pose, true_translation, atol=0.05,
                                   err_msg="ICP should recover true translation")
        assert residual < 0.1, f"Final residual {residual} too high"

    def test_icp_pure_translation_large(self, simple_scan):
        """Test ICP recovers larger pure translation."""
        true_translation = np.array([2.0, 1.5, 0.0])
        target = se2_apply(true_translation, simple_scan)

        pose, iters, residual, converged = icp_point_to_point(
            simple_scan,
            target,
            max_iterations=50,
            tolerance=1e-6,
        )

        # Assertions
        assert converged, "ICP should converge for large translation"
        np.testing.assert_allclose(pose, true_translation, atol=0.1,
                                   err_msg="ICP should recover true translation")
        assert residual < 1.0, f"Final residual {residual} too high"

    def test_icp_rotation_30deg(self, simple_scan):
        """Test ICP recovers 30-degree rotation WITH good initial guess."""
        true_pose = np.array([0.5, 0.5, np.pi / 6])  # 30 degrees
        target = se2_apply(true_pose, simple_scan)

        # ICP struggles with large rotations from zero initial guess
        # Provide a reasonable initial guess (within ~20 degrees)
        initial_guess = np.array([0.3, 0.3, np.pi / 8])
        
        pose, iters, residual, converged = icp_point_to_point(
            simple_scan,
            target,
            initial_pose=initial_guess,
            max_iterations=100,
            tolerance=1e-6,
        )

        # Assertions
        assert converged, "ICP should converge with rotation and good guess"
        assert iters < 50, f"ICP should converge reasonably fast, took {iters}"
        np.testing.assert_allclose(pose, true_pose, atol=0.15,
                                   err_msg="ICP should recover rotation + translation")
        assert residual < 1.0, f"Final residual {residual} too high"

    def test_icp_rotation_small(self, simple_scan):
        """Test ICP with moderate rotation (15 degrees)."""
        true_pose = np.array([0.8, 0.6, np.pi / 12])  # 15 degrees
        target = se2_apply(true_pose, simple_scan)

        # Moderate rotation with reasonable initial guess
        initial_guess = np.array([0.7, 0.5, np.pi / 15])  # 12 degrees
        
        pose, iters, residual, converged = icp_point_to_point(
            simple_scan,
            target,
            initial_pose=initial_guess,
            max_iterations=100,
            tolerance=1e-6,
        )

        # Assertions
        assert converged, "ICP should converge with moderate rotation"
        np.testing.assert_allclose(pose, true_pose, atol=0.15,
                                   err_msg="ICP should recover moderate rotation")
        assert residual < 1.0, f"Final residual {residual} too high"

    def test_icp_with_good_initial_guess(self, dense_scan):
        """Test ICP converges faster with good initial guess."""
        true_pose = np.array([1.5, 2.0, np.pi / 8])  # Smaller rotation (22.5 deg)
        target = se2_apply(true_pose, dense_scan)

        # With moderate initial guess
        moderate_guess = np.array([1.0, 1.5, 0.1])
        pose_moderate, iters_moderate, _, converged_moderate = icp_point_to_point(
            dense_scan, target, initial_pose=moderate_guess,
            max_iterations=100,
        )

        # With good initial guess
        good_guess = true_pose + np.array([0.1, 0.1, 0.05])
        pose_with_init, iters_with_init, _, converged_with_init = icp_point_to_point(
            dense_scan, target, initial_pose=good_guess,
            max_iterations=100,
        )

        # Assertions
        assert converged_moderate, "ICP should converge with moderate guess"
        assert converged_with_init, "ICP should converge with good guess"
        assert iters_with_init <= iters_moderate, \
            "Better initial guess should not increase iterations"
        
        # Good guess should be reasonably accurate
        np.testing.assert_allclose(pose_with_init, true_pose, atol=0.25)

    def test_icp_with_noise(self, simple_scan):
        """Test ICP robustness to measurement noise."""
        np.random.seed(999)
        true_pose = np.array([0.5, 0.3, np.pi / 16])  # Small rotation (11.25 deg)
        
        # Apply transformation and add noise
        target_clean = se2_apply(true_pose, simple_scan)
        noise = np.random.normal(0, 0.03, target_clean.shape)  # 3cm noise
        target_noisy = target_clean + noise

        # Provide reasonable initial guess to help convergence with noise
        initial_guess = np.array([0.4, 0.2, 0.1])
        
        pose, iters, residual, converged = icp_point_to_point(
            simple_scan,
            target_noisy,
            initial_pose=initial_guess,
            max_iterations=100,
            tolerance=1e-5,
        )

        # Assertions
        assert converged, "ICP should converge despite noise"
        # Allow larger tolerance due to noise
        np.testing.assert_allclose(pose, true_pose, atol=0.25,
                                   err_msg="ICP should approximately recover pose with noise")
        # Residual will be higher due to noise
        assert residual < 5.0, f"Residual {residual} unexpectedly high"

    def test_icp_partial_overlap(self, dense_scan):
        """Test ICP with partial scan overlap."""
        np.random.seed(555)
        true_pose = np.array([0.5, 0.5, 0.0])
        
        # Take only subset of points to simulate partial overlap
        source_subset = dense_scan[::2]  # Every other point
        target = se2_apply(true_pose, dense_scan)

        # Provide good initial guess to help with partial overlap
        initial_guess = np.array([0.4, 0.4, 0.0])
        
        pose, iters, residual, converged = icp_point_to_point(
            source_subset,
            target,
            initial_pose=initial_guess,
            max_iterations=100,
            tolerance=1e-6,
            max_correspondence_distance=1.5,  # More lenient for partial overlap
        )

        # Assertions
        assert converged, "ICP should handle partial overlap"
        np.testing.assert_allclose(pose, true_pose, atol=0.2,
                                   err_msg="ICP should work with partial overlap")

    def test_icp_fixed_seed_reproducibility(self, simple_scan):
        """Test that ICP is reproducible with fixed random seed."""
        np.random.seed(777)
        true_pose = np.array([1.0, 1.0, np.pi / 6])
        target = se2_apply(true_pose, simple_scan)

        # Run ICP twice with same conditions
        pose1, iters1, res1, conv1 = icp_point_to_point(
            simple_scan, target, max_iterations=50
        )
        pose2, iters2, res2, conv2 = icp_point_to_point(
            simple_scan, target, max_iterations=50
        )

        # Should be identical
        np.testing.assert_array_equal(pose1, pose2, err_msg="ICP should be deterministic")
        assert iters1 == iters2, "Iteration count should be identical"
        assert res1 == res2, "Residual should be identical"
        assert conv1 == conv2, "Convergence status should be identical"


class TestICPRegressionThresholds:
    """Regression tests with fixed RMSE thresholds."""

    def test_icp_accuracy_threshold_small_noise(self):
        """Regression: ICP should achieve <10cm RMSE on clean scans with good guess."""
        np.random.seed(100)
        
        # Generate structured scan
        scan = np.array([
            [i, j]
            for i in np.linspace(0, 5, 10)
            for j in np.linspace(0, 5, 10)
        ]) + np.random.normal(0, 0.01, (100, 2))

        true_pose = np.array([1.0, 1.5, np.pi / 16])  # ~11 degrees
        target = se2_apply(true_pose, scan)

        # Provide good initial guess
        initial_guess = np.array([0.9, 1.4, np.pi / 20])
        
        pose, _, residual, converged = icp_point_to_point(
            scan, target, initial_pose=initial_guess, max_iterations=100
        )

        # Regression threshold
        assert converged, "Must converge"
        pose_error = np.linalg.norm(pose[:2] - true_pose[:2])
        assert pose_error < 0.10, f"Position error {pose_error:.4f}m exceeds 10cm threshold"
        
        angle_error = np.abs(pose[2] - true_pose[2])
        assert angle_error < np.deg2rad(5), f"Angle error {np.rad2deg(angle_error):.2f}deg exceeds 5deg threshold"

    def test_icp_accuracy_threshold_with_noise(self):
        """Regression: ICP should achieve <15cm RMSE with 5cm noise."""
        np.random.seed(200)
        
        # Generate scan with moderate noise
        scan = np.random.rand(50, 2) * 5
        scan += np.random.normal(0, 0.01, scan.shape)

        true_pose = np.array([0.8, 1.2, np.pi / 8])
        target_clean = se2_apply(true_pose, scan)
        target = target_clean + np.random.normal(0, 0.05, target_clean.shape)

        pose, _, _, converged = icp_point_to_point(
            scan, target, max_iterations=100
        )

        # Regression threshold (more lenient due to noise)
        assert converged, "Must converge"
        pose_error = np.linalg.norm(pose[:2] - true_pose[:2])
        assert pose_error < 0.15, f"Position error {pose_error:.4f}m exceeds 15cm threshold with noise"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

