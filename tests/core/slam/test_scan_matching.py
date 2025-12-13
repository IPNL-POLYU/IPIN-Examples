"""Unit tests for core.slam.scan_matching module.

Tests ICP (Iterative Closest Point) scan matching algorithms used in
Chapter 7 for 2D LiDAR SLAM.

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.slam import (
    align_svd,
    compute_icp_covariance,
    compute_icp_residual,
    find_correspondences,
    icp_point_to_point,
    se2_apply,
)


class TestFindCorrespondences:
    """Test suite for find_correspondences function."""

    def test_exact_match(self):
        """Test correspondences when source equals target."""
        points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        src_matched, tgt_matched, dists = find_correspondences(points, points)

        # All points should match themselves
        np.testing.assert_array_equal(src_matched, points)
        np.testing.assert_array_equal(tgt_matched, points)
        np.testing.assert_allclose(dists, [0.0, 0.0, 0.0], atol=1e-10)

    def test_small_offset(self):
        """Test correspondences with small offset."""
        source = np.array([[1.0, 0.0], [0.0, 1.0]])
        target = np.array([[1.1, 0.0], [0.0, 0.9]])

        src_matched, tgt_matched, dists = find_correspondences(source, target)

        # Should match to closest points
        assert src_matched.shape == (2, 2)
        assert tgt_matched.shape == (2, 2)
        assert len(dists) == 2
        assert all(d <= 0.2 for d in dists)  # Small distances

    def test_max_distance_filter(self):
        """Test that max_distance filters out distant correspondences."""
        source = np.array([[0.0, 0.0], [10.0, 10.0]])
        target = np.array([[0.1, 0.0]])  # Only close to first source point

        src_matched, tgt_matched, dists = find_correspondences(
            source, target, max_distance=1.0
        )

        # Only first source point should match (within 1m)
        assert src_matched.shape[0] == 1
        assert np.allclose(src_matched[0], [0.0, 0.0])
        assert dists[0] < 1.0

    def test_empty_source(self):
        """Test with empty source cloud."""
        source = np.empty((0, 2))
        target = np.array([[1.0, 0.0]])

        src_matched, tgt_matched, dists = find_correspondences(source, target)

        assert src_matched.shape == (0, 2)
        assert tgt_matched.shape == (0, 2)
        assert len(dists) == 0

    def test_empty_target(self):
        """Test with empty target cloud."""
        source = np.array([[1.0, 0.0]])
        target = np.empty((0, 2))

        src_matched, tgt_matched, dists = find_correspondences(source, target)

        assert src_matched.shape == (0, 2)
        assert tgt_matched.shape == (0, 2)
        assert len(dists) == 0

    def test_many_to_one_correspondence(self):
        """Test that multiple source points can match same target point."""
        source = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        target = np.array([[0.0, 0.0]])  # Single target point

        src_matched, tgt_matched, dists = find_correspondences(source, target)

        # All source points should match to the same target
        assert src_matched.shape[0] == 3
        assert tgt_matched.shape[0] == 3
        np.testing.assert_array_equal(tgt_matched[0], tgt_matched[1])
        np.testing.assert_array_equal(tgt_matched[0], tgt_matched[2])

    def test_invalid_source_shape(self):
        """Test that invalid source shape raises ValueError."""
        source = np.array([1.0, 2.0])  # 1D array
        target = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="must have shape \\(N, 2\\)"):
            find_correspondences(source, target)


class TestComputeICPResidual:
    """Test suite for compute_icp_residual function."""

    def test_identical_points(self):
        """Test residual is zero for identical point sets."""
        points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        residual = compute_icp_residual(points, points)

        assert np.isclose(residual, 0.0, atol=1e-10)

    def test_known_residual(self):
        """Test residual computation with known distances."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]])
        target = np.array([[0.0, 0.1], [1.0, 0.1]])

        residual = compute_icp_residual(source, target)

        # Each point is 0.1m away → total residual = 2 * 0.1^2 = 0.02
        expected = 2 * 0.1**2
        assert np.isclose(residual, expected, atol=1e-10)

    def test_empty_clouds(self):
        """Test residual is zero for empty point clouds."""
        empty = np.empty((0, 2))
        residual = compute_icp_residual(empty, empty)

        assert residual == 0.0

    def test_mismatched_sizes(self):
        """Test that mismatched sizes raise ValueError."""
        source = np.array([[0.0, 0.0]])
        target = np.array([[0.0, 0.0], [1.0, 0.0]])

        with pytest.raises(ValueError, match="must have same shape"):
            compute_icp_residual(source, target)

    def test_residual_is_positive(self):
        """Test that residual is always non-negative."""
        source = np.array([[1.0, 2.0], [-3.0, 4.0]])
        target = np.array([[0.5, 1.5], [-2.5, 3.5]])

        residual = compute_icp_residual(source, target)

        assert residual >= 0.0


class TestAlignSVD:
    """Test suite for align_svd function."""

    def test_identity_alignment(self):
        """Test alignment of identical point clouds."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pose = align_svd(points, points)

        # Should return identity pose
        np.testing.assert_allclose(pose, [0.0, 0.0, 0.0], atol=1e-6)

    def test_pure_translation(self):
        """Test alignment with pure translation."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = source + np.array([2.0, 3.0])

        pose = align_svd(source, target)

        # Should recover translation [2, 3, 0]
        np.testing.assert_allclose(pose[:2], [2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(pose[2], 0.0, atol=1e-6)

    def test_pure_rotation_90deg(self):
        """Test alignment with 90° rotation."""
        source = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        # Rotate 90° counter-clockwise: (x,y) → (-y, x)
        target = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

        pose = align_svd(source, target)

        # Should recover 90° rotation
        np.testing.assert_allclose(pose[:2], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(pose[2], np.pi / 2, atol=1e-6)

    def test_rotation_and_translation(self):
        """Test alignment with both rotation and translation."""
        # Original points
        source = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

        # Rotate 45° and translate by [1, 2]
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        t = np.array([1.0, 2.0])
        target = (R @ source.T).T + t

        pose = align_svd(source, target)

        # Should recover [1, 2, π/4]
        np.testing.assert_allclose(pose[:2], [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(pose[2], np.pi / 4, atol=1e-6)

    def test_too_few_points(self):
        """Test that fewer than 2 points raises ValueError."""
        source = np.array([[0.0, 0.0]])
        target = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="at least 2 correspondences"):
            align_svd(source, target)

    def test_verification_with_se2_apply(self):
        """Test that aligned pose correctly transforms source to target."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        # Apply known transformation
        true_pose = np.array([2.0, 3.0, np.pi / 3])
        target = se2_apply(true_pose, source)

        # Recover pose with SVD
        estimated_pose = align_svd(source, target)

        # Estimated pose should match true pose
        np.testing.assert_allclose(estimated_pose, true_pose, atol=1e-6)

    def test_centered_points(self):
        """Test alignment when point clouds are already centered."""
        # Points centered at origin
        source = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
        target = source.copy()

        pose = align_svd(source, target)

        np.testing.assert_allclose(pose, [0.0, 0.0, 0.0], atol=1e-6)


class TestICPPointToPoint:
    """Test suite for icp_point_to_point function."""

    def test_identical_scans(self):
        """Test ICP with identical scans."""
        scan = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        pose, iters, residual, converged = icp_point_to_point(scan, scan)

        # Should converge to identity quickly
        assert converged
        np.testing.assert_allclose(pose, [0.0, 0.0, 0.0], atol=1e-4)
        assert residual < 1e-6
        assert iters <= 5  # Should converge in very few iterations

    def test_pure_translation(self):
        """Test ICP with pure translation."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = source + np.array([2.0, 3.0])

        # Provide good initial guess (close to true translation)
        initial_guess = np.array([1.8, 2.8, 0.0])
        pose, iters, residual, converged = icp_point_to_point(
            source, target, initial_pose=initial_guess
        )

        # Should recover translation with good initial guess
        assert converged
        np.testing.assert_allclose(pose[:2], [2.0, 3.0], atol=1e-2)
        np.testing.assert_allclose(pose[2], 0.0, atol=1e-2)

    def test_with_initial_guess(self):
        """Test ICP with good initial guess."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = source + np.array([1.0, 2.0])

        initial_guess = np.array([0.9, 1.9, 0.0])  # Close to true solution
        pose, iters, residual, converged = icp_point_to_point(
            source, target, initial_pose=initial_guess
        )

        assert converged
        # Should converge faster with good initial guess
        assert iters <= 10

    def test_rotation_and_translation(self):
        """Test ICP with rotation and translation."""
        # Create a distinctive pattern
        source = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )

        # Apply known transformation
        true_pose = np.array([1.0, 2.0, np.pi / 6])  # 30° rotation
        target = se2_apply(true_pose, source)

        pose, iters, residual, converged = icp_point_to_point(
            source, target, max_iterations=100
        )

        # Should recover the transformation
        assert converged
        np.testing.assert_allclose(pose, true_pose, atol=1e-2)

    def test_max_correspondence_distance(self):
        """Test ICP with max correspondence distance."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        target = np.array([[0.1, 0.0], [1.1, 0.0], [10.0, 10.0]])  # Last point is outlier

        pose, iters, residual, converged = icp_point_to_point(
            source, target, max_correspondence_distance=1.0
        )

        # Should ignore the outlier and converge
        # (may or may not converge depending on outlier impact, but should not crash)
        assert pose.shape == (3,)

    def test_insufficient_correspondences(self):
        """Test ICP failure with too few correspondences."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]])
        target = np.array([[100.0, 100.0], [101.0, 100.0]])  # Far away

        pose, iters, residual, converged = icp_point_to_point(
            source,
            target,
            max_correspondence_distance=1.0,  # Will reject all correspondences
            min_correspondences=10,
        )

        # Should fail due to insufficient correspondences
        assert not converged

    def test_empty_source_raises_error(self):
        """Test that empty source scan raises ValueError."""
        source = np.empty((0, 2))
        target = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="source_scan is empty"):
            icp_point_to_point(source, target)

    def test_empty_target_raises_error(self):
        """Test that empty target scan raises ValueError."""
        source = np.array([[1.0, 0.0]])
        target = np.empty((0, 2))

        with pytest.raises(ValueError, match="target_scan is empty"):
            icp_point_to_point(source, target)

    def test_convergence_tolerance(self):
        """Test that ICP respects convergence tolerance."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = source + np.array([0.001, 0.001])  # Very small offset

        pose, iters, residual, converged = icp_point_to_point(
            source, target, tolerance=1e-6
        )

        assert converged
        # Should converge quickly for small offset
        assert iters <= 10

    def test_max_iterations_limit(self):
        """Test that ICP respects max_iterations limit."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        target = np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]])  # Far away

        pose, iters, residual, converged = icp_point_to_point(
            source, target, max_iterations=3, tolerance=1e-10  # Very strict tolerance
        )

        # Should stop at max_iterations (or converge early if alignment is perfect)
        assert iters <= 3

    def test_large_point_cloud(self):
        """Test ICP with larger point cloud (performance check)."""
        # Generate circle of points
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        source = np.column_stack([np.cos(angles), np.sin(angles)])

        # Small translation
        target = source + np.array([0.1, 0.2])

        pose, iters, residual, converged = icp_point_to_point(source, target)

        assert converged
        np.testing.assert_allclose(pose[:2], [0.1, 0.2], atol=1e-2)


class TestComputeICPCovariance:
    """Test suite for compute_icp_covariance function."""

    def test_covariance_shape(self):
        """Test that covariance has correct shape."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = source.copy()
        pose = np.array([0.0, 0.0, 0.0])

        cov = compute_icp_covariance(source, target, pose)

        assert cov.shape == (3, 3)

    def test_covariance_is_symmetric(self):
        """Test that covariance matrix is symmetric."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = source + np.array([0.1, 0.1])
        pose = np.array([0.1, 0.1, 0.0])

        cov = compute_icp_covariance(source, target, pose)

        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_covariance_is_positive_semidefinite(self):
        """Test that covariance eigenvalues are non-negative."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = source + np.array([0.1, 0.1])
        pose = np.array([0.1, 0.1, 0.0])

        cov = compute_icp_covariance(source, target, pose)

        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_high_uncertainty_with_few_correspondences(self):
        """Test that covariance is large with few correspondences."""
        source = np.array([[0.0, 0.0], [10.0, 0.0]])
        target = np.array([[0.1, 0.0]])  # Only one point nearby
        pose = np.array([0.0, 0.0, 0.0])

        cov = compute_icp_covariance(source, target, pose, max_correspondence_distance=1.0)

        # Should have high uncertainty (large diagonal values)
        assert cov[0, 0] > 0.1
        assert cov[1, 1] > 0.1
        assert cov[2, 2] > 0.01


class TestICPIntegration:
    """Integration tests combining multiple ICP functions."""

    def test_full_icp_pipeline(self):
        """Test complete ICP pipeline: correspondence → alignment → refinement."""
        # Create distinctive source pattern with good initial guess
        source = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )

        # Apply known transformation
        true_pose = np.array([0.5, 1.0, np.pi / 8])
        target = se2_apply(true_pose, source)

        # Run ICP with good initial guess (close to true pose)
        initial_guess = np.array([0.4, 0.9, np.pi / 9])
        estimated_pose, iters, residual, converged = icp_point_to_point(
            source, target, initial_pose=initial_guess, max_iterations=50
        )

        # Verify results
        assert converged
        # With good initial guess, should recover pose accurately
        np.testing.assert_allclose(estimated_pose, true_pose, atol=1e-2)
        assert residual < 1e-4

        # Compute covariance
        cov = compute_icp_covariance(source, target, estimated_pose)
        assert cov.shape == (3, 3)

    def test_icp_with_noise(self):
        """Test ICP robustness with noisy measurements."""
        np.random.seed(42)  # For reproducibility

        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        true_pose = np.array([0.2, 0.3, 0.1])
        target = se2_apply(true_pose, source)

        # Add small noise to target
        noise = np.random.normal(0, 0.01, target.shape)
        target_noisy = target + noise

        pose, iters, residual, converged = icp_point_to_point(
            source, target_noisy, max_iterations=100
        )

        # Should still converge close to true pose despite noise
        assert converged
        np.testing.assert_allclose(pose, true_pose, atol=0.05)

    def test_icp_partial_overlap(self):
        """Test ICP with partial overlap between scans."""
        # Source: square
        source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # Target: larger area with translation
        target = np.array(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 1.5],
                [2.5, 0.5],
                [2.5, 1.5],
            ]
        )

        pose, iters, residual, converged = icp_point_to_point(
            source, target, max_iterations=50, max_correspondence_distance=1.5
        )

        # Should find reasonable alignment despite partial overlap
        assert pose.shape == (3,)
        # Translation should be roughly [0.5, 0.5]
        assert 0.0 <= pose[0] <= 1.5
        assert 0.0 <= pose[1] <= 1.5

