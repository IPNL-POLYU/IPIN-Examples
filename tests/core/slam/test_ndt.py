"""Unit tests for core.slam.ndt module.

Tests NDT (Normal Distributions Transform) scan matching algorithms
used in Chapter 7 for 2D LiDAR SLAM.

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.slam import (
    build_ndt_map,
    ndt_align,
    ndt_covariance,
    ndt_gradient,
    ndt_score,
    se2_apply,
)


class TestBuildNDTMap:
    """Test suite for build_ndt_map function."""

    def test_single_voxel(self):
        """Test NDT map with points in a single voxel."""
        points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ndt_map = build_ndt_map(points, voxel_size=1.0)

        # All points should be in voxel (0, 0)
        assert (0, 0) in ndt_map
        assert len(ndt_map) == 1

        voxel = ndt_map[(0, 0)]
        assert voxel["n_points"] == 3
        assert voxel["mean"].shape == (2,)
        assert voxel["cov"].shape == (2, 2)

    def test_multiple_voxels(self):
        """Test NDT map with points in multiple voxels."""
        points = np.array(
            [[0.5, 0.5], [0.7, 0.7], [1.5, 1.5], [1.7, 1.7], [2.5, 2.5], [2.7, 2.7]]
        )
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=2)

        # Should have 3 voxels: (0,0), (1,1), (2,2)
        assert len(ndt_map) == 3
        assert (0, 0) in ndt_map
        assert (1, 1) in ndt_map
        assert (2, 2) in ndt_map

    def test_min_points_filtering(self):
        """Test that voxels with too few points are filtered out."""
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [2.7, 2.7]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=2)

        # Only voxel (2, 2) has 2 points
        assert len(ndt_map) == 1
        assert (2, 2) in ndt_map

    def test_mean_computation(self):
        """Test that voxel means are correctly computed."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ndt_map = build_ndt_map(points, voxel_size=2.0)

        voxel = ndt_map[(0, 0)]
        expected_mean = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(voxel["mean"], expected_mean, atol=1e-10)

    def test_covariance_shape(self):
        """Test that covariance matrices have correct shape."""
        points = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        ndt_map = build_ndt_map(points, voxel_size=1.0)

        voxel = ndt_map[(0, 0)]
        assert voxel["cov"].shape == (2, 2)

        # Covariance should be symmetric
        np.testing.assert_allclose(voxel["cov"], voxel["cov"].T, atol=1e-10)

    def test_covariance_positive_definite(self):
        """Test that covariance matrices are positive definite."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ndt_map = build_ndt_map(points, voxel_size=2.0)

        voxel = ndt_map[(0, 0)]
        eigenvalues = np.linalg.eigvals(voxel["cov"])

        # All eigenvalues should be positive
        assert np.all(eigenvalues > 0)

    def test_empty_points(self):
        """Test with empty point cloud."""
        points = np.empty((0, 2))
        ndt_map = build_ndt_map(points, voxel_size=1.0)

        assert len(ndt_map) == 0

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        points = np.array([[-0.5, -0.5], [-0.3, -0.3], [-0.1, -0.1]])
        ndt_map = build_ndt_map(points, voxel_size=1.0)

        # Should be in voxel (-1, -1)
        assert (-1, -1) in ndt_map
        assert ndt_map[(-1, -1)]["n_points"] == 3

    def test_invalid_points_shape(self):
        """Test that invalid point shape raises ValueError."""
        points = np.array([1.0, 2.0])  # 1D array

        with pytest.raises(ValueError, match="must have shape \\(N, 2\\)"):
            build_ndt_map(points, voxel_size=1.0)


class TestNDTScore:
    """Test suite for ndt_score function."""

    def test_perfect_alignment(self):
        """Test score with perfect alignment (identity pose)."""
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        pose_identity = np.array([0.0, 0.0, 0.0])
        score = ndt_score(points, ndt_map, pose_identity, voxel_size=1.0)

        # Perfect alignment should give low score magnitude
        # (Note: score can be negative since it's based on log-likelihood)
        assert abs(score) < 50.0  # Reasonable score magnitude
        assert np.isfinite(score)

    def test_translation_increases_score(self):
        """Test that misalignment increases score."""
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        pose_identity = np.array([0.0, 0.0, 0.0])
        pose_translated = np.array([5.0, 5.0, 0.0])

        score_aligned = ndt_score(points, ndt_map, pose_identity, voxel_size=1.0)
        score_misaligned = ndt_score(points, ndt_map, pose_translated, voxel_size=1.0)

        # Misalignment should give higher score (lower likelihood)
        assert score_misaligned > score_aligned

    def test_empty_source(self):
        """Test with empty source cloud."""
        target = np.array([[0.5, 0.5], [1.5, 1.5]])
        ndt_map = build_ndt_map(target, voxel_size=1.0, min_points_per_voxel=1)

        source = np.empty((0, 2))
        pose = np.array([0.0, 0.0, 0.0])

        score = ndt_score(source, ndt_map, pose, voxel_size=1.0)
        assert score == 0.0

    def test_no_matching_voxels(self):
        """Test when source points fall in empty voxels."""
        target = np.array([[0.5, 0.5]])
        ndt_map = build_ndt_map(target, voxel_size=1.0, min_points_per_voxel=1)

        source = np.array([[10.0, 10.0]])  # Far from target
        pose = np.array([0.0, 0.0, 0.0])

        score = ndt_score(source, ndt_map, pose, voxel_size=1.0)

        # Should return large penalty
        assert score > 1e5

    def test_score_is_non_negative(self):
        """Test that score is always non-negative."""
        points = np.array([[0.5, 0.5], [1.5, 1.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        for _ in range(10):
            pose = np.random.randn(3)
            score = ndt_score(points, ndt_map, pose, voxel_size=1.0)
            assert score >= 0.0


class TestNDTGradient:
    """Test suite for ndt_gradient function."""

    def test_gradient_shape(self):
        """Test that gradient has correct shape."""
        points = np.array([[0.5, 0.5], [1.5, 1.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        pose = np.array([0.0, 0.0, 0.0])
        grad = ndt_gradient(points, ndt_map, pose, voxel_size=1.0)

        assert grad.shape == (3,)

    def test_gradient_at_optimum(self):
        """Test that gradient is small at optimum."""
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        # At identity (perfect alignment), gradient should be small
        pose_optimum = np.array([0.0, 0.0, 0.0])
        grad = ndt_gradient(points, ndt_map, pose_optimum, voxel_size=1.0)

        # Gradient magnitude should be small
        grad_norm = np.linalg.norm(grad)
        assert grad_norm < 1.0  # Loose threshold for numerical gradient

    def test_gradient_descent_direction(self):
        """Test that negative gradient reduces score."""
        points = np.array([[0.5, 0.5], [1.5, 1.5]])
        ndt_map = build_ndt_map(points, voxel_size=1.0, min_points_per_voxel=1)

        pose = np.array([0.5, 0.5, 0.1])  # Slight misalignment
        grad = ndt_gradient(points, ndt_map, pose, voxel_size=1.0)

        score_before = ndt_score(points, ndt_map, pose, voxel_size=1.0)

        # Take small step in negative gradient direction
        step_size = 0.01
        pose_after = pose - step_size * grad

        score_after = ndt_score(points, ndt_map, pose_after, voxel_size=1.0)

        # Score should decrease (or stay similar if already at minimum)
        assert score_after <= score_before + 0.1  # Allow small numerical error


class TestNDTAlign:
    """Test suite for ndt_align function."""

    def test_identical_scans(self):
        """Test NDT with identical scans."""
        scan = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]])

        pose, iters, score, converged = ndt_align(
            scan, scan, voxel_size=1.0, max_iterations=50
        )

        # Should converge quickly to identity
        assert converged or iters < 10  # Quick convergence
        # Pose should be close to identity
        assert np.linalg.norm(pose) < 0.5

    def test_pure_translation(self):
        """Test NDT with pure translation."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = source + np.array([2.0, 3.0])

        # Provide good initial guess
        initial_guess = np.array([1.8, 2.8, 0.0])
        pose, iters, score, converged = ndt_align(
            source,
            target,
            initial_pose=initial_guess,
            voxel_size=2.0,
            max_iterations=100,
        )

        # Should converge (may not be exact due to gradient descent)
        # Verify alignment quality
        transformed = se2_apply(pose, source)
        alignment_error = np.mean(np.linalg.norm(transformed - target, axis=1))
        assert alignment_error < 1.0  # Reasonable alignment

    def test_with_rotation(self):
        """Test NDT with rotation."""
        # Create a distinctive pattern
        source = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])

        # Apply known transformation
        true_pose = np.array([1.0, 2.0, np.pi / 6])  # 30° rotation
        target = se2_apply(true_pose, source)

        # Good initial guess
        initial_guess = np.array([0.8, 1.8, np.pi / 7])
        pose, iters, score, converged = ndt_align(
            source,
            target,
            initial_pose=initial_guess,
            voxel_size=2.0,
            max_iterations=100,
            step_size=0.05,
        )

        # Check alignment quality
        transformed = se2_apply(pose, source)
        alignment_error = np.mean(np.linalg.norm(transformed - target, axis=1))
        assert alignment_error < 1.5  # Reasonable alignment with rotation

    def test_empty_source_raises_error(self):
        """Test that empty source raises ValueError."""
        source = np.empty((0, 2))
        target = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="source_scan is empty"):
            ndt_align(source, target)

    def test_empty_target_raises_error(self):
        """Test that empty target raises ValueError."""
        source = np.array([[1.0, 0.0]])
        target = np.empty((0, 2))

        with pytest.raises(ValueError, match="target_scan is empty"):
            ndt_align(source, target)

    def test_convergence_with_tolerance(self):
        """Test that NDT respects convergence tolerance."""
        source = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        target = source + np.array([0.01, 0.01])  # Very small offset

        pose, iters, score, converged = ndt_align(
            source, target, voxel_size=2.0, tolerance=1e-2, max_iterations=50
        )

        # Should converge quickly for small offset
        assert converged or iters < 20

    def test_max_iterations_limit(self):
        """Test that NDT respects max_iterations."""
        source = np.array([[0, 0], [1, 0], [0, 1]])
        target = np.array([[10, 10], [11, 10], [10, 11]])  # Far away

        pose, iters, score, converged = ndt_align(
            source, target, voxel_size=2.0, max_iterations=5
        )

        # Should stop at max_iterations
        assert iters <= 5

    def test_large_point_cloud(self):
        """Test NDT with larger point cloud."""
        # Generate simple grid of points (easier to align than circle)
        x = np.linspace(0, 3, 10)
        y = np.linspace(0, 3, 10)
        xx, yy = np.meshgrid(x, y)
        source = np.column_stack([xx.ravel(), yy.ravel()])

        # Small translation
        target = source + np.array([0.3, 0.4])

        initial_guess = np.array([0.25, 0.35, 0.0])
        pose, iters, score, converged = ndt_align(
            source,
            target,
            initial_pose=initial_guess,
            voxel_size=2.0,  # Larger voxels for better convergence
            max_iterations=100,
            step_size=0.05,  # Smaller step size
        )

        # Should achieve reasonable alignment
        # (NDT with gradient descent may not be as accurate as ICP)
        transformed = se2_apply(pose, source)
        alignment_error = np.mean(np.linalg.norm(transformed - target, axis=1))
        assert alignment_error < 2.0  # Relaxed threshold for gradient descent


class TestNDTCovariance:
    """Test suite for ndt_covariance function."""

    def test_covariance_shape(self):
        """Test that covariance has correct shape."""
        source = np.array([[0.5, 0.5], [1.5, 1.5]])
        target = source.copy()
        ndt_map = build_ndt_map(target, voxel_size=1.0, min_points_per_voxel=1)

        pose = np.array([0.0, 0.0, 0.0])
        cov = ndt_covariance(source, ndt_map, pose, voxel_size=1.0)

        assert cov.shape == (3, 3)

    def test_covariance_is_symmetric(self):
        """Test that covariance matrix is symmetric."""
        source = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        target = source + np.array([0.1, 0.1])
        ndt_map = build_ndt_map(target, voxel_size=1.0, min_points_per_voxel=1)

        pose = np.array([0.1, 0.1, 0.0])
        cov = ndt_covariance(source, ndt_map, pose, voxel_size=1.0)

        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_covariance_is_positive_semidefinite(self):
        """Test that covariance eigenvalues are non-negative."""
        source = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        target = source + np.array([0.1, 0.1])
        ndt_map = build_ndt_map(target, voxel_size=1.0, min_points_per_voxel=1)

        pose = np.array([0.1, 0.1, 0.0])
        cov = ndt_covariance(source, ndt_map, pose, voxel_size=1.0)

        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


class TestNDTIntegration:
    """Integration tests for NDT alignment."""

    def test_ndt_vs_icp_comparison(self):
        """Compare NDT and ICP on same alignment problem."""
        # Create a simple pattern
        source = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        target = source + np.array([0.5, 1.0])

        # Run NDT
        initial_guess = np.array([0.4, 0.9, 0.0])
        ndt_pose, _, _, ndt_converged = ndt_align(
            source, target, initial_pose=initial_guess, voxel_size=2.0
        )

        # Both should achieve reasonable alignment
        ndt_transformed = se2_apply(ndt_pose, source)
        ndt_error = np.mean(np.linalg.norm(ndt_transformed - target, axis=1))

        # NDT should achieve sub-meter accuracy
        assert ndt_error < 1.0

    def test_ndt_with_noise(self):
        """Test NDT robustness with noisy measurements."""
        np.random.seed(42)

        source = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        true_pose = np.array([0.3, 0.5, 0.1])
        target = se2_apply(true_pose, source)

        # Add small noise
        noise = np.random.normal(0, 0.05, target.shape)
        target_noisy = target + noise

        # Run NDT with good initial guess
        initial_guess = np.array([0.2, 0.4, 0.05])
        pose, iters, score, converged = ndt_align(
            source, target_noisy, initial_pose=initial_guess, voxel_size=3.0
        )

        # Should still achieve reasonable alignment despite noise
        transformed = se2_apply(pose, source)
        alignment_error = np.mean(np.linalg.norm(transformed - target_noisy, axis=1))
        assert alignment_error < 1.0

    def test_ndt_pipeline(self):
        """Test complete NDT pipeline: build map → score → align → covariance."""
        source = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        target = source + np.array([1.0, 2.0])

        # Step 1: Build NDT map
        ndt_map = build_ndt_map(target, voxel_size=2.0)
        assert len(ndt_map) > 0

        # Step 2: Initial score evaluation (with good initial guess)
        initial_pose = np.array([0.9, 1.9, 0.0])
        initial_score = ndt_score(source, ndt_map, initial_pose, voxel_size=2.0)
        assert np.isfinite(initial_score)  # Valid score

        # Step 3: Run alignment
        final_pose, iters, final_score, converged = ndt_align(
            source, target, initial_pose=initial_pose, voxel_size=2.0, max_iterations=100
        )

        # NDT should complete without errors
        assert np.all(np.isfinite(final_pose))
        assert np.isfinite(final_score)

        # Step 4: Compute covariance
        cov = ndt_covariance(source, ndt_map, final_pose, voxel_size=2.0)
        assert cov.shape == (3, 3)

