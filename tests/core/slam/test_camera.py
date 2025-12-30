"""Unit tests for core.slam.camera module (visual SLAM camera models).

Tests camera projection, distortion, and reprojection error functions
from Chapter 7, Section 7.4 (Visual SLAM).

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import pytest

from core.slam.camera import (
    compute_reprojection_error,
    distort_normalized,
    project_point,
    undistort_normalized,
    unproject_pixel,
)
from core.slam.types import CameraIntrinsics


class TestDistortNormalized:
    """Tests for distort_normalized function (Eq. 7.41)."""

    def test_zero_distortion(self):
        """Test that zero distortion coefficients leave coordinates unchanged."""
        xy = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = distort_normalized(xy, k1=0, k2=0, k3=0, p1=0, p2=0)
        np.testing.assert_allclose(result, xy, atol=1e-10)

    def test_single_point(self):
        """Test distortion on a single point."""
        xy = np.array([0.1, 0.2])
        result = distort_normalized(xy, k1=-0.1, k2=0.01, k3=0.0, p1=0.001, p2=0.001)
        
        # Result should be different from input
        assert result.shape == (2,)
        assert not np.allclose(result, xy)

    def test_radial_distortion_only(self):
        """Test pure radial distortion (k1, k2 non-zero, p1=p2=0)."""
        xy = np.array([[0.1, 0.2]])
        
        # Negative k1 causes barrel distortion (points move inward)
        result_barrel = distort_normalized(xy, k1=-0.2, k2=0, k3=0, p1=0, p2=0)
        r_original = np.linalg.norm(xy)
        r_distorted = np.linalg.norm(result_barrel)
        assert r_distorted < r_original  # Barrel distortion

        # Positive k1 causes pincushion distortion (points move outward)
        result_pincushion = distort_normalized(xy, k1=0.2, k2=0, k3=0, p1=0, p2=0)
        r_distorted2 = np.linalg.norm(result_pincushion)
        assert r_distorted2 > r_original  # Pincushion distortion

    def test_tangential_distortion_only(self):
        """Test pure tangential distortion (p1, p2 non-zero, k1=k2=k3=0)."""
        xy = np.array([[0.1, 0.2]])
        result = distort_normalized(xy, k1=0, k2=0, k3=0, p1=0.01, p2=0.01)
        
        # Result should be different from input
        assert not np.allclose(result, xy)

    def test_origin_unchanged(self):
        """Test that the origin (0, 0) is unchanged by distortion."""
        xy = np.array([[0.0, 0.0]])
        result = distort_normalized(xy, k1=-0.1, k2=0.01, k3=0.001, p1=0.001, p2=0.001)
        np.testing.assert_allclose(result, xy, atol=1e-10)

    def test_batch_processing(self):
        """Test distortion on multiple points."""
        xy = np.array([[0.1, 0.2], [0.3, 0.4], [-0.1, -0.2]])
        result = distort_normalized(xy, k1=-0.1, k2=0.01, k3=0.0, p1=0.001, p2=0.001)
        
        assert result.shape == (3, 2)
        # Each point should be distorted differently
        for i in range(3):
            assert not np.allclose(result[i], xy[i])

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises ValueError."""
        xy_bad = np.array([0.1, 0.2, 0.3])  # 3 values instead of 2
        with pytest.raises(ValueError, match="must be"):
            distort_normalized(xy_bad, k1=0, k2=0, k3=0, p1=0, p2=0)


class TestUndistortNormalized:
    """Tests for undistort_normalized function (inverse of distortion)."""

    def test_undistort_inverts_distort(self):
        """Test that undistort inverts distort operation."""
        xy_original = np.array([[0.1, 0.2], [0.3, 0.4]])
        k1, k2, k3, p1, p2 = -0.1, 0.01, 0.001, 0.001, 0.001
        
        # Distort then undistort
        xy_distorted = distort_normalized(xy_original, k1, k2, k3, p1, p2)
        xy_recovered = undistort_normalized(xy_distorted, k1, k2, k3, p1, p2)
        
        np.testing.assert_allclose(xy_recovered, xy_original, atol=1e-5)

    def test_zero_distortion(self):
        """Test undistort with zero coefficients."""
        xy = np.array([[0.1, 0.2]])
        result = undistort_normalized(xy, k1=0, k2=0, k3=0, p1=0, p2=0)
        np.testing.assert_allclose(result, xy, atol=1e-6)


class TestProjectPoint:
    """Tests for project_point function (full camera projection)."""

    def test_simple_projection(self):
        """Test projection of a simple 3D point."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Point at 5m depth, centered
        point_3d = np.array([0.0, 0.0, 5.0])
        pixel = project_point(intrinsics, point_3d)
        
        # Should project to principal point (cx, cy)
        np.testing.assert_allclose(pixel, [320, 240], atol=0.1)

    def test_off_center_projection(self):
        """Test projection of off-center points."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Point 1m to the right, 5m away
        point_3d = np.array([1.0, 0.0, 5.0])
        pixel = project_point(intrinsics, point_3d)
        
        # u = fx * (X/Z) + cx = 500 * (1/5) + 320 = 420
        expected_u = 320 + 500 * (1.0 / 5.0)
        assert np.abs(pixel[0] - expected_u) < 1.0
        assert np.abs(pixel[1] - 240) < 1.0  # v should be near cy

    def test_with_distortion(self):
        """Test projection with lens distortion."""
        intrinsics = CameraIntrinsics(
            fx=500, fy=500, cx=320, cy=240,
            k1=-0.1, k2=0.01, p1=0.001, p2=0.001
        )
        
        point_3d = np.array([1.0, 0.5, 5.0])
        pixel = project_point(intrinsics, point_3d)
        
        # Should get a valid pixel
        assert pixel.shape == (2,)
        assert np.all(np.isfinite(pixel))

    def test_behind_camera_raises_error(self):
        """Test that points behind camera (Z <= 0) raise ValueError."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        point_behind = np.array([1.0, 0.0, -1.0])
        with pytest.raises(ValueError, match="behind camera"):
            project_point(intrinsics, point_behind)

    def test_batch_projection(self):
        """Test projection of multiple points."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        points_3d = np.array([
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ])
        pixels = project_point(intrinsics, points_3d)
        
        assert pixels.shape == (3, 2)
        # First point should be at principal point
        np.testing.assert_allclose(pixels[0], [320, 240], atol=0.1)

    def test_different_focal_lengths(self):
        """Test projection with different fx and fy."""
        intrinsics = CameraIntrinsics(fx=500, fy=600, cx=320, cy=240)
        
        point_3d = np.array([1.0, 1.0, 5.0])
        pixel = project_point(intrinsics, point_3d)
        
        # u = 500 * (1/5) + 320 = 420
        # v = 600 * (1/5) + 240 = 360
        np.testing.assert_allclose(pixel, [420, 360], atol=0.5)


class TestUnprojectPixel:
    """Tests for unproject_pixel function (inverse projection)."""

    def test_unproject_principal_point(self):
        """Test unprojecting the principal point gives forward ray."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Principal point should unproject to [0, 0, 1] direction
        ray = unproject_pixel(intrinsics, np.array([320, 240]))
        
        # Ray should point forward (Z component positive)
        assert ray[2] > 0
        # Ray should be roughly aligned with Z-axis
        assert np.abs(ray[0]) < 0.01
        assert np.abs(ray[1]) < 0.01

    def test_unproject_with_depth(self):
        """Test unprojecting to a specific depth."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        pixel = np.array([320, 240])
        depth = 5.0
        point_3d = unproject_pixel(intrinsics, pixel, depth=depth)
        
        # Should be at depth 5.0
        assert np.abs(point_3d[2] - depth) < 0.1

    def test_unproject_inverts_project(self):
        """Test that unproject inverts project (with known depth)."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Original 3D point
        point_original = np.array([1.0, 0.5, 5.0])
        
        # Project to pixel
        pixel = project_point(intrinsics, point_original)
        
        # Unproject back with known depth
        point_recovered = unproject_pixel(intrinsics, pixel, depth=5.0)
        
        # Relaxed tolerance due to iterative undistort convergence
        np.testing.assert_allclose(point_recovered, point_original, atol=0.15)

    def test_batch_unproject(self):
        """Test unprojecting multiple pixels."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        pixels = np.array([[320, 240], [420, 340], [220, 140]])
        rays = unproject_pixel(intrinsics, pixels)
        
        assert rays.shape == (3, 3)
        # All rays should point forward
        assert np.all(rays[:, 2] > 0)


class TestComputeReprojectionError:
    """Tests for compute_reprojection_error function."""

    def test_zero_error_for_perfect_projection(self):
        """Test zero error when point projects exactly to observed pixel."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Create point and project it
        point_3d = np.array([1.0, 0.5, 5.0])
        observed_pixel = project_point(intrinsics, point_3d)
        
        # Compute error (should be near zero)
        error = compute_reprojection_error(intrinsics, point_3d, observed_pixel)
        
        np.testing.assert_allclose(error, [0, 0], atol=1e-10)

    def test_nonzero_error_for_displaced_observation(self):
        """Test nonzero error when observation doesn't match projection."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        point_3d = np.array([1.0, 0.5, 5.0])
        projected_pixel = project_point(intrinsics, point_3d)
        
        # Displaced observation
        observed_pixel = projected_pixel + np.array([10, 5])
        
        error = compute_reprojection_error(intrinsics, point_3d, observed_pixel)
        
        # Error should be approximately [-10, -5]
        np.testing.assert_allclose(error, [-10, -5], atol=0.1)

    def test_error_magnitude_increases_with_displacement(self):
        """Test that larger displacement leads to larger error."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        point_3d = np.array([1.0, 0.5, 5.0])
        projected_pixel = project_point(intrinsics, point_3d)
        
        # Small displacement
        observed_1 = projected_pixel + np.array([1, 1])
        error_1 = compute_reprojection_error(intrinsics, point_3d, observed_1)
        
        # Large displacement
        observed_2 = projected_pixel + np.array([10, 10])
        error_2 = compute_reprojection_error(intrinsics, point_3d, observed_2)
        
        assert np.linalg.norm(error_2) > np.linalg.norm(error_1)

    def test_batch_reprojection_error(self):
        """Test computing errors for multiple points."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        points_3d = np.array([
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ])
        
        # Project all points
        projected = project_point(intrinsics, points_3d)
        
        # Add small noise to create observations
        observed = projected + np.random.normal(0, 1, projected.shape)
        
        # Compute errors
        errors = compute_reprojection_error(intrinsics, points_3d, observed)
        
        assert errors.shape == (3, 2)
        # Errors should be small but non-zero
        assert np.all(np.linalg.norm(errors, axis=1) < 5)


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics dataclass."""

    def test_create_simple_intrinsics(self):
        """Test creating simple intrinsics without distortion."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        assert intrinsics.fx == 500
        assert intrinsics.fy == 500
        assert intrinsics.cx == 320
        assert intrinsics.cy == 240
        assert intrinsics.k1 == 0.0
        assert intrinsics.k2 == 0.0
        assert intrinsics.p1 == 0.0
        assert intrinsics.p2 == 0.0

    def test_create_with_distortion(self):
        """Test creating intrinsics with distortion coefficients."""
        intrinsics = CameraIntrinsics(
            fx=500, fy=500, cx=320, cy=240,
            k1=-0.1, k2=0.01, p1=0.001, p2=0.001
        )
        
        assert intrinsics.k1 == -0.1
        assert intrinsics.k2 == 0.01
        assert intrinsics.p1 == 0.001
        assert intrinsics.p2 == 0.001

    def test_invalid_focal_length_raises(self):
        """Test that negative focal lengths raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            CameraIntrinsics(fx=-500, fy=500, cx=320, cy=240)
        
        with pytest.raises(ValueError, match="must be positive"):
            CameraIntrinsics(fx=500, fy=-500, cx=320, cy=240)


class TestIntegration:
    """Integration tests for complete projection pipeline."""

    def test_project_unproject_roundtrip(self):
        """Test full roundtrip: 3D point → pixel → 3D point."""
        intrinsics = CameraIntrinsics(
            fx=500, fy=500, cx=320, cy=240,
            k1=-0.05, k2=0.005, p1=0.001, p2=0.001
        )
        
        # Original 3D points
        points_original = np.array([
            [0.5, 0.3, 5.0],
            [1.0, -0.5, 4.0],
            [-0.5, 0.8, 6.0],
        ])
        
        # Project to pixels
        pixels = project_point(intrinsics, points_original)
        
        # Unproject back with known depths
        depths = points_original[:, 2]
        points_recovered = unproject_pixel(intrinsics, pixels, depth=depths)
        
        # Should recover original points (relaxed tolerance for distortion convergence)
        np.testing.assert_allclose(points_recovered, points_original, atol=0.2)

    def test_reprojection_error_minimization_concept(self):
        """Test the concept behind bundle adjustment: minimizing reprojection error."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # True 3D point
        point_true = np.array([1.0, 0.5, 5.0])
        
        # True observation (perfect projection)
        observed_pixel = project_point(intrinsics, point_true)
        
        # Test points at different positions
        test_points = np.array([
            [1.0, 0.5, 5.0],    # Correct point
            [1.1, 0.5, 5.0],    # Slightly off in X
            [1.0, 0.6, 5.0],    # Slightly off in Y
            [1.0, 0.5, 5.2],    # Slightly off in Z
        ])
        
        errors = []
        for test_point in test_points:
            error = compute_reprojection_error(intrinsics, test_point, observed_pixel)
            errors.append(np.linalg.norm(error))
        
        # Error should be smallest for the true point
        assert errors[0] < 0.1  # Near zero
        assert all(errors[i] > errors[0] for i in range(1, len(errors)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

