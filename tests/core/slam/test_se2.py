"""Unit tests for core.slam.se2 module.

Tests SE(2) operations (Special Euclidean group in 2D) used throughout
Chapter 7 for scan matching, pose graph optimization, and trajectory
representation.

Author: Li-Ta Hsu
Date: 2024
"""

import numpy as np
import pytest

from core.slam import (
    Pose2,
    se2_apply,
    se2_compose,
    se2_from_matrix,
    se2_inverse,
    se2_relative,
    se2_to_matrix,
    wrap_angle,
)


class TestWrapAngle:
    """Test suite for wrap_angle function."""

    def test_wrap_zero(self):
        """Test that zero maps to zero."""
        result = wrap_angle(0.0)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_wrap_pi(self):
        """Test that π maps to π."""
        result = wrap_angle(np.pi)
        assert np.isclose(result, np.pi, atol=1e-10)

    def test_wrap_negative_pi(self):
        """Test that -π maps to -π."""
        result = wrap_angle(-np.pi)
        assert np.isclose(result, -np.pi, atol=1e-10)

    def test_wrap_large_positive(self):
        """Test wrapping large positive angle (3π → ±π)."""
        result = wrap_angle(3 * np.pi)
        # 3π wraps to π (or -π, they're equivalent on the circle)
        assert np.isclose(abs(result), np.pi, atol=1e-10)

    def test_wrap_large_negative(self):
        """Test wrapping large negative angle (-3π → -π)."""
        result = wrap_angle(-3 * np.pi)
        assert np.isclose(result, -np.pi, atol=1e-10)

    def test_wrap_slightly_over_pi(self):
        """Test that π + ε wraps to negative side."""
        result = wrap_angle(np.pi + 0.1)
        assert result < 0
        assert np.isclose(result, -np.pi + 0.1, atol=1e-10)

    def test_wrap_array_of_angles(self):
        """Test wrapping multiple angles."""
        angles = np.array([0, np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 3 * np.pi / 2])
        wrapped = np.array([wrap_angle(a) for a in angles])

        # All should be in [-π, π]
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped <= np.pi)


class TestSE2Compose:
    """Test suite for se2_compose function."""

    def test_compose_identity_left(self):
        """Test that identity ⊕ p = p."""
        p_id = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, 2.0, np.pi / 4])

        result = se2_compose(p_id, p)
        np.testing.assert_allclose(result, p, atol=1e-10)

    def test_compose_identity_right(self):
        """Test that p ⊕ identity = p."""
        p = np.array([1.0, 2.0, np.pi / 4])
        p_id = np.array([0.0, 0.0, 0.0])

        result = se2_compose(p, p_id)
        np.testing.assert_allclose(result, p, atol=1e-10)

    def test_compose_translation_only(self):
        """Test composing two pure translations."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 3.0, 0.0])

        result = se2_compose(p1, p2)
        expected = np.array([3.0, 3.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_compose_rotation_90deg(self):
        """Test composition with 90° rotation."""
        # Rotate 90°, then move 1m forward (in rotated frame → becomes left)
        p1 = np.array([0.0, 0.0, np.pi / 2])  # 90° rotation
        p2 = np.array([1.0, 0.0, 0.0])  # 1m forward

        result = se2_compose(p1, p2)
        expected = np.array([0.0, 1.0, np.pi / 2])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_compose_rotation_180deg(self):
        """Test composition with 180° rotation."""
        p1 = np.array([0.0, 0.0, np.pi])  # 180° rotation
        p2 = np.array([1.0, 0.0, 0.0])  # 1m forward

        result = se2_compose(p1, p2)
        # After 180° rotation, forward becomes backward
        expected = np.array([-1.0, 0.0, np.pi])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_compose_with_pose2_dataclass(self):
        """Test composition using Pose2 dataclass."""
        p1 = Pose2(x=1.0, y=2.0, yaw=0.0)
        p2 = Pose2(x=3.0, y=4.0, yaw=0.0)

        result = se2_compose(p1, p2)
        expected = np.array([4.0, 6.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_compose_angle_wrapping(self):
        """Test that composed angles are wrapped to [-π, π]."""
        p1 = np.array([0.0, 0.0, 3 * np.pi / 4])
        p2 = np.array([0.0, 0.0, 3 * np.pi / 4])

        result = se2_compose(p1, p2)
        # 3π/4 + 3π/4 = 3π/2, which wraps to -π/2
        assert np.isclose(result[2], -np.pi / 2, atol=1e-10)

    def test_compose_invalid_shape(self):
        """Test that invalid pose shapes raise ValueError."""
        p1 = np.array([1.0, 2.0])  # Missing yaw
        p2 = np.array([3.0, 4.0, 0.0])

        with pytest.raises(ValueError, match="must have shape \\(3,\\)"):
            se2_compose(p1, p2)


class TestSE2Inverse:
    """Test suite for se2_inverse function."""

    def test_inverse_identity(self):
        """Test that inverse of identity is identity."""
        p_id = np.array([0.0, 0.0, 0.0])
        p_inv = se2_inverse(p_id)
        np.testing.assert_allclose(p_inv, p_id, atol=1e-10)

    def test_inverse_translation_only(self):
        """Test inverse of pure translation."""
        p = np.array([1.0, 2.0, 0.0])
        p_inv = se2_inverse(p)
        expected = np.array([-1.0, -2.0, 0.0])
        np.testing.assert_allclose(p_inv, expected, atol=1e-10)

    def test_inverse_rotation_only(self):
        """Test inverse of pure rotation."""
        p = np.array([0.0, 0.0, np.pi / 4])
        p_inv = se2_inverse(p)
        expected = np.array([0.0, 0.0, -np.pi / 4])
        np.testing.assert_allclose(p_inv, expected, atol=1e-10)

    def test_inverse_composition_is_identity(self):
        """Test that p ⊕ p⁻¹ = identity."""
        p = np.array([1.0, 2.0, np.pi / 4])
        p_inv = se2_inverse(p)

        result = se2_compose(p, p_inv)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_inverse_composition_reverse_is_identity(self):
        """Test that p⁻¹ ⊕ p = identity."""
        p = np.array([1.0, 2.0, np.pi / 4])
        p_inv = se2_inverse(p)

        result = se2_compose(p_inv, p)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_inverse_with_pose2_dataclass(self):
        """Test inverse using Pose2 dataclass."""
        p = Pose2(x=1.0, y=2.0, yaw=np.pi / 3)
        p_inv = se2_inverse(p)

        # Verify p ⊕ p⁻¹ = identity
        result = se2_compose(p.to_array(), p_inv)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_double_inverse(self):
        """Test that (p⁻¹)⁻¹ = p."""
        p = np.array([1.0, 2.0, np.pi / 4])
        p_inv_inv = se2_inverse(se2_inverse(p))
        np.testing.assert_allclose(p_inv_inv, p, atol=1e-10)


class TestSE2Apply:
    """Test suite for se2_apply function."""

    def test_apply_identity(self):
        """Test that identity transformation leaves points unchanged."""
        p_id = np.array([0.0, 0.0, 0.0])
        points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        result = se2_apply(p_id, points)
        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_apply_translation_only(self):
        """Test applying pure translation."""
        p = np.array([10.0, 5.0, 0.0])
        points = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = se2_apply(p, points)
        expected = np.array([[11.0, 5.0], [10.0, 6.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_apply_rotation_90deg(self):
        """Test applying 90° rotation."""
        p = np.array([0.0, 0.0, np.pi / 2])  # 90° rotation
        points = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = se2_apply(p, points)
        # [1,0] rotates to [0,1], [0,1] rotates to [-1,0]
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_apply_rotation_180deg(self):
        """Test applying 180° rotation."""
        p = np.array([0.0, 0.0, np.pi])  # 180° rotation
        points = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = se2_apply(p, points)
        # [1,0] → [-1,0], [0,1] → [0,-1]
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_apply_rotation_and_translation(self):
        """Test applying rotation + translation."""
        p = np.array([1.0, 2.0, np.pi / 2])  # 90° rotation + translation
        points = np.array([[1.0, 0.0]])

        result = se2_apply(p, points)
        # Point [1,0] rotates to [0,1], then translates by [1,2]
        expected = np.array([[1.0, 3.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_apply_with_pose2_dataclass(self):
        """Test applying transformation using Pose2 dataclass."""
        p = Pose2(x=5.0, y=3.0, yaw=0.0)
        points = np.array([[1.0, 1.0]])

        result = se2_apply(p, points)
        expected = np.array([[6.0, 4.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_apply_empty_points(self):
        """Test applying transformation to empty point cloud."""
        p = np.array([1.0, 2.0, np.pi / 4])
        points = np.empty((0, 2))

        result = se2_apply(p, points)
        assert result.shape == (0, 2)

    def test_apply_invalid_points_shape(self):
        """Test that invalid point shape raises ValueError."""
        p = np.array([1.0, 2.0, 0.0])
        points = np.array([1.0, 2.0])  # 1D array, should be 2D

        with pytest.raises(ValueError, match="must have shape \\(N, 2\\)"):
            se2_apply(p, points)


class TestSE2Relative:
    """Test suite for se2_relative function."""

    def test_relative_same_pose(self):
        """Test that relative pose from p to p is identity."""
        p = np.array([1.0, 2.0, np.pi / 4])
        rel = se2_relative(p, p)
        np.testing.assert_allclose(rel, [0.0, 0.0, 0.0], atol=1e-10)

    def test_relative_translation_only(self):
        """Test relative pose with pure translation."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([5.0, 3.0, 0.0])

        rel = se2_relative(p1, p2)
        expected = np.array([5.0, 3.0, 0.0])
        np.testing.assert_allclose(rel, expected, atol=1e-10)

    def test_relative_with_rotation(self):
        """Test relative pose with rotation."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, np.pi / 2])

        rel = se2_relative(p1, p2)
        expected = np.array([1.0, 1.0, np.pi / 2])
        np.testing.assert_allclose(rel, expected, atol=1e-10)

    def test_relative_compose_recovers_target(self):
        """Test that p_from ⊕ rel = p_to."""
        p1 = np.array([1.0, 2.0, np.pi / 6])
        p2 = np.array([3.0, 4.0, np.pi / 3])

        rel = se2_relative(p1, p2)
        p2_recovered = se2_compose(p1, rel)

        np.testing.assert_allclose(p2_recovered, p2, atol=1e-10)


class TestSE2MatrixConversion:
    """Test suite for se2_to_matrix and se2_from_matrix functions."""

    def test_to_matrix_identity(self):
        """Test identity pose to matrix."""
        p = np.array([0.0, 0.0, 0.0])
        T = se2_to_matrix(p)

        expected = np.eye(3)
        np.testing.assert_allclose(T, expected, atol=1e-10)

    def test_to_matrix_translation_only(self):
        """Test pure translation to matrix."""
        p = np.array([1.0, 2.0, 0.0])
        T = se2_to_matrix(p)

        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(T, expected, atol=1e-10)

    def test_to_matrix_rotation_only(self):
        """Test pure rotation to matrix."""
        p = np.array([0.0, 0.0, np.pi / 2])
        T = se2_to_matrix(p)

        # cos(π/2)=0, sin(π/2)=1
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(T, expected, atol=1e-10)

    def test_from_matrix_identity(self):
        """Test identity matrix to pose."""
        T = np.eye(3)
        p = se2_from_matrix(T)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_roundtrip_conversion(self):
        """Test that to_matrix ↔ from_matrix is identity."""
        p_original = np.array([1.0, 2.0, np.pi / 4])

        T = se2_to_matrix(p_original)
        p_recovered = se2_from_matrix(T)

        np.testing.assert_allclose(p_recovered, p_original, atol=1e-10)

    def test_matrix_last_row_is_identity(self):
        """Test that transformation matrix has [0, 0, 1] as last row."""
        p = np.array([1.5, 2.5, np.pi / 3])
        T = se2_to_matrix(p)

        np.testing.assert_allclose(T[2, :], [0.0, 0.0, 1.0], atol=1e-10)


class TestPose2Dataclass:
    """Test suite for Pose2 dataclass."""

    def test_creation_valid(self):
        """Test creating a valid Pose2."""
        p = Pose2(x=1.0, y=2.0, yaw=np.pi / 4)

        assert p.x == 1.0
        assert p.y == 2.0
        assert p.yaw == np.pi / 4

    def test_to_array(self):
        """Test converting Pose2 to array."""
        p = Pose2(x=1.0, y=2.0, yaw=np.pi / 4)
        arr = p.to_array()

        expected = np.array([1.0, 2.0, np.pi / 4])
        np.testing.assert_allclose(arr, expected)
        assert arr.shape == (3,)

    def test_from_array(self):
        """Test creating Pose2 from array."""
        arr = np.array([1.0, 2.0, np.pi / 4])
        p = Pose2.from_array(arr)

        assert p.x == 1.0
        assert p.y == 2.0
        assert np.isclose(p.yaw, np.pi / 4)

    def test_identity(self):
        """Test creating identity pose."""
        p = Pose2.identity()

        assert p.x == 0.0
        assert p.y == 0.0
        assert p.yaw == 0.0

    def test_validation_nan_x(self):
        """Test that NaN x raises ValueError."""
        with pytest.raises(ValueError, match="x must be finite"):
            Pose2(x=np.nan, y=0.0, yaw=0.0)

    def test_validation_inf_y(self):
        """Test that infinite y raises ValueError."""
        with pytest.raises(ValueError, match="y must be finite"):
            Pose2(x=0.0, y=np.inf, yaw=0.0)

    def test_validation_nan_yaw(self):
        """Test that NaN yaw raises ValueError."""
        with pytest.raises(ValueError, match="yaw must be finite"):
            Pose2(x=0.0, y=0.0, yaw=np.nan)

    def test_from_array_wrong_shape(self):
        """Test that wrong array shape raises ValueError."""
        arr = np.array([1.0, 2.0])  # Missing yaw

        with pytest.raises(ValueError, match="must have shape \\(3,\\)"):
            Pose2.from_array(arr)

    def test_repr(self):
        """Test string representation."""
        p = Pose2(x=1.2345, y=2.3456, yaw=np.pi / 4)
        repr_str = repr(p)

        assert "Pose2" in repr_str
        assert "1.2345" in repr_str
        assert "2.3456" in repr_str

