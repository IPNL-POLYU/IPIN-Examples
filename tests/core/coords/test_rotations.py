"""Unit tests for rotation representations and conversions.

This module tests the conversion functions between different rotation
representations: rotation matrices, quaternions, and Euler angles.

Test cases include:
- Round-trip conversions (e.g., Euler -> quaternion -> Euler)
- Identity rotations
- Known rotation values (90°, 180°, etc.)
- Orthogonality and normalization checks
- Gimbal lock handling

Reference: Chapter 2, Section 2.4 - Rotation Representations
"""

import unittest

import numpy as np

from core.coords.rotations import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat,
)


class TestEulerToRotationMatrix(unittest.TestCase):
    """Test cases for Euler angles to rotation matrix conversion."""

    def test_identity_rotation(self) -> None:
        """Test identity rotation (zero Euler angles)."""
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)

        expected = np.eye(3)
        np.testing.assert_allclose(R, expected, atol=1e-9)

    def test_rotation_matrix_properties(self) -> None:
        """Test that rotation matrix is orthogonal with det(R) = 1."""
        roll = 0.1
        pitch = 0.2
        yaw = 0.3

        R = euler_to_rotation_matrix(roll, pitch, yaw)

        # Check orthogonality: R^T @ R = I
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)

        # Check determinant: det(R) = 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=9)

    def test_90_degree_yaw(self) -> None:
        """Test 90° yaw rotation."""
        R = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)

        # 90° yaw should rotate x-axis to y-axis
        v_body = np.array([1.0, 0.0, 0.0])
        v_nav = R @ v_body

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v_nav, expected, atol=1e-9)

    def test_90_degree_pitch(self) -> None:
        """Test 90° pitch rotation."""
        R = euler_to_rotation_matrix(0.0, np.pi / 2.0, 0.0)

        # 90° pitch should rotate x-axis to -z-axis
        v_body = np.array([1.0, 0.0, 0.0])
        v_nav = R @ v_body

        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(v_nav, expected, atol=1e-9)

    def test_90_degree_roll(self) -> None:
        """Test 90° roll rotation."""
        R = euler_to_rotation_matrix(np.pi / 2.0, 0.0, 0.0)

        # 90° roll should rotate y-axis to z-axis
        v_body = np.array([0.0, 1.0, 0.0])
        v_nav = R @ v_body

        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(v_nav, expected, atol=1e-9)


class TestRotationMatrixToEuler(unittest.TestCase):
    """Test cases for rotation matrix to Euler angles conversion."""

    def test_identity_rotation(self) -> None:
        """Test identity rotation matrix."""
        R = np.eye(3)
        euler = rotation_matrix_to_euler(R)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(euler, expected, atol=1e-9)

    def test_90_degree_yaw(self) -> None:
        """Test extraction of 90° yaw."""
        R = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
        euler = rotation_matrix_to_euler(R)

        expected = np.array([0.0, 0.0, np.pi / 2.0])
        np.testing.assert_allclose(euler, expected, atol=1e-9)

    def test_gimbal_lock_positive(self) -> None:
        """Test gimbal lock at pitch = +90°."""
        R = euler_to_rotation_matrix(0.3, np.pi / 2.0, 0.5)
        euler = rotation_matrix_to_euler(R)

        # At gimbal lock, pitch should be π/2
        self.assertAlmostEqual(euler[1], np.pi / 2.0, places=9)

        # Roll is set to zero by convention
        self.assertAlmostEqual(euler[0], 0.0, places=9)

    def test_gimbal_lock_negative(self) -> None:
        """Test gimbal lock at pitch = -90°."""
        R = euler_to_rotation_matrix(0.3, -np.pi / 2.0, 0.5)
        euler = rotation_matrix_to_euler(R)

        # At gimbal lock, pitch should be -π/2
        self.assertAlmostEqual(euler[1], -np.pi / 2.0, places=9)

        # Roll is set to zero by convention
        self.assertAlmostEqual(euler[0], 0.0, places=9)

    def test_invalid_matrix_shape(self) -> None:
        """Test that invalid matrix shape raises ValueError."""
        R_invalid = np.eye(4)

        with self.assertRaises(ValueError):
            rotation_matrix_to_euler(R_invalid)


class TestRoundTripEulerRotationMatrix(unittest.TestCase):
    """Test round-trip conversions between Euler angles and rotation matrices."""

    def test_round_trip_multiple_angles(self) -> None:
        """Test Euler -> R -> Euler for multiple angle sets."""
        test_angles = [
            (0.0, 0.0, 0.0),
            (0.1, 0.2, 0.3),
            (np.pi / 4, np.pi / 6, np.pi / 3),
            (-0.5, 0.0, 0.5),
            (0.0, np.pi / 4, 0.0),
            (np.pi / 6, -np.pi / 6, np.pi / 4),
        ]

        for roll, pitch, yaw in test_angles:
            with self.subTest(roll=roll, pitch=pitch, yaw=yaw):
                # Forward: Euler -> R
                R = euler_to_rotation_matrix(roll, pitch, yaw)

                # Backward: R -> Euler
                euler_result = rotation_matrix_to_euler(R)

                # Check round-trip accuracy
                np.testing.assert_allclose(
                    euler_result, np.array([roll, pitch, yaw]), atol=1e-9
                )


class TestEulerToQuaternion(unittest.TestCase):
    """Test cases for Euler angles to quaternion conversion."""

    def test_identity_rotation(self) -> None:
        """Test identity rotation (zero Euler angles)."""
        q = euler_to_quat(0.0, 0.0, 0.0)

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_quaternion_normalization(self) -> None:
        """Test that quaternion is normalized."""
        roll = 0.5
        pitch = 0.3
        yaw = 0.7

        q = euler_to_quat(roll, pitch, yaw)

        # Check norm
        norm = np.linalg.norm(q)
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_90_degree_yaw(self) -> None:
        """Test 90° yaw rotation."""
        q = euler_to_quat(0.0, 0.0, np.pi / 2.0)

        # Expected: [cos(45°), 0, 0, sin(45°)]
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_180_degree_rotation(self) -> None:
        """Test 180° rotation about z-axis."""
        q = euler_to_quat(0.0, 0.0, np.pi)

        # Check norm
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=9)

        # qw should be 0, qz should be ±1
        self.assertAlmostEqual(q[0], 0.0, places=9)
        self.assertAlmostEqual(abs(q[3]), 1.0, places=9)


class TestQuaternionToEuler(unittest.TestCase):
    """Test cases for quaternion to Euler angles conversion."""

    def test_identity_quaternion(self) -> None:
        """Test identity quaternion."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        euler = quat_to_euler(q)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(euler, expected, atol=1e-9)

    def test_90_degree_yaw(self) -> None:
        """Test 90° yaw rotation."""
        q = euler_to_quat(0.0, 0.0, np.pi / 2.0)
        euler = quat_to_euler(q)

        expected = np.array([0.0, 0.0, np.pi / 2.0])
        np.testing.assert_allclose(euler, expected, atol=1e-9)

    def test_invalid_quaternion_shape(self) -> None:
        """Test that invalid quaternion shape raises ValueError."""
        q_invalid = np.array([1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            quat_to_euler(q_invalid)


class TestRoundTripEulerQuaternion(unittest.TestCase):
    """Test round-trip conversions between Euler angles and quaternions."""

    def test_round_trip_multiple_angles(self) -> None:
        """Test Euler -> quat -> Euler for multiple angle sets."""
        test_angles = [
            (0.0, 0.0, 0.0),
            (0.1, 0.2, 0.3),
            (np.pi / 4, np.pi / 6, np.pi / 3),
            (-0.5, 0.0, 0.5),
            (0.0, np.pi / 4, 0.0),
            (np.pi / 6, -np.pi / 6, np.pi / 4),
        ]

        for roll, pitch, yaw in test_angles:
            with self.subTest(roll=roll, pitch=pitch, yaw=yaw):
                # Forward: Euler -> quat
                q = euler_to_quat(roll, pitch, yaw)

                # Backward: quat -> Euler
                euler_result = quat_to_euler(q)

                # Check round-trip accuracy
                np.testing.assert_allclose(
                    euler_result, np.array([roll, pitch, yaw]), atol=1e-9
                )


class TestQuaternionToRotationMatrix(unittest.TestCase):
    """Test cases for quaternion to rotation matrix conversion."""

    def test_identity_quaternion(self) -> None:
        """Test identity quaternion."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotation_matrix(q)

        expected = np.eye(3)
        np.testing.assert_allclose(R, expected, atol=1e-9)

    def test_rotation_matrix_properties(self) -> None:
        """Test that rotation matrix is orthogonal with det(R) = 1."""
        q = np.array([0.9239, 0.2209, 0.1768, 0.2651])  # Random unit quaternion
        q = q / np.linalg.norm(q)  # Normalize

        R = quat_to_rotation_matrix(q)

        # Check orthogonality: R^T @ R = I
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)

        # Check determinant: det(R) = 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=9)

    def test_90_degree_yaw(self) -> None:
        """Test 90° yaw rotation."""
        q = euler_to_quat(0.0, 0.0, np.pi / 2.0)
        R = quat_to_rotation_matrix(q)

        # 90° yaw should rotate x-axis to y-axis
        v_body = np.array([1.0, 0.0, 0.0])
        v_nav = R @ v_body

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v_nav, expected, atol=1e-9)

    def test_invalid_quaternion_shape(self) -> None:
        """Test that invalid quaternion shape raises ValueError."""
        q_invalid = np.array([1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            quat_to_rotation_matrix(q_invalid)


class TestRotationMatrixToQuaternion(unittest.TestCase):
    """Test cases for rotation matrix to quaternion conversion."""

    def test_identity_rotation(self) -> None:
        """Test identity rotation matrix."""
        R = np.eye(3)
        q = rotation_matrix_to_quat(R)

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_quaternion_normalization(self) -> None:
        """Test that quaternion is normalized."""
        R = euler_to_rotation_matrix(0.5, 0.3, 0.7)
        q = rotation_matrix_to_quat(R)

        # Check norm
        norm = np.linalg.norm(q)
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_90_degree_yaw(self) -> None:
        """Test 90° yaw rotation."""
        R = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
        q = rotation_matrix_to_quat(R)

        # Expected: [cos(45°), 0, 0, sin(45°)] or negative (quaternion double cover)
        expected_magnitude = np.array(
            [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        )

        # Check that q is either +expected or -expected
        is_positive = np.allclose(q, expected_magnitude, atol=1e-9)
        is_negative = np.allclose(q, -expected_magnitude, atol=1e-9)

        self.assertTrue(is_positive or is_negative)

    def test_shepperd_branches(self) -> None:
        """Test Shepperd's method alternative branches."""
        # Branch 1: R[0,0] is largest
        R1 = euler_to_rotation_matrix(np.pi / 2.0, 0.0, 0.0)
        q1 = rotation_matrix_to_quat(R1)
        self.assertAlmostEqual(np.linalg.norm(q1), 1.0, places=9)

        # Branch 2: R[1,1] is largest
        R2 = euler_to_rotation_matrix(0.0, np.pi / 2.0, 0.0)
        q2 = rotation_matrix_to_quat(R2)
        self.assertAlmostEqual(np.linalg.norm(q2), 1.0, places=9)

        # Branch 3: R[2,2] is largest
        R3 = euler_to_rotation_matrix(0.0, 0.0, np.pi)
        q3 = rotation_matrix_to_quat(R3)
        self.assertAlmostEqual(np.linalg.norm(q3), 1.0, places=9)

    def test_invalid_matrix_shape(self) -> None:
        """Test that invalid matrix shape raises ValueError."""
        R_invalid = np.eye(4)

        with self.assertRaises(ValueError):
            rotation_matrix_to_quat(R_invalid)


class TestRoundTripQuaternionRotationMatrix(unittest.TestCase):
    """Test round-trip conversions between quaternions and rotation matrices."""

    def test_round_trip_multiple_quaternions(self) -> None:
        """Test quat -> R -> quat for multiple quaternions."""
        test_quaternions = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.9239, 0.2209, 0.1768, 0.2651]),
            np.array([0.7071, 0.7071, 0.0, 0.0]),
            np.array([0.7071, 0.0, 0.7071, 0.0]),
            np.array([0.7071, 0.0, 0.0, 0.7071]),
        ]

        for q in test_quaternions:
            q = q / np.linalg.norm(q)  # Normalize

            with self.subTest(q=q):
                # Forward: quat -> R
                R = quat_to_rotation_matrix(q)

                # Backward: R -> quat
                q_result = rotation_matrix_to_quat(R)

                # Quaternions have double cover: q and -q represent same rotation
                # Check that q_result is either +q or -q
                is_positive = np.allclose(q_result, q, atol=1e-9)
                is_negative = np.allclose(q_result, -q, atol=1e-9)

                self.assertTrue(is_positive or is_negative)


class TestCrossConversions(unittest.TestCase):
    """Test cross-conversions between all three representations."""

    def test_euler_to_quat_to_matrix_to_euler(self) -> None:
        """Test Euler -> quat -> matrix -> Euler."""
        roll = 0.3
        pitch = 0.4
        yaw = 0.5

        # Euler -> quat
        q = euler_to_quat(roll, pitch, yaw)

        # quat -> matrix
        R = quat_to_rotation_matrix(q)

        # matrix -> Euler
        euler_result = rotation_matrix_to_euler(R)

        # Check accuracy
        expected = np.array([roll, pitch, yaw])
        np.testing.assert_allclose(euler_result, expected, atol=1e-9)

    def test_matrix_to_quat_to_euler_to_matrix(self) -> None:
        """Test matrix -> quat -> Euler -> matrix."""
        R_original = euler_to_rotation_matrix(0.3, 0.4, 0.5)

        # matrix -> quat
        q = rotation_matrix_to_quat(R_original)

        # quat -> Euler
        euler = quat_to_euler(q)

        # Euler -> matrix
        R_result = euler_to_rotation_matrix(*euler)

        # Check accuracy
        np.testing.assert_allclose(R_result, R_original, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
