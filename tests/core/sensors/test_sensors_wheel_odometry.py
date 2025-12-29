"""
Unit tests for core/sensors/wheel_odometry.py (wheel dead reckoning).

Tests cover:
    - Skew-symmetric matrix (Eq. 6.12)
    - Lever arm compensation (Eq. 6.11)
    - Attitude to map velocity transform (Eq. 6.14)
    - Position update (Eq. 6.15)
    - Complete wheel odometry update loop
    - Edge cases and validation

Run with: pytest tests/test_sensors_wheel_odometry.py -v
"""

import unittest
import numpy as np
import pytest

from core.sensors.wheel_odometry import (
    skew,
    wheel_speed_to_attitude_velocity,
    attitude_to_map_velocity,
    odom_pos_update,
    wheel_odom_update,
)


class TestSkew(unittest.TestCase):
    """Test suite for skew-symmetric matrix (Eq. 6.12)."""

    def test_skew_zeros(self) -> None:
        """Test skew matrix for zero vector."""
        v = np.zeros(3)
        S = skew(v)

        np.testing.assert_array_equal(S, np.zeros((3, 3)))

    def test_skew_skew_symmetric(self) -> None:
        """Test that skew matrix is skew-symmetric (S^T = -S)."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)

        np.testing.assert_array_almost_equal(S.T, -S)

    def test_skew_trace_zero(self) -> None:
        """Test that trace(S) = 0."""
        v = np.array([5.0, -3.0, 7.0])
        S = skew(v)

        assert np.isclose(np.trace(S), 0.0)

    def test_skew_cross_product(self) -> None:
        """Test that skew matrix correctly computes cross product."""
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, -1.0, 2.0])

        # Cross product via skew matrix
        S = skew(v)
        cross_skew = S @ w

        # Cross product via NumPy
        cross_numpy = np.cross(v, w)

        np.testing.assert_array_almost_equal(cross_skew, cross_numpy)

    def test_skew_structure(self) -> None:
        """Test skew matrix has correct structure (Eq. 6.12)."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)

        # Expected structure from Eq. (6.12):
        # [  0   -vz    vy ]
        # [ vz    0   -vx ]
        # [-vy   vx    0  ]
        expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

        np.testing.assert_array_almost_equal(S, expected)

    def test_skew_invalid_shape(self) -> None:
        """Test that invalid v shape raises error."""
        v_bad = np.array([1.0, 2.0])  # Wrong: should be (3,)

        with pytest.raises(ValueError, match="v must have shape"):
            skew(v_bad)


class TestWheelSpeedToAttitudeVelocity(unittest.TestCase):
    """Test suite for lever arm compensation (Eq. 6.11)."""

    def test_lever_arm_zero(self) -> None:
        """Test that zero lever arm gives v^A = C_S^A @ v^S (rotation only)."""
        v_s = np.array([0.0, 5.0, 0.0])  # Book convention: y=forward
        omega = np.array([0.0, 0.0, 0.5])
        lever_arm = np.zeros(3)

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # With zero lever arm and default C_S^A=I: v^A = v^S
        np.testing.assert_array_almost_equal(v_a, v_s)

    def test_lever_arm_no_rotation(self) -> None:
        """Test that zero rotation gives v^A = C_S^A @ v^S (no lever arm effect)."""
        v_s = np.array([0.0, 5.0, 0.0])  # Book convention: y=forward
        omega = np.zeros(3)
        lever_arm = np.array([0.0, 0.5, 0.0])  # 0.5m forward (y-axis)

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # With zero rotation and default C_S^A=I: v^A = v^S
        np.testing.assert_array_almost_equal(v_a, v_s)

    def test_lever_arm_forward_yaw(self) -> None:
        """Test lever arm correction with forward offset and yaw rotation (book convention)."""
        v_s = np.array([0.0, 5.0, 0.0])  # 5 m/s forward (book: y=forward)
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw (turning left)
        lever_arm = np.array([0.0, 0.5, 0.0])  # wheel 0.5m forward of IMU (y-axis)

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # Expected: v^A = C_S^A @ v^S - [ω^A×] @ l^A
        # With C_S^A = I: v^A = v^S - [ω×] l
        # [ω×] l = skew([0, 0, 1]) @ [0, 0.5, 0] = [-0.5, 0, 0]
        # So v^A = [0, 5, 0] - [-0.5, 0, 0] = [0.5, 5, 0]
        expected = np.array([0.5, 5.0, 0.0])
        np.testing.assert_array_almost_equal(v_a, expected)

    def test_lever_arm_lateral_offset(self) -> None:
        """Test lever arm correction with lateral offset (book convention)."""
        v_s = np.array([0.0, 3.0, 0.0])  # 3 m/s forward (book: y=forward)
        omega = np.array([0.0, 0.0, 0.5])  # yaw rate
        lever_arm = np.array([1.0, 0.0, 0.0])  # wheel 1m to the right (x-axis)

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # [ω×] l = skew([0, 0, 0.5]) @ [1, 0, 0] = [0, 0.5, 0]
        # v^A = [0, 3, 0] - [0, 0.5, 0] = [0, 2.5, 0]
        expected = np.array([0.0, 2.5, 0.0])
        np.testing.assert_array_almost_equal(v_a, expected)

    def test_lever_arm_equation_6_11_aligned(self) -> None:
        """Test Eq. (6.11) with aligned frames: v^A = v^S - [ω×] l."""
        v_s = np.array([0.5, 2.0, 0.5])  # General velocity
        omega = np.array([0.1, -0.2, 0.3])
        lever_arm = np.array([0.3, 0.4, -0.1])

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # Manual calculation with C_S^A = I (default)
        omega_skew = skew(omega)
        expected = v_s - omega_skew @ lever_arm

        np.testing.assert_array_almost_equal(v_a, expected)

    def test_lever_arm_equation_6_11_misaligned(self) -> None:
        """Test Eq. (6.11) with misaligned frames: v^A = C_S^A @ v^S - [ω×] l."""
        v_s = np.array([0.0, 5.0, 0.0])  # 5 m/s forward in S-frame
        omega = np.array([0.0, 0.0, 0.5])  # yaw rate
        lever_arm = np.array([0.0, 0.5, 0.0])  # forward offset
        
        # Rotation matrix: 90° about z (S-frame rotated 90° relative to A-frame)
        angle = np.pi / 2
        C_S_A = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm, C_S_A)

        # Manual calculation: v^A = C_S^A @ v^S - [ω×] @ l
        rotated_v = C_S_A @ v_s  # [0, 5, 0] → [-5, 0, 0] after 90° rotation
        omega_skew = skew(omega)
        expected = rotated_v - omega_skew @ lever_arm  # [-5, 0, 0] - [-0.25, 0, 0]

        np.testing.assert_array_almost_equal(v_a, expected, decimal=5)

    def test_misaligned_frames_identity_rotation(self) -> None:
        """Test that explicit identity C_S^A gives same result as default."""
        v_s = np.array([0.0, 3.0, 0.0])
        omega = np.array([0.0, 0.0, 0.2])
        lever_arm = np.array([0.0, 0.3, 0.0])

        # Default (C_S^A = I implicitly)
        v_a_default = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # Explicit identity
        C_S_A_identity = np.eye(3)
        v_a_explicit = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm, C_S_A_identity)

        np.testing.assert_array_almost_equal(v_a_default, v_a_explicit)

    def test_misaligned_frames_180deg_rotation(self) -> None:
        """Test with 180° rotation between S and A frames."""
        v_s = np.array([0.0, 5.0, 0.0])  # Forward in S-frame
        omega = np.zeros(3)
        lever_arm = np.zeros(3)
        
        # 180° rotation about z-axis
        C_S_A = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm, C_S_A)

        # After 180° rotation: [0, 5, 0] → [0, -5, 0]
        expected = np.array([0.0, -5.0, 0.0])
        np.testing.assert_array_almost_equal(v_a, expected)


class TestAttitudeToMapVelocity(unittest.TestCase):
    """Test suite for attitude to map velocity transform (Eq. 6.14)."""

    def test_attitude_to_map_identity_quaternion(self) -> None:
        """Test that identity quaternion gives v^M = v^A (no rotation)."""
        v_a = np.array([0.0, 5.0, 0.0])  # Book convention: y=forward
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])

        v_m = attitude_to_map_velocity(v_a, q_identity)

        np.testing.assert_array_almost_equal(v_m, v_a)

    def test_attitude_to_map_90deg_yaw(self) -> None:
        """Test 90° yaw rotation (heading changes direction in map)."""
        v_a = np.array([0.0, 5.0, 0.0])  # forward in attitude frame (y-axis)
        # 90° yaw: q = [cos(45°), 0, 0, sin(45°)]
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])

        v_m = attitude_to_map_velocity(v_a, q)

        # After 90° yaw, y-axis in attitude frame maps to different direction in map
        # The exact result depends on quaternion convention
        expected = np.array([-5.0, 0.0, 0.0])  # y→-x for this rotation
        np.testing.assert_array_almost_equal(v_m, expected, decimal=5)

    def test_attitude_to_map_equation_6_14(self) -> None:
        """Test Eq. (6.14): v^M = C_A^M @ v^A."""
        v_a = np.array([3.0, 1.0, -0.5])
        q = np.array([0.9239, 0.3827, 0.0, 0.0])  # arbitrary rotation

        v_m = attitude_to_map_velocity(v_a, q)

        # Manual calculation using quat_to_rotmat
        from core.sensors.strapdown import quat_to_rotmat

        C_A_M = quat_to_rotmat(q)
        expected = C_A_M @ v_a

        np.testing.assert_array_almost_equal(v_m, expected)


class TestOdomPosUpdate(unittest.TestCase):
    """Test suite for position update (Eq. 6.15)."""

    def test_odom_pos_update_zero_velocity(self) -> None:
        """Test that position doesn't change with zero velocity."""
        p0 = np.array([10.0, 20.0, 5.0])
        v = np.zeros(3)
        dt = 0.1

        p1 = odom_pos_update(p0, v, dt)

        np.testing.assert_array_equal(p1, p0)

    def test_odom_pos_update_constant_velocity(self) -> None:
        """Test position update with constant velocity (Eq. 6.15)."""
        p0 = np.zeros(3)
        v = np.array([2.0, 0.0, 0.0])  # 2 m/s eastward
        dt = 0.1

        p1 = odom_pos_update(p0, v, dt)

        expected = np.array([0.2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(p1, expected)

    def test_odom_pos_update_multiple_steps(self) -> None:
        """Test position integration over multiple steps."""
        p = np.zeros(3)
        v = np.array([1.0, 0.5, 0.0])
        dt = 0.01
        n_steps = 100  # 1 second

        for _ in range(n_steps):
            p = odom_pos_update(p, v, dt)

        # After 1 s: p = v * 1.0
        expected = v * 1.0
        np.testing.assert_array_almost_equal(p, expected, decimal=5)

    def test_odom_pos_update_invalid_dt(self) -> None:
        """Test that invalid dt raises error."""
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="dt must be positive"):
            odom_pos_update(p, v, dt=0.0)


class TestWheelOdomUpdate(unittest.TestCase):
    """Test suite for complete wheel odometry update loop."""

    def test_wheel_odom_update_zero_lever_arm(self) -> None:
        """Test complete update with zero lever arm (book convention)."""
        p0 = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        v_s = np.array([0.0, 5.0, 0.0])  # 5 m/s forward (book: y=forward)
        omega = np.zeros(3)
        lever_arm = np.zeros(3)
        dt = 0.1

        p1 = wheel_odom_update(p0, q, v_s, omega, lever_arm, dt)

        # Should move 0.5 m forward (in y direction with identity q)
        expected = np.array([0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(p1, expected)

    def test_wheel_odom_update_with_lever_arm(self) -> None:
        """Test complete update with lever arm correction (book convention)."""
        p0 = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_s = np.array([0.0, 5.0, 0.0])  # Book: y=forward
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw
        lever_arm = np.array([0.0, 0.5, 0.0])  # 0.5m forward (y-axis)
        dt = 0.1

        p1 = wheel_odom_update(p0, q, v_s, omega, lever_arm, dt)

        # v^A = v^S - [ω×] l = [0,5,0] - [-0.5,0,0] = [0.5,5,0]
        # v^M = C @ v^A = v^A (identity q)
        # p1 = p0 + v^M * dt = [0.05, 0.5, 0]
        expected = np.array([0.05, 0.5, 0.0])
        np.testing.assert_array_almost_equal(p1, expected)

    def test_wheel_odom_update_with_rotation(self) -> None:
        """Test complete update with vehicle rotation (book convention)."""
        p0 = np.zeros(3)
        # 45° yaw: heading northeast
        q = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        v_s = np.array([0.0, 4.0, 0.0])  # 4 m/s forward (book: y=forward)
        omega = np.zeros(3)
        lever_arm = np.zeros(3)
        dt = 0.1

        p1 = wheel_odom_update(p0, q, v_s, omega, lever_arm, dt)

        # v^A = v^S = [0, 4, 0]
        # v^M = C(45°) @ [0, 4, 0]
        # After rotation, y-axis velocity maps to the map frame
        # The result depends on the quaternion rotation
        # For 45° yaw, [0,4,0] → approx [-2.83, 2.83, 0]
        expected_x = -4 * np.sin(np.pi / 4) * 0.1
        expected_y = 4 * np.cos(np.pi / 4) * 0.1
        expected = np.array([expected_x, expected_y, 0.0])

        np.testing.assert_array_almost_equal(p1, expected, decimal=5)

    def test_wheel_odom_update_multiple_steps(self) -> None:
        """Test wheel odometry integration over multiple steps (book convention)."""
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_s = np.array([0.0, 2.0, 0.0])  # constant 2 m/s (book: y=forward)
        omega = np.zeros(3)
        lever_arm = np.zeros(3)
        dt = 0.01
        n_steps = 100  # 1 second

        for _ in range(n_steps):
            p = wheel_odom_update(p, q, v_s, omega, lever_arm, dt)

        # After 1 s: should move 2 m forward (in y direction)
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(p, expected, decimal=5)

    def test_wheel_odom_update_turning_vehicle(self) -> None:
        """Test wheel odometry with turning vehicle (yaw rate, book convention)."""
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_s = np.array([0.0, 5.0, 0.0])  # 5 m/s forward (book: y=forward)
        omega = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s yaw (turning left)
        lever_arm = np.array([0.0, 1.0, 0.0])  # wheel 1m forward (y-axis)
        dt = 0.1

        p1 = wheel_odom_update(p, q, v_s, omega, lever_arm, dt)

        # With lever arm and rotation, position should be affected
        # v^A = [0, 5, 0] - [-0.5, 0, 0] = [0.5, 5, 0]
        # p1 = [0.05, 0.5, 0]
        assert p1[0] > 0  # slight rightward motion due to lever arm
        assert p1[1] > 0  # moved forward


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for wheel odometry."""

    def test_skew_commutativity(self) -> None:
        """Test that skew(v) @ w = -skew(w) @ v (anti-commutativity)."""
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, -1.0, 2.0])

        result1 = skew(v) @ w
        result2 = -skew(w) @ v

        np.testing.assert_array_almost_equal(result1, result2)

    def test_large_lever_arm(self) -> None:
        """Test with large lever arm (stress test, book convention)."""
        v_s = np.array([0.0, 10.0, 0.0])  # Book: y=forward
        omega = np.array([0.0, 0.0, 2.0])  # high yaw rate
        lever_arm = np.array([0.0, 5.0, 0.0])  # large offset (y-axis)

        v_a = wheel_speed_to_attitude_velocity(v_s, omega, lever_arm)

        # Should handle large values correctly
        assert not np.any(np.isnan(v_a))
        assert not np.any(np.isinf(v_a))

    def test_negative_velocity(self) -> None:
        """Test with negative (backward) velocity (book convention)."""
        p0 = np.array([10.0, 5.0, 2.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_s = np.array([0.0, -3.0, 0.0])  # backing up (book: y=forward)
        omega = np.zeros(3)
        lever_arm = np.zeros(3)
        dt = 0.1

        p1 = wheel_odom_update(p0, q, v_s, omega, lever_arm, dt)

        # Should move backward (negative y)
        assert p1[1] < p0[1]


if __name__ == "__main__":
    unittest.main()

