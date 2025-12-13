"""
Unit tests for core/sensors/imu_models.py (IMU measurement correction).

Tests cover:
    - Gyro and accel bias correction (Eqs. 6.6, 6.9)
    - Scale factor and misalignment correction (Eq. 6.59)
    - Gravity removal helper
    - Edge cases and validation

Run with: pytest tests/test_sensors_imu_models.py -v
"""

import unittest
import numpy as np
import pytest

from core.sensors.imu_models import (
    correct_gyro,
    correct_accel,
    apply_imu_scale_misalignment,
    remove_gravity_component,
)


class TestCorrectGyro(unittest.TestCase):
    """Test suite for gyro correction (Eq. 6.6)."""

    def test_correct_gyro_no_bias_no_noise(self) -> None:
        """Test gyro correction with zero bias and noise."""
        gyro_raw = np.array([0.1, 0.05, -0.02])
        bias = np.zeros(3)

        gyro_corrected = correct_gyro(gyro_raw, bias)

        np.testing.assert_array_almost_equal(gyro_corrected, gyro_raw)

    def test_correct_gyro_with_bias(self) -> None:
        """Test gyro correction removes bias (Eq. 6.6)."""
        gyro_raw = np.array([0.1, 0.05, -0.02])
        bias = np.array([0.001, -0.0005, 0.0002])

        gyro_corrected = correct_gyro(gyro_raw, bias)

        expected = gyro_raw - bias
        np.testing.assert_array_almost_equal(gyro_corrected, expected)

    def test_correct_gyro_with_bias_and_noise(self) -> None:
        """Test gyro correction with both bias and noise."""
        gyro_raw = np.array([0.1, 0.05, -0.02])
        bias = np.array([0.001, -0.0005, 0.0002])
        noise = np.array([0.0001, 0.00005, -0.00008])

        gyro_corrected = correct_gyro(gyro_raw, bias, noise)

        expected = gyro_raw - bias - noise
        np.testing.assert_array_almost_equal(gyro_corrected, expected)

    def test_correct_gyro_batch(self) -> None:
        """Test gyro correction on batch data (N, 3)."""
        n = 10
        gyro_raw = np.random.randn(n, 3) * 0.1
        bias = np.array([0.001, -0.0005, 0.0002])

        # Broadcast bias across batch
        gyro_corrected = correct_gyro(gyro_raw, bias)

        expected = gyro_raw - bias[np.newaxis, :]
        np.testing.assert_array_almost_equal(gyro_corrected, expected)

    def test_correct_gyro_zero_measurement(self) -> None:
        """Test correction when gyro reads zero (stationary case)."""
        gyro_raw = np.zeros(3)
        bias = np.array([0.001, -0.0005, 0.0002])

        gyro_corrected = correct_gyro(gyro_raw, bias)

        # Corrected should be -bias
        np.testing.assert_array_almost_equal(gyro_corrected, -bias)


class TestCorrectAccel(unittest.TestCase):
    """Test suite for accel correction (Eq. 6.9)."""

    def test_correct_accel_no_bias_no_noise(self) -> None:
        """Test accel correction with zero bias and noise."""
        accel_raw = np.array([0.05, 0.1, -9.81])
        bias = np.zeros(3)

        accel_corrected = correct_accel(accel_raw, bias)

        np.testing.assert_array_almost_equal(accel_corrected, accel_raw)

    def test_correct_accel_with_bias(self) -> None:
        """Test accel correction removes bias (Eq. 6.9)."""
        accel_raw = np.array([0.05, 0.1, -9.81])
        bias = np.array([0.01, -0.005, 0.02])

        accel_corrected = correct_accel(accel_raw, bias)

        expected = accel_raw - bias
        np.testing.assert_array_almost_equal(accel_corrected, expected)

    def test_correct_accel_with_bias_and_noise(self) -> None:
        """Test accel correction with both bias and noise."""
        accel_raw = np.array([0.05, 0.1, -9.81])
        bias = np.array([0.01, -0.005, 0.02])
        noise = np.array([0.001, 0.002, -0.003])

        accel_corrected = correct_accel(accel_raw, bias, noise)

        expected = accel_raw - bias - noise
        np.testing.assert_array_almost_equal(accel_corrected, expected)

    def test_correct_accel_stationary_sensor(self) -> None:
        """Test accel correction for stationary sensor (gravity only)."""
        # Stationary sensor in level position: measures -g in body z
        accel_raw = np.array([0.0, 0.0, -9.81])
        bias = np.array([0.01, -0.01, 0.05])  # small biases

        accel_corrected = correct_accel(accel_raw, bias)

        expected = np.array([-0.01, 0.01, -9.86])
        np.testing.assert_array_almost_equal(accel_corrected, expected, decimal=5)


class TestApplyImuScaleMisalignment(unittest.TestCase):
    """Test suite for scale and misalignment correction (Eq. 6.59)."""

    def test_scale_misalignment_identity(self) -> None:
        """Test with identity matrices (no correction)."""
        u_raw = np.array([0.1, 0.05, -0.02])
        M = np.eye(3)
        S = np.eye(3)
        b = np.zeros(3)

        u_corrected = apply_imu_scale_misalignment(u_raw, M, S, b)

        np.testing.assert_array_almost_equal(u_corrected, u_raw)

    def test_scale_misalignment_bias_only(self) -> None:
        """Test bias-only correction."""
        u_raw = np.array([0.1, 0.05, -0.02])
        M = np.eye(3)
        S = np.eye(3)
        b = np.array([0.01, -0.005, 0.002])

        u_corrected = apply_imu_scale_misalignment(u_raw, M, S, b)

        expected = u_raw - b
        np.testing.assert_array_almost_equal(u_corrected, expected)

    def test_scale_misalignment_scale_factors(self) -> None:
        """Test scale factor correction (Eq. 6.59)."""
        u_raw = np.array([1.0, 1.0, 1.0])
        M = np.eye(3)
        S = np.diag([1.02, 0.98, 1.01])  # ±2% scale errors
        b = np.zeros(3)

        u_corrected = apply_imu_scale_misalignment(u_raw, M, S, b)

        expected = np.array([1.02, 0.98, 1.01])
        np.testing.assert_array_almost_equal(u_corrected, expected)

    def test_scale_misalignment_full_correction(self) -> None:
        """Test full correction: bias + scale + misalignment."""
        u_raw = np.array([1.0, 0.5, -0.3])
        M = np.array(
            [
                [1.0, 0.01, 0.005],  # small misalignment
                [-0.01, 1.0, 0.002],
                [-0.005, -0.002, 1.0],
            ]
        )
        S = np.diag([1.02, 0.98, 1.01])
        b = np.array([0.01, -0.005, 0.002])

        u_corrected = apply_imu_scale_misalignment(u_raw, M, S, b)

        # Manual calculation: u_corrected = M @ S @ (u_raw - b)
        u_debias = u_raw - b
        u_scaled = S @ u_debias
        expected = M @ u_scaled

        np.testing.assert_array_almost_equal(u_corrected, expected)

    def test_scale_misalignment_batch(self) -> None:
        """Test batch processing (N, 3) data."""
        n = 5
        u_raw = np.random.randn(n, 3)
        M = np.eye(3)
        S = np.diag([1.02, 0.98, 1.01])
        b = np.array([0.01, -0.005, 0.002])

        u_corrected = apply_imu_scale_misalignment(u_raw, M, S, b)

        # Check each row manually
        for i in range(n):
            expected_i = M @ S @ (u_raw[i, :] - b)
            np.testing.assert_array_almost_equal(u_corrected[i, :], expected_i)

    def test_scale_misalignment_invalid_m_shape(self) -> None:
        """Test that invalid M shape raises error."""
        u_raw = np.array([1.0, 0.5, -0.3])
        M_bad = np.eye(2)  # Wrong shape
        S = np.eye(3)
        b = np.zeros(3)

        with pytest.raises(ValueError, match="M must be"):
            apply_imu_scale_misalignment(u_raw, M_bad, S, b)

    def test_scale_misalignment_invalid_s_shape(self) -> None:
        """Test that invalid S shape raises error."""
        u_raw = np.array([1.0, 0.5, -0.3])
        M = np.eye(3)
        S_bad = np.diag([1.0, 1.0])  # Wrong shape
        b = np.zeros(3)

        with pytest.raises(ValueError, match="S must be"):
            apply_imu_scale_misalignment(u_raw, M, S_bad, b)

    def test_scale_misalignment_invalid_b_shape(self) -> None:
        """Test that invalid b shape raises error."""
        u_raw = np.array([1.0, 0.5, -0.3])
        M = np.eye(3)
        S = np.eye(3)
        b_bad = np.zeros(2)  # Wrong shape

        with pytest.raises(ValueError, match="b must be"):
            apply_imu_scale_misalignment(u_raw, M, S, b_bad)


class TestRemoveGravityComponent(unittest.TestCase):
    """Test suite for gravity removal helper."""

    def test_remove_gravity_stationary_level(self) -> None:
        """Test gravity removal for stationary sensor at level orientation."""
        # Stationary, level: f_body = [0, 0, -9.81] (gravity only)
        accel_body = np.array([0.0, 0.0, -9.81])
        gravity_map = np.array([0.0, 0.0, -9.81])
        C_B_M = np.eye(3)  # body = map (level)

        a_true = remove_gravity_component(accel_body, gravity_map, C_B_M)

        # Should be zero (no motion)
        np.testing.assert_array_almost_equal(a_true, np.zeros(3), decimal=5)

    def test_remove_gravity_with_motion(self) -> None:
        """Test gravity removal with true acceleration."""
        # Sensor accelerating forward: f = a + (-g_body)
        a_true_body = np.array([1.0, 0.0, 0.0])  # 1 m/s² forward
        gravity_map = np.array([0.0, 0.0, -9.81])
        C_B_M = np.eye(3)  # level

        # Measured accel includes gravity
        accel_body = a_true_body + np.array([0.0, 0.0, -9.81])

        a_recovered = remove_gravity_component(accel_body, gravity_map, C_B_M)

        np.testing.assert_array_almost_equal(a_recovered, a_true_body, decimal=5)

    def test_remove_gravity_tilted_sensor(self) -> None:
        """Test gravity removal with tilted sensor orientation."""
        # 90° pitch (nose up): body z = map x
        C_B_M = np.array(
            [
                [0, 0, 1],  # body x = map z
                [0, 1, 0],  # body y = map y
                [-1, 0, 0],  # body z = -map x
            ]
        )
        gravity_map = np.array([0.0, 0.0, -9.81])

        # Stationary tilted sensor: measures gravity in body frame
        # g_body = C_M_B @ g_map = C_B_M.T @ [0, 0, -9.81] = [9.81, 0, 0]
        accel_body = np.array([9.81, 0.0, 0.0])

        a_true = remove_gravity_component(accel_body, gravity_map, C_B_M)

        # Should be zero (no motion, just rotated)
        np.testing.assert_array_almost_equal(a_true, np.zeros(3), decimal=4)

    def test_remove_gravity_batch(self) -> None:
        """Test gravity removal on batch data."""
        n = 10
        accel_batch = np.tile([0.0, 0.0, -9.81], (n, 1))  # all stationary
        gravity_map = np.array([0.0, 0.0, -9.81])
        C_B_M = np.eye(3)

        a_true_batch = remove_gravity_component(accel_batch, gravity_map, C_B_M)

        # All should be near zero
        np.testing.assert_array_almost_equal(a_true_batch, np.zeros((n, 3)), decimal=5)


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for IMU models."""

    def test_zero_bias_equivalence(self) -> None:
        """Test that zero bias is equivalent to no bias."""
        gyro_raw = np.array([0.1, 0.05, -0.02])
        
        result1 = correct_gyro(gyro_raw, np.zeros(3))
        result2 = correct_gyro(gyro_raw, np.zeros(3), None)

        np.testing.assert_array_equal(result1, result2)

    def test_large_bias(self) -> None:
        """Test correction with large bias (unusual but valid)."""
        gyro_raw = np.array([0.1, 0.05, -0.02])
        bias_large = np.array([1.0, 2.0, -0.5])  # unrealistically large

        gyro_corrected = correct_gyro(gyro_raw, bias_large)

        expected = gyro_raw - bias_large
        np.testing.assert_array_almost_equal(gyro_corrected, expected)

    def test_negative_measurements(self) -> None:
        """Test corrections work with negative measurements."""
        accel_raw = np.array([-1.0, -2.0, -9.81])
        bias = np.array([0.1, -0.2, 0.05])

        accel_corrected = correct_accel(accel_raw, bias)

        expected = accel_raw - bias
        np.testing.assert_array_almost_equal(accel_corrected, expected)


if __name__ == "__main__":
    unittest.main()

