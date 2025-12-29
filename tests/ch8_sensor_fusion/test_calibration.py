"""Unit tests for calibration functions (Section 8.4).

Tests IMU intrinsic calibration and extrinsic calibration methods.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest

import numpy as np

from ch8_sensor_fusion.calibration_demo import (
    estimate_imu_bias_stationary,
    calibrate_extrinsic_2d_least_squares,
    generate_synthetic_imu_stationary,
    generate_synthetic_extrinsic_data,
)


class TestIMUBiasEstimation(unittest.TestCase):
    """Test IMU intrinsic calibration (bias estimation)."""
    
    def test_perfect_stationary_data(self):
        """Test bias estimation with perfect (noiseless) data."""
        # Create perfect stationary data
        n_samples = 100
        true_accel_bias = np.array([0.05, -0.03, 0.02])
        true_gyro_bias = np.array([0.01, -0.005, 0.008])
        
        # Accelerometer reads gravity + bias (no noise)
        gravity = np.array([0, 0, -9.81])
        accel_samples = np.tile(gravity + true_accel_bias, (n_samples, 1))
        
        # Gyroscope reads bias (no noise)
        gyro_samples = np.tile(true_gyro_bias, (n_samples, 1))
        
        calibration = estimate_imu_bias_stationary(accel_samples, gyro_samples)
        
        # Should recover exact biases
        np.testing.assert_allclose(
            calibration['accel_bias'], true_accel_bias, atol=1e-10
        )
        np.testing.assert_allclose(
            calibration['gyro_bias'], true_gyro_bias, atol=1e-10
        )
        
        # Gravity axis should be Z (index 2)
        self.assertEqual(calibration['gravity_axis'], 2)
    
    def test_noisy_stationary_data(self):
        """Test bias estimation with noisy data."""
        np.random.seed(42)
        n_samples = 1000
        true_accel_bias = np.array([0.05, -0.03, 0.02])
        true_gyro_bias = np.array([0.01, -0.005, 0.008])
        
        # Add noise
        gravity = np.array([0, 0, -9.81])
        accel_samples = (
            np.tile(gravity + true_accel_bias, (n_samples, 1)) +
            np.random.randn(n_samples, 3) * 0.01
        )
        gyro_samples = (
            np.tile(true_gyro_bias, (n_samples, 1)) +
            np.random.randn(n_samples, 3) * 0.001
        )
        
        calibration = estimate_imu_bias_stationary(accel_samples, gyro_samples)
        
        # Should recover biases within noise tolerance
        # With 1000 samples, std_error = 0.01 / sqrt(1000) ≈ 0.0003
        np.testing.assert_allclose(
            calibration['accel_bias'], true_accel_bias, atol=0.001
        )
        np.testing.assert_allclose(
            calibration['gyro_bias'], true_gyro_bias, atol=0.0001
        )
    
    def test_different_gravity_orientations(self):
        """Test bias estimation with different gravity orientations."""
        n_samples = 100
        
        # Test X-axis gravity (sensor on side)
        gravity_x = np.array([9.81, 0, 0])
        accel_bias = np.array([0.05, -0.03, 0.02])
        accel_samples = np.tile(gravity_x + accel_bias, (n_samples, 1))
        gyro_samples = np.zeros((n_samples, 3))
        
        calibration = estimate_imu_bias_stationary(accel_samples, gyro_samples)
        self.assertEqual(calibration['gravity_axis'], 0)  # X-axis
        
        # Test Y-axis gravity
        gravity_y = np.array([0, -9.81, 0])
        accel_samples = np.tile(gravity_y + accel_bias, (n_samples, 1))
        
        calibration = estimate_imu_bias_stationary(accel_samples, gyro_samples)
        self.assertEqual(calibration['gravity_axis'], 1)  # Y-axis
    
    def test_synthetic_data_generation(self):
        """Test synthetic IMU data generation."""
        np.random.seed(42)
        
        data = generate_synthetic_imu_stationary(
            duration=5.0,
            rate=100.0,
            accel_bias=np.array([0.1, -0.05, 0.03]),
            gyro_bias=np.array([0.02, -0.01, 0.015])
        )
        
        # Check data structure
        self.assertIn('t', data)
        self.assertIn('accel', data)
        self.assertIn('gyro', data)
        self.assertIn('true_accel_bias', data)
        self.assertIn('true_gyro_bias', data)
        
        # Check dimensions
        expected_samples = int(5.0 * 100.0)
        self.assertEqual(len(data['t']), expected_samples)
        self.assertEqual(data['accel'].shape, (expected_samples, 3))
        self.assertEqual(data['gyro'].shape, (expected_samples, 3))
        
        # Verify biases match
        np.testing.assert_allclose(
            data['true_accel_bias'], np.array([0.1, -0.05, 0.03])
        )
        np.testing.assert_allclose(
            data['true_gyro_bias'], np.array([0.02, -0.01, 0.015])
        )


class TestExtrinsicCalibration(unittest.TestCase):
    """Test 2D extrinsic calibration (lever-arm + rotation)."""
    
    def test_identity_transformation(self):
        """Test calibration with identity transformation (no offset)."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([np.cos(t), np.sin(t)])
        
        # No transformation
        p_sensor2 = p_sensor1.copy()
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Should recover identity
        np.testing.assert_allclose(R_est, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(t_est, np.zeros(2), atol=1e-10)
    
    def test_pure_translation(self):
        """Test calibration with pure translation (no rotation)."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([np.cos(t), np.sin(t)])
        
        # Apply translation only
        true_t = np.array([2.0, -1.5])
        p_sensor2 = p_sensor1 + true_t
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Should recover identity rotation and true translation
        np.testing.assert_allclose(R_est, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(t_est, true_t, atol=1e-10)
    
    def test_pure_rotation(self):
        """Test calibration with pure rotation (no translation)."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([np.cos(t), np.sin(t)])
        
        # Apply rotation only
        angle = np.pi / 4  # 45 degrees
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        p_sensor2 = (R_true @ p_sensor1.T).T
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Should recover true rotation and zero translation
        np.testing.assert_allclose(R_est, R_true, atol=1e-10)
        np.testing.assert_allclose(t_est, np.zeros(2), atol=1e-10)
    
    def test_combined_rotation_and_translation(self):
        """Test calibration with both rotation and translation."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([np.cos(t), np.sin(t)])
        
        # Apply rotation and translation
        angle = np.pi / 6  # 30 degrees
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        t_true = np.array([1.0, 2.0])
        p_sensor2 = (R_true @ p_sensor1.T).T + t_true
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Should recover true transformation
        np.testing.assert_allclose(R_est, R_true, atol=1e-10)
        np.testing.assert_allclose(t_est, t_true, atol=1e-10)
    
    def test_with_measurement_noise(self):
        """Test calibration robustness with noisy measurements."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 500  # More samples to average out noise
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([5 * np.cos(t), 5 * np.sin(t)])
        
        # Apply known transformation
        angle = np.pi / 4
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        t_true = np.array([0.5, -0.3])
        p_sensor2 = (R_true @ p_sensor1.T).T + t_true
        
        # Add noise
        noise_std = 0.05
        p_sensor1 += np.random.randn(n_samples, 2) * noise_std
        p_sensor2 += np.random.randn(n_samples, 2) * noise_std
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Should recover approximately (tolerance based on noise)
        # With 500 samples and 0.05m noise, error should be < 0.01
        np.testing.assert_allclose(R_est, R_true, atol=0.01)
        np.testing.assert_allclose(t_est, t_true, atol=0.01)
    
    def test_rotation_matrix_properties(self):
        """Test that estimated rotation matrix has proper properties."""
        np.random.seed(42)
        
        # Generate trajectory
        n_samples = 100
        t = np.linspace(0, 10, n_samples)
        p_sensor1 = np.column_stack([np.cos(t), np.sin(t)])
        p_sensor2 = np.column_stack([np.sin(t), -np.cos(t)])
        
        R_est, t_est = calibrate_extrinsic_2d_least_squares(p_sensor1, p_sensor2)
        
        # Check rotation matrix properties
        # 1. Orthogonality: R @ R.T = I
        np.testing.assert_allclose(R_est @ R_est.T, np.eye(2), atol=1e-10)
        
        # 2. Determinant = 1 (proper rotation)
        self.assertAlmostEqual(np.linalg.det(R_est), 1.0, places=10)
        
        # 3. Preserves norm: ||R @ v|| = ||v||
        v = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            np.linalg.norm(R_est @ v),
            np.linalg.norm(v),
            atol=1e-10
        )
    
    def test_synthetic_data_generation(self):
        """Test synthetic extrinsic calibration data generation."""
        np.random.seed(42)
        
        data = generate_synthetic_extrinsic_data(
            duration=20.0,
            rate=10.0,
            lever_arm=np.array([1.0, 0.5]),
            rotation_angle=np.pi / 3  # 60 degrees
        )
        
        # Check data structure
        self.assertIn('t', data)
        self.assertIn('p_sensor1', data)
        self.assertIn('p_sensor2', data)
        self.assertIn('true_R', data)
        self.assertIn('true_t', data)
        self.assertIn('true_rotation_angle', data)
        
        # Check dimensions
        expected_samples = int(20.0 * 10.0)
        self.assertEqual(len(data['t']), expected_samples)
        self.assertEqual(data['p_sensor1'].shape, (expected_samples, 2))
        self.assertEqual(data['p_sensor2'].shape, (expected_samples, 2))
        
        # Check rotation matrix properties
        R = data['true_R']
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
        
        # Check angle
        angle_from_matrix = np.arctan2(R[1, 0], R[0, 0])
        self.assertAlmostEqual(angle_from_matrix, np.pi / 3, places=10)


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests for calibration workflow."""
    
    def test_imu_calibration_workflow(self):
        """Test full IMU calibration workflow."""
        np.random.seed(42)
        
        # Generate data
        data = generate_synthetic_imu_stationary(duration=5.0, rate=100.0)
        
        # Estimate biases
        calibration = estimate_imu_bias_stationary(data['accel'], data['gyro'])
        
        # Verify estimates are close to truth
        accel_error = np.linalg.norm(
            calibration['accel_bias'] - data['true_accel_bias']
        )
        gyro_error = np.linalg.norm(
            calibration['gyro_bias'] - data['true_gyro_bias']
        )
        
        # With 500 samples and reasonable noise, errors should be small
        self.assertLess(accel_error, 0.005)  # < 5 mm/s²
        self.assertLess(gyro_error, 0.0005)  # < 0.03 deg/s
    
    def test_extrinsic_calibration_workflow(self):
        """Test full extrinsic calibration workflow."""
        np.random.seed(42)
        
        # Generate data
        true_angle = np.pi / 4
        true_lever_arm = np.array([0.5, 0.3])
        
        data = generate_synthetic_extrinsic_data(
            duration=30.0,
            rate=10.0,
            lever_arm=true_lever_arm,
            rotation_angle=true_angle
        )
        
        # Estimate calibration
        R_est, t_est = calibrate_extrinsic_2d_least_squares(
            data['p_sensor1'],
            data['p_sensor2']
        )
        
        # Verify estimates
        np.testing.assert_allclose(R_est, data['true_R'], atol=0.01)
        np.testing.assert_allclose(t_est, data['true_t'], atol=0.01)
        
        # Verify alignment quality
        p1_transformed = (R_est @ data['p_sensor1'].T).T + t_est
        residuals = data['p_sensor2'] - p1_transformed
        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        
        # RMSE should be close to measurement noise (~0.05m)
        self.assertLess(rmse, 0.15)  # Allow some margin


if __name__ == '__main__':
    unittest.main()

