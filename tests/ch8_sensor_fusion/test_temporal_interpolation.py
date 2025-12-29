"""Unit tests for IMU interpolation and asynchronous measurement handling.

Tests the measurement-time handling and interpolation methods from Chapter 8,
Section 8.5.2.

Author: Li-Ta Hsu
Date: December 2025
References: Chapter 8, Section 8.5.2 (Measurement Timing and Interpolation)
"""

import unittest

import numpy as np

from ch8_sensor_fusion.tc_models import interpolate_imu_measurements


class TestIMUInterpolation(unittest.TestCase):
    """Test suite for IMU measurement interpolation."""
    
    def test_midpoint_interpolation(self) -> None:
        """Test interpolation at exact midpoint."""
        t_imu = np.array([0.0, 0.01, 0.02])
        accel = np.array([[1.0, 0.5], [1.2, 0.6], [1.4, 0.7]])
        gyro = np.array([0.1, 0.2, 0.3])
        
        # Query at midpoint between samples 1 and 2
        u, dt = interpolate_imu_measurements(0.015, t_imu, accel, gyro)
        
        # Should be average of [1.2, 0.6, 0.2] and [1.4, 0.7, 0.3]
        expected_u = np.array([1.3, 0.65, 0.25])
        np.testing.assert_array_almost_equal(u, expected_u)
        
        # Time since last IMU sample (0.01) should be 0.005s
        self.assertAlmostEqual(dt, 0.005, places=6)
    
    def test_quarter_point_interpolation(self) -> None:
        """Test interpolation at 1/4 point."""
        t_imu = np.array([0.0, 0.01, 0.02])
        accel = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        gyro = np.array([0.0, 0.1, 0.2])
        
        # Query at 1/4 point between samples 1 and 2 (t=0.0125)
        u, dt = interpolate_imu_measurements(0.0125, t_imu, accel, gyro)
        
        # alpha = (0.0125 - 0.01) / 0.01 = 0.25
        # u = 0.75 * [2.0, 1.0, 0.1] + 0.25 * [3.0, 2.0, 0.2]
        expected_u = np.array([2.25, 1.25, 0.125])
        np.testing.assert_array_almost_equal(u, expected_u, decimal=6)
        
        self.assertAlmostEqual(dt, 0.0025, places=6)
    
    def test_exact_sample_time(self) -> None:
        """Test query at exact IMU sample time."""
        t_imu = np.array([0.0, 0.01, 0.02, 0.03])
        accel = np.array([[1.0, 0.5], [1.2, 0.6], [1.4, 0.7], [1.6, 0.8]])
        gyro = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Query at exact sample time
        u, dt = interpolate_imu_measurements(0.02, t_imu, accel, gyro)
        
        # Should return exact sample value
        expected_u = np.array([1.4, 0.7, 0.3])
        np.testing.assert_array_almost_equal(u, expected_u)
        
        # dt is time from start of interval (t=0.02) to query (t=0.02), which is 0
        # This is correct: no propagation needed when query is at exact sample
        self.assertAlmostEqual(dt, 0.0, places=6)
    
    def test_first_interval(self) -> None:
        """Test interpolation in first interval."""
        t_imu = np.array([0.0, 0.01, 0.02])
        accel = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        gyro = np.array([0.0, 0.1, 0.2])
        
        # Query at t=0.005 (middle of first interval)
        u, dt = interpolate_imu_measurements(0.005, t_imu, accel, gyro)
        
        # Should be average of samples 0 and 1
        expected_u = np.array([0.5, 0.25, 0.05])
        np.testing.assert_array_almost_equal(u, expected_u)
        
        self.assertAlmostEqual(dt, 0.005, places=6)
    
    def test_last_interval(self) -> None:
        """Test interpolation in last interval."""
        t_imu = np.array([0.0, 0.01, 0.02])
        accel = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        gyro = np.array([0.0, 0.1, 0.2])
        
        # Query at t=0.015 (middle of last interval)
        u, dt = interpolate_imu_measurements(0.015, t_imu, accel, gyro)
        
        # Should be average of samples 1 and 2
        expected_u = np.array([1.5, 0.75, 0.15])
        np.testing.assert_array_almost_equal(u, expected_u)
        
        self.assertAlmostEqual(dt, 0.005, places=6)
    
    def test_query_before_range_raises(self) -> None:
        """Test that query before IMU range raises ValueError."""
        t_imu = np.array([0.01, 0.02, 0.03])
        accel = np.array([[1.0, 0.5], [1.2, 0.6], [1.4, 0.7]])
        gyro = np.array([0.1, 0.2, 0.3])
        
        with self.assertRaises(ValueError) as cm:
            interpolate_imu_measurements(0.005, t_imu, accel, gyro)
        
        self.assertIn("outside IMU range", str(cm.exception))
    
    def test_query_after_range_raises(self) -> None:
        """Test that query after IMU range raises ValueError."""
        t_imu = np.array([0.0, 0.01, 0.02])
        accel = np.array([[1.0, 0.5], [1.2, 0.6], [1.4, 0.7]])
        gyro = np.array([0.1, 0.2, 0.3])
        
        with self.assertRaises(ValueError) as cm:
            interpolate_imu_measurements(0.025, t_imu, accel, gyro)
        
        self.assertIn("outside IMU range", str(cm.exception))
    
    def test_realistic_imu_rate(self) -> None:
        """Test interpolation with realistic IMU rate (100 Hz)."""
        # 1 second of 100 Hz IMU data
        t_imu = np.arange(0, 1.0, 0.01)
        N = len(t_imu)
        
        # Sinusoidal accelerometer data
        accel = np.column_stack([
            np.sin(2 * np.pi * 1.0 * t_imu),  # 1 Hz sine
            np.cos(2 * np.pi * 1.0 * t_imu)
        ])
        gyro = 0.1 * np.sin(2 * np.pi * 0.5 * t_imu)  # 0.5 Hz sine
        
        # Query at 10.5 Hz (UWB rate with slight offset)
        t_query = 0.105  # Between 0.10 and 0.11
        
        u, dt = interpolate_imu_measurements(t_query, t_imu, accel, gyro)
        
        # Expected: linear interpolation between samples at 0.10 and 0.11
        # alpha = (0.105 - 0.10) / 0.01 = 0.5
        idx = 10  # t_imu[10] = 0.10
        expected_u = 0.5 * np.array([
            accel[idx, 0], accel[idx, 1], gyro[idx]
        ]) + 0.5 * np.array([
            accel[idx + 1, 0], accel[idx + 1, 1], gyro[idx + 1]
        ])
        
        np.testing.assert_array_almost_equal(u, expected_u, decimal=10)
        self.assertAlmostEqual(dt, 0.005, places=6)


class TestAsynchronousTimestampShift(unittest.TestCase):
    """Regression test for artificially shifted UWB timestamps."""
    
    def test_half_imu_dt_shift(self) -> None:
        """Test that fusion handles UWB timestamps shifted by half IMU dt.
        
        This is a regression test per the requirement: artificially shift
        UWB timestamps and ensure fusion still runs without collapsing.
        """
        # Create synthetic data
        dt_imu = 0.01  # 100 Hz
        dt_uwb = 0.1   # 10 Hz
        duration = 5.0
        
        t_imu = np.arange(0, duration, dt_imu)
        N_imu = len(t_imu)
        
        # Constant velocity motion
        accel = np.zeros((N_imu, 2))
        accel[:, 0] = 0.1  # Small constant acceleration
        gyro = np.zeros(N_imu)
        
        # Original UWB timestamps aligned with IMU
        t_uwb_aligned = np.arange(0, duration, dt_uwb)
        
        # Shifted UWB timestamps (half IMU dt = 5ms shift)
        t_uwb_shifted = t_uwb_aligned + dt_imu / 2
        
        # Test interpolation at shifted times
        for t_query in t_uwb_shifted:
            if t_query < t_imu[0] or t_query > t_imu[-1]:
                continue
            
            # Should not raise an exception
            u, dt = interpolate_imu_measurements(t_query, t_imu, accel, gyro)
            
            # Verify output is reasonable
            self.assertEqual(len(u), 3)
            self.assertGreaterEqual(dt, 0.0)
            self.assertLessEqual(dt, dt_imu)
            
            # Since constant acceleration, interpolated value should equal it
            np.testing.assert_array_almost_equal(u[:2], accel[0], decimal=10)
            self.assertAlmostEqual(u[2], 0.0, places=10)
    
    def test_random_timestamp_offsets(self) -> None:
        """Test fusion robustness with random timestamp perturbations."""
        # Create synthetic data
        dt_imu = 0.01  # 100 Hz
        dt_uwb = 0.1   # 10 Hz
        duration = 2.0
        
        t_imu = np.arange(0, duration, dt_imu)
        N_imu = len(t_imu)
        
        # Varying accelerometer data
        accel = np.column_stack([
            0.1 * np.sin(2 * np.pi * 0.5 * t_imu),
            0.1 * np.cos(2 * np.pi * 0.5 * t_imu)
        ])
        gyro = 0.05 * np.sin(2 * np.pi * 1.0 * t_imu)
        
        # UWB timestamps with random offsets (±5ms)
        np.random.seed(42)
        t_uwb_nominal = np.arange(0, duration, dt_uwb)
        t_uwb_perturbed = t_uwb_nominal + np.random.uniform(-0.005, 0.005, len(t_uwb_nominal))
        
        # Test interpolation at perturbed times
        for t_query in t_uwb_perturbed:
            if t_query < t_imu[0] or t_query > t_imu[-1]:
                continue
            
            # Should not raise an exception
            u, dt = interpolate_imu_measurements(t_query, t_imu, accel, gyro)
            
            # Verify output is reasonable
            self.assertEqual(len(u), 3)
            self.assertGreaterEqual(dt, 0.0)
            self.assertLessEqual(dt, dt_imu)
            
            # Check magnitude is reasonable (not NaN or inf)
            self.assertTrue(np.all(np.isfinite(u)))
            self.assertLess(np.linalg.norm(u), 1.0)  # Should be small


if __name__ == "__main__":
    unittest.main()

