"""
Unit tests for IMU unit conversion utilities.

Tests all conversion functions in core.sensors.units to ensure correct
unit handling for IMU specifications.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np
from core.sensors.units import (
    # Forward conversions
    deg_per_hour_to_rad_per_sec,
    deg_per_sec_to_rad_per_sec,
    deg_per_sqrt_hour_to_rad_per_sqrt_sec,
    rad_per_hour_to_rad_per_sec,
    mg_to_mps2,
    ug_to_mps2,
    mps_per_sqrt_hour_to_mps_per_sqrt_sec,
    # Reverse conversions
    rad_per_sec_to_deg_per_hour,
    rad_per_sec_to_deg_per_sec,
    rad_per_sqrt_sec_to_deg_per_sqrt_hour,
    mps2_to_mg,
    mps_per_sqrt_sec_to_mps_per_sqrt_hour,
    # Formatters
    format_gyro_bias,
    format_accel_bias,
    format_arw,
    format_vrw,
)


class TestGyroUnitConversions(unittest.TestCase):
    """Test gyroscope unit conversions."""
    
    def test_deg_per_hour_to_rad_per_sec(self):
        """Test conversion from deg/hr to rad/s."""
        # 10 deg/hr should be 10/(3600*180/π) rad/s ≈ 0.000048 rad/s
        result = deg_per_hour_to_rad_per_sec(10.0)
        expected = np.deg2rad(10.0) / 3600.0
        self.assertAlmostEqual(result, expected, places=10)
        
        # Also check that it's approximately 0.0028 deg/s
        result_deg_s = np.rad2deg(result)
        self.assertAlmostEqual(result_deg_s, 10.0/3600.0, places=6)
    
    def test_deg_per_hour_to_rad_per_sec_acceptance_criterion(self):
        """Test acceptance criterion: 10 deg/hr = 0.0028 deg/s."""
        bias_rad_s = deg_per_hour_to_rad_per_sec(10.0)
        bias_deg_s = np.rad2deg(bias_rad_s)
        
        # Should be approximately 0.002778 deg/s (10/3600)
        expected_deg_s = 10.0 / 3600.0
        self.assertAlmostEqual(bias_deg_s, expected_deg_s, places=6)
        
        # Check it's in the right ballpark for acceptance
        self.assertLess(abs(bias_deg_s - 0.0028), 0.0001)
    
    def test_deg_per_sec_to_rad_per_sec(self):
        """Test conversion from deg/s to rad/s."""
        result = deg_per_sec_to_rad_per_sec(180.0)
        expected = np.pi
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_deg_per_sqrt_hour_to_rad_per_sqrt_sec(self):
        """Test ARW conversion from deg/√hr to rad/√s."""
        result = deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1)
        expected = np.deg2rad(0.1) / np.sqrt(3600.0)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_rad_per_hour_to_rad_per_sec(self):
        """Test conversion from rad/hr to rad/s."""
        result = rad_per_hour_to_rad_per_sec(3600.0)
        expected = 1.0
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_gyro_round_trip(self):
        """Test round-trip conversion for gyro bias."""
        original = 10.0  # deg/hr
        rad_s = deg_per_hour_to_rad_per_sec(original)
        back = rad_per_sec_to_deg_per_hour(rad_s)
        self.assertAlmostEqual(original, back, places=6)
    
    def test_arw_round_trip(self):
        """Test round-trip conversion for ARW."""
        original = 0.1  # deg/√hr
        rad_sqrt_s = deg_per_sqrt_hour_to_rad_per_sqrt_sec(original)
        back = rad_per_sqrt_sec_to_deg_per_sqrt_hour(rad_sqrt_s)
        self.assertAlmostEqual(original, back, places=6)


class TestAccelUnitConversions(unittest.TestCase):
    """Test accelerometer unit conversions."""
    
    def test_mg_to_mps2(self):
        """Test conversion from mg to m/s²."""
        # 1 mg = 0.001 * 9.80665 m/s²
        result = mg_to_mps2(1.0)
        expected = 0.001 * 9.80665
        self.assertAlmostEqual(result, expected, places=6)
        
        # 10 mg = 0.098067 m/s²
        result = mg_to_mps2(10.0)
        expected = 0.01 * 9.80665
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_ug_to_mps2(self):
        """Test conversion from µg to m/s²."""
        result = ug_to_mps2(1.0)
        expected = 1e-6 * 9.80665
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_mps_per_sqrt_hour_to_mps_per_sqrt_sec(self):
        """Test VRW conversion from m/s/√hr to m/s/√s."""
        result = mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01)
        expected = 0.01 / np.sqrt(3600.0)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_accel_round_trip(self):
        """Test round-trip conversion for accel bias."""
        original = 10.0  # mg
        mps2 = mg_to_mps2(original)
        back = mps2_to_mg(mps2)
        self.assertAlmostEqual(original, back, places=6)
    
    def test_vrw_round_trip(self):
        """Test round-trip conversion for VRW."""
        original = 0.01  # m/s/√hr
        mps_sqrt_s = mps_per_sqrt_hour_to_mps_per_sqrt_sec(original)
        back = mps_per_sqrt_sec_to_mps_per_sqrt_hour(mps_sqrt_s)
        self.assertAlmostEqual(original, back, places=6)


class TestArrayConversions(unittest.TestCase):
    """Test that conversions work with numpy arrays."""
    
    def test_deg_per_hour_to_rad_per_sec_array(self):
        """Test array conversion for gyro bias."""
        input_array = np.array([1.0, 10.0, 100.0])
        result = deg_per_hour_to_rad_per_sec(input_array)
        
        self.assertEqual(result.shape, input_array.shape)
        
        # Check each element
        for i, val in enumerate(input_array):
            expected = deg_per_hour_to_rad_per_sec(val)
            self.assertAlmostEqual(result[i], expected, places=10)
    
    def test_mg_to_mps2_array(self):
        """Test array conversion for accel bias."""
        input_array = np.array([1.0, 10.0, 100.0])
        result = mg_to_mps2(input_array)
        
        self.assertEqual(result.shape, input_array.shape)
        
        # Check each element
        for i, val in enumerate(input_array):
            expected = mg_to_mps2(val)
            self.assertAlmostEqual(result[i], expected, places=10)


class TestFormatters(unittest.TestCase):
    """Test formatting functions for human-readable output."""
    
    def test_format_gyro_bias(self):
        """Test gyro bias formatting."""
        bias_rad_s = deg_per_hour_to_rad_per_sec(10.0)
        result = format_gyro_bias(bias_rad_s)
        
        # Should contain both units
        self.assertIn("deg/hr", result)
        self.assertIn("deg/s", result)
        self.assertIn("10.00", result)
        self.assertIn("0.0028", result)
    
    def test_format_accel_bias(self):
        """Test accel bias formatting."""
        bias_mps2 = mg_to_mps2(10.0)
        result = format_accel_bias(bias_mps2)
        
        # Should contain both units
        self.assertIn("mg", result)
        self.assertIn("m/s", result)
        self.assertIn("10.00", result)
    
    def test_format_arw(self):
        """Test ARW formatting."""
        arw_rad_sqrt_s = deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1)
        result = format_arw(arw_rad_sqrt_s)
        
        self.assertIn("deg", result)
        self.assertIn("0.10", result)
    
    def test_format_vrw(self):
        """Test VRW formatting."""
        vrw_mps_sqrt_s = mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01)
        result = format_vrw(vrw_mps_sqrt_s)
        
        self.assertIn("m/s", result)
        self.assertIn("0.01", result)


class TestPhysicalConsistency(unittest.TestCase):
    """Test physical consistency of conversions."""
    
    def test_consumer_grade_gyro_bias(self):
        """Test typical consumer-grade gyro bias values."""
        # Consumer grade: 10-100 deg/hr
        bias_deg_hr = 10.0
        bias_rad_s = deg_per_hour_to_rad_per_sec(bias_deg_hr)
        
        # Should be a small number in rad/s
        self.assertLess(bias_rad_s, 0.001)
        self.assertGreater(bias_rad_s, 0.00001)
    
    def test_tactical_grade_gyro_bias(self):
        """Test typical tactical-grade gyro bias values."""
        # Tactical grade: 0.1-10 deg/hr
        bias_deg_hr = 1.0
        bias_rad_s = deg_per_hour_to_rad_per_sec(bias_deg_hr)
        
        # Should be even smaller
        self.assertLess(bias_rad_s, 0.0001)
        self.assertGreater(bias_rad_s, 0.000001)
    
    def test_consumer_grade_accel_bias(self):
        """Test typical consumer-grade accel bias values."""
        # Consumer grade: 1-100 mg
        bias_mg = 10.0
        bias_mps2 = mg_to_mps2(bias_mg)
        
        # Should be around 0.1 m/s²
        self.assertLess(bias_mps2, 1.0)
        self.assertGreater(bias_mps2, 0.001)
    
    def test_arw_reasonable_values(self):
        """Test that ARW values are in reasonable range."""
        # Consumer grade: 0.1-1.0 deg/√hr
        arw_deg_sqrt_hr = 0.1
        arw_rad_sqrt_s = deg_per_sqrt_hour_to_rad_per_sqrt_sec(arw_deg_sqrt_hr)
        
        # Should be very small
        self.assertLess(arw_rad_sqrt_s, 0.001)
        self.assertGreater(arw_rad_sqrt_s, 0.00001)


if __name__ == "__main__":
    unittest.main()










