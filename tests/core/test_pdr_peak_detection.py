"""
Unit tests for PDR step detection using peak detection (Eqs. 6.46-6.47).

Tests the detect_steps_peak_detector() function which implements the book's
prescribed method for step detection.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np
from core.sensors.pdr import (
    total_accel_magnitude,
    remove_gravity_from_magnitude,
    detect_steps_peak_detector,
)


class TestPDRPeakDetection(unittest.TestCase):
    """Test PDR step detection with peak detector."""

    def test_total_accel_magnitude_stationary(self):
        """Test Eq. 6.46: stationary device should measure g."""
        # Stationary device with gravity pointing down in body frame
        accel_b = np.array([0.0, 0.0, -9.81])
        mag = total_accel_magnitude(accel_b)
        self.assertAlmostEqual(mag, 9.81, places=2)

    def test_total_accel_magnitude_walking(self):
        """Test Eq. 6.46: walking should have magnitude > g."""
        # Walking with additional motion acceleration
        accel_b = np.array([1.5, 0.5, -10.2])
        mag = total_accel_magnitude(accel_b)
        self.assertGreater(mag, 9.81)
        self.assertAlmostEqual(mag, 10.37, places=1)

    def test_remove_gravity_stationary(self):
        """Test Eq. 6.47: stationary should give ~0 after gravity removal."""
        a_mag_stationary = 9.81
        a_dynamic = remove_gravity_from_magnitude(a_mag_stationary, g=9.81)
        self.assertAlmostEqual(a_dynamic, 0.0, places=2)

    def test_remove_gravity_walking(self):
        """Test Eq. 6.47: walking should give positive value."""
        a_mag_walking = 12.0
        a_dynamic = remove_gravity_from_magnitude(a_mag_walking, g=9.81)
        self.assertAlmostEqual(a_dynamic, 2.19, places=1)
        self.assertGreater(a_dynamic, 0.0)

    def test_detect_steps_no_motion(self):
        """No steps should be detected for stationary signal."""
        # Stationary for 10 seconds at 100 Hz
        t = np.arange(0, 10, 0.01)
        N = len(t)
        
        # Constant gravity only (no motion)
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            -9.81 * np.ones(N)
        ])
        
        step_indices, accel_processed = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=1.0, min_peak_distance=0.3
        )
        
        # Should detect zero or very few steps
        self.assertLess(len(step_indices), 5, 
                        "Stationary signal should not detect many steps")

    def test_detect_steps_synthetic_walking(self):
        """Detect steps in synthetic walking pattern."""
        # 60 seconds at 100 Hz, 2 Hz step frequency
        t = np.arange(0, 60, 0.01)
        N = len(t)
        step_freq = 2.0  # Hz
        
        # Simulate vertical acceleration oscillations (walking)
        walking_amplitude = 2.5  # m/s²
        accel_z = -9.81 + walking_amplitude * np.sin(2 * np.pi * step_freq * t)
        
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            accel_z
        ])
        
        step_indices, accel_processed = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=1.0, min_peak_distance=0.3
        )
        
        # Expected: ~120 steps (60s * 2 Hz)
        expected_steps = 60 * step_freq
        detected_steps = len(step_indices)
        
        # Allow 50-200% of expected (detector tuning dependent)
        self.assertGreater(detected_steps, 0, "Should detect nonzero steps")
        self.assertGreater(detected_steps, expected_steps * 0.5,
                          f"Too few steps: {detected_steps} < {expected_steps * 0.5}")
        self.assertLess(detected_steps, expected_steps * 2.0,
                       f"Too many steps: {detected_steps} > {expected_steps * 2.0}")

    def test_detect_steps_refractory_period(self):
        """Test that min_peak_distance prevents double-counting."""
        # Signal with two close peaks (0.1s apart)
        t = np.arange(0, 10, 0.01)
        N = len(t)
        
        # Two wider Gaussian peaks at t=5.0s and t=5.1s (wider to survive filtering)
        accel_z = -9.81 + 5.0 * (
            np.exp(-((t - 5.0)**2) / 0.1) +
            np.exp(-((t - 5.1)**2) / 0.1)
        )
        
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            accel_z
        ])
        
        # With refractory period of 0.3s, should detect only 1 step
        step_indices, _ = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=1.0, min_peak_distance=0.3,
            lowpass_cutoff=None  # Disable filter to preserve sharp peaks
        )
        
        self.assertLessEqual(len(step_indices), 1,
                            "Refractory period should prevent double-counting")

    def test_detect_steps_without_filter(self):
        """Test peak detection without low-pass filter."""
        # Synthetic walking with noise
        t = np.arange(0, 10, 0.01)
        N = len(t)
        step_freq = 2.0
        
        walking_amplitude = 2.5
        accel_z = -9.81 + walking_amplitude * np.sin(2 * np.pi * step_freq * t)
        # Add noise
        noise = np.random.normal(0, 0.5, N)
        accel_z += noise
        
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            accel_z
        ])
        
        # Detect without filter (lowpass_cutoff=None)
        step_indices, accel_processed = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=1.0, min_peak_distance=0.3,
            lowpass_cutoff=None
        )
        
        # Should still detect steps (but possibly noisier)
        self.assertGreater(len(step_indices), 0)

    def test_detect_steps_with_filter(self):
        """Test peak detection with low-pass filter improves robustness."""
        # Synthetic walking with high-frequency noise
        t = np.arange(0, 10, 0.01)
        N = len(t)
        step_freq = 2.0
        
        walking_amplitude = 2.5
        accel_z = -9.81 + walking_amplitude * np.sin(2 * np.pi * step_freq * t)
        # Add high-frequency noise (20 Hz)
        noise = 0.8 * np.sin(2 * np.pi * 20 * t)
        accel_z += noise
        
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            accel_z
        ])
        
        # Detect with filter (5 Hz cutoff)
        step_indices_filtered, _ = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=1.0, min_peak_distance=0.3,
            lowpass_cutoff=5.0
        )
        
        # Expected: ~20 steps (10s * 2 Hz)
        expected_steps = 10 * step_freq
        detected_steps = len(step_indices_filtered)
        
        # With filtering, should be close to expected
        self.assertGreater(detected_steps, 0)
        self.assertLess(abs(detected_steps - expected_steps) / expected_steps, 0.5,
                       "Filtered detection should be within 50% of expected")

    def test_detect_steps_tunable_sensitivity(self):
        """Test that min_peak_height controls sensitivity."""
        # Weak walking signal
        t = np.arange(0, 10, 0.01)
        N = len(t)
        step_freq = 2.0
        
        # Lower amplitude (1.0 m/s²)
        walking_amplitude = 1.0
        accel_z = -9.81 + walking_amplitude * np.sin(2 * np.pi * step_freq * t)
        
        accel = np.column_stack([
            np.zeros(N),
            np.zeros(N),
            accel_z
        ])
        
        # High threshold (2.0 m/s²) - should detect few/no steps
        steps_high_thresh, _ = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=2.0, min_peak_distance=0.3
        )
        
        # Low threshold (0.5 m/s²) - should detect steps
        steps_low_thresh, _ = detect_steps_peak_detector(
            accel, dt=0.01, g=9.81,
            min_peak_height=0.5, min_peak_distance=0.3
        )
        
        # Low threshold should detect more steps
        self.assertGreaterEqual(len(steps_low_thresh), len(steps_high_thresh),
                               "Lower threshold should detect more steps")

    def test_detect_steps_input_validation(self):
        """Test input validation for detect_steps_peak_detector."""
        t = np.arange(0, 1, 0.01)
        N = len(t)
        accel = np.zeros((N, 3))
        
        # Invalid shape (1D instead of 2D)
        with self.assertRaises(ValueError):
            detect_steps_peak_detector(accel[:, 0], dt=0.01)
        
        # Invalid dt (negative)
        with self.assertRaises(ValueError):
            detect_steps_peak_detector(accel, dt=-0.01)
        
        # Invalid min_peak_distance (negative)
        with self.assertRaises(ValueError):
            detect_steps_peak_detector(accel, dt=0.01, min_peak_distance=-0.1)


if __name__ == "__main__":
    unittest.main()

