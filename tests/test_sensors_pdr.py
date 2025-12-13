"""
Unit tests for core/sensors/pdr.py (Pedestrian Dead Reckoning).

Tests cover:
    - Total acceleration magnitude (Eq. 6.46)
    - Gravity removal (Eq. 6.47)
    - Step frequency (Eq. 6.48)
    - Step length (Eq. 6.49)
    - 2D position update (Eq. 6.50)
    - Helper functions (step detection, heading integration)
    - Edge cases and validation

Run with: pytest tests/test_sensors_pdr.py -v
"""

import unittest
import numpy as np
import pytest

from core.sensors.pdr import (
    total_accel_magnitude,
    remove_gravity_from_magnitude,
    step_frequency,
    step_length,
    pdr_step_update,
    detect_step_simple,
    integrate_gyro_heading,
    wrap_heading,
)


class TestTotalAccelMagnitude(unittest.TestCase):
    """Test suite for total acceleration magnitude (Eq. 6.46)."""

    def test_accel_magnitude_stationary(self) -> None:
        """Test magnitude for stationary sensor (gravity only)."""
        accel = np.array([0.0, 0.0, -9.81])  # gravity pointing down
        mag = total_accel_magnitude(accel)

        assert np.isclose(mag, 9.81, atol=0.01)

    def test_accel_magnitude_pythagorean(self) -> None:
        """Test magnitude follows Pythagorean theorem."""
        accel = np.array([3.0, 4.0, 0.0])  # 3-4-5 triangle
        mag = total_accel_magnitude(accel)

        assert np.isclose(mag, 5.0)

    def test_accel_magnitude_tilted(self) -> None:
        """Test magnitude for tilted stationary sensor."""
        # 45° tilt: gravity has x and z components
        g = 9.81
        accel = np.array([g / np.sqrt(2), 0.0, -g / np.sqrt(2)])
        mag = total_accel_magnitude(accel)

        # Magnitude should still be g
        assert np.isclose(mag, g, atol=0.01)

    def test_accel_magnitude_walking(self) -> None:
        """Test magnitude during walking motion."""
        # Walking: includes dynamic component
        accel = np.array([1.5, 0.5, -10.2])  # gravity + motion
        mag = total_accel_magnitude(accel)

        # Should be > g
        assert mag > 9.81
        assert np.isclose(mag, 10.37, atol=0.1)

    def test_accel_magnitude_zero(self) -> None:
        """Test magnitude for zero acceleration (free fall)."""
        accel = np.zeros(3)
        mag = total_accel_magnitude(accel)

        assert mag == 0.0

    def test_accel_magnitude_invalid_shape(self) -> None:
        """Test that invalid shape raises error."""
        accel_bad = np.array([1.0, 2.0])  # Wrong: should be (3,)

        with pytest.raises(ValueError, match="must have shape"):
            total_accel_magnitude(accel_bad)


class TestRemoveGravityFromMagnitude(unittest.TestCase):
    """Test suite for gravity removal (Eq. 6.47)."""

    def test_gravity_removal_stationary(self) -> None:
        """Test gravity removal for stationary sensor."""
        a_mag = 9.81  # gravity only
        a_dyn = remove_gravity_from_magnitude(a_mag)

        assert np.isclose(a_dyn, 0.0, atol=0.01)

    def test_gravity_removal_walking_peak(self) -> None:
        """Test gravity removal during walking peak."""
        a_mag = 12.0  # gravity + dynamic
        a_dyn = remove_gravity_from_magnitude(a_mag)

        # Dynamic component
        assert np.isclose(a_dyn, 12.0 - 9.81, atol=0.01)

    def test_gravity_removal_walking_valley(self) -> None:
        """Test gravity removal during walking valley (free fall phase)."""
        a_mag = 8.0  # less than gravity
        a_dyn = remove_gravity_from_magnitude(a_mag)

        # Negative dynamic component
        assert np.isclose(a_dyn, 8.0 - 9.81, atol=0.01)
        assert a_dyn < 0

    def test_gravity_removal_custom_g(self) -> None:
        """Test gravity removal with custom gravity value."""
        a_mag = 10.0
        g_custom = 9.80665  # standard gravity
        a_dyn = remove_gravity_from_magnitude(a_mag, g=g_custom)

        assert np.isclose(a_dyn, 10.0 - 9.80665, atol=0.001)


class TestStepFrequency(unittest.TestCase):
    """Test suite for step frequency (Eq. 6.48)."""

    def test_step_frequency_normal_walking(self) -> None:
        """Test frequency for normal walking pace."""
        dt = 0.5  # 0.5 seconds between steps
        freq = step_frequency(dt)

        # 2 Hz = 120 steps/min
        assert np.isclose(freq, 2.0)

    def test_step_frequency_slow_walking(self) -> None:
        """Test frequency for slow walking."""
        dt = 0.7  # 0.7 seconds between steps
        freq = step_frequency(dt)

        # ~1.43 Hz = ~86 steps/min
        assert np.isclose(freq, 1.0 / 0.7)

    def test_step_frequency_fast_walking(self) -> None:
        """Test frequency for fast walking/jogging."""
        dt = 0.4  # 0.4 seconds between steps
        freq = step_frequency(dt)

        # 2.5 Hz = 150 steps/min
        assert np.isclose(freq, 2.5)

    def test_step_frequency_reciprocal(self) -> None:
        """Test that frequency is exact reciprocal of period."""
        dt = 0.6
        freq = step_frequency(dt)

        assert np.isclose(freq * dt, 1.0)

    def test_step_frequency_invalid_dt(self) -> None:
        """Test that zero or negative dt raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            step_frequency(0.0)

        with pytest.raises(ValueError):
            step_frequency(-0.5)


class TestStepLength(unittest.TestCase):
    """Test suite for step length (Eq. 6.49)."""

    def test_step_length_typical_person(self) -> None:
        """Test step length for typical person with Weinberg formula."""
        height = 1.75  # m
        freq = 2.0  # Hz (normal walking)

        L = step_length(height, freq)

        # Weinberg formula with default params gives ~1.4 m
        # (Note: typical step lengths are 0.6-0.8 m; c parameter may need calibration)
        assert 1.0 < L < 1.6

    def test_step_length_increases_with_height(self) -> None:
        """Test that taller people have longer steps."""
        freq = 2.0

        L_short = step_length(h=1.60, f_step=freq)
        L_tall = step_length(h=1.90, f_step=freq)

        assert L_tall > L_short

    def test_step_length_increases_with_frequency(self) -> None:
        """Test that faster walking gives longer steps."""
        height = 1.75

        L_slow = step_length(h=height, f_step=1.5)
        L_fast = step_length(h=height, f_step=2.5)

        assert L_fast > L_slow

    def test_step_length_weinberg_formula(self) -> None:
        """Test Weinberg formula explicitly."""
        h = 1.8
        f = 2.0
        a = 0.371
        b = 0.227
        c = 1.0

        L = step_length(h, f, a, b, c)

        # Manual calculation: L = c * h^a * f^b
        L_expected = c * (h**a) * (f**b)

        assert np.isclose(L, L_expected)

    def test_step_length_custom_parameters(self) -> None:
        """Test step length with custom calibration parameters."""
        h = 1.70
        f = 2.0
        c_custom = 1.1  # personal calibration factor

        L_default = step_length(h, f, c=1.0)
        L_custom = step_length(h, f, c=c_custom)

        # Custom should be 10% longer
        assert np.isclose(L_custom, L_default * 1.1)

    def test_step_length_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError, match="height"):
            step_length(h=0.0, f_step=2.0)

        with pytest.raises(ValueError, match="f_step"):
            step_length(h=1.75, f_step=0.0)

        with pytest.raises(ValueError, match="c"):
            step_length(h=1.75, f_step=2.0, c=0.0)


class TestPdrStepUpdate(unittest.TestCase):
    """Test suite for 2D position update (Eq. 6.50)."""

    def test_pdr_step_update_north(self) -> None:
        """Test step update heading north."""
        p0 = np.array([0.0, 0.0])
        step_len = 0.7  # m
        heading = np.pi / 2  # 90° = north

        p1 = pdr_step_update(p0, step_len, heading)

        # Should move north (y increases)
        expected = np.array([0.0, 0.7])
        np.testing.assert_array_almost_equal(p1, expected, decimal=5)

    def test_pdr_step_update_east(self) -> None:
        """Test step update heading east."""
        p0 = np.array([0.0, 0.0])
        step_len = 0.7
        heading = 0.0  # 0° = east

        p1 = pdr_step_update(p0, step_len, heading)

        # Should move east (x increases)
        expected = np.array([0.7, 0.0])
        np.testing.assert_array_almost_equal(p1, expected, decimal=5)

    def test_pdr_step_update_northeast(self) -> None:
        """Test step update heading northeast (45°)."""
        p0 = np.array([0.0, 0.0])
        step_len = 1.0  # m
        heading = np.pi / 4  # 45° = northeast

        p1 = pdr_step_update(p0, step_len, heading)

        # Should move diagonally
        expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        np.testing.assert_array_almost_equal(p1, expected, decimal=5)

    def test_pdr_step_update_west(self) -> None:
        """Test step update heading west (180°)."""
        p0 = np.array([5.0, 3.0])
        step_len = 0.8
        heading = np.pi  # 180° = west

        p1 = pdr_step_update(p0, step_len, heading)

        # Should move west (x decreases)
        expected = np.array([5.0 - 0.8, 3.0])
        np.testing.assert_array_almost_equal(p1, expected, decimal=5)

    def test_pdr_step_update_multiple_steps(self) -> None:
        """Test multiple consecutive steps."""
        p = np.array([0.0, 0.0])
        step_len = 0.7
        heading = 0.0  # east

        # Take 10 steps east
        for _ in range(10):
            p = pdr_step_update(p, step_len, heading)

        # Should be at (7.0, 0.0)
        expected = np.array([7.0, 0.0])
        np.testing.assert_array_almost_equal(p, expected, decimal=5)

    def test_pdr_step_update_zero_step_length(self) -> None:
        """Test that zero step length keeps position unchanged."""
        p0 = np.array([1.0, 2.0])
        step_len = 0.0
        heading = np.pi / 4

        p1 = pdr_step_update(p0, step_len, heading)

        np.testing.assert_array_equal(p1, p0)

    def test_pdr_step_update_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        p = np.array([0.0, 0.0])

        # Wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            pdr_step_update(np.array([0.0]), step_len=0.7, heading_rad=0.0)

        # Negative step length
        with pytest.raises(ValueError, match="non-negative"):
            pdr_step_update(p, step_len=-0.5, heading_rad=0.0)


class TestDetectStepSimple(unittest.TestCase):
    """Test suite for simple step detection."""

    def test_detect_step_stationary(self) -> None:
        """Test no step detection for stationary sensor."""
        mag_window = np.ones(20) * 9.81  # constant at gravity
        threshold = 11.0

        step_detected = detect_step_simple(mag_window, threshold)

        assert step_detected == False

    def test_detect_step_with_peak(self) -> None:
        """Test step detection when peak exceeds threshold."""
        # Create window with peak
        mag_window = np.concatenate(
            [np.ones(10) * 9.8, [12.0], np.ones(9) * 9.8]  # peak above threshold
        )
        threshold = 11.0

        step_detected = detect_step_simple(mag_window, threshold)

        assert step_detected == True

    def test_detect_step_below_threshold(self) -> None:
        """Test no detection when peak is below threshold."""
        mag_window = np.concatenate([np.ones(10) * 9.8, [10.5], np.ones(9) * 9.8])
        threshold = 11.0

        step_detected = detect_step_simple(mag_window, threshold)

        assert step_detected == False

    def test_detect_step_exactly_at_threshold(self) -> None:
        """Test step detection at exact threshold (should not detect)."""
        mag_window = np.concatenate([np.ones(10) * 9.8, [11.0], np.ones(9) * 9.8])
        threshold = 11.0

        step_detected = detect_step_simple(mag_window, threshold)

        # > threshold, not >=
        assert step_detected == False


class TestIntegrateGyroHeading(unittest.TestCase):
    """Test suite for gyro heading integration."""

    def test_integrate_heading_no_rotation(self) -> None:
        """Test heading doesn't change with zero yaw rate."""
        heading = 0.0
        omega_z = 0.0
        dt = 1.0

        heading_new = integrate_gyro_heading(heading, omega_z, dt)

        assert heading_new == 0.0

    def test_integrate_heading_turn_left(self) -> None:
        """Test heading update for left turn."""
        heading = 0.0  # facing east
        omega_z = 0.5  # rad/s (counter-clockwise)
        dt = 1.0

        heading_new = integrate_gyro_heading(heading, omega_z, dt)

        assert np.isclose(heading_new, 0.5)

    def test_integrate_heading_turn_right(self) -> None:
        """Test heading update for right turn."""
        heading = np.pi / 2  # facing north
        omega_z = -0.3  # rad/s (clockwise)
        dt = 1.0

        heading_new = integrate_gyro_heading(heading, omega_z, dt)

        assert np.isclose(heading_new, np.pi / 2 - 0.3)

    def test_integrate_heading_multiple_steps(self) -> None:
        """Test heading integration over multiple time steps."""
        heading = 0.0
        omega_z = 0.1  # rad/s
        dt = 0.1
        n_steps = 10

        for _ in range(n_steps):
            heading = integrate_gyro_heading(heading, omega_z, dt)

        # After 10 steps: heading = 0.1 * 0.1 * 10 = 0.1 rad
        assert np.isclose(heading, 0.1)

    def test_integrate_heading_invalid_dt(self) -> None:
        """Test that zero or negative dt raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            integrate_gyro_heading(0.0, 0.5, dt=0.0)


class TestWrapHeading(unittest.TestCase):
    """Test suite for heading wrapping."""

    def test_wrap_heading_in_range(self) -> None:
        """Test wrapping preserves angles already in [-π, π]."""
        heading = np.pi / 4
        wrapped = wrap_heading(heading)

        assert np.isclose(wrapped, heading)

    def test_wrap_heading_greater_than_pi(self) -> None:
        """Test wrapping for angle > π."""
        heading = 4.0  # > π
        wrapped = wrap_heading(heading)

        # Should wrap to negative
        assert -np.pi <= wrapped <= np.pi
        # 4.0 - 2π ≈ -2.28
        assert np.isclose(wrapped, 4.0 - 2 * np.pi, atol=0.01)

    def test_wrap_heading_less_than_minus_pi(self) -> None:
        """Test wrapping for angle < -π."""
        heading = -4.0  # < -π
        wrapped = wrap_heading(heading)

        assert -np.pi <= wrapped <= np.pi
        # -4.0 + 2π ≈ 2.28
        assert np.isclose(wrapped, -4.0 + 2 * np.pi, atol=0.01)

    def test_wrap_heading_multiple_revolutions(self) -> None:
        """Test wrapping for large angles (multiple revolutions)."""
        heading = 10.0  # several revolutions
        wrapped = wrap_heading(heading)

        assert -np.pi <= wrapped <= np.pi

    def test_wrap_heading_zero(self) -> None:
        """Test wrapping zero angle."""
        wrapped = wrap_heading(0.0)
        assert wrapped == 0.0

    def test_wrap_heading_pi(self) -> None:
        """Test wrapping ±π."""
        wrapped_pos = wrap_heading(np.pi)
        wrapped_neg = wrap_heading(-np.pi)

        # Both should be at boundary
        assert np.isclose(abs(wrapped_pos), np.pi)
        assert np.isclose(abs(wrapped_neg), np.pi)


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for PDR."""

    def test_step_length_extreme_values(self) -> None:
        """Test step length with extreme but valid inputs."""
        # Very tall person, very fast
        L_extreme = step_length(h=2.2, f_step=4.0)
        assert L_extreme > 0
        assert not np.isnan(L_extreme)

        # Very short person, very slow
        L_small = step_length(h=1.3, f_step=1.0)
        assert L_small > 0
        assert not np.isnan(L_small)

    def test_pdr_full_circle(self) -> None:
        """Test PDR completing a full circle."""
        p = np.array([0.0, 0.0])
        step_len = 0.1  # small steps
        n_steps = 100

        for i in range(n_steps):
            heading = 2 * np.pi * i / n_steps  # rotate around circle
            p = pdr_step_update(p, step_len, heading)

        # Should return close to origin (not exact due to discrete steps)
        distance_from_origin = np.linalg.norm(p)
        assert distance_from_origin < 1.0  # reasonable closure error

    def test_accel_magnitude_large_values(self) -> None:
        """Test magnitude with large accelerations (impact, jump)."""
        accel_large = np.array([5.0, 10.0, -20.0])
        mag = total_accel_magnitude(accel_large)

        assert mag > 20.0
        assert not np.isinf(mag)


if __name__ == "__main__":
    unittest.main()

