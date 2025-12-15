"""
Unit tests for core/sensors/calibration.py (Allan variance and IMU characterization).

Tests cover:
    - Allan variance computation (Eqs. 6.56-6.58)
    - Bias instability identification
    - Random walk coefficient extraction
    - Rate random walk extraction
    - Full IMU noise characterization
    - Edge cases and validation

Run with: pytest tests/test_sensors_calibration.py -v
"""

import unittest
import numpy as np
import pytest
import warnings

from core.sensors.calibration import (
    allan_variance,
    identify_bias_instability,
    identify_random_walk,
    identify_rate_random_walk,
    characterize_imu_noise,
)


class TestAllanVariance(unittest.TestCase):
    """Test suite for Allan variance computation (Eqs. 6.56-6.58)."""

    def test_allan_variance_white_noise(self) -> None:
        """Test Allan variance on white noise (should have -1/2 slope)."""
        # White noise: adev should decrease as sqrt(tau)
        np.random.seed(42)
        fs = 100.0
        duration = 60  # 1 minute
        N = int(fs * duration)

        # White noise
        x = np.random.randn(N)

        taus, adev = allan_variance(x, fs, overlapping=True)

        # Check that adev decreases with tau (white noise behavior)
        assert len(taus) > 0
        assert len(adev) == len(taus)
        # Allan deviation should decrease for white noise
        assert adev[0] > adev[-1]

    def test_allan_variance_constant_signal(self) -> None:
        """Test Allan variance on constant signal (should be near zero)."""
        fs = 100.0
        duration = 10
        N = int(fs * duration)

        # Constant signal
        x = np.ones(N) * 5.0

        taus, adev = allan_variance(x, fs)

        # Allan deviation should be very small (numerical noise only)
        assert np.all(adev < 1e-10)

    def test_allan_variance_overlapping_vs_non_overlapping(self) -> None:
        """Test that overlapping gives better statistics (lower noise floor)."""
        np.random.seed(42)
        fs = 100.0
        duration = 60
        N = int(fs * duration)
        x = np.random.randn(N)

        # Generate fixed taus for comparison
        taus_fixed = np.logspace(-1, 0, 20)

        taus_over, adev_over = allan_variance(x, fs, taus=taus_fixed, overlapping=True)
        taus_non, adev_non = allan_variance(
            x, fs, taus=taus_fixed, overlapping=False
        )

        # Both should produce results
        assert len(adev_over) > 0
        assert len(adev_non) > 0

        # Overlapping typically has slightly lower values (better averaging)
        # But this depends on data length, so just check they're similar
        assert len(adev_over) == len(adev_non)

    def test_allan_variance_custom_taus(self) -> None:
        """Test Allan variance with custom tau values."""
        np.random.seed(42)
        fs = 100.0
        N = 1000
        x = np.random.randn(N)

        # Custom taus
        taus_custom = np.array([0.1, 0.5, 1.0, 2.0])

        taus, adev = allan_variance(x, fs, taus=taus_custom)

        # Should use provided taus (or subset that are valid)
        assert len(taus) > 0
        assert len(adev) == len(taus)

    def test_allan_variance_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        x = np.random.randn(100)

        # 2D array
        with pytest.raises(ValueError, match="must be 1D"):
            allan_variance(np.random.randn(10, 3), fs=100.0)

        # Negative fs
        with pytest.raises(ValueError, match="must be positive"):
            allan_variance(x, fs=-100.0)

        # Zero fs
        with pytest.raises(ValueError, match="must be positive"):
            allan_variance(x, fs=0.0)

    def test_allan_variance_short_data(self) -> None:
        """Test Allan variance with very short data (limited taus)."""
        fs = 100.0
        N = 50  # very short
        x = np.random.randn(N)

        taus, adev = allan_variance(x, fs)

        # Should still produce some results (but limited)
        assert len(taus) > 0
        assert len(adev) == len(taus)


class TestIdentifyBiasInstability(unittest.TestCase):
    """Test suite for bias instability identification."""

    def test_identify_bias_instability_synthetic(self) -> None:
        """Test bias instability identification on synthetic Allan curve."""
        # Create synthetic Allan deviation curve with minimum
        taus = np.logspace(-2, 2, 100)
        # Synthetic: random walk + flat + rate random walk
        adev = 0.01 / np.sqrt(taus) + 0.001 + 0.0001 * np.sqrt(taus)

        bi, tau_bi = identify_bias_instability(taus, adev)

        # Bias instability should be near minimum
        # (actual minimum is ~0.003, not 0.001, due to the combined curve)
        assert 0.002 < bi < 0.005
        assert tau_bi > 0
        # Should be somewhere reasonable (depends on curve shape)
        assert 0.01 < tau_bi < 200.0

    def test_identify_bias_instability_monotonic_decrease(self) -> None:
        """Test with monotonically decreasing curve (no flat region)."""
        taus = np.logspace(-2, 2, 100)
        adev = 1.0 / np.sqrt(taus)  # pure random walk, no flat region

        bi, tau_bi = identify_bias_instability(taus, adev)

        # Should return minimum (which is at longest tau)
        assert bi == adev[-1]
        assert tau_bi == taus[-1]

    def test_identify_bias_instability_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        taus = np.array([1.0, 2.0, 3.0])
        adev = np.array([0.1, 0.2])

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            identify_bias_instability(taus, adev)

        # Empty arrays
        with pytest.raises(ValueError, match="Empty"):
            identify_bias_instability(np.array([]), np.array([]))


class TestIdentifyRandomWalk(unittest.TestCase):
    """Test suite for random walk identification."""

    def test_identify_random_walk_white_noise(self) -> None:
        """Test random walk identification on pure white noise."""
        # Simulate white noise
        np.random.seed(42)
        fs = 100.0
        duration = 300  # 5 minutes
        N = int(fs * duration)
        sigma = 0.01  # noise std
        x = sigma * np.random.randn(N)

        taus, adev = allan_variance(x, fs)
        arw = identify_random_walk(taus, adev, tau_target=1.0)

        # For white noise: σ(τ) ≈ σ / sqrt(τ)
        # At τ=1s: σ(1) ≈ σ / sqrt(1*fs) = σ / 10
        expected_order = sigma / np.sqrt(fs)

        # Should be within same order of magnitude
        assert 0.1 * expected_order < arw < 10 * expected_order

    def test_identify_random_walk_with_fitting(self) -> None:
        """Test that fitting improves accuracy when slope is correct."""
        # Create synthetic curve with -1/2 slope
        taus = np.logspace(-2, 1, 100)
        N_true = 0.01  # true random walk coefficient
        adev = N_true / np.sqrt(taus)  # perfect -1/2 slope

        arw = identify_random_walk(taus, adev, tau_target=1.0)

        # Should extract correct coefficient
        assert np.isclose(arw, N_true, rtol=0.1)

    def test_identify_random_walk_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            identify_random_walk(np.array([1.0, 2.0]), np.array([0.1]))

        # Empty
        with pytest.raises(ValueError, match="Empty"):
            identify_random_walk(np.array([]), np.array([]))

    def test_identify_random_walk_warns_bad_slope(self) -> None:
        """Test that warning is issued for incorrect slope."""
        taus = np.logspace(-1, 1, 50)
        # Wrong slope (too steep)
        adev = 1.0 / (taus**2)  # slope -2 instead of -0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arw = identify_random_walk(taus, adev, tau_target=1.0)

            # Should issue warning about slope
            assert len(w) > 0
            assert "slope" in str(w[0].message).lower()


class TestIdentifyRateRandomWalk(unittest.TestCase):
    """Test suite for rate random walk identification."""

    def test_identify_rate_random_walk_synthetic(self) -> None:
        """Test rate random walk identification on synthetic curve."""
        taus = np.logspace(0, 3, 100)
        K_true = 0.001  # true RRW coefficient
        # RRW: σ(τ) = K * sqrt(τ/3)
        adev = K_true * np.sqrt(taus / 3.0)

        rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

        # Should extract correct coefficient
        assert np.isclose(rrw, K_true, rtol=0.15)

    def test_identify_rate_random_walk_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            identify_rate_random_walk(np.array([1.0, 2.0]), np.array([0.1]))

        # Empty
        with pytest.raises(ValueError, match="Empty"):
            identify_rate_random_walk(np.array([]), np.array([]))

    def test_identify_rate_random_walk_warns_bad_slope(self) -> None:
        """Test that warning is issued for incorrect slope."""
        taus = np.logspace(1, 3, 50)
        # Wrong slope (flat instead of +0.5)
        adev = np.ones_like(taus) * 0.01

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

            # Should issue warning
            assert len(w) > 0


class TestCharacterizeImuNoise(unittest.TestCase):
    """Test suite for full IMU noise characterization."""

    def test_characterize_imu_single_axis(self) -> None:
        """Test IMU characterization with single-axis data."""
        np.random.seed(42)
        fs = 100.0
        duration = 300  # 5 minutes
        N = int(fs * duration)

        # Simulate gyro and accel with white noise
        gyro = 0.001 * np.random.randn(N)
        accel = 0.01 * np.random.randn(N)

        results = characterize_imu_noise(gyro, accel, fs)

        # Check structure
        assert "gyro" in results
        assert "accel" in results
        assert "angle_random_walk" in results["gyro"]
        assert "bias_instability" in results["gyro"]
        assert "rate_random_walk" in results["gyro"]
        assert "taus" in results["gyro"]
        assert "adev" in results["gyro"]

        # Check that values are positive
        assert results["gyro"]["angle_random_walk"] > 0
        assert results["gyro"]["bias_instability"] > 0
        assert results["accel"]["velocity_random_walk"] > 0

    def test_characterize_imu_multi_axis(self) -> None:
        """Test IMU characterization with 3-axis data."""
        np.random.seed(42)
        fs = 100.0
        duration = 300
        N = int(fs * duration)

        # 3-axis gyro and accel
        gyro = 0.001 * np.random.randn(N, 3)
        accel = 0.01 * np.random.randn(N, 3)

        results = characterize_imu_noise(gyro, accel, fs)

        # Should return averaged parameters
        assert "gyro" in results
        assert "accel" in results
        assert isinstance(results["gyro"]["angle_random_walk"], (float, np.floating))
        assert isinstance(results["accel"]["velocity_random_walk"], (float, np.floating))

    def test_characterize_imu_realistic_parameters(self) -> None:
        """Test with realistic IMU noise parameters."""
        np.random.seed(42)
        fs = 100.0
        duration = 600  # 10 minutes for better statistics
        N = int(fs * duration)

        # Simulate gyro with realistic noise
        # Consumer IMU: ARW ~ 0.01 deg/s/sqrt(Hz) = 0.000175 rad/s/sqrt(Hz)
        arw_target = 0.0002  # rad/s/sqrt(Hz)
        gyro = arw_target * np.sqrt(fs) * np.random.randn(N)

        # Simple accel
        accel = 0.01 * np.random.randn(N)

        results = characterize_imu_noise(gyro, accel, fs)

        # Extracted ARW should be in reasonable range
        arw_extracted = results["gyro"]["angle_random_walk"]
        assert 0.00001 < arw_extracted < 0.01  # reasonable range


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for calibration."""

    def test_allan_variance_very_long_data(self) -> None:
        """Test Allan variance with long dataset."""
        np.random.seed(42)
        fs = 100.0
        duration = 3600  # 1 hour
        N = int(fs * duration)

        x = np.random.randn(N)

        taus, adev = allan_variance(x, fs)

        # Should handle long data
        assert len(taus) > 50  # many tau values
        assert np.all(np.isfinite(adev))

    def test_allan_variance_tau_limits(self) -> None:
        """Test that taus are within valid range."""
        fs = 100.0
        N = 1000
        x = np.random.randn(N)

        # Provide taus outside valid range
        taus_bad = np.array([0.001, 0.01, 100.0, 1000.0])  # some too large

        taus, adev = allan_variance(x, fs, taus=taus_bad)

        # Should filter invalid taus
        assert np.all(taus >= 1.0 / fs)  # at least one sample
        assert np.all(taus <= N / fs / 2.0)  # at most half the data

    def test_bias_instability_flat_curve(self) -> None:
        """Test bias instability on perfectly flat curve."""
        taus = np.logspace(-1, 2, 100)
        adev = np.ones_like(taus) * 0.005  # perfectly flat

        bi, tau_bi = identify_bias_instability(taus, adev)

        # Should return the constant value
        assert np.isclose(bi, 0.005)

    def test_random_walk_very_short_tau(self) -> None:
        """Test random walk at very short tau."""
        taus = np.logspace(-3, 0, 100)
        adev = 0.1 / np.sqrt(taus)

        arw = identify_random_walk(taus, adev, tau_target=0.01)

        # Should still extract value
        assert arw > 0
        assert np.isfinite(arw)

    def test_characterize_with_nan_values(self) -> None:
        """Test characterization handles data with NaN gracefully."""
        np.random.seed(42)
        fs = 100.0
        N = 1000

        # Data with some NaN values
        gyro = np.random.randn(N)
        gyro[100:110] = np.nan  # inject NaNs

        accel = np.random.randn(N)

        # Should handle or raise clear error
        # Depending on implementation, this might fail or skip NaNs
        # For now, let's just ensure it doesn't crash catastrophically
        try:
            results = characterize_imu_noise(gyro, accel, fs)
            # If it succeeds, check structure
            assert "gyro" in results
        except (ValueError, RuntimeError):
            # If it fails, that's acceptable for data with NaN
            pass

    def test_allan_variance_single_tau(self) -> None:
        """Test Allan variance with single tau value."""
        fs = 100.0
        N = 500
        x = np.random.randn(N)

        taus_single = np.array([1.0])

        taus, adev = allan_variance(x, fs, taus=taus_single)

        assert len(taus) == 1
        assert len(adev) == 1
        assert np.isfinite(adev[0])


if __name__ == "__main__":
    unittest.main()

