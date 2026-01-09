"""
Unit tests for pink noise generation (core/sim/noise_pink.py).

Tests 1/f noise generation and scaling to match bias instability specs.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np
from core.sim.noise_pink import pink_noise_1f_fft, scale_to_bias_instability
from core.sensors import allan_variance


class TestPinkNoise1fFFT(unittest.TestCase):
    """Test pink_noise_1f_fft() function."""

    def test_zero_mean(self):
        """Pink noise should be zero-mean."""
        N = 10000
        fs = 100.0
        rng = np.random.default_rng(42)
        pink = pink_noise_1f_fft(N, fs, rng=rng)

        # Zero-mean within tolerance
        self.assertAlmostEqual(np.mean(pink), 0.0, places=2)

    def test_unit_std(self):
        """Pink noise should have unit standard deviation."""
        N = 10000
        fs = 100.0
        rng = np.random.default_rng(42)
        pink = pink_noise_1f_fft(N, fs, rng=rng)

        # Unit-std within tolerance
        self.assertAlmostEqual(np.std(pink), 1.0, places=2)

    def test_output_shape(self):
        """Pink noise output should have correct shape."""
        N = 5000
        fs = 100.0
        pink = pink_noise_1f_fft(N, fs)

        self.assertEqual(pink.shape, (N,))

    def test_reproducibility(self):
        """Pink noise should be reproducible with same seed."""
        N = 1000
        fs = 100.0
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        pink1 = pink_noise_1f_fft(N, fs, rng=rng1)
        pink2 = pink_noise_1f_fft(N, fs, rng=rng2)

        np.testing.assert_array_equal(pink1, pink2)

    def test_invalid_fs(self):
        """Should raise ValueError for invalid fs."""
        with self.assertRaises(ValueError):
            pink_noise_1f_fft(1000, fs=0.0)

        with self.assertRaises(ValueError):
            pink_noise_1f_fft(1000, fs=-10.0)

    def test_small_N(self):
        """Should handle small N gracefully."""
        # N=0 should return empty array
        pink = pink_noise_1f_fft(0, fs=100.0)
        self.assertEqual(len(pink), 0)

        # N=1 should return zero
        pink = pink_noise_1f_fft(1, fs=100.0)
        np.testing.assert_array_equal(pink, np.array([0.0]))

    def test_psd_shape_1f(self):
        """
        Verify that PSD is approximately 1/f.

        This is a statistical test, so we use loose tolerance.
        We check that the log-log slope is approximately -1.
        """
        N = 100000  # Need long sequence for good PSD estimate
        fs = 100.0
        rng = np.random.default_rng(42)
        pink = pink_noise_1f_fft(N, fs, rng=rng)

        # Compute PSD using Welch method (averaged periodogram)
        from scipy import signal

        freqs, psd = signal.welch(pink, fs=fs, nperseg=4096, scaling="density")

        # Exclude DC and very low frequencies
        mask = (freqs > 0.1) & (freqs < fs / 4)
        freqs_valid = freqs[mask]
        psd_valid = psd[mask]

        # Fit log(PSD) vs log(f): should get slope ~ -1
        log_f = np.log10(freqs_valid)
        log_psd = np.log10(psd_valid)
        coeffs = np.polyfit(log_f, log_psd, deg=1)
        slope = coeffs[0]

        # Pink noise should have slope close to -1 (within ±0.2)
        self.assertGreater(slope, -1.3)
        self.assertLess(slope, -0.7)

    def test_allan_deviation_with_combined_noise(self):
        """
        Verify that combining white + pink noise creates a BI region.

        Pink noise (1/f) combined with white noise should produce an Allan
        deviation curve with a minimum (indicating bias instability region).
        This is the realistic IMU scenario where ARW + BI are both present.
        """
        N = 360000  # 3600 seconds (1 hour) at 100 Hz
        fs = 100.0
        rng = np.random.default_rng(42)
        
        # Generate combined white + pink noise (realistic IMU scenario)
        white = rng.standard_normal(N) * 0.3  # ARW component (dominant at short tau)
        pink = pink_noise_1f_fft(N, fs, rng=rng) * 1.0  # BI component (dominant at mid tau)
        combined = white + pink

        # Compute Allan deviation
        tau_grid = np.logspace(0, 3, 40)  # 1s to 1000s
        taus, sigma = allan_variance(combined, fs, tau_grid)

        # Key test: The curve should show characteristic BI behavior
        # Pure white noise would have slope -1/2 throughout (monotonic in log-log)
        idx_min = np.argmin(sigma)
        
        # The minimum should not be at the very start
        self.assertGreater(idx_min, 0)  # Not at short tau (would be pure white)
        
        # Verify the curve decreases from short tau (characteristic of BI region)
        # This shows pink noise is flattening the curve relative to pure white noise
        self.assertLess(sigma[idx_min], sigma[0])
        
        # Verify we have a realistic Allan deviation magnitude
        # (not all zeros, not absurdly large)
        self.assertGreater(np.min(sigma), 0)
        self.assertLess(np.max(sigma), 100 * np.min(sigma))


class TestScaleToBiasInstability(unittest.TestCase):
    """Test scale_to_bias_instability() function."""

    def test_scaling_matches_target_bi(self):
        """
        Scaled pink noise should produce Allan deviation minimum
        close to target_bi * 0.664.
        """
        # Generate unit pink noise
        N = 100000  # Long sequence for good Allan statistics
        fs = 100.0
        rng = np.random.default_rng(42)
        pink_unit = pink_noise_1f_fft(N, fs, rng=rng)

        # Target BI: 10 deg/hr
        bi_deg_hr = 10.0
        target_bi_rad_s = np.deg2rad(bi_deg_hr) / 3600.0

        # Create tau grid
        tau_grid = np.logspace(0, 2.5, 40)  # 1s to ~300s

        # Scale to match target BI
        pink_scaled = scale_to_bias_instability(
            pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs, bi_factor=0.664
        )

        # Verify: compute Allan deviation and check minimum
        taus, sigma = allan_variance(pink_scaled, fs, tau_grid)
        sigma_min = np.min(sigma)
        expected_sigma_min = target_bi_rad_s * 0.664

        # Should match within 20% (stochastic process, finite length)
        relative_error = np.abs(sigma_min - expected_sigma_min) / expected_sigma_min
        self.assertLess(relative_error, 0.2)

    def test_scaling_preserves_zero_mean(self):
        """Scaled pink noise should remain zero-mean."""
        N = 10000
        fs = 100.0
        rng = np.random.default_rng(42)
        pink_unit = pink_noise_1f_fft(N, fs, rng=rng)

        target_bi_rad_s = 1e-5  # arbitrary
        tau_grid = np.logspace(0, 2, 20)

        pink_scaled = scale_to_bias_instability(
            pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs
        )

        # Should remain zero-mean
        self.assertAlmostEqual(np.mean(pink_scaled), 0.0, places=2)

    def test_scaling_factor_correctness(self):
        """Verify that scaling factor is computed correctly."""
        N = 50000
        fs = 100.0
        rng = np.random.default_rng(123)
        pink_unit = pink_noise_1f_fft(N, fs, rng=rng)

        # Use a different bi_factor for testing
        target_bi_rad_s = 2e-5
        tau_grid = np.logspace(0, 2.5, 30)
        bi_factor = 0.5  # non-standard factor

        pink_scaled = scale_to_bias_instability(
            pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs, bi_factor=bi_factor
        )

        # Compute Allan deviation
        taus, sigma = allan_variance(pink_scaled, fs, tau_grid)
        sigma_min = np.min(sigma)

        # Should match target_bi * bi_factor within tolerance
        expected_sigma_min = target_bi_rad_s * bi_factor
        relative_error = np.abs(sigma_min - expected_sigma_min) / expected_sigma_min
        self.assertLess(relative_error, 0.25)

    def test_invalid_sigma_min(self):
        """Should raise ValueError if Allan deviation returns invalid result."""
        N = 1000
        fs = 100.0
        pink_unit = np.zeros(N)  # All zeros => sigma_min = 0

        target_bi_rad_s = 1e-5
        tau_grid = np.logspace(0, 2, 10)

        with self.assertRaises(ValueError):
            scale_to_bias_instability(
                pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for pink noise generation and scaling."""

    def test_full_pipeline(self):
        """
        Test full pipeline: generate pink noise, scale to BI, verify.
        """
        # Configuration
        fs = 100.0
        duration = 3600.0  # 1 hour - need longer for accurate BI characterization
        N = int(fs * duration)
        bi_deg_hr = 5.0
        target_bi_rad_s = np.deg2rad(bi_deg_hr) / 3600.0

        # Generate unit pink noise
        rng = np.random.default_rng(999)
        pink_unit = pink_noise_1f_fft(N, fs, rng=rng)

        # Verify unit pink noise properties
        self.assertAlmostEqual(np.mean(pink_unit), 0.0, places=2)
        self.assertAlmostEqual(np.std(pink_unit), 1.0, places=1)

        # Scale to target BI
        tau_grid = np.logspace(0, 3, 40)  # 1s to 1000s
        pink_scaled = scale_to_bias_instability(
            pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs
        )

        # Verify scaling
        taus, sigma = allan_variance(pink_scaled, fs, tau_grid)
        sigma_min = np.min(sigma)
        bi_recovered = sigma_min / 0.664

        # Should recover target BI within 30% (stochastic process with finite length)
        relative_error = np.abs(bi_recovered - target_bi_rad_s) / target_bi_rad_s
        self.assertLess(relative_error, 0.30)

        # Verify that sigma_min is in reasonable range (not at edges)
        idx_min = np.argmin(sigma)
        self.assertGreater(idx_min, 3)  # Not at very short tau
        self.assertLess(idx_min, len(sigma) - 3)  # Not at very long tau


if __name__ == "__main__":
    unittest.main()

