"""Unit tests for preprocessing utilities.

Tests scan averaging, normalization, and preprocessing pipelines for
fingerprint-based positioning (Chapter 5).

Author: Li-Ta Hsu
Date: December 2024
"""

import numpy as np
import pytest

from core.fingerprinting import (
    average_scans,
    compute_normalization_params,
    normalize_fingerprint,
    preprocess_query,
)


class TestAverageScans:
    """Test scan averaging functions."""

    def test_average_scans_mean_perfect(self):
        """Test mean averaging with identical scans."""
        scans = np.array([
            [-50, -60, -70],
            [-50, -60, -70],
            [-50, -60, -70],
        ])
        
        avg = average_scans(scans, method="mean")
        
        assert avg.shape == (3,)
        np.testing.assert_array_almost_equal(avg, [-50, -60, -70])

    def test_average_scans_mean_with_noise(self):
        """Test mean averaging reduces noise."""
        np.random.seed(42)
        # True values + Gaussian noise
        true_values = np.array([-50, -60, -70])
        noise = np.random.randn(5, 3) * 2.0  # 2 dBm std
        scans = true_values + noise
        
        avg = average_scans(scans, method="mean")
        
        # Averaged should be closer to true values on average
        # Check that average error is smaller than median single-scan error
        avg_error = np.linalg.norm(avg - true_values)
        scan_errors = [np.linalg.norm(scans[i] - true_values) for i in range(len(scans))]
        median_scan_error = np.median(scan_errors)
        
        # Average should be better than median single scan
        assert avg_error <= median_scan_error

    def test_average_scans_median_robust_to_outliers(self):
        """Test median averaging is robust to outliers."""
        scans = np.array([
            [-50, -60, -70],
            [-51, -59, -71],
            [-49, -61, -69],
            [-20, -65, -72],  # Outlier in AP1
            [-52, -58, -68],
        ])
        
        avg_mean = average_scans(scans, method="mean")
        avg_median = average_scans(scans, method="median")
        
        # Median should be less affected by outlier
        # True value is around -50, median should be closer
        assert abs(avg_median[0] - (-50)) < abs(avg_mean[0] - (-50))

    def test_average_scans_trimmed_mean(self):
        """Test trimmed mean averaging."""
        scans = np.array([
            [-50, -60, -70],
            [-51, -59, -71],
            [-49, -61, -69],
            [-20, -65, -72],  # Outlier
            [-52, -58, -68],
        ])
        
        avg = average_scans(scans, method="trimmed_mean", trim_percent=0.2)
        
        # Should trim 1 value from each end (20% of 5 = 1)
        # Result should be between mean and median
        assert avg.shape == (3,)

    def test_average_scans_with_nan(self):
        """Test averaging with missing values (NaN)."""
        scans = np.array([
            [-50, np.nan, -70],
            [-52, -58, -72],
            [-48, -62, np.nan],
            [-51, -60, -69],
        ])
        
        avg = average_scans(scans, method="mean")
        
        # AP1: mean of [-50, -52, -48, -51] = -50.25
        # AP2: mean of [-58, -62, -60] = -60.0 (ignoring NaN)
        # AP3: mean of [-70, -72, -69] = -70.33 (ignoring NaN)
        assert abs(avg[0] - (-50.25)) < 0.01
        assert abs(avg[1] - (-60.0)) < 0.01
        assert abs(avg[2] - (-70.33)) < 0.01

    def test_average_scans_all_nan_feature(self):
        """Test averaging when all scans for a feature are NaN."""
        scans = np.array([
            [-50, np.nan, -70],
            [-52, np.nan, -72],
            [-48, np.nan, -68],
        ])
        
        avg = average_scans(scans, method="mean")
        
        # AP2 should be NaN (all scans missing)
        assert np.isnan(avg[1])
        assert not np.isnan(avg[0])
        assert not np.isnan(avg[2])

    def test_average_scans_invalid_shape(self):
        """Test that 1D or 3D arrays raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D array"):
            average_scans(np.array([-50, -60, -70]))
        
        with pytest.raises(ValueError, match="must be 2D array"):
            average_scans(np.ones((2, 3, 4)))

    def test_average_scans_invalid_method(self):
        """Test that invalid method raises ValueError."""
        scans = np.array([[-50, -60], [-51, -59]])
        
        with pytest.raises(ValueError, match="Unknown method"):
            average_scans(scans, method="invalid")

    def test_average_scans_invalid_trim_percent(self):
        """Test that invalid trim_percent raises ValueError."""
        scans = np.array([[-50, -60], [-51, -59]])
        
        with pytest.raises(ValueError, match="trim_percent must be in"):
            average_scans(scans, method="trimmed_mean", trim_percent=0.6)


class TestNormalizeFingerprint:
    """Test fingerprint normalization functions."""

    def test_normalize_zscore_simple(self):
        """Test z-score normalization."""
        z = np.array([-50, -60, -70, -80])
        
        z_norm, params = normalize_fingerprint(z, method="zscore")
        
        # Mean should be ~0, std should be ~1
        assert abs(np.mean(z_norm)) < 1e-10
        assert abs(np.std(z_norm, ddof=1) - 1.0) < 1e-10
        
        # Check params
        assert params["method"] == "zscore"
        assert abs(params["mean"] - (-65.0)) < 0.01
        # Std for [-50, -60, -70, -80] with mean -65 is sqrt(((15^2 + 5^2 + 5^2 + 15^2) / 3))
        # = sqrt(500/3) = sqrt(166.67) ≈ 12.91
        assert abs(params["std"] - 12.91) < 0.1

    def test_normalize_zscore_with_reference(self):
        """Test z-score normalization with reference statistics."""
        z = np.array([-55, -65, -75, -85])
        ref_mean = -65.0
        ref_std = 11.18
        
        z_norm, params = normalize_fingerprint(
            z, method="zscore", ref_mean=ref_mean, ref_std=ref_std
        )
        
        # Should use provided statistics
        assert params["mean"] == ref_mean
        assert params["std"] == ref_std
        
        # Verify normalization: (z - mean) / std
        expected = (z - ref_mean) / ref_std
        np.testing.assert_array_almost_equal(z_norm, expected)

    def test_normalize_minmax_simple(self):
        """Test minmax normalization."""
        z = np.array([-50, -60, -70, -80])
        
        z_norm, params = normalize_fingerprint(z, method="minmax")
        
        # Should be in [0, 1] range
        assert np.min(z_norm) == 0.0
        assert np.max(z_norm) == 1.0
        
        # Check params
        assert params["method"] == "minmax"
        assert params["min"] == -80.0
        assert params["max"] == -50.0
        assert params["range"] == 30.0

    def test_normalize_minmax_with_reference(self):
        """Test minmax normalization with reference statistics."""
        z = np.array([-55, -65, -75])
        
        z_norm, params = normalize_fingerprint(
            z, method="minmax", ref_min=-80.0, ref_max=-50.0
        )
        
        # Verify: (z - min) / range
        expected = (z - (-80.0)) / 30.0
        np.testing.assert_array_almost_equal(z_norm, expected)

    def test_normalize_none(self):
        """Test no normalization (pass through)."""
        z = np.array([-50, -60, -70])
        
        z_norm, params = normalize_fingerprint(z, method="none")
        
        np.testing.assert_array_equal(z_norm, z)
        assert params["method"] == "none"

    def test_normalize_with_nan(self):
        """Test normalization with missing values (NaN)."""
        z = np.array([-50, np.nan, -70, -80])
        
        z_norm, params = normalize_fingerprint(z, method="zscore")
        
        # NaN should remain NaN
        assert np.isnan(z_norm[1])
        # Other values should be normalized
        assert not np.isnan(z_norm[0])
        assert not np.isnan(z_norm[2])

    def test_normalize_constant_values(self):
        """Test normalization with constant values (zero std/range)."""
        z = np.array([-60, -60, -60, -60])
        
        # Z-score should not crash (std=0 -> use std=1.0)
        z_norm, params = normalize_fingerprint(z, method="zscore")
        # All values should be 0 (mean is -60, std is 1.0 by default)
        np.testing.assert_array_almost_equal(z_norm, [0, 0, 0, 0])
        
        # Minmax should not crash (range=0 -> use range=1.0)
        z_norm_mm, params_mm = normalize_fingerprint(z, method="minmax")
        # All values should be 0
        np.testing.assert_array_almost_equal(z_norm_mm, [0, 0, 0, 0])

    def test_normalize_invalid_shape(self):
        """Test that 2D array raises ValueError."""
        with pytest.raises(ValueError, match="must be 1D array"):
            normalize_fingerprint(np.ones((3, 4)))

    def test_normalize_invalid_method(self):
        """Test that invalid method raises ValueError."""
        z = np.array([-50, -60, -70])
        
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_fingerprint(z, method="invalid")


class TestPreprocessQuery:
    """Test combined preprocessing pipeline."""

    def test_preprocess_multiple_scans_no_norm(self):
        """Test preprocessing with multiple scans, no normalization."""
        scans = np.array([
            [-50, -60, -70],
            [-52, -58, -72],
            [-48, -62, -68],
        ])
        
        z_prep, info = preprocess_query(scans, normalization_method="none")
        
        # Should average scans
        expected_avg = np.mean(scans, axis=0)
        np.testing.assert_array_almost_equal(z_prep, expected_avg)
        
        # Check info
        assert info["averaging"]["n_scans"] == 3
        assert info["normalization"]["method"] == "none"

    def test_preprocess_multiple_scans_with_zscore(self):
        """Test preprocessing with averaging + z-score normalization."""
        scans = np.array([
            [-50, -60, -70],
            [-52, -58, -72],
            [-48, -62, -68],
        ])
        
        z_prep, info = preprocess_query(
            scans,
            averaging_method="mean",
            normalization_method="zscore"
        )
        
        # Should first average, then normalize
        z_avg = np.mean(scans, axis=0)
        # Z-score should have mean ~0, std ~1
        assert abs(np.mean(z_prep)) < 1e-10
        assert abs(np.std(z_prep, ddof=1) - 1.0) < 1e-10

    def test_preprocess_single_scan_with_norm(self):
        """Test preprocessing with single scan (just normalization)."""
        single_scan = np.array([-50, -60, -70, -80])
        
        z_prep, info = preprocess_query(
            single_scan,
            normalization_method="zscore"
        )
        
        assert info["averaging"]["n_scans"] == 1
        assert info["averaging"]["method"] == "single_scan"
        assert info["normalization"]["method"] == "zscore"

    def test_preprocess_with_reference_stats(self):
        """Test preprocessing with reference normalization statistics."""
        scans = np.array([
            [-55, -65, -75],
            [-56, -64, -74],
        ])
        
        # Use pre-computed normalization params (per-feature)
        ref_mean = np.array([-60.0, -70.0, -80.0])
        ref_std = np.array([10.0, 10.0, 10.0])
        
        z_prep, info = preprocess_query(
            scans,
            averaging_method="mean",
            normalization_method="zscore",
            ref_mean=ref_mean,
            ref_std=ref_std
        )
        
        # Check that reference stats were used (should be arrays)
        np.testing.assert_array_equal(info["normalization"]["mean"], ref_mean)
        np.testing.assert_array_equal(info["normalization"]["std"], ref_std)

    def test_preprocess_trimmed_mean_averaging(self):
        """Test preprocessing with trimmed mean averaging."""
        scans = np.array([
            [-50, -60, -70],
            [-51, -59, -71],
            [-20, -65, -72],  # Outlier
            [-49, -61, -69],
            [-52, -58, -68],
        ])
        
        z_prep, info = preprocess_query(
            scans,
            averaging_method="trimmed_mean",
            trim_percent=0.2,
            normalization_method="none"
        )
        
        assert info["averaging"]["method"] == "trimmed_mean"


class TestComputeNormalizationParams:
    """Test normalization parameter computation from database."""

    def test_compute_zscore_params(self):
        """Test computing z-score parameters from fingerprints."""
        # Simulate database features (M=5 RPs, N=3 APs)
        fingerprints = np.array([
            [-50, -60, -70],
            [-55, -65, -75],
            [-45, -55, -65],
            [-60, -70, -80],
            [-50, -60, -70],
        ])
        
        params = compute_normalization_params(fingerprints, method="zscore")
        
        assert params["method"] == "zscore"
        assert params["mean"].shape == (3,)
        assert params["std"].shape == (3,)
        
        # Verify mean
        expected_mean = np.mean(fingerprints, axis=0)
        np.testing.assert_array_almost_equal(params["mean"], expected_mean)
        
        # Verify std
        expected_std = np.std(fingerprints, axis=0, ddof=1)
        np.testing.assert_array_almost_equal(params["std"], expected_std)

    def test_compute_minmax_params(self):
        """Test computing minmax parameters from fingerprints."""
        fingerprints = np.array([
            [-50, -60, -70],
            [-55, -65, -75],
            [-45, -55, -65],
            [-60, -70, -80],
        ])
        
        params = compute_normalization_params(fingerprints, method="minmax")
        
        assert params["method"] == "minmax"
        assert params["min"].shape == (3,)
        assert params["max"].shape == (3,)
        
        # Verify min/max
        np.testing.assert_array_equal(params["min"], [-60, -70, -80])
        np.testing.assert_array_equal(params["max"], [-45, -55, -65])

    def test_compute_params_with_nan(self):
        """Test computing parameters with missing values (NaN)."""
        fingerprints = np.array([
            [-50, np.nan, -70],
            [-55, -65, -75],
            [-45, -55, np.nan],
            [-60, -70, -80],
        ])
        
        params = compute_normalization_params(fingerprints, method="zscore")
        
        # Should compute using available (non-NaN) values
        # AP1: all values available
        # AP2: 3 values available (ignoring first NaN)
        # AP3: 3 values available (ignoring third NaN)
        assert not np.isnan(params["mean"]).any()
        assert not np.isnan(params["std"]).any()

    def test_compute_params_constant_feature(self):
        """Test computing parameters when feature is constant (zero variance)."""
        fingerprints = np.array([
            [-50, -60, -70],
            [-55, -60, -75],
            [-45, -60, -65],
        ])
        
        params = compute_normalization_params(fingerprints, method="zscore")
        
        # AP2 has constant value -60, std should be set to 1.0
        assert params["std"][1] == 1.0

    def test_compute_params_invalid_shape(self):
        """Test that 1D array raises ValueError."""
        with pytest.raises(ValueError, match="must be 2D array"):
            compute_normalization_params(np.array([-50, -60, -70]))

    def test_compute_params_invalid_method(self):
        """Test that invalid method raises ValueError."""
        fingerprints = np.array([[-50, -60], [-55, -65]])
        
        with pytest.raises(ValueError, match="Unknown method"):
            compute_normalization_params(fingerprints, method="invalid")


class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing workflow."""
        # Step 1: Collect multiple scans (simulated)
        np.random.seed(42)
        true_fingerprint = np.array([-50.0, -60.0, -70.0, -80.0])
        noise = np.random.randn(10, 4) * 3.0
        scans = true_fingerprint + noise
        
        # Step 2: Compute normalization params from database
        db_features = np.array([
            [-50.0, -60.0, -70.0, -80.0],
            [-55.0, -65.0, -75.0, -85.0],
            [-45.0, -55.0, -65.0, -75.0],
            [-60.0, -70.0, -80.0, -90.0],
        ])
        norm_params = compute_normalization_params(db_features, method="zscore")
        
        # Step 3: Preprocess query (note: ref_mean and ref_std are arrays, not scalars)
        # We need to pass them element-wise or compute scalar normalization
        # Since normalize_fingerprint expects scalars when ref_mean/std are provided,
        # let's use the computed mean/std as overall statistics
        z_avg = average_scans(scans, method="mean")
        z_preprocessed, norm_info = normalize_fingerprint(
            z_avg,
            method="zscore",
            ref_mean=norm_params["mean"][0],  # Use first AP's mean as reference
            ref_std=norm_params["std"][0]     # Use first AP's std as reference
        )
        
        # Verify pipeline executed
        assert z_preprocessed.shape == (4,)
        assert norm_info["method"] == "zscore"

    def test_preprocessing_reduces_noise(self):
        """Test that preprocessing reduces measurement noise."""
        np.random.seed(42)
        true_values = np.array([-50, -60, -70])
        
        # Single scan (noisy)
        single_scan = true_values + np.random.randn(3) * 5.0
        
        # Multiple scans (average reduces noise)
        multiple_scans = true_values + np.random.randn(20, 3) * 5.0
        
        # Process single scan
        z_single, _ = preprocess_query(single_scan, normalization_method="none")
        
        # Process multiple scans
        z_multi, _ = preprocess_query(multiple_scans, normalization_method="none")
        
        # Multi-scan average should be closer to true values
        error_single = np.linalg.norm(z_single - true_values)
        error_multi = np.linalg.norm(z_multi - true_values)
        
        assert error_multi < error_single

    def test_normalization_handles_device_offset(self):
        """Test that normalization mitigates device calibration offset."""
        # Reference fingerprint (device A)
        ref_fingerprint = np.array([-50, -60, -70, -80])
        
        # Query fingerprint (device B with +5 dBm offset)
        query_fingerprint = ref_fingerprint + 5.0
        
        # Without normalization, error is large
        error_raw = np.linalg.norm(query_fingerprint - ref_fingerprint)
        
        # With z-score normalization (removes offset)
        query_norm, _ = normalize_fingerprint(query_fingerprint, method="zscore")
        ref_norm, _ = normalize_fingerprint(ref_fingerprint, method="zscore")
        
        error_norm = np.linalg.norm(query_norm - ref_norm)
        
        # Normalized error should be smaller
        assert error_norm < error_raw

