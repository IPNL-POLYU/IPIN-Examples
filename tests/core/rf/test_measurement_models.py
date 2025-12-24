"""
Unit tests for RF measurement models.

Tests TOA, TDOA, AOA, and RSS measurement functions.
"""

import numpy as np
import pytest

from core.rf.measurement_models import (
    SPEED_OF_LIGHT,
    aoa_angle_vector,
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    aoa_sin_elevation,
    aoa_tan_azimuth,
    clock_bias_meters_to_seconds,
    clock_bias_seconds_to_meters,
    range_to_rtt,
    rss_fading_to_distance_error,
    rss_pathloss,
    rss_to_distance,
    rtt_to_range,
    simulate_rss_measurement,
    simulate_rtt_measurement,
    tdoa_measurement_vector,
    tdoa_range_difference,
    toa_range,
    two_way_toa_range,
)


class TestTOA:
    """Test TOA measurement models."""

    def test_toa_range_basic(self):
        """Test basic TOA range calculation."""
        tx_pos = np.array([0.0, 0.0, 0.0])
        rx_pos = np.array([3.0, 4.0, 0.0])

        range_m = toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, 5.0), f"Expected 5.0, got {range_m}"

    def test_toa_range_with_clock_bias(self):
        """Test TOA range with clock bias in SECONDS."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([3.0, 4.0])
        clock_bias_s = 1e-9  # 1 nanosecond

        range_m = toa_range(tx_pos, rx_pos, clock_bias_s=clock_bias_s)

        expected = 5.0 + SPEED_OF_LIGHT * clock_bias_s
        assert np.isclose(range_m, expected)

    def test_toa_range_2d(self):
        """Test TOA in 2D."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([1.0, 1.0])

        range_m = toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, np.sqrt(2))

    def test_toa_range_clock_bias_positive(self):
        """Test that positive clock bias increases measured range."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([3.0, 4.0])  # 5m distance
        clock_bias_s = 10e-9  # 10 ns

        range_biased = toa_range(tx_pos, rx_pos, clock_bias_s=clock_bias_s)
        range_unbiased = toa_range(tx_pos, rx_pos)

        # Positive bias should increase range
        assert range_biased > range_unbiased
        # Difference should be c * Δt
        assert np.isclose(range_biased - range_unbiased, SPEED_OF_LIGHT * clock_bias_s)

    def test_toa_range_clock_bias_negative(self):
        """Test that negative clock bias decreases measured range."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([3.0, 4.0])  # 5m distance
        clock_bias_s = -5e-9  # -5 ns

        range_biased = toa_range(tx_pos, rx_pos, clock_bias_s=clock_bias_s)
        range_unbiased = toa_range(tx_pos, rx_pos)

        # Negative bias should decrease range
        assert range_biased < range_unbiased

    def test_two_way_toa(self):
        """Test two-way TOA (RTT) geometric distance."""
        tx_pos = np.array([0.0, 0.0, 0.0])
        rx_pos = np.array([10.0, 0.0, 0.0])

        range_m = two_way_toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, 10.0)


class TestClockBiasConversion:
    """Test clock bias unit conversion utilities."""

    def test_seconds_to_meters_basic(self):
        """Test converting nanoseconds to meters."""
        # 10 nanoseconds should be ~3.0 meters
        bias_s = 10e-9
        bias_m = clock_bias_seconds_to_meters(bias_s)

        expected_m = SPEED_OF_LIGHT * bias_s  # ~2.998 m
        assert np.isclose(bias_m, expected_m)
        assert np.isclose(bias_m, 2.998, rtol=1e-3)

    def test_meters_to_seconds_basic(self):
        """Test converting meters to nanoseconds."""
        # 3 meters should be ~10 nanoseconds
        bias_m = 3.0
        bias_s = clock_bias_meters_to_seconds(bias_m)

        expected_s = bias_m / SPEED_OF_LIGHT
        assert np.isclose(bias_s, expected_s)
        assert np.isclose(bias_s * 1e9, 10.0, rtol=1e-3)  # ~10 ns

    def test_roundtrip_conversion(self):
        """Test that s->m->s and m->s->m are identity."""
        # Seconds -> meters -> seconds
        bias_s_original = 15e-9  # 15 ns
        bias_m = clock_bias_seconds_to_meters(bias_s_original)
        bias_s_recovered = clock_bias_meters_to_seconds(bias_m)
        assert np.isclose(bias_s_original, bias_s_recovered)

        # Meters -> seconds -> meters
        bias_m_original = 10.0  # 10 m
        bias_s = clock_bias_meters_to_seconds(bias_m_original)
        bias_m_recovered = clock_bias_seconds_to_meters(bias_s)
        assert np.isclose(bias_m_original, bias_m_recovered)

    def test_zero_bias(self):
        """Test that zero converts to zero."""
        assert clock_bias_seconds_to_meters(0.0) == 0.0
        assert clock_bias_meters_to_seconds(0.0) == 0.0

    def test_negative_bias(self):
        """Test that negative values are handled correctly."""
        bias_s = -5e-9  # -5 ns
        bias_m = clock_bias_seconds_to_meters(bias_s)
        assert bias_m < 0  # Should be negative

        bias_m = -2.0  # -2 m
        bias_s = clock_bias_meters_to_seconds(bias_m)
        assert bias_s < 0  # Should be negative

    def test_scale_reference_values(self):
        """Test well-known scale relationships."""
        # 1 ns -> 0.3 m (approximately)
        bias_m = clock_bias_seconds_to_meters(1e-9)
        assert 0.29 < bias_m < 0.31, f"1 ns should be ~0.3 m, got {bias_m}"

        # 0.3 m -> 1 ns (approximately)
        bias_s = clock_bias_meters_to_seconds(0.299792458)
        assert np.isclose(bias_s * 1e9, 1.0, rtol=1e-6), f"0.3 m should be ~1 ns"


class TestRTTModel:
    """Test RTT timing-based measurement model (Eqs. 4.6-4.9)."""

    def test_rtt_to_range_basic(self):
        """Test basic RTT to range conversion (Eq. 4.7)."""
        # RTT for 15m distance: 2 * 15m / c = 100.07 ns
        distance = 15.0
        rtt = 2.0 * distance / SPEED_OF_LIGHT

        range_est = rtt_to_range(rtt)

        assert np.isclose(range_est, distance, atol=1e-6)

    def test_rtt_to_range_with_processing_time(self):
        """Test RTT to range with processing time (Eq. 4.7)."""
        distance = 15.0
        processing_time = 50e-9  # 50 ns

        # RTT includes processing time
        rtt = 2.0 * distance / SPEED_OF_LIGHT + processing_time

        # Without knowing processing time, estimate is wrong
        range_wrong = rtt_to_range(rtt)
        assert range_wrong > distance

        # With correct processing time, estimate is correct
        range_correct = rtt_to_range(rtt, processing_time=processing_time)
        assert np.isclose(range_correct, distance, atol=1e-6)

    def test_rtt_to_range_with_clock_drift(self):
        """Test RTT to range with clock drift (Eq. 4.8)."""
        distance = 15.0
        clock_drift = 10e-9  # 10 ns drift

        # RTT includes drift
        rtt = 2.0 * distance / SPEED_OF_LIGHT + clock_drift

        # With drift correction
        range_est = rtt_to_range(rtt, clock_drift=clock_drift)
        assert np.isclose(range_est, distance, atol=1e-6)

    def test_range_to_rtt_inverse(self):
        """Test range_to_rtt is inverse of rtt_to_range."""
        distance = 20.0
        processing_time = 30e-9

        # Convert to RTT and back
        rtt = range_to_rtt(distance, processing_time=processing_time)
        range_back = rtt_to_range(rtt, processing_time=processing_time)

        assert np.isclose(range_back, distance, atol=1e-6)

    def test_range_to_rtt_values(self):
        """Test range_to_rtt produces expected values."""
        distance = 15.0  # 15 meters

        # Expected RTT: 2 * 15 / 299792458 = 100.07 ns
        rtt = range_to_rtt(distance)
        expected_rtt = 2.0 * 15.0 / SPEED_OF_LIGHT

        assert np.isclose(rtt, expected_rtt)

    def test_simulate_rtt_no_noise(self):
        """Test simulate_rtt_measurement without noise."""
        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([15.0, 0.0, 0.0])

        rtt, info = simulate_rtt_measurement(anchor, agent)

        # True range should be 15m
        assert np.isclose(info['true_range'], 15.0)

        # RTT should be 2 * 15 / c
        expected_rtt = 2.0 * 15.0 / SPEED_OF_LIGHT
        assert np.isclose(rtt, expected_rtt)

        # Range estimate should match true range
        assert np.isclose(info['range_estimate'], 15.0, atol=1e-6)

    def test_simulate_rtt_with_processing_time(self):
        """Test simulate_rtt_measurement with processing time."""
        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([15.0, 0.0, 0.0])
        processing_time = 50e-9

        rtt, info = simulate_rtt_measurement(
            anchor, agent, processing_time=processing_time
        )

        # RTT should include processing time
        expected_rtt = 2.0 * 15.0 / SPEED_OF_LIGHT + processing_time
        assert np.isclose(rtt, expected_rtt)

        # Range estimate should still be correct
        assert np.isclose(info['range_estimate'], 15.0, atol=1e-6)

    def test_simulate_rtt_with_noise(self):
        """Test simulate_rtt_measurement with noise (Eq. 4.9)."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([15.0, 0.0, 0.0])

        # Simulate multiple measurements
        errors = []
        for _ in range(100):
            rtt, info = simulate_rtt_measurement(
                anchor, agent,
                processing_time=50e-9,
                processing_time_std=5e-9,
                clock_drift_std=2e-9,
            )
            errors.append(info['range_estimate'] - 15.0)

        errors = np.array(errors)

        # Errors should have non-zero std (noise present)
        assert np.std(errors) > 0.1  # At least 10cm std

        # Mean error should be reasonable (< 1m)
        assert np.abs(np.mean(errors)) < 1.0

    def test_timing_error_to_range_error(self):
        """Test that 1ns timing error ≈ 0.15m range error."""
        distance = 15.0
        timing_error = 1e-9  # 1 nanosecond

        rtt_true = 2.0 * distance / SPEED_OF_LIGHT
        rtt_with_error = rtt_true + timing_error

        range_error = rtt_to_range(rtt_with_error) - distance

        # 1 ns timing error -> 0.15m range error (c * 1ns / 2)
        expected_error = SPEED_OF_LIGHT * timing_error / 2
        assert np.isclose(range_error, expected_error)
        assert np.isclose(expected_error, 0.1499, atol=0.001)


class TestRSS:
    """Test RSS measurement models (Book Eqs. 4.10-4.13)."""

    def test_rss_pathloss_eq410(self):
        """Test RSS path-loss forward model (Eq. 4.10)."""
        # Eq. 4.10: p_R = p_ref - 10*eta*log10(d/d_ref)
        p_ref_dbm = -40.0  # Reference RSS at 1m
        distance = 10.0  # meters
        path_loss_exp = 2.5  # Indoor

        rss = rss_pathloss(p_ref_dbm, distance, path_loss_exp)

        # Expected: -40 - 10*2.5*log10(10/1) = -40 - 25 = -65 dBm
        expected_rss = -40.0 - 10 * 2.5 * np.log10(10.0)
        assert np.isclose(rss, expected_rss, atol=0.01)
        assert np.isclose(rss, -65.0, atol=0.01)

    def test_rss_pathloss_free_space(self):
        """Test RSS path-loss model in free space (eta=2)."""
        p_ref_dbm = 0.0  # dBm at 1m
        distance = 10.0  # meters
        path_loss_exp = 2.0  # Free space

        rss = rss_pathloss(p_ref_dbm, distance, path_loss_exp)

        # Free space: PL = 10*2*log10(10/1) = 20 dB
        expected_rss = 0.0 - 20.0  # -20 dBm
        assert np.isclose(rss, expected_rss, atol=0.1)

    def test_rss_to_distance_eq411(self):
        """Test RSS to distance inversion (Eq. 4.11)."""
        # Eq. 4.11: d = d_ref * 10^((p_ref - p_R) / (10*eta))
        p_ref_dbm = -40.0
        rss_dbm = -65.0
        path_loss_exp = 2.5

        distance = rss_to_distance(rss_dbm, p_ref_dbm, path_loss_exp)

        # Expected: 1 * 10^((-40 - (-65)) / (10*2.5)) = 10^(25/25) = 10 m
        assert np.isclose(distance, 10.0, atol=0.01)

    def test_rss_to_distance_roundtrip(self):
        """Test RSS to distance round-trip conversion."""
        p_ref_dbm = -40.0
        distance_true = 15.0
        path_loss_exp = 2.5

        # Convert distance to RSS (Eq. 4.10)
        rss = rss_pathloss(p_ref_dbm, distance_true, path_loss_exp)

        # Convert RSS back to distance (Eq. 4.11)
        distance_est = rss_to_distance(rss, p_ref_dbm, path_loss_exp)

        assert np.isclose(distance_est, distance_true, rtol=1e-3)

    def test_rss_invalid_distance(self):
        """Test RSS with invalid distance."""
        with pytest.raises(ValueError, match="Distance must be positive"):
            rss_pathloss(-40.0, 0.0, 2.5)


class TestRSSFading:
    """Test RSS fading model (Book Eqs. 4.12-4.13)."""

    def test_simulate_rss_no_fading(self):
        """Test simulate_rss_measurement without fading."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        rss, info = simulate_rss_measurement(
            anchor, agent, p_ref_dbm=-40.0, path_loss_exp=2.5
        )

        # True distance should be 10m
        assert np.isclose(info['true_distance'], 10.0)

        # RSS should match forward model
        expected_rss = -40.0 - 10 * 2.5 * np.log10(10.0)
        assert np.isclose(rss, expected_rss, atol=0.01)
        assert np.isclose(info['rss_true'], expected_rss, atol=0.01)

        # No fading
        assert info['omega_long_db'] == 0.0
        assert info['omega_short_db'] == 0.0

        # Distance estimate should match true distance
        assert np.isclose(info['distance_estimate'], 10.0, atol=0.01)

    def test_simulate_rss_with_long_term_fading(self):
        """Test simulate_rss_measurement with long-term fading (Eq. 4.12)."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        # Simulate with 6 dB long-term fading std
        rss, info = simulate_rss_measurement(
            anchor, agent,
            p_ref_dbm=-40.0,
            path_loss_exp=2.5,
            sigma_long_db=6.0,
        )

        # RSS should differ from true RSS by omega_long
        assert np.isclose(
            rss, info['rss_true'] + info['omega_long_db'], atol=1e-6
        )

        # omega_long should be non-zero
        assert info['omega_long_db'] != 0.0

    def test_simulate_rss_short_term_gaussian_averaging(self):
        """Test that Gaussian short-term fading is reduced by averaging."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        # No averaging
        errors_no_avg = []
        for _ in range(100):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_short_linear=4.0,  # Interpreted as dB std for gaussian_db
                n_samples_avg=1,
                short_fading_model="gaussian_db",
            )
            errors_no_avg.append(info['omega_short_db'])

        # With 10-sample averaging
        np.random.seed(42)
        errors_with_avg = []
        for _ in range(100):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_short_linear=4.0,
                n_samples_avg=10,
                short_fading_model="gaussian_db",
            )
            errors_with_avg.append(info['omega_short_db'])

        # Averaging should reduce std by sqrt(10) ≈ 3.16
        std_no_avg = np.std(errors_no_avg)
        std_with_avg = np.std(errors_with_avg)

        # With averaging, std should be ~3x smaller
        assert std_with_avg < std_no_avg / 2.0

    def test_simulate_rss_rayleigh_fading_statistics(self):
        """Test Rayleigh fading produces correct statistics."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        # Rayleigh scale for normalized mean power: σ = 1/sqrt(2) ≈ 0.707
        sigma_rayleigh = 1.0 / np.sqrt(2)

        fading_values_db = []
        for _ in range(1000):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_short_linear=sigma_rayleigh,
                n_samples_avg=1,
                short_fading_model="rayleigh",
            )
            fading_values_db.append(info['omega_short_db'])

        fading_values_db = np.array(fading_values_db)

        # For Rayleigh fading, the power envelope P = A² follows exponential.
        # When converted to dB relative to mean power, the expected mean
        # is -10*log10(e)*γ ≈ -2.51 dB (Euler-Mascheroni constant effect)
        # Our normalization divides by expected_mean_power, so the mean
        # of omega_short_db should be close to this value.
        # Allow ±1 dB tolerance around the theoretical value
        theoretical_mean_db = -10 * np.log10(np.e) * 0.5772  # ≈ -2.5 dB
        assert np.abs(np.mean(fading_values_db) - theoretical_mean_db) < 1.5

        # Rayleigh fading has wide spread (std ~ 5-6 dB for power in dB)
        assert np.std(fading_values_db) > 4.0  # Significant spread
        assert np.std(fading_values_db) < 8.0  # But not unreasonably large

    def test_simulate_rss_rayleigh_averaging_reduces_variance(self):
        """Test that averaging Rayleigh samples in power domain reduces variance."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])
        sigma_rayleigh = 0.707  # Normalized mean power

        # No averaging
        fading_no_avg = []
        for _ in range(200):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_short_linear=sigma_rayleigh,
                n_samples_avg=1,
                short_fading_model="rayleigh",
            )
            fading_no_avg.append(info['omega_short_db'])

        # With 10-sample averaging
        np.random.seed(42)
        fading_with_avg = []
        for _ in range(200):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_short_linear=sigma_rayleigh,
                n_samples_avg=10,
                short_fading_model="rayleigh",
            )
            fading_with_avg.append(info['omega_short_db'])

        # Averaging should significantly reduce variance
        std_no_avg = np.std(fading_no_avg)
        std_with_avg = np.std(fading_with_avg)

        # With 10 samples averaged in power domain, variance should be reduced
        assert std_with_avg < std_no_avg / 2.0

    def test_simulate_rss_fading_model_none(self):
        """Test that short_fading_model='none' disables short-term fading."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        rss, info = simulate_rss_measurement(
            anchor, agent,
            p_ref_dbm=-40.0,
            path_loss_exp=2.5,
            sigma_short_linear=1.0,  # Non-zero but model is 'none'
            short_fading_model="none",
        )

        # No short-term fading despite non-zero sigma
        assert info['omega_short_db'] == 0.0
        assert info['short_fading_model'] == "none"

    def test_simulate_rss_invalid_fading_model(self):
        """Test that invalid fading model raises ValueError."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        with pytest.raises(ValueError, match="short_fading_model must be one of"):
            simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                short_fading_model="invalid_model",
            )

    def test_simulate_rss_returns_short_fading_model_in_info(self):
        """Test that info dict contains the short_fading_model used."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        # Default model
        rss, info = simulate_rss_measurement(
            anchor, agent, p_ref_dbm=-40.0
        )
        assert info['short_fading_model'] == "rayleigh"

        # Explicit Gaussian
        rss, info = simulate_rss_measurement(
            anchor, agent, p_ref_dbm=-40.0, short_fading_model="gaussian_db"
        )
        assert info['short_fading_model'] == "gaussian_db"

    def test_rss_fading_to_distance_error_eq413(self):
        """Test fading to distance error conversion (Eq. 4.13)."""
        from core.rf.measurement_models import rss_fading_to_distance_error

        # Eq. 4.13: d̃/d = 10^(-omega / (10*eta))
        path_loss_exp = 2.5

        # +6 dB fading (stronger signal) -> underestimate distance
        factor_pos = rss_fading_to_distance_error(6.0, path_loss_exp)
        # 10^(-6/(10*2.5)) = 10^(-0.24) ≈ 0.575
        expected_pos = 10 ** (-6.0 / 25.0)
        assert np.isclose(factor_pos, expected_pos, atol=0.01)
        assert factor_pos < 1.0  # Underestimate

        # -6 dB fading (weaker signal) -> overestimate distance
        factor_neg = rss_fading_to_distance_error(-6.0, path_loss_exp)
        expected_neg = 10 ** (6.0 / 25.0)
        assert np.isclose(factor_neg, expected_neg, atol=0.01)
        assert factor_neg > 1.0  # Overestimate

        # 0 dB fading -> no error
        factor_zero = rss_fading_to_distance_error(0.0, path_loss_exp)
        assert np.isclose(factor_zero, 1.0)

    def test_multiplicative_distance_error_eq413(self):
        """Test that distance error is multiplicative per Eq. 4.13."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        rss, info = simulate_rss_measurement(
            anchor, agent,
            p_ref_dbm=-40.0,
            path_loss_exp=2.5,
            sigma_long_db=6.0,
        )

        # Check distance_error_factor matches Eq. 4.13
        total_fading = info['omega_long_db'] + info['omega_short_db']
        expected_factor = 10 ** (-total_fading / 25.0)
        assert np.isclose(info['distance_error_factor'], expected_factor, atol=1e-6)

        # Check actual distance estimate follows multiplicative error
        # d̃ ≈ d * distance_error_factor
        expected_distance = info['true_distance'] * info['distance_error_factor']
        assert np.isclose(info['distance_estimate'], expected_distance, rtol=1e-3)

    def test_simulate_rss_monte_carlo(self):
        """Test RSS simulation produces expected statistics."""
        np.random.seed(42)

        anchor = np.array([0.0, 0.0])
        agent = np.array([10.0, 0.0])

        # Run Monte Carlo simulation
        distance_errors = []
        for _ in range(200):
            rss, info = simulate_rss_measurement(
                anchor, agent,
                p_ref_dbm=-40.0,
                path_loss_exp=2.5,
                sigma_long_db=6.0,
            )
            relative_error = (info['distance_estimate'] - 10.0) / 10.0
            distance_errors.append(relative_error)

        distance_errors = np.array(distance_errors)

        # Errors should have significant spread due to 6 dB fading
        assert np.std(distance_errors) > 0.2  # At least 20% relative std


class TestTDOA:
    """Test TDOA measurement models."""

    def test_tdoa_range_difference_basic(self):
        """Test basic TDOA range difference."""
        anchor_1 = np.array([0.0, 0.0])
        anchor_2 = np.array([10.0, 0.0])
        agent = np.array([5.0, 5.0])

        rd = tdoa_range_difference(anchor_1, anchor_2, agent)

        # Distance from agent to anchor_1: sqrt(25 + 25) = 7.071
        # Distance from agent to anchor_2: sqrt(25 + 25) = 7.071
        # Range difference: 0.0
        assert np.isclose(rd, 0.0, atol=1e-3)

    def test_tdoa_range_difference_asymmetric(self):
        """Test TDOA with asymmetric position."""
        anchor_1 = np.array([0.0, 0.0])
        anchor_2 = np.array([10.0, 0.0])
        agent = np.array([2.0, 0.0])

        rd = tdoa_range_difference(anchor_1, anchor_2, agent)

        # Distance to anchor_1: 2.0
        # Distance to anchor_2: 8.0
        # Range difference: 2.0 - 8.0 = -6.0
        assert np.isclose(rd, -6.0)

    def test_tdoa_measurement_vector(self):
        """Test TDOA measurement vector generation."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        agent = np.array([5, 5])

        tdoa = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)

        # Should return (N-1) measurements
        assert tdoa.shape == (3,)

    def test_tdoa_measurement_vector_different_reference(self):
        """Test TDOA with different reference anchor."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        # Use asymmetric position to avoid all-zero measurements
        agent = np.array([3, 7])

        tdoa_ref0 = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)
        tdoa_ref1 = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=1)

        # Different reference should give different measurements
        assert not np.allclose(tdoa_ref0, tdoa_ref1)

    def test_tdoa_invalid_reference(self):
        """Test TDOA with invalid reference index."""
        anchors = np.array([[0, 0], [10, 0], [10, 10]])
        agent = np.array([5, 5])

        with pytest.raises(ValueError, match="reference_anchor_idx"):
            tdoa_measurement_vector(anchors, agent, reference_anchor_idx=5)


class TestAOA:
    """Test AOA measurement models with ENU convention (book Eqs. 4.63-4.65)."""

    def test_aoa_azimuth_north(self):
        """Test azimuth for anchor directly North of agent (ψ = 0°)."""
        anchor = np.array([0.0, 10.0])  # North of agent
        agent = np.array([0.0, 0.0])

        azimuth = aoa_azimuth(anchor, agent)

        # Anchor is directly North: ψ = atan2(0, 10) = 0
        assert np.isclose(azimuth, 0.0, atol=1e-6)

    def test_aoa_azimuth_east(self):
        """Test azimuth for anchor directly East of agent (ψ = 90°)."""
        anchor = np.array([10.0, 0.0])  # East of agent
        agent = np.array([0.0, 0.0])

        azimuth = aoa_azimuth(anchor, agent)

        # Anchor is directly East: ψ = atan2(10, 0) = π/2
        assert np.isclose(azimuth, np.pi / 2, atol=1e-6)

    def test_aoa_azimuth_south(self):
        """Test azimuth for anchor directly South of agent (ψ = ±180°)."""
        anchor = np.array([0.0, -10.0])  # South of agent
        agent = np.array([0.0, 0.0])

        azimuth = aoa_azimuth(anchor, agent)

        # Anchor is directly South: ψ = atan2(0, -10) = π or -π
        assert np.isclose(np.abs(azimuth), np.pi, atol=1e-6)

    def test_aoa_azimuth_west(self):
        """Test azimuth for anchor directly West of agent (ψ = -90°)."""
        anchor = np.array([-10.0, 0.0])  # West of agent
        agent = np.array([0.0, 0.0])

        azimuth = aoa_azimuth(anchor, agent)

        # Anchor is directly West: ψ = atan2(-10, 0) = -π/2
        assert np.isclose(azimuth, -np.pi / 2, atol=1e-6)

    def test_aoa_azimuth_northeast_45deg(self):
        """Test azimuth for anchor at 45° (Northeast)."""
        anchor = np.array([10.0, 10.0])  # Northeast
        agent = np.array([0.0, 0.0])

        azimuth = aoa_azimuth(anchor, agent)

        # Anchor is Northeast: ψ = atan2(10, 10) = π/4
        assert np.isclose(azimuth, np.pi / 4, atol=1e-6)

    def test_aoa_tan_azimuth_basic(self):
        """Test tan(ψ) calculation per Eq. 4.64."""
        anchor = np.array([10.0, 10.0])  # Northeast
        agent = np.array([0.0, 0.0])

        tan_psi = aoa_tan_azimuth(anchor, agent)

        # tan(ψ) = ΔE / ΔN = 10/10 = 1
        assert np.isclose(tan_psi, 1.0, atol=1e-6)

    def test_aoa_tan_azimuth_singularity(self):
        """Test tan(ψ) when anchor is directly East (ΔN = 0)."""
        anchor = np.array([10.0, 0.0])  # Due East
        agent = np.array([0.0, 0.0])

        tan_psi = aoa_tan_azimuth(anchor, agent)

        # tan(ψ) should be very large (approaches infinity)
        assert tan_psi > 1e9

    def test_aoa_elevation_anchor_above(self):
        """Test elevation when anchor is above agent (θ > 0)."""
        anchor = np.array([10.0, 0.0, 10.0])  # Above and East
        agent = np.array([0.0, 0.0, 0.0])

        elevation = aoa_elevation(anchor, agent)

        # θ = arctan2(10, 10) = 45°
        assert np.isclose(elevation, np.pi / 4, atol=1e-6)

    def test_aoa_elevation_anchor_below(self):
        """Test elevation when anchor is below agent (θ < 0)."""
        anchor = np.array([10.0, 0.0, -10.0])  # Below and East
        agent = np.array([0.0, 0.0, 0.0])

        elevation = aoa_elevation(anchor, agent)

        # θ = arctan2(-10, 10) = -45°
        assert np.isclose(elevation, -np.pi / 4, atol=1e-6)

    def test_aoa_elevation_same_height(self):
        """Test elevation when anchor and agent at same height (θ = 0)."""
        anchor = np.array([10.0, 0.0, 5.0])
        agent = np.array([0.0, 0.0, 5.0])

        elevation = aoa_elevation(anchor, agent)

        # Same height: θ = 0
        assert np.isclose(elevation, 0.0, atol=1e-6)

    def test_aoa_elevation_directly_above(self):
        """Test elevation when anchor directly above agent (θ = 90°)."""
        anchor = np.array([0.0, 0.0, 10.0])  # Directly above
        agent = np.array([0.0, 0.0, 0.0])

        elevation = aoa_elevation(anchor, agent)

        # θ = 90°
        assert np.isclose(elevation, np.pi / 2, atol=1e-6)

    def test_aoa_sin_elevation_basic(self):
        """Test sin(θ) calculation per Eq. 4.63."""
        anchor = np.array([0.0, 0.0, 10.0])  # Directly above
        agent = np.array([0.0, 0.0, 0.0])

        sin_theta = aoa_sin_elevation(anchor, agent)

        # sin(θ) = ΔU / d = 10/10 = 1.0
        assert np.isclose(sin_theta, 1.0, atol=1e-6)

    def test_aoa_sin_elevation_45deg(self):
        """Test sin(θ) at 45° elevation."""
        anchor = np.array([10.0, 0.0, 10.0])
        agent = np.array([0.0, 0.0, 0.0])

        sin_theta = aoa_sin_elevation(anchor, agent)

        # d = sqrt(100 + 0 + 100) = sqrt(200)
        # sin(θ) = 10 / sqrt(200) = 1/sqrt(2)
        expected = 10.0 / np.sqrt(200.0)
        assert np.isclose(sin_theta, expected, atol=1e-6)

    def test_aoa_elevation_requires_3d(self):
        """Test that elevation requires 3D positions."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="3D positions required"):
            aoa_elevation(anchor, agent)

    def test_aoa_sin_elevation_requires_3d(self):
        """Test that sin_elevation requires 3D positions."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="3D positions required"):
            aoa_sin_elevation(anchor, agent)

    def test_aoa_measurement_vector_2d(self):
        """Test AOA measurement vector in 2D (tan(ψ) only) per Eq. 4.65."""
        # Anchors at cardinal directions from agent at origin
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]])  # E, N, W, S
        agent = np.array([0.0, 0.0])

        z = aoa_measurement_vector(anchors, agent, include_elevation=False)

        # Should return N tan(ψ) values
        assert z.shape == (4,)

        # tan(ψ) for each anchor:
        # E: tan(90°) -> very large positive
        # N: tan(0°) = 0
        # W: tan(-90°) -> very large negative
        # S: tan(180°) = 0
        assert z[1] == pytest.approx(0.0, abs=1e-6)  # North anchor
        assert z[3] == pytest.approx(0.0, abs=1e-6)  # South anchor

    def test_aoa_measurement_vector_3d(self):
        """Test AOA measurement vector with elevation per Eq. 4.65."""
        # Anchors at corners, 5m above agent
        anchors = np.array([[10, 10, 5], [10, -10, 5], [-10, -10, 5], [-10, 10, 5]])
        agent = np.array([0.0, 0.0, 0.0])

        z = aoa_measurement_vector(anchors, agent, include_elevation=True)

        # Should return 2*N values: [sin(θ_1), tan(ψ_1), sin(θ_2), tan(ψ_2), ...]
        assert z.shape == (8,)

        # Check structure: even indices are sin(θ), odd indices are tan(ψ)
        for i in range(4):
            sin_theta = z[2 * i]
            tan_psi = z[2 * i + 1]
            # All anchors at same height, should have same sin(θ)
            # d = sqrt(100 + 100 + 25) = 15
            expected_sin = 5.0 / 15.0
            assert np.isclose(sin_theta, expected_sin, atol=1e-6)

    def test_aoa_angle_vector_2d(self):
        """Test raw angle vector (not sin/tan)."""
        anchors = np.array([[10, 0], [0, 10]])  # East, North
        agent = np.array([0.0, 0.0])

        angles = aoa_angle_vector(anchors, agent, include_elevation=False)

        # Should return raw azimuth angles
        assert angles.shape == (2,)
        assert np.isclose(angles[0], np.pi / 2, atol=1e-6)  # East -> ψ = 90°
        assert np.isclose(angles[1], 0.0, atol=1e-6)  # North -> ψ = 0°

    def test_aoa_handcheck_geometry(self):
        """Hand-check geometry verification from book equations.

        Beacon at (0, 10, 5) ENU, Agent at (5, 5, 0) ENU.

        Eq. 4.63: sin(θ) = (x_u^i - x_u,a) / ||x_a - x^i||
        Eq. 4.64: tan(ψ) = (x_e^i - x_e,a) / (x_n^i - x_n,a)
        """
        anchor = np.array([0.0, 10.0, 5.0])  # (E=0, N=10, U=5)
        agent = np.array([5.0, 5.0, 0.0])  # (E=5, N=5, U=0)

        # ΔE = 0 - 5 = -5
        # ΔN = 10 - 5 = 5
        # ΔU = 5 - 0 = 5
        # distance = sqrt(25 + 25 + 25) = sqrt(75)

        distance = np.sqrt(75.0)

        # Eq. 4.63: sin(θ) = ΔU / d = 5 / sqrt(75)
        expected_sin_theta = 5.0 / distance
        sin_theta = aoa_sin_elevation(anchor, agent)
        assert np.isclose(sin_theta, expected_sin_theta, atol=1e-6)

        # Eq. 4.64: tan(ψ) = ΔE / ΔN = -5 / 5 = -1
        expected_tan_psi = -5.0 / 5.0
        tan_psi = aoa_tan_azimuth(anchor, agent)
        assert np.isclose(tan_psi, expected_tan_psi, atol=1e-6)

        # Azimuth: ψ = atan2(-5, 5) = -45° = -π/4
        expected_azimuth = np.arctan2(-5.0, 5.0)
        azimuth = aoa_azimuth(anchor, agent)
        assert np.isclose(azimuth, expected_azimuth, atol=1e-6)
        assert np.isclose(azimuth, -np.pi / 4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

