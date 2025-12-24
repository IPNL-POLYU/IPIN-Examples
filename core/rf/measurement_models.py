"""
RF measurement models for indoor positioning.

This module implements measurement models for various RF positioning techniques:
- TOA (Time of Arrival)
- RSS (Received Signal Strength)
- TDOA (Time Difference of Arrival)
- AOA (Angle of Arrival)

All functions implement equations from Chapter 4 of the IPIN book.
"""

from typing import Optional, Tuple

import numpy as np

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


# =============================================================================
# Clock Bias Unit Conversion Utilities
# =============================================================================
def clock_bias_seconds_to_meters(
    bias_seconds: float,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Convert clock bias from seconds to meters.

    The TOA model includes a clock bias term: d_measured = d_true + c*Δt
    where Δt is the clock offset in seconds.

    For positioning algorithms (Eqs. 4.24-4.26), the state vector uses
    the clock bias in meters (c*Δt) because this simplifies the Jacobian
    (the partial derivative ∂h/∂b = 1).

    Args:
        bias_seconds: Clock bias in seconds (Δt).
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.

    Returns:
        Clock bias in meters (c * Δt).

    Example:
        >>> # 10 nanoseconds clock offset
        >>> bias_s = 10e-9
        >>> bias_m = clock_bias_seconds_to_meters(bias_s)
        >>> print(f"{bias_m:.2f} m")  # ~3.0 m

    Notes:
        - 1 nanosecond ≈ 0.3 meters (one-way)
        - 10 ns → ~3 m, 100 ns → ~30 m

    References:
        Chapter 4, Eq. (4.24): State vector x = [x, y, z, c*Δt]^T
    """
    return c * bias_seconds


def clock_bias_meters_to_seconds(
    bias_meters: float,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Convert clock bias from meters to seconds.

    The positioning algorithms estimate clock bias in meters (c*Δt) for
    mathematical convenience. Use this function to convert back to seconds
    for interpretation or hardware calibration.

    Args:
        bias_meters: Clock bias in meters (c * Δt).
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.

    Returns:
        Clock bias in seconds (Δt).

    Example:
        >>> # Estimated 3 meters clock bias
        >>> bias_m = 3.0
        >>> bias_s = clock_bias_meters_to_seconds(bias_m)
        >>> print(f"{bias_s*1e9:.2f} ns")  # ~10 ns

    Notes:
        - 0.3 meters ≈ 1 nanosecond
        - 3 m → ~10 ns, 30 m → ~100 ns

    References:
        Chapter 4, Eq. (4.24): State vector x = [x, y, z, c*Δt]^T
    """
    return bias_meters / c


def toa_range(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    c: float = SPEED_OF_LIGHT,
    clock_bias_s: float = 0.0,
) -> float:
    """
    Compute Time of Arrival (TOA) range measurement.

    Implements Eq. (4.1)-(4.3) from Chapter 4:
        d_measured = ||p_tx - p_rx|| + c * Δt

    where:
        d_measured: measured pseudorange from anchor to agent
        p_tx, p_rx: transmitter/receiver positions
        c: speed of light (m/s)
        Δt: clock bias in SECONDS

    **Unit Convention:**

    - Input clock bias is in SECONDS (timing domain)
    - Internally converted to meters: bias_m = c * Δt
    - Output range is in METERS

    To work with clock bias in meters (as in positioning solvers),
    use the conversion utilities:
    - `clock_bias_seconds_to_meters()`
    - `clock_bias_meters_to_seconds()`

    Args:
        tx_pos: Transmitter (anchor) position [x, y, z] in meters.
        rx_pos: Receiver (agent) position [x, y, z] in meters.
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.
        clock_bias_s: Clock bias in SECONDS (Δt). Defaults to 0.0.
                     Positive bias = receiver clock ahead of system time
                     → measured range larger than true distance.

    Returns:
        Range measurement in METERS, including clock bias effect.

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([3.0, 4.0, 0.0])
        >>> range_m = toa_range(anchor, agent)
        >>> print(f"Range: {range_m:.2f} m")
        Range: 5.00 m

        >>> # With 10 ns clock bias (adds ~3 m)
        >>> range_biased = toa_range(anchor, agent, clock_bias_s=10e-9)
        >>> print(f"Biased range: {range_biased:.2f} m")
        Biased range: 8.00 m

    References:
        Chapter 4, Eq. (4.1)-(4.3): TOA measurement model
    """
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    # Geometric distance
    distance = np.linalg.norm(rx_pos - tx_pos)

    # Add clock bias effect: convert time offset (seconds) to distance (meters)
    # d_measured = d_true + c * Δt
    bias_meters = c * clock_bias_s
    distance_with_bias = distance + bias_meters

    return distance_with_bias


def two_way_toa_range(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
) -> float:
    """
    Compute geometric distance for two-way TOA (RTT) positioning.

    This function computes the true geometric distance between two positions,
    which represents the ideal range that RTT-based methods aim to estimate.

    For the actual RTT measurement model per book Eqs. (4.6)-(4.9), use:
        - `rtt_to_range()`: Convert RTT timing to range
        - `simulate_rtt_measurement()`: Simulate realistic RTT with noise

    Args:
        tx_pos: Transmitter (anchor) position [x, y, z] in meters.
        rx_pos: Receiver (agent) position [x, y, z] in meters.

    Returns:
        Geometric distance in meters.

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([10.0, 0.0, 0.0])
        >>> range_m = two_way_toa_range(anchor, agent)
        >>> print(f"True Range: {range_m:.2f} m")
        True Range: 10.00 m
    """
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    # Geometric distance (one-way)
    return np.linalg.norm(rx_pos - tx_pos)


def rtt_to_range(
    rtt: float,
    processing_time: float = 0.0,
    clock_drift: float = 0.0,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Convert Round-Trip Time (RTT) measurement to range estimate.

    Implements Eqs. (4.7)-(4.8) from Chapter 4:
        d_a^i = c * (t_arrive - t_depart - Δt_proc - Δt_drift) / 2

    The RTT measurement includes:
        - Signal travel time (to beacon and back)
        - Beacon processing time (Δt_proc)
        - Agent clock drift error (Δt_drift)

    Args:
        rtt: Measured round-trip time in seconds.
             This is (t_arrive - t_depart) from the agent's perspective.
        processing_time: Beacon processing time in seconds (Δt_proc^i).
                        Should be calibrated or provided by beacon manufacturer.
                        Defaults to 0.0.
        clock_drift: Agent clock drift error in seconds (Δt_drift).
                    Accumulated drift over RTT measurement period.
                    Defaults to 0.0.
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.

    Returns:
        Estimated range in meters.

    Example:
        >>> # 100ns RTT corresponds to ~15m range (30m round-trip)
        >>> rtt = 100e-9  # 100 nanoseconds
        >>> range_m = rtt_to_range(rtt)
        >>> print(f"Range: {range_m:.2f} m")
        Range: 14.99 m

        >>> # With 10ns processing time
        >>> range_m = rtt_to_range(rtt, processing_time=10e-9)
        >>> print(f"Range: {range_m:.2f} m")
        Range: 13.49 m

    Notes:
        - 1 nanosecond timing error ≈ 0.15 meter range error (one-way)
        - Processing time estimation is critical for accuracy
        - Clock drift accumulates over measurement time

    References:
        Chapter 4, Section 4.2.1.2, Eqs. (4.6)-(4.8): Two-way TOA
    """
    # Eq. 4.8: d = c * (RTT - Δt_proc - Δt_drift) / 2
    effective_travel_time = rtt - processing_time - clock_drift
    range_estimate = c * effective_travel_time / 2.0

    return range_estimate


def simulate_rtt_measurement(
    anchor_pos: np.ndarray,
    agent_pos: np.ndarray,
    processing_time: float = 0.0,
    processing_time_std: float = 0.0,
    clock_drift: float = 0.0,
    clock_drift_std: float = 0.0,
    c: float = SPEED_OF_LIGHT,
) -> Tuple[float, dict]:
    """
    Simulate a realistic RTT measurement with noise.

    Implements Eq. (4.9) from Chapter 4:
        d̃ = c * (t_arrive - t_depart - Δt_proc - Δt_drift) / 2
            + ω_proc + ω_drift

    This function simulates the RTT measurement process including:
        - True signal propagation time (from geometry)
        - Beacon processing delay (with optional noise)
        - Agent clock drift (with optional noise)

    Args:
        anchor_pos: Beacon/anchor position [x, y, z] in meters.
        agent_pos: Agent position [x, y, z] in meters.
        processing_time: Mean beacon processing time in seconds.
                        Defaults to 0.0.
        processing_time_std: Std dev of processing time noise (σ_proc).
                            Defaults to 0.0 (no noise).
        clock_drift: Mean clock drift in seconds over measurement period.
                    Defaults to 0.0.
        clock_drift_std: Std dev of clock drift noise (σ_drift).
                        Defaults to 0.0 (no noise).
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.

    Returns:
        rtt: Simulated RTT measurement in seconds.
        info: Dictionary with breakdown:
            - 'true_range': Actual geometric distance (m)
            - 'true_travel_time': True one-way propagation time (s)
            - 'processing_time_actual': Actual processing time with noise (s)
            - 'clock_drift_actual': Actual clock drift with noise (s)
            - 'range_estimate': Range estimated using rtt_to_range()

    Example:
        >>> anchor = np.array([0.0, 0.0, 0.0])
        >>> agent = np.array([15.0, 0.0, 0.0])  # 15m away
        >>> rtt, info = simulate_rtt_measurement(
        ...     anchor, agent,
        ...     processing_time=50e-9,  # 50ns processing
        ...     processing_time_std=5e-9,  # 5ns std
        ... )
        >>> print(f"RTT: {rtt*1e9:.1f} ns")
        >>> print(f"True range: {info['true_range']:.2f} m")

    Notes:
        - For Wi-Fi FTM, typical processing time is 10-100 ns
        - For UWB, processing time can be sub-nanosecond
        - TCXO clock drift is typically ±1-2 ppm

    References:
        Chapter 4, Section 4.2.1.2, Eq. (4.9): Two-way TOA with noise
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # True geometric distance
    true_range = np.linalg.norm(agent_pos - anchor_pos)

    # True one-way propagation time
    true_travel_time = true_range / c

    # Add noise to processing time if std > 0
    if processing_time_std > 0:
        processing_time_actual = processing_time + np.random.randn() * processing_time_std
    else:
        processing_time_actual = processing_time

    # Add noise to clock drift if std > 0
    if clock_drift_std > 0:
        clock_drift_actual = clock_drift + np.random.randn() * clock_drift_std
    else:
        clock_drift_actual = clock_drift

    # RTT = 2 * travel_time + processing_time + drift
    # (from agent's perspective, the clock drift affects the measured arrival time)
    rtt = 2.0 * true_travel_time + processing_time_actual + clock_drift_actual

    # Estimate range using known/estimated processing time and drift
    # (In practice, the actual values with noise are unknown; here we use
    # the nominal values for estimation to show the effect of noise)
    range_estimate = rtt_to_range(
        rtt, processing_time=processing_time, clock_drift=clock_drift, c=c
    )

    info = {
        'true_range': true_range,
        'true_travel_time': true_travel_time,
        'processing_time_actual': processing_time_actual,
        'clock_drift_actual': clock_drift_actual,
        'range_estimate': range_estimate,
    }

    return rtt, info


def range_to_rtt(
    distance: float,
    processing_time: float = 0.0,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Convert range to ideal RTT (inverse of rtt_to_range).

    Useful for generating synthetic RTT measurements from known geometry.

    Args:
        distance: One-way distance in meters.
        processing_time: Beacon processing time in seconds.
                        Defaults to 0.0.
        c: Speed of light in m/s. Defaults to SPEED_OF_LIGHT.

    Returns:
        Ideal RTT in seconds.

    Example:
        >>> distance = 15.0  # 15 meters
        >>> rtt = range_to_rtt(distance)
        >>> print(f"RTT: {rtt*1e9:.2f} ns")
        RTT: 100.07 ns
    """
    # RTT = 2 * (distance / c) + processing_time
    travel_time = distance / c
    rtt = 2.0 * travel_time + processing_time

    return rtt


def rss_pathloss(
    p_ref_dbm: float,
    distance: float,
    path_loss_exp: float = 2.0,
    d_ref: float = 1.0,
) -> float:
    """
    Compute RSS using the log-distance path-loss model (book Eq. 4.10).

    Implements Eq. (4.10) from Chapter 4:
        p_R,a^i = p_ref - 10*η*log10(d_a^i / d_ref)

    where:
        p_R,a^i: received signal power at agent a from beacon i (dBm)
        p_ref: reference RSS measured at distance d_ref (dBm)
        η (eta): path-loss exponent
        d_a^i: distance from beacon i to agent a
        d_ref: reference distance (typically 1m)

    Args:
        p_ref_dbm: Reference RSS at distance d_ref in dBm.
                  This is the measured power at a reference location.
        distance: Distance from beacon to agent in meters (d_a^i).
        path_loss_exp: Path-loss exponent η. Defaults to 2.0 (free space).
                      Typical indoor values: 2.5-4.0.
        d_ref: Reference distance in meters. Defaults to 1.0.

    Returns:
        Received signal strength in dBm (p_R,a^i).

    Example:
        >>> # RSS at 10m with p_ref=-40dBm at 1m, η=2.5
        >>> rss = rss_pathloss(p_ref_dbm=-40.0, distance=10.0, path_loss_exp=2.5)
        >>> print(f"RSS: {rss:.2f} dBm")
        RSS: -65.00 dBm

    References:
        Chapter 4, Section 4.2.1.3, Eq. (4.10): RSS path-loss model
    """
    if distance <= 0:
        raise ValueError("Distance must be positive")

    # Eq. 4.10: p_R = p_ref - 10*η*log10(d/d_ref)
    rss_dbm = p_ref_dbm - 10 * path_loss_exp * np.log10(distance / d_ref)

    return rss_dbm


def rss_to_distance(
    rss_dbm: float,
    p_ref_dbm: float,
    path_loss_exp: float = 2.0,
    d_ref: float = 1.0,
) -> float:
    """
    Estimate distance from RSS using the inverse path-loss model (book Eq. 4.11).

    Implements Eq. (4.11) from Chapter 4:
        d_a^i = d_ref * 10^((p_ref - p_R,a^i) / (10*η))

    Args:
        rss_dbm: Received signal strength in dBm (p_R,a^i).
        p_ref_dbm: Reference RSS at distance d_ref in dBm.
        path_loss_exp: Path-loss exponent η. Defaults to 2.0.
        d_ref: Reference distance in meters. Defaults to 1.0.

    Returns:
        Estimated distance in meters.

    Example:
        >>> # Distance from RSS=-65dBm with p_ref=-40dBm, η=2.5
        >>> distance = rss_to_distance(rss_dbm=-65.0, p_ref_dbm=-40.0, path_loss_exp=2.5)
        >>> print(f"Distance: {distance:.2f} m")
        Distance: 10.00 m

    References:
        Chapter 4, Section 4.2.1.3, Eq. (4.11): RSS-to-distance inversion
    """
    # Eq. 4.11: d = d_ref * 10^((p_ref - p_R) / (10*η))
    exponent = (p_ref_dbm - rss_dbm) / (10 * path_loss_exp)
    distance = d_ref * (10**exponent)

    return distance


def simulate_rss_measurement(
    anchor_pos: np.ndarray,
    agent_pos: np.ndarray,
    p_ref_dbm: float,
    path_loss_exp: float = 2.5,
    d_ref: float = 1.0,
    sigma_long_db: float = 0.0,
    sigma_short_linear: float = 0.0,
    n_samples_avg: int = 1,
    short_fading_model: str = "rayleigh",
) -> Tuple[float, dict]:
    """
    Simulate an RSS measurement with fading noise (book Eqs. 4.10, 4.12).

    Implements Eqs. (4.10) and (4.12) from Chapter 4:
        p̃_R,a^i = p_R,a^i + ω_long(x_a) + ω_short(t)

    **Fading Model Summary:**

    - **ω_long(x_a)**: Long-term fading (shadowing), location-dependent.
      Modeled as Gaussian in dB: ω_long ~ N(0, σ_long_db²).
      Cannot be reduced by temporal averaging.

    - **ω_short(t)**: Short-term fading (multipath), time-varying.
      Default model: Rayleigh fading (amplitude domain).
      - Rayleigh: Amplitude A ~ Rayleigh(σ), power P = A² ~ Exponential
      - In dB: ω_short_db = 10*log10(P) has a specific distribution
      - Can be reduced by averaging multiple samples in linear power domain.

    **Physical Interpretation:**

    - Rayleigh fading occurs when there is no dominant LOS path and the
      received signal is the sum of many scattered components.
    - The parameter `sigma_short_linear` is the Rayleigh scale parameter σ.
    - For normalized mean power (E[P] = 1), use σ = 1/sqrt(2) ≈ 0.707.
    - Typical values: σ ≈ 0.5-1.0 (produces ~5-10 dB spread in power).

    Args:
        anchor_pos: Beacon/anchor position [x, y] or [x, y, z] in meters.
        agent_pos: Agent position [x, y] or [x, y, z] in meters.
        p_ref_dbm: Reference RSS at distance d_ref in dBm.
        path_loss_exp: Path-loss exponent η. Defaults to 2.5 (indoor).
        d_ref: Reference distance in meters. Defaults to 1.0.
        sigma_long_db: Std dev of long-term fading in dB (σ_long).
                      Typical indoor values: 4-8 dB. Defaults to 0.0.
        sigma_short_linear: Short-term fading parameter (Rayleigh scale σ).
                           For Rayleigh: scale parameter in linear amplitude.
                           For Gaussian: std dev in dB.
                           Typical Rayleigh σ: 0.5-1.0. Defaults to 0.0.
        n_samples_avg: Number of samples to average for short-term fading
                      reduction. Averaging is done in linear power domain.
                      Defaults to 1 (no averaging).
        short_fading_model: Short-term fading model. Options:
                           - "rayleigh": Rayleigh amplitude fading (default)
                           - "gaussian_db": Gaussian fading in dB domain
                           - "none": No short-term fading

    Returns:
        rss_measured: Measured RSS in dBm (with fading noise).
        info: Dictionary with breakdown:
            - 'true_distance': Actual geometric distance (m)
            - 'rss_true': True RSS without fading (dBm)
            - 'omega_long_db': Long-term fading sample (dB)
            - 'omega_short_db': Short-term fading after averaging (dB)
            - 'omega_short_samples': Individual short-term samples (dB) if n>1
            - 'distance_estimate': Distance estimated from measured RSS (m)
            - 'distance_error_factor': Multiplicative error d̃/d using BOTH terms
              (Eq. 4.13): 10^(-(ω_long + ω_short) / (10*η))
            - 'short_fading_model': Model used for short-term fading

    Example:
        >>> np.random.seed(42)
        >>> anchor = np.array([0.0, 0.0])
        >>> agent = np.array([10.0, 0.0])
        >>> # Rayleigh fading with σ=0.7 (normalized mean power)
        >>> rss, info = simulate_rss_measurement(
        ...     anchor, agent,
        ...     p_ref_dbm=-40.0,
        ...     path_loss_exp=2.5,
        ...     sigma_long_db=6.0,
        ...     sigma_short_linear=0.707,
        ...     n_samples_avg=5,
        ...     short_fading_model="rayleigh",
        ... )
        >>> print(f"True RSS: {info['rss_true']:.1f} dBm")
        >>> print(f"Short-term fading: {info['omega_short_db']:.2f} dB")

    Notes:
        - Long-term fading: Location-dependent, cannot be reduced by averaging
        - Short-term fading: Time-dependent, reduced by averaging n samples
          - Rayleigh: Average in LINEAR POWER, then convert to dB
          - Gaussian: Std reduces by sqrt(n) in dB domain directly
        - Multiplicative distance error (Eq. 4.13, generalized form):
          d̃ = d * 10^(-(ω_long + ω_short) / (10*η))
          The book shows only ω_long because it assumes ω_short is averaged
          out. We use both terms for accurate simulation.

    References:
        Chapter 4, Section 4.2.1.3, Eqs. (4.10), (4.12), (4.13): RSS with fading
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    valid_models = {"rayleigh", "gaussian_db", "none"}
    if short_fading_model not in valid_models:
        raise ValueError(
            f"short_fading_model must be one of {valid_models}, "
            f"got '{short_fading_model}'"
        )

    # True geometric distance
    true_distance = np.linalg.norm(agent_pos - anchor_pos)

    if true_distance <= 0:
        raise ValueError("Agent and anchor positions must be different")

    # True RSS without fading (Eq. 4.10)
    rss_true = rss_pathloss(p_ref_dbm, true_distance, path_loss_exp, d_ref)

    # Long-term fading (location-dependent, cannot be averaged)
    # Modeled as Gaussian in dB per book
    omega_long_db = 0.0
    if sigma_long_db > 0:
        omega_long_db = np.random.randn() * sigma_long_db

    # Short-term fading (time-varying, can be reduced by averaging)
    omega_short_db = 0.0
    omega_short_samples = []

    if sigma_short_linear > 0 and short_fading_model != "none":
        if short_fading_model == "rayleigh":
            # Rayleigh fading in amplitude domain
            # A ~ Rayleigh(σ), P = A² ~ Exponential
            # Generate n_samples_avg independent Rayleigh samples
            amplitudes = np.random.rayleigh(
                scale=sigma_short_linear, size=n_samples_avg
            )
            # Convert to power (linear)
            powers_linear = amplitudes**2

            # Store individual samples in dB for diagnostics
            # Avoid log(0) by clipping
            powers_clipped = np.maximum(powers_linear, 1e-10)
            omega_short_samples = 10 * np.log10(powers_clipped)

            # Average in LINEAR POWER domain (physically correct for power averaging)
            avg_power_linear = np.mean(powers_linear)

            # Convert average power to dB relative to mean power
            # For Rayleigh with scale σ, E[A²] = 2σ²
            # The mean power is 2σ², so we normalize relative to expected mean
            expected_mean_power = 2 * sigma_short_linear**2
            if expected_mean_power > 0:
                normalized_power = avg_power_linear / expected_mean_power
                omega_short_db = 10 * np.log10(max(normalized_power, 1e-10))
            else:
                omega_short_db = 0.0

        elif short_fading_model == "gaussian_db":
            # Gaussian fading directly in dB domain (legacy model)
            # sigma_short_linear is interpreted as std dev in dB
            sigma_short_db = sigma_short_linear
            if n_samples_avg > 1:
                # Generate n samples and average
                samples_db = np.random.randn(n_samples_avg) * sigma_short_db
                omega_short_samples = samples_db.tolist()

                # For Gaussian in dB: averaging in dB is NOT physically correct
                # but we do it for this legacy model. More correct would be:
                # convert to linear, average, convert back.
                # Here we use the simple approximation: std reduces by sqrt(n)
                effective_std = sigma_short_db / np.sqrt(n_samples_avg)
                omega_short_db = np.random.randn() * effective_std
            else:
                omega_short_db = np.random.randn() * sigma_short_db
                omega_short_samples = [omega_short_db]

    # Measured RSS (Eq. 4.12)
    rss_measured = rss_true + omega_long_db + omega_short_db

    # Estimate distance from measured RSS (Eq. 4.11)
    distance_estimate = rss_to_distance(rss_measured, p_ref_dbm, path_loss_exp, d_ref)

    # Multiplicative distance error factor (Eq. 4.13)
    # d̃ = d * 10^(-ω_total / (10*η))
    total_fading_db = omega_long_db + omega_short_db
    distance_error_factor = 10 ** (-total_fading_db / (10 * path_loss_exp))

    info = {
        'true_distance': true_distance,
        'rss_true': rss_true,
        'omega_long_db': omega_long_db,
        'omega_short_db': omega_short_db,
        'omega_short_samples': omega_short_samples,
        'distance_estimate': distance_estimate,
        'distance_error_factor': distance_error_factor,
        'short_fading_model': short_fading_model,
    }

    return rss_measured, info


def rss_fading_to_distance_error(
    omega_db: float,
    path_loss_exp: float = 2.5,
) -> float:
    """
    Convert RSS fading (dB) to multiplicative distance error factor (Eq. 4.13).

    Implements the generalized form of Eq. (4.13) from Chapter 4:
        d̃ = d * 10^(-ω_total / (10*η))

    where ω_total = ω_long + ω_short (total fading in dB).

    **Book vs Implementation:**
    - The book's Eq. 4.13 shows only ω_long because it assumes ω_short
      is mitigated by time-averaging multiple RSS samples.
    - This function accepts ANY fading value (ω_long, ω_short, or their sum)
      for flexibility. Pass total_fading = ω_long + ω_short for full accuracy.

    This shows how RSS fading in dB translates to a multiplicative
    distance error. For example:
        - +6 dB fading → ~0.55× distance (underestimate)
        - -6 dB fading → ~1.82× distance (overestimate)

    Args:
        omega_db: Fading noise in dB (positive = stronger signal).
                 Can be ω_long only (book approximation after averaging)
                 or ω_long + ω_short (full model).
        path_loss_exp: Path-loss exponent η. Defaults to 2.5.

    Returns:
        Multiplicative distance error factor (d̃/d).

    Example:
        >>> # How does 6 dB total fading affect distance estimate?
        >>> factor = rss_fading_to_distance_error(6.0, path_loss_exp=2.5)
        >>> print(f"Distance factor: {factor:.2f}x")
        Distance factor: 0.55x

        >>> # -6 dB fading (weaker signal)
        >>> factor = rss_fading_to_distance_error(-6.0, path_loss_exp=2.5)
        >>> print(f"Distance factor: {factor:.2f}x")
        Distance factor: 1.82x

        >>> # With both fading terms from simulate_rss_measurement
        >>> total_fading = info['omega_long_db'] + info['omega_short_db']
        >>> factor = rss_fading_to_distance_error(total_fading)

    References:
        Chapter 4, Section 4.2.1.3, Eq. (4.13): RSS fading distance error
    """
    # Eq. 4.13 (generalized): d̃/d = 10^(-ω_total / (10*η))
    return 10 ** (-omega_db / (10 * path_loss_exp))


def tdoa_range_difference(
    anchor_i: np.ndarray,
    anchor_j: np.ndarray,
    agent_pos: np.ndarray,
    c: float = SPEED_OF_LIGHT,
) -> float:
    """
    Compute TDOA range difference between two anchors.

    Implements Eqs. (4.27)-(4.33) from Chapter 4:
        d_i,j^a = d_i^a - d_j^a = ||p_i - p^a|| - ||p_j - p^a||

    where:
        d_i,j^a: range difference (TDOA measurement)
        p_i, p_j: anchor positions
        p^a: agent position

    Args:
        anchor_i: Position of anchor i [x, y, z] in meters.
        anchor_j: Position of anchor j [x, y, z] in meters.
        agent_pos: Agent position [x, y, z] in meters.
        c: Speed of light (not used, for API consistency).

    Returns:
        Range difference in meters.

    Example:
        >>> anchor_1 = np.array([0.0, 0.0, 0.0])
        >>> anchor_2 = np.array([10.0, 0.0, 0.0])
        >>> agent = np.array([5.0, 5.0, 0.0])
        >>> rd = tdoa_range_difference(anchor_1, anchor_2, agent)
        >>> print(f"Range difference: {rd:.3f} m")
        Range difference: -2.929 m
    """
    anchor_i = np.asarray(anchor_i, dtype=float)
    anchor_j = np.asarray(anchor_j, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # Distance from agent to anchor i
    dist_i = np.linalg.norm(agent_pos - anchor_i)

    # Distance from agent to anchor j
    dist_j = np.linalg.norm(agent_pos - anchor_j)

    # Range difference
    range_diff = dist_i - dist_j

    return range_diff


def tdoa_measurement_vector(
    anchors: np.ndarray,
    agent_pos: np.ndarray,
    reference_anchor_idx: int = 0,
) -> np.ndarray:
    """
    Compute TDOA measurement vector for all anchor pairs.

    Implements Eqs. (4.27)-(4.33): stacks range differences relative to
    a reference anchor.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        agent_pos: Agent position [x, y] or [x, y, z].
        reference_anchor_idx: Index of reference anchor. Defaults to 0.

    Returns:
        TDOA measurement vector of shape (N-1,), containing range differences
        between each anchor and the reference anchor.

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> agent = np.array([5, 5])
        >>> tdoa = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)
        >>> print(tdoa.shape)
        (3,)
    """
    anchors = np.asarray(anchors, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    n_anchors = anchors.shape[0]
    if reference_anchor_idx < 0 or reference_anchor_idx >= n_anchors:
        raise ValueError(
            f"reference_anchor_idx must be in [0, {n_anchors-1}], "
            f"got {reference_anchor_idx}"
        )

    reference_anchor = anchors[reference_anchor_idx]

    # Compute range differences relative to reference
    tdoa_measurements = []
    for i in range(n_anchors):
        if i == reference_anchor_idx:
            continue
        range_diff = tdoa_range_difference(
            anchors[i], reference_anchor, agent_pos
        )
        tdoa_measurements.append(range_diff)

    return np.array(tdoa_measurements)


def aoa_azimuth(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute azimuth angle from agent toward anchor (ENU convention).

    Implements Eq. (4.64) from Chapter 4:
        tan(ψ_i) = (x_e^i - x_e,a) / (x_n^i - x_n,a)

    where:
        ψ_i: azimuth angle measured from North, positive CCW (ENU convention)
        (x_e^i, x_n^i): anchor position (East, North)
        (x_e,a, x_n,a): agent position (East, North)

    The azimuth is computed as ψ = atan2(ΔE, ΔN) where:
        ΔE = anchor_E - agent_E
        ΔN = anchor_N - agent_N

    Args:
        anchor_pos: Anchor (beacon) position [E, N] or [E, N, U] in ENU.
        agent_pos: Agent position [E, N] or [E, N, U] in ENU.

    Returns:
        Azimuth angle ψ in radians, range [-π, π].
        - ψ = 0: anchor is directly North of agent
        - ψ = π/2: anchor is directly East of agent
        - ψ = π or -π: anchor is directly South of agent
        - ψ = -π/2: anchor is directly West of agent

    Example:
        >>> anchor = np.array([0.0, 10.0])  # North of agent
        >>> agent = np.array([0.0, 0.0])
        >>> azimuth = aoa_azimuth(anchor, agent)
        >>> print(f"Azimuth: {np.rad2deg(azimuth):.2f}°")
        Azimuth: 0.00°

        >>> anchor = np.array([10.0, 0.0])  # East of agent
        >>> agent = np.array([0.0, 0.0])
        >>> azimuth = aoa_azimuth(anchor, agent)
        >>> print(f"Azimuth: {np.rad2deg(azimuth):.2f}°")
        Azimuth: 90.00°
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # Compute differences (anchor - agent) per Eq. (4.64)
    delta_e = anchor_pos[0] - agent_pos[0]  # ΔE = x_e^i - x_e,a
    delta_n = anchor_pos[1] - agent_pos[1]  # ΔN = x_n^i - x_n,a

    # Azimuth from North, positive CCW: atan2(ΔE, ΔN)
    azimuth = np.arctan2(delta_e, delta_n)

    return azimuth


def aoa_elevation(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute elevation angle from agent toward anchor (ENU convention).

    Implements Eq. (4.63) from Chapter 4:
        sin(θ_i) = (x_u^i - x_u,a) / ||x_a - x^i||

    where:
        θ_i: elevation angle (positive when anchor is above agent)
        (x_e^i, x_n^i, x_u^i): anchor position (East, North, Up)
        (x_e,a, x_n,a, x_u,a): agent position (East, North, Up)

    The elevation θ is computed as:
        θ = arcsin((anchor_U - agent_U) / distance)

    Or equivalently:
        θ = arctan2(ΔU, horizontal_distance)

    where ΔU = anchor_U - agent_U.

    Args:
        anchor_pos: Anchor (beacon) position [E, N, U] in ENU.
        agent_pos: Agent position [E, N, U] in ENU.

    Returns:
        Elevation angle θ in radians, range [-π/2, π/2].
        - θ > 0: anchor is above agent
        - θ = 0: anchor is at same height as agent
        - θ < 0: anchor is below agent

    Example:
        >>> anchor = np.array([0.0, 0.0, 10.0])  # Above agent
        >>> agent = np.array([0.0, 0.0, 0.0])
        >>> elevation = aoa_elevation(anchor, agent)
        >>> print(f"Elevation: {np.rad2deg(elevation):.2f}°")
        Elevation: 90.00°

        >>> anchor = np.array([10.0, 0.0, 10.0])  # Above and East
        >>> agent = np.array([0.0, 0.0, 0.0])
        >>> elevation = aoa_elevation(anchor, agent)
        >>> print(f"Elevation: {np.rad2deg(elevation):.2f}°")
        Elevation: 45.00°
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    if len(anchor_pos) < 3 or len(agent_pos) < 3:
        raise ValueError("3D positions required for elevation angle")

    # Compute differences (anchor - agent) per Eq. (4.63)
    delta_e = anchor_pos[0] - agent_pos[0]  # ΔE
    delta_n = anchor_pos[1] - agent_pos[1]  # ΔN
    delta_u = anchor_pos[2] - agent_pos[2]  # ΔU = x_u^i - x_u,a

    # Horizontal distance
    horizontal_dist = np.sqrt(delta_e**2 + delta_n**2)

    # Elevation angle: arctan2(ΔU, horizontal_dist)
    elevation = np.arctan2(delta_u, horizontal_dist)

    return elevation


def aoa_sin_elevation(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute sine of elevation angle from agent toward anchor (ENU convention).

    Implements Eq. (4.63) from Chapter 4:
        sin(θ_i) = (x_u^i - x_u,a) / ||x_a - x^i||

    where:
        θ_i: elevation angle
        (x_e^i, x_n^i, x_u^i): anchor position (East, North, Up)
        (x_e,a, x_n,a, x_u,a): agent position (East, North, Up)

    This function returns sin(θ) directly as required for the I-WLS
    measurement vector (Eq. 4.65).

    Args:
        anchor_pos: Anchor (beacon) position [E, N, U] in ENU.
        agent_pos: Agent position [E, N, U] in ENU.

    Returns:
        sin(θ): Sine of elevation angle, range [-1, 1].

    Example:
        >>> anchor = np.array([0.0, 0.0, 10.0])  # Directly above
        >>> agent = np.array([0.0, 0.0, 0.0])
        >>> sin_theta = aoa_sin_elevation(anchor, agent)
        >>> print(f"sin(θ): {sin_theta:.4f}")
        sin(θ): 1.0000
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    if len(anchor_pos) < 3 or len(agent_pos) < 3:
        raise ValueError("3D positions required for elevation angle")

    # Compute ΔU = anchor_U - agent_U per Eq. (4.63)
    delta_u = anchor_pos[2] - agent_pos[2]

    # Compute 3D distance ||x_a - x^i||
    distance = np.linalg.norm(agent_pos - anchor_pos)

    if distance < 1e-10:
        return 0.0  # Avoid division by zero

    # sin(θ) = ΔU / distance per Eq. (4.63)
    sin_theta = delta_u / distance

    return sin_theta


def aoa_tan_azimuth(anchor_pos: np.ndarray, agent_pos: np.ndarray) -> float:
    """
    Compute tangent of azimuth angle from agent toward anchor (ENU convention).

    Implements Eq. (4.64) from Chapter 4:
        tan(ψ_i) = (x_e^i - x_e,a) / (x_n^i - x_n,a)

    where:
        ψ_i: azimuth angle from North, positive CCW
        (x_e^i, x_n^i): anchor position (East, North)
        (x_e,a, x_n,a): agent position (East, North)

    This function returns tan(ψ) directly as required for the I-WLS
    measurement vector (Eq. 4.65).

    Note: tan(ψ) is undefined when ΔN = 0 (anchor directly East or West).
    In such cases, this function returns ±inf or a large value.

    Args:
        anchor_pos: Anchor (beacon) position [E, N] or [E, N, U] in ENU.
        agent_pos: Agent position [E, N] or [E, N, U] in ENU.

    Returns:
        tan(ψ): Tangent of azimuth angle.

    Example:
        >>> anchor = np.array([10.0, 10.0])  # Northeast (45°)
        >>> agent = np.array([0.0, 0.0])
        >>> tan_psi = aoa_tan_azimuth(anchor, agent)
        >>> print(f"tan(ψ): {tan_psi:.4f}")
        tan(ψ): 1.0000
    """
    anchor_pos = np.asarray(anchor_pos, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    # Compute differences per Eq. (4.64)
    delta_e = anchor_pos[0] - agent_pos[0]  # ΔE = x_e^i - x_e,a
    delta_n = anchor_pos[1] - agent_pos[1]  # ΔN = x_n^i - x_n,a

    # tan(ψ) = ΔE / ΔN per Eq. (4.64)
    if np.abs(delta_n) < 1e-10:
        # Handle singularity when anchor is directly East or West
        # Return large value with correct sign
        return np.sign(delta_e) * 1e10 if np.abs(delta_e) > 1e-10 else 0.0

    tan_psi = delta_e / delta_n

    return tan_psi


def aoa_measurement_vector(
    anchors: np.ndarray,
    agent_pos: np.ndarray,
    include_elevation: bool = True,
) -> np.ndarray:
    """
    Compute AOA measurement vector for all anchors (book Eq. 4.65 format).

    Implements Eq. (4.65) from Chapter 4:
        z_a = [sin(θ_1), tan(ψ_1), sin(θ_2), tan(ψ_2), ..., sin(θ_I), tan(ψ_I)]^T

    where:
        θ_i: elevation angle (Eq. 4.63)
        ψ_i: azimuth angle from North (Eq. 4.64)
        I: number of anchors

    The measurement vector uses sin(θ) and tan(ψ) as required for I-WLS
    linearization in the book's formulation.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3) in ENU.
        agent_pos: Agent position [E, N] or [E, N, U] in ENU.
        include_elevation: If True (default), include sin(elevation) for 3D.
                          If False, return only tan(azimuth) values.
                          Requires 3D positions when True.

    Returns:
        AOA measurement vector:
        - If include_elevation=True (3D): shape (2*N,) with
          [sin(θ_1), tan(ψ_1), sin(θ_2), tan(ψ_2), ...]
        - If include_elevation=False (2D): shape (N,) with
          [tan(ψ_1), tan(ψ_2), ...]

    Example:
        >>> # 3D case with elevation
        >>> anchors = np.array([[10, 0, 5], [0, 10, 5], [-10, 0, 5]])
        >>> agent = np.array([0.0, 0.0, 0.0])
        >>> z = aoa_measurement_vector(anchors, agent, include_elevation=True)
        >>> print(f"Measurement vector shape: {z.shape}")
        Measurement vector shape: (6,)

        >>> # 2D case (azimuth only)
        >>> anchors_2d = np.array([[10, 0], [0, 10], [-10, 0]])
        >>> agent_2d = np.array([0.0, 0.0])
        >>> z = aoa_measurement_vector(anchors_2d, agent_2d, include_elevation=False)
        >>> print(f"Measurement vector shape: {z.shape}")
        Measurement vector shape: (3,)
    """
    anchors = np.asarray(anchors, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    n_anchors = anchors.shape[0]
    measurements = []

    for i in range(n_anchors):
        if include_elevation:
            # 3D case: [sin(θ_i), tan(ψ_i)] per Eq. (4.65)
            sin_theta = aoa_sin_elevation(anchors[i], agent_pos)
            tan_psi = aoa_tan_azimuth(anchors[i], agent_pos)
            measurements.extend([sin_theta, tan_psi])
        else:
            # 2D case: only tan(ψ_i)
            tan_psi = aoa_tan_azimuth(anchors[i], agent_pos)
            measurements.append(tan_psi)

    return np.array(measurements)


def aoa_angle_vector(
    anchors: np.ndarray,
    agent_pos: np.ndarray,
    include_elevation: bool = False,
) -> np.ndarray:
    """
    Compute AOA angle vector for all anchors (raw angles, not sin/tan).

    Returns the raw azimuth and elevation angles rather than sin(θ) and tan(ψ).
    Useful for visualization and debugging.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3) in ENU.
        agent_pos: Agent position [E, N] or [E, N, U] in ENU.
        include_elevation: If True, include elevation angles (requires 3D).

    Returns:
        AOA angle vector:
        - If include_elevation=False: shape (N,) with [ψ_1, ψ_2, ...]
        - If include_elevation=True: shape (2*N,) with [θ_1, ψ_1, θ_2, ψ_2, ...]

    Example:
        >>> anchors = np.array([[10, 0], [0, 10]])
        >>> agent = np.array([0.0, 0.0])
        >>> angles = aoa_angle_vector(anchors, agent, include_elevation=False)
        >>> print(f"Azimuth angles (deg): {np.rad2deg(angles)}")
        Azimuth angles (deg): [90. 0.]
    """
    anchors = np.asarray(anchors, dtype=float)
    agent_pos = np.asarray(agent_pos, dtype=float)

    n_anchors = anchors.shape[0]
    angles = []

    for i in range(n_anchors):
        azimuth = aoa_azimuth(anchors[i], agent_pos)

        if include_elevation:
            elevation = aoa_elevation(anchors[i], agent_pos)
            angles.extend([elevation, azimuth])
        else:
            angles.append(azimuth)

    return np.array(angles)



