"""
1/f (pink) noise generation for IMU bias instability simulation.

This module provides tools for generating and scaling 1/f (pink) noise,
which produces the flat "bias instability" region in Allan deviation plots.

Implements FFT-based frequency shaping to create noise with power spectral
density (PSD) proportional to 1/f, which is characteristic of flicker noise
and bias instability in MEMS IMU sensors.

Key concepts:
    - Pink noise: PSD ~ 1/f (between white noise 1/f^0 and brown noise 1/f^2)
    - Allan deviation: Pink noise produces a flat region (slope ~ 0)
    - Bias instability (BI): Minimum of Allan deviation curve

References:
    - IEEE Std 952-1997: Allan variance
    - Chapter 6: IMU error modeling and Allan variance analysis

Author: Li-Ta Hsu
Date: December 2025
"""

from typing import Optional, Callable
import numpy as np


def pink_noise_1f_fft(
    N: int,
    fs: float,
    fmin: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate zero-mean, unit-std approximate 1/f (pink) noise using FFT shaping.

    Pink noise has a power spectral density (PSD) proportional to 1/f.
    This is achieved by shaping the amplitude spectrum as 1/sqrt(f), so that
    PSD = |X(f)|^2 ~ 1/f.

    In Allan deviation analysis, pink noise produces a flat region (slope ~ 0)
    corresponding to bias instability, which is the characteristic behavior
    of flicker noise in MEMS gyroscopes and accelerometers.

    Args:
        N: Number of samples to generate.
        fs: Sampling frequency in Hz.
        fmin: Low-frequency floor to prevent divergence at f→0.
              If None, defaults to fs/N (inverse of total duration).
              This prevents DC and very low frequencies from dominating.
        rng: Random number generator for reproducibility.
             If None, uses np.random.default_rng().

    Returns:
        Pink noise array of shape (N,), zero-mean, unit standard deviation.

    Raises:
        ValueError: If fs <= 0.

    Example:
        >>> import numpy as np
        >>> from core.sim.noise_pink import pink_noise_1f_fft
        >>> 
        >>> # Generate 1 hour of pink noise at 100 Hz
        >>> fs = 100.0  # Hz
        >>> duration = 3600.0  # seconds
        >>> N = int(fs * duration)
        >>> 
        >>> rng = np.random.default_rng(42)
        >>> pink = pink_noise_1f_fft(N, fs, rng=rng)
        >>> 
        >>> # Verify properties
        >>> assert np.abs(np.mean(pink)) < 0.01  # zero-mean
        >>> assert np.abs(np.std(pink) - 1.0) < 0.01  # unit-std

    Notes:
        - The DC component (f=0) is explicitly set to zero
        - For even N, the Nyquist bin is forced to be real
        - The low-frequency floor fmin prevents 1/f divergence at f→0
        - Output is normalized to zero-mean, unit standard deviation
        - Allan deviation of pink noise shows flat region (bias instability)

    Implementation:
        1. Generate frequency bins using rfftfreq
        2. Create amplitude shaping: |X(f)| ∝ 1/sqrt(f)
        3. Apply random Gaussian phases (real and imaginary parts)
        4. Force DC to zero and ensure real Nyquist bin
        5. Inverse FFT back to time domain
        6. Normalize to zero-mean, unit-std
    """
    if rng is None:
        rng = np.random.default_rng()

    if N < 2:
        return np.zeros(N)

    if fs <= 0:
        raise ValueError("fs must be > 0")

    # Frequency bins for real FFT (N//2 + 1 bins)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Low-frequency floor: prevents huge DC/near-DC blowup
    # Default to 1/duration = fs/N (one cycle over entire record)
    if fmin is None:
        fmin = fs / N

    # Build shaping: |X(f)| ∝ 1/sqrt(f) => PSD ∝ 1/f
    # Avoid f=0 by replacing it with fmin
    f = np.copy(freqs)
    f[0] = fmin
    f = np.maximum(f, fmin)
    shape = 1.0 / np.sqrt(f)

    # Random complex spectrum with Gaussian real and imaginary parts
    # rFFT spectrum length = N//2 + 1
    re = rng.standard_normal(len(freqs))
    im = rng.standard_normal(len(freqs))
    spectrum = (re + 1j * im) * shape

    # Force DC to zero (good hygiene for drift-free noise)
    spectrum[0] = 0.0 + 0.0j

    # Enforce real signal constraint: Nyquist bin must be real if N is even
    if N % 2 == 0:
        spectrum[-1] = spectrum[-1].real + 0.0j

    # Inverse FFT back to time domain
    x = np.fft.irfft(spectrum, n=N)

    # Normalize to zero-mean, unit-std
    x -= np.mean(x)
    std = np.std(x)
    if std > 0:
        x /= std

    return x


def scale_to_bias_instability(
    pink_unit: np.ndarray,
    target_bi_rad_s: float,
    allan_sigma_func: Callable,
    tau_grid_s: np.ndarray,
    fs: float,
    bi_factor: float = 0.664,
) -> np.ndarray:
    """
    Scale unit pink noise so Allan deviation minimum matches target BI.

    Bias instability (BI) is conventionally defined from the minimum of the
    Allan deviation curve. The relationship is:
        BI ≈ sigma_min / 0.664
    where sigma_min is the minimum Allan deviation and 0.664 is the
    conventional factor from Allan variance theory.

    This function:
        1. Computes Allan deviation for unit pink noise
        2. Finds sigma_min (minimum of Allan deviation)
        3. Scales so that sigma_min_scaled ≈ target_bi_rad_s * 0.664
        4. Returns the scaled pink noise

    Args:
        pink_unit: Unit-std pink noise sequence (from pink_noise_1f_fft).
                   Shape: (N,).
        target_bi_rad_s: Desired bias instability in rad/s (for gyro)
                        or m/s² (for accel). This should be converted
                        from deg/hr using: np.deg2rad(bi_deg_hr) / 3600.0
        allan_sigma_func: Function that computes Allan deviation.
                         Signature: sigma = func(x, fs, taus)
                         Should return (taus, sigma).
        tau_grid_s: Tau values for Allan deviation computation.
                   Shape: (M,). Units: seconds.
        fs: Sampling frequency in Hz.
        bi_factor: Convention factor for BI extraction (default: 0.664).
                  From Allan variance theory: BI = sigma_min / bi_factor.

    Returns:
        Scaled pink noise with Allan deviation minimum matching target BI.
        Shape: (N,).

    Raises:
        ValueError: If sigma_min <= 0 (indicates allan_sigma_func failure).

    Example:
        >>> import numpy as np
        >>> from core.sim.noise_pink import pink_noise_1f_fft, scale_to_bias_instability
        >>> from core.sensors import allan_variance
        >>> 
        >>> # Generate unit pink noise
        >>> fs = 100.0  # Hz
        >>> N = 360000  # 1 hour at 100 Hz
        >>> pink_unit = pink_noise_1f_fft(N, fs)
        >>> 
        >>> # Target BI: 10 deg/hr
        >>> bi_deg_hr = 10.0
        >>> target_bi_rad_s = np.deg2rad(bi_deg_hr) / 3600.0
        >>> 
        >>> # Create tau grid for Allan deviation
        >>> tau_grid = np.logspace(0, 3, 50)  # 1s to 1000s
        >>> 
        >>> # Scale pink noise to match target BI
        >>> pink_scaled = scale_to_bias_instability(
        ...     pink_unit, target_bi_rad_s, allan_variance, tau_grid, fs
        ... )
        >>> 
        >>> # Verify: compute Allan deviation and check minimum
        >>> taus, sigma = allan_variance(pink_scaled, fs, tau_grid)
        >>> sigma_min = np.min(sigma)
        >>> bi_recovered = sigma_min / 0.664
        >>> print(f"Target BI: {target_bi_rad_s:.2e} rad/s")
        >>> print(f"Recovered BI: {bi_recovered:.2e} rad/s")

    Notes:
        - The bi_factor of 0.664 is conventional from Allan variance analysis
        - Actual sigma_min may vary slightly due to finite data length
        - For best results, use long pink noise sequences (hours)
        - tau_grid should span the expected BI region (typically 10-1000s)
    """
    # Compute Allan deviation for unit pink noise
    taus, sigma = allan_sigma_func(pink_unit, fs, tau_grid_s)
    sigma_min = np.min(sigma)

    if sigma_min <= 0:
        raise ValueError(
            "Allan sigma_min <= 0; check allan_sigma_func implementation."
        )

    # Scale so that: target_bi_rad_s ≈ sigma_min_scaled / bi_factor
    # => sigma_min_scaled ≈ target_bi_rad_s * bi_factor
    scale = (target_bi_rad_s * bi_factor) / sigma_min

    return pink_unit * scale

