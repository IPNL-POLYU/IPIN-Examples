"""
IMU calibration utilities: Allan variance and noise characterization (Chapter 6).

This module implements calibration and characterization tools for IMU sensors:
    - Allan variance analysis (Eqs. (6.56)-(6.58))
    - Noise parameter extraction from Allan deviation plots
    - IMU scale/misalignment correction (Eq. (6.59) - implemented in imu_models.py)

Allan variance is the gold standard for characterizing IMU noise sources:
    - Quantization noise
    - Angle/velocity random walk
    - Bias instability
    - Rate random walk
    - Rate ramp

The Allan variance technique analyzes how the variance of sensor output
changes with averaging time (tau), revealing different noise processes.

References:
    Chapter 6, Section 6.5: IMU calibration
    Eq. (6.56): Cluster averages (binning)
    Eq. (6.57): Allan variance definition
    Eq. (6.58): Allan deviation (square root of variance)
    IEEE Std 952-1997: Allan variance standard
"""

from typing import Tuple, Optional
import numpy as np
import warnings


def allan_variance(
    x: np.ndarray,
    fs: float,
    taus: Optional[np.ndarray] = None,
    overlapping: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Allan variance (and Allan deviation) for IMU data.

    Implements Eqs. (6.56)-(6.58) in Chapter 6:
        1. Cluster/bin the data into averaging intervals tau (Eq. 6.56)
        2. Compute Allan variance: σ²(τ) = (1/2) E[(θ̄_{k+1} - θ̄_k)²] (Eq. 6.57)
        3. Allan deviation: σ(τ) = √(σ²(τ)) (Eq. 6.58)

    where:
        x: time series of sensor measurements (e.g., gyro, accel)
        τ (tau): averaging time (cluster duration)
        θ̄_k: average of cluster k
        σ²(τ): Allan variance at averaging time τ
        σ(τ): Allan deviation (ADEV)

    Allan variance reveals noise characteristics:
        - Quantization noise: slope -1 on log-log plot
        - Angle/velocity random walk: slope -1/2
        - Bias instability: flat region (minimum of curve)
        - Rate random walk: slope +1/2
        - Rate ramp: slope +1

    Args:
        x: Time series of sensor measurements.
           Shape: (N,). Units: rad/s (gyro) or m/s² (accel).
           Should be high-rate stationary data (e.g., 100-1000 Hz, several hours).
        fs: Sampling frequency.
            Units: Hz. Must match the data rate.
        taus: Array of averaging times (tau values) to evaluate.
              Shape: (M,). Units: seconds.
              If None, auto-generate logarithmically spaced taus.
              Typical range: [1/fs, N/(10*fs)].
        overlapping: If True, use overlapping clusters (more data, better statistics).
                     If False, use non-overlapping clusters (faster, independent samples).
                     Default: True (recommended for better noise floor).

    Returns:
        Tuple of (taus, adev):
            taus: Array of averaging times used.
                  Shape: (M,). Units: seconds.
            adev: Allan deviation σ(τ) at each tau.
                  Shape: (M,). Units: match input (rad/s or m/s²).

    Notes:
        - Requires stationary data (sensor at rest, stable temperature).
        - Longer datasets (hours) give better characterization of long-term noise.
        - Log-log plot of (tau, adev) reveals noise characteristics.
        - Overlapping gives sqrt(2) improvement in noise floor.
        - For gyros: units are rad/s, identify bias instability and random walk.
        - For accels: units are m/s², identify bias and velocity random walk.

    Example:
        >>> import numpy as np
        >>> # Simulate gyro data with bias instability and random walk
        >>> fs = 100.0  # Hz
        >>> duration = 3600  # 1 hour
        >>> N = int(fs * duration)
        >>> t = np.arange(N) / fs
        >>> 
        >>> # White noise + bias drift
        >>> gyro = 0.001 * np.random.randn(N)  # angle random walk
        >>> gyro += 0.01 * np.cumsum(np.random.randn(N)) / fs  # bias drift
        >>> 
        >>> # Compute Allan deviation
        >>> taus, adev = allan_variance(gyro, fs)
        >>> 
        >>> # Plot (log-log) to identify noise sources
        >>> # import matplotlib.pyplot as plt
        >>> # plt.loglog(taus, adev)
        >>> # plt.xlabel('Averaging time τ [s]')
        >>> # plt.ylabel('Allan deviation [rad/s]')
        >>> # plt.grid(True, which='both')

    Related Equations:
        - Eq. (6.56): Cluster averages (binning)
        - Eq. (6.57): Allan variance definition
        - Eq. (6.58): Allan deviation (THIS FUNCTION)
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    N = len(x)
    dt = 1.0 / fs

    # Generate tau values if not provided
    if taus is None:
        # Logarithmically spaced from 1 sample to N/10 samples
        tau_min = dt  # minimum averaging time (1 sample)
        tau_max = N * dt / 10.0  # maximum averaging time (10% of data length)
        num_taus = 100  # number of tau values to evaluate
        taus = np.logspace(np.log10(tau_min), np.log10(tau_max), num_taus)

    # Ensure taus are within valid range
    taus = taus[taus >= dt]
    taus = taus[taus <= N * dt / 2.0]

    if len(taus) == 0:
        raise ValueError("No valid tau values in specified range")

    adev_values = np.zeros(len(taus))

    for i, tau in enumerate(taus):
        # Number of samples per cluster (Eq. 6.56)
        m = int(np.round(tau * fs))
        m = max(1, m)  # at least 1 sample per cluster

        if overlapping:
            # Overlapping Allan variance (better statistics)
            # Number of overlapping clusters
            num_clusters = N - 2 * m + 1

            if num_clusters < 1:
                adev_values[i] = np.nan
                continue

            # Compute cluster averages (Eq. 6.56): θ̄_k = (1/m) Σ x[j]
            # Using cumsum for efficiency
            cumsum = np.cumsum(np.concatenate(([0], x)))
            cluster_avgs = (cumsum[m:] - cumsum[:-m]) / m

            # Allan variance (Eq. 6.57): σ²(τ) = (1/2) E[(θ̄_{k+1} - θ̄_k)²]
            diffs = cluster_avgs[m:] - cluster_avgs[:-m]
            allan_var = np.mean(diffs**2) / 2.0

        else:
            # Non-overlapping Allan variance (faster, independent samples)
            # Number of non-overlapping clusters
            num_clusters = N // m

            if num_clusters < 2:
                adev_values[i] = np.nan
                continue

            # Reshape into clusters and compute averages
            x_trimmed = x[: num_clusters * m]
            clusters = x_trimmed.reshape(num_clusters, m)
            cluster_avgs = np.mean(clusters, axis=1)  # Eq. 6.56

            # Allan variance (Eq. 6.57)
            diffs = np.diff(cluster_avgs)
            allan_var = np.mean(diffs**2) / 2.0

        # Allan deviation (Eq. 6.58): σ(τ) = √(σ²(τ))
        adev_values[i] = np.sqrt(allan_var)

    # Remove any NaN values (from invalid taus)
    valid_mask = ~np.isnan(adev_values)
    taus = taus[valid_mask]
    adev_values = adev_values[valid_mask]

    return taus, adev_values


def identify_bias_instability(
    taus: np.ndarray,
    adev: np.ndarray,
) -> Tuple[float, float]:
    """
    Identify bias instability from Allan deviation curve.

    Bias instability appears as the minimum (flat region) of the Allan
    deviation curve on a log-log plot. It represents the best achievable
    sensor stability over medium time scales (typically 10-1000 seconds).

    Args:
        taus: Averaging times from allan_variance().
              Shape: (M,). Units: seconds.
        adev: Allan deviation values from allan_variance().
              Shape: (M,). Units: rad/s (gyro) or m/s² (accel).

    Returns:
        Tuple of (bias_instability, tau_at_min):
            bias_instability: Minimum Allan deviation (bias instability value).
                              Units: match adev input.
            tau_at_min: Averaging time at which minimum occurs.
                        Units: seconds.

    Notes:
        - Bias instability is a key specification for IMU quality.
        - Consumer IMUs: ~0.01-0.1 deg/s (gyro), ~0.001-0.01 m/s² (accel).
        - Tactical IMUs: ~0.001-0.01 deg/s (gyro), ~0.0001-0.001 m/s² (accel).
        - Navigation grade: <0.001 deg/s (gyro), <0.0001 m/s² (accel).

    Example:
        >>> taus, adev = allan_variance(gyro_data, fs=100.0)
        >>> bias_instab, tau_bi = identify_bias_instability(taus, adev)
        >>> print(f"Bias instability: {np.rad2deg(bias_instab):.4f} deg/s at τ={tau_bi:.1f}s")
    """
    if len(taus) != len(adev):
        raise ValueError("taus and adev must have same length")
    if len(taus) == 0:
        raise ValueError("Empty arrays provided")

    # Find minimum Allan deviation
    min_idx = np.argmin(adev)
    bias_instability = adev[min_idx]
    tau_at_min = taus[min_idx]

    return bias_instability, tau_at_min


def identify_random_walk(
    taus: np.ndarray,
    adev: np.ndarray,
    tau_target: float = 1.0,
) -> float:
    """
    Identify angle/velocity random walk coefficient from Allan deviation.

    Angle random walk (ARW) or velocity random walk (VRW) appears as a
    slope of -1/2 on the log-log Allan deviation plot at short averaging
    times. It characterizes the white noise in the sensor output.

    The random walk coefficient N is found from:
        σ(τ=1s) = N

    where N has units:
        - rad/√s for gyro (angle random walk)
        - m/(s^(3/2)) for accel (velocity random walk)

    Args:
        taus: Averaging times from allan_variance().
              Shape: (M,). Units: seconds.
        adev: Allan deviation values from allan_variance().
              Shape: (M,). Units: rad/s (gyro) or m/s² (accel).
        tau_target: Target averaging time for evaluation.
                    Units: seconds. Default: 1.0 second.

    Returns:
        Random walk coefficient N.
        Units: rad/√s (gyro) or m/s^(3/2) (accel).

    Notes:
        - Random walk is read at τ = 1 second on the -1/2 slope region.
        - For gyro: angle random walk (ARW) in rad/√s or deg/√hr.
        - For accel: velocity random walk (VRW) in m/s/√s.
        - Conversion: 1 rad/√s = 3437.75 deg/√hr.
        - Smaller random walk = better short-term accuracy.

    Example:
        >>> taus, adev = allan_variance(gyro_data, fs=100.0)
        >>> arw = identify_random_walk(taus, adev, tau_target=1.0)
        >>> print(f"Angle random walk: {np.rad2deg(arw):.4f} deg/√s")
        >>> print(f"                   {np.rad2deg(arw)*60:.2f} deg/√hr")
    """
    if len(taus) != len(adev):
        raise ValueError("taus and adev must have same length")
    if len(taus) == 0:
        raise ValueError("Empty arrays provided")

    # Find Allan deviation at tau_target (interpolate if needed)
    # Random walk coefficient: N = σ(τ) * √τ (for -1/2 slope region)
    # At τ = 1s: N = σ(1s)

    # Find closest tau or interpolate
    idx = np.argmin(np.abs(taus - tau_target))

    # For better accuracy, fit a line in log-log space near tau_target
    # and extrapolate to tau_target
    # Select region around tau_target (e.g., tau_target/3 to 3*tau_target)
    tau_min = tau_target / 3.0
    tau_max = tau_target * 3.0
    mask = (taus >= tau_min) & (taus <= tau_max)

    if np.sum(mask) >= 2:
        # Fit line in log-log space: log(adev) = a + b*log(tau)
        # For -1/2 slope: b should be close to -0.5
        log_taus = np.log10(taus[mask])
        log_adev = np.log10(adev[mask])

        # Linear fit
        coeffs = np.polyfit(log_taus, log_adev, deg=1)
        # Extrapolate to tau_target
        log_adev_at_target = coeffs[0] * np.log10(tau_target) + coeffs[1]
        adev_at_target = 10**log_adev_at_target

        # Check if slope is reasonable for random walk (-0.7 to -0.3)
        slope = coeffs[0]
        if not (-0.7 < slope < -0.3):
            warnings.warn(
                f"Slope {slope:.2f} is not typical for random walk (-0.5 expected). "
                "Result may be inaccurate.",
                UserWarning,
            )
    else:
        # Not enough points, use nearest neighbor
        adev_at_target = adev[idx]
        warnings.warn(
            f"Insufficient data near τ={tau_target}s. Using nearest value.",
            UserWarning,
        )

    # Random walk coefficient (units: rad/√s or m/s^(3/2))
    N = adev_at_target

    return N


def identify_rate_random_walk(
    taus: np.ndarray,
    adev: np.ndarray,
    tau_target: float = 100.0,
) -> float:
    """
    Identify rate random walk coefficient from Allan deviation.

    Rate random walk (RRW) appears as a slope of +1/2 on the log-log
    Allan deviation plot at long averaging times. It characterizes the
    random walk in the sensor bias.

    The rate random walk coefficient K is found from:
        σ(τ) = K * √(τ/3)

    where K has units:
        - rad/(s^(3/2)) for gyro
        - m/(s^(5/2)) for accel

    Args:
        taus: Averaging times from allan_variance().
              Shape: (M,). Units: seconds.
        adev: Allan deviation values from allan_variance().
              Shape: (M,). Units: rad/s (gyro) or m/s² (accel).
        tau_target: Target averaging time for evaluation.
                    Units: seconds. Default: 100.0 seconds.

    Returns:
        Rate random walk coefficient K.
        Units: rad/s^(3/2) (gyro) or m/s^(5/2) (accel).

    Notes:
        - Rate random walk characterizes bias instability at long time scales.
        - Read at large τ (e.g., 100-1000 seconds) on +1/2 slope region.
        - K = σ(τ) * √(3/τ)
        - Requires long datasets (hours) to characterize accurately.

    Example:
        >>> taus, adev = allan_variance(gyro_data, fs=100.0)
        >>> rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)
        >>> print(f"Rate random walk: {np.rad2deg(rrw):.6f} deg/s^(3/2)")
    """
    if len(taus) != len(adev):
        raise ValueError("taus and adev must have same length")
    if len(taus) == 0:
        raise ValueError("Empty arrays provided")

    # Find Allan deviation at tau_target
    idx = np.argmin(np.abs(taus - tau_target))

    # Fit line in log-log space around tau_target
    tau_min = tau_target / 3.0
    tau_max = tau_target * 3.0
    mask = (taus >= tau_min) & (taus <= tau_max)

    if np.sum(mask) >= 2:
        log_taus = np.log10(taus[mask])
        log_adev = np.log10(adev[mask])

        coeffs = np.polyfit(log_taus, log_adev, deg=1)
        log_adev_at_target = coeffs[0] * np.log10(tau_target) + coeffs[1]
        adev_at_target = 10**log_adev_at_target

        # Check slope (should be around +0.5 for rate random walk)
        slope = coeffs[0]
        if not (0.3 < slope < 0.7):
            warnings.warn(
                f"Slope {slope:.2f} is not typical for rate random walk (+0.5 expected). "
                "Result may be inaccurate.",
                UserWarning,
            )
    else:
        adev_at_target = adev[idx]
        warnings.warn(
            f"Insufficient data near τ={tau_target}s. Using nearest value.",
            UserWarning,
        )

    # Rate random walk coefficient: K = σ(τ) * √(3/τ)
    K = adev_at_target * np.sqrt(3.0 / tau_target)

    return K


def characterize_imu_noise(
    gyro_data: np.ndarray,
    accel_data: np.ndarray,
    fs: float,
) -> dict:
    """
    Complete IMU noise characterization using Allan variance.

    Analyzes both gyroscope and accelerometer data to extract key noise
    parameters: random walk, bias instability, and rate random walk.

    Args:
        gyro_data: Gyroscope measurements (3-axis or single axis).
                   Shape: (N, 3) or (N,). Units: rad/s.
                   Should be stationary data (several hours recommended).
        accel_data: Accelerometer measurements (3-axis or single axis).
                    Shape: (N, 3) or (N,). Units: m/s².
                    Should be stationary data.
        fs: Sampling frequency.
            Units: Hz.

    Returns:
        Dictionary containing noise characterization results:
            {
                'gyro': {
                    'angle_random_walk': float,  # rad/√s
                    'bias_instability': float,   # rad/s
                    'rate_random_walk': float,   # rad/s^(3/2)
                    'taus': np.ndarray,          # s
                    'adev': np.ndarray,          # rad/s
                },
                'accel': {
                    'velocity_random_walk': float,  # m/s^(3/2)
                    'bias_instability': float,      # m/s²
                    'rate_random_walk': float,      # m/s^(5/2)
                    'taus': np.ndarray,             # s
                    'adev': np.ndarray,             # m/s²
                },
            }

    Example:
        >>> # Collect stationary IMU data for several hours
        >>> noise_params = characterize_imu_noise(gyro, accel, fs=100.0)
        >>> print(f"Gyro ARW: {np.rad2deg(noise_params['gyro']['angle_random_walk'])*60:.2f} deg/√hr")
        >>> print(f"Gyro BI:  {np.rad2deg(noise_params['gyro']['bias_instability']):.4f} deg/s")
    """
    results = {}

    # Process gyroscope
    if gyro_data.ndim == 2:
        # Multi-axis: analyze each axis separately and report average
        arw_list = []
        bi_list = []
        rrw_list = []
        for axis in range(gyro_data.shape[1]):
            taus, adev = allan_variance(gyro_data[:, axis], fs)
            arw = identify_random_walk(taus, adev, tau_target=1.0)
            bi, _ = identify_bias_instability(taus, adev)
            rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

            arw_list.append(arw)
            bi_list.append(bi)
            rrw_list.append(rrw)

        # Use first axis for plotting, average for parameters
        taus, adev = allan_variance(gyro_data[:, 0], fs)
        results["gyro"] = {
            "angle_random_walk": np.mean(arw_list),
            "bias_instability": np.mean(bi_list),
            "rate_random_walk": np.mean(rrw_list),
            "taus": taus,
            "adev": adev,
        }
    else:
        # Single axis
        taus, adev = allan_variance(gyro_data, fs)
        arw = identify_random_walk(taus, adev, tau_target=1.0)
        bi, _ = identify_bias_instability(taus, adev)
        rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

        results["gyro"] = {
            "angle_random_walk": arw,
            "bias_instability": bi,
            "rate_random_walk": rrw,
            "taus": taus,
            "adev": adev,
        }

    # Process accelerometer (similar to gyro)
    if accel_data.ndim == 2:
        vrw_list = []
        bi_list = []
        rrw_list = []
        for axis in range(accel_data.shape[1]):
            taus, adev = allan_variance(accel_data[:, axis], fs)
            vrw = identify_random_walk(taus, adev, tau_target=1.0)
            bi, _ = identify_bias_instability(taus, adev)
            rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

            vrw_list.append(vrw)
            bi_list.append(bi)
            rrw_list.append(rrw)

        taus, adev = allan_variance(accel_data[:, 0], fs)
        results["accel"] = {
            "velocity_random_walk": np.mean(vrw_list),
            "bias_instability": np.mean(bi_list),
            "rate_random_walk": np.mean(rrw_list),
            "taus": taus,
            "adev": adev,
        }
    else:
        taus, adev = allan_variance(accel_data, fs)
        vrw = identify_random_walk(taus, adev, tau_target=1.0)
        bi, _ = identify_bias_instability(taus, adev)
        rrw = identify_rate_random_walk(taus, adev, tau_target=100.0)

        results["accel"] = {
            "velocity_random_walk": vrw,
            "bias_instability": bi,
            "rate_random_walk": rrw,
            "taus": taus,
            "adev": adev,
        }

    return results

