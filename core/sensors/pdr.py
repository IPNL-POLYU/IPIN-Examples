"""
Pedestrian Dead Reckoning (PDR) algorithms (Chapter 6).

This module implements step-and-heading based pedestrian navigation:
    - Acceleration magnitude computation (Eq. (6.46))
    - Gravity removal from magnitude (Eq. (6.47))
    - Step frequency estimation (Eq. (6.48))
    - Step length estimation (Eq. (6.49))
    - 2D position update from step (Eq. (6.50))

PDR is a lightweight dead reckoning approach for pedestrians that uses:
    - Step detection from accelerometer magnitude peaks
    - Step length models (empirical formulas based on height and frequency)
    - Heading from gyro integration or magnetometer

Frame Conventions:
    - B: Body frame (IMU/phone frame)
    - M: Map frame (horizontal plane, ENU or NED defined by FrameConvention)
    - PDR operates primarily in 2D horizontal plane
    - Heading convention must match frame (0=East for ENU, 0=North for NED)

All position update functions accept an optional FrameConvention parameter.

References:
    Chapter 6, Section 6.3: Pedestrian dead reckoning
    Eq. (6.46): Total acceleration magnitude
    Eq. (6.47): Gravity-removed magnitude
    Eq. (6.48): Step frequency
    Eq. (6.49): Step length (Weinberg model)
    Eq. (6.50): 2D position update
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal

# Import FrameConvention for type hints
from core.sensors.types import FrameConvention
from core.sensors.gravity import gravity_magnitude


def total_accel_magnitude(accel_b: np.ndarray) -> float:
    """
    Compute total acceleration magnitude from 3D accelerometer measurement.

    Implements Eq. (6.46) in Chapter 6:
        a_mag = ||a|| = √(ax² + ay² + az²)

    where a = [ax, ay, az]^T is the accelerometer measurement in body frame B.

    This magnitude is used for step detection: peaks in a_mag correspond to
    foot strikes (or hand swings for phone-based PDR). The magnitude removes
    dependence on device orientation.

    Args:
        accel_b: Accelerometer measurement in body frame B.
                 Shape: (3,). Units: m/s².
                 Raw measurement (includes gravity).

    Returns:
        Acceleration magnitude a_mag.
        Units: m/s². Scalar (always non-negative).

    Notes:
        - For stationary device: a_mag ≈ g = 9.81 m/s² (gravity only).
        - During walking: a_mag oscillates around g with amplitude ~2-5 m/s².
        - Step detection typically uses peak finding on smoothed a_mag time series.
        - This is the L2 norm (Euclidean norm) of the acceleration vector.

    Example:
        >>> import numpy as np
        >>> # Stationary: only gravity (pointing down in body frame)
        >>> accel = np.array([0.0, 0.0, -9.81])
        >>> mag = total_accel_magnitude(accel)
        >>> print(f"{mag:.2f}")  # 9.81
        >>> 
        >>> # Walking: includes motion + gravity
        >>> accel = np.array([1.5, 0.5, -10.2])
        >>> mag = total_accel_magnitude(accel)
        >>> print(f"{mag:.2f}")  # ~10.37

    Related Equations:
        - Eq. (6.46): Total acceleration magnitude (THIS FUNCTION)
        - Eq. (6.47): Gravity removal (see remove_gravity_from_magnitude)
    """
    if accel_b.shape != (3,):
        raise ValueError(f"accel_b must have shape (3,), got {accel_b.shape}")

    # Eq. (6.46): a_mag = ||a|| = sqrt(ax^2 + ay^2 + az^2)
    a_mag = np.linalg.norm(accel_b)

    return a_mag


def remove_gravity_from_magnitude(
    a_mag: float,
    g: float = 9.81,
    lat_rad: Optional[float] = None,
) -> float:
    """
    Remove gravity component from acceleration magnitude.

    Implements Eq. (6.47) in Chapter 6:
        a_dynamic = a_mag - g

    where:
        a_mag: total acceleration magnitude (Eq. 6.46) [m/s²]
        g: gravity magnitude (from Eq. 6.8 if latitude provided) [m/s²]
        a_dynamic: dynamic acceleration magnitude [m/s²]

    This approximation assumes that the static component (gravity) can be
    removed by simple subtraction. More accurate methods would account for
    device tilt, but this simple approach works reasonably well for PDR.

    Args:
        a_mag: Total acceleration magnitude from total_accel_magnitude().
               Units: m/s². Must be non-negative.
        g: Gravity magnitude (fallback when lat_rad=None).
           Units: m/s². Default: 9.81 m/s² (standard gravity).
        lat_rad: Geodetic latitude in radians (optional).
                 If provided, uses Eq. (6.8) for gravity magnitude.
                 If None, uses g parameter (backward compatible).

    Returns:
        Dynamic acceleration magnitude (gravity removed).
        Units: m/s². Can be negative if a_mag < g.

    Notes:
        - For stationary: a_dynamic ≈ 0 (a_mag ≈ g).
        - For walking: a_dynamic oscillates around 0 with peaks during steps.
        - This is an approximation; exact gravity removal requires attitude.
        - Negative values can occur during "free fall" phases of gait.
        - Typically used after high-pass filtering or smoothing.

    Example:
        >>> # Stationary
        >>> a_mag_stationary = 9.81
        >>> a_dyn = remove_gravity_from_magnitude(a_mag_stationary)
        >>> print(f"{a_dyn:.2f}")  # 0.00
        >>> 
        >>> # Walking (peak acceleration)
        >>> a_mag_walking = 12.0
        >>> a_dyn = remove_gravity_from_magnitude(a_mag_walking)
        >>> print(f"{a_dyn:.2f}")  # 2.19

    Related Equations:
        - Eq. (6.46): Total acceleration magnitude
        - Eq. (6.47): Gravity removal (THIS FUNCTION)
        - Eq. (6.48): Step frequency (uses processed a_dynamic)
        - Eq. (6.8): Gravity magnitude (used when latitude provided)
    """
    # Eq. (6.47): a_dynamic = a_mag - g
    # where g is from Eq. (6.8) if latitude provided
    g_mag = gravity_magnitude(lat_rad=lat_rad, default_g=g)
    a_dynamic = a_mag - g_mag

    return a_dynamic


def detect_steps_peak_detector(
    accel_series: np.ndarray,
    dt: float,
    g: float = 9.81,
    min_peak_height: float = 1.0,
    min_peak_distance: float = 0.3,
    lowpass_cutoff: Optional[float] = 5.0,
    lat_rad: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect steps using peak detection on gravity-removed acceleration magnitude.
    
    Implements the book's approach (Eqs. 6.46-6.47):
    1. Compute total acceleration magnitude (Eq. 6.46)
    2. Remove gravity (Eq. 6.47)
    3. Optionally apply low-pass filter
    4. Find peaks with minimum height and distance constraints
    
    This is the proper peak detection method described in Chapter 6, Section 6.3.2.
    
    Args:
        accel_series: Accelerometer time series in body frame.
                      Shape: (N, 3). Units: m/s².
                      Must include gravity (raw measurements).
        dt: Time step between samples. Units: seconds.
        g: Gravity magnitude (fallback when lat_rad=None).
           Default: 9.81 m/s². Units: m/s².
        min_peak_height: Minimum height of peaks above zero (after gravity removal).
                         Units: m/s². Default: 1.0 m/s².
                         Typical range: 0.5-2.0 m/s².
        min_peak_distance: Minimum time between peaks (refractory period).
                           Units: seconds. Default: 0.3 s.
                           Typical range: 0.2-0.5 s (corresponds to 2-5 steps/s).
        lowpass_cutoff: Low-pass filter cutoff frequency. Units: Hz.
                        Default: 5.0 Hz. Set to None to disable filtering.
                        Typical range: 3-10 Hz.
        lat_rad: Geodetic latitude in radians (optional).
                 If provided, uses Eq. (6.8) for gravity magnitude.
                 If None, uses g parameter (backward compatible).
    
    Returns:
        Tuple of (step_indices, accel_mag_filtered):
            step_indices: Indices of detected steps in accel_series.
                          Shape: (n_steps,). These are the peak locations.
            accel_mag_filtered: Processed acceleration magnitude time series.
                                Shape: (N,). Units: m/s².
                                After gravity removal and optional filtering.
    
    Notes:
        - Follows Eq. (6.46): Compute ||a|| = sqrt(ax² + ay² + az²)
        - Follows Eq. (6.47): Remove gravity: a_dynamic = ||a|| - g
        - Uses scipy.signal.find_peaks for robust peak detection
        - min_peak_distance prevents detecting same step multiple times
        - Low-pass filter reduces high-frequency noise from sensors
        - Peaks correspond to foot strikes or hand swings (device-dependent)
    
    Example:
        >>> import numpy as np
        >>> # Synthetic walking: 60s at 2 Hz step frequency
        >>> t = np.arange(0, 60, 0.01)  # 100 Hz sampling
        >>> # Simulate vertical acceleration with steps
        >>> accel_z = -9.81 + 2.0 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz steps
        >>> accel = np.column_stack([np.zeros_like(t), np.zeros_like(t), accel_z])
        >>> 
        >>> step_indices, accel_processed = detect_steps_peak_detector(
        ...     accel, dt=0.01, min_peak_height=0.5, min_peak_distance=0.4
        ... )
        >>> print(f"Detected {len(step_indices)} steps in 60s")
        >>> print(f"Expected ~120 steps (2 steps/s * 60s)")
    
    Related Equations:
        - Eq. (6.46): Total acceleration magnitude
        - Eq. (6.47): Gravity removal
        - Eq. (6.48): Step frequency (computed from detected peaks)
    
    References:
        Chapter 6, Section 6.3.2: Pedestrian Dead Reckoning
        Figure 6.12: Total accelerations during walking (shows peak pattern)
    """
    if accel_series.ndim != 2 or accel_series.shape[1] != 3:
        raise ValueError(
            f"accel_series must have shape (N, 3), got {accel_series.shape}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if min_peak_distance <= 0:
        raise ValueError(f"min_peak_distance must be positive, got {min_peak_distance}")
    
    N = len(accel_series)
    
    # Step 1: Compute total acceleration magnitude (Eq. 6.46)
    # a_mag[k] = ||a_k|| = sqrt(ax² + ay² + az²)
    accel_mag = np.linalg.norm(accel_series, axis=1)  # Shape: (N,)
    
    # Step 2: Remove gravity (Eq. 6.47)
    # a_dynamic[k] = a_mag[k] - g
    # where g is from Eq. (6.8) if latitude provided
    g_mag = gravity_magnitude(lat_rad=lat_rad, default_g=g)
    accel_dynamic = accel_mag - g_mag  # Shape: (N,)
    
    # Step 3: Optional low-pass filter to reduce noise
    if lowpass_cutoff is not None:
        # Design Butterworth low-pass filter
        fs = 1.0 / dt  # Sampling frequency
        nyquist = fs / 2.0
        normalized_cutoff = lowpass_cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            # Cutoff too high, skip filtering
            accel_filtered = accel_dynamic
        else:
            # Apply 4th order Butterworth filter
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            accel_filtered = signal.filtfilt(b, a, accel_dynamic)
    else:
        accel_filtered = accel_dynamic
    
    # Step 4: Find peaks using scipy.signal.find_peaks
    # Convert min_peak_distance from seconds to samples
    min_distance_samples = int(min_peak_distance / dt)
    
    # Find peaks with constraints
    peak_indices, peak_properties = signal.find_peaks(
        accel_filtered,
        height=min_peak_height,  # Minimum peak height
        distance=min_distance_samples,  # Minimum distance between peaks
    )
    
    return peak_indices, accel_filtered


def step_frequency(delta_t: float) -> float:
    """
    Compute step frequency from inter-step time interval.

    Implements Eq. (6.48) in Chapter 6:
        f_step = 1 / Δt

    where:
        Δt: time between consecutive steps (inter-step interval) [s]
        f_step: step frequency [Hz, or steps/second]

    Step frequency is inversely related to walking speed: faster walking
    produces higher frequency. Typical walking: 1.5-2.5 Hz (90-150 steps/min).

    Args:
        delta_t: Time interval between consecutive steps.
                 Units: seconds. Must be positive.
                 Typical range: 0.4-0.8 s (1.25-2.5 Hz).

    Returns:
        Step frequency f_step.
        Units: Hz (steps per second). Always positive.

    Notes:
        - Typical walking: 1.5-2.0 Hz (90-120 steps/min).
        - Fast walking/jogging: 2.0-3.0 Hz (120-180 steps/min).
        - Running: 2.5-4.0 Hz (150-240 steps/min).
        - Used in step length models (Eq. 6.49).
        - Can be estimated from peak-to-peak intervals in a_mag signal.

    Example:
        >>> # Normal walking: 0.5 s between steps
        >>> dt = 0.5
        >>> freq = step_frequency(dt)
        >>> print(f"{freq:.2f} Hz")  # 2.00 Hz (120 steps/min)
        >>> 
        >>> # Slow walking: 0.7 s between steps
        >>> dt = 0.7
        >>> freq = step_frequency(dt)
        >>> print(f"{freq:.2f} Hz")  # 1.43 Hz (86 steps/min)

    Related Equations:
        - Eq. (6.48): Step frequency (THIS FUNCTION)
        - Eq. (6.49): Step length (uses f_step as input)
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")

    # Eq. (6.48): f_step = 1 / Δt
    f_step = 1.0 / delta_t

    return f_step


def step_length(
    h: float,
    f_step: float,
    a: float = 0.371,
    b: float = 0.227,
    c: float = 1.0,
) -> float:
    """
    Estimate step length using empirical power-law model.

    **DEPRECATED**: This function implements a generic power-law formula
    L = c * h^a * f^b that is neither the book's Eq. (6.49) nor the actual
    Weinberg model. Use `step_length_book_eq6_49()` or `step_length_weinberg()`
    for mathematically correct implementations.

    Formula:
        L = c * h^a * f_step^b

    where:
        L: step length [m]
        h: user height [m]
        f_step: step frequency [Hz]
        a, b, c: model parameters (empirically determined)

    This empirical model relates step length to user height and step frequency.
    Typical values: a ≈ 0.371, b ≈ 0.227, c ≈ 1.0 (varies by user and gait).

    Args:
        h: User height.
           Units: meters. Typical range: 1.5-2.0 m.
        f_step: Step frequency.
                Units: Hz (steps per second). Typical range: 1.5-3.0 Hz.
        a: Height exponent. Default: 0.371 (from literature).
        b: Frequency exponent. Default: 0.227 (from literature).
        c: Scaling constant. Default: 1.0 (personal calibration factor).

    Returns:
        Step length L.
        Units: meters. Typical range: 0.5-1.0 m.

    Notes:
        - Model parameters (a, b, c) are typically calibrated per user.
        - Alternative models exist (e.g., constant step length, linear models).
        - Accuracy: ±10-20% without calibration, ±5% with calibration.
        - c is a personal factor that accounts for individual gait patterns.
        - Higher frequency (faster walking) → longer steps (nonlinear).
        - Taller people → longer steps (nonlinear).

    Example:
        >>> # Person: 1.75m tall, walking at 2 Hz
        >>> height = 1.75  # m
        >>> freq = 2.0  # Hz
        >>> L = step_length(height, freq)
        >>> print(f"Step length: {L:.2f} m")  # ~0.73 m

    Recommended Alternatives:
        - `step_length_book_eq6_49()`: Matches book Eq. (6.49) exactly
        - `step_length_weinberg()`: Actual Weinberg model with per-step ptp
    """
    if h <= 0:
        raise ValueError(f"h (height) must be positive, got {h}")
    if f_step <= 0:
        raise ValueError(f"f_step must be positive, got {f_step}")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")

    # Generic power-law (NOT Eq. 6.49 or Weinberg)
    L = c * (h**a) * (f_step**b)

    return L


def step_length_book_eq6_49(
    h: float,
    SF: float,
    a: float = 0.371,
    b: float = 0.227,
    c: float = 1.0,
    h_ref: float = 1.75,
    SF_ref: float = 1.79,
    L_offset: float = 0.7,
) -> float:
    """
    Estimate step length using book Eq. (6.49).

    Implements the actual formula from Chapter 6, Eq. (6.49):
        L = L_offset + c * ((h / h_ref)^a) * ((SF / SF_ref)^b)

    where:
        L: step length [m]
        h: user height [m]
        SF: step frequency [Hz]
        h_ref: reference height (1.75 m)
        SF_ref: reference step frequency (1.79 Hz)
        L_offset: base step length (0.7 m)
        a, b, c: empirical exponents

    This is a height + frequency empirical model with an offset and reference
    normalization terms, NOT a simple power-law.

    Args:
        h: User height.
           Units: meters. Typical range: 1.5-2.0 m.
        SF: Step frequency.
            Units: Hz (steps per second). Typical range: 1.5-3.0 Hz.
        a: Height exponent. Default: 0.371 (book value).
        b: Frequency exponent. Default: 0.227 (book value).
        c: Scaling constant. Default: 1.0 (personal calibration factor).
        h_ref: Reference height. Default: 1.75 m (book value).
        SF_ref: Reference step frequency. Default: 1.79 Hz (book value).
        L_offset: Base step length offset. Default: 0.7 m (book value).

    Returns:
        Step length L.
        Units: meters. Typical range: 0.6-1.0 m.

    Notes:
        - This matches the book's Eq. (6.49) exactly
        - The offset and reference terms make this model more physically realistic
        - At h=h_ref and SF=SF_ref with c=1, L ≈ L_offset + 1.0 m
        - Parameters can be calibrated per user for best accuracy

    Example:
        >>> # Person at reference height and frequency
        >>> h, SF = 1.75, 1.79
        >>> L = step_length_book_eq6_49(h, SF)
        >>> print(f"Step length: {L:.2f} m")  # ~1.7 m
        >>> 
        >>> # Taller person, faster walking
        >>> L_tall = step_length_book_eq6_49(h=1.90, SF=2.2)
        >>> print(f"Step length (tall): {L_tall:.2f} m")

    Related Equations:
        - Eq. (6.48): Step frequency (SF)
        - Eq. (6.49): Step length (THIS FUNCTION)
        - Eq. (6.50): Position update (uses L)
    
    Author: Li-Ta Hsu
    Date: December 2025
    """
    if h <= 0:
        raise ValueError(f"h (height) must be positive, got {h}")
    if SF <= 0:
        raise ValueError(f"SF (step frequency) must be positive, got {SF}")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if h_ref <= 0:
        raise ValueError(f"h_ref must be positive, got {h_ref}")
    if SF_ref <= 0:
        raise ValueError(f"SF_ref must be positive, got {SF_ref}")

    # Book Eq. (6.49): L = L_offset + c * (h/h_ref)^a * (SF/SF_ref)^b
    h_norm = h / h_ref
    SF_norm = SF / SF_ref
    L = L_offset + c * (h_norm**a) * (SF_norm**b)

    return L


def step_length_weinberg(
    f_step_window: np.ndarray,
    G_w: float,
    power: float = 0.25,
    eps: float = 1e-6,
) -> float:
    """
    Weinberg step-length model (peak-to-peak specific force).

    Implements the actual Weinberg model used in practice:
        SL = G_w * (max(f) - min(f))^(1/4)

    where:
        SL: step length [m]
        f: specific force (or gravity-removed accel magnitude) samples over ONE step
        G_w: gain parameter (user-calibrated, typically 0.3-0.5)
        power: exponent (default 0.25 = 1/4)

    This model relates step length to the peak-to-peak amplitude of vertical
    acceleration during a step, which correlates biomechanically with step
    energy and thus step length.

    Args:
        f_step_window: Specific force (or gravity-removed accel magnitude) samples
                       for ONE step. Shape: (M,). Units should be consistent
                       (typically m/s² or g-units).
                       Obtained from detect_steps_peak_detector() output.
        G_w: Gain parameter (user-calibrated).
             Units: depends on f_step_window units. Typical: 0.3-0.5 if f is in m/s².
        power: Exponent for ptp amplitude. Default: 0.25 (quarter-power law).
        eps: Floor value for numerical stability (avoids zero ptp). Default: 1e-6.

    Returns:
        SL: Estimated step length.
        Units: meters (if G_w is calibrated for meters).

    Raises:
        ValueError: If f_step_window is not 1D or has < 2 samples.

    Notes:
        - This is the ACTUAL Weinberg model, not a power-law on height/frequency
        - Requires per-step acceleration segments (not just frequency)
        - G_w must be calibrated using `calibrate_weinberg_gain()`
        - More accurate than frequency-only models but requires proper segmentation
        - Works best with foot-mounted or waist-worn IMUs

    Example:
        >>> # Get step segments from detector
        >>> step_indices, accel_filtered = detect_steps_peak_detector(accel, dt=0.01)
        >>> 
        >>> # Calibrate gain (assuming known distance)
        >>> ptp_per_step = []
        >>> for i in range(len(step_indices)-1):
        >>>     seg = accel_filtered[step_indices[i]:step_indices[i+1]]
        >>>     ptp_per_step.append(np.ptp(seg))
        >>> G_w = calibrate_weinberg_gain(np.array(ptp_per_step), distance_m=50.0)
        >>> 
        >>> # Compute step length per step
        >>> for i in range(len(step_indices)-1):
        >>>     seg = accel_filtered[step_indices[i]:step_indices[i+1]]
        >>>     L = step_length_weinberg(seg, G_w)
        >>>     print(f"Step {i}: {L:.2f} m")

    Related Functions:
        - `calibrate_weinberg_gain()`: Calibrate G_w from known distance
        - `detect_steps_peak_detector()`: Provides step windows and accel_filtered
    
    References:
        Weinberg, H. (2002). "Using the ADXL202 in Pedometer and Personal Navigation
        Applications." Analog Devices AN-602 Application Note.
    
    Author: Li-Ta Hsu
    Date: December 2025
    """
    if f_step_window.ndim != 1 or f_step_window.size < 2:
        raise ValueError(
            f"f_step_window must be 1D with at least 2 samples, got shape {f_step_window.shape}"
        )

    # Compute peak-to-peak amplitude (max - min)
    ptp = float(np.ptp(f_step_window))
    ptp = max(ptp, eps)  # Apply floor for numerical stability

    # Weinberg formula: SL = G_w * ptp^(1/4)
    SL = G_w * (ptp ** power)

    return SL


def calibrate_weinberg_gain(
    ptp_per_step: np.ndarray,
    distance_m: float,
    power: float = 0.25,
    eps: float = 1e-6,
) -> float:
    """
    Calibrate Weinberg gain parameter from known distance.

    Given a calibration walk with known distance D and measured per-step
    peak-to-peak amplitudes, computes the gain:

        G_w = D / sum_i(ptp_i^(1/4))

    such that sum of step lengths equals the known distance.

    Args:
        ptp_per_step: Peak-to-peak amplitudes for each step.
                      Shape: (n_steps,). Units: m/s² (or consistent with sensor).
        distance_m: Known total distance traveled.
                    Units: meters. Must be positive.
        power: Exponent for ptp (default 0.25 for quarter-power law).
        eps: Floor value for numerical stability. Default: 1e-6.

    Returns:
        G_w: Calibrated gain parameter.
        Units: meters / (sensor_units^power). Typical: 0.3-0.5 for m/s².

    Raises:
        ValueError: If distance_m <= 0 or denominator is invalid.

    Notes:
        - Requires a calibration walk with known distance (e.g., measured corridor)
        - More steps provide better calibration (recommend 20+ steps)
        - G_w is user-specific and should be recalibrated if sensor changes
        - Store G_w for future use with same user + sensor setup

    Example:
        >>> # After walking 50m in a corridor
        >>> ptp_per_step = np.array([3.2, 3.5, 3.1, 3.4, ...])  # from detector
        >>> known_distance = 50.0  # meters
        >>> G_w = calibrate_weinberg_gain(ptp_per_step, known_distance)
        >>> print(f"Calibrated gain: {G_w:.3f}")
        >>> # Save G_w for future use

    Related Functions:
        - `step_length_weinberg()`: Uses calibrated G_w
        - `detect_steps_peak_detector()`: Provides accel_filtered for ptp computation
    
    Author: Li-Ta Hsu
    Date: December 2025
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be positive, got {distance_m}")

    # Apply floor to ptp values
    ptp = np.maximum(ptp_per_step.astype(float), eps)

    # Compute denominator: sum of ptp^power
    denom = float(np.sum(ptp ** power))

    if denom <= 0:
        raise ValueError(
            f"Invalid denominator {denom}; check ptp_per_step values or eps"
        )

    # Compute gain: G_w = D / sum(ptp^power)
    G_w = distance_m / denom

    return G_w


def pdr_step_update(
    p_prev_xy: np.ndarray,
    step_len: float,
    heading_rad: float,
    frame: Optional[FrameConvention] = None,
) -> np.ndarray:
    """
    Update 2D position from a detected step (step-and-heading PDR).

    Implements Eq. (6.50) specialized to 2D (horizontal plane):
        p_k = p_{k-1} + L * [cos(ψ), sin(ψ)]^T

    where:
        p_k: position after step k [m]
        p_{k-1}: position before step [m]
        L: step length [m]
        ψ: heading angle (yaw) in horizontal plane [radians]

    This is the core update equation for step-and-heading PDR. Each detected
    step moves the position by step_len in the direction of heading.

    Args:
        p_prev_xy: Previous 2D position (before step).
                   Shape: (2,). Units: m. Components: [x, y] or [E, N].
        step_len: Step length (from step_length()).
                  Units: meters. Typical: 0.5-1.0 m.
        heading_rad: Current heading (yaw angle) in horizontal plane.
                     Units: radians.
                     Convention determined by frame:
                         ENU: 0 = East, π/2 = North (default)
                         NED: 0 = North, π/2 = East
        frame: Frame convention defining heading interpretation.
               Default: None (creates ENU).

    Returns:
        Updated 2D position p_k after the step.
        Shape: (2,). Units: m.

    Notes:
        - Assumes horizontal plane motion (no altitude change).
        - Heading can come from: gyro integration, magnetometer, or fusion.
        - Heading drift is the primary error source in PDR.
        - Step length errors accumulate linearly with number of steps.
        - Heading errors cause position error to grow quadratically.
        - Typical PDR accuracy: 2-5% of distance traveled (without constraints).

    Example:
        >>> import numpy as np
        >>> # Start at origin
        >>> p0 = np.array([0.0, 0.0])
        >>> 
        >>> # Take a step north (heading = π/2 = 90°)
        >>> L = 0.7  # m
        >>> heading = np.pi / 2  # north
        >>> p1 = pdr_step_update(p0, L, heading)
        >>> print(p1)  # [0.0, 0.7] (moved 0.7m north)
        >>> 
        >>> # Take another step northeast (heading = π/4 = 45°)
        >>> heading_ne = np.pi / 4
        >>> p2 = pdr_step_update(p1, L, heading_ne)
        >>> print(p2)  # [0.495, 1.195] (moved diagonally)

    Related Equations:
        - Eq. (6.49): Step length computation
        - Eq. (6.50): Position update (THIS FUNCTION)
        - Eqs. (6.51)-(6.53): Magnetometer heading (alternative to gyro)
    """
    if p_prev_xy.shape != (2,):
        raise ValueError(f"p_prev_xy must have shape (2,), got {p_prev_xy.shape}")
    if step_len < 0:
        raise ValueError(f"step_len must be non-negative, got {step_len}")

    if frame is None:
        frame = FrameConvention.create_enu()

    # Eq. (6.50) in 2D: p_k = p_{k-1} + L * [cos(ψ), sin(ψ)]
    # The direction vector is determined by the frame convention:
    # - ENU: heading 0 = East (+x), π/2 = North (+y)
    # - NED: heading 0 = North (+x), π/2 = East (+y)
    # Both follow the same formula: [cos(ψ), sin(ψ)] by design
    direction = frame.heading_to_unit_vector(heading_rad)
    displacement = step_len * direction
    p_next_xy = p_prev_xy + displacement

    return p_next_xy


def detect_step_simple(
    accel_mag_window: np.ndarray,
    threshold: float = 11.0,
) -> bool:
    """
    Simple step detection using peak finding in acceleration magnitude.

    Detects a step if the maximum acceleration magnitude in the window
    exceeds a threshold. This is a simplified approach; production systems
    use more sophisticated algorithms (zero-crossing, template matching).

    Args:
        accel_mag_window: Recent acceleration magnitude samples.
                          Shape: (N,). Units: m/s². N typically 10-50 samples.
        threshold: Magnitude threshold for step detection.
                   Units: m/s². Default: 11.0 m/s² (g + ~1.2 m/s² dynamic).
                   Typical range: 10.5-12.0 m/s².

    Returns:
        True if step detected (peak above threshold), False otherwise.

    Notes:
        - This is a VERY simplified detector for demonstration.
        - Production PDR uses: zero-crossing, autocorrelation, ML, etc.
        - Should include: peak prominence, inter-step timing, etc.
        - False positives from: sudden motion, bumps, stairs.
        - False negatives from: very slow walking, smooth gait.

    Example:
        >>> import numpy as np
        >>> # No motion (stationary)
        >>> mag_stationary = np.ones(20) * 9.81
        >>> step = detect_step_simple(mag_stationary, threshold=11.0)
        >>> print(step)  # False
        >>> 
        >>> # Motion with step (peak at 12 m/s²)
        >>> mag_with_step = np.concatenate([
        >>>     np.ones(10) * 9.8,
        >>>     [12.0],  # peak (step)
        >>>     np.ones(9) * 9.8
        >>> ])
        >>> step = detect_step_simple(mag_with_step, threshold=11.0)
        >>> print(step)  # True
    """
    if accel_mag_window.ndim != 1:
        raise ValueError(
            f"accel_mag_window must be 1D, got shape {accel_mag_window.shape}"
        )

    # Simple peak detection: max > threshold
    max_mag = np.max(accel_mag_window)
    step_detected = max_mag > threshold

    return step_detected


def integrate_gyro_heading(
    heading_prev: float,
    omega_z: float,
    dt: float,
) -> float:
    """
    Integrate gyro yaw rate to update heading (dead reckoning).

    Simple heading update:
        ψ_k = ψ_{k-1} + ω_z * Δt

    where:
        ψ: heading (yaw angle) [radians]
        ω_z: yaw rate (angular velocity about z-axis) [rad/s]
        Δt: time step [s]

    Args:
        heading_prev: Previous heading.
                      Units: radians. Range: typically wrapped to [-π, π].
        omega_z: Yaw rate (z-axis angular velocity).
                 Units: rad/s. Positive = counter-clockwise (right-hand rule).
        dt: Time step.
            Units: seconds.

    Returns:
        Updated heading.
        Units: radians. Unwrapped (may exceed [-π, π]).

    Notes:
        - Gyro-based heading drifts over time due to bias and noise.
        - Should be fused with magnetometer or corrected periodically.
        - Consider wrapping to [-π, π] using np.arctan2 or modulo.
        - For PDR, heading is more critical than step length (errors grow faster).

    Example:
        >>> # Start heading east (0 radians)
        >>> heading = 0.0
        >>> # Turn left at 0.5 rad/s for 1 second
        >>> omega = 0.5  # rad/s
        >>> dt = 1.0
        >>> heading_new = integrate_gyro_heading(heading, omega, dt)
        >>> print(f"{heading_new:.2f} rad")  # 0.50 rad (~28.6°)

    Related:
        - Eq. (6.2): Quaternion integration (full 3D attitude)
        - Eqs. (6.51)-(6.53): Magnetometer heading (alternative/complementary)
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Simple heading integration (dead reckoning)
    heading_next = heading_prev + omega_z * dt

    return heading_next


def wrap_heading(heading_rad: float) -> float:
    """
    Wrap heading angle to [-π, π] range.

    Helper function to keep heading angle in standard range.

    Args:
        heading_rad: Heading angle (possibly outside [-π, π]).
                     Units: radians.

    Returns:
        Wrapped heading in range [-π, π].
        Units: radians.

    Example:
        >>> heading = 7.0  # > 2π
        >>> wrapped = wrap_heading(heading)
        >>> print(f"{wrapped:.2f}")  # 0.72 (wrapped)
    """
    # Wrap to [-π, π]
    wrapped = np.arctan2(np.sin(heading_rad), np.cos(heading_rad))
    return wrapped


