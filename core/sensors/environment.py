"""
Environmental sensor models: magnetometer and barometer (Chapter 6).

This module implements environmental sensor algorithms for heading and altitude:
    - Magnetometer heading with tilt compensation (Eqs. (6.51)-(6.53))
    - Barometric altitude from pressure (Eq. (6.54))
    - Optional smoothing filter helper (Eq. (6.55))

Environmental sensors complement proprioceptive sensors (IMU, wheel) by
providing absolute measurements (heading, altitude) that can reduce drift.

Frame Conventions:
    - B: Body frame (sensor/device frame)
    - M: Map frame (typically ENU with z = Up)
    - Magnetic field is measured in body frame, heading computed in horizontal plane

References:
    Chapter 6, Section 6.4: Environmental sensors
    Eq. (6.51): Magnetometer heading definition
    Eq. (6.52): Tilt compensation
    Eq. (6.53): Heading computation from tilt-compensated field
    Eq. (6.54): Barometric altitude formula
    Eq. (6.55): Generic state/measurement model for smoothing
"""

from typing import Optional
import numpy as np


def mag_tilt_compensate(
    mag_b: np.ndarray,
    roll: float,
    pitch: float,
) -> np.ndarray:
    """
    Apply tilt compensation to magnetometer measurement.

    Implements Eq. (6.52) in Chapter 6:
        mag_h = R_y(-pitch) @ R_x(-roll) @ mag_b

    where:
        mag_b: magnetic field in body frame [μT]
        roll, pitch: attitude angles [radians]
        mag_h: magnetic field projected to horizontal plane [μT]

    Tilt compensation rotates the magnetic field vector from the tilted
    body frame to the horizontal plane, removing the effect of device
    orientation (pitch and roll). This is essential for accurate heading
    when the device is not held level.

    Args:
        mag_b: Magnetic field vector in body frame B.
               Shape: (3,). Units: μT (microtesla) or normalized.
               Components: [mx, my, mz] measured by magnetometer.
        roll: Roll angle (rotation about x-axis).
              Units: radians. Positive = right wing down.
              Typically from IMU attitude estimation.
        pitch: Pitch angle (rotation about y-axis).
               Units: radians. Positive = nose up.
               Typically from IMU attitude estimation.

    Returns:
        Tilt-compensated magnetic field in horizontal plane.
        Shape: (3,). Units match input (μT or normalized).
        The z-component (vertical) should be small after compensation.

    Notes:
        - Requires accurate roll and pitch from IMU.
        - Yaw (heading) is what we're solving for, so it's not an input.
        - Rotation order: roll first, then pitch (R_y(-pitch) @ R_x(-roll)).
        - Sign convention: negative angles undo the body tilt.
        - Indoor magnetic disturbances (steel, electronics) can corrupt results.

    Example:
        >>> import numpy as np
        >>> # Level device: magnetic field points north-down
        >>> mag = np.array([20.0, 0.0, -40.0])  # μT (north, down components)
        >>> roll = 0.0  # level
        >>> pitch = 0.0
        >>> mag_comp = mag_tilt_compensate(mag, roll, pitch)
        >>> print(mag_comp)  # [20, 0, -40] (unchanged when level)
        >>> 
        >>> # Tilted device: 30° pitch
        >>> pitch_30 = np.deg2rad(30)
        >>> mag_comp = mag_tilt_compensate(mag, 0.0, pitch_30)
        >>> # x-component should increase, z-component decrease

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition
        - Eq. (6.52): Tilt compensation (THIS FUNCTION)
        - Eq. (6.53): Heading from tilt-compensated field
    """
    if mag_b.shape != (3,):
        raise ValueError(f"mag_b must have shape (3,), got {mag_b.shape}")

    # Rotation matrices for roll and pitch (negative to undo body rotation)
    # R_x(-roll): rotation about x-axis
    c_roll = np.cos(-roll)
    s_roll = np.sin(-roll)
    R_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    # R_y(-pitch): rotation about y-axis
    c_pitch = np.cos(-pitch)
    s_pitch = np.sin(-pitch)
    R_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    # Tilt compensation (Eq. 6.52): mag_h = R_y @ R_x @ mag_b
    mag_compensated = R_y @ R_x @ mag_b

    return mag_compensated


def mag_heading(
    mag_b: np.ndarray,
    roll: float,
    pitch: float,
    declination: float = 0.0,
) -> float:
    """
    Compute heading (yaw) from magnetometer with tilt compensation.

    Implements Eqs. (6.51)-(6.53) in Chapter 6:
        1. Tilt compensation: mag_h = R(roll, pitch) @ mag_b    (Eq. 6.52)
        2. Heading computation: ψ = atan2(mag_hy, mag_hx)       (Eq. 6.53)
        3. Apply magnetic declination correction                (Eq. 6.51)

    where:
        mag_b: magnetic field in body frame [μT]
        roll, pitch: device attitude [radians]
        mag_h: tilt-compensated field (horizontal) [μT]
        ψ: heading (yaw) angle [radians]
        declination: magnetic declination (true north correction) [radians]

    The magnetometer measures the Earth's magnetic field, which points toward
    magnetic north (not true north). Tilt compensation + declination give
    true heading in the horizontal plane.

    Args:
        mag_b: Magnetic field in body frame B.
               Shape: (3,). Units: μT (microtesla) or normalized.
        roll: Roll angle. Units: radians. From IMU attitude.
        pitch: Pitch angle. Units: radians. From IMU attitude.
        declination: Magnetic declination (magnetic north → true north offset).
                     Units: radians. Default: 0.0 (assume magnetic = true north).
                     Varies by location: -25° to +25° (≈ ±0.44 rad) globally.

    Returns:
        Heading ψ (yaw angle) in horizontal plane.
        Units: radians. Range: [-π, π].
        Convention: 0 = North, π/2 = East (ENU), or depends on frame.

    Notes:
        - Indoor magnetic disturbances (steel, electronics) can corrupt heading.
        - Should be fused with gyro (complementary filter) for stability.
        - Declination varies by location; use IGRF model or local lookup.
        - Requires accurate roll/pitch from IMU (attitude estimation).
        - Sign conventions depend on sensor frame and map frame choice.

    Example:
        >>> import numpy as np
        >>> # Magnetic field pointing north (horizontal) in level device
        >>> mag = np.array([20.0, 0.0, -40.0])  # north + downward (typical)
        >>> roll = 0.0
        >>> pitch = 0.0
        >>> heading = mag_heading(mag, roll, pitch)
        >>> print(f"Heading: {np.rad2deg(heading):.1f}°")  # Should be near 0° (north)

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition (with declination)
        - Eq. (6.52): Tilt compensation (see mag_tilt_compensate)
        - Eq. (6.53): Heading computation (THIS FUNCTION)
    """
    if mag_b.shape != (3,):
        raise ValueError(f"mag_b must have shape (3,), got {mag_b.shape}")

    # Step 1: Tilt compensation (Eq. 6.52)
    mag_h = mag_tilt_compensate(mag_b, roll, pitch)

    # Step 2: Heading from horizontal components (Eq. 6.53)
    # Heading: ψ = atan2(mag_hy, mag_hx)
    # Convention: depends on sensor/map frame alignment
    # Typical: atan2(y, x) gives angle from x-axis (east) counter-clockwise
    psi = np.arctan2(mag_h[1], mag_h[0])

    # Step 3: Apply magnetic declination correction (Eq. 6.51)
    heading = psi + declination

    # Wrap to [-π, π]
    heading = np.arctan2(np.sin(heading), np.cos(heading))

    return heading


def pressure_to_altitude(
    p: float,
    p0: float = 101325.0,
    T: float = 288.15,
) -> float:
    """
    Convert barometric pressure to altitude.

    Implements Eq. (6.54) in Chapter 6 (barometric formula):
        h = (T / L) * (1 - (p / p0)^(R * L / (g * M)))

    Simplified approximation (valid for small altitude changes):
        h ≈ (T / L) * (1 - (p / p0)^α)

    where:
        h: altitude above reference [m]
        p: measured pressure [Pa]
        p0: reference pressure (e.g., sea level or building entrance) [Pa]
        T: temperature [K]
        L: temperature lapse rate ≈ 0.0065 K/m
        R: universal gas constant
        g: gravity
        M: molar mass of air
        α: exponent ≈ 0.190263 (for standard atmosphere)

    Args:
        p: Measured atmospheric pressure.
           Units: Pa (Pascals). Typical range: 95000-105000 Pa.
        p0: Reference pressure (e.g., at known altitude or sea level).
            Units: Pa. Default: 101325 Pa (standard sea level pressure).
        T: Temperature.
           Units: Kelvin. Default: 288.15 K (15°C, standard temp).

    Returns:
        Altitude h above reference level.
        Units: meters. Positive = above p0 level.

    Notes:
        - Assumes standard atmosphere model (reasonable for indoor use).
        - Temperature T should be ambient temperature for accuracy.
        - Pressure changes ~12 Pa per meter (~120 Pa per floor).
        - Barometers drift over time; need periodic reference updates.
        - Weather changes affect p0; track or calibrate regularly.
        - Typical barometer resolution: 1 Pa ≈ 0.08 m altitude.

    Example:
        >>> # At sea level (p = p0)
        >>> h = pressure_to_altitude(p=101325, p0=101325)
        >>> print(f"{h:.1f} m")  # 0.0 m
        >>> 
        >>> # One floor up (~3m, pressure drops ~36 Pa)
        >>> p_floor1 = 101325 - 36
        >>> h = pressure_to_altitude(p=p_floor1, p0=101325)
        >>> print(f"{h:.1f} m")  # ~3.0 m

    Related Equations:
        - Eq. (6.54): Barometric altitude formula (THIS FUNCTION)
        - Eq. (6.55): Generic state/measurement model for smoothing
    """
    if p <= 0:
        raise ValueError(f"p (pressure) must be positive, got {p}")
    if p0 <= 0:
        raise ValueError(f"p0 (reference pressure) must be positive, got {p0}")
    if T <= 0:
        raise ValueError(f"T (temperature) must be positive, got {T}")

    # Simplified barometric formula (Eq. 6.54)
    # Standard atmosphere parameters
    L = 0.0065  # K/m (temperature lapse rate)
    R = 8.31432  # J/(mol·K) (universal gas constant)
    g = 9.80665  # m/s² (standard gravity)
    M = 0.0289644  # kg/mol (molar mass of air)

    # Exponent: α = (R * L) / (g * M) ≈ 0.190263
    alpha = (R * L) / (g * M)

    # Altitude (Eq. 6.54): h = (T / L) * (1 - (p / p0)^α)
    h = (T / L) * (1.0 - (p / p0) ** alpha)

    return h


def detect_floor_change(
    altitude_prev: float,
    altitude_current: float,
    floor_height: float = 3.0,
    threshold: float = 1.5,
) -> int:
    """
    Detect floor change from altitude measurements.

    Simple floor change detector based on altitude difference. Returns the
    estimated floor change (+1 for up, -1 for down, 0 for no change).

    Args:
        altitude_prev: Previous altitude estimate.
                       Units: meters.
        altitude_current: Current altitude estimate.
                          Units: meters.
        floor_height: Typical floor height (floor-to-floor).
                      Units: meters. Default: 3.0 m (typical building).
        threshold: Minimum altitude change to trigger detection.
                   Units: meters. Default: 1.5 m (half floor).

    Returns:
        Floor change: +1 (up one floor), -1 (down), 0 (no change).

    Notes:
        - Very simplified approach for demonstration.
        - Production systems use: hysteresis, smoothing, building models.
        - Barometer noise can cause false detections; use filtering.
        - Multi-floor changes (stairs, elevators) need more sophisticated logic.

    Example:
        >>> # No significant change
        >>> change = detect_floor_change(10.0, 10.2, floor_height=3.0)
        >>> print(change)  # 0
        >>> 
        >>> # Went up one floor
        >>> change = detect_floor_change(10.0, 13.5, floor_height=3.0)
        >>> print(change)  # +1
    """
    delta_h = altitude_current - altitude_prev

    if abs(delta_h) < threshold:
        return 0
    elif delta_h > 0:
        # Moved up: estimate number of floors (simplified to ±1)
        return +1
    else:
        # Moved down
        return -1


def smooth_measurement_simple(
    x_prev: float,
    z: float,
    alpha: float = 0.1,
) -> float:
    """
    Simple exponential smoothing for scalar measurements.

    Implements a lightweight smoothing filter consistent with the generic
    state/measurement model concept in Eq. (6.55):
        x_k = (1 - α) * x_{k-1} + α * z_k

    where:
        x: smoothed state (e.g., altitude, heading)
        z: raw measurement
        α: smoothing factor (0 < α < 1)

    This is a first-order low-pass filter (exponential moving average).
    For more sophisticated smoothing, use KF/EKF from core/estimators.

    Args:
        x_prev: Previous smoothed estimate.
                Units: depend on measurement (m, rad, etc.).
        z: Current raw measurement.
           Units: match x_prev.
        alpha: Smoothing factor.
               Range: (0, 1). Default: 0.1.
               α → 0: heavy smoothing (slow response).
               α → 1: minimal smoothing (track raw measurement).

    Returns:
        Smoothed estimate x_k.
        Units: match input.

    Notes:
        - This is NOT a full Kalman filter; just exponential smoothing.
        - For heading, use circular statistics (not implemented here).
        - For better performance, use core/estimators KalmanFilter with
          state/measurement model defined per Eq. (6.55).
        - Common α values: 0.05-0.2 for slow sensors, 0.3-0.5 for responsive.

    Example:
        >>> # Smooth noisy altitude measurements
        >>> alt_smoothed = 10.0
        >>> alt_raw = 10.5  # noisy measurement
        >>> alt_smoothed = smooth_measurement_simple(alt_smoothed, alt_raw, alpha=0.1)
        >>> print(f"{alt_smoothed:.2f}")  # 10.05 (mostly kept previous value)

    Related Equations:
        - Eq. (6.55): Generic state/measurement model for smoothing
        - Chapter 3: Kalman filter (for full statistical filtering)
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Exponential smoothing (simplified form of Eq. 6.55)
    x_next = (1.0 - alpha) * x_prev + alpha * z

    return x_next


def estimate_magnetic_declination(
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
) -> float:
    """
    Estimate magnetic declination for a given location.

    Simplified placeholder that returns zero. Production systems would use
    IGRF (International Geomagnetic Reference Field) model or lookup tables.

    Magnetic declination is the angle between true north and magnetic north,
    which varies by location and time. It's needed to convert magnetometer
    heading (magnetic) to true heading (geographic).

    Args:
        latitude: Geographic latitude.
                  Units: degrees. Range: [-90, 90].
        longitude: Geographic longitude.
                   Units: degrees. Range: [-180, 180].
        altitude: Altitude above sea level.
                  Units: meters. Default: 0.0.

    Returns:
        Magnetic declination.
        Units: radians. Range: typically [-0.5, 0.5] rad (≈ ±30°).
        Positive = magnetic north is east of true north.

    Notes:
        - THIS IS A PLACEHOLDER returning 0.0 for simplicity.
        - Real implementation needs IGRF model or WMM (World Magnetic Model).
        - Declination varies: US East Coast ~-15°, US West Coast ~+15°.
        - Changes slowly over time (~0.1° per year).
        - For indoor positioning in small areas, can use constant lookup value.

    Example:
        >>> # New York City (approximate)
        >>> dec = estimate_magnetic_declination(lat=40.7, lon=-74.0)
        >>> print(f"Declination: {np.rad2deg(dec):.1f}°")  # 0.0 (placeholder)
    """
    # TODO: Implement IGRF model or use lookup table
    # For now, return zero (assume magnetic north = true north)
    return 0.0


def compensate_hard_iron(
    mag_raw: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """
    Correct magnetometer hard-iron bias.

    Hard-iron distortion is a constant offset in the magnetic field measurement
    caused by nearby ferromagnetic materials (e.g., speaker magnets in phones).

    Correction:
        mag_corrected = mag_raw - offset

    Args:
        mag_raw: Raw magnetometer measurement in body frame.
                 Shape: (3,). Units: μT or normalized.
        offset: Hard-iron offset (bias) in body frame.
                Shape: (3,). Units: match mag_raw.
                Determined during calibration (e.g., figure-8 motion).

    Returns:
        Hard-iron corrected magnetic field.
        Shape: (3,). Units: match input.

    Notes:
        - Hard-iron calibration: rotate device in 3D, fit sphere center.
        - Offset should be re-calibrated periodically or when environment changes.
        - Soft-iron (scale/rotation matrix) is more complex (not implemented).

    Example:
        >>> import numpy as np
        >>> mag_raw = np.array([25.0, 5.0, -35.0])
        >>> offset = np.array([5.0, 0.0, 5.0])  # bias
        >>> mag_corrected = compensate_hard_iron(mag_raw, offset)
        >>> print(mag_corrected)  # [20, 5, -40]
    """
    if mag_raw.shape != (3,):
        raise ValueError(f"mag_raw must have shape (3,), got {mag_raw.shape}")
    if offset.shape != (3,):
        raise ValueError(f"offset must have shape (3,), got {offset.shape}")

    # Hard-iron correction: remove constant offset
    mag_corrected = mag_raw - offset

    return mag_corrected

