"""
Unit conversion utilities for IMU sensor specifications.

This module provides explicit conversion functions for IMU noise and bias parameters,
eliminating ambiguity in unit handling. All function names explicitly state both
the input and output units.

Common IMU specifications use these units:
    - Gyro bias: deg/hr (degrees per hour) → rad/s (radians per second)
    - Gyro ARW: deg/√hr (angular random walk) → rad/√s
    - Accel bias: mg (milligravity) → m/s²
    - Accel VRW: m/s/√hr (velocity random walk) → m/s/√s

Key insight: Always be explicit about units in variable names and conversions!

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from typing import Union

# Type alias for numeric types
Numeric = Union[float, np.ndarray]


# ============================================================================
# Gyroscope Unit Conversions
# ============================================================================

def deg_per_hour_to_rad_per_sec(deg_per_hr: Numeric) -> Numeric:
    """
    Convert gyroscope bias from deg/hr to rad/s.
    
    This is the standard conversion for gyro bias instability.
    
    Args:
        deg_per_hr: Bias in degrees per hour.
    
    Returns:
        Bias in radians per second.
    
    Example:
        >>> bias_deg_hr = 10.0  # 10 deg/hr (consumer grade)
        >>> bias_rad_s = deg_per_hour_to_rad_per_sec(bias_deg_hr)
        >>> print(f"{bias_rad_s:.6f} rad/s")
        0.000048 rad/s
        >>> print(f"{np.rad2deg(bias_rad_s):.6f} deg/s")
        0.002778 deg/s
    """
    return np.deg2rad(deg_per_hr) / 3600.0


def deg_per_sec_to_rad_per_sec(deg_per_s: Numeric) -> Numeric:
    """
    Convert angular velocity from deg/s to rad/s.
    
    Args:
        deg_per_s: Angular velocity in degrees per second.
    
    Returns:
        Angular velocity in radians per second.
    """
    return np.deg2rad(deg_per_s)


def deg_per_sqrt_hour_to_rad_per_sqrt_sec(deg_per_sqrt_hr: Numeric) -> Numeric:
    """
    Convert gyroscope Angular Random Walk (ARW) from deg/√hr to rad/√s.
    
    ARW is the angle random walk coefficient, measured in deg/√hr in datasheets.
    To use it in simulation/analysis, convert to rad/√s.
    
    Args:
        deg_per_sqrt_hr: ARW in degrees per square root hour.
    
    Returns:
        ARW in radians per square root second.
    
    Example:
        >>> arw_deg_sqrt_hr = 0.1  # 0.1 deg/√hr (consumer grade)
        >>> arw_rad_sqrt_s = deg_per_sqrt_hour_to_rad_per_sqrt_sec(arw_deg_sqrt_hr)
        >>> print(f"{arw_rad_sqrt_s:.8f} rad/√s")
        0.00002909 rad/√s
    """
    return np.deg2rad(deg_per_sqrt_hr) / np.sqrt(3600.0)


def rad_per_hour_to_rad_per_sec(rad_per_hr: Numeric) -> Numeric:
    """
    Convert angular rate from rad/hr to rad/s.
    
    Args:
        rad_per_hr: Angular rate in radians per hour.
    
    Returns:
        Angular rate in radians per second.
    """
    return rad_per_hr / 3600.0


# ============================================================================
# Accelerometer Unit Conversions
# ============================================================================

def mg_to_mps2(mg: Numeric) -> Numeric:
    """
    Convert acceleration from milligravity (mg) to m/s².
    
    1 mg = 0.001 * 9.80665 m/s² (standard gravity).
    
    Args:
        mg: Acceleration in milligravity.
    
    Returns:
        Acceleration in m/s².
    
    Example:
        >>> bias_mg = 10.0  # 10 mg (consumer grade)
        >>> bias_mps2 = mg_to_mps2(bias_mg)
        >>> print(f"{bias_mps2:.6f} m/s²")
        0.098067 m/s²
    """
    STANDARD_GRAVITY = 9.80665  # m/s² (ISO 80000-3:2006)
    return mg * 0.001 * STANDARD_GRAVITY


def ug_to_mps2(ug: Numeric) -> Numeric:
    """
    Convert acceleration from microgravity (µg) to m/s².
    
    1 µg = 0.000001 * 9.80665 m/s².
    
    Args:
        ug: Acceleration in microgravity.
    
    Returns:
        Acceleration in m/s².
    """
    STANDARD_GRAVITY = 9.80665  # m/s²
    return ug * 1e-6 * STANDARD_GRAVITY


def mps_per_sqrt_hour_to_mps_per_sqrt_sec(mps_per_sqrt_hr: Numeric) -> Numeric:
    """
    Convert accelerometer Velocity Random Walk (VRW) from m/s/√hr to m/s/√s.
    
    VRW is the velocity random walk coefficient.
    
    Args:
        mps_per_sqrt_hr: VRW in (m/s) per square root hour.
    
    Returns:
        VRW in (m/s) per square root second.
    
    Example:
        >>> vrw_mps_sqrt_hr = 0.01  # 0.01 m/s/√hr
        >>> vrw_mps_sqrt_s = mps_per_sqrt_hour_to_mps_per_sqrt_sec(vrw_mps_sqrt_hr)
        >>> print(f"{vrw_mps_sqrt_s:.8f} m/s/√s")
        0.00016667 m/s/√s
    """
    return mps_per_sqrt_hr / np.sqrt(3600.0)


# ============================================================================
# Reverse Conversions (for display/diagnostics)
# ============================================================================

def rad_per_sec_to_deg_per_hour(rad_per_s: Numeric) -> Numeric:
    """
    Convert gyroscope bias from rad/s to deg/hr.
    
    Reverse of deg_per_hour_to_rad_per_sec, useful for display.
    
    Args:
        rad_per_s: Bias in radians per second.
    
    Returns:
        Bias in degrees per hour.
    """
    return np.rad2deg(rad_per_s) * 3600.0


def rad_per_sec_to_deg_per_sec(rad_per_s: Numeric) -> Numeric:
    """
    Convert angular velocity from rad/s to deg/s.
    
    Args:
        rad_per_s: Angular velocity in radians per second.
    
    Returns:
        Angular velocity in degrees per second.
    """
    return np.rad2deg(rad_per_s)


def rad_per_sqrt_sec_to_deg_per_sqrt_hour(rad_per_sqrt_s: Numeric) -> Numeric:
    """
    Convert gyroscope ARW from rad/√s to deg/√hr.
    
    Reverse of deg_per_sqrt_hour_to_rad_per_sqrt_sec, useful for display.
    
    Args:
        rad_per_sqrt_s: ARW in radians per square root second.
    
    Returns:
        ARW in degrees per square root hour.
    """
    return np.rad2deg(rad_per_sqrt_s) * np.sqrt(3600.0)


def mps2_to_mg(mps2: Numeric) -> Numeric:
    """
    Convert acceleration from m/s² to milligravity (mg).
    
    Reverse of mg_to_mps2, useful for display.
    
    Args:
        mps2: Acceleration in m/s².
    
    Returns:
        Acceleration in milligravity.
    """
    STANDARD_GRAVITY = 9.80665  # m/s²
    return mps2 / (0.001 * STANDARD_GRAVITY)


def mps_per_sqrt_sec_to_mps_per_sqrt_hour(mps_per_sqrt_s: Numeric) -> Numeric:
    """
    Convert accelerometer VRW from m/s/√s to m/s/√hr.
    
    Reverse of mps_per_sqrt_hour_to_mps_per_sqrt_sec, useful for display.
    
    Args:
        mps_per_sqrt_s: VRW in (m/s) per square root second.
    
    Returns:
        VRW in (m/s) per square root hour.
    """
    return mps_per_sqrt_s * np.sqrt(3600.0)


# ============================================================================
# Noise PSD Conversions (for Allan Variance analysis)
# ============================================================================

def arw_to_gyro_noise_psd(arw_rad_sqrt_s: Numeric) -> Numeric:
    """
    Convert ARW to gyro noise power spectral density (PSD).
    
    PSD_gyro = ARW² (rad²/s)
    
    Args:
        arw_rad_sqrt_s: Angular Random Walk in rad/√s.
    
    Returns:
        Noise PSD in rad²/s.
    """
    return arw_rad_sqrt_s**2


def vrw_to_accel_noise_psd(vrw_mps_sqrt_s: Numeric) -> Numeric:
    """
    Convert VRW to accelerometer noise power spectral density (PSD).
    
    PSD_accel = VRW² ((m/s)²/s) = m²/s³
    
    Args:
        vrw_mps_sqrt_s: Velocity Random Walk in m/s/√s.
    
    Returns:
        Noise PSD in m²/s³.
    """
    return vrw_mps_sqrt_s**2


# ============================================================================
# Utility Functions
# ============================================================================

def format_gyro_bias(bias_rad_s: float) -> str:
    """
    Format gyro bias for human-readable display.
    
    Args:
        bias_rad_s: Bias in rad/s.
    
    Returns:
        Formatted string with both deg/hr and deg/s.
    
    Example:
        >>> bias = deg_per_hour_to_rad_per_sec(10.0)
        >>> print(format_gyro_bias(bias))
        10.00 deg/hr (0.0028 deg/s)
    """
    deg_hr = rad_per_sec_to_deg_per_hour(bias_rad_s)
    deg_s = rad_per_sec_to_deg_per_sec(bias_rad_s)
    return f"{deg_hr:.2f} deg/hr ({deg_s:.4f} deg/s)"


def format_accel_bias(bias_mps2: float) -> str:
    """
    Format accelerometer bias for human-readable display.
    
    Args:
        bias_mps2: Bias in m/s².
    
    Returns:
        Formatted string with both mg and m/s².
    
    Example:
        >>> bias = mg_to_mps2(10.0)
        >>> print(format_accel_bias(bias))
        10.00 mg (0.0981 m/s²)
    """
    mg = mps2_to_mg(bias_mps2)
    return f"{mg:.2f} mg ({bias_mps2:.4f} m/s²)"


def format_arw(arw_rad_sqrt_s: float) -> str:
    """
    Format gyro ARW for human-readable display.
    
    Args:
        arw_rad_sqrt_s: ARW in rad/sqrt(s).
    
    Returns:
        Formatted string with deg/sqrt(hr).
    
    Example:
        >>> arw = deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.1)
        >>> print(format_arw(arw))
        0.10 deg/sqrt(hr)
    """
    deg_sqrt_hr = rad_per_sqrt_sec_to_deg_per_sqrt_hour(arw_rad_sqrt_s)
    return f"{deg_sqrt_hr:.2f} deg/sqrt(hr)"


def format_vrw(vrw_mps_sqrt_s: float) -> str:
    """
    Format accelerometer VRW for human-readable display.
    
    Args:
        vrw_mps_sqrt_s: VRW in m/s/sqrt(s).
    
    Returns:
        Formatted string with m/s/sqrt(hr).
    
    Example:
        >>> vrw = mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01)
        >>> print(format_vrw(vrw))
        0.01 m/s/sqrt(hr)
    """
    mps_sqrt_hr = mps_per_sqrt_sec_to_mps_per_sqrt_hour(vrw_mps_sqrt_s)
    return f"{mps_sqrt_hr:.4f} m/s/sqrt(hr)"

