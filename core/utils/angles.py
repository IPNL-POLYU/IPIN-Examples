"""
Angle wrapping and manipulation utilities.

Provides functions for handling angular quantities and ensuring they remain
within proper bounds (typically [-π, π] for radians).

Critical for:
- Bearing measurements in positioning systems
- Heading angles in navigation
- Angular innovations in Kalman filters
"""

import numpy as np
from typing import Union


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-π, π] range.
    
    This is critical for bearing measurements and angular innovations
    in Kalman filters. Without wrapping, angles near ±180° can cause
    large incorrect innovations (e.g., -179° vs +179° = 358° error
    instead of 2° error).
    
    Args:
        angle: Angle in radians (can be any value)
    
    Returns:
        Wrapped angle in range [-π, π]
    
    Example:
        >>> wrap_angle(3.5 * np.pi)  # 630° -> -90°
        -1.5707963267948966
        >>> wrap_angle(-3.5 * np.pi)  # -630° -> 90°
        1.5707963267948966
    
    References:
        Used in Extended Kalman Filter bearing measurement updates
    """
    # Use atan2 trick for robust wrapping
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """
    Wrap array of angles to [-π, π] range.
    
    Vectorized version of wrap_angle() for efficiency.
    
    Args:
        angles: Array of angles in radians
    
    Returns:
        Array of wrapped angles in range [-π, π]
    
    Example:
        >>> angles = np.array([0, np.pi/2, np.pi, -np.pi, 3*np.pi])
        >>> wrap_angle_array(angles)
        array([ 0.        ,  1.57079633,  3.14159265, -3.14159265, -3.14159265])
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def angle_diff(angle1: Union[float, np.ndarray], 
               angle2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the shortest angular difference between two angles.
    
    Returns angle1 - angle2, wrapped to [-π, π]. This is the innovation
    for bearing measurements in EKF/UKF.
    
    Args:
        angle1: First angle in radians (measured)
        angle2: Second angle in radians (predicted)
    
    Returns:
        Shortest signed difference angle1 - angle2 in [-π, π]
    
    Example:
        >>> angle_diff(np.pi - 0.1, -np.pi + 0.1)  # Nearly opposite
        -0.2
        >>> angle_diff(0.1, -0.1)  # Small difference
        0.2
    
    Notes:
        This is critical for EKF bearing updates:
        innovation = angle_diff(measured_bearing, predicted_bearing)
        
        Without this, bearings near ±180° would have huge innovations:
        measured = +179°, predicted = -179° -> innovation = 358° (WRONG!)
        with angle_diff: innovation = 2° (CORRECT!)
    
    References:
        EKF bearing measurement update (Chapter 3)
    """
    if isinstance(angle1, np.ndarray) or isinstance(angle2, np.ndarray):
        return wrap_angle_array(np.asarray(angle1) - np.asarray(angle2))
    else:
        return wrap_angle(angle1 - angle2)


def degrees_to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def radians_to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert radians to degrees."""
    return np.rad2deg(radians)



