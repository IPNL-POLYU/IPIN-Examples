"""
Gravity magnitude computation (Chapter 6).

This module implements latitude-dependent gravity model from Book Eq. (6.8),
providing a single source of truth for gravity magnitude across all Ch6 algorithms.

The WGS-84 gravity formula (Eq. 6.8) accounts for:
    - Earth's oblate spheroid shape (equatorial bulge)
    - Centrifugal force from Earth's rotation
    - Latitude-dependent variation (±0.05 m/s² from equator to poles)

This module is used by:
    - Strapdown propagation (Eq. 6.7): Velocity update with gravity compensation
    - IMU forward model: Synthetic IMU generation from trajectories
    - PDR gravity removal (Eq. 6.47): Step detection preprocessing
    - ZUPT detector (Eq. 6.44): Stationary detection test statistic

Design Philosophy:
    - Single source of truth: All gravity computations route through this module
    - Backward compatible: Falls back to default g=9.81 when latitude unavailable
    - Explicit radians: Latitude must be in radians to avoid conversion errors

Author: Li-Ta Hsu
Date: December 2025
"""

from typing import Optional
import numpy as np


def gravity_magnitude_eq6_8(lat_rad: float) -> float:
    """
    Compute gravity magnitude using WGS-84 model (Book Eq. 6.8).
    
    Implements Eq. (6.8) from Chapter 6:
        g(φ) = 9.7803 * (1 + 0.0053024·sin²(φ) - 0.000005·sin²(2φ))
    
    where φ is geodetic latitude in radians.
    
    This formula models gravity variation due to:
        1. Earth's oblate spheroid shape (flattening at poles)
        2. Centrifugal force from Earth's rotation (stronger at equator)
        3. Mass distribution in Earth's interior
    
    Physical Interpretation:
        - Equator (φ=0°):   g ≈ 9.780 m/s² (minimum, strongest centrifugal effect)
        - 45° latitude:     g ≈ 9.806 m/s² (mid-range)
        - Poles (φ=±90°):   g ≈ 9.832 m/s² (maximum, no centrifugal effect)
        - Total variation:  ≈ 0.052 m/s² (≈0.5% of g)
    
    Args:
        lat_rad: Geodetic latitude in radians.
                 Range: [-π/2, +π/2] (South Pole to North Pole).
                 Examples:
                     0.0 rad = Equator (0°)
                     π/4 rad = 45° North
                     π/2 rad = North Pole (90°)
    
    Returns:
        Gravity magnitude g in m/s².
        Range: approximately [9.78, 9.84] m/s².
    
    Notes:
        - This is WGS-84 gravity formula (standard for GPS/GNSS)
        - Assumes sea level; altitude correction not included (Eq. 6.8 scope)
        - For indoor positioning: latitude variation is more significant than
          altitude variation (buildings rarely exceed ±100m → ±0.03 m/s² effect)
        - More accurate than constant g=9.81 m/s² for high-precision navigation
    
    Example:
        >>> import numpy as np
        >>> # Equator
        >>> g_equator = gravity_magnitude_eq6_8(0.0)
        >>> print(f"Equator: {g_equator:.4f} m/s²")  # 9.7803
        >>> 
        >>> # 45° North (typical mid-latitude)
        >>> g_45n = gravity_magnitude_eq6_8(np.deg2rad(45.0))
        >>> print(f"45°N: {g_45n:.4f} m/s²")  # ~9.8062
        >>> 
        >>> # North Pole
        >>> g_pole = gravity_magnitude_eq6_8(np.pi / 2)
        >>> print(f"Pole: {g_pole:.4f} m/s²")  # ~9.8322
    
    Related Equations:
        - Eq. (6.8): Gravity magnitude (THIS FUNCTION)
        - Eq. (6.7): Velocity update (uses g from this function)
        - Eq. (6.47): PDR gravity removal (uses g from this function)
        - Eq. (6.44): ZUPT test statistic (uses g from this function)
    
    References:
        Chapter 6, Eq. (6.8): Gravity vector definition
        WGS-84 Earth Gravitational Model
    """
    # Eq. (6.8): g(φ) = 9.7803 * (1 + 0.0053024·sin²(φ) - 0.000005·sin²(2φ))
    sin_lat = np.sin(lat_rad)
    sin_lat_sq = sin_lat * sin_lat
    sin_2lat = np.sin(2.0 * lat_rad)
    sin_2lat_sq = sin_2lat * sin_2lat
    
    g = 9.7803 * (1.0 + 0.0053024 * sin_lat_sq - 0.000005 * sin_2lat_sq)
    
    return g


def gravity_magnitude(
    lat_rad: Optional[float] = None,
    default_g: float = 9.81,
) -> float:
    """
    Compute gravity magnitude with automatic fallback.
    
    This is the recommended interface for all Chapter 6 algorithms.
    Provides backward compatibility while enabling book-accurate Eq. (6.8).
    
    Behavior:
        - If lat_rad is provided: Use Eq. (6.8) for latitude-dependent gravity
        - If lat_rad is None: Return default_g (backward compatible)
    
    Args:
        lat_rad: Geodetic latitude in radians (optional).
                 If None, returns default_g.
                 If provided, computes g using Eq. (6.8).
        default_g: Fallback gravity magnitude when lat_rad is None.
                   Default: 9.81 m/s² (standard gravity).
                   Typical range: 9.78-9.84 m/s².
    
    Returns:
        Gravity magnitude in m/s².
        Either from Eq. (6.8) or default_g.
    
    Notes:
        - Use this function in production code for flexibility
        - Use gravity_magnitude_eq6_8() directly if latitude is always known
        - default_g allows matching legacy behavior or custom gravity values
    
    Example:
        >>> import numpy as np
        >>> # Without latitude (backward compatible)
        >>> g = gravity_magnitude()
        >>> print(g)  # 9.81 (default)
        >>> 
        >>> # With latitude (book-accurate)
        >>> lat_deg = 40.0  # New York City latitude
        >>> lat_rad = np.deg2rad(lat_deg)
        >>> g = gravity_magnitude(lat_rad=lat_rad)
        >>> print(f"{g:.4f}")  # ~9.8018 (from Eq. 6.8)
        >>> 
        >>> # Custom default (e.g., for specific location)
        >>> g_tokyo = gravity_magnitude(default_g=9.798)
        >>> print(g_tokyo)  # 9.798 (custom)
    
    Usage in Chapter 6:
        >>> # Strapdown propagation
        >>> g_mag = gravity_magnitude(lat_rad, default_g=9.81)
        >>> g_vec = frame.gravity_vector(g_mag)
        >>> 
        >>> # IMU forward model
        >>> g_mag = gravity_magnitude(lat_rad, default_g=9.81)
        >>> accel_body, gyro_body = generate_imu_from_trajectory(..., g=g_mag)
        >>> 
        >>> # PDR step detection
        >>> g_mag = gravity_magnitude(lat_rad, default_g=9.81)
        >>> a_dynamic = a_mag - g_mag
    
    Related Functions:
        - gravity_magnitude_eq6_8(): Direct Eq. (6.8) implementation
    """
    if lat_rad is None:
        # Backward compatible: use default gravity
        return default_g
    else:
        # Book-accurate: use Eq. (6.8)
        return gravity_magnitude_eq6_8(lat_rad)


def gravity_magnitude_from_lat_deg(lat_deg: float) -> float:
    """
    Convenience wrapper: Compute gravity from latitude in degrees.
    
    Automatically converts degrees to radians before calling Eq. (6.8).
    Useful for user-facing APIs where degrees are more intuitive.
    
    Args:
        lat_deg: Geodetic latitude in degrees.
                 Range: [-90, +90] (South Pole to North Pole).
                 Examples: 0 (Equator), 45 (mid-latitude), 90 (North Pole).
    
    Returns:
        Gravity magnitude in m/s² from Eq. (6.8).
    
    Example:
        >>> # Tokyo, Japan (35.6762° N)
        >>> g_tokyo = gravity_magnitude_from_lat_deg(35.6762)
        >>> print(f"Tokyo: {g_tokyo:.4f} m/s²")  # ~9.7976
        >>> 
        >>> # Singapore (1.3521° N, near equator)
        >>> g_singapore = gravity_magnitude_from_lat_deg(1.3521)
        >>> print(f"Singapore: {g_singapore:.4f} m/s²")  # ~9.7804
    
    Related Functions:
        - gravity_magnitude_eq6_8(): Takes radians (more explicit)
        - gravity_magnitude(): Main interface with fallback
    """
    lat_rad = np.deg2rad(lat_deg)
    return gravity_magnitude_eq6_8(lat_rad)

