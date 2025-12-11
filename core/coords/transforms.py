"""Coordinate transformations between LLH, ECEF, and ENU frames.

This module implements transformations between geodetic (LLH),
Earth-Centered Earth-Fixed (ECEF), and local East-North-Up (ENU)
coordinate systems.

WGS84 ellipsoid parameters:
- Semi-major axis (a): 6378137.0 m
- Flattening (f): 1/298.257223563
- Semi-minor axis (b): 6356752.314245 m
- First eccentricity squared (e²): 0.00669437999014

Reference: Chapter 2, Section 2.3 - Coordinate Transformations
"""

import numpy as np
from numpy.typing import NDArray

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis (m)
WGS84_E2 = 1.0 - (WGS84_B / WGS84_A) ** 2  # First eccentricity squared


def llh_to_ecef(
    lat: float,
    lon: float,
    height: float,
) -> NDArray[np.float64]:
    """Convert geodetic coordinates (LLH) to ECEF Cartesian coordinates.

    Transforms latitude, longitude, and height above the WGS84 ellipsoid
    to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates.

    Args:
        lat: Latitude in radians (positive north).
        lon: Longitude in radians (positive east).
        height: Height above WGS84 ellipsoid in meters.

    Returns:
        ECEF coordinates as numpy array [x, y, z] in meters.

    Example:
        >>> import numpy as np
        >>> # Greenwich Observatory: 51.4769°N, 0°E, 0m
        >>> lat = np.deg2rad(51.4769)
        >>> lon = np.deg2rad(0.0)
        >>> xyz = llh_to_ecef(lat, lon, 0.0)
        >>> print(f"ECEF: {xyz}")

    Reference:
        Chapter 2, Eq. (2.1) - LLH to ECEF transformation
    """
    # Radius of curvature in the prime vertical
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    # ECEF coordinates
    x = (N + height) * cos_lat * np.cos(lon)
    y = (N + height) * cos_lat * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + height) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_llh(
    x: float,
    y: float,
    z: float,
    tol: float = 1e-12,
    max_iter: int = 10,
) -> NDArray[np.float64]:
    """Convert ECEF Cartesian coordinates to geodetic coordinates (LLH).

    Transforms Earth-Centered Earth-Fixed (ECEF) coordinates to
    latitude, longitude, and height using an iterative algorithm.

    Args:
        x: ECEF x-coordinate in meters.
        y: ECEF y-coordinate in meters.
        z: ECEF z-coordinate in meters.
        tol: Convergence tolerance for iterative solution (meters).
        max_iter: Maximum number of iterations.

    Returns:
        Geodetic coordinates as numpy array [lat, lon, height] where
        lat and lon are in radians, height is in meters.

    Example:
        >>> xyz = np.array([3980574.247, 0.0, 4966824.522])
        >>> llh = ecef_to_llh(*xyz)
        >>> print(f"LLH: lat={np.rad2deg(llh[0]):.4f}°, "
        ...       f"lon={np.rad2deg(llh[1]):.4f}°, h={llh[2]:.2f}m")

    Reference:
        Chapter 2, Eq. (2.2) - ECEF to LLH transformation (iterative)
    """
    # Longitude (exact)
    lon = np.arctan2(y, x)

    # Distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Special case: pole (p ≈ 0)
    if p < 1e-10:
        lat = np.copysign(np.pi / 2.0, z)  # ±90° based on z sign
        height = abs(z) - WGS84_B
        return np.array([lat, lon, height], dtype=np.float64)

    # Initial latitude estimate (assumes height = 0)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))

    # Iterative refinement of latitude and height
    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

        height = p / np.cos(lat) - N

        lat_new = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + height)))

        # Check convergence
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break

        lat = lat_new

    # Final height calculation
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    height = p / np.cos(lat) - N

    return np.array([lat, lon, height], dtype=np.float64)


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    lat_ref: float,
    lon_ref: float,
    height_ref: float,
) -> NDArray[np.float64]:
    """Convert ECEF coordinates to local ENU coordinates.

    Transforms ECEF coordinates to East-North-Up coordinates relative
    to a reference point on the WGS84 ellipsoid.

    Args:
        x: ECEF x-coordinate in meters.
        y: ECEF y-coordinate in meters.
        z: ECEF z-coordinate in meters.
        lat_ref: Reference latitude in radians (origin of ENU frame).
        lon_ref: Reference longitude in radians (origin of ENU frame).
        height_ref: Reference height in meters (origin of ENU frame).

    Returns:
        ENU coordinates as numpy array [east, north, up] in meters.

    Example:
        >>> # Convert point to ENU relative to reference location
        >>> xyz = np.array([3980574.247, 0.0, 4966824.522])
        >>> lat_ref = np.deg2rad(51.0)
        >>> lon_ref = np.deg2rad(0.0)
        >>> enu = ecef_to_enu(*xyz, lat_ref, lon_ref, 0.0)
        >>> print(f"ENU: {enu}")

    Reference:
        Chapter 2, Eq. (2.3) - ECEF to ENU transformation
    """
    # Reference point in ECEF
    xyz_ref = llh_to_ecef(lat_ref, lon_ref, height_ref)

    # Vector from reference to target in ECEF
    dx = x - xyz_ref[0]
    dy = y - xyz_ref[1]
    dz = z - xyz_ref[2]

    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ENU rotation matrix (R_ENU_ECEF)
    R = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )

    # Transform to ENU
    enu = R @ np.array([dx, dy, dz], dtype=np.float64)

    return enu


def enu_to_ecef(
    east: float,
    north: float,
    up: float,
    lat_ref: float,
    lon_ref: float,
    height_ref: float,
) -> NDArray[np.float64]:
    """Convert local ENU coordinates to ECEF coordinates.

    Transforms East-North-Up coordinates relative to a reference point
    back to Earth-Centered Earth-Fixed (ECEF) coordinates.

    Args:
        east: East coordinate in meters.
        north: North coordinate in meters.
        up: Up coordinate in meters.
        lat_ref: Reference latitude in radians (origin of ENU frame).
        lon_ref: Reference longitude in radians (origin of ENU frame).
        height_ref: Reference height in meters (origin of ENU frame).

    Returns:
        ECEF coordinates as numpy array [x, y, z] in meters.

    Example:
        >>> # Convert ENU displacement to ECEF
        >>> enu = np.array([100.0, 200.0, 0.0])  # 100m east, 200m north
        >>> lat_ref = np.deg2rad(51.0)
        >>> lon_ref = np.deg2rad(0.0)
        >>> xyz = enu_to_ecef(*enu, lat_ref, lon_ref, 0.0)
        >>> print(f"ECEF: {xyz}")

    Reference:
        Chapter 2, Eq. (2.4) - ENU to ECEF transformation
    """
    # Reference point in ECEF
    xyz_ref = llh_to_ecef(lat_ref, lon_ref, height_ref)

    # Rotation matrix from ENU to ECEF (transpose of ECEF to ENU)
    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    sin_lon = np.sin(lon_ref)
    cos_lon = np.cos(lon_ref)

    # ECEF rotation matrix (R_ECEF_ENU = R_ENU_ECEF^T)
    R = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ],
        dtype=np.float64,
    )

    # Transform to ECEF displacement
    dxyz = R @ np.array([east, north, up], dtype=np.float64)

    # Add reference point
    xyz = xyz_ref + dxyz

    return xyz
