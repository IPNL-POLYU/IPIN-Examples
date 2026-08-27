"""Coordinate transformations between LLH, ECEF, and ENU frames.

This module implements transformations between geodetic (LLH),
Earth-Centered Earth-Fixed (ECEF), and local East-North-Up (ENU)
coordinate systems.

WGS84 ellipsoid parameters:
- Semi-major axis (a): 6378137.0 m
- Flattening (f): 1/298.257223563
- Semi-minor axis (b): 6356752.314245 m
- First eccentricity squared (e²): 0.00669437999014

Reference: Chapter 2, Section 2.1 - Coordinate Systems and Transformations
"""

import numpy as np
from numpy.typing import NDArray

from core.coords.rotations import euler_to_rotation_matrix

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
        Chapter 2, Eq. (2.9) - LLH to ECEF transformation
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
        Chapter 2, Section 2.1.4 - inverse of Eq. (2.9), computed iteratively.
        The book states this transform is done via ECEF with an iteration
        method and refers to Kaplan & Hegarty [2]; no closed form is given.
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


def enu_to_llh_offset(
    east: float,
    north: float,
    lat: float,
) -> NDArray[np.float64]:
    """Convert a local East/North displacement in metres to a geodetic offset.

    Returns the offset in **radians**, ready to add to a latitude and longitude
    that are already in radians. This is the first-order local tangent
    approximation: divide by the meridian radius of curvature M going north,
    and by the prime-vertical radius N scaled by cos(lat) going east.

    Use this instead of a "metres per degree" constant. Such constants have to
    be run through :func:`numpy.deg2rad` before they touch a radian coordinate,
    and they are only valid at the latitude they were computed for -- one
    degree of longitude spans 78.8 km at 45 deg but 88.1 km at 37.8 deg. Both
    mistakes have shipped here: an example printed 6405.80 m as "100m East",
    and the Chapter 2 dataset generator sampled a 50 m building footprint
    across 2.7 km. Taking metres in and returning radians leaves no unit for a
    caller to mislay.

    Args:
        east: Eastward displacement in meters.
        north: Northward displacement in meters.
        lat: Latitude the offset is taken about, in radians.

    Returns:
        Offset as numpy array [dlat, dlon] in radians.

    Raises:
        ValueError: If the eastward displacement is not local at this latitude
            — see below.

    Example:
        >>> import numpy as np
        >>> lat = np.deg2rad(37.7749)
        >>> dlat, dlon = enu_to_llh_offset(100.0, 0.0, lat)
        >>> # lon + dlon is 100 m east of lon, to within the curvature term.

    Reference:
        Chapter 2, Section 2.1 - local tangent linearisation of Eq. (2.9).
        Exact to first order; the neglected term grows as the square of the
        offset (about 1 mm at 100 m, 99 mm at 1 km).
    """
    sin_lat = np.sin(lat)
    denom = 1.0 - WGS84_E2 * sin_lat**2
    # Prime-vertical radius N (east) and meridian radius M (north).
    n_radius = WGS84_A / np.sqrt(denom)
    m_radius = WGS84_A * (1.0 - WGS84_E2) / denom**1.5

    dlat = north / m_radius
    # Radius of the parallel through this latitude. It shrinks to zero at the
    # poles, where an eastward metre stops corresponding to any bounded change
    # in longitude.
    parallel_radius = n_radius * np.cos(lat)
    dlon = east / parallel_radius if parallel_radius != 0.0 else np.inf

    # Refuse a displacement that is not local, rather than returning a number.
    # `cos(pi/2)` is 6.1e-17 rather than 0, so the pole divides by something
    # tiny-but-finite and yields no warning at all: `enu_to_llh_offset(100, 0,
    # pi/2)` returned 2.55e+11 rad, a silent absurdity of exactly the kind a
    # local linearisation invites. At any latitude a reader would call "local",
    # pi/2 of longitude is thousands of kilometres away, so this cannot fire on
    # legitimate use.
    if not np.isfinite(dlon) or abs(dlon) > np.pi / 2:
        raise ValueError(
            f"east={east} m at latitude {np.rad2deg(lat):.4f} deg implies "
            f"{dlon:.3e} rad of longitude, which is not a local offset. The "
            f"parallel through this latitude has radius {parallel_radius:.3e} "
            f"m; the tangent-plane approximation this function makes does not "
            f"hold near the poles."
        )

    return np.array([dlat, dlon], dtype=np.float64)


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
        Chapter 2, Eq. (2.10) - ECEF to ENU transformation
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
        Chapter 2, inverse of Eq. (2.10) - ENU to ECEF transformation
        (transpose of the ECEF->ENU rotation plus the reference offset).
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


def map_to_body(
    x_map: NDArray[np.float64],
    yaw: float,
    body_origin_map: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transform a point from the local map frame to the local body frame.

    Implements the yaw-only case where the XY-planes of the map and body
    frames are parallel: ``x_body = Rz(yaw) @ (x_map - x_map_B)``.

    Args:
        x_map: Point in the local map frame, array [x, y, z] (meters).
        yaw: Yaw angle psi in radians from the map X/Y axes to the body X/Y
            axes (about the shared Z-axis).
        body_origin_map: Position of the body-frame origin B expressed in the
            map frame. Defaults to the origin (coincident frames).

    Returns:
        Point expressed in the local body frame, array [x, y, z] (meters).

    Reference:
        Chapter 2, Eq. (2.3) - local map to local body transformation.
    """
    cpsi, spsi = np.cos(yaw), np.sin(yaw)
    rz = np.array(
        [[cpsi, spsi, 0.0], [-spsi, cpsi, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d = x_map if body_origin_map is None else x_map - body_origin_map
    return rz @ np.asarray(d, dtype=np.float64)


def body_to_map(
    x_body: NDArray[np.float64],
    yaw: float,
    body_origin_map: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transform a point from the local body frame back to the local map frame.

    Inverse of :func:`map_to_body`:
    ``x_map = Rz(yaw)^T @ x_body + x_map_B``.

    Args:
        x_body: Point in the local body frame, array [x, y, z] (meters).
        yaw: Yaw angle psi in radians (about the shared Z-axis).
        body_origin_map: Position of the body-frame origin B expressed in the
            map frame. Defaults to the origin (coincident frames).

    Returns:
        Point expressed in the local map frame, array [x, y, z] (meters).

    Reference:
        Chapter 2, inverse of Eq. (2.3) - local body to local map.
    """
    cpsi, spsi = np.cos(yaw), np.sin(yaw)
    rz = np.array(
        [[cpsi, spsi, 0.0], [-spsi, cpsi, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    x_map = rz.T @ np.asarray(x_body, dtype=np.float64)
    if body_origin_map is not None:
        x_map = x_map + body_origin_map
    return x_map


# Rotation matrix C^NED_ENU that swaps E<->N and flips U->D (Eq. (2.5)).
# It is symmetric and its own inverse, so ENU->NED and NED->ENU share it.
_C_ENU_NED = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


def enu_to_ned(x_enu: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform a point from the ENU frame to the NED frame.

    ``x_ned = C^NED_ENU @ x_enu`` with the axis swap E<->N and U->D.

    Args:
        x_enu: Point in the ENU frame, array [east, north, up] (meters).

    Returns:
        Point in the NED frame, array [north, east, down] (meters).

    Reference:
        Chapter 2, Eq. (2.5) - ENU to NED transformation.
    """
    return _C_ENU_NED @ np.asarray(x_enu, dtype=np.float64)


def ned_to_enu(x_ned: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform a point from the NED frame to the ENU frame.

    Inverse of :func:`enu_to_ned`; the transform matrix is its own inverse.

    Args:
        x_ned: Point in the NED frame, array [north, east, down] (meters).

    Returns:
        Point in the ENU frame, array [east, north, up] (meters).

    Reference:
        Chapter 2, Eq. (2.5) - NED to ENU (self-inverse of ENU to NED).
    """
    return _C_ENU_NED @ np.asarray(x_ned, dtype=np.float64)


def enu_to_body(
    x_enu: NDArray[np.float64],
    roll: float,
    pitch: float,
    yaw: float,
    body_origin_enu: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transform a point from the ENU frame to the local body frame.

    ``x_body = C^BODY_ENU @ (x_enu - x_enu_B)``, where the rotation matrix is
    built from the body attitude via :func:`euler_to_rotation_matrix`.

    Args:
        x_enu: Point in the ENU frame, array [east, north, up] (meters).
        roll: Body roll angle phi in radians (about the Y-axis).
        pitch: Body pitch angle theta in radians (about the X-axis).
        yaw: Body yaw angle psi in radians (about the Z-axis).
        body_origin_enu: Position of the body-frame origin expressed in ENU.
            Defaults to the origin (coincident frames).

    Returns:
        Point expressed in the local body frame, array [x, y, z] (meters).

    Notes:
        - CONVENTION WARNING: ``body_origin_enu`` is the body origin expressed
          in ENU -- NOT the same quantity as :func:`body_to_enu`'s
          ``enu_origin_body``, which is the ENU origin expressed in the body
          frame. Passing the identical array to both functions is the obvious
          thing a reader tries and does NOT round-trip; it silently returns a
          wrong point. The two are related by
          ``enu_origin_body = -C @ body_origin_enu``, where
          ``C = euler_to_rotation_matrix(roll, pitch, yaw)``. See
          ``tests/core/coords/test_transforms.py::TestEnuBody`` for a pinned
          failure of the naive (same-array) usage and the corrected round
          trip.

    Reference:
        Chapter 2, Eq. (2.6) - ENU to local body transformation.
    """
    c_body_enu = euler_to_rotation_matrix(roll, pitch, yaw)
    d = x_enu if body_origin_enu is None else x_enu - body_origin_enu
    return c_body_enu @ np.asarray(d, dtype=np.float64)


def body_to_enu(
    x_body: NDArray[np.float64],
    roll: float,
    pitch: float,
    yaw: float,
    enu_origin_body: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transform a point from the local body frame to the ENU frame.

    ``x_enu = C^ENU_BODY @ (x_body - x_body_R)``, where
    ``C^ENU_BODY = (C^BODY_ENU)^T`` is the transpose of the ENU->body matrix.

    Args:
        x_body: Point in the local body frame, array [x, y, z] (meters).
        roll: Body roll angle phi in radians (about the Y-axis).
        pitch: Body pitch angle theta in radians (about the X-axis).
        yaw: Body yaw angle psi in radians (about the Z-axis).
        enu_origin_body: Position of the ENU-frame origin expressed in the body
            frame. Defaults to the origin (coincident frames).

    Returns:
        Point expressed in the ENU frame, array [east, north, up] (meters).

    Notes:
        - CONVENTION WARNING: ``enu_origin_body`` is the ENU origin expressed
          in the body frame -- NOT the same quantity as :func:`enu_to_body`'s
          ``body_origin_enu``, which is the body origin expressed in ENU.
          Passing the identical array to both functions is the obvious thing
          a reader tries and does NOT round-trip; it silently returns a wrong
          point. The two are related by
          ``enu_origin_body = -C @ body_origin_enu``, where
          ``C = euler_to_rotation_matrix(roll, pitch, yaw)``. See
          ``tests/core/coords/test_transforms.py::TestEnuBody`` for a pinned
          failure of the naive (same-array) usage and the corrected round
          trip.

    Reference:
        Chapter 2, Eq. (2.7) - local body to ENU transformation.
    """
    c_enu_body = euler_to_rotation_matrix(roll, pitch, yaw).T
    d = x_body if enu_origin_body is None else x_body - enu_origin_body
    return c_enu_body @ np.asarray(d, dtype=np.float64)
