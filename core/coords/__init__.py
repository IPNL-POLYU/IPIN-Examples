"""Coordinate systems and transformations for indoor positioning.

This module provides functions and classes for working with different
coordinate frames and transformations commonly used in navigation:
- LLH (Latitude, Longitude, Height) geodetic coordinates
- ECEF (Earth-Centered Earth-Fixed) Cartesian coordinates
- ENU (East-North-Up) local tangent plane coordinates
- NED (North-East-Down) local tangent plane coordinates
- Rotation representations (quaternions, matrices, Euler angles)

Reference: Chapter 2 - Coordinate Systems
"""

from core.coords.frames import Frame, FrameType
from core.coords.rotations import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat,
)
from core.coords.transforms import ecef_to_enu, ecef_to_llh, enu_to_ecef, llh_to_ecef

__all__ = [
    # Frames
    "Frame",
    "FrameType",
    # Transforms
    "llh_to_ecef",
    "ecef_to_llh",
    "ecef_to_enu",
    "enu_to_ecef",
    # Rotations
    "euler_to_quat",
    "euler_to_rotation_matrix",
    "quat_to_euler",
    "quat_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_quat",
]
