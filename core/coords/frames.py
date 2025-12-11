"""Coordinate frame definitions for indoor positioning and navigation.

This module defines the coordinate frames used in indoor positioning:
- ENU (East-North-Up): Local tangent plane with origin at reference point
- NED (North-East-Down): Local tangent plane (aerospace convention)
- ECEF (Earth-Centered Earth-Fixed): Global Cartesian frame
- LLH (Latitude-Longitude-Height): Geodetic coordinates
- Body: Vehicle/sensor frame (forward-right-down)
- Map: World frame for indoor environments

Reference: Chapter 2, Section 2.2 - Coordinate Frames
"""

from enum import Enum
from typing import NamedTuple


class FrameType(Enum):
    """Enumeration of coordinate frame types.

    Attributes:
        ENU: East-North-Up local tangent plane frame.
        NED: North-East-Down local tangent plane frame.
        ECEF: Earth-Centered Earth-Fixed Cartesian frame.
        LLH: Latitude-Longitude-Height geodetic frame.
        BODY: Vehicle/sensor body frame (forward-right-down).
        MAP: World frame for indoor positioning.
    """

    ENU = "enu"
    NED = "ned"
    ECEF = "ecef"
    LLH = "llh"
    BODY = "body"
    MAP = "map"


class Frame(NamedTuple):
    """Representation of a coordinate frame.

    Attributes:
        frame_type: Type of coordinate frame.
        description: Human-readable description of the frame.
    """

    frame_type: FrameType
    description: str

    def __repr__(self) -> str:
        """Return string representation of frame."""
        return f"Frame({self.frame_type.value}: {self.description})"


# Common frame definitions
FRAME_ENU = Frame(
    FrameType.ENU,
    "East-North-Up local tangent plane (x=East, y=North, z=Up)",
)

FRAME_NED = Frame(
    FrameType.NED,
    "North-East-Down local tangent plane (x=North, y=East, z=Down)",
)

FRAME_ECEF = Frame(
    FrameType.ECEF,
    "Earth-Centered Earth-Fixed (x=0°E 0°N, y=90°E 0°N, z=North Pole)",
)

FRAME_LLH = Frame(
    FrameType.LLH,
    "Latitude-Longitude-Height geodetic coordinates",
)

FRAME_BODY = Frame(
    FrameType.BODY,
    "Body frame (x=forward, y=right, z=down)",
)

FRAME_MAP = Frame(
    FrameType.MAP,
    "Map/world frame for indoor positioning",
)
