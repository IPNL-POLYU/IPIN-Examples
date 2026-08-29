"""Coordinate frame definitions for indoor positioning and navigation.

This module defines the coordinate frames used in indoor positioning:
- ENU (East-North-Up): Local tangent plane with origin at reference point
- NED (North-East-Down): Local tangent plane (aerospace convention)
- ECEF (Earth-Centered Earth-Fixed): Global Cartesian frame
- LLH (Latitude-Longitude-Height): Geodetic coordinates
- Body: Vehicle/sensor-local frames. Prefer the explicit BODY_CH2 or BODY_FRD
  names below; BODY is retained as a legacy generic label.
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
        BODY: Legacy generic body-frame label. Ambiguous on purpose for
            backward compatibility; prefer BODY_CH2 or BODY_FRD.
        BODY_CH2: Chapter 2 book body frame (x=right, y=forward, z=up).
        BODY_FRD: Aerospace/vehicle body frame (x=forward, y=right, z=down).
        MAP: World frame for indoor positioning.
    """

    ENU = "enu"
    NED = "ned"
    ECEF = "ecef"
    LLH = "llh"
    BODY = "body"
    BODY_CH2 = "body_ch2"
    BODY_FRD = "body_frd"
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
    "Legacy generic body frame label; prefer FRAME_BODY_CH2 or FRAME_BODY_FRD",
)

FRAME_BODY_CH2 = Frame(
    FrameType.BODY_CH2,
    "Chapter 2 body frame (x=right, y=forward, z=up; roll about Y, pitch about X)",
)

FRAME_BODY_FRD = Frame(
    FrameType.BODY_FRD,
    "Aerospace/vehicle body frame (x=forward, y=right, z=down)",
)

FRAME_MAP = Frame(
    FrameType.MAP,
    "Map/world frame for indoor positioning",
)
