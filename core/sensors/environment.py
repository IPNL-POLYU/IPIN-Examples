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
    - M: Map frame (ENU or NED, defined by FrameConvention)
    - Magnetic field is measured in body frame, heading computed in horizontal plane
    - Heading convention must match the frame (0=East for ENU, 0=North for NED)

All functions accept an optional FrameConvention parameter to ensure consistency.

References:
    Chapter 6, Section 6.4: Environmental sensors
    Eq. (6.51): Magnetometer heading definition
    Eq. (6.52): Tilt compensation
    Eq. (6.53): Heading computation from tilt-compensated field
    Eq. (6.54): Barometric altitude formula
    Eq. (6.55): Generic state/measurement model for smoothing
"""

from typing import Optional, cast
import numpy as np

# Import FrameConvention for type hints
from core.sensors.types import FrameConvention

#: Reference Earth magnetic field for the Chapter 6 examples and datasets.
#:
#: Representative values for an urban subtropical site (Hong Kong, roughly
#: 22.3 deg N / 114.2 deg E): total intensity about 45 uT, inclination about
#: +32 deg below horizontal, declination a few degrees west.  These are
#: round numbers chosen to be realistic, not an IGRF evaluation -- what the
#: examples need is a field of the right *shape* (a horizontal component that
#: does not move with the platform, and a dip large enough that tilt
#: compensation has something to do), and a single place to change it.
#:
#: The magnitude matters less than people expect: `mag_heading` reads a
#: direction, so scaling the whole field changes nothing.  The dip matters a
#: great deal, because it decides how much of the field leaks into the
#: horizontal plane when the device tilts.
EARTH_FIELD_INTENSITY_UT = 45.0
EARTH_FIELD_INCLINATION_RAD = float(np.deg2rad(32.0))
EARTH_FIELD_DECLINATION_RAD = float(np.deg2rad(-3.0))


def earth_field_map(
    intensity_ut: float = EARTH_FIELD_INTENSITY_UT,
    inclination_rad: float = EARTH_FIELD_INCLINATION_RAD,
    declination_rad: float = 0.0,
    frame: FrameConvention | None = None,
) -> np.ndarray:
    """
    Earth's magnetic field as a FIXED vector in the map frame.

    This is the field a magnetometer actually sits in: it points at magnetic
    north and dips into the ground, and it does **not** move when the platform
    turns.  Rotating the platform is what changes the *body-frame* reading, and
    that is the entire mechanism `mag_heading` inverts.

    Args:
        intensity_ut: Total field intensity F. Units: uT. Default:
                      `EARTH_FIELD_INTENSITY_UT`.
        inclination_rad: Dip angle I, positive when the field points *into*
                         the ground (northern hemisphere). Units: radians.
        declination_rad: Declination D, positive when magnetic north lies east
                         of true north. Units: radians. Default 0.0, so the
                         field points at true north and no correction is owed.
        frame: Frame the components are returned in. Default: None (ENU,
               components ordered East, North, Up). NED returns
               (North, East, Down).

    Returns:
        Field vector in map-frame component order. Shape: (3,). Units: uT.

    Notes:
        - Horizontal magnitude is ``F cos(I)``; vertical is ``F sin(I)``,
          pointing down, which is -Up in ENU and +Down in NED.
        - Pair with `earth_field_body` to synthesise a magnetometer reading.
          Do not hand-roll the rotation: six places in Chapter 6 used to --
          two generators, three examples and a notebook cell -- and every
          one of them built a field that co-rotated with the platform,
          which is not a magnetic field at all.

    Example:
        >>> import numpy as np
        >>> field = earth_field_map(declination_rad=0.0)
        >>> bool(np.isclose(field[0], 0.0))  # nothing along East
        True
        >>> bool(field[1] > 0 and field[2] < 0)  # north and downward
        True

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition (declination enters here)
    """
    if frame is None:
        frame = FrameConvention.create_enu()

    horizontal = intensity_ut * np.cos(inclination_rad)
    vertical_down = intensity_ut * np.sin(inclination_rad)

    # Magnetic north sits at this heading in the map frame; the frame owns the
    # sign, because ENU heading runs counter-clockwise and NED clockwise.
    north_heading = frame.magnetic_north_heading(declination_rad)
    horizontal_xy = horizontal * frame.heading_to_unit_vector(north_heading)

    # Down is -z in ENU (gravity_direction -1) and +z in NED (+1), which is the
    # same sign the frame already publishes for gravity.
    vertical = frame.gravity_direction * vertical_down

    return np.array([horizontal_xy[0], horizontal_xy[1], vertical])


def earth_field_body(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
    field_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Resolve a fixed map-frame magnetic field into the body frame.

    The single place Chapter 6 synthesises a magnetometer reading. It applies
    ``C_M^B = (Rz(yaw) Ry(pitch) Rx(roll))^T`` -- the transpose of the attitude
    matrix `core.sensors.strapdown.quat_to_rotmat` produces -- so a synthetic
    reading and the strapdown integrator cannot disagree about what an
    attitude means.

    Args:
        roll_rad: Roll about body x, positive right-side-down. Units: radians.
        pitch_rad: Pitch about body y, positive nose-up. Units: radians.
        yaw_rad: Heading in the map frame. Units: radians. ENU: 0 = East,
                 increasing toward North.
        field_map: Map-frame field, shape (3,). Default: None, meaning
                   `earth_field_map()` with zero declination.

    Returns:
        Magnetic field in the body frame. Shape: (3,). Units: match field_map.

    Notes:
        - `mag_heading` inverts exactly this, so the pair is a round trip that
          returns ``yaw_rad`` to machine precision -- pinned over 500 random
          attitudes in tests/core/sensors/test_mag_chain_recovers_enu_heading.py.
        - The field a magnetometer reports does NOT rotate with the platform.
          Building one as ``[cos(yaw), sin(yaw), 0]`` describes a field that
          follows the device around, and makes a *compass* readout look like
          an ENU heading. All six Chapter 6 sites did that; the resulting
          chain was self-consistent and wrong by ``90 deg - psi``.

    Example:
        >>> import numpy as np
        >>> level_east = earth_field_body(0.0, 0.0, 0.0)
        >>> facing_north = earth_field_body(0.0, 0.0, np.pi / 2)
        >>> # Same field, different body: the reading moves, the field does not.
        >>> bool(np.isclose(np.linalg.norm(level_east), np.linalg.norm(facing_north)))
        True

    Related Equations:
        - Eq. (6.52): Tilt compensation undoes the roll/pitch part of this
        - Eq. (6.53): Heading computation undoes the yaw part
    """
    if field_map is None:
        field_map = earth_field_map()
    field_map = np.asarray(field_map, dtype=float)
    if field_map.shape != (3,):
        raise ValueError(f"field_map must have shape (3,), got {field_map.shape}")

    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]])
    rot_y = np.array([[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0], [-sin_p, 0.0, cos_p]])
    rot_z = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])

    body_to_map = rot_z @ rot_y @ rot_x
    return cast(np.ndarray, body_to_map.T @ field_map)


def wrap_angle_diff(
    angle1: float | np.ndarray, angle2: float | np.ndarray
) -> float | np.ndarray:
    """
    Compute the smallest signed difference between two angles.

    Returns angle1 - angle2 wrapped to [-π, π].
    This ensures the result is always the shortest angular distance.

    Args:
        angle1: First angle (radians). Arrays work and wrap elementwise.
        angle2: Second angle (radians). Arrays work and wrap elementwise.

    Returns:
        Signed difference angle1 - angle2 in range [-π, π], scalar or array
        to match the input.
        Positive means angle1 is counter-clockwise from angle2.

    Example:
        >>> # 350° - 10° should give -20° (not +340°)
        >>> diff = wrap_angle_diff(np.deg2rad(350), np.deg2rad(10))
        >>> print(f"{np.rad2deg(diff):.1f}°")  # -20.0°

        >>> # 10° - 350° should give +20° (not -340°)
        >>> diff = wrap_angle_diff(np.deg2rad(10), np.deg2rad(350))
        >>> print(f"{np.rad2deg(diff):.1f}°")  # 20.0°
    """
    diff = angle1 - angle2
    # Wrap to [-π, π] using atan2 trick
    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
    return cast("float | np.ndarray", wrapped_diff)


def mag_tilt_compensate(
    mag_b: np.ndarray,
    roll: float,
    pitch: float,
) -> np.ndarray:
    """
    Apply tilt compensation to magnetometer measurement.

    Implements Eq. (6.52) in Chapter 6:
        M_x = m̃_x cos(θ) + m̃_y sin(ϕ)sin(θ) + m̃_z cos(ϕ)sin(θ)
        M_y = m̃_y cos(ϕ) - m̃_z sin(ϕ)

    where θ = pitch, ϕ = roll, and [m̃_x, m̃_y, m̃_z] = mag_b. The third
    (vertical) component is
    Mz = -m̃_x sin(θ) + m̃_y sin(ϕ)cos(θ) + m̃_z cos(ϕ)cos(θ).

    Tilt compensation rotates the magnetic field vector from the tilted
    body frame back into the horizontal plane, removing the effect of device
    orientation (pitch and roll). This is essential for accurate heading
    when the device is not held level.

    ROTATION ORDER, and why it is this one. The expression above is
    R_y(pitch) @ R_x(roll) @ mag_b, the exact inverse of the roll/pitch part
    of the attitude matrix the rest of Chapter 6 uses -- strapdown integration
    builds C_B^M = R_z(yaw) R_y(pitch) R_x(roll) (see
    docs/ch6_frame_conventions.md), so a body reading is
    m_b = R_x(-roll) R_y(-pitch) m_level and undoing it needs R_y then R_x,
    in that order.

    This module previously composed the same two rotations the other way round,
    R_x(roll) @ R_y(pitch), which is the tilt-compensation form printed in
    several sensor application notes -- correct for an attitude convention that
    applies pitch outside roll, and wrong for this one. The two agree only when
    roll or pitch is zero, so every level test passed. Settled by measurement
    rather than by algebra, because three independent derivations of "the right
    order" produced three different answers: over 500 random attitudes
    (|roll|, |pitch| <= 45 deg) the implemented order recovers yaw to 3.7e-14
    deg, R_x(roll) R_y(pitch) misses by up to 32.8 deg (median 6.0 deg), and
    the two negated-angle orders by 177.9 and 178.4 deg. See
    tests/core/sensors/test_mag_chain_recovers_enu_heading.py.

    Args:
        mag_b: Magnetic field vector in body frame B.
               Shape: (3,). Units: μT (microtesla) or normalized.
               Components: [mx, my, mz] measured by magnetometer.
        roll: Roll angle ϕ (rotation about x-axis).
              Units: radians. Positive = right wing down.
              Typically from IMU attitude estimation.
        pitch: Pitch angle θ (rotation about y-axis).
               Units: radians. Positive = nose up.
               Typically from IMU attitude estimation.

    Returns:
        Tilt-compensated magnetic field in horizontal plane [M_x, M_y, M_z].
        Shape: (3,). Units match input (μT or normalized).
        The z-component (vertical) should be small after compensation.

    Notes:
        - Requires accurate roll and pitch from IMU.
        - Yaw (heading) is what we're solving for, so it's not an input.
        - Rotation order: R_y(pitch) @ R_x(roll) -- see above; it is fixed by
          the chapter's attitude convention and is not a free choice.
        - The output is the field in the LEVEL frame: the map frame turned by
          the platform's yaw, with the tilt taken out. Its horizontal part
          therefore points at magnetic north as seen from the platform, which
          is what Eq. (6.53) turns into a heading.
        - Indoor magnetic disturbances (steel, electronics) can corrupt results.

    Example:
        >>> import numpy as np
        >>> # Level device: the level-frame field is the reading, unchanged.
        >>> mag = np.array([20.0, 0.0, -40.0])  # μT
        >>> mag_comp = mag_tilt_compensate(mag, 0.0, 0.0)
        >>> bool(np.allclose(mag_comp, mag))
        True
        >>> # Tilting the device must NOT move the level-frame field.
        >>> from core.sensors.environment import earth_field_body
        >>> level = mag_tilt_compensate(earth_field_body(0.0, 0.0, 0.9), 0.0, 0.0)
        >>> tilted_reading = earth_field_body(0.3, -0.2, 0.9)
        >>> bool(np.allclose(mag_tilt_compensate(tilted_reading, 0.3, -0.2), level))
        True

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition
        - Eq. (6.52): Tilt compensation (THIS FUNCTION)
        - Eq. (6.53): Heading from tilt-compensated field
    """
    if mag_b.shape != (3,):
        raise ValueError(f"mag_b must have shape (3,), got {mag_b.shape}")

    mx, my, mz = mag_b
    ct, st = np.cos(pitch), np.sin(pitch)  # theta = pitch
    cr, sr = np.cos(roll), np.sin(roll)  # phi = roll

    # Eq. (6.52), written out componentwise rather than as a matrix product so
    # the book's expression is readable in the code. It is R_y(pitch) R_x(roll),
    # the inverse of the roll/pitch part of C_B^M = R_z R_y R_x; see the
    # ROTATION ORDER note above for why the other composition is wrong here.
    Mx = mx * ct + my * sr * st + mz * cr * st
    My = my * cr - mz * sr
    Mz = -mx * st + my * sr * ct + mz * cr * ct

    return np.array([Mx, My, Mz])


def mag_heading(
    mag_b: np.ndarray,
    roll: float,
    pitch: float,
    declination: float = 0.0,
    frame: Optional[FrameConvention] = None,
) -> float:
    """
    Compute heading (yaw) from magnetometer with tilt compensation.

    Implements Eqs. (6.51)-(6.53) in Chapter 6:
        1. Tilt compensation: mag_h = R_y(pitch) R_x(roll) mag_b   (Eq. 6.52)
        2. Heading of magnetic north as the platform sees it,
           ψ_level = atan2(mag_h_y, mag_h_x)                       (Eq. 6.53)
        3. Heading of magnetic north in the map frame, which is where
           declination enters: ψ_map = frame.magnetic_north_heading(D)  (Eq. 6.51)
        4. The platform's heading is the gap between them, ψ = ψ_map - ψ_level

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
        mag_b: Magnetic field in body frame B, with components ordered to
               match `frame.map_axes`. For the ENU default that is
               [East, North, Up]; for NED it is [North, East, Down].
               Shape: (3,). Units: μT (microtesla) or normalized.
        roll: Roll angle. Units: radians. From IMU attitude.
        pitch: Pitch angle. Units: radians. From IMU attitude.
        declination: Magnetic declination, positive when magnetic north lies
                     **east** of true north. Units: radians. Default: 0.0
                     (assume magnetic = true north).
                     Varies by location: -25° to +25° (≈ ±0.44 rad) globally;
                     Hong Kong is about -3°.
                     NOTE the sense: the familiar rule "true = magnetic + east
                     declination" is the *compass* form, and it reverses in ENU
                     because ENU heading runs counter-clockwise. The frame owns
                     that sign (`FrameConvention.magnetic_north_heading`), so
                     the same argument is correct for ENU and NED.
        frame: Frame convention the reading is expressed in and the heading is
               reported in. Default: None (ENU: 0 = East, π/2 = North).
               NED gives 0 = North, π/2 = East.

               **This does not transform mag_b.** Both conventions measure
               heading from the first map axis toward the second, so Eq. (6.53)
               is the same expression either way and only the axis order of the
               input changes. Passing NED to a function handed an ENU-ordered
               reading returns an answer that is wrong by a reflection, and no
               check here can catch it -- three unlabelled numbers carry no
               frame. Order the components yourself to match.

               The angle itself comes from `frame.unit_vector_to_heading`, the
               inverse of the `frame.heading_to_unit_vector` that PDR walks
               along, so the two stay consistent by construction.

    Returns:
        Heading ψ (yaw angle) in horizontal plane.
        Units: radians. Range: [-π, π].
        Convention determined by frame:
            ENU: 0 = East, π/2 = North (default)
            NED: 0 = North, π/2 = East

    Notes:
        - Indoor magnetic disturbances (steel, electronics) can corrupt heading.
        - Should be fused with gyro (complementary filter) for stability.
        - Declination varies by location; use IGRF model or local lookup.
        - Requires accurate roll/pitch from IMU (attitude estimation).
        - Heading convention MUST match frame used in strapdown/PDR.

    Example:
        >>> import numpy as np
        >>> from core.sensors import FrameConvention, earth_field_body
        >>> frame_enu = FrameConvention.create_enu()  # map_axes = E, N, U
        >>> # A level platform facing North: the Earth field, which points
        >>> # North, therefore lands on the platform's own +x (forward) axis.
        >>> mag = earth_field_body(0.0, 0.0, np.pi / 2)
        >>> bool(np.isclose(mag[1], 0.0))  # nothing on the body's left axis
        True
        >>> float(np.round(np.rad2deg(mag_heading(mag, 0.0, 0.0, frame=frame_enu)), 6))
        90.0
        >>> # Reading a body-frame field of [0, H, -Z] instead -- the field on
        >>> # the platform's LEFT -- means the platform faces East, not North:
        >>> float(np.round(np.rad2deg(mag_heading(np.array([0.0, 20.0, -40.0]), 0.0, 0.0)), 6))
        0.0
        >>> # The same three numbers read as NED would be [North, East, Down],
        >>> # a different physical situation. Reordering, not relabelling, is
        >>> # what converts them.

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition (with declination)
        - Eq. (6.52): Tilt compensation (see mag_tilt_compensate)
        - Eq. (6.53): Heading computation (THIS FUNCTION)
        - Eq. (6.50): PDR update (heading convention must match)
    """
    if mag_b.shape != (3,):
        raise ValueError(f"mag_b must have shape (3,), got {mag_b.shape}")

    if frame is None:
        frame = FrameConvention.create_enu()

    # Step 1: Tilt compensation (Eq. 6.52). The result is the field in the
    # LEVEL frame -- the map frame turned by the platform's own heading.
    mag_h = mag_tilt_compensate(mag_b, roll, pitch)

    # Step 2: Heading from horizontal components (Eqs. 6.51, 6.53).
    #
    # The heading is a DIFFERENCE of two directions, and writing it that way
    # is what makes it come out in the map frame's convention:
    #
    #   * where magnetic north sits in the MAP frame -- a known constant, and
    #     the only place declination enters;
    #   * where the platform sees it, in the LEVEL frame -- the measurement.
    #
    # The level frame is the map frame rotated by the heading, so the gap
    # between the two readings IS the heading. Both are read by the same
    # frame method that pdr_step_update walks along, so the two cannot drift
    # apart.
    #
    # This used to be `atan2(m_y, m_x) + declination`, which is the *compass*
    # bearing: clockwise from North, where this function's docstring, its
    # `frame` argument and every downstream caller expect counter-clockwise
    # from East. At level the two differ by exactly 90 deg - psi, a reflection
    # rather than an offset, so it could not be absorbed by a constant. It
    # survived because every synthetic magnetometer in Chapter 6 -- six of
    # them, in two generators, three examples and a notebook -- was built to
    # make it come out right, as a field co-rotating with the platform. Each
    # file was self-consistent, so nothing that compared a file against
    # itself could see either half of the error.
    #
    # The contract that remains on the caller is the axis order of the input:
    # mag_b must already be in frame.map_axes order. Nothing in three
    # unlabelled numbers could tell this function which order it was handed.
    magnetic_north_in_map = frame.magnetic_north_heading(declination)
    magnetic_north_in_level = frame.unit_vector_to_heading(mag_h)
    heading = magnetic_north_in_map - magnetic_north_in_level

    # Wrap to [-π, π]
    heading = np.arctan2(np.sin(heading), np.cos(heading))

    return float(heading)


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

    return float(h)


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

    return cast(np.ndarray, mag_corrected)
