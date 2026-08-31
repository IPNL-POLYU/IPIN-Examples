"""The magnetometer chain must return the heading the rest of Chapter 6 uses.

`mag_heading` documents an ENU heading -- 0 = East, increasing toward North --
and `pdr_step_update` walks along ``[cos psi, sin psi]`` in exactly that
convention.  It used to return ``90 deg - psi``: a *compass* heading, clockwise
from North.  Nothing caught it because every synthetic magnetometer in the
chapter -- all six of them, in two generators, three examples and a notebook
cell -- was built to make the wrong formula come out right, as a field that
co-rotates with the platform rather than a fixed Earth field. The two errors
cancelled inside each file, so nothing that compared a file against itself
could see either.

So this file refuses to synthesise a field from anything the chain itself
believes.  It starts from physics:

    1. a FIXED Earth field in the map frame (Hong Kong-ish: 45 uT, dip 32 deg,
       declination -3 deg);
    2. rotated into the body frame by the transpose of the attitude matrix the
       chapter's own strapdown integrator uses, ``Rz(yaw) Ry(pitch) Rx(roll)``;
    3. handed to the chain, whose answer must be the yaw we started from;
    4. and finally walked by `pdr_step_update`, which must step in the
       direction the platform is actually facing.

Step 2 is the part that makes this a *truth* harness rather than a round trip.
`Rz Ry Rx` is not asserted here on faith either -- `test_the_attitude_matrix_is
_the_one_strapdown_uses` pins it against `quat_to_rotmat`, so if the chapter's
attitude convention ever moves, this file goes red rather than quietly
following it.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from core.coords.rotations import euler_to_quat
from core.sensors import (
    FrameConvention,
    earth_field_body,
    earth_field_map,
    mag_heading,
    pdr_step_update,
)
from core.sensors.strapdown import quat_to_rotmat

N_ATTITUDES = 500
#: Zero-noise budget.  The chain is a handful of trig calls on float64, so the
#: honest bound is a few ulp of a radian, not a modelling tolerance.  Measured
#: below at 5.1e-14 deg; the loser orders miss by 33 deg and 178 deg, so this
#: sits fourteen orders below the defect it exists to catch.
MAX_HEADING_ERROR_DEG = 1e-9


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_map(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """C_B^M for the Chapter 6 attitude convention, built here independently."""
    return _rotation_z(yaw_rad) @ _rotation_y(pitch_rad) @ _rotation_x(roll_rad)


def random_attitudes(count: int = N_ATTITUDES) -> np.ndarray:
    """Level-ish device attitudes: full yaw, +-45 deg of roll and pitch."""
    rng = np.random.default_rng(20260830)
    return np.column_stack(
        [
            rng.uniform(-np.pi / 4, np.pi / 4, count),
            rng.uniform(-np.pi / 4, np.pi / 4, count),
            rng.uniform(-np.pi, np.pi, count),
        ]
    )


def wrapped_error_deg(recovered_rad: float, truth_rad: float) -> float:
    difference = recovered_rad - truth_rad
    return float(abs(np.rad2deg(np.arctan2(np.sin(difference), np.cos(difference)))))


def test_the_attitude_matrix_is_the_one_strapdown_uses() -> None:
    """Rz(yaw) Ry(pitch) Rx(roll), the premise everything below rests on."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        roll, pitch, yaw = rng.uniform(-1.2, 1.2, 3)
        # The documented Chapter 2 -> Chapter 6 conversion swaps roll and pitch;
        # see docs/ch6_frame_conventions.md.
        quaternion = euler_to_quat(pitch, roll, yaw)
        worst = max(
            worst,
            float(
                np.abs(quat_to_rotmat(quaternion) - body_to_map(roll, pitch, yaw)).max()
            ),
        )
    assert (
        worst < 1e-12
    ), f"strapdown attitude convention moved: max |diff| = {worst:.3e}"


def test_level_platform_recovers_its_own_yaw() -> None:
    """The level case, where the answer is readable by eye.

    This is the assertion the old chain failed most legibly: it returned
    ``90 deg - psi``, so a platform facing East (psi = 0) was reported as
    facing North.
    """
    field_map = earth_field_map()
    for yaw_deg in [-135.0, -90.0, 0.0, 30.0, 45.0, 90.0, 135.0, 179.0]:
        yaw = np.deg2rad(yaw_deg)
        mag_b = body_to_map(0.0, 0.0, yaw).T @ field_map
        recovered = mag_heading(mag_b, 0.0, 0.0, declination=0.0)
        assert (
            wrapped_error_deg(recovered, yaw) < MAX_HEADING_ERROR_DEG
        ), f"level yaw {yaw_deg} deg came back as {np.rad2deg(recovered):.3f} deg"


def test_tilted_platform_recovers_its_own_yaw() -> None:
    """The sweep that settles the tilt-compensation rotation order.

    Three plausible orders were proposed for Eq. (6.52) while this was being
    fixed, and analysis alone picked a different one each time.  Measured over
    ``N_ATTITUDES`` attitudes: the implemented order lands at 3.7e-14 deg,
    ``Rx(roll) Ry(pitch)`` (the order the code carried) at 32.8 deg (median
    6.0 deg), and the two negated orders at 177.9 and 178.4 deg.
    """
    field_map = earth_field_map()
    worst_deg = 0.0
    worst_attitude = None
    for roll, pitch, yaw in random_attitudes():
        mag_b = body_to_map(roll, pitch, yaw).T @ field_map
        error_deg = wrapped_error_deg(mag_heading(mag_b, roll, pitch), yaw)
        if error_deg > worst_deg:
            worst_deg, worst_attitude = error_deg, (roll, pitch, yaw)
    assert worst_deg < MAX_HEADING_ERROR_DEG, (
        f"max heading error {worst_deg:.4f} deg over {N_ATTITUDES} attitudes, "
        f"worst at roll/pitch/yaw = {np.rad2deg(np.asarray(worst_attitude))} deg"
    )


def test_the_generator_helper_agrees_with_the_attitude_matrix() -> None:
    """`earth_field_body` is the same rotation, not a second opinion.

    Every synthetic magnetometer in Chapter 6 now goes through that helper, so
    if it drifted from `body_to_map` above the examples would agree with a
    chain that had stopped being physical -- which is the exact failure this
    file exists to end.
    """
    field_map = earth_field_map()
    worst = 0.0
    for roll, pitch, yaw in random_attitudes(64):
        expected = body_to_map(roll, pitch, yaw).T @ field_map
        worst = max(
            worst, float(np.abs(earth_field_body(roll, pitch, yaw) - expected).max())
        )
    assert worst < 1e-12, f"earth_field_body disagrees by {worst:.3e} uT"


@pytest.mark.parametrize("declination_deg", [-20.0, -3.0, 0.0, 7.5, 20.0])
def test_declination_returns_true_heading(declination_deg: float) -> None:
    """A field built with declination D still reports the TRUE heading.

    In the compass convention east declination adds; in this ENU convention it
    subtracts, because ENU heading runs counter-clockwise while a compass
    bearing runs clockwise.  Measured, not argued: without the correction the
    recovered heading is off by exactly D.
    """
    declination = np.deg2rad(declination_deg)
    field_map = earth_field_map(declination_rad=declination)
    for yaw_deg in [-120.0, 0.0, 37.0, 160.0]:
        yaw = np.deg2rad(yaw_deg)
        mag_b = body_to_map(0.2, -0.1, yaw).T @ field_map
        recovered = mag_heading(mag_b, 0.2, -0.1, declination=declination)
        assert wrapped_error_deg(recovered, yaw) < MAX_HEADING_ERROR_DEG

        uncorrected = mag_heading(mag_b, 0.2, -0.1, declination=0.0)
        assert (
            abs(wrapped_error_deg(uncorrected, yaw) - abs(declination_deg))
            < MAX_HEADING_ERROR_DEG
        ), "skipping the declination correction should cost exactly D"


def test_the_heading_walks_the_direction_the_platform_faces() -> None:
    """The end of the chain: `pdr_step_update` consumes what `mag_heading` gives.

    An angle is only right relative to what reads it.  A platform facing map
    +y must step in +y, and the step the old chain took was 90 deg - psi off.
    """
    field_map = earth_field_map()
    step_length_m = 0.7
    for yaw_deg, expected_step in [
        (0.0, np.array([step_length_m, 0.0])),
        (90.0, np.array([0.0, step_length_m])),
        (180.0, np.array([-step_length_m, 0.0])),
        (-90.0, np.array([0.0, -step_length_m])),
    ]:
        yaw = np.deg2rad(yaw_deg)
        mag_b = earth_field_body(0.0, 0.0, yaw, field_map=field_map)
        heading = mag_heading(mag_b, 0.0, 0.0)
        stepped = pdr_step_update(np.zeros(2), step_length_m, heading)
        assert np.allclose(
            stepped, expected_step, atol=1e-9
        ), f"facing {yaw_deg} deg the walker stepped {stepped}, not {expected_step}"


def test_ned_recovers_its_own_yaw_too() -> None:
    """The other frame, because `frame` is supposed to be load-bearing.

    NED reads the same three numbers as (North, East, Down) and measures
    heading clockwise from North, so the declination correction changes sign
    between the two conventions.  Both come out of `FrameConvention`.
    """
    frame = FrameConvention.create_ned()
    declination = np.deg2rad(-3.0)
    field_map_ned = earth_field_map(declination_rad=declination, frame=frame)
    worst_deg = 0.0
    for roll, pitch, yaw in random_attitudes(128):
        mag_b = body_to_map(roll, pitch, yaw).T @ field_map_ned
        recovered = mag_heading(
            mag_b, roll, pitch, declination=declination, frame=frame
        )
        worst_deg = max(worst_deg, wrapped_error_deg(recovered, yaw))
    assert (
        worst_deg < MAX_HEADING_ERROR_DEG
    ), f"NED max heading error {worst_deg:.4f} deg"
