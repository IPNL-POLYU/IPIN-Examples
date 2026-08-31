"""`quat_to_euler` must be right *at* roll = +/-90, not merely near it.

The two recovery paths -- `rotation_matrix_to_euler` and `quat_to_euler` --
invert the same composition, so they have the same singularity: at
roll = +/-90 deg the yaw and pitch axes coincide and only their sum survives.
The matrix path has always carried an explicit branch for it (fix yaw = 0,
solve pitch, sign it with `copysign`). The quaternion path did not, and the
failure was **discontinuous rather than gradual**:

- at roll = 89.999 deg it reconstructs C to 6e-12;
- at roll = 90.0 deg exactly it returned [90, 180, 180] deg, whose matrix
  differs from the input by 0.5 -- a different attitude, not one of the
  gimbal-equivalent triples.

The mechanism is signed zeros, which is why "degrades near the lock" was the
wrong description. At the lock both `atan2` arguments for pitch and for yaw
collapse to (0.0, -2.2e-16): a negative x with a zero y, so `atan2` returns
+pi rather than 0, twice. It is not a precision loss that grows -- it is a
branch that only exists exactly on the boundary.

So the assertion that matters is the *reconstruction*: whatever triple comes
back, `euler_to_rotation_matrix` of it must reproduce the matrix the
quaternion stands for. That is the only statement about a gimbal-locked
attitude that is convention-free, since infinitely many (yaw, pitch) pairs
name it.

Author: Li-Ta Hsu
Reference: Chapter 2, Eq. (2.22) - quaternion to Euler angles.
"""

import numpy as np
import pytest

from core.coords.rotations import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
)

# (roll, pitch, yaw) in degrees, roll exactly at the lock on both branches.
# Several (yaw, pitch) pairs, because the lock couples them: a fix that only
# happens to work for yaw = 0 would pass a single case.
LOCKED_ANGLES_DEG = [
    (90.0, 0.0, 0.0),
    (90.0, -30.0, 0.0),
    (90.0, 0.0, 30.0),
    (90.0, 20.0, 50.0),
    (90.0, -75.0, 110.0),
    (-90.0, 0.0, 0.0),
    (-90.0, -30.0, 0.0),
    (-90.0, 0.0, 30.0),
    (-90.0, 20.0, 50.0),
    (-90.0, -75.0, 110.0),
]


def _radians(angles_deg):
    return tuple(np.deg2rad(a) for a in angles_deg)


@pytest.mark.parametrize("angles_deg", LOCKED_ANGLES_DEG)
def test_the_recovered_triple_reconstructs_the_same_attitude(angles_deg):
    """The property that survives the lock: C in, C out."""
    roll, pitch, yaw = _radians(angles_deg)
    C = euler_to_rotation_matrix(roll, pitch, yaw)
    q = euler_to_quat(roll, pitch, yaw)

    recovered = quat_to_euler(q)
    C_again = euler_to_rotation_matrix(*recovered)

    residual = np.abs(C_again - C).max()
    assert residual < 1e-12, (
        f"quat_to_euler at {angles_deg} deg returned "
        f"{np.round(np.rad2deg(recovered), 4)} deg, which reconstructs a "
        f"matrix differing from the input by {residual:.3g}. At the lock the "
        "triple is not unique, but the attitude it names is."
    )


@pytest.mark.parametrize("angles_deg", LOCKED_ANGLES_DEG)
def test_roll_comes_back_at_the_lock_with_its_sign(angles_deg):
    """Roll is the one angle the lock does not make ambiguous."""
    roll, pitch, yaw = _radians(angles_deg)
    recovered = quat_to_euler(euler_to_quat(roll, pitch, yaw))

    assert recovered[0] == pytest.approx(np.copysign(np.pi / 2.0, roll), abs=1e-12)


@pytest.mark.parametrize("angles_deg", LOCKED_ANGLES_DEG)
def test_both_recovery_paths_pick_the_same_branch(angles_deg):
    """The quaternion path resolves the lock the way the matrix path does.

    Two functions may each be defensible alone and still leave a caller that
    converts one way and checks the other reading a spurious difference, so
    the convention -- yaw pinned to 0, the remainder folded into pitch -- has
    to be one convention rather than two.
    """
    roll, pitch, yaw = _radians(angles_deg)
    C = euler_to_rotation_matrix(roll, pitch, yaw)
    q = euler_to_quat(roll, pitch, yaw)

    np.testing.assert_allclose(
        quat_to_euler(q), rotation_matrix_to_euler(C), atol=1e-12
    )
    assert quat_to_euler(q)[2] == pytest.approx(0.0, abs=1e-12)


# The lock branch fires at |sin(roll)| >= 1 - LOCK_EPS, so it can snap roll by
# at most delta where cos(delta) = 1 - LOCK_EPS, i.e. delta = sqrt(2*LOCK_EPS)
# = 1.41e-6 rad. That snap is the whole cost of the branch and it bounds the
# reconstruction residual; the bound is derived here rather than written down,
# so changing the threshold moves the tolerance with it.
LOCK_EPS = 1e-12
SNAP_RAD = np.sqrt(2.0 * LOCK_EPS)


def test_the_lock_is_a_boundary_and_not_a_neighbourhood():
    """Approaching the lock must not degrade, and arriving must not jump.

    Written as a sweep because the defect was invisible at every offset
    tested: 89.999 deg was exact to 6e-12 while 90.0 was wrong by 0.5. A test
    that samples "near the lock" and stops there confirms the wrong thing.

    Both bounds are measured against the noise and against the defect, per
    CLAUDE.md. Worst observed on this machine: 1.6e-7 of reconstruction (a
    mid-band offset paying the full snap) against a 2.8e-6 gate and a 0.906
    defect, and 6.4e-11 of path disagreement against a 1e-9 gate and a
    3.14 rad defect -- the broken exact-lock case put the two paths a
    half-turn apart in pitch and in yaw.
    """
    worst_residual = 0.0
    worst_disagreement = 0.0
    for offset_deg in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9, 1e-12, 0.0):
        for sign in (+1.0, -1.0):
            roll = sign * np.deg2rad(90.0 - offset_deg)
            pitch, yaw = np.deg2rad(-40.0), np.deg2rad(25.0)
            C = euler_to_rotation_matrix(roll, pitch, yaw)
            recovered = quat_to_euler(euler_to_quat(roll, pitch, yaw))
            worst_residual = max(
                worst_residual, np.abs(euler_to_rotation_matrix(*recovered) - C).max()
            )
            worst_disagreement = max(
                worst_disagreement,
                np.abs(recovered - rotation_matrix_to_euler(C)).max(),
            )

    assert worst_residual < 2.0 * SNAP_RAD, (
        f"worst reconstruction residual through the lock: {worst_residual:.3g}, "
        f"against the {2.0 * SNAP_RAD:.3g} the threshold's own snap allows"
    )
    assert worst_disagreement < 1e-9, (
        "the two recovery paths part company through the lock by "
        f"{worst_disagreement:.3g} rad"
    )


def test_the_quaternion_the_test_feeds_really_is_the_attitude():
    """Guard the guard: `euler_to_quat` itself must be exact at the lock.

    Everything above reads `quat_to_euler` through a quaternion built by
    `euler_to_quat`. If that construction were the broken half, these tests
    would be measuring the wrong function and could be satisfied by a change
    in the wrong place.
    """
    for angles_deg in LOCKED_ANGLES_DEG:
        roll, pitch, yaw = _radians(angles_deg)
        C = euler_to_rotation_matrix(roll, pitch, yaw)
        C_from_q = quat_to_rotation_matrix(euler_to_quat(roll, pitch, yaw))
        np.testing.assert_allclose(C_from_q, C, atol=1e-14)


def test_away_from_the_lock_the_triple_itself_is_recovered():
    """Off the singularity the answer is unique, so pin the angles, not just C.

    A lock branch that fired too eagerly would pass every test above by
    flattening ordinary attitudes onto yaw = 0, so the threshold needs a test
    on the other side of it.
    """
    rng = np.random.default_rng(2)
    for _ in range(400):
        roll = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        pitch = rng.uniform(-np.pi, np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        recovered = quat_to_euler(euler_to_quat(roll, pitch, yaw))
        np.testing.assert_allclose(recovered, [roll, pitch, yaw], atol=1e-9)
