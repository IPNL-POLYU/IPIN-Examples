"""Simulated ground truth must describe a motion something could perform.

A dataset that no platform could execute makes every estimator downstream look
bad, and the resulting number gets reported as an accuracy. Chapter 8's fusion
RMSE was 0.739 m against 0.035 m ranging entirely because its trajectory turned
90 degrees inside one sample -- 9000 deg/s, which the IMU forward model
rendered as 5.1 g. Rounding the corners took the same filter to 0.167 m with
its median unchanged: nothing about the estimator had ever been wrong.

That is not a figure defect, so `030-figures-and-claims` does not catch it, and
it is not visible in any plot -- the trajectory looked like a tidy rectangle.
The only way to see it is to differentiate the truth and ask whether the answer
is achievable.

Author: Li-Ta Hsu
"""

import glob
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generous bounds: these are smoke alarms for discontinuities, not fidelity
# standards. A person turns at well under 180 deg/s and an indoor platform
# rarely exceeds 1 g, so anything past these is a step rather than a manoeuvre.
MAX_TURN_RATE_DEG_S = 360.0
MAX_ACCEL_M_S2 = 5.0 * 9.81

# Datasets known to contain a discontinuity, predating this check.
#
# THIS LIST MUST ONLY SHRINK -- same ratchet as tests/test_repo_conventions.py.
#
# ch6_foot_zupt_walk: the walk/stance generator steps velocity between
# 1.75 m/s and 0 within one 10 ms sample, implying 8.9 g. Real gait does stop
# the foot, but it ramps; and a foot-mounted IMU legitimately sees several g at
# heel strike, so the entry is about the step, not the magnitude. Left alone
# because fixing it changes Chapter 6's stance-detection figures, which is a
# deliberate change and not a side effect of adding this test.
KNOWN_DISCONTINUOUS = {
    "ch6_foot_zupt_walk",
}


def _truth_datasets():
    """Every simulated truth file carrying a time base and a velocity."""
    found = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "data/sim/*/truth.npz"))):
        with np.load(path) as data:
            if {"t", "v_xy"}.issubset(set(data.files)):
                found.append(path)
    return found


def _dataset_name(path):
    """Directory name, used as the test id and the allowlist key."""
    return os.path.basename(os.path.dirname(path))


@pytest.mark.parametrize("path", _truth_datasets(), ids=_dataset_name)
def test_truth_acceleration_is_achievable(path):
    """Differentiating the truth must not imply an impossible acceleration.

    This is the check that would have caught the Chapter 8 corners before they
    were mistaken for a fusion result.
    """
    name = _dataset_name(path)
    with np.load(path) as data:
        t, v_xy = data["t"], data["v_xy"]

    accel = np.linalg.norm(np.gradient(v_xy, t, axis=0), axis=1).max()

    if name in KNOWN_DISCONTINUOUS:
        pytest.skip(f"known pre-existing discontinuity ({name})")

    assert accel < MAX_ACCEL_M_S2, (
        f"{name} implies {accel:.1f} m/s^2 ({accel / 9.81:.1f} g) at a peak "
        f"speed of {np.linalg.norm(v_xy, axis=1).max():.2f} m/s. That is a step "
        f"in the velocity, not a manoeuvre, and no estimator can follow it -- "
        f"the error it causes will be reported as the estimator's."
    )


@pytest.mark.parametrize("path", _truth_datasets(), ids=_dataset_name)
def test_truth_turn_rate_is_achievable(path):
    """Heading must not jump between samples.

    Separate from acceleration because a trajectory can rotate without
    translating, and Chapter 8's did: the yaw step was 9000 deg/s.
    """
    name = _dataset_name(path)
    with np.load(path) as data:
        if "yaw" not in data.files:
            pytest.skip(f"{name} carries no heading")
        t, yaw = data["t"], data["yaw"]

    turn_rate = np.degrees(np.abs(np.gradient(np.unwrap(yaw), t))).max()

    if name in KNOWN_DISCONTINUOUS:
        pytest.skip(f"known pre-existing discontinuity ({name})")

    assert turn_rate < MAX_TURN_RATE_DEG_S, (
        f"{name} turns at {turn_rate:.0f} deg/s. A pedestrian manages well "
        f"under 180; this is a heading step, and the IMU forward model will "
        f"faithfully turn it into an impossible gyro reading."
    )


@pytest.mark.parametrize("path", _truth_datasets(), ids=_dataset_name)
def test_position_and_velocity_agree(path):
    """The stored velocity must be the derivative of the stored position.

    Cheap, and it guards the two tests above: they read v_xy, so a generator
    that wrote a plausible velocity next to an unrelated position would pass
    both while still being unusable.
    """
    name = _dataset_name(path)
    with np.load(path) as data:
        t, p_xy, v_xy = data["t"], data["p_xy"], data["v_xy"]

    if name in KNOWN_DISCONTINUOUS:
        # A velocity step also breaks this test, and for the same reason: a
        # central difference straddling the step returns half of it. Skipping
        # rather than loosening, so the entry disappears from all three tests
        # together when the generator is fixed.
        pytest.skip(f"known pre-existing discontinuity ({name})")

    derived = np.gradient(p_xy, t, axis=0)
    speed = np.linalg.norm(v_xy, axis=1)
    scale = max(float(speed.max()), 1e-6)
    # Compare in the interior: np.gradient's one-sided ends differ at a
    # discontinuity, which the tests above are the right place to catch.
    disagreement = np.linalg.norm(derived - v_xy, axis=1)[1:-1].max()

    assert disagreement < 0.1 * scale, (
        f"{name}: stored velocity disagrees with d(position)/dt by "
        f"{disagreement:.3f} m/s against a peak speed of {scale:.2f} m/s."
    )
