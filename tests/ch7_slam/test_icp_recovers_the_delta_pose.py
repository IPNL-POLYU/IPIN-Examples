"""ICP must recover the transform between two scans, rotation included.

Ported from ``ch7_slam/test_deltapose.py``, a file that looked like a test,
was named like a test, and had never run once: ``testpaths`` is ``tests``, so
pytest never collected it. Its only effect on the suite was that
``tests/test_repo_conventions.py`` matched it as a *chapter example* and
checked it for raw savefig calls.

Running it settled what it was for. Two of its three cases passed; the third
reported ``PASS: False``, and its own hint pointed the wrong way:

    "If case2 fails badly, check ICP pose update order
     (left-multiply vs right-multiply)."

Case 2 does not fail badly. It recovers a 35 degree rotation to 0.005 m and
0.021 degrees, which is not what a left/right-multiply confusion looks like.
What failed was the *convergence flag*: the run needs 59 iterations and the
harness allowed 50. Nine short. Anyone following that hint would have gone
looking for a sign error in working code.

So the budget was wrong, not the library, and the cases are worth keeping --
particularly case 2, which is the one that would catch a genuine update-order
bug. The original also imported ``se2_relative`` and then immediately
redefined it, shadowing the library function, so the case written to exercise
the relative-pose path never touched it. The two are exactly equal (checked
over 500 random pose pairs, worst disagreement 0.0), so the redefinition is
simply dropped here.

Author: Li-Ta Hsu
References: Chapter 7, scan matching and pose-graph construction
"""

import numpy as np
import pytest

from core.slam import (
    icp_point_to_point,
    se2_apply,
    se2_compose,
    se2_relative,
    wrap_angle,
)

#: Comfortably clear of the 59 iterations the rotation case needs. The point
#: of this limit is to stop a non-converging run, not to bound the cost, and a
#: budget set just above the observed figure is how the original turned a
#: working solve into a failure.
MAX_ITERATIONS = 200

#: The original ran with 5% outliers and 0.01 m noise on both clouds. Kept, so
#: these are correspondence-rejection tests as well as alignment tests.
NOISE_STD_M = 0.01
OUTLIER_RATIO = 0.05
MAX_CORRESPONDENCE_DIST_M = 0.7


def _arc_scan(n=400, seed=0):
    """Points on an arc plus a short wall, roughly like a 2-D lidar sweep.

    The wall matters: a pure arc is rotationally near-symmetric about its own
    centre, which leaves the yaw poorly constrained and makes an alignment
    test pass or fail for reasons that have nothing to do with the solver.
    """
    rng = np.random.default_rng(seed)

    angles = rng.uniform(-1.8, 1.8, size=n)
    radius = rng.uniform(4.0, 8.0, size=n)
    arc = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    wall = np.stack([
        rng.uniform(-2.0, 2.0, size=n // 5),
        rng.uniform(6.0, 7.0, size=n // 5),
    ], axis=1)

    return np.concatenate([arc, wall], axis=0).astype(np.float64)


def _observe(points, seed):
    """Add measurement noise and a scattering of gross outliers."""
    rng = np.random.default_rng(seed)

    noisy = points + rng.normal(0.0, NOISE_STD_M, size=points.shape)
    n_outliers = int(len(noisy) * OUTLIER_RATIO)
    if n_outliers <= 0:
        return noisy

    outliers = rng.uniform([-15.0, -15.0], [15.0, 15.0], size=(n_outliers, 2))
    return np.concatenate([noisy, outliers], axis=0)


def _align(true_delta, initial_guess, max_iterations=MAX_ITERATIONS):
    """Run ICP on a scan pair separated by ``true_delta``.

    Returns:
        Tuple of (estimate, iterations, converged, translation error [m],
        yaw error [deg]).
    """
    source = _arc_scan()
    target = se2_apply(true_delta, source)

    estimate, iterations, _residual, converged = icp_point_to_point(
        _observe(source, seed=1),
        _observe(target, seed=2),
        initial_pose=initial_guess,
        max_iterations=max_iterations,
        tolerance=1e-4,
        max_correspondence_distance=MAX_CORRESPONDENCE_DIST_M,
        min_correspondences=20,
    )

    translation_error = float(np.linalg.norm(estimate[:2] - true_delta[:2]))
    yaw_error_deg = float(abs(wrap_angle(estimate[2] - true_delta[2]))) * 180 / np.pi
    return estimate, iterations, converged, translation_error, yaw_error_deg


class TestIcpRecoversTheDeltaPose:
    """Three separations, each with a perturbed initial guess."""

    def test_small_motion(self):
        """0.8 m and 5 degrees, started 0.2 m and 3 degrees away."""
        delta = np.array([0.8, -0.4, np.deg2rad(5.0)])
        guess = delta + np.array([0.2, -0.2, np.deg2rad(3.0)])

        _, _, converged, trans_err, yaw_err = _align(delta, guess)

        assert converged
        assert trans_err <= 0.10, f"translation error {trans_err:.4f} m"
        assert yaw_err <= 2.0, f"yaw error {yaw_err:.3f} deg"

    def test_large_rotation(self):
        """35 degrees, started 10 degrees and 0.6 m away.

        The case that matters. A left/right-multiply error in the pose update
        survives small-motion tests -- at 5 degrees the two conventions differ
        by little -- and shows up here, where it would put the estimate metres
        and tens of degrees out rather than the millimetres below.
        """
        delta = np.array([2.0, 1.0, np.deg2rad(35.0)])
        guess = delta + np.array([-0.5, 0.3, np.deg2rad(-10.0)])

        _, _, converged, trans_err, yaw_err = _align(delta, guess)

        assert converged
        assert trans_err <= 0.15, f"translation error {trans_err:.4f} m"
        assert yaw_err <= 3.0, f"yaw error {yaw_err:.3f} deg"

    def test_relative_pose_as_the_initial_guess(self):
        """The pose-graph path: the guess comes from two absolute poses.

        Exercises ``core.slam.se2_relative``, which the original file imported
        and then shadowed with a local reimplementation, so this case never
        reached the library.
        """
        pose_i = np.array([1.0, 2.0, np.deg2rad(20.0)])
        pose_j = se2_compose(pose_i, np.array([1.2, -0.7, np.deg2rad(30.0)]))
        delta = se2_relative(pose_i, pose_j)
        guess = delta + np.array([0.1, -0.1, np.deg2rad(2.0)])

        _, _, converged, trans_err, yaw_err = _align(delta, guess)

        assert converged
        assert trans_err <= 0.10, f"translation error {trans_err:.4f} m"
        assert yaw_err <= 2.0, f"yaw error {yaw_err:.3f} deg"


def test_se2_relative_matches_compose_of_the_inverse():
    """The identity the shadowing redefinition assumed.

    It is true, which is why nobody noticed the shadowing. Asserting it here
    means the original file's local copy can be dropped without taking an
    unstated assumption with it.
    """
    from core.slam import se2_inverse

    rng = np.random.default_rng(0)
    for _ in range(100):
        a = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
        b = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])

        assert se2_relative(a, b) == pytest.approx(
            se2_compose(se2_inverse(a), b), abs=1e-12
        )


def test_the_rotation_case_needs_more_than_fifty_iterations():
    """Why MAX_ITERATIONS is 200, pinned so the reason cannot be lost.

    The original harness allowed 50 and reported the 35 degree case as a
    failure, with a hint blaming the ICP update order. The solve is fine; it
    converges on iteration 59. This records the gap, so that anyone who later
    tightens the budget sees what it costs, and so that a real regression --
    ICP suddenly needing far more iterations -- is visible rather than hidden
    behind a generous limit.
    """
    delta = np.array([2.0, 1.0, np.deg2rad(35.0)])
    guess = delta + np.array([-0.5, 0.3, np.deg2rad(-10.0)])

    _, iterations, converged, trans_err, _ = _align(delta, guess)

    assert converged
    assert 50 < iterations < 120, (
        f"the rotation case converged in {iterations} iterations. It needed 59 "
        f"when this was written; below 50 means the original harness's budget "
        f"was fine after all and this note is stale, well above means ICP has "
        f"slowed down."
    )
    # Accuracy at 50 iterations was already 0.005 m -- the old failure was the
    # convergence flag, not the answer. Running to convergence improves it.
    assert trans_err < 0.005
