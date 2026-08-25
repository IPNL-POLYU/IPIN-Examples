"""Angle differences must be wrapped, including where nothing wraps today.

An audit of every angular reduction in `core/` and the chapter examples,
following the same defect in the Ch2 dataset generator (reported 360 deg of
rotation round-trip error, which is the identity) and the Ch6 environmental
sensors generator (`min(|d|, 2pi - |d|)`, which returned negative errors and
understated the mean by 24%).

**No chapter example had a live defect.** Every angular error that is actually
reported already went through a correct helper. What the sweep found was one
real bug in `core/` and four latent subtractions whose inputs happen not to
cross the branch cut today:

- `core/estimators/factor_graph.py` built a bearing residual as
  `predicted - z` with no wrap. Both terms come from `atan2` and so lie in
  (-pi, pi], making the raw difference reach 2pi across the cut -- the exact
  case `core.utils.angle_diff`'s own docstring names ("measured = +179 deg,
  predicted = -179 deg -> innovation = 358 deg (WRONG!)"). An optimiser
  minimising that gets a spurious gradient wherever the anchor lies roughly
  west of its estimate. Latent only because that demo's three anchors sit at
  bearings of -128.7, -39.8 and +78.7 deg.
- `ch2_coords` gated its Euler round-trip `PASS if error < 1e-9` on an
  unwrapped difference, so a reader swapping in a yaw of 200 deg would see
  `FAIL` for a perfect round-trip.
- `ch7_slam` measured loop closure as `abs(end_yaw - start_yaw)`; a closed loop
  returns to its start heading modulo 2pi, so a perfect closure can print
  360 deg.
- `ch7_slam` also fed an unwrapped yaw difference to the 1e-6 gate that exists
  to catch the front-end returning odometry untouched -- the check CLAUDE.md
  records as having been written to fail when the front-end was repaired. Two
  representations of one heading differ by 2pi, so an unwrapped diff could call
  identical poses different and let a no-op through.
- `ch8_sensor_fusion` reported its extrinsic rotation error the same way, safe
  at the 30 deg it uses.

All five are fixed. This file guards the property rather than the instances:
it asserts the shared helpers agree and are correct across the branch cut, so
any future site that routes through them is right by construction. **The
argument for wrapping is not that today's numbers are wrong -- four of the five
printed identical output before and after -- but that a latent branch-cut bug
is invisible until someone changes an angle, and then it lies.**

Author: Li-Ta Hsu
References: Chapters 2, 3, 6, 7, 8
"""

import unittest

import numpy as np

from core.sensors import wrap_angle_diff
from core.sensors.pdr import wrap_heading
from core.utils import angle_diff, wrap_angle

#: Every wrap helper in the repo. They are separate functions with separate
#: docstrings, and a caller picks whichever the neighbouring code used, so they
#: had better agree.
DIFF_HELPERS = {
    "core.utils.angle_diff": angle_diff,
    "core.sensors.wrap_angle_diff": wrap_angle_diff,
}


class TestWrapHelpersAgree(unittest.TestCase):
    """The helpers a caller might reach for are interchangeable and correct."""

    #: (a, b, expected a - b in degrees). The first three straddle the cut.
    CASES = [
        (179.0, -179.0, 358.0 - 360.0),
        (-179.0, 179.0, 360.0 - 358.0),
        (180.0, -180.0, 0.0),
        (10.0, -10.0, 20.0),
        (0.0, 0.0, 0.0),
        # A difference beyond a full turn, which `min(|d|, 2pi - |d|)` handles
        # by returning a negative number. This is the ch6 case.
        (720.0 + 5.0, 0.0, 5.0),
        (-720.0 - 5.0, 0.0, -5.0),
    ]

    def test_difference_helpers_are_correct_across_the_branch_cut(self) -> None:
        for name, fn in DIFF_HELPERS.items():
            for a_deg, b_deg, want_deg in self.CASES:
                with self.subTest(helper=name, a=a_deg, b=b_deg):
                    got = np.rad2deg(fn(np.deg2rad(a_deg), np.deg2rad(b_deg)))
                    self.assertAlmostEqual(
                        float(got),
                        want_deg,
                        places=9,
                        msg=f"{name}({a_deg}, {b_deg}) = {got}, want {want_deg}",
                    )

    def test_single_angle_helpers_agree_with_the_difference_helpers(self) -> None:
        """`wrap_angle(a - b)` and `wrap_heading(a - b)` must match too.

        The ch7 fix uses the first and the ch6 PDR example the second, so a
        divergence between them would be a real inconsistency in the repo.
        """
        for a_deg, b_deg, want_deg in self.CASES:
            d = np.deg2rad(a_deg) - np.deg2rad(b_deg)
            with self.subTest(a=a_deg, b=b_deg):
                self.assertAlmostEqual(
                    float(np.rad2deg(wrap_angle(d))), want_deg, places=9
                )
                self.assertAlmostEqual(
                    float(np.rad2deg(wrap_heading(d))), want_deg, places=9
                )

    def test_the_helpers_vectorise(self) -> None:
        """Several call sites pass whole arrays; none may silently degrade.

        `ch6_dead_reckoning/example_pdr.py` calls `wrap_heading` on an array of
        1800 samples, and the ch2 round-trip gate now calls `angle_diff` on a
        3-vector of Euler angles.
        """
        a = np.deg2rad(np.array([179.0, -179.0, 10.0, 725.0]))
        b = np.deg2rad(np.array([-179.0, 179.0, -10.0, 0.0]))
        want = np.array([-2.0, 2.0, 20.0, 5.0])

        np.testing.assert_allclose(np.rad2deg(angle_diff(a, b)), want, atol=1e-9)
        np.testing.assert_allclose(np.rad2deg(wrap_angle(a - b)), want, atol=1e-9)
        np.testing.assert_allclose(np.rad2deg(wrap_heading(a - b)), want, atol=1e-9)

    def test_the_naive_reductions_really_are_wrong(self) -> None:
        """The two forms this sweep removed, kept so the difference is visible.

        Without this the file only says the helpers work, which proves nothing
        about why the call sites had to change. Both naive forms are checked on
        the input that actually broke them, so the assertions below would fail
        if someone "simplified" a helper back toward either.
        """
        # The ch2/ch7/ch8 form: a raw subtraction across the branch cut.
        raw = np.deg2rad(179.0) - np.deg2rad(-179.0)
        self.assertAlmostEqual(float(np.rad2deg(raw)), 358.0, places=6)
        self.assertAlmostEqual(
            float(np.rad2deg(angle_diff(np.deg2rad(179.0), np.deg2rad(-179.0)))),
            -2.0,
            places=9,
        )

        # The ch6 form: min(|d|, 2pi - |d|), negative once |d| exceeds 2pi.
        d = abs(np.deg2rad(725.0) - 0.0)
        naive = min(d, 2 * np.pi - d)
        self.assertLess(naive, 0.0, "the ch6 form no longer misbehaves at 725 deg")
        self.assertAlmostEqual(
            float(np.rad2deg(abs(wrap_angle(np.deg2rad(725.0))))), 5.0, places=9
        )


if __name__ == "__main__":
    unittest.main()
