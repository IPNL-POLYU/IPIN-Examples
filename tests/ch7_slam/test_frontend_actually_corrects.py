"""The SLAM front-end corrects odometry, and the pipeline's gates check that.

This file used to assert the opposite. `example_pose_graph_slam` ran
`SlamFrontend2D.step()` for all 145 scans and returned poses bit-identical to
the odometry it was given; the example printed "[WARNING] Frontend poses
identical to odometry - ICP not working!" and nothing acted on it. The hard gate
asserted `rmse.frontend <= rmse.odom`, which a no-op satisfies exactly, so a
stage that did nothing passed a gate designed to prove it did something -- the
same shape as Chapter 6's frozen ZUPT detector, which scored well by never
moving. The tests were written to fail when the ICP was repaired, and they did.

Two bugs were behind it, both in units rather than in the algorithm:

  - `icp_point_to_point` returned Eq. (7.10)'s objective, a *sum* of squared
    errors, while every caller gated it with a threshold named and documented
    in metres. The sum grows with the correspondence count, so matching a
    360-point scan against a submap voxelised at 0.2 m cost ~1.2 from
    quantisation alone, and the front-end's `max_icp_residual=1.5` was
    demanding 0.065 m RMS against a 0.058 m floor. It rejected every alignment
    it was ever handed and fell back to the odometry prediction each time. The
    function now reports RMS per correspondence, in metres.
  - `_scan_to_map_alignment` never passed `max_correspondence_distance`, so
    correspondence gating (Eq. 7.11) was off. Scan points with no nearby map
    point were paired with distant ones, and ICP diverged on individual steps
    to residuals of 3.9e3 and 2.1e13.

Fixing the units also changed what the loop-closure threshold meant, and that
had to be retuned in the same breath: at the old 0.30 it admitted nine
geometrically wrong closures and the back-end optimised to 1.26 m, worse than
the 0.85 m odometry baseline it started from. The two populations separate
cleanly -- correct closures top out at 0.054 m RMS, wrong ones start at
0.150 m -- so 0.10 takes all 147 correct closures and no wrong ones.

  odometry 0.8488 m -> front-end 0.5344 m (+37.0%) -> optimised 0.3045 m (+64.1%)

Author: Li-Ta Hsu
References: Chapter 7, Section 7.3 (scan matching), Section 7.5 (pose graph)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

from tests.ch7_slam.slam_example_runner import (
    parse_slam_summary,
    run_pose_graph_example,
)


class TestFrontendActuallyCorrects(unittest.TestCase):
    """What the pipeline claims its middle stage does."""

    @classmethod
    def setUpClass(cls):
        run = run_pose_graph_example()
        cls.summary = parse_slam_summary(run.process.stdout)
        cls.stdout = run.process.stdout

    def test_the_frontend_runs_on_every_scan(self):
        """It is called; that part of the claim was always true."""
        self.assertEqual(self.summary["n_frontend_steps"], self.summary["n_scans"])

    def test_the_frontend_correction_is_reported(self):
        """The summary must expose the quantity a gate can test.

        frontend_used only says step() was called. Whether it changed anything
        is a different question, and it has its own field.
        """
        self.assertIn("frontend_correction_m", self.summary)

    def test_the_frontend_changes_the_poses(self):
        """The assertion this file exists for, now the right way round."""
        self.assertGreater(self.summary["frontend_correction_m"], 0.0)

    def test_the_example_no_longer_warns(self):
        """The warning was the example telling us; it must be gone, not muted."""
        self.assertNotIn("ICP not working", self.stdout)

    def test_the_frontend_strictly_improves_on_odometry(self):
        """Promoted from `<=`, which a no-op satisfied exactly.

        The margin is deliberately loose: the point is that scan-to-map
        alignment earns something, not that it earns 37% on this seed.
        """
        rmse = self.summary["rmse"]

        self.assertLess(rmse["frontend"], 0.9 * rmse["odom"])

    def test_the_backend_improves_on_the_frontend(self):
        """Each stage has to pay for itself, not ride on the one before it."""
        rmse = self.summary["rmse"]

        self.assertLess(rmse["optimized"], rmse["frontend"])
        self.assertLess(rmse["optimized"], 0.6 * rmse["odom"])

    def test_loop_closures_are_not_admitted_by_the_hundred(self):
        """A guard on the threshold that retuning could quietly loosen.

        Accepting every candidate is not free: at the old gate the back-end
        took nine wrong closures and finished worse than odometry. Detection
        finds ~147 correct ones here, so a large excess means the residual
        threshold has drifted back above the 0.054/0.150 m separation.
        """
        self.assertLessEqual(self.summary["n_loop_closures"], 155)


if __name__ == "__main__":
    unittest.main()
