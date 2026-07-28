"""The SLAM front-end is a no-op, and the pipeline's gates do not notice.

example_pose_graph_slam runs SlamFrontend2D.step() for every scan -- 145 of
145 -- and the poses it returns are bit-identical to the odometry it was given.
The example knows: it computes max|frontend - odom|, gets 0.0000 m, and prints
"[WARNING] Frontend poses identical to odometry - ICP not working!". Its module
docstring states the requirement outright: "frontend_poses MUST differ from
odom_poses (ICP must be working)".

Nothing acts on any of that. The hard gate in test_example_pose_graph_inline
asserts rmse.frontend <= rmse.odom, which a no-op satisfies exactly, and the
printed summary reports "Frontend improvement: -0.00%" without treating it as a
failure. So a stage that does nothing passes a gate designed to prove it does
something -- the same shape as Chapter 6's frozen ZUPT detector, which scored
well by never moving.

These tests assert the front-end is STILL a no-op, so that repairing the ICP
turns them red and forces the gate to be promoted rather than leaving a silent
allowance behind. See tests/ch6_dead_reckoning/test_methods_actually_move.py
for the same construction.

Author: Li-Ta Hsu
References: Chapter 7, Section 7.3 (scan matching), Section 7.5 (pose graph)
"""

import json
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

from tests.ch7_slam.slam_example_runner import (
    parse_slam_summary,
    run_pose_graph_example,
)

# The front-end is known to return odometry unchanged. Remove this once the
# ICP is fixed; the tests below are written to fail at that moment.
FRONTEND_IS_KNOWN_NO_OP = True


class TestFrontendActuallyCorrects(unittest.TestCase):
    """What the pipeline claims its middle stage does."""

    @classmethod
    def setUpClass(cls):
        run = run_pose_graph_example()
        cls.summary = parse_slam_summary(run.process.stdout)
        cls.stdout = run.process.stdout

    def test_the_frontend_runs_on_every_scan(self):
        """It is called; that part of the claim is true."""
        self.assertEqual(self.summary["n_frontend_steps"], self.summary["n_scans"])

    def test_the_frontend_correction_is_reported(self):
        """The summary must expose the quantity a gate can test.

        frontend_used only says step() was called. Whether it changed anything
        is a different question, and it now has its own field.
        """
        self.assertIn("frontend_correction_m", self.summary)

    def test_the_frontend_still_changes_nothing(self):
        """Pinned as broken, so a fix flips this test rather than passing quietly.

        When the ICP starts correcting, this fails with the correction it
        achieved; set FRONTEND_IS_KNOWN_NO_OP to False and turn the assertion
        around, and promote the hard gate from rmse.frontend <= rmse.odom to a
        strict improvement.
        """
        correction = self.summary["frontend_correction_m"]

        if FRONTEND_IS_KNOWN_NO_OP:
            self.assertEqual(
                correction,
                0.0,
                f"The front-end now corrects {correction} m. Flip "
                f"FRONTEND_IS_KNOWN_NO_OP and tighten the hard gate.",
            )
        else:
            self.assertGreater(correction, 0.0)

    def test_the_example_says_so_out_loud(self):
        """The warning is printed; this pins that it is not quietly dropped.

        If the front-end is fixed, this fails too -- both halves of the story
        move together.
        """
        if FRONTEND_IS_KNOWN_NO_OP:
            self.assertIn("ICP not working", self.stdout)

    def test_the_backend_is_what_earns_the_improvement(self):
        """The pipeline result is real, and it comes from the pose graph.

        Worth separating: the chapter's headline is not wrong, it is just
        attributable to one stage rather than three.
        """
        rmse = self.summary["rmse"]

        self.assertEqual(rmse["frontend"], rmse["odom"])
        self.assertLess(rmse["optimized"], 0.6 * rmse["odom"])


if __name__ == "__main__":
    unittest.main()
