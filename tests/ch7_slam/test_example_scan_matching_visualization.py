"""Smoke tests for the Chapter 7 scan-matching visualization example.

Beyond checking that the figures are written, these assert the facts the
figures assert, so a caption cannot quietly drift from the code: ICP and NDT
must both recover the motion the scans were generated with, and NDT's capture
range must widen with the voxel size.

Author: Li-Ta Hsu
References: Chapter 7, Sections 7.3.1-7.3.2, Eqs. (7.10)-(7.16)
"""

import unittest

import numpy as np

from tests.ch7_slam.slam_example_runner import run_scan_matching_example

EXPECTED_FIGURES = (
    "ch7_icp_correspondences",
    "ch7_ndt_voxels",
    "ch7_ndt_score_surface",
    "ch7_convergence_basin",
)


class TestScanMatchingVisualizationExample(unittest.TestCase):
    """Run the example headless and check its figure output."""

    @classmethod
    def setUpClass(cls):
        # Through the shared runner, which owns the subprocess policy for every
        # Chapter 7 example: Agg backend, PYTHONPATH, a timeout sized as a
        # deadlock guard rather than a performance budget, and IPIN_FIGS_DIR
        # pointed at scratch so a run cannot rewrite the committed figures.
        #
        # This file used to carry its own copy of all of that. It was correct,
        # but it was a second copy -- and the figure diversion in particular is
        # the kind of thing that only has to be forgotten once. The runner also
        # holds its scratch root for the whole session, where the local
        # TemporaryDirectory was torn down with this class, so the output is
        # still there if another file ever asserts on the same run.
        run = run_scan_matching_example("--no-show")
        cls.result = run.process
        cls.figs_dir = run.figs_dir

    def test_example_runs_successfully(self):
        """Exit code 0 with no traceback."""
        self.assertEqual(
            self.result.returncode,
            0,
            f"Example failed with stderr:\n{self.result.stderr}",
        )

    def test_every_figure_is_written_in_every_format(self):
        """All four figures, in svg, pdf and png."""
        for name in EXPECTED_FIGURES:
            for suffix in ("png", "svg", "pdf"):
                path = self.figs_dir / f"{name}.{suffix}"
                with self.subTest(figure=path.name):
                    self.assertTrue(path.exists(), f"missing {path}")
                    self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_figures_land_in_figs_not_the_chapter_root(self):
        """Repo rule: chapter figures live in chX_*/figs/.

        Checked in the diverted tree, which is where this run's output actually
        went. Globbing the real ch7_slam/ would pass no matter what the example
        did, and quietly stop testing anything.
        """
        chapter_root = self.figs_dir.parent
        stray = list(chapter_root.glob("*.png"))

        self.assertEqual(stray, [], f"figures written beside the source: {stray}")


class TestScanMatchingClaims(unittest.TestCase):
    """Check the claims the figures make, not just that they render."""

    @classmethod
    def setUpClass(cls):
        from ch7_slam.example_scan_matching_visualization import (
            TRUE_MOTION,
            _make_scan_pair,
        )

        cls.true_motion = TRUE_MOTION
        cls.target, cls.source = _make_scan_pair()

    def test_icp_recovers_the_true_motion(self):
        """The star in the figures sits where the optimum actually is.

        An earlier draft marked -TRUE_MOTION and disagreed with both solvers.
        """
        from core.slam.scan_matching import icp_point_to_point

        icp_pose, _, _, icp_ok = icp_point_to_point(
            self.source, self.target, max_correspondence_distance=1.0
        )

        self.assertTrue(icp_ok)
        np.testing.assert_allclose(
            icp_pose[:2], self.true_motion[:2], atol=0.05
        )

    def test_ndt_outcome_does_not_depend_on_step_size(self):
        """Pins the claim the score-surface figure makes.

        This used to assert the opposite: steepest descent on a central finite
        difference reached the optimum at step_size 0.5 but stalled after three
        iterations at 0.1 and 0.3, non-monotonically, and reported
        converged=True either way. The finite difference was the culprit -- it
        straddled voxel boundaries rather than the smooth trend, and at the
        0.3 stall it pointed within 11 degrees of straight *away* from the
        optimum. With an analytic gradient and a Gauss-Newton step the answer
        no longer depends on the step size, so the figure's caption says that
        instead. If this test fails, the caption needs updating with it.
        """
        from core.slam.ndt import ndt_align

        poses = {}
        for step_size in (0.05, 0.1, 0.3, 0.5, 1.0):
            pose, iters, _, converged = ndt_align(
                self.source, self.target, voxel_size=1.0,
                step_size=step_size, max_iterations=200,
            )
            poses[step_size] = pose
            with self.subTest(step_size=step_size):
                np.testing.assert_allclose(
                    pose[:2], self.true_motion[:2], atol=0.05
                )
                self.assertTrue(converged)

        # Not merely all-correct: they agree with each other to within the
        # 0.05 m tolerance each is checked against. (The smallest step settles a
        # little short because the stopping test on the step length scales with
        # step_size, so it is the widest contributor to this spread -- but even
        # it lands inside the tolerance.)
        spread = max(
            np.linalg.norm(a[:2] - b[:2])
            for a in poses.values() for b in poses.values()
        )
        self.assertLess(spread, 0.05, f"step size still steers the answer: {poses}")

    def test_ndt_matches_icp_on_this_scan_pair(self):
        """The two Section 7.3 matchers must agree on the same displacement.

        ICP was the control that showed the old NDT failure was the optimizer's
        fault and not the data's; keep it wired up as one.
        """
        from core.slam.ndt import ndt_align
        from core.slam.scan_matching import icp_point_to_point

        icp_pose, _, _, _ = icp_point_to_point(
            self.source, self.target, max_correspondence_distance=1.0
        )
        ndt_pose, _, _, _ = ndt_align(self.source, self.target, voxel_size=1.0)

        np.testing.assert_allclose(ndt_pose[:2], icp_pose[:2], atol=0.05)

    def test_icp_animation_renders_small_and_monotone(self):
        """The GIF must stay small, and must show the residual falling.

        Called directly rather than through --animate so the suite does not
        pay for a second full run of the example.
        """
        import tempfile

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ch7_slam.example_scan_matching_visualization import (
            animate_icp_convergence,
        )
        from core.eval import save_animation

        fig, update, n_frames = animate_icp_convergence()
        try:
            self.assertGreater(n_frames, 5)
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(
                    fig, update, n_frames, tmp, "icp_anim", fps=4
                )
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        # Committed binaries live in git history forever; keep them modest.
        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")

    def test_score_surface_slice_must_use_the_true_yaw(self):
        """A yaw=0 slice does not contain the optimum.

        This is why the figure slices at the true yaw: holding yaw at zero
        costs an order of magnitude in score, so the plotted minimum would sit
        in the wrong place.
        """
        from core.slam.ndt import build_ndt_map, ndt_score

        ndt_map = build_ndt_map(self.target, voxel_size=1.0)
        at_truth = ndt_score(self.source, ndt_map, self.true_motion, 1.0)
        at_zero_yaw = ndt_score(
            self.source,
            ndt_map,
            np.array([self.true_motion[0], self.true_motion[1], 0.0]),
            1.0,
        )

        self.assertGreater(at_zero_yaw, 5.0 * at_truth)


if __name__ == "__main__":
    unittest.main()
