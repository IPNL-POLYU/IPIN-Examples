"""Smoke tests for the Chapter 7 scan-matching visualization example.

Beyond checking that the figures are written, these assert the facts the
figures assert, so a caption cannot quietly drift from the code: ICP and NDT
must both recover the motion the scans were generated with, and NDT's capture
range must widen with the voxel size.

Author: Li-Ta Hsu
References: Chapter 7, Sections 7.3.1-7.3.2, Eqs. (7.10)-(7.16)
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

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
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "PYTHONPATH": str(WORKSPACE_ROOT)})

        cls.result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ch7_slam.example_scan_matching_visualization",
                "--no-show",
            ],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        cls.figs_dir = WORKSPACE_ROOT / "ch7_slam" / "figs"

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
        """Repo rule: chapter figures live in chX_*/figs/."""
        stray = list((WORKSPACE_ROOT / "ch7_slam").glob("*.png"))

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

    def test_ndt_outcome_depends_on_step_size(self):
        """Pins the claim the score-surface figure makes.

        On this stepped objective ndt_align reaches the optimum at step_size
        0.5 but stalls at the 0.1 default, and *both* report converged=True --
        "converged" means the line search stopped improving, not that the
        answer is right. If this test ever fails because the 0.1 run starts
        succeeding, the figure's caption needs updating with it.
        """
        from core.slam.ndt import ndt_align

        converging_pose, _, _, _ = ndt_align(
            self.source, self.target, voxel_size=1.0, step_size=0.5
        )
        stalled_pose, stalled_iters, _, stalled_flag = ndt_align(
            self.source, self.target, voxel_size=1.0, step_size=0.1
        )

        np.testing.assert_allclose(
            converging_pose[:2], self.true_motion[:2], atol=0.05
        )
        self.assertGreater(
            np.linalg.norm(stalled_pose[:2] - self.true_motion[:2]), 0.25
        )
        self.assertLess(stalled_iters, 10)
        self.assertTrue(stalled_flag, "the stalled run still claims success")

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
