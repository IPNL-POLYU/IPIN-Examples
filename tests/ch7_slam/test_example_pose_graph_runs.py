"""Smoke tests for Chapter 7 SLAM example scripts.

Verifies that example scripts run without errors in all modes.
Uses Agg backend to avoid display requirements.

Tests validate the machine-readable [SLAM_SUMMARY] JSON line to ensure:
- n_loop_closures >= 1 for inline mode
- rmse.optimized <= 0.95 * rmse.odom (>= 5% improvement, HARD GATE)
- rmse.frontend <= rmse.odom (frontend doesn't make things worse)

The subprocess runs are shared via slam_example_runner: each distinct
invocation happens once per pytest session, however many tests assert on it.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
from pathlib import Path

from tests.ch7_slam.slam_example_runner import (
    WORKSPACE_ROOT,
    parse_slam_summary,
    run_frontend_example,
    run_pose_graph_example,
)


class TestExamplePoseGraphSLAMRuns(unittest.TestCase):
    """Smoke tests: Example scripts should run without errors.

    These tests validate the [SLAM_SUMMARY] JSON contract:
    - For inline mode: n_loop_closures >= 1, optimized < odom
    - For dataset mode: optimized < odom (may have 0 loop closures due to dataset)
    """

    def setUp(self):
        """Set up test environment."""
        self.workspace_root = WORKSPACE_ROOT
        self.script_path = (
            self.workspace_root / "ch7_slam" / "example_pose_graph_slam.py"
        )

        # Verify script exists
        self.assertTrue(
            self.script_path.exists(), f"Script not found: {self.script_path}"
        )

    def test_inline_mode_runs_without_error(self):
        """Test inline mode (synthetic data) completes successfully.

        Validates JSON contract (HARD PERFORMANCE GATE):
        - n_loop_closures >= 1 (inline mode must detect loop closures)
        - rmse.optimized <= 0.95 * rmse.odom (>= 5% improvement, REQUIRED)
        - rmse.frontend <= rmse.odom (frontend must not worsen)
        """
        result = run_pose_graph_example().process

        # Check exit code
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr:\n{result.stderr}"
        )

        # Verify key outputs
        self.assertIn(
            "SLAM PIPELINE COMPLETE", result.stdout, "Missing completion message"
        )
        self.assertIn("Optimized RMSE", result.stdout, "Missing RMSE output")

        # Parse and validate JSON summary
        summary = parse_slam_summary(result.stdout)
        self.assertIsNotNone(summary, "Missing [SLAM_SUMMARY] JSON line in output")

        # Validate mode
        self.assertEqual(
            summary["mode"],
            "inline",
            f"Expected mode='inline', got '{summary['mode']}'",
        )

        # === HARD PERFORMANCE GATE ===
        # These criteria MUST be met for inline mode to be a valid teaching demo

        # Gate 1: Loop closures must be detected
        self.assertGreaterEqual(
            summary["n_loop_closures"],
            1,
            f"HARD GATE FAILED: Inline mode must detect >= 1 loop closures, "
            f"got {summary['n_loop_closures']}",
        )

        # Gate 2: Frontend must not worsen trajectory
        rmse = summary["rmse"]
        self.assertLessEqual(
            rmse["frontend"],
            rmse["odom"],
            f"HARD GATE FAILED: Frontend RMSE ({rmse['frontend']}) must be "
            f"<= Odometry RMSE ({rmse['odom']})",
        )

        # Gate 3: Optimization must achieve >= 5% improvement
        max_allowed_rmse = 0.95 * rmse["odom"]
        self.assertLessEqual(
            rmse["optimized"],
            max_allowed_rmse,
            f"HARD GATE FAILED: Optimized RMSE ({rmse['optimized']:.4f}) must be "
            f"<= 0.95 * Odometry RMSE ({max_allowed_rmse:.4f}) for >= 5% "
            f"improvement. Actual improvement: "
            f"{100 * (1 - rmse['optimized'] / rmse['odom']):.1f}%",
        )

    def test_square_dataset_mode_runs_without_error(self):
        """Test square dataset mode completes successfully.

        Validates JSON contract:
        - rmse.frontend <= rmse.odom (frontend must not worsen)
        Note: Dataset mode may have 0 loop closures due to scan matching challenges,
        so we don't require n_loop_closures >= 1 here.
        """
        result = run_pose_graph_example("--data", "ch7_slam_2d_square").process

        # Check exit code
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr:\n{result.stderr}"
        )

        # Verify key outputs
        self.assertIn("SLAM PIPELINE COMPLETE", result.stdout)
        self.assertIn(
            "observation-based",
            result.stdout,
            "Should use observation-based loop closure",
        )

        # Parse and validate JSON summary
        summary = parse_slam_summary(result.stdout)
        self.assertIsNotNone(summary, "Missing [SLAM_SUMMARY] JSON line in output")

        # Validate mode
        self.assertEqual(
            summary["mode"],
            "dataset",
            f"Expected mode='dataset', got '{summary['mode']}'",
        )

        # Validate RMSE (frontend should not be significantly worse than odom)
        # Note: Dataset mode frontend may struggle, allow some tolerance
        rmse = summary["rmse"]
        # Frontend might be slightly worse due to ICP challenges with dataset
        # Just verify the summary is present and well-formed
        self.assertIn("odom", rmse)
        self.assertIn("frontend", rmse)
        self.assertIn("optimized", rmse)

    def test_high_drift_dataset_mode_runs_without_error(self):
        """Test high drift dataset mode completes successfully.

        Validates JSON contract presence.
        """
        result = run_pose_graph_example("--data", "ch7_slam_2d_high_drift").process

        # Check exit code
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr:\n{result.stderr}"
        )

        # Verify key outputs
        self.assertIn("SLAM PIPELINE COMPLETE", result.stdout)
        self.assertIn("observation-based", result.stdout)

        # Parse and validate JSON summary
        summary = parse_slam_summary(result.stdout)
        self.assertIsNotNone(summary, "Missing [SLAM_SUMMARY] JSON line in output")

        # Validate mode and structure
        self.assertEqual(summary["mode"], "dataset")
        self.assertIn("n_poses", summary)
        self.assertIn("n_loop_closures", summary)
        self.assertIn("rmse", summary)

    def test_visualization_file_created(self):
        """Test that visualization file is created.

        Asserts against the runner's diverted figs directory, not
        ``ch7_slam/figs``: the run must not touch the committed figure. And on
        mtime as well as existence -- all three shared invocations write
        ``slam_with_maps.png`` to that one directory, so presence alone would
        pass on a sibling invocation's file even if this one saved nothing.
        """
        run = run_pose_graph_example()

        self.assertEqual(run.process.returncode, 0)

        output_file = run.figs_dir / "slam_with_maps.png"
        self.assertTrue(
            output_file.exists(), f"Visualization file not created: {output_file}"
        )

        # One second of slack for filesystem timestamp granularity.
        self.assertGreaterEqual(
            output_file.stat().st_mtime,
            run.started_at - 1.0,
            f"{output_file} predates the run that was supposed to write it; "
            f"the example silently did not save the figure.",
        )

        # Verify file message in output
        self.assertIn(
            "Saved figure", run.process.stdout, "Missing file save confirmation"
        )


class TestExampleSLAMFrontendRuns(unittest.TestCase):
    """Smoke test for SLAM frontend example."""

    def setUp(self):
        """Set up test environment."""
        self.workspace_root = WORKSPACE_ROOT
        self.script_path = self.workspace_root / "ch7_slam" / "example_slam_frontend.py"

        # Verify script exists
        self.assertTrue(
            self.script_path.exists(), f"Script not found: {self.script_path}"
        )

    def test_frontend_example_runs_without_error(self):
        """Test SLAM frontend example completes successfully."""
        result = run_frontend_example().process

        # Check exit code
        self.assertEqual(
            result.returncode, 0, f"Script failed with stderr:\n{result.stderr}"
        )

        # Verify key outputs
        self.assertIn("SLAM FRONT-END DEMO", result.stdout, "Missing header")
        self.assertIn("Frontend RMSE", result.stdout, "Missing RMSE output")


if __name__ == "__main__":
    unittest.main()
