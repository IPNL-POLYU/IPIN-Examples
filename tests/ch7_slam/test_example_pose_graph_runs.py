"""Smoke tests for Chapter 7 SLAM example scripts.

Verifies that example scripts run without errors in all modes.
Uses Agg backend to avoid display requirements.

Author: Li-Ta Hsu
Date: December 2025
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestExamplePoseGraphSLAMRuns(unittest.TestCase):
    """Smoke tests: Example scripts should run without errors."""

    def setUp(self):
        """Set up test environment."""
        self.python_exe = sys.executable
        self.workspace_root = Path(__file__).parent.parent.parent
        self.script_path = self.workspace_root / "ch7_slam" / "example_pose_graph_slam.py"
        
        # Verify script exists
        self.assertTrue(self.script_path.exists(), 
                       f"Script not found: {self.script_path}")

    def test_inline_mode_runs_without_error(self):
        """Test inline mode (synthetic data) completes successfully."""
        import os
        env = os.environ.copy()
        env.update({
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(self.workspace_root),
        })
        
        result = subprocess.run(
            [self.python_exe, "-m", "ch7_slam.example_pose_graph_slam"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        # Check exit code
        self.assertEqual(result.returncode, 0, 
                        f"Script failed with stderr:\n{result.stderr}")
        
        # Verify key outputs
        self.assertIn("SLAM PIPELINE COMPLETE", result.stdout,
                     "Missing completion message")
        self.assertIn("Optimized RMSE", result.stdout,
                     "Missing RMSE output")

    def test_square_dataset_mode_runs_without_error(self):
        """Test square dataset mode completes successfully."""
        import os
        env = os.environ.copy()
        env.update({
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(self.workspace_root),
        })
        
        result = subprocess.run(
            [self.python_exe, "-m", "ch7_slam.example_pose_graph_slam", 
             "--data", "ch7_slam_2d_square"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        # Check exit code
        self.assertEqual(result.returncode, 0,
                        f"Script failed with stderr:\n{result.stderr}")
        
        # Verify key outputs
        self.assertIn("SLAM PIPELINE COMPLETE", result.stdout)
        self.assertIn("observation-based", result.stdout,
                     "Should use observation-based loop closure")
        self.assertIn("improvement", result.stdout,
                     "Should show RMSE improvement")

    def test_high_drift_dataset_mode_runs_without_error(self):
        """Test high drift dataset mode completes successfully."""
        import os
        env = os.environ.copy()
        env.update({
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(self.workspace_root),
        })
        
        result = subprocess.run(
            [self.python_exe, "-m", "ch7_slam.example_pose_graph_slam",
             "--data", "ch7_slam_2d_high_drift"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        # Check exit code
        self.assertEqual(result.returncode, 0,
                        f"Script failed with stderr:\n{result.stderr}")
        
        # Verify key outputs
        self.assertIn("SLAM PIPELINE COMPLETE", result.stdout)
        self.assertIn("observation-based", result.stdout)

    def test_visualization_file_created(self):
        """Test that visualization file is created."""
        import os
        env = os.environ.copy()
        env.update({
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(self.workspace_root),
        })
        
        # Run script
        result = subprocess.run(
            [self.python_exe, "-m", "ch7_slam.example_pose_graph_slam"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        self.assertEqual(result.returncode, 0)
        
        # Check output file exists
        output_file = self.workspace_root / "ch7_slam" / "figs" / "slam_with_maps.png"
        self.assertTrue(output_file.exists(),
                       f"Visualization file not created: {output_file}")
        
        # Verify file message in output
        self.assertIn("Saved figure", result.stdout,
                     "Missing file save confirmation")


class TestExampleSLAMFrontendRuns(unittest.TestCase):
    """Smoke test for SLAM frontend example."""

    def setUp(self):
        """Set up test environment."""
        self.python_exe = sys.executable
        self.workspace_root = Path(__file__).parent.parent.parent
        self.script_path = self.workspace_root / "ch7_slam" / "example_slam_frontend.py"
        
        # Verify script exists
        self.assertTrue(self.script_path.exists(),
                       f"Script not found: {self.script_path}")

    def test_frontend_example_runs_without_error(self):
        """Test SLAM frontend example completes successfully."""
        import os
        env = os.environ.copy()
        env.update({
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(self.workspace_root),
        })
        
        result = subprocess.run(
            [self.python_exe, "-m", "ch7_slam.example_slam_frontend"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        # Check exit code
        self.assertEqual(result.returncode, 0,
                        f"Script failed with stderr:\n{result.stderr}")
        
        # Verify key outputs
        self.assertIn("SLAM FRONT-END DEMO", result.stdout,
                     "Missing header")
        self.assertIn("Frontend RMSE", result.stdout,
                     "Missing RMSE output")


if __name__ == "__main__":
    unittest.main()
