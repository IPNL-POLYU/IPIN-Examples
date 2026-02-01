"""Strict inline mode tests for Chapter 7 SLAM example.

This test file enforces the HARD PERFORMANCE GATES for inline mode:
- frontend_used == True (SlamFrontend2D.step() must be called)
- n_frontend_steps == n_scans (every scan must go through frontend)
- n_loop_closures >= 1 (loop closures must be detected)
- rmse.frontend <= rmse.odom (frontend must not worsen)
- rmse.optimized <= 0.95 * rmse.odom (>= 5% improvement required)

If any assertion fails, the PR MUST be rejected.

Author: Li-Ta Hsu
Date: December 2025
"""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, Optional


def parse_slam_summary(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse the [SLAM_SUMMARY] JSON line from script output.
    
    Args:
        stdout: Standard output from the SLAM script.
    
    Returns:
        Parsed JSON dictionary, or None if not found.
    
    Raises:
        ValueError: If the summary line is malformed.
    """
    match = re.search(r'\[SLAM_SUMMARY\]\s*(\{.*\})', stdout)
    if not match:
        return None
    
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed SLAM_SUMMARY JSON: {e}")


class TestInlineModeHardGates(unittest.TestCase):
    """Hard performance gates for inline mode.
    
    These tests ensure inline mode demonstrates a valid SLAM teaching pipeline.
    All assertions are NON-NEGOTIABLE - if they fail, the implementation is broken.
    """

    def setUp(self):
        """Set up test environment."""
        self.python_exe = sys.executable
        self.workspace_root = Path(__file__).parent.parent.parent
        
    def test_inline_mode_hard_gates(self):
        """Test inline mode meets ALL hard performance gates.
        
        HARD GATES (all must pass):
        1. frontend_used == True
        2. n_frontend_steps == n_scans
        3. n_loop_closures >= 1
        4. rmse.frontend <= rmse.odom
        5. rmse.optimized <= 0.95 * rmse.odom
        """
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
            timeout=60,
            env=env,
        )
        
        # Must complete successfully
        self.assertEqual(result.returncode, 0,
            f"Script failed with exit code {result.returncode}\n"
            f"stderr: {result.stderr}")
        
        # Parse JSON summary
        summary = parse_slam_summary(result.stdout)
        self.assertIsNotNone(summary,
            "CRITICAL: Missing [SLAM_SUMMARY] JSON line in output.\n"
            "The script must print a machine-readable summary.")
        
        # === HARD GATE 1: Frontend was actually used ===
        self.assertIn("frontend_used", summary,
            "HARD GATE 1 FAILED: Missing 'frontend_used' in summary")
        self.assertTrue(summary["frontend_used"],
            "HARD GATE 1 FAILED: frontend_used must be True. "
            "SlamFrontend2D.step() must be executed for every pose.")
        
        # === HARD GATE 2: Frontend processed all scans ===
        self.assertIn("n_scans", summary,
            "HARD GATE 2 FAILED: Missing 'n_scans' in summary")
        self.assertIn("n_frontend_steps", summary,
            "HARD GATE 2 FAILED: Missing 'n_frontend_steps' in summary")
        self.assertEqual(summary["n_frontend_steps"], summary["n_scans"],
            f"HARD GATE 2 FAILED: n_frontend_steps ({summary['n_frontend_steps']}) "
            f"must equal n_scans ({summary['n_scans']}). "
            "Every scan must go through the frontend.")
        
        # === HARD GATE 3: Loop closures detected ===
        self.assertIn("n_loop_closures", summary,
            "HARD GATE 3 FAILED: Missing 'n_loop_closures' in summary")
        self.assertGreaterEqual(summary["n_loop_closures"], 1,
            f"HARD GATE 3 FAILED: n_loop_closures ({summary['n_loop_closures']}) "
            "must be >= 1. The inline scenario must include a loop that can be detected.")
        
        # === HARD GATE 4: Frontend improves or maintains trajectory ===
        rmse = summary.get("rmse", {})
        self.assertIn("odom", rmse,
            "HARD GATE 4 FAILED: Missing 'rmse.odom' in summary")
        self.assertIn("frontend", rmse,
            "HARD GATE 4 FAILED: Missing 'rmse.frontend' in summary")
        self.assertLessEqual(rmse["frontend"], rmse["odom"],
            f"HARD GATE 4 FAILED: Frontend RMSE ({rmse['frontend']:.4f}) "
            f"must be <= Odometry RMSE ({rmse['odom']:.4f}). "
            "If frontend makes things worse, scan-to-map ICP is not working.")
        
        # === HARD GATE 5: Optimization achieves >= 5% improvement ===
        self.assertIn("optimized", rmse,
            "HARD GATE 5 FAILED: Missing 'rmse.optimized' in summary")
        max_allowed_rmse = 0.95 * rmse["odom"]
        improvement_pct = 100 * (1 - rmse["optimized"] / rmse["odom"])
        self.assertLessEqual(rmse["optimized"], max_allowed_rmse,
            f"HARD GATE 5 FAILED: Optimized RMSE ({rmse['optimized']:.4f}) "
            f"must be <= 0.95 * Odometry RMSE ({max_allowed_rmse:.4f}). "
            f"Actual improvement: {improvement_pct:.1f}% (need >= 5%).")
        
        # Print success summary
        print(f"\n=== ALL HARD GATES PASSED ===")
        print(f"  frontend_used: {summary['frontend_used']}")
        print(f"  n_scans: {summary['n_scans']}")
        print(f"  n_frontend_steps: {summary['n_frontend_steps']}")
        print(f"  n_loop_closures: {summary['n_loop_closures']}")
        print(f"  Frontend RMSE: {rmse['frontend']:.4f} (vs Odom: {rmse['odom']:.4f})")
        print(f"  Optimized RMSE: {rmse['optimized']:.4f} ({improvement_pct:.1f}% improvement)")


if __name__ == "__main__":
    unittest.main()
