"""Smoke tests for the Chapter 2 attitude visualization example.

Figure code rots quietly: nothing fails until someone opens the PNG. These
tests run the example headless and assert that every advertised figure is
actually written, in every format.

Author: Li-Ta Hsu
References: Chapter 2, Sections 2.1-2.2
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FIGURES = (
    "ch2_euler_convention",
    "ch2_passive_vs_active",
    "ch2_gimbal_lock",
    "ch2_frame_chain",
)


class TestAttitudeVisualizationExample(unittest.TestCase):
    """Run the example end to end and check its figure output."""

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "PYTHONPATH": str(WORKSPACE_ROOT)})

        cls.result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ch2_coords.example_attitude_visualization",
                "--no-show",
            ],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        cls.figs_dir = WORKSPACE_ROOT / "ch2_coords" / "figs"

    def test_example_runs_successfully(self):
        """Exit code 0 with no traceback."""
        self.assertEqual(
            self.result.returncode,
            0,
            f"Example failed with stderr:\n{self.result.stderr}",
        )

    def test_reports_the_book_convention(self):
        """The output states the non-standard axis assignment."""
        self.assertIn("roll about Y", self.result.stdout)
        self.assertIn("pitch about X", self.result.stdout)

    def test_every_figure_is_written_in_every_format(self):
        """Chapter 2 previously produced no figures at all."""
        for name in EXPECTED_FIGURES:
            for suffix in ("png", "svg", "pdf"):
                path = self.figs_dir / f"{name}.{suffix}"
                with self.subTest(figure=path.name):
                    self.assertTrue(path.exists(), f"missing {path}")
                    self.assertGreater(
                        path.stat().st_size, 0, f"empty {path}"
                    )


if __name__ == "__main__":
    unittest.main()
