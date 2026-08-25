"""Tests for the Chapter 7 SLAM front-end demo figure.

The committed figure existed for one reason: to show odometry drifting away
from the truth while scan-to-map alignment holds onto it. It did not. The walk
runs 4.5 m along x and about 0.13 m across it, ``plot_trajectory_2d`` applied
equal axes, and matplotlib stretched the y range to roughly [-2, 2] to match --
so all three tracks landed inside a 15-pixel band on a 500-pixel panel, 97% of
which was white. The front-end track sat within about 4 px of the truth, near
enough its own stroke width to be unreadable. No test caught it, because every
property anyone had thought to assert (the figure exists, the RMSE improves)
was still true.

So these tests are written in *display* coordinates: not "is the drift in the
data" but "can a reader tell the lines apart on the page".

Author: Li-Ta Hsu
References: Chapter 7, Section 7.4 (SLAM front end)
"""

import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch7_slam.example_slam_frontend import (
    DEFAULT_SEED,
    FIGURE_NAME,
    build_figure,
    run_frontend_demo,
)
from core.eval import plot_trajectory_2d
from core.eval.plots import UNEQUAL_AXES_NOTE

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _vertical_pixel_gap(ax, first_xy, second_xy):
    """Largest on-page vertical separation between two tracks, in pixels."""
    ax.figure.canvas.draw()
    first = ax.transData.transform(np.asarray(first_xy))
    second = ax.transData.transform(np.asarray(second_xy))
    return float(np.max(np.abs(first[:, 1] - second[:, 1])))


def _height_fraction(ax, *tracks):
    """Share of the panel's height that the drawn series actually occupy."""
    ax.figure.canvas.draw()
    ys = ax.transData.transform(np.vstack(tracks))[:, 1]
    return float(np.ptp(ys) / ax.get_window_extent().height)


class TestTrajectoryPanelIsReadable(unittest.TestCase):
    """The left panel must resolve the drift it exists to show."""

    @classmethod
    def setUpClass(cls):
        cls.demo = run_frontend_demo()

    def setUp(self):
        self.fig = build_figure(self.demo)
        self.addCleanup(plt.close, self.fig)
        self.panel = self.fig.axes[0]

    def _tracks(self):
        """Every series drawn on the trajectory panel."""
        return (
            self.demo["true_xy"],
            self.demo["odom_xy"],
            self.demo["frontend_xy"],
        )

    def test_the_walk_really_is_near_1d(self):
        """Guards the premise: if the demo path gains width, revisit the fix.

        A path with genuine y extent should go back to equal axes, which are
        the honest default for a trajectory.
        """
        truth = self.demo["true_xy"]
        x_span = np.ptp(truth[:, 0])
        y_span = max(np.ptp(truth[:, 1]), np.ptp(self.demo["odom_xy"][:, 1]))

        self.assertGreater(x_span, 4.0)
        self.assertGreater(x_span / y_span, 10.0)

    def test_odometry_drift_is_resolvable_on_the_page(self):
        """The headline claim, measured in pixels rather than metres.

        The two estimates differ by under 0.1 m. Whether that is the point of
        the figure or invisible noise is entirely a question of the panel's
        limits and aspect, which is why this goes through transData.
        """
        gap = _vertical_pixel_gap(
            self.panel, self.demo["odom_xy"], self.demo["frontend_xy"]
        )

        self.assertGreater(gap, 100.0, f"odometry and front-end are {gap:.1f} px apart")

    def test_the_frontend_track_clears_the_ground_truth(self):
        """The harder half: the *good* estimate has structure worth seeing.

        Scan-to-map holds to within about 0.02 m, which under equal axes put
        the red track roughly 4 px from the black one -- close to its own
        stroke width, so its shape was unreadable even where the odometry
        drift was faintly visible.
        """
        gap = _vertical_pixel_gap(
            self.panel, self.demo["frontend_xy"], self.demo["true_xy"]
        )

        self.assertGreater(gap, 30.0, f"front-end sits {gap:.1f} px from truth")

    def test_the_panel_is_not_mostly_white(self):
        """The y limits frame the data, not an aspect ratio.

        The old panel opened y to about [-2, 2] for 0.13 m of motion, so all
        three tracks shared a band under 3% of the panel's height.
        """
        share = _height_fraction(self.panel, *self._tracks())

        self.assertGreater(share, 0.5, f"data occupies only {share:.1%} of the panel")

    def test_equal_axes_would_still_break_this_panel(self):
        """Pins the defect itself, on the demo's own data.

        Without this, the tests above could start passing for some unrelated
        reason and nobody would notice that the flag had stopped mattering.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        try:
            plot_trajectory_2d(
                self.demo["true_xy"],
                {
                    "odom": self.demo["odom_xy"],
                    "frontend": self.demo["frontend_xy"],
                },
                ax=axes[0],
            )
            share = _height_fraction(axes[0], *self._tracks())
            frontend_gap = _vertical_pixel_gap(
                axes[0], self.demo["frontend_xy"], self.demo["true_xy"]
            )
        finally:
            plt.close(fig)

        self.assertLess(share, 0.1, f"expected a hairline, got {share:.1%}")
        self.assertLess(
            frontend_gap, 8.0, f"front-end already legible at {frontend_gap:.1f} px"
        )

    def test_panel_discloses_that_it_is_not_to_scale(self):
        """Unequal axes draw a shape the robot never traced; say so."""
        self.assertIn(UNEQUAL_AXES_NOTE, [t.get_text() for t in self.panel.texts])

    def test_error_panel_is_indexed_by_step_not_seconds(self):
        """The shared primitive labels time in seconds; this series is poses."""
        self.assertEqual(self.fig.axes[1].get_xlabel(), "Step Index")


class TestFrontendDemoClaims(unittest.TestCase):
    """The facts the figure asserts, independent of how it is drawn."""

    @classmethod
    def setUpClass(cls):
        cls.demo = run_frontend_demo()

    def test_scan_to_map_beats_raw_odometry(self):
        """The caption's claim: correction is worth roughly an order of magnitude."""
        odom_rmse = np.sqrt(np.mean(self.demo["odom_errors"] ** 2))
        frontend_rmse = np.sqrt(np.mean(self.demo["frontend_errors"] ** 2))

        self.assertLess(frontend_rmse, odom_rmse / 5.0)

    def test_odometry_actually_drifts(self):
        """Otherwise the left panel has nothing to resolve and passes vacuously."""
        drift = np.linalg.norm(self.demo["odom_xy"][-1] - self.demo["true_xy"][-1])

        self.assertGreater(drift, 0.05)

    def test_the_run_is_seed_controlled(self):
        """The figure is committed, so it has to regenerate byte for byte.

        save_figure already writes reproducible output; that is worthless if
        the simulation feeding it draws from the unseeded global RNG.
        """
        first = run_frontend_demo(seed=DEFAULT_SEED)
        second = run_frontend_demo(seed=DEFAULT_SEED)

        for key in ("true_xy", "odom_xy", "frontend_xy"):
            with self.subTest(series=key):
                np.testing.assert_array_equal(first[key], second[key])

    def test_the_seed_is_wired_through(self):
        """A hardcoded draw would satisfy the test above while ignoring its argument."""
        other = run_frontend_demo(seed=DEFAULT_SEED + 1)

        self.assertFalse(np.array_equal(other["odom_xy"], self.demo["odom_xy"]))


class TestFigureOutputPath(unittest.TestCase):
    """The repo rule: chapter figures go to chX_*/figs via core.eval.save_figure."""

    def test_committed_figure_exists_in_every_format(self):
        """png alone is not enough -- the book sets from svg/pdf."""
        figs_dir = WORKSPACE_ROOT / "ch7_slam" / "figs"

        for suffix in ("png", "svg", "pdf"):
            path = figs_dir / f"{FIGURE_NAME}.{suffix}"
            with self.subTest(figure=path.name):
                self.assertTrue(path.exists(), f"missing {path}")
                self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_writing_the_figure_twice_is_byte_identical(self):
        """A committed figure diff must mean the picture changed.

        save_figure fixes the svg hash salt and drops creation timestamps;
        going through plt.savefig instead -- which is what this example used
        to do -- gives a fresh several-hundred-line diff on every run.
        """
        import tempfile

        from core.eval import save_figure

        demo = run_frontend_demo()
        fig = build_figure(demo)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                first = save_figure(fig, Path(tmp) / "a", FIGURE_NAME)
                second = save_figure(fig, Path(tmp) / "b", FIGURE_NAME)

                for lhs, rhs in zip(first, second, strict=True):
                    with self.subTest(fmt=lhs.suffix):
                        self.assertEqual(lhs.read_bytes(), rhs.read_bytes())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
