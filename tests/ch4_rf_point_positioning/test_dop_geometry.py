"""Tests for the Chapter 4 dilution-of-precision example.

The example's whole claim is that DOP predicts real position error, so that is
what gets tested against the Monte-Carlo simulation the example itself runs --
not just that a figure appears.

Author: Li-Ta Hsu
References: Chapter 4, Section 4.5, Eqs. (4.30)-(4.33)
"""

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np

from ch4_rf_point_positioning.example_dop_geometry import (
    ANCHORS,
    RANGE_STD,
    _position_covariance,
    animate_dop,
    compute_dop_field,
    gdop_at,
    plot_dop_summary,
    run_walk,
)
from core.eval import save_animation


class TestDopField(unittest.TestCase):
    """Basic properties of the GDOP field."""

    def test_dop_is_low_near_the_cluster_high_far_away(self):
        """The field must actually degrade with distance, or there is no story."""
        centre = ANCHORS.mean(axis=0)
        near = gdop_at(centre + np.array([20.0, 20.0]))
        far = gdop_at(centre + np.array([180.0, 180.0]))

        self.assertLess(near, 4.0)
        self.assertGreater(far, 10.0)
        self.assertGreater(far, 3.0 * near)

    def test_field_shape_matches_resolution(self):
        field, xs, ys = compute_dop_field()
        self.assertEqual(field.shape, (len(ys), len(xs)))
        self.assertTrue(np.all(field >= 1.0 - 1e-9))


class TestDopPredictsError(unittest.TestCase):
    """The load-bearing claim: GDOP x range_std tracks the actual RMS error."""

    @classmethod
    def setUpClass(cls):
        cls.walk = run_walk()

    def test_gdop_grows_monotonically_along_the_walk(self):
        """The receiver walks steadily into worse geometry."""
        gdop = self.walk["gdop"]
        # Allow tiny non-monotone wobble from the discrete grid; the trend and
        # the endpoints are what matter.
        self.assertGreater(gdop[-1], 5.0 * gdop[0])
        self.assertGreater(np.corrcoef(np.arange(len(gdop)), gdop)[0, 1], 0.98)

    def test_prediction_matches_monte_carlo(self):
        """GDOP x range_std agrees with the TOA solver's RMS to a few percent."""
        predicted = self.walk["predicted"]
        measured = self.walk["mc_rms"]

        relative = np.abs(predicted - measured) / predicted
        self.assertLess(relative.mean(), 0.10)
        self.assertLess(relative.max(), 0.20)

    def test_predicted_equals_gdop_times_range_std(self):
        """Guards the definition itself."""
        np.testing.assert_allclose(
            self.walk["predicted"], self.walk["gdop"] * RANGE_STD
        )

    def test_error_ellipse_is_tangential_to_the_cluster_line(self):
        """The error is worst across the line of sight, not along it.

        Each range constrains the radial direction (distance to an anchor).
        Clustered anchors all lie in nearly the same direction, so they pin the
        distance to the cluster well but give almost no bearing information --
        the error ellipse is elongated *perpendicular* to the line to the
        cluster, and its long axis is highly dominant.
        """
        cluster = ANCHORS.mean(axis=0)
        far_position = self.walk["walk"][-1]
        cov = _position_covariance(far_position)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        long_axis = eigenvectors[:, -1]
        to_cluster = cluster - far_position
        to_cluster /= np.linalg.norm(to_cluster)

        # Long axis perpendicular to the line of sight: |cos angle| near 0.
        alignment = abs(float(np.dot(long_axis, to_cluster)))
        self.assertLess(alignment, 0.1)
        # And genuinely a cigar, not a near-circle.
        self.assertGreater(eigenvalues[-1] / eigenvalues[0], 20.0)

    def test_reproducible(self):
        first = run_walk()
        second = run_walk()
        np.testing.assert_allclose(first["mc_rms"], second["mc_rms"])


class TestDopFigures(unittest.TestCase):
    """Rendering checks, including the single-colorbar regression."""

    @classmethod
    def setUpClass(cls):
        cls.walk = run_walk()

    def test_summary_renders(self):
        fig = plot_dop_summary(self.walk)
        try:
            self.assertGreaterEqual(len(fig.axes), 2)
        finally:
            plt.close(fig)

    def test_animation_has_a_single_colorbar(self):
        """A per-frame colorbar would stack one per step; exactly one allowed."""
        fig, update, n_frames = animate_dop(self.walk)
        try:
            for frame in range(min(4, n_frames)):
                update(frame)
            self.assertEqual(len(fig.axes), 3)  # 2 panels + 1 colorbar
        finally:
            plt.close(fig)

    def test_animation_stays_small(self):
        fig, update, n_frames = animate_dop(self.walk)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = save_animation(fig, update, n_frames, tmp, "dop", fps=4)
                size_mb = path.stat().st_size / (1024 * 1024)
        finally:
            plt.close(fig)

        self.assertLess(size_mb, 1.5, f"GIF grew to {size_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()
