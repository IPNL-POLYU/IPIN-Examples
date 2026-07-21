"""Unit tests for core.eval.plots 3-D frame primitives.

The frame drawing carries a real convention claim -- that a passive rotation
matrix stores the new frame's axes in its *rows* -- so it is tested like any
other equation-bearing code rather than eyeballed in a figure.

Author: Li-Ta Hsu
References: Chapter 2, Section 2.2 (attitude), Eqs. (2.14)-(2.17), (2.21)
"""

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.coords import euler_to_rotation_matrix
from core.eval import plot_frame_3d, save_figure, set_axes_equal_3d


@pytest.fixture
def ax3d():
    """A bare 3-D axes, closed after the test."""
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    yield axes
    plt.close(fig)


def _drawn_axes(ax):
    """Return the three drawn axis vectors as rows of a (3, 3) array."""
    vectors = []
    for line in ax.lines:
        xs, ys, zs = line.get_data_3d()
        vectors.append([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
    return np.asarray(vectors)


class TestPlotFrame3D:
    """Test the coordinate-triad primitive."""

    def test_draws_three_axes(self, ax3d):
        """One line per axis."""
        plot_frame_3d(ax3d)

        assert len(ax3d.lines) == 3

    def test_identity_draws_canonical_axes(self, ax3d):
        """With C = I the triad is the reference frame itself."""
        plot_frame_3d(ax3d, np.eye(3), scale=1.0)

        np.testing.assert_allclose(_drawn_axes(ax3d), np.eye(3), atol=1e-12)

    def test_uses_rows_not_columns_of_passive_matrix(self, ax3d):
        """The convention claim: rows of a passive C are the new axes.

        Drawing columns instead would rotate every frame the wrong way -- the
        same passive/active transpose confusion that Chapter 6's strapdown
        module warns about. For a yaw rotation the two differ in the sign of
        the off-diagonal terms, so this test tells them apart.
        """
        C = euler_to_rotation_matrix(0.0, 0.0, np.deg2rad(40.0))
        plot_frame_3d(ax3d, C, scale=1.0)

        drawn = _drawn_axes(ax3d)
        np.testing.assert_allclose(drawn, C, atol=1e-12)
        assert not np.allclose(drawn, C.T, atol=1e-6)

    def test_rotation_axis_stays_invariant(self, ax3d):
        """Roll turns about Y (Eq. 2.15), so the drawn y axis must not move."""
        C = euler_to_rotation_matrix(np.deg2rad(35.0), 0.0, 0.0)
        plot_frame_3d(ax3d, C, scale=1.0)

        drawn = _drawn_axes(ax3d)
        np.testing.assert_allclose(drawn[1], [0.0, 1.0, 0.0], atol=1e-12)
        # ...while the other two do move.
        assert not np.allclose(drawn[0], [1.0, 0.0, 0.0], atol=1e-3)
        assert not np.allclose(drawn[2], [0.0, 0.0, 1.0], atol=1e-3)

    def test_scale_and_origin_are_applied(self, ax3d):
        """Arrows start at the origin and have the requested length."""
        origin = np.array([1.0, -2.0, 0.5])
        plot_frame_3d(ax3d, np.eye(3), origin=origin, scale=2.0)

        np.testing.assert_allclose(
            _drawn_axes(ax3d), 2.0 * np.eye(3), atol=1e-12
        )
        for line in ax3d.lines:
            xs, ys, zs = line.get_data_3d()
            np.testing.assert_allclose([xs[0], ys[0], zs[0]], origin, atol=1e-12)

    def test_labels_can_be_suppressed(self, ax3d):
        """show_labels=False leaves no text, for a faint reference triad."""
        plot_frame_3d(ax3d, np.eye(3), show_labels=False)
        assert len(ax3d.texts) == 0

        plot_frame_3d(ax3d, np.eye(3), show_labels=True)
        assert len(ax3d.texts) == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"C": np.eye(2)},
            {"C": np.zeros((3, 4))},
            {"origin": np.zeros(2)},
        ],
    )
    def test_invalid_shapes_raise(self, ax3d, kwargs):
        """Wrong shapes fail loudly rather than drawing nonsense."""
        with pytest.raises(ValueError):
            plot_frame_3d(ax3d, **kwargs)


class TestSetAxesEqual3D:
    """Test the equal-aspect helper."""

    def test_limits_are_symmetric_and_equal(self, ax3d):
        """An orthonormal triad must not render sheared."""
        set_axes_equal_3d(ax3d, radius=2.0)

        for get_lim in (ax3d.get_xlim, ax3d.get_ylim, ax3d.get_zlim):
            low, high = get_lim()
            assert low == pytest.approx(-2.0)
            assert high == pytest.approx(2.0)


class TestSaveFigure:
    """Test the shared figure output path."""

    def test_writes_every_requested_format(self, tmp_path):
        """save_figure is the single output path for all chapters."""
        fig = plt.figure()
        try:
            paths = save_figure(fig, tmp_path, "demo", formats=("png", "svg"))
        finally:
            plt.close(fig)

        assert [p.name for p in paths] == ["demo.png", "demo.svg"]
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    def test_creates_missing_output_directory(self, tmp_path):
        """A chapter's figs/ directory need not exist yet."""
        target = tmp_path / "figs" / "nested"
        fig = plt.figure()
        try:
            paths = save_figure(fig, target, "demo", formats=("png",))
        finally:
            plt.close(fig)

        assert paths[0].exists()
