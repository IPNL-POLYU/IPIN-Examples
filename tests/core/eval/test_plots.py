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
from core.eval import (
    plot_error_cdf,
    plot_error_magnitude_time,
    plot_frame_3d,
    plot_trajectory_2d,
    save_animation,
    save_figure,
    set_axes_equal_3d,
)


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

    def test_output_is_byte_reproducible(self, tmp_path):
        """Regenerating an unchanged figure must not produce a diff.

        matplotlib stamps SVG and PDF with the current time and randomises
        internal element ids, which made every regeneration a several-hundred
        line diff per file -- noise that buries real changes in a repository
        where tracked figures are already over half the bytes.
        """
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1, 2], [0, 1, 4])
        try:
            first = save_figure(
                fig, tmp_path / "a", "demo", formats=("svg", "pdf", "png")
            )
            second = save_figure(
                fig, tmp_path / "b", "demo", formats=("svg", "pdf", "png")
            )
        finally:
            plt.close(fig)

        for lhs, rhs in zip(first, second):
            assert lhs.read_bytes() == rhs.read_bytes(), (
                f"{lhs.suffix} output is not reproducible"
            )


class TestPlotTrajectory2D:
    """Test the trajectory primitive's frame flexibility."""

    def test_axis_labels_are_configurable(self):
        """Chapters plotting in ENU need East/North, not X/Y.

        Hardcoded "X (m)"/"Y (m)" was the only thing standing between the
        Chapter 6 comparison example and reusing this function, so it had
        reimplemented the whole plot locally.
        """
        truth = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        est = {"method": truth + 0.1}

        fig = plot_trajectory_2d(
            truth, est, axis_labels=("East [m]", "North [m]")
        )
        try:
            axes = fig.axes[0]
            assert axes.get_xlabel() == "East [m]"
            assert axes.get_ylabel() == "North [m]"
        finally:
            plt.close(fig)

    def test_default_axis_labels_are_frame_agnostic(self):
        """Existing callers keep the previous labels."""
        truth = np.array([[0.0, 0.0], [1.0, 1.0]])

        fig = plot_trajectory_2d(truth, {"m": truth})
        try:
            assert fig.axes[0].get_xlabel() == "X (m)"
        finally:
            plt.close(fig)

    def test_single_panel_by_default(self):
        """zoom_to_truth is opt-in; existing callers keep one pair of axes."""
        truth = np.array([[0.0, 0.0], [1.0, 1.0]])

        fig = plot_trajectory_2d(truth, {"m": truth})
        try:
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_estimates_are_drawn_above_the_truth(self):
        """An estimate that tracks well lies on the truth and must stay visible.

        With the truth on top, the best method was the one you could not see.
        """
        truth = np.array([[0.0, 0.0], [1.0, 1.0]])

        fig = plot_trajectory_2d(truth, {"m": truth})
        try:
            by_label = {line.get_label(): line for line in fig.axes[0].lines}
            assert by_label["m"].get_zorder() > by_label["Ground Truth"].get_zorder()
        finally:
            plt.close(fig)

    def test_zoom_panel_frames_the_truth_not_the_outlier(self):
        """The zoomed axes bound the truth and exclude the diverging estimate.

        This is the wide-dynamic-range case: comparing an unbounded estimator
        against bounded ones, equal axes force one pair of limits to span the
        divergence, and everything near the truth collapses to a blob.
        """
        truth = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
        diverging = np.array([[0.0, 0.0], [500.0, 0.0], [900.0, 0.0]])

        fig = plot_trajectory_2d(truth, {"drift": diverging}, zoom_to_truth=True)
        try:
            assert len(fig.axes) == 2
            zoom_x = fig.axes[1].get_xlim()
            zoom_y = fig.axes[1].get_ylim()
            # Contains the truth ...
            assert zoom_x[0] <= 0.0 and zoom_x[1] >= 10.0
            assert zoom_y[0] <= 0.0 and zoom_y[1] >= 5.0
            # ... but is nowhere near the far end of the drift.
            assert zoom_x[1] < 100.0
            # The full-extent panel keeps the drift, which is usually the point.
            assert fig.axes[0].get_xlim()[1] >= 900.0
        finally:
            plt.close(fig)

    def test_draws_into_a_supplied_axes(self):
        """As one panel of a composite, without creating a second figure."""
        truth = np.array([[0.0, 0.0], [1.0, 1.0]])
        host, (left, right) = plt.subplots(1, 2)

        try:
            before = len(plt.get_fignums())
            fig = plot_trajectory_2d(truth, {"m": truth}, ax=left)

            assert fig is host
            assert len(plt.get_fignums()) == before  # no stray figure
            assert len(left.lines) > 0
            assert len(right.lines) == 0  # drew into the axes it was given
        finally:
            plt.close(host)

    def test_supplied_axes_and_zoom_are_mutually_exclusive(self):
        """Two panels cannot share one supplied axes, so say so.

        Silently honouring either one would betray the other: the caller's
        composite would lose its zoom, or gain a figure it does not contain.
        """
        truth = np.array([[0.0, 0.0], [1.0, 1.0]])
        host, host_ax = plt.subplots()

        try:
            with pytest.raises(ValueError, match="zoom_to_truth"):
                plot_trajectory_2d(
                    truth, {"m": truth}, ax=host_ax, zoom_to_truth=True
                )
        finally:
            plt.close(host)

    def test_zoom_title_weight_is_configurable(self):
        """title_fontweight reaches the figure title in the two-panel layout.

        The panel captions stay plain either way -- they label parts, they do
        not announce a figure.
        """
        truth = np.array([[0.0, 0.0], [10.0, 5.0]])

        fig = plot_trajectory_2d(
            truth, {"m": truth}, zoom_to_truth=True, title_fontweight="normal"
        )
        try:
            assert fig._suptitle.get_fontweight() == "normal"
        finally:
            plt.close(fig)

    def test_zoom_handles_a_stationary_truth(self):
        """A degenerate truth still yields orderable limits, not a zero span."""
        truth = np.zeros((5, 2))

        fig = plot_trajectory_2d(truth, {"drift": np.ones((5, 2))}, zoom_to_truth=True)
        try:
            x_lo, x_hi = fig.axes[1].get_xlim()
            y_lo, y_hi = fig.axes[1].get_ylim()
            assert x_hi > x_lo
            assert y_hi > y_lo
        finally:
            plt.close(fig)


class TestPlotErrorMagnitudeTime:
    """Test the error-magnitude primitive.

    Distinct from plot_position_error_time, which splits the error per axis.
    Every chapter that compares methods wanted the scalar magnitude and so
    wrote it by hand -- 43 sites computed norm(est - truth, axis=1).
    """

    def test_accepts_vector_and_scalar_errors(self):
        """(N, 2) vectors are normed; (N,) magnitudes are taken as given."""
        vector = np.array([[3.0, 4.0], [6.0, 8.0]])

        fig = plot_error_magnitude_time({"vec": vector, "scalar": np.array([5.0, 10.0])})
        try:
            lines = fig.axes[0].lines
            assert len(lines) == 2
            # norm([3,4]) = 5, norm([6,8]) = 10 -- both series coincide.
            np.testing.assert_allclose(lines[0].get_ydata(), [5.0, 10.0])
            np.testing.assert_allclose(lines[1].get_ydata(), [5.0, 10.0])
        finally:
            plt.close(fig)

    def test_uses_supplied_time_vector(self):
        """A non-uniform or offset time base must be honoured, not re-derived."""
        t = np.array([10.0, 12.5, 20.0])

        fig = plot_error_magnitude_time({"m": np.array([1.0, 2.0, 3.0])}, t=t)
        try:
            np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), t)
        finally:
            plt.close(fig)

    def test_log_scale_clamps_the_axis_to_max_decades(self):
        """A run starting at ground truth must not open the axis to 1e-5.

        The hand-rolled versions all did: error starts near zero, matplotlib
        fits it, and every curve is squeezed into the top third of the figure.
        """
        errors = {"m": np.array([1e-9, 1.0, 500.0])}

        fig = plot_error_magnitude_time(errors, log_scale=True, max_decades=4.0)
        try:
            bottom, top = fig.axes[0].get_ylim()
            assert bottom == pytest.approx(500.0 / 1e4)
            assert top == pytest.approx(1000.0)
            # ...and the near-zero sample is excluded rather than driving the axis.
            assert bottom > 1e-9
        finally:
            plt.close(fig)

    def test_linear_scale_leaves_the_axis_alone(self):
        """The clamp is a log-axis concern only."""
        fig = plot_error_magnitude_time({"m": np.array([0.0, 1.0])}, log_scale=False)
        try:
            assert fig.axes[0].get_yscale() == "linear"
        finally:
            plt.close(fig)


class TestCompositePanelSupport:
    """Test that the primitives can draw into an existing axes.

    This is what unlocked reuse at all: nearly every figure in the book is a
    multi-panel composite, so a function that insists on making its own figure
    can only serve a standalone plot. Every chapter but one had therefore
    reimplemented these plots locally.
    """

    TRUTH = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    ERRORS = {"m": np.array([0.5, 1.0, 0.25])}

    def _calls(self):
        return [
            (plot_trajectory_2d, (self.TRUTH, {"est": self.TRUTH + 0.1}), {}),
            (plot_error_magnitude_time, (self.ERRORS,), {}),
            (plot_error_cdf, (self.ERRORS,), {}),
        ]

    def test_draws_into_the_supplied_axes(self):
        """Content lands on the given panel, and no extra figure is made."""
        for fn, args, kwargs in self._calls():
            fig, axes = plt.subplots(1, 2)
            try:
                before = plt.get_fignums()
                returned = fn(*args, ax=axes[1], **kwargs)

                assert plt.get_fignums() == before, f"{fn.__name__} made a figure"
                assert returned is fig, f"{fn.__name__} returned the wrong figure"
                assert len(axes[1].lines) > 0, f"{fn.__name__} drew nothing"
                assert len(axes[0].lines) == 0, f"{fn.__name__} drew on the wrong axes"
            finally:
                plt.close(fig)

    def test_creates_its_own_figure_when_ax_is_omitted(self):
        """Backward compatible: existing standalone callers are unaffected."""
        for fn, args, kwargs in self._calls():
            fig = fn(*args, **kwargs)
            try:
                assert isinstance(fig, plt.Figure)
                assert len(fig.axes) == 1
            finally:
                plt.close(fig)

    def test_title_weight_is_selectable(self):
        """A shared panel must not look louder than its unbold siblings."""
        for fn, args, kwargs in self._calls():
            fig, ax = plt.subplots()
            try:
                fn(*args, ax=ax, title_fontweight="normal", **kwargs)
                assert ax.title.get_fontweight() == "normal"
            finally:
                plt.close(fig)

            fig, ax = plt.subplots()
            try:
                fn(*args, ax=ax, **kwargs)
                assert ax.title.get_fontweight() == "bold"
            finally:
                plt.close(fig)


class TestSaveAnimation:
    """Test the shared GIF output path."""

    def test_writes_a_gif_with_the_requested_frames(self, tmp_path):
        """Frames render and the file lands in the given directory."""
        from PIL import Image

        fig = plt.figure()
        axes = fig.add_subplot(111)

        def update(frame):
            axes.clear()
            axes.plot([0, 1], [0, frame])
            return axes

        try:
            path = save_animation(fig, update, 4, tmp_path, "anim", fps=4)
        finally:
            plt.close(fig)

        assert path.exists() and path.suffix == ".gif"
        with Image.open(path) as image:
            assert image.n_frames == 4
