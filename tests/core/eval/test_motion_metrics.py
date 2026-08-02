"""Tests for the degenerate-baseline metrics.

These exist because of a specific failure: two of Chapter 6's four dead
reckoning methods were pinned at the start point, tracing 0.5 m and 2.3 m
against a 100 m walk, and every metric the example reported called them good.
Final position error did, because the ground truth is a closed loop and an
estimator that never moves is exactly right at loop closure. RMSE did too,
because for a stationary estimate it just measures how far the truth wanders
from the start.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from core.eval import compute_position_rmse, compute_rmse, motion_ratio, path_length


class TestPathLength:
    """Total distance travelled along a path."""

    def test_sums_segment_lengths(self):
        """A 3-4-5 leg plus a unit leg."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]])

        assert path_length(positions) == pytest.approx(6.0)

    def test_closed_loop_is_the_perimeter_not_zero(self):
        """The distinction the whole metric exists for.

        A path returning to its start has zero *displacement*; its length is
        the perimeter. Confusing the two is what let a stationary estimator
        look like a good one.
        """
        square = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
        )

        assert path_length(square) == pytest.approx(40.0)
        np.testing.assert_allclose(square[-1] - square[0], [0.0, 0.0])

    def test_stationary_path_is_zero(self):
        """The degenerate case being detected."""
        assert path_length(np.zeros((50, 2))) == pytest.approx(0.0)

    @pytest.mark.parametrize("positions", [np.zeros((0, 2)), np.zeros((1, 2))])
    def test_too_short_to_have_length(self, positions):
        """Fewer than two samples means no segments, not an error."""
        assert path_length(positions) == 0.0

    def test_works_in_three_dimensions(self):
        """Callers pass (N, 3) as often as (N, 2)."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])

        assert path_length(positions) == pytest.approx(3.0)

    def test_rejects_wrong_rank(self):
        """A 1-D array is a common slicing mistake; fail loudly."""
        with pytest.raises(ValueError):
            path_length(np.array([1.0, 2.0, 3.0]))


class TestMotionRatio:
    """How far an estimator moved, relative to the truth."""

    def test_perfect_tracking_is_one(self):
        truth = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

        assert motion_ratio(truth, truth) == pytest.approx(1.0)

    def test_frozen_estimator_is_near_zero(self):
        """The Chapter 6 signature: a method that never leaves the start.

        Reproduces the real numbers -- 0.5 m traced against a 100 m walk.
        """
        truth = np.array(
            [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0], [0.0, 0.0]]
        )
        frozen = np.array([[0.0, 0.0], [0.3, 0.1], [0.1, 0.0]])

        assert path_length(truth) == pytest.approx(100.0)
        assert motion_ratio(frozen, truth) < 0.05

    def test_drifting_estimator_exceeds_one(self):
        """An unaided integrator accumulates spurious motion."""
        truth = np.array([[0.0, 0.0], [10.0, 0.0]])
        drifting = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 150.0]])

        assert motion_ratio(drifting, truth) > 10.0

    def test_lengths_need_not_match(self):
        """Estimates are often decimated relative to the truth."""
        truth = np.linspace([0.0, 0.0], [10.0, 0.0], 500)
        coarse = np.linspace([0.0, 0.0], [10.0, 0.0], 5)

        assert motion_ratio(coarse, truth) == pytest.approx(1.0)

    def test_stationary_truth_does_not_divide_by_zero(self):
        """Degenerate truth: report inf if the estimate moved, else 0."""
        stationary = np.zeros((10, 2))
        moving = np.linspace([0.0, 0.0], [5.0, 0.0], 10)

        assert motion_ratio(moving, stationary) == float("inf")
        assert motion_ratio(stationary, stationary) == 0.0


class TestComputePositionRmse:
    """RMS of the error magnitude, versus the per-axis RMS trap."""

    def test_norms_before_averaging(self):
        """Magnitudes 5 and 10 give sqrt((25+100)/2)."""
        errors = np.array([[3.0, 4.0], [6.0, 8.0]])

        assert compute_position_rmse(errors) == pytest.approx(np.sqrt(62.5))

    def test_differs_from_compute_rmse_by_exactly_sqrt_two(self):
        """The silent failure this function exists to prevent.

        compute_rmse on (N, 2) vectors averages over 2N components instead of
        N positions. In 2-D that is smaller by exactly sqrt(2) -- a number
        that still looks plausible, just 29% too small, which is how four
        Chapter 8 examples shipped under-reported position RMSE.
        """
        rng = np.random.default_rng(7)
        errors = rng.standard_normal((500, 2))

        ratio = compute_position_rmse(errors) / compute_rmse(errors)
        assert ratio == pytest.approx(np.sqrt(2.0))

    def test_accepts_magnitudes_already_normed(self):
        """Callers that normed first must get the same answer."""
        errors = np.array([[3.0, 4.0], [6.0, 8.0]])
        magnitudes = np.linalg.norm(errors, axis=1)

        assert compute_position_rmse(magnitudes) == pytest.approx(
            compute_position_rmse(errors)
        )

    def test_works_in_three_dimensions(self):
        """Position error is not always planar."""
        errors = np.array([[1.0, 2.0, 2.0]])

        assert compute_position_rmse(errors) == pytest.approx(3.0)

    def test_rejects_wrong_rank(self):
        """A (2, 2, N) slip should fail loudly, not average silently."""
        with pytest.raises(ValueError):
            compute_position_rmse(np.zeros((2, 2, 2)))
