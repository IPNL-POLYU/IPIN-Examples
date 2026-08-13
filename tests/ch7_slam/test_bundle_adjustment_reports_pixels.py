"""Bundle adjustment reported a percentage of a weighted sum of squares.

`example_bundle_adjustment` printed:

    Initial reprojection error: 4940699.453155
    Final reprojection error: 52.481083
    Error reduction: 100.00%

Two problems. The reduction is 99.998938%, printed as "100.00%" by a `.2f`
format -- a partial result shown as a total one. And neither of the numbers it
is computed from is a reprojection error: `graph.compute_error()` returns the
Eq. (3.38) cost, sum_i r_i^T Lambda_i r_i, which sums over all 46 observations
and scales each by 1/sigma^2 = 4. It grows with the number of observations and
with the confidence in them, so it is not an accuracy and cannot be compared
against anything a reader knows. 4.9e6 says only that some landmark was badly
initialised.

In pixels the same solve reads 163.9 -> 0.53 px RMS against a 0.5 px
measurement noise floor, which is the statement worth making: the optimiser did
not merely improve, it reached the best any estimator could.

This is the same defect as the ICP residual in `core/slam/scan_matching.py`
(see tests/core/slam/test_icp_residual_units.py) -- a sum of squares standing
in for a distance -- and it is worth recognising on sight. The tell is a number
with no unit in its name that is being compared against a threshold or turned
into a percentage.

Author: Li-Ta Hsu
References: Chapter 7, Section 7.4 (bundle adjustment), Eq. (3.38)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch7_slam.example_bundle_adjustment import (
    PIXEL_NOISE_STD,
    reprojection_residuals_px,
    rms,
)
from core.estimators.factor_graph import Factor, FactorGraph


def _graph_with_known_residuals(reprojection, prior):
    """A graph whose factors return fixed residuals, so the maths is checkable.

    `reprojection_residuals_px` only reads `variable_ids` and `residual_func`,
    so synthetic factors exercise its contract exactly.
    """
    graph = FactorGraph()
    graph.add_variable(0, np.zeros(2))
    for r in reprojection:
        graph.add_factor(Factor(
            variable_ids=[0],
            residual_func=lambda _x, r=np.asarray(r, dtype=float): r,
            jacobian_func=lambda _x: [np.zeros((2, 2))],
            information=np.eye(2) / (PIXEL_NOISE_STD ** 2),
        ))
    # The gauge prior is appended last and its residual is in metres.
    graph.add_factor(Factor(
        variable_ids=[0],
        residual_func=lambda _x, r=np.asarray(prior, dtype=float): r,
        jacobian_func=lambda _x: [np.zeros((2, 2))],
        information=np.eye(2),
    ))
    return graph


class TestResidualsAreReportedInPixels(unittest.TestCase):

    def test_returns_one_magnitude_per_observation(self):
        graph = _graph_with_known_residuals([[3.0, 4.0], [0.0, 1.0]], [0.0, 0.0])

        residuals = reprojection_residuals_px(graph, 2)

        np.testing.assert_allclose(residuals, [5.0, 1.0])

    def test_the_gauge_prior_is_excluded(self):
        """Its residual is in metres; averaging it in would be a unit error.

        The prior is deliberately given a huge residual here, so forgetting to
        exclude it cannot pass by coincidence.
        """
        graph = _graph_with_known_residuals([[3.0, 4.0]], [900.0, 0.0])

        residuals = reprojection_residuals_px(graph, 1)

        self.assertEqual(len(residuals), 1)
        self.assertAlmostEqual(float(residuals[0]), 5.0)

    def test_the_cost_is_not_the_pixel_error(self):
        """The conflation this file exists to prevent.

        Same graph, two quantities: the cost weights by 1/sigma^2 and sums,
        the RMS does neither. They are not interchangeable and the gap grows
        with the observation count.
        """
        graph = _graph_with_known_residuals([[3.0, 4.0]] * 8, [0.0, 0.0])

        cost = graph.compute_error()
        pixels = rms(reprojection_residuals_px(graph, 8))

        self.assertAlmostEqual(pixels, 5.0)
        self.assertAlmostEqual(cost, 8 * 25.0 / PIXEL_NOISE_STD ** 2)
        self.assertGreater(cost / pixels, 100.0)

    def test_the_cost_grows_with_observation_count_and_the_rms_does_not(self):
        """Why a percentage of the cost is not a statement about accuracy."""
        few = _graph_with_known_residuals([[3.0, 4.0]] * 4, [0.0, 0.0])
        many = _graph_with_known_residuals([[3.0, 4.0]] * 16, [0.0, 0.0])

        self.assertAlmostEqual(many.compute_error() / few.compute_error(), 4.0)
        self.assertAlmostEqual(
            rms(reprojection_residuals_px(many, 16)),
            rms(reprojection_residuals_px(few, 4)),
        )


class TestTheReportedReductionIsNotOneHundredPercent(unittest.TestCase):
    """The rounding that turned 99.998938% into a claim of totality."""

    def test_two_decimals_hide_the_remaining_error(self):
        initial, final = 4940699.453155, 52.481083
        reduction = (1 - final / initial) * 100

        self.assertEqual(f"{reduction:.2f}", "100.00")
        self.assertLess(reduction, 100.0)

    def test_the_ratio_says_it_without_rounding_to_a_falsehood(self):
        """What the example prints instead: a factor, plus enough decimals."""
        initial, final = 4940699.453155, 52.481083

        self.assertEqual(f"{initial / final:,.0f}x", "94,142x")
        self.assertEqual(f"{(1 - final / initial) * 100:.4f}", "99.9989")


if __name__ == "__main__":
    unittest.main()
