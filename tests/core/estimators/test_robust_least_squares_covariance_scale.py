"""
Regression tests for robust_least_squares's covariance scale.

core.estimators.least_squares.robust_least_squares used to return
P = (A'WA)^-1 built from the raw IRLS weights -- dimensionless and capped at
1 -- straight through weighted_least_squares(). That formula is only a
covariance when W is supplied in absolute units (w_i = 1/sigma_i^2), so P's
magnitude depended on which loss function (l2/huber/cauchy/gm) was chosen,
even on clean data with no outliers at all: measured before the fix, huber,
cauchy and gm reported an uncertainty roughly 2-2.5x the true a-priori value
that l2 (correctly) landed near.

The fix scales (A'WA)^-1 a posteriori from the final IRLS weights and
residuals, the same way linear_least_squares() scales (A'A)^-1 -- see that
function's docstring Note for the measured residual bias this does not
remove.

Book Reference: Chapter 3, Section 3.1.1 (Table 3.1, robust estimators)
"""

import unittest

import numpy as np

from core.estimators.least_squares import linear_least_squares, robust_least_squares


def _clean_dataset(n_points=300, sigma=0.5, seed=1):
    """Outlier-free linear data: A = [1, x], b = A @ true_params + noise."""
    x = np.linspace(0, 10, n_points)
    A = np.column_stack([np.ones(n_points), x])
    true_params = np.array([2.0, 1.5])
    rng = np.random.default_rng(seed)
    b = A @ true_params + rng.normal(0, sigma, n_points)
    ata_inv = np.linalg.inv(A.T @ A)
    true_sqrt_p00 = sigma * np.sqrt(ata_inv[0, 0])
    return A, b, true_sqrt_p00


def _outlier_dataset(
    n_points=60, sigma=0.5, seed=1, outlier_index=10, outlier_size=15.0
):
    """Same recipe as _clean_dataset, plus one severe outlier."""
    x = np.linspace(0, 10, n_points)
    A = np.column_stack([np.ones(n_points), x])
    true_params = np.array([2.0, 1.5])
    rng = np.random.default_rng(seed)
    b = A @ true_params + rng.normal(0, sigma, n_points)
    b_outlier = b.copy()
    b_outlier[outlier_index] += outlier_size
    return A, b_outlier, outlier_index


class TestRobustCovarianceAgreesAcrossMethods(unittest.TestCase):
    """On clean (outlier-free) data, P's scale should not depend on `method`."""

    def test_all_methods_land_near_the_true_a_priori_uncertainty(self):
        """sqrt(P[0, 0]) for l2/huber/cauchy/gm should be the same order of
        magnitude as sigma * sqrt((A'A)^-1)[0, 0], not off by 2-2.5x depending
        on which robust loss was picked.

        [0.6, 1.3] is not the book's "roughly 15%" -- it is the measured
        result. This session's measurement (N=300, seed=1): l2=0.924,
        huber=0.852, cauchy=0.805, gm=0.720 (ratio to true). huber and l2 are
        within ~15%; cauchy and gm are not, and a Monte Carlo check (N=60,
        300 replicates, see least_squares.py's robust_least_squares docstring)
        shows that is a real bias from continuous downweighting of clean
        Gaussian residuals, not sampling noise. [0.6, 1.3] covers the
        measured spread with margin while still failing hard against the
        pre-fix code, which landed huber/cauchy/gm at 2.0-2.5x here.
        """
        A, b, true_sqrt_p00 = _clean_dataset()

        for method in ["l2", "huber", "cauchy", "gm"]:
            _, P, _ = robust_least_squares(A, b, method=method)
            ratio = np.sqrt(P[0, 0]) / true_sqrt_p00
            self.assertTrue(
                0.6 <= ratio <= 1.3,
                f"{method}: sqrt(P00)/true = {ratio:.3f}, expected roughly in [0.6, 1.3]",
            )

    def test_methods_agree_with_each_other_not_just_with_the_true_value(self):
        """The spread across methods (max/min of sqrt(P00)) should be modest.

        Measured this session (N=300, seed=1): 1.28x after the fix against
        2.67x before it. The fix does not reach perfect parity -- cauchy and
        gm carry a real residual bias, see the Note in robust_least_squares's
        docstring -- but it roughly halves the spread, which is the bar this
        test pins.
        """
        A, b, _ = _clean_dataset()

        sqrt_p00 = [
            np.sqrt(robust_least_squares(A, b, method=method)[1][0, 0])
            for method in ["l2", "huber", "cauchy", "gm"]
        ]

        spread = max(sqrt_p00) / min(sqrt_p00)
        self.assertLess(
            spread, 1.8, f"methods disagree by {spread:.2f}x, expected < 1.8x"
        )


class TestRobustCovarianceSmallerThanL2WithOutlier(unittest.TestCase):
    """With a genuine outlier present, robust P should not inherit L2's
    outlier-inflated sigma2_hat."""

    def test_robust_p_smaller_than_l2_p_with_outlier(self):
        """L2's sigma2_hat is inflated by the outlier's huge residual; robust
        methods downweight it and should report smaller uncertainty.

        Note: this specific comparison holds both before and after the fix
        (the raw, unscaled (A'WA)^-1 is already bounded while L2's blows up
        with the outlier), so it is a real property worth pinning but is not
        by itself a regression guard for the covariance-scale bug -- see
        test_robust_p_lands_near_l2_fit_on_the_outlier_removed_data below for
        the assertion that actually distinguishes the two.
        """
        A, b_outlier, _ = _outlier_dataset()
        _, P_l2 = linear_least_squares(A, b_outlier)

        for method in ["huber", "cauchy", "gm"]:
            _, P_robust, _ = robust_least_squares(A, b_outlier, method=method)
            self.assertLess(
                P_robust[0, 0],
                P_l2[0, 0],
                f"{method}: robust P00={P_robust[0, 0]:.4f} should be < L2's {P_l2[0, 0]:.4f}",
            )

    def test_robust_p_lands_near_l2_fit_on_the_outlier_removed_data(self):
        """A sharper check: the fixed, scaled P should land within the same
        order of magnitude as an L2 fit on the data with the outlier simply
        removed -- the outcome "the robust fit shouldn't see the outlier"
        actually implies for the reported uncertainty.

        Measured this session: the pre-fix raw (A'WA)^-1 does not have this
        property at all (5.3-7.5x the outlier-removed L2 variance); the fix
        lands at 0.5-1.7x. [0.3, 3.0] comfortably covers the fixed code and
        excludes the pre-fix one.
        """
        A, b_outlier, outlier_index = _outlier_dataset()
        n_points = A.shape[0]
        mask = np.ones(n_points, dtype=bool)
        mask[outlier_index] = False
        _, P_clean = linear_least_squares(A[mask], b_outlier[mask])

        for method in ["huber", "cauchy", "gm"]:
            _, P_robust, _ = robust_least_squares(A, b_outlier, method=method)
            ratio = P_robust[0, 0] / P_clean[0, 0]
            self.assertTrue(
                0.3 <= ratio <= 3.0,
                f"{method}: P00/clean-L2-P00 = {ratio:.3f}, expected roughly in [0.3, 3.0]",
            )


if __name__ == "__main__":
    unittest.main()
