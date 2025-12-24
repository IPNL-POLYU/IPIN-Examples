"""
Unit tests for least squares estimation algorithms.

Tests cover:
    - Linear least squares (LS)
    - Weighted least squares (WLS) with diagonal/sigma inputs
    - Iterative least squares (Gauss-Newton)
    - Robust least squares (IRLS with L2, Huber, Cauchy, G-M, Tukey)

Book Reference: Chapter 3, Section 3.1 (Table 3.1 for robust estimators)
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from core.estimators.least_squares import (
    iterative_least_squares,
    linear_least_squares,
    robust_least_squares,
    weighted_least_squares,
)


class TestLinearLeastSquares(unittest.TestCase):
    """Test cases for standard linear least squares."""

    def test_exact_fit(self):
        """Test LS with exact data (no noise)."""
        # y = 2x + 1
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([3, 5, 7])  # y values

        x_hat, P = linear_least_squares(A, b)

        # Should recover exact parameters [intercept, slope] = [1, 2]
        expected = np.array([1.0, 2.0])
        assert_allclose(x_hat, expected, atol=1e-10)

        # Covariance should be finite
        self.assertIsNotNone(P)
        self.assertEqual(P.shape, (2, 2))

    def test_overdetermined_system(self):
        """Test LS with more equations than unknowns."""
        # 5 measurements, 2 unknowns
        A = np.random.randn(5, 2)
        x_true = np.array([1.0, -0.5])
        b = A @ x_true + 0.01 * np.random.randn(5)

        x_hat, P = linear_least_squares(A, b)

        # Should be close to true value
        assert_allclose(x_hat, x_true, atol=0.1)

        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_identity_matrix(self):
        """Test LS with identity design matrix."""
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])

        x_hat, P = linear_least_squares(A, b)

        # Should return b exactly
        assert_allclose(x_hat, b, atol=1e-10)

    def test_rank_deficient_raises_error(self):
        """Test that rank-deficient A raises ValueError."""
        # Rank-deficient matrix (columns are linearly dependent)
        A = np.array([[1, 2], [2, 4], [3, 6]])
        b = np.array([1, 2, 3])

        with self.assertRaises(ValueError) as context:
            linear_least_squares(A, b)

        self.assertIn("rank deficient", str(context.exception).lower())

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched dimensions raise ValueError."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2, 3])  # Wrong size

        with self.assertRaises(ValueError):
            linear_least_squares(A, b)

    def test_underdetermined_raises_error(self):
        """Test that underdetermined system (m < n) raises ValueError."""
        A = np.array([[1, 2, 3]])  # 1 row, 3 columns
        b = np.array([1])

        with self.assertRaises(ValueError) as context:
            linear_least_squares(A, b)

        self.assertIn("underdetermined", str(context.exception).lower())


class TestWeightedLeastSquares(unittest.TestCase):
    """Test cases for weighted least squares."""

    def test_equal_weights(self):
        """Test WLS with equal weights matches LS."""
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        b = np.array([3, 5, 7, 9])
        W = np.eye(4)  # Equal weights

        x_wls, _ = weighted_least_squares(A, b, W)
        x_ls, _ = linear_least_squares(A, b)

        assert_allclose(x_wls, x_ls, atol=1e-10)

    def test_high_weight_on_accurate_measurement(self):
        """Test that high weight emphasizes accurate measurements."""
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([1.0, 2.0, 3.5])  # Last measurement is noisy

        # High weight on first two, low on third
        W = np.diag([10.0, 10.0, 0.1])

        x_hat, P = weighted_least_squares(A, b, W)

        # Should be close to [1, 2] and far from [1.75, 1.75]
        self.assertLess(abs(x_hat[0] - 1.0), 0.2)
        self.assertLess(abs(x_hat[1] - 2.0), 0.2)

    def test_weight_matrix_symmetry(self):
        """Test that asymmetric weight matrix raises error."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2])
        W = np.array([[1, 0.5], [0.3, 1]])  # Asymmetric

        with self.assertRaises(ValueError) as context:
            weighted_least_squares(A, b, W)

        self.assertIn("symmetric", str(context.exception).lower())

    def test_non_positive_definite_raises_error(self):
        """Test that non-positive definite W raises error."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2])
        W = np.array([[1, 0], [0, -1]])  # Negative eigenvalue

        with self.assertRaises(ValueError) as context:
            weighted_least_squares(A, b, W)

        self.assertIn("positive semi-definite", str(context.exception).lower())

    def test_covariance_computation(self):
        """Test covariance computation in WLS."""
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([1.0, 2.0, 3.0])
        W = np.eye(3)

        x_hat, P = weighted_least_squares(A, b, W, return_covariance=True)

        # Check covariance is symmetric
        assert_allclose(P, P.T, atol=1e-10)

        # Check positive definite
        eigenvalues = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_diagonal_weights_1d_input(self):
        """Test WLS with 1D diagonal weights array."""
        # Synthetic linear system: y = 2x + 1 with varying noise
        A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        x_true = np.array([1.0, 2.0])  # intercept=1, slope=2
        b = A @ x_true  # Clean data

        # 1D diagonal weights
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        x_hat, P = weighted_least_squares(A, b, weights)

        # Should recover exact parameters
        assert_allclose(x_hat, x_true, atol=1e-10)

        # Should match full weight matrix approach
        W_full = np.diag(weights)
        x_hat_full, _ = weighted_least_squares(A, b, W_full)
        assert_allclose(x_hat, x_hat_full, atol=1e-10)

    def test_sigma_input_converts_to_weights(self):
        """Test WLS with sigma input (wᵢ = 1/σᵢ² as per book Section 3.1.1)."""
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([1.0, 2.0, 3.5])

        # Measurement standard deviations
        sigma = np.array([0.1, 0.1, 0.5])  # Third is less accurate

        # Using sigma input
        x_sigma, P_sigma = weighted_least_squares(A, b, sigma, is_sigma=True)

        # Expected weights: wᵢ = 1/σᵢ²
        expected_weights = 1.0 / sigma**2
        W_expected = np.diag(expected_weights)
        x_expected, P_expected = weighted_least_squares(A, b, W_expected)

        # Should match
        assert_allclose(x_sigma, x_expected, atol=1e-10)
        assert_allclose(P_sigma, P_expected, atol=1e-10)

    def test_sigma_weighting_favors_accurate_measurements(self):
        """Test that low sigma (high weight) measurements dominate."""
        # Two measurements: one accurate, one noisy
        A = np.array([[1], [1]])
        b = np.array([1.0, 5.0])  # True value is 1.0, second is way off
        sigma = np.array([0.1, 10.0])  # First very accurate, second very noisy

        x_hat, _ = weighted_least_squares(A, b, sigma, is_sigma=True)

        # Should be much closer to 1.0 than 5.0
        self.assertLess(abs(x_hat[0] - 1.0), 0.1)
        self.assertGreater(abs(x_hat[0] - 5.0), 3.0)

    def test_negative_sigma_raises_error(self):
        """Test that negative sigma values raise error."""
        A = np.array([[1], [1]])
        b = np.array([1.0, 2.0])
        sigma = np.array([0.1, -0.5])  # Invalid negative sigma

        with self.assertRaises(ValueError) as context:
            weighted_least_squares(A, b, sigma, is_sigma=True)

        self.assertIn("positive", str(context.exception).lower())

    def test_negative_weights_raise_error(self):
        """Test that negative weights raise error."""
        A = np.array([[1], [1]])
        b = np.array([1.0, 2.0])
        weights = np.array([1.0, -0.5])  # Invalid negative weight

        with self.assertRaises(ValueError):
            weighted_least_squares(A, b, weights)


class TestIterativeLeastSquares(unittest.TestCase):
    """Test cases for iterative (Gauss-Newton) least squares."""

    def test_range_positioning_2d(self):
        """Test 2D positioning from range measurements (nonlinear)."""
        # 4 anchors at corners of unit square
        anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        true_position = np.array([0.3, 0.4])

        # Measurement function: ranges to anchors
        def f(x):
            return np.linalg.norm(anchors - x, axis=1)

        # Jacobian: (x - anchor) / range
        def jacobian(x):
            diff = x - anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            # Avoid division by zero
            ranges = np.maximum(ranges, 1e-10)
            return diff / ranges

        # Generate measurements
        ranges = f(true_position)

        # Initial guess
        x0 = np.array([0.5, 0.5])

        # Estimate position
        x_hat, P, iters = iterative_least_squares(f, jacobian, ranges, x0)

        # Should converge to true position
        assert_allclose(x_hat, true_position, atol=1e-6)

        # Should converge in < 10 iterations
        self.assertLess(iters, 10)

        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_convergence_with_noise(self):
        """Test convergence with noisy measurements."""
        anchors = np.array([[0, 0], [1, 0], [0, 1]])
        true_position = np.array([0.5, 0.5])

        def f(x):
            return np.linalg.norm(anchors - x, axis=1)

        def jacobian(x):
            diff = x - anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            ranges = np.maximum(ranges, 1e-10)
            return diff / ranges

        # Noisy measurements
        ranges = f(true_position) + 0.01 * np.random.randn(3)

        x0 = np.array([0.3, 0.3])
        x_hat, P, iters = iterative_least_squares(f, jacobian, ranges, x0)

        # Should be close to true position (within noise level)
        error = np.linalg.norm(x_hat - true_position)
        self.assertLess(error, 0.1)

    def test_linear_problem(self):
        """Test that iterative LS works on linear problems too."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        x_true = np.array([1.0, 2.0])
        b = A @ x_true

        def f(x):
            return A @ x

        def jacobian(x):
            return A

        x0 = np.zeros(2)
        x_hat, P, iters = iterative_least_squares(f, jacobian, b, x0)

        # Should converge to true solution
        assert_allclose(x_hat, x_true, atol=1e-6)

        # Should converge quickly for linear problem
        self.assertLess(iters, 5)

    def test_max_iterations(self):
        """Test that max_iter is respected."""

        def f(x):
            return np.array([x[0] ** 2, x[1] ** 2])

        def jacobian(x):
            return np.array([[2 * x[0], 0], [0, 2 * x[1]]])

        b = np.array([1.0, 4.0])
        x0 = np.array([10.0, 10.0])  # Bad initial guess

        # Run with limited iterations
        x_hat, P, iters = iterative_least_squares(
            f, jacobian, b, x0, max_iter=3, tol=1e-10
        )

        # Should stop at max_iter
        self.assertEqual(iters, 3)


class TestRobustLeastSquares(unittest.TestCase):
    """Test cases for robust least squares."""

    def test_huber_with_outliers(self):
        """Test Huber robust LS downweights outliers."""
        # Data with one outlier
        A = np.array([[1], [1], [1], [1], [1]])
        b = np.array([1.0, 1.1, 0.9, 1.05, 5.0])  # Last is outlier

        x_robust, _, weights = robust_least_squares(A, b, method="huber")

        # Outlier should have low weight
        self.assertLess(weights[-1], weights[0])

        # Estimate should be close to 1.0, not pulled by outlier
        self.assertLess(abs(x_robust[0] - 1.0), 0.3)

    def test_cauchy_with_outliers(self):
        """Test Cauchy robust LS."""
        A = np.vstack([np.eye(3), np.eye(3)])
        b = np.array([1.0, 2.0, 3.0, 1.1, 2.1, 10.0])  # Last is outlier

        x_robust, P, weights = robust_least_squares(
            A, b, method="cauchy", threshold=2.0
        )

        # Outlier weight should be small
        self.assertLess(weights[-1], 0.5)

        # Estimates for first two should be close to truth
        assert_allclose(x_robust[:2], [1.05, 2.05], atol=0.2)

    def test_tukey_with_outliers(self):
        """Test Tukey biweight robust LS."""
        A = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        b = np.array([1.0, 1.1, 5.0, 2.0, 2.1, -3.0])  # Two outliers

        x_robust, P, weights = robust_least_squares(
            A, b, method="tukey", threshold=2.0
        )

        # Outliers should have zero or near-zero weight
        self.assertLess(weights[2], 0.1)
        self.assertLess(weights[5], 0.1)

    def test_robust_vs_standard_ls(self):
        """Compare robust LS to standard LS with clean data."""
        # Clean data - both should agree
        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        x_true = np.array([1.0, 2.0])
        b = A @ x_true + 0.01 * np.random.randn(4)

        x_ls, _ = linear_least_squares(A, b)
        x_robust, _, _ = robust_least_squares(A, b, method="huber")

        # Should be very similar with no outliers
        assert_allclose(x_ls, x_robust, atol=0.1)

    def test_invalid_method_raises_error(self):
        """Test that invalid robust method raises error."""
        A = np.array([[1, 2]])
        b = np.array([1])

        with self.assertRaises(ValueError) as context:
            robust_least_squares(A, b, method="invalid")

        self.assertIn("unknown method", str(context.exception).lower())

    def test_all_weights_converge(self):
        """Test that IRLS converges to stable weights."""
        np.random.seed(42)
        A = np.random.randn(10, 2)
        x_true = np.array([1.0, -0.5])
        b = A @ x_true + 0.1 * np.random.randn(10)

        # Add one outlier
        b[5] += 3.0

        x_hat, P, weights = robust_least_squares(A, b, method="huber", max_iter=20)

        # All weights should be in valid range
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1.0))

        # Outlier weight should be < 0.5
        self.assertLess(weights[5], 0.5)

    def test_geman_mcclure_downweights_large_residuals(self):
        """Test G-M (Geman-McClure) robust LS from Table 3.1.

        G-M loss: e(x) = ½ ‖r(x)‖² / (1 + ‖r(x)‖²)
        Weight: w = 1 / (1 + r²)²

        This provides even stronger outlier rejection than Cauchy.
        """
        # Simple 1D problem with outliers
        A = np.ones((6, 1))  # 6 measurements of same quantity
        b = np.array([1.0, 1.1, 0.95, 1.05, 10.0, -5.0])  # Two severe outliers

        x_gm, P_gm, weights_gm = robust_least_squares(
            A, b, method="gm", threshold=2.0
        )

        # Outliers (indices 4, 5) should have very low weights
        self.assertLess(weights_gm[4], 0.1)  # 10.0 outlier
        self.assertLess(weights_gm[5], 0.1)  # -5.0 outlier

        # Good measurements should have higher weights
        self.assertGreater(weights_gm[0], weights_gm[4])
        self.assertGreater(weights_gm[1], weights_gm[5])

        # Estimate should be close to 1.0 (mean of good measurements)
        self.assertLess(abs(x_gm[0] - 1.0), 0.2)

    def test_geman_mcclure_alias(self):
        """Test that 'geman_mcclure' is accepted as alias for 'gm'."""
        A = np.array([[1], [1], [1], [1]])
        b = np.array([1.0, 1.1, 1.0, 5.0])

        x_gm, _, weights_gm = robust_least_squares(A, b, method="gm")
        x_geman, _, weights_geman = robust_least_squares(A, b, method="geman_mcclure")

        assert_allclose(x_gm, x_geman, atol=1e-10)
        assert_allclose(weights_gm, weights_geman, atol=1e-10)

    def test_l2_equivalent_to_standard_ls(self):
        """Test L2 robust loss is equivalent to standard LS (Table 3.1).

        L2 loss: e(x) = ½‖r(x)‖² (no outlier rejection)
        """
        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        b = np.array([5.0, 11.0, 17.1, 23.0])  # Includes slight noise

        x_l2, _, weights_l2 = robust_least_squares(A, b, method="l2")
        x_ls, _ = linear_least_squares(A, b)

        # L2 should match standard LS exactly
        assert_allclose(x_l2, x_ls, atol=1e-10)

        # All weights should be 1.0 for L2
        assert_allclose(weights_l2, np.ones(4), atol=1e-10)

    def test_gm_stronger_than_cauchy_on_outliers(self):
        """Test G-M provides stronger outlier rejection than Cauchy.

        G-M weight: w = 1/(1+r²)²
        Cauchy weight: w = 1/(1+r²)

        For same |r| > 1, G-M weight is always smaller.
        """
        A = np.ones((5, 1))
        b = np.array([1.0, 1.0, 1.0, 1.0, 20.0])  # Extreme outlier

        _, _, weights_cauchy = robust_least_squares(
            A, b, method="cauchy", threshold=1.5
        )
        _, _, weights_gm = robust_least_squares(
            A, b, method="gm", threshold=1.5
        )

        # G-M should downweight the outlier more than Cauchy
        self.assertLess(weights_gm[-1], weights_cauchy[-1])

    def test_robust_methods_table_3_1_coverage(self):
        """Verify all Table 3.1 methods are available: L2, Cauchy, Huber, G-M."""
        A = np.array([[1], [1], [1]])
        b = np.array([1.0, 1.1, 5.0])

        # All Table 3.1 methods should work without error
        for method in ["l2", "cauchy", "huber", "gm"]:
            x, P, weights = robust_least_squares(A, b, method=method)
            self.assertEqual(len(x), 1)
            self.assertEqual(len(weights), 3)

        # Tukey (extra, not in Table 3.1) should also work
        x, P, weights = robust_least_squares(A, b, method="tukey")
        self.assertEqual(len(x), 1)


if __name__ == "__main__":
    unittest.main()

