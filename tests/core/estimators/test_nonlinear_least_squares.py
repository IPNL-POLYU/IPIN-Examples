"""
Unit tests for nonlinear least squares solvers.

Tests cover:
    - Gauss-Newton method (Eq. 3.52)
    - Levenberg-Marquardt method (Eq. 3.53, Algorithm 3.2)
    - Weighted nonlinear LS
    - Robust nonlinear LS (Table 3.1 loss functions)

Book Reference: Chapter 3, Section 3.4.1 (Numerical Optimization Methods)
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from core.estimators.nonlinear_least_squares import (
    gauss_newton,
    levenberg_marquardt,
    robust_gauss_newton,
    solve_nonlinear_ls,
    NonlinearLSResult,
)


class TestGaussNewton2DRangePositioning(unittest.TestCase):
    """Test Gauss-Newton on canonical 2D range positioning problem.

    This tests hᵢ(x) = ‖x - aᵢ‖ (range from position x to anchor aᵢ).
    """

    def setUp(self):
        """Setup 4 anchors at corners of 10x10 area."""
        self.anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        self.true_pos = np.array([3.0, 4.0])

        # Measurement model: ranges to anchors
        def h(x):
            return np.linalg.norm(self.anchors - x, axis=1)

        # Jacobian: ∂hᵢ/∂x = (x - aᵢ) / ‖x - aᵢ‖
        def jacobian(x):
            diff = x - self.anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        self.h = h
        self.jacobian = jacobian
        self.y_clean = h(self.true_pos)

    def test_exact_measurements_convergence(self):
        """Test GN converges to true position with exact measurements."""
        x0 = np.array([5.0, 5.0])

        result = gauss_newton(self.h, self.jacobian, self.y_clean, x0)

        assert_allclose(result.x, self.true_pos, atol=1e-6)
        self.assertTrue(result.converged)
        self.assertLess(result.iterations, 10)

    def test_noisy_measurements(self):
        """Test GN with noisy measurements stays close to true position."""
        np.random.seed(42)
        noise = 0.1 * np.random.randn(4)
        y_noisy = self.y_clean + noise

        x0 = np.array([5.0, 5.0])
        result = gauss_newton(self.h, self.jacobian, y_noisy, x0)

        # Should be within ~0.5m of true position given 0.1m noise
        error = np.linalg.norm(result.x - self.true_pos)
        self.assertLess(error, 0.5)

    def test_different_initial_guesses(self):
        """Test GN converges from different initial guesses."""
        initial_guesses = [
            np.array([0.0, 0.0]),
            np.array([5.0, 5.0]),
            np.array([8.0, 8.0]),
            np.array([1.0, 9.0]),
        ]

        for x0 in initial_guesses:
            result = gauss_newton(self.h, self.jacobian, self.y_clean, x0)
            assert_allclose(result.x, self.true_pos, atol=1e-4)

    def test_covariance_is_positive_semidefinite(self):
        """Test that returned covariance is positive semi-definite.
        
        Note: With exact measurements (m=n), covariance may have near-zero
        eigenvalues due to numerical precision. We check for positive
        semi-definiteness with a tolerance.
        """
        # Use noisy measurements to ensure m > effective n for covariance
        np.random.seed(42)
        y_noisy = self.y_clean + 0.1 * np.random.randn(4)
        x0 = np.array([5.0, 5.0])
        result = gauss_newton(self.h, self.jacobian, y_noisy, x0)

        self.assertIsNotNone(result.covariance)
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        # Allow small negative eigenvalues due to numerical precision
        self.assertTrue(np.all(eigenvalues > -1e-10))

    def test_result_dataclass_fields(self):
        """Test NonlinearLSResult contains all expected fields."""
        x0 = np.array([5.0, 5.0])
        result = gauss_newton(self.h, self.jacobian, self.y_clean, x0)

        self.assertIsInstance(result, NonlinearLSResult)
        self.assertEqual(len(result.x), 2)
        self.assertEqual(result.covariance.shape, (2, 2))
        self.assertGreater(result.iterations, 0)
        self.assertEqual(len(result.residuals), 4)
        self.assertGreaterEqual(result.cost, 0)
        self.assertIsInstance(result.converged, bool)


class TestLevenbergMarquardt(unittest.TestCase):
    """Test Levenberg-Marquardt solver (Algorithm 3.2)."""

    def setUp(self):
        """Setup same 2D positioning problem."""
        self.anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        self.true_pos = np.array([3.0, 4.0])

        def h(x):
            return np.linalg.norm(self.anchors - x, axis=1)

        def jacobian(x):
            diff = x - self.anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        self.h = h
        self.jacobian = jacobian
        self.y_clean = h(self.true_pos)

    def test_lm_converges_from_poor_initial_guess(self):
        """Test LM handles poor initial guesses better than pure GN."""
        # Very poor initial guess at corner
        x0 = np.array([0.0, 0.0])

        result = levenberg_marquardt(self.h, self.jacobian, self.y_clean, x0)

        assert_allclose(result.x, self.true_pos, atol=1e-4)
        self.assertTrue(result.converged)

    def test_lm_matches_gn_for_good_initial_guess(self):
        """Test LM gives same result as GN when starting close to solution."""
        x0 = np.array([3.5, 4.5])  # Close to true position

        result_gn = gauss_newton(self.h, self.jacobian, self.y_clean, x0)
        result_lm = levenberg_marquardt(self.h, self.jacobian, self.y_clean, x0)

        assert_allclose(result_gn.x, result_lm.x, atol=1e-6)

    def test_lm_damping_parameter(self):
        """Test that custom mu0 is respected."""
        x0 = np.array([5.0, 5.0])

        # Large mu0 should make it more conservative (more iterations typically)
        result_large_mu = levenberg_marquardt(
            self.h, self.jacobian, self.y_clean, x0, mu0=10.0
        )

        # Should still converge
        assert_allclose(result_large_mu.x, self.true_pos, atol=1e-4)


class TestWeightedNonlinearLS(unittest.TestCase):
    """Test weighted nonlinear least squares."""

    def setUp(self):
        """Setup positioning problem with varying measurement quality."""
        self.anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        self.true_pos = np.array([3.0, 4.0])

        def h(x):
            return np.linalg.norm(self.anchors - x, axis=1)

        def jacobian(x):
            diff = x - self.anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        self.h = h
        self.jacobian = jacobian

    def test_weights_emphasize_accurate_measurements(self):
        """Test that high weights favor accurate measurements."""
        # First 3 accurate, last one has large error
        y = self.h(self.true_pos)
        y[3] += 3.0  # Add 3m error to last measurement

        # Without weights (uniform)
        result_unweighted = gauss_newton(
            self.h, self.jacobian, y, np.array([5.0, 5.0])
        )

        # With weights: downweight the bad measurement
        weights = np.array([1.0, 1.0, 1.0, 0.01])
        result_weighted = gauss_newton(
            self.h, self.jacobian, y, np.array([5.0, 5.0]), weights=weights
        )

        # Weighted should be closer to true position
        error_unweighted = np.linalg.norm(result_unweighted.x - self.true_pos)
        error_weighted = np.linalg.norm(result_weighted.x - self.true_pos)

        self.assertLess(error_weighted, error_unweighted)

    def test_weights_validation(self):
        """Test that invalid weights raise errors."""
        y = self.h(self.true_pos)
        x0 = np.array([5.0, 5.0])

        # Wrong length
        with self.assertRaises(ValueError):
            gauss_newton(self.h, self.jacobian, y, x0, weights=np.ones(3))

        # Negative weights
        with self.assertRaises(ValueError):
            gauss_newton(self.h, self.jacobian, y, x0, weights=np.array([1, 1, 1, -1]))


class TestRobustNonlinearLS(unittest.TestCase):
    """Test robust nonlinear LS with Table 3.1 loss functions."""

    def setUp(self):
        """Setup positioning problem with outliers."""
        self.anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        self.true_pos = np.array([3.0, 4.0])

        def h(x):
            return np.linalg.norm(self.anchors - x, axis=1)

        def jacobian(x):
            diff = x - self.anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        self.h = h
        self.jacobian = jacobian
        self.y_clean = h(self.true_pos)

    def test_huber_rejects_outlier(self):
        """Test Huber loss downweights outlier measurement."""
        # Use more anchors to provide redundancy for robust estimation
        anchors_extended = np.array([
            [0, 0], [10, 0], [0, 10], [10, 10],
            [5, 0], [0, 5], [10, 5], [5, 10]
        ], dtype=float)
        true_pos = np.array([3.0, 4.0])

        def h_ext(x):
            return np.linalg.norm(anchors_extended - x, axis=1)

        def jac_ext(x):
            diff = x - anchors_extended
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        y_clean = h_ext(true_pos)
        y_outlier = y_clean.copy()
        y_outlier[3] += 5.0  # 5m NLOS error on one anchor

        x0 = np.array([5.0, 5.0])

        # Standard LS (pulled by outlier)
        result_standard = gauss_newton(h_ext, jac_ext, y_outlier, x0)

        # Robust LS with Huber
        result_robust = robust_gauss_newton(
            h_ext, jac_ext, y_outlier, x0, loss="huber", loss_param=1.0
        )

        # Robust should be closer to true position
        error_standard = np.linalg.norm(result_standard.x - true_pos)
        error_robust = np.linalg.norm(result_robust.x - true_pos)

        self.assertLess(error_robust, error_standard)

        # Outlier should have lower weight than clean measurements
        clean_weights = [result_robust.weights[i] for i in [0, 1, 2, 4, 5, 6, 7]]
        self.assertLess(result_robust.weights[3], np.mean(clean_weights))

    def test_all_table_3_1_losses(self):
        """Test all robust loss functions from Table 3.1 are available."""
        y_outlier = self.y_clean.copy()
        y_outlier[3] += 3.0

        x0 = np.array([5.0, 5.0])

        for loss in ["l2", "huber", "cauchy", "gm"]:
            result = robust_gauss_newton(
                self.h, self.jacobian, y_outlier, x0, loss=loss
            )
            # All should converge and give reasonable result
            self.assertIsNotNone(result.x)
            error = np.linalg.norm(result.x - self.true_pos)
            self.assertLess(error, 3.0)  # Reasonable even with outlier

    def test_gm_stronger_outlier_rejection_than_cauchy(self):
        """Test G-M has stronger outlier rejection than Cauchy."""
        y_outlier = self.y_clean.copy()
        y_outlier[3] += 5.0

        x0 = np.array([5.0, 5.0])

        result_cauchy = robust_gauss_newton(
            self.h, self.jacobian, y_outlier, x0, loss="cauchy"
        )
        result_gm = robust_gauss_newton(
            self.h, self.jacobian, y_outlier, x0, loss="gm"
        )

        # G-M should downweight outlier more
        self.assertLess(result_gm.weights[3], result_cauchy.weights[3])

    def test_l2_equivalent_to_standard(self):
        """Test L2 loss gives same result as standard LS."""
        x0 = np.array([5.0, 5.0])

        result_standard = gauss_newton(self.h, self.jacobian, self.y_clean, x0)
        result_l2 = robust_gauss_newton(
            self.h, self.jacobian, self.y_clean, x0, loss="l2"
        )

        assert_allclose(result_standard.x, result_l2.x, atol=1e-6)


class TestSolveNonlinearLS(unittest.TestCase):
    """Test the convenience solve_nonlinear_ls function."""

    def setUp(self):
        """Setup standard positioning problem."""
        self.anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        self.true_pos = np.array([3.0, 4.0])

        def h(x):
            return np.linalg.norm(self.anchors - x, axis=1)

        def jacobian(x):
            diff = x - self.anchors
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        self.h = h
        self.jacobian = jacobian
        self.y = h(self.true_pos)

    def test_method_gn(self):
        """Test method='gn' uses Gauss-Newton."""
        x0 = np.array([5.0, 5.0])
        result = solve_nonlinear_ls(
            self.h, self.jacobian, self.y, x0, method="gn"
        )
        assert_allclose(result.x, self.true_pos, atol=1e-6)

    def test_method_lm(self):
        """Test method='lm' uses Levenberg-Marquardt."""
        x0 = np.array([5.0, 5.0])
        result = solve_nonlinear_ls(
            self.h, self.jacobian, self.y, x0, method="lm"
        )
        assert_allclose(result.x, self.true_pos, atol=1e-6)

    def test_robust_loss_parameter(self):
        """Test robust_loss parameter activates robust estimation."""
        # Use more anchors for redundancy
        anchors_ext = np.array([
            [0, 0], [10, 0], [0, 10], [10, 10],
            [5, 0], [0, 5]
        ], dtype=float)
        true_pos = np.array([3.0, 4.0])

        def h_ext(x):
            return np.linalg.norm(anchors_ext - x, axis=1)

        def jac_ext(x):
            diff = x - anchors_ext
            ranges = np.linalg.norm(diff, axis=1, keepdims=True)
            return diff / np.maximum(ranges, 1e-10)

        y_outlier = h_ext(true_pos)
        y_outlier[3] += 5.0  # Large outlier

        x0 = np.array([5.0, 5.0])
        result = solve_nonlinear_ls(
            h_ext, jac_ext, y_outlier, x0, robust_loss="huber", loss_param=1.0
        )

        # Should have weights
        self.assertIsNotNone(result.weights)
        # Outlier weight should be reduced compared to average of clean
        clean_weights = [result.weights[i] for i in [0, 1, 2, 4, 5]]
        self.assertLess(result.weights[3], np.mean(clean_weights))

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        x0 = np.array([5.0, 5.0])
        with self.assertRaises(ValueError):
            solve_nonlinear_ls(
                self.h, self.jacobian, self.y, x0, method="invalid"
            )


class TestInputValidation(unittest.TestCase):
    """Test input validation for all solvers."""

    def setUp(self):
        """Setup simple measurement model."""
        def h(x):
            return np.array([x[0], x[1]])

        def jacobian(x):
            return np.eye(2)

        self.h = h
        self.jacobian = jacobian

    def test_y_must_be_1d(self):
        """Test that 2D y raises error."""
        with self.assertRaises(ValueError):
            gauss_newton(
                self.h, self.jacobian,
                y=np.array([[1], [2]]),  # 2D
                x0=np.array([0, 0])
            )

    def test_x0_must_be_1d(self):
        """Test that 2D x0 raises error."""
        with self.assertRaises(ValueError):
            gauss_newton(
                self.h, self.jacobian,
                y=np.array([1, 2]),
                x0=np.array([[0], [0]])  # 2D
            )

    def test_jacobian_shape_validation(self):
        """Test that wrong Jacobian shape raises error."""
        def bad_jacobian(x):
            return np.eye(3)  # Wrong shape

        with self.assertRaises(ValueError):
            gauss_newton(
                self.h, bad_jacobian,
                y=np.array([1, 2]),
                x0=np.array([0, 0])
            )

    def test_h_output_shape_validation(self):
        """Test that wrong h output shape raises error."""
        def bad_h(x):
            return np.array([x[0]])  # Wrong length

        with self.assertRaises(ValueError):
            gauss_newton(
                bad_h, self.jacobian,
                y=np.array([1, 2]),
                x0=np.array([0, 0])
            )


if __name__ == "__main__":
    unittest.main()

