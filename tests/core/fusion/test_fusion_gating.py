"""Unit tests for core.fusion.gating module.

Tests chi-square gating and Mahalanobis distance functions
implementing Eqs. (8.8)-(8.9) from Chapter 8.

Author: Navigation Engineer
References: Chapter 8, Section 8.3 (Tuning and Robustness)
"""

import unittest

import numpy as np
from scipy import stats

from core.fusion.gating import (
    chi_square_bounds,
    chi_square_gate,
    chi_square_threshold,
    mahalanobis_distance_squared,
)


class TestMahalanobisDistanceSquared(unittest.TestCase):
    """Test suite for mahalanobis_distance_squared function (Eq. 8.8)."""
    
    def test_scalar_case(self) -> None:
        """Test Mahalanobis distance for scalar innovation."""
        y = np.array([2.0])
        S = np.array([[4.0]])
        
        d_sq = mahalanobis_distance_squared(y, S)
        
        # d^2 = y^T S^{-1} y = 2.0 * (1/4) * 2.0 = 1.0
        self.assertAlmostEqual(d_sq, 1.0, places=10)
    
    def test_identity_covariance(self) -> None:
        """Test Mahalanobis distance with identity covariance."""
        y = np.array([3.0, 4.0])
        S = np.eye(2)
        
        d_sq = mahalanobis_distance_squared(y, S)
        
        # d^2 = 3^2 + 4^2 = 25
        self.assertAlmostEqual(d_sq, 25.0, places=10)
    
    def test_diagonal_covariance(self) -> None:
        """Test Mahalanobis distance with diagonal covariance."""
        y = np.array([2.0, 3.0])
        S = np.diag([4.0, 9.0])
        
        d_sq = mahalanobis_distance_squared(y, S)
        
        # d^2 = 2^2/4 + 3^2/9 = 1 + 1 = 2
        self.assertAlmostEqual(d_sq, 2.0, places=10)
    
    def test_zero_innovation(self) -> None:
        """Test that zero innovation gives zero distance."""
        y = np.array([0.0, 0.0, 0.0])
        S = np.diag([1.0, 2.0, 3.0])
        
        d_sq = mahalanobis_distance_squared(y, S)
        
        self.assertAlmostEqual(d_sq, 0.0, places=10)
    
    def test_correlated_covariance(self) -> None:
        """Test Mahalanobis distance with correlated covariance."""
        y = np.array([1.0, 1.0])
        S = np.array([[2.0, 1.0], [1.0, 2.0]])
        
        d_sq = mahalanobis_distance_squared(y, S)
        
        # S^{-1} = (1/3) * [[2, -1], [-1, 2]]
        # d^2 = [1, 1] @ (1/3) * [[2, -1], [-1, 2]] @ [1, 1]
        #     = (1/3) * [1, 1] @ [1, 1] = (1/3) * 2 = 2/3
        self.assertAlmostEqual(d_sq, 2.0/3.0, places=10)
    
    def test_dimension_mismatch_raises(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        y = np.array([1.0, 2.0])
        S = np.eye(3)
        
        with self.assertRaises(ValueError):
            mahalanobis_distance_squared(y, S)
    
    def test_singular_covariance_raises(self) -> None:
        """Test that singular covariance raises ValueError."""
        y = np.array([1.0, 1.0])
        S = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
        
        with self.assertRaises(ValueError):
            mahalanobis_distance_squared(y, S)
    
    def test_not_1d_innovation_raises(self) -> None:
        """Test that 2D innovation raises ValueError."""
        y = np.array([[1.0, 2.0]])
        S = np.eye(2)
        
        with self.assertRaises(ValueError):
            mahalanobis_distance_squared(y, S)
    
    def test_positive_result(self) -> None:
        """Test that result is always non-negative."""
        # Random test cases
        np.random.seed(42)
        
        for _ in range(10):
            m = np.random.randint(1, 5)
            y = np.random.randn(m)
            
            # Generate positive definite S
            A = np.random.randn(m, m)
            S = A @ A.T + 0.1 * np.eye(m)
            
            d_sq = mahalanobis_distance_squared(y, S)
            
            self.assertGreaterEqual(d_sq, 0.0)


class TestChiSquareGate(unittest.TestCase):
    """Test suite for chi_square_gate function (Eq. 8.9)."""
    
    def test_small_innovation_accepted(self) -> None:
        """Test that small innovation is accepted."""
        y = np.array([0.1, 0.2])
        S = np.eye(2)
        
        accept = chi_square_gate(y, S, alpha=0.05)
        
        self.assertTrue(accept)
    
    def test_large_innovation_rejected(self) -> None:
        """Test that large innovation is rejected."""
        y = np.array([5.0, 5.0])
        S = np.eye(2)
        
        accept = chi_square_gate(y, S, alpha=0.05)
        
        self.assertFalse(accept)
    
    def test_threshold_case(self) -> None:
        """Test behavior near the chi-square threshold."""
        # For m=2, alpha=0.05, chi2 critical value ≈ 5.99
        # So d^2 just below should accept, just above should reject
        
        S = np.eye(2)
        
        # d^2 = 5.0 < 5.99 → accept
        y_accept = np.array([np.sqrt(2.5), np.sqrt(2.5)])  # d^2 = 5.0
        self.assertTrue(chi_square_gate(y_accept, S, alpha=0.05))
        
        # d^2 = 7.0 > 5.99 → reject
        y_reject = np.array([np.sqrt(3.5), np.sqrt(3.5)])  # d^2 = 7.0
        self.assertFalse(chi_square_gate(y_reject, S, alpha=0.05))
    
    def test_conservative_gating(self) -> None:
        """Test that lower alpha (more conservative) is more restrictive."""
        y = np.array([2.5, 0.0])  # d^2 = 6.25
        S = np.eye(2)
        
        # At 90% confidence (alpha=0.10), chi2(2, 0.10) ≈ 4.61
        # d^2 = 6.25 > 4.61 → reject
        accept_90 = chi_square_gate(y, S, alpha=0.10)
        
        # At 95% confidence (alpha=0.05), chi2(2, 0.05) ≈ 5.99
        # d^2 = 6.25 > 5.99 → reject
        accept_95 = chi_square_gate(y, S, alpha=0.05)
        
        # At 99% confidence (alpha=0.01), chi2(2, 0.01) ≈ 9.21
        # d^2 = 6.25 < 9.21 → accept
        accept_99 = chi_square_gate(y, S, alpha=0.01)
        
        self.assertFalse(accept_90)
        self.assertFalse(accept_95)
        self.assertTrue(accept_99)
    
    def test_zero_innovation_always_accepted(self) -> None:
        """Test that zero innovation is always accepted."""
        y = np.array([0.0, 0.0, 0.0])
        S = np.eye(3)
        
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            self.assertTrue(chi_square_gate(y, S, alpha=alpha))
    
    def test_invalid_alpha_raises(self) -> None:
        """Test that alpha outside (0, 1) raises ValueError."""
        y = np.array([1.0])
        S = np.array([[1.0]])
        
        with self.assertRaises(ValueError):
            chi_square_gate(y, S, alpha=0.0)
        
        with self.assertRaises(ValueError):
            chi_square_gate(y, S, alpha=1.0)
        
        with self.assertRaises(ValueError):
            chi_square_gate(y, S, alpha=-0.1)
        
        with self.assertRaises(ValueError):
            chi_square_gate(y, S, alpha=1.5)
    
    def test_scalar_measurement(self) -> None:
        """Test gating for scalar measurement."""
        # For m=1, alpha=0.05, chi2 ≈ 3.84
        S = np.array([[1.0]])
        
        # d^2 = 1.0 < 3.84 → accept
        y_small = np.array([1.0])
        self.assertTrue(chi_square_gate(y_small, S, alpha=0.05))
        
        # d^2 = 9.0 > 3.84 → reject
        y_large = np.array([3.0])
        self.assertFalse(chi_square_gate(y_large, S, alpha=0.05))
    
    def test_different_covariances(self) -> None:
        """Test that scaling covariance affects gating decision."""
        y = np.array([3.0, 4.0])  # d^2 with I = 25
        
        # Small covariance → large d^2 → reject
        S_small = 0.1 * np.eye(2)
        self.assertFalse(chi_square_gate(y, S_small, alpha=0.05))
        
        # Large covariance → small d^2 → accept
        S_large = 10.0 * np.eye(2)
        self.assertTrue(chi_square_gate(y, S_large, alpha=0.05))


class TestChiSquareThreshold(unittest.TestCase):
    """Test suite for chi_square_threshold function."""
    
    def test_known_values(self) -> None:
        """Test against known chi-square critical values."""
        # m=1, alpha=0.05: chi2 ≈ 3.841
        threshold = chi_square_threshold(dof=1, alpha=0.05)
        self.assertAlmostEqual(threshold, 3.841, places=2)
        
        # m=2, alpha=0.05: chi2 ≈ 5.991
        threshold = chi_square_threshold(dof=2, alpha=0.05)
        self.assertAlmostEqual(threshold, 5.991, places=2)
        
        # m=3, alpha=0.05: chi2 ≈ 7.815
        threshold = chi_square_threshold(dof=3, alpha=0.05)
        self.assertAlmostEqual(threshold, 7.815, places=2)
    
    def test_higher_dof_higher_threshold(self) -> None:
        """Test that threshold increases with degrees of freedom."""
        alpha = 0.05
        
        threshold_1 = chi_square_threshold(dof=1, alpha=alpha)
        threshold_2 = chi_square_threshold(dof=2, alpha=alpha)
        threshold_5 = chi_square_threshold(dof=5, alpha=alpha)
        
        self.assertLess(threshold_1, threshold_2)
        self.assertLess(threshold_2, threshold_5)
    
    def test_lower_alpha_higher_threshold(self) -> None:
        """Test that lower alpha (more conservative) gives higher threshold."""
        dof = 2
        
        threshold_10 = chi_square_threshold(dof=dof, alpha=0.10)  # 90% confidence
        threshold_05 = chi_square_threshold(dof=dof, alpha=0.05)  # 95% confidence
        threshold_01 = chi_square_threshold(dof=dof, alpha=0.01)  # 99% confidence
        
        self.assertLess(threshold_10, threshold_05)
        self.assertLess(threshold_05, threshold_01)
    
    def test_invalid_dof_raises(self) -> None:
        """Test that invalid degrees of freedom raise ValueError."""
        with self.assertRaises(ValueError):
            chi_square_threshold(dof=0, alpha=0.05)
        
        with self.assertRaises(ValueError):
            chi_square_threshold(dof=-1, alpha=0.05)
    
    def test_invalid_alpha_raises(self) -> None:
        """Test that invalid alpha raises ValueError."""
        with self.assertRaises(ValueError):
            chi_square_threshold(dof=2, alpha=0.0)
        
        with self.assertRaises(ValueError):
            chi_square_threshold(dof=2, alpha=1.0)


class TestChiSquareBounds(unittest.TestCase):
    """Test suite for chi_square_bounds function."""
    
    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple of two values."""
        lower, upper = chi_square_bounds(dof=2, alpha=0.05)
        
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
    
    def test_lower_less_than_upper(self) -> None:
        """Test that lower bound < upper bound."""
        lower, upper = chi_square_bounds(dof=3, alpha=0.05)
        
        self.assertLess(lower, upper)
    
    def test_known_values(self) -> None:
        """Test bounds against known chi-square quantiles."""
        # For m=2, alpha=0.05 (95% interval):
        # Lower (2.5 percentile) ≈ 0.051
        # Upper (97.5 percentile) ≈ 7.378
        lower, upper = chi_square_bounds(dof=2, alpha=0.05)
        
        self.assertAlmostEqual(lower, 0.051, places=2)
        self.assertAlmostEqual(upper, 7.378, places=2)
    
    def test_symmetry_around_mean(self) -> None:
        """Test that bounds are approximately symmetric around the mean."""
        # Chi-square mean = dof
        dof = 10
        lower, upper = chi_square_bounds(dof=dof, alpha=0.05)
        
        # For large dof, chi-square approaches Gaussian
        # So bounds should be roughly symmetric around mean
        mean = float(dof)
        
        # Check that mean is between bounds
        self.assertLess(lower, mean)
        self.assertLess(mean, upper)
    
    def test_lower_bound_positive(self) -> None:
        """Test that lower bound is always non-negative."""
        for dof in [1, 2, 5, 10]:
            lower, _ = chi_square_bounds(dof=dof, alpha=0.05)
            self.assertGreaterEqual(lower, 0.0)
    
    def test_different_confidence_levels(self) -> None:
        """Test that wider confidence (smaller alpha) gives wider bounds."""
        dof = 3
        
        lower_90, upper_90 = chi_square_bounds(dof=dof, alpha=0.10)  # 90%
        lower_95, upper_95 = chi_square_bounds(dof=dof, alpha=0.05)  # 95%
        lower_99, upper_99 = chi_square_bounds(dof=dof, alpha=0.01)  # 99%
        
        # 90% interval should be narrowest
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        
        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)
    
    def test_invalid_dof_raises(self) -> None:
        """Test that invalid degrees of freedom raise ValueError."""
        with self.assertRaises(ValueError):
            chi_square_bounds(dof=0, alpha=0.05)
    
    def test_invalid_alpha_raises(self) -> None:
        """Test that invalid alpha raises ValueError."""
        with self.assertRaises(ValueError):
            chi_square_bounds(dof=2, alpha=0.0)
        
        with self.assertRaises(ValueError):
            chi_square_bounds(dof=2, alpha=1.0)
    
    def test_consistency_with_threshold(self) -> None:
        """Test that upper bound matches single-sided threshold."""
        dof = 2
        alpha = 0.05
        
        _, upper = chi_square_bounds(dof=dof, alpha=alpha)
        threshold = chi_square_threshold(dof=dof, alpha=alpha)
        
        # For two-sided test with alpha=0.05, upper bound is at 97.5 percentile
        # For one-sided test with alpha=0.05, threshold is at 95 percentile
        # So upper > threshold
        self.assertGreater(upper, threshold)


if __name__ == "__main__":
    unittest.main()

