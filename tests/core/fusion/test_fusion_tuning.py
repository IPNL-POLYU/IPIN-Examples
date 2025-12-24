"""Unit tests for core.fusion.tuning module.

Tests innovation monitoring and robust covariance tuning functions
implementing Eqs. (8.5)-(8.7) from Chapter 8.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3 (Tuning and Robustness)
"""

import unittest

import numpy as np

from core.fusion.tuning import (
    cauchy_weight,
    compute_normalized_innovation,
    huber_weight,
    innovation,
    innovation_covariance,
    scale_measurement_covariance,
)


class TestInnovation(unittest.TestCase):
    """Test suite for innovation function (Eq. 8.5)."""
    
    def test_scalar_innovation(self) -> None:
        """Test innovation computation for scalar measurement."""
        z = np.array([5.2])
        z_pred = np.array([5.0])
        y = innovation(z, z_pred)
        
        np.testing.assert_array_almost_equal(y, [0.2])
    
    def test_vector_innovation(self) -> None:
        """Test innovation computation for vector measurement."""
        z = np.array([5.2, 3.1, 2.8])
        z_pred = np.array([5.0, 3.0, 3.0])
        y = innovation(z, z_pred)
        
        np.testing.assert_array_almost_equal(y, [0.2, 0.1, -0.2])
    
    def test_zero_innovation(self) -> None:
        """Test that perfect prediction gives zero innovation."""
        z = np.array([1.0, 2.0, 3.0])
        z_pred = np.array([1.0, 2.0, 3.0])
        y = innovation(z, z_pred)
        
        np.testing.assert_array_almost_equal(y, [0.0, 0.0, 0.0])
    
    def test_negative_innovation(self) -> None:
        """Test that overestimated prediction gives negative innovation."""
        z = np.array([1.0])
        z_pred = np.array([2.0])
        y = innovation(z, z_pred)
        
        self.assertLess(y[0], 0.0)
        np.testing.assert_array_almost_equal(y, [-1.0])
    
    def test_dimension_mismatch_raises(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        z = np.array([1.0, 2.0])
        z_pred = np.array([1.0])
        
        with self.assertRaises(ValueError):
            innovation(z, z_pred)
    
    def test_works_with_lists(self) -> None:
        """Test that function accepts lists (converted to arrays)."""
        y = innovation([1.0, 2.0], [0.5, 1.5])
        np.testing.assert_array_almost_equal(y, [0.5, 0.5])


class TestInnovationCovariance(unittest.TestCase):
    """Test suite for innovation_covariance function (Eq. 8.6)."""
    
    def test_identity_observation(self) -> None:
        """Test S = P + R for identity observation matrix."""
        H = np.eye(2)
        P_pred = np.diag([0.5, 0.3])
        R = np.diag([0.1, 0.1])
        
        S = innovation_covariance(H, P_pred, R)
        
        expected = np.diag([0.6, 0.4])
        np.testing.assert_array_almost_equal(S, expected)
    
    def test_scalar_case(self) -> None:
        """Test innovation covariance for scalar measurement."""
        H = np.array([[1.0, 0.0]])  # observe first state only
        P_pred = np.diag([2.0, 1.0])
        R = np.array([[0.5]])
        
        S = innovation_covariance(H, P_pred, R)
        
        # S = H P H^T + R = 1*2*1 + 0.5 = 2.5
        np.testing.assert_array_almost_equal(S, [[2.5]])
    
    def test_partial_observation(self) -> None:
        """Test innovation covariance for partial state observation."""
        # Observe only first component of 2D state
        H = np.array([[1.0, 0.0]])
        P_pred = np.array([[1.0, 0.5], [0.5, 2.0]])
        R = np.array([[0.1]])
        
        S = innovation_covariance(H, P_pred, R)
        
        # S = [1, 0] @ [[1, 0.5], [0.5, 2]] @ [1, 0]^T + 0.1
        #   = [1, 0] @ [1, 0.5]^T + 0.1 = 1.0 + 0.1 = 1.1
        np.testing.assert_array_almost_equal(S, [[1.1]])
    
    def test_symmetry(self) -> None:
        """Test that output covariance is symmetric."""
        H = np.random.randn(3, 5)
        P_pred = np.eye(5) * 0.5
        R = np.eye(3) * 0.2
        
        S = innovation_covariance(H, P_pred, R)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(S, S.T)
    
    def test_dimension_validation(self) -> None:
        """Test that dimension mismatches raise ValueError."""
        # H is 2x3, P should be 3x3
        H = np.random.randn(2, 3)
        P_pred = np.eye(4)  # wrong size
        R = np.eye(2)
        
        with self.assertRaises(ValueError):
            innovation_covariance(H, P_pred, R)
    
    def test_measurement_dimension_validation(self) -> None:
        """Test R dimension must match measurement dimension."""
        H = np.random.randn(2, 3)
        P_pred = np.eye(3)
        R = np.eye(3)  # should be 2x2
        
        with self.assertRaises(ValueError):
            innovation_covariance(H, P_pred, R)
    
    def test_not_2d_raises(self) -> None:
        """Test that 1D inputs raise ValueError."""
        with self.assertRaises(ValueError):
            innovation_covariance(
                np.array([1.0, 0.0]),  # 1D
                np.eye(2),
                np.eye(2)
            )


class TestScaleMeasurementCovariance(unittest.TestCase):
    """Test suite for scale_measurement_covariance function (Eq. 8.7)."""
    
    def test_unit_weight(self) -> None:
        """Test that weight=1 leaves covariance unchanged."""
        R = np.diag([0.1, 0.2])
        R_scaled = scale_measurement_covariance(R, weight=1.0)
        
        np.testing.assert_array_almost_equal(R_scaled, R)
    
    def test_upweighting(self) -> None:
        """Test that weight > 1 inflates covariance (reduces confidence)."""
        R = np.diag([0.1, 0.2])
        weight = 2.0
        
        R_scaled = scale_measurement_covariance(R, weight)
        
        expected = np.diag([0.2, 0.4])
        np.testing.assert_array_almost_equal(R_scaled, expected)
    
    def test_downweighting(self) -> None:
        """Test that weight < 1 reduces covariance (increases confidence)."""
        R = np.diag([1.0, 2.0])
        weight = 0.5
        
        R_scaled = scale_measurement_covariance(R, weight)
        
        expected = np.diag([0.5, 1.0])
        np.testing.assert_array_almost_equal(R_scaled, expected)
    
    def test_outlier_case(self) -> None:
        """Test aggressive outlier down-weighting (inflate by 100x)."""
        R = np.diag([0.1, 0.1])
        weight = 100.0
        
        R_scaled = scale_measurement_covariance(R, weight)
        
        expected = np.diag([10.0, 10.0])
        np.testing.assert_array_almost_equal(R_scaled, expected)
    
    def test_zero_weight(self) -> None:
        """Test that zero weight gives zero covariance."""
        R = np.diag([1.0, 2.0])
        R_scaled = scale_measurement_covariance(R, weight=0.0)
        
        np.testing.assert_array_almost_equal(R_scaled, np.zeros((2, 2)))
    
    def test_negative_weight_raises(self) -> None:
        """Test that negative weight raises ValueError."""
        R = np.eye(2)
        
        with self.assertRaises(ValueError):
            scale_measurement_covariance(R, weight=-1.0)
    
    def test_non_diagonal_covariance(self) -> None:
        """Test scaling works for non-diagonal covariance."""
        R = np.array([[1.0, 0.5], [0.5, 2.0]])
        weight = 3.0
        
        R_scaled = scale_measurement_covariance(R, weight)
        
        expected = np.array([[3.0, 1.5], [1.5, 6.0]])
        np.testing.assert_array_almost_equal(R_scaled, expected)
    
    def test_invalid_dimension_raises(self) -> None:
        """Test that 1D input raises ValueError."""
        with self.assertRaises(ValueError):
            scale_measurement_covariance(np.array([1.0, 2.0]), weight=2.0)


class TestHuberWeight(unittest.TestCase):
    """Test suite for Huber robust weight function."""
    
    def test_inlier_weight_is_one(self) -> None:
        """Test that inliers (|r| <= k) get weight 1."""
        self.assertEqual(huber_weight(0.0, threshold=1.345), 1.0)
        self.assertEqual(huber_weight(0.5, threshold=1.345), 1.0)
        self.assertEqual(huber_weight(1.0, threshold=1.345), 1.0)
        self.assertEqual(huber_weight(1.345, threshold=1.345), 1.0)
    
    def test_outlier_weight_decreases(self) -> None:
        """Test that outliers (|r| > k) get weight < 1."""
        w = huber_weight(3.0, threshold=1.345)
        self.assertLess(w, 1.0)
        self.assertGreater(w, 0.0)
        self.assertAlmostEqual(w, 1.345 / 3.0, places=10)
    
    def test_larger_outlier_smaller_weight(self) -> None:
        """Test that larger residuals get smaller weights."""
        w1 = huber_weight(2.0, threshold=1.345)
        w2 = huber_weight(5.0, threshold=1.345)
        
        self.assertLess(w2, w1)
    
    def test_negative_residual(self) -> None:
        """Test that negative residuals work correctly (absolute value)."""
        w_pos = huber_weight(3.0, threshold=1.345)
        w_neg = huber_weight(-3.0, threshold=1.345)
        
        self.assertEqual(w_pos, w_neg)
    
    def test_threshold_effect(self) -> None:
        """Test that higher threshold is more permissive."""
        residual = 2.0
        
        w_strict = huber_weight(residual, threshold=1.0)   # strict
        w_loose = huber_weight(residual, threshold=3.0)    # loose
        
        # Strict threshold treats 2.0 as outlier, loose treats as inlier
        self.assertLess(w_strict, 1.0)
        self.assertEqual(w_loose, 1.0)


class TestCauchyWeight(unittest.TestCase):
    """Test suite for Cauchy robust weight function."""
    
    def test_zero_residual_weight_is_one(self) -> None:
        """Test that zero residual gets weight 1."""
        self.assertEqual(cauchy_weight(0.0, scale=2.385), 1.0)
    
    def test_weight_at_scale(self) -> None:
        """Test that weight at r=c is 0.5."""
        w = cauchy_weight(2.385, scale=2.385)
        self.assertAlmostEqual(w, 0.5, places=10)
    
    def test_outlier_weight_decreases(self) -> None:
        """Test that large residuals get small weights."""
        w = cauchy_weight(10.0, scale=2.385)
        self.assertLess(w, 0.2)
        self.assertGreater(w, 0.0)
    
    def test_negative_residual(self) -> None:
        """Test that negative residuals work correctly."""
        w_pos = cauchy_weight(5.0, scale=2.385)
        w_neg = cauchy_weight(-5.0, scale=2.385)
        
        self.assertAlmostEqual(w_pos, w_neg, places=10)
    
    def test_cauchy_stronger_than_huber(self) -> None:
        """Test that Cauchy down-weights outliers more aggressively than Huber."""
        residual = 5.0
        
        w_huber = huber_weight(residual, threshold=1.345)
        w_cauchy = cauchy_weight(residual, scale=2.385)
        
        # Cauchy should give smaller weight for large outliers
        self.assertLess(w_cauchy, w_huber)
    
    def test_scale_effect(self) -> None:
        """Test that larger scale is more permissive."""
        residual = 3.0
        
        w_small_scale = cauchy_weight(residual, scale=1.0)
        w_large_scale = cauchy_weight(residual, scale=5.0)
        
        self.assertLess(w_small_scale, w_large_scale)


class TestComputeNormalizedInnovation(unittest.TestCase):
    """Test suite for compute_normalized_innovation function."""
    
    def test_scalar_normalization(self) -> None:
        """Test normalization for scalar innovation."""
        y = np.array([2.0])
        S = np.array([[4.0]])  # std = 2.0
        
        y_norm = compute_normalized_innovation(y, S)
        
        # 2.0 / 2.0 = 1.0
        np.testing.assert_array_almost_equal(y_norm, [1.0])
    
    def test_diagonal_covariance(self) -> None:
        """Test normalization with diagonal covariance."""
        y = np.array([2.0, 3.0])
        S = np.diag([4.0, 9.0])  # std = [2.0, 3.0]
        
        y_norm = compute_normalized_innovation(y, S)
        
        # [2.0/2.0, 3.0/3.0] = [1.0, 1.0]
        np.testing.assert_array_almost_equal(y_norm, [1.0, 1.0])
    
    def test_correlated_covariance(self) -> None:
        """Test normalization with correlated (non-diagonal) covariance."""
        y = np.array([1.0, 1.0])
        S = np.array([[2.0, 1.0], [1.0, 2.0]])
        
        y_norm = compute_normalized_innovation(y, S)
        
        # Cholesky: S = L L^T where L = [[sqrt(2), 0], [1/sqrt(2), sqrt(3/2)]]
        # Result should be unit-ish but not [1, 1] due to correlation
        self.assertEqual(len(y_norm), 2)
    
    def test_zero_innovation(self) -> None:
        """Test that zero innovation normalizes to zero."""
        y = np.array([0.0, 0.0])
        S = np.diag([1.0, 2.0])
        
        y_norm = compute_normalized_innovation(y, S)
        
        np.testing.assert_array_almost_equal(y_norm, [0.0, 0.0])
    
    def test_singular_covariance_raises(self) -> None:
        """Test that singular covariance raises ValueError."""
        y = np.array([1.0, 1.0])
        S = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, not positive definite
        
        with self.assertRaises(ValueError):
            compute_normalized_innovation(y, S)
    
    def test_dimension_mismatch_raises(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        y = np.array([1.0, 2.0])
        S = np.eye(3)  # wrong size
        
        with self.assertRaises(ValueError):
            compute_normalized_innovation(y, S)
    
    def test_not_1d_innovation_raises(self) -> None:
        """Test that 2D innovation raises ValueError."""
        y = np.array([[1.0, 2.0]])
        S = np.eye(2)
        
        with self.assertRaises(ValueError):
            compute_normalized_innovation(y, S)


if __name__ == "__main__":
    unittest.main()

