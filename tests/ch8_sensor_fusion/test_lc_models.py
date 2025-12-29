"""Unit tests for LC (Loosely Coupled) fusion models.

Tests the improved WLS position solver with proper covariance handling.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest

import numpy as np

from ch8_sensor_fusion.lc_models import solve_uwb_position_wls


class TestSolveUWBPositionWLS(unittest.TestCase):
    """Test the improved WLS position solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Square anchor layout (20m x 15m)
        self.anchors = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 15.0],
            [0.0, 15.0]
        ])
        
        # True position at center
        self.true_pos = np.array([10.0, 7.5])
        
        # Compute true ranges
        self.true_ranges = np.linalg.norm(
            self.anchors - self.true_pos, axis=1
        )
    
    def test_nominal_case_converges(self):
        """Test that WLS converges on noise-free measurements."""
        pos, cov, converged = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.05
        )
        
        self.assertIsNotNone(pos)
        self.assertIsNotNone(cov)
        self.assertTrue(converged)
        
        # Position should be very close to truth
        np.testing.assert_allclose(pos, self.true_pos, atol=1e-3)
        
        # Covariance should be positive definite
        self.assertTrue(np.all(np.linalg.eigvals(cov) > 0))
    
    def test_with_measurement_noise(self):
        """Test WLS with noisy measurements."""
        np.random.seed(42)
        noise_std = 0.1
        noisy_ranges = self.true_ranges + np.random.randn(4) * noise_std
        
        pos, cov, converged = solve_uwb_position_wls(
            ranges=noisy_ranges,
            anchor_positions=self.anchors,
            range_noise_std=noise_std
        )
        
        self.assertIsNotNone(pos)
        self.assertIsNotNone(cov)
        self.assertTrue(converged)
        
        # Position should be reasonably close (within a few sigmas)
        error = np.linalg.norm(pos - self.true_pos)
        self.assertLess(error, 0.5, "Position error too large with noise")
    
    def test_covariance_floor_enforced(self):
        """Test that covariance floor prevents overconfidence."""
        # Use very small noise to trigger covariance floor
        pos, cov, converged = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.001,  # Very small noise
            cov_floor_std=0.2  # Floor at 0.2m std
        )
        
        self.assertIsNotNone(pos)
        self.assertIsNotNone(cov)
        
        # Standard deviations should be at least the floor value
        std_x = np.sqrt(cov[0, 0])
        std_y = np.sqrt(cov[1, 1])
        
        self.assertGreaterEqual(std_x, 0.2 - 1e-6, "X std below floor")
        self.assertGreaterEqual(std_y, 0.2 - 1e-6, "Y std below floor")
    
    def test_anchor_dependent_noise(self):
        """Test WLS with anchor-dependent noise levels."""
        # Simulate one degraded anchor (e.g., NLOS)
        anchor_stds = np.array([0.05, 0.5, 0.05, 0.05])  # Anchor 1 degraded
        
        pos, cov, converged = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            anchor_noise_std=anchor_stds
        )
        
        self.assertIsNotNone(pos)
        self.assertIsNotNone(cov)
        self.assertTrue(converged)
        
        # Position should still converge
        np.testing.assert_allclose(pos, self.true_pos, atol=1e-3)
        
        # Covariance should be larger than uniform-noise case
        # (degraded anchor reduces overall precision)
        pos_uniform, cov_uniform, _ = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.05
        )
        
        # At least one axis should have larger uncertainty
        self.assertTrue(
            cov[0, 0] > cov_uniform[0, 0] or cov[1, 1] > cov_uniform[1, 1],
            "Degraded anchor should increase covariance"
        )
    
    def test_dropout_handling(self):
        """Test WLS with some NaN ranges (anchor dropouts)."""
        ranges_with_dropout = self.true_ranges.copy()
        ranges_with_dropout[1] = np.nan  # Anchor 1 dropout
        
        pos, cov, converged = solve_uwb_position_wls(
            ranges=ranges_with_dropout,
            anchor_positions=self.anchors,
            range_noise_std=0.05
        )
        
        self.assertIsNotNone(pos)
        self.assertIsNotNone(cov)
        self.assertTrue(converged)
        
        # Should still converge with 3 anchors
        error = np.linalg.norm(pos - self.true_pos)
        self.assertLess(error, 0.1, "Position error too large with one dropout")
    
    def test_insufficient_anchors_fails(self):
        """Test that WLS fails gracefully with < 3 valid anchors."""
        ranges_insufficient = self.true_ranges.copy()
        ranges_insufficient[1:] = np.nan  # Only 1 valid range
        
        pos, cov, converged = solve_uwb_position_wls(
            ranges=ranges_insufficient,
            anchor_positions=self.anchors,
            range_noise_std=0.05
        )
        
        self.assertIsNone(pos)
        self.assertIsNone(cov)
        self.assertFalse(converged)
    
    def test_divergence_detection(self):
        """Test that WLS detects and rejects divergent solutions."""
        # Create inconsistent ranges that would cause divergence
        bad_ranges = np.array([1.0, 1.0, 50.0, 50.0])
        
        pos, cov, converged = solve_uwb_position_wls(
            ranges=bad_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.05,
            max_iterations=5
        )
        
        # Should either fail or produce position within reasonable bounds
        if pos is not None:
            # Check position is within anchor bounds + margin
            anchor_min = np.min(self.anchors, axis=0)
            anchor_max = np.max(self.anchors, axis=0)
            margin = 50.0
            
            self.assertTrue(
                np.all(pos >= anchor_min - margin) and 
                np.all(pos <= anchor_max + margin),
                "Divergent position not rejected"
            )
    
    def test_covariance_realism(self):
        """Test that covariance scales realistically with noise and geometry."""
        # Test 1: Higher noise -> larger covariance
        pos_low, cov_low, _ = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.05,
            cov_floor_std=0.0  # Disable floor for this test
        )
        
        pos_high, cov_high, _ = solve_uwb_position_wls(
            ranges=self.true_ranges,
            anchor_positions=self.anchors,
            range_noise_std=0.2,
            cov_floor_std=0.0
        )
        
        # Higher noise should give larger covariance
        self.assertGreater(
            np.trace(cov_high), np.trace(cov_low),
            "Covariance should increase with noise"
        )
        
        # Test 2: Covariance should scale approximately with σ²
        # (for good geometry, Cov ≈ (H^T H)^{-1} σ²)
        ratio = np.trace(cov_high) / np.trace(cov_low)
        expected_ratio = (0.2 / 0.05) ** 2  # 16
        
        # Allow some tolerance due to nonlinearity and floor
        self.assertGreater(ratio, 10, "Covariance scaling too weak")
        self.assertLess(ratio, 20, "Covariance scaling too strong")


class TestWLSIntegrationWithGating(unittest.TestCase):
    """Integration tests for WLS + chi-square gating."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.anchors = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 15.0],
            [0.0, 15.0]
        ])
        self.true_pos = np.array([10.0, 7.5])
        self.true_ranges = np.linalg.norm(
            self.anchors - self.true_pos, axis=1
        )
    
    def test_realistic_covariance_for_gating(self):
        """Test that WLS covariance is realistic enough for chi-square gating.
        
        This is a regression test for the issue where overconfident WLS
        covariance caused gating to reject too many valid measurements.
        """
        from core.fusion import chi_square_gate, innovation_covariance
        
        # Simulate a sequence of position fixes with realistic noise
        np.random.seed(42)
        n_fixes = 20
        noise_std = 0.1
        
        # Simulate EKF state covariance (reasonable after some propagation)
        P_ekf = np.diag([0.5**2, 0.5**2, 0.2**2, 0.2**2, 0.1**2])
        
        # Measurement Jacobian H for position measurement
        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        
        n_accepted = 0
        n_rejected = 0
        
        for _ in range(n_fixes):
            # Simulate noisy ranges
            noisy_ranges = self.true_ranges + np.random.randn(4) * noise_std
            
            # Solve WLS
            pos_uwb, cov_uwb, converged = solve_uwb_position_wls(
                ranges=noisy_ranges,
                anchor_positions=self.anchors,
                range_noise_std=noise_std,
                cov_floor_std=0.2  # Realistic floor
            )
            
            if not converged:
                continue
            
            # Compute innovation (assume EKF prediction is at true position)
            y = pos_uwb - self.true_pos
            
            # Innovation covariance S = H P H^T + R
            R = cov_uwb
            S = innovation_covariance(H, P_ekf, R)
            
            # Gating test
            accept = chi_square_gate(y, S, confidence=0.95)
            
            if accept:
                n_accepted += 1
            else:
                n_rejected += 1
        
        # With realistic covariance, rejection rate should be low (< 20%)
        # For 95% confidence gate, expect ~5% false rejections
        total = n_accepted + n_rejected
        rejection_rate = n_rejected / total if total > 0 else 0
        
        self.assertGreater(n_accepted, 0, "No measurements accepted")
        self.assertLess(
            rejection_rate, 0.20,
            f"Rejection rate {rejection_rate:.1%} too high - covariance likely overconfident"
        )


if __name__ == '__main__':
    unittest.main()

