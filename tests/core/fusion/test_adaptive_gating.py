"""Unit tests for adaptive gating mechanisms.

Tests the AdaptiveGatingManager for robustness features:
- Consecutive reject tracking and covariance inflation
- NIS-based consistency monitoring and R scaling
- Adaptive parameter adjustment

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest

import numpy as np

from core.fusion.adaptive import AdaptiveGatingManager


class TestAdaptiveGatingManager(unittest.TestCase):
    """Test adaptive gating manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mgr = AdaptiveGatingManager(
            dof=1,
            consecutive_reject_limit=3,
            nis_window_size=10,
            nis_scale_threshold=1.5,
            P_inflation_factor=2.0,
            R_scale_factor=1.2,
        )
    
    def test_initial_state(self):
        """Test initial state of adaptive manager."""
        self.assertEqual(self.mgr.consecutive_rejects, 0)
        self.assertEqual(len(self.mgr.nis_history), 0)
        self.assertEqual(self.mgr.current_R_scale, 1.0)
        self.assertEqual(self.mgr.total_measurements, 0)
        self.assertEqual(self.mgr.total_accepts, 0)
        self.assertEqual(self.mgr.total_rejects, 0)
        self.assertEqual(self.mgr.total_adaptations, 0)
    
    def test_accept_resets_consecutive_rejects(self):
        """Test that accepting a measurement resets consecutive reject counter."""
        # Reject twice
        self.mgr.update(nis_value=10.0, gated_accept=False)
        self.mgr.update(nis_value=10.0, gated_accept=False)
        self.assertEqual(self.mgr.consecutive_rejects, 2)
        
        # Accept once
        accept, action = self.mgr.update(nis_value=1.0, gated_accept=True)
        self.assertEqual(self.mgr.consecutive_rejects, 0)
        self.assertTrue(accept)
        self.assertIsNone(action)
    
    def test_consecutive_reject_triggers_inflation(self):
        """Test that reaching consecutive reject limit triggers P inflation."""
        # Reject 3 times (reaches limit)
        for i in range(3):
            accept, action = self.mgr.update(nis_value=10.0, gated_accept=False)
            
            if i < 2:
                # First two rejects: no action yet
                self.assertFalse(accept)
                self.assertIsNone(action)
            else:
                # Third reject: triggers inflation and forces accept
                self.assertTrue(accept)  # Forced accept after adaptation
                self.assertEqual(action, 'inflate_P')
                self.assertEqual(self.mgr.total_adaptations, 1)
        
        # Counter should reset after adaptation
        self.assertEqual(self.mgr.consecutive_rejects, 0)
    
    def test_covariance_inflation(self):
        """Test covariance inflation application."""
        P = np.diag([1.0, 2.0, 3.0])
        P_inflated = self.mgr.inflate_covariance(P)
        
        # Should be scaled by inflation factor (2.0)
        np.testing.assert_allclose(P_inflated, 2.0 * P)
    
    def test_nis_monitoring_scales_R_up(self):
        """Test that high NIS triggers R scaling."""
        # Feed high NIS values (DOF=1, so expected=1, but we give 3)
        for _ in range(10):  # Fill window
            self.mgr.update(nis_value=3.0, gated_accept=True)
        
        # Mean NIS = 3.0, expected = 1.0
        # Ratio = 3.0 > threshold (1.5), so R should scale up
        self.assertGreater(self.mgr.current_R_scale, 1.0)
    
    def test_nis_monitoring_scales_R_down(self):
        """Test that low NIS scales R back down."""
        # First increase R scale artificially
        self.mgr.current_R_scale = 2.0
        
        # Feed low NIS values (below 0.7 * expected)
        for _ in range(10):  # Fill window
            self.mgr.update(nis_value=0.5, gated_accept=True)
        
        # Mean NIS = 0.5, expected = 1.0
        # Ratio = 0.5 < 0.7, so R should scale down
        self.assertLess(self.mgr.current_R_scale, 2.0)
    
    def test_R_scale_limits(self):
        """Test that R scale stays within bounds."""
        # Try to push R scale to max
        for _ in range(50):  # Many iterations
            self.mgr.update(nis_value=10.0, gated_accept=True)
        
        # Should be capped at max_R_scale (5.0)
        self.assertLessEqual(self.mgr.current_R_scale, 5.0)
        
        # Try to push R scale to min
        self.mgr.current_R_scale = 1.5  # Start above 1
        for _ in range(50):
            self.mgr.update(nis_value=0.1, gated_accept=True)
        
        # Should be capped at min_R_scale (1.0)
        self.assertGreaterEqual(self.mgr.current_R_scale, 1.0)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        # Process some measurements
        self.mgr.update(nis_value=1.0, gated_accept=True)
        self.mgr.update(nis_value=2.0, gated_accept=False)
        self.mgr.update(nis_value=1.5, gated_accept=True)
        
        stats = self.mgr.get_stats()
        
        self.assertEqual(stats['total_measurements'], 3)
        self.assertEqual(stats['total_accepts'], 2)
        self.assertEqual(stats['total_rejects'], 1)
        self.assertAlmostEqual(stats['acceptance_rate'], 2.0 / 3.0)
        self.assertEqual(stats['expected_nis'], 1)
        self.assertAlmostEqual(stats['mean_nis'], (1.0 + 2.0 + 1.5) / 3.0)
    
    def test_reset(self):
        """Test reset functionality."""
        # Process some measurements
        self.mgr.update(nis_value=1.0, gated_accept=True)
        self.mgr.update(nis_value=2.0, gated_accept=False)
        self.mgr.current_R_scale = 2.0
        
        # Reset
        self.mgr.reset()
        
        # Should be back to initial state
        self.assertEqual(self.mgr.consecutive_rejects, 0)
        self.assertEqual(len(self.mgr.nis_history), 0)
        self.assertEqual(self.mgr.current_R_scale, 1.0)
        self.assertEqual(self.mgr.total_measurements, 0)
        self.assertEqual(self.mgr.total_accepts, 0)
        self.assertEqual(self.mgr.total_rejects, 0)


class TestAdaptiveGatingIntegration(unittest.TestCase):
    """Integration tests for adaptive gating in fusion scenarios."""
    
    def test_prevents_filter_starvation(self):
        """Test that adaptive gating prevents filter from starving on consecutive rejects."""
        mgr = AdaptiveGatingManager(
            dof=2,
            consecutive_reject_limit=5,
            nis_window_size=20,
        )
        
        # Simulate scenario where all measurements are initially rejected
        forced_accepts = 0
        total_rejects = 0
        
        for i in range(20):
            accept, action = mgr.update(nis_value=100.0, gated_accept=False)
            
            if accept:  # Forced accept after adaptation
                forced_accepts += 1
            else:
                total_rejects += 1
        
        # Should have triggered some forced accepts to prevent starvation
        self.assertGreater(forced_accepts, 0)
        self.assertGreater(mgr.total_adaptations, 0)
    
    def test_restores_consistency_with_overconfident_filter(self):
        """Test that NIS monitoring restores consistency when filter is overconfident."""
        mgr = AdaptiveGatingManager(
            dof=1,
            nis_window_size=10,
            nis_scale_threshold=1.5,
        )
        
        # Simulate overconfident filter (high NIS values)
        for _ in range(15):
            mgr.update(nis_value=5.0, gated_accept=True)  # Mean NIS = 5x expected
        
        # R scale should have increased to compensate
        self.assertGreater(mgr.current_R_scale, 1.0)
        
        # Verify stats show the issue
        stats = mgr.get_stats()
        self.assertGreater(stats['mean_nis'], stats['expected_nis'] * 1.5)


if __name__ == '__main__':
    unittest.main()

