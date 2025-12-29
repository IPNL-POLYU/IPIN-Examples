"""Unit tests for TC batch update mode.

Tests that batch update mode correctly processes multiple UWB ranges
at the same timestamp, matching the book's "m+n measurements" description.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
from pathlib import Path

import numpy as np

from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset, run_tc_fusion


class TestBatchUpdate(unittest.TestCase):
    """Test batch update mode for TC fusion."""
    
    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests."""
        dataset_path = "data/sim/ch8_fusion_2d_imu_uwb"
        if not Path(dataset_path).exists():
            raise unittest.SkipTest(f"Dataset not found: {dataset_path}")
        
        cls.dataset = load_fusion_dataset(dataset_path)
    
    def test_sequential_mode_runs(self):
        """Test that sequential mode runs without errors."""
        history = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=False,
            verbose=False
        )
        
        self.assertGreater(len(history['t']), 0)
        self.assertGreater(history['n_uwb_accepted'], 0)
        self.assertEqual(history['n_uwb_rejected'], 0)  # No gating
    
    def test_batch_mode_runs(self):
        """Test that batch mode runs without errors."""
        history = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=True,
            verbose=False
        )
        
        self.assertGreater(len(history['t']), 0)
        self.assertGreater(history['n_uwb_accepted'], 0)
        self.assertEqual(history['n_uwb_rejected'], 0)  # No gating
    
    def test_batch_vs_sequential_update_count(self):
        """Test that batch mode has fewer updates than sequential (epochs vs ranges)."""
        history_seq = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=False,
            verbose=False
        )
        
        history_batch = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=True,
            verbose=False
        )
        
        # Batch mode should have ~4x fewer updates (4 anchors → 1 batch per epoch)
        self.assertLess(
            history_batch['n_uwb_accepted'],
            history_seq['n_uwb_accepted'] / 2
        )
    
    def test_batch_accuracy_comparable(self):
        """Test that batch mode achieves similar accuracy to sequential mode."""
        history_seq = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=False,
            verbose=False
        )
        
        history_batch = run_tc_fusion(
            self.dataset,
            use_gating=False,
            batch_update=True,
            verbose=False
        )
        
        # Compute RMSE for both
        truth = self.dataset['truth']
        
        # Interpolate truth to estimated timestamps
        def compute_rmse(history):
            p_true_interp = np.column_stack([
                np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
                np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
            ])
            errors = history['x_est'][:, :2] - p_true_interp
            return np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        rmse_seq = compute_rmse(history_seq)
        rmse_batch = compute_rmse(history_batch)
        
        # Both should achieve sub-meter accuracy
        self.assertLess(rmse_seq, 1.0)
        self.assertLess(rmse_batch, 1.0)
        
        # They should be within 50% of each other
        ratio = max(rmse_seq, rmse_batch) / min(rmse_seq, rmse_batch)
        self.assertLess(ratio, 1.5)
    
    def test_batch_with_gating(self):
        """Test that batch mode works with adaptive gating."""
        history = run_tc_fusion(
            self.dataset,
            use_gating=True,
            batch_update=True,
            verbose=False
        )
        
        self.assertGreater(len(history['t']), 0)
        self.assertGreater(history['n_uwb_accepted'], 0)
        
        # Should have some rejections with gating
        total_measurements = history['n_uwb_accepted'] + history['n_uwb_rejected']
        acceptance_rate = history['n_uwb_accepted'] / total_measurements
        
        # Acceptance rate should be reasonable (> 50%)
        self.assertGreater(acceptance_rate, 0.5)
    
    def test_batch_improves_gating_performance(self):
        """Test that batch mode improves accuracy when using gating.
        
        This is the key advantage of batch mode: applying chi-square test
        to the full measurement vector is more statistically sound.
        """
        history_seq = run_tc_fusion(
            self.dataset,
            use_gating=True,
            batch_update=False,
            verbose=False
        )
        
        history_batch = run_tc_fusion(
            self.dataset,
            use_gating=True,
            batch_update=True,
            verbose=False
        )
        
        # Compute RMSE
        truth = self.dataset['truth']
        
        def compute_rmse(history):
            p_true_interp = np.column_stack([
                np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
                np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
            ])
            errors = history['x_est'][:, :2] - p_true_interp
            return np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        rmse_seq = compute_rmse(history_seq)
        rmse_batch = compute_rmse(history_batch)
        
        # With gating, batch mode should be better (or at least not significantly worse)
        # Allow batch to be up to 20% worse in worst case, but it should usually be better
        self.assertLess(rmse_batch, rmse_seq * 1.2)


class TestBatchUpdateMeasurementConstruction(unittest.TestCase):
    """Test that batch measurements are constructed correctly."""
    
    def test_batch_measurement_structure(self):
        """Test that batch measurements have correct structure."""
        # Create dummy data
        uwb_t = np.array([1.0, 2.0])
        uwb_ranges = np.array([
            [5.0, np.nan, 7.0, 6.0],  # Epoch 1: anchor 1 dropout
            [5.1, 7.2, np.nan, 6.1],  # Epoch 2: anchor 2 dropout
        ])
        
        # Simulate what batch mode does
        for i in range(len(uwb_t)):
            ranges_at_epoch = uwb_ranges[i, :]
            valid_mask = ~np.isnan(ranges_at_epoch)
            
            if np.any(valid_mask):
                valid_ranges = ranges_at_epoch[valid_mask]
                valid_anchors = np.where(valid_mask)[0]
                
                if i == 0:
                    # Epoch 1: anchors 0, 2, 3 valid
                    self.assertEqual(len(valid_ranges), 3)
                    np.testing.assert_array_equal(valid_anchors, [0, 2, 3])
                    np.testing.assert_array_almost_equal(valid_ranges, [5.0, 7.0, 6.0])
                
                elif i == 1:
                    # Epoch 2: anchors 0, 1, 3 valid
                    self.assertEqual(len(valid_ranges), 3)
                    np.testing.assert_array_equal(valid_anchors, [0, 1, 3])
                    np.testing.assert_array_almost_equal(valid_ranges, [5.1, 7.2, 6.1])


if __name__ == '__main__':
    unittest.main()

