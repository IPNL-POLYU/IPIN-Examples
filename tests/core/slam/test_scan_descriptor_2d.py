"""Unit tests for scan_descriptor_2d module.

Tests cover:
- Descriptor computation (range histogram)
- Descriptor similarity metrics
- Batch computation
- Edge cases and input validation

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np

from core.slam.scan_descriptor_2d import (
    compute_scan_descriptor,
    compute_descriptor_similarity,
    batch_compute_descriptors,
)


class TestComputeScanDescriptor(unittest.TestCase):
    """Test compute_scan_descriptor function."""
    
    def test_basic_descriptor_shape(self):
        """Test descriptor has correct shape."""
        scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        
        self.assertEqual(desc.shape, (8,))
    
    def test_descriptor_is_normalized(self):
        """Test descriptor sums to 1."""
        scan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        
        self.assertAlmostEqual(np.sum(desc), 1.0)
    
    def test_empty_scan_returns_zero_vector(self):
        """Test empty scan returns zero descriptor."""
        scan = np.array([]).reshape(0, 2)
        desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        
        self.assertEqual(desc.shape, (8,))
        self.assertEqual(np.sum(desc), 0.0)
    
    def test_single_point_scan(self):
        """Test scan with single point."""
        scan = np.array([[1.0, 0.0]])
        desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        
        self.assertAlmostEqual(np.sum(desc), 1.0)
        self.assertEqual(np.count_nonzero(desc), 1)  # Only one bin should be non-zero
    
    def test_rotation_invariance(self):
        """Test descriptor is rotation-invariant."""
        # Original scan
        scan1 = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        desc1 = compute_scan_descriptor(scan1, n_bins=8, max_range=5.0)
        
        # Rotated by 90 degrees
        scan2 = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        desc2 = compute_scan_descriptor(scan2, n_bins=8, max_range=5.0)
        
        # Descriptors should be identical (rotation doesn't change ranges)
        np.testing.assert_allclose(desc1, desc2, atol=1e-10)
    
    def test_different_scans_different_descriptors(self):
        """Test different scans produce different descriptors."""
        scan1 = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # All at range 1
        scan2 = np.array([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])  # All at range 3
        
        desc1 = compute_scan_descriptor(scan1, n_bins=8, max_range=5.0)
        desc2 = compute_scan_descriptor(scan2, n_bins=8, max_range=5.0)
        
        # Descriptors should be different
        self.assertGreater(np.linalg.norm(desc1 - desc2), 0.1)
    
    def test_points_beyond_max_range(self):
        """Test points beyond max_range are clipped to last bin."""
        # All points beyond max_range
        scan = np.array([[10.0, 0.0], [15.0, 0.0], [20.0, 0.0]])
        desc = compute_scan_descriptor(scan, n_bins=8, max_range=5.0)
        
        # All points should be clipped and placed in the last bin
        self.assertAlmostEqual(desc[-1], 1.0)
        self.assertAlmostEqual(np.sum(desc[:-1]), 0.0)
    
    def test_invalid_scan_shape_raises_error(self):
        """Test invalid scan shape raises ValueError."""
        invalid_scan = np.array([1.0, 2.0, 3.0])  # 1D array
        
        with self.assertRaises(ValueError):
            compute_scan_descriptor(invalid_scan)
    
    def test_invalid_n_bins_raises_error(self):
        """Test invalid n_bins raises ValueError."""
        scan = np.array([[1.0, 0.0]])
        
        with self.assertRaises(ValueError):
            compute_scan_descriptor(scan, n_bins=0)
    
    def test_invalid_max_range_raises_error(self):
        """Test invalid max_range raises ValueError."""
        scan = np.array([[1.0, 0.0]])
        
        with self.assertRaises(ValueError):
            compute_scan_descriptor(scan, max_range=-1.0)


class TestComputeDescriptorSimilarity(unittest.TestCase):
    """Test compute_descriptor_similarity function."""
    
    def test_identical_descriptors_cosine(self):
        """Test identical descriptors have similarity 1.0 (cosine)."""
        desc1 = np.array([0.5, 0.3, 0.2])
        desc2 = np.array([0.5, 0.3, 0.2])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        self.assertAlmostEqual(sim, 1.0)
    
    def test_orthogonal_descriptors_cosine(self):
        """Test orthogonal descriptors have similarity 0.0 (cosine)."""
        desc1 = np.array([1.0, 0.0, 0.0])
        desc2 = np.array([0.0, 1.0, 0.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_opposite_descriptors_cosine(self):
        """Test opposite descriptors have similarity -1.0 (cosine)."""
        desc1 = np.array([1.0, 0.0])
        desc2 = np.array([-1.0, 0.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        self.assertAlmostEqual(sim, -1.0, places=5)
    
    def test_zero_descriptor_cosine(self):
        """Test zero descriptors return 0.0 similarity (cosine)."""
        desc1 = np.zeros(8)
        desc2 = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        self.assertEqual(sim, 0.0)
    
    def test_correlation_method(self):
        """Test correlation similarity method."""
        desc1 = np.array([1.0, 2.0, 3.0])
        desc2 = np.array([1.0, 2.0, 3.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="correlation")
        
        # Perfect correlation should be 1.0
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_l2_method(self):
        """Test L2 distance method."""
        desc1 = np.array([1.0, 0.0])
        desc2 = np.array([1.0, 0.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="l2")
        
        # Identical descriptors: L2 distance = 0, so -distance = 0
        self.assertEqual(sim, 0.0)
    
    def test_l2_method_different_descriptors(self):
        """Test L2 distance for different descriptors."""
        desc1 = np.array([1.0, 0.0])
        desc2 = np.array([0.0, 1.0])
        
        sim = compute_descriptor_similarity(desc1, desc2, method="l2")
        
        # L2 distance = sqrt(2), so -distance = -sqrt(2) ≈ -1.414
        expected = -np.sqrt(2)
        self.assertAlmostEqual(sim, expected, places=5)
    
    def test_invalid_method_raises_error(self):
        """Test invalid similarity method raises ValueError."""
        desc1 = np.array([1.0, 0.0])
        desc2 = np.array([0.0, 1.0])
        
        with self.assertRaises(ValueError):
            compute_descriptor_similarity(desc1, desc2, method="invalid")
    
    def test_mismatched_shapes_raises_error(self):
        """Test mismatched descriptor shapes raise ValueError."""
        desc1 = np.array([1.0, 0.0])
        desc2 = np.array([0.0, 1.0, 0.0])
        
        with self.assertRaises(ValueError):
            compute_descriptor_similarity(desc1, desc2)


class TestBatchComputeDescriptors(unittest.TestCase):
    """Test batch_compute_descriptors function."""
    
    def test_batch_computation_shape(self):
        """Test batch computation returns correct shape."""
        scans = [
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            np.array([[3.0, 0.0], [4.0, 0.0]]),
            np.array([[5.0, 0.0], [6.0, 0.0]]),
        ]
        
        descriptors = batch_compute_descriptors(scans, n_bins=8, max_range=10.0)
        
        self.assertEqual(descriptors.shape, (3, 8))
    
    def test_batch_computation_normalized(self):
        """Test all batch descriptors are normalized."""
        scans = [
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            np.array([[3.0, 0.0], [4.0, 0.0]]),
        ]
        
        descriptors = batch_compute_descriptors(scans, n_bins=8, max_range=10.0)
        
        for desc in descriptors:
            self.assertAlmostEqual(np.sum(desc), 1.0)
    
    def test_batch_with_empty_scan(self):
        """Test batch computation handles empty scans."""
        scans = [
            np.array([[1.0, 0.0]]),
            np.array([]).reshape(0, 2),  # Empty scan
            np.array([[3.0, 0.0]]),
        ]
        
        descriptors = batch_compute_descriptors(scans, n_bins=8, max_range=10.0)
        
        self.assertEqual(descriptors.shape, (3, 8))
        self.assertEqual(np.sum(descriptors[1]), 0.0)  # Empty scan descriptor


class TestDescriptorIntegration(unittest.TestCase):
    """Integration tests for descriptor-based matching."""
    
    def test_similar_scans_high_similarity(self):
        """Test similar scans produce high similarity."""
        # Two similar scans (same structure, slightly different)
        scan1 = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        scan2 = np.array([[1.1, 0.1], [2.1, 0.1], [2.9, 0.1]])
        
        desc1 = compute_scan_descriptor(scan1, n_bins=16, max_range=10.0)
        desc2 = compute_scan_descriptor(scan2, n_bins=16, max_range=10.0)
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        # Should have high similarity
        self.assertGreater(sim, 0.8)
    
    def test_different_scans_low_similarity(self):
        """Test very different scans produce low similarity."""
        # Scan at close range
        scan1 = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]])
        
        # Scan at far range
        scan2 = np.array([[10.0, 0.0], [10.0, 0.1], [10.0, -0.1]])
        
        desc1 = compute_scan_descriptor(scan1, n_bins=16, max_range=15.0)
        desc2 = compute_scan_descriptor(scan2, n_bins=16, max_range=15.0)
        
        sim = compute_descriptor_similarity(desc1, desc2, method="cosine")
        
        # Should have low similarity
        self.assertLess(sim, 0.5)


if __name__ == "__main__":
    unittest.main()
