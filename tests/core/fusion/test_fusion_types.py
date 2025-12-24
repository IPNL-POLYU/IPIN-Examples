"""Unit tests for core.fusion.types module.

Tests StampedMeasurement and TimeSyncModel data structures for Chapter 8
sensor fusion examples.

Author: Li-Ta Hsu
References: Chapter 8 - Sensor Fusion
"""

import unittest
import warnings

import numpy as np

from core.fusion.types import StampedMeasurement, TimeSyncModel


class TestStampedMeasurement(unittest.TestCase):
    """Test suite for StampedMeasurement dataclass."""
    
    def test_valid_scalar_measurement(self) -> None:
        """Test creation of valid scalar measurement."""
        meas = StampedMeasurement(
            t=1.234,
            sensor="uwb_range",
            z=np.array([5.67]),
            R=np.array([[0.01]]),
            meta={"anchor_id": 3}
        )
        
        self.assertEqual(meas.t, 1.234)
        self.assertEqual(meas.sensor, "uwb_range")
        np.testing.assert_array_equal(meas.z, [5.67])
        np.testing.assert_array_equal(meas.R, [[0.01]])
        self.assertEqual(meas.meta["anchor_id"], 3)
    
    def test_valid_vector_measurement(self) -> None:
        """Test creation of valid vector measurement."""
        meas = StampedMeasurement(
            t=2.5,
            sensor="imu_accel",
            z=np.array([0.1, 0.05, 9.81]),
            R=np.diag([0.01, 0.01, 0.02]),
            meta={"frame": "body"}
        )
        
        self.assertEqual(len(meas.z), 3)
        self.assertEqual(meas.R.shape, (3, 3))
        self.assertTrue(np.allclose(meas.R, meas.R.T))  # symmetric
    
    def test_empty_meta_default(self) -> None:
        """Test that meta defaults to empty dict."""
        meas = StampedMeasurement(
            t=1.0,
            sensor="test",
            z=np.array([1.0]),
            R=np.array([[1.0]])
        )
        
        self.assertEqual(meas.meta, {})
    
    def test_negative_timestamp_raises(self) -> None:
        """Test that negative timestamp raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=-1.0,
                sensor="test",
                z=np.array([1.0]),
                R=np.array([[1.0]])
            )
    
    def test_invalid_timestamp_type_raises(self) -> None:
        """Test that non-numeric timestamp raises TypeError."""
        with self.assertRaises(TypeError):
            StampedMeasurement(
                t="invalid",  # type: ignore
                sensor="test",
                z=np.array([1.0]),
                R=np.array([[1.0]])
            )
    
    def test_empty_sensor_name_raises(self) -> None:
        """Test that empty sensor name raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="",
                z=np.array([1.0]),
                R=np.array([[1.0]])
            )
    
    def test_measurement_not_1d_raises(self) -> None:
        """Test that 2D measurement array raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="test",
                z=np.array([[1.0, 2.0]]),  # 2D
                R=np.array([[1.0]])
            )
    
    def test_covariance_not_2d_raises(self) -> None:
        """Test that 1D covariance array raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="test",
                z=np.array([1.0]),
                R=np.array([1.0])  # 1D
            )
    
    def test_covariance_dimension_mismatch_raises(self) -> None:
        """Test that mismatched z and R dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="test",
                z=np.array([1.0, 2.0]),  # 2D measurement
                R=np.array([[1.0]])       # 1x1 covariance
            )
    
    def test_asymmetric_covariance_raises(self) -> None:
        """Test that asymmetric covariance raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="test",
                z=np.array([1.0, 2.0]),
                R=np.array([[1.0, 0.5], [0.3, 1.0]])  # not symmetric
            )
    
    def test_negative_definite_covariance_raises(self) -> None:
        """Test that negative definite covariance raises ValueError."""
        with self.assertRaises(ValueError):
            StampedMeasurement(
                t=1.0,
                sensor="test",
                z=np.array([1.0, 2.0]),
                R=np.array([[-1.0, 0.0], [0.0, 1.0]])  # negative eigenvalue
            )
    
    def test_positive_semidefinite_covariance_allowed(self) -> None:
        """Test that positive semi-definite (singular) covariance is allowed."""
        # This can occur in practice (e.g., perfectly correlated measurements)
        meas = StampedMeasurement(
            t=1.0,
            sensor="test",
            z=np.array([1.0, 2.0]),
            R=np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, positive semi-definite
        )
        
        # Should not raise
        self.assertEqual(meas.z.shape, (2,))


class TestTimeSyncModel(unittest.TestCase):
    """Test suite for TimeSyncModel dataclass."""
    
    def test_identity_transform(self) -> None:
        """Test that default parameters give identity transform."""
        sync = TimeSyncModel()
        
        self.assertEqual(sync.offset, 0.0)
        self.assertEqual(sync.drift, 0.0)
        self.assertTrue(sync.is_synchronized())
        
        # Identity: t_fusion = t_sensor
        self.assertEqual(sync.to_fusion_time(10.0), 10.0)
        self.assertEqual(sync.to_fusion_time(0.0), 0.0)
        self.assertEqual(sync.to_fusion_time(100.5), 100.5)
    
    def test_offset_only(self) -> None:
        """Test time synchronization with offset only (no drift)."""
        sync = TimeSyncModel(offset=0.5, drift=0.0)
        
        # t_fusion = t_sensor + 0.5
        self.assertEqual(sync.to_fusion_time(10.0), 10.5)
        self.assertEqual(sync.to_fusion_time(0.0), 0.5)
        self.assertEqual(sync.to_fusion_time(100.0), 100.5)
    
    def test_drift_only(self) -> None:
        """Test time synchronization with drift only (no offset)."""
        sync = TimeSyncModel(offset=0.0, drift=0.001)
        
        # t_fusion = 1.001 * t_sensor
        self.assertAlmostEqual(sync.to_fusion_time(10.0), 10.01, places=10)
        self.assertAlmostEqual(sync.to_fusion_time(100.0), 100.1, places=10)
        self.assertAlmostEqual(sync.to_fusion_time(1000.0), 1001.0, places=10)
    
    def test_offset_and_drift(self) -> None:
        """Test time synchronization with both offset and drift."""
        sync = TimeSyncModel(offset=0.2, drift=0.001)
        
        # t_fusion = 1.001 * t_sensor + 0.2
        self.assertAlmostEqual(sync.to_fusion_time(0.0), 0.2, places=10)
        self.assertAlmostEqual(sync.to_fusion_time(100.0), 100.3, places=10)
    
    def test_negative_offset(self) -> None:
        """Test that negative offset (sensor behind) works correctly."""
        sync = TimeSyncModel(offset=-0.5, drift=0.0)
        
        # t_fusion = t_sensor - 0.5
        self.assertEqual(sync.to_fusion_time(10.0), 9.5)
        self.assertEqual(sync.to_fusion_time(1.0), 0.5)
    
    def test_inverse_transform(self) -> None:
        """Test that to_sensor_time inverts to_fusion_time."""
        sync = TimeSyncModel(offset=0.5, drift=0.001)
        
        t_sensor = 100.0
        t_fusion = sync.to_fusion_time(t_sensor)
        t_sensor_recovered = sync.to_sensor_time(t_fusion)
        
        self.assertAlmostEqual(t_sensor_recovered, t_sensor, places=10)
    
    def test_inverse_transform_identity(self) -> None:
        """Test inverse transform for identity (no offset, no drift)."""
        sync = TimeSyncModel()
        
        for t in [0.0, 10.0, 100.5, 1234.567]:
            self.assertEqual(sync.to_sensor_time(t), t)
    
    def test_is_synchronized_tolerance(self) -> None:
        """Test is_synchronized with custom tolerance."""
        # Within default tolerance (1e-6)
        sync = TimeSyncModel(offset=1e-7, drift=1e-7)
        self.assertTrue(sync.is_synchronized())
        
        # Outside default tolerance
        sync = TimeSyncModel(offset=1e-5, drift=0.0)
        self.assertFalse(sync.is_synchronized())
        
        # But within larger tolerance
        self.assertTrue(sync.is_synchronized(tolerance=1e-4))
    
    def test_large_drift_warning(self) -> None:
        """Test that unrealistically large drift triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            sync = TimeSyncModel(offset=0.0, drift=0.02)  # 20,000 ppm
            
            # Check that a warning was issued
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("unusually large", str(w[0].message).lower())
    
    def test_realistic_drift_no_warning(self) -> None:
        """Test that realistic drift (< 100 ppm) does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            sync = TimeSyncModel(offset=0.1, drift=0.00005)  # 50 ppm
            
            # No warning should be issued
            self.assertEqual(len(w), 0)
    
    def test_invalid_offset_type_raises(self) -> None:
        """Test that non-numeric offset raises TypeError."""
        with self.assertRaises(TypeError):
            TimeSyncModel(offset="invalid", drift=0.0)  # type: ignore
    
    def test_invalid_drift_type_raises(self) -> None:
        """Test that non-numeric drift raises TypeError."""
        with self.assertRaises(TypeError):
            TimeSyncModel(offset=0.0, drift="invalid")  # type: ignore
    
    def test_round_trip_multiple_times(self) -> None:
        """Test that multiple round trips preserve accuracy."""
        sync = TimeSyncModel(offset=1.234, drift=-0.0001)
        
        t_original = 500.0
        
        # Forward and back 10 times
        t = t_original
        for _ in range(10):
            t_fus = sync.to_fusion_time(t)
            t = sync.to_sensor_time(t_fus)
        
        self.assertAlmostEqual(t, t_original, places=8)


if __name__ == "__main__":
    unittest.main()

