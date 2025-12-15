"""
Unit tests for core/sensors/types.py data structures.

Tests cover:
    - Dataclass instantiation and immutability
    - Shape validation for all sensor series
    - Quaternion normalization warnings
    - Metadata handling
    - Edge cases and error conditions

All tests use pytest and follow the naming convention:
    test_<class_name>_<aspect>

Run with: pytest tests/test_sensors_types.py -v
"""

import unittest
import warnings
import numpy as np
import pytest

from core.sensors.types import (
    ImuSeries,
    WheelSpeedSeries,
    MagnetometerSeries,
    BarometerSeries,
    NavStateQPVP,
    NavStateQPVPBias,
)


class TestImuSeries(unittest.TestCase):
    """Test suite for ImuSeries dataclass."""

    def test_imu_series_valid_construction(self) -> None:
        """Test that ImuSeries can be constructed with valid data."""
        n = 100
        t = np.linspace(0, 1, n)
        accel = np.random.randn(n, 3)
        gyro = np.random.randn(n, 3)
        meta = {"sample_rate_hz": 100.0, "sensor_id": "imu_01"}

        imu = ImuSeries(t=t, accel=accel, gyro=gyro, meta=meta)

        assert imu.t.shape == (n,)
        assert imu.accel.shape == (n, 3)
        assert imu.gyro.shape == (n, 3)
        assert imu.meta == meta

    def test_imu_series_immutability(self) -> None:
        """Test that ImuSeries is frozen (immutable)."""
        n = 10
        imu = ImuSeries(
            t=np.linspace(0, 1, n),
            accel=np.zeros((n, 3)),
            gyro=np.zeros((n, 3)),
            meta={},
        )

        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
            imu.t = np.ones(n)

    def test_imu_series_invalid_t_shape(self) -> None:
        """Test that ImuSeries rejects non-1D time array."""
        n = 10
        with pytest.raises(ValueError, match="must be 1D array"):
            ImuSeries(
                t=np.zeros((n, 1)),  # Wrong: should be (n,)
                accel=np.zeros((n, 3)),
                gyro=np.zeros((n, 3)),
                meta={},
            )

    def test_imu_series_invalid_accel_shape(self) -> None:
        """Test that ImuSeries rejects wrong accel shape."""
        n = 10
        with pytest.raises(ValueError, match="accel must have shape"):
            ImuSeries(
                t=np.linspace(0, 1, n),
                accel=np.zeros((n, 2)),  # Wrong: should be (n, 3)
                gyro=np.zeros((n, 3)),
                meta={},
            )

    def test_imu_series_invalid_gyro_shape(self) -> None:
        """Test that ImuSeries rejects wrong gyro shape."""
        n = 10
        with pytest.raises(ValueError, match="gyro must have shape"):
            ImuSeries(
                t=np.linspace(0, 1, n),
                accel=np.zeros((n, 3)),
                gyro=np.zeros((n, 4)),  # Wrong: should be (n, 3)
                meta={},
            )

    def test_imu_series_mismatched_lengths(self) -> None:
        """Test that ImuSeries rejects mismatched array lengths."""
        with pytest.raises(ValueError):
            ImuSeries(
                t=np.linspace(0, 1, 10),
                accel=np.zeros((15, 3)),  # Mismatch: t has 10 samples
                gyro=np.zeros((10, 3)),
                meta={},
            )

    def test_imu_series_empty_arrays(self) -> None:
        """Test ImuSeries with zero samples (edge case)."""
        imu = ImuSeries(
            t=np.array([]),
            accel=np.zeros((0, 3)),
            gyro=np.zeros((0, 3)),
            meta={"note": "empty"},
        )
        assert imu.t.shape == (0,)
        assert imu.accel.shape == (0, 3)
        assert imu.gyro.shape == (0, 3)


class TestWheelSpeedSeries(unittest.TestCase):
    """Test suite for WheelSpeedSeries dataclass."""

    def test_wheel_speed_series_valid_construction(self) -> None:
        """Test valid WheelSpeedSeries construction."""
        n = 50
        t = np.linspace(0, 5, n)
        v_s = np.random.randn(n, 3)
        meta = {"lever_arm_b": np.array([0.5, 0.0, -0.1]), "frame": "speed"}

        wheel = WheelSpeedSeries(t=t, v_s=v_s, meta=meta)

        assert wheel.t.shape == (n,)
        assert wheel.v_s.shape == (n, 3)
        assert "lever_arm_b" in wheel.meta

    def test_wheel_speed_series_immutability(self) -> None:
        """Test that WheelSpeedSeries is frozen."""
        n = 10
        wheel = WheelSpeedSeries(
            t=np.linspace(0, 1, n), v_s=np.zeros((n, 3)), meta={}
        )

        with pytest.raises(Exception):
            wheel.v_s = np.ones((n, 3))

    def test_wheel_speed_series_invalid_v_s_shape(self) -> None:
        """Test WheelSpeedSeries rejects wrong v_s shape."""
        n = 10
        with pytest.raises(ValueError, match="v_s must have shape"):
            WheelSpeedSeries(
                t=np.linspace(0, 1, n),
                v_s=np.zeros((n, 2)),  # Wrong: should be (n, 3)
                meta={},
            )


class TestMagnetometerSeries(unittest.TestCase):
    """Test suite for MagnetometerSeries dataclass."""

    def test_magnetometer_series_valid_construction(self) -> None:
        """Test valid MagnetometerSeries construction."""
        n = 30
        t = np.linspace(0, 3, n)
        mag = np.random.randn(n, 3) * 50  # Typical μT values
        meta = {"frame": "body", "units": "uT"}

        mag_series = MagnetometerSeries(t=t, mag=mag, meta=meta)

        assert mag_series.t.shape == (n,)
        assert mag_series.mag.shape == (n, 3)
        assert mag_series.meta["units"] == "uT"

    def test_magnetometer_series_invalid_mag_shape(self) -> None:
        """Test MagnetometerSeries rejects wrong mag shape."""
        n = 10
        with pytest.raises(ValueError, match="mag must have shape"):
            MagnetometerSeries(
                t=np.linspace(0, 1, n),
                mag=np.zeros((n, 4)),  # Wrong: should be (n, 3)
                meta={},
            )


class TestBarometerSeries(unittest.TestCase):
    """Test suite for BarometerSeries dataclass."""

    def test_barometer_series_valid_construction(self) -> None:
        """Test valid BarometerSeries construction."""
        n = 20
        t = np.linspace(0, 2, n)
        pressure = 101325.0 + np.random.randn(n) * 10  # Near sea level, Pa
        meta = {"units": "Pa", "p0": 101325.0, "T": 288.15}

        baro = BarometerSeries(t=t, pressure=pressure, meta=meta)

        assert baro.t.shape == (n,)
        assert baro.pressure.shape == (n,)
        assert baro.meta["units"] == "Pa"

    def test_barometer_series_missing_units(self) -> None:
        """Test that BarometerSeries enforces units field in meta."""
        n = 10
        with pytest.raises(ValueError, match="must include 'units' field"):
            BarometerSeries(
                t=np.linspace(0, 1, n),
                pressure=np.ones(n) * 101325,
                meta={},  # Missing 'units'
            )

    def test_barometer_series_invalid_pressure_shape(self) -> None:
        """Test BarometerSeries rejects wrong pressure shape."""
        n = 10
        with pytest.raises(ValueError, match="pressure must have shape"):
            BarometerSeries(
                t=np.linspace(0, 1, n),
                pressure=np.zeros((n, 1)),  # Wrong: should be (n,)
                meta={"units": "Pa"},
            )


class TestNavStateQPVP(unittest.TestCase):
    """Test suite for NavStateQPVP (minimal navigation state)."""

    def test_nav_state_qpvp_valid_construction(self) -> None:
        """Test valid NavStateQPVP construction with identity quaternion."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (scalar-first)
        v = np.zeros(3)
        p = np.zeros(3)

        state = NavStateQPVP(q=q, v=v, p=p)

        assert state.q.shape == (4,)
        assert state.v.shape == (3,)
        assert state.p.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(state.q), 1.0)

    def test_nav_state_qpvp_mutability(self) -> None:
        """Test that NavStateQPVP is mutable (not frozen)."""
        state = NavStateQPVP(
            q=np.array([1.0, 0.0, 0.0, 0.0]), v=np.zeros(3), p=np.zeros(3)
        )

        # Should allow in-place modification
        state.v = np.array([1.0, 2.0, 3.0])
        state.p = np.array([10.0, 20.0, 30.0])

        np.testing.assert_array_equal(state.v, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(state.p, [10.0, 20.0, 30.0])

    def test_nav_state_qpvp_non_unit_quaternion_warning(self) -> None:
        """Test that non-unit quaternion triggers a warning."""
        q_bad = np.array([0.5, 0.0, 0.0, 0.0])  # Not normalized

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = NavStateQPVP(q=q_bad, v=np.zeros(3), p=np.zeros(3))

            assert len(w) == 1
            assert "non-unit quaternion" in str(w[0].message).lower()

    def test_nav_state_qpvp_invalid_q_shape(self) -> None:
        """Test NavStateQPVP rejects wrong quaternion shape."""
        with pytest.raises(ValueError, match="q must have shape"):
            NavStateQPVP(q=np.array([1.0, 0.0, 0.0]), v=np.zeros(3), p=np.zeros(3))

    def test_nav_state_qpvp_invalid_v_shape(self) -> None:
        """Test NavStateQPVP rejects wrong velocity shape."""
        with pytest.raises(ValueError, match="v must have shape"):
            NavStateQPVP(
                q=np.array([1.0, 0.0, 0.0, 0.0]), v=np.zeros(2), p=np.zeros(3)
            )

    def test_nav_state_qpvp_invalid_p_shape(self) -> None:
        """Test NavStateQPVP rejects wrong position shape."""
        with pytest.raises(ValueError, match="p must have shape"):
            NavStateQPVP(
                q=np.array([1.0, 0.0, 0.0, 0.0]), v=np.zeros(3), p=np.zeros(4)
            )


class TestNavStateQPVPBias(unittest.TestCase):
    """Test suite for NavStateQPVPBias (augmented state with biases)."""

    def test_nav_state_qpvp_bias_valid_construction(self) -> None:
        """Test valid NavStateQPVPBias construction."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.zeros(3)
        p = np.zeros(3)
        b_g = np.array([1e-4, -5e-5, 2e-5])  # rad/s
        b_a = np.array([0.01, -0.005, 0.02])  # m/s^2

        state = NavStateQPVPBias(q=q, v=v, p=p, b_g=b_g, b_a=b_a)

        assert state.q.shape == (4,)
        assert state.v.shape == (3,)
        assert state.p.shape == (3,)
        assert state.b_g.shape == (3,)
        assert state.b_a.shape == (3,)

    def test_nav_state_qpvp_bias_mutability(self) -> None:
        """Test that NavStateQPVPBias is mutable."""
        state = NavStateQPVPBias(
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            v=np.zeros(3),
            p=np.zeros(3),
            b_g=np.zeros(3),
            b_a=np.zeros(3),
        )

        # Should allow modification
        state.b_g = np.array([1e-3, 2e-3, 3e-3])
        state.b_a = np.array([0.1, 0.2, 0.3])

        np.testing.assert_array_equal(state.b_g, [1e-3, 2e-3, 3e-3])
        np.testing.assert_array_equal(state.b_a, [0.1, 0.2, 0.3])

    def test_nav_state_qpvp_bias_non_unit_quaternion_warning(self) -> None:
        """Test that non-unit quaternion triggers a warning."""
        q_bad = np.array([2.0, 0.0, 0.0, 0.0])  # ||q|| = 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = NavStateQPVPBias(
                q=q_bad, v=np.zeros(3), p=np.zeros(3), b_g=np.zeros(3), b_a=np.zeros(3)
            )

            assert len(w) == 1
            assert "non-unit quaternion" in str(w[0].message).lower()

    def test_nav_state_qpvp_bias_invalid_b_g_shape(self) -> None:
        """Test NavStateQPVPBias rejects wrong gyro bias shape."""
        with pytest.raises(ValueError, match="b_g must have shape"):
            NavStateQPVPBias(
                q=np.array([1.0, 0.0, 0.0, 0.0]),
                v=np.zeros(3),
                p=np.zeros(3),
                b_g=np.zeros(4),  # Wrong
                b_a=np.zeros(3),
            )

    def test_nav_state_qpvp_bias_invalid_b_a_shape(self) -> None:
        """Test NavStateQPVPBias rejects wrong accel bias shape."""
        with pytest.raises(ValueError, match="b_a must have shape"):
            NavStateQPVPBias(
                q=np.array([1.0, 0.0, 0.0, 0.0]),
                v=np.zeros(3),
                p=np.zeros(3),
                b_g=np.zeros(3),
                b_a=np.zeros(2),  # Wrong
            )


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for all data structures."""

    def test_large_imu_series(self) -> None:
        """Test ImuSeries with large number of samples."""
        n = 100000  # 100k samples
        t = np.linspace(0, 1000, n)
        accel = np.random.randn(n, 3)
        gyro = np.random.randn(n, 3)

        imu = ImuSeries(t=t, accel=accel, gyro=gyro, meta={})

        assert imu.t.shape == (n,)
        assert imu.accel.shape == (n, 3)

    def test_nav_state_zero_initialization(self) -> None:
        """Test navigation states with all-zero arrays (except quaternion)."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        state_simple = NavStateQPVP(q=q, v=np.zeros(3), p=np.zeros(3))
        state_bias = NavStateQPVPBias(
            q=q, v=np.zeros(3), p=np.zeros(3), b_g=np.zeros(3), b_a=np.zeros(3)
        )

        np.testing.assert_array_equal(state_simple.v, 0.0)
        np.testing.assert_array_equal(state_bias.b_g, 0.0)

    def test_metadata_flexibility(self) -> None:
        """Test that meta dict accepts arbitrary keys."""
        n = 10
        custom_meta = {
            "device_serial": "ABC123",
            "calibration_date": "2025-12-13",
            "notes": "Test sensor",
            "custom_array": np.array([1, 2, 3]),
            "nested_dict": {"key": "value"},
        }

        imu = ImuSeries(
            t=np.linspace(0, 1, n),
            accel=np.zeros((n, 3)),
            gyro=np.zeros((n, 3)),
            meta=custom_meta,
        )

        assert imu.meta["device_serial"] == "ABC123"
        assert "nested_dict" in imu.meta


if __name__ == "__main__":
    unittest.main()

