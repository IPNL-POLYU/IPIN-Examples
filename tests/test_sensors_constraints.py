"""
Unit tests for core/sensors/constraints.py (drift correction constraints).

Tests cover:
    - ZUPT detector (Eq. 6.44)
    - ZUPT measurement model (Eq. 6.45)
    - ZARU measurement model (Eq. 6.60)
    - NHC measurement model (Eq. 6.61)
    - Edge cases and validation

Run with: pytest tests/test_sensors_constraints.py -v
"""

import unittest
import numpy as np
import pytest

from core.sensors.constraints import (
    detect_zupt,
    ZuptMeasurementModel,
    ZaruMeasurementModel,
    NhcMeasurementModel,
)


class TestDetectZupt(unittest.TestCase):
    """Test suite for ZUPT detector (Eq. 6.44)."""

    def test_detect_zupt_stationary(self) -> None:
        """Test that stationary sensor is detected (both conditions met)."""
        # Stationary: low rotation, accel ≈ gravity
        gyro = np.array([0.01, -0.005, 0.002])  # very small
        accel = np.array([0.0, 0.0, -9.81])  # gravity only
        delta_omega = 0.05  # 0.05 rad/s threshold
        delta_f = 0.5  # 0.5 m/s² threshold

        is_stationary = detect_zupt(gyro, accel, delta_omega, delta_f)

        assert is_stationary == True

    def test_detect_zupt_rotating(self) -> None:
        """Test that rotating sensor is not detected (omega too high)."""
        gyro = np.array([0.5, 0.2, -0.1])  # rotating
        accel = np.array([0.0, 0.0, -9.81])  # gravity only
        delta_omega = 0.05
        delta_f = 0.5

        is_stationary = detect_zupt(gyro, accel, delta_omega, delta_f)

        assert is_stationary == False

    def test_detect_zupt_accelerating(self) -> None:
        """Test that accelerating sensor is not detected (accel too high)."""
        gyro = np.array([0.01, 0.0, 0.0])  # not rotating
        accel = np.array([2.0, 0.5, -9.0])  # accelerating
        delta_omega = 0.05
        delta_f = 0.5

        is_stationary = detect_zupt(gyro, accel, delta_omega, delta_f)

        assert is_stationary == False

    def test_detect_zupt_both_moving(self) -> None:
        """Test that moving sensor fails both conditions."""
        gyro = np.array([0.3, -0.2, 0.5])  # rotating
        accel = np.array([1.5, -0.8, -10.5])  # accelerating
        delta_omega = 0.05
        delta_f = 0.5

        is_stationary = detect_zupt(gyro, accel, delta_omega, delta_f)

        assert is_stationary == False

    def test_detect_zupt_threshold_boundary(self) -> None:
        """Test detection at threshold boundaries."""
        # Exactly at omega threshold
        gyro_boundary = np.array([0.05, 0.0, 0.0])
        accel = np.array([0.0, 0.0, -9.81])
        delta_omega = 0.05
        delta_f = 0.5

        # Should not detect (< threshold required, not <=)
        is_stationary = detect_zupt(gyro_boundary, accel, delta_omega, delta_f)
        assert is_stationary == False

        # Slightly below threshold
        gyro_below = np.array([0.049, 0.0, 0.0])
        is_stationary = detect_zupt(gyro_below, accel, delta_omega, delta_f)
        assert is_stationary == True

    def test_detect_zupt_tilted_stationary(self) -> None:
        """Test ZUPT detection for tilted but stationary sensor."""
        # Stationary sensor tilted 45° (accel has x and z components)
        gyro = np.array([0.01, 0.0, 0.0])  # minimal rotation
        # Tilted: gravity appears as [g/√2, 0, -g/√2]
        g = 9.81
        accel = np.array([g / np.sqrt(2), 0.0, -g / np.sqrt(2)])
        delta_omega = 0.05
        delta_f = 0.5

        is_stationary = detect_zupt(gyro, accel, delta_omega, delta_f)

        # ||accel|| should still be ≈ g
        assert np.isclose(np.linalg.norm(accel), g, atol=0.01)
        assert is_stationary == True


class TestZuptMeasurementModel(unittest.TestCase):
    """Test suite for ZUPT measurement model (Eq. 6.45)."""

    def test_zupt_model_initialization(self) -> None:
        """Test ZUPT model can be initialized."""
        model = ZuptMeasurementModel(sigma_zupt=0.05)
        assert model.sigma_zupt == 0.05

    def test_zupt_model_invalid_sigma(self) -> None:
        """Test that invalid sigma raises error."""
        with pytest.raises(ValueError, match="sigma_zupt must be positive"):
            ZuptMeasurementModel(sigma_zupt=0.0)

        with pytest.raises(ValueError):
            ZuptMeasurementModel(sigma_zupt=-0.1)

    def test_zupt_h_function(self) -> None:
        """Test ZUPT measurement function h(x) returns velocity."""
        model = ZuptMeasurementModel()

        # State: [q (4), v (3), p (3), ...]
        x = np.array([1.0, 0.0, 0.0, 0.0, 1.5, 2.0, -0.5, 10.0, 20.0, 5.0])

        h = model.h(x)

        # Should extract velocity (indices 4:7)
        expected = np.array([1.5, 2.0, -0.5])
        np.testing.assert_array_equal(h, expected)

    def test_zupt_h_jacobian(self) -> None:
        """Test ZUPT Jacobian H = ∂h/∂x."""
        model = ZuptMeasurementModel()

        # State: 10 dimensions
        x = np.zeros(10)
        H = model.H(x)

        # H should be (3, 10) with identity at velocity indices
        assert H.shape == (3, 10)

        # Check that H selects velocity (indices 4:7)
        expected_H = np.zeros((3, 10))
        expected_H[:, 4:7] = np.eye(3)

        np.testing.assert_array_equal(H, expected_H)

    def test_zupt_r_covariance(self) -> None:
        """Test ZUPT measurement noise covariance R."""
        sigma = 0.1
        model = ZuptMeasurementModel(sigma_zupt=sigma)

        R = model.R()

        # Should be σ² I
        expected = (sigma**2) * np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_zupt_model_short_state(self) -> None:
        """Test that short state vector raises error."""
        model = ZuptMeasurementModel()
        x_short = np.zeros(6)  # Too short (needs at least 7)

        with pytest.raises(ValueError, match="at least 7 elements"):
            model.h(x_short)


class TestZaruMeasurementModel(unittest.TestCase):
    """Test suite for ZARU measurement model (Eq. 6.60)."""

    def test_zaru_model_initialization(self) -> None:
        """Test ZARU model can be initialized."""
        model = ZaruMeasurementModel(sigma_zaru=0.01)
        assert model.sigma_zaru == 0.01

    def test_zaru_model_invalid_sigma(self) -> None:
        """Test that invalid sigma raises error."""
        with pytest.raises(ValueError, match="sigma_zaru must be positive"):
            ZaruMeasurementModel(sigma_zaru=0.0)

    def test_zaru_h_function(self) -> None:
        """Test ZARU measurement function h(x) returns zeros."""
        model = ZaruMeasurementModel()

        x = np.zeros(13)  # State with gyro bias
        h = model.h(x)

        # ZARU expects zero angular velocity
        np.testing.assert_array_equal(h, np.zeros(3))

    def test_zaru_h_jacobian(self) -> None:
        """Test ZARU Jacobian H (simplified implementation)."""
        model = ZaruMeasurementModel()

        # State: 13 dimensions (q, v, p, b_g, ...)
        x = np.zeros(13)
        H = model.H(x)

        # H should be (3, 13)
        assert H.shape == (3, 13)

        # Should affect gyro bias at indices 10:13 (∂h/∂b_g = -I)
        assert np.allclose(H[:, 10:13], -np.eye(3))

    def test_zaru_r_covariance(self) -> None:
        """Test ZARU measurement noise covariance R."""
        sigma = 0.02
        model = ZaruMeasurementModel(sigma_zaru=sigma)

        R = model.R()

        expected = (sigma**2) * np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)


class TestNhcMeasurementModel(unittest.TestCase):
    """Test suite for NHC measurement model (Eq. 6.61)."""

    def test_nhc_model_initialization(self) -> None:
        """Test NHC model can be initialized."""
        model = NhcMeasurementModel(sigma_lateral=0.1, sigma_vertical=0.05)
        assert model.sigma_lateral == 0.1
        assert model.sigma_vertical == 0.05

    def test_nhc_model_invalid_sigma(self) -> None:
        """Test that invalid sigmas raise errors."""
        with pytest.raises(ValueError, match="sigma_lateral must be positive"):
            NhcMeasurementModel(sigma_lateral=0.0, sigma_vertical=0.05)

        with pytest.raises(ValueError, match="sigma_vertical must be positive"):
            NhcMeasurementModel(sigma_lateral=0.1, sigma_vertical=-0.01)

    def test_nhc_h_function_level_forward(self) -> None:
        """Test NHC h(x) for level vehicle moving forward."""
        model = NhcMeasurementModel()

        # State: [q (identity), v (forward), p, ...]
        q = np.array([1.0, 0.0, 0.0, 0.0])  # level, heading east
        v_map = np.array([5.0, 0.0, 0.0])  # 5 m/s eastward
        p = np.zeros(3)
        x = np.concatenate([q, v_map, p])

        h = model.h(x)

        # For level vehicle moving forward: lateral and vertical velocity = 0
        expected = np.zeros(2)
        np.testing.assert_array_almost_equal(h, expected, decimal=5)

    def test_nhc_h_function_lateral_velocity(self) -> None:
        """Test NHC h(x) detects lateral velocity."""
        model = NhcMeasurementModel()

        # State with lateral (y) velocity in map frame
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_map = np.array([0.0, 2.0, 0.0])  # 2 m/s northward (lateral in body)
        p = np.zeros(3)
        x = np.concatenate([q, v_map, p])

        h = model.h(x)

        # With identity q, body = map, so lateral velocity = 2 m/s
        # h = [v_y_body, v_z_body] = [2.0, 0.0]
        expected = np.array([2.0, 0.0])
        np.testing.assert_array_almost_equal(h, expected, decimal=5)

    def test_nhc_h_function_vertical_velocity(self) -> None:
        """Test NHC h(x) detects vertical velocity."""
        model = NhcMeasurementModel()

        # State with vertical (z) velocity
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_map = np.array([0.0, 0.0, 1.5])  # 1.5 m/s upward
        p = np.zeros(3)
        x = np.concatenate([q, v_map, p])

        h = model.h(x)

        # h = [v_y_body, v_z_body] = [0.0, 1.5]
        expected = np.array([0.0, 1.5])
        np.testing.assert_array_almost_equal(h, expected, decimal=5)

    def test_nhc_h_function_rotated_vehicle(self) -> None:
        """Test NHC h(x) with rotated vehicle."""
        model = NhcMeasurementModel()

        # Vehicle rotated 90° (heading north instead of east)
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # 90° yaw
        v_map = np.array([5.0, 0.0, 0.0])  # 5 m/s eastward in map
        p = np.zeros(3)
        x = np.concatenate([q, v_map, p])

        h = model.h(x)

        # After rotation, eastward velocity appears as lateral (negative y) in body
        # h[0] (lateral) should be ≈ -5.0
        assert h[0] < 0  # moving to the right in body frame
        assert np.isclose(np.abs(h[0]), 5.0, atol=0.1)

    def test_nhc_h_jacobian(self) -> None:
        """Test NHC Jacobian H = ∂h/∂x."""
        model = NhcMeasurementModel()

        # State: [q, v, p]
        x = np.zeros(10)
        x[0] = 1.0  # q0 = 1 (identity quaternion)

        H = model.H(x)

        # H should be (2, 10)
        assert H.shape == (2, 10)

        # For identity quaternion, ∂h/∂v should extract y and z components
        # (rows 1 and 2 of C_M^B = identity)
        expected_dh_dv = np.array([[0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(H[:, 4:7], expected_dh_dv)

    def test_nhc_r_covariance(self) -> None:
        """Test NHC measurement noise covariance R."""
        sigma_lat = 0.2
        sigma_vert = 0.1
        model = NhcMeasurementModel(sigma_lateral=sigma_lat, sigma_vertical=sigma_vert)

        R = model.R()

        # Should be diagonal with different variances
        expected = np.diag([sigma_lat**2, sigma_vert**2])
        np.testing.assert_array_almost_equal(R, expected)


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for constraints."""

    def test_zupt_detector_different_thresholds(self) -> None:
        """Test ZUPT detector behavior with different thresholds."""
        gyro = np.array([0.03, 0.0, 0.0])
        accel = np.array([0.0, 0.0, -9.81])

        # Strict threshold
        is_stat_strict = detect_zupt(gyro, accel, delta_omega=0.02, delta_f=0.3)
        assert is_stat_strict == False

        # Loose threshold
        is_stat_loose = detect_zupt(gyro, accel, delta_omega=0.1, delta_f=1.0)
        assert is_stat_loose == True

    def test_measurement_models_with_long_state(self) -> None:
        """Test measurement models work with extended state vectors."""
        # Extended state: [q, v, p, b_g, b_a, ...]
        x = np.zeros(16)
        x[0] = 1.0  # q0
        x[4:7] = [1.0, 2.0, 3.0]  # velocity

        zupt_model = ZuptMeasurementModel()
        h_zupt = zupt_model.h(x)
        H_zupt = zupt_model.H(x)

        assert h_zupt.shape == (3,)
        assert H_zupt.shape == (3, 16)

    def test_nhc_zero_velocity(self) -> None:
        """Test NHC with zero velocity (should give zero measurement)."""
        model = NhcMeasurementModel()

        q = np.array([1.0, 0.0, 0.0, 0.0])
        v_map = np.zeros(3)  # stationary
        p = np.zeros(3)
        x = np.concatenate([q, v_map, p])

        h = model.h(x)

        # Zero velocity should give zero lateral and vertical components
        np.testing.assert_array_almost_equal(h, np.zeros(2))


if __name__ == "__main__":
    unittest.main()

