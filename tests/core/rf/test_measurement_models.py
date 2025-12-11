"""
Unit tests for RF measurement models.

Tests TOA, TDOA, AOA, and RSS measurement functions.
"""

import numpy as np
import pytest

from core.rf.measurement_models import (
    SPEED_OF_LIGHT,
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    rss_pathloss,
    rss_to_distance,
    tdoa_measurement_vector,
    tdoa_range_difference,
    toa_range,
    two_way_toa_range,
)


class TestTOA:
    """Test TOA measurement models."""

    def test_toa_range_basic(self):
        """Test basic TOA range calculation."""
        tx_pos = np.array([0.0, 0.0, 0.0])
        rx_pos = np.array([3.0, 4.0, 0.0])

        range_m = toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, 5.0), f"Expected 5.0, got {range_m}"

    def test_toa_range_with_clock_bias(self):
        """Test TOA range with clock bias."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([3.0, 4.0])
        clock_bias = 1e-9  # 1 nanosecond

        range_m = toa_range(tx_pos, rx_pos, clock_bias=clock_bias)

        expected = 5.0 + SPEED_OF_LIGHT * clock_bias
        assert np.isclose(range_m, expected)

    def test_toa_range_2d(self):
        """Test TOA in 2D."""
        tx_pos = np.array([0.0, 0.0])
        rx_pos = np.array([1.0, 1.0])

        range_m = toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, np.sqrt(2))

    def test_two_way_toa(self):
        """Test two-way TOA (RTT)."""
        tx_pos = np.array([0.0, 0.0, 0.0])
        rx_pos = np.array([10.0, 0.0, 0.0])

        range_m = two_way_toa_range(tx_pos, rx_pos)

        assert np.isclose(range_m, 10.0)


class TestRSS:
    """Test RSS measurement models."""

    def test_rss_pathloss_free_space(self):
        """Test RSS path-loss model in free space (n=2)."""
        tx_power = 0.0  # dBm
        distance = 10.0  # meters
        path_loss_exp = 2.0

        rss = rss_pathloss(tx_power, distance, path_loss_exp)

        # Free space: PL = 10*2*log10(10/1) = 20 dB
        expected_rss = 0.0 - 20.0  # -20 dBm
        assert np.isclose(rss, expected_rss, atol=0.1)

    def test_rss_pathloss_indoor(self):
        """Test RSS with indoor path-loss exponent."""
        tx_power = 20.0  # dBm
        distance = 5.0  # meters
        path_loss_exp = 3.0  # Indoor

        rss = rss_pathloss(tx_power, distance, path_loss_exp)

        # PL = 10*3*log10(5/1) ≈ 20.97 dB
        expected_rss = 20.0 - 10 * 3.0 * np.log10(5.0)
        assert np.isclose(rss, expected_rss, atol=0.1)

    def test_rss_to_distance_roundtrip(self):
        """Test RSS to distance conversion (round-trip)."""
        tx_power = 0.0
        distance_true = 15.0
        path_loss_exp = 2.5

        # Convert distance to RSS
        rss = rss_pathloss(tx_power, distance_true, path_loss_exp)

        # Convert RSS back to distance
        distance_est = rss_to_distance(rss, tx_power, path_loss_exp)

        assert np.isclose(distance_est, distance_true, rtol=1e-3)

    def test_rss_invalid_distance(self):
        """Test RSS with invalid distance."""
        with pytest.raises(ValueError, match="Distance must be positive"):
            rss_pathloss(0.0, 0.0, 2.0)


class TestTDOA:
    """Test TDOA measurement models."""

    def test_tdoa_range_difference_basic(self):
        """Test basic TDOA range difference."""
        anchor_1 = np.array([0.0, 0.0])
        anchor_2 = np.array([10.0, 0.0])
        agent = np.array([5.0, 5.0])

        rd = tdoa_range_difference(anchor_1, anchor_2, agent)

        # Distance from agent to anchor_1: sqrt(25 + 25) = 7.071
        # Distance from agent to anchor_2: sqrt(25 + 25) = 7.071
        # Range difference: 0.0
        assert np.isclose(rd, 0.0, atol=1e-3)

    def test_tdoa_range_difference_asymmetric(self):
        """Test TDOA with asymmetric position."""
        anchor_1 = np.array([0.0, 0.0])
        anchor_2 = np.array([10.0, 0.0])
        agent = np.array([2.0, 0.0])

        rd = tdoa_range_difference(anchor_1, anchor_2, agent)

        # Distance to anchor_1: 2.0
        # Distance to anchor_2: 8.0
        # Range difference: 2.0 - 8.0 = -6.0
        assert np.isclose(rd, -6.0)

    def test_tdoa_measurement_vector(self):
        """Test TDOA measurement vector generation."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        agent = np.array([5, 5])

        tdoa = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)

        # Should return (N-1) measurements
        assert tdoa.shape == (3,)

    def test_tdoa_measurement_vector_different_reference(self):
        """Test TDOA with different reference anchor."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        # Use asymmetric position to avoid all-zero measurements
        agent = np.array([3, 7])

        tdoa_ref0 = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=0)
        tdoa_ref1 = tdoa_measurement_vector(anchors, agent, reference_anchor_idx=1)

        # Different reference should give different measurements
        assert not np.allclose(tdoa_ref0, tdoa_ref1)

    def test_tdoa_invalid_reference(self):
        """Test TDOA with invalid reference index."""
        anchors = np.array([[0, 0], [10, 0], [10, 10]])
        agent = np.array([5, 5])

        with pytest.raises(ValueError, match="reference_anchor_idx"):
            tdoa_measurement_vector(anchors, agent, reference_anchor_idx=5)


class TestAOA:
    """Test AOA measurement models."""

    def test_aoa_azimuth_basic(self):
        """Test basic azimuth calculation."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([1.0, 1.0])

        azimuth = aoa_azimuth(anchor, agent)

        # 45 degrees
        assert np.isclose(azimuth, np.pi / 4)

    def test_aoa_azimuth_quadrants(self):
        """Test azimuth in all quadrants."""
        anchor = np.array([0.0, 0.0])

        # Quadrant I (0° to 90°)
        agent_q1 = np.array([1.0, 1.0])
        az_q1 = aoa_azimuth(anchor, agent_q1)
        assert 0 < az_q1 < np.pi / 2

        # Quadrant II (90° to 180°)
        agent_q2 = np.array([-1.0, 1.0])
        az_q2 = aoa_azimuth(anchor, agent_q2)
        assert np.pi / 2 < az_q2 < np.pi

        # Quadrant III (-180° to -90°)
        agent_q3 = np.array([-1.0, -1.0])
        az_q3 = aoa_azimuth(anchor, agent_q3)
        assert -np.pi < az_q3 < -np.pi / 2

        # Quadrant IV (-90° to 0°)
        agent_q4 = np.array([1.0, -1.0])
        az_q4 = aoa_azimuth(anchor, agent_q4)
        assert -np.pi / 2 < az_q4 < 0

    def test_aoa_elevation_basic(self):
        """Test basic elevation calculation."""
        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([1.0, 0.0, 1.0])

        elevation = aoa_elevation(anchor, agent)

        # 45 degrees elevation
        assert np.isclose(elevation, np.pi / 4)

    def test_aoa_elevation_horizontal(self):
        """Test elevation for horizontal position."""
        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([10.0, 0.0, 0.0])

        elevation = aoa_elevation(anchor, agent)

        # 0 degrees elevation (horizontal)
        assert np.isclose(elevation, 0.0, atol=1e-6)

    def test_aoa_elevation_vertical(self):
        """Test elevation for vertical position."""
        anchor = np.array([0.0, 0.0, 0.0])
        agent = np.array([0.0, 0.0, 10.0])

        elevation = aoa_elevation(anchor, agent)

        # 90 degrees elevation (vertical)
        assert np.isclose(elevation, np.pi / 2, atol=1e-6)

    def test_aoa_elevation_requires_3d(self):
        """Test that elevation requires 3D positions."""
        anchor = np.array([0.0, 0.0])
        agent = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="3D positions required"):
            aoa_elevation(anchor, agent)

    def test_aoa_measurement_vector_2d(self):
        """Test AOA measurement vector in 2D (azimuth only)."""
        anchors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        agent = np.array([0.5, 0.5])

        aoa = aoa_measurement_vector(anchors, agent, include_elevation=False)

        # Should return N azimuth angles
        assert aoa.shape == (4,)
        # All angles should be in [-π, π]
        assert np.all((-np.pi <= aoa) & (aoa <= np.pi))

    def test_aoa_measurement_vector_3d(self):
        """Test AOA measurement vector with elevation."""
        anchors = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        agent = np.array([0.5, 0.5, 1.0])

        aoa = aoa_measurement_vector(anchors, agent, include_elevation=True)

        # Should return 2*N values (azimuth and elevation for each anchor)
        assert aoa.shape == (8,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

