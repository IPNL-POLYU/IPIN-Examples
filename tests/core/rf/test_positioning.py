"""
Unit tests for RF positioning algorithms.

Tests TOA, TDOA, and AOA positioning algorithms.
"""

import numpy as np
import pytest

from core.rf.positioning import (
    AOAPositioner,
    TDOAPositioner,
    TOAPositioner,
    toa_solve_with_clock_bias,
)


class TestTOAPositioning:
    """Test TOA positioning algorithms."""

    def test_toa_positioning_perfect_measurements(self):
        """Test TOA positioning with perfect measurements."""
        # Square anchor layout
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        # True position
        true_pos = np.array([5.0, 5.0])

        # Compute true ranges
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        # Solve
        positioner = TOAPositioner(anchors, method="ls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_positioning_with_noise(self):
        """Test TOA positioning with measurement noise."""
        np.random.seed(42)

        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([3.0, 7.0])

        # True ranges with noise
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        ranges = true_ranges + np.random.randn(4) * 0.1  # 10cm noise

        positioner = TOAPositioner(anchors, method="iwls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([5.0, 5.0])
        )

        # Should be close to true position
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 0.5  # Within 50cm for 10cm noise

    def test_toa_positioning_3d(self):
        """Test TOA positioning in 3D."""
        # Cube anchor layout
        anchors = np.array(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 10]],
            dtype=float,
        )

        true_pos = np.array([5.0, 5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors, method="ls")
        estimated_pos, info = positioner.solve(
            ranges, initial_guess=np.array([6.0, 6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_toa_with_clock_bias(self):
        """Test TOA positioning with unknown clock bias."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        true_clock_bias_m = 2.0  # 2 meters

        # Compute ranges with clock bias
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        ranges = true_ranges + true_clock_bias_m

        # Solve with clock bias estimation
        initial_guess = np.array([6.0, 6.0, 0.0])  # [x, y, clock_bias]
        pos, bias, info = toa_solve_with_clock_bias(
            anchors, ranges, initial_guess
        )

        # Check position accuracy
        assert np.linalg.norm(pos - true_pos) < 1e-3

        # Check clock bias accuracy
        assert np.abs(bias - true_clock_bias_m) < 1e-3

    def test_toa_convergence_history(self):
        """Test that iteration history is recorded."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])
        ranges = np.linalg.norm(anchors - true_pos, axis=1)

        positioner = TOAPositioner(anchors)
        _, info = positioner.solve(ranges, initial_guess=np.array([2.0, 2.0]))

        # History should have initial guess + iterations
        assert "history" in info
        assert len(info["history"]) >= 2  # At least initial + one iteration


class TestTDOAPositioning:
    """Test TDOA positioning algorithms."""

    def test_tdoa_positioning_perfect_measurements(self):
        """Test TDOA positioning with perfect measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Compute TDOA measurements (reference anchor = 0)
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        # Solve
        positioner = TDOAPositioner(anchors, reference_idx=0)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([6.0, 6.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_tdoa_positioning_with_noise(self):
        """Test TDOA positioning with measurement noise."""
        np.random.seed(43)

        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([3.0, 7.0])

        # True TDOA with noise
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa) + np.random.randn(3) * 0.05  # 5cm noise

        positioner = TDOAPositioner(anchors, reference_idx=0)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([5.0, 5.0])
        )

        # Should be close to true position
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 0.5  # Within 50cm for 5cm noise

    def test_tdoa_different_reference(self):
        """Test TDOA with different reference anchors."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        # Test with reference anchor = 1
        dist_ref = np.linalg.norm(true_pos - anchors[1])
        tdoa = []
        for i in range(len(anchors)):
            if i == 1:
                continue
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        positioner = TDOAPositioner(anchors, reference_idx=1)
        estimated_pos, info = positioner.solve(
            tdoa, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3


class TestAOAPositioning:
    """Test AOA positioning algorithms."""

    def test_aoa_positioning_perfect_measurements(self):
        """Test AOA positioning with perfect measurements."""
        # Anchors at cardinal directions
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        true_pos = np.array([3.0, 4.0])

        # Compute true azimuth angles
        aoa = []
        for anchor in anchors:
            dx = true_pos[0] - anchor[0]
            dy = true_pos[1] - anchor[1]
            aoa.append(np.arctan2(dy, dx))
        aoa = np.array(aoa)

        # Solve
        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0])
        )

        # Should converge to true position
        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3

    def test_aoa_positioning_with_noise(self):
        """Test AOA positioning with angle noise."""
        np.random.seed(44)

        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        true_pos = np.array([2.0, 3.0])

        # True angles with noise
        aoa = []
        for anchor in anchors:
            dx = true_pos[0] - anchor[0]
            dy = true_pos[1] - anchor[1]
            aoa.append(np.arctan2(dy, dx))
        aoa = np.array(aoa) + np.random.randn(4) * np.deg2rad(
            2.0
        )  # 2 degree noise

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([5.0, 5.0])
        )

        # Should be reasonably close
        error = np.linalg.norm(estimated_pos - true_pos)
        assert error < 1.0  # Within 1m for 2° angle noise

    def test_aoa_positioning_square_anchors(self):
        """Test AOA with square anchor layout."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        true_pos = np.array([5.0, 5.0])

        aoa = []
        for anchor in anchors:
            dx = true_pos[0] - anchor[0]
            dy = true_pos[1] - anchor[1]
            aoa.append(np.arctan2(dy, dx))
        aoa = np.array(aoa)

        positioner = AOAPositioner(anchors)
        estimated_pos, info = positioner.solve(
            aoa, initial_guess=np.array([6.0, 6.0])
        )

        assert info["converged"]
        assert np.linalg.norm(estimated_pos - true_pos) < 1e-3


class TestPositioningEdgeCases:
    """Test edge cases and error conditions."""

    def test_toa_insufficient_anchors(self):
        """Test TOA with insufficient anchors."""
        # Only 2 anchors in 2D (need at least 3)
        anchors = np.array([[0, 0], [10, 0]], dtype=float)
        ranges = np.array([5.0, 5.0])

        positioner = TOAPositioner(anchors)
        # Should not crash, but may not converge well
        pos, info = positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))

        # At minimum, should complete without exception
        assert pos.shape == (2,)

    def test_tdoa_wrong_measurement_size(self):
        """Test TDOA with wrong number of measurements."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        tdoa = np.array([1.0, 2.0])  # Should be 3 measurements

        positioner = TDOAPositioner(anchors)
        with pytest.raises(ValueError, match="Expected 3 TDOA measurements"):
            positioner.solve(tdoa, initial_guess=np.array([5.0, 5.0]))

    def test_aoa_wrong_measurement_size(self):
        """Test AOA with wrong number of measurements."""
        anchors = np.array([[10, 0], [0, 10], [-10, 0]], dtype=float)
        aoa = np.array([0.0, np.pi / 2])  # Should be 3 measurements

        positioner = AOAPositioner(anchors)
        with pytest.raises(ValueError, match="Expected 3 AOA measurements"):
            positioner.solve(aoa, initial_guess=np.array([0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


