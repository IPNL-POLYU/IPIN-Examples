"""
Unit tests for DOP (Dilution of Precision) utilities.

Tests geometry matrix computation and DOP metrics.
Validates book equations 4.103-4.108.
"""

import numpy as np
import pytest

from core.rf.dop import (
    compute_dop,
    compute_dop_map,
    compute_geometry_matrix,
    position_error_from_dop,
)


class TestGeometryMatrix:
    """Test geometry matrix computation."""

    def test_geometry_matrix_toa_2d(self):
        """Test TOA geometry matrix in 2D."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Should have shape (N, 2)
        assert H.shape == (4, 2)

        # Unit vectors should point from position to anchors
        # Check one manually
        expected_dir = -(anchors[0] - position) / np.linalg.norm(
            anchors[0] - position
        )
        assert np.allclose(H[0], expected_dir)

    def test_geometry_matrix_toa_3d(self):
        """Test TOA geometry matrix in 3D."""
        anchors = np.array(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float
        )
        position = np.array([5.0, 5.0, 2.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Should have shape (N, 3)
        assert H.shape == (4, 3)

    def test_geometry_matrix_tdoa(self):
        """Test TDOA geometry matrix."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "tdoa")

        # TDOA: N-1 measurements relative to reference
        assert H.shape == (3, 2)

    def test_geometry_matrix_aoa(self):
        """Test AOA geometry matrix."""
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)
        position = np.array([3.0, 4.0])

        H = compute_geometry_matrix(anchors, position, "aoa")

        # AOA: N azimuth measurements
        assert H.shape == (4, 2)

    def test_geometry_matrix_invalid_type(self):
        """Test geometry matrix with invalid measurement type."""
        anchors = np.array([[0, 0], [10, 0]], dtype=float)
        position = np.array([5.0, 5.0])

        with pytest.raises(ValueError, match="measurement_type"):
            compute_geometry_matrix(anchors, position, "invalid")

    def test_geometry_matrix_coincident_position(self):
        """Test geometry matrix when position coincides with anchor."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([0.0, 0.0])  # Same as first anchor

        H = compute_geometry_matrix(anchors, position, "toa")

        # First row should be zero (undefined direction)
        assert np.allclose(H[0], 0.0)


class TestDOP:
    """Test DOP computation."""

    def test_dop_ideal_geometry(self):
        """Test DOP with ideal anchor geometry."""
        # Square anchor layout at unit distance
        anchors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        position = np.array([0.0, 0.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # Ideal geometry should have low DOP
        assert dop["GDOP"] < 3.0
        assert dop["HDOP"] < 2.0
        assert dop["PDOP"] < 2.0

    def test_dop_poor_geometry(self):
        """Test DOP with poor anchor geometry."""
        # Collinear anchors (poor geometry)
        anchors = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
        position = np.array([15.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # Poor geometry should have higher DOP than ideal
        # Even collinear has some cross-range sensitivity
        assert dop["GDOP"] > 0.8

    def test_dop_3d(self):
        """Test DOP computation in 3D."""
        # Cube corners
        anchors = np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
        )
        position = np.array([5.0, 5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # Check all DOP values are present
        assert "GDOP" in dop
        assert "PDOP" in dop
        assert "HDOP" in dop
        assert "VDOP" in dop
        assert dop["VDOP"] is not None

    def test_dop_2d_no_vdop(self):
        """Test that VDOP is None for 2D."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # 2D should not have VDOP
        assert dop["VDOP"] is None

    def test_dop_weighted(self):
        """Test DOP with custom weights."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Equal weights
        dop_equal = compute_dop(H, weights=None)

        # Higher weight on first measurement
        W = np.diag([10.0, 1.0, 1.0, 1.0])
        dop_weighted = compute_dop(H, weights=W)

        # Weighted DOP should be different
        assert dop_equal["GDOP"] != dop_weighted["GDOP"]

    def test_dop_singular_matrix(self):
        """Test DOP with singular geometry matrix."""
        # Only 1 anchor (truly singular for 2D positioning)
        anchors = np.array([[0, 0]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # Should return infinite DOP (singular matrix)
        assert dop["GDOP"] == np.inf
        assert dop["HDOP"] == np.inf


class TestDOPMap:
    """Test DOP map computation."""

    def test_dop_map_2d(self):
        """Test DOP map computation over 2D grid."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        # Create grid
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Compute DOP map
        hdop_map = compute_dop_map(anchors, grid_points, "toa")

        # Should have one value per grid point
        assert hdop_map.shape == (121,)  # 11x11 grid

        # DOP should be lowest near center
        center_idx = 60  # Middle of 11x11 grid (index 60 is at (5,5))
        center_dop = hdop_map[center_idx]

        # Center should have lower DOP than average
        assert center_dop < np.median(hdop_map)

    def test_dop_map_tdoa(self):
        """Test DOP map for TDOA."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        grid_points = np.array([[5, 5], [2, 2], [8, 8]])

        hdop_map = compute_dop_map(anchors, grid_points, "tdoa")

        assert hdop_map.shape == (3,)
        assert np.all(hdop_map > 0)  # All DOPs should be positive

    def test_dop_map_aoa(self):
        """Test DOP map for AOA."""
        anchors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=float)

        grid_points = np.array([[0, 0], [5, 5], [-5, 5]])

        hdop_map = compute_dop_map(anchors, grid_points, "aoa")

        assert hdop_map.shape == (3,)


class TestDOPComparisons:
    """Test comparisons between different geometries."""

    def test_dop_more_anchors_better(self):
        """Test that more anchors generally give better DOP."""
        position = np.array([5.0, 5.0])

        # 4 anchors
        anchors_4 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        H_4 = compute_geometry_matrix(anchors_4, position, "toa")
        dop_4 = compute_dop(H_4)

        # 8 anchors (add intermediate points)
        anchors_8 = np.array(
            [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
                [5, 0],
                [10, 5],
                [5, 10],
                [0, 5],
            ],
            dtype=float,
        )
        H_8 = compute_geometry_matrix(anchors_8, position, "toa")
        dop_8 = compute_dop(H_8)

        # More anchors should give better (lower) DOP
        assert dop_8["GDOP"] <= dop_4["GDOP"]

    def test_dop_center_vs_edge(self):
        """Test that DOP is better at center than at edge."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        # Center position
        H_center = compute_geometry_matrix(
            anchors, np.array([5.0, 5.0]), "toa"
        )
        dop_center = compute_dop(H_center)

        # Edge position
        H_edge = compute_geometry_matrix(
            anchors, np.array([1.0, 1.0]), "toa"
        )
        dop_edge = compute_dop(H_edge)

        # Center should have better DOP
        assert dop_center["HDOP"] < dop_edge["HDOP"]


class TestBookDOPFormulas:
    """Test DOP computation matches book equations 4.103-4.108."""

    def test_dop_from_q_matrix_book_eq_4107(self):
        """Verify GDOP = sqrt(trace(Q)) per book Eq. 4.107."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Manually compute Q = (H^T H)^{-1}
        Q = np.linalg.inv(H.T @ H)

        # GDOP should equal sqrt(trace(Q))
        expected_gdop = np.sqrt(np.trace(Q))
        dop = compute_dop(H)

        assert np.isclose(dop["GDOP"], expected_gdop)

    def test_hdop_vdop_book_eq_4108(self):
        """Verify HDOP = sqrt(κ_ee + κ_nn) and VDOP = sqrt(κ_uu) per Eq. 4.108."""
        anchors = np.array(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 10]],
            dtype=float,
        )
        position = np.array([5.0, 5.0, 0.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Manually compute Q
        Q = np.linalg.inv(H.T @ H)

        # Book notation: κ_ee = Q[0,0], κ_nn = Q[1,1], κ_uu = Q[2,2]
        expected_hdop = np.sqrt(Q[0, 0] + Q[1, 1])
        expected_vdop = np.sqrt(Q[2, 2])

        dop = compute_dop(H)

        assert np.isclose(dop["HDOP"], expected_hdop)
        assert np.isclose(dop["VDOP"], expected_vdop)

    def test_position_error_from_dop_eq_4107(self):
        """Verify σ_position = DOP × σ_measurement per book Eq. 4.107."""
        hdop = 1.5
        sigma_range = 0.3  # meters

        expected_sigma_horizontal = 0.45  # 1.5 * 0.3

        sigma_horizontal = position_error_from_dop(hdop, sigma_range)

        assert np.isclose(sigma_horizontal, expected_sigma_horizontal)

    def test_dop_covariance_relationship(self):
        """Verify C(x_a) = Q × σ_z² per book Eq. 4.103."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])
        sigma_z = 0.3  # measurement noise std

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # Position error should equal HDOP * sigma_z
        expected_sigma_horizontal = dop["HDOP"] * sigma_z
        computed_sigma = position_error_from_dop(dop["HDOP"], sigma_z)

        assert np.isclose(computed_sigma, expected_sigma_horizontal)

    def test_pdop_equals_gdop_for_pure_positioning(self):
        """PDOP = GDOP when no clock bias is estimated."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # For pure positioning, PDOP = GDOP
        assert dop["PDOP"] == dop["GDOP"]

    def test_2d_hdop_equals_gdop(self):
        """In 2D, HDOP = GDOP (no vertical component)."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")
        dop = compute_dop(H)

        # 2D: HDOP = GDOP (only horizontal components)
        assert np.isclose(dop["HDOP"], dop["GDOP"])
        assert dop["VDOP"] is None

    def test_weighted_dop_book_eq_4103(self):
        """Test DOP with weight matrix W = Σ^{-1}."""
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        position = np.array([5.0, 5.0])

        H = compute_geometry_matrix(anchors, position, "toa")

        # Different measurement noise per anchor
        sigmas = np.array([0.2, 0.3, 0.4, 0.5])
        W = np.diag(1.0 / sigmas**2)

        # Compute Q = (H^T W H)^{-1}
        Q_weighted = np.linalg.inv(H.T @ W @ H)
        expected_hdop = np.sqrt(Q_weighted[0, 0] + Q_weighted[1, 1])

        dop = compute_dop(H, weights=W)

        assert np.isclose(dop["HDOP"], expected_hdop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

