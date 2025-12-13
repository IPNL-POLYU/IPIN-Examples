"""Unit tests for core.fingerprinting.deterministic module.

Tests NN and k-NN fingerprinting algorithms (Eqs. 5.1 and 5.2).

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    distance,
    knn_localize,
    nn_localize,
    pairwise_distances,
)


class TestDistance:
    """Test suite for distance() function."""

    def test_euclidean_distance_simple(self):
        """Test Euclidean distance on simple vectors."""
        z = np.array([0.0, 0.0, 0.0])
        f = np.array([3.0, 4.0, 0.0])

        d = distance(z, f, metric="euclidean")

        assert d == pytest.approx(5.0)  # 3-4-5 triangle

    def test_euclidean_distance_rss(self):
        """Test Euclidean distance on RSS fingerprints."""
        z = np.array([-50.0, -60.0, -70.0])
        f = np.array([-52.0, -58.0, -72.0])

        d = distance(z, f, metric="euclidean")

        expected = np.sqrt(2**2 + 2**2 + 2**2)
        assert d == pytest.approx(expected)

    def test_manhattan_distance_simple(self):
        """Test Manhattan distance on simple vectors."""
        z = np.array([0.0, 0.0, 0.0])
        f = np.array([3.0, 4.0, 0.0])

        d = distance(z, f, metric="manhattan")

        assert d == pytest.approx(7.0)  # |3| + |4| + |0|

    def test_manhattan_distance_rss(self):
        """Test Manhattan distance on RSS fingerprints."""
        z = np.array([-50.0, -60.0, -70.0])
        f = np.array([-52.0, -58.0, -72.0])

        d = distance(z, f, metric="manhattan")

        assert d == pytest.approx(6.0)  # |2| + |2| + |2|

    def test_distance_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        z = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0])  # Different length

        with pytest.raises(ValueError, match="must have same shape"):
            distance(z, f)

    def test_distance_invalid_metric_error(self):
        """Test that invalid metric raises ValueError."""
        z = np.array([1.0, 2.0])
        f = np.array([3.0, 4.0])

        with pytest.raises(ValueError, match="Unsupported metric"):
            distance(z, f, metric="cosine")


class TestPairwiseDistances:
    """Test suite for pairwise_distances() function."""

    def test_pairwise_euclidean_simple(self):
        """Test pairwise Euclidean distances."""
        z = np.array([0.0, 0.0])
        F = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])

        distances = pairwise_distances(z, F, metric="euclidean")

        expected = np.array([1.0, 1.0, 5.0])
        np.testing.assert_array_almost_equal(distances, expected)

    def test_pairwise_manhattan_simple(self):
        """Test pairwise Manhattan distances."""
        z = np.array([0.0, 0.0])
        F = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])

        distances = pairwise_distances(z, F, metric="manhattan")

        expected = np.array([1.0, 1.0, 7.0])
        np.testing.assert_array_almost_equal(distances, expected)

    def test_pairwise_rss_fingerprints(self):
        """Test pairwise distances on RSS fingerprints."""
        z = np.array([-50, -60, -70])
        F = np.array([[-50, -60, -70], [-52, -58, -72], [-48, -62, -68]])

        distances = pairwise_distances(z, F, metric="euclidean")

        # First RP is identical
        assert distances[0] == pytest.approx(0.0)
        # Second and third should be non-zero
        assert distances[1] > 0
        assert distances[2] > 0

    def test_pairwise_query_dimension_error(self):
        """Test that 2D query raises ValueError."""
        z = np.array([[1, 2], [3, 4]])  # 2D, should be 1D
        F = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Query z must be 1D array"):
            pairwise_distances(z, F)

    def test_pairwise_features_dimension_error(self):
        """Test that 1D features raise ValueError."""
        z = np.array([1, 2])
        F = np.array([1, 2, 3, 4])  # 1D, should be 2D

        with pytest.raises(ValueError, match="Reference F must be 2D array"):
            pairwise_distances(z, F)

    def test_pairwise_incompatible_dimensions_error(self):
        """Test that incompatible feature dimensions raise ValueError."""
        z = np.array([1, 2, 3])  # 3 features
        F = np.array([[1, 2], [3, 4]])  # 2 features per RP

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            pairwise_distances(z, F)

    def test_pairwise_invalid_metric_error(self):
        """Test that invalid metric raises ValueError."""
        z = np.array([1, 2])
        F = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Unsupported metric"):
            pairwise_distances(z, F, metric="hamming")


@pytest.fixture
def simple_database():
    """Create a simple 2D single-floor database for testing."""
    return FingerprintDatabase(
        locations=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
        features=np.array(
            [[-50, -60, -70], [-60, -50, -80], [-70, -80, -50], [-55, -55, -55]]
        ),
        floor_ids=np.array([0, 0, 0, 0]),
        meta={"ap_ids": ["AP1", "AP2", "AP3"]},
    )


@pytest.fixture
def multifloor_database():
    """Create a multi-floor database for testing."""
    return FingerprintDatabase(
        locations=np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],  # Floor 0
                [0.0, 5.0],
                [5.0, 5.0],
                [10.0, 5.0],  # Floor 1
                [0.0, 10.0],
                [5.0, 10.0],  # Floor 2
            ]
        ),
        features=np.array(
            [
                [-50, -60, -70],
                [-52, -58, -72],
                [-54, -56, -74],  # Floor 0
                [-55, -65, -75],
                [-57, -63, -77],
                [-59, -61, -79],  # Floor 1
                [-60, -70, -80],
                [-62, -68, -82],  # Floor 2
            ]
        ),
        floor_ids=np.array([0, 0, 0, 1, 1, 1, 2, 2]),
        meta={"ap_ids": ["AP1", "AP2", "AP3"], "building_id": "test_building"},
    )


class TestNNLocalize:
    """Test suite for nn_localize() function (Eq. 5.1)."""

    def test_nn_exact_match(self, simple_database):
        """Test NN when query exactly matches a reference point."""
        # Query exactly matches RP at [0, 0]
        z = np.array([-50, -60, -70])

        x_hat = nn_localize(z, simple_database)

        np.testing.assert_array_almost_equal(x_hat, [0.0, 0.0])

    def test_nn_approximate_match(self, simple_database):
        """Test NN with query close to but not exactly matching an RP."""
        # Query close to RP at [10, 0]: [-60, -50, -80]
        z = np.array([-61, -51, -81])

        x_hat = nn_localize(z, simple_database)

        np.testing.assert_array_almost_equal(x_hat, [10.0, 0.0])

    def test_nn_manhattan_metric(self, simple_database):
        """Test NN with Manhattan distance metric."""
        z = np.array([-52, -58, -72])

        x_hat = nn_localize(z, simple_database, metric="manhattan")

        # Should find RP at [0, 0]: [-50, -60, -70]
        # Manhattan distance = |2| + |2| + |2| = 6
        np.testing.assert_array_almost_equal(x_hat, [0.0, 0.0])

    def test_nn_multifloor_no_constraint(self, multifloor_database):
        """Test NN across all floors without floor constraint."""
        # Query close to floor 1 RP at [5, 5]: [-57, -63, -77]
        z = np.array([-58, -64, -78])

        x_hat = nn_localize(z, multifloor_database, floor_id=None)

        # Should find best match across all floors
        np.testing.assert_array_almost_equal(x_hat, [5.0, 5.0])

    def test_nn_floor_constraint(self, multifloor_database):
        """Test NN with floor constraint."""
        # Query close to floor 2 RP, but search only floor 0
        z = np.array([-61, -69, -81])

        x_hat = nn_localize(z, multifloor_database, floor_id=0)

        # Should find best match on floor 0 only (one of the floor 0 RPs)
        assert x_hat[1] == 0.0  # All floor 0 RPs have y=0

    def test_nn_query_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong number of features raises ValueError."""
        z = np.array([-50, -60])  # Only 2 features, need 3

        with pytest.raises(ValueError, match="Query fingerprint has 2 features"):
            nn_localize(z, simple_database)

    def test_nn_invalid_floor_error(self, simple_database):
        """Test that invalid floor_id raises ValueError."""
        z = np.array([-50, -60, -70])

        with pytest.raises(ValueError, match="Floor 5 not found"):
            nn_localize(z, simple_database, floor_id=5)


class TestKNNLocalize:
    """Test suite for knn_localize() function (Eq. 5.2)."""

    def test_knn_k1_matches_nn(self, simple_database):
        """Test that k=1 gives same result as NN."""
        z = np.array([-52, -58, -72])

        x_hat_nn = nn_localize(z, simple_database)
        x_hat_knn = knn_localize(z, simple_database, k=1)

        np.testing.assert_array_almost_equal(x_hat_knn, x_hat_nn)

    def test_knn_k3_inverse_distance(self, simple_database):
        """Test k=3 with inverse distance weighting."""
        # Query at geometric center (not exact RP)
        z = np.array([-55, -55, -65])

        x_hat = knn_localize(z, simple_database, k=3, weighting="inverse_distance")

        # Should be weighted average of 3 nearest RPs
        # Result should be somewhere in the middle of the floor
        assert 0 <= x_hat[0] <= 10
        assert 0 <= x_hat[1] <= 10

    def test_knn_k3_uniform_weights(self, simple_database):
        """Test k=3 with uniform weights (simple average)."""
        z = np.array([-55, -55, -65])

        x_hat = knn_localize(z, simple_database, k=3, weighting="uniform")

        # With uniform weights, result is simple average of 3 nearest locations
        assert 0 <= x_hat[0] <= 10
        assert 0 <= x_hat[1] <= 10

    def test_knn_all_rps(self, simple_database):
        """Test k=M using all reference points."""
        z = np.array([-55, -55, -65])

        # Use all 4 RPs
        x_hat = knn_localize(z, simple_database, k=4, weighting="uniform")

        # With uniform weights and all 4 corner RPs, should get center
        expected = np.array([5.0, 5.0])  # Center of square
        np.testing.assert_array_almost_equal(x_hat, expected)

    def test_knn_manhattan_metric(self, simple_database):
        """Test k-NN with Manhattan distance."""
        z = np.array([-52, -58, -72])

        x_hat = knn_localize(z, simple_database, k=2, metric="manhattan")

        # Should use Manhattan distance for neighbor selection
        assert 0 <= x_hat[0] <= 10
        assert 0 <= x_hat[1] <= 10

    def test_knn_multifloor_constraint(self, multifloor_database):
        """Test k-NN with floor constraint."""
        z = np.array([-56, -64, -76])

        # Use only floor 1 RPs (3 RPs available)
        x_hat = knn_localize(z, multifloor_database, k=2, floor_id=1)

        # Result should have y=5 (all floor 1 RPs)
        assert x_hat[1] == pytest.approx(5.0)

    def test_knn_k_less_than_1_error(self, simple_database):
        """Test that k < 1 raises ValueError."""
        z = np.array([-50, -60, -70])

        with pytest.raises(ValueError, match="k must be >= 1"):
            knn_localize(z, simple_database, k=0)

    def test_knn_k_exceeds_m_error(self, simple_database):
        """Test that k > M raises ValueError."""
        z = np.array([-50, -60, -70])

        # Database has only 4 RPs
        with pytest.raises(ValueError, match="k=10 exceeds"):
            knn_localize(z, simple_database, k=10)

    def test_knn_invalid_weighting_error(self, simple_database):
        """Test that invalid weighting method raises ValueError."""
        z = np.array([-50, -60, -70])

        with pytest.raises(ValueError, match="Unsupported weighting method"):
            knn_localize(z, simple_database, k=3, weighting="gaussian")

    def test_knn_query_dimension_mismatch_error(self, simple_database):
        """Test that query with wrong number of features raises ValueError."""
        z = np.array([-50, -60])  # Only 2 features, need 3

        with pytest.raises(ValueError, match="Query fingerprint has 2 features"):
            knn_localize(z, simple_database, k=3)

    def test_knn_eps_prevents_zero_division(self, simple_database):
        """Test that eps parameter prevents division by zero for exact matches."""
        # Query exactly matches first RP
        z = np.array([-50, -60, -70])

        # With eps, this should not crash even though distance is 0
        x_hat = knn_localize(z, simple_database, k=3, eps=1e-6)

        # Should heavily weight the exact match
        # Result should be very close to [0, 0]
        assert np.linalg.norm(x_hat - np.array([0.0, 0.0])) < 2.0


class TestIntegration:
    """Integration tests comparing NN and k-NN behavior."""

    def test_nn_vs_knn_consistency(self, simple_database):
        """Test that k=1 with inverse_distance matches NN."""
        z = np.array([-51, -61, -71])

        x_nn = nn_localize(z, simple_database, metric="euclidean")
        x_knn = knn_localize(
            z, simple_database, k=1, metric="euclidean", weighting="inverse_distance"
        )

        np.testing.assert_array_almost_equal(x_nn, x_knn)

    def test_increasing_k_smoothing_effect(self, multifloor_database):
        """Test that increasing k produces smoother (more averaged) estimates."""
        z = np.array([-51, -61, -71])

        x_k1 = knn_localize(z, multifloor_database, k=1, floor_id=0)
        x_k3 = knn_localize(z, multifloor_database, k=3, floor_id=0)

        # k=1 should be at an exact RP location
        # k=3 should be averaged, likely between RPs
        # This is a qualitative check - just verify both are valid
        assert 0 <= x_k1[0] <= 10 and 0 <= x_k1[1] <= 10
        assert 0 <= x_k3[0] <= 10 and 0 <= x_k3[1] <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

