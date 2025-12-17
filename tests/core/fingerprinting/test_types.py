"""Unit tests for core.fingerprinting.types module.

Tests the FingerprintDatabase dataclass and its validation logic.

Author: Navigation Engineer
Date: 2024
"""

import numpy as np
import pytest

from core.fingerprinting.types import Fingerprint, FingerprintDatabase, Location


class TestFingerprintDatabase:
    """Test suite for FingerprintDatabase dataclass."""

    def test_creation_valid_data(self):
        """Test creating a valid database with multi-floor data."""
        # Create a simple 2D database with 6 RPs on 2 floors
        db = FingerprintDatabase(
            locations=np.array(
                [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [0.0, 5.0], [5.0, 5.0], [10.0, 5.0]]
            ),
            features=np.array(
                [
                    [-50, -60, -70],
                    [-60, -50, -80],
                    [-70, -80, -50],
                    [-55, -65, -75],
                    [-65, -55, -85],
                    [-75, -85, -55],
                ]
            ),
            floor_ids=np.array([0, 0, 0, 1, 1, 1]),
            meta={
                "ap_ids": ["AP1", "AP2", "AP3"],
                "unit": "dBm",
                "building_id": "test_building",
            },
        )

        # Verify properties
        assert db.n_reference_points == 6
        assert db.n_features == 3
        assert db.location_dim == 2
        assert db.n_floors == 2
        np.testing.assert_array_equal(db.floor_list, [0, 1])

    def test_creation_single_floor(self):
        """Test creating a database with a single floor."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [10, 0], [10, 10]]),
            features=np.array([[-50, -60], [-60, -50], [-55, -55]]),
            floor_ids=np.array([0, 0, 0]),
            meta={},
        )

        assert db.n_reference_points == 3
        assert db.n_features == 2
        assert db.n_floors == 1
        np.testing.assert_array_equal(db.floor_list, [0])

    def test_creation_3d_locations(self):
        """Test creating a database with 3D location coordinates."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0, 0], [10, 0, 0]]),
            features=np.array([[-50, -60], [-60, -50]]),
            floor_ids=np.array([0, 0]),
            meta={},
        )

        assert db.location_dim == 3
        assert db.n_reference_points == 2

    def test_validation_type_error_locations(self):
        """Test that non-ndarray locations raise TypeError."""
        with pytest.raises(TypeError, match="locations must be a NumPy ndarray"):
            FingerprintDatabase(
                locations=[[0, 0], [10, 0]],  # Python list, not ndarray
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_validation_type_error_features(self):
        """Test that non-ndarray features raise TypeError."""
        with pytest.raises(TypeError, match="features must be a NumPy ndarray"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=[[-50, -60], [-60, -50]],  # Python list
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_validation_type_error_floor_ids(self):
        """Test that non-ndarray floor_ids raise TypeError."""
        with pytest.raises(TypeError, match="floor_ids must be a NumPy ndarray"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=[0, 0],  # Python list
                meta={},
            )

    def test_validation_dimension_error_locations(self):
        """Test that 1D locations array raises ValueError."""
        with pytest.raises(ValueError, match="locations must be 2D array"):
            FingerprintDatabase(
                locations=np.array([0, 0, 10, 0]),  # 1D, should be (M, d)
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_validation_dimension_error_features(self):
        """Test that 1D features array raises ValueError."""
        with pytest.raises(ValueError, match="features must be 2D array"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=np.array([-50, -60, -60, -50]),  # 1D
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_validation_dimension_error_floor_ids(self):
        """Test that 2D floor_ids array raises ValueError."""
        with pytest.raises(ValueError, match="floor_ids must be 1D array"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=np.array([[0], [0]]),  # 2D, should be 1D
                meta={},
            )

    def test_validation_inconsistent_size(self):
        """Test that inconsistent array sizes raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent number of reference points"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),  # 2 RPs
                features=np.array([[-50, -60], [-60, -50], [-55, -55]]),  # 3 RPs
                floor_ids=np.array([0, 0]),  # 2 RPs
                meta={},
            )

    def test_validation_floor_ids_not_integer(self):
        """Test that non-integer floor_ids raise TypeError."""
        with pytest.raises(TypeError, match="floor_ids must have integer dtype"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=np.array([0.0, 0.5]),  # float, should be int
                meta={},
            )

    def test_validation_nan_in_locations(self):
        """Test that NaN in locations raises ValueError."""
        with pytest.raises(ValueError, match="locations contain NaN values"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [np.nan, 0]]),
                features=np.array([[-50, -60], [-60, -50]]),
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_validation_nan_in_features(self):
        """Test that NaN in features raises ValueError."""
        with pytest.raises(ValueError, match="features contain NaN values"):
            FingerprintDatabase(
                locations=np.array([[0, 0], [10, 0]]),
                features=np.array([[-50, np.nan], [-60, -50]]),
                floor_ids=np.array([0, 0]),
                meta={},
            )

    def test_get_floor_mask_valid(self):
        """Test getting a floor mask for an existing floor."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0], [10, 0], [0, 5]]),
            features=np.array([[-50, -60], [-60, -50], [-55, -55], [-65, -65]]),
            floor_ids=np.array([0, 0, 1, 1]),
            meta={},
        )

        mask_floor0 = db.get_floor_mask(0)
        mask_floor1 = db.get_floor_mask(1)

        np.testing.assert_array_equal(mask_floor0, [True, True, False, False])
        np.testing.assert_array_equal(mask_floor1, [False, False, True, True])

    def test_get_floor_mask_invalid_floor(self):
        """Test that getting mask for non-existent floor raises ValueError."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [10, 0]]),
            features=np.array([[-50, -60], [-60, -50]]),
            floor_ids=np.array([0, 0]),
            meta={},
        )

        with pytest.raises(ValueError, match="Floor 5 not found"):
            db.get_floor_mask(5)

    def test_filter_by_floor(self):
        """Test filtering database to a single floor."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0], [10, 0], [0, 5], [5, 5]]),
            features=np.array(
                [[-50, -60], [-60, -50], [-55, -55], [-65, -65], [-70, -70]]
            ),
            floor_ids=np.array([0, 0, 1, 1, 2]),
            meta={"building": "test"},
        )

        # Filter to floor 1
        db_floor1 = db.filter_by_floor(1)

        assert db_floor1.n_reference_points == 2
        assert db_floor1.n_floors == 1
        np.testing.assert_array_equal(db_floor1.floor_ids, [1, 1])
        np.testing.assert_array_equal(db_floor1.locations, [[10, 0], [0, 5]])
        assert db_floor1.meta["building"] == "test"  # Metadata copied

    def test_repr(self):
        """Test string representation of database."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [10, 0], [0, 10]]),
            features=np.array([[-50, -60], [-60, -50], [-55, -55]]),
            floor_ids=np.array([0, 0, 1]),
            meta={},
        )

        repr_str = repr(db)
        assert "FingerprintDatabase" in repr_str
        assert "n_rps=3" in repr_str
        assert "n_features=2" in repr_str
        assert "location_dim=2" in repr_str
        assert "floors=[0, 1]" in repr_str


class TestTypeAliases:
    """Test type aliases are properly defined."""

    def test_location_is_ndarray(self):
        """Test that Location is an alias for ndarray."""
        loc: Location = np.array([1.0, 2.0])
        assert isinstance(loc, np.ndarray)

    def test_fingerprint_is_ndarray(self):
        """Test that Fingerprint is an alias for ndarray."""
        fp: Fingerprint = np.array([-50.0, -60.0, -70.0])
        assert isinstance(fp, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


