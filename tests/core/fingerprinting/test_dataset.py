"""Unit tests for core.fingerprinting.dataset module.

Tests load/save functions and database validation.

Author: Navigation Engineer
Date: 2024
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.fingerprinting import (
    FingerprintDatabase,
    load_fingerprint_database,
    print_database_summary,
    save_fingerprint_database,
    validate_database,
)


@pytest.fixture
def sample_database():
    """Create a sample multi-floor database for testing."""
    return FingerprintDatabase(
        locations=np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
                [0.0, 5.0],
                [5.0, 5.0],
                [10.0, 5.0],
                [0.0, 10.0],
                [5.0, 10.0],
            ]
        ),
        features=np.array(
            [
                [-50, -60, -70],
                [-60, -50, -80],
                [-70, -80, -50],
                [-55, -65, -75],
                [-65, -55, -85],
                [-75, -85, -55],
                [-52, -62, -72],
                [-62, -52, -82],
            ]
        ),
        floor_ids=np.array([0, 0, 0, 1, 1, 1, 2, 2]),
        meta={
            "ap_ids": ["AP1", "AP2", "AP3"],
            "unit": "dBm",
            "building_id": "test_building",
            "coordinate_frame": "ENU",
        },
    )


class TestLoadSaveFunctions:
    """Test load/save functionality for fingerprint databases."""

    def test_save_and_load_roundtrip(self, sample_database):
        """Test that save followed by load preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save database
            save_path = Path(tmpdir) / "test_db"
            save_fingerprint_database(sample_database, save_path, format="npz")

            # Verify files were created
            assert (save_path / "locations.npy").exists()
            assert (save_path / "features.npy").exists()
            assert (save_path / "floor_ids.npy").exists()
            assert (save_path / "metadata.json").exists()

            # Load database
            loaded_db = load_fingerprint_database(save_path, format="npz")

            # Verify data integrity
            np.testing.assert_array_equal(
                loaded_db.locations, sample_database.locations
            )
            np.testing.assert_array_equal(loaded_db.features, sample_database.features)
            np.testing.assert_array_equal(
                loaded_db.floor_ids, sample_database.floor_ids
            )
            assert loaded_db.meta == sample_database.meta

            # Verify properties
            assert loaded_db.n_reference_points == sample_database.n_reference_points
            assert loaded_db.n_features == sample_database.n_features
            assert loaded_db.n_floors == sample_database.n_floors

    def test_save_creates_directory(self, sample_database):
        """Test that save creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "test_db"
            assert not save_path.exists()

            save_fingerprint_database(sample_database, save_path)

            assert save_path.exists()
            assert save_path.is_dir()

    def test_load_missing_file_error(self):
        """Test that loading from non-existent directory raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent"

            with pytest.raises(FileNotFoundError, match="Required file not found"):
                load_fingerprint_database(missing_path)

    def test_load_missing_locations_file(self, sample_database):
        """Test that missing locations.npy raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "incomplete_db"
            save_path.mkdir()

            # Save only features and floor_ids (missing locations)
            np.save(save_path / "features.npy", sample_database.features)
            np.save(save_path / "floor_ids.npy", sample_database.floor_ids)
            with open(save_path / "metadata.json", "w") as f:
                json.dump(sample_database.meta, f)

            with pytest.raises(FileNotFoundError, match="locations.npy"):
                load_fingerprint_database(save_path)

    def test_save_unsupported_format(self, sample_database):
        """Test that unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_db"

            with pytest.raises(ValueError, match="Unsupported format: hdf5"):
                save_fingerprint_database(sample_database, save_path, format="hdf5")

    def test_load_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "test_db"

            with pytest.raises(ValueError, match="Unsupported format: csv"):
                load_fingerprint_database(load_path, format="csv")


class TestValidateDatabase:
    """Test database validation functions."""

    def test_validate_valid_database(self, sample_database):
        """Test that a valid database passes all checks."""
        result = validate_database(sample_database, strict=True)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["stats"]["n_reference_points"] == 8
        assert result["stats"]["n_features"] == 3
        assert result["stats"]["n_floors"] == 3
        assert result["stats"]["floor_ids"] == [0, 1, 2]

    def test_validate_floor_coverage(self):
        """Test validation detects floors with insufficient RPs."""
        # Create database with sparse floor coverage
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0], [10, 0]]),
            features=np.array([[-50, -60], [-60, -50], [-55, -55]]),
            floor_ids=np.array([0, 0, 1]),  # Floor 1 has only 1 RP
            meta={},
        )

        result = validate_database(db, strict=False)

        assert result["valid"] is True  # Not an error, but warning
        assert len(result["warnings"]) > 0
        assert any("Floor 1 has only 1 RP" in w for w in result["warnings"])

    def test_validate_constant_feature(self):
        """Test validation detects constant (zero variance) features."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0], [10, 0]]),
            features=np.array(
                [
                    [-50, -60, -100],  # Third feature is constant
                    [-60, -50, -100],
                    [-55, -55, -100],
                ]
            ),
            floor_ids=np.array([0, 0, 0]),
            meta={},
        )

        result = validate_database(db, strict=False)

        assert result["valid"] is True  # Warning, not error
        assert any("zero variance" in w for w in result["warnings"])

    def test_validate_duplicate_locations(self):
        """Test validation detects duplicate reference point locations."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0], [5, 0]]),  # Duplicate at [5, 0]
            features=np.array([[-50, -60], [-60, -50], [-55, -55]]),
            floor_ids=np.array([0, 0, 0]),
            meta={},
        )

        result = validate_database(db, strict=False)

        assert result["valid"] is True  # Warning, not error
        assert any("duplicate location" in w for w in result["warnings"])

    def test_validate_rss_positive_warning(self):
        """Test strict validation warns about positive RSS values."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0]]),
            features=np.array([[10, -60], [-60, -50]]),  # Positive RSS (unusual)
            floor_ids=np.array([0, 0]),
            meta={"unit": "dBm"},
        )

        result = validate_database(db, strict=True)

        assert result["valid"] is True  # Warning, not error
        assert any("positive" in w.lower() for w in result["warnings"])

    def test_validate_rss_very_weak_warning(self):
        """Test strict validation warns about very weak RSS values."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [5, 0]]),
            features=np.array([[-130, -60], [-60, -50]]),  # Very weak signal
            floor_ids=np.array([0, 0]),
            meta={"unit": "dBm"},
        )

        result = validate_database(db, strict=True)

        assert result["valid"] is True  # Warning, not error
        assert any("below -120 dBm" in w for w in result["warnings"])

    def test_validate_non_finite_locations_error(self):
        """Test validation errors on non-finite location values."""
        db = FingerprintDatabase(
            locations=np.array([[0, 0], [np.inf, 0]]),  # Infinite coordinate
            features=np.array([[-50, -60], [-60, -50]]),
            floor_ids=np.array([0, 0]),
            meta={},
        )

        result = validate_database(db, strict=True)

        assert result["valid"] is False
        assert any("Non-finite values" in e for e in result["errors"])

    def test_validate_stats_returned(self, sample_database):
        """Test that validation returns complete statistics."""
        result = validate_database(sample_database, strict=False)

        stats = result["stats"]
        assert "n_reference_points" in stats
        assert "n_features" in stats
        assert "n_floors" in stats
        assert "floor_ids" in stats
        assert "rps_per_floor" in stats
        assert "feature_std_min" in stats
        assert "feature_std_max" in stats


class TestPrintDatabaseSummary:
    """Test summary printing function."""

    def test_print_summary_runs_without_error(self, sample_database, capsys):
        """Test that print_database_summary executes without errors."""
        # This should not raise any exceptions
        print_database_summary(sample_database)

        # Capture output
        captured = capsys.readouterr()

        # Verify key information appears in output
        assert "Fingerprint Database Summary" in captured.out
        assert "Reference Points: 8" in captured.out
        assert "Features per RP:  3" in captured.out
        assert "Location Dim:     2" in captured.out
        assert "Floors:           [0, 1, 2]" in captured.out
        assert "Floor 0:" in captured.out
        assert "Floor 1:" in captured.out
        assert "Floor 2:" in captured.out

    def test_print_summary_multifloor_breakdown(self, sample_database, capsys):
        """Test that summary shows per-floor breakdown."""
        print_database_summary(sample_database)
        captured = capsys.readouterr()

        # Verify per-floor RPs are shown
        assert "Floor 0: 3 RPs" in captured.out  # 3 RPs on floor 0
        assert "Floor 1: 3 RPs" in captured.out  # 3 RPs on floor 1
        assert "Floor 2: 2 RPs" in captured.out  # 2 RPs on floor 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


