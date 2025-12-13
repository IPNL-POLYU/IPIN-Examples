"""Dataset utilities for loading, saving, and validating fingerprint databases.

This module provides I/O functions for FingerprintDatabase objects, supporting
multiple file formats (NPZ, HDF5) and validation utilities for data quality checks.

Author: Navigation Engineer
Date: 2024
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .types import FingerprintDatabase


def load_fingerprint_database(
    data_dir: Union[str, Path], format: str = "npz"
) -> FingerprintDatabase:
    """
    Load a fingerprint database from disk.

    Expected directory structure for NPZ format:
        data_dir/
        ├── locations.npy      # (M, d) array
        ├── features.npy       # (M, N) array
        ├── floor_ids.npy      # (M,) array
        └── metadata.json      # dict

    Args:
        data_dir: Path to directory containing database files.
        format: File format, currently only 'npz' is supported.

    Returns:
        FingerprintDatabase object loaded from disk.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data validation fails.

    Examples:
        >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
        >>> print(db)
        FingerprintDatabase(n_rps=75, n_features=6, location_dim=2, floors=[0, 1, 2])
    """
    data_dir = Path(data_dir)

    if format == "npz":
        return _load_npz_database(data_dir)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz'.")


def _load_npz_database(data_dir: Path) -> FingerprintDatabase:
    """Load database from individual .npy files and metadata.json."""
    # Required files
    locations_file = data_dir / "locations.npy"
    features_file = data_dir / "features.npy"
    floor_ids_file = data_dir / "floor_ids.npy"
    metadata_file = data_dir / "metadata.json"

    # Check existence
    for filepath in [locations_file, features_file, floor_ids_file, metadata_file]:
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")

    # Load arrays
    locations = np.load(locations_file)
    features = np.load(features_file)
    floor_ids = np.load(floor_ids_file)

    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Create database (validation happens in __post_init__)
    db = FingerprintDatabase(
        locations=locations, features=features, floor_ids=floor_ids, meta=meta
    )

    return db


def save_fingerprint_database(
    db: FingerprintDatabase, data_dir: Union[str, Path], format: str = "npz"
) -> None:
    """
    Save a fingerprint database to disk.

    Creates directory structure:
        data_dir/
        ├── locations.npy
        ├── features.npy
        ├── floor_ids.npy
        └── metadata.json

    Args:
        db: FingerprintDatabase to save.
        data_dir: Destination directory (will be created if it doesn't exist).
        format: File format, currently only 'npz' is supported.

    Examples:
        >>> save_fingerprint_database(db, 'data/sim/my_database')
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        _save_npz_database(db, data_dir)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz'.")


def _save_npz_database(db: FingerprintDatabase, data_dir: Path) -> None:
    """Save database as individual .npy files and metadata.json."""
    # Save arrays
    np.save(data_dir / "locations.npy", db.locations)
    np.save(data_dir / "features.npy", db.features)
    np.save(data_dir / "floor_ids.npy", db.floor_ids)

    # Save metadata
    with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(db.meta, f, indent=2)


def validate_database(db: FingerprintDatabase, strict: bool = True) -> dict:
    """
    Perform comprehensive validation checks on a fingerprint database.

    Validation checks include:
    - Data type and shape consistency (already enforced by __post_init__)
    - Floor coverage: at least 1 RP per floor
    - Feature statistics: check for constant features (no variance)
    - Location coverage: check for duplicate locations
    - Optional strict checks: verify reasonable value ranges

    Args:
        db: FingerprintDatabase to validate.
        strict: If True, perform additional checks on value ranges.

    Returns:
        Dictionary with validation results and warnings:
            {
                'valid': bool,
                'errors': list of error messages,
                'warnings': list of warning messages,
                'stats': dict with database statistics
            }

    Examples:
        >>> result = validate_database(db)
        >>> if not result['valid']:
        ...     print("Errors:", result['errors'])
    """
    errors = []
    warnings = []
    stats = {}

    # Basic statistics
    stats["n_reference_points"] = db.n_reference_points
    stats["n_features"] = db.n_features
    stats["location_dim"] = db.location_dim
    stats["n_floors"] = db.n_floors
    stats["floor_ids"] = db.floor_list.tolist()

    # Check 1: Floor coverage
    rps_per_floor = {
        int(floor_id): np.sum(db.floor_ids == floor_id) for floor_id in db.floor_list
    }
    stats["rps_per_floor"] = rps_per_floor

    for floor_id, count in rps_per_floor.items():
        if count == 0:
            errors.append(f"Floor {floor_id} has no reference points")
        elif count < 3:
            warnings.append(
                f"Floor {floor_id} has only {count} RP(s); "
                f"may be insufficient for localization"
            )

    # Check 2: Feature variance (detect constant features)
    feature_std = np.std(db.features, axis=0)
    constant_features = np.where(feature_std < 1e-6)[0]
    if len(constant_features) > 0:
        warnings.append(
            f"Features {constant_features.tolist()} have zero variance "
            f"(constant across all RPs)"
        )
    stats["feature_std_min"] = float(np.min(feature_std))
    stats["feature_std_max"] = float(np.max(feature_std))

    # Check 3: Duplicate locations
    unique_locations = np.unique(db.locations, axis=0)
    n_duplicates = db.n_reference_points - len(unique_locations)
    if n_duplicates > 0:
        warnings.append(
            f"Found {n_duplicates} duplicate location(s); "
            f"multiple RPs at same coordinates"
        )

    # Check 4: Strict value range checks (optional)
    if strict:
        # RSS values typically in range [-100, 0] dBm
        if "unit" in db.meta and db.meta["unit"] == "dBm":
            if np.any(db.features > 0):
                warnings.append("Some RSS values are positive (unusual for dBm)")
            if np.any(db.features < -120):
                warnings.append("Some RSS values below -120 dBm (very weak signal)")

        # Location coordinates should be finite
        if not np.all(np.isfinite(db.locations)):
            errors.append("Non-finite values detected in locations")

    # Overall validity
    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def print_database_summary(db: FingerprintDatabase) -> None:
    """
    Print a human-readable summary of the database.

    Args:
        db: FingerprintDatabase to summarize.

    Examples:
        >>> print_database_summary(db)
        Fingerprint Database Summary
        ============================
        Reference Points: 75
        Features per RP:  6
        Location Dim:     2
        Floors:           [0, 1, 2]
        ...
    """
    print("Fingerprint Database Summary")
    print("=" * 50)
    print(f"Reference Points: {db.n_reference_points}")
    print(f"Features per RP:  {db.n_features}")
    print(f"Location Dim:     {db.location_dim}")
    print(f"Floors:           {db.floor_list.tolist()}")
    print()

    # Per-floor breakdown
    print("Reference Points per Floor:")
    for floor_id in db.floor_list:
        count = np.sum(db.floor_ids == floor_id)
        print(f"  Floor {floor_id}: {count} RPs")
    print()

    # Feature statistics
    print("Feature Statistics:")
    print(f"  Mean (across all RPs): {np.mean(db.features, axis=0)}")
    print(f"  Std  (across all RPs): {np.std(db.features, axis=0)}")
    print()

    # Location bounds
    print("Location Bounds:")
    for dim in range(db.location_dim):
        min_val = np.min(db.locations[:, dim])
        max_val = np.max(db.locations[:, dim])
        print(f"  Dimension {dim}: [{min_val:.2f}, {max_val:.2f}]")
    print()

    # Metadata
    print("Metadata:")
    for key, value in db.meta.items():
        # Truncate long values for display
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        print(f"  {key}: {value_str}")

