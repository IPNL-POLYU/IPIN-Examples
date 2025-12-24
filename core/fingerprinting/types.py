"""Type definitions and data structures for fingerprint-based localization.

This module defines the core data structures used throughout Chapter 5
(Fingerprinting) of the book, including the FingerprintDatabase class
and related type aliases.

Author: Li-Ta Hsu
Date: 2024
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Type aliases for clarity and documentation
Location = np.ndarray  # Shape (d,), typically d=2 or d=3 (x, y) or (x, y, z)
Fingerprint = np.ndarray  # Shape (N,), feature vector (e.g., RSS from N APs)


@dataclass
class FingerprintDatabase:
    """
    Fingerprint database for positioning with multi-floor support.

    This data structure represents the offline survey / radio map used in
    Chapter 5 fingerprinting methods. It stores reference point (RP)
    locations, their corresponding fingerprint feature vectors, and floor
    labels for multi-floor buildings.

    Attributes:
        locations: Reference point coordinates, shape (M, d).
                   Each row is a location vector x_i where i=1..M.
                   Typically d=2 for (x, y) or d=3 for (x, y, z).
        features: Fingerprint feature vectors. Supports two formats:
                  - Single sample: shape (M, N) - one fingerprint per RP
                  - Multiple samples: shape (M, S, N) - S samples per RP
                  Each feature vector corresponds to location x_i.
                  Typically N = number of access points (APs) or signal sources.
                  Example: RSS values [RSSI_AP1, RSSI_AP2, ..., RSSI_APN].
                  For multiple samples, axis 1 indexes repeated measurements.
        floor_ids: Floor labels for each reference point, shape (M,).
                   Integer floor identifiers (e.g., 0, 1, 2 for floors 0-2).
                   Used to constrain searches to specific floors in multi-floor
                   buildings.
        meta: Metadata dictionary containing auxiliary information.
              Recommended keys:
                  - 'ap_ids': list of AP identifiers (length N)
                  - 'ap_positions': AP coordinates, shape (N, d) or None
                  - 'building_id': string identifier
                  - 'coordinate_frame': 'ENU', 'local', etc.
                  - 'floor_heights': dict mapping floor_id -> height in meters
                  - 'survey_date': timestamp or string
                  - 'unit': 'dBm' for RSS, 'm' for locations, etc.

    Examples:
        >>> # Create a simple 2D database with 3 RPs on floor 0 (single sample)
        >>> db = FingerprintDatabase(
        ...     locations=np.array([[0, 0], [10, 0], [10, 10]]),
        ...     features=np.array([[-50, -60, -70], [-60, -50, -80], [-70, -80, -50]]),
        ...     floor_ids=np.array([0, 0, 0]),
        ...     meta={'ap_ids': ['AP1', 'AP2', 'AP3'], 'unit': 'dBm'}
        ... )
        >>> print(f"Database has {db.n_reference_points} RPs on {db.n_floors} floor(s)")
        Database has 3 RPs on 1 floor(s)
        
        >>> # Create database with multiple samples per RP (shape: M, S, N)
        >>> features_multi = np.array([
        ...     [[-50, -60, -70], [-51, -59, -71], [-49, -61, -69]],  # RP1: 3 samples
        ...     [[-60, -50, -80], [-61, -51, -79], [-59, -49, -81]],  # RP2: 3 samples
        ... ])  # shape: (2, 3, 3) = (2 RPs, 3 samples, 3 APs)
        >>> db_multi = FingerprintDatabase(
        ...     locations=np.array([[0, 0], [10, 0]]),
        ...     features=features_multi,
        ...     floor_ids=np.array([0, 0]),
        ...     meta={'ap_ids': ['AP1', 'AP2', 'AP3'], 'unit': 'dBm', 'n_samples_per_rp': 3}
        ... )

    Notes:
        - All reference points must have the same dimensionality d.
        - All fingerprints must have the same number of features N.
        - For multi-sample format, all RPs must have same number of samples S.
        - floor_ids are required (not optional) to support multi-floor scenarios.
        - Missing or unavailable AP measurements can be represented as NaN.
          Distance and likelihood computations will handle missing values by
          computing only over overlapping (non-NaN) dimensions.
    """

    locations: np.ndarray
    features: np.ndarray
    floor_ids: np.ndarray
    meta: dict

    def __post_init__(self) -> None:
        """Validate data structure consistency after initialization."""
        # Check that all arrays are NumPy arrays
        if not isinstance(self.locations, np.ndarray):
            raise TypeError("locations must be a NumPy ndarray")
        if not isinstance(self.features, np.ndarray):
            raise TypeError("features must be a NumPy ndarray")
        if not isinstance(self.floor_ids, np.ndarray):
            raise TypeError("floor_ids must be a NumPy ndarray")

        # Check dimensions
        if self.locations.ndim != 2:
            raise ValueError(
                f"locations must be 2D array (M, d), got shape {self.locations.shape}"
            )
        if self.features.ndim not in [2, 3]:
            raise ValueError(
                f"features must be 2D (M, N) or 3D (M, S, N) array, "
                f"got shape {self.features.shape}"
            )
        if self.floor_ids.ndim != 1:
            raise ValueError(
                f"floor_ids must be 1D array (M,), got shape {self.floor_ids.shape}"
            )

        # Check consistency: all must have same number of reference points M
        M_loc = self.locations.shape[0]
        M_feat = self.features.shape[0]
        M_floor = self.floor_ids.shape[0]

        if not (M_loc == M_feat == M_floor):
            raise ValueError(
                f"Inconsistent number of reference points: "
                f"locations={M_loc}, features={M_feat}, floor_ids={M_floor}"
            )

        # Check that floor_ids are integers
        if not np.issubdtype(self.floor_ids.dtype, np.integer):
            raise TypeError(
                f"floor_ids must have integer dtype, got {self.floor_ids.dtype}"
            )

        # Check for NaN values in locations (not allowed)
        if np.any(np.isnan(self.locations)):
            raise ValueError("locations contain NaN values (not allowed)")
        
        # Note: NaN values in features are allowed (represent missing AP readings)

    @property
    def n_reference_points(self) -> int:
        """Number of reference points (M) in the database."""
        return self.locations.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (N) per fingerprint (e.g., number of APs)."""
        return self.features.shape[-1]  # Last dimension is always N

    @property
    def n_samples_per_rp(self) -> Optional[int]:
        """
        Number of samples (S) per reference point.
        
        Returns:
            int: Number of samples if features is 3D (M, S, N).
            None: If features is 2D (M, N) indicating single sample.
        """
        return self.features.shape[1] if self.features.ndim == 3 else None

    @property
    def has_multiple_samples(self) -> bool:
        """True if database contains multiple samples per RP (3D features)."""
        return self.features.ndim == 3

    @property
    def location_dim(self) -> int:
        """Dimensionality (d) of location vectors (typically 2 or 3)."""
        return self.locations.shape[1]

    @property
    def n_floors(self) -> int:
        """Number of unique floors in the database."""
        return len(np.unique(self.floor_ids))

    @property
    def floor_list(self) -> np.ndarray:
        """Sorted array of unique floor identifiers."""
        return np.sort(np.unique(self.floor_ids))

    def get_floor_mask(self, floor_id: int) -> np.ndarray:
        """
        Get boolean mask for reference points on a specific floor.

        Args:
            floor_id: Floor identifier (must exist in database).

        Returns:
            Boolean mask of shape (M,), True for RPs on the specified floor.

        Raises:
            ValueError: If floor_id does not exist in the database.

        Examples:
            >>> mask = db.get_floor_mask(floor_id=0)
            >>> floor0_locations = db.locations[mask]
        """
        if floor_id not in self.floor_ids:
            raise ValueError(
                f"Floor {floor_id} not found in database. "
                f"Available floors: {self.floor_list}"
            )
        return self.floor_ids == floor_id

    def filter_by_floor(self, floor_id: int) -> "FingerprintDatabase":
        """
        Create a new database containing only RPs from a specific floor.

        Args:
            floor_id: Floor identifier.

        Returns:
            New FingerprintDatabase with only the specified floor's data.

        Examples:
            >>> db_floor0 = db.filter_by_floor(floor_id=0)
        """
        mask = self.get_floor_mask(floor_id)
        return FingerprintDatabase(
            locations=self.locations[mask],
            features=self.features[mask],
            floor_ids=self.floor_ids[mask],
            meta=self.meta.copy(),  # Shallow copy of metadata
        )

    def get_mean_features(self) -> np.ndarray:
        """
        Get mean features across samples.
        
        Handles NaN values (missing AP readings) using nanmean, which
        computes the mean while ignoring NaN values.
        
        Returns:
            Mean feature array of shape (M, N).
            If single-sample format, returns features as-is.
            If multi-sample format, returns mean over samples axis (ignoring NaN).
        """
        if self.has_multiple_samples:
            return np.nanmean(self.features, axis=1)  # Average over S, ignore NaN
        else:
            return self.features

    def get_std_features(self, min_std: float = 0.0) -> np.ndarray:
        """
        Get standard deviation of features across samples.
        
        Handles NaN values (missing AP readings) using nanstd, which
        computes the standard deviation while ignoring NaN values.
        
        Args:
            min_std: Minimum std to return (floor for numerical stability).
        
        Returns:
            Std array of shape (M, N).
            If single-sample format, returns array filled with min_std.
            If multi-sample format, returns std over samples axis (ignoring NaN).
        """
        if self.has_multiple_samples:
            stds = np.nanstd(self.features, axis=1, ddof=1)  # Sample std over S, ignore NaN
            # Apply floor
            stds = np.maximum(stds, min_std)
            # If all samples at an RP for a feature are NaN, nanstd returns NaN
            # Replace those with min_std
            stds = np.where(np.isnan(stds), min_std, stds)
            return stds
        else:
            # Single sample: return min_std everywhere
            return np.full((self.n_reference_points, self.n_features), min_std)

    def __repr__(self) -> str:
        """Readable string representation."""
        samples_str = f", samples_per_rp={self.n_samples_per_rp}" if self.has_multiple_samples else ""
        return (
            f"FingerprintDatabase("
            f"n_rps={self.n_reference_points}, "
            f"n_features={self.n_features}{samples_str}, "
            f"location_dim={self.location_dim}, "
            f"floors={self.floor_list.tolist()})"
        )

