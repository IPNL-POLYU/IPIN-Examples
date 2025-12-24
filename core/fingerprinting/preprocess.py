"""Preprocessing utilities for fingerprint-based positioning.

This module implements preprocessing techniques discussed in Chapter 5 of the book,
including scan averaging to reduce measurement noise and feature normalization to
mitigate device-specific offsets.

Key preprocessing steps:
    - Scan averaging: Reduce short-term fading by averaging multiple RSS samples
    - Normalization: Subtract mean/divide by std to handle device calibration differences
    - Outlier removal: Filter anomalous measurements before averaging

Author: Li-Ta Hsu
Date: December 2024
"""

from typing import Literal, Optional, Tuple

import numpy as np


def average_scans(
    scans: np.ndarray,
    method: Literal["mean", "median", "trimmed_mean"] = "mean",
    trim_percent: float = 0.1,
) -> np.ndarray:
    """
    Average multiple RSS scans to reduce measurement noise.

    This function implements scan averaging as discussed in Chapter 5, which
    helps mitigate short-term fading effects (e.g., Rayleigh fading) by
    averaging multiple independent measurements. The book notes that averaging
    S samples reduces variance by a factor of ~sqrt(S).

    Args:
        scans: RSS measurements, shape (S, N) where S is number of scans,
               N is number of features (APs). Can contain NaN for missing APs.
        method: Averaging method:
                - "mean": Arithmetic mean (default, optimal for Gaussian noise)
                - "median": Median (robust to outliers)
                - "trimmed_mean": Mean after removing top/bottom percentiles
        trim_percent: Percentage to trim from each end for trimmed_mean (0.0-0.5).

    Returns:
        Averaged fingerprint, shape (N,). NaN if all scans for an AP are NaN.

    Raises:
        ValueError: If scans is not 2D, trim_percent is invalid, or method is unknown.

    Examples:
        >>> # Average 5 scans with Gaussian noise
        >>> scans = np.array([
        ...     [-50, -60, -70],
        ...     [-52, -58, -72],
        ...     [-48, -62, -68],
        ...     [-51, -59, -71],
        ...     [-49, -61, -69],
        ... ])
        >>> avg = average_scans(scans, method="mean")
        >>> print(avg)  # [-50.0, -60.0, -70.0]

        >>> # Average with missing values (NaN)
        >>> scans_missing = np.array([
        ...     [-50, np.nan, -70],
        ...     [-52, -58, -72],
        ...     [-48, -62, np.nan],
        ... ])
        >>> avg_missing = average_scans(scans_missing)
        >>> # Returns [-50, -60, -71] using available samples

        >>> # Robust averaging with median (outlier rejection)
        >>> scans_outlier = np.array([
        ...     [-50, -60, -70],
        ...     [-51, -59, -71],
        ...     [-20, -65, -72],  # Outlier in AP1
        ...     [-49, -58, -69],
        ... ])
        >>> avg_robust = average_scans(scans_outlier, method="median")

    References:
        Chapter 5, Section 5.1: Discusses averaging multiple scans to reduce
        measurement variance and mitigate short-term fading effects.
    """
    # Validate input
    if scans.ndim != 2:
        raise ValueError(f"scans must be 2D array (S, N), got shape {scans.shape}")

    if method == "mean":
        # Arithmetic mean, ignoring NaN values
        # For Gaussian noise, this is the optimal estimator
        return np.nanmean(scans, axis=0)

    elif method == "median":
        # Median is robust to outliers
        # More robust than mean but less efficient for Gaussian noise
        return np.nanmedian(scans, axis=0)

    elif method == "trimmed_mean":
        # Trimmed mean: remove extreme values, then average
        # Good balance between robustness and efficiency
        if not 0.0 <= trim_percent < 0.5:
            raise ValueError(
                f"trim_percent must be in [0.0, 0.5), got {trim_percent}"
            )

        S, N = scans.shape
        result = np.zeros(N)

        for j in range(N):
            # Get valid (non-NaN) values for this feature
            valid_values = scans[:, j][~np.isnan(scans[:, j])]

            if len(valid_values) == 0:
                result[j] = np.nan
                continue

            # Compute number of values to trim from each end
            n_trim = int(len(valid_values) * trim_percent)

            if n_trim == 0 or len(valid_values) <= 2 * n_trim:
                # Not enough samples to trim, use mean
                result[j] = np.mean(valid_values)
            else:
                # Sort, trim, and average
                sorted_values = np.sort(valid_values)
                trimmed = sorted_values[n_trim:-n_trim]
                result[j] = np.mean(trimmed)

        return result

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'mean', 'median', or 'trimmed_mean'."
        )


def normalize_fingerprint(
    z: np.ndarray,
    method: Literal["zscore", "minmax", "none"] = "zscore",
    ref_mean: Optional[np.ndarray] = None,
    ref_std: Optional[np.ndarray] = None,
    ref_min: Optional[np.ndarray] = None,
    ref_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize fingerprint features to mitigate device-specific offsets.

    This function implements feature normalization as discussed in Chapter 5,
    which helps reduce the impact of device calibration differences. The book
    notes that different devices may report RSS values with systematic offsets
    (e.g., ±5 dBm), and normalization can improve cross-device performance.

    Args:
        z: Fingerprint vector, shape (N,). Can contain NaN for missing APs.
        method: Normalization method:
                - "zscore": Standardize to zero mean, unit variance (default)
                  z_norm = (z - mean) / std
                - "minmax": Scale to [0, 1] range
                  z_norm = (z - min) / (max - min)
                - "none": No normalization (pass through)
        ref_mean: Reference mean for z-score (computed from z if None).
        ref_std: Reference std for z-score (computed from z if None).
        ref_min: Reference min for minmax (computed from z if None).
        ref_max: Reference max for minmax (computed from z if None).

    Returns:
        Tuple of (normalized_fingerprint, normalization_params):
            - normalized_fingerprint: shape (N,), same as input
            - normalization_params: dict with keys 'method', 'mean', 'std', etc.

    Raises:
        ValueError: If z is not 1D or method is unknown.

    Examples:
        >>> # Z-score normalization (zero mean, unit variance)
        >>> z = np.array([-50, -60, -70, -80])
        >>> z_norm, params = normalize_fingerprint(z, method="zscore")
        >>> print(z_norm)  # [1.34, 0.45, -0.45, -1.34]
        >>> print(params)  # {'method': 'zscore', 'mean': -65.0, 'std': 11.18}

        >>> # Minmax normalization (scale to [0, 1])
        >>> z_minmax, params = normalize_fingerprint(z, method="minmax")
        >>> print(z_minmax)  # [1.0, 0.67, 0.33, 0.0]

        >>> # Use reference statistics (e.g., from training database)
        >>> z_new = np.array([-55, -65, -75, -85])
        >>> z_norm_new, _ = normalize_fingerprint(
        ...     z_new, method="zscore",
        ...     ref_mean=params['mean'], ref_std=params['std']
        ... )

        >>> # Handles missing values (NaN)
        >>> z_missing = np.array([-50, np.nan, -70, -80])
        >>> z_norm_missing, _ = normalize_fingerprint(z_missing)
        >>> # NaN values remain NaN after normalization

    References:
        Chapter 5, Section 5.1: Discusses feature normalization to handle
        device heterogeneity and calibration differences.
    """
    # Validate input
    if z.ndim != 1:
        raise ValueError(f"z must be 1D array (N,), got shape {z.shape}")

    if method == "none":
        # No normalization
        return z.copy(), {"method": "none"}

    elif method == "zscore":
        # Z-score normalization: (z - mean) / std
        # Compute statistics from non-NaN values
        if ref_mean is None:
            ref_mean = np.nanmean(z)
        else:
            # Ensure ref_mean is scalar or convert array to scalar
            ref_mean = np.asarray(ref_mean)
            if ref_mean.ndim == 0:
                ref_mean = float(ref_mean)
            elif ref_mean.shape == z.shape:
                # Per-feature normalization (ref_mean is array)
                pass
            else:
                raise ValueError(f"ref_mean shape {ref_mean.shape} incompatible with z shape {z.shape}")
        
        if ref_std is None:
            ref_std = np.nanstd(z, ddof=1)
            if ref_std == 0 or np.isnan(ref_std):
                ref_std = 1.0  # Avoid division by zero
        else:
            # Ensure ref_std is scalar or convert array to scalar
            ref_std = np.asarray(ref_std)
            if ref_std.ndim == 0:
                ref_std = float(ref_std)
            elif ref_std.shape == z.shape:
                # Per-feature normalization (ref_std is array)
                # Replace zeros/NaNs with 1.0
                ref_std = np.where((ref_std == 0) | np.isnan(ref_std), 1.0, ref_std)
            else:
                raise ValueError(f"ref_std shape {ref_std.shape} incompatible with z shape {z.shape}")

        # Normalize
        z_norm = (z - ref_mean) / ref_std

        # Convert to scalar for params if needed
        if isinstance(ref_mean, np.ndarray):
            params = {
                "method": "zscore",
                "mean": ref_mean.copy(),
                "std": ref_std.copy() if isinstance(ref_std, np.ndarray) else ref_std,
            }
        else:
            params = {
                "method": "zscore",
                "mean": float(ref_mean),
                "std": float(ref_std),
            }

        return z_norm, params

    elif method == "minmax":
        # Min-max normalization: (z - min) / (max - min)
        if ref_min is None:
            ref_min = np.nanmin(z)
        else:
            ref_min = np.asarray(ref_min)
            if ref_min.ndim > 0 and ref_min.shape != z.shape:
                raise ValueError(f"ref_min shape {ref_min.shape} incompatible with z shape {z.shape}")
        
        if ref_max is None:
            ref_max = np.nanmax(z)
        else:
            ref_max = np.asarray(ref_max)
            if ref_max.ndim > 0 and ref_max.shape != z.shape:
                raise ValueError(f"ref_max shape {ref_max.shape} incompatible with z shape {z.shape}")

        # Avoid division by zero
        range_val = ref_max - ref_min
        if np.isscalar(range_val):
            if range_val == 0 or np.isnan(range_val):
                range_val = 1.0
        else:
            range_val = np.where((range_val == 0) | np.isnan(range_val), 1.0, range_val)

        # Normalize
        z_norm = (z - ref_min) / range_val

        # Convert to scalar/array for params
        if isinstance(ref_min, np.ndarray) and ref_min.ndim > 0:
            params = {
                "method": "minmax",
                "min": ref_min.copy(),
                "max": ref_max.copy(),
                "range": range_val.copy() if isinstance(range_val, np.ndarray) else range_val,
            }
        else:
            params = {
                "method": "minmax",
                "min": float(ref_min),
                "max": float(ref_max),
                "range": float(range_val),
            }

        return z_norm, params

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'zscore', 'minmax', or 'none'."
        )


def preprocess_query(
    scans: np.ndarray,
    averaging_method: Literal["mean", "median", "trimmed_mean"] = "mean",
    normalization_method: Literal["zscore", "minmax", "none"] = "none",
    ref_mean: Optional[np.ndarray] = None,
    ref_std: Optional[np.ndarray] = None,
    ref_min: Optional[np.ndarray] = None,
    ref_max: Optional[np.ndarray] = None,
    trim_percent: float = 0.1,
) -> Tuple[np.ndarray, dict]:
    """
    Preprocess query fingerprint: averaging + normalization.

    This is a convenience function that combines scan averaging and normalization
    into a single pipeline, as commonly used in Chapter 5 fingerprinting methods.

    Args:
        scans: RSS measurements, shape (S, N) for multiple scans or (N,) for single scan.
        averaging_method: Method for averaging scans (if scans is 2D).
        normalization_method: Method for normalization.
        ref_mean: Reference mean for z-score normalization.
        ref_std: Reference std for z-score normalization.
        ref_min: Reference min for minmax normalization.
        ref_max: Reference max for minmax normalization.
        trim_percent: Trim percentage for trimmed_mean averaging.

    Returns:
        Tuple of (preprocessed_fingerprint, preprocessing_info):
            - preprocessed_fingerprint: shape (N,)
            - preprocessing_info: dict with 'averaging' and 'normalization' sub-dicts

    Examples:
        >>> # Multiple scans, average then normalize
        >>> scans = np.array([
        ...     [-50, -60, -70],
        ...     [-52, -58, -72],
        ...     [-48, -62, -68],
        ... ])
        >>> z_preprocessed, info = preprocess_query(
        ...     scans,
        ...     averaging_method="mean",
        ...     normalization_method="zscore"
        ... )

        >>> # Single scan, just normalize
        >>> single_scan = np.array([-50, -60, -70])
        >>> z_norm, info = preprocess_query(
        ...     single_scan,
        ...     normalization_method="zscore"
        ... )

    References:
        Chapter 5, Section 5.1: Discusses preprocessing pipeline for fingerprinting.
    """
    # Step 1: Averaging (if multiple scans provided)
    if scans.ndim == 2:
        z_avg = average_scans(scans, method=averaging_method, trim_percent=trim_percent)
        avg_info = {"method": averaging_method, "n_scans": scans.shape[0]}
    elif scans.ndim == 1:
        z_avg = scans.copy()
        avg_info = {"method": "single_scan", "n_scans": 1}
    else:
        raise ValueError(f"scans must be 1D or 2D array, got shape {scans.shape}")

    # Step 2: Normalization
    z_norm, norm_params = normalize_fingerprint(
        z_avg,
        method=normalization_method,
        ref_mean=ref_mean,
        ref_std=ref_std,
        ref_min=ref_min,
        ref_max=ref_max,
    )

    # Combine info
    info = {"averaging": avg_info, "normalization": norm_params}

    return z_norm, info


def compute_normalization_params(
    fingerprints: np.ndarray, method: Literal["zscore", "minmax"] = "zscore"
) -> dict:
    """
    Compute normalization parameters from a set of fingerprints (e.g., database).

    This function computes statistics from a reference dataset (typically the
    offline survey database) that can be used to normalize query fingerprints
    consistently.

    Args:
        fingerprints: Fingerprint array, shape (M, N) where M is number of samples,
                      N is number of features. Can contain NaN for missing APs.
        method: Normalization method ('zscore' or 'minmax').

    Returns:
        Dictionary with normalization parameters:
            - For 'zscore': {'method': 'zscore', 'mean': (N,), 'std': (N,)}
            - For 'minmax': {'method': 'minmax', 'min': (N,), 'max': (N,)}

    Examples:
        >>> # Compute normalization params from database
        >>> db_features = db.get_mean_features()  # (M, N)
        >>> norm_params = compute_normalization_params(db_features, method="zscore")
        >>> print(norm_params['mean'].shape)  # (N,)
        >>> print(norm_params['std'].shape)   # (N,)

        >>> # Use params to normalize query
        >>> z_query = np.array([-55, -65, -75])
        >>> z_norm, _ = normalize_fingerprint(
        ...     z_query, method="zscore",
        ...     ref_mean=norm_params['mean'],
        ...     ref_std=norm_params['std']
        ... )

    References:
        Chapter 5, Section 5.1: Discusses computing statistics from offline database
        for consistent normalization.
    """
    if fingerprints.ndim != 2:
        raise ValueError(
            f"fingerprints must be 2D array (M, N), got shape {fingerprints.shape}"
        )

    if method == "zscore":
        # Compute mean and std across all RPs (axis=0), ignoring NaN
        mean = np.nanmean(fingerprints, axis=0)
        std = np.nanstd(fingerprints, axis=0, ddof=1)

        # Replace zero/NaN stds with 1.0 to avoid division by zero
        std = np.where((std == 0) | np.isnan(std), 1.0, std)

        return {"method": "zscore", "mean": mean, "std": std}

    elif method == "minmax":
        # Compute min and max across all RPs (axis=0), ignoring NaN
        min_val = np.nanmin(fingerprints, axis=0)
        max_val = np.nanmax(fingerprints, axis=0)

        # Compute range
        range_val = max_val - min_val

        # Replace zero/NaN ranges with 1.0
        range_val = np.where((range_val == 0) | np.isnan(range_val), 1.0, range_val)

        return {
            "method": "minmax",
            "min": min_val,
            "max": max_val,
            "range": range_val,
        }

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'minmax'.")

