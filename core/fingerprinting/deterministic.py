"""Deterministic fingerprinting methods for Chapter 5.

This module implements nearest-neighbor (NN) and k-nearest-neighbor (k-NN)
fingerprinting algorithms as described in Section 5.1 of the book.

Key equations:
    - Eq. (5.1): NN decision rule i* = argmin_i D(z, f_i)
    - Eq. (5.2): k-NN weighted average x̂ = Σ w_i x_i / Σ w_i

Author: Navigation Engineer
Date: 2024
"""

from typing import Optional

import numpy as np

from .types import Fingerprint, FingerprintDatabase, Location


def distance(z: np.ndarray, f: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute distance D(z, f) between query and reference fingerprint.

    This function implements the distance metric D(·, ·) used in Eq. (5.1)
    and Eq. (5.2) of Chapter 5.

    Args:
        z: Query fingerprint vector, shape (N,).
        f: Reference fingerprint vector, shape (N,).
        metric: Distance metric, either 'euclidean' or 'manhattan'.

    Returns:
        Distance value D(z, f) as a scalar float.

    Raises:
        ValueError: If metric is not supported or if z and f have different shapes.

    Examples:
        >>> z = np.array([-50, -60, -70])
        >>> f = np.array([-52, -58, -72])
        >>> d_eucl = distance(z, f, metric='euclidean')
        >>> d_manh = distance(z, f, metric='manhattan')
        >>> print(f"Euclidean: {d_eucl:.2f}, Manhattan: {d_manh:.2f}")
        Euclidean: 3.46, Manhattan: 6.00

    References:
        Chapter 5, Eqs. (5.1)-(5.2): Distance metrics for fingerprinting.
    """
    # Validate inputs
    if z.shape != f.shape:
        raise ValueError(
            f"Query and reference fingerprints must have same shape: "
            f"z.shape={z.shape}, f.shape={f.shape}"
        )

    # Compute distance based on metric
    if metric == "euclidean":
        return float(np.linalg.norm(z - f))
    elif metric == "manhattan":
        return float(np.sum(np.abs(z - f)))
    else:
        raise ValueError(
            f"Unsupported metric: '{metric}'. Use 'euclidean' or 'manhattan'."
        )


def pairwise_distances(
    z: np.ndarray, F: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute distances D(z, f_i) for all fingerprints f_i in F.

    This function evaluates the distance metric required in Eq. (5.1)
    across all reference fingerprints i = 1, ..., M.

    Args:
        z: Query fingerprint vector, shape (N,).
        F: Reference fingerprints matrix, shape (M, N).
           Each row F[i] is a reference fingerprint f_i.
        metric: Distance metric, either 'euclidean' or 'manhattan'.

    Returns:
        Array of distances, shape (M,), where element i is D(z, f_i).

    Raises:
        ValueError: If z and F have incompatible dimensions or metric is invalid.

    Examples:
        >>> z = np.array([-50, -60, -70])
        >>> F = np.array([[-52, -58, -72],
        ...               [-48, -62, -68],
        ...               [-55, -55, -75]])
        >>> distances = pairwise_distances(z, F, metric='euclidean')
        >>> print(distances)
        [3.46 4.47 7.07]

    References:
        Chapter 5, Eq. (5.1): Distance computation for all reference points.
    """
    # Validate dimensions
    if z.ndim != 1:
        raise ValueError(f"Query z must be 1D array, got shape {z.shape}")
    if F.ndim != 2:
        raise ValueError(f"Reference F must be 2D array (M, N), got shape {F.shape}")
    if z.shape[0] != F.shape[1]:
        raise ValueError(
            f"Incompatible dimensions: z has {z.shape[0]} features, "
            f"F has {F.shape[1]} features per row"
        )

    # Compute pairwise distances efficiently
    if metric == "euclidean":
        # Vectorized Euclidean distance: ||z - f_i||_2 for all i
        return np.linalg.norm(F - z, axis=1)
    elif metric == "manhattan":
        # Vectorized Manhattan distance: Σ |z_j - f_i,j| for all i
        return np.sum(np.abs(F - z), axis=1)
    else:
        raise ValueError(
            f"Unsupported metric: '{metric}'. Use 'euclidean' or 'manhattan'."
        )


def nn_localize(
    z: Fingerprint,
    db: FingerprintDatabase,
    metric: str = "euclidean",
    floor_id: Optional[int] = None,
) -> Location:
    """
    Nearest-neighbor (NN) deterministic fingerprinting.

    Implements Eq. (5.1) in Chapter 5:
        i* = argmin_{i=1..M} D(z, f_i)
    and returns x̂ = x_{i*}.

    If floor_id is provided, the search is constrained to reference points
    on that floor only. Otherwise, searches across all floors.

    Args:
        z: Query fingerprint vector, shape (N,).
        db: FingerprintDatabase containing M reference points.
        metric: Distance metric ('euclidean' or 'manhattan').
        floor_id: Optional floor constraint. If None, searches all floors.

    Returns:
        Estimated location x̂, shape (d,), where d is location_dim.

    Raises:
        ValueError: If floor_id is provided but doesn't exist in database,
                    or if query fingerprint dimension doesn't match database.

    Examples:
        >>> # Single-floor localization
        >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
        >>> z_query = np.array([-51, -61, -71])
        >>> x_hat = nn_localize(z_query, db, floor_id=0)
        >>> print(f"Estimated position: {x_hat}")

        >>> # Multi-floor search (no floor constraint)
        >>> x_hat_global = nn_localize(z_query, db, floor_id=None)

    References:
        Chapter 5, Eq. (5.1): Nearest-neighbor decision rule.
    """
    # Validate query dimension
    if z.shape[0] != db.n_features:
        raise ValueError(
            f"Query fingerprint has {z.shape[0]} features, "
            f"but database expects {db.n_features} features"
        )

    # Filter by floor if specified
    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        locations = db.locations[mask]
        features = db.features[mask]

        if len(locations) == 0:
            raise ValueError(
                f"Floor {floor_id} exists but has no reference points "
                f"(this should not happen in a valid database)"
            )
    else:
        locations = db.locations
        features = db.features

    # Compute distances to all reference points
    # Implements: D(z, f_i) for i = 1, ..., M
    distances = pairwise_distances(z, features, metric=metric)

    # Find nearest neighbor
    # Implements: i* = argmin_i D(z, f_i) from Eq. (5.1)
    i_star = np.argmin(distances)

    # Return location of nearest neighbor
    # Implements: x̂ = x_{i*} from Eq. (5.1)
    return locations[i_star]


def knn_localize(
    z: Fingerprint,
    db: FingerprintDatabase,
    k: int = 3,
    metric: str = "euclidean",
    weighting: str = "inverse_distance",
    eps: float = 1e-6,
    floor_id: Optional[int] = None,
) -> Location:
    """
    k-nearest-neighbor (k-NN) fingerprinting with weighted interpolation.

    Implements Eq. (5.2) in Chapter 5:
        x̂ = Σ_{i ∈ K(z)} w_i x_i / Σ_{i ∈ K(z)} w_i

    where K(z) is the set of k nearest neighbors, and weights w_i are
    typically defined as:
        w_i = 1 / (D(z, f_i) + ε)

    Args:
        z: Query fingerprint vector, shape (N,).
        db: FingerprintDatabase containing M reference points.
        k: Number of nearest neighbors to use (must be >= 1).
        metric: Distance metric ('euclidean' or 'manhattan').
        weighting: Weight computation method. Currently supports:
                   - 'inverse_distance': w_i = 1 / (D(z, f_i) + eps)
                   - 'uniform': w_i = 1 (simple average)
        eps: Small constant added to distances to avoid division by zero.
        floor_id: Optional floor constraint. If None, searches all floors.

    Returns:
        Estimated location x̂, shape (d,), weighted average of k-NN locations.

    Raises:
        ValueError: If k < 1, k > M, invalid weighting method, or dimension mismatch.

    Examples:
        >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
        >>> z_query = np.array([-51, -61, -71])
        >>> 
        >>> # Standard k-NN with k=3
        >>> x_hat = knn_localize(z_query, db, k=3, floor_id=0)
        >>> 
        >>> # k-NN with uniform weights (simple average)
        >>> x_hat_uniform = knn_localize(z_query, db, k=5, weighting='uniform')
        >>> 
        >>> # k-NN with inverse distance weighting (default)
        >>> x_hat_weighted = knn_localize(z_query, db, k=5, weighting='inverse_distance')

    References:
        Chapter 5, Eq. (5.2): k-NN weighted position estimate.
    """
    # Validate query dimension
    if z.shape[0] != db.n_features:
        raise ValueError(
            f"Query fingerprint has {z.shape[0]} features, "
            f"but database expects {db.n_features} features"
        )

    # Validate k
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")

    # Filter by floor if specified
    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        locations = db.locations[mask]
        features = db.features[mask]

        if len(locations) == 0:
            raise ValueError(
                f"Floor {floor_id} exists but has no reference points "
                f"(this should not happen in a valid database)"
            )
    else:
        locations = db.locations
        features = db.features

    # Check that k doesn't exceed available reference points
    M = len(locations)
    if k > M:
        raise ValueError(
            f"k={k} exceeds number of available reference points M={M}. "
            f"Use k <= {M}."
        )

    # Compute distances to all reference points
    # Implements: D(z, f_i) for i = 1, ..., M
    distances = pairwise_distances(z, features, metric=metric)

    # Find k nearest neighbors
    # Implements: K(z) = indices of k smallest D(z, f_i)
    k_indices = np.argpartition(distances, k - 1)[:k]

    # Get distances and locations for k-NN
    k_distances = distances[k_indices]
    k_locations = locations[k_indices]

    # Compute weights based on weighting scheme
    if weighting == "inverse_distance":
        # Implements: w_i = 1 / (D(z, f_i) + ε) from Eq. (5.2) discussion
        weights = 1.0 / (k_distances + eps)
    elif weighting == "uniform":
        # Uniform weights: simple average (all w_i = 1)
        weights = np.ones(k)
    else:
        raise ValueError(
            f"Unsupported weighting method: '{weighting}'. "
            f"Use 'inverse_distance' or 'uniform'."
        )

    # Normalize weights
    # Prepare for: Σ_{i ∈ K(z)} w_i x_i / Σ_{i ∈ K(z)} w_i
    weights_sum = np.sum(weights)

    # Compute weighted average position
    # Implements Eq. (5.2): x̂ = Σ w_i x_i / Σ w_i
    x_hat = np.sum(weights[:, np.newaxis] * k_locations, axis=0) / weights_sum

    return x_hat

