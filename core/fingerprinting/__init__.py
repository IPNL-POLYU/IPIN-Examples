"""Fingerprint-based localization algorithms for Chapter 5.

This module implements deterministic and probabilistic fingerprinting methods
for indoor positioning, as described in Chapter 5 of the book:
Principles of Indoor Positioning and Indoor Navigation.

Main components:
    - FingerprintDatabase: Core data structure for radio maps
    - load/save functions: I/O utilities for databases
    - validate_database: Data quality checks
    - nn_localize, knn_localize: Deterministic fingerprinting (Eqs. 5.1-5.2)
    - map_localize, posterior_mean_localize: Probabilistic (Eqs. 5.3-5.5)

Example usage:
    >>> from core.fingerprinting import (
    ...     FingerprintDatabase,
    ...     load_fingerprint_database,
    ...     nn_localize,
    ...     knn_localize,
    ...     fit_gaussian_naive_bayes,
    ...     map_localize
    ... )
    >>> db = load_fingerprint_database('data/sim/ch5_wifi_fingerprint_grid')
    >>> z_query = np.array([-50, -60, -70])
    >>> 
    >>> # Deterministic
    >>> x_hat_nn = nn_localize(z_query, db, floor_id=0)
    >>> 
    >>> # Probabilistic
    >>> model = fit_gaussian_naive_bayes(db)
    >>> x_hat_map = map_localize(z_query, model, floor_id=0)

Author: Li-Ta Hsu
Date: 2024
"""

from .classification import (
    ClassificationLocalizer,
    fit_classifier,
    hierarchical_localize,
)
from .dataset import (
    load_fingerprint_database,
    print_database_summary,
    save_fingerprint_database,
    validate_database,
)
from .deterministic import distance, knn_localize, nn_localize, pairwise_distances
from .pattern_recognition import LinearRegressionLocalizer
from .preprocess import (
    average_scans,
    compute_normalization_params,
    normalize_fingerprint,
    preprocess_query,
)
from .probabilistic import (
    NaiveBayesFingerprintModel,
    fit_gaussian_naive_bayes,
    log_likelihood,
    log_posterior,
    map_localize,
    posterior_mean_localize,
)
from .types import Fingerprint, FingerprintDatabase, Location

__all__ = [
    # Core types
    "FingerprintDatabase",
    "Location",
    "Fingerprint",
    # Dataset I/O
    "load_fingerprint_database",
    "save_fingerprint_database",
    "validate_database",
    "print_database_summary",
    # Deterministic methods
    "distance",
    "pairwise_distances",
    "nn_localize",
    "knn_localize",
    # Probabilistic methods
    "NaiveBayesFingerprintModel",
    "fit_gaussian_naive_bayes",
    "log_likelihood",
    "log_posterior",
    "map_localize",
    "posterior_mean_localize",
    # Pattern recognition methods (regression)
    "LinearRegressionLocalizer",
    # Pattern recognition methods (classification)
    "ClassificationLocalizer",
    "fit_classifier",
    "hierarchical_localize",
    # Preprocessing
    "average_scans",
    "normalize_fingerprint",
    "preprocess_query",
    "compute_normalization_params",
]

__version__ = "0.1.0"

