"""Fingerprint-based localization algorithms for Chapter 5.

This module implements deterministic and probabilistic fingerprinting methods
for indoor positioning, as described in Chapter 5 of the book:
Principles of Indoor Positioning and Indoor Navigation.

Main components:
    - FingerprintDatabase: Core data structure for radio maps
    - load/save functions: I/O utilities for databases
    - validate_database: Data quality checks

Example usage:
    >>> from core.fingerprinting import (
    ...     FingerprintDatabase,
    ...     load_fingerprint_database,
    ...     save_fingerprint_database
    ... )
    >>> db = load_fingerprint_database('data/sim/wifi_fingerprint_grid')
    >>> print(db)

Author: Navigation Engineer
Date: 2024
"""

from .dataset import (
    load_fingerprint_database,
    print_database_summary,
    save_fingerprint_database,
    validate_database,
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
]

__version__ = "0.1.0"

