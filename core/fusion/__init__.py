"""Sensor fusion utilities for Chapter 8.

This package provides practical multi-sensor fusion tools including:
- Time-stamped measurement types and temporal synchronization
- Innovation monitoring and covariance tuning (Eqs. 8.5-8.7)
- Chi-square gating for outlier rejection (Eqs. 8.8-8.9)

These utilities support the Chapter 8 examples demonstrating loosely vs
tightly coupled fusion, observability, tuning, calibration, and temporal
synchronization.

Author: Li-Ta Hsu
References: Chapter 8 - Sensor Fusion
"""

from core.fusion.gating import (
    chi_square_bounds,
    chi_square_gate,
    chi_square_threshold,
    mahalanobis_distance_squared,
)
from core.fusion.tuning import (
    cauchy_weight,
    compute_normalized_innovation,
    huber_weight,
    innovation,
    innovation_covariance,
    scale_measurement_covariance,
)
from core.fusion.types import StampedMeasurement, TimeSyncModel

__all__ = [
    # Types
    "StampedMeasurement",
    "TimeSyncModel",
    # Tuning (Eqs. 8.5-8.7)
    "innovation",
    "innovation_covariance",
    "scale_measurement_covariance",
    "huber_weight",
    "cauchy_weight",
    "compute_normalized_innovation",
    # Gating (Eqs. 8.8-8.9)
    "mahalanobis_distance_squared",
    "chi_square_gate",
    "chi_square_threshold",
    "chi_square_bounds",
]


