"""Sensor fusion utilities for Chapter 8.

This package provides practical multi-sensor fusion tools including:
- Time-stamped measurement types and temporal synchronization
- Innovation monitoring and covariance tuning (Eqs. 8.5-8.7)
- Chi-square gating for outlier rejection (Eqs. 8.8-8.9)
- Adaptive gating with covariance inflation and NIS monitoring

These utilities support the Chapter 8 examples demonstrating loosely vs
tightly coupled fusion, observability, tuning, calibration, and temporal
synchronization.

Author: Li-Ta Hsu
References: Chapter 8 - Sensor Fusion
"""

from core.fusion.adaptive import (
    AdaptiveGatingManager,
    create_adaptive_manager_for_lc,
    create_adaptive_manager_for_tc,
)
from core.fusion.gating import (
    chi_square_bounds,
    chi_square_gate,
    chi_square_threshold,
    mahalanobis_distance_squared,
)
from core.fusion.tuning import (
    cauchy_R_scale,
    cauchy_weight,
    compute_normalized_innovation,
    huber_R_scale,
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
    "huber_R_scale",
    "cauchy_R_scale",
    "compute_normalized_innovation",
    # Deprecated (use R_scale versions for Eq. 8.7)
    "huber_weight",
    "cauchy_weight",
    # Gating (Eqs. 8.8-8.9)
    "mahalanobis_distance_squared",
    "chi_square_gate",
    "chi_square_threshold",
    "chi_square_bounds",
    # Adaptive gating (Sec. 8.3.2)
    "AdaptiveGatingManager",
    "create_adaptive_manager_for_tc",
    "create_adaptive_manager_for_lc",
]


