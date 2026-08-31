"""Sensor fusion for Chapter 8.

This package provides:
- Time-stamped measurement types and temporal synchronization
- Innovation monitoring and covariance tuning (Eqs. 8.5-8.7)
- Chi-square gating for outlier rejection (Eqs. 8.8-8.9)
- Adaptive gating with covariance inflation and NIS monitoring
- The dataset loader and both fusion architectures themselves,
  ``run_lc_fusion`` (Sec. 8.1.1) and ``run_tc_fusion`` (Sec. 8.1.2)

The last of those used to live in ``ch8_sensor_fusion/``, where four demos
imported them from a fifth. Every other chapter keeps its shared
implementation here and its examples as leaves; this one does now too.

Author: Li-Ta Hsu
References: Chapter 8 - Sensor Fusion
"""

from core.fusion.adaptive import (
    AdaptiveGatingManager,
    create_adaptive_manager_for_lc,
    create_adaptive_manager_for_tc,
)
from core.fusion.dataset import FusionDataset, load_fusion_dataset
from core.fusion.gating import (
    chi_square_bounds,
    chi_square_gate,
    chi_square_threshold,
    mahalanobis_distance_squared,
)
from core.fusion.loosely_coupled import run_lc_fusion
from core.fusion.tightly_coupled import run_tc_fusion
from core.fusion.tuning import (
    cauchy_R_scale,
    cauchy_weight,
    compute_normalized_innovation,
    huber_R_scale,
    huber_weight,
    innovation,
    innovation_covariance,
    kalman_update,
    scale_measurement_covariance,
)
from core.fusion.types import (
    SENSOR_IMU,
    SENSOR_UWB_RANGE,
    SENSOR_UWB_RANGES_BATCH,
    SENSOR_UWB_RANGES_EPOCH,
    FusionHistory,
    StampedMeasurement,
    TimeSyncModel,
)

__all__ = [
    # Datasets and the two fusion architectures (Sec. 8.1)
    "load_fusion_dataset",
    "FusionDataset",
    "run_lc_fusion",
    "run_tc_fusion",
    # Types
    "SENSOR_IMU",
    "SENSOR_UWB_RANGE",
    "SENSOR_UWB_RANGES_EPOCH",
    "SENSOR_UWB_RANGES_BATCH",
    "FusionHistory",
    "StampedMeasurement",
    "TimeSyncModel",
    # Tuning (Eqs. 8.5-8.7)
    "innovation",
    "innovation_covariance",
    "kalman_update",
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
