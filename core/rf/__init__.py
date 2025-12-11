"""
RF (Radio Frequency) positioning module.

This module implements RF signal measurement models and positioning algorithms
for Chapter 4 of "Principles of Indoor Positioning and Indoor Navigation".

Submodules:
    measurement_models: TOA, TDOA, AOA, RSS measurement functions
    positioning: Positioning algorithms (LS, I-WLS, closed-form)
    dop: Dilution of Precision utilities
"""

from core.rf.dop import compute_dop, compute_dop_map, compute_geometry_matrix
from core.rf.measurement_models import (
    SPEED_OF_LIGHT,
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    rss_pathloss,
    rss_to_distance,
    tdoa_measurement_vector,
    tdoa_range_difference,
    toa_range,
    two_way_toa_range,
)
from core.rf.positioning import (
    AOAPositioner,
    TDOAPositioner,
    TOAPositioner,
    toa_solve_with_clock_bias,
)

__all__ = [
    # Constants
    "SPEED_OF_LIGHT",
    # Measurement models
    "toa_range",
    "two_way_toa_range",
    "rss_pathloss",
    "rss_to_distance",
    "tdoa_range_difference",
    "tdoa_measurement_vector",
    "aoa_azimuth",
    "aoa_elevation",
    "aoa_measurement_vector",
    # Positioning
    "TOAPositioner",
    "TDOAPositioner",
    "AOAPositioner",
    "toa_solve_with_clock_bias",
    # DOP
    "compute_geometry_matrix",
    "compute_dop",
    "compute_dop_map",
]

