"""
RF (Radio Frequency) positioning module.

This module implements RF signal measurement models and positioning algorithms
for Chapter 4 of "Principles of Indoor Positioning and Indoor Navigation".

Submodules:
    measurement_models: TOA, TDOA, AOA, RSS measurement functions
    positioning: Positioning algorithms (LS, I-WLS, closed-form)
    dop: Dilution of Precision utilities
"""

from core.rf.dop import (
    compute_dop,
    compute_dop_map,
    compute_geometry_matrix,
    position_error_from_dop,
)
from core.rf.measurement_models import (
    SPEED_OF_LIGHT,
    aoa_angle_vector,
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    aoa_sin_elevation,
    aoa_tan_azimuth,
    clock_bias_meters_to_seconds,
    clock_bias_seconds_to_meters,
    range_to_rtt,
    rss_fading_to_distance_error,
    rss_pathloss,
    rss_to_distance,
    rtt_to_range,
    simulate_rss_measurement,
    simulate_rtt_measurement,
    tdoa_measurement_vector,
    tdoa_range_difference,
    toa_range,
    two_way_toa_range,
)
from core.rf.positioning import (
    AOAPositioner,
    TDOAPositioner,
    TOAPositioner,
    aoa_ove_solve,
    aoa_ple_solve_2d,
    aoa_ple_solve_3d,
    build_tdoa_covariance,
    tdoa_chan_solver,
    toa_fang_solver,
    toa_solve_with_clock_bias,
)

__all__ = [
    # Constants
    "SPEED_OF_LIGHT",
    # Clock bias conversion utilities
    "clock_bias_seconds_to_meters",
    "clock_bias_meters_to_seconds",
    # Measurement models
    "toa_range",
    "two_way_toa_range",
    "rtt_to_range",
    "range_to_rtt",
    "simulate_rtt_measurement",
    "rss_pathloss",
    "rss_to_distance",
    "simulate_rss_measurement",
    "rss_fading_to_distance_error",
    "tdoa_range_difference",
    "tdoa_measurement_vector",
    "aoa_azimuth",
    "aoa_elevation",
    "aoa_sin_elevation",
    "aoa_tan_azimuth",
    "aoa_measurement_vector",
    "aoa_angle_vector",
    # Positioning
    "TOAPositioner",
    "TDOAPositioner",
    "AOAPositioner",
    "toa_solve_with_clock_bias",
    # TDOA utilities
    "build_tdoa_covariance",
    # TOA/TDOA Closed-form solvers
    "toa_fang_solver",
    "tdoa_chan_solver",
    # AOA Closed-form solvers
    "aoa_ove_solve",
    "aoa_ple_solve_2d",
    "aoa_ple_solve_3d",
    # DOP
    "compute_geometry_matrix",
    "compute_dop",
    "compute_dop_map",
    "position_error_from_dop",
]

