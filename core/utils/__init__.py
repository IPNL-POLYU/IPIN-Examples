"""
Utility functions for navigation algorithms.

This module provides common utility functions used across the codebase,
including angle operations, singularity handling, and observability checks.
"""

from .angles import wrap_angle, wrap_angle_array, angle_diff
from .paths import resolve_data_path, repo_root
from .geometry import normalize_jacobian_singularities, check_anchor_geometry, compute_gdop_2d
from .observability import (
    check_observability, 
    compute_observability_matrix,
    check_range_only_observability_2d,
    estimate_observability_time_constant
)

__all__ = [
    'resolve_data_path',
    'repo_root',
    'wrap_angle',
    'wrap_angle_array',
    'angle_diff',
    'normalize_jacobian_singularities',
    'check_anchor_geometry',
    'compute_gdop_2d',
    'check_observability',
    'compute_observability_matrix',
    'check_range_only_observability_2d',
    'estimate_observability_time_constant',
]

