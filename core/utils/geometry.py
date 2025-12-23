"""
Geometric utilities for positioning and navigation.

Provides functions for:
- Singularity handling in Jacobians
- Anchor geometry checking
- Positioning quality metrics
"""

import numpy as np
from typing import Tuple, Optional
import warnings


# Singularity threshold constants
EPSILON_RANGE = 1e-10  # Minimum range for Jacobian computation (10 picometers)
EPSILON_COLINEAR = 1e-6  # Threshold for colinearity detection


def normalize_jacobian_singularities(
    diff: np.ndarray,
    ranges: np.ndarray,
    epsilon: float = EPSILON_RANGE
) -> np.ndarray:
    """
    Safely compute normalized Jacobian, avoiding singularities.
    
    Computes H[i] = diff[i] / range[i] with protection against division by zero
    when range → 0 (receiver at anchor position).
    
    Args:
        diff: Difference vectors (receiver - anchor), shape (N, d)
        ranges: Range values, shape (N,) or (N, 1)
        epsilon: Minimum range threshold (default: 1e-10 meters = 10 pm)
    
    Returns:
        Normalized Jacobian H = diff / range, shape (N, d)
        At singularities (range < epsilon), returns zero vector
    
    Example:
        >>> diff = np.array([[1.0, 0.0], [1e-12, 1e-12], [3.0, 4.0]])
        >>> ranges = np.array([1.0, 1e-12, 5.0])
        >>> H = normalize_jacobian_singularities(diff, ranges)
        >>> H[1]  # Singularity -> zero vector
        array([0., 0.])
    
    References:
        Used in range-bearing EKF measurement Jacobians (Chapter 3)
    """
    ranges = np.asarray(ranges).reshape(-1, 1)  # Ensure column vector
    diff = np.asarray(diff)
    
    # Clamp ranges to epsilon to avoid division by zero
    ranges_safe = np.maximum(ranges, epsilon)
    
    # Compute normalized Jacobian
    H = diff / ranges_safe
    
    # Zero out rows where range is below epsilon (true singularity)
    singular_mask = (ranges < epsilon).flatten()
    if np.any(singular_mask):
        H[singular_mask, :] = 0.0
        warnings.warn(
            f"{np.sum(singular_mask)} measurement(s) at singularity (range < {epsilon}m). "
            "Setting Jacobian rows to zero. Check anchor-receiver geometry.",
            RuntimeWarning
        )
    
    return H


def check_anchor_geometry(
    anchors: np.ndarray,
    position: Optional[np.ndarray] = None,
    min_anchors_2d: int = 3,
    min_anchors_3d: int = 4,
    warn_degenerate: bool = True
) -> Tuple[bool, str]:
    """
    Check if anchor geometry is suitable for positioning.
    
    Performs geometric checks:
    1. Sufficient number of anchors
    2. Anchors are not colinear (2D) / coplanar (3D)
    3. Position (if given) has reasonable geometry with anchors
    
    Args:
        anchors: Anchor positions, shape (N, d) where d=2 or 3
        position: Optional receiver position to check geometry, shape (d,)
        min_anchors_2d: Minimum anchors for 2D positioning (default: 3)
        min_anchors_3d: Minimum anchors for 3D positioning (default: 4)
        warn_degenerate: If True, issue warnings for degenerate cases
    
    Returns:
        Tuple of (is_valid, message):
            - is_valid: True if geometry is acceptable
            - message: Description of geometry issue (empty if valid)
    
    Example:
        >>> # Good 2D geometry (triangle)
        >>> anchors = np.array([[0, 0], [10, 0], [5, 10]])
        >>> is_valid, msg = check_anchor_geometry(anchors)
        >>> is_valid
        True
        
        >>> # Bad 2D geometry (colinear)
        >>> anchors = np.array([[0, 0], [5, 0], [10, 0]])
        >>> is_valid, msg = check_anchor_geometry(anchors)
        >>> is_valid
        False
        >>> 'colinear' in msg.lower()
        True
    
    References:
        DOP analysis (Chapter 4), Observability (Chapter 8)
    """
    anchors = np.asarray(anchors)
    
    if anchors.ndim != 2:
        return False, f"Anchors must be 2D array (N, d), got shape {anchors.shape}"
    
    n_anchors, dim = anchors.shape
    
    if dim not in [2, 3]:
        return False, f"Only 2D or 3D positioning supported, got dim={dim}"
    
    # Check 1: Sufficient number of anchors
    min_required = min_anchors_2d if dim == 2 else min_anchors_3d
    if n_anchors < min_required:
        return False, (
            f"Insufficient anchors: need at least {min_required} for {dim}D positioning, "
            f"got {n_anchors}"
        )
    
    # Check 2: Anchors not degenerate (colinear in 2D, coplanar in 3D)
    # Compute centered anchor matrix
    anchors_centered = anchors - np.mean(anchors, axis=0)
    
    # Check rank via SVD
    singular_values = np.linalg.svd(anchors_centered, compute_uv=False)
    rank = np.sum(singular_values > EPSILON_COLINEAR * singular_values[0])
    
    if rank < dim:
        if dim == 2:
            msg = f"Anchors are colinear (rank {rank} < 2). Positioning will fail."
        else:
            msg = f"Anchors are coplanar (rank {rank} < 3). 3D positioning will fail."
        
        if warn_degenerate:
            warnings.warn(msg, RuntimeWarning)
        return False, msg
    
    # Check 3: If position provided, check if it's surrounded by anchors
    if position is not None:
        position = np.asarray(position)
        if position.shape != (dim,):
            return False, f"Position dimension mismatch: expected {dim}, got {position.shape}"
        
        # Check if position is within convex hull (rough approximation)
        # For 2D: check if position is within bounding box extended by 20%
        if dim == 2:
            min_bounds = anchors.min(axis=0)
            max_bounds = anchors.max(axis=0)
            margin = 0.2 * (max_bounds - min_bounds)
            
            if np.any(position < min_bounds - margin) or np.any(position > max_bounds + margin):
                msg = (
                    f"Position {position} is far outside anchor region "
                    f"[{min_bounds}, {max_bounds}]. This may cause poor DOP."
                )
                if warn_degenerate:
                    warnings.warn(msg, RuntimeWarning)
                # Don't fail, just warn - extrapolation can still work
    
    return True, ""


def compute_gdop_2d(anchors: np.ndarray, position: np.ndarray) -> float:
    """
    Compute Geometric Dilution of Precision for 2D positioning.
    
    GDOP quantifies how anchor geometry affects positioning accuracy:
    - GDOP < 2: Excellent
    - GDOP 2-5: Good
    - GDOP 5-10: Moderate
    - GDOP > 10: Poor
    
    Args:
        anchors: Anchor positions, shape (N, 2)
        position: Receiver position, shape (2,)
    
    Returns:
        GDOP value (dimensionless)
    
    Example:
        >>> # Square geometry
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> position = np.array([5.0, 5.0])  # Center
        >>> gdop = compute_gdop_2d(anchors, position)
        >>> gdop < 2.0  # Should be excellent
        True
    
    References:
        DOP analysis (core/rf/dop.py), Chapter 4
    """
    from core.rf.dop import compute_geometry_matrix, compute_dop
    
    H = compute_geometry_matrix(anchors, position, measurement_type="toa")
    dop = compute_dop(H)
    
    return dop.get("GDOP", np.inf)


