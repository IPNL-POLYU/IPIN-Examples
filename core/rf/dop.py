"""
Dilution of Precision (DOP) utilities for RF positioning.

This module provides functions to compute geometry matrices and various
DOP metrics (GDOP, PDOP, HDOP, VDOP) for positioning systems.

References Chapter 4 of the IPIN book.
"""

from typing import Dict, Optional

import numpy as np


def compute_geometry_matrix(
    anchors: np.ndarray,
    position: np.ndarray,
    measurement_type: str = "toa",
) -> np.ndarray:
    """
    Compute geometry matrix for DOP calculation.

    The geometry matrix H relates measurement errors to position errors:
        Δz = H * Δp
    where Δz is measurement error and Δp is position error.

    Args:
        anchors: Array of anchor positions, shape (N, d) where d=2 or 3.
        position: Position at which to compute geometry, shape (d,).
        measurement_type: Type of measurement ('toa', 'tdoa', 'aoa').
                         Defaults to 'toa'.

    Returns:
        Geometry matrix H, shape (N, d) for TOA/AOA or (N-1, d) for TDOA.

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> position = np.array([5, 5])
        >>> H = compute_geometry_matrix(anchors, position, 'toa')
        >>> print(H.shape)
        (4, 2)
    """
    anchors = np.asarray(anchors, dtype=float)
    position = np.asarray(position, dtype=float)

    n_anchors = anchors.shape[0]
    dim = anchors.shape[1]

    measurement_type = measurement_type.lower()

    if measurement_type == "toa":
        # TOA geometry: H[i,:] = -(p_i - p) / ||p_i - p||
        H = np.zeros((n_anchors, dim))
        for i in range(n_anchors):
            diff = anchors[i] - position
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                H[i] = -diff / dist
            else:
                # Anchor coincides with position, undefined direction
                H[i] = 0.0

    elif measurement_type == "tdoa":
        # TDOA geometry: relative to reference anchor (index 0)
        reference = anchors[0]
        dist_ref = np.linalg.norm(position - reference)

        H = np.zeros((n_anchors - 1, dim))
        for i in range(1, n_anchors):
            dist_i = np.linalg.norm(position - anchors[i])

            if dist_i > 1e-10 and dist_ref > 1e-10:
                H[i - 1] = (position - anchors[i]) / dist_i - (
                    position - reference
                ) / dist_ref
            else:
                H[i - 1] = 0.0

    elif measurement_type == "aoa":
        # AOA geometry (azimuth only): H[i,:] = [-dy/r^2, dx/r^2]
        H = np.zeros((n_anchors, dim))
        for i in range(n_anchors):
            dx = position[0] - anchors[i, 0]
            dy = position[1] - anchors[i, 1]
            dist_sq = dx**2 + dy**2

            if dist_sq > 1e-10:
                if dim == 2:
                    H[i] = np.array([-dy / dist_sq, dx / dist_sq])
                else:
                    # For 3D AOA, this is simplified (azimuth only)
                    H[i, :2] = np.array([-dy / dist_sq, dx / dist_sq])
            else:
                H[i] = 0.0

    else:
        raise ValueError(
            f"measurement_type must be 'toa', 'tdoa', or 'aoa', "
            f"got {measurement_type}"
        )

    return H


def compute_dop(
    geometry_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute Dilution of Precision (DOP) metrics.

    DOP quantifies the effect of geometry on positioning accuracy:
        GDOP: Geometric DOP (overall 3D accuracy)
        PDOP: Position DOP (3D position only)
        HDOP: Horizontal DOP (2D horizontal accuracy)
        VDOP: Vertical DOP (height accuracy)

    The covariance of the position estimate is:
        Σ_p = σ_meas^2 * (H^T W H)^{-1}

    where σ_meas is the measurement noise std and W is the weight matrix.

    Args:
        geometry_matrix: Geometry matrix H, shape (N, d).
        weights: Optional weight matrix W, shape (N, N). If None, identity is used.

    Returns:
        Dictionary containing DOP values:
            - 'GDOP': Geometric DOP
            - 'PDOP': Position DOP
            - 'HDOP': Horizontal DOP
            - 'VDOP': Vertical DOP (only for 3D)

    Example:
        >>> H = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) * 0.1
        >>> dop = compute_dop(H)
        >>> print(f"HDOP: {dop['HDOP']:.2f}")
    """
    H = np.asarray(geometry_matrix, dtype=float)
    n_meas, dim = H.shape

    # Weight matrix
    if weights is None:
        W = np.eye(n_meas)
    else:
        W = np.asarray(weights, dtype=float)

    # Compute covariance matrix (scaled by σ_meas^2)
    # Q = (H^T W H)^{-1}
    try:
        Q = np.linalg.inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        # Singular matrix, return infinite DOP
        return {
            "GDOP": np.inf,
            "PDOP": np.inf,
            "HDOP": np.inf,
            "VDOP": np.inf if dim >= 3 else None,
        }

    # Compute DOP values
    dop_dict = {}

    # GDOP: sqrt(trace(Q)) for all dimensions
    dop_dict["GDOP"] = np.sqrt(np.trace(Q))

    # PDOP: sqrt(sum of position variances)
    dop_dict["PDOP"] = np.sqrt(np.sum(np.diag(Q)))

    # HDOP: sqrt(sum of horizontal variances)
    if dim >= 2:
        dop_dict["HDOP"] = np.sqrt(Q[0, 0] + Q[1, 1])
    else:
        dop_dict["HDOP"] = None

    # VDOP: sqrt(vertical variance)
    if dim >= 3:
        dop_dict["VDOP"] = np.sqrt(Q[2, 2])
    else:
        dop_dict["VDOP"] = None

    return dop_dict


def compute_dop_map(
    anchors: np.ndarray,
    grid_points: np.ndarray,
    measurement_type: str = "toa",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute DOP values over a grid of positions.

    This is useful for visualizing how positioning geometry affects
    accuracy across a space.

    Args:
        anchors: Array of anchor positions, shape (N, d).
        grid_points: Array of grid positions, shape (M, d).
        measurement_type: Type of measurement ('toa', 'tdoa', 'aoa').
                         Defaults to 'toa'.
        weights: Optional weight matrix, shape (N, N).

    Returns:
        Array of HDOP values at each grid point, shape (M,).

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> x = np.linspace(0, 10, 20)
        >>> y = np.linspace(0, 10, 20)
        >>> xx, yy = np.meshgrid(x, y)
        >>> grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        >>> hdop_map = compute_dop_map(anchors, grid_points, 'toa')
        >>> print(hdop_map.shape)
        (400,)
    """
    grid_points = np.asarray(grid_points, dtype=float)
    n_points = grid_points.shape[0]

    hdop_values = np.zeros(n_points)

    for i in range(n_points):
        position = grid_points[i]

        # Compute geometry matrix at this position
        H = compute_geometry_matrix(anchors, position, measurement_type)

        # Compute DOP
        dop = compute_dop(H, weights)

        # Store HDOP (or GDOP if HDOP not available)
        hdop_values[i] = (
            dop["HDOP"] if dop["HDOP"] is not None else dop["GDOP"]
        )

    return hdop_values

