"""
Dilution of Precision (DOP) utilities for RF positioning.

This module provides functions to compute geometry matrices and various
DOP metrics (GDOP, HDOP, VDOP, PDOP) for positioning systems.

Implements the DOP formulations from Chapter 4, Section 4.5 of the IPIN book.

Book DOP Definitions (Eqs. 4.103-4.108):
========================================

The position error covariance is related to measurement error covariance by:
    C(x_a) = (H_a^T H_a)^{-1} * σ_z^2    (Eq. 4.103)

where H_a is the geometry (design) matrix and σ_z is the measurement noise
standard deviation. The DOP factors map measurement noise to position noise:

    σ_position = DOP * σ_measurement     (Eq. 4.107)

Defining Q = (H_a^T H_a)^{-1} with elements:
    Q = [κ_ee  κ_en  κ_eu]
        [κ_ne  κ_nn  κ_nu]
        [κ_ue  κ_un  κ_uu]

The DOP metrics are defined as (Eq. 4.107-4.108):
    GDOP = √(κ_ee + κ_nn + κ_uu) = √(trace(Q))     (overall 3D position)
    HDOP = √(κ_ee + κ_nn)                           (horizontal position)
    VDOP = √(κ_uu)                                  (vertical position)

For 2D positioning:
    HDOP = √(κ_ee + κ_nn) = √(trace(Q))            (2D position = GDOP)

IMPORTANT: σ symbols in the book represent STANDARD DEVIATIONS, not variances.
    - σ_z: measurement noise std (meters for TOA/TDOA)
    - σ_position: position error std (meters)
    - The relationship is: σ_position = DOP × σ_z

Example relationship:
    If HDOP = 1.5 and σ_z = 0.3 m, then σ_horizontal = 1.5 × 0.3 = 0.45 m

PDOP vs GDOP:
    - In pure positioning (no clock bias), PDOP = GDOP
    - When clock bias is estimated, GDOP includes time error contribution:
        GDOP = √(PDOP² + TDOP²) where TDOP is time DOP
    - This module computes PDOP (position-only) which equals GDOP for
      TOA/TDOA positioning without explicit clock bias estimation.

References:
    Chapter 4, Section 4.5: Distribution of Beacons and DOP
    Equations: 4.99-4.108
"""

from typing import Dict, Optional

import numpy as np


def compute_geometry_matrix(
    anchors: np.ndarray,
    position: np.ndarray,
    measurement_type: str = "toa",
) -> np.ndarray:
    """
    Compute geometry matrix H for DOP calculation.

    The geometry matrix H (also called the design matrix or LOS matrix)
    relates measurement errors to position errors via:
        δz = H * δx    (linearized measurement model)

    For TOA, H contains normalized line-of-sight vectors from position to anchors.
    This is the same matrix used in Eq. (4.18)-(4.19) for iterative LS.

    Args:
        anchors: Array of anchor positions, shape (N, d) where d=2 or 3.
        position: Position at which to compute geometry, shape (d,).
        measurement_type: Type of measurement ('toa', 'tdoa', 'aoa').
                         Defaults to 'toa'.

    Returns:
        Geometry matrix H:
            - TOA: shape (N, d), H[i,:] = (x_a - x^i) / ||x_a - x^i||
            - TDOA: shape (N-1, d), differenced LOS vectors
            - AOA: shape (N, d), azimuth-based geometry

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> position = np.array([5.0, 5.0])
        >>> H = compute_geometry_matrix(anchors, position, 'toa')
        >>> print(H.shape)
        (4, 2)

    Notes:
        The book's H_a (Eq. 4.18) uses (x_a - x^i)/d convention.
        This implementation follows that convention for TOA.
    """
    anchors = np.asarray(anchors, dtype=float)
    position = np.asarray(position, dtype=float)

    n_anchors = anchors.shape[0]
    dim = anchors.shape[1]

    measurement_type = measurement_type.lower()

    if measurement_type == "toa":
        # TOA geometry matrix (Eq. 4.18):
        # h_a^i = [(x_e,a - x_e^i)/d, (x_n,a - x_n^i)/d, (x_u,a - x_u^i)/d]
        # This is the normalized LOS vector from anchor to agent
        H = np.zeros((n_anchors, dim))
        for i in range(n_anchors):
            diff = position - anchors[i]  # x_a - x^i
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                H[i] = diff / dist
            else:
                # Anchor coincides with position, undefined direction
                H[i] = 0.0

    elif measurement_type == "tdoa":
        # TDOA geometry: difference of LOS vectors (Eq. 4.38)
        # h_a^(i,j) = h_a^i - h_a^j (reference anchor j=0)
        reference = anchors[0]
        dist_ref = np.linalg.norm(position - reference)

        H = np.zeros((n_anchors - 1, dim))
        for i in range(1, n_anchors):
            diff_i = position - anchors[i]
            diff_ref = position - reference
            dist_i = np.linalg.norm(diff_i)

            if dist_i > 1e-10 and dist_ref > 1e-10:
                H[i - 1] = diff_i / dist_i - diff_ref / dist_ref
            else:
                H[i - 1] = 0.0

    elif measurement_type == "aoa":
        # AOA geometry (azimuth only): linearized tan(psi) Jacobian (Eq. 4.72-4.73)
        # For DOP purposes, we use simplified 2D bearing geometry
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
            f"got '{measurement_type}'"
        )

    return H


def compute_dop(
    geometry_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute Dilution of Precision (DOP) metrics.

    Implements the book's DOP definitions (Eqs. 4.103-4.108):
        Q = (H^T W H)^{-1}
        GDOP = sqrt(trace(Q)) = sqrt(κ_ee + κ_nn + κ_uu)
        HDOP = sqrt(κ_ee + κ_nn)
        VDOP = sqrt(κ_uu)

    DOP quantifies how measurement noise maps to position error:
        σ_position = DOP × σ_measurement    (Eq. 4.107)

    For example, if HDOP=1.5 and σ_range=0.3m, then σ_horizontal=0.45m.

    Args:
        geometry_matrix: Geometry matrix H, shape (N, d).
        weights: Optional weight matrix W, shape (N, N).
                 If None, identity is used (uniform weights).
                 For WLS with covariance Σ, use W = Σ^{-1}.

    Returns:
        Dictionary containing DOP values:
            - 'GDOP': Geometric DOP = sqrt(trace(Q))
                     (overall position error, = PDOP for pure positioning)
            - 'PDOP': Position DOP = sqrt(trace(Q))
                     (equivalent to GDOP when no clock bias)
            - 'HDOP': Horizontal DOP = sqrt(Q[0,0] + Q[1,1])
                     (horizontal position error)
            - 'VDOP': Vertical DOP = sqrt(Q[2,2])
                     (vertical position error, None for 2D)

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> pos = np.array([5.0, 5.0])
        >>> H = compute_geometry_matrix(anchors, pos, 'toa')
        >>> dop = compute_dop(H)
        >>> print(f"HDOP: {dop['HDOP']:.2f}")
        HDOP: 1.41
        >>> # Position error = HDOP * measurement noise
        >>> sigma_range = 0.3  # meters
        >>> sigma_horizontal = dop['HDOP'] * sigma_range
        >>> print(f"σ_horizontal: {sigma_horizontal:.2f} m")
        σ_horizontal: 0.42 m

    Notes:
        - For 2D positioning, HDOP = GDOP = PDOP (all equivalent)
        - σ symbols represent STANDARD DEVIATIONS, not variances
        - Lower DOP values indicate better geometry
        - DOP ≈ 1.0 is excellent, DOP > 6 indicates poor geometry

    References:
        Chapter 4, Section 4.5, Eqs. 4.103-4.108
    """
    H = np.asarray(geometry_matrix, dtype=float)
    n_meas, dim = H.shape

    # Weight matrix
    if weights is None:
        W = np.eye(n_meas)
    else:
        W = np.asarray(weights, dtype=float)

    # Compute DOP matrix Q = (H^T W H)^{-1}
    # This is the covariance factor: C(x) = Q * σ_z^2
    try:
        HtWH = H.T @ W @ H
        Q = np.linalg.inv(HtWH)
    except np.linalg.LinAlgError:
        # Singular matrix, return infinite DOP
        return {
            "GDOP": np.inf,
            "PDOP": np.inf,
            "HDOP": np.inf,
            "VDOP": np.inf if dim >= 3 else None,
        }

    # Compute DOP values using book definitions (Eq. 4.107-4.108)
    dop_dict = {}

    # GDOP: sqrt(trace(Q)) = sqrt(κ_ee + κ_nn + κ_uu)
    # Overall 3D position DOP (Eq. 4.107)
    gdop = np.sqrt(np.trace(Q))
    dop_dict["GDOP"] = gdop

    # PDOP: Position DOP (equivalent to GDOP for pure positioning)
    # When clock bias is included, GDOP^2 = PDOP^2 + TDOP^2
    # For TOA/TDOA without explicit clock estimation, PDOP = GDOP
    dop_dict["PDOP"] = gdop

    # HDOP: sqrt(κ_ee + κ_nn) - Horizontal DOP (Eq. 4.108)
    if dim >= 2:
        dop_dict["HDOP"] = np.sqrt(Q[0, 0] + Q[1, 1])
    else:
        dop_dict["HDOP"] = np.sqrt(Q[0, 0])  # 1D case

    # VDOP: sqrt(κ_uu) - Vertical DOP (Eq. 4.108)
    if dim >= 3:
        dop_dict["VDOP"] = np.sqrt(Q[2, 2])
    else:
        dop_dict["VDOP"] = None  # No vertical component in 2D

    return dop_dict


def compute_dop_map(
    anchors: np.ndarray,
    grid_points: np.ndarray,
    measurement_type: str = "toa",
    weights: Optional[np.ndarray] = None,
    dop_type: str = "HDOP",
) -> np.ndarray:
    """
    Compute DOP values over a grid of positions.

    This is useful for visualizing how beacon geometry affects
    positioning accuracy across a space.

    Args:
        anchors: Array of anchor positions, shape (N, d).
        grid_points: Array of grid positions, shape (M, d).
        measurement_type: Type of measurement ('toa', 'tdoa', 'aoa').
                         Defaults to 'toa'.
        weights: Optional weight matrix, shape (N, N).
        dop_type: Which DOP metric to return ('GDOP', 'HDOP', 'VDOP').
                  Defaults to 'HDOP'.

    Returns:
        Array of DOP values at each grid point, shape (M,).

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> x = np.linspace(0, 10, 20)
        >>> y = np.linspace(0, 10, 20)
        >>> xx, yy = np.meshgrid(x, y)
        >>> grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        >>> hdop_map = compute_dop_map(anchors, grid_points, 'toa', dop_type='HDOP')
        >>> print(f"Min HDOP: {hdop_map.min():.2f}, Max HDOP: {hdop_map.max():.2f}")

    Notes:
        - DOP varies across space depending on beacon geometry
        - Optimal placement minimizes average DOP across the coverage area
        - See Figure 4.8 in the book for DOP map examples
    """
    grid_points = np.asarray(grid_points, dtype=float)
    n_points = grid_points.shape[0]

    dop_values = np.zeros(n_points)

    for i in range(n_points):
        position = grid_points[i]

        # Compute geometry matrix at this position
        H = compute_geometry_matrix(anchors, position, measurement_type)

        # Compute DOP
        dop = compute_dop(H, weights)

        # Get requested DOP type
        dop_value = dop.get(dop_type)
        if dop_value is None:
            # Fallback to GDOP if requested type not available
            dop_value = dop.get("GDOP", np.inf)

        dop_values[i] = dop_value

    return dop_values


def position_error_from_dop(
    dop_value: float,
    measurement_noise_std: float,
) -> float:
    """
    Compute expected position error from DOP and measurement noise.

    Implements the fundamental DOP relationship (Eq. 4.107):
        σ_position = DOP × σ_measurement

    Args:
        dop_value: DOP value (HDOP, VDOP, or GDOP).
        measurement_noise_std: Measurement noise standard deviation (meters).

    Returns:
        Expected position error standard deviation (meters).

    Example:
        >>> hdop = 1.5
        >>> sigma_range = 0.3  # 30cm range noise
        >>> sigma_horizontal = position_error_from_dop(hdop, sigma_range)
        >>> print(f"Expected horizontal error: {sigma_horizontal:.2f} m")
        Expected horizontal error: 0.45 m

    Notes:
        - This is the 1-sigma (68%) position error
        - For 95% confidence, multiply by ~2 (assuming Gaussian errors)
        - σ_measurement for TOA/TDOA is typically in meters
        - σ_measurement for AOA would be angle-dependent
    """
    return dop_value * measurement_noise_std
