"""Tuning and robustness utilities for sensor fusion (Chapter 8).

This module implements innovation monitoring, covariance tuning, and robust
measurement weighting as described in Chapter 8, Section 8.3.

Author: Navigation Engineer
References: Chapter 8, Section 8.3 (Tuning and Robustness)
"""

from typing import Callable

import numpy as np


def innovation(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    """Compute measurement innovation (residual).
    
    Implements Eq. (8.5) in Chapter 8:
        y_k = z_k - h(x̂_{k|k-1})
    
    The innovation represents the difference between the actual measurement
    and the predicted measurement from the current state estimate.
    
    Args:
        z: Actual measurement vector (m,).
        z_pred: Predicted measurement h(x̂_{k|k-1}) (m,).
    
    Returns:
        Innovation vector y_k (m,).
    
    Raises:
        ValueError: If z and z_pred have different shapes.
    
    Example:
        >>> z = np.array([5.2, 3.1])
        >>> z_pred = np.array([5.0, 3.0])
        >>> y = innovation(z, z_pred)
        >>> np.allclose(y, [0.2, 0.1])
        True
    
    References:
        Eq. (8.5) in Chapter 8
    """
    z = np.asarray(z)
    z_pred = np.asarray(z_pred)
    
    if z.shape != z_pred.shape:
        raise ValueError(
            f"Measurement z and prediction z_pred must have same shape, "
            f"got {z.shape} and {z_pred.shape}"
        )
    
    return z - z_pred


def innovation_covariance(
    H: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """Compute innovation covariance matrix.
    
    Implements Eq. (8.6) in Chapter 8:
        S_k = H_k P_{k|k-1} H_k^T + R_k
    
    The innovation covariance quantifies the uncertainty in the innovation,
    combining prediction uncertainty (P) and measurement noise (R).
    
    Args:
        H: Measurement Jacobian matrix (m × n).
        P_pred: Predicted state covariance P_{k|k-1} (n × n).
        R: Measurement noise covariance R_k (m × m).
    
    Returns:
        Innovation covariance S_k (m × m).
    
    Raises:
        ValueError: If matrix dimensions are incompatible.
    
    Example:
        >>> H = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity observation
        >>> P_pred = np.diag([0.5, 0.3])
        >>> R = np.diag([0.1, 0.1])
        >>> S = innovation_covariance(H, P_pred, R)
        >>> np.allclose(S, [[0.6, 0.0], [0.0, 0.4]])
        True
    
    References:
        Eq. (8.6) in Chapter 8
    """
    H = np.asarray(H)
    P_pred = np.asarray(P_pred)
    R = np.asarray(R)
    
    # Validate dimensions
    if H.ndim != 2:
        raise ValueError(f"H must be 2D matrix, got shape {H.shape}")
    if P_pred.ndim != 2:
        raise ValueError(f"P_pred must be 2D matrix, got shape {P_pred.shape}")
    if R.ndim != 2:
        raise ValueError(f"R must be 2D matrix, got shape {R.shape}")
    
    m, n = H.shape
    
    if P_pred.shape != (n, n):
        raise ValueError(
            f"P_pred shape {P_pred.shape} incompatible with H shape {H.shape}, "
            f"expected ({n}, {n})"
        )
    
    if R.shape != (m, m):
        raise ValueError(
            f"R shape {R.shape} incompatible with H shape {H.shape}, "
            f"expected ({m}, {m})"
        )
    
    # Eq. (8.6): S_k = H_k P_{k|k-1} H_k^T + R_k
    S = H @ P_pred @ H.T + R
    
    # Ensure symmetry (numerical stability)
    S = 0.5 * (S + S.T)
    
    return S


def scale_measurement_covariance(
    R: np.ndarray,
    weight: float
) -> np.ndarray:
    """Apply robust scaling to measurement covariance.
    
    Implements Eq. (8.7) in Chapter 8:
        R_k ← w(y_k) * R_k
    
    Robust weighting down-weights measurements with large innovations,
    reducing their influence on the estimate. This is an alternative to
    hard gating (rejection).
    
    Args:
        R: Original measurement covariance (m × m).
        weight: Scalar weight w(y_k) ∈ [0, ∞). Values > 1 inflate the
                covariance (reduce measurement confidence). Typical robust
                loss functions produce weights in (0, 1] for inliers and
                weights >> 1 for outliers.
    
    Returns:
        Scaled covariance R_scaled = w * R (m × m).
    
    Raises:
        ValueError: If weight is negative or R is not 2D.
    
    Example:
        >>> R = np.diag([0.1, 0.2])
        >>> weight = 2.0  # reduce confidence by 2x
        >>> R_scaled = scale_measurement_covariance(R, weight)
        >>> np.allclose(R_scaled, [[0.2, 0.0], [0.0, 0.4]])
        True
        
        >>> # Outlier case: inflate covariance by 100x
        >>> R_outlier = scale_measurement_covariance(R, 100.0)
        >>> np.allclose(R_outlier, [[10.0, 0.0], [0.0, 20.0]])
        True
    
    Notes:
        Common robust weight functions include:
        - Huber: w = 1 if |y| < k, else k/|y|
        - Cauchy: w = 1 + (y/c)^2
        - Tukey: w = (1 - (y/c)^2)^2 if |y| < c, else 0 (hard rejection)
        
        These are typically applied element-wise to normalized innovations
        and then combined into a scalar weight.
    
    References:
        Eq. (8.7) in Chapter 8
    """
    R = np.asarray(R)
    
    if R.ndim != 2:
        raise ValueError(f"R must be 2D matrix, got shape {R.shape}")
    
    if not isinstance(weight, (int, float, np.number)):
        raise TypeError(f"Weight must be numeric, got {type(weight)}")
    
    if weight < 0:
        raise ValueError(f"Weight must be non-negative, got {weight}")
    
    # Eq. (8.7): R_k ← w(y_k) * R_k
    R_scaled = weight * R
    
    return R_scaled


def huber_weight(
    residual: float,
    threshold: float
) -> float:
    """Compute Huber robust weight for a scalar residual.
    
    The Huber weight function provides robust down-weighting for outliers:
        w(r) = 1                if |r| ≤ k
        w(r) = k / |r|          if |r| > k
    
    This is a helper function for computing the weight in Eq. (8.7).
    
    Args:
        residual: Normalized residual (e.g., innovation / sqrt(variance)).
        threshold: Huber threshold k (typical values: 1.345 for 95% efficiency
                   on Gaussian data, or 2.0-3.0 for more aggressive downweighting).
    
    Returns:
        Weight w(r) ∈ (0, 1].
    
    Example:
        >>> huber_weight(0.5, threshold=1.345)  # inlier
        1.0
        >>> huber_weight(3.0, threshold=1.345)  # outlier
        0.448...
    
    References:
        Chapter 8, Section 8.3 (Robust Loss Functions)
        Related to Eq. (8.7)
    """
    abs_residual = abs(residual)
    
    if abs_residual <= threshold:
        return 1.0
    else:
        return threshold / abs_residual


def cauchy_weight(
    residual: float,
    scale: float
) -> float:
    """Compute Cauchy robust weight for a scalar residual.
    
    The Cauchy weight function provides stronger outlier rejection than Huber:
        w(r) = 1 / (1 + (r/c)^2)
    
    This is a helper function for computing the weight in Eq. (8.7).
    As an inflation factor for R (Eq. 8.7), use 1/w(r).
    
    Args:
        residual: Normalized residual (e.g., innovation / sqrt(variance)).
        scale: Cauchy scale parameter c (typical values: 2.385 for 95% efficiency).
    
    Returns:
        Weight w(r) ∈ (0, 1].
    
    Example:
        >>> cauchy_weight(0.0, scale=2.385)
        1.0
        >>> cauchy_weight(2.385, scale=2.385)
        0.5
        >>> cauchy_weight(10.0, scale=2.385)  # strong outlier
        0.053...
    
    References:
        Chapter 8, Section 8.3 (Robust Loss Functions)
        Related to Eq. (8.7)
    """
    normalized = residual / scale
    return 1.0 / (1.0 + normalized**2)


def compute_normalized_innovation(
    y: np.ndarray,
    S: np.ndarray
) -> np.ndarray:
    """Compute normalized (whitened) innovation for robust weighting.
    
    Normalizes the innovation vector by the innovation covariance, producing
    a dimensionless residual suitable for robust weight computation.
    
    For a scalar measurement, this is simply y / sqrt(S).
    For vector measurements, this uses Cholesky decomposition: y_norm = L^{-1} y
    where S = L L^T.
    
    Args:
        y: Innovation vector (m,).
        S: Innovation covariance (m × m), must be positive definite.
    
    Returns:
        Normalized innovation vector (m,).
    
    Raises:
        ValueError: If dimensions are incompatible or S is not positive definite.
    
    Example:
        >>> y = np.array([2.0])
        >>> S = np.array([[4.0]])
        >>> y_norm = compute_normalized_innovation(y, S)
        >>> np.allclose(y_norm, [1.0])  # 2.0 / sqrt(4.0) = 1.0
        True
        
        >>> # Vector case
        >>> y = np.array([2.0, 1.0])
        >>> S = np.diag([4.0, 1.0])
        >>> y_norm = compute_normalized_innovation(y, S)
        >>> np.allclose(y_norm, [1.0, 1.0])
        True
    
    Notes:
        This function is used to prepare innovations for element-wise robust
        weight computation (Huber, Cauchy, etc.).
    """
    y = np.asarray(y)
    S = np.asarray(S)
    
    if y.ndim != 1:
        raise ValueError(f"Innovation y must be 1D, got shape {y.shape}")
    if S.ndim != 2:
        raise ValueError(f"Covariance S must be 2D, got shape {S.shape}")
    
    m = len(y)
    if S.shape != (m, m):
        raise ValueError(
            f"Innovation dimension {m} incompatible with S shape {S.shape}"
        )
    
    # Cholesky decomposition: S = L L^T
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Innovation covariance S is not positive definite: {e}")
    
    # Solve L * y_norm = y for y_norm
    y_normalized = np.linalg.solve(L, y)
    
    return y_normalized


