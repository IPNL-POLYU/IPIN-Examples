"""
Evaluation Metrics for Indoor Positioning.

This module provides functions to compute error metrics and consistency
statistics for positioning algorithms.

Author: Navigation Engineering Team
Date: December 2025
"""

from typing import Dict, Optional, Union

import numpy as np


def compute_position_errors(
    truth: np.ndarray, estimated: np.ndarray
) -> np.ndarray:
    """
    Compute position errors between true and estimated positions.

    Args:
        truth: True positions, shape (N, 2) or (N, 3)
        estimated: Estimated positions, shape (N, 2) or (N, 3)

    Returns:
        errors: Position error vectors, shape (N, 2) or (N, 3)

    Raises:
        ValueError: If inputs have incompatible shapes
    """
    truth = np.asarray(truth)
    estimated = np.asarray(estimated)

    if truth.shape != estimated.shape:
        raise ValueError(
            f"Shape mismatch: truth {truth.shape} vs estimated {estimated.shape}"
        )

    return estimated - truth


def compute_rmse(errors: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Compute Root Mean Square Error (RMSE).

    Args:
        errors: Error vectors, shape (N, d) or (N,)
        axis: Axis along which to compute RMSE
              None: scalar RMSE across all dimensions
              0: per-dimension RMSE
              1: per-sample RMSE

    Returns:
        rmse: RMSE value(s)
    """
    errors = np.asarray(errors)

    if axis is None:
        # Scalar RMSE across all dimensions
        return np.sqrt(np.mean(errors**2))
    else:
        # Per-axis or per-sample RMSE
        return np.sqrt(np.mean(errors**2, axis=axis))


def compute_error_stats(errors: np.ndarray) -> Dict[str, float]:
    """
    Compute error statistics.

    Args:
        errors: Error vectors, shape (N, d) or (N,)

    Returns:
        stats: Dictionary with keys:
               - 'mean': Mean error
               - 'median': Median error
               - 'std': Standard deviation
               - 'rmse': Root mean square error
               - 'p50': 50th percentile (median)
               - 'p75': 75th percentile
               - 'p90': 90th percentile
               - 'p95': 95th percentile
               - 'max': Maximum error
    """
    errors = np.asarray(errors)

    # Compute error magnitudes if multi-dimensional
    if errors.ndim > 1:
        error_magnitudes = np.linalg.norm(errors, axis=1)
    else:
        error_magnitudes = np.abs(errors)

    stats = {
        "mean": float(np.mean(error_magnitudes)),
        "median": float(np.median(error_magnitudes)),
        "std": float(np.std(error_magnitudes)),
        "rmse": float(np.sqrt(np.mean(error_magnitudes**2))),
        "p50": float(np.percentile(error_magnitudes, 50)),
        "p75": float(np.percentile(error_magnitudes, 75)),
        "p90": float(np.percentile(error_magnitudes, 90)),
        "p95": float(np.percentile(error_magnitudes, 95)),
        "max": float(np.max(error_magnitudes)),
    }

    return stats


def compute_nees(
    truth: np.ndarray, estimated: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """
    Compute Normalized Estimation Error Squared (NEES).

    NEES is a consistency metric for filter performance:
        NEES = (x_true - x_est)^T P^{-1} (x_true - x_est)

    For consistent estimators, NEES follows chi-squared distribution
    with n degrees of freedom (state dimension).

    Args:
        truth: True states, shape (N, n)
        estimated: Estimated states, shape (N, n)
        covariance: Estimation covariances, shape (N, n, n)

    Returns:
        nees: NEES values, shape (N,)

    Raises:
        ValueError: If inputs have incompatible shapes
    """
    truth = np.asarray(truth)
    estimated = np.asarray(estimated)
    covariance = np.asarray(covariance)

    if truth.shape != estimated.shape:
        raise ValueError("truth and estimated must have same shape")

    N, n = truth.shape

    if covariance.shape != (N, n, n):
        raise ValueError(
            f"covariance must have shape ({N}, {n}, {n}), "
            f"got {covariance.shape}"
        )

    nees = np.zeros(N)
    for i in range(N):
        error = estimated[i] - truth[i]
        try:
            P_inv = np.linalg.inv(covariance[i])
            nees[i] = error @ P_inv @ error
        except np.linalg.LinAlgError:
            nees[i] = np.nan

    return nees


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Innovation Squared (NIS).

    NIS is a consistency metric for measurement updates:
        NIS = nu^T S^{-1} nu

    where nu is the innovation and S is the innovation covariance.

    For consistent estimators, NIS follows chi-squared distribution
    with m degrees of freedom (measurement dimension).

    Args:
        innovation: Innovation vectors, shape (N, m)
        S: Innovation covariances, shape (N, m, m)

    Returns:
        nis: NIS values, shape (N,)

    Raises:
        ValueError: If inputs have incompatible shapes
    """
    innovation = np.asarray(innovation)
    S = np.asarray(S)

    if innovation.ndim == 1:
        innovation = innovation.reshape(-1, 1)

    N, m = innovation.shape

    if S.shape != (N, m, m):
        raise ValueError(
            f"S must have shape ({N}, {m}, {m}), got {S.shape}"
        )

    nis = np.zeros(N)
    for i in range(N):
        try:
            S_inv = np.linalg.inv(S[i])
            nis[i] = innovation[i] @ S_inv @ innovation[i]
        except np.linalg.LinAlgError:
            nis[i] = np.nan

    return nis


