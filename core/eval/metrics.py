"""
Evaluation Metrics for Indoor Positioning.

This module provides functions to compute error metrics and consistency
statistics for positioning algorithms.

Author: Li-Ta Hsu
Date: December 2025
"""

from typing import Dict, Optional, Union

import numpy as np


def compute_position_errors(truth: np.ndarray, estimated: np.ndarray) -> np.ndarray:
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


def compute_rmse(
    errors: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
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
            f"covariance must have shape ({N}, {n}, {n}), " f"got {covariance.shape}"
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
        raise ValueError(f"S must have shape ({N}, {m}, {m}), got {S.shape}")

    nis = np.zeros(N)
    for i in range(N):
        try:
            S_inv = np.linalg.inv(S[i])
            nis[i] = innovation[i] @ S_inv @ innovation[i]
        except np.linalg.LinAlgError:
            nis[i] = np.nan

    return nis


def path_length(positions: np.ndarray) -> float:
    """
    Total distance travelled along a path.

    Args:
        positions: Positions, shape (N, D). Any dimension; pass ``pos[:, :2]``
            for the horizontal path.

    Returns:
        Sum of the distances between consecutive samples [m]. Zero for a
        single sample or an empty array.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2-D of shape (N, D), got {positions.shape}"
        )
    if len(positions) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def motion_ratio(est: np.ndarray, truth: np.ndarray) -> float:
    """
    How far an estimator travelled, relative to the ground truth.

    Guards against a degenerate estimator -- one that barely moves and so
    scores well by accident. On a closed-loop ground truth (a path returning
    to its start), *final position error* rewards standing still perfectly:
    Chapter 6's comparison reported final errors of 0.32 m and 1.13 m for two
    methods that had traced 0.5 m and 2.3 m against a 100 m walk, and the
    example printed "90-95% error reduction" on that basis. RMSE does not
    catch it either -- for a stationary estimate it merely measures the mean
    distance of the truth from the start point.

    A working estimator returns a ratio near 1. Well below 1 means it is not
    moving; well above 1 means it is accumulating spurious motion, which is
    the signature of an unaided integrator drifting.

    Args:
        est: Estimated positions, shape (N, D).
        truth: Ground-truth positions, shape (M, D). Need not match ``est``
            in length; only the traced distances are compared.

    Returns:
        ``path_length(est) / path_length(truth)``, or ``inf`` if the truth is
        stationary while the estimate is not (0.0 if neither moves).
    """
    truth_distance = path_length(truth)
    est_distance = path_length(est)

    if truth_distance == 0.0:
        return 0.0 if est_distance == 0.0 else float("inf")

    return est_distance / truth_distance
