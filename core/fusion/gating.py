"""Innovation gating utilities for sensor fusion (Chapter 8).

This module implements chi-square statistical gating for outlier detection
and measurement validation, as described in Chapter 8, Section 8.3.

Author: Navigation Engineer
References: Chapter 8, Section 8.3 (Tuning and Robustness)
"""

import numpy as np
from scipy import stats


def mahalanobis_distance_squared(
    y: np.ndarray,
    S: np.ndarray
) -> float:
    """Compute squared Mahalanobis distance of innovation.
    
    Implements Eq. (8.8) in Chapter 8:
        d_k^2 = y_k^T S_k^{-1} y_k
    
    This is the same quantity as the Normalized Innovation Squared (NIS)
    computed in core.eval.metrics.compute_nis. The squared Mahalanobis
    distance follows a chi-square distribution under the hypothesis that
    the measurement is consistent with the predicted state.
    
    Args:
        y: Innovation vector (m,).
        S: Innovation covariance matrix (m × m), must be positive definite.
    
    Returns:
        Squared Mahalanobis distance d^2 (scalar).
    
    Raises:
        ValueError: If dimensions are incompatible or S is not positive definite.
    
    Example:
        >>> y = np.array([1.0, 0.0])
        >>> S = np.diag([1.0, 1.0])
        >>> d_sq = mahalanobis_distance_squared(y, S)
        >>> np.allclose(d_sq, 1.0)
        True
        
        >>> # Larger innovation or smaller covariance → larger distance
        >>> y = np.array([3.0, 4.0])
        >>> S = np.diag([1.0, 1.0])
        >>> d_sq = mahalanobis_distance_squared(y, S)
        >>> np.allclose(d_sq, 25.0)  # 3^2 + 4^2
        True
    
    References:
        Eq. (8.8) in Chapter 8
        See also core.eval.metrics.compute_nis (equivalent computation)
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
    
    # Eq. (8.8): d_k^2 = y_k^T S_k^{-1} y_k
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Innovation covariance S is singular: {e}")
    
    d_squared = y.T @ S_inv @ y
    
    return float(d_squared)


def chi_square_gate(
    y: np.ndarray,
    S: np.ndarray,
    alpha: float = 0.05
) -> bool:
    """Chi-square gating decision for measurement validation.
    
    Implements Eq. (8.9) in Chapter 8:
        Accept measurement if d_k^2 < χ²(m, α)
        Reject measurement if d_k^2 ≥ χ²(m, α)
    
    where m is the measurement dimension and χ²(m, α) is the chi-square
    critical value at significance level α.
    
    Args:
        y: Innovation vector (m,).
        S: Innovation covariance matrix (m × m), must be positive definite.
        alpha: Significance level for the chi-square test (default 0.05).
               Typical values:
               - 0.01 (99% confidence, conservative gating)
               - 0.05 (95% confidence, standard gating)
               - 0.10 (90% confidence, aggressive gating)
    
    Returns:
        True if measurement should be accepted (innovation is consistent).
        False if measurement should be rejected (likely outlier).
    
    Raises:
        ValueError: If dimensions are incompatible, S is not positive definite,
                    or alpha is not in (0, 1).
    
    Example:
        >>> # Small innovation → accept
        >>> y = np.array([0.1, 0.2])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, alpha=0.05)
        True
        
        >>> # Large innovation → reject
        >>> y = np.array([5.0, 5.0])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, alpha=0.05)
        False
        
        >>> # More conservative gating (lower alpha) → easier to reject
        >>> y = np.array([2.5, 0.0])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, alpha=0.10)  # 90% confidence
        True
        >>> chi_square_gate(y, S, alpha=0.01)  # 99% confidence
        False
    
    Notes:
        The chi-square critical values for common cases:
        - m=1, α=0.05: χ² = 3.84
        - m=2, α=0.05: χ² = 5.99
        - m=3, α=0.05: χ² = 7.81
        
        Lower alpha (more conservative) leads to lower critical values,
        making it easier to reject measurements.
    
    References:
        Eq. (8.9) in Chapter 8
    """
    if not (0 < alpha < 1):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
    
    # Compute squared Mahalanobis distance (Eq. 8.8)
    d_squared = mahalanobis_distance_squared(y, S)
    
    # Degrees of freedom = measurement dimension
    m = len(y)
    
    # Chi-square critical value at significance level alpha
    # Note: scipy.stats.chi2.ppf(1 - alpha, m) gives the (1-alpha) quantile
    chi2_critical = stats.chi2.ppf(1.0 - alpha, m)
    
    # Eq. (8.9): Accept if d_k^2 < χ²(m, α)
    accept = d_squared < chi2_critical
    
    return bool(accept)


def chi_square_threshold(
    dof: int,
    alpha: float = 0.05
) -> float:
    """Get chi-square critical value for a given significance level.
    
    Computes χ²(m, α), the critical value for chi-square gating with
    m degrees of freedom at significance level α.
    
    Args:
        dof: Degrees of freedom m (measurement dimension).
        alpha: Significance level (default 0.05).
    
    Returns:
        Chi-square critical value.
    
    Example:
        >>> threshold = chi_square_threshold(dof=2, alpha=0.05)
        >>> np.allclose(threshold, 5.991, atol=0.01)
        True
        
        >>> # More conservative (lower alpha) → lower threshold
        >>> chi_square_threshold(dof=2, alpha=0.01)
        9.21...
    
    References:
        Used in Eq. (8.9) in Chapter 8
    """
    if dof < 1:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if not (0 < alpha < 1):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
    
    return float(stats.chi2.ppf(1.0 - alpha, dof))


def chi_square_bounds(
    dof: int,
    alpha: float = 0.05
) -> tuple[float, float]:
    """Get lower and upper chi-square bounds for consistency monitoring.
    
    Computes the symmetric confidence interval [χ²_lower, χ²_upper] for
    chi-square distributed statistics. Useful for NIS/NEES consistency plots.
    
    Args:
        dof: Degrees of freedom m.
        alpha: Significance level (default 0.05 for 95% confidence).
               The returned bounds cover the central (1 - alpha) probability mass.
    
    Returns:
        Tuple (lower_bound, upper_bound).
    
    Example:
        >>> lower, upper = chi_square_bounds(dof=2, alpha=0.05)
        >>> # For 2 DOF, 95% interval is approximately [0.05, 5.99]
        >>> 0.0 < lower < 0.2
        True
        >>> 5.5 < upper < 6.5
        True
    
    Notes:
        For consistency monitoring (e.g., NIS plots), the statistic should
        fall within these bounds approximately (1-alpha)% of the time if the
        filter is well-tuned.
    
    References:
        Chapter 8, Section 8.3 (Filter Consistency Monitoring)
    """
    if dof < 1:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if not (0 < alpha < 1):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
    
    # Two-sided interval: [alpha/2, 1 - alpha/2]
    lower = float(stats.chi2.ppf(alpha / 2.0, dof))
    upper = float(stats.chi2.ppf(1.0 - alpha / 2.0, dof))
    
    return lower, upper

