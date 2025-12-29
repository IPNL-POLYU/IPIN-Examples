"""Innovation gating utilities for sensor fusion (Chapter 8).

This module implements chi-square statistical gating for outlier detection
and measurement validation, as described in Chapter 8, Section 8.3.

The API uses 'confidence' parameter (e.g., 0.95 for 95% confidence) to match
the book's notation in Equations 8.8-8.9, where α represents the upper quantile
of the chi-square distribution.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3 (Tuning and Robustness), Equations 8.8-8.9
"""

import warnings
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
    confidence: float = None,
    alpha: float = None
) -> bool:
    """Chi-square gating decision for measurement validation.
    
    Implements Eq. (8.9) in Chapter 8:
        Accept measurement if d_k^2 < χ²(m, α)
        Reject measurement if d_k^2 ≥ χ²(m, α)
    
    where m is the measurement dimension and χ²(m, α) is the chi-square
    critical value at confidence level α (e.g., α=0.95 for 95% confidence).
    
    Args:
        y: Innovation vector (m,).
        S: Innovation covariance matrix (m × m), must be positive definite.
        confidence: Confidence level α (default 0.95 for 95% confidence).
                    Typical values:
                    - 0.99 (99% confidence, very conservative)
                    - 0.95 (95% confidence, standard)
                    - 0.90 (90% confidence, less conservative)
        alpha: DEPRECATED. Use 'confidence' instead. If provided, treated as
               significance level (1 - confidence) for backward compatibility.
    
    Returns:
        True if measurement should be accepted (innovation is consistent).
        False if measurement should be rejected (likely outlier).
    
    Raises:
        ValueError: If dimensions are incompatible, S is not positive definite,
                    or confidence is not in (0, 1).
    
    Example:
        >>> # Small innovation → accept (95% confidence)
        >>> y = np.array([0.1, 0.2])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, confidence=0.95)
        True
        
        >>> # Large innovation → reject (95% confidence)
        >>> y = np.array([5.0, 5.0])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, confidence=0.95)
        False
        
        >>> # More conservative gating (higher confidence) → easier to reject
        >>> y = np.array([2.5, 0.0])
        >>> S = np.diag([1.0, 1.0])
        >>> chi_square_gate(y, S, confidence=0.90)  # 90% confidence
        True
        >>> chi_square_gate(y, S, confidence=0.99)  # 99% confidence
        False
    
    Notes:
        The chi-square critical values for common cases (95% confidence):
        - m=1, α=0.95: χ² ≈ 3.841
        - m=2, α=0.95: χ² ≈ 5.991
        - m=3, α=0.95: χ² ≈ 7.815
        
        Higher confidence (larger α) leads to higher critical values,
        making it harder to reject measurements (more conservative).
    
    References:
        Eq. (8.9) in Chapter 8
    """
    # Handle backward compatibility
    if alpha is not None and confidence is None:
        warnings.warn(
            "Parameter 'alpha' is deprecated and will be removed in a future version. "
            "Use 'confidence' instead. Note: 'alpha' was interpreted as significance "
            "level (1 - confidence). To maintain equivalent behavior, use "
            f"confidence={1.0 - alpha:.2f} instead of alpha={alpha:.2f}.",
            DeprecationWarning,
            stacklevel=2
        )
        confidence = 1.0 - alpha
    elif confidence is None:
        # Default to 95% confidence
        confidence = 0.95
    
    if not (0 < confidence < 1):
        raise ValueError(
            f"Confidence level must be in (0, 1), got {confidence}"
        )
    
    # Compute squared Mahalanobis distance (Eq. 8.8)
    d_squared = mahalanobis_distance_squared(y, S)
    
    # Degrees of freedom = measurement dimension
    m = len(y)
    
    # Chi-square critical value at confidence level α
    # Book notation: α is the upper quantile
    chi2_critical = chi_square_threshold(dof=m, confidence=confidence)
    
    # Eq. (8.9): Accept if d_k^2 < χ²(m, α)
    accept = d_squared < chi2_critical
    
    return bool(accept)


def chi_square_threshold(
    dof: int,
    confidence: float = None,
    alpha: float = None
) -> float:
    """Get chi-square critical value for a given confidence level.
    
    Computes χ²(m, α), the critical value for chi-square gating with
    m degrees of freedom at confidence level α (Chapter 8, Eq. 8.9).
    
    In the book's notation, α is the upper quantile (e.g., α=0.95 for 95%
    confidence), not the significance level.
    
    Args:
        dof: Degrees of freedom m (measurement dimension).
        confidence: Confidence level α (default 0.95 for 95% confidence).
                    Common values:
                    - 0.99 (99% confidence, very conservative)
                    - 0.95 (95% confidence, standard)
                    - 0.90 (90% confidence, less conservative)
        alpha: DEPRECATED. Use 'confidence' instead. If provided, treated as
               significance level (1 - confidence) for backward compatibility.
    
    Returns:
        Chi-square critical value χ²(m, α).
    
    Example:
        >>> # Standard 95% confidence for 1 DOF
        >>> threshold = chi_square_threshold(dof=1, confidence=0.95)
        >>> np.allclose(threshold, 3.841, atol=0.01)
        True
        
        >>> # Standard 95% confidence for 2 DOF
        >>> threshold = chi_square_threshold(dof=2, confidence=0.95)
        >>> np.allclose(threshold, 5.991, atol=0.01)
        True
        
        >>> # More conservative (higher confidence) → higher threshold
        >>> chi_square_threshold(dof=2, confidence=0.99)
        9.21...
    
    References:
        Eq. (8.9) in Chapter 8: Accept if d_k^2 < χ²(m, α)
    """
    # Handle backward compatibility
    if alpha is not None and confidence is None:
        warnings.warn(
            "Parameter 'alpha' is deprecated and will be removed in a future version. "
            "Use 'confidence' instead. Note: 'alpha' was interpreted as significance "
            "level (1 - confidence). To maintain equivalent behavior, use "
            f"confidence={1.0 - alpha:.2f} instead of alpha={alpha:.2f}.",
            DeprecationWarning,
            stacklevel=2
        )
        confidence = 1.0 - alpha
    elif confidence is None:
        # Default to 95% confidence
        confidence = 0.95
    
    if dof < 1:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if not (0 < confidence < 1):
        raise ValueError(
            f"Confidence level must be in (0, 1), got {confidence}"
        )
    
    # Book notation: α is the upper quantile (confidence level)
    # scipy.stats.chi2.ppf(confidence, dof) gives the α-quantile
    return float(stats.chi2.ppf(confidence, dof))


def chi_square_bounds(
    dof: int,
    confidence: float = None,
    alpha: float = None
) -> tuple[float, float]:
    """Get lower and upper chi-square bounds for consistency monitoring.
    
    Computes the symmetric confidence interval [χ²_lower, χ²_upper] for
    chi-square distributed statistics. Useful for NIS/NEES consistency plots.
    
    Args:
        dof: Degrees of freedom m.
        confidence: Confidence level (default 0.95 for 95% confidence).
                    The returned bounds cover the central 'confidence'
                    probability mass.
        alpha: DEPRECATED. Use 'confidence' instead. If provided, treated as
               significance level (1 - confidence) for backward compatibility.
    
    Returns:
        Tuple (lower_bound, upper_bound).
    
    Example:
        >>> lower, upper = chi_square_bounds(dof=2, confidence=0.95)
        >>> # For 2 DOF, 95% interval is approximately [0.05, 5.99]
        >>> 0.0 < lower < 0.2
        True
        >>> 5.5 < upper < 6.5
        True
        
        >>> # For 1 DOF, 95% interval
        >>> lower, upper = chi_square_bounds(dof=1, confidence=0.95)
        >>> 0.0 < lower < 0.01
        True
        >>> 3.5 < upper < 4.0
        True
    
    Notes:
        For consistency monitoring (e.g., NIS plots), the statistic should
        fall within these bounds approximately 'confidence'% of the time if
        the filter is well-tuned.
        
        The bounds are computed as the symmetric two-sided interval:
        - lower = ppf((1 - confidence) / 2)
        - upper = ppf((1 + confidence) / 2)
    
    References:
        Chapter 8, Section 8.3 (Filter Consistency Monitoring)
    """
    # Handle backward compatibility
    if alpha is not None and confidence is None:
        warnings.warn(
            "Parameter 'alpha' is deprecated and will be removed in a future version. "
            "Use 'confidence' instead. Note: 'alpha' was interpreted as significance "
            "level (1 - confidence). To maintain equivalent behavior, use "
            f"confidence={1.0 - alpha:.2f} instead of alpha={alpha:.2f}.",
            DeprecationWarning,
            stacklevel=2
        )
        confidence = 1.0 - alpha
    elif confidence is None:
        # Default to 95% confidence
        confidence = 0.95
    
    if dof < 1:
        raise ValueError(f"Degrees of freedom must be positive, got {dof}")
    if not (0 < confidence < 1):
        raise ValueError(
            f"Confidence level must be in (0, 1), got {confidence}"
        )
    
    # Two-sided interval: [(1-conf)/2, (1+conf)/2]
    # This is the central 'confidence' interval
    lower = float(stats.chi2.ppf((1.0 - confidence) / 2.0, dof))
    upper = float(stats.chi2.ppf((1.0 + confidence) / 2.0, dof))
    
    return lower, upper


