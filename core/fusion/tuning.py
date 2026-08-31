"""Tuning and robustness utilities for sensor fusion (Chapter 8).

This module implements innovation monitoring, covariance tuning, and robust
measurement weighting as described in Chapter 8, Section 8.3.

Robust Covariance Scaling (Eq. 8.7):
    R_k ← w_R(y_k) * R_k

    Where w_R >= 1 is a covariance scale factor that **inflates** R for outliers.
    This reduces the influence of measurements with large innovations without
    completely rejecting them.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3 (Tuning and Robustness), Equation 8.7
"""

import warnings
from typing import NamedTuple, cast

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

    return cast(np.ndarray, z - z_pred)


def innovation_covariance(
    H: np.ndarray, P_pred: np.ndarray, R: np.ndarray
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

    return cast(np.ndarray, S)


class KalmanUpdateResult(NamedTuple):
    """Joseph-form measurement update output; unpacks as ``(state, covariance)``."""

    updated_state_estimate: np.ndarray
    updated_state_covariance: np.ndarray


def kalman_update(
    state_estimate: np.ndarray,
    state_covariance: np.ndarray,
    innovation: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
) -> KalmanUpdateResult:
    """Apply one measurement update in Joseph form, deriving S from ``P``.

    Returns new arrays; ``x`` and ``P`` are not modified.

        K = P H^T S^-1,  S = H P H^T + R
        x <- x + K y
        P <- (I - K H) P (I - K H)^T + K R K^T

    Two properties are the point of this function, and Chapter 8's fusion
    runners lost both by hand-rolling the update three times over:

    **S is computed here rather than accepted as an argument.** A caller that
    computes S for the chi-square gate, then inflates P, then reuses that S
    for the gain is applying a gain matched to a covariance the filter no
    longer has. `core.fusion.adaptive.AdaptiveGatingManager` inflates P by
    lambda after S is formed, which is exactly that sequence.

    **The Joseph form, not the short form (I - K H) P.** The short form is
    algebraically equal to this one *only at the optimal gain*. Off it, the
    variance along H scales by 1 - lambda*HPH'/(HPH' + R), which is negative
    whenever lambda*HPH' > HPH' + R. The loosely coupled run measured
    min trace(P) = -0.0814 with 88 of 6587 samples negative, worst at
    t = 57.6 s, and drew it in the covariance panel of
    `ch8_sensor_fusion/figs/lc_uwb_imu_results.png`. Joseph is a sum of two
    congruence transforms, so it is positive semidefinite for *any* K --
    which is the reason `core/estimators/extended_kalman_filter.py:196` uses
    it, and why the runners now share this one implementation instead of
    three copies of the short form.

    Args:
        x: Current state estimate (n,).
        P: Current state covariance (n, n).
        y: Innovation z - h(x), shape (m,).
        H: Measurement Jacobian (m, n).
        R: Measurement noise covariance (m, m).

    Returns:
        Tuple of (x_updated, P_updated).

    Example:
        >>> x = np.array([0.0, 0.0])
        >>> P = np.diag([1.0, 1.0])
        >>> H = np.array([[1.0, 0.0]])
        >>> R = np.array([[1.0]])
        >>> x_new, P_new = kalman_update(x, P, np.array([2.0]), H, R)
        >>> np.allclose(x_new, [1.0, 0.0])
        True
        >>> np.allclose(P_new, np.diag([0.5, 1.0]))
        True

        Joseph stays positive semidefinite even at a badly mismatched gain,
        where the short form does not:

        >>> P_inflated = 100.0 * P
        >>> _, P_out = kalman_update(x, P_inflated, np.array([2.0]), H, R)
        >>> float(np.linalg.eigvalsh(P_out).min()) >= 0.0
        True

    References:
        Chapter 8, Section 8.2; Joseph form correcting Eq. (3.19) (see docs/book_errata.md E-01)
    """
    # Book notation (docs/api_naming_conventions.md): x = state estimate,
    # P = state covariance, y = innovation, H = measurement matrix,
    # R = measurement covariance.
    x = state_estimate
    P = state_covariance
    y = innovation
    H = measurement_matrix
    R = measurement_covariance
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float)

    S = innovation_covariance(H, P, R)
    K = P @ H.T @ np.linalg.inv(S)

    x_updated = x + (K @ y).flatten()

    I_KH = np.eye(P.shape[0]) - K @ H
    P_updated = I_KH @ P @ I_KH.T + K @ R @ K.T

    return KalmanUpdateResult(
        x_updated, cast(np.ndarray, 0.5 * (P_updated + P_updated.T))
    )


def scale_measurement_covariance(R: np.ndarray, scale_factor: float) -> np.ndarray:
    """Apply robust scaling to measurement covariance (Eq. 8.7).

    Implements Eq. (8.7) in Chapter 8:
        R_k ← w_R(y_k) * R_k

    where w_R >= 1 is a covariance scale factor that **inflates** R for outliers.
    This reduces the influence of measurements with large innovations without
    completely rejecting them (softer alternative to chi-square gating).

    Args:
        R: Original measurement covariance (m × m).
        scale_factor: Covariance inflation factor w_R >= 1.
                      - w_R = 1: no inflation (inlier)
                      - w_R > 1: inflate covariance (outlier)
                      Typical robust functions return values in [1, ∞).

    Returns:
        Scaled covariance R_scaled = w_R * R (m × m).

    Raises:
        ValueError: If scale_factor < 1 or R is not 2D.

    Example:
        >>> R = np.diag([0.1, 0.2])

        >>> # Inlier: no scaling
        >>> R_inlier = scale_measurement_covariance(R, 1.0)
        >>> np.allclose(R_inlier, R)
        True

        >>> # Moderate outlier: inflate by 2x
        >>> R_scaled = scale_measurement_covariance(R, 2.0)
        >>> np.allclose(R_scaled, [[0.2, 0.0], [0.0, 0.4]])
        True

        >>> # Strong outlier: inflate by 100x (nearly reject)
        >>> R_outlier = scale_measurement_covariance(R, 100.0)
        >>> np.allclose(R_outlier, [[10.0, 0.0], [0.0, 20.0]])
        True

    Notes:
        **Key Point:** Outliers get **larger** covariance (inflated R), which
        reduces their weight in the Kalman gain K = P H^T S^{-1}.

        Use the companion functions to compute scale factors:
        - `huber_R_scale(r, delta)` for Huber robust loss
        - `cauchy_R_scale(r, c)` for Cauchy robust loss

    References:
        Eq. (8.7) in Chapter 8
    """
    R = np.asarray(R)

    if R.ndim != 2:
        raise ValueError(f"R must be 2D matrix, got shape {R.shape}")

    if not isinstance(scale_factor, (int, float, np.number)):
        raise TypeError(f"Scale factor must be numeric, got {type(scale_factor)}")

    if scale_factor < 1.0:
        raise ValueError(
            f"Scale factor must be >= 1 (inflate covariance for outliers), "
            f"got {scale_factor}"
        )

    # Eq. (8.7): R_k ← w_R(y_k) * R_k
    R_scaled = scale_factor * R

    return R_scaled


def huber_R_scale(residual: float, delta: float = 1.345) -> float:
    """Compute Huber covariance scale factor for Eq. 8.7.

    Returns a scale factor w_R >= 1 that inflates measurement covariance R
    for large residuals (outliers). Implements the Huber robust loss function
    as a covariance inflation strategy.

    Scale factor:
        w_R(r) = 1                if |r| ≤ δ  (inlier: no inflation)
        w_R(r) = |r| / δ          if |r| > δ  (outlier: inflate by ratio)

    Args:
        residual: Normalized residual, r = innovation / sigma. **Not an
            innovation in the measurement's own units** -- delta is compared
            against it directly, so handing this metres silently disables the
            function whenever the innovations are smaller than delta. Prefer a
            *fixed* sigma, normally sqrt(R): sqrt(S) tracks the filter's own
            confidence, so an over-confident filter shrinks its own scale,
            inflates R, drifts, and enlarges the next residual. See
            ch8_sensor_fusion/example_robust_tuning.py, which measures both.
        delta: Huber threshold δ, in the same units as ``residual`` -- so in
            multiples of sigma. The default 1.345 gives 95% efficiency on
            *Gaussian* residuals; a filter whose innovations are heavy-tailed
            needs a threshold measured against its own clean-data tail, which
            can be far larger.

    Returns:
        Covariance scale factor w_R >= 1.

    Example:
        >>> # Inlier: no inflation
        >>> huber_R_scale(0.5, delta=1.345)
        1.0

        >>> # Moderate outlier: inflate proportionally
        >>> huber_R_scale(2.69, delta=1.345)
        2.0

        >>> # Strong outlier: large inflation
        >>> scale = huber_R_scale(10.0, delta=1.345)
        >>> scale > 7.0
        True

    Notes:
        For Eq. 8.7 application: R_robust = huber_R_scale(r, delta) * R

        The Huber function provides a **linear** inflation for outliers,
        making it less aggressive than Cauchy.

    References:
        Chapter 8, Section 8.3.2 (Robust Loss Functions)
        Eq. (8.7): R_k ← w_R(y_k) * R_k
    """
    abs_residual = abs(residual)

    if abs_residual <= delta:
        return 1.0
    else:
        # Outliers: inflate R proportional to residual magnitude
        return abs_residual / delta


def cauchy_R_scale(residual: float, c: float = 2.385) -> float:
    """Compute Cauchy covariance scale factor for Eq. 8.7.

    Returns a scale factor w_R >= 1 that inflates measurement covariance R
    for large residuals (outliers). Implements the Cauchy robust loss function
    as a covariance inflation strategy.

    Scale factor:
        w_R(r) = 1 + (r / c)²

    Args:
        residual: Normalized residual, r = innovation / sigma, exactly as for
            ``huber_R_scale`` -- read that argument's note, including why a
            fixed sqrt(R) is preferred over sqrt(S).
        c: Cauchy scale parameter, in the same units as ``residual`` -- so in
            multiples of sigma. Larger values are more tolerant of outliers.
            The default 2.385 gives 95% efficiency on *Gaussian* residuals.
            Note there is no dead zone here: w_R exceeds 1 for every nonzero
            residual, so unlike Huber this cannot be made free on clean data,
            only cheap -- choose c against the clean-data residual tail.

    Returns:
        Covariance scale factor w_R >= 1.

    Example:
        >>> # Inlier: minimal inflation
        >>> cauchy_R_scale(0.0, c=2.385)
        1.0

        >>> # Moderate outlier
        >>> scale = cauchy_R_scale(2.385, c=2.385)
        >>> np.isclose(scale, 2.0)
        True

        >>> # Strong outlier: quadratic inflation
        >>> scale = cauchy_R_scale(10.0, c=2.385)
        >>> scale > 17.0
        True

    Notes:
        For Eq. 8.7 application: R_robust = cauchy_R_scale(r, c) * R

        The Cauchy function provides **quadratic** inflation for outliers,
        making it more aggressive than Huber. It strongly down-weights
        measurements with large innovations.

    References:
        Chapter 8, Section 8.3.2 (Robust Loss Functions)
        Eq. (8.7): R_k ← w_R(y_k) * R_k
    """
    normalized = residual / c
    # Outliers: inflate R quadratically
    return 1.0 + normalized**2


def huber_weight(residual: float, threshold: float) -> float:
    """Compute Huber robust weight for a scalar residual.

    **DEPRECATED:** Use `huber_R_scale()` for Eq. 8.7 covariance inflation.

    This function returns IRLS-style weights w ∈ (0, 1] that are used in
    iterative optimization. For Kalman filtering with Eq. 8.7, use the
    inverse relationship: R_scale = 1 / w, which gives inflation factors >= 1.

    The Huber weight function:
        w(r) = 1                if |r| ≤ k  (inlier)
        w(r) = k / |r|          if |r| > k  (outlier: weight < 1)

    Args:
        residual: Normalized residual (e.g., innovation / sqrt(variance)).
        threshold: Huber threshold k (typical values: 1.345 for 95% efficiency
                   on Gaussian data, or 2.0-3.0 for more tolerant).

    Returns:
        IRLS weight w(r) ∈ (0, 1]. For Eq. 8.7, use R_scale = 1/w.

    Example:
        >>> huber_weight(0.5, threshold=1.345)  # inlier
        1.0
        >>> w = huber_weight(3.0, threshold=1.345)  # outlier
        >>> np.isclose(w, 0.448, atol=0.01)
        True
        >>> # For Eq. 8.7: R_scale = 1/w ≈ 2.23 (inflate R)

    References:
        Chapter 8, Section 8.3 (Robust Loss Functions)
    """
    warnings.warn(
        "huber_weight() returns IRLS weights (0,1] which can be confusing for "
        "covariance inflation. Use huber_R_scale() instead for Eq. 8.7, which "
        "returns scale factors >= 1 that directly inflate R.",
        DeprecationWarning,
        stacklevel=2,
    )

    abs_residual = abs(residual)

    if abs_residual <= threshold:
        return 1.0
    else:
        return threshold / abs_residual


def cauchy_weight(residual: float, scale: float) -> float:
    """Compute Cauchy robust weight for a scalar residual.

    **DEPRECATED:** Use `cauchy_R_scale()` for Eq. 8.7 covariance inflation.

    This function returns IRLS-style weights w ∈ (0, 1] that are used in
    iterative optimization. For Kalman filtering with Eq. 8.7, use the
    inverse relationship: R_scale = 1 / w, which gives inflation factors >= 1.

    The Cauchy weight function:
        w(r) = 1 / (1 + (r/c)²)  (outliers get weight << 1)

    Args:
        residual: Normalized residual (e.g., innovation / sqrt(variance)).
        scale: Cauchy scale parameter c (typical values: 2.385 for 95% efficiency).

    Returns:
        IRLS weight w(r) ∈ (0, 1]. For Eq. 8.7, use R_scale = 1/w.

    Example:
        >>> cauchy_weight(0.0, scale=2.385)
        1.0
        >>> cauchy_weight(2.385, scale=2.385)
        0.5
        >>> w = cauchy_weight(10.0, scale=2.385)  # strong outlier
        >>> w < 0.06
        True
        >>> # For Eq. 8.7: R_scale = 1/w ≈ 18.6 (strongly inflate R)

    References:
        Chapter 8, Section 8.3 (Robust Loss Functions)
    """
    warnings.warn(
        "cauchy_weight() returns IRLS weights (0,1] which can be confusing for "
        "covariance inflation. Use cauchy_R_scale() instead for Eq. 8.7, which "
        "returns scale factors >= 1 that directly inflate R.",
        DeprecationWarning,
        stacklevel=2,
    )

    normalized = residual / scale
    return 1.0 / (1.0 + normalized**2)


def compute_normalized_innovation(y: np.ndarray, S: np.ndarray) -> np.ndarray:
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
