"""
Least Squares estimation algorithms.

This module implements various least squares methods for state estimation,
as described in Chapter 3, Section 3.1 of the IPIN book.

Functions:
    - linear_least_squares: Standard LS (Eq. 3.2-3.3)
    - weighted_least_squares: Weighted LS with measurement weights (Section 3.1.1)
    - iterative_least_squares: Iterative LS for nonlinear problems (Eq. 3.4)
    - robust_least_squares: Robust LS with outlier rejection (Table 3.1)

Book Reference:
    - Section 3.1: Least Squares Estimation
    - Section 3.1.1: Robust Estimators
    - Table 3.1: Robust estimator error functions (L2, Cauchy, Huber, G-M)
"""

from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np


def linear_least_squares(
    A: np.ndarray, b: np.ndarray, return_covariance: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standard linear least squares estimation.

    Solves: x_hat = argmin ||Ax - b||²
    Solution: x_hat = (A'A)^(-1) A'b

    **Implements Eq. (3.2)-(3.3)** from Chapter 3:
        - Eq. (3.2): Normal equations H'Hx̂ = H'y
        - Eq. (3.3): Closed-form solution x̂ = (H'H)⁻¹H'y

    Args:
        A: Design matrix (m × n), where m ≥ n.
        b: Observation vector (m × 1).
        return_covariance: If True, compute covariance matrix.

    Returns:
        Tuple of:
            - x_hat: Estimated state vector (n × 1).
            - P: Covariance matrix (n × n), or None if return_covariance is False.

    Raises:
        ValueError: If A and b dimensions don't match or A is rank deficient.

    Example:
        >>> import numpy as np
        >>> # Estimate position from 4 range measurements
        >>> A = np.array([[1, 0], [0, 1], [1, 1], [1, -1]])  # Design matrix
        >>> b = np.array([1.0, 2.0, 3.5, -0.5])  # Measured ranges
        >>> x_hat, P = linear_least_squares(A, b)
        >>> print(f"Position estimate: {x_hat}")
    """
    # Validate inputs
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(f"A must be 2D and b must be 1D. Got A: {A.shape}, b: {b.shape}")

    m, n = A.shape
    if m < n:
        raise ValueError(f"Underdetermined system: m={m} < n={n}. Need m ≥ n.")

    if len(b) != m:
        raise ValueError(f"Dimension mismatch: A has {m} rows, b has {len(b)} elements")

    # Check rank
    rank = np.linalg.matrix_rank(A)
    if rank < n:
        raise ValueError(
            f"A is rank deficient: rank={rank} < n={n}. " f"System has no unique solution."
        )

    # Compute normal equations: A'A x = A'b
    ATA = A.T @ A
    ATb = A.T @ b

    # Solve for state estimate
    try:
        x_hat = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to solve normal equations: {e}")

    # Compute covariance if requested
    P = None
    if return_covariance:
        # Residuals
        residuals = b - A @ x_hat
        # Estimated measurement variance (unbiased)
        if m > n:
            sigma2 = np.sum(residuals**2) / (m - n)
        else:
            sigma2 = 1.0  # Exact fit case

        # Covariance: P = sigma² (A'A)^(-1)
        P = sigma2 * np.linalg.inv(ATA)

    return x_hat, P


def weighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    W_or_sigma: np.ndarray,
    is_sigma: bool = False,
    return_covariance: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Weighted least squares estimation with measurement weights or covariance.

    Solves: x_hat = argmin (Ax - b)' W (Ax - b)
    Solution: x_hat = (A'WA)^(-1) A'Wb

    **Implements weighted LS** as described in Section 3.1.1 of Chapter 3:
        "Weighted least-squares assigns a weight wᵢ to each residual...
        setting wᵢ = 1/σᵢ² (the inverse of noise variance) yields the
        best linear unbiased estimate in the linear case."

    Args:
        A: Design matrix (m × n), where m ≥ n.
        b: Observation vector (m × 1).
        W_or_sigma: Weight specification, one of:
            - 2D array (m × m): Full weight matrix W
            - 1D array (m,): Diagonal weights wᵢ (if is_sigma=False)
            - 1D array (m,): Measurement std devs σᵢ (if is_sigma=True)
        is_sigma: If True, interpret 1D W_or_sigma as σᵢ and compute wᵢ = 1/σᵢ².
        return_covariance: If True, compute covariance matrix.

    Returns:
        Tuple of:
            - x_hat: Estimated state vector (n × 1).
            - P: Covariance matrix (n × n), or None if return_covariance is False.

    Raises:
        ValueError: If dimensions don't match or matrices are invalid.

    Example:
        >>> import numpy as np
        >>> # Estimate with different measurement uncertainties
        >>> A = np.array([[1, 0], [0, 1], [1, 1]])
        >>> b = np.array([1.0, 2.0, 3.2])
        >>> # Using measurement standard deviations (σᵢ)
        >>> sigma = np.array([0.1, 0.1, 0.5])  # Third measurement less accurate
        >>> x_hat, P = weighted_least_squares(A, b, sigma, is_sigma=True)
        >>>
        >>> # Or using diagonal weights directly
        >>> weights = np.array([100.0, 100.0, 4.0])  # wᵢ = 1/σᵢ²
        >>> x_hat, P = weighted_least_squares(A, b, weights, is_sigma=False)
        >>>
        >>> # Or using full weight matrix
        >>> W = np.diag([100.0, 100.0, 4.0])
        >>> x_hat, P = weighted_least_squares(A, b, W)
    """
    # Validate inputs
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(
            f"Invalid dimensions: A must be 2D, b must be 1D. "
            f"Got A={A.shape}, b={b.shape}"
        )

    m, n = A.shape
    if len(b) != m:
        raise ValueError(
            f"Dimension mismatch: A has {m} rows, b has {len(b)} elements"
        )

    # Process W_or_sigma into full weight matrix W
    W_or_sigma = np.asarray(W_or_sigma)

    if W_or_sigma.ndim == 1:
        # 1D input: diagonal weights or sigmas
        if len(W_or_sigma) != m:
            raise ValueError(
                f"W_or_sigma length mismatch: expected {m}, got {len(W_or_sigma)}"
            )

        if is_sigma:
            # Convert σᵢ to wᵢ = 1/σᵢ²
            if np.any(W_or_sigma <= 0):
                raise ValueError("Sigma values must be positive")
            weights = 1.0 / (W_or_sigma ** 2)
        else:
            weights = W_or_sigma
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative")

        W = np.diag(weights)

    elif W_or_sigma.ndim == 2:
        # 2D input: full weight matrix
        if W_or_sigma.shape != (m, m):
            raise ValueError(
                f"Weight matrix shape mismatch: expected ({m}, {m}), "
                f"got {W_or_sigma.shape}"
            )
        W = W_or_sigma

        # Check if W is symmetric
        if not np.allclose(W, W.T):
            raise ValueError("Weight matrix W must be symmetric")

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(W)
        if np.any(eigenvalues < -1e-10):  # Allow small numerical errors
            raise ValueError("Weight matrix W must be positive semi-definite")
    else:
        raise ValueError(
            f"W_or_sigma must be 1D or 2D array, got {W_or_sigma.ndim}D"
        )

    # Compute weighted normal equations: A'WA x = A'Wb
    ATWA = A.T @ W @ A
    ATWb = A.T @ W @ b

    # Check rank
    rank = np.linalg.matrix_rank(ATWA)
    if rank < n:
        raise ValueError(f"A'WA is rank deficient: rank={rank} < n={n}")

    # Solve for state estimate
    try:
        x_hat = np.linalg.solve(ATWA, ATWb)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to solve weighted normal equations: {e}")

    # Compute covariance if requested
    P = None
    if return_covariance:
        # Covariance: P = (A'WA)^(-1)
        P = np.linalg.inv(ATWA)

    return x_hat, P


def iterative_least_squares(
    f: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6,
    return_covariance: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Iterative least squares for nonlinear measurement models (Gauss-Newton).

    Solves: x_hat = argmin ||f(x) - b||²
    Iterates: x_{k+1} = x_k - (J'J)^(-1) J'(f(x_k) - b)

    **Implements Eq. (3.4)** from Chapter 3:
        "Σᵢ(yᵢ - hᵢ(x))∇hᵢ(x) = 0"
        Solved iteratively via Gauss-Newton method.

    Args:
        f: Nonlinear measurement function f: R^n -> R^m.
        jacobian: Function returning Jacobian matrix J = df/dx (m × n).
        b: Observation vector (m × 1).
        x0: Initial state estimate (n × 1).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on ||x_{k+1} - x_k||.
        return_covariance: If True, compute covariance at final estimate.

    Returns:
        Tuple of:
            - x_hat: Estimated state vector (n × 1).
            - P: Covariance matrix (n × n), or None if return_covariance is False.
            - iterations: Number of iterations performed.

    Example:
        >>> import numpy as np
        >>> # Range-only positioning (nonlinear)
        >>> anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> def f(x):
        ...     return np.linalg.norm(anchors - x, axis=1)
        >>> def jacobian(x):
        ...     diff = x - anchors
        ...     ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        ...     return diff / ranges
        >>> ranges = np.array([1.0, 0.5, 0.5, 0.7])  # Measured ranges
        >>> x0 = np.array([0.5, 0.5])  # Initial guess
        >>> x_hat, P, iters = iterative_least_squares(f, jacobian, ranges, x0)
    """
    # Validate inputs
    if not callable(f) or not callable(jacobian):
        raise ValueError("f and jacobian must be callable functions")

    if b.ndim != 1:
        raise ValueError(f"b must be 1D array, got shape {b.shape}")

    if x0.ndim != 1:
        raise ValueError(f"x0 must be 1D array, got shape {x0.shape}")

    x = x0.copy()
    m = len(b)

    for iteration in range(max_iter):
        # Evaluate function and Jacobian at current estimate
        fx = f(x)
        if len(fx) != m:
            raise ValueError(
                f"Function f returned wrong dimension: expected {m}, got {len(fx)}"
            )

        J = jacobian(x)
        if J.shape != (m, len(x)):
            raise ValueError(
                f"Jacobian has wrong shape: expected ({m}, {len(x)}), got {J.shape}"
            )

        # Compute residual
        r = b - fx

        # Gauss-Newton update: Δx = (J'J)^(-1) J'r
        JTJ = J.T @ J
        JTr = J.T @ r

        try:
            delta_x = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            delta_x = np.linalg.lstsq(JTJ, JTr, rcond=None)[0]

        # Update estimate
        x = x + delta_x

        # Check convergence
        if np.linalg.norm(delta_x) < tol:
            break

    # Compute final covariance if requested
    P = None
    if return_covariance:
        # Evaluate Jacobian at final estimate
        J = jacobian(x)
        JTJ = J.T @ J

        # Estimate measurement variance
        fx = f(x)
        residuals = b - fx
        n = len(x)
        if m > n:
            sigma2 = np.sum(residuals**2) / (m - n)
        else:
            sigma2 = 1.0

        # Covariance: P = sigma² (J'J)^(-1)
        try:
            P = sigma2 * np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            P = sigma2 * np.linalg.pinv(JTJ)

    return x, P, iteration + 1


# Type alias for robust loss methods
RobustLossMethod = Literal["l2", "huber", "cauchy", "gm", "geman_mcclure", "tukey"]


def robust_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    method: RobustLossMethod = "huber",
    threshold: float = 1.5,
    max_iter: int = 10,
    tol: float = 1e-4,
    return_covariance: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Robust least squares with outlier rejection using IRLS.

    Solves: x_hat = argmin Σ ρ(rᵢ) where rᵢ = (Ax - b)ᵢ
    Uses Iteratively Reweighted Least Squares (IRLS).

    **Implements robust estimators from Table 3.1** in Chapter 3, Section 3.1.1.

    Robust Loss Functions (Table 3.1):
    ---------------------------------
    | Method   | Error Function e(x)                              |
    |----------|--------------------------------------------------|
    | L2       | e(x) = ½‖r(x)‖²                                  |
    | Cauchy   | e(x) = ½ ln(1 + ‖r(x)‖²)                         |
    | Huber    | e(x) = { ½‖r‖² if |r|≤δ; δ(|r|-½δ) otherwise }  |
    | G-M      | e(x) = ½ ‖r(x)‖² / (1 + ‖r(x)‖²)                 |

    The `threshold` parameter serves as residual normalization scale (δ in Huber,
    c in others), allowing the formulas to match the book while providing
    practical tuning capability.

    Args:
        A: Design matrix (m × n).
        b: Observation vector (m × 1).
        method: Robust loss function from Table 3.1:
            - "l2": Standard L2 norm (no outlier rejection)
            - "huber": Huber loss (quadratic near zero, linear for large residuals)
            - "cauchy": Cauchy/Lorentzian loss (heavy-tailed)
            - "gm" or "geman_mcclure": Geman-McClure loss (strong outlier rejection)
            - "tukey": Tukey biweight (*Extra, not in book Table 3.1*)
        threshold: Scale parameter for residual normalization (default 1.5).
            For normalized residuals r̃ = r/σ, the loss is computed on r̃/threshold.
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance on ||x_{k+1} - x_k||.
        return_covariance: If True, compute covariance.

    Returns:
        Tuple of:
            - x_hat: Estimated state vector (n × 1).
            - P: Covariance matrix (n × n), or None.
            - weights: Final weights for each measurement (m × 1).

    Note:
        - Residuals are internally normalized using MAD (Median Absolute Deviation)
          for robust scale estimation: σ = 1.4826 × MAD(residuals).
        - Weights are computed from the derivative of ρ(r): w(r) = (1/r) × dρ/dr.

    Example:
        >>> import numpy as np
        >>> # Data with outliers
        >>> A = np.vstack([np.eye(3), np.eye(3)])
        >>> b = np.array([1.0, 2.0, 3.0, 1.1, 2.2, 10.0])  # Last is outlier
        >>> x_hat, P, weights = robust_least_squares(A, b, method="huber")
        >>> print(f"Outlier weight: {weights[-1]:.3f}")  # Should be < 1.0
        >>>
        >>> # Using Geman-McClure for stronger outlier rejection
        >>> x_hat_gm, _, weights_gm = robust_least_squares(A, b, method="gm")
        >>> print(f"G-M outlier weight: {weights_gm[-1]:.4f}")  # Even smaller
    """
    # Validate inputs
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(f"A must be 2D and b must be 1D. Got A: {A.shape}, b: {b.shape}")

    m, n = A.shape
    if len(b) != m:
        raise ValueError(f"Dimension mismatch: A has {m} rows, b has {len(b)} elements")

    # Normalize method name
    method_lower = method.lower()
    if method_lower == "geman_mcclure":
        method_lower = "gm"

    valid_methods = ["l2", "huber", "cauchy", "gm", "tukey"]
    if method_lower not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use one of: {valid_methods} (or 'geman_mcclure' for 'gm')"
        )

    # L2 is just standard least squares
    if method_lower == "l2":
        x, P = linear_least_squares(A, b, return_covariance=return_covariance)
        weights = np.ones(m)
        return x, P, weights

    # Initial estimate with standard LS
    x, _ = linear_least_squares(A, b, return_covariance=False)

    # IRLS iterations
    weights = np.ones(m)
    for iteration in range(max_iter):
        # Compute residuals
        residuals = b - A @ x

        # Robust standard deviation (MAD-based)
        # σ = 1.4826 × median(|r - median(r)|)
        # The factor 1.4826 makes MAD consistent with std dev for Gaussian data
        sigma = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        if sigma < 1e-10:
            sigma = 1.0  # Avoid division by zero

        # Normalized residuals: r̃ = r / σ
        normalized_res = residuals / sigma

        # Compute weights based on robust function
        # Weight w(r) = (1/r) × dρ/dr for IRLS
        weights = _compute_robust_weights(normalized_res, method_lower, threshold)

        # Weighted least squares update
        W = np.diag(weights)
        x_new, _ = weighted_least_squares(A, b, W, return_covariance=False)

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    # Final covariance if requested
    P = None
    if return_covariance:
        W = np.diag(weights)
        _, P = weighted_least_squares(A, b, W, return_covariance=True)

    return x, P, weights


def _compute_robust_weights(
    normalized_res: np.ndarray, method: str, threshold: float
) -> np.ndarray:
    """
    Compute IRLS weights for robust estimation.

    For IRLS, the weight function is w(r) = (1/r) × dρ/dr, where ρ is the
    robust loss function from Table 3.1.

    Args:
        normalized_res: Residuals normalized by robust scale estimate.
        method: One of "huber", "cauchy", "gm", "tukey".
        threshold: Scale parameter (c or δ in the book formulas).

    Returns:
        weights: Weight for each residual (m,).
    """
    # Scaled residual: u = r / c
    u = normalized_res / threshold
    abs_u = np.abs(u)

    if method == "huber":
        # Huber loss: ρ(r) = { ½r² if |r|≤c; c|r|-½c² otherwise }
        # Weight: w = { 1 if |r|≤c; c/|r| otherwise }
        weights = np.where(abs_u <= 1.0, 1.0, 1.0 / abs_u)

    elif method == "cauchy":
        # Cauchy loss: ρ(r) = ½ ln(1 + r²)
        # Weight: w = 1 / (1 + r²)
        weights = 1.0 / (1.0 + u ** 2)

    elif method == "gm":
        # Geman-McClure (G-M) loss: ρ(r) = ½ r² / (1 + r²)
        # Weight: w = 1 / (1 + r²)²
        weights = 1.0 / (1.0 + u ** 2) ** 2

    elif method == "tukey":
        # Tukey biweight (*Extra, not in book Table 3.1*)
        # ρ(r) = { (c²/6)[1-(1-(r/c)²)³] if |r|≤c; c²/6 otherwise }
        # Weight: w = { (1-(r/c)²)² if |r|≤c; 0 otherwise }
        weights = np.where(abs_u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
        # Add small epsilon to avoid singular weight matrix
        weights = np.maximum(weights, 1e-10)

    else:
        # Should not reach here due to validation
        raise ValueError(f"Unknown method: {method}")

    return weights
