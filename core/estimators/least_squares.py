"""
Least Squares estimation algorithms.

This module implements various least squares methods for state estimation,
as described in Chapter 3 of the IPIN book.

Functions:
    - linear_least_squares: Standard LS (Eq. 3.1)
    - weighted_least_squares: Weighted LS with measurement covariance (Eq. 3.2)
    - iterative_least_squares: Iterative LS for nonlinear problems (Eq. 3.3)
    - robust_least_squares: Robust LS with outlier rejection (Eq. 3.4)
"""

from typing import Callable, Optional, Tuple

import numpy as np


def linear_least_squares(
    A: np.ndarray, b: np.ndarray, return_covariance: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standard linear least squares estimation.

    Solves: x_hat = argmin ||Ax - b||²
    Solution: x_hat = (A'A)^(-1) A'b

    **Implements Eq. (3.1)** from Chapter 3.

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
    A: np.ndarray, b: np.ndarray, W: np.ndarray, return_covariance: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Weighted least squares estimation with measurement covariance.

    Solves: x_hat = argmin (Ax - b)' W (Ax - b)
    Solution: x_hat = (A'WA)^(-1) A'Wb

    **Implements Eq. (3.2)** from Chapter 3.

    Args:
        A: Design matrix (m × n), where m ≥ n.
        b: Observation vector (m × 1).
        W: Weight matrix (m × m). Typically W = R^(-1) where R is measurement covariance.
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
        >>> # Higher weight for more accurate measurements
        >>> W = np.diag([1.0, 1.0, 0.5])  # Third measurement less accurate
        >>> x_hat, P = weighted_least_squares(A, b, W)
    """
    # Validate inputs
    if A.ndim != 2 or b.ndim != 1 or W.ndim != 2:
        raise ValueError(
            f"Invalid dimensions: A={A.shape}, b={b.shape}, W={W.shape}"
        )

    m, n = A.shape
    if len(b) != m or W.shape != (m, m):
        raise ValueError(
            f"Dimension mismatch: A=({m},{n}), b={len(b)}, W={W.shape}"
        )

    # Check if W is symmetric and positive semi-definite
    if not np.allclose(W, W.T):
        raise ValueError("Weight matrix W must be symmetric")

    eigenvalues = np.linalg.eigvalsh(W)
    if np.any(eigenvalues < -1e-10):  # Allow small numerical errors
        raise ValueError("Weight matrix W must be positive semi-definite")

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
    Iterative least squares for nonlinear measurement models.

    Solves: x_hat = argmin ||f(x) - b||²
    Iterates: x_{k+1} = x_k - (J'J)^(-1) J'(f(x_k) - b)

    **Implements Eq. (3.3)** from Chapter 3 (Gauss-Newton method).

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


def robust_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "huber",
    threshold: float = 1.5,
    max_iter: int = 10,
    tol: float = 1e-4,
    return_covariance: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Robust least squares with outlier rejection.

    Solves: x_hat = argmin Σ ρ(r_i) where r_i = (Ax - b)_i
    Uses Iteratively Reweighted Least Squares (IRLS).

    **Implements Eq. (3.4)** from Chapter 3.

    Args:
        A: Design matrix (m × n).
        b: Observation vector (m × 1).
        method: Robust loss function - "huber", "cauchy", or "tukey".
        threshold: Threshold parameter for robust function (c in the book).
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance.
        return_covariance: If True, compute covariance.

    Returns:
        Tuple of:
            - x_hat: Estimated state vector (n × 1).
            - P: Covariance matrix (n × n), or None.
            - weights: Final weights for each measurement (m × 1).

    Example:
        >>> import numpy as np
        >>> # Data with outliers
        >>> A = np.vstack([np.eye(3), np.eye(3)])
        >>> b = np.array([1.0, 2.0, 3.0, 1.1, 2.2, 10.0])  # Last is outlier
        >>> x_hat, P, weights = robust_least_squares(A, b, method="huber")
        >>> print(f"Outlier weight: {weights[-1]:.3f}")  # Should be < 1.0
    """
    # Validate inputs
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(f"A must be 2D and b must be 1D. Got A: {A.shape}, b: {b.shape}")

    m, n = A.shape
    if len(b) != m:
        raise ValueError(f"Dimension mismatch: A has {m} rows, b has {len(b)} elements")

    if method not in ["huber", "cauchy", "tukey"]:
        raise ValueError(f"Unknown method '{method}'. Use 'huber', 'cauchy', or 'tukey'")

    # Initial estimate with standard LS
    x, _ = linear_least_squares(A, b, return_covariance=False)

    # IRLS iterations
    weights = np.ones(m)
    for iteration in range(max_iter):
        # Compute residuals
        residuals = b - A @ x

        # Robust standard deviation (MAD-based)
        sigma = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        if sigma < 1e-10:
            sigma = 1.0  # Avoid division by zero

        # Normalized residuals
        normalized_res = residuals / sigma

        # Compute weights based on robust function
        if method == "huber":
            # Huber: w = 1 if |r| ≤ c, else w = c/|r|
            weights = np.where(
                np.abs(normalized_res) <= threshold,
                1.0,
                threshold / np.abs(normalized_res),
            )
        elif method == "cauchy":
            # Cauchy: w = 1 / (1 + (r/c)²)
            weights = 1.0 / (1.0 + (normalized_res / threshold) ** 2)
        elif method == "tukey":
            # Tukey biweight: w = (1-(r/c)²)² if |r| ≤ c, else 0
            weights = np.where(
                np.abs(normalized_res) <= threshold,
                (1.0 - (normalized_res / threshold) ** 2) ** 2,
                0.0,
            )
            # Add small epsilon to avoid singular weight matrix
            weights = np.maximum(weights, 1e-10)

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

