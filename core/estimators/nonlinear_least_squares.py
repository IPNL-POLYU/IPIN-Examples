"""
Nonlinear Least Squares solver using Gauss-Newton and Levenberg-Marquardt.

This module implements iterative optimization methods for nonlinear least squares
problems as described in Chapter 3, Section 3.4.1 of the IPIN book.

Book Reference:
    - Section 3.1: Nonlinear LS problem formulation (Eq. 3.4)
    - Section 3.4.1.2: Gauss-Newton method (Eq. 3.47-3.52)
    - Section 3.4.1.3: Levenberg-Marquardt method (Eq. 3.53-3.56, Algorithm 3.2)
    - Table 3.1: Robust estimator loss functions

Mathematical Formulation:
    Given observations y and measurement model h(x), we seek:
        x̂ = argmin ½‖r(x)‖²_W
    where r(x) = y - h(x) is the residual vector.

    Gauss-Newton update (Eq. 3.52):
        (J'WJ) Δx = J'W r  →  x ← x + Δx

    Levenberg-Marquardt update (Eq. 3.53):
        (J'WJ + μI) Δx = J'W r
    where μ is an adaptive damping parameter.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class NonlinearLSResult:
    """Result container for nonlinear least squares optimization.

    Attributes:
        x: Estimated state vector.
        covariance: Covariance matrix (n × n), or None.
        iterations: Number of iterations performed.
        residuals: Final residuals r = y - h(x̂).
        cost: Final cost value ½‖r‖²_W.
        converged: Whether the solver converged within tolerance.
        weights: Final measurement weights (for robust estimation).
    """

    x: np.ndarray
    covariance: Optional[np.ndarray]
    iterations: int
    residuals: np.ndarray
    cost: float
    converged: bool
    weights: Optional[np.ndarray] = None


def gauss_newton(
    h: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 20,
    tol: float = 1e-8,
    return_covariance: bool = True,
) -> NonlinearLSResult:
    """
    Gauss-Newton solver for nonlinear least squares.

    Solves: x̂ = argmin ½‖y - h(x)‖²_W

    **Implements Eq. (3.52)** from Chapter 3, Section 3.4.1.2:
        (J'WJ) Δx = J'W r  →  x ← x + Δx
    where J = ∂h/∂x is the Jacobian and r = y - h(x) is the residual.

    Args:
        h: Measurement model function h: R^n → R^m.
            Returns predicted measurements given state x.
        jacobian: Function returning Jacobian matrix J = ∂h/∂x (m × n).
        y: Observation vector (m,).
        x0: Initial state estimate (n,).
        weights: Optional measurement weights (m,) for weighted LS.
            If None, uses uniform weights (standard LS).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on ‖Δx‖.
        return_covariance: If True, compute covariance at final estimate.

    Returns:
        NonlinearLSResult containing estimate, covariance, and diagnostics.

    Example:
        >>> import numpy as np
        >>> # 2D range positioning with 4 anchors
        >>> anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        >>> true_pos = np.array([3.0, 4.0])
        >>> def h(x):
        ...     return np.linalg.norm(anchors - x, axis=1)
        >>> def jac(x):
        ...     diff = x - anchors
        ...     ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        ...     return diff / np.maximum(ranges, 1e-10)
        >>> y = h(true_pos) + 0.1 * np.random.randn(4)  # Noisy ranges
        >>> result = gauss_newton(h, jac, y, x0=np.array([5.0, 5.0]))
        >>> print(f"Estimate: {result.x}, Iterations: {result.iterations}")
    """
    return _solve_nonlinear_ls(
        h=h,
        jacobian=jacobian,
        y=y,
        x0=x0,
        weights=weights,
        method="gn",
        max_iter=max_iter,
        tol=tol,
        return_covariance=return_covariance,
    )


def levenberg_marquardt(
    h: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    mu0: float = 1e-3,
    return_covariance: bool = True,
) -> NonlinearLSResult:
    """
    Levenberg-Marquardt solver for nonlinear least squares.

    Solves: x̂ = argmin ½‖y - h(x)‖²_W

    **Implements Eq. (3.53)** from Chapter 3, Section 3.4.1.3:
        (J'WJ + μI) Δx = J'W r
    where μ is an adaptive damping parameter.

    LM combines Gauss-Newton (fast near solution) with gradient descent
    (robust far from solution) by adaptively adjusting μ:
        - Small μ: Gauss-Newton behavior (quadratic convergence)
        - Large μ: Gradient descent behavior (global convergence)

    **Algorithm 3.2** adaptive μ update using gain ratio.

    Args:
        h: Measurement model function h: R^n → R^m.
        jacobian: Function returning Jacobian matrix J = ∂h/∂x (m × n).
        y: Observation vector (m,).
        x0: Initial state estimate (n,).
        weights: Optional measurement weights (m,) for weighted LS.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on ‖Δx‖.
        mu0: Initial damping parameter (default 1e-3).
        return_covariance: If True, compute covariance at final estimate.

    Returns:
        NonlinearLSResult containing estimate, covariance, and diagnostics.

    Example:
        >>> import numpy as np
        >>> # Same 2D positioning problem, but with poor initial guess
        >>> anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        >>> def h(x):
        ...     return np.linalg.norm(anchors - x, axis=1)
        >>> def jac(x):
        ...     diff = x - anchors
        ...     ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        ...     return diff / np.maximum(ranges, 1e-10)
        >>> y = np.array([5.0, 7.07, 7.07, 5.0])  # True position (5, 5)
        >>> # Poor initial guess far from solution
        >>> result = levenberg_marquardt(h, jac, y, x0=np.array([0.0, 0.0]))
        >>> print(f"Estimate: {result.x}, Converged: {result.converged}")
    """
    return _solve_nonlinear_ls(
        h=h,
        jacobian=jacobian,
        y=y,
        x0=x0,
        weights=weights,
        method="lm",
        max_iter=max_iter,
        tol=tol,
        mu0=mu0,
        return_covariance=return_covariance,
    )


def robust_gauss_newton(
    h: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray,
    loss: Literal["l2", "huber", "cauchy", "gm", "tukey"] = "huber",
    loss_param: float = 1.5,
    max_iter: int = 30,
    max_irls_iter: int = 5,
    tol: float = 1e-8,
    return_covariance: bool = True,
) -> NonlinearLSResult:
    """
    Robust Gauss-Newton solver with outlier rejection.

    Combines nonlinear LS with robust M-estimation using IRLS
    (Iteratively Reweighted Least Squares).

    **Uses robust loss functions from Table 3.1** in Chapter 3:
        - L2: Standard LS (no outlier rejection)
        - Huber: Quadratic near zero, linear for large residuals
        - Cauchy: Heavy-tailed, gradual outlier downweighting
        - G-M (Geman-McClure): Strong outlier rejection
        - Tukey: Hard rejection beyond threshold

    Args:
        h: Measurement model function h: R^n → R^m.
        jacobian: Function returning Jacobian matrix J = ∂h/∂x (m × n).
        y: Observation vector (m,).
        x0: Initial state estimate (n,).
        loss: Robust loss function from Table 3.1.
        loss_param: Scale parameter for robust loss (threshold/scale).
        max_iter: Maximum Gauss-Newton iterations per IRLS step.
        max_irls_iter: Maximum IRLS outer iterations.
        tol: Convergence tolerance.
        return_covariance: If True, compute covariance at final estimate.

    Returns:
        NonlinearLSResult with final weights indicating outlier rejection.

    Example:
        >>> import numpy as np
        >>> anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        >>> def h(x):
        ...     return np.linalg.norm(anchors - x, axis=1)
        >>> def jac(x):
        ...     diff = x - anchors
        ...     ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        ...     return diff / np.maximum(ranges, 1e-10)
        >>> # True position (3, 4), but one measurement has NLOS error
        >>> y = np.array([5.0, 7.07, 7.21, 9.0])  # Last is outlier (+2m)
        >>> result = robust_gauss_newton(h, jac, y, x0=np.array([5.0, 5.0]),
        ...                              loss="huber")
        >>> print(f"Outlier weight: {result.weights[-1]:.3f}")
    """
    # Input validation
    y = np.asarray(y)
    x0 = np.asarray(x0)

    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")
    if x0.ndim != 1:
        raise ValueError(f"x0 must be 1D array, got shape {x0.shape}")

    m = len(y)
    x = x0.copy()
    weights = np.ones(m)

    # L2 is just standard Gauss-Newton
    if loss == "l2":
        result = gauss_newton(h, jacobian, y, x0, weights=None,
                              max_iter=max_iter, tol=tol,
                              return_covariance=return_covariance)
        result.weights = np.ones(m)
        return result

    # IRLS outer loop
    prev_weights = np.zeros(m)
    for irls_iter in range(max_irls_iter):
        # Run weighted Gauss-Newton with current weights
        result = gauss_newton(
            h, jacobian, y, x,
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            return_covariance=False,
        )
        x = result.x

        # Compute residuals and update weights
        residuals = y - h(x)

        # Robust scale estimate (MAD)
        # Use absolute residuals for scale estimation
        abs_residuals = np.abs(residuals)
        sigma = 1.4826 * np.median(abs_residuals)
        if sigma < 1e-10:
            # If residuals are tiny, use a fixed scale based on measurement precision
            sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1.0

        # Normalized residuals: u = r / (σ × c)
        u = residuals / (sigma * loss_param)

        # Compute weights based on robust loss
        weights = _compute_robust_weights(u, loss)

        # Check for IRLS convergence (weights stable)
        weight_change = np.max(np.abs(weights - prev_weights))
        if weight_change < 0.001:
            break
        prev_weights = weights.copy()

    # Final solve with converged weights
    result = gauss_newton(
        h, jacobian, y, x,
        weights=weights,
        max_iter=max_iter,
        tol=tol,
        return_covariance=return_covariance,
    )
    result.weights = weights

    return result


def _solve_nonlinear_ls(
    h: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray,
    weights: Optional[np.ndarray],
    method: str,
    max_iter: int,
    tol: float,
    mu0: float = 1e-3,
    return_covariance: bool = True,
) -> NonlinearLSResult:
    """Internal solver implementing both Gauss-Newton and Levenberg-Marquardt."""
    # Input validation
    y = np.asarray(y)
    x0 = np.asarray(x0)

    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")
    if x0.ndim != 1:
        raise ValueError(f"x0 must be 1D array, got shape {x0.shape}")

    m = len(y)
    n = len(x0)
    x = x0.copy()

    # Setup weight matrix
    if weights is None:
        W = np.eye(m)
    else:
        weights = np.asarray(weights)
        if weights.ndim != 1 or len(weights) != m:
            raise ValueError(f"weights must be 1D array of length {m}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        W = np.diag(weights)

    # LM-specific initialization
    mu = mu0
    nu = 2.0

    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # Evaluate model and Jacobian
        hx = h(x)
        if len(hx) != m:
            raise ValueError(f"h(x) returned {len(hx)} elements, expected {m}")

        J = jacobian(x)
        if J.shape != (m, n):
            raise ValueError(f"Jacobian shape {J.shape}, expected ({m}, {n})")

        # Residual: r = y - h(x)
        r = y - hx

        # Weighted normal equations: (J'WJ) Δx = J'Wr
        JtW = J.T @ W
        JtWJ = JtW @ J
        JtWr = JtW @ r

        # Cost function: f = ½ r'Wr
        cost = 0.5 * r @ W @ r

        if method == "gn":
            # Gauss-Newton: solve (J'WJ) Δx = J'Wr
            try:
                delta_x = np.linalg.solve(JtWJ, JtWr)
            except np.linalg.LinAlgError:
                # Singular - use pseudo-inverse
                delta_x = np.linalg.lstsq(JtWJ, JtWr, rcond=None)[0]

            x = x + delta_x

        elif method == "lm":
            # Levenberg-Marquardt: solve (J'WJ + μI) Δx = J'Wr
            while True:
                # Add damping
                JtWJ_damped = JtWJ + mu * np.eye(n)

                try:
                    delta_x = np.linalg.solve(JtWJ_damped, JtWr)
                except np.linalg.LinAlgError:
                    delta_x = np.linalg.lstsq(JtWJ_damped, JtWr, rcond=None)[0]

                # Evaluate new cost
                x_new = x + delta_x
                r_new = y - h(x_new)
                cost_new = 0.5 * r_new @ W @ r_new

                # Gain ratio (Eq. 3.56 simplified)
                # Predicted decrease: ½ Δx'(μΔx + J'Wr)
                predicted_decrease = 0.5 * delta_x @ (mu * delta_x + JtWr)
                actual_decrease = cost - cost_new

                if predicted_decrease > 1e-15:
                    gain_ratio = actual_decrease / predicted_decrease
                else:
                    gain_ratio = 0.0

                if gain_ratio > 0:
                    # Accept step
                    x = x_new
                    # Decrease damping (more GN-like)
                    mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3)
                    nu = 2.0
                    break
                else:
                    # Reject step, increase damping (more GD-like)
                    mu = mu * nu
                    nu = 2.0 * nu

                    # Prevent infinite loop with very large damping
                    if mu > 1e10:
                        break

        # Check convergence
        step_norm = np.linalg.norm(delta_x)
        if step_norm < tol:
            converged = True
            break

    # Final evaluation
    hx = h(x)
    r = y - hx
    cost = 0.5 * r @ W @ r

    # Covariance estimation
    P = None
    if return_covariance:
        J = jacobian(x)
        JtWJ = J.T @ W @ J

        # Estimate residual variance
        if m > n:
            sigma2 = (r @ W @ r) / (m - n)
        else:
            sigma2 = 1.0

        try:
            P = sigma2 * np.linalg.inv(JtWJ)
        except np.linalg.LinAlgError:
            P = sigma2 * np.linalg.pinv(JtWJ)

    return NonlinearLSResult(
        x=x,
        covariance=P,
        iterations=iteration + 1,
        residuals=r,
        cost=cost,
        converged=converged,
    )


def _compute_robust_weights(u: np.ndarray, loss: str) -> np.ndarray:
    """
    Compute IRLS weights for robust loss functions (Table 3.1).

    Args:
        u: Normalized residuals (r / (σ × c)).
        loss: Loss function name.

    Returns:
        weights: Weight for each measurement.
    """
    abs_u = np.abs(u)

    if loss == "huber":
        # Huber: w = min(1, 1/|u|)
        weights = np.where(abs_u <= 1.0, 1.0, 1.0 / abs_u)

    elif loss == "cauchy":
        # Cauchy: w = 1 / (1 + u²)
        weights = 1.0 / (1.0 + u ** 2)

    elif loss == "gm" or loss == "geman_mcclure":
        # Geman-McClure: w = 1 / (1 + u²)²
        weights = 1.0 / (1.0 + u ** 2) ** 2

    elif loss == "tukey":
        # Tukey biweight: w = (1-u²)² if |u| ≤ 1, else 0
        weights = np.where(abs_u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
        weights = np.maximum(weights, 1e-10)  # Avoid singularity

    else:
        raise ValueError(f"Unknown loss function: {loss}")

    return weights


# Convenience function matching book terminology
def solve_nonlinear_ls(
    h: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: Literal["gn", "lm"] = "gn",
    robust_loss: Optional[Literal["l2", "huber", "cauchy", "gm", "tukey"]] = None,
    loss_param: float = 1.5,
    max_iter: int = 30,
    tol: float = 1e-8,
    return_covariance: bool = True,
    **kwargs,
) -> NonlinearLSResult:
    """
    General nonlinear least squares solver.

    This is a convenience function that dispatches to the appropriate solver
    based on the specified method and robust loss.

    **Book Reference:**
        - Gauss-Newton: Eq. (3.52), Section 3.4.1.2
        - Levenberg-Marquardt: Eq. (3.53), Algorithm 3.2, Section 3.4.1.3
        - Robust losses: Table 3.1, Section 3.1.1

    Args:
        h: Measurement model h(x) returning predicted observations.
        jacobian: Jacobian function J = ∂h/∂x.
        y: Observations (m,).
        x0: Initial state estimate (n,).
        weights: Optional measurement weights for WLS (m,).
        method: Optimization method - "gn" (Gauss-Newton) or "lm" (Levenberg-Marquardt).
        robust_loss: Optional robust loss function from Table 3.1.
            If None, standard LS is used. Options: "l2", "huber", "cauchy", "gm", "tukey".
        loss_param: Scale parameter for robust loss (default 1.5).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        return_covariance: If True, compute covariance at solution.
        **kwargs: Additional arguments passed to the solver (e.g., mu0 for LM).

    Returns:
        NonlinearLSResult with solution, covariance, and diagnostics.

    Example:
        >>> import numpy as np
        >>> # 2D positioning from ranges to 4 anchors
        >>> anchors = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        >>> true_pos = np.array([3.0, 4.0])
        >>>
        >>> def h(x):
        ...     return np.linalg.norm(anchors - x, axis=1)
        >>>
        >>> def jacobian(x):
        ...     diff = x - anchors
        ...     ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        ...     return diff / np.maximum(ranges, 1e-10)
        >>>
        >>> # Clean measurements
        >>> y = h(true_pos)
        >>> result = solve_nonlinear_ls(h, jacobian, y, x0=np.array([5, 5]))
        >>> print(f"Position: {result.x}")
        >>>
        >>> # With outlier and robust estimation
        >>> y_noisy = h(true_pos) + np.array([0.1, 0.1, 0.1, 3.0])  # Outlier
        >>> result = solve_nonlinear_ls(h, jacobian, y_noisy, x0=np.array([5, 5]),
        ...                             robust_loss="huber")
        >>> print(f"Robust estimate: {result.x}")
    """
    if robust_loss is not None:
        # Use robust solver with IRLS
        return robust_gauss_newton(
            h=h,
            jacobian=jacobian,
            y=y,
            x0=x0,
            loss=robust_loss,
            loss_param=loss_param,
            max_iter=max_iter,
            tol=tol,
            return_covariance=return_covariance,
        )
    elif method == "gn":
        return gauss_newton(
            h=h,
            jacobian=jacobian,
            y=y,
            x0=x0,
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            return_covariance=return_covariance,
        )
    elif method == "lm":
        mu0 = kwargs.get("mu0", 1e-3)
        return levenberg_marquardt(
            h=h,
            jacobian=jacobian,
            y=y,
            x0=x0,
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            mu0=mu0,
            return_covariance=return_covariance,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gn' or 'lm'.")

