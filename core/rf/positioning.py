"""
RF positioning algorithms.

This module implements positioning algorithms for TOA, TDOA, and AOA measurements:
- Iterative Weighted Least Squares (I-WLS)
- Closed-form solutions (Fang, Chan)
- Linear Least Squares

All algorithms implement equations from Chapter 4 of the IPIN book.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class TOAPositioner:
    """
    Time of Arrival (TOA) positioning using Iterative Weighted Least Squares.

    Implements Eqs. (4.14)-(4.23) from Chapter 4:
        Nonlinear TOA positioning via linearization and iterative refinement.

    Attributes:
        anchors: Array of anchor positions, shape (N, d) where d=2 or 3.
        method: Positioning method ('ls' or 'iwls').
    """

    def __init__(self, anchors: np.ndarray, method: str = "iwls"):
        """
        Initialize TOA positioner.

        Args:
            anchors: Array of anchor positions, shape (N, 2) or (N, 3).
            method: 'ls' for Least Squares, 'iwls' for Iterative WLS.
                   Defaults to 'iwls'.
        """
        self.anchors = np.asarray(anchors, dtype=float)
        self.n_anchors = self.anchors.shape[0]
        self.dim = self.anchors.shape[1]
        self.method = method.lower()

        if self.method not in ["ls", "iwls"]:
            raise ValueError(f"method must be 'ls' or 'iwls', got {method}")

    def solve(
        self,
        ranges: np.ndarray,
        initial_guess: np.ndarray,
        weights: Optional[np.ndarray] = None,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve TOA positioning problem.

        Implements Eqs. (4.14)-(4.23): Linearize the nonlinear range equations
        and iteratively refine the solution.

        Args:
            ranges: Measured ranges from anchors, shape (N,).
            initial_guess: Initial position estimate, shape (d,).
            weights: Measurement weights (optional), shape (N,) or (N, N).
                    If None, uniform weights are used.
            max_iters: Maximum number of iterations. Defaults to 10.
            tol: Convergence tolerance in meters. Defaults to 1e-6.

        Returns:
            position: Estimated position, shape (d,).
            info: Dictionary with convergence information:
                - 'iterations': number of iterations
                - 'converged': True if converged
                - 'residual': final residual norm
                - 'history': position history

        Example:
            >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
            >>> ranges = np.array([5.0, 7.07, 7.07, 5.0])
            >>> positioner = TOAPositioner(anchors)
            >>> pos, info = positioner.solve(ranges, initial_guess=np.array([5, 5]))
            >>> print(f"Position: {pos}")
        """
        ranges = np.asarray(ranges, dtype=float)
        position = np.asarray(initial_guess, dtype=float).copy()

        if len(ranges) != self.n_anchors:
            raise ValueError(
                f"Expected {self.n_anchors} ranges, got {len(ranges)}"
            )

        # Initialize weights
        if weights is None:
            W = np.eye(self.n_anchors)
        elif weights.ndim == 1:
            W = np.diag(weights)
        else:
            W = weights

        # Iteration history
        history = [position.copy()]
        converged = False

        for iteration in range(max_iters):
            # Compute predicted ranges and residuals (Eq. 4.16)
            predicted_ranges = np.linalg.norm(
                self.anchors - position, axis=1
            )
            residuals = ranges - predicted_ranges

            # Check convergence
            residual_norm = np.linalg.norm(residuals)
            if residual_norm < tol:
                converged = True
                break

            # Compute Jacobian (linearization, Eq. 4.17)
            # H[i, j] = -(p_i[j] - p[j]) / ||p_i - p||
            H = np.zeros((self.n_anchors, self.dim))
            for i in range(self.n_anchors):
                diff = self.anchors[i] - position
                dist = predicted_ranges[i]
                if dist > 1e-10:  # Avoid division by zero
                    H[i] = -diff / dist

            # Solve weighted least squares (Eq. 4.18-4.20)
            # Δp = (H^T W H)^{-1} H^T W r
            try:
                if self.method == "iwls":
                    # Update weights based on current estimate
                    # W_ii = 1 / σ_i^2, assuming σ_i ∝ d_i
                    diag_weights = 1.0 / np.maximum(predicted_ranges**2, 1e-6)
                    W = np.diag(diag_weights)

                # Weighted LS solution
                delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residuals)
                position = position + delta
                history.append(position.copy())

            except np.linalg.LinAlgError:
                # Singular matrix, stop iteration
                break

            # Check step size convergence
            if np.linalg.norm(delta) < tol:
                converged = True
                break

        info = {
            "iterations": iteration + 1,
            "converged": converged,
            "residual": residual_norm if converged else np.linalg.norm(residuals),
            "history": np.array(history),
        }

        return position, info


def toa_solve_with_clock_bias(
    anchors: np.ndarray,
    ranges: np.ndarray,
    initial_guess: np.ndarray,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Solve TOA positioning with unknown clock bias.

    Implements Eqs. (4.24)-(4.26) from Chapter 4:
        Extended state vector: x_aug = [x, y, z, c*b]
        where c*b is the clock bias in meters.

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        ranges: Measured ranges from anchors, shape (N,).
        initial_guess: Initial position and clock bias estimate, shape (d+1,).
                      Format: [x, y, (z), clock_bias_m]
        max_iters: Maximum number of iterations. Defaults to 10.
        tol: Convergence tolerance. Defaults to 1e-6.

    Returns:
        position: Estimated position, shape (d,).
        clock_bias_m: Estimated clock bias in meters.
        info: Dictionary with convergence information.

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> # Ranges with 2m clock bias
        >>> true_ranges = np.array([5.0, 7.07, 7.07, 5.0])
        >>> ranges = true_ranges + 2.0
        >>> initial = np.array([5, 5, 0])  # [x, y, clock_bias]
        >>> pos, bias, info = toa_solve_with_clock_bias(anchors, ranges, initial)
        >>> print(f"Clock bias: {bias:.2f} m")
    """
    anchors = np.asarray(anchors, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    state = np.asarray(initial_guess, dtype=float).copy()

    n_anchors = anchors.shape[0]
    dim = anchors.shape[1]

    if len(state) != dim + 1:
        raise ValueError(
            f"initial_guess must have length {dim+1} (position + clock bias)"
        )

    history = [state.copy()]
    converged = False

    for iteration in range(max_iters):
        position = state[:dim]
        clock_bias_m = state[dim]

        # Predicted ranges including clock bias (Eq. 4.24)
        geometric_ranges = np.linalg.norm(anchors - position, axis=1)
        predicted_ranges = geometric_ranges + clock_bias_m
        residuals = ranges - predicted_ranges

        # Check convergence
        residual_norm = np.linalg.norm(residuals)
        if residual_norm < tol:
            converged = True
            break

        # Compute Jacobian (Eq. 4.25)
        # H = [∂h/∂x, ∂h/∂y, (∂h/∂z), ∂h/∂b]
        H = np.zeros((n_anchors, dim + 1))

        for i in range(n_anchors):
            diff = anchors[i] - position
            dist = geometric_ranges[i]
            if dist > 1e-10:
                # Partial derivatives w.r.t. position
                H[i, :dim] = -diff / dist
            # Partial derivative w.r.t. clock bias is 1
            H[i, dim] = 1.0

        # Solve LS (Eq. 4.26)
        try:
            delta = np.linalg.solve(H.T @ H, H.T @ residuals)
            state = state + delta
            history.append(state.copy())
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(delta) < tol:
            converged = True
            break

    position_final = state[:dim]
    clock_bias_final = state[dim]

    info = {
        "iterations": iteration + 1,
        "converged": converged,
        "residual": residual_norm if converged else np.linalg.norm(residuals),
        "history": np.array(history),
    }

    return position_final, clock_bias_final, info


class TDOAPositioner:
    """
    TDOA positioning using Least Squares or Weighted Least Squares.

    Implements Eqs. (4.34)-(4.42) from Chapter 4:
        Linearized TDOA positioning.

    Attributes:
        anchors: Array of anchor positions.
        reference_idx: Index of reference anchor.
    """

    def __init__(self, anchors: np.ndarray, reference_idx: int = 0):
        """
        Initialize TDOA positioner.

        Args:
            anchors: Array of anchor positions, shape (N, 2) or (N, 3).
            reference_idx: Index of reference anchor. Defaults to 0.
        """
        self.anchors = np.asarray(anchors, dtype=float)
        self.n_anchors = self.anchors.shape[0]
        self.dim = self.anchors.shape[1]
        self.reference_idx = reference_idx
        self.reference_anchor = self.anchors[reference_idx]

    def solve(
        self,
        tdoa_measurements: np.ndarray,
        initial_guess: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve TDOA positioning problem.

        Implements Eqs. (4.34)-(4.42): Linearized TDOA LS/WLS.

        Args:
            tdoa_measurements: TDOA measurements (range differences), shape (N-1,).
            initial_guess: Initial position estimate, shape (d,).
            covariance: Measurement covariance matrix (optional), shape (N-1, N-1).
            max_iters: Maximum iterations. Defaults to 10.
            tol: Convergence tolerance. Defaults to 1e-6.

        Returns:
            position: Estimated position.
            info: Convergence information dictionary.
        """
        tdoa_measurements = np.asarray(tdoa_measurements, dtype=float)
        position = np.asarray(initial_guess, dtype=float).copy()

        if len(tdoa_measurements) != self.n_anchors - 1:
            raise ValueError(
                f"Expected {self.n_anchors-1} TDOA measurements, "
                f"got {len(tdoa_measurements)}"
            )

        # Weight matrix
        if covariance is not None:
            W = np.linalg.inv(covariance)
        else:
            W = np.eye(self.n_anchors - 1)

        history = [position.copy()]
        converged = False

        for iteration in range(max_iters):
            # Compute predicted TDOA (Eq. 4.34)
            dist_ref = np.linalg.norm(position - self.reference_anchor)
            predicted_tdoa = []
            H = []

            idx = 0
            for i in range(self.n_anchors):
                if i == self.reference_idx:
                    continue

                dist_i = np.linalg.norm(position - self.anchors[i])
                predicted_tdoa.append(dist_i - dist_ref)

                # Jacobian row (Eq. 4.37-4.38)
                if dist_i > 1e-10 and dist_ref > 1e-10:
                    h_i = (position - self.anchors[i]) / dist_i - (
                        position - self.reference_anchor
                    ) / dist_ref
                else:
                    h_i = np.zeros(self.dim)

                H.append(h_i)
                idx += 1

            predicted_tdoa = np.array(predicted_tdoa)
            H = np.array(H)

            # Residuals
            residuals = tdoa_measurements - predicted_tdoa
            residual_norm = np.linalg.norm(residuals)

            if residual_norm < tol:
                converged = True
                break

            # Weighted LS (Eq. 4.39-4.41)
            try:
                delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residuals)
                position = position + delta
                history.append(position.copy())
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(delta) < tol:
                converged = True
                break

        info = {
            "iterations": iteration + 1,
            "converged": converged,
            "residual": residual_norm if converged else np.linalg.norm(residuals),
            "history": np.array(history),
        }

        return position, info


class AOAPositioner:
    """
    AOA positioning using Least Squares.

    Implements Eqs. (4.63)-(4.67) from Chapter 4:
        Linearized AOA positioning.

    Attributes:
        anchors: Array of anchor positions.
    """

    def __init__(self, anchors: np.ndarray):
        """
        Initialize AOA positioner.

        Args:
            anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        """
        self.anchors = np.asarray(anchors, dtype=float)
        self.n_anchors = self.anchors.shape[0]
        self.dim = self.anchors.shape[1]

    def solve(
        self,
        aoa_measurements: np.ndarray,
        initial_guess: np.ndarray,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve AOA positioning problem.

        Implements Eqs. (4.63)-(4.67): Iterative LS for AOA.

        Args:
            aoa_measurements: Azimuth angles in radians, shape (N,).
            initial_guess: Initial position estimate, shape (2,) for 2D.
            max_iters: Maximum iterations. Defaults to 10.
            tol: Convergence tolerance. Defaults to 1e-6.

        Returns:
            position: Estimated position.
            info: Convergence information dictionary.
        """
        aoa_measurements = np.asarray(aoa_measurements, dtype=float)
        position = np.asarray(initial_guess, dtype=float).copy()

        if len(aoa_measurements) != self.n_anchors:
            raise ValueError(
                f"Expected {self.n_anchors} AOA measurements, "
                f"got {len(aoa_measurements)}"
            )

        history = [position.copy()]
        converged = False

        for iteration in range(max_iters):
            # Compute predicted azimuth angles
            predicted_aoa = []
            H = []

            for i in range(self.n_anchors):
                dx = position[0] - self.anchors[i, 0]
                dy = position[1] - self.anchors[i, 1]
                dist_sq = dx**2 + dy**2

                # Predicted angle (Eq. 4.63)
                predicted_angle = np.arctan2(dy, dx)
                predicted_aoa.append(predicted_angle)

                # Jacobian (Eq. 4.67)
                if dist_sq > 1e-10:
                    h_i = np.array([-dy / dist_sq, dx / dist_sq])
                else:
                    h_i = np.zeros(2)

                H.append(h_i)

            predicted_aoa = np.array(predicted_aoa)
            H = np.array(H)

            # Handle angle wrapping
            residuals = aoa_measurements - predicted_aoa
            residuals = np.arctan2(np.sin(residuals), np.cos(residuals))

            residual_norm = np.linalg.norm(residuals)

            if residual_norm < tol:
                converged = True
                break

            # Least squares
            try:
                delta = np.linalg.solve(H.T @ H, H.T @ residuals)
                position = position + delta
                history.append(position.copy())
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(delta) < tol:
                converged = True
                break

        info = {
            "iterations": iteration + 1,
            "converged": converged,
            "residual": residual_norm if converged else np.linalg.norm(residuals),
            "history": np.array(history),
        }

        return position, info

