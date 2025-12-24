"""
RF positioning algorithms.

This module implements positioning algorithms for TOA, TDOA, and AOA measurements:
- Iterative Weighted Least Squares (I-WLS)
- Closed-form solutions (Fang, Chan)
- Linear Least Squares

All algorithms implement equations from Chapter 4 of the IPIN book.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np


def build_tdoa_covariance(
    sigmas: np.ndarray,
    ref_idx: int = 0,
) -> np.ndarray:
    """
    Build correlated covariance matrix for TDOA measurements.

    Implements the covariance structure from Eq. (4.42) in Chapter 4.

    For TDOA measurements z = [d^{1,ref}, d^{2,ref}, ..., d^{I-1,ref}]^T
    where d^{k,ref} = d^k - d^{ref} (range difference from anchor k to reference):

        var(d^{k,ref}) = sigma_k^2 + sigma_ref^2  (diagonal terms)
        cov(d^{k,ref}, d^{m,ref}) = sigma_ref^2   (off-diagonal terms, k != m)

    The off-diagonal correlation arises because all TDOA measurements share
    the same reference anchor, whose measurement noise contributes to all
    range differences.

    Args:
        sigmas: Per-anchor range measurement standard deviations, shape (N,).
                sigmas[i] is the std dev of range measurement to anchor i.
        ref_idx: Index of reference anchor (default 0).
                This anchor is used as the base for all TDOA differences.

    Returns:
        Covariance matrix Sigma of shape (N-1, N-1) for TDOA measurements.
        The matrix structure matches Eq. (4.42):
            - Diagonal: sigma_k^2 + sigma_ref^2
            - Off-diagonal: sigma_ref^2

    Example:
        >>> # 4 anchors with different noise levels
        >>> sigmas = np.array([0.1, 0.2, 0.15, 0.25])  # [ref, anc1, anc2, anc3]
        >>> cov = build_tdoa_covariance(sigmas, ref_idx=0)
        >>> print(cov.shape)  # (3, 3)
        >>> # Diagonal: [0.2^2 + 0.1^2, 0.15^2 + 0.1^2, 0.25^2 + 0.1^2]
        >>> # Off-diagonal: 0.1^2 = 0.01

    Notes:
        - The correlation between TDOA measurements comes from sharing the
          reference anchor. This must be accounted for in WLS estimation.
        - Using identity weighting (ignoring correlation) leads to suboptimal
          estimates, especially when sigma_ref is large relative to other sigmas.
        - For heterogeneous anchor noise, proper covariance modeling can
          significantly improve positioning accuracy.

    References:
        Chapter 4, Eq. (4.42): TDOA covariance matrix structure
    """
    sigmas = np.asarray(sigmas, dtype=float)
    n_anchors = len(sigmas)

    if ref_idx < 0 or ref_idx >= n_anchors:
        raise ValueError(
            f"ref_idx must be in [0, {n_anchors-1}], got {ref_idx}"
        )

    # Reference anchor variance
    sigma_ref_sq = sigmas[ref_idx] ** 2

    # Number of TDOA measurements (all anchors except reference)
    n_tdoa = n_anchors - 1

    # Build covariance matrix
    # Indices of non-reference anchors
    non_ref_indices = [i for i in range(n_anchors) if i != ref_idx]

    # Initialize covariance matrix
    cov = np.zeros((n_tdoa, n_tdoa))

    for i, anchor_i in enumerate(non_ref_indices):
        for j, anchor_j in enumerate(non_ref_indices):
            if i == j:
                # Diagonal: var(d^{k,ref}) = sigma_k^2 + sigma_ref^2
                cov[i, j] = sigmas[anchor_i] ** 2 + sigma_ref_sq
            else:
                # Off-diagonal: cov(d^{k,ref}, d^{m,ref}) = sigma_ref^2
                cov[i, j] = sigma_ref_sq

    return cov


class TOAPositioner:
    """
    Time of Arrival (TOA) positioning using Iterative Least Squares.

    Implements Eqs. (4.14)-(4.23) from Chapter 4:
        Nonlinear TOA positioning via linearization and iterative refinement.

    The TOA range equations are nonlinear, so we linearize via Taylor series
    expansion (Eq. 4.17) and iteratively update the position estimate until
    convergence. This is the "Iterative" part.

    **Methods:**

    - `"iterative_ls"` (default): Iterative Least Squares with uniform weights
      (W = I). Matches Eq. (4.20): δx = (H^T H)^{-1} H^T δz

    - `"iterative_wls"`: Iterative Weighted LS using user-provided covariance.
      Requires passing `covariance` to solve(). Matches Eq. (4.23):
      δx = (H^T W H)^{-1} H^T W δz, where W = Σ^{-1}.

    - `"range_weighted"`: Heuristic weighting where W_ii = 1/d_i^2.
      This is NOT from the book but is sometimes used in practice to
      down-weight far anchors. Use with caution.

    - `"ls"`, `"iwls"`: Deprecated aliases for `"iterative_ls"` and
      `"range_weighted"`, respectively. Will be removed in future versions.

    Attributes:
        anchors: Array of anchor positions, shape (N, d) where d=2 or 3.
        method: Positioning method.

    References:
        Chapter 4, Section 4.2.2, Eqs. (4.14)-(4.23): TOA LS/WLS estimation
    """

    # Method name aliases for backward compatibility
    _METHOD_ALIASES = {
        "ls": "iterative_ls",
        "iwls": "range_weighted",  # Legacy behavior used 1/d^2 weights
    }

    _VALID_METHODS = {"iterative_ls", "iterative_wls", "range_weighted"}

    def __init__(self, anchors: np.ndarray, method: str = "iterative_ls"):
        """
        Initialize TOA positioner.

        Args:
            anchors: Array of anchor positions, shape (N, 2) or (N, 3).
            method: Positioning method. One of:
                - "iterative_ls" (default): Iterative LS with W=I (Eq. 4.20)
                - "iterative_wls": Iterative WLS with user covariance (Eq. 4.23)
                - "range_weighted": Heuristic 1/d^2 weighting (not in book)
                - "ls", "iwls": Deprecated aliases (backward compatibility)
        """
        self.anchors = np.asarray(anchors, dtype=float)
        self.n_anchors = self.anchors.shape[0]
        self.dim = self.anchors.shape[1]

        # Handle method aliases
        method_lower = method.lower()
        if method_lower in self._METHOD_ALIASES:
            self.method = self._METHOD_ALIASES[method_lower]
        elif method_lower in self._VALID_METHODS:
            self.method = method_lower
        else:
            valid = list(self._VALID_METHODS) + list(self._METHOD_ALIASES.keys())
            raise ValueError(f"method must be one of {valid}, got {method}")

    def solve(
        self,
        ranges: np.ndarray,
        initial_guess: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve TOA positioning problem via iterative linearization.

        Implements Eqs. (4.14)-(4.23): Linearize the nonlinear range equations
        and iteratively refine the solution until convergence.

        **Algorithm:**

        1. Linearize measurement model via Taylor expansion (Eq. 4.17)
        2. Compute Jacobian H (Eq. 4.18)
        3. Solve LS or WLS for position update (Eq. 4.20 or 4.23)
        4. Update position and repeat until convergence (Eq. 4.21-4.22)

        **Weighting depends on method:**

        - `iterative_ls`: W = I (uniform weights, book default)
        - `iterative_wls`: W = Σ^{-1} (requires `covariance` parameter)
        - `range_weighted`: W_ii = 1/d_i^2 (heuristic, not in book)

        Args:
            ranges: Measured ranges from anchors, shape (N,).
            initial_guess: Initial position estimate, shape (d,).
            covariance: Measurement covariance matrix for iterative_wls method.
                       Shape (N, N). Required for method="iterative_wls".
                       Diagonal: σ_i^2 for range measurement to anchor i.
            max_iters: Maximum number of iterations. Defaults to 10.
            tol: Convergence tolerance in meters. Defaults to 1e-6.

        Returns:
            position: Estimated position, shape (d,).
            info: Dictionary with convergence information:
                - 'iterations': number of iterations
                - 'converged': True if converged
                - 'residual': final residual norm
                - 'history': position history
                - 'method': method used

        Example:
            >>> # Basic iterative LS (book default)
            >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
            >>> ranges = np.array([5.0, 7.07, 7.07, 5.0])
            >>> positioner = TOAPositioner(anchors, method="iterative_ls")
            >>> pos, info = positioner.solve(ranges, initial_guess=np.array([5, 5]))

            >>> # With known measurement covariance (Eq. 4.23)
            >>> sigmas = np.array([0.1, 0.2, 0.15, 0.1])  # range stds
            >>> cov = np.diag(sigmas**2)
            >>> positioner = TOAPositioner(anchors, method="iterative_wls")
            >>> pos, info = positioner.solve(ranges, np.array([5, 5]), covariance=cov)

        References:
            Chapter 4, Section 4.2.2, Eqs. (4.14)-(4.23): TOA I-LS/I-WLS
        """
        ranges = np.asarray(ranges, dtype=float)
        position = np.asarray(initial_guess, dtype=float).copy()

        if len(ranges) != self.n_anchors:
            raise ValueError(
                f"Expected {self.n_anchors} ranges, got {len(ranges)}"
            )

        # Initialize weight matrix based on method
        if self.method == "iterative_ls":
            # Eq. 4.20: W = I (uniform weights)
            W = np.eye(self.n_anchors)
        elif self.method == "iterative_wls":
            # Eq. 4.23: W = Σ^{-1}
            if covariance is None:
                raise ValueError(
                    "method='iterative_wls' requires covariance parameter"
                )
            W = np.linalg.inv(covariance)
        elif self.method == "range_weighted":
            # Heuristic: will be updated each iteration
            W = np.eye(self.n_anchors)
        else:
            W = np.eye(self.n_anchors)

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

            # Compute Jacobian (linearization, Eq. 4.17-4.18)
            # H[i, :] = (x_a - x^i) / d_a^i
            H = np.zeros((self.n_anchors, self.dim))
            for i in range(self.n_anchors):
                diff = self.anchors[i] - position
                dist = predicted_ranges[i]
                if dist > 1e-10:  # Avoid division by zero
                    H[i] = -diff / dist

            # Solve weighted least squares (Eq. 4.20 or 4.23)
            # Δx = (H^T W H)^{-1} H^T W r
            try:
                if self.method == "range_weighted":
                    # Heuristic: W_ii = 1/d_i^2 (down-weight far anchors)
                    # NOTE: This is NOT from the book!
                    diag_weights = 1.0 / np.maximum(predicted_ranges**2, 1e-6)
                    W = np.diag(diag_weights)

                # Weighted LS solution
                delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residuals)
                position = position + delta
                history.append(position.copy())

            except np.linalg.LinAlgError:
                # Singular matrix, stop iteration
                break

            # Check step size convergence (Eq. 4.22)
            if np.linalg.norm(delta) < tol:
                converged = True
                break

        info = {
            "iterations": iteration + 1,
            "converged": converged,
            "residual": residual_norm if converged else np.linalg.norm(residuals),
            "history": np.array(history),
            "method": self.method,
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
        Extended state vector: x = [x_e, x_n, (x_u), c*Δt]^T

    **Unit Convention:**

    The clock bias is estimated in METERS (c*Δt), not seconds.
    This is the book's convention (Eq. 4.24) because it simplifies the
    Jacobian matrix: ∂h/∂(c*Δt) = 1 (see Eq. 4.26).

    To convert to/from seconds, use:
    - `clock_bias_meters_to_seconds(bias_m)` → seconds
    - `clock_bias_seconds_to_meters(bias_s)` → meters

    **Physical interpretation:**

    - Positive bias_m: Receiver clock is AHEAD of system time
      → Measured ranges are LARGER than true distances
    - Negative bias_m: Receiver clock is BEHIND system time
      → Measured ranges are SMALLER than true distances
    - Scale: 1 nanosecond ≈ 0.3 meters

    Args:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        ranges: Measured ranges from anchors, shape (N,).
                These are pseudoranges: d_measured = d_true + c*Δt
        initial_guess: Initial estimate, shape (d+1,).
                      Format: [x, y, (z), bias_m] where bias_m is in METERS.
        max_iters: Maximum number of iterations. Defaults to 10.
        tol: Convergence tolerance. Defaults to 1e-6.

    Returns:
        position: Estimated position in meters, shape (d,).
        clock_bias_m: Estimated clock bias in METERS (c*Δt).
                     Convert to seconds: bias_s = bias_m / c
        info: Dictionary with convergence information.

    Example:
        >>> from core.rf import clock_bias_meters_to_seconds
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> # Ranges with 3m clock bias (~10 ns)
        >>> true_ranges = np.array([5.0, 7.07, 7.07, 5.0])
        >>> ranges = true_ranges + 3.0  # Add 3m bias
        >>> initial = np.array([5.0, 5.0, 0.0])  # [x, y, bias_m]
        >>> pos, bias_m, info = toa_solve_with_clock_bias(anchors, ranges, initial)
        >>> print(f"Clock bias: {bias_m:.2f} m")
        Clock bias: 3.00 m
        >>> bias_ns = clock_bias_meters_to_seconds(bias_m) * 1e9
        >>> print(f"Clock bias: {bias_ns:.2f} ns")
        Clock bias: 10.01 ns

    References:
        Chapter 4, Eqs. (4.24)-(4.26): Joint position and clock bias estimation
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
    AOA positioning using Iterative Weighted Least Squares (I-WLS).

    Implements Eqs. (4.63)-(4.78) from Chapter 4:
        - Measurement model: Eq. (4.63)-(4.65)
            f_i(x) = sin(θ_i) = (x_u^i - x_u,a) / d_i  (elevation)
            g_i(x) = tan(ψ_i) = (x_e^i - x_e,a) / (x_n^i - x_n,a)  (azimuth)
        - Jacobians: Eq. (4.68)-(4.70) for f_i, Eq. (4.72)-(4.74) for g_i
        - I-WLS solution: Eq. (4.77)-(4.78)

    Supports both 2D and 3D positioning:
        - 2D: Uses tan(ψ) measurements only (azimuth)
        - 3D: Uses [sin(θ), tan(ψ)] measurements (elevation + azimuth)

    Attributes:
        anchors: Array of anchor positions, shape (N, 2) or (N, 3).
        n_anchors: Number of anchors.
        dim: Dimension (2 or 3).
        is_3d: True if 3D positioning.
    """

    # Threshold for handling singularity when Δn ≈ 0
    _SINGULARITY_THRESHOLD = 1e-10

    def __init__(self, anchors: np.ndarray):
        """
        Initialize AOA positioner.

        Args:
            anchors: Array of anchor positions, shape (N, 2) or (N, 3).
                    - 2D: [E, N] coordinates (East, North)
                    - 3D: [E, N, U] coordinates (East, North, Up)
        """
        self.anchors = np.asarray(anchors, dtype=float)
        self.n_anchors = self.anchors.shape[0]
        self.dim = self.anchors.shape[1]
        self.is_3d = self.dim == 3

        if self.dim not in [2, 3]:
            raise ValueError(f"Anchors must be 2D or 3D, got dim={self.dim}")

    def _compute_predicted_and_jacobian_3d(
        self, position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predicted measurements and Jacobian for 3D case.

        Implements:
            - f_i(x) = sin(θ_i) = Δu / d  (Eq. 4.63)
            - g_i(x) = tan(ψ_i) = Δe / Δn  (Eq. 4.64)
            - Jacobians from Eqs. 4.68-4.70, 4.72-4.74

        Args:
            position: Current position estimate [x_e, x_n, x_u].

        Returns:
            predicted: Predicted measurement vector [sin(θ_1), tan(ψ_1), ...].
            H: Jacobian matrix of shape (2*N, 3).
        """
        n_meas = 2 * self.n_anchors
        predicted = np.zeros(n_meas)
        H = np.zeros((n_meas, 3))

        for i in range(self.n_anchors):
            # Differences: anchor - agent (book convention)
            delta_e = self.anchors[i, 0] - position[0]  # x_e^i - x_e,a
            delta_n = self.anchors[i, 1] - position[1]  # x_n^i - x_n,a
            delta_u = self.anchors[i, 2] - position[2]  # x_u^i - x_u,a

            # 3D distance
            d_sq = delta_e**2 + delta_n**2 + delta_u**2
            d = np.sqrt(d_sq)
            d_cubed = d**3

            # Horizontal distance squared
            horiz_sq = delta_e**2 + delta_n**2

            # Row indices for this anchor
            row_f = 2 * i  # sin(θ) row
            row_g = 2 * i + 1  # tan(ψ) row

            # === f_i = sin(θ_i) = Δu / d (Eq. 4.63) ===
            if d > self._SINGULARITY_THRESHOLD:
                predicted[row_f] = delta_u / d

                # Jacobians for f_i (Eqs. 4.68-4.70)
                # ∂f/∂x_e,a = Δu * Δe / d³
                H[row_f, 0] = delta_u * delta_e / d_cubed
                # ∂f/∂x_n,a = Δu * Δn / d³
                H[row_f, 1] = delta_u * delta_n / d_cubed
                # ∂f/∂x_u,a = -(Δe² + Δn²) / d³
                H[row_f, 2] = -horiz_sq / d_cubed
            else:
                # Agent very close to anchor - use zero Jacobian
                predicted[row_f] = 0.0
                H[row_f, :] = 0.0

            # === g_i = tan(ψ_i) = Δe / Δn (Eq. 4.64) ===
            if np.abs(delta_n) > self._SINGULARITY_THRESHOLD:
                predicted[row_g] = delta_e / delta_n

                # Jacobians for g_i (Eqs. 4.72-4.74)
                # ∂g/∂x_e,a = -1 / Δn
                H[row_g, 0] = -1.0 / delta_n
                # ∂g/∂x_n,a = Δe / Δn²
                H[row_g, 1] = delta_e / (delta_n**2)
                # ∂g/∂x_u,a = 0
                H[row_g, 2] = 0.0
            else:
                # Anchor directly East or West - singularity
                # Use a large value with correct sign
                sign_e = np.sign(delta_e) if np.abs(delta_e) > 1e-10 else 0.0
                predicted[row_g] = sign_e * 1e10

                # Use regularized Jacobian
                reg_delta_n = np.sign(delta_n) * self._SINGULARITY_THRESHOLD
                if reg_delta_n == 0:
                    reg_delta_n = self._SINGULARITY_THRESHOLD
                H[row_g, 0] = -1.0 / reg_delta_n
                H[row_g, 1] = delta_e / (reg_delta_n**2)
                H[row_g, 2] = 0.0

        return predicted, H

    def _compute_predicted_and_jacobian_2d(
        self, position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predicted measurements and Jacobian for 2D case.

        Uses tan(ψ) measurements only (no elevation).

        Args:
            position: Current position estimate [x_e, x_n].

        Returns:
            predicted: Predicted measurement vector [tan(ψ_1), tan(ψ_2), ...].
            H: Jacobian matrix of shape (N, 2).
        """
        predicted = np.zeros(self.n_anchors)
        H = np.zeros((self.n_anchors, 2))

        for i in range(self.n_anchors):
            # Differences: anchor - agent (book convention)
            delta_e = self.anchors[i, 0] - position[0]  # x_e^i - x_e,a
            delta_n = self.anchors[i, 1] - position[1]  # x_n^i - x_n,a

            # g_i = tan(ψ_i) = Δe / Δn (Eq. 4.64)
            if np.abs(delta_n) > self._SINGULARITY_THRESHOLD:
                predicted[i] = delta_e / delta_n

                # Jacobians for g_i (Eqs. 4.72-4.73)
                H[i, 0] = -1.0 / delta_n  # ∂g/∂x_e,a
                H[i, 1] = delta_e / (delta_n**2)  # ∂g/∂x_n,a
            else:
                # Singularity handling
                sign_e = np.sign(delta_e) if np.abs(delta_e) > 1e-10 else 0.0
                predicted[i] = sign_e * 1e10

                reg_delta_n = np.sign(delta_n) * self._SINGULARITY_THRESHOLD
                if reg_delta_n == 0:
                    reg_delta_n = self._SINGULARITY_THRESHOLD
                H[i, 0] = -1.0 / reg_delta_n
                H[i, 1] = delta_e / (reg_delta_n**2)

        return predicted, H

    def _angles_to_sin_tan(self, aoa_measurements: np.ndarray) -> np.ndarray:
        """
        Convert raw angle measurements to sin/tan measurement vector.

        Args:
            aoa_measurements: Raw angles.
                - 2D: (N,) array of azimuth angles ψ
                - 3D: (2*N,) array of [θ_1, ψ_1, θ_2, ψ_2, ...]

        Returns:
            Measurement vector z:
                - 2D: [tan(ψ_1), tan(ψ_2), ...]
                - 3D: [sin(θ_1), tan(ψ_1), sin(θ_2), tan(ψ_2), ...]
        """
        if self.is_3d:
            # 3D: input is [θ_1, ψ_1, θ_2, ψ_2, ...]
            z = np.zeros(2 * self.n_anchors)
            for i in range(self.n_anchors):
                theta_i = aoa_measurements[2 * i]  # elevation
                psi_i = aoa_measurements[2 * i + 1]  # azimuth
                z[2 * i] = np.sin(theta_i)
                z[2 * i + 1] = np.tan(psi_i)
            return z
        else:
            # 2D: input is [ψ_1, ψ_2, ...]
            return np.tan(aoa_measurements)

    def _compute_weight_matrix(
        self,
        aoa_measurements: np.ndarray,
        sigma_theta: Optional[Union[float, np.ndarray]] = None,
        sigma_psi: Optional[Union[float, np.ndarray]] = None,
        sigma_sin_theta: Optional[Union[float, np.ndarray]] = None,
        sigma_tan_psi: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Compute AOA weight matrix W_a = Σ_a^{-1} from measurement noise.

        Implements the book's AOA weighting using first-order error propagation
        from angle domain to sin/tan domain:
            var(sin θ) ≈ (cos θ)² * var(θ)
            var(tan ψ) ≈ (sec² ψ)² * var(ψ) = (1 + tan² ψ)² * var(ψ)

        Args:
            aoa_measurements: Raw angle measurements in radians.
                - 2D: shape (N,) with azimuth angles ψ
                - 3D: shape (2*N,) with [θ_1, ψ_1, θ_2, ψ_2, ...]
            sigma_theta: Std of elevation angle θ in radians (3D only).
                        Scalar for uniform noise, or (N,) array per anchor.
            sigma_psi: Std of azimuth angle ψ in radians.
                      Scalar for uniform noise, or (N,) array per anchor.
            sigma_sin_theta: Direct std of sin(θ) measurement (overrides sigma_theta).
            sigma_tan_psi: Direct std of tan(ψ) measurement (overrides sigma_psi).

        Returns:
            Weight matrix W_a of shape (n_meas, n_meas).
            If no sigma inputs provided, returns identity matrix.

        Notes:
            - For robustness near ψ = ±π/2, var(tan ψ) is capped at a maximum.
            - The weight matrix is diagonal (uncorrelated measurements assumed).
            - When sigma_theta/sigma_psi are provided in angle domain, variances
              are computed via first-order error propagation at the given angles.
        """
        if self.is_3d:
            n_meas = 2 * self.n_anchors
        else:
            n_meas = self.n_anchors

        # Check if any sigma input is provided
        has_angle_domain = sigma_theta is not None or sigma_psi is not None
        has_transformed = sigma_sin_theta is not None or sigma_tan_psi is not None

        if not has_angle_domain and not has_transformed:
            # No noise info provided, return identity
            return np.eye(n_meas)

        # Initialize variance array
        variances = np.ones(n_meas)

        # Maximum variance for regularization (prevents numerical issues)
        MAX_VARIANCE = 1e8
        MIN_VARIANCE = 1e-12

        if self.is_3d:
            # 3D case: measurements are [sin(θ_1), tan(ψ_1), ...]
            for i in range(self.n_anchors):
                theta_i = aoa_measurements[2 * i]  # elevation
                psi_i = aoa_measurements[2 * i + 1]  # azimuth

                idx_sin = 2 * i  # sin(θ) index
                idx_tan = 2 * i + 1  # tan(ψ) index

                # === sin(θ) variance ===
                if sigma_sin_theta is not None:
                    # Direct transformed-domain std
                    if np.isscalar(sigma_sin_theta):
                        variances[idx_sin] = sigma_sin_theta**2
                    else:
                        variances[idx_sin] = sigma_sin_theta[i] ** 2
                elif sigma_theta is not None:
                    # First-order error propagation: var(sin θ) ≈ cos²(θ) * var(θ)
                    if np.isscalar(sigma_theta):
                        var_theta = sigma_theta**2
                    else:
                        var_theta = sigma_theta[i] ** 2
                    cos_theta = np.cos(theta_i)
                    variances[idx_sin] = cos_theta**2 * var_theta
                # else: keep default variance = 1

                # === tan(ψ) variance ===
                if sigma_tan_psi is not None:
                    # Direct transformed-domain std
                    if np.isscalar(sigma_tan_psi):
                        variances[idx_tan] = sigma_tan_psi**2
                    else:
                        variances[idx_tan] = sigma_tan_psi[i] ** 2
                elif sigma_psi is not None:
                    # First-order error propagation:
                    # var(tan ψ) ≈ sec⁴(ψ) * var(ψ) = (1 + tan²ψ)² * var(ψ)
                    if np.isscalar(sigma_psi):
                        var_psi = sigma_psi**2
                    else:
                        var_psi = sigma_psi[i] ** 2
                    tan_psi = np.tan(psi_i)
                    sec_sq = 1.0 + tan_psi**2  # sec²(ψ)
                    variances[idx_tan] = sec_sq**2 * var_psi
                # else: keep default variance = 1

        else:
            # 2D case: measurements are [tan(ψ_1), tan(ψ_2), ...]
            for i in range(self.n_anchors):
                psi_i = aoa_measurements[i]

                if sigma_tan_psi is not None:
                    # Direct transformed-domain std
                    if np.isscalar(sigma_tan_psi):
                        variances[i] = sigma_tan_psi**2
                    else:
                        variances[i] = sigma_tan_psi[i] ** 2
                elif sigma_psi is not None:
                    # First-order error propagation:
                    # var(tan ψ) ≈ sec⁴(ψ) * var(ψ)
                    if np.isscalar(sigma_psi):
                        var_psi = sigma_psi**2
                    else:
                        var_psi = sigma_psi[i] ** 2
                    tan_psi = np.tan(psi_i)
                    sec_sq = 1.0 + tan_psi**2  # sec²(ψ)
                    variances[i] = sec_sq**2 * var_psi
                # else: keep default variance = 1

        # Clamp variances for numerical stability
        variances = np.clip(variances, MIN_VARIANCE, MAX_VARIANCE)

        # Build weight matrix W = Σ^{-1} (diagonal)
        weights = 1.0 / variances
        W = np.diag(weights)

        return W

    def solve(
        self,
        aoa_measurements: np.ndarray,
        initial_guess: np.ndarray,
        weights: Optional[np.ndarray] = None,
        sigma_theta: Optional[Union[float, np.ndarray]] = None,
        sigma_psi: Optional[Union[float, np.ndarray]] = None,
        sigma_sin_theta: Optional[Union[float, np.ndarray]] = None,
        sigma_tan_psi: Optional[Union[float, np.ndarray]] = None,
        recompute_weights: bool = True,
        max_iters: int = 20,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve AOA positioning problem using I-WLS.

        Implements Eqs. (4.63)-(4.78): Book-consistent I-WLS for AOA.

        The measurement model uses:
            - f_i(x) = sin(θ_i) = (x_u^i - x_u,a) / d_i  (Eq. 4.63)
            - g_i(x) = tan(ψ_i) = (x_e^i - x_e,a) / (x_n^i - x_n,a)  (Eq. 4.64)

        Linearization uses Jacobians from Eqs. 4.68-4.70 and 4.72-4.74.

        **Book-Default Weighting (W_a = Σ_a^{-1}):**
        When sigma parameters are provided, the weight matrix is computed via
        first-order error propagation from angle domain to sin/tan domain:
            var(sin θ) ≈ cos²(θ) * var(θ)
            var(tan ψ) ≈ sec⁴(ψ) * var(ψ) = (1 + tan²ψ)² * var(ψ)

        This properly accounts for heterogeneous noise and angle-dependent
        variance amplification in the transformed measurement domain.

        Args:
            aoa_measurements: AOA measurements in radians.
                - 2D: shape (N,) with azimuth angles ψ
                - 3D: shape (2*N,) with [θ_1, ψ_1, θ_2, ψ_2, ...]
                  where θ = elevation, ψ = azimuth
            initial_guess: Initial position estimate.
                - 2D: shape (2,) as [E, N]
                - 3D: shape (3,) as [E, N, U]
            weights: Optional explicit weight matrix (overrides sigma inputs).
                - 2D: shape (N,) or (N, N)
                - 3D: shape (2*N,) or (2*N, 2*N)
                If None and no sigma inputs, uniform weights are used.
            sigma_theta: Std of elevation angle θ in radians (3D only).
                        Scalar for uniform noise, or (N,) array per anchor.
            sigma_psi: Std of azimuth angle ψ in radians.
                      Scalar for uniform noise, or (N,) array per anchor.
            sigma_sin_theta: Direct std of sin(θ) (overrides sigma_theta).
            sigma_tan_psi: Direct std of tan(ψ) (overrides sigma_psi).
            recompute_weights: If True (default), recompute W_a each iteration
                              using predicted angles at current estimate.
                              If False, compute W_a once using initial measurements.
            max_iters: Maximum iterations. Defaults to 20.
            tol: Convergence tolerance. Defaults to 1e-6.

        Returns:
            position: Estimated position, shape (2,) or (3,).
            info: Dictionary with convergence information:
                - 'iterations': number of iterations
                - 'converged': True if converged
                - 'residual': final residual norm
                - 'history': position history
                - 'final_weights': final weight matrix used

        Example:
            >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
            >>> # Generate azimuth measurements to agent at (4, 6)
            >>> from core.rf import aoa_angle_vector
            >>> aoa = aoa_angle_vector(anchors, np.array([4.0, 6.0]))
            >>> positioner = AOAPositioner(anchors)
            >>> # Without weighting (identity)
            >>> pos, info = positioner.solve(aoa, initial_guess=np.array([5, 5]))
            >>> # With book-default weighting (sigma in radians)
            >>> pos, info = positioner.solve(
            ...     aoa, initial_guess=np.array([5, 5]),
            ...     sigma_psi=np.deg2rad(2.0)  # 2° azimuth std
            ... )

        Notes:
            - When sigma_psi is provided for 2D (or sigma_psi + sigma_theta for 3D),
              the solver uses W_a ≠ I, which improves accuracy when measurements
              have heterogeneous noise or when angles are near singularities.
            - For angles near ψ = ±90°, the tan(ψ) variance becomes very large,
              effectively down-weighting those measurements (correct behavior).
            - If both `weights` and sigma inputs are provided, `weights` takes
              precedence.
        """
        aoa_measurements = np.asarray(aoa_measurements, dtype=float)
        position = np.asarray(initial_guess, dtype=float).copy()

        # Validate input dimensions
        if self.is_3d:
            expected_meas = 2 * self.n_anchors
            expected_dim = 3
        else:
            expected_meas = self.n_anchors
            expected_dim = 2

        if len(aoa_measurements) != expected_meas:
            raise ValueError(
                f"Expected {expected_meas} AOA measurements for "
                f"{'3D' if self.is_3d else '2D'}, got {len(aoa_measurements)}"
            )

        if len(position) != expected_dim:
            raise ValueError(
                f"Expected initial_guess of length {expected_dim}, "
                f"got {len(position)}"
            )

        # Convert raw angles to sin/tan measurement vector
        z_measured = self._angles_to_sin_tan(aoa_measurements)

        # Determine weight matrix source
        n_meas = len(z_measured)
        use_sigma_weights = (
            weights is None and
            (sigma_theta is not None or sigma_psi is not None or
             sigma_sin_theta is not None or sigma_tan_psi is not None)
        )

        # Initialize weight matrix
        if weights is not None:
            # Explicit weights provided (take precedence)
            if weights.ndim == 1:
                W = np.diag(weights)
            else:
                W = weights
            use_sigma_weights = False
        elif use_sigma_weights:
            # Compute initial weights from sigma inputs
            W = self._compute_weight_matrix(
                aoa_measurements,
                sigma_theta=sigma_theta,
                sigma_psi=sigma_psi,
                sigma_sin_theta=sigma_sin_theta,
                sigma_tan_psi=sigma_tan_psi,
            )
        else:
            # Default: identity weights
            W = np.eye(n_meas)

        # Iteration history
        history = [position.copy()]
        converged = False
        residual_norm = np.inf

        for iteration in range(max_iters):
            # Compute predicted measurements and Jacobian
            if self.is_3d:
                z_predicted, H = self._compute_predicted_and_jacobian_3d(position)
            else:
                z_predicted, H = self._compute_predicted_and_jacobian_2d(position)

            # Recompute weights at current estimate (if using sigma-based weights)
            if use_sigma_weights and recompute_weights and iteration > 0:
                # Compute predicted angles at current position
                predicted_angles = self._predicted_to_angles(z_predicted)
                W = self._compute_weight_matrix(
                    predicted_angles,
                    sigma_theta=sigma_theta,
                    sigma_psi=sigma_psi,
                    sigma_sin_theta=sigma_sin_theta,
                    sigma_tan_psi=sigma_tan_psi,
                )

            # Compute residuals (no angle wrapping needed for sin/tan)
            residuals = z_measured - z_predicted

            # Check convergence
            residual_norm = np.linalg.norm(residuals)
            if residual_norm < tol:
                converged = True
                break

            # Weighted Least Squares solution (Eq. 4.77)
            # δx = (H^T W H)^{-1} H^T W (z - h(x))
            try:
                HtWH = H.T @ W @ H
                HtW_residuals = H.T @ W @ residuals

                # Check for singular matrix
                if np.linalg.cond(HtWH) > 1e12:
                    # Use pseudo-inverse for ill-conditioned case
                    delta = np.linalg.lstsq(HtWH, HtW_residuals, rcond=None)[0]
                else:
                    delta = np.linalg.solve(HtWH, HtW_residuals)

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
            "residual": residual_norm,
            "history": np.array(history),
            "final_weights": W,
        }

        return position, info

    def _predicted_to_angles(self, z_predicted: np.ndarray) -> np.ndarray:
        """
        Convert predicted sin/tan measurements back to angles.

        This is used to recompute weights at the current position estimate.

        Args:
            z_predicted: Predicted measurement vector.
                - 2D: [tan(ψ_1), tan(ψ_2), ...]
                - 3D: [sin(θ_1), tan(ψ_1), sin(θ_2), tan(ψ_2), ...]

        Returns:
            Angle vector in radians.
                - 2D: [ψ_1, ψ_2, ...]
                - 3D: [θ_1, ψ_1, θ_2, ψ_2, ...]
        """
        if self.is_3d:
            angles = np.zeros(2 * self.n_anchors)
            for i in range(self.n_anchors):
                sin_theta = z_predicted[2 * i]
                tan_psi = z_predicted[2 * i + 1]
                # Clamp sin_theta to valid range [-1, 1]
                sin_theta = np.clip(sin_theta, -1.0, 1.0)
                angles[2 * i] = np.arcsin(sin_theta)  # θ
                angles[2 * i + 1] = np.arctan(tan_psi)  # ψ
            return angles
        else:
            return np.arctan(z_predicted)


def aoa_ove_solve(
    anchors: np.ndarray,
    elevation_angles: np.ndarray,
    azimuth_angles: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Orthogonal Vector Estimator (OVE) for 3D AOA positioning.

    Implements Eqs. (4.79)-(4.85) from Chapter 4:
        Closed-form 3D positioning using orthogonal projection.

    The OVE constructs unit direction vectors from each beacon to the agent,
    then solves a linear system by projecting onto orthogonal vectors.

    Measurement model (Eq. 4.79):
        eta_i = [cos(theta_i)*sin(psi_i), cos(theta_i)*cos(psi_i), sin(theta_i)]^T

    Orthogonal vector (Eq. 4.82):
        s_i = [-sin(theta_i)*sin(psi_i), -sin(theta_i)*cos(psi_i), cos(theta_i)]^T

    Linear system (Eq. 4.84):
        S_a * x_a = z_a

    LS solution (Eq. 4.85):
        x_a = (S_a^T * S_a)^{-1} * S_a^T * z_a

    Args:
        anchors: Array of anchor positions, shape (N, 3) as [E, N, U].
        elevation_angles: Elevation angles in radians, shape (N,).
                         Positive when anchor is above agent.
        azimuth_angles: Azimuth angles in radians, shape (N,).
                       Measured from North, positive CCW (ENU convention).

    Returns:
        position: Estimated 3D position [E, N, U].
        info: Dictionary with solver information:
            - 'method': 'OVE'
            - 'condition_number': condition number of S_a^T * S_a

    Example:
        >>> anchors = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]])
        >>> # Agent at (5, 5, 0), compute angles from each anchor
        >>> from core.rf import aoa_elevation, aoa_azimuth
        >>> theta = [aoa_elevation(a, np.array([5, 5, 0])) for a in anchors]
        >>> psi = [aoa_azimuth(a, np.array([5, 5, 0])) for a in anchors]
        >>> pos, info = aoa_ove_solve(anchors, np.array(theta), np.array(psi))

    Note:
        OVE is a biased estimator. The bias increases with measurement noise
        and distance from beacons (see book Section 4.4.2).
    """
    anchors = np.asarray(anchors, dtype=float)
    elevation_angles = np.asarray(elevation_angles, dtype=float)
    azimuth_angles = np.asarray(azimuth_angles, dtype=float)

    n_anchors = anchors.shape[0]

    if anchors.shape[1] != 3:
        raise ValueError("OVE requires 3D anchors with shape (N, 3)")
    if len(elevation_angles) != n_anchors or len(azimuth_angles) != n_anchors:
        raise ValueError(
            f"Expected {n_anchors} elevation and azimuth angles each"
        )

    # Build the S_a matrix and z_a vector (Eq. 4.84)
    S_a = np.zeros((n_anchors, 3))
    z_a = np.zeros(n_anchors)

    for i in range(n_anchors):
        theta_i = elevation_angles[i]
        psi_i = azimuth_angles[i]

        sin_theta = np.sin(theta_i)
        cos_theta = np.cos(theta_i)
        sin_psi = np.sin(psi_i)
        cos_psi = np.cos(psi_i)

        # Orthogonal vector s_i (Eq. 4.82)
        # s_i = [-sin(theta)*sin(psi), -sin(theta)*cos(psi), cos(theta)]
        S_a[i, 0] = -sin_theta * sin_psi  # coefficient for x_e
        S_a[i, 1] = -sin_theta * cos_psi  # coefficient for x_n
        S_a[i, 2] = cos_theta  # coefficient for x_u

        # Right-hand side (Eq. 4.83)
        # z_i = s_i^T * x^i
        x_e_i, x_n_i, x_u_i = anchors[i]
        z_a[i] = (
            -sin_theta * sin_psi * x_e_i
            - sin_theta * cos_psi * x_n_i
            + cos_theta * x_u_i
        )

    # Solve LS (Eq. 4.85): x_a = (S_a^T * S_a)^{-1} * S_a^T * z_a
    StS = S_a.T @ S_a
    cond_num = np.linalg.cond(StS)

    try:
        if cond_num > 1e12:
            # Ill-conditioned, use pseudo-inverse
            position = np.linalg.lstsq(S_a, z_a, rcond=None)[0]
        else:
            position = np.linalg.solve(StS, S_a.T @ z_a)
    except np.linalg.LinAlgError:
        # Singular matrix
        position = np.linalg.lstsq(S_a, z_a, rcond=None)[0]

    info = {
        "method": "OVE",
        "condition_number": cond_num,
    }

    return position, info


def aoa_ple_solve_2d(
    anchors: np.ndarray,
    azimuth_angles: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    2D Pseudolinear Estimator (PLE) for AOA positioning.

    Implements Eqs. (4.86)-(4.91) from Chapter 4:
        Closed-form 2D positioning using line-of-bearing equations.

    The PLE converts each azimuth measurement to a line equation and
    finds the intersection via least squares.

    Line-of-bearing form (Eq. 4.89):
        cos(psi_i)*x_e - sin(psi_i)*x_n = cos(psi_i)*x_e^i - sin(psi_i)*x_n^i

    Linear system (Eq. 4.90):
        S_a * [x_e; x_n] = z_a
        where S_a[i, :] = [cos(psi_i), -sin(psi_i)]
              z_a[i] = cos(psi_i)*x_e^i - sin(psi_i)*x_n^i

    LS solution (Eq. 4.91):
        [x_e; x_n] = (S_a^T * S_a)^{-1} * S_a^T * z_a

    Args:
        anchors: Array of anchor positions, shape (N, 2) as [E, N].
        azimuth_angles: Azimuth angles in radians, shape (N,).
                       Measured from North, positive CCW (ENU convention).

    Returns:
        position: Estimated 2D position [E, N].
        info: Dictionary with solver information:
            - 'method': 'PLE_2D'
            - 'condition_number': condition number of S_a^T * S_a
            - 'geometry_warning': True if near-parallel bearings detected

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> # Agent at (5, 5), compute azimuth angles from each anchor
        >>> from core.rf import aoa_azimuth
        >>> psi = [aoa_azimuth(a, np.array([5, 5])) for a in anchors]
        >>> pos, info = aoa_ple_solve_2d(anchors, np.array(psi))

    Note:
        PLE is biased, especially with:
        - High measurement noise
        - Poor geometry (near-parallel bearings from aligned anchors)
        PLE is often used as an initial guess for iterative methods.
    """
    anchors = np.asarray(anchors, dtype=float)
    azimuth_angles = np.asarray(azimuth_angles, dtype=float)

    n_anchors = anchors.shape[0]

    if anchors.shape[1] != 2:
        raise ValueError("PLE_2D requires 2D anchors with shape (N, 2)")
    if len(azimuth_angles) != n_anchors:
        raise ValueError(f"Expected {n_anchors} azimuth angles")

    # Build the S_a matrix and z_a vector (Eq. 4.90)
    S_a = np.zeros((n_anchors, 2))
    z_a = np.zeros(n_anchors)

    for i in range(n_anchors):
        psi_i = azimuth_angles[i]

        cos_psi = np.cos(psi_i)
        sin_psi = np.sin(psi_i)

        # Line-of-bearing coefficients (Eq. 4.89)
        # cos(psi)*x_e - sin(psi)*x_n = cos(psi)*x_e^i - sin(psi)*x_n^i
        S_a[i, 0] = cos_psi  # coefficient for x_e
        S_a[i, 1] = -sin_psi  # coefficient for x_n

        # Right-hand side
        x_e_i, x_n_i = anchors[i]
        z_a[i] = cos_psi * x_e_i - sin_psi * x_n_i

    # Check for geometry warning (near-parallel lines)
    # Parallel lines occur when bearings are similar
    geometry_warning = False
    if n_anchors >= 2:
        # Check angle spread
        angle_diffs = []
        for i in range(n_anchors):
            for j in range(i + 1, n_anchors):
                diff = np.abs(azimuth_angles[i] - azimuth_angles[j])
                diff = min(diff, np.pi - diff % np.pi)  # Map to [0, pi/2]
                angle_diffs.append(diff)
        min_angle_diff = min(angle_diffs) if angle_diffs else np.pi / 2
        if min_angle_diff < np.deg2rad(10):  # Less than 10 degree spread
            geometry_warning = True

    # Solve LS (Eq. 4.91): [x_e; x_n] = (S_a^T * S_a)^{-1} * S_a^T * z_a
    StS = S_a.T @ S_a
    cond_num = np.linalg.cond(StS)

    try:
        if cond_num > 1e12:
            # Ill-conditioned, use pseudo-inverse
            position = np.linalg.lstsq(S_a, z_a, rcond=None)[0]
        else:
            position = np.linalg.solve(StS, S_a.T @ z_a)
    except np.linalg.LinAlgError:
        # Singular matrix
        position = np.linalg.lstsq(S_a, z_a, rcond=None)[0]

    info = {
        "method": "PLE_2D",
        "condition_number": cond_num,
        "geometry_warning": geometry_warning,
    }

    return position, info


def aoa_ple_solve_3d(
    anchors: np.ndarray,
    elevation_angles: np.ndarray,
    azimuth_angles: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    3D Pseudolinear Estimator (PLE) for AOA positioning.

    Implements Eqs. (4.92)-(4.95) from Chapter 4:
        Two-step estimation: 2D PLE for (E, N), then elevation averaging for U.

    Step 1: Solve 2D position using azimuth angles only (Eqs. 4.86-4.91)
    Step 2: Estimate U using elevation angles (Eqs. 4.92-4.95):
        x_u,a = (1/I) * sum(x_u^i - ||x_a(1:2) - x^i(1:2)|| * tan(theta_i))

    Args:
        anchors: Array of anchor positions, shape (N, 3) as [E, N, U].
        elevation_angles: Elevation angles in radians, shape (N,).
                         Positive when anchor is above agent.
        azimuth_angles: Azimuth angles in radians, shape (N,).
                       Measured from North, positive CCW (ENU convention).

    Returns:
        position: Estimated 3D position [E, N, U].
        info: Dictionary with solver information:
            - 'method': 'PLE_3D'
            - 'condition_number': condition number from 2D step
            - 'geometry_warning': True if poor 2D geometry
            - 'u_estimates': individual U estimates from each beacon

    Example:
        >>> anchors = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]])
        >>> # Agent at (5, 5, 0), compute angles from each anchor
        >>> from core.rf import aoa_elevation, aoa_azimuth
        >>> theta = [aoa_elevation(a, np.array([5, 5, 0])) for a in anchors]
        >>> psi = [aoa_azimuth(a, np.array([5, 5, 0])) for a in anchors]
        >>> pos, info = aoa_ple_solve_3d(anchors, np.array(theta), np.array(psi))

    Note:
        The 3D PLE inherits bias from 2D PLE and may have additional error
        if horizontal position is biased or geometry varies significantly.
    """
    anchors = np.asarray(anchors, dtype=float)
    elevation_angles = np.asarray(elevation_angles, dtype=float)
    azimuth_angles = np.asarray(azimuth_angles, dtype=float)

    n_anchors = anchors.shape[0]

    if anchors.shape[1] != 3:
        raise ValueError("PLE_3D requires 3D anchors with shape (N, 3)")
    if len(elevation_angles) != n_anchors or len(azimuth_angles) != n_anchors:
        raise ValueError(
            f"Expected {n_anchors} elevation and azimuth angles each"
        )

    # Step 1: Solve 2D position using azimuth only
    anchors_2d = anchors[:, :2]
    pos_2d, info_2d = aoa_ple_solve_2d(anchors_2d, azimuth_angles)

    # Step 2: Estimate U using elevation angles (Eqs. 4.92-4.95)
    # x_u,a^i = x_u^i - ||x_a(1:2) - x^i(1:2)|| * tan(theta_i)
    u_estimates = []
    for i in range(n_anchors):
        # Horizontal distance from estimated position to anchor
        horiz_dist = np.linalg.norm(pos_2d - anchors[i, :2])

        # Estimate U from this beacon (Eq. 4.93)
        # Note: theta > 0 means anchor above agent, so we subtract
        x_u_i = anchors[i, 2] - horiz_dist * np.tan(elevation_angles[i])
        u_estimates.append(x_u_i)

    # Average the U estimates (Eq. 4.95)
    x_u = np.mean(u_estimates)

    # Combine to 3D position
    position = np.array([pos_2d[0], pos_2d[1], x_u])

    info = {
        "method": "PLE_3D",
        "condition_number": info_2d["condition_number"],
        "geometry_warning": info_2d["geometry_warning"],
        "u_estimates": np.array(u_estimates),
    }

    return position, info


def toa_fang_solver(
    anchors: np.ndarray,
    ranges: np.ndarray,
    ref_idx: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Fang's closed-form TOA positioning algorithm.

    Implements Eqs. (4.43)-(4.49) from Chapter 4:
        Linearizes TOA equations using squared range differences to obtain
        a direct algebraic solution without iteration.

    Algorithm:
        1. Choose a reference anchor (default: first anchor)
        2. Form squared range differences: d_i² - d_ref²
        3. Rearrange into linear system: H_a * x_a = y_a (Eq. 4.48)
        4. Solve via least squares: x_a = (H_a^T H_a)^{-1} H_a^T y_a

    Linear system (Eq. 4.46-4.47):
        h_e^i = -2 * (x_e^i - x_e^ref)
        h_n^i = -2 * (x_n^i - x_n^ref)
        y^i = d_i² - d_ref² - (x_e^i² - x_e^ref²) - (x_n^i² - x_n^ref²)

    Args:
        anchors: Array of anchor positions, shape (N, 2) for 2D.
                Format: [[x_e^1, x_n^1], [x_e^2, x_n^2], ...]
        ranges: TOA range measurements, shape (N,).
                ranges[i] is the measured distance to anchor i.
        ref_idx: Index of reference anchor (default 0).

    Returns:
        position: Estimated 2D position [x_e, x_n].
        info: Dictionary with solver information:
            - 'method': 'Fang_TOA'
            - 'condition_number': condition number of H_a^T H_a
            - 'residual': residual norm of the linear system

    Raises:
        ValueError: If fewer than 3 anchors provided (minimum for 2D).
        ValueError: If anchors are not 2D.

    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> true_pos = np.array([5.0, 5.0])
        >>> ranges = np.linalg.norm(anchors - true_pos, axis=1)
        >>> pos, info = toa_fang_solver(anchors, ranges)
        >>> print(f"Position: {pos}")  # Should be close to [5, 5]

    Notes:
        - Requires at least 3 anchors for 2D positioning.
        - Non-iterative: provides direct solution without initial guess.
        - Sensitive to measurement noise; no inherent filtering.
        - Poor anchor geometry amplifies errors (GDOP effect).
        - Well-suited for real-time applications where speed is critical.

    References:
        Chapter 4, Section 4.3.3, Eqs. (4.43)-(4.49): Fang's Algorithm
        B.T. Fang, "Simple solutions for hyperbolic and related position fixes"
    """
    anchors = np.asarray(anchors, dtype=float)
    ranges = np.asarray(ranges, dtype=float)

    n_anchors = anchors.shape[0]
    dim = anchors.shape[1]

    # Validate inputs
    if dim != 2:
        raise ValueError(
            f"Fang's algorithm currently supports 2D only, got dim={dim}"
        )
    if n_anchors < 3:
        raise ValueError(
            f"Fang's algorithm requires at least 3 anchors, got {n_anchors}"
        )
    if len(ranges) != n_anchors:
        raise ValueError(
            f"Expected {n_anchors} ranges, got {len(ranges)}"
        )
    if ref_idx < 0 or ref_idx >= n_anchors:
        raise ValueError(
            f"ref_idx must be in [0, {n_anchors-1}], got {ref_idx}"
        )

    # Reference anchor position and range
    x_ref = anchors[ref_idx]  # [x_e^ref, x_n^ref]
    d_ref = ranges[ref_idx]

    # Build linear system H_a * x_a = y_a (Eq. 4.48)
    # Skip reference anchor
    non_ref_indices = [i for i in range(n_anchors) if i != ref_idx]
    n_eqs = len(non_ref_indices)

    H_a = np.zeros((n_eqs, 2))
    y_a = np.zeros(n_eqs)

    for row, i in enumerate(non_ref_indices):
        x_i = anchors[i]  # [x_e^i, x_n^i]
        d_i = ranges[i]

        # Eq. 4.47: h_e^i = -2 * (x_e^i - x_e^ref)
        H_a[row, 0] = -2.0 * (x_i[0] - x_ref[0])
        # Eq. 4.47: h_n^i = -2 * (x_n^i - x_n^ref)
        H_a[row, 1] = -2.0 * (x_i[1] - x_ref[1])

        # Eq. 4.47: y^i = d_i² - d_ref² - (x_e^i² - x_e^ref²) - (x_n^i² - x_n^ref²)
        y_a[row] = (
            d_i**2 - d_ref**2
            - (x_i[0]**2 - x_ref[0]**2)
            - (x_i[1]**2 - x_ref[1]**2)
        )

    # Solve: x_a = (H_a^T H_a)^{-1} H_a^T y_a (Eq. 4.49)
    HtH = H_a.T @ H_a
    cond_num = np.linalg.cond(HtH)

    try:
        if cond_num > 1e12:
            # Ill-conditioned, use pseudo-inverse
            position = np.linalg.lstsq(H_a, y_a, rcond=None)[0]
        else:
            position = np.linalg.solve(HtH, H_a.T @ y_a)
    except np.linalg.LinAlgError:
        # Singular matrix
        position = np.linalg.lstsq(H_a, y_a, rcond=None)[0]

    # Compute residual
    residual = np.linalg.norm(H_a @ position - y_a)

    info = {
        "method": "Fang_TOA",
        "condition_number": cond_num,
        "residual": residual,
    }

    return position, info


def tdoa_chan_solver(
    anchors: np.ndarray,
    tdoa_measurements: np.ndarray,
    ref_idx: int = 0,
    covariance: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Chan's closed-form TDOA positioning algorithm.

    Implements Eqs. (4.50)-(4.62) from Chapter 4:
        Two-step WLS solution using an auxiliary variable (reference distance)
        to linearize the hyperbolic TDOA equations.

    Algorithm:
        Step 1: Solve linearized LS problem (Eq. 4.58-4.60)
            - State vector: x_a = [x_e, x_n, d_ref]^T (includes ref distance)
            - Linear system: H_a * x_a = y_a
            - Initial LS: x_a = (H_a^T H_a)^{-1} H_a^T y_a

        Step 2: WLS refinement (Eq. 4.61-4.62)
            - Apply covariance weighting to account for correlated errors
            - x_a = (H_a^T Σ^{-1} H_a)^{-1} H_a^T Σ^{-1} y_a

    Linear system (Eq. 4.58-4.60):
        h_e^i = 2 * (x_e^ref - x_e^i)
        h_n^i = 2 * (x_n^ref - x_n^i)
        h_d^i = 2 * Δd^i  (where Δd^i = d_i - d_ref is the TDOA measurement)
        y^i = (Δd^i)² - [(x_e^i² + x_n^i²) - (x_e^ref² + x_n^ref²)]

    Args:
        anchors: Array of anchor positions, shape (N, 2) for 2D.
                Format: [[x_e^1, x_n^1], [x_e^2, x_n^2], ...]
        tdoa_measurements: TDOA measurements (range differences), shape (N-1,).
                          tdoa_measurements[k] = d^{k+1} - d^{ref} if ref_idx=0
                          (range to anchor k+1 minus range to reference)
        ref_idx: Index of reference anchor (default 0).
        covariance: Optional covariance matrix for WLS, shape (N-1, N-1).
                   If None, uses identity (LS solution).
                   For proper WLS, use build_tdoa_covariance() to construct
                   the correlated covariance matrix (Eq. 4.62).

    Returns:
        position: Estimated 2D position [x_e, x_n].
        info: Dictionary with solver information:
            - 'method': 'Chan_TDOA'
            - 'condition_number': condition number of H_a^T W H_a
            - 'reference_distance': estimated distance to reference anchor (d_ref)
            - 'residual': residual norm

    Raises:
        ValueError: If fewer than 4 anchors provided (minimum for 2D TDOA).
        ValueError: If wrong number of TDOA measurements.

    Example:
        >>> anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        >>> true_pos = np.array([8.0, 12.0])
        >>> # Compute TDOA measurements (relative to anchor 0)
        >>> d_ref = np.linalg.norm(true_pos - anchors[0])
        >>> tdoa = [np.linalg.norm(true_pos - anchors[i]) - d_ref
        ...         for i in range(1, len(anchors))]
        >>> pos, info = tdoa_chan_solver(anchors, np.array(tdoa))
        >>> print(f"Position: {pos}")  # Should be close to [8, 12]

    Notes:
        - Requires at least 4 anchors for 2D (3 non-ref + 1 ref).
        - Non-iterative: no initial guess required.
        - WLS refinement (Step 2) improves accuracy with known noise statistics.
        - May produce two symmetric solutions; algorithm selects positive d_ref.
        - For correlated TDOA noise, use build_tdoa_covariance() to construct Σ.

    References:
        Chapter 4, Section 4.3.3, Eqs. (4.50)-(4.62): Chan's Algorithm
        Y.T. Chan and K.C. Ho, "A simple and efficient estimator for
        hyperbolic location" (1994)
    """
    anchors = np.asarray(anchors, dtype=float)
    tdoa_measurements = np.asarray(tdoa_measurements, dtype=float)

    n_anchors = anchors.shape[0]
    dim = anchors.shape[1]

    # Validate inputs
    if dim != 2:
        raise ValueError(
            f"Chan's algorithm currently supports 2D only, got dim={dim}"
        )
    if n_anchors < 4:
        raise ValueError(
            f"Chan's algorithm requires at least 4 anchors, got {n_anchors}"
        )
    if len(tdoa_measurements) != n_anchors - 1:
        raise ValueError(
            f"Expected {n_anchors-1} TDOA measurements, got {len(tdoa_measurements)}"
        )
    if ref_idx < 0 or ref_idx >= n_anchors:
        raise ValueError(
            f"ref_idx must be in [0, {n_anchors-1}], got {ref_idx}"
        )

    # Reference anchor position
    x_ref = anchors[ref_idx]  # [x_e^ref, x_n^ref]

    # Non-reference anchor indices
    non_ref_indices = [i for i in range(n_anchors) if i != ref_idx]
    n_eqs = len(non_ref_indices)

    # Build linear system H_a * x_a = y_a (Eq. 4.59-4.60)
    # State: x_a = [x_e, x_n, d_ref]^T
    H_a = np.zeros((n_eqs, 3))
    y_a = np.zeros(n_eqs)

    for row, i in enumerate(non_ref_indices):
        x_i = anchors[i]  # [x_e^i, x_n^i]
        # TDOA measurement: Δd^i = d^i - d^ref
        delta_d = tdoa_measurements[row]

        # Eq. 4.60: h_e^i = 2 * (x_e^ref - x_e^i)
        H_a[row, 0] = 2.0 * (x_ref[0] - x_i[0])
        # Eq. 4.60: h_n^i = 2 * (x_n^ref - x_n^i)
        H_a[row, 1] = 2.0 * (x_ref[1] - x_i[1])
        # Eq. 4.59: third column is 2 * Δd^i (for d_ref term)
        H_a[row, 2] = 2.0 * delta_d

        # Eq. 4.60: y^i = (Δd^i)² - [(x_e^i² + x_n^i²) - (x_e^ref² + x_n^ref²)]
        y_a[row] = (
            delta_d**2
            - (x_i[0]**2 + x_i[1]**2)
            + (x_ref[0]**2 + x_ref[1]**2)
        )

    # Step 1: Initial LS solution (Eq. 4.49 applied to Chan's formulation)
    # x_a = (H_a^T H_a)^{-1} H_a^T y_a
    HtH = H_a.T @ H_a

    try:
        state_ls = np.linalg.solve(HtH, H_a.T @ y_a)
    except np.linalg.LinAlgError:
        state_ls = np.linalg.lstsq(H_a, y_a, rcond=None)[0]

    # Step 2: WLS refinement (Eq. 4.61)
    # If covariance provided, use WLS; otherwise use LS result
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (n_eqs, n_eqs):
            raise ValueError(
                f"Covariance must be ({n_eqs}, {n_eqs}), got {covariance.shape}"
            )
        try:
            W = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # Singular covariance, fall back to LS
            W = np.eye(n_eqs)

        HtWH = H_a.T @ W @ H_a
        cond_num = np.linalg.cond(HtWH)

        try:
            state_wls = np.linalg.solve(HtWH, H_a.T @ W @ y_a)
        except np.linalg.LinAlgError:
            state_wls = np.linalg.lstsq(H_a, y_a, rcond=None)[0]

        state = state_wls
    else:
        cond_num = np.linalg.cond(HtH)
        state = state_ls

    # Extract position and reference distance
    position = state[:2]  # [x_e, x_n]
    d_ref = state[2]  # Reference distance

    # Handle negative reference distance (ambiguity)
    # Book notes: select positive root for d_ref
    if d_ref < 0:
        # This can happen with poor geometry; take absolute value
        d_ref = abs(d_ref)

    # Compute residual
    residual = np.linalg.norm(H_a @ state - y_a)

    info = {
        "method": "Chan_TDOA",
        "condition_number": cond_num,
        "reference_distance": d_ref,
        "residual": residual,
    }

    return position, info
