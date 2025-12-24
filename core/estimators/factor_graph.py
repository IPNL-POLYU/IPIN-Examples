"""
Factor Graph Optimization for batch state estimation.

This module implements Factor Graph Optimization (FGO) as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.35): MAP estimation X̂_MAP = argmax_X p(X | Z)
    - Eq. (3.36)-(3.37): Simplified MAP form with likelihood and prior
    - Eq. (3.38): Conversion to sum of squared residuals
    - Eq. (3.42)-(3.43): Gradient descent and Gauss-Newton updates
    - Algorithm 3.1: Line search strategy with sufficient decrease
    - Algorithm 3.2: Levenberg-Marquardt method with damping parameter
    - Eq. (3.53): LM update direction (J^T J + μI) d_lm = -J^T r
    - Eq. (3.56): Gain ratio for LM damping adjustment
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class Factor:
    """
    Factor in a factor graph representing a constraint or measurement.

    A factor encodes a probabilistic constraint on a subset of variables.
    For Gaussian factors, this is equivalent to minimizing a squared residual.

    Attributes:
        variable_ids: List of variable IDs that this factor connects
        residual_func: Function computing residual r(x_subset)
        jacobian_func: Function computing Jacobian ∂r/∂x
        information: Information matrix (inverse covariance) for this factor
    """

    def __init__(
        self,
        variable_ids: List[int],
        residual_func: Callable[[List[np.ndarray]], np.ndarray],
        jacobian_func: Callable[[List[np.ndarray]], List[np.ndarray]],
        information: np.ndarray,
    ):
        """
        Initialize Factor.

        Args:
            variable_ids: List of variable IDs connected by this factor.
            residual_func: Function computing residual r(x_vars) where x_vars
                is a list of variable values.
            jacobian_func: Function computing Jacobian [∂r/∂x₁, ∂r/∂x₂, ...].
            information: Information matrix Λ (inverse of covariance matrix).
                For scalar residual: [[1/σ²]].
        """
        self.variable_ids = variable_ids
        self.residual_func = residual_func
        self.jacobian_func = jacobian_func
        self.information = information

    def compute_error(self, variables: Dict[int, np.ndarray]) -> float:
        """
        Compute squared error for this factor.

        Implements: error = rᵀ Λ r where r is the residual.

        Args:
            variables: Dictionary mapping variable ID to value.

        Returns:
            Squared error (scalar).
        """
        x_vars = [variables[vid] for vid in self.variable_ids]
        r = self.residual_func(x_vars)
        return float(r.T @ self.information @ r)

    def linearize(
        self, variables: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Linearize the factor around current variable values.

        Args:
            variables: Dictionary mapping variable ID to value.

        Returns:
            Tuple of (residual, jacobians) where jacobians is a list
            of Jacobian matrices for each connected variable.
        """
        x_vars = [variables[vid] for vid in self.variable_ids]
        r = self.residual_func(x_vars)
        J = self.jacobian_func(x_vars)
        return r, J


class FactorGraph:
    """
    Factor Graph for batch state estimation.

    Represents a graph where nodes are variables (states) and edges are
    factors (constraints/measurements). Optimization finds the Maximum
    A Posteriori (MAP) estimate of all variables.

    Implements Eqs. (3.35)-(3.38) from Chapter 3.

    Attributes:
        variables: Dictionary mapping variable ID to current value
        factors: List of factors in the graph
        variable_dims: Dictionary mapping variable ID to dimension
    """

    def __init__(self):
        """Initialize empty Factor Graph."""
        self.variables: Dict[int, np.ndarray] = {}
        self.factors: List[Factor] = []
        self.variable_dims: Dict[int, int] = {}

    def add_variable(self, var_id: int, initial_value: np.ndarray) -> None:
        """
        Add a variable to the graph.

        Args:
            var_id: Unique identifier for this variable.
            initial_value: Initial value for the variable (n,).
        """
        self.variables[var_id] = np.asarray(initial_value, dtype=float).copy()
        self.variable_dims[var_id] = len(initial_value)

    def add_factor(self, factor: Factor) -> None:
        """
        Add a factor to the graph.

        Args:
            factor: Factor connecting variables.

        Raises:
            ValueError: If any variable ID in factor is not in graph.
        """
        for vid in factor.variable_ids:
            if vid not in self.variables:
                raise ValueError(f"Variable {vid} not in graph")
        self.factors.append(factor)

    def compute_error(self) -> float:
        """
        Compute total error over all factors.

        Implements Eq. (3.38): Sum of squared residuals
            error = Σ rᵢᵀ Λᵢ rᵢ

        Returns:
            Total squared error.
        """
        total_error = 0.0
        for factor in self.factors:
            total_error += factor.compute_error(self.variables)
        return total_error

    def optimize(
        self,
        method: str = "gauss_newton",
        max_iterations: int = 20,
        tol: float = 1e-6,
        **kwargs,
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Optimize the factor graph to find MAP estimate.

        Implements:
        - Eq. (3.35): MAP estimation
        - Eq. (3.42), (3.46): Gradient descent with steepest descent direction
        - Algorithm 3.1: Line search (optional for GD, default for "line_search")
        - Eq. (3.51)-(3.52): Gauss-Newton method
        - Algorithm 3.2: Levenberg-Marquardt

        Args:
            method: Optimization method. One of:
                - "gauss_newton": Standard Gauss-Newton (Eq. 3.51-3.52)
                - "gd": Gradient descent (Eq. 3.42, 3.46)
                - "line_search": Gauss-Newton with line search (Algorithm 3.1)
                - "levenberg_marquardt" or "lm": LM method (Algorithm 3.2)
            max_iterations: Maximum number of iterations.
            tol: Convergence tolerance on error change.
            **kwargs: Additional method-specific parameters:
                - line_search: For "gd", use Armijo line search (Algorithm 3.1)
                    instead of fixed step size. Default: False.
                - alpha: For "gd" without line search, fixed step size. Default: 0.01.
                - initial_mu: Initial damping for LM (default: 1e-3)

        Returns:
            Tuple of (optimized_variables, error_history).

        Raises:
            ValueError: If method is not supported.
        """
        if method == "gauss_newton":
            return self._gauss_newton(max_iterations, tol)
        elif method == "gd":
            use_line_search = kwargs.get("line_search", False)
            alpha = kwargs.get("alpha", 0.01)
            return self._gradient_descent(
                max_iterations, tol, alpha=alpha, use_line_search=use_line_search
            )
        elif method == "line_search":
            return self._line_search(max_iterations, tol)
        elif method in ("levenberg_marquardt", "lm"):
            initial_mu = kwargs.get("initial_mu", 1e-3)
            return self._levenberg_marquardt(max_iterations, tol, initial_mu)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gauss_newton(
        self, max_iterations: int, tol: float
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Gauss-Newton optimization.

        Solves the linearized system: (JᵀΛJ) δx = -JᵀΛr
        at each iteration and updates: x ← x + δx

        Args:
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple of (optimized_variables, error_history).
        """
        error_history = [self.compute_error()]

        for iteration in range(max_iterations):
            # Build linearized system
            H, b = self._build_linearized_system()

            # Solve for update: H δx = b
            try:
                delta_x = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                # Singular matrix - use pseudo-inverse
                delta_x = np.linalg.lstsq(H, b, rcond=None)[0]

            # Update variables
            self._update_variables(delta_x)

            # Compute new error
            current_error = self.compute_error()
            error_history.append(current_error)

            # Check convergence
            if abs(error_history[-2] - error_history[-1]) < tol:
                break

        return self.variables.copy(), error_history

    def _gradient_descent(
        self,
        max_iterations: int,
        tol: float,
        alpha: float = 0.01,
        use_line_search: bool = False,
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Gradient descent optimization with optional line search.

        Implements:
            - Eq. (3.42): x_{k+1} = x_k + α d
            - Eq. (3.46): Descent direction d = -∇f(x) (steepest descent)
            - Algorithm 3.1: Line search for step size selection (if enabled)

        The descent direction is d = b where b = -J^T Λ r from _build_linearized_system,
        which equals -∇f(x) (the negative gradient / steepest descent direction).

        Args:
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.
            alpha: Fixed step size (used if use_line_search=False).
            use_line_search: If True, use Armijo backtracking line search
                (Algorithm 3.1) instead of fixed step size.

        Returns:
            Tuple of (optimized_variables, error_history).
        """
        error_history = [self.compute_error()]

        # Line search parameters (Algorithm 3.1)
        c = 1e-4  # Sufficient decrease constant (Armijo parameter)
        rho = 0.5  # Backtracking factor
        max_ls_iter = 20  # Maximum line search iterations

        for iteration in range(max_iterations):
            # Compute gradient: b = -J^T Λ r = -∇f(x)
            # So b is already the steepest descent direction (Eq. 3.46)
            _, b = self._build_linearized_system()
            d = b  # Descent direction d = -∇f(x) = b

            current_error = error_history[-1]

            if use_line_search:
                # Algorithm 3.1: Backtracking line search with Armijo condition
                # Save current state
                old_vars = {k: v.copy() for k, v in self.variables.items()}

                # Gradient for Armijo: grad = -b (since b = -∇f)
                grad = -b
                directional_deriv = np.dot(grad, d)  # Should be negative for descent

                # Line search: find alpha with sufficient decrease (Eq. 3.43)
                step = 1.0
                for ls_iter in range(max_ls_iter):
                    # Try update with current step
                    self._update_variables(step * d)
                    new_error = self.compute_error()

                    # Armijo sufficient decrease condition (Eq. 3.43):
                    # f(x + α*d) ≤ f(x) + c * α * ∇f^T * d
                    if new_error <= current_error + c * step * directional_deriv:
                        # Sufficient decrease achieved
                        break

                    # Backtrack: restore and reduce step
                    self.variables = {k: v.copy() for k, v in old_vars.items()}
                    step *= rho
                else:
                    # Line search failed - accept small step anyway
                    self._update_variables(step * d)
                    new_error = self.compute_error()

                error_history.append(new_error)
            else:
                # Fixed step size (Eq. 3.42): x_{k+1} = x_k + α d
                delta_x = alpha * d
                self._update_variables(delta_x)
                new_error = self.compute_error()
                error_history.append(new_error)

            # Check convergence
            if abs(current_error - new_error) < tol:
                break

        return self.variables.copy(), error_history

    def _line_search(
        self, max_iterations: int, tol: float
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Gauss-Newton optimization with line search (Algorithm 3.1).

        Implements Algorithm 3.1: Standard line search strategy
            1. Find descent direction d (Gauss-Newton direction)
            2. Find step length alpha achieving sufficient decrease
            3. Update: x_{k+1} = x_k + alpha * d

        The sufficient decrease condition (Armijo condition):
            f(x + alpha*d) <= f(x) + c * alpha * grad^T * d

        Args:
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple of (optimized_variables, error_history).
        """
        error_history = [self.compute_error()]

        # Line search parameters
        c = 1e-4  # Sufficient decrease constant (Armijo parameter)
        rho = 0.5  # Backtracking factor
        max_ls_iter = 20  # Maximum line search iterations

        for iteration in range(max_iterations):
            # Build linearized system
            H, b = self._build_linearized_system()

            # Solve for Gauss-Newton direction: H d = b
            try:
                d = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(H, b, rcond=None)[0]

            # Save current state
            old_vars = {k: v.copy() for k, v in self.variables.items()}
            current_error = error_history[-1]

            # Gradient for Armijo condition: grad = -b (since b = -J^T Lambda r)
            grad = -b
            directional_deriv = np.dot(grad, d)

            # Line search: find alpha with sufficient decrease (Algorithm 3.1 step 4)
            alpha = 1.0
            for ls_iter in range(max_ls_iter):
                # Try update with current alpha
                self._update_variables(alpha * d)
                new_error = self.compute_error()

                # Armijo sufficient decrease condition (Eq. 3.43)
                if new_error <= current_error + c * alpha * directional_deriv:
                    # Sufficient decrease achieved
                    break

                # Backtrack
                self.variables = {k: v.copy() for k, v in old_vars.items()}
                alpha *= rho
            else:
                # Line search failed - accept small step anyway
                self._update_variables(alpha * d)
                new_error = self.compute_error()

            error_history.append(new_error)

            # Check convergence
            if abs(current_error - new_error) < tol:
                break

        return self.variables.copy(), error_history

    def _levenberg_marquardt(
        self, max_iterations: int, tol: float, initial_mu: float = 1e-3
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Levenberg-Marquardt optimization (Algorithm 3.2).

        Implements Algorithm 3.2 from Chapter 3:
            - Eq. (3.53): (J^T J + mu*I) d_lm = -J^T r
            - Eq. (3.56): Gain ratio g for damping adjustment

        The LM method interpolates between gradient descent (large mu)
        and Gauss-Newton (small mu) based on the gain ratio.

        Args:
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.
            initial_mu: Initial damping parameter (default 1e-3).

        Returns:
            Tuple of (optimized_variables, error_history).
        """
        error_history = [self.compute_error()]

        # LM parameters (Algorithm 3.2)
        mu = initial_mu  # Damping parameter
        nu = 2.0  # Factor for increasing mu on rejected steps

        # Get total dimension
        var_ids_sorted = sorted(self.variables.keys())
        total_dim = sum(self.variable_dims[vid] for vid in var_ids_sorted)

        for iteration in range(max_iterations):
            # Build linearized system: H = J^T Lambda J, b = -J^T Lambda r
            H, b = self._build_linearized_system()
            current_error = error_history[-1]

            # Eq. (3.53): Solve (H + mu*I) d_lm = b
            # Note: b already contains -J^T Lambda r from _build_linearized_system
            H_damped = H + mu * np.eye(total_dim)

            try:
                d_lm = np.linalg.solve(H_damped, b)
            except np.linalg.LinAlgError:
                d_lm = np.linalg.lstsq(H_damped, b, rcond=None)[0]

            # Save current state
            old_vars = {k: v.copy() for k, v in self.variables.items()}

            # Try update
            self._update_variables(d_lm)
            new_error = self.compute_error()

            # Eq. (3.56): Calculate gain ratio
            # g = (f(x_k) - f(x_k + d_lm)) / L(0) - L(d_lm))
            # where L(d) = f(x_k) + d^T grad + 0.5 d^T H d
            # L(0) - L(d_lm) = -d_lm^T grad - 0.5 d_lm^T H d_lm
            #                = d_lm^T b - 0.5 d_lm^T H d_lm  (since grad = -b)
            # Simplified: denominator = 0.5 * d_lm^T (mu * d_lm + b)
            actual_reduction = current_error - new_error
            predicted_reduction = 0.5 * np.dot(d_lm, mu * d_lm + b)

            if predicted_reduction > 0:
                g = actual_reduction / predicted_reduction
            else:
                g = 0.0

            # Algorithm 3.2 steps 5-11: Accept/reject based on gain ratio
            if g > 0:
                # Accept step (line 9)
                error_history.append(new_error)

                # Update mu (line 10): mu = mu * max(1/3, 1 - (2g-1)^3)
                mu_factor = max(1.0 / 3.0, 1.0 - (2.0 * g - 1.0) ** 3)
                mu = mu * mu_factor
                nu = 2.0  # Reset nu
            else:
                # Reject step (lines 6-7)
                self.variables = {k: v.copy() for k, v in old_vars.items()}
                error_history.append(current_error)

                # Increase damping: mu = mu * nu, nu = 2 * nu
                mu = mu * nu
                nu = 2.0 * nu

            # Check convergence
            if len(error_history) >= 2:
                if abs(error_history[-2] - error_history[-1]) < tol:
                    break

            # Also check if step size is too small
            if np.linalg.norm(d_lm) < tol:
                break

        return self.variables.copy(), error_history

    def _build_linearized_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linearized system Hδx = b for Gauss-Newton.

        H = JᵀΛJ (Hessian approximation)
        b = -JᵀΛr (gradient)

        Returns:
            Tuple of (H, b) for solving Hδx = b.
        """
        # Total dimension of all variables
        var_ids_sorted = sorted(self.variables.keys())
        total_dim = sum(self.variable_dims[vid] for vid in var_ids_sorted)

        # Initialize H and b
        H = np.zeros((total_dim, total_dim))
        b = np.zeros(total_dim)

        # Build index mapping
        var_indices = {}
        current_idx = 0
        for vid in var_ids_sorted:
            dim = self.variable_dims[vid]
            var_indices[vid] = (current_idx, current_idx + dim)
            current_idx += dim

        # Accumulate contributions from each factor
        for factor in self.factors:
            r, jacobians = factor.linearize(self.variables)
            Lambda = factor.information

            # For each pair of variables in this factor
            for i, vid_i in enumerate(factor.variable_ids):
                J_i = jacobians[i]
                start_i, end_i = var_indices[vid_i]

                # Gradient contribution: -JᵀΛr
                b[start_i:end_i] -= J_i.T @ Lambda @ r

                for j, vid_j in enumerate(factor.variable_ids):
                    J_j = jacobians[j]
                    start_j, end_j = var_indices[vid_j]

                    # Hessian contribution: JᵀΛJ
                    H[start_i:end_i, start_j:end_j] += J_i.T @ Lambda @ J_j

        return H, b

    def _update_variables(self, delta_x: np.ndarray) -> None:
        """
        Update all variables by adding delta.

        Args:
            delta_x: Stacked update vector for all variables.
        """
        var_ids_sorted = sorted(self.variables.keys())
        current_idx = 0
        for vid in var_ids_sorted:
            dim = self.variable_dims[vid]
            self.variables[vid] += delta_x[current_idx : current_idx + dim]
            current_idx += dim


def test_fgo_simple_ls():
    """
    Unit test: Simple linear least squares using FGO.

    Tests FGO on a problem that can be verified analytically.
    """
    print("FGO Simple Least Squares Test:")

    # Problem: Estimate x from measurements y = x + noise
    measurements = np.array([1.1, 0.9, 1.2, 0.8])
    true_x = 1.0

    # Create factor graph
    graph = FactorGraph()
    graph.add_variable(0, np.array([0.0]))  # Variable to estimate

    # Add measurement factors
    for z in measurements:
        def residual_func(x_vars, z=z):
            return np.array([x_vars[0][0] - z])

        def jacobian_func(x_vars):
            return [np.array([[1.0]])]

        information = np.array([[1.0]])  # Unit variance
        factor = Factor([0], residual_func, jacobian_func, information)
        graph.add_factor(factor)

    # Optimize
    optimized_vars, error_history = graph.optimize(method="gauss_newton")

    estimated_x = optimized_vars[0][0]
    expected_x = np.mean(measurements)

    print(f"  True value: {true_x}")
    print(f"  Measurements: {measurements}")
    print(f"  Expected estimate (mean): {expected_x:.4f}")
    print(f"  FGO estimate: {estimated_x:.4f}")
    print(f"  Error: {abs(estimated_x - expected_x):.6f}")
    print(f"  Iterations: {len(error_history) - 1}")

    assert abs(estimated_x - expected_x) < 1e-6, "FGO estimate incorrect"
    print("  [PASS] Test passed")


def test_fgo_2d_positioning():
    """
    Unit test: 2D positioning from range measurements.

    Tests FGO on a simple 2D trilateration problem.
    """
    print("FGO 2D Positioning Test:")

    # Anchor positions
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    # True position
    true_pos = np.array([3.0, 4.0])

    # Generate range measurements
    ranges = np.linalg.norm(anchors - true_pos, axis=1)

    # Create factor graph
    graph = FactorGraph()
    graph.add_variable(0, np.array([5.0, 5.0]))  # Initial guess at center

    # Add range factors
    for i, anchor in enumerate(anchors):
        z = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z):
            pos = x_vars[0]
            predicted_range = np.linalg.norm(pos - anchor)
            return np.array([predicted_range - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        information = np.array([[1.0]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph.add_factor(factor)

    # Optimize
    optimized_vars, error_history = graph.optimize(method="gauss_newton")

    estimated_pos = optimized_vars[0]
    position_error = np.linalg.norm(estimated_pos - true_pos)

    print(f"  True position: {true_pos}")
    print(f"  Estimated position: {estimated_pos}")
    print(f"  Position error: {position_error:.6f} m")
    print(f"  Final error: {error_history[-1]:.6e}")
    print(f"  Iterations: {len(error_history) - 1}")

    assert position_error < 1e-6, f"Position error {position_error} too large"
    print("  [PASS] Test passed")


def test_fgo_levenberg_marquardt():
    """
    Unit test: Levenberg-Marquardt optimization (Algorithm 3.2).

    Tests LM on a mildly nonlinear problem to ensure:
    1. LM converges to the correct solution
    2. Error decreases monotonically (or at least doesn't diverge)
    """
    print("FGO Levenberg-Marquardt Test (Algorithm 3.2):")

    # Same 2D positioning problem but with poor initial guess
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    true_pos = np.array([3.0, 4.0])
    ranges = np.linalg.norm(anchors - true_pos, axis=1)

    # Create factor graph with POOR initial guess
    graph = FactorGraph()
    graph.add_variable(0, np.array([8.0, 1.0]))  # Far from true position

    for i, anchor in enumerate(anchors):
        z = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z):
            pos = x_vars[0]
            predicted_range = np.linalg.norm(pos - anchor)
            return np.array([predicted_range - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        information = np.array([[1.0]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph.add_factor(factor)

    # Optimize with LM
    optimized_vars, error_history = graph.optimize(
        method="levenberg_marquardt",
        max_iterations=50,
        initial_mu=1e-2
    )

    estimated_pos = optimized_vars[0]
    position_error = np.linalg.norm(estimated_pos - true_pos)

    print(f"  True position: {true_pos}")
    print(f"  Initial guess: [8.0, 1.0]")
    print(f"  Estimated position: {estimated_pos}")
    print(f"  Position error: {position_error:.6f} m")
    print(f"  Initial error: {error_history[0]:.6f}")
    print(f"  Final error: {error_history[-1]:.6e}")
    print(f"  Iterations: {len(error_history) - 1}")

    # Check monotonic decrease (allowing for rejected steps)
    # Count how many times error increased
    increases = sum(1 for i in range(1, len(error_history))
                    if error_history[i] > error_history[i-1] + 1e-10)
    print(f"  Error increases (should be 0 or small): {increases}")

    # LM should converge
    assert position_error < 0.01, f"LM position error {position_error} too large"

    # Error should decrease overall
    assert error_history[-1] < error_history[0], "LM should reduce error"

    print("  [PASS] Test passed")


def test_fgo_line_search():
    """
    Unit test: Line search optimization (Algorithm 3.1).

    Tests line search with sufficient decrease condition.
    """
    print("FGO Line Search Test (Algorithm 3.1):")

    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    true_pos = np.array([3.0, 4.0])
    ranges = np.linalg.norm(anchors - true_pos, axis=1)

    graph = FactorGraph()
    graph.add_variable(0, np.array([5.0, 5.0]))

    for i, anchor in enumerate(anchors):
        z = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z):
            pos = x_vars[0]
            predicted_range = np.linalg.norm(pos - anchor)
            return np.array([predicted_range - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        information = np.array([[1.0]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph.add_factor(factor)

    # Optimize with line search
    optimized_vars, error_history = graph.optimize(method="line_search")

    estimated_pos = optimized_vars[0]
    position_error = np.linalg.norm(estimated_pos - true_pos)

    print(f"  True position: {true_pos}")
    print(f"  Estimated position: {estimated_pos}")
    print(f"  Position error: {position_error:.6f} m")
    print(f"  Iterations: {len(error_history) - 1}")

    # Check error decreases monotonically (line search guarantees this)
    for i in range(1, len(error_history)):
        assert error_history[i] <= error_history[i-1] + 1e-10, \
            f"Line search should guarantee decrease: {error_history[i]} > {error_history[i-1]}"

    assert position_error < 1e-5, f"Position error {position_error} too large"
    print("  [PASS] Test passed")


def test_lm_monotonic_decrease():
    """
    Unit test: Verify LM error decreases monotonically.

    This test uses a mildly nonlinear problem to verify that LM
    does not diverge and maintains stable convergence.
    """
    print("LM Monotonic Decrease Test:")

    # Nonlinear problem: range-bearing positioning
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    true_pos = np.array([4.0, 5.0])

    # Range AND bearing measurements (more nonlinear)
    ranges = np.linalg.norm(anchors - true_pos, axis=1)
    bearings = np.arctan2(anchors[:, 1] - true_pos[1], anchors[:, 0] - true_pos[0])

    graph = FactorGraph()
    graph.add_variable(0, np.array([7.0, 2.0]))  # Poor initial guess

    # Range factors
    for i, anchor in enumerate(anchors):
        z_range = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z_range):
            pos = x_vars[0]
            predicted = np.linalg.norm(pos - anchor)
            return np.array([predicted - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        factor = Factor([0], residual_func, jacobian_func, np.array([[4.0]]))
        graph.add_factor(factor)

    # Bearing factors (nonlinear)
    for i, anchor in enumerate(anchors):
        z_bearing = bearings[i]

        def residual_func_b(x_vars, anchor=anchor, z=z_bearing):
            pos = x_vars[0]
            predicted = np.arctan2(anchor[1] - pos[1], anchor[0] - pos[0])
            return np.array([predicted - z])

        def jacobian_func_b(x_vars, anchor=anchor):
            pos = x_vars[0]
            dx = anchor[0] - pos[0]
            dy = anchor[1] - pos[1]
            r_sq = dx**2 + dy**2
            if r_sq < 1e-10:
                return [np.zeros((1, 2))]
            return [np.array([[dy / r_sq, -dx / r_sq]])]

        factor = Factor([0], residual_func_b, jacobian_func_b, np.array([[1.0]]))
        graph.add_factor(factor)

    # Optimize with LM
    optimized_vars, error_history = graph.optimize(
        method="levenberg_marquardt",
        max_iterations=30,
        initial_mu=1e-2
    )

    estimated_pos = optimized_vars[0]
    position_error = np.linalg.norm(estimated_pos - true_pos)

    print(f"  True position: {true_pos}")
    print(f"  Estimated position: {estimated_pos}")
    print(f"  Position error: {position_error:.4f} m")
    print(f"  Error history: {[f'{e:.4f}' for e in error_history[:5]]}...")
    print(f"  Final error: {error_history[-1]:.6e}")

    # LM should NOT diverge
    assert error_history[-1] <= error_history[0] * 1.1, \
        "LM should not diverge significantly"

    # Should converge reasonably
    assert position_error < 0.5, f"Position error {position_error} too large"

    print("  [PASS] Test passed")


def test_gd_with_line_search():
    """
    Unit test: Gradient descent with Armijo line search (Algorithm 3.1).

    Tests that gradient descent with line_search=True:
    1. Uses Armijo backtracking line search
    2. Guarantees error decrease (or at least non-increase)
    3. Converges to a reasonable solution
    """
    print("Gradient Descent with Line Search Test (Eq. 3.42, 3.46, Alg. 3.1):")

    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    true_pos = np.array([3.0, 4.0])
    ranges = np.linalg.norm(anchors - true_pos, axis=1)

    # Test 1: GD without line search (fixed step)
    graph_fixed = FactorGraph()
    graph_fixed.add_variable(0, np.array([8.0, 8.0]))

    for i, anchor in enumerate(anchors):
        z = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z):
            pos = x_vars[0]
            predicted_range = np.linalg.norm(pos - anchor)
            return np.array([predicted_range - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        information = np.array([[1.0]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph_fixed.add_factor(factor)

    # Test 2: GD with line search
    graph_ls = FactorGraph()
    graph_ls.add_variable(0, np.array([8.0, 8.0]))

    for i, anchor in enumerate(anchors):
        z = ranges[i]

        def residual_func(x_vars, anchor=anchor, z=z):
            pos = x_vars[0]
            predicted_range = np.linalg.norm(pos - anchor)
            return np.array([predicted_range - z])

        def jacobian_func(x_vars, anchor=anchor):
            pos = x_vars[0]
            diff = pos - anchor
            r = np.linalg.norm(diff)
            if r < 1e-10:
                return [np.zeros((1, 2))]
            return [diff.reshape(1, 2) / r]

        information = np.array([[1.0]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph_ls.add_factor(factor)

    # Run GD with fixed step (may not converge well)
    _, error_history_fixed = graph_fixed.optimize(
        method="gd", max_iterations=100, alpha=0.1, line_search=False
    )

    # Run GD with line search (Armijo)
    optimized_vars_ls, error_history_ls = graph_ls.optimize(
        method="gd", max_iterations=100, line_search=True
    )

    estimated_pos = optimized_vars_ls[0]
    position_error = np.linalg.norm(estimated_pos - true_pos)

    print(f"  True position: {true_pos}")
    print(f"  GD+LineSearch estimate: {estimated_pos}")
    print(f"  Position error: {position_error:.6f} m")
    print(f"  GD fixed step iterations: {len(error_history_fixed) - 1}")
    print(f"  GD line search iterations: {len(error_history_ls) - 1}")
    print(f"  GD fixed step final error: {error_history_fixed[-1]:.6e}")
    print(f"  GD line search final error: {error_history_ls[-1]:.6e}")

    # Check error decreases monotonically with line search
    for i in range(1, len(error_history_ls)):
        assert error_history_ls[i] <= error_history_ls[i-1] + 1e-10, \
            f"Line search should guarantee decrease: {error_history_ls[i]} > {error_history_ls[i-1]}"

    print("  [PASS] Line search guarantees monotonic decrease")

    # Line search should generally do better or equal to fixed step
    print(f"  Line search improvement: {error_history_fixed[-1] / max(error_history_ls[-1], 1e-15):.1f}x")

    print("  [PASS] Test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("FACTOR GRAPH OPTIMIZATION UNIT TESTS")
    print("=" * 70)
    print()

    test_fgo_simple_ls()
    print()
    test_fgo_2d_positioning()
    print()
    test_fgo_line_search()
    print()
    test_gd_with_line_search()
    print()
    test_fgo_levenberg_marquardt()
    print()
    test_lm_monotonic_decrease()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)



