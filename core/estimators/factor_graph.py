"""
Factor Graph Optimization for batch state estimation.

This module implements Factor Graph Optimization (FGO) as described in Chapter 3
of Principles of Indoor Positioning and Indoor Navigation.

Implements:
    - Eq. (3.35): MAP estimation X̂_MAP = argmax_X p(X | Z)
    - Eq. (3.36)-(3.37): Simplified MAP form with likelihood and prior
    - Eq. (3.38): Conversion to sum of squared residuals
    - Eq. (3.42)-(3.43): Gradient descent and Gauss-Newton updates
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
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Optimize the factor graph to find MAP estimate.

        Implements:
        - Eq. (3.35): MAP estimation
        - Eq. (3.42): Gradient descent (if method="gd")
        - Gauss-Newton method (if method="gauss_newton")

        Args:
            method: Optimization method ("gauss_newton" or "gd").
            max_iterations: Maximum number of iterations.
            tol: Convergence tolerance on error change.

        Returns:
            Tuple of (optimized_variables, error_history).

        Raises:
            ValueError: If method is not supported.
        """
        if method == "gauss_newton":
            return self._gauss_newton(max_iterations, tol)
        elif method == "gd":
            return self._gradient_descent(max_iterations, tol)
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
        self, max_iterations: int, tol: float, alpha: float = 0.01
    ) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """
        Gradient descent optimization.

        Implements Eq. (3.42): x_{k+1} = x_k + α d

        Args:
            max_iterations: Maximum iterations.
            tol: Convergence tolerance.
            alpha: Step size.

        Returns:
            Tuple of (optimized_variables, error_history).
        """
        error_history = [self.compute_error()]

        for iteration in range(max_iterations):
            # Compute gradient
            _, gradient = self._build_linearized_system()

            # Eq. (3.42): Update with gradient descent
            delta_x = alpha * gradient

            # Update variables
            self._update_variables(delta_x)

            # Compute new error
            current_error = self.compute_error()
            error_history.append(current_error)

            # Check convergence
            if abs(error_history[-2] - error_history[-1]) < tol:
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


if __name__ == "__main__":
    print("=" * 70)
    print("FACTOR GRAPH OPTIMIZATION UNIT TESTS")
    print("=" * 70)
    print()

    test_fgo_simple_ls()
    print()
    test_fgo_2d_positioning()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


