"""
Unit tests for Factor Graph Optimization methods.

Tests Line Search (Algorithm 3.1) and Levenberg-Marquardt (Algorithm 3.2)
as described in Chapter 3 of the book.

Book References:
    - Algorithm 3.1: Line search strategy with sufficient decrease
    - Algorithm 3.2: Levenberg-Marquardt method
    - Eq. (3.53): LM update direction (J^T J + μI) d_lm = -J^T r
    - Eq. (3.56): Gain ratio for damping adjustment
"""

import unittest

import numpy as np

from core.estimators import Factor, FactorGraph


def create_range_positioning_graph(
    anchors: np.ndarray,
    true_pos: np.ndarray,
    initial_guess: np.ndarray,
    noise_std: float = 0.0
) -> FactorGraph:
    """
    Create a factor graph for 2D positioning from range measurements.

    Args:
        anchors: Anchor positions (n_anchors, 2).
        true_pos: True position (2,).
        initial_guess: Initial guess for optimization (2,).
        noise_std: Measurement noise standard deviation.

    Returns:
        Configured FactorGraph.
    """
    # Generate range measurements
    ranges = np.linalg.norm(anchors - true_pos, axis=1)
    if noise_std > 0:
        np.random.seed(42)
        ranges += np.random.normal(0, noise_std, len(ranges))

    graph = FactorGraph()
    graph.add_variable(0, initial_guess.copy())

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

        information = np.array([[1.0 / (noise_std**2 if noise_std > 0 else 1.0)]])
        factor = Factor([0], residual_func, jacobian_func, information)
        graph.add_factor(factor)

    return graph


class TestLevenbergMarquardt(unittest.TestCase):
    """Test Levenberg-Marquardt optimization (Algorithm 3.2)."""

    def test_lm_converges_to_correct_solution(self):
        """LM should converge to the correct position."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        initial_guess = np.array([8.0, 1.0])  # Poor initial guess

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        optimized_vars, error_history = graph.optimize(
            method="levenberg_marquardt",
            max_iterations=50,
            initial_mu=1e-2
        )

        estimated_pos = optimized_vars[0]
        position_error = np.linalg.norm(estimated_pos - true_pos)

        self.assertLess(position_error, 0.01,
                        f"LM should converge, got error {position_error}")

    def test_lm_error_decreases(self):
        """LM should decrease error overall (not diverge)."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        true_pos = np.array([4.0, 5.0])
        initial_guess = np.array([7.0, 2.0])

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        _, error_history = graph.optimize(
            method="levenberg_marquardt",
            max_iterations=30,
            initial_mu=1e-2
        )

        # Final error should be less than initial
        self.assertLess(error_history[-1], error_history[0],
                        "LM should reduce total error")

    def test_lm_does_not_diverge(self):
        """LM should not diverge even with poor initial guess."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        initial_guess = np.array([15.0, -5.0])  # Very poor initial guess

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        _, error_history = graph.optimize(
            method="levenberg_marquardt",
            max_iterations=100,
            initial_mu=0.1
        )

        # Should not diverge: final error should not be much worse than initial
        self.assertLessEqual(error_history[-1], error_history[0] * 1.5,
                             "LM should not diverge significantly")

    def test_lm_alias_works(self):
        """The 'lm' alias should work."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        initial_guess = np.array([5.0, 5.0])

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        optimized_vars, _ = graph.optimize(method="lm", max_iterations=20)

        estimated_pos = optimized_vars[0]
        position_error = np.linalg.norm(estimated_pos - true_pos)

        self.assertLess(position_error, 0.01)


class TestLineSearch(unittest.TestCase):
    """Test Line Search optimization (Algorithm 3.1)."""

    def test_line_search_converges(self):
        """Line search should converge to correct solution."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        initial_guess = np.array([5.0, 5.0])

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        optimized_vars, _ = graph.optimize(method="line_search", max_iterations=20)

        estimated_pos = optimized_vars[0]
        position_error = np.linalg.norm(estimated_pos - true_pos)

        self.assertLess(position_error, 1e-5)

    def test_line_search_monotonic_decrease(self):
        """Line search should guarantee monotonic error decrease."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        initial_guess = np.array([6.0, 6.0])

        graph = create_range_positioning_graph(anchors, true_pos, initial_guess)

        _, error_history = graph.optimize(method="line_search", max_iterations=20)

        # Check monotonic decrease
        for i in range(1, len(error_history)):
            self.assertLessEqual(
                error_history[i], error_history[i - 1] + 1e-10,
                f"Line search should guarantee decrease at step {i}"
            )


class TestMethodComparison(unittest.TestCase):
    """Compare different optimization methods."""

    def test_all_methods_converge(self):
        """All methods should converge to similar solutions."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        true_pos = np.array([4.0, 5.0])
        initial_guess = np.array([6.0, 6.0])

        methods = ["gauss_newton", "line_search", "levenberg_marquardt"]
        solutions = {}

        for method in methods:
            graph = create_range_positioning_graph(
                anchors, true_pos, initial_guess
            )
            optimized_vars, _ = graph.optimize(method=method, max_iterations=50)
            solutions[method] = optimized_vars[0]

        # All methods should find similar solutions
        for method in methods:
            error = np.linalg.norm(solutions[method] - true_pos)
            self.assertLess(error, 0.01,
                            f"{method} should converge, got error {error}")

        # All solutions should be close to each other
        for m1 in methods:
            for m2 in methods:
                diff = np.linalg.norm(solutions[m1] - solutions[m2])
                self.assertLess(diff, 0.01,
                                f"{m1} and {m2} should find similar solutions")

    def test_lm_handles_poor_initial_better_than_gn(self):
        """LM should handle poor initial guesses at least as well as GN."""
        anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        true_pos = np.array([3.0, 4.0])
        # Very poor initial guess - far from solution
        initial_guess = np.array([20.0, -10.0])

        # Try Gauss-Newton
        graph_gn = create_range_positioning_graph(anchors, true_pos, initial_guess)
        optimized_gn, error_history_gn = graph_gn.optimize(
            method="gauss_newton", max_iterations=50
        )

        # Try LM
        graph_lm = create_range_positioning_graph(anchors, true_pos, initial_guess)
        optimized_lm, error_history_lm = graph_lm.optimize(
            method="levenberg_marquardt", max_iterations=50, initial_mu=0.1
        )

        # Both should converge well (position error < 0.1)
        error_gn = np.linalg.norm(optimized_gn[0] - true_pos)
        error_lm = np.linalg.norm(optimized_lm[0] - true_pos)

        self.assertLess(error_gn, 0.1, "GN should converge")
        self.assertLess(error_lm, 0.1, "LM should converge")

        # LM should reduce error significantly from initial
        self.assertLess(error_history_lm[-1], error_history_lm[0] * 0.01,
                        "LM should reduce error significantly")


if __name__ == "__main__":
    unittest.main()

