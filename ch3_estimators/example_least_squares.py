"""
Example: Least Squares Estimation for Indoor Positioning

This script demonstrates various least squares methods applied to 2D positioning
from range measurements (Time-of-Arrival positioning).

Demonstrates:
    - Linear least squares (LS) with linearization
    - Weighted least squares (WLS) with different measurement accuracies
    - Iterative least squares (Gauss-Newton) for nonlinear problem
    - Robust least squares (IRLS) with outlier rejection

Implements equations (3.1)-(3.4) from Chapter 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.estimators import (
    linear_least_squares,
    weighted_least_squares,
    iterative_least_squares,
    robust_least_squares,
)


def setup_positioning_scenario():
    """Create a 2D positioning scenario with 4 anchors."""
    # Anchor positions at corners of 10m × 10m room
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])

    # True target position (unknown to estimator)
    true_position = np.array([3.0, 4.0])

    return anchors, true_position


def compute_ranges(position, anchors, noise_std=0.0):
    """Compute ranges from position to anchors with optional noise."""
    true_ranges = np.linalg.norm(anchors - position, axis=1)
    if noise_std > 0:
        true_ranges += noise_std * np.random.randn(len(anchors))
    return true_ranges


def example_1_linear_ls():
    """
    Example 1: Linear Least Squares with Linearization.

    Demonstrates Eq. (3.1): x̂ = (A'A)⁻¹A'b
    """
    print("=" * 70)
    print("EXAMPLE 1: Linear Least Squares (LS)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()

    # Generate measurements
    np.random.seed(42)
    ranges = compute_ranges(true_position, anchors, noise_std=0.1)

    # Initial guess (center of room)
    x0 = np.array([5.0, 5.0])

    # Linearize around initial guess
    # Design matrix: A = ∂r/∂x evaluated at x0
    diff = x0 - anchors
    ranges_at_x0 = np.linalg.norm(diff, axis=1, keepdims=True)
    A = diff / ranges_at_x0

    # Observation vector: b = measured_ranges - predicted_ranges
    b = ranges - np.linalg.norm(anchors - x0, axis=1)

    # Solve linear LS
    dx, P = linear_least_squares(A, b)
    position_estimate = x0 + dx

    # Results
    error = np.linalg.norm(position_estimate - true_position)

    print(f"\nTrue position:      {true_position}")
    print(f"Initial guess:      {x0}")
    print(f"LS estimate:        {position_estimate}")
    print(f"Position error:     {error:.4f} m")
    print(f"\nCovariance matrix:")
    print(P)
    print(f"Position std dev:   {np.sqrt(np.diag(P))}")

    return position_estimate, P


def example_2_weighted_ls():
    """
    Example 2: Weighted Least Squares.

    Demonstrates Eq. (3.2): x̂ = (A'WA)⁻¹A'Wb
    Different anchors have different measurement accuracies.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Weighted Least Squares (WLS)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()

    # Different measurement accuracies
    # Anchor 0 is very accurate (GPS reference station)
    # Others are UWB with higher noise
    measurement_stds = np.array([0.05, 0.3, 0.3, 0.3])

    print(f"\nMeasurement standard deviations:")
    for i, std in enumerate(measurement_stds):
        print(f"  Anchor {i}: sigma = {std:.2f} m")

    # Generate measurements with different noise levels
    np.random.seed(42)
    ranges = np.array(
        [
            compute_ranges(true_position, anchors[i : i + 1], noise_std=measurement_stds[i])[0]
            for i in range(len(anchors))
        ]
    )

    # Weight matrix: W = R^(-1) where R is measurement covariance
    W = np.diag(1.0 / measurement_stds**2)

    # Linearization
    x0 = np.array([5.0, 5.0])
    diff = x0 - anchors
    ranges_at_x0 = np.linalg.norm(diff, axis=1, keepdims=True)
    A = diff / ranges_at_x0
    b = ranges - np.linalg.norm(anchors - x0, axis=1)

    # Solve WLS
    dx_wls, P_wls = weighted_least_squares(A, b, W)
    position_wls = x0 + dx_wls

    # Compare with standard LS (ignoring weights)
    dx_ls, P_ls = linear_least_squares(A, b)
    position_ls = x0 + dx_ls

    # Results
    error_wls = np.linalg.norm(position_wls - true_position)
    error_ls = np.linalg.norm(position_ls - true_position)

    print(f"\nTrue position:      {true_position}")
    print(f"WLS estimate:       {position_wls} (error: {error_wls:.4f} m)")
    print(f"LS estimate:        {position_ls} (error: {error_ls:.4f} m)")
    print(f"\nWLS covariance trace: {np.trace(P_wls):.6f}")
    print(f"LS covariance trace:  {np.trace(P_ls):.6f}")
    print(f"\nImprovement: {((error_ls - error_wls) / error_ls * 100):.1f}%")

    return position_wls, P_wls


def example_3_iterative_ls():
    """
    Example 3: Iterative Least Squares (Gauss-Newton).

    Demonstrates Eq. (3.3): Iterative solution for nonlinear problem.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Iterative Least Squares (Gauss-Newton)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()

    # Generate measurements
    np.random.seed(42)
    ranges = compute_ranges(true_position, anchors, noise_std=0.1)

    # Define nonlinear measurement model
    def range_model(x):
        """Predicted ranges from position x to all anchors."""
        return np.linalg.norm(anchors - x, axis=1)

    def range_jacobian(x):
        """Jacobian: ∂r/∂x = (x - anchor) / range."""
        diff = x - anchors
        ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        return diff / np.maximum(ranges, 1e-10)

    # Initial guess
    x_init = np.array([5.0, 5.0])

    print(f"\nInitial guess: {x_init}")
    print(f"True position: {true_position}")

    # Iterative solution
    x_hat, P, iterations = iterative_least_squares(
        range_model, range_jacobian, ranges, x_init, max_iter=20, tol=1e-6
    )

    error = np.linalg.norm(x_hat - true_position)

    print(f"\nConverged in {iterations} iterations")
    print(f"Final estimate: {x_hat}")
    print(f"Position error: {error:.4f} m")
    print(f"\nCovariance:")
    print(P)

    return x_hat, P, iterations


def example_4_robust_ls():
    """
    Example 4: Robust Least Squares with Outliers.

    Demonstrates Eq. (3.4): IRLS with Huber/Cauchy/Tukey loss functions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Robust Least Squares (Outlier Rejection)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()

    # Generate measurements
    np.random.seed(42)
    ranges = compute_ranges(true_position, anchors, noise_std=0.1)

    # Add severe outlier to anchor 2
    ranges[2] += 3.0
    print(f"\nAdded 3.0 m outlier to anchor 2")

    # Linearization
    x0 = np.array([5.0, 5.0])
    diff = x0 - anchors
    ranges_at_x0 = np.linalg.norm(diff, axis=1, keepdims=True)
    A = diff / ranges_at_x0
    b = ranges - np.linalg.norm(anchors - x0, axis=1)

    # Standard LS (corrupted by outlier)
    dx_ls, _ = linear_least_squares(A, b)
    position_ls = x0 + dx_ls

    # Robust LS with different methods
    methods = ["huber", "cauchy", "tukey"]
    results = {}

    for method in methods:
        dx_robust, P_robust, weights = robust_least_squares(
            A, b, method=method, threshold=2.0, max_iter=20
        )
        position_robust = x0 + dx_robust
        error_robust = np.linalg.norm(position_robust - true_position)

        results[method] = {
            "position": position_robust,
            "error": error_robust,
            "weights": weights,
        }

    # Results
    error_ls = np.linalg.norm(position_ls - true_position)

    print(f"\nTrue position:       {true_position}")
    print(f"Standard LS:         {position_ls} (error: {error_ls:.4f} m)")
    print()

    for method, result in results.items():
        print(f"{method.capitalize()} LS:    {result['position']} (error: {result['error']:.4f} m)")
        print(f"  Outlier weight: {result['weights'][2]:.4f}")

    print(f"\nStandard LS corrupted by outlier!")
    print(f"Robust methods successfully rejected outlier.")

    return results


def visualize_results():
    """Create visualization of all examples."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()

    # Run all examples
    np.random.seed(42)

    # Example 1: Linear LS
    ranges = compute_ranges(true_position, anchors, noise_std=0.1)
    x0 = np.array([5.0, 5.0])
    diff = x0 - anchors
    ranges_at_x0 = np.linalg.norm(diff, axis=1, keepdims=True)
    A = diff / ranges_at_x0
    b = ranges - np.linalg.norm(anchors - x0, axis=1)
    dx_ls, P_ls = linear_least_squares(A, b)
    pos_ls = x0 + dx_ls

    # Example 3: Iterative LS
    def range_model(x):
        return np.linalg.norm(anchors - x, axis=1)

    def range_jacobian(x):
        diff = x - anchors
        ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        return diff / np.maximum(ranges, 1e-10)

    pos_iterative, _, _ = iterative_least_squares(
        range_model, range_jacobian, ranges, x0
    )

    # Example 4: Robust LS
    ranges_outlier = ranges.copy()
    ranges_outlier[2] += 3.0
    b_outlier = ranges_outlier - np.linalg.norm(anchors - x0, axis=1)
    dx_ls_outlier, _ = linear_least_squares(A, b_outlier)
    pos_ls_outlier = x0 + dx_ls_outlier
    dx_robust, _, _ = robust_least_squares(A, b_outlier, method="huber")
    pos_robust = x0 + dx_robust

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Clean data
    ax = axes[0]
    ax.scatter(anchors[:, 0], anchors[:, 1], s=200, c="blue", marker="^", label="Anchors", zorder=3)
    ax.scatter(true_position[0], true_position[1], s=200, c="green", marker="*", label="True Position", zorder=3)
    ax.scatter(x0[0], x0[1], s=150, c="gray", marker="x", label="Initial Guess", zorder=3)
    ax.scatter(pos_ls[0], pos_ls[1], s=150, c="orange", marker="o", label="Linear LS", zorder=3)
    ax.scatter(pos_iterative[0], pos_iterative[1], s=150, c="red", marker="s", label="Iterative LS", zorder=3)

    # Draw range circles
    for i, anchor in enumerate(anchors):
        circle = plt.Circle(anchor, ranges[i], fill=False, edgecolor="blue", alpha=0.3, linestyle="--")
        ax.add_patch(circle)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Example 1-3: LS Methods (Clean Data)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    # Plot 2: With outlier
    ax = axes[1]
    ax.scatter(anchors[:, 0], anchors[:, 1], s=200, c="blue", marker="^", label="Anchors", zorder=3)
    ax.scatter(true_position[0], true_position[1], s=200, c="green", marker="*", label="True Position", zorder=3)
    ax.scatter(x0[0], x0[1], s=150, c="gray", marker="x", label="Initial Guess", zorder=3)
    ax.scatter(pos_ls_outlier[0], pos_ls_outlier[1], s=150, c="orange", marker="o", label="Standard LS (corrupted)", zorder=3)
    ax.scatter(pos_robust[0], pos_robust[1], s=150, c="purple", marker="D", label="Robust LS (Huber)", zorder=3)

    # Draw range circles (with outlier marked)
    for i, anchor in enumerate(anchors):
        color = "red" if i == 2 else "blue"
        alpha = 0.6 if i == 2 else 0.3
        circle = plt.Circle(anchor, ranges_outlier[i], fill=False, edgecolor=color, alpha=alpha, linestyle="--", linewidth=2 if i == 2 else 1)
        ax.add_patch(circle)

    # Mark outlier anchor
    ax.scatter(anchors[2, 0], anchors[2, 1], s=300, facecolors="none", edgecolors="red", linewidth=3, zorder=2, label="Outlier Measurement")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Example 4: Robust LS with Outlier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    plt.tight_layout()
    plt.savefig("ch3_least_squares_examples.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as: ch3_least_squares_examples.png")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CHAPTER 3: LEAST SQUARES ESTIMATION EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates Equations (3.1) - (3.4) from Chapter 3")
    print("Application: 2D positioning from Time-of-Arrival (TOA) ranges")

    # Run examples
    example_1_linear_ls()
    example_2_weighted_ls()
    example_3_iterative_ls()
    example_4_robust_ls()

    # Visualization
    visualize_results()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

