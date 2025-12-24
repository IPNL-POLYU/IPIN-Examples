"""
Example: Least Squares Estimation for Indoor Positioning.

This script demonstrates various least squares methods applied to 2D positioning
from range measurements (Time-of-Arrival positioning), following Chapter 3.

Run from repository root:
    python ch3_estimators/example_least_squares.py

Demonstrates:
    - Linear least squares (LS) with linearization
    - Weighted least squares (WLS) with w_i = 1/sigma_i^2
    - Gauss-Newton iterative LS for nonlinear problems
    - Levenberg-Marquardt for robust convergence
    - Robust least squares (IRLS) with Table 3.1 loss functions

Book Reference (Chapter 3):
    Section 3.1 - Least Squares Estimation:
        - Eq. (3.1): Cost function J(x) = sum_i (y_i - h_i(x))^2
        - Eq. (3.2): Normal equations (H^T H) x = H^T y
        - Eq. (3.3): Closed-form solution x = (H^T H)^{-1} H^T y
        - Eq. (3.4): First-order optimality condition dJ/dx = 0 (stationarity)
        - Table 3.1: Robust estimators (L2, Cauchy, Huber, Geman-McClure)

    Section 3.4.1 - Numerical Optimization:
        - Eq. (3.42)-(3.43): Line search x_{k+1} = x_k + alpha*d
        - Eq. (3.51)-(3.52): Gauss-Newton normal equations (J^T J) dx = J^T r
        - Eq. (3.53)-(3.56): Levenberg-Marquardt with damping mu
        - Algorithm 3.1: Line search strategy
        - Algorithm 3.2: Levenberg-Marquardt method

Measurement Model:
    h_i(x) = ||x - a_i||  (range from position x to anchor a_i)
    Residual: r_i(x) = y_i - h_i(x)  (observation minus prediction)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.estimators import (
    linear_least_squares,
    weighted_least_squares,
    gauss_newton,
    levenberg_marquardt,
    robust_gauss_newton,
)


def setup_positioning_scenario():
    """Create a 2D positioning scenario with 4 anchors.

    Returns:
        anchors: (4, 2) array of anchor positions at room corners.
        true_position: (2,) array of true target position.
    """
    # Anchor positions at corners of 10m × 10m room
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])

    # True target position (unknown to estimator)
    true_position = np.array([3.0, 4.0])

    return anchors, true_position


def create_range_model(anchors: np.ndarray):
    """Create measurement model and Jacobian functions for range positioning.

    This implements the book's formulation (Section 3.1):
        h_i(x) = ||x - a_i||  (nonlinear range model)
        J_i = ∂h_i/∂x = (x - a_i) / ||x - a_i||  (Jacobian for Eq. 3.51-3.52)

    Args:
        anchors: (m, 2) array of anchor positions.

    Returns:
        h: Measurement model function h(x) -> predicted ranges.
        jacobian: Jacobian function J(x) -> ∂h/∂x matrix.
    """

    def h(x: np.ndarray) -> np.ndarray:
        """Predicted ranges from position x to all anchors.

        h_i(x) = ||x - a_i||  (range measurement model)
        """
        return np.linalg.norm(anchors - x, axis=1)

    def jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of range model.

        J_i = ∂h_i/∂x = (x - a_i) / ||x - a_i||

        This is the direction vector from anchor to position, used in
        Gauss-Newton iteration (Section 3.4.1.2).
        """
        diff = x - anchors
        ranges = np.linalg.norm(diff, axis=1, keepdims=True)
        return diff / np.maximum(ranges, 1e-10)

    return h, jacobian


def compute_ranges(position: np.ndarray, anchors: np.ndarray,
                   noise_std: float = 0.0) -> np.ndarray:
    """Compute ranges from position to anchors with optional noise.

    Args:
        position: (2,) target position.
        anchors: (m, 2) anchor positions.
        noise_std: Standard deviation of range noise (meters).

    Returns:
        ranges: (m,) measured ranges.
    """
    true_ranges = np.linalg.norm(anchors - position, axis=1)
    if noise_std > 0:
        true_ranges += noise_std * np.random.randn(len(anchors))
    return true_ranges


def example_1_linear_ls():
    """
    Example 1: Linear Least Squares with Linearization.

    Demonstrates Eq. (3.2)-(3.3): Normal equations x̂ = (A'A)⁻¹A'b

    For nonlinear range model h(x), we linearize around initial guess x0:
        h(x) ~ h(x0) + J|_{x0} (x - x0)
    Leading to linear system: A dx = b
    where A = J|_{x0}, b = y - h(x0), dx = x - x0
    """
    print("=" * 70)
    print("EXAMPLE 1: Linear Least Squares (Eq. 3.2-3.3)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate noisy measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Initial guess (center of room)
    x0 = np.array([5.0, 5.0])

    # Linearize around initial guess
    # Residual: r = y - h(x₀) (observation minus prediction)
    # Jacobian: A = J|_{x₀}
    r = y - h(x0)
    A = jacobian(x0)

    # Solve linear LS: A'A dx = A'r (Eq. 3.2)
    dx, P = linear_least_squares(A, r)
    position_estimate = x0 + dx

    # Results
    error = np.linalg.norm(position_estimate - true_position)

    print(f"\nMeasurement model: h_i(x) = ||x - a_i|| (range to anchor)")
    print(f"Residual: r_i = y_i - h_i(x)  (book convention)")
    print(f"\nTrue position:      {true_position}")
    print(f"Initial guess:      {x0}")
    print(f"LS estimate:        {position_estimate}")
    print(f"Position error:     {error:.4f} m")
    print(f"\nCovariance matrix (Eq. 3.3):")
    print(P)
    print(f"Position std dev:   {np.sqrt(np.diag(P))}")

    return position_estimate, P


def example_2_weighted_ls():
    """
    Example 2: Weighted Least Squares.

    Demonstrates Section 3.1.1: w_i = 1/sigma_i^2 weighting.

    WLS minimizes: J(x) = sum w_i (y_i - h_i(x))^2
    Solution: x̂ = (A'WA)⁻¹A'Wb
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Weighted Least Squares (Section 3.1.1)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Different measurement accuracies
    # Anchor 0 is very accurate (GPS reference station)
    # Others are UWB with higher noise
    measurement_stds = np.array([0.05, 0.3, 0.3, 0.3])

    print(f"\nMeasurement standard deviations (sigma_i):")
    for i, std in enumerate(measurement_stds):
        print(f"  Anchor {i}: sigma = {std:.2f} m -> w = 1/sigma^2 = {1/std**2:.1f}")

    # Generate measurements with different noise levels
    np.random.seed(42)
    y = np.array([
        compute_ranges(true_position, anchors[i:i + 1], noise_std=measurement_stds[i])[0]
        for i in range(len(anchors))
    ])

    # Weight matrix: W = diag(1/sigma^2) (book Section 3.1.1)
    W = np.diag(1.0 / measurement_stds ** 2)

    # Linearization
    x0 = np.array([5.0, 5.0])
    r = y - h(x0)  # Residual: y - h(x₀)
    A = jacobian(x0)

    # Solve WLS: (A'WA) dx = A'W r
    dx_wls, P_wls = weighted_least_squares(A, r, W)
    position_wls = x0 + dx_wls

    # Compare with standard LS (ignoring weights)
    dx_ls, P_ls = linear_least_squares(A, r)
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


def example_3_gauss_newton():
    """
    Example 3: Gauss-Newton Iterative Least Squares.

    Demonstrates Section 3.4.1.2 Gauss-Newton method:
        - Eq. (3.4): First-order optimality condition dJ/dx = 0
        - Eq. (3.51): Linearized normal equations
        - Eq. (3.52): Gauss-Newton update (J'J) dx = J'r  ->  x <- x + dx

    This is the book's standard Gauss-Newton formulation for nonlinear LS.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gauss-Newton Nonlinear LS (Eq. 3.51-3.52)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate noisy measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Initial guess
    x0 = np.array([5.0, 5.0])

    print(f"\nMeasurement model: h_i(x) = ||x - a_i||")
    print(f"Residual:          r_i = y_i - h_i(x)")
    print(f"Jacobian:          J_i = (x - a_i) / ||x - a_i||")
    print(f"\nGauss-Newton update (Eq. 3.52):")
    print(f"  (J'J) dx = J'r  ->  x <- x + dx")
    print(f"\nInitial guess: {x0}")
    print(f"True position: {true_position}")

    # Gauss-Newton solution using core module
    result = gauss_newton(h, jacobian, y, x0, max_iter=20, tol=1e-8)

    print(f"\nConverged in {result.iterations} iterations")
    print(f"Final estimate:  {result.x}")
    print(f"Position error:  {np.linalg.norm(result.x - true_position):.6f} m")
    print(f"Final residuals: {result.residuals}")
    print(f"Final cost:      {result.cost:.6e}")
    print(f"\nCovariance:")
    print(result.covariance)

    return result


def example_4_levenberg_marquardt():
    """
    Example 4: Levenberg-Marquardt for Robust Convergence.

    Demonstrates Eq. (3.53) and Algorithm 3.2:
        (J'J + mu*I) dx = J'r

    LM combines Gauss-Newton (fast near solution) with gradient descent
    (robust far from solution) via adaptive damping parameter mu.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Levenberg-Marquardt (Eq. 3.53, Algorithm 3.2)")
    print("=" * 70)

    anchors, true_position = setup_positioning_scenario()
    h, jacobian = create_range_model(anchors)

    # Generate measurements
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)

    # Poor initial guess (far from true position)
    x0_poor = np.array([0.0, 0.0])

    print(f"\nLM update (Eq. 3.53): (J'J + mu*I) dx = J'r")
    print(f"  mu large -> gradient descent behavior (global convergence)")
    print(f"  mu small -> Gauss-Newton behavior (fast local convergence)")
    print(f"\nPoor initial guess: {x0_poor} (far from true position)")
    print(f"True position:      {true_position}")

    # Compare GN vs LM from poor initial guess
    print("\n--- Gauss-Newton from poor guess ---")
    result_gn = gauss_newton(h, jacobian, y, x0_poor, max_iter=50)
    error_gn = np.linalg.norm(result_gn.x - true_position)
    print(f"Result: {result_gn.x}, error: {error_gn:.4f} m, "
          f"iters: {result_gn.iterations}, converged: {result_gn.converged}")

    print("\n--- Levenberg-Marquardt from poor guess ---")
    result_lm = levenberg_marquardt(h, jacobian, y, x0_poor, max_iter=50, mu0=1e-3)
    error_lm = np.linalg.norm(result_lm.x - true_position)
    print(f"Result: {result_lm.x}, error: {error_lm:.4f} m, "
          f"iters: {result_lm.iterations}, converged: {result_lm.converged}")

    if error_lm < error_gn:
        print(f"\n[OK] LM converged better than GN from poor initial guess")
    else:
        print(f"\n(Both methods converged similarly)")

    return result_lm


def example_5_robust_ls():
    """
    Example 5: Robust Least Squares with Outliers.

    Demonstrates Table 3.1 robust estimators from Section 3.1.1:
        - L2:     e(x) = 0.5*||r||^2  (standard LS)
        - Cauchy: e(x) = 0.5*ln(1 + ||r||^2)
        - Huber:  e(x) = 0.5*||r||^2 if |r|<=delta, else delta(|r| - 0.5*delta)
        - G-M:    e(x) = 0.5*||r||^2 / (1 + ||r||^2)  (Geman-McClure)

    These robust loss functions reduce the influence of outliers (e.g., NLOS
    measurements in indoor environments).

    Note: Uses 8 anchors for sufficient redundancy to isolate outliers.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Robust LS with Table 3.1 Estimators")
    print("=" * 70)

    # Use more anchors for robust estimation (need redundancy!)
    anchors = np.array([
        [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0],  # Corners
        [5.0, 0.0], [5.0, 10.0], [0.0, 5.0], [10.0, 5.0]     # Midpoints
    ])
    true_position = np.array([3.0, 4.0])
    h, jacobian = create_range_model(anchors)

    # Generate measurements with one severe outlier
    np.random.seed(42)
    y = compute_ranges(true_position, anchors, noise_std=0.1)
    y[2] += 5.0  # 5m NLOS error on anchor 2

    print(f"\nScenario: 2D positioning from 8 anchors")
    print(f"Added 5.0 m NLOS outlier to anchor 2")
    print(f"(8 anchors provide redundancy for outlier rejection)")
    print(f"\nTable 3.1 Robust Estimators:")
    print(f"  L2:     e(x) = 0.5*||r||^2        (standard, sensitive to outliers)")
    print(f"  Cauchy: e(x) = 0.5*ln(1+||r||^2) (soft downweighting)")
    print(f"  Huber:  e(x) = quadratic/linear  (threshold at delta)")
    print(f"  G-M:    e(x) = 0.5*||r||^2/(1+||r||^2) (strong outlier rejection)")

    x0 = np.array([5.0, 5.0])

    # Robust methods from Table 3.1
    # Note: L2 is included in the loop below for comparison
    table_3_1_methods = {
        "L2 (Table 3.1)": "l2",
        "Cauchy (Table 3.1)": "cauchy",
        "Huber (Table 3.1)": "huber",
        "G-M (Table 3.1)": "gm",
        "Tukey (extra)": "tukey",  # Not in Table 3.1, but available
    }

    results = {}
    for label, method in table_3_1_methods.items():
        result = robust_gauss_newton(
            h, jacobian, y, x0,
            loss=method,
            loss_param=1.5,
            max_iter=30,
            max_irls_iter=10,
        )
        error = np.linalg.norm(result.x - true_position)
        results[label] = {
            "position": result.x,
            "error": error,
            "weights": result.weights,
            "outlier_weight": result.weights[2],
        }

    # Results
    print(f"\nTrue position: {true_position}")
    print(f"\n{'Method':<20} {'Position':<25} {'Error (m)':<10} {'Outlier w':<10}")
    print("-" * 65)

    for label, res in results.items():
        pos_str = f"[{res['position'][0]:.3f}, {res['position'][1]:.3f}]"
        print(f"{label:<20} {pos_str:<25} {res['error']:<10.4f} {res['outlier_weight']:<10.4f}")

    print(f"\nKey insight from Table 3.1:")
    print(f"  - L2 is corrupted by the outlier (no downweighting)")
    print(f"  - Cauchy provides soft downweighting")
    print(f"  - Huber transitions from quadratic to linear")
    print(f"  - G-M provides strongest outlier rejection")

    return results, anchors, true_position, y


def visualize_results():
    """Create visualization of all examples with Table 3.1 labels."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    # Setup
    anchors_4, true_position = setup_positioning_scenario()
    h_4, jac_4 = create_range_model(anchors_4)

    # 8 anchors for robust example
    anchors_8 = np.array([
        [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0],
        [5.0, 0.0], [5.0, 10.0], [0.0, 5.0], [10.0, 5.0]
    ])
    h_8, jac_8 = create_range_model(anchors_8)

    np.random.seed(42)

    # Generate clean measurements (4 anchors)
    y_clean = compute_ranges(true_position, anchors_4, noise_std=0.1)
    x0 = np.array([5.0, 5.0])

    # Example 1: Linear LS (one iteration)
    r = y_clean - h_4(x0)
    A = jac_4(x0)
    dx_ls, _ = linear_least_squares(A, r)
    pos_linear = x0 + dx_ls

    # Example 3: Gauss-Newton
    result_gn = gauss_newton(h_4, jac_4, y_clean, x0)
    pos_gn = result_gn.x

    # Example 4: LM from poor guess
    result_lm = levenberg_marquardt(h_4, jac_4, y_clean, np.array([0.0, 0.0]))
    pos_lm = result_lm.x

    # Example 5: Robust LS with outlier (8 anchors)
    y_outlier = compute_ranges(true_position, anchors_8, noise_std=0.1)
    y_outlier[2] += 5.0  # Outlier

    pos_l2 = gauss_newton(h_8, jac_8, y_outlier, x0).x

    # Robust methods from Table 3.1
    pos_cauchy = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="cauchy").x
    pos_huber = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="huber").x
    pos_gm = robust_gauss_newton(h_8, jac_8, y_outlier, x0, loss="gm").x

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ----- Plot 1: Clean data (Examples 1-4) -----
    ax = axes[0]

    # Anchors
    ax.scatter(anchors_4[:, 0], anchors_4[:, 1], s=200, c="blue", marker="^",
               label="Anchors", zorder=5)

    # True position
    ax.scatter(true_position[0], true_position[1], s=250, c="green", marker="*",
               label="True Position", zorder=5)

    # Estimates
    ax.scatter(x0[0], x0[1], s=150, c="gray", marker="x", label="Initial Guess", zorder=4)
    ax.scatter(pos_linear[0], pos_linear[1], s=150, c="orange", marker="o",
               label="Linear LS (Eq. 3.2)", zorder=4)
    ax.scatter(pos_gn[0], pos_gn[1], s=150, c="red", marker="s",
               label="Gauss-Newton (Eq. 3.52)", zorder=4)
    ax.scatter(pos_lm[0], pos_lm[1], s=150, c="purple", marker="D",
               label="LM (Eq. 3.53)", zorder=4)

    # Range circles
    for i, anchor in enumerate(anchors_4):
        circle = plt.Circle(anchor, y_clean[i], fill=False,
                            edgecolor="blue", alpha=0.3, linestyle="--")
        ax.add_patch(circle)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Examples 1-4: LS Methods (Clean Data)\n"
                 "Gauss-Newton (Eq. 3.52) & LM (Eq. 3.53, Alg. 3.2)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    # ----- Plot 2: Robust LS with outlier (Table 3.1) -----
    ax = axes[1]

    # Anchors
    ax.scatter(anchors_8[:, 0], anchors_8[:, 1], s=200, c="blue", marker="^",
               label="Anchors (8)", zorder=5)

    # Mark outlier anchor
    ax.scatter(anchors_8[2, 0], anchors_8[2, 1], s=350, facecolors="none",
               edgecolors="red", linewidth=3, zorder=4, label="Outlier Anchor")

    # True position
    ax.scatter(true_position[0], true_position[1], s=250, c="green", marker="*",
               label="True Position", zorder=5)

    # Table 3.1 estimator results
    ax.scatter(pos_l2[0], pos_l2[1], s=150, c="orange", marker="o",
               label="L2 (Table 3.1) - corrupted", zorder=4)
    ax.scatter(pos_cauchy[0], pos_cauchy[1], s=150, c="cyan", marker="s",
               label="Cauchy (Table 3.1)", zorder=4)
    ax.scatter(pos_huber[0], pos_huber[1], s=150, c="magenta", marker="^",
               label="Huber (Table 3.1)", zorder=4)
    ax.scatter(pos_gm[0], pos_gm[1], s=150, c="purple", marker="D",
               label="G-M (Table 3.1)", zorder=4)

    # Range circles (show first 4 anchors only to reduce clutter)
    for i in range(4):
        anchor = anchors_8[i]
        color = "red" if i == 2 else "blue"
        alpha = 0.6 if i == 2 else 0.2
        lw = 2.5 if i == 2 else 1
        circle = plt.Circle(anchor, y_outlier[i], fill=False,
                            edgecolor=color, alpha=alpha, linestyle="--", linewidth=lw)
        ax.add_patch(circle)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Example 5: Robust LS with Table 3.1 Estimators\n"
                 "(Section 3.1.1: Outlier Rejection)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)

    plt.tight_layout()

    # Save to figs directory
    output_dir = Path(__file__).parent / "figs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch3_least_squares_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved as: {output_path}")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CHAPTER 3: LEAST SQUARES ESTIMATION EXAMPLES")
    print("=" * 70)
    print("\nBook Reference: Section 3.1 (Least Squares Estimation)")
    print("               Section 3.1.1 (Robust Estimators, Table 3.1)")
    print("               Section 3.4.1 (Gauss-Newton & Levenberg-Marquardt)")
    print("\nApplication: 2D positioning from Time-of-Arrival (TOA) ranges")
    print("Measurement model: h_i(x) = ||x - a_i||")
    print("Residual:          r_i = y_i - h_i(x)  (observation - prediction)")

    # Run examples
    example_1_linear_ls()
    example_2_weighted_ls()
    example_3_gauss_newton()
    example_4_levenberg_marquardt()
    example_5_robust_ls()

    # Visualization
    visualize_results()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
