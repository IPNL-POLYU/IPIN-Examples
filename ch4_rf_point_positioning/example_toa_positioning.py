"""
TOA and RSS Positioning Example.

This script demonstrates Time of Arrival (TOA) and RSS-based positioning
using Iterative Weighted Least Squares (I-WLS).

Implements:
    - Eq. (4.1)-(4.3): TOA range measurements
    - Eq. (4.11)-(4.13): RSS path-loss model
    - Eq. (4.14)-(4.23): Nonlinear TOA I-WLS positioning
    - Eq. (4.24)-(4.26): Joint position + clock bias estimation

Author: Navigation Engineering Team
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import TOAPositioner, rss_pathloss, rss_to_distance, toa_range, toa_solve_with_clock_bias


def example_toa_perfect():
    """Example 1: TOA positioning with perfect measurements."""
    print("=" * 70)
    print("Example 1: TOA Positioning with Perfect Measurements")
    print("=" * 70)

    # Square anchor layout
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    print(f"\nAnchor positions:\n{anchors}")
    print(f"True position: {true_pos}")

    # Compute true ranges
    ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
    print(f"\nTrue ranges: {ranges}")

    # Solve using I-WLS
    positioner = TOAPositioner(anchors, method="iwls")
    estimated_pos, info = positioner.solve(
        ranges, initial_guess=np.array([6.0, 6.0])
    )

    # Results
    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"\nEstimated position: {estimated_pos}")
    print(f"Position error: {error:.6f} m")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Residual: {info['residual']:.2e}")

    return anchors, true_pos, estimated_pos, info


def example_toa_with_noise():
    """Example 2: TOA positioning with measurement noise."""
    print("\n" + "=" * 70)
    print("Example 2: TOA Positioning with Measurement Noise")
    print("=" * 70)

    np.random.seed(42)

    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([3.0, 7.0])

    print(f"\nTrue position: {true_pos}")

    # Add Gaussian noise to ranges
    true_ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
    noise_std = 0.1  # 10 cm standard deviation
    ranges_noisy = true_ranges + np.random.randn(4) * noise_std

    print(f"Range noise std: {noise_std} m")
    print(f"True ranges:  {true_ranges}")
    print(f"Noisy ranges: {ranges_noisy}")

    # Solve using I-WLS
    positioner = TOAPositioner(anchors, method="iwls")
    estimated_pos, info = positioner.solve(
        ranges_noisy, initial_guess=np.array([5.0, 5.0])
    )

    # Results
    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"\nEstimated position: {estimated_pos}")
    print(f"Position error: {error:.3f} m")
    print(f"Error/Noise ratio: {error/noise_std:.2f}")
    print(f"Iterations: {info['iterations']}")

    return anchors, true_pos, estimated_pos


def example_toa_with_clock_bias():
    """Example 3: TOA positioning with unknown clock bias."""
    print("\n" + "=" * 70)
    print("Example 3: Joint Position and Clock Bias Estimation")
    print("=" * 70)

    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])
    true_clock_bias_m = 2.0  # 2 meters (about 6.7 nanoseconds)

    print(f"\nTrue position: {true_pos}")
    print(f"True clock bias: {true_clock_bias_m} m")

    # Compute ranges with clock bias
    true_ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
    ranges_biased = true_ranges + true_clock_bias_m

    print(f"\nTrue ranges (unbiased):  {true_ranges}")
    print(f"Measured ranges (biased): {ranges_biased}")

    # Solve with clock bias estimation
    initial_guess = np.array([6.0, 6.0, 0.0])  # [x, y, clock_bias]
    pos, bias, info = toa_solve_with_clock_bias(
        anchors, ranges_biased, initial_guess
    )

    # Results
    pos_error = np.linalg.norm(pos - true_pos)
    bias_error = abs(bias - true_clock_bias_m)

    print(f"\nEstimated position: {pos}")
    print(f"Estimated clock bias: {bias:.3f} m")
    print(f"Position error: {pos_error:.6f} m")
    print(f"Clock bias error: {bias_error:.6f} m")
    print(f"Iterations: {info['iterations']}")

    return anchors, true_pos, pos


def example_rss_positioning():
    """Example 4: RSS-based ranging and positioning."""
    print("\n" + "=" * 70)
    print("Example 4: RSS-Based Ranging")
    print("=" * 70)

    # Transmitter parameters
    tx_power_dbm = 0.0  # dBm
    path_loss_exp = 2.5  # Indoor environment

    distances = np.array([1.0, 5.0, 10.0, 20.0])

    print(f"\nTx power: {tx_power_dbm} dBm")
    print(f"Path loss exponent: {path_loss_exp}")
    print("\nDistance -> RSS -> Estimated Distance:")

    for dist in distances:
        # Compute RSS
        rss = rss_pathloss(tx_power_dbm, dist, path_loss_exp)

        # Invert to estimate distance
        dist_est = rss_to_distance(rss, tx_power_dbm, path_loss_exp)

        print(f"  {dist:5.1f} m -> {rss:7.2f} dBm -> {dist_est:5.1f} m")

    # RSS-based positioning example
    print("\nRSS-Based Positioning:")
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    # Compute RSS at each anchor
    rss_measurements = []
    for anchor in anchors:
        dist = np.linalg.norm(anchor - true_pos)
        rss = rss_pathloss(tx_power_dbm, dist, path_loss_exp)
        rss_measurements.append(rss)

    rss_measurements = np.array(rss_measurements)
    print(f"RSS measurements: {rss_measurements}")

    # Convert RSS to ranges
    ranges_from_rss = np.array(
        [
            rss_to_distance(rss, tx_power_dbm, path_loss_exp)
            for rss in rss_measurements
        ]
    )
    print(f"Estimated ranges: {ranges_from_rss}")

    # Position from RSS-derived ranges
    positioner = TOAPositioner(anchors, method="iwls")
    estimated_pos, info = positioner.solve(
        ranges_from_rss, initial_guess=np.array([6.0, 6.0])
    )

    error = np.linalg.norm(estimated_pos - true_pos)
    print(f"Estimated position: {estimated_pos}")
    print(f"Position error: {error:.3f} m")


def plot_toa_positioning(anchors, true_pos, estimated_pos, history=None):
    """Visualize TOA positioning results."""
    plt.figure(figsize=(8, 8))

    # Plot anchors
    plt.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=200,
        c="red",
        marker="^",
        label="Anchors",
        zorder=5,
    )

    # Label anchors
    for i, anchor in enumerate(anchors):
        plt.text(
            anchor[0],
            anchor[1] + 0.5,
            f"A{i+1}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Plot true position
    plt.scatter(
        true_pos[0],
        true_pos[1],
        s=150,
        c="green",
        marker="o",
        label="True Position",
        zorder=4,
    )

    # Plot estimated position
    plt.scatter(
        estimated_pos[0],
        estimated_pos[1],
        s=150,
        c="blue",
        marker="x",
        label="Estimated Position",
        linewidths=3,
        zorder=4,
    )

    # Plot iteration history if available
    if history is not None and len(history) > 1:
        plt.plot(
            history[:, 0],
            history[:, 1],
            "b--",
            alpha=0.5,
            label="Convergence Path",
            zorder=3,
        )

    # Plot range circles
    for anchor in anchors:
        dist = np.linalg.norm(anchor - true_pos)
        circle = plt.Circle(
            anchor, dist, fill=False, color="red", alpha=0.2, linestyle="--"
        )
        plt.gca().add_patch(circle)

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("East (m)", fontsize=12)
    plt.ylabel("North (m)", fontsize=12)
    plt.title("TOA Positioning", fontsize=14, fontweight="bold")
    plt.legend(loc="best")

    plt.tight_layout()
    return plt.gcf()


def main():
    """Run all TOA positioning examples."""
    print("\n" + "=" * 70)
    print("Chapter 4: TOA and RSS Positioning Examples")
    print("=" * 70)

    # Example 1: Perfect measurements
    anchors1, true_pos1, est_pos1, info1 = example_toa_perfect()

    # Example 2: With noise
    anchors2, true_pos2, est_pos2 = example_toa_with_noise()

    # Example 3: With clock bias
    anchors3, true_pos3, est_pos3 = example_toa_with_clock_bias()

    # Example 4: RSS-based
    example_rss_positioning()

    # Visualization
    print("\n" + "=" * 70)
    print("Generating visualization...")
    print("=" * 70)

    fig = plot_toa_positioning(anchors1, true_pos1, est_pos1, info1["history"])
    plt.savefig(
        "ch4_rf_point_positioning/toa_positioning_example.png", dpi=150, bbox_inches="tight"
    )
    print("\nFigure saved: toa_positioning_example.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()


