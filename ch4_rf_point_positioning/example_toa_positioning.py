"""
TOA and RSS Positioning Example.

This script demonstrates Time of Arrival (TOA) and RSS-based positioning
using Iterative Weighted Least Squares (I-WLS).

Implements:
    - Eq. (4.1)-(4.3): TOA range measurements
    - Eq. (4.6)-(4.9): Two-way TOA / RTT measurement model
    - Eq. (4.11)-(4.13): RSS path-loss model
    - Eq. (4.14)-(4.23): Nonlinear TOA I-WLS positioning
    - Eq. (4.24)-(4.26): Joint position + clock bias estimation

Author: Navigation Engineering Team
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import (
    SPEED_OF_LIGHT,
    TOAPositioner,
    range_to_rtt,
    rss_pathloss,
    rss_to_distance,
    rtt_to_range,
    simulate_rtt_measurement,
    toa_range,
    toa_solve_with_clock_bias,
)


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

    # Solve using iterative LS (book default: Eq. 4.20)
    positioner = TOAPositioner(anchors, method="iterative_ls")
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

    # Solve using iterative LS (book default: Eq. 4.20)
    positioner = TOAPositioner(anchors, method="iterative_ls")
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
    """
    Example 3: TOA positioning with unknown clock bias.

    Demonstrates the unit convention for clock bias:
    - Measurement model (`toa_range`): clock_bias_s in SECONDS
    - Positioning solver: clock_bias_m in METERS (book Eq. 4.24)
    - Conversion: bias_m = c * bias_s, bias_s = bias_m / c

    The book uses meters because the Jacobian ∂h/∂(c*Δt) = 1 (Eq. 4.26).
    """
    print("\n" + "=" * 70)
    print("Example 3: Joint Position and Clock Bias Estimation")
    print("=" * 70)

    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos = np.array([5.0, 5.0])

    # Define clock bias in SECONDS (timing domain)
    # Then convert to METERS for the solver
    true_clock_bias_s = 10e-9  # 10 nanoseconds
    true_clock_bias_m = true_clock_bias_s * SPEED_OF_LIGHT  # ~3.0 meters

    print(f"\nTrue position: {true_pos}")
    print(f"True clock bias: {true_clock_bias_s*1e9:.2f} ns = {true_clock_bias_m:.3f} m")
    print(f"  (1 ns = {SPEED_OF_LIGHT*1e-9:.3f} m, 1 m = {1e9/SPEED_OF_LIGHT:.3f} ns)")

    # Compute ranges WITH clock bias using measurement model
    # toa_range() takes clock_bias_s in SECONDS
    ranges_biased = np.array([
        toa_range(anchor, true_pos, clock_bias_s=true_clock_bias_s)
        for anchor in anchors
    ])

    # Also compute true geometric ranges (no bias)
    true_ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])

    print(f"\nTrue geometric ranges: {true_ranges}")
    print(f"Measured pseudoranges: {ranges_biased}")
    print(f"Difference (bias_m):   {ranges_biased - true_ranges}")

    # Solve with clock bias estimation
    # The solver estimates bias in METERS (book convention)
    initial_guess = np.array([6.0, 6.0, 0.0])  # [x, y, bias_m]
    pos, bias_m, info = toa_solve_with_clock_bias(
        anchors, ranges_biased, initial_guess
    )

    # Convert estimated bias from meters to seconds for interpretation
    bias_s = bias_m / SPEED_OF_LIGHT

    # Results
    pos_error = np.linalg.norm(pos - true_pos)
    bias_error_m = abs(bias_m - true_clock_bias_m)
    bias_error_ns = abs(bias_s - true_clock_bias_s) * 1e9

    print(f"\n--- Results ---")
    print(f"Estimated position: {pos}")
    print(f"Position error: {pos_error:.6f} m")
    print(f"\nEstimated clock bias:")
    print(f"  In meters:  {bias_m:.6f} m (error: {bias_error_m:.6f} m)")
    print(f"  In seconds: {bias_s*1e9:.3f} ns (error: {bias_error_ns:.3f} ns)")
    print(f"Iterations: {info['iterations']}, Converged: {info['converged']}")

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
    positioner = TOAPositioner(anchors, method="iterative_ls")
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


def example_rtt_measurement():
    """Example 5: Two-way TOA / RTT measurement model (Eqs. 4.6-4.9)."""
    print("\n" + "=" * 70)
    print("Example 5: Two-Way TOA / RTT Measurement Model (Eqs. 4.6-4.9)")
    print("=" * 70)

    print("\n--- Basic RTT to Range Conversion (Eq. 4.7) ---")

    # Example: Wi-Fi FTM measurement
    print("\nWi-Fi FTM Example:")
    print(f"  Speed of light: {SPEED_OF_LIGHT:.0f} m/s")
    print(f"  1 nanosecond timing -> {SPEED_OF_LIGHT * 1e-9 / 2:.3f} m range error")

    # RTT for 15m distance
    distance = 15.0
    rtt_ideal = range_to_rtt(distance)
    print(f"\n  True distance: {distance:.1f} m")
    print(f"  Ideal RTT: {rtt_ideal * 1e9:.2f} ns")

    # Convert back to range
    range_est = rtt_to_range(rtt_ideal)
    print(f"  Range from RTT: {range_est:.6f} m")

    print("\n--- RTT with Processing Time (Eq. 4.7) ---")

    # Typical Wi-Fi FTM processing time: 10-100 ns
    processing_time = 50e-9  # 50 ns
    print(f"\n  Beacon processing time: {processing_time * 1e9:.0f} ns")

    # RTT includes processing time
    rtt_with_proc = range_to_rtt(distance, processing_time=processing_time)
    print(f"  RTT with processing: {rtt_with_proc * 1e9:.2f} ns")

    # Without correction: overestimate distance
    range_wrong = rtt_to_range(rtt_with_proc)
    print(f"\n  Range without correction: {range_wrong:.2f} m (ERROR: +{range_wrong - distance:.2f} m)")

    # With correction: correct distance
    range_correct = rtt_to_range(rtt_with_proc, processing_time=processing_time)
    print(f"  Range with correction: {range_correct:.6f} m")

    print("\n--- RTT with Clock Drift (Eq. 4.8) ---")

    # TCXO clock drift example: ~1-2 ppm
    # For 100ns RTT, 1 ppm drift -> 0.1 ns error
    clock_drift = 5e-9  # 5 ns drift
    print(f"\n  Agent clock drift: {clock_drift * 1e9:.0f} ns")

    rtt_with_drift = rtt_with_proc + clock_drift
    print(f"  RTT with drift: {rtt_with_drift * 1e9:.2f} ns")

    # Correct for both processing and drift
    range_corrected = rtt_to_range(
        rtt_with_drift, processing_time=processing_time, clock_drift=clock_drift
    )
    print(f"  Range with full correction: {range_corrected:.6f} m")

    print("\n--- Simulated RTT Measurement with Noise (Eq. 4.9) ---")

    np.random.seed(42)

    anchor = np.array([0.0, 0.0, 0.0])
    agent = np.array([15.0, 0.0, 0.0])

    # Single measurement
    rtt, info = simulate_rtt_measurement(
        anchor, agent,
        processing_time=50e-9,
        processing_time_std=5e-9,  # 5 ns std
        clock_drift_std=2e-9,      # 2 ns std
    )

    print(f"\n  True range: {info['true_range']:.2f} m")
    print(f"  Processing time (actual): {info['processing_time_actual'] * 1e9:.2f} ns")
    print(f"  Clock drift (actual): {info['clock_drift_actual'] * 1e9:.2f} ns")
    print(f"  Measured RTT: {rtt * 1e9:.2f} ns")
    print(f"  Estimated range: {info['range_estimate']:.3f} m")
    print(f"  Range error: {info['range_estimate'] - 15.0:.3f} m")

    # Monte Carlo simulation
    print("\n  Monte Carlo (100 trials):")
    errors = []
    for _ in range(100):
        _, info = simulate_rtt_measurement(
            anchor, agent,
            processing_time=50e-9,
            processing_time_std=5e-9,
            clock_drift_std=2e-9,
        )
        errors.append(info['range_estimate'] - 15.0)

    errors = np.array(errors)
    print(f"    Mean error: {np.mean(errors):.4f} m")
    print(f"    Std dev: {np.std(errors):.4f} m")
    print(f"    RMSE: {np.sqrt(np.mean(errors**2)):.4f} m")

    print("\n--- RTT-Based Positioning Example ---")

    # Multiple anchors
    anchors = np.array([
        [0, 0, 0],
        [20, 0, 0],
        [20, 20, 0],
        [0, 20, 0],
    ], dtype=float)
    true_pos = np.array([8.0, 12.0, 0.0])

    print(f"\n  True position: {true_pos[:2]}")

    # Simulate RTT measurements from each anchor
    ranges_from_rtt = []
    for i, anchor in enumerate(anchors):
        rtt, info = simulate_rtt_measurement(
            anchor, true_pos,
            processing_time=50e-9,
            processing_time_std=3e-9,
        )
        ranges_from_rtt.append(info['range_estimate'])
        print(f"  Anchor {i+1}: RTT={rtt*1e9:.1f}ns -> Range={info['range_estimate']:.3f}m "
              f"(true: {info['true_range']:.2f}m)")

    ranges_from_rtt = np.array(ranges_from_rtt)

    # Position using TOA solver
    positioner = TOAPositioner(anchors[:, :2], method='iterative_ls')
    est_pos, info = positioner.solve(
        ranges_from_rtt, initial_guess=np.array([10.0, 10.0])
    )

    error = np.linalg.norm(est_pos - true_pos[:2])
    print(f"\n  Estimated position: {est_pos}")
    print(f"  Position error: {error:.3f} m")

    return


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

    # Example 5: RTT model
    example_rtt_measurement()

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



