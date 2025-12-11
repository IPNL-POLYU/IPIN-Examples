"""
AOA Positioning Examples.

This script demonstrates Angle of Arrival (AOA) positioning algorithms
from Chapter 4.

Implements:
    - AOA measurement model (Eqs. 4.63-4.67)
    - AOA I-WLS positioning
    - OVE algorithm (Optimal Velocity Estimator)
    - 3D PLE algorithm (3D Position Line Estimation)

Author: Navigation Engineering Team
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import AOAPositioner, aoa_3dple_solver, aoa_azimuth, aoa_ove_solver


def demo_aoa_basic():
    """Demonstrate basic AOA positioning with I-WLS."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic AOA Positioning (I-WLS)")
    print("=" * 70)

    # Setup anchors (4 anchors at corners)
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    # True position
    true_position = np.array([4.0, 6.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Generate AOA measurements (azimuth angles)
    aoa_measurements = np.array(
        [aoa_azimuth(anchor, true_position) for anchor in anchors]
    )

    print(f"\nAOA measurements (radians): {aoa_measurements}")
    print(f"AOA measurements (degrees): {np.rad2deg(aoa_measurements)}")

    # Solve using I-WLS
    positioner = AOAPositioner(anchors)
    estimated_position, info = positioner.solve(
        aoa_measurements, initial_guess=np.array([5.0, 5.0])
    )

    print(f"\nEstimated position: {estimated_position}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(
        f"Position error: {np.linalg.norm(estimated_position - true_position):.6f} m"
    )

    return anchors, true_position, aoa_measurements


def demo_aoa_with_noise():
    """Demonstrate AOA positioning with measurement noise."""
    print("\n" + "=" * 70)
    print("Demo 2: AOA Positioning with Measurement Noise")
    print("=" * 70)

    # Setup
    anchors = np.array([[0, 0], [15, 0], [15, 15], [0, 15]], dtype=float)
    true_position = np.array([6.0, 9.0])

    # Generate noiseless AOA
    aoa_true = np.array(
        [aoa_azimuth(anchor, true_position) for anchor in anchors]
    )

    # Test different noise levels (in degrees)
    noise_levels_deg = [0.0, 1.0, 5.0, 10.0]
    results = []

    print(f"\nTrue position: {true_position}")
    print(f"Testing {len(noise_levels_deg)} noise levels...")

    for noise_deg in noise_levels_deg:
        noise_rad = np.deg2rad(noise_deg)

        # Add noise
        aoa_noisy = aoa_true + np.random.randn(len(aoa_true)) * noise_rad

        # Solve
        positioner = AOAPositioner(anchors)
        est_pos, info = positioner.solve(
            aoa_noisy, initial_guess=np.array([7.5, 7.5])
        )

        if info["converged"]:
            error = np.linalg.norm(est_pos - true_position)
            results.append(
                {
                    "noise": noise_deg,
                    "position": est_pos,
                    "error": error,
                    "iterations": info["iterations"],
                }
            )
        else:
            results.append(
                {
                    "noise": noise_deg,
                    "position": None,
                    "error": np.inf,
                    "iterations": info["iterations"],
                }
            )

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Noise (deg)':<15} {'Est. Position':<25} {'Error (m)':<12} {'Iters':<8}")
    print("-" * 70)
    for r in results:
        pos_str = (
            f"[{r['position'][0]:.3f}, {r['position'][1]:.3f}]"
            if r["position"] is not None
            else "FAILED"
        )
        error_str = f"{r['error']:.4f}" if r["error"] != np.inf else "FAILED"
        print(
            f"{r['noise']:<15.1f} {pos_str:<25} {error_str:<12} {r['iterations']:<8}"
        )

    return results


def demo_closed_form_algorithms():
    """Demonstrate OVE and 3D PLE closed-form AOA algorithms."""
    print("\n" + "=" * 70)
    print("Demo 3: Closed-Form AOA Algorithms (OVE & 3D PLE)")
    print("=" * 70)

    # Setup (at least 3 anchors needed)
    anchors = np.array([[0, 0], [20, 0], [10, 20]], dtype=float)
    true_position = np.array([10.0, 8.0])

    # Generate AOA measurements
    aoa_measurements = np.array(
        [aoa_azimuth(anchor, true_position) for anchor in anchors]
    )

    print(f"\nTrue position: {true_position}")
    print(f"AOA measurements (deg): {np.rad2deg(aoa_measurements)}")

    # OVE algorithm
    print("\n--- OVE Algorithm ---")
    try:
        ove_position = aoa_ove_solver(anchors, aoa_measurements)
        ove_error = np.linalg.norm(ove_position - true_position)
        print(f"Estimated position: {ove_position}")
        print(f"Position error: {ove_error:.6f} m")
    except Exception as e:
        print(f"OVE algorithm failed: {e}")
        ove_position = None
        ove_error = np.inf

    # 3D PLE algorithm
    print("\n--- 3D PLE Algorithm ---")
    try:
        ple_position = aoa_3dple_solver(anchors, aoa_measurements)
        ple_error = np.linalg.norm(ple_position - true_position)
        print(f"Estimated position: {ple_position}")
        print(f"Position error: {ple_error:.6f} m")
    except Exception as e:
        print(f"3D PLE algorithm failed: {e}")
        ple_position = None
        ple_error = np.inf

    # I-WLS for comparison
    print("\n--- I-WLS (for comparison) ---")
    positioner = AOAPositioner(anchors)
    iwls_position, info = positioner.solve(
        aoa_measurements, initial_guess=np.array([10.0, 10.0])
    )
    iwls_error = np.linalg.norm(iwls_position - true_position)
    print(f"Estimated position: {iwls_position}")
    print(f"Position error: {iwls_error:.6f} m")
    print(f"Iterations: {info['iterations']}")

    # Summary
    print("\n" + "-" * 70)
    print("Algorithm Comparison:")
    print("-" * 70)
    print(f"{'Method':<20} {'Position Error (m)':<20}")
    print("-" * 70)
    if ove_position is not None:
        print(f"{'OVE (closed-form)':<20} {ove_error:<20.6f}")
    else:
        print(f"{'OVE (closed-form)':<20} {'FAILED':<20}")
    if ple_position is not None:
        print(f"{'3D PLE (closed-form)':<20} {ple_error:<20.6f}")
    else:
        print(f"{'3D PLE (closed-form)':<20} {'FAILED':<20}")
    print(f"{'I-WLS (iterative)':<20} {iwls_error:<20.6f}")

    return anchors, true_position, aoa_measurements


def demo_minimum_anchors():
    """Demonstrate AOA positioning with minimum number of anchors."""
    print("\n" + "=" * 70)
    print("Demo 4: Minimum Anchors for AOA Positioning")
    print("=" * 70)

    true_position = np.array([5.0, 7.0])

    # Test with 2, 3, and 4 anchors
    anchor_configs = {
        "2 anchors": np.array([[0, 0], [10, 0]], dtype=float),
        "3 anchors": np.array([[0, 0], [10, 0], [5, 10]], dtype=float),
        "4 anchors": np.array(
            [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float
        ),
    }

    print(f"\nTrue position: {true_position}")

    for config_name, anchors in anchor_configs.items():
        print(f"\n--- {config_name} ---")

        # Generate AOA measurements
        aoa = np.array(
            [aoa_azimuth(anchor, true_position) for anchor in anchors]
        )

        # Try to solve
        try:
            positioner = AOAPositioner(anchors)
            est_pos, info = positioner.solve(
                aoa, initial_guess=np.array([5.0, 5.0])
            )

            if info["converged"]:
                error = np.linalg.norm(est_pos - true_position)
                print(f"Estimated position: {est_pos}")
                print(f"Position error: {error:.6f} m")
                print(f"Iterations: {info['iterations']}")
            else:
                print("Failed to converge")
        except Exception as e:
            print(f"Failed: {e}")

    print(
        "\nNote: At least 2 anchors are theoretically needed, "
        "but 3+ anchors improve accuracy and robustness."
    )


def visualize_aoa_geometry():
    """Visualize AOA positioning geometry."""
    print("\n" + "=" * 70)
    print("Demo 5: AOA Geometry Visualization")
    print("=" * 70)

    # Setup
    anchors = np.array([[0, 0], [12, 0], [12, 12], [0, 12]], dtype=float)
    true_position = np.array([5.0, 7.0])

    # Generate AOA
    aoa = np.array([aoa_azimuth(anchor, true_position) for anchor in anchors])

    # Add small noise
    aoa_noisy = aoa + np.random.randn(len(aoa)) * np.deg2rad(2.0)

    # Solve
    positioner = AOAPositioner(anchors)
    est_pos, info = positioner.solve(
        aoa_noisy, initial_guess=np.array([6.0, 6.0])
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot anchors
    ax.plot(
        anchors[:, 0],
        anchors[:, 1],
        "s",
        color="blue",
        markersize=12,
        label="Anchors",
    )
    for i, anchor in enumerate(anchors):
        ax.text(
            anchor[0] - 0.5,
            anchor[1] - 0.8,
            f"A{i}",
            fontsize=10,
            color="blue",
        )

    # Plot true position
    ax.plot(
        true_position[0],
        true_position[1],
        "o",
        color="green",
        markersize=15,
        label="True Position",
    )

    # Plot estimated position
    ax.plot(
        est_pos[0],
        est_pos[1],
        "x",
        color="red",
        markersize=15,
        linewidth=3,
        label="Estimated Position",
    )

    # Plot angle lines from anchors to true position
    for i, (anchor, angle) in enumerate(zip(anchors, aoa_noisy)):
        # Draw line in the direction of the measured angle
        line_length = 15
        end_x = anchor[0] + line_length * np.cos(angle)
        end_y = anchor[1] + line_length * np.sin(angle)
        ax.plot(
            [anchor[0], end_x],
            [anchor[1], end_y],
            "--",
            color="gray",
            alpha=0.6,
            linewidth=1,
        )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        "AOA Positioning Geometry", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.set_xlim([-2, 14])
    ax.set_ylim([-2, 14])

    plt.tight_layout()
    plt.savefig("ch4_aoa_geometry.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: ch4_aoa_geometry.png")

    plt.show()

    error = np.linalg.norm(est_pos - true_position)
    print(f"\nTrue position: {true_position}")
    print(f"Estimated position: {est_pos}")
    print(f"Position error: {error:.4f} m")


def main():
    """Run all AOA positioning examples."""
    print("\n" + "=" * 70)
    print("Chapter 4: AOA Positioning Examples")
    print("=" * 70)

    # Run demos
    demo_aoa_basic()
    demo_aoa_with_noise()
    demo_closed_form_algorithms()
    demo_minimum_anchors()
    visualize_aoa_geometry()

    print("\n" + "=" * 70)
    print("All AOA examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

