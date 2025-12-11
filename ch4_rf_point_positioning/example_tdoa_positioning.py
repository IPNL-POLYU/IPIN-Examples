"""
TDOA Positioning Examples.

This script demonstrates TDOA positioning algorithms from Chapter 4.

Implements:
    - TDOA measurement model (Eqs. 4.27-4.33)
    - TDOA I-WLS (Eqs. 4.34-4.42)
    - Fang's closed-form algorithm (Eqs. 4.43-4.48)
    - Chan's two-step algorithm (Eq. 4.58)

Author: Navigation Engineering Team
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import TDOAPositioner, tdoa_chan_solver, tdoa_fang_solver


def demo_tdoa_basic():
    """Demonstrate basic TDOA positioning with I-WLS."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic TDOA Positioning (I-WLS)")
    print("=" * 70)

    # Setup anchors (5 anchors in a larger area)
    anchors = np.array(
        [[0, 0], [15, 0], [15, 15], [0, 15], [7.5, 7.5]], dtype=float
    )

    # True position
    true_position = np.array([5.0, 8.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Generate TDOA measurements (reference anchor = 0)
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_measurements = []
    for i in range(1, len(anchors)):
        dist_i = np.linalg.norm(true_position - anchors[i])
        tdoa = dist_i - dist_ref
        tdoa_measurements.append(tdoa)
    tdoa_measurements = np.array(tdoa_measurements)

    print(f"\nTDOA measurements (m): {tdoa_measurements}")

    # Solve using I-WLS
    positioner = TDOAPositioner(anchors, reference_idx=0)
    estimated_position, info = positioner.solve(
        tdoa_measurements, initial_guess=np.array([7.5, 7.5])
    )

    print(f"\nEstimated position: {estimated_position}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(
        f"Position error: {np.linalg.norm(estimated_position - true_position):.6f} m"
    )

    return anchors, true_position, tdoa_measurements


def demo_tdoa_with_noise():
    """Demonstrate TDOA positioning with measurement noise."""
    print("\n" + "=" * 70)
    print("Demo 2: TDOA Positioning with Measurement Noise")
    print("=" * 70)

    # Setup
    anchors = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float
    )
    true_position = np.array([7.0, 12.0])

    # Generate noiseless TDOA
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    results = []

    print(f"\nTrue position: {true_position}")
    print(f"Testing {len(noise_levels)} noise levels...")

    for noise_std in noise_levels:
        # Add noise
        tdoa_noisy = tdoa_true + np.random.randn(len(tdoa_true)) * noise_std

        # Solve
        positioner = TDOAPositioner(anchors, reference_idx=0)
        est_pos, info = positioner.solve(
            tdoa_noisy, initial_guess=np.array([10.0, 10.0])
        )

        if info["converged"]:
            error = np.linalg.norm(est_pos - true_position)
            results.append(
                {
                    "noise": noise_std,
                    "position": est_pos,
                    "error": error,
                    "iterations": info["iterations"],
                }
            )
        else:
            results.append(
                {
                    "noise": noise_std,
                    "position": None,
                    "error": np.inf,
                    "iterations": info["iterations"],
                }
            )

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Noise (m)':<15} {'Est. Position':<25} {'Error (m)':<12} {'Iters':<8}")
    print("-" * 70)
    for r in results:
        pos_str = (
            f"[{r['position'][0]:.3f}, {r['position'][1]:.3f}]"
            if r["position"] is not None
            else "FAILED"
        )
        error_str = f"{r['error']:.4f}" if r["error"] != np.inf else "FAILED"
        print(
            f"{r['noise']:<15.2f} {pos_str:<25} {error_str:<12} {r['iterations']:<8}"
        )

    return results


def demo_closed_form_algorithms():
    """Demonstrate Fang and Chan closed-form TDOA algorithms."""
    print("\n" + "=" * 70)
    print("Demo 3: Closed-Form TDOA Algorithms (Fang & Chan)")
    print("=" * 70)

    # Setup (4 anchors for Fang, more for Chan)
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    true_position = np.array([8.0, 12.0])

    # Generate TDOA measurements
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_measurements = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    print(f"\nTrue position: {true_position}")
    print(f"TDOA measurements: {tdoa_measurements}")

    # Fang's algorithm
    print("\n--- Fang's Algorithm ---")
    try:
        fang_position = tdoa_fang_solver(anchors, tdoa_measurements)
        fang_error = np.linalg.norm(fang_position - true_position)
        print(f"Estimated position: {fang_position}")
        print(f"Position error: {fang_error:.6f} m")
    except Exception as e:
        print(f"Fang's algorithm failed: {e}")
        fang_position = None
        fang_error = np.inf

    # Chan's algorithm
    print("\n--- Chan's Algorithm ---")
    try:
        chan_position = tdoa_chan_solver(anchors, tdoa_measurements)
        chan_error = np.linalg.norm(chan_position - true_position)
        print(f"Estimated position: {chan_position}")
        print(f"Position error: {chan_error:.6f} m")
    except Exception as e:
        print(f"Chan's algorithm failed: {e}")
        chan_position = None
        chan_error = np.inf

    # I-WLS for comparison
    print("\n--- I-WLS (for comparison) ---")
    positioner = TDOAPositioner(anchors, reference_idx=0)
    iwls_position, info = positioner.solve(
        tdoa_measurements, initial_guess=np.array([10.0, 10.0])
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
    if fang_position is not None:
        print(f"{'Fang (closed-form)':<20} {fang_error:<20.6f}")
    else:
        print(f"{'Fang (closed-form)':<20} {'FAILED':<20}")
    if chan_position is not None:
        print(f"{'Chan (closed-form)':<20} {chan_error:<20.6f}")
    else:
        print(f"{'Chan (closed-form)':<20} {'FAILED':<20}")
    print(f"{'I-WLS (iterative)':<20} {iwls_error:<20.6f}")

    return anchors, true_position, tdoa_measurements


def demo_geometry_effect():
    """Demonstrate the effect of anchor geometry on TDOA accuracy."""
    print("\n" + "=" * 70)
    print("Demo 4: Effect of Anchor Geometry on TDOA Accuracy")
    print("=" * 70)

    true_position = np.array([5.0, 5.0])

    # Good geometry: anchors surrounding the target
    good_anchors = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float
    )

    # Poor geometry: anchors on one side
    poor_anchors = np.array(
        [[0, 0], [2, 0], [4, 0], [6, 0]], dtype=float
    )

    noise_std = 0.2  # meters

    print(f"\nTrue position: {true_position}")
    print(f"Noise std: {noise_std} m")

    # Test with good geometry
    print("\n--- Good Geometry (surrounding) ---")
    dist_ref = np.linalg.norm(true_position - good_anchors[0])
    tdoa_good = np.array(
        [
            np.linalg.norm(true_position - good_anchors[i]) - dist_ref
            for i in range(1, len(good_anchors))
        ]
    )
    tdoa_good_noisy = tdoa_good + np.random.randn(len(tdoa_good)) * noise_std

    positioner_good = TDOAPositioner(good_anchors, reference_idx=0)
    est_good, info_good = positioner_good.solve(
        tdoa_good_noisy, initial_guess=np.array([5.0, 5.0])
    )

    if info_good["converged"]:
        error_good = np.linalg.norm(est_good - true_position)
        print(f"Estimated position: {est_good}")
        print(f"Position error: {error_good:.4f} m")
    else:
        print("Failed to converge")
        error_good = np.inf

    # Test with poor geometry
    print("\n--- Poor Geometry (collinear) ---")
    dist_ref = np.linalg.norm(true_position - poor_anchors[0])
    tdoa_poor = np.array(
        [
            np.linalg.norm(true_position - poor_anchors[i]) - dist_ref
            for i in range(1, len(poor_anchors))
        ]
    )
    tdoa_poor_noisy = tdoa_poor + np.random.randn(len(tdoa_poor)) * noise_std

    positioner_poor = TDOAPositioner(poor_anchors, reference_idx=0)
    est_poor, info_poor = positioner_poor.solve(
        tdoa_poor_noisy, initial_guess=np.array([5.0, 5.0])
    )

    if info_poor["converged"]:
        error_poor = np.linalg.norm(est_poor - true_position)
        print(f"Estimated position: {est_poor}")
        print(f"Position error: {error_poor:.4f} m")
    else:
        print("Failed to converge")
        error_poor = np.inf

    # Summary
    print("\n" + "-" * 70)
    print("Geometry Comparison:")
    print("-" * 70)
    print(f"{'Geometry':<30} {'Position Error (m)':<20}")
    print("-" * 70)
    if error_good != np.inf:
        print(f"{'Good (surrounding)':<30} {error_good:<20.4f}")
    else:
        print(f"{'Good (surrounding)':<30} {'FAILED':<20}")
    if error_poor != np.inf:
        print(f"{'Poor (collinear)':<30} {error_poor:<20.4f}")
    else:
        print(f"{'Poor (collinear)':<30} {'FAILED':<20}")

    print(
        "\nNote: Poor geometry leads to higher errors and potential convergence issues."
    )


def main():
    """Run all TDOA positioning examples."""
    print("\n" + "=" * 70)
    print("Chapter 4: TDOA Positioning Examples")
    print("=" * 70)

    # Run demos
    demo_tdoa_basic()
    demo_tdoa_with_noise()
    demo_closed_form_algorithms()
    demo_geometry_effect()

    print("\n" + "=" * 70)
    print("All TDOA examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

