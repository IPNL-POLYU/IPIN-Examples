"""
AOA Positioning Examples.

This script demonstrates Angle of Arrival (AOA) positioning algorithms
from Chapter 4.

Implements:
    - AOA measurement model (Eqs. 4.63-4.65)
        - Eq. 4.63: sin(theta) = (x_u^i - x_u,a) / ||x_a - x^i||
        - Eq. 4.64: tan(psi) = (x_e^i - x_e,a) / (x_n^i - x_n,a)
        - Eq. 4.65: z = [sin(theta_1), tan(psi_1), ..., sin(theta_I), tan(psi_I)]^T
    - AOA I-WLS positioning (Eqs. 4.67-4.78)
    - Orthogonal Vector Estimator (OVE) - 3D closed-form (Eqs. 4.79-4.85)
    - Pseudolinear Estimator (PLE) - 2D/3D closed-form (Eqs. 4.86-4.95)

ENU Convention:
    - Azimuth psi is measured from North (+N), positive CCW
    - psi = atan2(dE, dN) where dE = anchor_E - agent_E, dN = anchor_N - agent_N
    - Elevation theta is positive when anchor is above agent

Author: Li-Ta Hsu
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from core.rf import (
    AOAPositioner,
    aoa_angle_vector,
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    aoa_ove_solve,
    aoa_ple_solve_2d,
    aoa_ple_solve_3d,
    aoa_sin_elevation,
    aoa_tan_azimuth,
)


def demo_aoa_basic():
    """Demonstrate basic AOA positioning with I-WLS."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic AOA Positioning (I-WLS)")
    print("=" * 70)

    # Setup anchors (4 anchors at corners) in ENU coordinates
    # E=x, N=y in 2D
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    # True position
    true_position = np.array([4.0, 6.0])
    print(f"\nTrue position (E, N): {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Generate AOA measurements using ENU convention (Eq. 4.64)
    # psi = atan2(dE, dN) where dE = anchor_E - agent_E, dN = anchor_N - agent_N
    aoa_measurements = aoa_angle_vector(anchors, true_position, include_elevation=False)

    print(f"\nAOA azimuth angles (radians): {aoa_measurements}")
    print(f"AOA azimuth angles (degrees): {np.rad2deg(aoa_measurements)}")
    print("  Note: Azimuth from North, positive CCW (ENU convention)")

    # Show tan(psi) values per Eq. 4.64
    tan_psi = np.array([aoa_tan_azimuth(anchor, true_position) for anchor in anchors])
    print(f"\ntan(psi) values (Eq. 4.64): {tan_psi}")

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

    # Setup in ENU coordinates
    anchors = np.array([[0, 0], [15, 0], [15, 15], [0, 15]], dtype=float)
    true_position = np.array([6.0, 9.0])

    # Generate noiseless AOA using new convention (Eq. 4.64)
    aoa_true = aoa_angle_vector(anchors, true_position, include_elevation=False)

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


def demo_measurement_vector():
    """Demonstrate the AOA measurement vector per Eq. 4.65."""
    print("\n" + "=" * 70)
    print("Demo 3: AOA Measurement Vector (Eq. 4.65)")
    print("=" * 70)

    # Setup 3D scenario with anchors in ENU coordinates
    anchors_3d = np.array(
        [[0, 0, 5], [20, 0, 5], [10, 20, 5]], dtype=float
    )  # Anchors 5m above ground
    true_position_3d = np.array([10.0, 8.0, 0.0])  # Agent at ground level

    print(f"\nTrue agent position (E, N, U): {true_position_3d}")
    print(f"Anchor positions:")
    for i, anchor in enumerate(anchors_3d):
        print(f"  Anchor {i}: {anchor}")

    # Generate measurement vector per Eq. 4.65: [sin(theta_i), tan(psi_i), ...]
    z = aoa_measurement_vector(anchors_3d, true_position_3d, include_elevation=True)

    print(f"\nMeasurement vector z (Eq. 4.65):")
    print(f"  Shape: {z.shape}")
    for i in range(len(anchors_3d)):
        sin_theta = z[2 * i]
        tan_psi = z[2 * i + 1]
        print(f"  Anchor {i}: sin(theta)={sin_theta:.4f}, tan(psi)={tan_psi:.4f}")

    # Verify with individual functions
    print("\n--- Verification with individual functions ---")
    for i, anchor in enumerate(anchors_3d):
        sin_theta = aoa_sin_elevation(anchor, true_position_3d)
        tan_psi = aoa_tan_azimuth(anchor, true_position_3d)
        azimuth = aoa_azimuth(anchor, true_position_3d)
        print(
            f"  Anchor {i}: sin(theta)={sin_theta:.4f}, tan(psi)={tan_psi:.4f}, "
            f"psi={np.rad2deg(azimuth):.1f} deg"
        )

    # 2D case
    print("\n--- 2D Case (azimuth only) ---")
    anchors_2d = anchors_3d[:, :2]
    true_position_2d = true_position_3d[:2]

    z_2d = aoa_measurement_vector(anchors_2d, true_position_2d, include_elevation=False)
    print(f"2D measurement vector (tan(psi) only): {z_2d}")

    return anchors_3d, true_position_3d, z


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

    print(f"\nTrue position (E, N): {true_position}")

    for config_name, anchors in anchor_configs.items():
        print(f"\n--- {config_name} ---")

        # Generate AOA measurements using ENU convention (Eq. 4.64)
        aoa = aoa_angle_vector(anchors, true_position, include_elevation=False)

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
    """Visualize AOA positioning geometry with ENU convention."""
    print("\n" + "=" * 70)
    print("Demo 5: AOA Geometry Visualization (ENU Convention)")
    print("=" * 70)

    # Setup in ENU coordinates (E=x-axis, N=y-axis)
    anchors = np.array([[0, 0], [12, 0], [12, 12], [0, 12]], dtype=float)
    true_position = np.array([5.0, 7.0])

    # Generate AOA using new convention (Eq. 4.64: psi from North)
    aoa = aoa_angle_vector(anchors, true_position, include_elevation=False)

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
        label="Anchors (Beacons)",
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

    # Plot bearing lines from anchors toward agent
    # psi = atan2(dE, dN) is angle from North, so we convert to x-y plot
    for i, (anchor, psi) in enumerate(zip(anchors, aoa_noisy)):
        # In ENU: psi is from North (+y), so direction is:
        # E-component = sin(psi), N-component = cos(psi)
        line_length = 15
        end_e = anchor[0] + line_length * np.sin(psi)
        end_n = anchor[1] + line_length * np.cos(psi)
        ax.plot(
            [anchor[0], end_e],
            [anchor[1], end_n],
            "--",
            color="gray",
            alpha=0.6,
            linewidth=1,
        )
        # Label angle
        ax.text(
            anchor[0] + 2 * np.sin(psi),
            anchor[1] + 2 * np.cos(psi),
            f"psi={np.rad2deg(psi):.0f} deg",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title(
        "AOA Positioning Geometry (ENU Convention)\n"
        "psi = azimuth from North, positive CCW",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.set_xlim([-2, 14])
    ax.set_ylim([-2, 14])

    plt.tight_layout()
    plt.savefig(
        "ch4_rf_point_positioning/figs/ch4_aoa_geometry.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("\nFigure saved: ch4_rf_point_positioning/figs/ch4_aoa_geometry.png")

    plt.show()

    error = np.linalg.norm(est_pos - true_position)
    print(f"\nTrue position (E, N): {true_position}")
    print(f"Estimated position: {est_pos}")
    print(f"Position error: {error:.4f} m")


def demo_closed_form_algorithms():
    """Demonstrate closed-form AOA solvers (OVE and PLE)."""
    print("\n" + "=" * 70)
    print("Demo 6: Closed-Form AOA Solvers (OVE & PLE)")
    print("=" * 70)
    print("\nAlgorithms compared:")
    print("  - I-WLS: Iterative Weighted Least Squares (Eqs. 4.63-4.78)")
    print("  - OVE: Orthogonal Vector Estimator, 3D (Eqs. 4.79-4.85)")
    print("  - PLE: Pseudolinear Estimator, 2D (Eqs. 4.86-4.91)")

    # === 2D Comparison ===
    print("\n--- 2D Comparison (I-WLS vs PLE) ---")
    anchors_2d = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos_2d = np.array([4.0, 6.0])

    # Generate azimuth angles
    azimuths = np.array([aoa_azimuth(a, true_pos_2d) for a in anchors_2d])

    print(f"\nAnchors: {anchors_2d.tolist()}")
    print(f"True position (E, N): {true_pos_2d}")

    # I-WLS
    aoa_meas = aoa_angle_vector(anchors_2d, true_pos_2d, include_elevation=False)
    positioner = AOAPositioner(anchors_2d)
    pos_iwls, info_iwls = positioner.solve(aoa_meas, initial_guess=np.array([5.0, 5.0]))
    err_iwls = np.linalg.norm(pos_iwls - true_pos_2d)

    # PLE 2D
    pos_ple, info_ple = aoa_ple_solve_2d(anchors_2d, azimuths)
    err_ple = np.linalg.norm(pos_ple - true_pos_2d)

    print(f"\nResults (perfect measurements):")
    print(f"  I-WLS: pos={pos_iwls}, error={err_iwls:.6f} m, iters={info_iwls['iterations']}")
    print(f"  PLE:   pos={pos_ple}, error={err_ple:.6f} m (closed-form)")

    # === 3D Comparison ===
    print("\n--- 3D Comparison (I-WLS vs OVE vs PLE) ---")
    anchors_3d = np.array(
        [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
    )
    true_pos_3d = np.array([4.0, 6.0, 0.0])

    # Generate angles
    elevations = np.array([aoa_elevation(a, true_pos_3d) for a in anchors_3d])
    azimuths_3d = np.array([aoa_azimuth(a, true_pos_3d) for a in anchors_3d])

    print(f"\nAnchors (3D): {anchors_3d.tolist()}")
    print(f"True position (E, N, U): {true_pos_3d}")

    # I-WLS 3D
    aoa_meas_3d = aoa_angle_vector(anchors_3d, true_pos_3d, include_elevation=True)
    positioner_3d = AOAPositioner(anchors_3d)
    pos_iwls_3d, info_iwls_3d = positioner_3d.solve(
        aoa_meas_3d, initial_guess=np.array([5.0, 5.0, 1.0])
    )
    err_iwls_3d = np.linalg.norm(pos_iwls_3d - true_pos_3d)

    # OVE 3D
    pos_ove, info_ove = aoa_ove_solve(anchors_3d, elevations, azimuths_3d)
    err_ove = np.linalg.norm(pos_ove - true_pos_3d)

    # PLE 3D
    pos_ple_3d, info_ple_3d = aoa_ple_solve_3d(anchors_3d, elevations, azimuths_3d)
    err_ple_3d = np.linalg.norm(pos_ple_3d - true_pos_3d)

    print(f"\nResults (perfect measurements):")
    print(
        f"  I-WLS: pos={pos_iwls_3d}, error={err_iwls_3d:.6f} m, "
        f"iters={info_iwls_3d['iterations']}"
    )
    print(f"  OVE:   pos={pos_ove}, error={err_ove:.6f} m (closed-form)")
    print(f"  PLE:   pos={pos_ple_3d}, error={err_ple_3d:.6f} m (closed-form)")

    # === With noise ===
    print("\n--- With 2 deg measurement noise (Monte Carlo, 100 trials) ---")
    np.random.seed(42)
    noise_deg = 2.0
    n_trials = 100
    errors = {"I-WLS": [], "OVE": [], "PLE": []}

    for _ in range(n_trials):
        # Add noise
        elev_noisy = elevations + np.random.randn(len(elevations)) * np.deg2rad(noise_deg)
        azim_noisy = azimuths_3d + np.random.randn(len(azimuths_3d)) * np.deg2rad(noise_deg)

        # I-WLS
        aoa_noisy = np.zeros(2 * len(anchors_3d))
        for i in range(len(anchors_3d)):
            aoa_noisy[2 * i] = elev_noisy[i]
            aoa_noisy[2 * i + 1] = azim_noisy[i]
        try:
            pos, info = positioner_3d.solve(aoa_noisy, initial_guess=np.array([5.0, 5.0, 1.0]))
            if info["converged"]:
                errors["I-WLS"].append(np.linalg.norm(pos - true_pos_3d))
        except Exception:
            pass

        # OVE
        try:
            pos, _ = aoa_ove_solve(anchors_3d, elev_noisy, azim_noisy)
            errors["OVE"].append(np.linalg.norm(pos - true_pos_3d))
        except Exception:
            pass

        # PLE
        try:
            pos, _ = aoa_ple_solve_3d(anchors_3d, elev_noisy, azim_noisy)
            errors["PLE"].append(np.linalg.norm(pos - true_pos_3d))
        except Exception:
            pass

    print(f"\nRMSE over {n_trials} trials with {noise_deg} deg noise:")
    for method, errs in errors.items():
        if errs:
            rmse = np.sqrt(np.mean(np.array(errs) ** 2))
            print(f"  {method}: RMSE={rmse:.4f} m (success rate={100*len(errs)/n_trials:.0f}%)")
        else:
            print(f"  {method}: No successful trials")


def demo_geometry_sensitivity():
    """Demonstrate PLE degradation with poor geometry (aligned beacons)."""
    print("\n" + "=" * 70)
    print("Demo 7: Geometry Sensitivity (PLE Degradation)")
    print("=" * 70)
    print("\nAs noted in Section 4.4.3, PLE is biased and sensitive to:")
    print("  - Poor geometry (near-parallel bearings from aligned anchors)")
    print("  - Large bearing noise")

    true_pos = np.array([5.0, 7.0])
    np.random.seed(123)
    noise_deg = 2.0

    # Define different anchor geometries
    geometries = {
        "Square (good)": np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
        "Triangle (good)": np.array([[0, 0], [10, 0], [5, 10]], dtype=float),
        "Linear (poor)": np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=float),
        "Near-collinear (very poor)": np.array(
            [[0, 0], [5, 0.1], [10, 0], [15, 0.1]], dtype=float
        ),
    }

    print(f"\nTrue position: {true_pos}")
    print(f"Noise level: {noise_deg} deg")
    print("\n" + "-" * 80)
    print(
        f"{'Geometry':<25} {'I-WLS Error (m)':<18} {'PLE Error (m)':<18} "
        f"{'PLE Cond#':<15} {'Warning':<10}"
    )
    print("-" * 80)

    for name, anchors in geometries.items():
        # Generate angles
        azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors])
        azimuths_noisy = azimuths + np.random.randn(len(azimuths)) * np.deg2rad(noise_deg)

        # I-WLS
        aoa_meas = aoa_angle_vector(anchors, true_pos, include_elevation=False)
        aoa_noisy = aoa_meas + np.random.randn(len(aoa_meas)) * np.deg2rad(noise_deg)
        positioner = AOAPositioner(anchors)
        try:
            pos_iwls, info_iwls = positioner.solve(
                aoa_noisy, initial_guess=np.array([5.0, 5.0])
            )
            if info_iwls["converged"]:
                err_iwls = np.linalg.norm(pos_iwls - true_pos)
                iwls_str = f"{err_iwls:.4f}"
            else:
                iwls_str = "FAIL"
        except Exception:
            iwls_str = "FAIL"

        # PLE
        try:
            pos_ple, info_ple = aoa_ple_solve_2d(anchors, azimuths_noisy)
            err_ple = np.linalg.norm(pos_ple - true_pos)
            ple_str = f"{err_ple:.4f}"
            cond_str = f"{info_ple['condition_number']:.2e}"
            warn_str = "YES" if info_ple["geometry_warning"] else "no"
        except Exception:
            ple_str = "FAIL"
            cond_str = "N/A"
            warn_str = "N/A"

        print(f"{name:<25} {iwls_str:<18} {ple_str:<18} {cond_str:<15} {warn_str:<10}")

    print("\nKey observations:")
    print("  - With aligned (linear) anchors, bearings are near-parallel")
    print("  - This causes high condition number and large PLE errors")
    print("  - I-WLS is more robust but may also struggle with poor geometry")
    print("  - The 'geometry_warning' flag indicates potential issues")


def demo_ove_vs_ple_3d():
    """Compare OVE and PLE in 3D with varying noise."""
    print("\n" + "=" * 70)
    print("Demo 8: OVE vs PLE 3D Noise Sensitivity")
    print("=" * 70)

    anchors_3d = np.array(
        [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float
    )
    true_pos = np.array([5.0, 5.0, 0.0])

    elevations = np.array([aoa_elevation(a, true_pos) for a in anchors_3d])
    azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors_3d])

    print(f"\nTrue position (E, N, U): {true_pos}")
    print(f"Anchors at height 5m above agent")

    noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0]
    n_trials = 50

    print("\n" + "-" * 70)
    print(f"{'Noise (deg)':<15} {'OVE RMSE (m)':<20} {'PLE RMSE (m)':<20}")
    print("-" * 70)

    for noise_deg in noise_levels:
        ove_errors = []
        ple_errors = []

        np.random.seed(42)
        for _ in range(n_trials if noise_deg > 0 else 1):
            if noise_deg > 0:
                elev_noisy = elevations + np.random.randn(4) * np.deg2rad(noise_deg)
                azim_noisy = azimuths + np.random.randn(4) * np.deg2rad(noise_deg)
            else:
                elev_noisy = elevations
                azim_noisy = azimuths

            # OVE
            try:
                pos_ove, _ = aoa_ove_solve(anchors_3d, elev_noisy, azim_noisy)
                ove_errors.append(np.linalg.norm(pos_ove - true_pos))
            except Exception:
                pass

            # PLE
            try:
                pos_ple, _ = aoa_ple_solve_3d(anchors_3d, elev_noisy, azim_noisy)
                ple_errors.append(np.linalg.norm(pos_ple - true_pos))
            except Exception:
                pass

        ove_rmse = np.sqrt(np.mean(np.array(ove_errors) ** 2)) if ove_errors else np.nan
        ple_rmse = np.sqrt(np.mean(np.array(ple_errors) ** 2)) if ple_errors else np.nan

        print(f"{noise_deg:<15.1f} {ove_rmse:<20.4f} {ple_rmse:<20.4f}")

    print("\nNote: Both OVE and PLE are biased closed-form estimators.")
    print("OVE uses 3D geometry directly, while PLE decouples horizontal and vertical.")


def main():
    """Run all AOA positioning examples."""
    print("\n" + "=" * 70)
    print("Chapter 4: AOA Positioning Examples")
    print("=" * 70)
    print("\nENU Convention:")
    print("  - Azimuth psi: from North, positive CCW (Eq. 4.64)")
    print("  - Elevation theta: positive when anchor above agent (Eq. 4.63)")
    print("  - Measurement vector: [sin(theta_i), tan(psi_i)] stacked (Eq. 4.65)")

    # Run demos
    demo_aoa_basic()
    demo_aoa_with_noise()
    demo_measurement_vector()  # New demo for Eq. 4.65
    demo_minimum_anchors()
    demo_closed_form_algorithms()  # OVE and PLE comparison
    demo_geometry_sensitivity()  # PLE degradation with poor geometry
    demo_ove_vs_ple_3d()  # OVE vs PLE noise sensitivity
    visualize_aoa_geometry()

    print("\n" + "=" * 70)
    print("All AOA examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()



