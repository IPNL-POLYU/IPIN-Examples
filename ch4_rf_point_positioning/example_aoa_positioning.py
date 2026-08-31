"""
AOA Positioning Examples.

This script demonstrates Angle of Arrival (AOA) positioning algorithms
from Chapter 4.

Implements:
    - AOA measurement model (Eqs. 4.63-4.65)
        - Eq. 4.63: sin(theta) = (x_u^i - x_u,a) / ||x_a - x^i||
        - Eq. 4.64: tan(psi) = (x_e^i - x_e,a) / (x_n^i - x_n,a)
        - Eq. 4.65: z = [sin(theta_1), tan(psi_1), ..., sin(theta_I), tan(psi_I)]^T
    - AOA iterative LS positioning (Eqs. 4.67-4.78 at uniform weights)
    - Orthogonal Vector Estimator (OVE) - 3D closed-form (Eqs. 4.79-4.85)
    - Pseudolinear Estimator (PLE) - 2D/3D closed-form (Eqs. 4.86-4.95)

ENU Convention:
    - Azimuth psi is measured from North (+N), positive CCW
    - psi = atan2(dE, dN) where dE = anchor_E - agent_E, dN = anchor_N - agent_N
    - Elevation theta is positive when anchor is above agent

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import save_figure, show_figures_if_requested
from core.rf import (
    AOAPositioner,
    aoa_angle_vector,
    aoa_azimuth,
    aoa_elevation,
    aoa_ove_solve,
    aoa_ple_solve_2d,
    aoa_ple_solve_3d,
    aoa_sin_elevation,
    aoa_sin_tan_vector,
    aoa_tan_azimuth,
)


def demo_aoa_basic():
    """Demonstrate basic AOA positioning with iterative LS.

    `AOAPositioner(anchors)` is built with no `sigma_*`, which the class
    documents as uniform weights -- and a uniform weight matrix is a multiple
    of the identity, so it cancels out of (H' W H)^-1 H' W. Checked rather
    than assumed: no sigma and a scalar sigma give bit-identical positions,
    and only a *per-anchor* sigma moves the answer. So this is the
    Eqs. (4.63)-(4.78) solver run unweighted -- iterative LS.
    `example_comparison --compare-geometry` is where a per-anchor sigma is
    supplied and the "W" earns its letter.
    """
    print("\n" + "=" * 70)
    print("Demo 1: Basic AOA Positioning (iterative LS, uniform weights)")
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

    # Solve unweighted: no sigma given, so W is a multiple of the identity
    positioner = AOAPositioner(anchors)
    estimated_position, info = positioner.solve_angles_rad(
        aoa_measurements, initial_guess=np.array([5.0, 5.0])
    )

    print(f"\nEstimated position: {estimated_position}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Position error: {np.linalg.norm(estimated_position - true_position):.6f} m")

    return anchors, true_position, aoa_measurements


# Seed for the Monte Carlo in Demo 2.
SEED = 42


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

    # RMS over repeated draws, not one solve per noise level. A single draw
    # made this table contradict itself: it read 1.74 m at 5 deg and 2.04 m at
    # 10 deg, an apparent saturation, when AOA position error is linear in the
    # angular noise. The OVE/PLE table further down this same file always
    # averaged, and shows that linearity cleanly -- its RMSE divided by the
    # noise is constant to within 1.5%.
    trials = 300
    rng = np.random.default_rng(SEED)

    for noise_deg in noise_levels_deg:
        noise_rad = np.deg2rad(noise_deg)

        errors = []
        last_pos, last_iters = None, 0
        n_failed = 0
        for _ in range(trials if noise_deg > 0 else 1):
            aoa_noisy = aoa_true + rng.standard_normal(len(aoa_true)) * noise_rad
            positioner = AOAPositioner(anchors)
            est_pos, info = positioner.solve_angles_rad(
                aoa_noisy, initial_guess=np.array([7.5, 7.5])
            )
            if info["converged"]:
                errors.append(np.linalg.norm(est_pos - true_position))
                last_pos, last_iters = est_pos, info["iterations"]
            else:
                n_failed += 1

        if errors:
            # Median, not RMS. Averaging revealed something a single draw hid:
            # at 10 deg a handful of solves blow up to 1e14 m *while reporting
            # convergence*, and they alone move the RMS to 9e8. The median is
            # unaffected and shows the actual law -- 0.25, 1.21, 2.44 m for
            # 1, 5, 10 deg, i.e. 0.24 m per degree throughout. Both facts are
            # reported rather than one being allowed to hide the other.
            errors = np.asarray(errors)
            results.append(
                {
                    "noise": noise_deg,
                    "position": last_pos,
                    "error": float(np.median(errors)),
                    "iterations": last_iters,
                    "n_failed": n_failed,
                    "n_gross": int(np.sum(errors > 100.0)),
                }
            )
        else:
            results.append(
                {
                    "noise": noise_deg,
                    "position": None,
                    "error": np.inf,
                    "iterations": 0,
                    "n_failed": n_failed,
                    "n_gross": 0,
                }
            )

    # Print results
    print("\n" + "-" * 70)
    print(
        f"Median position error over {trials} draws per noise level "
        f"(Eq. 4.64 geometry)."
    )
    print(
        f"{'Noise (deg)':<13} {'Median err (m)':<16} {'m/deg':<9} "
        f"{'no-converge':<13} {'>100 m':<8}"
    )
    print("-" * 70)
    for r in results:
        error_str = f"{r['error']:.4f}" if r["error"] != np.inf else "FAILED"
        # Error per degree of angular noise: constant if the relationship is
        # linear, which is the check this table is worth running.
        slope = (
            f"{r['error'] / r['noise']:.4f}"
            if r["noise"] > 0 and r["error"] != np.inf
            else "-"
        )
        print(
            f"{r['noise']:<13.1f} {error_str:<16} {slope:<9} "
            f"{r['n_failed']:<13d} {r['n_gross']:<8d}"
        )

    print()
    print("  Median error is linear in the angular noise, as the geometry")
    print("  implies. The last two columns are why this table reports a median")
    print("  rather than an RMS: at 10 deg a few solves diverge, and some of")
    print("  those report convergence while landing over 100 m away, which is")
    print("  enough to move an RMS by eight orders of magnitude. Treat the")
    print("  solver's converged flag as necessary, not sufficient, at high")
    print("  angular noise.")

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
    print("Anchor positions:")
    for i, anchor in enumerate(anchors_3d):
        print(f"  Anchor {i}: {anchor}")

    # Generate measurement vector per Eq. 4.65: [sin(theta_i), tan(psi_i), ...]
    z = aoa_sin_tan_vector(anchors_3d, true_position_3d, include_elevation=True)

    print("\nMeasurement vector z (Eq. 4.65):")
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

    z_2d = aoa_sin_tan_vector(anchors_2d, true_position_2d, include_elevation=False)
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
        "4 anchors": np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
    }

    print(f"\nTrue position (E, N): {true_position}")

    for config_name, anchors in anchor_configs.items():
        print(f"\n--- {config_name} ---")

        # Generate AOA measurements using ENU convention (Eq. 4.64)
        aoa = aoa_angle_vector(anchors, true_position, include_elevation=False)

        # Try to solve
        try:
            positioner = AOAPositioner(anchors)
            est_pos, info = positioner.solve_angles_rad(
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
    est_pos, info = positioner.solve_angles_rad(
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

    # Plot bearing lines from anchors toward agent.
    #
    # psi = atan2(dE, dN) is the angle from North, but aoa_azimuth defines it
    # with dE = anchor - agent: it is the bearing *from the agent toward the
    # anchor*. Drawing (sin psi, cos psi) outward from the anchor therefore
    # points directly away from the agent, which is what this figure did --
    # all four rays left the plot on the far side, and none passed through the
    # position they are supposed to intersect at.
    #
    # Negating gives the anchor-to-agent direction, so the rays now meet at
    # the target the way an AOA geometry figure is meant to show.
    for anchor, psi in zip(anchors, aoa_noisy, strict=True):
        # In ENU: psi is from North (+y), so the agent lies from the anchor at
        # E-component = -sin(psi), N-component = -cos(psi)
        line_length = 15
        end_e = anchor[0] - line_length * np.sin(psi)
        end_n = anchor[1] - line_length * np.cos(psi)
        ax.plot(
            [anchor[0], end_e],
            [anchor[1], end_n],
            "--",
            color="gray",
            alpha=0.6,
            linewidth=1,
        )
        # Label angle, placed along the ray it belongs to
        ax.text(
            anchor[0] - 2 * np.sin(psi),
            anchor[1] - 2 * np.cos(psi),
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
    paths = save_figure(fig, Path(__file__).parent / "figs", "ch4_aoa_geometry")
    print(f"\nFigure saved: {paths[0]}")

    show_figures_if_requested()

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
    print("  - I-LS: the Eqs. (4.63)-(4.78) iterative solver, run unweighted")
    print("  - OVE: Orthogonal Vector Estimator, 3D (Eqs. 4.79-4.85)")
    print("  - PLE: Pseudolinear Estimator, 2D (Eqs. 4.86-4.91)")

    # === 2D Comparison ===
    print("\n--- 2D Comparison (I-LS vs PLE) ---")
    anchors_2d = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    true_pos_2d = np.array([4.0, 6.0])

    # Generate azimuth angles
    azimuths = np.array([aoa_azimuth(a, true_pos_2d) for a in anchors_2d])

    print(f"\nAnchors: {anchors_2d.tolist()}")
    print(f"True position (E, N): {true_pos_2d}")

    # Iterative LS (unweighted)
    aoa_meas = aoa_angle_vector(anchors_2d, true_pos_2d, include_elevation=False)
    positioner = AOAPositioner(anchors_2d)
    pos_ils, info_ils = positioner.solve_angles_rad(
        aoa_meas, initial_guess=np.array([5.0, 5.0])
    )
    err_ils = np.linalg.norm(pos_ils - true_pos_2d)

    # PLE 2D
    pos_ple, info_ple = aoa_ple_solve_2d(anchors_2d, azimuths)
    err_ple = np.linalg.norm(pos_ple - true_pos_2d)

    print("\nResults (perfect measurements):")
    print(
        f"  I-LS:  pos={pos_ils}, error={err_ils:.6f} m, iters={info_ils['iterations']}"
    )
    print(f"  PLE:   pos={pos_ple}, error={err_ple:.6f} m (closed-form)")

    # === 3D Comparison ===
    print("\n--- 3D Comparison (I-LS vs OVE vs PLE) ---")
    anchors_3d = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
    true_pos_3d = np.array([4.0, 6.0, 0.0])

    # Generate angles
    elevations = np.array([aoa_elevation(a, true_pos_3d) for a in anchors_3d])
    azimuths_3d = np.array([aoa_azimuth(a, true_pos_3d) for a in anchors_3d])

    print(f"\nAnchors (3D): {anchors_3d.tolist()}")
    print(f"True position (E, N, U): {true_pos_3d}")

    # Iterative LS, 3D (unweighted)
    aoa_meas_3d = aoa_angle_vector(anchors_3d, true_pos_3d, include_elevation=True)
    positioner_3d = AOAPositioner(anchors_3d)
    pos_ils_3d, info_ils_3d = positioner_3d.solve(
        aoa_meas_3d, initial_guess=np.array([5.0, 5.0, 1.0])
    )
    err_ils_3d = np.linalg.norm(pos_ils_3d - true_pos_3d)

    # OVE 3D
    pos_ove, info_ove = aoa_ove_solve(anchors_3d, elevations, azimuths_3d)
    err_ove = np.linalg.norm(pos_ove - true_pos_3d)

    # PLE 3D
    pos_ple_3d, info_ple_3d = aoa_ple_solve_3d(anchors_3d, elevations, azimuths_3d)
    err_ple_3d = np.linalg.norm(pos_ple_3d - true_pos_3d)

    print("\nResults (perfect measurements):")
    print(
        f"  I-LS:  pos={pos_ils_3d}, error={err_ils_3d:.6f} m, "
        f"iters={info_ils_3d['iterations']}"
    )
    print(f"  OVE:   pos={pos_ove}, error={err_ove:.6f} m (closed-form)")
    print(f"  PLE:   pos={pos_ple_3d}, error={err_ple_3d:.6f} m (closed-form)")

    # === With noise ===
    print("\n--- With 2 deg measurement noise (Monte Carlo, 100 trials) ---")
    np.random.seed(42)
    noise_deg = 2.0
    n_trials = 100
    errors = {"I-LS": [], "OVE": [], "PLE": []}

    for _ in range(n_trials):
        # Add noise
        elev_noisy = elevations + np.random.randn(len(elevations)) * np.deg2rad(
            noise_deg
        )
        azim_noisy = azimuths_3d + np.random.randn(len(azimuths_3d)) * np.deg2rad(
            noise_deg
        )

        # Iterative LS (unweighted)
        aoa_noisy = np.zeros(2 * len(anchors_3d))
        for i in range(len(anchors_3d)):
            aoa_noisy[2 * i] = elev_noisy[i]
            aoa_noisy[2 * i + 1] = azim_noisy[i]
        try:
            pos, info = positioner_3d.solve(
                aoa_noisy, initial_guess=np.array([5.0, 5.0, 1.0])
            )
            if info["converged"]:
                errors["I-LS"].append(np.linalg.norm(pos - true_pos_3d))
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
            print(
                f"  {method}: RMSE={rmse:.4f} m (success rate={100*len(errs)/n_trials:.0f}%)"
            )
        else:
            print(f"  {method}: No successful trials")


#: Draws per row in Demo 7. One draw per row, to four decimals, is what this
#: table used to report -- and it drew a *separate* one for each of its two
#: columns, so the two solvers were never compared on the same measurements.
GEOMETRY_TRIALS = 500

#: Where Demo 7's second table walks the agent. Anchors on a line do not by
#: themselves make bearings near-parallel; standing near that line does.
LINE_WALK = ((5.0, 7.0), (5.0, 0.5), (30.0, 0.2))


def _bearing_spread_deg(azimuths):
    """Smallest pairwise bearing separation, in degrees, folded to [0, 90].

    This is the quantity ``aoa_ple_solve_2d`` thresholds at 10 degrees for its
    ``geometry_warning``, recomputed here so the table can show what the flag
    is looking at rather than only what it decided.
    """
    diffs = [
        min(
            abs(azimuths[i] - azimuths[j]),
            np.pi - abs(azimuths[i] - azimuths[j]) % np.pi,
        )
        for i in range(len(azimuths))
        for j in range(i + 1, len(azimuths))
    ]
    return float(np.rad2deg(min(diffs))) if diffs else 90.0


def _aoa_geometry_row(anchors, true_pos, noise_deg, trials, seed=2024):
    """Median I-LS and PLE error over `trials` draws of the SAME bearings.

    Both solvers see one noise realisation per trial. They used to be handed
    independent draws -- `azimuths_noisy` for PLE and a freshly drawn
    `aoa_noisy` for I-LS -- so the two columns of a row differed by the noise
    as much as by the method, at one draw each.
    """
    rng = np.random.default_rng(seed)
    psi_true = np.array([aoa_azimuth(a, true_pos) for a in anchors])
    err_ils, err_ple, n_failed = [], [], 0

    for _ in range(trials):
        psi_noisy = psi_true + rng.standard_normal(len(psi_true)) * np.deg2rad(
            noise_deg
        )
        try:
            pos_ils, info_ils = AOAPositioner(anchors).solve_angles_rad(
                psi_noisy, initial_guess=np.array([5.0, 5.0])
            )
            if info_ils["converged"]:
                err_ils.append(float(np.linalg.norm(pos_ils - true_pos)))
            else:
                n_failed += 1
        except Exception:  # noqa: BLE001 - counted as a failure, not hidden
            n_failed += 1
        pos_ple, _ = aoa_ple_solve_2d(anchors, psi_noisy)
        err_ple.append(float(np.linalg.norm(pos_ple - true_pos)))

    _, info_clean = aoa_ple_solve_2d(anchors, psi_true)
    return {
        "ils": float(np.median(err_ils)) if err_ils else float("inf"),
        "ple": float(np.median(err_ple)),
        "ple_errors": np.asarray(err_ple),
        "cond": float(info_clean["condition_number"]),
        "warning": bool(info_clean["geometry_warning"]),
        "spread_deg": _bearing_spread_deg(psi_true),
        "n_failed": n_failed,
        "mean_range_m": float(np.linalg.norm(anchors - true_pos, axis=1).mean()),
    }


def demo_geometry_sensitivity():
    """Demo 7: what geometry actually costs a bearing-only fix.

    This demo used to be titled "PLE degradation" and print, under its own
    table, that aligned anchors "cause high condition number and large PLE
    errors". Its four rows said otherwise every time it ran: the near-collinear
    array gave the *smallest* PLE error of the four, the square had the lowest
    condition number, and the ``geometry_warning`` column read "no" in every
    row. Two defects were hiding that.

    **The two columns were different experiments.** PLE was solved on one draw
    of ``azimuths_noisy`` and I-LS on a *separately drawn* ``aoa_noisy``, so
    the comparison carried a full realisation of noise on each side, at one
    draw per row. Both now solve the same bearings, and the table reports a
    median over 500 draws.

    **Collinear anchors are not the same thing as near-parallel bearings.**
    From the target at (5, 7) the four "poor" arrays still spread their
    bearings by 19-20 degrees, which is why the warning never fired: it
    thresholds the *bearing* spread at the agent, not the anchor layout. On
    equal draws all four geometries land within 30% of each other, and that is
    the honest headline -- bearings tolerate a collinear array, exactly as
    ``example_comparison --compare-geometry`` reports (AOA is the *best* of
    the three methods on the collinear beacons, where ranges cannot separate a
    target from its mirror image).

    The second table is where the warning earns itself: walking the agent onto
    the beacon line collapses the spread, and what it costs scales with range.
    """
    print("\n" + "=" * 70)
    print("Demo 7: Geometry Sensitivity (what actually costs a bearing fix)")
    print("=" * 70)
    print("\nSection 4.4.3 lists PLE as sensitive to poor geometry and to large")
    print("bearing noise. This demo measures the first of those.")

    true_pos = np.array([5.0, 7.0])
    noise_deg = 2.0

    geometries = {
        "Square (spread out)": np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float),
        "Triangle": np.array([[0, 0], [10, 0], [5, 10]], float),
        "Linear anchors": np.array([[0, 0], [5, 0], [10, 0], [15, 0]], float),
        "Near-collinear anchors": np.array(
            [[0, 0], [5, 0.1], [10, 0], [15, 0.1]], float
        ),
    }

    print(f"\nTrue position: {true_pos}")
    print(f"Bearing noise: {noise_deg} deg, {GEOMETRY_TRIALS} draws per row")
    print("Both columns of a row solve the SAME draws.")
    print("\n" + "-" * 94)
    print(
        f"{'Anchor layout':<24} {'I-LS med (m)':<14} {'PLE med (m)':<13} "
        f"{'PLE cond#':<11} {'spread (deg)':<14} {'warning':<9} {'I-LS fail':<9}"
    )
    print("-" * 94)

    rows = {}
    for name, anchors in geometries.items():
        row = _aoa_geometry_row(anchors, true_pos, noise_deg, GEOMETRY_TRIALS)
        rows[name] = row
        print(
            f"{name:<24} {row['ils']:<14.4f} {row['ple']:<13.4f} "
            f"{row['cond']:<11.2f} {row['spread_deg']:<14.1f} "
            f"{('YES' if row['warning'] else 'no'):<9} {row['n_failed']:<9d}"
        )

    ple_values = [r["ple"] for r in rows.values()]
    print(
        f"\nThe four PLE medians span {min(ple_values):.4f}-{max(ple_values):.4f} m, "
        f"a factor of {max(ple_values) / min(ple_values):.2f}. Collinear anchors"
    )
    print("do not degrade a bearing fix: every bearing still points somewhere")
    print("different, because the agent is 7 m off the line. The minimum")
    print("pairwise spread is 19.5 deg there against the 10 deg the warning")
    print("thresholds -- so 'no' in that column is the flag being right.")
    print("This is the same result --compare-geometry reports from the other")
    print("side: on collinear beacons AOA is the best of the three methods,")
    print("because reflecting a position flips every azimuth while leaving")
    print("every range unchanged.")

    # Where near-parallel bearings do cost something: walk the agent in.
    print("\n" + "-" * 94)
    print("The same linear array, with the agent walked toward and along the line:")
    print("-" * 94)
    print(
        f"{'Agent':<14} {'range (m)':<11} {'spread (deg)':<14} {'warn':<6} "
        f"{'PLE med (m)':<13} {'along bias':<12} {'along sd':<10} {'cross sd':<10}"
    )
    print("-" * 94)

    linear = geometries["Linear anchors"]
    for point in LINE_WALK:
        agent = np.array(point, dtype=float)
        row = _aoa_geometry_row(linear, agent, noise_deg, GEOMETRY_TRIALS)
        rng = np.random.default_rng(2024)
        psi_true = np.array([aoa_azimuth(a, agent) for a in linear])
        residuals = np.array(
            [
                aoa_ple_solve_2d(
                    linear,
                    psi_true
                    + rng.standard_normal(len(psi_true)) * np.deg2rad(noise_deg),
                )[0]
                - agent
                for _ in range(GEOMETRY_TRIALS)
            ]
        )
        print(
            f"{str(tuple(point)):<14} {row['mean_range_m']:<11.1f} "
            f"{row['spread_deg']:<14.2f} {('YES' if row['warning'] else 'no'):<6} "
            f"{row['ple']:<13.4f} {residuals[:, 0].mean():<+12.3f} "
            f"{residuals[:, 0].std():<10.3f} {residuals[:, 1].std():<10.3f}"
        )

    print("\nKey observations:")
    print("  - What a bearing fix needs is spread between its bearings, and")
    print("    that is a property of where the AGENT stands, not only of how")
    print("    the anchors are laid out.")
    print("  - A small spread is cheap up close and ruinous far away. On the")
    print("    line but inside the array the spread is 2.85 deg and PLE still")
    print("    reaches 0.10 m; 30 m out the spread is 0.08 deg and the median")
    print("    error is 22 m. sigma_position ~ range x sigma_angle / spread,")
    print("    and the table above moves the way that says it should.")
    print("  - Read the last row's three right-hand columns together. The")
    print("    22 m is almost all BIAS along the beacon line (-22.3 m), with")
    print("    4.6 m of scatter beside it, while across the line the estimate")
    print("    is good to 0.13 m. A near-parallel set of bearings pins you")
    print("    across their common direction and says almost nothing along it,")
    print("    so the estimator collapses back toward the array. A single")
    print("    Euclidean error cannot say that, which is why the split is")
    print("    printed: a 22 m bias and 22 m of noise are different failures.")
    print("  - 'geometry_warning' fires on the middle and last rows and not on")
    print("    the first, which is the flag doing its job. It is a warning")
    print("    about conditioning, not a prediction of error: read it beside")
    print("    the range, not on its own.")


def demo_ove_vs_ple_3d():
    """Compare OVE and PLE in 3D with varying noise."""
    print("\n" + "=" * 70)
    print("Demo 8: OVE vs PLE 3D Noise Sensitivity")
    print("=" * 70)

    anchors_3d = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
    true_pos = np.array([5.0, 5.0, 0.0])

    elevations = np.array([aoa_elevation(a, true_pos) for a in anchors_3d])
    azimuths = np.array([aoa_azimuth(a, true_pos) for a in anchors_3d])

    print(f"\nTrue position (E, N, U): {true_pos}")
    print("Anchors at height 5m above agent")

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
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

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
