"""
Comparison of RF Positioning Methods.

This script compares TOA, TDOA, AOA, and RSS positioning methods
under various conditions.

Implements:
    - TOA positioning (Eqs. 4.14-4.23)
    - TDOA positioning (Eqs. 4.34-4.42)
    - AOA positioning (Eqs. 4.63-4.67)
    - RSS positioning (Eqs. 4.11-4.13)

Author: Navigation Engineering Team
Date: December 2025
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.rf import (
    AOAPositioner,
    TDOAPositioner,
    TOAPositioner,
    aoa_azimuth,
    rss_pathloss,
    rss_to_distance,
    toa_range,
)


def generate_scenario(seed=42):
    """Generate a test scenario with anchors and true positions."""
    np.random.seed(seed)

    # Square anchor layout (10m x 10m area)
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    # Generate test positions
    n_points = 50
    x = np.random.uniform(1, 9, n_points)
    y = np.random.uniform(1, 9, n_points)
    true_positions = np.column_stack([x, y])

    return anchors, true_positions


def toa_positioning_test(anchors, true_positions, noise_std=0.0):
    """Test TOA positioning."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  TOA", leave=False, unit="pt"):
        # Generate TOA measurements
        ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])

        # Add noise
        if noise_std > 0:
            ranges += np.random.randn(len(ranges)) * noise_std

        # Solve
        try:
            positioner = TOAPositioner(anchors, method="iwls")
            est_pos, info = positioner.solve(
                ranges, initial_guess=np.array([5.0, 5.0])
            )

            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def tdoa_positioning_test(anchors, true_positions, noise_std=0.0):
    """Test TDOA positioning."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  TDOA", leave=False, unit="pt"):
        # Generate TDOA measurements
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        # Add noise
        if noise_std > 0:
            tdoa += np.random.randn(len(tdoa)) * noise_std

        # Solve
        try:
            positioner = TDOAPositioner(anchors, reference_idx=0)
            est_pos, info = positioner.solve(
                tdoa, initial_guess=np.array([5.0, 5.0])
            )

            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def aoa_positioning_test(anchors, true_positions, noise_std=0.0):
    """Test AOA positioning."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  AOA", leave=False, unit="pt"):
        # Generate AOA measurements
        aoa = np.array([aoa_azimuth(anchor, true_pos) for anchor in anchors])

        # Add noise (in radians)
        if noise_std > 0:
            aoa += np.random.randn(len(aoa)) * noise_std

        # Solve
        try:
            positioner = AOAPositioner(anchors)
            est_pos, info = positioner.solve(
                aoa, initial_guess=np.array([5.0, 5.0])
            )

            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def rss_positioning_test(
    anchors, true_positions, rss_noise_std=0.0, path_loss_exp=2.5
):
    """Test RSS positioning."""
    errors = []
    tx_power_dbm = 0.0

    for true_pos in tqdm(true_positions, desc="  RSS", leave=False, unit="pt"):
        # Generate RSS measurements
        rss = []
        for anchor in anchors:
            dist = np.linalg.norm(anchor - true_pos)
            rss_val = rss_pathloss(tx_power_dbm, dist, path_loss_exp)
            rss.append(rss_val)
        rss = np.array(rss)

        # Add noise (in dB)
        if rss_noise_std > 0:
            rss += np.random.randn(len(rss)) * rss_noise_std

        # Convert RSS to ranges
        ranges = np.array(
            [
                rss_to_distance(r, tx_power_dbm, path_loss_exp)
                for r in rss
            ]
        )

        # Solve using TOA
        try:
            positioner = TOAPositioner(anchors, method="iwls")
            est_pos, info = positioner.solve(
                ranges, initial_guess=np.array([5.0, 5.0])
            )

            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def compare_methods():
    """Compare all RF positioning methods."""
    print("=" * 70)
    print("RF Positioning Methods Comparison")
    print("=" * 70)

    # Generate scenario
    print("\n--- Setting up test scenario ---")
    anchors, true_positions = generate_scenario(seed=42)
    print(f"✓ Test scenario created:")
    print(f"  Anchors: {len(anchors)}")
    print(f"  Test points: {len(true_positions)}")
    print(f"  Area: 10m x 10m")

    # Test different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = {
        "TOA": [],
        "TDOA": [],
        "AOA": [],
        "RSS": [],
    }

    print("\nTesting noise levels...")
    start_time = time.time()
    
    for i, noise in enumerate(tqdm(noise_levels, desc="Overall progress", unit="level")):
        print(f"\n[{i+1}/{len(noise_levels)}] Noise: {noise:.2f} m (TOA/TDOA), {np.rad2deg(noise):.2f}° (AOA), {noise*30:.1f} dB (RSS)")

        # TOA
        toa_errors = toa_positioning_test(anchors, true_positions, noise)
        results["TOA"].append(toa_errors)

        # TDOA
        tdoa_errors = tdoa_positioning_test(anchors, true_positions, noise)
        results["TDOA"].append(tdoa_errors)

        # AOA (convert distance noise to angle noise approximately)
        angle_noise = noise / 5.0  # rough conversion at 5m distance
        aoa_errors = aoa_positioning_test(
            anchors, true_positions, angle_noise
        )
        results["AOA"].append(aoa_errors)

        # RSS (convert distance noise to dB noise approximately)
        rss_noise = noise * 30  # rough conversion (10*2.5*log10(noise ratio))
        rss_errors = rss_positioning_test(
            anchors, true_positions, rss_noise
        )
        results["RSS"].append(rss_errors)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ All tests completed in {elapsed_time:.2f}s")

    # Print statistics
    print("\n" + "=" * 70)
    print("Results Summary (RMSE in meters)")
    print("=" * 70)
    print(f"{'Noise':<12} {'TOA':<12} {'TDOA':<12} {'AOA':<12} {'RSS':<12}")
    print("-" * 70)

    for i, noise in enumerate(noise_levels):
        noise_str = f"{noise:.2f} m"
        toa_rmse = (
            np.sqrt(np.mean(results["TOA"][i] ** 2))
            if len(results["TOA"][i]) > 0
            else np.nan
        )
        tdoa_rmse = (
            np.sqrt(np.mean(results["TDOA"][i] ** 2))
            if len(results["TDOA"][i]) > 0
            else np.nan
        )
        aoa_rmse = (
            np.sqrt(np.mean(results["AOA"][i] ** 2))
            if len(results["AOA"][i]) > 0
            else np.nan
        )
        rss_rmse = (
            np.sqrt(np.mean(results["RSS"][i] ** 2))
            if len(results["RSS"][i]) > 0
            else np.nan
        )

        print(
            f"{noise_str:<12} {toa_rmse:<12.3f} {tdoa_rmse:<12.3f} "
            f"{aoa_rmse:<12.3f} {rss_rmse:<12.3f}"
        )

    return noise_levels, results


def plot_comparison(noise_levels, results):
    """Plot comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "RF Positioning Methods Comparison", fontsize=16, fontweight="bold"
    )

    methods = ["TOA", "TDOA", "AOA", "RSS"]
    colors = ["blue", "red", "green", "orange"]

    # 1. RMSE vs Noise
    ax1 = axes[0, 0]
    for method, color in zip(methods, colors):
        rmse_values = []
        for errors in results[method]:
            if len(errors) > 0:
                rmse = np.sqrt(np.mean(errors**2))
                rmse_values.append(rmse)
            else:
                rmse_values.append(np.nan)

        ax1.plot(
            noise_levels, rmse_values, "o-", label=method, color=color, linewidth=2
        )

    ax1.set_xlabel("Measurement Noise (m)", fontsize=12)
    ax1.set_ylabel("RMSE (m)", fontsize=12)
    ax1.set_title("RMSE vs Measurement Noise", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Error CDF (at medium noise level)
    ax2 = axes[0, 1]
    noise_idx = 2  # 0.1m noise
    for method, color in zip(methods, colors):
        errors = results[method][noise_idx]
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(sorted_errors, cdf, label=method, color=color, linewidth=2)

    ax2.set_xlabel("Position Error (m)", fontsize=12)
    ax2.set_ylabel("CDF", fontsize=12)
    ax2.set_title(
        f"Error CDF (Noise = {noise_levels[noise_idx]:.2f}m)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)

    # 3. Error Distribution (boxplot)
    ax3 = axes[1, 0]
    noise_idx = 2  # 0.1m noise
    data_for_boxplot = []
    labels_for_boxplot = []
    for method in methods:
        errors = results[method][noise_idx]
        if len(errors) > 0:
            data_for_boxplot.append(errors)
            labels_for_boxplot.append(method)

    bp = ax3.boxplot(
        data_for_boxplot,
        labels=labels_for_boxplot,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors[: len(data_for_boxplot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_ylabel("Position Error (m)", fontsize=12)
    ax3.set_title(
        f"Error Distribution (Noise = {noise_levels[noise_idx]:.2f}m)",
        fontsize=13,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Success Rate vs Noise
    ax4 = axes[1, 1]
    for method, color in zip(methods, colors):
        success_rates = []
        total_points = 50  # from generate_scenario
        for errors in results[method]:
            success_rate = len(errors) / total_points * 100
            success_rates.append(success_rate)

        ax4.plot(
            noise_levels,
            success_rates,
            "o-",
            label=method,
            color=color,
            linewidth=2,
        )

    ax4.set_xlabel("Measurement Noise (m)", fontsize=12)
    ax4.set_ylabel("Success Rate (%)", fontsize=12)
    ax4.set_title("Convergence Success Rate", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    return fig


def main():
    """Run RF positioning comparison."""
    overall_start = time.time()
    
    print("\n" + "=" * 70)
    print("Chapter 4: RF Positioning Methods Comparison")
    print("=" * 70)

    # Run comparison
    noise_levels, results = compare_methods()

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    plot_start = time.time()
    fig = plot_comparison(noise_levels, results)

    # Save figure
    output_file = "ch4_rf_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plot_time = time.time() - plot_start
    print(f"✓ Figure saved: {output_file} (plotting took {plot_time:.2f}s)")

    plt.show()

    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("Comparison completed successfully!")
    print("=" * 70)
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()


