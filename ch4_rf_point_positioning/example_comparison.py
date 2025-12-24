"""
Comparison of RF Positioning Methods.

This script compares TOA, TDOA, AOA, and RSS positioning methods
under various conditions using pre-generated datasets.

Can run with:
    - Pre-generated dataset: python example_comparison.py --data ch4_rf_2d_square
    - Inline data (default): python example_comparison.py
    - Compare geometries: python example_comparison.py --compare-geometry

Implements:
    - TOA positioning (Eqs. 4.14-4.23)
    - TDOA positioning (Eqs. 4.34-4.42)
    - AOA positioning (Eqs. 4.63-4.67)
    - DOP analysis (Section 4.5)

Author: Navigation Engineering Team
Date: December 2025
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_rf_dataset(data_dir: str) -> Dict:
    """Load RF positioning dataset.
    
    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch4_rf_2d_square')
    
    Returns:
        Dictionary with beacons, positions, measurements, and config
    """
    path = Path(data_dir)
    
    data = {
        'beacons': np.loadtxt(path / 'beacons.txt'),
        'positions': np.loadtxt(path / 'ground_truth_positions.txt'),
        'toa_ranges': np.loadtxt(path / 'toa_ranges.txt'),
        'tdoa_diffs': np.loadtxt(path / 'tdoa_diffs.txt'),
        'aoa_angles': np.loadtxt(path / 'aoa_angles.txt'),
        'gdop_toa': np.loadtxt(path / 'gdop_toa.txt'),
        'gdop_tdoa': np.loadtxt(path / 'gdop_tdoa.txt'),
        'gdop_aoa': np.loadtxt(path / 'gdop_aoa.txt'),
    }
    
    with open(path / 'config.json') as f:
        data['config'] = json.load(f)
    
    return data


def run_with_dataset(data_dir: str, verbose: bool = True) -> Dict:
    """Run RF positioning comparison using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
        verbose: Print detailed output
    
    Returns:
        Dictionary with results for each method
    """
    if verbose:
        print("=" * 70)
        print("Chapter 4: RF Positioning Methods Comparison")
        print(f"Using dataset: {data_dir}")
        print("=" * 70)
    
    # Load dataset
    data = load_rf_dataset(data_dir)
    config = data['config']
    
    beacons = data['beacons']
    positions = data['positions']
    n_points = len(positions)
    
    if verbose:
        print(f"\nDataset Info:")
        print(f"  Geometry: {config.get('geometry', {}).get('type', 'unknown')}")
        print(f"  Beacons: {len(beacons)}")
        print(f"  Test points: {n_points}")
        print(f"  TOA noise: {config.get('measurements', {}).get('toa_noise_std_m', 'N/A')} m")
        print(f"  AOA noise: {config.get('measurements', {}).get('aoa_noise_std_deg', 'N/A')}°")
    
    results = {
        'TOA': {'errors': [], 'converged': 0},
        'TDOA': {'errors': [], 'converged': 0},
        'AOA': {'errors': [], 'converged': 0},
    }
    
    # Run TOA positioning
    if verbose:
        print("\n--- Running TOA Positioning ---")
    toa_positioner = TOAPositioner(beacons, method="iterative_ls")
    
    for i in tqdm(range(n_points), desc="TOA", disable=not verbose):
        try:
            est_pos, info = toa_positioner.solve(
                data['toa_ranges'][i],
                initial_guess=np.mean(beacons, axis=0)
            )
            if info["converged"]:
                error = np.linalg.norm(est_pos - positions[i])
                results['TOA']['errors'].append(error)
                results['TOA']['converged'] += 1
        except Exception:
            pass
    
    # Run TDOA positioning
    if verbose:
        print("\n--- Running TDOA Positioning ---")
    tdoa_positioner = TDOAPositioner(beacons, reference_idx=0)
    
    for i in tqdm(range(n_points), desc="TDOA", disable=not verbose):
        try:
            est_pos, info = tdoa_positioner.solve(
                data['tdoa_diffs'][i],
                initial_guess=np.mean(beacons, axis=0)
            )
            if info["converged"]:
                error = np.linalg.norm(est_pos - positions[i])
                results['TDOA']['errors'].append(error)
                results['TDOA']['converged'] += 1
        except Exception:
            pass
    
    # Run AOA positioning
    if verbose:
        print("\n--- Running AOA Positioning ---")
    aoa_positioner = AOAPositioner(beacons)
    
    for i in tqdm(range(n_points), desc="AOA", disable=not verbose):
        try:
            est_pos, info = aoa_positioner.solve(
                data['aoa_angles'][i],
                initial_guess=np.mean(beacons, axis=0)
            )
            if info["converged"]:
                error = np.linalg.norm(est_pos - positions[i])
                results['AOA']['errors'].append(error)
                results['AOA']['converged'] += 1
        except Exception:
            pass
    
    # Convert to numpy arrays
    for method in results:
        results[method]['errors'] = np.array(results[method]['errors'])
    
    # Add GDOP info
    results['gdop'] = {
        'TOA': data['gdop_toa'],
        'TDOA': data['gdop_tdoa'],
        'AOA': data['gdop_aoa'],
    }
    results['n_points'] = n_points
    results['beacons'] = beacons
    results['positions'] = positions
    results['config'] = config
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        print(f"{'Method':<10} {'RMSE (m)':<12} {'Mean (m)':<12} {'Max (m)':<12} {'Converged':<12} {'GDOP (mean)':<12}")
        print("-" * 70)
        
        for method in ['TOA', 'TDOA', 'AOA']:
            errors = results[method]['errors']
            if len(errors) > 0:
                rmse = np.sqrt(np.mean(errors**2))
                mean_err = np.mean(errors)
                max_err = np.max(errors)
            else:
                rmse = mean_err = max_err = np.nan
            
            gdop_mean = np.mean(results['gdop'][method])
            conv_rate = results[method]['converged'] / n_points * 100
            
            print(f"{method:<10} {rmse:<12.3f} {mean_err:<12.3f} {max_err:<12.3f} "
                  f"{conv_rate:<12.1f}% {gdop_mean:<12.2f}")
    
    return results


def compare_geometries(verbose: bool = True) -> Dict:
    """Compare positioning performance across different beacon geometries.
    
    Uses ch4_rf_2d_square, ch4_rf_2d_optimal, and ch4_rf_2d_linear datasets.
    """
    if verbose:
        print("=" * 70)
        print("Chapter 4: Beacon Geometry Comparison")
        print("=" * 70)
    
    geometries = [
        ('ch4_rf_2d_square', 'Square (4 corners)'),
        ('ch4_rf_2d_optimal', 'Optimal (circular)'),
        ('ch4_rf_2d_linear', 'Linear (poor)'),
    ]
    
    all_results = {}
    
    for dataset_name, geometry_label in geometries:
        data_path = Path("data/sim") / dataset_name
        if not data_path.exists():
            if verbose:
                print(f"\nSkipping {dataset_name} (not found)")
            continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Geometry: {geometry_label}")
            print(f"{'='*70}")
        
        results = run_with_dataset(str(data_path), verbose=False)
        all_results[geometry_label] = results
        
        # Print summary for this geometry
        if verbose:
            for method in ['TOA', 'TDOA', 'AOA']:
                errors = results[method]['errors']
                if len(errors) > 0:
                    rmse = np.sqrt(np.mean(errors**2))
                    gdop = np.mean(results['gdop'][method])
                    print(f"  {method}: RMSE={rmse:.3f}m, GDOP={gdop:.2f}")
    
    if verbose and len(all_results) > 0:
        print("\n" + "=" * 70)
        print("KEY INSIGHT: Geometry Impact on TOA RMSE")
        print("=" * 70)
        for geom, res in all_results.items():
            if len(res['TOA']['errors']) > 0:
                rmse = np.sqrt(np.mean(res['TOA']['errors']**2))
                gdop = np.mean(res['gdop']['TOA'])
                print(f"  {geom}: {rmse:.3f}m (GDOP={gdop:.2f})")
        print("\nGeometry can cause 10× variation in positioning accuracy!")
    
    return all_results


def generate_scenario(seed=42):
    """Generate a test scenario with anchors and true positions (inline mode)."""
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
    """Test TOA positioning (inline mode)."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  TOA", leave=False, unit="pt"):
        ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
        if noise_std > 0:
            ranges += np.random.randn(len(ranges)) * noise_std

        try:
            positioner = TOAPositioner(anchors, method="iterative_ls")
            est_pos, info = positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def tdoa_positioning_test(anchors, true_positions, noise_std=0.0):
    """Test TDOA positioning (inline mode)."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  TDOA", leave=False, unit="pt"):
        dist_ref = np.linalg.norm(true_pos - anchors[0])
        tdoa = []
        for i in range(1, len(anchors)):
            dist_i = np.linalg.norm(true_pos - anchors[i])
            tdoa.append(dist_i - dist_ref)
        tdoa = np.array(tdoa)

        if noise_std > 0:
            tdoa += np.random.randn(len(tdoa)) * noise_std

        try:
            positioner = TDOAPositioner(anchors, reference_idx=0)
            est_pos, info = positioner.solve(tdoa, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def aoa_positioning_test(anchors, true_positions, noise_std=0.0):
    """Test AOA positioning (inline mode)."""
    errors = []

    for true_pos in tqdm(true_positions, desc="  AOA", leave=False, unit="pt"):
        aoa = np.array([aoa_azimuth(anchor, true_pos) for anchor in anchors])
        if noise_std > 0:
            aoa += np.random.randn(len(aoa)) * noise_std

        try:
            positioner = AOAPositioner(anchors)
            est_pos, info = positioner.solve(aoa, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def rss_positioning_test(
    anchors,
    true_positions,
    sigma_long_db=0.0,
    sigma_short_linear=0.0,
    n_samples_avg=1,
    short_fading_model="rayleigh",
    path_loss_exp=2.5,
):
    """
    Test RSS positioning with fading noise per book model (Eqs. 4.10-4.13).

    Args:
        anchors: Anchor positions.
        true_positions: True agent positions to test.
        sigma_long_db: Long-term fading std in dB (per Eq. 4.12).
                      Typical indoor values: 4-8 dB.
        sigma_short_linear: Short-term fading parameter (Rayleigh scale sigma).
                           For Rayleigh: typical σ = 0.5-1.0. Defaults to 0.0.
        n_samples_avg: Number of samples to average for short-term fading
                      reduction. Defaults to 1 (no averaging).
        short_fading_model: Short-term fading model ("rayleigh", "gaussian_db", "none").
                           Defaults to "rayleigh".
        path_loss_exp: Path-loss exponent (eta). Defaults to 2.5.

    Returns:
        Array of position errors.
    """
    from core.rf import simulate_rss_measurement

    errors = []
    p_ref_dbm = -40.0  # Reference RSS at d_ref=1m (typical Wi-Fi beacon)

    for true_pos in tqdm(true_positions, desc="  RSS", leave=False, unit="pt"):
        ranges = []
        for anchor in anchors:
            # Use simulate_rss_measurement for full fading model (Eq. 4.12)
            rss_meas, info = simulate_rss_measurement(
                anchor, true_pos,
                p_ref_dbm=p_ref_dbm,
                path_loss_exp=path_loss_exp,
                sigma_long_db=sigma_long_db,
                sigma_short_linear=sigma_short_linear,
                n_samples_avg=n_samples_avg,
                short_fading_model=short_fading_model,
            )
            # Invert RSS to range (Eq. 4.11)
            range_est = rss_to_distance(rss_meas, p_ref_dbm, path_loss_exp)
            ranges.append(range_est)

        ranges = np.array(ranges)

        try:
            positioner = TOAPositioner(anchors, method="iterative_ls")
            est_pos, info = positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def run_inline_comparison():
    """Run comparison with inline generated data (original behavior)."""
    print("=" * 70)
    print("RF Positioning Methods Comparison")
    print("(Using inline generated data)")
    print("=" * 70)

    print("\n--- Setting up test scenario ---")
    anchors, true_positions = generate_scenario(seed=42)
    print(f"Test scenario created:")
    print(f"  Anchors: {len(anchors)}")
    print(f"  Test points: {len(true_positions)}")
    print(f"  Area: 10m x 10m")

    # Noise levels for different methods
    # TOA/TDOA: range noise in meters
    # AOA: angle noise in radians
    # RSS: fading noise in dB (per book Eq. 4.12)
    toa_noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]  # meters
    rss_noise_levels = [0.0, 2.0, 4.0, 6.0, 8.0]   # dB (typical indoor: 4-8 dB)

    results = {"TOA": [], "TDOA": [], "AOA": [], "RSS": []}

    print("\nNoise configuration:")
    print("  TOA/TDOA: Range noise (meters)")
    print("  AOA: Angle noise (radians) = TOA_noise / 5")
    print("  RSS: Long-term fading (dB) + Rayleigh short-term fading (Eq. 4.12)")
    print("       - Long-term fading: Gaussian in dB (location-dependent)")
    print("       - Short-term fading: Rayleigh amplitude (mitigated by averaging)")
    print("       - 5 samples averaged to reduce short-term fading variance")

    # RSS fading configuration
    sigma_short_linear = 0.5  # Rayleigh scale (moderate short-term fading)
    n_samples_avg = 5  # Average 5 samples to reduce short-term fading

    print("\nTesting noise levels...")
    start_time = time.time()

    for i, (toa_noise, rss_fading_db) in enumerate(
        tqdm(
            list(zip(toa_noise_levels, rss_noise_levels)),
            desc="Overall progress",
            unit="level",
        )
    ):
        print(f"\n[{i+1}/{len(toa_noise_levels)}] TOA/TDOA: {toa_noise:.2f}m, "
              f"RSS long-term: {rss_fading_db:.1f}dB")

        results["TOA"].append(toa_positioning_test(anchors, true_positions, toa_noise))
        results["TDOA"].append(tdoa_positioning_test(anchors, true_positions, toa_noise))

        angle_noise = toa_noise / 5.0  # radians
        results["AOA"].append(aoa_positioning_test(anchors, true_positions, angle_noise))

        # RSS uses full fading model per book (Eq. 4.12):
        # - omega_long: Gaussian in dB (location-dependent shadowing)
        # - omega_short: Rayleigh amplitude fading (multipath, time-varying)
        # Averaging n samples reduces short-term fading variance
        results["RSS"].append(
            rss_positioning_test(
                anchors,
                true_positions,
                sigma_long_db=rss_fading_db,
                sigma_short_linear=sigma_short_linear,
                n_samples_avg=n_samples_avg,
                short_fading_model="rayleigh",
            )
        )

    elapsed_time = time.time() - start_time
    print(f"\nAll tests completed in {elapsed_time:.2f}s")

    print("\n" + "=" * 70)
    print("Results Summary (RMSE in meters)")
    print("=" * 70)
    print(f"  RSS config: Rayleigh short-term (sigma={sigma_short_linear}), "
          f"{n_samples_avg} samples averaged")
    print(f"{'TOA/TDOA':<12} {'RSS Long':<12} {'TOA':<10} {'TDOA':<10} {'AOA':<10} {'RSS':<10}")
    print("-" * 70)

    for i, (toa_noise, rss_fading_db) in enumerate(
        zip(toa_noise_levels, rss_noise_levels)
    ):
        toa_str = f"{toa_noise:.2f}m"
        rss_str = f"{rss_fading_db:.1f}dB"
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
            f"{toa_str:<12} {rss_str:<12} {toa_rmse:<10.3f} {tdoa_rmse:<10.3f} "
            f"{aoa_rmse:<10.3f} {rss_rmse:<10.3f}"
        )

    return toa_noise_levels, results


def plot_dataset_results(results: Dict, output_file: str = None):
    """Plot results from dataset-based comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF Positioning: Dataset Analysis", fontsize=16, fontweight="bold")
    
    beacons = results['beacons']
    positions = results['positions']
    
    # 1. Beacon geometry and test points
    ax1 = axes[0, 0]
    ax1.scatter(beacons[:, 0], beacons[:, 1], s=200, c='red', marker='^', 
                label='Beacons', zorder=10, edgecolors='black', linewidths=2)
    ax1.scatter(positions[:, 0], positions[:, 1], s=20, c='blue', alpha=0.5, label='Test Points')
    for i, b in enumerate(beacons):
        ax1.annotate(f'B{i}', (b[0], b[1]), xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Beacon Geometry & Test Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Error CDF
    ax2 = axes[0, 1]
    colors = {'TOA': 'blue', 'TDOA': 'red', 'AOA': 'green'}
    for method, color in colors.items():
        errors = results[method]['errors']
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(sorted_errors, cdf, label=method, color=color, linewidth=2)
    ax2.set_xlabel('Position Error (m)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution of Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    # 3. GDOP distribution
    ax3 = axes[1, 0]
    gdop_data = [results['gdop']['TOA'], results['gdop']['TDOA'], results['gdop']['AOA']]
    bp = ax3.boxplot(gdop_data, labels=['TOA', 'TDOA', 'AOA'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'red', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('GDOP')
    ax3.set_title('Geometric Dilution of Precision')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Error vs GDOP scatter
    ax4 = axes[1, 1]
    for method, color in colors.items():
        errors = results[method]['errors']
        gdop = results['gdop'][method]
        if len(errors) > 0:
            # Match lengths (some points may have failed)
            n = min(len(errors), len(gdop))
            ax4.scatter(gdop[:n], errors[:n], alpha=0.5, label=method, color=color, s=20)
    ax4.set_xlabel('GDOP')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Error vs GDOP (lower GDOP = better geometry)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved: {output_file}")
    
    return fig


def plot_inline_comparison(noise_levels, results):
    """Plot comparison results (inline mode)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF Positioning Methods Comparison", fontsize=16, fontweight="bold")

    methods = ["TOA", "TDOA", "AOA", "RSS"]
    colors = ["blue", "red", "green", "orange"]

    # 1. RMSE vs Noise
    ax1 = axes[0, 0]
    for method, color in zip(methods, colors):
        rmse_values = [np.sqrt(np.mean(e**2)) if len(e) > 0 else np.nan for e in results[method]]
        ax1.plot(noise_levels, rmse_values, "o-", label=method, color=color, linewidth=2)
    ax1.set_xlabel("Measurement Noise (m)")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("RMSE vs Measurement Noise")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Error CDF
    ax2 = axes[0, 1]
    noise_idx = 2
    for method, color in zip(methods, colors):
        errors = results[method][noise_idx]
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(sorted_errors, cdf, label=method, color=color, linewidth=2)
    ax2.set_xlabel("Position Error (m)")
    ax2.set_ylabel("CDF")
    ax2.set_title(f"Error CDF (Noise = {noise_levels[noise_idx]:.2f}m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)

    # 3. Boxplot
    ax3 = axes[1, 0]
    data = [results[m][noise_idx] for m in methods if len(results[m][noise_idx]) > 0]
    labels = [m for m in methods if len(results[m][noise_idx]) > 0]
    bp = ax3.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title(f"Error Distribution (Noise = {noise_levels[noise_idx]:.2f}m)")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Success Rate
    ax4 = axes[1, 1]
    total_points = 50
    for method, color in zip(methods, colors):
        rates = [len(e) / total_points * 100 for e in results[method]]
        ax4.plot(noise_levels, rates, "o-", label=method, color=color, linewidth=2)
    ax4.set_xlabel("Measurement Noise (m)")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Convergence Success Rate")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    return fig


def main():
    """Run RF positioning comparison."""
    parser = argparse.ArgumentParser(
        description="Chapter 4: RF Positioning Methods Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python example_comparison.py
  
  # Run with pre-generated dataset
  python example_comparison.py --data ch4_rf_2d_square
  
  # Compare different beacon geometries
  python example_comparison.py --compare-geometry
  
  # Compare NLOS vs baseline
  python example_comparison.py --data ch4_rf_2d_nlos
        """
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name or path (e.g., 'ch4_rf_2d_square' or full path)"
    )
    parser.add_argument(
        "--compare-geometry", action="store_true",
        help="Compare positioning across different beacon geometries"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for figure (default: ch4_rf_point_positioning/figs/ch4_rf_comparison.png)"
    )
    
    args = parser.parse_args()
    
    overall_start = time.time()
    
    if args.compare_geometry:
        # Compare different geometries
        all_results = compare_geometries(verbose=True)
        
        if len(all_results) > 0:
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            methods = ['TOA', 'TDOA', 'AOA']
            x = np.arange(len(all_results))
            width = 0.25
            
            for i, method in enumerate(methods):
                rmse_vals = []
                for geom, res in all_results.items():
                    errors = res[method]['errors']
                    rmse = np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0
                    rmse_vals.append(rmse)
                ax.bar(x + i*width, rmse_vals, width, label=method)
            
            ax.set_ylabel('RMSE (m)')
            ax.set_title('Positioning Error by Beacon Geometry')
            ax.set_xticks(x + width)
            ax.set_xticklabels(all_results.keys())
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            output_file = args.output or "ch4_rf_point_positioning/figs/ch4_geometry_comparison.png"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n✓ Figure saved: {output_file}")
            plt.show()
    
    elif args.data:
        # Run with dataset
        data_path = Path(args.data)
        if not data_path.exists():
            data_path = Path("data/sim") / args.data
        if not data_path.exists():
            print(f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'")
            print("\nAvailable datasets:")
            sim_dir = Path("data/sim")
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch4"):
                        print(f"  - {d.name}")
            return
        
        results = run_with_dataset(str(data_path), verbose=True)
        
        output_file = args.output or "ch4_rf_point_positioning/figs/ch4_rf_comparison.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plot_dataset_results(results, output_file)
        plt.show()
    
    else:
        # Run with inline data (original behavior)
        print("\n" + "=" * 70)
        print("Chapter 4: RF Positioning Methods Comparison")
        print("=" * 70)
        print("\nTip: Run with --data ch4_rf_2d_square to use pre-generated dataset")
        print("     Run with --compare-geometry to compare beacon layouts\n")

        noise_levels, results = run_inline_comparison()

        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)

        fig = plot_inline_comparison(noise_levels, results)

        output_file = args.output or "ch4_rf_point_positioning/figs/ch4_rf_comparison.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {output_file}")
        plt.show()

    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("Comparison completed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
