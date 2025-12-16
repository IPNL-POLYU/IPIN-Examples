"""Compare multiple fusion dataset variants side-by-side.

Creates side-by-side comparison plots of different dataset configurations
(e.g., baseline vs. NLOS vs. time offset).

Usage:
    python tools/compare_fusion_variants.py \
        data/sim/ch8_fusion_2d_imu_uwb \
        data/sim/ch8_fusion_2d_imu_uwb_nlos \
        data/sim/ch8_fusion_2d_imu_uwb_timeoffset

    python tools/compare_fusion_variants.py \
        data/sim/ch8_fusion_2d_imu_uwb \
        data/sim/ch8_fusion_2d_imu_uwb_nlos \
        --output comparison.svg

Author: Navigation Engineer
Date: December 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_fusion_dataset(dataset_path: Path) -> Dict:
    """Load a fusion dataset from directory.
    
    Args:
        dataset_path: Path to dataset directory.
    
    Returns:
        Dictionary with 'truth', 'imu', 'uwb', 'anchors', 'config' keys.
    """
    data = {}
    data['name'] = dataset_path.name
    data['truth'] = dict(np.load(dataset_path / 'truth.npz'))
    data['imu'] = dict(np.load(dataset_path / 'imu.npz'))
    data['uwb'] = dict(np.load(dataset_path / 'uwb_ranges.npz'))
    data['anchors'] = np.load(dataset_path / 'uwb_anchors.npy')
    
    with open(dataset_path / 'config.json') as f:
        data['config'] = json.load(f)
    
    return data


def compare_trajectories(datasets: List[Dict], output_file: str = None, show: bool = False):
    """Compare trajectories from multiple datasets.
    
    Args:
        datasets: List of dataset dictionaries.
        output_file: Output file path (None = display only).
        show: If True, display plot interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot trajectories
    for i, data in enumerate(datasets):
        truth = data['truth']
        color = colors[i % len(colors)]
        
        ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
                '-', color=color, linewidth=2, alpha=0.7,
                label=data['name'])
    
    # Plot anchors (from first dataset, assuming same for all)
    anchors = datasets[0]['anchors']
    ax.plot(anchors[:, 0], anchors[:, 1],
            'k^', markersize=15, label='UWB Anchors', zorder=5)
    
    for i, anchor in enumerate(anchors):
        ax.text(anchor[0], anchor[1] + 1, f'A{i}',
                ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('East (m)', fontsize=13)
    ax.set_ylabel('North (m)', fontsize=13)
    ax.set_title('Trajectory Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_range_errors(datasets: List[Dict], output_file: str = None, show: bool = False):
    """Compare range error distributions across datasets.
    
    Args:
        datasets: List of dataset dictionaries.
        output_file: Output file path.
        show: If True, display plot interactively.
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 4, figsize=(16, 4*n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for row, data in enumerate(datasets):
        truth = data['truth']
        uwb = data['uwb']
        anchors = data['anchors']
        config = data['config']
        
        t_uwb = uwb['t']
        ranges = uwb['ranges']
        nlos_anchors = config.get('uwb', {}).get('nlos_anchors', [])
        
        # Interpolate truth to UWB timestamps
        p_xy_uwb = np.column_stack([
            np.interp(t_uwb, truth['t'], truth['p_xy'][:, 0]),
            np.interp(t_uwb, truth['t'], truth['p_xy'][:, 1])
        ])
        
        # Compute true ranges
        ranges_true = np.array([
            np.linalg.norm(p_xy_uwb - anchor, axis=1)
            for anchor in anchors
        ]).T
        
        for col in range(4):
            ax = axes[row, col]
            ranges_i = ranges[:, col]
            valid_mask = ~np.isnan(ranges_i)
            
            errors = ranges_i[valid_mask] - ranges_true[valid_mask, col]
            
            color = 'red' if col in nlos_anchors else 'blue'
            nlos_label = ' (NLOS)' if col in nlos_anchors else ''
            
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color=color)
            ax.axvline(0, color='k', linestyle='--', linewidth=2)
            ax.axvline(np.mean(errors), color='r', linestyle='-',
                       linewidth=2, label=f'μ={np.mean(errors):.3f}m')
            
            ax.set_xlabel('Range Error (m)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            
            if row == 0:
                ax.set_title(f'Anchor {col}{nlos_label}', fontsize=11, fontweight='bold')
            
            if col == 0:
                ax.text(-0.15, 0.5, data['name'], transform=ax.transAxes,
                        fontsize=11, fontweight='bold', rotation=90,
                        va='center', ha='right')
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Range Error Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_imu_noise(datasets: List[Dict], output_file: str = None, show: bool = False):
    """Compare IMU noise characteristics across datasets.
    
    Args:
        datasets: List of dataset dictionaries.
        output_file: Output file path.
        show: If True, display plot interactively.
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4*n_datasets), sharex=True)
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for row, data in enumerate(datasets):
        imu = data['imu']
        t = imu['t']
        accel_xy = imu['accel_xy']
        gyro_z = imu['gyro_z']
        
        # Accelerometer X
        axes[row, 0].plot(t, accel_xy[:, 0], 'b-', linewidth=0.3, alpha=0.7)
        axes[row, 0].set_ylabel('Accel X (m/s²)', fontsize=9)
        if row == 0:
            axes[row, 0].set_title('Accelerometer X', fontsize=11, fontweight='bold')
        if row == 0:
            axes[row, 0].text(-0.15, 0.5, data['name'], transform=axes[row, 0].transAxes,
                              fontsize=11, fontweight='bold', rotation=90,
                              va='center', ha='right')
        axes[row, 0].grid(True, alpha=0.3)
        
        # Accelerometer Y
        axes[row, 1].plot(t, accel_xy[:, 1], 'g-', linewidth=0.3, alpha=0.7)
        axes[row, 1].set_ylabel('Accel Y (m/s²)', fontsize=9)
        if row == 0:
            axes[row, 1].set_title('Accelerometer Y', fontsize=11, fontweight='bold')
        axes[row, 1].grid(True, alpha=0.3)
        
        # Gyroscope Z
        axes[row, 2].plot(t, gyro_z, 'r-', linewidth=0.3, alpha=0.7)
        axes[row, 2].set_ylabel('Gyro Z (rad/s)', fontsize=9)
        if row == 0:
            axes[row, 2].set_title('Gyroscope Z', fontsize=11, fontweight='bold')
        axes[row, 2].grid(True, alpha=0.3)
        
        if row == n_datasets - 1:
            axes[row, 0].set_xlabel('Time (s)', fontsize=10)
            axes[row, 1].set_xlabel('Time (s)', fontsize=10)
            axes[row, 2].set_xlabel('Time (s)', fontsize=10)
    
    fig.suptitle('IMU Measurements Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_comparison_summary(datasets: List[Dict]):
    """Print summary comparison table of datasets.
    
    Args:
        datasets: List of dataset dictionaries.
    """
    print("\n" + "="*80)
    print("DATASET COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Parameter':<30} ", end="")
    for data in datasets:
        print(f"{data['name'][:20]:<22}", end="")
    print()
    print("-" * 80)
    
    # Duration
    print(f"{'Duration (s)':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['dataset_info']['duration_sec']:<22.1f}", end="")
    print()
    
    # IMU samples
    print(f"{'IMU samples':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['dataset_info']['imu_samples']:<22}", end="")
    print()
    
    # UWB samples
    print(f"{'UWB samples':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['dataset_info']['uwb_samples']:<22}", end="")
    print()
    
    print("-" * 80)
    
    # IMU parameters
    print(f"{'Accel noise (m/s²)':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['imu']['accel_noise_std_m_s2']:<22.3f}", end="")
    print()
    
    print(f"{'Gyro noise (rad/s)':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['imu']['gyro_noise_std_rad_s']:<22.4f}", end="")
    print()
    
    # UWB parameters
    print(f"{'Range noise (m)':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['uwb']['range_noise_std_m']:<22.3f}", end="")
    print()
    
    print(f"{'NLOS anchors':<30} ", end="")
    for data in datasets:
        nlos = data['config']['uwb'].get('nlos_anchors', [])
        print(f"{str(nlos):<22}", end="")
    print()
    
    print(f"{'NLOS bias (m)':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['uwb'].get('nlos_bias_m', 0.0):<22.2f}", end="")
    print()
    
    print(f"{'Dropout rate':<30} ", end="")
    for data in datasets:
        print(f"{data['config']['uwb']['dropout_rate']:<22.2f}", end="")
    print()
    
    # Temporal calibration
    print(f"{'Time offset (ms)':<30} ", end="")
    for data in datasets:
        offset = data['config']['temporal_calibration'].get('time_offset_sec', 0.0)
        print(f"{offset*1000:<22.1f}", end="")
    print()
    
    print(f"{'Clock drift (ppm)':<30} ", end="")
    for data in datasets:
        drift = data['config']['temporal_calibration'].get('clock_drift', 0.0)
        print(f"{drift*1e6:<22.1f}", end="")
    print()
    
    print("="*80 + "\n")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Compare multiple fusion dataset variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 3 standard variants
  python %(prog)s \
      data/sim/ch8_fusion_2d_imu_uwb \
      data/sim/ch8_fusion_2d_imu_uwb_nlos \
      data/sim/ch8_fusion_2d_imu_uwb_timeoffset

  # Compare baseline vs. NLOS only
  python %(prog)s \
      data/sim/ch8_fusion_2d_imu_uwb \
      data/sim/ch8_fusion_2d_imu_uwb_nlos \
      --output comparison_baseline_vs_nlos

  # Save as PNG instead of SVG
  python %(prog)s \
      data/sim/ch8_fusion_2d_imu_uwb \
      data/sim/ch8_fusion_2d_imu_uwb_nlos \
      --format png

  # Display interactively
  python %(prog)s \
      data/sim/ch8_fusion_2d_imu_uwb \
      data/sim/ch8_fusion_2d_imu_uwb_nlos \
      --show
        """
    )
    
    parser.add_argument(
        'datasets',
        type=str,
        nargs='+',
        help='Paths to dataset directories to compare'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='comparison',
        help='Output file prefix (default: comparison)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['svg', 'png', 'pdf'],
        default='svg',
        help='Output format (default: svg)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    if len(args.datasets) < 2:
        parser.error("At least 2 datasets required for comparison")
    
    # Load all datasets
    print(f"\nLoading {len(args.datasets)} datasets...")
    datasets = []
    for dataset_path in args.datasets:
        path = Path(dataset_path)
        print(f"  Loading: {path.name}")
        data = load_fusion_dataset(path)
        datasets.append(data)
    
    # Print summary
    print_comparison_summary(datasets)
    
    # Generate comparison plots
    print("Generating comparison plots...")
    
    # 1. Trajectory comparison
    print("  1/3: Trajectory comparison")
    output_file1 = f"{args.output}_trajectories.{args.format}"
    compare_trajectories(datasets, output_file=output_file1, show=args.show)
    
    # 2. Range error comparison
    print("  2/3: Range error comparison")
    output_file2 = f"{args.output}_range_errors.{args.format}"
    compare_range_errors(datasets, output_file=output_file2, show=args.show)
    
    # 3. IMU noise comparison
    print("  3/3: IMU measurements comparison")
    output_file3 = f"{args.output}_imu_measurements.{args.format}"
    compare_imu_noise(datasets, output_file=output_file3, show=args.show)
    
    print(f"\nAll comparison plots saved with prefix: {args.output}")
    print(f"Format: {args.format.upper()}\n")


if __name__ == "__main__":
    main()


