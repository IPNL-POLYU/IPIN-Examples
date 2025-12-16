"""Visualize 2D IMU + UWB fusion datasets.

Creates comprehensive plots showing trajectory, measurements, noise characteristics,
and data quality metrics.

Usage:
    python tools/plot_fusion_dataset.py data/sim/fusion_2d_imu_uwb
    python tools/plot_fusion_dataset.py data/sim/fusion_2d_imu_uwb --output my_plots
    python tools/plot_fusion_dataset.py data/sim/fusion_2d_imu_uwb --format png

Author: Navigation Engineer
Date: December 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

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
    
    # Load ground truth
    truth_file = dataset_path / 'truth.npz'
    if truth_file.exists():
        data['truth'] = dict(np.load(truth_file))
    else:
        raise FileNotFoundError(f"truth.npz not found in {dataset_path}")
    
    # Load IMU
    imu_file = dataset_path / 'imu.npz'
    if imu_file.exists():
        data['imu'] = dict(np.load(imu_file))
    else:
        raise FileNotFoundError(f"imu.npz not found in {dataset_path}")
    
    # Load UWB ranges
    uwb_file = dataset_path / 'uwb_ranges.npz'
    if uwb_file.exists():
        data['uwb'] = dict(np.load(uwb_file))
    else:
        raise FileNotFoundError(f"uwb_ranges.npz not found in {dataset_path}")
    
    # Load UWB anchors
    anchors_file = dataset_path / 'uwb_anchors.npy'
    if anchors_file.exists():
        data['anchors'] = np.load(anchors_file)
    else:
        raise FileNotFoundError(f"uwb_anchors.npy not found in {dataset_path}")
    
    # Load config
    config_file = dataset_path / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            data['config'] = json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found in {dataset_path}")
    
    return data


def plot_trajectory(data: Dict, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot 2D trajectory with anchors.
    
    Args:
        data: Dataset dictionary from load_fusion_dataset.
        ax: Optional existing axes (creates new if None).
    
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    truth = data['truth']
    anchors = data['anchors']
    config = data['config']
    
    # Plot trajectory
    ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    
    # Mark start and end
    ax.plot(truth['p_xy'][0, 0], truth['p_xy'][0, 1],
            'go', markersize=12, label='Start', zorder=10)
    ax.plot(truth['p_xy'][-1, 0], truth['p_xy'][-1, 1],
            'ro', markersize=12, label='End', zorder=10)
    
    # Plot anchors
    nlos_anchors = config.get('uwb', {}).get('nlos_anchors', [])
    
    for i, anchor in enumerate(anchors):
        if i in nlos_anchors:
            color = 'red'
            marker = '^'
            label = f'Anchor {i} (NLOS)' if i == nlos_anchors[0] else None
        else:
            color = 'green'
            marker = '^'
            label = f'Anchor {i} (Clean)' if i == 0 or (i == 1 and not nlos_anchors) else None
        
        ax.plot(anchor[0], anchor[1], marker, markersize=15, 
                color=color, label=label, zorder=5)
        ax.text(anchor[0], anchor[1] + 1, f'A{i}',
                ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title('2D Trajectory and UWB Anchors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return ax


def plot_velocity_heading(data: Dict, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """Plot velocity and heading over time.
    
    Args:
        data: Dataset dictionary.
        axes: Optional 2x1 axes array (creates new if None).
    
    Returns:
        Axes array.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    truth = data['truth']
    t = truth['t']
    v_xy = truth['v_xy']
    yaw = truth['yaw']
    
    # Velocity magnitude
    v_mag = np.linalg.norm(v_xy, axis=1)
    axes[0].plot(t, v_mag, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Speed (m/s)', fontsize=11)
    axes[0].set_title('Velocity and Heading Over Time', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Heading
    axes[1].plot(t, np.rad2deg(yaw), 'r-', linewidth=1.5)
    axes[1].set_ylabel('Heading (degrees)', fontsize=11)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    return axes


def plot_imu_measurements(data: Dict, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """Plot IMU measurements (accel, gyro).
    
    Args:
        data: Dataset dictionary.
        axes: Optional 3x1 axes array (creates new if None).
    
    Returns:
        Axes array.
    """
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    imu = data['imu']
    t = imu['t']
    accel_xy = imu['accel_xy']
    gyro_z = imu['gyro_z']
    
    # Accelerometer X
    axes[0].plot(t, accel_xy[:, 0], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Accel X (m/s²)', fontsize=10)
    axes[0].set_title('IMU Measurements', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Accelerometer Y
    axes[1].plot(t, accel_xy[:, 1], 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Accel Y (m/s²)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Gyroscope Z
    axes[2].plot(t, gyro_z, 'r-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Gyro Z (rad/s)', fontsize=10)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    return axes


def plot_uwb_ranges(data: Dict, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """Plot UWB range measurements per anchor.
    
    Args:
        data: Dataset dictionary.
        axes: Optional 2x2 axes array (creates new if None).
    
    Returns:
        Axes array.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    uwb = data['uwb']
    anchors = data['anchors']
    config = data['config']
    
    t_uwb = uwb['t']
    ranges = uwb['ranges']
    nlos_anchors = config.get('uwb', {}).get('nlos_anchors', [])
    
    for i in range(min(4, ranges.shape[1])):
        ranges_i = ranges[:, i]
        valid_mask = ~np.isnan(ranges_i)
        
        color = 'red' if i in nlos_anchors else 'blue'
        nlos_label = ' (NLOS)' if i in nlos_anchors else ''
        
        axes[i].plot(t_uwb[valid_mask], ranges_i[valid_mask],
                     'o', color=color, markersize=3, alpha=0.6,
                     label=f'Measured{nlos_label}')
        
        axes[i].set_ylabel('Range (m)', fontsize=10)
        axes[i].set_title(f'Anchor {i} Ranges', fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
        
        if i >= 2:
            axes[i].set_xlabel('Time (s)', fontsize=10)
    
    return axes


def plot_range_errors(data: Dict, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """Plot range measurement errors (measured - true).
    
    Args:
        data: Dataset dictionary.
        axes: Optional 2x2 axes array (creates new if None).
    
    Returns:
        Axes array.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
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
    
    for i in range(min(4, ranges.shape[1])):
        ranges_i = ranges[:, i]
        valid_mask = ~np.isnan(ranges_i)
        
        errors = ranges_i[valid_mask] - ranges_true[valid_mask, i]
        
        nlos_label = ' (NLOS +bias)' if i in nlos_anchors else ' (Clean)'
        
        axes[i].hist(errors, bins=50, alpha=0.7, edgecolor='black',
                     color='red' if i in nlos_anchors else 'blue')
        axes[i].axvline(0, color='k', linestyle='--', linewidth=2)
        axes[i].axvline(np.mean(errors), color='r', linestyle='-',
                        linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
        
        axes[i].set_xlabel('Range Error (m)', fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].set_title(f'Anchor {i}{nlos_label}', fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    return axes


def plot_dataset_overview(dataset_path: str, output_dir: Optional[str] = None,
                          fmt: str = 'svg', show: bool = False):
    """Create comprehensive overview plots for a fusion dataset.
    
    Args:
        dataset_path: Path to dataset directory.
        output_dir: Output directory for plots (default: same as dataset).
        fmt: Output format ('svg', 'png', 'pdf').
        show: If True, display plots interactively.
    """
    dataset_path = Path(dataset_path)
    
    if output_dir is None:
        output_dir = dataset_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_fusion_dataset(dataset_path)
    
    config = data['config']
    print(f"  Duration: {config['dataset_info']['duration_sec']} s")
    print(f"  IMU samples: {config['dataset_info']['imu_samples']}")
    print(f"  UWB samples: {config['dataset_info']['uwb_samples']}")
    
    # Create plots
    print(f"\nGenerating plots...")
    
    # 1. Trajectory with anchors
    print("  1/5: Trajectory")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    plot_trajectory(data, ax=ax1)
    fig1.tight_layout()
    output_file1 = output_dir / f"trajectory.{fmt}"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"       Saved: {output_file1}")
    
    # 2. Velocity and heading
    print("  2/5: Velocity and heading")
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_velocity_heading(data, axes=axes2)
    fig2.tight_layout()
    output_file2 = output_dir / f"velocity_heading.{fmt}"
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"       Saved: {output_file2}")
    
    # 3. IMU measurements
    print("  3/5: IMU measurements")
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_imu_measurements(data, axes=axes3)
    fig3.tight_layout()
    output_file3 = output_dir / f"imu_measurements.{fmt}"
    fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"       Saved: {output_file3}")
    
    # 4. UWB ranges
    print("  4/5: UWB ranges")
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    plot_uwb_ranges(data, axes=axes4)
    fig4.tight_layout()
    output_file4 = output_dir / f"uwb_ranges.{fmt}"
    fig4.savefig(output_file4, dpi=150, bbox_inches='tight')
    print(f"       Saved: {output_file4}")
    
    # 5. Range errors
    print("  5/5: Range errors")
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    plot_range_errors(data, axes=axes5)
    fig5.suptitle('UWB Range Measurement Errors', fontsize=14, fontweight='bold')
    fig5.tight_layout()
    output_file5 = output_dir / f"range_errors.{fmt}"
    fig5.savefig(output_file5, dpi=150, bbox_inches='tight')
    print(f"       Saved: {output_file5}")
    
    if show:
        plt.show()
    else:
        plt.close('all')
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Format: {fmt.upper()}\n")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize 2D IMU + UWB fusion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for baseline dataset
  python %(prog)s data/sim/fusion_2d_imu_uwb

  # Save to custom output directory
  python %(prog)s data/sim/fusion_2d_imu_uwb --output plots/baseline

  # Generate PNG instead of SVG
  python %(prog)s data/sim/fusion_2d_imu_uwb --format png

  # Display plots interactively
  python %(prog)s data/sim/fusion_2d_imu_uwb --show
        """
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to dataset directory (e.g., data/sim/fusion_2d_imu_uwb)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots (default: same as dataset)'
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
        help='Display plots interactively (default: save only)'
    )
    
    args = parser.parse_args()
    
    # Generate plots
    plot_dataset_overview(
        dataset_path=args.dataset,
        output_dir=args.output,
        fmt=args.format,
        show=args.show
    )


if __name__ == "__main__":
    main()


