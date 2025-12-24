"""Generate synthetic 2D IMU + UWB fusion dataset for Chapter 8 examples.

Creates a realistic indoor fusion scenario with:
    - 2D rectangular walking trajectory
    - High-rate IMU measurements (100 Hz): accel_xy, gyro_z
    - Low-rate UWB range measurements (10 Hz) to 4 corner anchors
    - Ground truth: position, velocity, heading
    - Configurable noise, time offset, and NLOS bias

Saves to: data/sim/ch8_fusion_2d_imu_uwb/

Author: Li-Ta Hsu
Date: December 2025
References: Chapter 8 - Sensor Fusion
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'baseline': {
        'description': 'Standard configuration with nominal parameters',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'range_noise_std': 0.05,
        'nlos_anchors': [],
        'nlos_bias': 0.0,
        'dropout_rate': 0.05,
        'time_offset_sec': 0.0,
    },
    'nlos_severe': {
        'description': 'Severe NLOS on 2 anchors to test robust loss functions',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'range_noise_std': 0.05,
        'nlos_anchors': [1, 2],
        'nlos_bias': 1.5,
        'dropout_rate': 0.05,
        'time_offset_sec': 0.0,
    },
    'high_dropout': {
        'description': 'High dropout rate to test multi-rate fusion',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'range_noise_std': 0.05,
        'nlos_anchors': [],
        'nlos_bias': 0.0,
        'dropout_rate': 0.3,
        'time_offset_sec': 0.0,
    },
    'degraded_imu': {
        'description': 'Poor IMU quality (MEMS-grade) to test IMU drift',
        'accel_noise_std': 0.5,
        'gyro_noise_std': 0.05,
        'range_noise_std': 0.05,
        'nlos_anchors': [],
        'nlos_bias': 0.0,
        'dropout_rate': 0.05,
        'time_offset_sec': 0.0,
    },
    'time_offset_50ms': {
        'description': 'UWB 50ms behind IMU with clock drift',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'range_noise_std': 0.05,
        'nlos_anchors': [],
        'nlos_bias': 0.0,
        'dropout_rate': 0.05,
        'time_offset_sec': -0.05,
        'clock_drift': 0.0001,
    },
    'tactical_imu': {
        'description': 'Tactical-grade IMU (low noise)',
        'accel_noise_std': 0.01,
        'gyro_noise_std': 0.001,
        'range_noise_std': 0.05,
        'nlos_anchors': [],
        'nlos_bias': 0.0,
        'dropout_rate': 0.05,
        'time_offset_sec': 0.0,
    },
}


def generate_rectangular_trajectory(
    width: float = 20.0,
    height: float = 15.0,
    speed: float = 1.0,
    dt: float = 0.01,
    duration: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D rectangular walking trajectory.
    
    Walks a rectangle starting from (0, 0), counter-clockwise:
        (0, 0) → (width, 0) → (width, height) → (0, height) → (0, 0)
    
    Args:
        width: Rectangle width (meters).
        height: Rectangle height (meters).
        speed: Walking speed (m/s).
        dt: Time step (seconds).
        duration: Total duration (seconds).
    
    Returns:
        Tuple of (t, p_xy, v_xy, yaw):
            t: timestamps (N,)
            p_xy: positions (N, 2) in meters
            v_xy: velocities (N, 2) in m/s
            yaw: heading angles (N,) in radians
    """
    # Time array
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Rectangle perimeter
    perimeter = 2 * (width + height)
    
    # Distance traveled per step
    distances = speed * t
    distances = distances % perimeter  # wrap around
    
    # Allocate arrays
    p_xy = np.zeros((N, 2))
    v_xy = np.zeros((N, 2))
    yaw = np.zeros(N)
    
    for i, d in enumerate(distances):
        if d < width:
            # Side 1: (0,0) → (width, 0), heading = 0
            p_xy[i] = [d, 0]
            v_xy[i] = [speed, 0]
            yaw[i] = 0.0
        elif d < width + height:
            # Side 2: (width, 0) → (width, height), heading = π/2
            p_xy[i] = [width, d - width]
            v_xy[i] = [0, speed]
            yaw[i] = np.pi / 2
        elif d < 2 * width + height:
            # Side 3: (width, height) → (0, height), heading = π
            p_xy[i] = [width - (d - width - height), height]
            v_xy[i] = [-speed, 0]
            yaw[i] = np.pi
        else:
            # Side 4: (0, height) → (0, 0), heading = 3π/2
            p_xy[i] = [0, height - (d - 2 * width - height)]
            v_xy[i] = [0, -speed]
            yaw[i] = 3 * np.pi / 2
    
    return t, p_xy, v_xy, yaw


def generate_imu_measurements(
    t: np.ndarray,
    v_xy: np.ndarray,
    yaw: np.ndarray,
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01,
    accel_bias: np.ndarray = None,
    gyro_bias: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic IMU measurements from ground truth.
    
    Args:
        t: timestamps (N,)
        v_xy: velocities (N, 2) in m/s
        yaw: heading angles (N,) in radians
        accel_noise_std: Accelerometer noise std (m/s²)
        gyro_noise_std: Gyroscope noise std (rad/s)
        accel_bias: Accelerometer bias (2,) in m/s² (default [0, 0])
        gyro_bias: Gyroscope bias in rad/s
    
    Returns:
        Tuple of (t_imu, accel_xy, gyro_z):
            t_imu: IMU timestamps (N,)
            accel_xy: 2D accelerations (N, 2) in m/s²
            gyro_z: Yaw rate (N,) in rad/s
    """
    N = len(t)
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    
    if accel_bias is None:
        accel_bias = np.zeros(2)
    
    # Compute true accelerations (derivative of velocity)
    accel_xy_true = np.gradient(v_xy, axis=0) / dt[:, None]
    
    # Compute true yaw rate (derivative of yaw)
    yaw_unwrapped = np.unwrap(yaw)  # handle 2π wraps
    gyro_z_true = np.gradient(yaw_unwrapped) / dt
    
    # Add noise and bias
    accel_xy = (
        accel_xy_true 
        + accel_bias 
        + np.random.randn(N, 2) * accel_noise_std
    )
    
    gyro_z = (
        gyro_z_true 
        + gyro_bias 
        + np.random.randn(N) * gyro_noise_std
    )
    
    return t, accel_xy, gyro_z


def generate_uwb_measurements(
    t: np.ndarray,
    p_xy: np.ndarray,
    anchor_positions: np.ndarray,
    uwb_rate: float = 10.0,
    range_noise_std: float = 0.05,
    nlos_anchors: list = None,
    nlos_bias: float = 0.5,
    dropout_rate: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic UWB range measurements.
    
    Args:
        t: Ground truth timestamps (N,)
        p_xy: Ground truth positions (N, 2)
        anchor_positions: Anchor positions (A, 2)
        uwb_rate: UWB measurement rate (Hz)
        range_noise_std: Range noise std (meters)
        nlos_anchors: List of anchor indices with NLOS bias (default: [])
        nlos_bias: NLOS bias added to ranges (meters)
        dropout_rate: Probability of measurement dropout per anchor
    
    Returns:
        Tuple of (t_uwb, ranges):
            t_uwb: UWB timestamps (M,)
            ranges: Range measurements (M, A) with NaN for dropouts
    """
    if nlos_anchors is None:
        nlos_anchors = []
    
    # Subsample to UWB rate
    dt_uwb = 1.0 / uwb_rate
    t_uwb = np.arange(t[0], t[-1], dt_uwb)
    M = len(t_uwb)
    
    # Interpolate positions at UWB timestamps
    p_xy_uwb = np.column_stack([
        np.interp(t_uwb, t, p_xy[:, 0]),
        np.interp(t_uwb, t, p_xy[:, 1])
    ])
    
    # Compute true ranges
    A = anchor_positions.shape[0]
    ranges_true = np.zeros((M, A))
    
    for i, anchor in enumerate(anchor_positions):
        ranges_true[:, i] = np.linalg.norm(p_xy_uwb - anchor, axis=1)
    
    # Add noise
    ranges = ranges_true + np.random.randn(M, A) * range_noise_std
    
    # Add NLOS bias
    for anchor_idx in nlos_anchors:
        if 0 <= anchor_idx < A:
            ranges[:, anchor_idx] += nlos_bias
    
    # Add dropouts
    for i in range(A):
        dropout_mask = np.random.rand(M) < dropout_rate
        ranges[dropout_mask, i] = np.nan
    
    return t_uwb, ranges


def generate_fusion_2d_imu_uwb_dataset(
    output_dir: str = "data/sim/ch8_fusion_2d_imu_uwb",
    seed: int = 42,
    # Trajectory parameters
    width: float = 20.0,
    height: float = 15.0,
    speed: float = 1.0,
    duration: float = 60.0,
    dt_imu: float = 0.01,  # 100 Hz
    # IMU parameters
    accel_noise_std: float = 0.1,  # m/s²
    gyro_noise_std: float = 0.01,  # rad/s
    accel_bias: np.ndarray = None,
    gyro_bias: float = 0.0,
    # UWB parameters
    uwb_rate: float = 10.0,  # Hz
    range_noise_std: float = 0.05,  # meters
    nlos_anchors: list = None,
    nlos_bias: float = 0.5,
    dropout_rate: float = 0.05,
    # Temporal calibration (for advanced demos)
    time_offset_sec: float = 0.0,
    clock_drift: float = 0.0,
) -> None:
    """Generate and save fusion dataset.
    
    Args:
        output_dir: Output directory path
        seed: Random seed for reproducibility
        (other parameters documented above)
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Generating 2D IMU + UWB Fusion Dataset")
    print(f"{'='*70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate ground truth trajectory
    print(f"\n1. Generating trajectory...")
    print(f"   Rectangle: {width}m × {height}m")
    print(f"   Speed: {speed} m/s")
    print(f"   Duration: {duration} s")
    print(f"   IMU rate: {1/dt_imu:.0f} Hz")
    
    t, p_xy, v_xy, yaw = generate_rectangular_trajectory(
        width=width,
        height=height,
        speed=speed,
        dt=dt_imu,
        duration=duration
    )
    
    print(f"   Generated {len(t)} samples")
    print(f"   Perimeter: {2*(width+height):.1f}m, {(2*(width+height)/speed):.1f}s per lap")
    
    # Save ground truth
    np.savez(
        output_path / "truth.npz",
        t=t,
        p_xy=p_xy,
        v_xy=v_xy,
        yaw=yaw
    )
    print(f"   Saved: truth.npz")
    
    # 2. Generate IMU measurements
    print(f"\n2. Generating IMU measurements...")
    print(f"   Accel noise: {accel_noise_std} m/s²")
    print(f"   Gyro noise: {gyro_noise_std} rad/s")
    
    if accel_bias is None:
        accel_bias = np.zeros(2)
    
    t_imu, accel_xy, gyro_z = generate_imu_measurements(
        t, v_xy, yaw,
        accel_noise_std=accel_noise_std,
        gyro_noise_std=gyro_noise_std,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias
    )
    
    np.savez(
        output_path / "imu.npz",
        t=t_imu,
        accel_xy=accel_xy,
        gyro_z=gyro_z
    )
    print(f"   Generated {len(t_imu)} IMU samples")
    print(f"   Saved: imu.npz")
    
    # 3. Place UWB anchors at corners (plus center offset)
    print(f"\n3. Generating UWB measurements...")
    anchor_positions = np.array([
        [0.0, 0.0],           # Bottom-left
        [width, 0.0],         # Bottom-right
        [width, height],      # Top-right
        [0.0, height]         # Top-left
    ])
    
    np.save(output_path / "uwb_anchors.npy", anchor_positions)
    print(f"   Anchors: {anchor_positions.shape[0]} at corners")
    for i, pos in enumerate(anchor_positions):
        print(f"      Anchor {i}: ({pos[0]:.1f}, {pos[1]:.1f}) m")
    
    # Generate UWB ranges
    print(f"   UWB rate: {uwb_rate} Hz")
    print(f"   Range noise: {range_noise_std} m")
    if nlos_anchors:
        print(f"   NLOS anchors: {nlos_anchors} (bias +{nlos_bias} m)")
    
    t_uwb, ranges = generate_uwb_measurements(
        t, p_xy, anchor_positions,
        uwb_rate=uwb_rate,
        range_noise_std=range_noise_std,
        nlos_anchors=nlos_anchors,
        nlos_bias=nlos_bias,
        dropout_rate=dropout_rate
    )
    
    np.savez(
        output_path / "uwb_ranges.npz",
        t=t_uwb,
        ranges=ranges
    )
    
    # Count dropouts
    n_dropouts = np.sum(np.isnan(ranges))
    dropout_percent = 100 * n_dropouts / ranges.size
    print(f"   Generated {len(t_uwb)} UWB samples")
    print(f"   Dropouts: {n_dropouts}/{ranges.size} ({dropout_percent:.1f}%)")
    print(f"   Saved: uwb_ranges.npz")
    
    # 4. Save configuration
    print(f"\n4. Saving configuration...")
    
    config = {
        "dataset_info": {
            "description": "2D IMU + UWB fusion dataset for Chapter 8",
            "seed": seed,
            "duration_sec": duration,
            "imu_samples": int(len(t_imu)),
            "uwb_samples": int(len(t_uwb))
        },
        "trajectory": {
            "type": "rectangular_walk",
            "width_m": width,
            "height_m": height,
            "speed_m_s": speed
        },
        "imu": {
            "rate_hz": 1 / dt_imu,
            "dt_sec": dt_imu,
            "accel_noise_std_m_s2": accel_noise_std,
            "gyro_noise_std_rad_s": gyro_noise_std,
            "accel_bias_m_s2": accel_bias.tolist(),
            "gyro_bias_rad_s": gyro_bias
        },
        "uwb": {
            "rate_hz": uwb_rate,
            "n_anchors": anchor_positions.shape[0],
            "range_noise_std_m": range_noise_std,
            "nlos_anchors": nlos_anchors if nlos_anchors else [],
            "nlos_bias_m": nlos_bias,
            "dropout_rate": dropout_rate
        },
        "temporal_calibration": {
            "time_offset_sec": time_offset_sec,
            "clock_drift": clock_drift,
            "note": "Use TimeSyncModel to apply these during fusion"
        },
        "coordinate_frame": {
            "description": "ENU (East-North-Up)",
            "origin": "Bottom-left corner (0, 0)",
            "units": "meters"
        }
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"   Saved: config.json")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Dataset generation complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"\nFiles created:")
    print(f"  - truth.npz        : Ground truth (t, p_xy, v_xy, yaw)")
    print(f"  - imu.npz          : IMU measurements (t, accel_xy, gyro_z)")
    print(f"  - uwb_anchors.npy  : UWB anchor positions")
    print(f"  - uwb_ranges.npz   : UWB range measurements (with NaN dropouts)")
    print(f"  - config.json      : Dataset configuration")
    print(f"\nDataset statistics:")
    print(f"  Duration        : {duration:.1f} s")
    print(f"  IMU samples     : {len(t_imu)} ({1/dt_imu:.0f} Hz)")
    print(f"  UWB samples     : {len(t_uwb)} ({uwb_rate:.0f} Hz)")
    print(f"  UWB anchors     : {anchor_positions.shape[0]}")
    print(f"  Trajectory laps : {duration * speed / (2*(width+height)):.1f}")
    print(f"\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate 2D IMU + UWB fusion dataset for Chapter 8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default parameters
  python %(prog)s

  # Use a preset configuration
  python %(prog)s --preset baseline

  # Generate NLOS variant
  python %(prog)s --preset nlos_severe --output data/sim/fusion_nlos_test

  # Custom parameters
  python %(prog)s --accel-noise 0.5 --gyro-noise 0.05 --duration 120

  # Generate all standard variants
  python %(prog)s --all-variants

  # High dropout test
  python %(prog)s --dropout-rate 0.3 --output data/sim/fusion_high_dropout

Available presets: """ + ", ".join(PRESETS.keys())
    )
    
    # Preset configuration
    parser.add_argument(
        '--preset',
        type=str,
        choices=PRESETS.keys(),
        help='Use preset configuration (overrides individual parameters)'
    )
    
    parser.add_argument(
        '--all-variants',
        action='store_true',
        help='Generate all 3 standard variants (baseline, nlos, timeoffset)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/sim/ch8_fusion_2d_imu_uwb',
        help='Output directory (default: data/sim/ch8_fusion_2d_imu_uwb)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Trajectory parameters
    traj_group = parser.add_argument_group('Trajectory Parameters')
    traj_group.add_argument(
        '--width',
        type=float,
        default=20.0,
        help='Rectangle width in meters (default: 20.0)'
    )
    traj_group.add_argument(
        '--height',
        type=float,
        default=15.0,
        help='Rectangle height in meters (default: 15.0)'
    )
    traj_group.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Walking speed in m/s (default: 1.0)'
    )
    traj_group.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Trajectory duration in seconds (default: 60.0)'
    )
    traj_group.add_argument(
        '--dt-imu',
        type=float,
        default=0.01,
        help='IMU time step in seconds (default: 0.01, i.e., 100 Hz)'
    )
    
    # IMU parameters
    imu_group = parser.add_argument_group('IMU Parameters')
    imu_group.add_argument(
        '--accel-noise',
        type=float,
        default=0.1,
        dest='accel_noise_std',
        help='Accelerometer noise std in m/s² (default: 0.1)'
    )
    imu_group.add_argument(
        '--gyro-noise',
        type=float,
        default=0.01,
        dest='gyro_noise_std',
        help='Gyroscope noise std in rad/s (default: 0.01)'
    )
    imu_group.add_argument(
        '--accel-bias-x',
        type=float,
        default=0.0,
        help='Accelerometer X-axis bias in m/s² (default: 0.0)'
    )
    imu_group.add_argument(
        '--accel-bias-y',
        type=float,
        default=0.0,
        help='Accelerometer Y-axis bias in m/s² (default: 0.0)'
    )
    imu_group.add_argument(
        '--gyro-bias',
        type=float,
        default=0.0,
        help='Gyroscope Z-axis bias in rad/s (default: 0.0)'
    )
    
    # UWB parameters
    uwb_group = parser.add_argument_group('UWB Parameters')
    uwb_group.add_argument(
        '--uwb-rate',
        type=float,
        default=10.0,
        help='UWB measurement rate in Hz (default: 10.0)'
    )
    uwb_group.add_argument(
        '--range-noise',
        type=float,
        default=0.05,
        dest='range_noise_std',
        help='UWB range noise std in meters (default: 0.05)'
    )
    uwb_group.add_argument(
        '--nlos-anchors',
        type=int,
        nargs='+',
        default=[],
        help='List of NLOS anchor indices (e.g., --nlos-anchors 1 2)'
    )
    uwb_group.add_argument(
        '--nlos-bias',
        type=float,
        default=0.5,
        help='NLOS positive bias in meters (default: 0.5)'
    )
    uwb_group.add_argument(
        '--dropout-rate',
        type=float,
        default=0.05,
        help='Measurement dropout probability per anchor (default: 0.05)'
    )
    
    # Temporal calibration
    temporal_group = parser.add_argument_group('Temporal Calibration Parameters')
    temporal_group.add_argument(
        '--time-offset',
        type=float,
        default=0.0,
        dest='time_offset_sec',
        help='UWB time offset in seconds (negative = UWB behind) (default: 0.0)'
    )
    temporal_group.add_argument(
        '--clock-drift',
        type=float,
        default=0.0,
        help='Relative clock drift (e.g., 0.0001 = 100 ppm) (default: 0.0)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle --all-variants special case
    if args.all_variants:
        print(f"\n{'='*70}")
        print("Generating all 3 standard variants...")
        print(f"{'='*70}\n")
        
        # Baseline
        print("1/3: BASELINE (no offset, no NLOS)")
        generate_fusion_2d_imu_uwb_dataset(
            output_dir="data/sim/ch8_fusion_2d_imu_uwb",
            seed=args.seed,
            duration=args.duration,
            width=args.width,
            height=args.height,
            speed=args.speed,
            dt_imu=args.dt_imu,
            accel_noise_std=args.accel_noise_std,
            gyro_noise_std=args.gyro_noise_std,
            uwb_rate=args.uwb_rate,
            range_noise_std=args.range_noise_std,
            nlos_anchors=[],
            time_offset_sec=0.0,
            clock_drift=0.0
        )
        
        # NLOS variant
        print("\n2/3: NLOS variant (anchors 1,2 biased +0.8m)")
        generate_fusion_2d_imu_uwb_dataset(
            output_dir="data/sim/ch8_fusion_2d_imu_uwb_nlos",
            seed=args.seed,
            duration=args.duration,
            width=args.width,
            height=args.height,
            speed=args.speed,
            dt_imu=args.dt_imu,
            accel_noise_std=args.accel_noise_std,
            gyro_noise_std=args.gyro_noise_std,
            uwb_rate=args.uwb_rate,
            range_noise_std=args.range_noise_std,
            nlos_anchors=[1, 2],
            nlos_bias=0.8,
            dropout_rate=args.dropout_rate,
            time_offset_sec=0.0,
            clock_drift=0.0
        )
        
        # Time offset variant
        print("\n3/3: TIME OFFSET variant (UWB 50ms behind, 100ppm drift)")
        generate_fusion_2d_imu_uwb_dataset(
            output_dir="data/sim/ch8_fusion_2d_imu_uwb_timeoffset",
            seed=args.seed,
            duration=args.duration,
            width=args.width,
            height=args.height,
            speed=args.speed,
            dt_imu=args.dt_imu,
            accel_noise_std=args.accel_noise_std,
            gyro_noise_std=args.gyro_noise_std,
            uwb_rate=args.uwb_rate,
            range_noise_std=args.range_noise_std,
            nlos_anchors=[],
            dropout_rate=args.dropout_rate,
            time_offset_sec=-0.05,
            clock_drift=0.0001
        )
        
        print(f"\n{'='*70}")
        print("All 3 variants generated successfully!")
        print(f"{'='*70}\n")
        return
    
    # If preset is specified, override with preset values
    if args.preset:
        preset_config = PRESETS[args.preset]
        print(f"\nUsing preset: '{args.preset}'")
        print(f"Description: {preset_config['description']}\n")
        
        # Override parameters with preset values
        for key, value in preset_config.items():
            if key != 'description' and hasattr(args, key):
                setattr(args, key, value)
    
    # Validate parameters
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.dt_imu <= 0 or args.dt_imu > args.duration:
        parser.error("IMU time step must be positive and less than duration")
    if args.speed <= 0:
        parser.error("Speed must be positive")
    if args.accel_noise_std < 0 or args.gyro_noise_std < 0:
        parser.error("Noise parameters must be non-negative")
    if args.range_noise_std < 0:
        parser.error("Range noise must be non-negative")
    if args.dropout_rate < 0 or args.dropout_rate > 1:
        parser.error("Dropout rate must be in [0, 1]")
    if args.nlos_anchors:
        if any(a < 0 or a > 3 for a in args.nlos_anchors):
            parser.error("NLOS anchor indices must be in [0, 3]")
    
    # Build accel_bias from individual components
    accel_bias = np.array([args.accel_bias_x, args.accel_bias_y])
    
    # Generate dataset
    generate_fusion_2d_imu_uwb_dataset(
        output_dir=args.output,
        seed=args.seed,
        width=args.width,
        height=args.height,
        speed=args.speed,
        duration=args.duration,
        dt_imu=args.dt_imu,
        accel_noise_std=args.accel_noise_std,
        gyro_noise_std=args.gyro_noise_std,
        accel_bias=accel_bias,
        gyro_bias=args.gyro_bias,
        uwb_rate=args.uwb_rate,
        range_noise_std=args.range_noise_std,
        nlos_anchors=args.nlos_anchors,
        nlos_bias=args.nlos_bias,
        dropout_rate=args.dropout_rate,
        time_offset_sec=args.time_offset_sec,
        clock_drift=args.clock_drift
    )


if __name__ == "__main__":
    main()

