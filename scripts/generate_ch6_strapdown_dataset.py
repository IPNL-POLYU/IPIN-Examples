"""Generate IMU strapdown integration dataset for Chapter 6 examples.

Creates pure IMU trajectory to demonstrate unbounded drift from inertial integration:
    - 2D circular trajectory
    - High-rate IMU measurements (100 Hz): accel_xy, gyro_z
    - Configurable noise and bias levels
    - Ground truth: position, velocity, heading
    - No external corrections (demonstrates drift)

Saves to: data/sim/ch6_strapdown_basic/

Author: Navigation Engineer
Date: December 2025
References: Chapter 6 - Dead Reckoning (Sections 6.1-6.2)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'tactical': {
        'description': 'Tactical-grade IMU (low noise, minimal drift)',
        'accel_noise_std': 0.01,
        'gyro_noise_std': 0.001,
        'accel_bias_x': 0.0,
        'accel_bias_y': 0.0,
        'gyro_bias': 0.0,
    },
    'consumer': {
        'description': 'Consumer-grade IMU (smartphone-like, moderate drift)',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'accel_bias_x': 0.0,
        'accel_bias_y': 0.0,
        'gyro_bias': 0.0,
    },
    'mems': {
        'description': 'MEMS-grade IMU (high noise, significant drift)',
        'accel_noise_std': 0.5,
        'gyro_noise_std': 0.05,
        'accel_bias_x': 0.0,
        'accel_bias_y': 0.0,
        'gyro_bias': 0.0,
    },
    'biased_consumer': {
        'description': 'Consumer IMU with constant bias (systematic drift)',
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
        'accel_bias_x': 0.05,
        'accel_bias_y': 0.03,
        'gyro_bias': 0.002,
    },
}


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_circular_trajectory(
    radius: float = 10.0,
    speed: float = 1.0,
    dt: float = 0.01,
    duration: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D circular trajectory at constant speed.
    
    Args:
        radius: Circle radius (meters).
        speed: Constant speed (m/s).
        dt: Time step (seconds).
        duration: Total duration (seconds).
    
    Returns:
        Tuple of (t, p_xy, v_xy, yaw):
            t: timestamps (N,)
            p_xy: positions (N, 2) in meters
            v_xy: velocities (N, 2) in m/s
            yaw: heading angles (N,) in radians
    
    References:
        Circular motion for Ch6 drift demonstration.
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Angular velocity for circular motion
    omega = speed / radius  # rad/s
    
    # Position (center at origin)
    theta = omega * t
    p_xy = np.zeros((N, 2))
    p_xy[:, 0] = radius * np.cos(theta)  # x = r cos(θ)
    p_xy[:, 1] = radius * np.sin(theta)  # y = r sin(θ)
    
    # Velocity (tangent to circle)
    v_xy = np.zeros((N, 2))
    v_xy[:, 0] = -radius * omega * np.sin(theta)  # vx = -rω sin(θ)
    v_xy[:, 1] = radius * omega * np.cos(theta)   # vy = rω cos(θ)
    
    # Heading (tangent direction)
    yaw = theta + np.pi / 2  # perpendicular to radius
    
    return t, p_xy, v_xy, yaw


# ============================================================================
# IMU MEASUREMENT GENERATION
# ============================================================================

def generate_imu_measurements(
    t: np.ndarray,
    v_xy: np.ndarray,
    yaw: np.ndarray,
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01,
    accel_bias_x: float = 0.0,
    accel_bias_y: float = 0.0,
    gyro_bias: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic IMU measurements from ground truth.
    
    Args:
        t: timestamps (N,)
        v_xy: velocities (N, 2) in m/s
        yaw: heading angles (N,) in radians
        accel_noise_std: Accelerometer noise std (m/s²)
        gyro_noise_std: Gyroscope noise std (rad/s)
        accel_bias_x: X-axis accelerometer bias (m/s²)
        accel_bias_y: Y-axis accelerometer bias (m/s²)
        gyro_bias: Z-axis gyroscope bias (rad/s)
    
    Returns:
        Tuple of (t_imu, accel_xy, gyro_z):
            t_imu: IMU timestamps (N,)
            accel_xy: 2D accelerations (N, 2) in m/s²
            gyro_z: Yaw rate (N,) in rad/s
    
    References:
        IMU error model from Ch6, Eqs. (6.5), (6.9).
    """
    N = len(t)
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    
    # Compute true accelerations (derivative of velocity)
    accel_xy_true = np.gradient(v_xy, axis=0) / dt[:, None]
    
    # Compute true yaw rate (derivative of yaw)
    yaw_unwrapped = np.unwrap(yaw)
    gyro_z_true = np.gradient(yaw_unwrapped) / dt
    
    # Add noise and bias
    accel_bias = np.array([accel_bias_x, accel_bias_y])
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


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_ch6_strapdown_dataset(
    output_dir: str = "data/sim/ch6_strapdown_basic",
    seed: int = 42,
    # Trajectory parameters
    radius: float = 10.0,
    speed: float = 1.0,
    duration: float = 60.0,
    dt: float = 0.01,  # 100 Hz
    # IMU parameters
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01,
    accel_bias_x: float = 0.0,
    accel_bias_y: float = 0.0,
    gyro_bias: float = 0.0,
) -> None:
    """Generate and save IMU strapdown dataset.
    
    Args:
        output_dir: Output directory path.
        seed: Random seed for reproducibility.
        radius: Circle radius (meters).
        speed: Constant speed (m/s).
        duration: Dataset duration (seconds).
        dt: Time step (seconds).
        accel_noise_std: Accelerometer noise std (m/s²).
        gyro_noise_std: Gyroscope noise std (rad/s).
        accel_bias_x: X-axis accel bias (m/s²).
        accel_bias_y: Y-axis accel bias (m/s²).
        gyro_bias: Z-axis gyro bias (rad/s).
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Generating Ch6 IMU Strapdown Dataset")
    print(f"{'='*70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate ground truth trajectory
    print(f"\n1. Generating circular trajectory...")
    print(f"   Radius: {radius} m")
    print(f"   Speed: {speed} m/s")
    print(f"   Duration: {duration} s")
    print(f"   IMU rate: {1/dt:.0f} Hz")
    
    t, p_xy, v_xy, yaw = generate_circular_trajectory(
        radius=radius,
        speed=speed,
        dt=dt,
        duration=duration
    )
    
    print(f"   Generated {len(t)} samples")
    omega = speed / radius
    period = 2 * np.pi / omega
    print(f"   Circular motion: omega={omega:.3f} rad/s, period={period:.1f}s")
    
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
    print(f"   Accel bias: [{accel_bias_x}, {accel_bias_y}] m/s²")
    print(f"   Gyro bias: {gyro_bias} rad/s")
    
    t_imu, accel_xy, gyro_z = generate_imu_measurements(
        t, v_xy, yaw,
        accel_noise_std=accel_noise_std,
        gyro_noise_std=gyro_noise_std,
        accel_bias_x=accel_bias_x,
        accel_bias_y=accel_bias_y,
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
    
    # 3. Save configuration
    print(f"\n3. Saving configuration...")
    
    config = {
        "dataset_info": {
            "description": "IMU strapdown integration dataset for Ch6",
            "seed": seed,
            "duration_sec": duration,
            "num_samples": int(len(t))
        },
        "trajectory": {
            "type": "circular",
            "radius_m": radius,
            "speed_m_s": speed
        },
        "imu": {
            "rate_hz": 1 / dt,
            "dt_sec": dt,
            "accel_noise_std_m_s2": accel_noise_std,
            "gyro_noise_std_rad_s": gyro_noise_std,
            "accel_bias_m_s2": [accel_bias_x, accel_bias_y],
            "gyro_bias_rad_s": gyro_bias
        },
        "coordinate_frame": {
            "description": "ENU (East-North-Up)",
            "origin": "Circle center at (0, 0)",
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
    print(f"  - truth.npz   : Ground truth (t, p_xy, v_xy, yaw)")
    print(f"  - imu.npz     : IMU measurements (t, accel_xy, gyro_z)")
    print(f"  - config.json : Dataset configuration")
    print(f"\nDataset statistics:")
    print(f"  Duration    : {duration:.1f} s")
    print(f"  Samples     : {len(t)} ({1/dt:.0f} Hz)")
    print(f"  Laps        : {duration / period:.2f}")
    print(f"\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate IMU strapdown integration dataset for Chapter 6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default parameters (consumer-grade IMU)
  python %(prog)s

  # Use a preset configuration
  python %(prog)s --preset tactical

  # Generate MEMS-grade IMU dataset
  python %(prog)s --preset mems --output data/sim/ch6_strapdown_mems

  # Custom parameters with bias
  python %(prog)s --accel-noise 0.2 --accel-bias-x 0.05 --gyro-bias 0.002

  # Longer trajectory
  python %(prog)s --duration 120 --output data/sim/ch6_strapdown_long

Available presets: """ + ", ".join(PRESETS.keys())
    )
    
    # Preset configuration
    parser.add_argument(
        '--preset',
        type=str,
        choices=PRESETS.keys(),
        help='Use preset configuration (overrides individual parameters)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/sim/ch6_strapdown_basic',
        help='Output directory (default: data/sim/ch6_strapdown_basic)'
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
        '--radius',
        type=float,
        default=10.0,
        help='Circle radius in meters (default: 10.0)'
    )
    traj_group.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Constant speed in m/s (default: 1.0)'
    )
    traj_group.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Trajectory duration in seconds (default: 60.0)'
    )
    traj_group.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step in seconds (default: 0.01, i.e., 100 Hz)'
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
    
    # Parse arguments
    args = parser.parse_args()
    
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
    if args.dt <= 0 or args.dt > args.duration:
        parser.error("Time step must be positive and less than duration")
    if args.speed <= 0:
        parser.error("Speed must be positive")
    if args.radius <= 0:
        parser.error("Radius must be positive")
    if args.accel_noise_std < 0 or args.gyro_noise_std < 0:
        parser.error("Noise parameters must be non-negative")
    
    # Generate dataset
    generate_ch6_strapdown_dataset(
        output_dir=args.output,
        seed=args.seed,
        radius=args.radius,
        speed=args.speed,
        duration=args.duration,
        dt=args.dt,
        accel_noise_std=args.accel_noise_std,
        gyro_noise_std=args.gyro_noise_std,
        accel_bias_x=args.accel_bias_x,
        accel_bias_y=args.accel_bias_y,
        gyro_bias=args.gyro_bias
    )


if __name__ == "__main__":
    main()

