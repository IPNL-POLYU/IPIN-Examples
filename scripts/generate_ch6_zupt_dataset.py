"""Generate foot-mounted IMU with ZUPT dataset for Chapter 6 examples.

Creates walking trajectory with stance phases for Zero-Velocity UPdaTe (ZUPT):
    - 2D walking path with stops
    - High-rate IMU measurements (100 Hz): accel_xy, gyro_z
    - Stance phase detection labels
    - Ground truth: position, velocity, heading
    - Demonstrates dramatic drift reduction with ZUPT constraints

Saves to: data/sim/ch6_foot_zupt_walk/

Author: Navigation Engineer
Date: December 2025
References: Chapter 6 - Dead Reckoning (Sections 6.3, Eqs. 6.44-6.45)
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
    'baseline': {
        'description': 'Standard walking with clear stance phases',
        'step_length': 0.7,
        'step_duration': 0.6,
        'stance_duration': 0.2,
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
    },
    'fast_walk': {
        'description': 'Fast walking with shorter stance phases',
        'step_length': 0.9,
        'step_duration': 0.5,
        'stance_duration': 0.15,
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
    },
    'slow_walk': {
        'description': 'Slow walking with longer stance phases',
        'step_length': 0.5,
        'step_duration': 0.8,
        'stance_duration': 0.3,
        'accel_noise_std': 0.1,
        'gyro_noise_std': 0.01,
    },
    'noisy_imu': {
        'description': 'Degraded IMU to test ZUPT robustness',
        'step_length': 0.7,
        'step_duration': 0.6,
        'stance_duration': 0.2,
        'accel_noise_std': 0.3,
        'gyro_noise_std': 0.03,
    },
}


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_walking_trajectory(
    num_steps: int = 20,
    step_length: float = 0.7,
    step_duration: float = 0.6,
    stance_duration: float = 0.2,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D walking trajectory with stance phases.
    
    Args:
        num_steps: Number of steps to take.
        step_length: Distance per step (meters).
        step_duration: Time per step including stance (seconds).
        stance_duration: Time foot is stationary per step (seconds).
        dt: Time step (seconds).
    
    Returns:
        Tuple of (t, p_xy, v_xy, yaw, is_stance):
            t: timestamps (N,)
            p_xy: positions (N, 2) in meters
            v_xy: velocities (N, 2) in m/s
            yaw: heading angles (N,) in radians (constant, walking forward)
            is_stance: stance phase indicator (N,) boolean
    
    References:
        Walking gait model for ZUPT demonstration (Ch6, Section 6.3).
    """
    # Total duration
    total_duration = num_steps * step_duration
    t = np.arange(0, total_duration, dt)
    N = len(t)
    
    # Initialize arrays
    p_xy = np.zeros((N, 2))
    v_xy = np.zeros((N, 2))
    yaw = np.zeros(N)
    is_stance = np.zeros(N, dtype=bool)
    
    # Walking forward (constant heading)
    heading = 0.0  # East direction
    yaw[:] = heading
    
    # Generate step-by-step motion
    current_pos = 0.0  # Position along x-axis
    
    for i, time in enumerate(t):
        # Determine which step we're in
        step_idx = int(time / step_duration)
        time_in_step = time - step_idx * step_duration
        
        if step_idx >= num_steps:
            # After all steps, stationary
            p_xy[i] = [current_pos, 0.0]
            v_xy[i] = [0.0, 0.0]
            is_stance[i] = True
        elif time_in_step < stance_duration:
            # Stance phase (foot stationary)
            if step_idx > 0:
                current_pos = step_idx * step_length
            p_xy[i] = [current_pos, 0.0]
            v_xy[i] = [0.0, 0.0]
            is_stance[i] = True
        else:
            # Swing phase (foot moving)
            swing_duration = step_duration - stance_duration
            time_in_swing = time_in_step - stance_duration
            progress = time_in_swing / swing_duration  # 0 to 1
            
            # Position during swing (linear interpolation)
            start_pos = step_idx * step_length
            end_pos = (step_idx + 1) * step_length
            current_pos = start_pos + progress * (end_pos - start_pos)
            p_xy[i] = [current_pos, 0.0]
            
            # Velocity during swing (constant)
            v_xy[i] = [step_length / swing_duration, 0.0]
            is_stance[i] = False
    
    return t, p_xy, v_xy, yaw, is_stance


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

def generate_ch6_zupt_dataset(
    output_dir: str = "data/sim/ch6_foot_zupt_walk",
    seed: int = 42,
    # Trajectory parameters
    num_steps: int = 20,
    step_length: float = 0.7,
    step_duration: float = 0.6,
    stance_duration: float = 0.2,
    dt: float = 0.01,  # 100 Hz
    # IMU parameters
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01,
    accel_bias_x: float = 0.0,
    accel_bias_y: float = 0.0,
    gyro_bias: float = 0.0,
) -> None:
    """Generate and save foot-mounted IMU with ZUPT dataset.
    
    Args:
        output_dir: Output directory path.
        seed: Random seed for reproducibility.
        num_steps: Number of steps to take.
        step_length: Distance per step (meters).
        step_duration: Time per step including stance (seconds).
        stance_duration: Time foot is stationary per step (seconds).
        dt: Time step (seconds).
        accel_noise_std: Accelerometer noise std (m/s²).
        gyro_noise_std: Gyroscope noise std (rad/s).
        accel_bias_x: X-axis accel bias (m/s²).
        accel_bias_y: Y-axis accel bias (m/s²).
        gyro_bias: Z-axis gyro bias (rad/s).
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Generating Ch6 Foot-Mounted IMU with ZUPT Dataset")
    print(f"{'='*70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate ground truth trajectory
    print(f"\n1. Generating walking trajectory...")
    print(f"   Steps: {num_steps}")
    print(f"   Step length: {step_length} m")
    print(f"   Step duration: {step_duration} s")
    print(f"   Stance duration: {stance_duration} s")
    print(f"   IMU rate: {1/dt:.0f} Hz")
    
    t, p_xy, v_xy, yaw, is_stance = generate_walking_trajectory(
        num_steps=num_steps,
        step_length=step_length,
        step_duration=step_duration,
        stance_duration=stance_duration,
        dt=dt
    )
    
    print(f"   Generated {len(t)} samples")
    total_distance = num_steps * step_length
    duration = t[-1]
    stance_ratio = np.sum(is_stance) / len(is_stance)
    print(f"   Total distance: {total_distance:.1f} m")
    print(f"   Duration: {duration:.1f} s")
    print(f"   Stance phases: {stance_ratio*100:.1f}% of time")
    
    # Save ground truth
    np.savez(
        output_path / "truth.npz",
        t=t,
        p_xy=p_xy,
        v_xy=v_xy,
        yaw=yaw,
        is_stance=is_stance
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
    
    swing_duration = step_duration - stance_duration
    config = {
        "dataset_info": {
            "description": "Foot-mounted IMU with ZUPT for Ch6",
            "seed": seed,
            "duration_sec": float(duration),
            "num_samples": int(len(t)),
            "total_distance_m": float(total_distance)
        },
        "trajectory": {
            "type": "walking_linear",
            "num_steps": num_steps,
            "step_length_m": step_length,
            "step_duration_sec": step_duration,
            "stance_duration_sec": stance_duration,
            "swing_duration_sec": float(swing_duration),
            "stance_ratio": float(stance_ratio)
        },
        "imu": {
            "rate_hz": 1 / dt,
            "dt_sec": dt,
            "accel_noise_std_m_s2": accel_noise_std,
            "gyro_noise_std_rad_s": gyro_noise_std,
            "accel_bias_m_s2": [accel_bias_x, accel_bias_y],
            "gyro_bias_rad_s": gyro_bias
        },
        "zupt": {
            "stance_threshold_description": "Use is_stance from truth.npz for ideal ZUPT",
            "detection_note": "In practice, detect stance from IMU statistics"
        },
        "coordinate_frame": {
            "description": "ENU (East-North-Up)",
            "origin": "Starting position at (0, 0)",
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
    print(f"  - truth.npz   : Ground truth (t, p_xy, v_xy, yaw, is_stance)")
    print(f"  - imu.npz     : IMU measurements (t, accel_xy, gyro_z)")
    print(f"  - config.json : Dataset configuration")
    print(f"\nDataset statistics:")
    print(f"  Duration        : {duration:.1f} s")
    print(f"  Samples         : {len(t)} ({1/dt:.0f} Hz)")
    print(f"  Steps           : {num_steps}")
    print(f"  Total distance  : {total_distance:.1f} m")
    print(f"  Stance phases   : {np.sum(is_stance)} samples ({stance_ratio*100:.1f}%)")
    print(f"\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate foot-mounted IMU with ZUPT dataset for Chapter 6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default parameters
  python %(prog)s

  # Use a preset configuration
  python %(prog)s --preset baseline

  # Fast walking with shorter stance phases
  python %(prog)s --preset fast_walk

  # Custom parameters
  python %(prog)s --num-steps 30 --step-length 0.8 --stance-duration 0.15

  # Test ZUPT with noisy IMU
  python %(prog)s --preset noisy_imu --output data/sim/ch6_zupt_noisy

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
        default='data/sim/ch6_foot_zupt_walk',
        help='Output directory (default: data/sim/ch6_foot_zupt_walk)'
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
        '--num-steps',
        type=int,
        default=20,
        help='Number of steps to take (default: 20)'
    )
    traj_group.add_argument(
        '--step-length',
        type=float,
        default=0.7,
        help='Distance per step in meters (default: 0.7)'
    )
    traj_group.add_argument(
        '--step-duration',
        type=float,
        default=0.6,
        help='Time per step in seconds (default: 0.6)'
    )
    traj_group.add_argument(
        '--stance-duration',
        type=float,
        default=0.2,
        help='Time foot is stationary per step in seconds (default: 0.2)'
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
            if key != 'description':
                # Handle both underscore and hyphen versions
                key_underscore = key.replace('-', '_')
                if hasattr(args, key_underscore):
                    setattr(args, key_underscore, value)
    
    # Validate parameters
    if args.num_steps <= 0:
        parser.error("Number of steps must be positive")
    if args.step_length <= 0:
        parser.error("Step length must be positive")
    if args.step_duration <= 0:
        parser.error("Step duration must be positive")
    if args.stance_duration < 0 or args.stance_duration >= args.step_duration:
        parser.error("Stance duration must be >= 0 and < step duration")
    if args.dt <= 0:
        parser.error("Time step must be positive")
    if args.accel_noise_std < 0 or args.gyro_noise_std < 0:
        parser.error("Noise parameters must be non-negative")
    
    # Generate dataset
    generate_ch6_zupt_dataset(
        output_dir=args.output,
        seed=args.seed,
        num_steps=args.num_steps,
        step_length=args.step_length,
        step_duration=args.step_duration,
        stance_duration=args.stance_duration,
        dt=args.dt,
        accel_noise_std=args.accel_noise_std,
        gyro_noise_std=args.gyro_noise_std,
        accel_bias_x=args.accel_bias_x,
        accel_bias_y=args.accel_bias_y,
        gyro_bias=args.gyro_bias
    )


if __name__ == "__main__":
    main()


