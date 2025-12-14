"""Generate synthetic 2D IMU + UWB fusion dataset for Chapter 8 examples.

Creates a realistic indoor fusion scenario with:
    - 2D rectangular walking trajectory
    - High-rate IMU measurements (100 Hz): accel_xy, gyro_z
    - Low-rate UWB range measurements (10 Hz) to 4 corner anchors
    - Ground truth: position, velocity, heading
    - Configurable noise, time offset, and NLOS bias

Saves to: data/sim/fusion_2d_imu_uwb/

Author: Navigation Engineer
Date: December 2025
References: Chapter 8 - Sensor Fusion
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


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
    output_dir: str = "data/sim/fusion_2d_imu_uwb",
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


if __name__ == "__main__":
    # Generate baseline dataset (no time offset, no NLOS)
    print("\nGenerating BASELINE dataset (no offset, no NLOS)...")
    generate_fusion_2d_imu_uwb_dataset(
        output_dir="data/sim/fusion_2d_imu_uwb",
        seed=42,
        duration=60.0,
        nlos_anchors=[],
        time_offset_sec=0.0,
        clock_drift=0.0
    )
    
    # Optional: Generate variant with NLOS for robust demo
    print("\n" + "="*70)
    print("Generating NLOS variant for robust loss demo...")
    generate_fusion_2d_imu_uwb_dataset(
        output_dir="data/sim/fusion_2d_imu_uwb_nlos",
        seed=42,
        duration=60.0,
        nlos_anchors=[1, 2],  # Anchors 1 and 2 have NLOS
        nlos_bias=0.8,
        time_offset_sec=0.0,
        clock_drift=0.0
    )
    
    # Optional: Generate variant with time offset for temporal calibration demo
    print("\n" + "="*70)
    print("Generating TIME OFFSET variant for temporal calibration demo...")
    generate_fusion_2d_imu_uwb_dataset(
        output_dir="data/sim/fusion_2d_imu_uwb_timeoffset",
        seed=42,
        duration=60.0,
        nlos_anchors=[],
        time_offset_sec=-0.05,  # UWB 50ms behind IMU
        clock_drift=0.0001,     # 100 ppm drift
    )
    
    print("\n" + "="*70)
    print("All dataset variants generated successfully!")
    print("="*70)

