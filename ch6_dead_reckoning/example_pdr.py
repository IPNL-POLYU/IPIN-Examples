"""
Example: Pedestrian Dead Reckoning (PDR) - Step-and-Heading

Demonstrates step-and-heading pedestrian navigation showing the critical
importance of accurate heading estimation.

Can run with:
    - Pre-generated dataset: python example_pdr.py --data ch6_pdr_corridor_walk
    - Inline data (default): python example_pdr.py

Implements:
    - Step detection (Eq. 6.46)
    - Step length estimation - Weinberg model (Eq. 6.49)
    - 2D position update (Eq. 6.50)
    - Gyro heading integration vs magnetometer heading

Key Insight: Heading errors DOMINATE PDR accuracy. 1° heading error
            causes ~1.7% position error per step!

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    total_accel_magnitude,
    detect_steps_peak_detector,
    step_length,
    pdr_step_update,
    detect_step_simple,
    integrate_gyro_heading,
    wrap_heading,
    mag_heading,
    units,
)
from core.sim import generate_imu_from_trajectory


def load_pdr_dataset(data_dir: str) -> Dict:
    """Load PDR dataset from directory.
    
    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch6_pdr_corridor_walk')
    
    Returns:
        Dictionary with time, ground truth, and sensor measurements
    """
    path = Path(data_dir)
    
    data = {
        't': np.loadtxt(path / 'time.txt'),
        'pos_true': np.loadtxt(path / 'ground_truth_position.txt'),
        'heading_true': np.loadtxt(path / 'ground_truth_heading.txt'),
        'accel_meas': np.loadtxt(path / 'accel.txt'),
        'gyro_meas': np.loadtxt(path / 'gyro.txt'),
        'mag_meas': np.loadtxt(path / 'magnetometer.txt'),
        'step_times': np.loadtxt(path / 'step_times.txt'),
    }
    
    # Load config if available
    config_path = path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            data['config'] = json.load(f)
    
    return data


def run_pdr_from_dataset(data: Dict, height: float = 1.75) -> Dict:
    """Run PDR algorithm on loaded dataset.
    
    Uses the book's peak detection method (Eqs. 6.46-6.47) for step detection:
    1. Compute total acceleration magnitude (6.46)
    2. Subtract gravity (6.47)
    3. Filter the signal
    4. Detect peaks
    
    Args:
        data: Dataset dictionary from load_pdr_dataset
        height: Pedestrian height in meters
    
    Returns:
        Dictionary with estimated positions and headings
    """
    t = data['t']
    accel_meas = data['accel_meas']
    gyro_meas = data['gyro_meas']
    mag_meas = data['mag_meas']
    
    N = len(t)
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    fs = 1.0 / dt  # Sampling frequency
    
    # Detect steps using peak detector (Eqs. 6.46-6.47)
    # Tune parameters based on dataset sampling rate:
    # - min_peak_height: 1.0 m/s² (typical walking peak above gravity)
    # - min_peak_distance: 0.3s minimum (max ~3.3 steps/s for fast walking)
    # - lowpass_cutoff: 5 Hz (removes high-frequency noise, preserves step dynamics)
    print(f"  Detecting steps using peak detector (Eqs. 6.46-6.47) at {fs:.1f} Hz...")
    step_indices, accel_processed = detect_steps_peak_detector(
        accel_meas,
        dt=dt,
        g=9.81,
        min_peak_height=1.0,  # m/s² above gravity
        min_peak_distance=0.3,  # seconds between steps
        lowpass_cutoff=5.0  # Hz low-pass filter
    )
    
    print(f"  Detected {len(step_indices)} steps")
    
    # Initialize outputs
    pos_gyro = np.zeros((N, 2))
    pos_mag = np.zeros((N, 2))
    heading_gyro = np.zeros(N)
    heading_mag = np.zeros(N)
    
    # Initialize headings
    heading_gyro[0] = 0.0
    heading_mag[0] = mag_heading(mag_meas[0], roll=0.0, pitch=0.0, declination=0.0)
    
    # Run PDR with gyro heading
    for k in range(1, N):
        # Integrate gyro heading
        heading_gyro[k] = integrate_gyro_heading(heading_gyro[k-1], gyro_meas[k, 2], dt)
        heading_gyro[k] = wrap_heading(heading_gyro[k])
        
        # Update position on step events
        if k in step_indices:
            # Find previous step for delta_t calculation
            prev_steps = step_indices[step_indices < k]
            if len(prev_steps) > 0:
                last_step_idx = prev_steps[-1]
                delta_t = t[k] - t[last_step_idx]
                f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            else:
                f_step = 2.0  # Default for first step
            
            # Step length (Eq. 6.49 - Weinberg model)
            L = step_length(height, f_step)
            
            # Update position (Eq. 6.50)
            pos_gyro[k] = pdr_step_update(pos_gyro[k-1], L, heading_gyro[k-1])
        else:
            pos_gyro[k] = pos_gyro[k-1]
    
    # Run PDR with magnetometer heading
    for k in range(1, N):
        # Magnetometer heading (Eqs. 6.51-6.53)
        heading_mag[k] = mag_heading(mag_meas[k], roll=0.0, pitch=0.0, declination=0.0)
        
        # Update position on step events
        if k in step_indices:
            # Find previous step for delta_t calculation
            prev_steps = step_indices[step_indices < k]
            if len(prev_steps) > 0:
                last_step_idx = prev_steps[-1]
                delta_t = t[k] - t[last_step_idx]
                f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            else:
                f_step = 2.0  # Default for first step
            
            # Step length (Eq. 6.49)
            L = step_length(height, f_step)
            
            # Update position (Eq. 6.50)
            pos_mag[k] = pdr_step_update(pos_mag[k-1], L, heading_mag[k-1])
        else:
            pos_mag[k] = pos_mag[k-1]
    
    return {
        't': t,
        'pos_gyro': pos_gyro,
        'pos_mag': pos_mag,
        'heading_gyro': heading_gyro,
        'heading_mag': heading_mag,
        'step_count_gyro': len(step_indices),
        'step_count_mag': len(step_indices),
        'step_indices': step_indices,
    }


def run_with_dataset(data_dir: str, height: float = 1.75) -> None:
    """Run PDR example using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
        height: Pedestrian height in meters
    """
    print("\n" + "="*70)
    print("Chapter 6: Pedestrian Dead Reckoning (PDR)")
    print(f"Using dataset: {data_dir}")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    data = load_pdr_dataset(data_dir)
    config = data.get('config', {})
    
    t = data['t']
    pos_true = data['pos_true']
    heading_true = data['heading_true']
    step_times = data['step_times']
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    
    print(f"\nDataset Info:")
    print(f"  Duration: {t[-1]:.1f} s")
    print(f"  Total distance: {total_dist:.1f} m")
    print(f"  True steps: {len(step_times)}")
    print(f"  User height: {height} m")
    
    # Run PDR
    print("\nRunning PDR algorithms...")
    start = time.time()
    results = run_pdr_from_dataset(data, height)
    elapsed = time.time() - start
    
    print(f"  Processing time: {elapsed:.3f} s")
    print(f"  Steps detected (gyro): {results['step_count_gyro']}")
    print(f"  Steps detected (mag): {results['step_count_mag']}")
    
    # Compute errors
    error_gyro = np.linalg.norm(results['pos_gyro'] - pos_true, axis=1)
    error_mag = np.linalg.norm(results['pos_mag'] - pos_true, axis=1)
    
    rmse_gyro = np.sqrt(np.mean(error_gyro**2))
    rmse_mag = np.sqrt(np.mean(error_mag**2))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"PDR (Gyro Heading - drifts unbounded):")
    print(f"  Final error:  {error_gyro[-1]:.1f} m ({error_gyro[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_gyro:.1f} m")
    print()
    print(f"PDR (Magnetometer Heading - absolute but noisy):")
    print(f"  Final error:  {error_mag[-1]:.1f} m ({error_mag[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_mag:.1f} m")
    
    # Plot results
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PDR: Dataset Analysis', fontsize=14, fontweight='bold')
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True Path')
    ax.plot(results['pos_gyro'][:, 0], results['pos_gyro'][:, 1], 'r--', linewidth=2, alpha=0.7, label='PDR (Gyro)')
    ax.plot(results['pos_mag'][:, 0], results['pos_mag'][:, 1], 'b-', linewidth=2, label='PDR (Mag)')
    ax.scatter(0, 0, c='g', s=150, marker='o', label='Start', zorder=5)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('PDR Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Position error
    ax = axes[0, 1]
    ax.plot(t, error_gyro, 'r-', linewidth=2, label='Gyro Heading')
    ax.plot(t, error_mag, 'b-', linewidth=2, label='Mag Heading')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Position Error vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heading comparison
    ax = axes[1, 0]
    ax.plot(t, np.rad2deg(heading_true), 'k-', linewidth=2, label='True')
    ax.plot(t, np.rad2deg(results['heading_gyro']), 'r--', linewidth=2, alpha=0.7, label='Gyro')
    ax.plot(t, np.rad2deg(results['heading_mag']), 'b-', linewidth=1.5, alpha=0.7, label='Mag')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading [deg]')
    ax.set_title('Heading Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heading error
    ax = axes[1, 1]
    heading_error_gyro = np.abs(wrap_heading(results['heading_gyro'] - heading_true))
    heading_error_mag = np.abs(wrap_heading(results['heading_mag'] - heading_true))
    ax.plot(t, np.rad2deg(heading_error_gyro), 'r-', linewidth=2, label='Gyro Error')
    ax.plot(t, np.rad2deg(heading_error_mag), 'b-', linewidth=2, label='Mag Error')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading Error [deg]')
    ax.set_title('Heading Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = figs_dir / 'pdr_dataset_results.svg'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Heading errors DOMINATE PDR accuracy!")
    print("             Gyro drifts unbounded -> unusable alone.")
    print("             Magnetometer provides absolute reference (with noise).")
    print("="*70)


def generate_corridor_walk(duration=120.0, dt=0.01, step_freq=2.0, frame=None):
    """
    Generate rectangular corridor walk with turns and synthetic walking dynamics.
    Uses correct IMU forward model with added vertical oscillations for step detection.
    
    Returns: t, pos_true, heading_true, accel_body, gyro_body, mag_body, expected_steps
    """
    if frame is None:
        frame = FrameConvention.create_enu()
    
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Corridor: 40m x 20m rectangle
    waypoints = np.array([
        [0, 0], [40, 0], [40, 20], [0, 20], [0, 0]  # Rectangle + return to start
    ])
    
    # Walking speed
    v_walk = 1.4  # m/s (typical walking speed)
    
    # Generate trajectory (2D horizontal + z=0)
    pos_2d = np.zeros((N, 2))
    heading_true = np.zeros(N)
    vel_2d = np.zeros((N, 2))
    
    # Distribute time across segments
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = segment_lengths / v_walk
    
    current_time = 0
    current_wp = 0
    
    for k in range(N):
        if current_wp >= len(waypoints) - 1:
            # Finished, stay at final position
            pos_2d[k] = waypoints[-1]
            heading_true[k] = heading_true[k-1] if k > 0 else 0
            continue
        
        # Time within current segment
        t_seg = t[k] - current_time
        
        if t_seg >= segment_times[current_wp]:
            # Move to next waypoint
            current_time += segment_times[current_wp]
            current_wp += 1
            if current_wp >= len(waypoints) - 1:
                pos_2d[k] = waypoints[-1]
                continue
            t_seg = 0
        
        # Interpolate along current segment
        alpha = t_seg / segment_times[current_wp]
        pos_2d[k] = (1-alpha) * waypoints[current_wp] + alpha * waypoints[current_wp+1]
        
        # Heading
        delta = waypoints[current_wp+1] - waypoints[current_wp]
        heading_true[k] = np.arctan2(delta[1], delta[0])
        
        # Velocity
        vel_2d[k] = v_walk * np.array([np.cos(heading_true[k]), np.sin(heading_true[k])])
    
    # Convert to 3D trajectory (z=0, vz=0)
    pos_map = np.column_stack([pos_2d, np.zeros(N)])
    vel_map = np.column_stack([vel_2d, np.zeros(N)])
    
    # Create quaternion trajectory (yaw only, roll/pitch = 0)
    quat_b_to_m = np.column_stack([
        np.cos(heading_true / 2),
        np.zeros(N),
        np.zeros(N),
        np.sin(heading_true / 2)
    ])
    
    # Add synthetic walking accelerations (vertical oscillations for step detection)
    # Walking creates periodic vertical accelerations at step frequency
    # Amplitude: ~2-3 m/s² (typical for walking)
    walking_accel_amplitude = 2.5  # m/s²
    walking_accel_z = walking_accel_amplitude * np.sin(2 * np.pi * step_freq * t)
    
    # Modify velocity to include these oscillations (integrate accel)
    # This is a simplified model - real walking has complex 3D motion
    vel_map_with_steps = vel_map.copy()
    vel_map_with_steps[:, 2] += walking_accel_amplitude / (2 * np.pi * step_freq) * np.cos(2 * np.pi * step_freq * t)
    
    # Generate IMU measurements using correct forward model
    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_map,
        vel_map=vel_map_with_steps,
        quat_b_to_m=quat_b_to_m,
        dt=dt,
        frame=frame,
        g=9.81
    )
    
    # Generate magnetometer measurements (points to magnetic north in body frame)
    mag_body = np.zeros((N, 3))
    mag_north_map = np.array([1.0, 0.0, 0.0])  # North = x-axis in ENU map frame (conventionally)
    
    for k in range(N):
        # Rotate north vector from map to body frame
        # C_M^B = (C_B^M)^T
        yaw = heading_true[k]
        C_yaw = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        mag_body[k] = C_yaw.T @ mag_north_map
    
    # Compute expected number of steps based on trajectory
    # Total walking time * step frequency = number of steps
    total_distance = np.sum(segment_lengths)
    walking_time = total_distance / v_walk
    expected_steps = int(walking_time * step_freq)
    
    return t, pos_2d, heading_true, accel_body, gyro_body, mag_body, expected_steps


def add_sensor_noise(accel_body, gyro_body, mag_body, dt, imu_params: IMUNoiseParams):
    """Add realistic sensor noise with explicit units."""
    N = len(accel_body)
    
    # IMU noise and biases
    gyro_bias = np.random.randn(3) * imu_params.gyro_bias_rad_s
    gyro_noise_std = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    gyro_noise = np.random.randn(N, 3) * gyro_noise_std
    
    accel_noise_std = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    accel_noise = np.random.randn(N, 3) * accel_noise_std
    
    # Magnetometer noise + disturbances
    mag_noise = np.random.randn(N, 3) * 0.05
    mag_disturbance = np.zeros((N, 3))
    # Add disturbances at specific times (simulating steel structures)
    disturb_intervals = [(20, 30), (70, 80)]  # seconds
    for start, end in disturb_intervals:
        mask = (np.arange(N)*dt >= start) & (np.arange(N)*dt < end)
        mag_disturbance[mask] = np.random.randn(np.sum(mask), 3) * 0.3
    
    gyro_meas = gyro_body + gyro_bias + gyro_noise
    accel_meas = accel_body + accel_noise
    mag_meas = mag_body + mag_noise + mag_disturbance
    
    return accel_meas, gyro_meas, mag_meas


def run_pdr_gyro_heading(t, accel_meas, gyro_meas, height=1.75):
    """
    Run PDR with gyro-integrated heading (drifts).
    
    Uses proper peak detection (Eqs. 6.46-6.47) instead of threshold crossing.
    """
    N = len(t)
    dt = t[1] - t[0]
    
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    
    # Step detection using peak detector (Eqs. 6.46-6.47)
    print("  Detecting steps using peak detector (Eqs. 6.46-6.47)...")
    step_indices, accel_processed = detect_steps_peak_detector(
        accel_meas,
        dt=dt,
        g=9.81,
        min_peak_height=1.0,  # 1 m/s² above gravity
        min_peak_distance=0.3,  # 0.3s between steps (max ~3.3 steps/s)
        lowpass_cutoff=5.0  # 5 Hz low-pass filter
    )
    
    step_count = len(step_indices)
    print(f"  Detected {step_count} steps using peak detection")
    
    # Initialize heading
    heading_est[0] = 0.0
    
    # Process time series
    for k in range(1, N):
        # Integrate gyro heading
        heading_est[k] = integrate_gyro_heading(heading_est[k-1], gyro_meas[k, 2], dt)
        heading_est[k] = wrap_heading(heading_est[k])
        
        # Update position on step events
        if k in step_indices:
            # Find previous step for delta_t calculation
            prev_steps = step_indices[step_indices < k]
            if len(prev_steps) > 0:
                last_step_idx = prev_steps[-1]
                delta_t = t[k] - t[last_step_idx]
                f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            else:
                f_step = 2.0  # Default for first step
            
            # Step length (Eq. 6.49 - Weinberg model)
            L = step_length(height, f_step)
            
            # Update position (Eq. 6.50)
            pos_est[k] = pdr_step_update(pos_est[k-1], L, heading_est[k-1])
        else:
            pos_est[k] = pos_est[k-1]
    
    return pos_est, heading_est, step_count


def run_pdr_mag_heading(t, accel_meas, gyro_meas, mag_meas, height=1.75):
    """
    Run PDR with magnetometer heading (absolute but noisy).
    
    Uses proper peak detection (Eqs. 6.46-6.47) instead of threshold crossing.
    """
    N = len(t)
    dt = t[1] - t[0]
    
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    
    # Step detection using peak detector (Eqs. 6.46-6.47)
    print("  Detecting steps using peak detector (Eqs. 6.46-6.47)...")
    step_indices, accel_processed = detect_steps_peak_detector(
        accel_meas,
        dt=dt,
        g=9.81,
        min_peak_height=1.0,  # 1 m/s² above gravity
        min_peak_distance=0.3,  # 0.3s between steps
        lowpass_cutoff=5.0  # 5 Hz low-pass filter
    )
    
    step_count = len(step_indices)
    print(f"  Detected {step_count} steps using peak detection")
    
    # Initialize heading
    heading_est[0] = mag_heading(mag_meas[0], roll=0.0, pitch=0.0, declination=0.0)
    
    # Process time series
    for k in range(1, N):
        # Magnetometer heading (Eqs. 6.51-6.53)
        # Assume level (roll=pitch=0 for simplicity)
        heading_est[k] = mag_heading(mag_meas[k], roll=0.0, pitch=0.0, declination=0.0)
        
        # Update position on step events
        if k in step_indices:
            # Find previous step for delta_t calculation
            prev_steps = step_indices[step_indices < k]
            if len(prev_steps) > 0:
                last_step_idx = prev_steps[-1]
                delta_t = t[k] - t[last_step_idx]
                f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            else:
                f_step = 2.0  # Default for first step
            
            # Step length (Eq. 6.49)
            L = step_length(height, f_step)
            
            # Update position (Eq. 6.50)
            pos_est[k] = pdr_step_update(pos_est[k-1], L, heading_est[k-1])
        else:
            pos_est[k] = pos_est[k-1]
    
    return pos_est, heading_est, step_count


def plot_results(t, pos_true, pos_gyro, pos_mag, heading_true, heading_gyro, heading_mag, figs_dir):
    """Generate publication-quality plots."""
    
    error_gyro = np.linalg.norm(pos_gyro - pos_true, axis=1)
    error_mag = np.linalg.norm(pos_mag - pos_true, axis=1)
    
    # Figure 1: Trajectory
    fig1, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True Path')
    ax.plot(pos_gyro[:, 0], pos_gyro[:, 1], 'r--', linewidth=2, alpha=0.7, label='PDR (Gyro Heading)')
    ax.plot(pos_mag[:, 0], pos_mag[:, 1], 'b-', linewidth=2, label='PDR (Mag Heading)')
    ax.scatter(0, 0, c='g', s=150, marker='o', label='Start', zorder=5)
    ax.set_xlabel('East [m]', fontsize=12)
    ax.set_ylabel('North [m]', fontsize=12)
    ax.set_title('PDR Example: Corridor Walk (Rectangular Path)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    fig1.savefig(figs_dir / 'pdr_trajectory.svg', dpi=300, bbox_inches='tight')
    fig1.savefig(figs_dir / 'pdr_trajectory.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'pdr_trajectory.svg'}")
    
    # Figure 2: Heading comparison
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(t, np.rad2deg(heading_true), 'k-', linewidth=2, label='True Heading')
    ax1.plot(t, np.rad2deg(heading_gyro), 'r--', linewidth=2, alpha=0.7, label='Gyro Integrated')
    ax1.plot(t, np.rad2deg(heading_mag), 'b-', linewidth=1.5, alpha=0.7, label='Magnetometer')
    ax1.set_ylabel('Heading [deg]', fontsize=12)
    ax1.set_title('PDR Example: Heading Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    heading_error_gyro = np.abs(wrap_heading(heading_gyro - heading_true))
    heading_error_mag = np.abs(wrap_heading(heading_mag - heading_true))
    ax2.plot(t, np.rad2deg(heading_error_gyro), 'r-', linewidth=2, label='Gyro Error')
    ax2.plot(t, np.rad2deg(heading_error_mag), 'b-', linewidth=2, label='Mag Error')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Heading Error [deg]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])
    
    plt.tight_layout()
    fig2.savefig(figs_dir / 'pdr_heading.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(figs_dir / 'pdr_heading.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'pdr_heading.svg'}")
    
    # Figure 3: Position error
    fig3, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, error_gyro, 'r-', linewidth=2, label='PDR (Gyro Heading)')
    ax.plot(t, error_mag, 'b-', linewidth=2, label='PDR (Mag Heading)')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Position Error [m]', fontsize=12)
    ax.set_title('PDR Example: Position Error vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])
    plt.tight_layout()
    fig3.savefig(figs_dir / 'pdr_error.svg', dpi=300, bbox_inches='tight')
    fig3.savefig(figs_dir / 'pdr_error.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'pdr_error.svg'}")
    
    plt.close('all')
    
    return error_gyro, error_mag


def run_with_inline_data():
    """Run with inline generated data (original behavior)."""
    print("\n" + "="*70)
    print("Chapter 6: Pedestrian Dead Reckoning (PDR) - Step-and-Heading")
    print("(Using inline generated data)")
    print("="*70)
    print("\nDemonstrates the critical importance of heading accuracy in PDR.")
    print("Key equations: 6.46-6.50 (step detection, length, position update)\n")
    
    duration = 120.0
    dt = 0.01
    height = 1.75  # meters
    frame = FrameConvention.create_enu()  # Use ENU frame
    # Use higher gyro bias for PDR to show heading drift
    imu_params = IMUNoiseParams(
        gyro_bias_rad_s=units.deg_per_hour_to_rad_per_sec(50.0),  # 50 deg/hr
        gyro_arw_rad_sqrt_s=units.deg_per_sqrt_hour_to_rad_per_sqrt_sec(0.5),
        gyro_rrw_rad_s_sqrt_s=0.0,
        accel_bias_mps2=units.mg_to_mps2(10.0),
        accel_vrw_mps_sqrt_s=units.mps_per_sqrt_hour_to_mps_per_sqrt_sec(0.01),
        grade='consumer (high gyro drift)'
    )
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  User Height:     {height} m")
    print(f"  Trajectory:      40m x 20m rectangular corridor")
    print(f"  Frame:           {frame.map_frame}\n")
    
    # Print IMU specifications
    print(imu_params.format_specs())
    print()
    
    print("Generating trajectory with correct IMU forward model...")
    t, pos_true, heading_true, accel_body, gyro_body, mag_body, expected_steps = generate_corridor_walk(
        duration, dt, step_freq=2.0, frame=frame
    )
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    print(f"  Total distance:  {total_dist:.1f} m")
    print(f"  Expected steps:  {expected_steps} (at 2.0 Hz step frequency)")
    
    print("\nAdding sensor noise...")
    accel_meas, gyro_meas, mag_meas = add_sensor_noise(accel_body, gyro_body, mag_body, dt, imu_params)
    
    print("\nRunning PDR with gyro heading...")
    start = time.time()
    pos_gyro, heading_gyro, steps_gyro = run_pdr_gyro_heading(t, accel_meas, gyro_meas, height)
    print(f"  Time: {time.time()-start:.3f} s, Steps detected: {steps_gyro}")
    
    print("\nRunning PDR with magnetometer heading...")
    start = time.time()
    pos_mag, heading_mag, steps_mag = run_pdr_mag_heading(t, accel_meas, gyro_meas, mag_meas, height)
    print(f"  Time: {time.time()-start:.3f} s, Steps detected: {steps_mag}")
    
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    error_gyro, error_mag = plot_results(
        t, pos_true, pos_gyro, pos_mag, heading_true, heading_gyro, heading_mag, figs_dir
    )
    
    # Metrics
    rmse_gyro = np.sqrt(np.mean(error_gyro**2))
    rmse_mag = np.sqrt(np.mean(error_mag**2))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"PDR (Gyro Heading - drifts unbounded):")
    print(f"  Final error:  {error_gyro[-1]:.1f} m ({error_gyro[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_gyro:.1f} m")
    print()
    print(f"PDR (Magnetometer Heading - absolute but noisy):")
    print(f"  Final error:  {error_mag[-1]:.1f} m ({error_mag[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_mag:.1f} m")
    print()
    print(f"Figures saved to: {figs_dir}/")
    print()
    print("="*70)
    print("KEY INSIGHT: Heading errors DOMINATE PDR accuracy!")
    print("             Gyro drifts unbounded -> unusable alone.")
    print("             Magnetometer provides absolute reference (with noise).")
    print("             Best practice: Complementary filter (gyro + mag).")
    print("="*70)
    print("\nTip: Run with --data ch6_pdr_corridor_walk to use pre-generated dataset")


def main():
    """Main execution with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chapter 6: Pedestrian Dead Reckoning (PDR) Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python example_pdr.py
  
  # Run with pre-generated dataset
  python example_pdr.py --data ch6_pdr_corridor_walk
  
  # Specify pedestrian height
  python example_pdr.py --data ch6_pdr_corridor_walk --height 1.80
        """
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name or path (e.g., 'ch6_pdr_corridor_walk' or full path)"
    )
    parser.add_argument(
        "--height", type=float, default=1.75,
        help="Pedestrian height in meters (default: 1.75)"
    )
    
    args = parser.parse_args()
    
    if args.data:
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
                    if d.is_dir() and d.name.startswith("ch6"):
                        print(f"  - {d.name}")
            return
        
        run_with_dataset(str(data_path), height=args.height)
    else:
        run_with_inline_data()


if __name__ == "__main__":
    main()

