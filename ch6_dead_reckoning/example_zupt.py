"""
Example: Zero-Velocity Update (ZUPT) for Foot-Mounted IMU

Demonstrates drift correction using Zero-Velocity Updates during stance phases.
Shows the dramatic improvement ZUPT provides over pure IMU integration.

Implements:
    - IMU strapdown integration (Eqs. 6.2-6.10)
    - ZUPT detection (Eq. 6.44)
    - ZUPT pseudo-measurement (Eq. 6.45)

Key Insight: ZUPT eliminates velocity drift during stops, preventing
            unbounded position drift. Essential for foot-mounted IMU!

Author: Li-Ta Hsu
Date: December 2025
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    strapdown_update,
    detect_zupt_windowed,
    zupt_test_statistic,
    NavStateQPVP,
    units,
)
from core.sensors.ins_ekf import ZUPT_EKF, INSState
from core.sim import generate_imu_from_trajectory


def generate_walking_trajectory(
    duration=60.0, dt=0.01, step_freq=2.0, step_length=0.7, frame=None
):
    """
    Generate walking trajectory with periodic stops (stance phases).
    Uses correct IMU forward model.
    
    Args:
        duration: Total duration [s].
        dt: Time step [s].
        step_freq: Steps per second when walking.
        step_length: Length per step [m].
        frame: Frame convention (default: ENU).
    
    Returns:
        Tuple of (t, pos_true, vel_true, quat_true, accel_body, gyro_body, stance_mask).
    """
    if frame is None:
        frame = FrameConvention.create_enu()
    
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Walking pattern: walk-stop-walk-stop cycles
    # Walk for 5s, stop for 2s, repeat
    cycle_duration = 7.0  # 5s walk + 2s stop
    
    pos_true = np.zeros((N, 3))
    vel_true = np.zeros((N, 3))
    stance_mask = np.zeros(N, dtype=bool)
    
    current_pos = np.array([0.0, 0.0, 0.0])
    current_heading = 0.0  # Forward (ENU: 0=East)
    
    for k in range(N):
        t_cycle = t[k] % cycle_duration
        
        if t_cycle < 5.0:  # Walking phase
            # Walking velocity (horizontal)
            v_forward = step_length * step_freq
            vel_true[k, 0] = v_forward * np.cos(current_heading)
            vel_true[k, 1] = v_forward * np.sin(current_heading)
            
            # No vertical motion (z velocity is 0)
            vel_true[k, 2] = 0.0
            
            stance_mask[k] = False
            
            # Heading changes slightly (curved path)
            heading_rate = 0.05  # rad/s (slight turn)
            current_heading += heading_rate * dt
            
        else:  # Stance phase (stopped)
            vel_true[k] = np.array([0.0, 0.0, 0.0])
            stance_mask[k] = True
        
        # Update position
        if k > 0:
            current_pos += vel_true[k] * dt
        pos_true[k] = current_pos
    
    # Create quaternion trajectory (yaw follows heading, roll/pitch = 0)
    # Compute yaw from velocity
    yaw = np.zeros(N)
    for k in range(N):
        if np.linalg.norm(vel_true[k, :2]) > 0.01:
            yaw[k] = np.arctan2(vel_true[k, 1], vel_true[k, 0])
        elif k > 0:
            yaw[k] = yaw[k - 1]  # Maintain previous heading during stance
    
    # Convert to quaternions (scalar-first, body-to-map)
    quat_true = np.column_stack([
        np.cos(yaw / 2),
        np.zeros(N),
        np.zeros(N),
        np.sin(yaw / 2)
    ])
    
    # Generate IMU measurements using correct forward model
    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_true,
        vel_map=vel_true,
        quat_b_to_m=quat_true,
        dt=dt,
        frame=frame,
        g=9.81
    )
    
    return t, pos_true, vel_true, quat_true, accel_body, gyro_body, stance_mask


def add_imu_noise(accel_body, gyro_body, dt, imu_params: IMUNoiseParams):
    """Add realistic foot-mounted IMU noise with explicit units."""
    N = len(accel_body)
    
    # Biases
    gyro_bias = np.random.randn(3) * imu_params.gyro_bias_rad_s
    accel_bias = np.random.randn(3) * imu_params.accel_bias_mps2
    
    # White noise
    gyro_noise_std = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    accel_noise_std = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    
    gyro_noise = np.random.randn(N, 3) * gyro_noise_std
    accel_noise = np.random.randn(N, 3) * accel_noise_std
    
    gyro_meas = gyro_body + gyro_bias + gyro_noise
    accel_meas = accel_body + accel_bias + accel_noise
    
    return accel_meas, gyro_meas


def run_imu_only(t, accel_meas, gyro_meas, initial_state, frame):
    """Run pure IMU (no ZUPT)."""
    N = len(t)
    dt = t[1] - t[0]
    
    q, v, p = initial_state.q.copy(), initial_state.v.copy(), initial_state.p.copy()
    
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    
    pos_est[0], vel_est[0] = p, v
    
    for k in range(1, N):
        q, v, p = strapdown_update(q, v, p, gyro_meas[k-1], accel_meas[k-1], dt, frame=frame)
        pos_est[k], vel_est[k] = p, v
    
    return pos_est, vel_est


def run_imu_with_zupt(
    t, accel_meas, gyro_meas, initial_state, frame, imu_params,
    window_size=10, gamma=1e6
):
    """
    Run IMU with ZUPT corrections using windowed detector (Eq. 6.44).
    
    Args:
        t: Time array [s].
        accel_meas: Measured acceleration [m/s²], shape (N, 3).
        gyro_meas: Measured angular velocity [rad/s], shape (N, 3).
        initial_state: Initial NavStateQPVP.
        frame: FrameConvention.
        imu_params: IMUNoiseParams with noise specifications.
        window_size: ZUPT detector window size (samples). Default: 10.
        gamma: ZUPT detection threshold. Default: 1e6.
    
    Returns:
        Tuple of (pos_est, vel_est, zupt_detections).
    """
    N = len(t)
    dt = t[1] - t[0]
    
    q, v, p = initial_state.q.copy(), initial_state.v.copy(), initial_state.p.copy()
    
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    zupt_detections = np.zeros(N, dtype=bool)
    
    pos_est[0], vel_est[0] = p, v
    
    # Compute noise std devs for ZUPT detector (scale by sample rate)
    # IMU noise parameters are in continuous time, need to scale for discrete time
    sigma_a = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    sigma_g = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    
    for k in range(1, N):
        # Propagate
        q, v, p = strapdown_update(q, v, p, gyro_meas[k-1], accel_meas[k-1], dt, frame=frame)
        
        # ZUPT detection using windowed test statistic (Eq. 6.44)
        # Build window centered at current sample
        window_start = max(0, k - window_size // 2)
        window_end = min(N, k + window_size // 2 + 1)
        
        accel_window = accel_meas[window_start:window_end]
        gyro_window = gyro_meas[window_start:window_end]
        
        # Detect ZUPT if window has enough samples
        if len(accel_window) >= window_size // 2:
            is_stationary = detect_zupt_windowed(
                accel_window, gyro_window,
                sigma_a=sigma_a,
                sigma_g=sigma_g,
                gamma=gamma,
                g=9.81
            )
        else:
            is_stationary = False
        
        zupt_detections[k] = is_stationary
        
        # Apply ZUPT correction (Eq. 6.45): force velocity to zero
        if is_stationary:
            v = np.zeros(3)  # Simple implementation: just zero velocity
        
        pos_est[k], vel_est[k] = p, v
    
    return pos_est, vel_est, zupt_detections


def run_imu_with_zupt_ekf(
    t, accel_meas, gyro_meas, initial_state, frame, imu_params,
    window_size=10, gamma=10.0
):
    """
    Run IMU with ZUPT corrections using EKF (Eqs. 6.40-6.43 + 6.45).
    
    This is the proper implementation that uses Kalman filter measurement
    update instead of hard-coding v=0.
    
    Args:
        t: Time array [s].
        accel_meas: Measured acceleration [m/s²], shape (N, 3).
        gyro_meas: Measured angular velocity [rad/s], shape (N, 3).
        initial_state: Initial NavStateQPVP.
        frame: FrameConvention.
        imu_params: IMUNoiseParams with noise specifications.
        window_size: ZUPT detector window size (samples). Default: 10.
        gamma: ZUPT detection threshold. Default: 10.0.
    
    Returns:
        Tuple of (pos_est, vel_est, zupt_detections).
    """
    N = len(t)
    dt = t[1] - t[0]
    
    # Initialize EKF (sigma_zupt = 0.001 makes ZUPT measurements highly trusted)
    ekf = ZUPT_EKF(frame=frame, imu_params=imu_params, sigma_zupt=0.001)
    state = ekf.initialize(
        p0=initial_state.p.copy(),
        v0=initial_state.v.copy(),
        q0=initial_state.q.copy()
    )
    
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    zupt_detections = np.zeros(N, dtype=bool)
    
    pos_est[0], vel_est[0] = state.p, state.v
    
    # Compute noise std devs for ZUPT detector
    sigma_a = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    sigma_g = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    
    for k in range(1, N):
        # EKF Prediction Step
        state = ekf.predict(state, gyro_meas[k-1], accel_meas[k-1], dt)
        
        # ZUPT detection using raw measurements (bias-agnostic detector)
        window_start = max(0, k - window_size // 2)
        window_end = min(N, k + window_size // 2 + 1)
        
        accel_window = accel_meas[window_start:window_end]
        gyro_window = gyro_meas[window_start:window_end]
        
        # Detect ZUPT if window has enough samples
        if len(accel_window) >= window_size // 2:
            is_stationary = detect_zupt_windowed(
                accel_window, gyro_window,
                sigma_a=sigma_a,
                sigma_g=sigma_g,
                gamma=gamma,
                g=9.81
            )
        else:
            is_stationary = False
        
        zupt_detections[k] = is_stationary
        
        # EKF Update Step (Eqs. 6.40-6.43 + 6.45)
        if is_stationary:
            state = ekf.update_zupt(state)
        
        pos_est[k], vel_est[k] = state.p, state.v
    
    return pos_est, vel_est, zupt_detections


def plot_results(t, pos_true, pos_imu, pos_zupt, vel_imu, vel_zupt, 
                 zupt_detections, stance_mask, figs_dir):
    """Generate publication-quality plots."""
    
    # Compute errors
    error_imu = np.linalg.norm(pos_imu - pos_true, axis=1)
    error_zupt = np.linalg.norm(pos_zupt - pos_true, axis=1)
    
    # Figure 1: Trajectory comparison
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True Trajectory', zorder=1)
    ax1.plot(pos_imu[:, 0], pos_imu[:, 1], 'r--', linewidth=2, alpha=0.7, label='IMU only (no ZUPT)', zorder=2)
    ax1.plot(pos_zupt[:, 0], pos_zupt[:, 1], 'g-', linewidth=2, label='IMU + ZUPT', zorder=3)
    ax1.scatter(pos_true[0, 0], pos_true[0, 1], c='blue', s=150, marker='o', label='Start', zorder=5)
    ax1.scatter(pos_true[-1, 0], pos_true[-1, 1], c='red', s=150, marker='s', label='End', zorder=5)
    ax1.set_xlabel('East [m]', fontsize=12)
    ax1.set_ylabel('North [m]', fontsize=12)
    ax1.set_title('ZUPT Example: Trajectory Comparison (Walking with Stops)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.tight_layout()
    fig1.savefig(figs_dir / 'zupt_trajectory.svg', dpi=300, bbox_inches='tight')
    fig1.savefig(figs_dir / 'zupt_trajectory.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'zupt_trajectory.svg'}")
    
    # Figure 2: Position error comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t, error_imu, 'r-', linewidth=2, label='IMU only (no ZUPT)', alpha=0.8)
    ax2.plot(t, error_zupt, 'g-', linewidth=2, label='IMU + ZUPT')
    ax2.fill_between(t, 0, np.max(error_imu)*1.1, where=stance_mask, 
                     alpha=0.2, color='gray', label='True stance phases')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Position Error [m]', fontsize=12)
    ax2.set_title('ZUPT Example: Position Error vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])
    plt.tight_layout()
    fig2.savefig(figs_dir / 'zupt_error_time.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(figs_dir / 'zupt_error_time.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'zupt_error_time.svg'}")
    
    # Figure 3: ZUPT detector performance
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Velocity magnitude
    vel_mag_imu = np.linalg.norm(vel_imu, axis=1)
    vel_mag_zupt = np.linalg.norm(vel_zupt, axis=1)
    ax3a.plot(t, vel_mag_imu, 'r-', linewidth=1.5, label='IMU only', alpha=0.7)
    ax3a.plot(t, vel_mag_zupt, 'g-', linewidth=2, label='IMU + ZUPT')
    ax3a.fill_between(t, 0, np.max(vel_mag_imu)*1.1, where=stance_mask,
                      alpha=0.2, color='gray', label='True stance')
    ax3a.set_ylabel('Velocity [m/s]', fontsize=12)
    ax3a.set_title('ZUPT Example: Velocity and Detector Timeline', fontsize=14, fontweight='bold')
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)
    
    # ZUPT detections
    ax3b.fill_between(t, 0, 1, where=stance_mask, alpha=0.3, color='gray', label='True stance')
    ax3b.fill_between(t, 0, 1, where=zupt_detections, alpha=0.5, color='green', label='ZUPT detected')
    ax3b.set_xlabel('Time [s]', fontsize=12)
    ax3b.set_ylabel('Detection', fontsize=12)
    ax3b.set_ylim([-0.1, 1.1])
    ax3b.set_yticks([0, 1])
    ax3b.set_yticklabels(['Moving', 'Stationary'])
    ax3b.legend(fontsize=10, loc='upper right')
    ax3b.grid(True, alpha=0.3, axis='x')
    ax3b.set_xlim([0, t[-1]])
    
    plt.tight_layout()
    fig3.savefig(figs_dir / 'zupt_detector_timeline.svg', dpi=300, bbox_inches='tight')
    fig3.savefig(figs_dir / 'zupt_detector_timeline.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'zupt_detector_timeline.svg'}")
    
    plt.close('all')
    
    return error_imu, error_zupt


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("\n" + "="*70)
    print("Chapter 6: Zero-Velocity Update (ZUPT) for Foot-Mounted IMU")
    print("="*70)
    print("\nDemonstrates drift elimination using ZUPT during stance phases.")
    print("Key equations: 6.44 (ZUPT detection), 6.45 (ZUPT correction)\n")
    
    # Configuration
    duration = 60.0  # seconds
    dt = 0.01  # 100 Hz IMU
    step_freq = 2.0  # steps/second when walking
    step_length = 0.7  # meters per step
    frame = FrameConvention.create_enu()  # Use ENU frame
    imu_params = IMUNoiseParams.consumer_grade()  # Consumer-grade IMU
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  IMU Rate:        {1/dt:.0f} Hz")
    print(f"  Walking Pattern: 5s walk + 2s stop (repeated)")
    print(f"  Step Rate:       {step_freq} Hz")
    print(f"  Step Length:     {step_length} m")
    print(f"  Frame:           {frame.map_frame}\n")
    
    # Print IMU specifications
    print(imu_params.format_specs())
    print()
    
    # Generate trajectory with correct IMU forward model
    print("Generating walking trajectory with stance phases...")
    t, pos_true, vel_true, quat_true, accel_body, gyro_body, stance_mask = generate_walking_trajectory(
        duration, dt, step_freq, step_length, frame
    )
    
    total_distance = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    stance_ratio = np.sum(stance_mask) / len(stance_mask) * 100
    print(f"  Total distance:  {total_distance:.1f} m")
    print(f"  Stance time:     {stance_ratio:.1f}% of trajectory")
    
    # Add IMU noise
    print("\nAdding IMU noise...")
    accel_meas, gyro_meas = add_imu_noise(accel_body, gyro_body, dt, imu_params)
    
    # Initial state (perfect knowledge)
    initial_state = NavStateQPVP(
        q=quat_true[0].copy(),
        v=vel_true[0].copy(),
        p=pos_true[0].copy(),
    )
    
    # Run IMU-only (no ZUPT)
    print("\nRunning IMU-only integration (no ZUPT)...")
    start = time.time()
    pos_imu, vel_imu = run_imu_only(t, accel_meas, gyro_meas, initial_state, frame)
    elapsed_imu = time.time() - start
    print(f"  Computation time: {elapsed_imu:.3f} s")
    
    # Run IMU with ZUPT-EKF (proper Kalman update, Eqs. 6.40-6.43 + 6.45)
    print("\nRunning IMU + ZUPT-EKF (Kalman filter update)...")
    start = time.time()
    pos_zupt, vel_zupt, zupt_detections = run_imu_with_zupt_ekf(
        t, accel_meas, gyro_meas, initial_state, frame, imu_params,
        window_size=10, gamma=1000.0  # Much higher threshold for noisy consumer IMU
    )
    elapsed_zupt = time.time() - start
    detection_rate = np.sum(zupt_detections) / len(zupt_detections) * 100
    print(f"  Computation time: {elapsed_zupt:.3f} s")
    print(f"  ZUPT detections:  {detection_rate:.1f}% of samples")
    print(f"  Method:           EKF measurement update (not hard-coded v=0)")
    
    # Create output directory
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    error_imu, error_zupt = plot_results(
        t, pos_true, pos_imu, pos_zupt, vel_imu, vel_zupt,
        zupt_detections, stance_mask, figs_dir
    )
    
    # Compute metrics
    final_error_imu = error_imu[-1]
    final_error_zupt = error_zupt[-1]
    rmse_imu = np.sqrt(np.mean(error_imu**2))
    rmse_zupt = np.sqrt(np.mean(error_zupt**2))
    improvement = (1 - rmse_zupt/rmse_imu) * 100
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"IMU-only (no ZUPT):")
    print(f"  Final error:  {final_error_imu:.2f} m ({final_error_imu/total_distance*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_imu:.2f} m")
    print()
    print(f"IMU + ZUPT:")
    print(f"  Final error:  {final_error_zupt:.2f} m ({final_error_zupt/total_distance*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_zupt:.2f} m")
    print()
    print(f"Improvement:    {improvement:.1f}% reduction in RMSE")
    print()
    print(f"Figures saved to: {figs_dir}/")
    print()
    print("="*70)
    print("KEY INSIGHT: ZUPT-EKF corrects velocity drift using Kalman updates!")
    print("             Eqs. 6.40-6.43 (Kalman filter) + Eq. 6.45 (ZUPT measurement)")
    print("             Essential for foot-mounted IMU navigation.")
    print("             Typical improvement: >90% error reduction.")
    print("="*70)
    print()


if __name__ == "__main__":
    main()


