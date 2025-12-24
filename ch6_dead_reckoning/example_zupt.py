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
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    strapdown_update,
    detect_zupt,
    NavStateQPVP,
)


def generate_walking_trajectory(duration=60.0, dt=0.01, step_freq=2.0, step_length=0.7):
    """
    Generate walking trajectory with periodic stops (stance phases).
    
    Args:
        duration: Total duration [s].
        dt: Time step [s].
        step_freq: Steps per second when walking.
        step_length: Length per step [m].
    
    Returns:
        Tuple of (t, pos_true, vel_true, accel_true, gyro_true, stance_mask).
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Walking pattern: walk-stop-walk-stop cycles
    # Walk for 5s, stop for 2s, repeat
    cycle_duration = 7.0  # 5s walk + 2s stop
    
    pos_true = np.zeros((N, 3))
    vel_true = np.zeros((N, 3))
    accel_true = np.zeros((N, 3))
    gyro_true = np.zeros((N, 3))
    stance_mask = np.zeros(N, dtype=bool)
    
    current_pos = np.array([0.0, 0.0, 0.0])
    current_heading = 0.0  # Forward (north)
    
    for k in range(N):
        t_cycle = t[k] % cycle_duration
        
        if t_cycle < 5.0:  # Walking phase
            # Walking velocity
            v_forward = step_length * step_freq
            vel_true[k] = v_forward * np.array([np.cos(current_heading), np.sin(current_heading), 0.0])
            
            # Vertical oscillation (simulates gait)
            phase = 2 * np.pi * step_freq * t[k]
            accel_z = 2.0 * np.sin(phase)  # Vertical acceleration
            accel_true[k] = np.array([0.0, 0.0, accel_z])
            
            stance_mask[k] = False
            
        else:  # Stance phase (stopped)
            vel_true[k] = np.array([0.0, 0.0, 0.0])
            accel_true[k] = np.array([0.0, 0.0, 0.0])
            stance_mask[k] = True
        
        # Update position
        if k > 0:
            current_pos += vel_true[k] * dt
        pos_true[k] = current_pos
        
        # Heading changes slightly (curved path)
        if t_cycle < 5.0:
            heading_rate = 0.05  # rad/s (slight turn)
            gyro_true[k, 2] = heading_rate
            current_heading += heading_rate * dt
    
    return t, pos_true, vel_true, accel_true, gyro_true, stance_mask


def add_imu_noise(accel_true, gyro_true, dt):
    """Add realistic foot-mounted IMU noise."""
    N = len(accel_true)
    
    # Consumer-grade foot IMU
    gyro_bias = np.random.randn(3) * np.deg2rad(10.0)
    gyro_noise = np.random.randn(N, 3) * np.deg2rad(0.1) * np.sqrt(1/dt)
    
    accel_bias = np.random.randn(3) * 0.01  # 10 mg
    accel_noise = np.random.randn(N, 3) * 0.001 * np.sqrt(1/dt)
    
    gyro_meas = gyro_true + gyro_bias + gyro_noise
    accel_meas = accel_true + accel_bias + accel_noise
    
    return accel_meas, gyro_meas


def run_imu_only(t, accel_meas, gyro_meas, initial_state):
    """Run pure IMU (no ZUPT)."""
    N = len(t)
    dt = t[1] - t[0]
    
    q, v, p = initial_state.q.copy(), initial_state.v.copy(), initial_state.p.copy()
    
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    
    pos_est[0], vel_est[0] = p, v
    
    for k in range(1, N):
        q, v, p = strapdown_update(q, v, p, gyro_meas[k-1], accel_meas[k-1], dt)
        pos_est[k], vel_est[k] = p, v
    
    return pos_est, vel_est


def run_imu_with_zupt(t, accel_meas, gyro_meas, initial_state, delta_omega=0.05, delta_f=0.5):
    """Run IMU with ZUPT corrections."""
    N = len(t)
    dt = t[1] - t[0]
    
    q, v, p = initial_state.q.copy(), initial_state.v.copy(), initial_state.p.copy()
    
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    zupt_detections = np.zeros(N, dtype=bool)
    
    pos_est[0], vel_est[0] = p, v
    
    for k in range(1, N):
        # Propagate
        q, v, p = strapdown_update(q, v, p, gyro_meas[k-1], accel_meas[k-1], dt)
        
        # ZUPT detection (Eq. 6.44)
        is_stationary = detect_zupt(gyro_meas[k], accel_meas[k], delta_omega, delta_f)
        zupt_detections[k] = is_stationary
        
        # Apply ZUPT correction (Eq. 6.45): force velocity to zero
        if is_stationary:
            v = np.zeros(3)  # Simple implementation: just zero velocity
        
        pos_est[k], vel_est[k] = p, v
    
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
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  IMU Rate:        {1/dt:.0f} Hz")
    print(f"  Walking Pattern: 5s walk + 2s stop (repeated)")
    print(f"  Step Rate:       {step_freq} Hz")
    print(f"  Step Length:     {step_length} m\n")
    
    # Generate trajectory
    print("Generating walking trajectory with stance phases...")
    t, pos_true, vel_true, accel_true, gyro_true, stance_mask = generate_walking_trajectory(
        duration, dt, step_freq, step_length
    )
    
    total_distance = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    stance_ratio = np.sum(stance_mask) / len(stance_mask) * 100
    print(f"  Total distance:  {total_distance:.1f} m")
    print(f"  Stance time:     {stance_ratio:.1f}% of trajectory")
    
    # Add IMU noise
    print("\nAdding IMU noise...")
    accel_meas, gyro_meas = add_imu_noise(accel_true, gyro_true, dt)
    
    # Initial state
    initial_state = NavStateQPVP(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        v=vel_true[0],
        p=pos_true[0],
    )
    
    # Run IMU-only (no ZUPT)
    print("\nRunning IMU-only integration (no ZUPT)...")
    start = time.time()
    pos_imu, vel_imu = run_imu_only(t, accel_meas, gyro_meas, initial_state)
    elapsed_imu = time.time() - start
    print(f"  Computation time: {elapsed_imu:.3f} s")
    
    # Run IMU with ZUPT
    print("\nRunning IMU + ZUPT integration...")
    start = time.time()
    pos_zupt, vel_zupt, zupt_detections = run_imu_with_zupt(
        t, accel_meas, gyro_meas, initial_state,
        delta_omega=0.05, delta_f=0.5
    )
    elapsed_zupt = time.time() - start
    detection_rate = np.sum(zupt_detections) / len(zupt_detections) * 100
    print(f"  Computation time: {elapsed_zupt:.3f} s")
    print(f"  ZUPT detections:  {detection_rate:.1f}% of samples")
    
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
    print("KEY INSIGHT: ZUPT eliminates velocity drift during stops!")
    print("             Essential for foot-mounted IMU navigation.")
    print("             Typical improvement: >90% error reduction.")
    print("="*70)
    print()


if __name__ == "__main__":
    main()


