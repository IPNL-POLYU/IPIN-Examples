"""
Example: Pedestrian Dead Reckoning (PDR) - Step-and-Heading

Demonstrates step-and-heading pedestrian navigation showing the critical
importance of accurate heading estimation.

Implements:
    - Step detection (Eq. 6.46)
    - Step length estimation - Weinberg model (Eq. 6.49)
    - 2D position update (Eq. 6.50)
    - Gyro heading integration vs magnetometer heading

Key Insight: Heading errors DOMINATE PDR accuracy. 1° heading error
            causes ~1.7% position error per step!

Author: Navigation Engineer
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    total_accel_magnitude,
    step_length,
    pdr_step_update,
    detect_step_simple,
    integrate_gyro_heading,
    wrap_heading,
    mag_heading,
)


def generate_corridor_walk(duration=120.0, dt=0.01, step_freq=2.0):
    """
    Generate rectangular corridor walk with turns.
    
    Returns: t, pos_true, accel_true, gyro_true, mag_true, step_events_true
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Corridor: 40m x 20m rectangle
    waypoints = np.array([
        [0, 0], [40, 0], [40, 20], [0, 20], [0, 0]  # Rectangle + return to start
    ])
    
    # Walking speed
    v_walk = 1.4  # m/s (typical walking speed)
    
    # Generate trajectory
    pos_true = np.zeros((N, 2))
    heading_true = np.zeros(N)
    vel_true = np.zeros((N, 2))
    
    # Distribute time across segments
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    segment_times = segment_lengths / v_walk
    
    current_time = 0
    current_wp = 0
    
    for k in range(N):
        if current_wp >= len(waypoints) - 1:
            # Finished, stay at final position
            pos_true[k] = waypoints[-1]
            heading_true[k] = heading_true[k-1] if k > 0 else 0
            continue
        
        # Time within current segment
        t_seg = t[k] - current_time
        
        if t_seg >= segment_times[current_wp]:
            # Move to next waypoint
            current_time += segment_times[current_wp]
            current_wp += 1
            if current_wp >= len(waypoints) - 1:
                pos_true[k] = waypoints[-1]
                continue
            t_seg = 0
        
        # Interpolate along current segment
        alpha = t_seg / segment_times[current_wp]
        pos_true[k] = (1-alpha) * waypoints[current_wp] + alpha * waypoints[current_wp+1]
        
        # Heading
        delta = waypoints[current_wp+1] - waypoints[current_wp]
        heading_true[k] = np.arctan2(delta[1], delta[0])
        
        # Velocity
        vel_true[k] = v_walk * np.array([np.cos(heading_true[k]), np.sin(heading_true[k])])
    
    # Generate IMU signals
    accel_true = np.zeros((N, 3))
    gyro_true = np.zeros((N, 3))
    mag_true = np.zeros((N, 3))
    step_events_true = np.zeros(N, dtype=bool)
    
    # Vertical oscillation from gait
    for k in range(N):
        phase = 2 * np.pi * step_freq * t[k]
        accel_true[k, 2] = 2.0 * np.sin(phase)  # Vertical accel
        
        # Step events (peaks in vertical accel)
        if k > 0 and accel_true[k-1, 2] < 0 and accel_true[k, 2] >= 0:
            step_events_true[k] = True
        
        # Gyro (heading rate)
        if k > 0:
            gyro_true[k, 2] = (heading_true[k] - heading_true[k-1]) / dt
        
        # Magnetometer (points north in horizontal plane)
        roll, pitch = 0, 0  # Assume level
        mag_north = np.array([1.0, 0.0, 0.0])  # North = [1,0,0] in map frame
        # Rotate to body frame
        R_yaw = np.array([
            [np.cos(heading_true[k]), np.sin(heading_true[k]), 0],
            [-np.sin(heading_true[k]), np.cos(heading_true[k]), 0],
            [0, 0, 1]
        ])
        mag_true[k] = R_yaw.T @ mag_north
    
    return t, pos_true, heading_true, accel_true, gyro_true, mag_true, step_events_true


def add_sensor_noise(accel_true, gyro_true, mag_true, dt):
    """Add realistic sensor noise."""
    N = len(accel_true)
    
    # IMU noise
    gyro_bias = np.random.randn(3) * np.deg2rad(50.0)  # High drift for consumer
    gyro_noise = np.random.randn(N, 3) * np.deg2rad(0.5) * np.sqrt(1/dt)
    
    accel_noise = np.random.randn(N, 3) * 0.01 * np.sqrt(1/dt)
    
    # Magnetometer noise + disturbances
    mag_noise = np.random.randn(N, 3) * 0.05
    mag_disturbance = np.zeros((N, 3))
    # Add disturbances at specific times (simulating steel structures)
    disturb_intervals = [(20, 30), (70, 80)]  # seconds
    for start, end in disturb_intervals:
        mask = (np.arange(N)*dt >= start) & (np.arange(N)*dt < end)
        mag_disturbance[mask] = np.random.randn(np.sum(mask), 3) * 0.3
    
    gyro_meas = gyro_true + gyro_bias + gyro_noise
    accel_meas = accel_true + accel_noise
    mag_meas = mag_true + mag_noise + mag_disturbance
    
    return accel_meas, gyro_meas, mag_meas


def run_pdr_gyro_heading(t, accel_meas, gyro_meas, height=1.75):
    """Run PDR with gyro-integrated heading (drifts)."""
    N = len(t)
    dt = t[1] - t[0]
    
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    step_count = 0
    
    last_step_time = 0
    last_a_mag = 10.0
    
    for k in range(1, N):
        # Step detection (Eq. 6.46) - simple peak crossing
        a_mag = total_accel_magnitude(accel_meas[k])
        is_step = (last_a_mag < 11.0 and a_mag >= 11.0)
        last_a_mag = a_mag
        
        if is_step and (t[k] - last_step_time) > 0.3:  # Minimum 0.3s between steps
            # Step detected!
            step_count += 1
            delta_t = t[k] - last_step_time
            last_step_time = t[k]
            
            # Step frequency (Eq. 6.48)
            f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            
            # Step length (Eq. 6.49 - Weinberg model)
            L = step_length(height, f_step)
            
            # Update position (Eq. 6.50)
            pos_est[k] = pdr_step_update(pos_est[k-1], L, heading_est[k-1])
        else:
            pos_est[k] = pos_est[k-1]
        
        # Integrate gyro heading
        heading_est[k] = integrate_gyro_heading(heading_est[k-1], gyro_meas[k, 2], dt)
        heading_est[k] = wrap_heading(heading_est[k])
    
    return pos_est, heading_est, step_count


def run_pdr_mag_heading(t, accel_meas, gyro_meas, mag_meas, height=1.75):
    """Run PDR with magnetometer heading (absolute but noisy)."""
    N = len(t)
    dt = t[1] - t[0]
    
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    step_count = 0
    
    last_step_time = 0
    last_a_mag = 10.0
    
    for k in range(1, N):
        # Step detection - simple peak crossing
        a_mag = total_accel_magnitude(accel_meas[k])
        is_step = (last_a_mag < 11.0 and a_mag >= 11.0)
        last_a_mag = a_mag
        
        if is_step and (t[k] - last_step_time) > 0.3:
            step_count += 1
            delta_t = t[k] - last_step_time
            last_step_time = t[k]
            
            f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            L = step_length(height, f_step)
            
            pos_est[k] = pdr_step_update(pos_est[k-1], L, heading_est[k-1])
        else:
            pos_est[k] = pos_est[k-1]
        
        # Magnetometer heading (Eqs. 6.51-6.53)
        # Assume level (roll=pitch=0 for simplicity)
        heading_est[k] = mag_heading(mag_meas[k], roll=0.0, pitch=0.0, declination=0.0)
    
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


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("Chapter 6: Pedestrian Dead Reckoning (PDR) - Step-and-Heading")
    print("="*70)
    print("\nDemonstrates the critical importance of heading accuracy in PDR.")
    print("Key equations: 6.46-6.50 (step detection, length, position update)\n")
    
    duration = 120.0
    dt = 0.01
    height = 1.75  # meters
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  User Height:     {height} m")
    print(f"  Trajectory:      40m x 20m rectangular corridor\n")
    
    print("Generating trajectory...")
    t, pos_true, heading_true, accel_true, gyro_true, mag_true, steps_true = generate_corridor_walk(duration, dt)
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    n_steps_true = np.sum(steps_true)
    print(f"  Total distance:  {total_dist:.1f} m")
    print(f"  True steps:      {n_steps_true}")
    
    print("\nAdding sensor noise...")
    accel_meas, gyro_meas, mag_meas = add_sensor_noise(accel_true, gyro_true, mag_true, dt)
    
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
    print()


if __name__ == "__main__":
    main()

