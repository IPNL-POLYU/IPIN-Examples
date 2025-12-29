"""
Example: Comprehensive Comparison of Dead Reckoning Methods

Compares all Chapter 6 dead reckoning approaches on a common trajectory:
    1. IMU Strapdown (pure, no corrections)
    2. IMU + ZUPT (foot-mounted with stance detection)
    3. Wheel Odometry (vehicle)
    4. Pedestrian DR (step-and-heading with magnetometer)

Demonstrates the trade-offs between different approaches and the critical
importance of drift correction.

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
    wheel_odom_update,
    total_accel_magnitude,
    step_length,
    pdr_step_update,
    detect_step_simple,
    mag_heading,
    NavStateQPVP,
    units,
)
from core.sim import generate_imu_from_trajectory


def generate_mixed_trajectory(duration=120.0, dt=0.01, frame=None):
    """
    Generate trajectory suitable for multiple DR methods.
    Walking-style motion with periodic stops.
    Uses correct IMU forward model.
    """
    if frame is None:
        frame = FrameConvention.create_enu()
    
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Simple rectangular path: 30m x 20m
    waypoints = np.array([[0, 0], [30, 0], [30, 20], [0, 20], [0, 0]])
    v_walk = 1.2  # m/s
    
    # Calculate segment properties
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = segment_lengths / v_walk
    
    # Add stops between segments
    stop_duration = 3.0  # seconds
    
    pos_true = np.zeros((N, 3))
    vel_true = np.zeros((N, 3))
    heading_true = np.zeros(N)
    stance_mask = np.zeros(N, dtype=bool)
    wheel_speed_true = np.zeros((N, 3))
    
    current_time = 0
    current_wp = 0
    in_stop = False
    stop_start = 0
    
    for k in range(N):
        if current_wp >= len(waypoints) - 1:
            # Finished
            if k == 0:
                pos_true[k, :2] = waypoints[-1]
            else:
                pos_true[k] = pos_true[k-1]
            stance_mask[k] = True
            continue
        
        # Check if in stop phase
        if in_stop:
            if t[k] - stop_start >= stop_duration:
                # End stop, move to next segment
                in_stop = False
                current_time = t[k]
                current_wp += 1
                if current_wp >= len(waypoints) - 1:
                    pos_true[k, :2] = waypoints[-1]
                    stance_mask[k] = True
                    continue
            else:
                # Still stopped
                if k > 0:
                    pos_true[k] = pos_true[k-1]
                else:
                    pos_true[k, :2] = waypoints[current_wp]
                stance_mask[k] = True
                heading_true[k] = heading_true[k-1] if k > 0 else 0
                continue
        
        # Walking phase
        t_seg = t[k] - current_time
        
        if t_seg >= segment_times[current_wp]:
            # Reached waypoint, enter stop
            in_stop = True
            stop_start = t[k]
            pos_true[k, :2] = waypoints[current_wp + 1]
            stance_mask[k] = True
            continue
        
        # Interpolate along segment
        alpha = t_seg / segment_times[current_wp]
        pos_true[k, :2] = (1-alpha) * waypoints[current_wp] + alpha * waypoints[current_wp+1]
        
        # Heading
        delta = waypoints[current_wp+1] - waypoints[current_wp]
        heading_true[k] = np.arctan2(delta[1], delta[0])
        
        # Velocity
        vel_true[k, :2] = v_walk * np.array([np.cos(heading_true[k]), np.sin(heading_true[k])])
        
        # Wheel speed (for vehicle scenario, book convention: y=forward)
        wheel_speed_true[k] = np.array([0, v_walk, 0])
    
    # Create quaternion trajectory (yaw only, roll/pitch = 0)
    quat_true = np.column_stack([
        np.cos(heading_true / 2),
        np.zeros(N),
        np.zeros(N),
        np.sin(heading_true / 2)
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
    
    # Generate magnetometer measurements (points to magnetic north in body frame)
    mag_body = np.zeros((N, 3))
    mag_north_map = np.array([1.0, 0.0, 0.0])  # North = x-axis in ENU (conventionally)
    
    for k in range(N):
        # Rotate north vector from map to body frame
        yaw = heading_true[k]
        C_yaw = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        mag_body[k] = C_yaw.T @ mag_north_map
    
    return t, pos_true, vel_true, accel_body, gyro_body, heading_true, mag_body, stance_mask, wheel_speed_true


def add_sensor_noise(accel_body, gyro_body, mag_body, wheel_true, dt, imu_params: IMUNoiseParams):
    """Add noise to all sensors with explicit units."""
    N = len(accel_body)
    
    # IMU noise and biases
    gyro_bias = np.random.randn(3) * imu_params.gyro_bias_rad_s
    gyro_noise_std = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    gyro_noise = np.random.randn(N, 3) * gyro_noise_std
    
    accel_bias = np.random.randn(3) * imu_params.accel_bias_mps2
    accel_noise_std = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    accel_noise = np.random.randn(N, 3) * accel_noise_std
    
    gyro_meas = gyro_body + gyro_bias + gyro_noise
    accel_meas = accel_body + accel_bias + accel_noise
    
    # Magnetometer noise
    mag_noise = np.random.randn(N, 3) * 0.05
    mag_meas = mag_body + mag_noise
    
    # Wheel encoder noise
    wheel_noise = np.random.randn(N, 3) * 0.05
    wheel_meas = wheel_true + wheel_noise
    
    return accel_meas, gyro_meas, mag_meas, wheel_meas


def run_imu_only(t, accel, gyro, initial, frame):
    """Method 1: Pure IMU strapdown."""
    N, dt = len(t), t[1] - t[0]
    q, v, p = initial.q.copy(), initial.v.copy(), initial.p.copy()
    pos = np.zeros((N, 3))
    pos[0] = p
    
    for k in range(1, N):
        q, v, p = strapdown_update(q, v, p, gyro[k-1], accel[k-1], dt, frame=frame)
        pos[k] = p
    return pos


def run_imu_zupt(t, accel, gyro, initial, frame, imu_params, window_size=10, gamma=1e6):
    """Method 2: IMU + ZUPT (windowed detector, Eq. 6.44)."""
    N, dt = len(t), t[1] - t[0]
    q, v, p = initial.q.copy(), initial.v.copy(), initial.p.copy()
    pos = np.zeros((N, 3))
    pos[0] = p
    
    # Compute noise std devs for ZUPT detector
    sigma_a = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    sigma_g = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    
    for k in range(1, N):
        q, v, p = strapdown_update(q, v, p, gyro[k-1], accel[k-1], dt, frame=frame)
        
        # Windowed ZUPT detection
        window_start = max(0, k - window_size // 2)
        window_end = min(N, k + window_size // 2 + 1)
        accel_window = accel[window_start:window_end]
        gyro_window = gyro[window_start:window_end]
        
        if len(accel_window) >= window_size // 2:
            if detect_zupt_windowed(accel_window, gyro_window, sigma_a, sigma_g, gamma):
                v = np.zeros(3)
        
        pos[k] = p
    return pos


def run_wheel_odom(t, wheel, gyro, initial, lever_arm):
    """Method 3: Wheel odometry."""
    N, dt = len(t), t[1] - t[0]
    p = initial.p.copy()
    q = initial.q.copy()
    pos = np.zeros((N, 3))
    pos[0] = p
    
    for k in range(1, N):
        p = wheel_odom_update(p, q, wheel[k-1], gyro[k-1], lever_arm, dt)
        # Update quaternion from gyro
        q_new = q.copy()
        dq = 0.5 * dt * np.array([
            -q[1]*gyro[k-1,0] - q[2]*gyro[k-1,1] - q[3]*gyro[k-1,2],
            q[0]*gyro[k-1,0] + q[2]*gyro[k-1,2] - q[3]*gyro[k-1,1],
            q[0]*gyro[k-1,1] - q[1]*gyro[k-1,2] + q[3]*gyro[k-1,0],
            q[0]*gyro[k-1,2] + q[1]*gyro[k-1,1] - q[2]*gyro[k-1,0]
        ])
        q = q_new + dq
        q = q / np.linalg.norm(q)
        pos[k] = p
    return pos


def run_pdr(t, accel, gyro, mag, height=1.75):
    """Method 4: Pedestrian DR with magnetometer heading."""
    N, dt = len(t), t[1] - t[0]
    pos = np.zeros((N, 2))
    heading = 0
    last_step_time = 0
    last_a_mag = 10.0
    
    for k in range(1, N):
        a_mag = total_accel_magnitude(accel[k])
        
        # Simple step detection: peak crossing
        is_step = (last_a_mag < 11.0 and a_mag >= 11.0)
        last_a_mag = a_mag
        
        if is_step and (t[k] - last_step_time) > 0.3:
            delta_t = t[k] - last_step_time
            last_step_time = t[k]
            f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            L = step_length(height, f_step)
            pos[k] = pdr_step_update(pos[k-1], L, heading)
        else:
            pos[k] = pos[k-1]
        
        heading = mag_heading(mag[k], 0, 0, 0)
    
    pos_3d = np.column_stack([pos, np.zeros(N)])
    return pos_3d


def plot_comparison(t, pos_true, results, figs_dir):
    """Generate comprehensive comparison plots."""
    
    # Figure 1: All trajectories
    fig1, ax = plt.subplots(figsize=(14, 10))
    ax.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True Trajectory', zorder=1)
    
    colors = ['red', 'green', 'blue', 'orange']
    styles = ['--', '-', '-.', ':']
    
    for i, (name, pos) in enumerate(results.items()):
        ax.plot(pos[:, 0], pos[:, 1], color=colors[i], linestyle=styles[i], 
                linewidth=2, alpha=0.8, label=name, zorder=2+i)
    
    ax.scatter(0, 0, c='lime', s=200, marker='o', edgecolors='black', linewidth=2, 
               label='Start', zorder=10)
    ax.set_xlabel('East [m]', fontsize=13)
    ax.set_ylabel('North [m]', fontsize=13)
    ax.set_title('Chapter 6 Comparison: All Dead Reckoning Methods', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    fig1.savefig(figs_dir / 'comparison_trajectories.svg', dpi=300, bbox_inches='tight')
    fig1.savefig(figs_dir / 'comparison_trajectories.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'comparison_trajectories.svg'}")
    
    # Figure 2: Error vs time
    fig2, ax = plt.subplots(figsize=(14, 7))
    
    for i, (name, pos) in enumerate(results.items()):
        error = np.linalg.norm(pos - pos_true, axis=1)
        ax.plot(t, error, color=colors[i], linestyle=styles[i], 
                linewidth=2, label=name)
    
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Position Error [m]', fontsize=13)
    ax.set_title('Chapter 6 Comparison: Position Error vs Time', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])
    ax.set_yscale('log')
    plt.tight_layout()
    fig2.savefig(figs_dir / 'comparison_error_time.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(figs_dir / 'comparison_error_time.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'comparison_error_time.svg'}")
    
    # Figure 3: Error CDF
    fig3, ax = plt.subplots(figsize=(12, 7))
    
    for i, (name, pos) in enumerate(results.items()):
        error = np.linalg.norm(pos - pos_true, axis=1)
        sorted_err = np.sort(error)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
        ax.plot(sorted_err, cdf*100, color=colors[i], linestyle=styles[i],
                linewidth=2, label=name)
    
    ax.set_xlabel('Position Error [m]', fontsize=13)
    ax.set_ylabel('CDF [%]', fontsize=13)
    ax.set_title('Chapter 6 Comparison: Error CDF', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 100])
    plt.tight_layout()
    fig3.savefig(figs_dir / 'comparison_error_cdf.svg', dpi=300, bbox_inches='tight')
    fig3.savefig(figs_dir / 'comparison_error_cdf.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'comparison_error_cdf.svg'}")
    
    plt.close('all')
    
    # Compute metrics
    metrics = {}
    for name, pos in results.items():
        error = np.linalg.norm(pos - pos_true, axis=1)
        metrics[name] = {
            'rmse': np.sqrt(np.mean(error**2)),
            'final': error[-1],
            'median': np.median(error),
            'p90': np.percentile(error, 90),
        }
    
    return metrics


def main():
    """Main execution."""
    print("\n" + "="*75)
    print("Chapter 6: COMPREHENSIVE COMPARISON of Dead Reckoning Methods")
    print("="*75)
    print("\nCompares all major DR approaches on a common trajectory.")
    print("Demonstrates trade-offs and the critical need for drift correction.\n")
    
    duration = 120.0
    dt = 0.01
    height = 1.75
    frame = FrameConvention.create_enu()  # Use ENU frame
    imu_params = IMUNoiseParams.consumer_grade()  # Consumer-grade IMU
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  Trajectory:      30m x 20m rectangular path with stops")
    print(f"  IMU Rate:        {1/dt:.0f} Hz")
    print(f"  Frame:           {frame.map_frame}\n")
    
    # Print IMU specifications
    print(imu_params.format_specs())
    print()
    
    print("Generating trajectory with correct IMU forward model...")
    t, pos_true, vel_true, accel_body, gyro_body, heading_true, mag_body, stance, wheel_true = \
        generate_mixed_trajectory(duration, dt, frame)
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    print(f"  Total distance:  {total_dist:.1f} m")
    
    print("\nAdding sensor noise...")
    accel_meas, gyro_meas, mag_meas, wheel_meas = add_sensor_noise(
        accel_body, gyro_body, mag_body, wheel_true, dt, imu_params)
    
    initial = NavStateQPVP(q=np.array([1, 0, 0, 0]), v=vel_true[0], p=pos_true[0])
    lever_arm = np.array([1.0, 0, -0.2])
    
    # Run all methods
    print("\nRunning all methods...")
    methods = {}
    
    print("  1. IMU only (pure strapdown)...")
    start = time.time()
    methods['IMU Only'] = run_imu_only(t, accel_meas, gyro_meas, initial, frame)
    print(f"     Time: {time.time()-start:.3f} s")
    
    print("  2. IMU + ZUPT (windowed, Eq. 6.44)...")
    start = time.time()
    methods['IMU + ZUPT'] = run_imu_zupt(t, accel_meas, gyro_meas, initial, frame, imu_params)
    print(f"     Time: {time.time()-start:.3f} s")
    
    print("  3. Wheel Odometry...")
    start = time.time()
    methods['Wheel Odom'] = run_wheel_odom(t, wheel_meas, gyro_meas, initial, lever_arm)
    print(f"     Time: {time.time()-start:.3f} s")
    
    print("  4. PDR (step-and-heading)...")
    start = time.time()
    methods['PDR (Mag)'] = run_pdr(t, accel_meas, gyro_meas, mag_meas, height)
    print(f"     Time: {time.time()-start:.3f} s")
    
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    print("\nGenerating comparison plots...")
    metrics = plot_comparison(t, pos_true, methods, figs_dir)
    
    # Print comparison table
    print("\n" + "="*75)
    print("RESULTS - Performance Comparison")
    print("="*75)
    print(f"\n{'Method':<20} {'RMSE [m]':>10} {'Final [m]':>10} {'Median [m]':>10} {'90% [m]':>10} {'% Dist':>8}")
    print("-"*75)
    
    for name in ['IMU Only', 'IMU + ZUPT', 'Wheel Odom', 'PDR (Mag)']:
        m = metrics[name]
        pct = (m['rmse'] / total_dist) * 100
        print(f"{name:<20} {m['rmse']:>10.2f} {m['final']:>10.2f} {m['median']:>10.2f} {m['p90']:>10.2f} {pct:>7.1f}%")
    
    print(f"\nFigures saved to: {figs_dir}/")
    print()
    print("="*75)
    print("KEY INSIGHTS:")
    print("  1. IMU-only: UNBOUNDED drift (unusable without corrections)")
    print("  2. IMU+ZUPT: Dramatic improvement (~90-95% error reduction)")
    print("  3. Wheel Odom: BOUNDED drift (~1-5% of distance)")
    print("  4. PDR: BOUNDED, heading-limited (~2-5% of distance)")
    print()
    print("  Conclusion: Dead reckoning REQUIRES corrections or fusion!")
    print("             - Use ZUPT for foot-mounted IMU")
    print("             - Use wheel encoders for vehicles")
    print("             - Use magnetometer for heading reference")
    print("             - Best: Multi-sensor fusion (Chapter 8)")
    print("="*75)
    print()


if __name__ == "__main__":
    main()

