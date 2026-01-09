"""
Example: Wheel Odometry Dead Reckoning for Vehicles

Demonstrates vehicle dead reckoning using wheel encoders with lever arm
compensation. Shows bounded but drift-prone behavior, especially during
wheel slip.

Implements:
    - Lever arm compensation (Eq. 6.11)
    - Skew-symmetric matrix (Eq. 6.12)
    - Frame transformations (Eq. 6.14)
    - Position update (Eq. 6.15)

Key Insight: Wheel odometry drift is BOUNDED (proportional to distance,
            not time) but very sensitive to wheel slip!

Author: Li-Ta Hsu
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import wheel_odom_update, NavStateQPVP


def generate_vehicle_trajectory(shape='square', duration=80.0, dt=0.01):
    """
    Generate vehicle trajectory (square or circle).
    
    Returns: t, pos_true, vel_true, quat_true, wheel_speed_true, gyro_true
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    pos_true = np.zeros((N, 3))
    vel_true = np.zeros((N, 3))
    quat_true = np.zeros((N, 4))
    wheel_speed_true = np.zeros((N, 3))  # Speed frame: [0, v_forward, 0] (book convention)
    gyro_true = np.zeros((N, 3))
    
    if shape == 'square':
        # Square: 20m sides, 5 m/s speed, 90° turns
        side_length = 20.0
        v_drive = 5.0  # m/s
        
        side_time = side_length / v_drive  # 4 seconds per side
        turn_time = 2.0  # 2 seconds for 90° turn
        segment_time = side_time + turn_time  # 6 seconds per segment
        
        current_pos = np.array([0.0, 0.0, 0.0])
        # Initial heading: 0° = East in ENU frame (see docs/ch6_frame_conventions.md)
        # NOTE FOR STUDENTS: This choice is arbitrary for simulation purposes.
        # In real applications, you would estimate initial heading from sensors
        # (magnetometer, GPS velocity, manual calibration). The wheel odometry
        # algorithm works identically regardless of initial heading!
        current_heading = 0.0  # Start facing East (0° in ENU)
        
        for k in range(N):
            t_cycle = t[k] % (segment_time * 4)  # 4 sides
            segment = int(t_cycle / segment_time)
            t_seg = t_cycle - segment * segment_time
            
            if t_seg < side_time:  # Driving straight
                wheel_speed_true[k] = np.array([0, v_drive, 0])  # Book: y=forward
                gyro_true[k, 2] = 0
            else:  # Turning
                wheel_speed_true[k] = np.array([0, 0, 0])  # Stop to turn
                gyro_true[k, 2] = np.pi/2 / turn_time  # 90°/2s
            
            # Update state
            if k > 0:
                current_heading += gyro_true[k, 2] * dt
                v_map = wheel_speed_true[k, 1] * np.array([np.cos(current_heading), np.sin(current_heading), 0])
                current_pos += v_map * dt
            
            pos_true[k] = current_pos
            vel_true[k, :2] = wheel_speed_true[k, 1] * np.array([np.cos(current_heading), np.sin(current_heading)])
            
            # Quaternion (yaw only)
            quat_true[k] = np.array([np.cos(current_heading/2), 0, 0, np.sin(current_heading/2)])
    
    else:  # circle
        omega = 2*np.pi / duration  # One full circle
        radius = 15.0
        v_drive = radius * omega
        
        for k in range(N):
            angle = omega * t[k]
            pos_true[k] = np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
            vel_true[k] = v_drive * np.array([-np.sin(angle), np.cos(angle), 0])
            wheel_speed_true[k] = np.array([0, v_drive, 0])  # Book: y=forward
            gyro_true[k, 2] = omega
            quat_true[k] = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    
    return t, pos_true, vel_true, quat_true, wheel_speed_true, gyro_true


def add_wheel_noise(wheel_speed_true, gyro_true, add_slip=False, slip_intervals=None):
    """Add wheel encoder noise and optional slip."""
    N = len(wheel_speed_true)
    
    # Scale error (wrong wheel radius) - systematic bias
    scale_error = 0.02  # 2% scale error (typical)
    
    # Random noise
    noise_std = 0.05  # m/s
    wheel_noise = np.random.randn(N, 3) * noise_std
    gyro_noise = np.random.randn(N, 3) * np.deg2rad(0.5)
    
    wheel_meas = wheel_speed_true * (1 + scale_error) + wheel_noise
    gyro_meas = gyro_true + gyro_noise
    
    # Add wheel slip during turns
    if add_slip and slip_intervals:
        for start, end in slip_intervals:
            mask = (np.arange(N)*0.01 >= start) & (np.arange(N)*0.01 < end)
            # During slip, wheel speed overestimates actual motion
            wheel_meas[mask, 1] *= 1.3  # 30% overestimate (y-component)
    
    return wheel_meas, gyro_meas


def run_wheel_odometry(t, wheel_speed, gyro, initial_state, lever_arm):
    """Run wheel dead reckoning with lever arm."""
    N = len(t)
    dt = t[1] - t[0]
    
    p = initial_state.p.copy()
    q = initial_state.q.copy()
    
    pos_est = np.zeros((N, 3))
    pos_est[0] = p
    
    for k in range(1, N):
        # Wheel odometry update (Eqs. 6.11-6.15)
        p = wheel_odom_update(
            p=p,
            q=q,
            v_s=wheel_speed[k-1],
            omega_a=gyro[k-1],
            lever_arm_a=lever_arm,
            dt=dt
        )
        
        # Update quaternion
        q_new = q.copy()
        dq = 0.5 * dt * np.array([
            -q[1]*gyro[k-1,0] - q[2]*gyro[k-1,1] - q[3]*gyro[k-1,2],
            q[0]*gyro[k-1,0] + q[2]*gyro[k-1,2] - q[3]*gyro[k-1,1],
            q[0]*gyro[k-1,1] - q[1]*gyro[k-1,2] + q[3]*gyro[k-1,0],
            q[0]*gyro[k-1,2] + q[1]*gyro[k-1,1] - q[2]*gyro[k-1,0]
        ])
        q = q_new + dq
        q = q / np.linalg.norm(q)
        
        pos_est[k] = p
    
    return pos_est


def plot_results(t, pos_true, pos_odom, pos_odom_slip, figs_dir):
    """Generate plots."""
    
    error_odom = np.linalg.norm(pos_odom - pos_true, axis=1)
    error_slip = np.linalg.norm(pos_odom_slip - pos_true, axis=1)
    
    # Figure 1: Trajectory
    fig1, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True Trajectory')
    ax.plot(pos_odom[:, 0], pos_odom[:, 1], 'b--', linewidth=2, label='Wheel Odom (no slip)')
    ax.plot(pos_odom_slip[:, 0], pos_odom_slip[:, 1], 'r--', linewidth=2, alpha=0.7, label='Wheel Odom (with slip)')
    ax.scatter(0, 0, c='g', s=150, marker='o', label='Start', zorder=5)
    ax.set_xlabel('East [m]', fontsize=12)
    ax.set_ylabel('North [m]', fontsize=12)
    ax.set_title('Wheel Odometry Example: Square Path', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    fig1.savefig(figs_dir / 'wheel_odom_trajectory.svg', dpi=300, bbox_inches='tight')
    fig1.savefig(figs_dir / 'wheel_odom_trajectory.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'wheel_odom_trajectory.svg'}")
    
    # Figure 2: Error
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, error_odom, 'b-', linewidth=2, label='No Slip')
    ax.plot(t, error_slip, 'r-', linewidth=2, label='With Slip')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Position Error [m]', fontsize=12)
    ax.set_title('Wheel Odometry Example: Position Error (Bounded Drift)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])
    plt.tight_layout()
    fig2.savefig(figs_dir / 'wheel_odom_error.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(figs_dir / 'wheel_odom_error.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'wheel_odom_error.svg'}")
    
    plt.close('all')
    return error_odom, error_slip


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("Chapter 6: Wheel Odometry Dead Reckoning for Vehicles")
    print("="*70)
    print("\nDemonstrates bounded drift and sensitivity to wheel slip.")
    print("Key equations: 6.11-6.15 (lever arm, frame transform, position)\n")
    
    duration = 80.0
    dt = 0.01
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  Trajectory:      20m square with turns")
    print(f"  Lever Arm:       [1.5, 0, -0.3] m\n")
    
    print("Generating trajectory...")
    t, pos_true, vel_true, quat_true, wheel_true, gyro_true = generate_vehicle_trajectory('square', duration, dt)
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    print(f"  Total distance:  {total_dist:.1f} m")
    
    # Add noise (no slip)
    print("\nAdding wheel encoder noise...")
    wheel_meas, gyro_meas = add_wheel_noise(wheel_true, gyro_true, add_slip=False)
    
    # Add noise WITH slip during turns
    print("Adding wheel encoder noise + slip...")
    slip_intervals = [(4, 6), (10, 12), (16, 18), (22, 24)]  # During turns
    wheel_slip, gyro_slip = add_wheel_noise(wheel_true, gyro_true, add_slip=True, slip_intervals=slip_intervals)
    
    # Initial state
    initial = NavStateQPVP(q=quat_true[0], v=vel_true[0], p=pos_true[0])
    lever_arm = np.array([1.5, 0, -0.3])  # Sensor offset from vehicle center
    
    print("\nRunning wheel odometry (no slip)...")
    start = time.time()
    pos_odom = run_wheel_odometry(t, wheel_meas, gyro_meas, initial, lever_arm)
    print(f"  Time: {time.time()-start:.3f} s")
    
    print("\nRunning wheel odometry (with slip)...")
    start = time.time()
    pos_odom_slip = run_wheel_odometry(t, wheel_slip, gyro_slip, initial, lever_arm)
    print(f"  Time: {time.time()-start:.3f} s")
    
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    error_odom, error_slip = plot_results(t, pos_true, pos_odom, pos_odom_slip, figs_dir)
    
    rmse_odom = np.sqrt(np.mean(error_odom**2))
    rmse_slip = np.sqrt(np.mean(error_slip**2))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Wheel Odometry (no slip):")
    print(f"  Final error:  {error_odom[-1]:.2f} m ({error_odom[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_odom:.2f} m")
    print()
    print(f"Wheel Odometry (with slip during turns):")
    print(f"  Final error:  {error_slip[-1]:.2f} m ({error_slip[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_slip:.2f} m")
    print()
    print(f"Figures saved to: {figs_dir}/")
    print()
    print("="*70)
    print("KEY INSIGHT: Wheel odometry drift is BOUNDED!")
    print("             Errors ~1-5% of distance (vs unbounded for IMU).")
    print("             BUT very sensitive to wheel slip (turns, ice, etc).")
    print("="*70)
    print()


if __name__ == "__main__":
    main()

