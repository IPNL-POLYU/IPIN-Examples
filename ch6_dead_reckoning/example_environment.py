"""
Example: Environmental Sensors (Magnetometer + Barometer)

Demonstrates magnetometer heading and barometric altitude for indoor
navigation. Shows absolute reference capability and failure modes.

Implements:
    - Magnetometer tilt compensation (Eq. 6.52)
    - Magnetometer heading (Eqs. 6.51-6.53)
    - Barometric altitude (Eq. 6.54)
    - Floor change detection

Key Insight: Environmental sensors provide absolute references to bound
            drift, but suffer from indoor disturbances (magnetic, weather).

Author: Li-Ta Hsu
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    mag_heading,
    mag_tilt_compensate,
    wrap_angle_diff,
    pressure_to_altitude,
    detect_floor_change,
    smooth_measurement_simple,
)


def generate_building_walk(duration=180.0, dt=0.1):
    """
    Generate multi-floor building walk with floor changes.
    
    Returns: t, pos_true, att_true, mag_true, pressure_true, floor_true
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    pos_true = np.zeros((N, 3))
    att_true = np.zeros((N, 3))  # roll, pitch, yaw
    mag_true = np.zeros((N, 3))
    pressure_true = np.zeros(N)
    floor_true = np.zeros(N, dtype=int)
    
    # Building: 3 floors, 3.5m per floor
    floor_height = 3.5  # meters
    p0 = 101325.0  # Sea level pressure [Pa]
    T = 288.15  # Temperature [K]
    
    # Walk pattern: ground floor → 2nd floor → 3rd floor → ground
    floor_schedule = [
        (0, 60, 0),    # Ground floor for 60s
        (60, 70, 1),   # Climb to floor 1
        (70, 110, 1),  # Floor 1 for 40s
        (110, 120, 2), # Climb to floor 2
        (120, 150, 2), # Floor 2 for 30s
        (150, 160, 1), # Descend to floor 1
        (160, 170, 0), # Descend to ground
        (170, 180, 0), # Ground floor
    ]
    
    # Heading: continuously rotating while walking
    # NOTE FOR STUDENTS: This trajectory has CONTINUOUSLY CHANGING heading
    # (20°/s rotation) to test magnetometer heading estimation under dynamic
    # conditions. This is different from fixed-heading examples because we're
    # specifically demonstrating magnetometer performance with magnetic disturbances.
    heading_rate = np.deg2rad(20.0)  # 20 deg/s continuous rotation
    
    for k in range(N):
        # Determine current floor
        current_floor = 0
        for start, end, floor in floor_schedule:
            if start <= t[k] < end:
                current_floor = floor
                break
        floor_true[k] = current_floor
        
        # Position (x,y random walk, z from floor)
        if k > 0:
            pos_true[k, 0] = pos_true[k-1, 0] + 0.1*np.random.randn()
            pos_true[k, 1] = pos_true[k-1, 1] + 0.1*np.random.randn()
        pos_true[k, 2] = current_floor * floor_height
        
        # Attitude (device orientation changes)
        att_true[k, 0] = 0.1 * np.sin(2*np.pi*t[k]/10)  # Roll oscillation
        att_true[k, 1] = 0.05 * np.sin(2*np.pi*t[k]/15)  # Pitch oscillation
        att_true[k, 2] = heading_rate * t[k]  # Continuous rotation
        
        # Magnetometer (north vector in body frame)
        roll, pitch, yaw = att_true[k]
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R_body_to_map = R_yaw @ R_pitch @ R_roll
        mag_north_map = np.array([1.0, 0.0, 0.0])  # North in map frame
        mag_true[k] = R_body_to_map.T @ mag_north_map
        
        # Barometric pressure from altitude
        h = pos_true[k, 2]
        # Standard atmosphere model (Eq. 6.54 inverted)
        L = 0.0065  # Temperature lapse rate [K/m]
        g = 9.81
        M = 0.029  # Molar mass of air [kg/mol]
        R_gas = 8.314  # Gas constant [J/(mol·K)]
        pressure_true[k] = p0 * (1 - L*h/T)**(g*M/(R_gas*L))
    
    return t, pos_true, att_true, mag_true, pressure_true, floor_true


def add_env_sensor_noise(mag_true, pressure_true, t, dt):
    """Add realistic environmental sensor noise + disturbances."""
    N = len(mag_true)
    duration = t[-1]
    
    # Magnetometer noise
    mag_noise = np.random.randn(N, 3) * 0.05  # Gaussian noise
    
    # Magnetic disturbances (steel structures)
    mag_disturbance = np.zeros((N, 3))
    disturbance_times = [(30, 50), (100, 120)]  # Near steel at these times
    for start, end in disturbance_times:
        mask = (t >= start) & (t < end)
        # Strong disturbance: shifts mag field direction
        mag_disturbance[mask] = np.random.randn(np.sum(mask), 3) * 0.5
    
    # Hard-iron offset (constant bias from device)
    hard_iron = np.array([0.1, -0.15, 0.05])
    
    mag_meas = mag_true + mag_noise + mag_disturbance + hard_iron
    
    # Barometer noise
    pressure_noise = np.random.randn(N) * 10.0  # Pa (typical noise)
    
    # Slow weather drift (pressure changes over time)
    weather_drift = 50.0 * np.sin(2*np.pi*t/(duration))  # ±50 Pa drift
    
    pressure_meas = pressure_true + pressure_noise + weather_drift
    
    return mag_meas, pressure_meas


def run_mag_heading(t, mag_meas, att_true):
    """Compute heading from magnetometer."""
    N = len(t)
    heading_est = np.zeros(N)
    
    for k in range(N):
        roll, pitch, _ = att_true[k]
        heading_est[k] = mag_heading(mag_meas[k], roll, pitch, declination=0.0)
    
    return heading_est


def run_baro_altitude(pressure_meas, p0=101325.0, T=288.15):
    """Compute altitude from barometer."""
    alt_est = np.array([pressure_to_altitude(p, p0, T) for p in pressure_meas])
    return alt_est


def plot_results(t, att_true, mag_meas, heading_est, pressure_meas, alt_est, 
                 floor_true, floor_detected, figs_dir):
    """Generate plots."""
    
    heading_true = att_true[:, 2]  # yaw
    
    # Figure 1: Magnetometer heading
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(t, np.rad2deg(heading_true), 'k-', linewidth=2, label='True Heading')
    ax1.plot(t, np.rad2deg(heading_est), 'b-', linewidth=2, alpha=0.7, label='Mag Heading')
    ax1.set_ylabel('Heading [deg]', fontsize=12)
    ax1.set_title('Magnetometer Example: Heading with Disturbances', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Compute heading error with proper angle wrapping
    heading_error_rad = np.array([wrap_angle_diff(heading_est[i], heading_true[i]) 
                                   for i in range(len(heading_est))])
    heading_error = np.abs(np.rad2deg(heading_error_rad))  # Absolute error in degrees
    # Note: By using wrap_angle_diff, heading_error is guaranteed to be <= 180°
    ax2.plot(t, heading_error, 'r-', linewidth=2)
    ax2.axhline(10, color='orange', linestyle='--', label='10° threshold')
    ax2.fill_between(t, 0, 100, where=(t>=30) & (t<50), alpha=0.2, color='red', label='Disturbance zone')
    ax2.fill_between(t, 0, 100, where=(t>=100) & (t<120), alpha=0.2, color='red')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Heading Error [deg]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])
    
    plt.tight_layout()
    fig1.savefig(figs_dir / 'environment_mag_heading.svg', dpi=300, bbox_inches='tight')
    fig1.savefig(figs_dir / 'environment_mag_heading.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'environment_mag_heading.svg'}")
    
    # Figure 2: Barometric altitude and floor detection
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    floor_heights = floor_true * 3.5
    ax1.plot(t, floor_heights, 'k-', linewidth=3, label='True Altitude', drawstyle='steps-post')
    ax1.plot(t, alt_est, 'b-', linewidth=1.5, alpha=0.7, label='Baro Altitude')
    ax1.set_ylabel('Altitude [m]', fontsize=12)
    ax1.set_title('Barometer Example: Altitude and Floor Detection', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, floor_true, 'k-', linewidth=3, label='True Floor', drawstyle='steps-post')
    ax2.plot(t, floor_detected, 'b--', linewidth=2, marker='o', markersize=3, label='Detected Floor')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Floor Number', fontsize=12)
    ax2.set_yticks([0, 1, 2])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])
    
    plt.tight_layout()
    fig2.savefig(figs_dir / 'environment_baro_altitude.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(figs_dir / 'environment_baro_altitude.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / 'environment_baro_altitude.svg'}")
    
    plt.close('all')
    
    return heading_error


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("Chapter 6: Environmental Sensors (Magnetometer + Barometer)")
    print("="*70)
    print("\nDemonstrates absolute references with indoor challenges.")
    print("Key equations: 6.51-6.54 (mag heading, tilt comp, baro altitude)\n")
    
    duration = 180.0
    dt = 0.1
    
    print(f"Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  Scenario:        Multi-floor building walk")
    print(f"  Floor Height:    3.5 m\n")
    
    print("Generating trajectory...")
    t, pos_true, att_true, mag_true, pressure_true, floor_true = generate_building_walk(duration, dt)
    print(f"  Floors visited:  {np.unique(floor_true)}")
    
    print("\nAdding sensor noise + disturbances...")
    mag_meas, pressure_meas = add_env_sensor_noise(mag_true, pressure_true, t, dt)
    
    print("\nComputing magnetometer heading...")
    start = time.time()
    heading_est = run_mag_heading(t, mag_meas, att_true)
    print(f"  Time: {time.time()-start:.3f} s")
    
    print("\nComputing barometric altitude...")
    start = time.time()
    alt_est = run_baro_altitude(pressure_meas)
    
    # Smooth altitude
    alt_smooth = np.zeros_like(alt_est)
    alt_smooth[0] = alt_est[0]
    for k in range(1, len(alt_est)):
        alt_smooth[k] = smooth_measurement_simple(alt_smooth[k-1], alt_est[k], alpha=0.1)
    
    # Detect floor changes
    floor_detected = np.zeros_like(floor_true)
    current_floor = 0
    for k in range(1, len(t)):
        delta_floor = detect_floor_change(alt_smooth[k-1], alt_smooth[k], floor_height=3.5, threshold=1.5)
        current_floor += delta_floor
        floor_detected[k] = max(0, min(2, current_floor))  # Clamp to [0, 2]
    
    print(f"  Time: {time.time()-start:.3f} s")
    
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    heading_error = plot_results(
        t, att_true, mag_meas, heading_est, pressure_meas, alt_smooth,
        floor_true, floor_detected, figs_dir
    )
    
    # Metrics
    heading_rmse = np.sqrt(np.mean(heading_error**2))
    alt_error = np.abs(alt_smooth - floor_true*3.5)
    alt_rmse = np.sqrt(np.mean(alt_error**2))
    floor_accuracy = np.sum(floor_detected == floor_true) / len(floor_true) * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Magnetometer Heading:")
    print(f"  RMSE:             {heading_rmse:.1f}°")
    print(f"  Max error:        {np.max(heading_error):.1f}°")
    print(f"  (Note: Large errors during disturbances at 30-50s, 100-120s)")
    print()
    print(f"Barometric Altitude:")
    print(f"  RMSE:             {alt_rmse:.2f} m")
    print(f"  Floor Accuracy:   {floor_accuracy:.1f}%")
    print()
    print(f"Figures saved to: {figs_dir}/")
    print()
    print("="*70)
    print("KEY INSIGHT: Environmental sensors provide absolute references!")
    print("             Magnetometer: bounds heading drift (when clean).")
    print("             Barometer: provides floor-level positioning.")
    print("             BUT sensitive to indoor disturbances (steel, weather).")
    print("="*70)
    print()


if __name__ == "__main__":
    main()

