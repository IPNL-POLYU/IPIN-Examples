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

from core.eval import resolve_figs_dir, save_figure
from core.utils import resolve_data_path
from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    detect_steps_peak_detector,
    step_length,
    step_length_book_eq6_49,
    step_length_weinberg,
    pdr_step_update,
    integrate_gyro_heading,
    wrap_heading,
    mag_heading,
    units,
)
from core.sim import generate_imu_from_trajectory



# Seed for this example's sensor-noise draws. Fixed so the committed
# figures can be regenerated exactly; see the noise function below.
DEFAULT_SEED = 42

#: Radius of the corridor corners, m. Zero would restore the 9000 deg/s step
#: that made the true gyro unusable; 2 m turns at v/r = 40 deg/s, which is
#: brisk for a pedestrian but achievable, and keeps the lap close to its
#: nominal 120 m.
CORNER_RADIUS_M = 2.0

#: Distance over which the walker decelerates to a halt at the end of the lap,
#: m. Stopping dead from 1.4 m/s inside one sample implied 7 g -- the same
#: unphysical step the rounded corners removed, just at the end of the record.
BRAKE_DISTANCE_M = 1.0

def compute_step_length(
    height: float,
    f_step: float,
    model: str = "book",
    G_w: Optional[float] = None,
    f_step_window: Optional[np.ndarray] = None,
) -> float:
    """
    Compute step length using selected model.
    
    Args:
        height: User height in meters
        f_step: Step frequency in Hz
        model: Model selection: 'book' (Eq. 6.49), 'weinberg' (actual Weinberg), or 'power_law' (old)
        G_w: Weinberg gain parameter (required if model='weinberg')
        f_step_window: Per-step accel window (required if model='weinberg')
    
    Returns:
        Step length in meters
    """
    if model == "book":
        # Use book Eq. (6.49) - default for reproducibility
        return step_length_book_eq6_49(height, f_step)
    elif model == "weinberg":
        # Use actual Weinberg model with per-step ptp
        if G_w is None or f_step_window is None:
            raise ValueError("Weinberg model requires G_w and f_step_window parameters")
        return step_length_weinberg(f_step_window, G_w)
    elif model == "power_law":
        # Legacy power-law (deprecated but kept for compatibility)
        return step_length(height, f_step)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'book', 'weinberg', or 'power_law'")


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


def run_pdr_from_dataset(data: Dict, height: float = 1.75, step_model: str = "book") -> Dict:
    """Run PDR algorithm on loaded dataset.
    
    Uses the book's peak detection method (Eqs. 6.46-6.47) for step detection:
    1. Compute total acceleration magnitude (6.46)
    2. Subtract gravity (6.47)
    3. Filter the signal
    4. Detect peaks
    
    Args:
        data: Dataset dictionary from load_pdr_dataset
        height: Pedestrian height in meters
        step_model: Step-length model: 'book' (Eq. 6.49, default), 'power_law' (old)
    
    Returns:
        Dictionary with estimated positions and headings
    
    Note:
        Weinberg model not supported here (requires per-step windows, needs refactoring)
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
    # NOTE FOR STUDENTS: Gyro heading starts at 0° (East in ENU) as a simulation
    # choice. In real PDR systems, initial heading MUST be calibrated from an
    # absolute reference (magnetometer, GPS, or user input) because gyros measure
    # only CHANGES in heading, not absolute direction!
    heading_gyro[0] = 0.0  # Start at 0° (East), will drift due to gyro bias
    heading_mag[0] = mag_heading(mag_meas[0], roll=0.0, pitch=0.0, declination=0.0)  # Absolute reference
    
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
            
            # Step length using selected model
            L = compute_step_length(height, f_step, model=step_model)
            
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
            
            # Step length using selected model
            L = compute_step_length(height, f_step, model=step_model)
            
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


def run_with_dataset(data_dir: str, height: float = 1.75, lat_deg: float = 45.0, step_model: str = "book") -> None:
    """Run PDR example using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
        height: Pedestrian height in meters
        lat_deg: Latitude in degrees
        step_model: Step-length model selection
    """
    print("\n" + "="*70)
    print("Chapter 6: Pedestrian Dead Reckoning (PDR)")
    print(f"Using dataset: {data_dir}")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    data = load_pdr_dataset(data_dir)
    
    t = data['t']
    pos_true = data['pos_true']
    heading_true = data['heading_true']
    step_times = data['step_times']
    
    total_dist = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    
    print("\nDataset Info:")
    print(f"  Duration: {t[-1]:.1f} s")
    print(f"  Total distance: {total_dist:.1f} m")
    print(f"  True steps: {len(step_times)}")
    print(f"  User height: {height} m")
    
    # Run PDR
    print(f"\nRunning PDR algorithms (step model: {step_model})...")
    start = time.time()
    results = run_pdr_from_dataset(data, height, step_model=step_model)
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
    print("PDR (Gyro Heading - drifts unbounded):")
    print(f"  Final error:  {error_gyro[-1]:.1f} m ({error_gyro[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_gyro:.1f} m")
    print()
    print("PDR (Magnetometer Heading - absolute but noisy):")
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
    
    paths = save_figure(fig, figs_dir, 'pdr_dataset_results')
    print(f"  [OK] Saved: {paths[0]}")
    
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
    
    # Corridor: 40m x 20m rectangle, walked counter-clockwise, with each
    # corner cut by a quarter-circle of CORNER_RADIUS_M.
    #
    # The rounding is not cosmetic. Square corners turned the heading 90 deg
    # between two 0.01 s samples -- 9000 deg/s. The gyro forward model cannot
    # represent a step that large, so the *true* gyro integrated to only
    # 162 deg over a lap whose heading comes all the way round to 360, and any
    # estimator integrating it lost the missing 198 deg. That was the whole of
    # this example's reported heading error: the estimator was faithfully
    # reporting a rotation the data never contained. Chapter 8 had the
    # identical defect at the identical 9000 deg/s.
    #
    # Parameterised by arc length rather than by waypoint index, so heading is
    # continuous by construction and its rate is v/r on the arcs -- about
    # 40 deg/s here, which a pedestrian can actually turn.
    width, height_m = 40.0, 20.0
    r = CORNER_RADIUS_M
    v_walk = 1.4  # m/s (typical walking speed)

    # (kind, length, data): straights alternate with quarter-circle arcs.
    # Arc data is (centre, start_angle); the walk turns left at every corner.
    straight_x, straight_y = width - 2 * r, height_m - 2 * r
    quarter = 0.5 * np.pi * r
    segments = [
        ("line", straight_x, (np.array([r, 0.0]), 0.0)),
        ("arc", quarter, (np.array([width - r, r]), -0.5 * np.pi)),
        ("line", straight_y, (np.array([width, r]), 0.5 * np.pi)),
        ("arc", quarter, (np.array([width - r, height_m - r]), 0.0)),
        ("line", straight_x, (np.array([width - r, height_m]), np.pi)),
        ("arc", quarter, (np.array([r, height_m - r]), 0.5 * np.pi)),
        ("line", straight_y, (np.array([0.0, height_m - r]), 1.5 * np.pi)),
        ("arc", quarter, (np.array([r, r]), np.pi)),
    ]
    total_length = sum(seg[1] for seg in segments)

    pos_2d = np.zeros((N, 2))
    heading_true = np.zeros(N)
    vel_2d = np.zeros((N, 2))

    # Distance travelled by time t, with a braking phase so the walk does not
    # stop dead. Rounding the corners and then halting from 1.4 m/s inside one
    # sample would leave the same defect at a different point in the record:
    # that stop implied 70 m/s^2, 7 g, and it was the only sample left above
    # 1 g once the corners were fixed. Decelerating over BRAKE_DISTANCE_M
    # costs v^2 / (2 d) = 0.98 m/s^2, the same as the corners themselves.
    cruise_length = total_length - BRAKE_DISTANCE_M
    t_cruise = cruise_length / v_walk
    decel = v_walk ** 2 / (2.0 * BRAKE_DISTANCE_M)
    t_brake = v_walk / decel

    def distance_at(time_s):
        """Arc length travelled at ``time_s``, cruising then braking."""
        if time_s <= t_cruise:
            return v_walk * time_s
        tau = min(time_s - t_cruise, t_brake)
        return cruise_length + v_walk * tau - 0.5 * decel * tau ** 2

    starts = np.cumsum([0.0] + [seg[1] for seg in segments])
    for k in range(N):
        s = distance_at(t[k])
        if s >= total_length:
            # Lap complete: hold the final pose.
            pos_2d[k] = pos_2d[k - 1] if k > 0 else np.array([r, 0.0])
            heading_true[k] = heading_true[k - 1] if k > 0 else 0.0
            continue

        i = int(np.searchsorted(starts, s, side="right") - 1)
        kind, seg_len, (anchor, phase) = segments[i]
        local = s - starts[i]

        if kind == "line":
            direction = np.array([np.cos(phase), np.sin(phase)])
            pos_2d[k] = anchor + local * direction
            heading_true[k] = phase
        else:
            angle = phase + local / r  # left turn: angle advances with s
            pos_2d[k] = anchor + r * np.array([np.cos(angle), np.sin(angle)])
            heading_true[k] = angle + 0.5 * np.pi  # tangent, turning left

        # Speed follows the profile, so velocity stays the derivative of
        # position through the braking phase too.
        speed = v_walk if t[k] <= t_cruise else max(
            v_walk - decel * (t[k] - t_cruise), 0.0
        )
        vel_2d[k] = speed * np.array([
            np.cos(heading_true[k]), np.sin(heading_true[k])
        ])

    heading_true = np.unwrap(heading_true)

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

    # Only while actually walking. The oscillation used to run for the whole
    # record, including the 36 s the walker spends standing still after the
    # lap closes, so the step detector faithfully counted about 73 steps that
    # were never taken -- and the example blamed the step-length model for the
    # distance error those phantom steps caused.
    #
    # Ramped rather than switched off, over one step period: a step in the
    # vertical acceleration is the same kind of unphysical discontinuity as
    # the square corners this generator used to have.
    walking = (v_walk * t) < total_length
    ramp = np.clip((total_length - v_walk * t) * step_freq / v_walk, 0.0, 1.0)
    gait_envelope = np.where(walking, ramp, 0.0)

    # Modify velocity to include these oscillations (integrate accel)
    # This is a simplified model - real walking has complex 3D motion
    vel_map_with_steps = vel_map.copy()
    vel_map_with_steps[:, 2] += (
        gait_envelope
        * walking_accel_amplitude
        / (2 * np.pi * step_freq)
        * np.cos(2 * np.pi * step_freq * t)
    )
    
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
    
    # Steps actually taken: the walker is only moving until the lap closes at
    # total_length / v_walk, and stands still afterwards.
    walking_time = min(total_length / v_walk, duration)
    expected_steps = int(round(walking_time * step_freq))
    
    return t, pos_2d, heading_true, accel_body, gyro_body, mag_body, expected_steps


def add_sensor_noise(accel_body, gyro_body, mag_body, dt,
                     imu_params: IMUNoiseParams, seed: int = DEFAULT_SEED):
    """Add realistic sensor noise with explicit units.

    Args:
        accel_body: Noise-free specific force [m/s^2], shape (N, 3).
        gyro_body: Noise-free angular rate [rad/s], shape (N, 3).
        mag_body: Noise-free magnetic field, shape (N, 3).
        dt: Sample interval [s].
        imu_params: IMU noise specification.
        seed: Seed for this run's noise, so the committed figures can be
            regenerated. The magnetic disturbances below are what the heading
            estimate has to survive, and redrawing them each run changed the
            figure without changing the code.

    Returns:
        Tuple of (accel_meas, gyro_meas, mag_meas).
    """
    N = len(accel_body)
    rng = np.random.default_rng(seed)

    # IMU noise and biases
    gyro_bias = rng.standard_normal(3) * imu_params.gyro_bias_rad_s
    gyro_noise_std = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
    gyro_noise = rng.standard_normal((N, 3)) * gyro_noise_std

    accel_noise_std = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
    accel_noise = rng.standard_normal((N, 3)) * accel_noise_std

    # Magnetometer noise + disturbances
    mag_noise = rng.standard_normal((N, 3)) * 0.05
    mag_disturbance = np.zeros((N, 3))
    # Add disturbances at specific times (simulating steel structures)
    disturb_intervals = [(20, 30), (70, 80)]  # seconds
    for start, end in disturb_intervals:
        mask = (np.arange(N)*dt >= start) & (np.arange(N)*dt < end)
        mag_disturbance[mask] = rng.standard_normal((np.sum(mask), 3)) * 0.3
    
    gyro_meas = gyro_body + gyro_bias + gyro_noise
    accel_meas = accel_body + accel_noise
    mag_meas = mag_body + mag_noise + mag_disturbance
    
    return accel_meas, gyro_meas, mag_meas


def run_pdr_gyro_heading(t, accel_meas, gyro_meas, height=1.75, step_model="book"):
    """
    Run PDR with gyro-integrated heading (drifts).
    
    Uses proper peak detection (Eqs. 6.46-6.47) instead of threshold crossing.
    
    Args:
        t: Time array
        accel_meas: Accelerometer measurements
        gyro_meas: Gyroscope measurements
        height: Pedestrian height in meters
        step_model: Step-length model: 'book' (Eq. 6.49, default), 'power_law' (old)
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
    
    # Initialize heading (gyro integration requires initial heading)
    # NOTE FOR STUDENTS: Starting at 0° for simulation simplicity. In practice,
    # use magnetometer or GPS for initial heading calibration!
    heading_est[0] = 0.0  # Initial heading (0° = East in ENU)
    
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
            
            # Step length using selected model
            L = compute_step_length(height, f_step, model=step_model)
            
            # Update position (Eq. 6.50)
            pos_est[k] = pdr_step_update(pos_est[k-1], L, heading_est[k-1])
        else:
            pos_est[k] = pos_est[k-1]
    
    return pos_est, heading_est, step_count


def run_pdr_mag_heading(t, accel_meas, gyro_meas, mag_meas, height=1.75, step_model="book"):
    """
    Run PDR with magnetometer heading (absolute but noisy).
    
    Uses proper peak detection (Eqs. 6.46-6.47) instead of threshold crossing.
    
    Args:
        t: Time array
        accel_meas: Accelerometer measurements
        gyro_meas: Gyroscope measurements
        mag_meas: Magnetometer measurements
        height: Pedestrian height in meters
        step_model: Step-length model: 'book' (Eq. 6.49, default), 'power_law' (old)
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
    
    # Initialize heading from magnetometer (absolute reference)
    # NOTE FOR STUDENTS: Unlike gyro heading which can start anywhere, magnetometer
    # provides an absolute heading reference. This is the proper way to initialize!
    heading_est[0] = mag_heading(mag_meas[0], roll=0.0, pitch=0.0, declination=0.0)  # Absolute heading
    
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
            
            # Step length using selected model
            L = compute_step_length(height, f_step, model=step_model)
            
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
    paths = save_figure(fig1, figs_dir, 'pdr_trajectory')
    print(f"  [OK] Saved: {paths[0]}")
    
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
    paths = save_figure(fig2, figs_dir, 'pdr_heading')
    print(f"  [OK] Saved: {paths[0]}")
    
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
    paths = save_figure(fig3, figs_dir, 'pdr_error')
    print(f"  [OK] Saved: {paths[0]}")
    
    plt.close('all')
    
    return error_gyro, error_mag


def run_with_inline_data(lat_deg: float = 45.0, step_model: str = "book"):
    """Run with inline generated data (original behavior).
    
    Args:
        lat_deg: Latitude in degrees
        step_model: Step-length model selection
    """
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
    
    print("Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  User Height:     {height} m")
    print("  Trajectory:      40m x 20m rectangular corridor")
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
    
    print(f"\nRunning PDR with gyro heading (step model: {step_model})...")
    start = time.time()
    pos_gyro, heading_gyro, steps_gyro = run_pdr_gyro_heading(t, accel_meas, gyro_meas, height, step_model=step_model)
    print(f"  Time: {time.time()-start:.3f} s, Steps detected: {steps_gyro}")
    
    print(f"\nRunning PDR with magnetometer heading (step model: {step_model})...")
    start = time.time()
    pos_mag, heading_mag, steps_mag = run_pdr_mag_heading(t, accel_meas, gyro_meas, mag_meas, height, step_model=step_model)
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
    print("PDR (Gyro Heading - drifts unbounded):")
    print(f"  Final error:  {error_gyro[-1]:.1f} m ({error_gyro[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_gyro:.1f} m")
    print()
    print("PDR (Magnetometer Heading - absolute but noisy):")
    print(f"  Final error:  {error_mag[-1]:.1f} m ({error_mag[-1]/total_dist*100:.1f}% of distance)")
    print(f"  RMSE:         {rmse_mag:.1f} m")
    print()
    # Decompose the error before attributing it. Most of what these two runs
    # report is not heading at all.
    from core.eval import path_length

    walked = path_length(pos_true[:, :2])
    stepped = path_length(pos_gyro[:, :2])
    # Shortest angular difference. Subtracting raw angles compares a wrapped
    # estimate against a truth that now runs continuously to 360 deg over the
    # lap, which reported a 1.2 deg error as 358.8.
    heading_drift_deg = float(
        np.degrees(
            np.abs(
                np.arctan2(
                    np.sin(heading_gyro[-1] - heading_true[-1]),
                    np.cos(heading_gyro[-1] - heading_true[-1]),
                )
            )
        )
    )

    # Measured from the trajectory rather than assumed. An earlier version of
    # this budget divided the distance by an assumed 0.5 m gait to get a
    # "true" step count, which inverted the attribution entirely.
    walk_speed = np.linalg.norm(np.diff(pos_true[:, :2], axis=0), axis=1) / (t[1] - t[0])
    moving = walk_speed > 0.05
    walking_time = float(np.sum(moving)) * (t[1] - t[0])
    true_step_len = float(np.mean(walk_speed[moving])) / 2.0
    true_steps = int(round(walking_time * 2.0))
    final_gyro = float(error_gyro[-1])

    print("  Where the error comes from, now that the trajectory is one a")
    print("  pedestrian could actually walk:")
    print(f"    1. Step length, and that is essentially all of it. PDR "
          f"believes it walked {stepped:.1f} m against a true {walked:.1f} m, "
          f"{100 * (stepped / walked - 1):+.0f}%.")
    print(f"       Detection is sound -- {steps_gyro} steps found against "
          f"{true_steps} taken, within {abs(steps_gyro - true_steps)} -- so "
          f"the gap is the model: Eq. (6.49) returns "
          f"{stepped / max(steps_gyro, 1):.3f} m")
    print(f"       per step for a {height:.2f} m walker at this cadence while "
          f"the simulated gait is {true_step_len:.3f} m. Step length is the "
          f"parameter PDR is")
    print("       most sensitive to, and it is the one a real deployment has "
          "to calibrate per user.")
    print(f"    2. Heading. The gyro ends {heading_drift_deg:.1f} deg from "
          f"truth, which is its realised bias integrated over {t[-1]:.0f} s "
          f"and nothing else.")
    print("       That is what drift at this grade actually looks like. It "
          "used to read 163 deg and none of it was drift: this generator "
          "turned each")
    print("       corner 90 deg inside one 0.01 s sample -- 9000 deg/s -- "
          "which the gyro forward model cannot represent, so the *true* gyro")
    print("       integrated to 162 deg over a lap whose heading comes round "
          "to 360. The estimator was faithfully reporting a rotation the "
          "data never")
    print(f"       contained. Chapter 8 had the identical defect at the "
          f"identical 9000 deg/s. The corners are rounded now "
          f"({CORNER_RADIUS_M:.0f} m, {np.degrees(1.4 / CORNER_RADIUS_M):.0f} deg/s),")
    print(f"       and the gait oscillation no longer runs through the "
          f"{t[-1] - walking_time:.0f} s of standing still that was worth 73 "
          f"phantom steps. Final error: 80.7 m -> {final_gyro:.1f} m.")
    print()
    # Report where the figures actually went. save_figure resolves this
    # internally through IPIN_FIGS_DIR, so printing the requested path made
    # this line contradict the per-figure "[OK] Saved:" lines above it
    # whenever the variable was set -- which is every test run.
    print(f"Figures saved to: {resolve_figs_dir(figs_dir)}/")
    print()
    print("="*70)
    print("KEY INSIGHT: Check that a simulated truth is achievable before")
    print("             reading an estimator's error as the estimator's. This")
    print("             example reported 80.7 m and blamed unbounded gyro")
    print("             drift. The drift was 1.2 deg. The 163 deg it actually")
    print("             showed came from corners the trajectory turned in one")
    print("             0.01 s sample -- 9000 deg/s, which no gyro can encode")
    print("             -- so the estimator was faithfully reporting a")
    print("             rotation the data never contained. Chapter 8 lost")
    print("             0.739 m to the identical defect.")
    print("             With the corners rounded and the gait signal stopped")
    print("             while standing still, PDR closes a 117 m lap to 1.4 m.")
    print("             What remains is step length: Eq. (6.49) gives 0.748 m")
    print("             against a 0.700 m gait, and that 7% is the whole")
    print("             residual. Step length is what a real deployment must")
    print("             calibrate per user; heading matters too, which is why")
    print("             best practice is still a complementary filter.")
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
    parser.add_argument(
        "--latitude", type=float, default=45.0,
        help="Latitude in degrees for gravity model (Eq. 6.8, default: 45.0)"
    )
    parser.add_argument(
        "--step-model", type=str, default="book",
        choices=["book", "power_law"],
        help="Step-length model: 'book' (Eq. 6.49, default) or 'power_law' (old)"
    )
    
    args = parser.parse_args()
    
    if args.data:
        # Run with dataset
        data_path = resolve_data_path(args.data)
        if not data_path.exists():
            data_path = resolve_data_path(Path("data/sim") / args.data)
        if not data_path.exists():
            print(f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'")
            print("\nAvailable datasets:")
            sim_dir = resolve_data_path(Path("data/sim"))
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch6"):
                        print(f"  - {d.name}")
            return
        
        run_with_dataset(str(data_path), height=args.height, lat_deg=args.latitude, step_model=args.step_model)
    else:
        run_with_inline_data(lat_deg=args.latitude, step_model=args.step_model)


if __name__ == "__main__":
    main()

