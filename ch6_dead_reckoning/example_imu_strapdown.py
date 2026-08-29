"""
Example: IMU Strapdown Integration (Pure, No Corrections)

Demonstrates pure IMU strapdown integration showing unbounded drift.
This example illustrates WHY dead reckoning needs corrections.

Implements:
    - Quaternion attitude integration (Eqs. 6.2-6.4)
    - Velocity integration with gravity (Eq. 6.7)
    - Position integration (Eq. 6.10)
    - IMU error models (Eqs. 6.6, 6.9)

Key Insight: IMU drift is UNBOUNDED without external corrections!

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# Import Chapter 6 sensor algorithms
# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import resolve_figs_dir, save_figure, show_figures_if_requested
from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    NavStateQPVP,
    random_walk_to_rate_sample_std,
    strapdown_update,
    units,
)

# Import IMU forward model
from core.sim import generate_imu_from_trajectory

# Seed for this example's sensor-noise draws. Fixed so the committed
# figures can be regenerated exactly; see the noise function below.
DEFAULT_SEED = 42


class Figure8Trajectory(NamedTuple):
    """Figure-8 ground truth and ideal IMU signals.

    NamedTuple keeps legacy tuple unpacking while making semantic fields
    discoverable. Positions/velocities are in the selected local map frame.
    """

    timestamps_s: np.ndarray
    true_positions_map_m: np.ndarray
    true_velocities_map_mps: np.ndarray
    true_attitudes_quat_wxyz: np.ndarray
    specific_force_body_mps2: np.ndarray
    angular_rates_body_rad_s: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return self.timestamps_s

    @property
    def pos_true(self) -> np.ndarray:
        return self.true_positions_map_m

    @property
    def vel_true(self) -> np.ndarray:
        return self.true_velocities_map_mps

    @property
    def quat_true(self) -> np.ndarray:
        return self.true_attitudes_quat_wxyz

    @property
    def accel_body(self) -> np.ndarray:
        return self.specific_force_body_mps2

    @property
    def gyro_body(self) -> np.ndarray:
        return self.angular_rates_body_rad_s


def generate_figure8_trajectory(duration=100.0, dt=0.01, frame=None, lat_deg=45.0):
    """
    Generate a figure-8 trajectory with correct IMU forward model.

    Args:
        duration: Total duration [s].
        dt: Time step [s].
        frame: Frame convention (default: ENU).
        lat_deg: Latitude in degrees for gravity model (default: 45.0°).

    Returns:
        Figure8Trajectory with semantic fields and tuple-compatible order:
        timestamps_s, true_positions_map_m, true_velocities_map_mps,
        true_attitudes_quat_wxyz, specific_force_body_mps2,
        angular_rates_body_rad_s.
    """
    if frame is None:
        frame = FrameConvention.create_enu()

    # Convert latitude to radians for Eq. (6.8)
    lat_rad = np.deg2rad(lat_deg)

    t = np.arange(0, duration, dt)

    # Figure-8 parametric curve
    # x(t) = A*sin(ωt), y(t) = B*sin(2ωt)
    omega = 2 * np.pi / 40.0  # 40-second period
    A, B = 15.0, 10.0  # meters

    # Position (2D, z=0)
    x = A * np.sin(omega * t)
    y = B * np.sin(2 * omega * t)
    z = np.zeros_like(t)
    pos_true = np.column_stack([x, y, z])

    # Velocity (derivative)
    vx = A * omega * np.cos(omega * t)
    vy = 2 * B * omega * np.cos(2 * omega * t)
    vz = np.zeros_like(t)
    vel_true = np.column_stack([vx, vy, vz])

    # Attitude (yaw follows velocity direction, roll/pitch = 0)
    # NOTE FOR STUDENTS: Here, heading is DERIVED from the velocity vector, not
    # preset to a fixed value. This creates a naturally varying heading trajectory
    # that demonstrates IMU attitude tracking. The figure-8 path continuously
    # changes direction, testing the gyroscope integration accuracy.
    yaw = np.arctan2(vy, vx)  # Heading = direction of velocity vector

    # Convert to quaternions (scalar-first, body-to-map)
    quat_true = np.column_stack(
        [np.cos(yaw / 2), np.zeros_like(t), np.zeros_like(t), np.sin(yaw / 2)]
    )

    # Generate IMU measurements using correct forward model with Eq. (6.8)
    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_true,
        vel_map=vel_true,
        quat_b_to_m=quat_true,
        dt=dt,
        frame=frame,
        g=9.81,
        lat_rad=lat_rad,
    )

    return Figure8Trajectory(
        timestamps_s=t,
        true_positions_map_m=pos_true,
        true_velocities_map_mps=vel_true,
        true_attitudes_quat_wxyz=quat_true,
        specific_force_body_mps2=accel_body,
        angular_rates_body_rad_s=gyro_body,
    )


def add_imu_noise(
    accel_true, gyro_true, dt, imu_params: IMUNoiseParams, seed: int = DEFAULT_SEED
):
    """
    Add realistic IMU noise and biases using explicit unit conversions.

    Args:
        accel_true: True acceleration [m/s²], shape (N, 3).
        gyro_true: True angular velocity [rad/s], shape (N, 3).
        dt: Time step [s].
        imu_params: IMU noise parameters with explicit units.
        seed: Seed for this run's noise. The bias below is a single draw that
            sets the direction the unaided solution walks off in, so without a
            seed the committed figure could not be regenerated and every diff
            was noise.

    Returns:
        Tuple of (accel_meas, gyro_meas, accel_bias, gyro_bias).
    """
    N = len(accel_true)
    rng = np.random.default_rng(seed)

    # Constant biases (slowly varying in reality, but constant for this example)
    # Sample from zero-mean Gaussian with std = bias_instability
    gyro_bias = rng.standard_normal(3) * imu_params.gyro_bias_rad_s
    accel_bias = rng.standard_normal(3) * imu_params.accel_bias_mps2

    # White noise (scale with sqrt(1/dt) for discrete-time PSD)
    # For continuous-time noise with PSD σ², discrete noise std is σ/√dt
    gyro_noise_std = random_walk_to_rate_sample_std(imu_params.gyro_arw_rad_sqrt_s, dt)
    accel_noise_std = random_walk_to_rate_sample_std(
        imu_params.accel_vrw_mps_sqrt_s, dt
    )

    gyro_noise = rng.standard_normal((N, 3)) * gyro_noise_std
    accel_noise = rng.standard_normal((N, 3)) * accel_noise_std

    # Measured = true + bias + noise (Eq. 6.5, Eq. 6.9)
    gyro_meas = gyro_true + gyro_bias + gyro_noise
    accel_meas = accel_true + accel_bias + accel_noise

    return accel_meas, gyro_meas, accel_bias, gyro_bias


def run_imu_strapdown(
    t, accel_meas, gyro_meas, initial_state, frame, lat_rad=np.deg2rad(45.0)
):
    """
    Run pure IMU strapdown integration (no corrections).

    Args:
        t: Time array [s], shape (N,).
        accel_meas: Measured acceleration [m/s²], shape (N, 3).
        gyro_meas: Measured angular velocity [rad/s], shape (N, 3).
        initial_state: Initial NavStateQPVP.
        frame: Frame convention.
        lat_rad: Latitude [rad] for the Eq. (6.8) gravity model. Must match the
            latitude the trajectory was generated at, or the integrated gravity
            will not cancel the simulated specific force.

    Returns:
        Tuple of (pos_est, vel_est, quat_est).
    """
    N = len(t)
    dt = t[1] - t[0]

    # Initialize
    q = initial_state.q.copy()
    v = initial_state.v.copy()
    p = initial_state.p.copy()

    # Storage
    pos_est = np.zeros((N, 3))
    vel_est = np.zeros((N, 3))
    quat_est = np.zeros((N, 4))

    pos_est[0] = p
    vel_est[0] = v
    quat_est[0] = q

    # Propagate
    for k in range(1, N):
        # Get IMU measurements
        omega_b = gyro_meas[k - 1]
        f_b = accel_meas[k - 1]

        # Strapdown update (Eqs. 6.2-6.10) with Eq. (6.8) gravity
        q, v, p = strapdown_update(
            q=q, v=v, p=p, omega_b=omega_b, f_b=f_b, dt=dt, frame=frame, lat_rad=lat_rad
        )

        # Store
        pos_est[k] = p
        vel_est[k] = v
        quat_est[k] = q

    return pos_est, vel_est, quat_est


def quat_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion [q0, q1, q2, q3], shape (4,) or (N, 4).

    Returns:
        Euler angles [roll, pitch, yaw] in radians, shape (3,) or (N, 3).
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    # Pitch (y-axis rotation)
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))

    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    euler = np.column_stack([roll, pitch, yaw])

    if squeeze:
        return euler[0]
    return euler


def plot_results(
    t, pos_true, pos_est, vel_true, vel_est, quat_true, quat_est, figs_dir
):
    """
    Generate publication-quality plots.

    Args:
        t: Time array [s].
        pos_true: True position [m], shape (N, 3).
        pos_est: Estimated position [m], shape (N, 3).
        vel_true: True velocity [m/s], shape (N, 3).
        vel_est: Estimated velocity [m/s], shape (N, 3).
        quat_true: True quaternion, shape (N, 4).
        quat_est: Estimated quaternion, shape (N, 4).
        figs_dir: Directory to save figures.
    """
    # Convert quaternions to Euler
    att_true = quat_to_euler(quat_true)
    att_est = quat_to_euler(quat_est)

    # Compute errors
    pos_error = np.linalg.norm(pos_est - pos_true, axis=1)
    vel_error = np.linalg.norm(vel_est - vel_true, axis=1)
    att_error = np.abs(att_est - att_true)
    att_error = np.rad2deg(att_error)  # Convert to degrees

    # Figure 1: Trajectory (2D)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(pos_true[:, 0], pos_true[:, 1], "k-", linewidth=2, label="True Trajectory")
    ax1.plot(
        pos_est[:, 0],
        pos_est[:, 1],
        "r--",
        linewidth=2,
        label="IMU Estimated (no corrections)",
    )
    ax1.scatter(
        pos_true[0, 0],
        pos_true[0, 1],
        c="g",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax1.scatter(
        pos_true[-1, 0],
        pos_true[-1, 1],
        c="b",
        s=100,
        marker="s",
        label="End (true)",
        zorder=5,
    )
    ax1.scatter(
        pos_est[-1, 0],
        pos_est[-1, 1],
        c="r",
        s=100,
        marker="x",
        label="End (estimated)",
        zorder=5,
    )
    ax1.set_xlabel("East [m]", fontsize=12)
    ax1.set_ylabel("North [m]", fontsize=12)
    ax1.set_title(
        "IMU Strapdown: Trajectory (Pure Integration, No Corrections)", fontsize=14
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    plt.tight_layout()
    paths = save_figure(fig1, figs_dir, "imu_strapdown_trajectory")
    print(f"  [OK] Saved: {paths[0]}")

    # Figure 2: Position Error vs Time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t, pos_error, "r-", linewidth=2)
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Position Error [m]", fontsize=12)
    ax2.set_title(
        "IMU Strapdown: Position Error vs Time (Unbounded Drift)", fontsize=14
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])
    plt.tight_layout()
    paths = save_figure(fig2, figs_dir, "imu_strapdown_error_time")
    print(f"  [OK] Saved: {paths[0]}")

    # Figure 3: Attitude Evolution
    fig3, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, np.rad2deg(att_true[:, 0]), "k-", linewidth=2, label="True")
    axes[0].plot(t, np.rad2deg(att_est[:, 0]), "r--", linewidth=2, label="Estimated")
    axes[0].set_ylabel("Roll [deg]", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("IMU Strapdown: Attitude Evolution", fontsize=14)

    axes[1].plot(t, np.rad2deg(att_true[:, 1]), "k-", linewidth=2, label="True")
    axes[1].plot(t, np.rad2deg(att_est[:, 1]), "r--", linewidth=2, label="Estimated")
    axes[1].set_ylabel("Pitch [deg]", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, np.rad2deg(att_true[:, 2]), "k-", linewidth=2, label="True")
    axes[2].plot(t, np.rad2deg(att_est[:, 2]), "r--", linewidth=2, label="Estimated")
    axes[2].set_ylabel("Yaw [deg]", fontsize=12)
    axes[2].set_xlabel("Time [s]", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    paths = save_figure(fig3, figs_dir, "imu_strapdown_attitude")
    print(f"  [OK] Saved: {paths[0]}")

    plt.close("all")

    return pos_error, vel_error, att_error


def main():
    """Main execution function."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    print("\n" + "=" * 60)
    print("Chapter 6: IMU Strapdown Integration (Pure, No Corrections)")
    print("=" * 60)
    print("\nThis example demonstrates UNBOUNDED DRIFT in pure IMU integration.")
    print("Key equations: 6.2-6.4 (quaternion), 6.7 (velocity), 6.10 (position)\n")

    # Configuration
    duration = 100.0  # seconds
    dt = 0.01  # 100 Hz IMU
    imu_params = IMUNoiseParams.consumer_grade()  # Use consumer-grade IMU
    frame = FrameConvention.create_enu()  # Use ENU frame

    # Latitude for gravity model (Eq. 6.8)
    # Example: 45° North (mid-latitude, typical for many cities)
    lat_deg = 45.0  # degrees
    lat_rad = np.deg2rad(lat_deg)
    print(f"Using latitude: {lat_deg}° N for Eq. (6.8) gravity model")

    print("Configuration:")
    print(f"  Duration:        {duration} s")
    print(f"  IMU Rate:        {1 / dt:.0f} Hz")
    print(f"  IMU Grade:       {imu_params.grade}")
    print("  Trajectory:      Figure-8 pattern")
    print(f"  Frame:           {frame.map_frame}\n")

    # Print IMU specifications with explicit units
    print(imu_params.format_specs())
    print()

    # Generate true trajectory with correct IMU forward model
    print("Generating trajectory...")
    t, pos_true, vel_true, quat_true, accel_body, gyro_body = (
        generate_figure8_trajectory(
            duration=duration, dt=dt, frame=frame, lat_deg=lat_deg
        )
    )

    total_distance = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    print(f"  Total distance:  {total_distance:.1f} m")

    # Add IMU noise
    print("\nAdding IMU noise and biases...")
    accel_meas, gyro_meas, accel_bias, gyro_bias = add_imu_noise(
        accel_body, gyro_body, dt, imu_params
    )
    # Print realized bias values (random samples from the bias distribution)
    print(
        f"  Gyro bias (realized):  {units.format_gyro_bias(np.linalg.norm(gyro_bias))}"
    )
    print(
        f"  Accel bias (realized): {units.format_accel_bias(np.linalg.norm(accel_bias))}"
    )

    # Initial state (perfect knowledge)
    initial_state = NavStateQPVP(
        q=quat_true[0].copy(),
        v=vel_true[0].copy(),
        p=pos_true[0].copy(),
    )

    # Run IMU strapdown integration
    print("\nRunning IMU strapdown integration (no corrections)...")
    start_time = time.time()
    pos_est, vel_est, quat_est = run_imu_strapdown(
        t, accel_meas, gyro_meas, initial_state, frame, lat_rad=lat_rad
    )
    elapsed = time.time() - start_time
    print(f"  Computation time: {elapsed:.3f} s ({len(t) / elapsed:.0f}x real-time)")

    # Create output directory
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    pos_error, vel_error, att_error = plot_results(
        t, pos_true, pos_est, vel_true, vel_est, quat_true, quat_est, figs_dir
    )

    # Compute metrics
    final_pos_error = pos_error[-1]
    max_vel_error = np.max(vel_error)
    max_att_error = np.max(att_error, axis=0)
    drift_rate = final_pos_error / duration  # m/s
    drift_percent = (final_pos_error / total_distance) * 100

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (IMU-only, no corrections)")
    print("=" * 60)
    print(
        f"  Final Position Error:  {final_pos_error:.1f} m ({drift_percent:.1f}% of distance)"
    )
    print(f"  Max Velocity Error:    {max_vel_error:.2f} m/s")
    print("  Max Attitude Error:")
    print(f"    Roll:   {max_att_error[0]:.1f}°")
    print(f"    Pitch:  {max_att_error[1]:.1f}°")
    print(f"    Yaw:    {max_att_error[2]:.1f}°")
    print(f"  Drift Rate:            {drift_rate:.3f} m/s (UNBOUNDED!)")
    print()
    print(f"Figures saved to: {resolve_figs_dir(figs_dir)}/")
    print()
    print("=" * 60)
    print("KEY INSIGHT: IMU drift is UNBOUNDED without corrections!")
    print("             Velocity errors integrate to position errors.")
    print("             Errors grow without bound over time.")
    print("             Solutions: ZUPT, wheel fusion, GPS, etc.")
    print("=" * 60)
    print()
    show_figures_if_requested()


if __name__ == "__main__":
    main()
