"""
Generate Ch6 Wheel Odometry Dead Reckoning Dataset.

This script generates synthetic wheel odometry datasets for vehicle dead
reckoning using wheel encoders with IMU integration. Demonstrates bounded
drift characteristics and sensitivity to wheel slip.

Key Learning Objectives:
    - Wheel odometry provides BOUNDED drift (proportional to distance)
    - Lever arm compensation affects accuracy (Eq. 6.11)
    - Wheel slip during turns causes significant errors
    - Combined wheel+IMU is more robust than IMU-only

Implements Equations:
    - Eq. (6.11): Lever arm compensation for wheel speed
    - Eq. (6.12): Skew-symmetric matrix for cross products
    - Eq. (6.14): Attitude to map frame velocity transform
    - Eq. (6.15): Position update from wheel odometry

Author: Navigation Engineer
Date: December 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensors import wheel_odom_update, NavStateQPVP
from core.sensors.strapdown import quat_integrate


def generate_square_trajectory(
    side_length: float = 20.0,
    speed: float = 5.0,
    num_laps: int = 2,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate square trajectory for vehicle.

    Args:
        side_length: Length of square side in meters.
        speed: Forward speed in m/s.
        num_laps: Number of square laps.
        dt: Time step in seconds.

    Returns:
        Tuple of (t, pos, vel, quat, wheel_speed, gyro):
            - t: Time array [N]
            - pos: True positions [N, 3] in meters
            - vel: True velocities [N, 3] in m/s
            - quat: True quaternions [N, 4] (scalar-first)
            - wheel_speed: Wheel speed in speed frame [N, 3] m/s
            - gyro: Angular velocity [N, 3] rad/s
    """
    # Square timing: straight (side/speed) + turn (90deg at omega)
    turn_omega = 0.3  # rad/s (about 17 deg/s)
    turn_time = (np.pi / 2) / turn_omega  # Time for 90 degree turn

    straight_time = side_length / speed
    lap_time = 4 * straight_time + 4 * turn_time
    duration = num_laps * lap_time

    t = np.arange(0, duration, dt)
    N = len(t)

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    quat = np.zeros((N, 4))
    wheel_speed = np.zeros((N, 3))
    gyro = np.zeros((N, 3))

    # Initial state
    x, y, yaw = 0.0, 0.0, 0.0
    quat[0] = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    for i in range(1, N):
        time_in_lap = t[i] % lap_time

        # Determine if in straight or turn
        if time_in_lap < straight_time:  # Side 1: East
            omega_z = 0.0
            v_forward = speed
            yaw_target = 0.0
        elif time_in_lap < straight_time + turn_time:  # Turn 1: CCW
            omega_z = turn_omega
            v_forward = speed * 0.8  # Slow down in turns
            yaw_target = (time_in_lap - straight_time) * turn_omega
        elif time_in_lap < 2 * straight_time + turn_time:  # Side 2: North
            omega_z = 0.0
            v_forward = speed
            yaw_target = np.pi / 2
        elif time_in_lap < 2 * straight_time + 2 * turn_time:  # Turn 2
            omega_z = turn_omega
            v_forward = speed * 0.8
            yaw_target = np.pi / 2 + (time_in_lap - 2 * straight_time - turn_time) * turn_omega
        elif time_in_lap < 3 * straight_time + 2 * turn_time:  # Side 3: West
            omega_z = 0.0
            v_forward = speed
            yaw_target = np.pi
        elif time_in_lap < 3 * straight_time + 3 * turn_time:  # Turn 3
            omega_z = turn_omega
            v_forward = speed * 0.8
            yaw_target = np.pi + (time_in_lap - 3 * straight_time - 2 * turn_time) * turn_omega
        elif time_in_lap < 4 * straight_time + 3 * turn_time:  # Side 4: South
            omega_z = 0.0
            v_forward = speed
            yaw_target = 3 * np.pi / 2
        else:  # Turn 4
            omega_z = turn_omega
            v_forward = speed * 0.8
            yaw_target = 3 * np.pi / 2 + (time_in_lap - 4 * straight_time - 3 * turn_time) * turn_omega

        # Update yaw
        yaw = yaw_target

        # Normalize yaw to [-pi, pi]
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi

        # Update velocity (body frame)
        vx = v_forward * np.cos(yaw)
        vy = v_forward * np.sin(yaw)

        vel[i] = np.array([vx, vy, 0.0])

        # Update position
        x += vx * dt
        y += vy * dt
        pos[i] = np.array([x, y, 0.0])

        # Update quaternion
        quat[i] = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

        # Wheel speed in speed frame (forward only)
        wheel_speed[i] = np.array([v_forward, 0.0, 0.0])

        # Gyro (body frame)
        gyro[i] = np.array([0.0, 0.0, omega_z])

    return t, pos, vel, quat, wheel_speed, gyro


def add_wheel_noise(
    wheel_speed_true: np.ndarray,
    gyro_true: np.ndarray,
    encoder_noise: float = 0.05,
    gyro_noise: float = 0.001,
    wheel_bias: float = 0.01,
    gyro_bias: float = 0.0005,
    add_slip: bool = False,
    slip_intervals: Optional[List[Tuple[float, float]]] = None,
    slip_magnitude: float = 0.3,
    dt: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic noise and slip to wheel/gyro measurements.

    Args:
        wheel_speed_true: True wheel speeds [N, 3] in m/s.
        gyro_true: True gyro [N, 3] in rad/s.
        encoder_noise: Wheel encoder noise std dev (m/s).
        gyro_noise: Gyro noise std dev (rad/s).
        wheel_bias: Wheel speed bias (m/s).
        gyro_bias: Gyro bias (rad/s).
        add_slip: Whether to add wheel slip during turns.
        slip_intervals: List of (start_time, end_time) for slip events.
        slip_magnitude: Slip magnitude (fraction of speed).
        dt: Time step.
        seed: Random seed.

    Returns:
        Tuple of (wheel_meas, gyro_meas).
    """
    rng = np.random.default_rng(seed)
    N = len(wheel_speed_true)

    # Add white noise + bias
    wheel_noise = rng.normal(0, encoder_noise, wheel_speed_true.shape)
    gyro_noise_vec = rng.normal(0, gyro_noise, gyro_true.shape)

    wheel_meas = wheel_speed_true + wheel_noise + wheel_bias
    gyro_meas = gyro_true + gyro_noise_vec + gyro_bias

    # Add wheel slip during specified intervals
    if add_slip and slip_intervals is not None:
        t = np.arange(N) * dt
        for start_t, end_t in slip_intervals:
            mask = (t >= start_t) & (t <= end_t)
            # Slip: reduce measured wheel speed (wheel spins but vehicle doesn't move)
            wheel_meas[mask, 0] *= (1.0 + slip_magnitude)

    return wheel_meas, gyro_meas


def run_wheel_odometry(
    t: np.ndarray,
    wheel_meas: np.ndarray,
    gyro_meas: np.ndarray,
    initial_state: NavStateQPVP,
    lever_arm: np.ndarray,
) -> np.ndarray:
    """
    Run wheel odometry dead reckoning.

    Args:
        t: Time array [N].
        wheel_meas: Wheel speed measurements [N, 3] m/s.
        gyro_meas: Gyro measurements [N, 3] rad/s.
        initial_state: Initial navigation state.
        lever_arm: Lever arm from IMU to wheel center [3] m.

    Returns:
        Estimated positions [N, 3] m.
    """
    N = len(t)
    pos = np.zeros((N, 3))
    quat = np.zeros((N, 4))

    pos[0] = initial_state.p
    quat[0] = initial_state.q

    for i in range(1, N):
        dt = t[i] - t[i - 1]

        # Update attitude using gyro
        quat[i] = quat_integrate(quat[i - 1], gyro_meas[i - 1], dt)

        # Update position using wheel odometry with lever arm
        pos[i] = wheel_odom_update(
            pos[i - 1], quat[i], wheel_meas[i], gyro_meas[i], lever_arm, dt
        )

    return pos


def save_dataset(
    output_dir: Path,
    t: np.ndarray,
    pos_true: np.ndarray,
    vel_true: np.ndarray,
    quat_true: np.ndarray,
    wheel_true: np.ndarray,
    gyro_true: np.ndarray,
    wheel_meas: np.ndarray,
    gyro_meas: np.ndarray,
    config: Dict,
) -> None:
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth
    np.savetxt(output_dir / "time.txt", t, fmt="%.6f", header="time (s)")
    np.savetxt(
        output_dir / "ground_truth_position.txt",
        pos_true,
        fmt="%.6f",
        header="x (m), y (m), z (m)",
    )
    np.savetxt(
        output_dir / "ground_truth_velocity.txt",
        vel_true,
        fmt="%.6f",
        header="vx (m/s), vy (m/s), vz (m/s)",
    )
    np.savetxt(
        output_dir / "ground_truth_quaternion.txt",
        quat_true,
        fmt="%.6f",
        header="q0, q1, q2, q3 (scalar-first)",
    )

    # Save noisy measurements
    np.savetxt(
        output_dir / "wheel_speed.txt",
        wheel_meas,
        fmt="%.6f",
        header="v_forward (m/s), v_lateral (m/s), v_vertical (m/s)",
    )
    np.savetxt(
        output_dir / "gyro.txt",
        gyro_meas,
        fmt="%.6f",
        header="omega_x (rad/s), omega_y (rad/s), omega_z (rad/s)",
    )

    # Save clean signals for reference
    np.savetxt(
        output_dir / "wheel_speed_clean.txt",
        wheel_true,
        fmt="%.6f",
        header="v_forward (m/s), v_lateral (m/s), v_vertical (m/s)",
    )
    np.savetxt(
        output_dir / "gyro_clean.txt",
        gyro_true,
        fmt="%.6f",
        header="omega_x (rad/s), omega_y (rad/s), omega_z (rad/s)",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print(f"    Files: 9 files (time, GT x4, measurements x2, clean x2, config)")
    print(f"    Samples: {len(t)}")


def generate_dataset(
    output_dir: str,
    preset: Optional[str] = None,
    side_length: float = 20.0,
    speed: float = 5.0,
    num_laps: int = 2,
    dt: float = 0.01,
    encoder_noise: float = 0.05,
    gyro_noise: float = 0.001,
    wheel_bias: float = 0.01,
    gyro_bias: float = 0.0005,
    lever_arm: Tuple[float, float, float] = (1.5, 0.0, -0.3),
    add_slip: bool = False,
    slip_magnitude: float = 0.3,
    seed: int = 42,
) -> None:
    """
    Generate wheel odometry dead reckoning dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        side_length: Square side length (m).
        speed: Forward speed (m/s).
        num_laps: Number of square laps.
        dt: Time step (s).
        encoder_noise: Wheel encoder noise std dev (m/s).
        gyro_noise: Gyro noise std dev (rad/s).
        wheel_bias: Wheel speed bias (m/s).
        gyro_bias: Gyro bias (rad/s).
        lever_arm: Lever arm from IMU to wheel center (m).
        add_slip: Whether to add wheel slip.
        slip_magnitude: Slip magnitude (fraction).
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "baseline":
        # Clean measurements, no slip
        encoder_noise = 0.03
        gyro_noise = 0.0005
        wheel_bias = 0.005
        gyro_bias = 0.0002
        add_slip = False
        output_dir = "data/sim/ch6_wheel_odom_square"
    elif preset == "noisy":
        # Higher noise
        encoder_noise = 0.1
        gyro_noise = 0.002
        wheel_bias = 0.02
        gyro_bias = 0.001
        add_slip = False
        output_dir = "data/sim/ch6_wheel_odom_noisy"
    elif preset == "slip":
        # Add wheel slip during turns
        encoder_noise = 0.05
        gyro_noise = 0.001
        wheel_bias = 0.01
        gyro_bias = 0.0005
        add_slip = True
        slip_magnitude = 0.3
        output_dir = "data/sim/ch6_wheel_odom_slip"
    elif preset == "poor":
        # Poor quality sensors + slip
        encoder_noise = 0.15
        gyro_noise = 0.003
        wheel_bias = 0.05
        gyro_bias = 0.002
        add_slip = True
        slip_magnitude = 0.5
        output_dir = "data/sim/ch6_wheel_odom_poor"

    print("\n" + "=" * 70)
    print(f"Generating Ch6 Wheel Odometry Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Generate trajectory
    print("\nStep 1: Generating square trajectory...")
    t, pos_true, vel_true, quat_true, wheel_true, gyro_true = generate_square_trajectory(
        side_length=side_length,
        speed=speed,
        num_laps=num_laps,
        dt=dt,
    )

    total_distance = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    duration = t[-1]

    print(f"  Duration: {duration:.1f} s")
    print(f"  Distance: {total_distance:.1f} m")
    print(f"  Samples: {len(t)}")

    # Determine slip intervals (during turns)
    slip_intervals = None
    if add_slip:
        # Turns occur at specific times (calculated from trajectory generation)
        straight_time = side_length / speed
        turn_omega = 0.3
        turn_time = (np.pi / 2) / turn_omega
        lap_time = 4 * straight_time + 4 * turn_time

        slip_intervals = []
        for lap in range(num_laps):
            t_offset = lap * lap_time
            # 4 turns per lap
            for turn_idx in range(4):
                t_start = t_offset + (turn_idx + 1) * straight_time + turn_idx * turn_time
                t_end = t_start + turn_time
                slip_intervals.append((t_start, t_end))

    # Add noise and slip
    print("\nStep 2: Adding wheel/gyro noise...")
    print(f"  Encoder noise: {encoder_noise:.4f} m/s")
    print(f"  Gyro noise: {gyro_noise:.6f} rad/s")
    print(f"  Wheel bias: {wheel_bias:.4f} m/s")
    print(f"  Wheel slip: {'YES' if add_slip else 'NO'}")
    if add_slip:
        print(f"  Slip magnitude: {slip_magnitude:.1%}")
        print(f"  Slip events: {len(slip_intervals)}")

    wheel_meas, gyro_meas = add_wheel_noise(
        wheel_true,
        gyro_true,
        encoder_noise=encoder_noise,
        gyro_noise=gyro_noise,
        wheel_bias=wheel_bias,
        gyro_bias=gyro_bias,
        add_slip=add_slip,
        slip_intervals=slip_intervals,
        slip_magnitude=slip_magnitude,
        dt=dt,
        seed=seed,
    )

    # Run wheel odometry
    print("\nStep 3: Running wheel odometry...")
    initial_state = NavStateQPVP(q=quat_true[0], v=vel_true[0], p=pos_true[0])
    lever_arm_vec = np.array(lever_arm)

    start = time.time()
    pos_est = run_wheel_odometry(t, wheel_meas, gyro_meas, initial_state, lever_arm_vec)
    elapsed = time.time() - start

    # Compute error
    error = np.linalg.norm(pos_est - pos_true, axis=1)
    final_error = error[-1]
    mean_error = np.mean(error)
    max_error = np.max(error)

    print(f"  Execution time: {elapsed:.3f} s")
    print(f"\nResults:")
    print(f"  Final error: {final_error:.3f} m")
    print(f"  Mean error: {mean_error:.3f} m")
    print(f"  Max error: {max_error:.3f} m")
    print(f"  Drift rate: {final_error / total_distance * 100:.2f}% of distance")

    # Save dataset
    config = {
        "dataset": "ch6_wheel_odometry",
        "preset": preset,
        "trajectory": {
            "shape": "square",
            "side_length_m": side_length,
            "speed_m_s": speed,
            "num_laps": num_laps,
            "duration_s": float(duration),
            "total_distance_m": float(total_distance),
        },
        "dt_s": dt,
        "sample_rate_hz": 1.0 / dt,
        "num_samples": len(t),
        "encoder": {
            "noise_std_m_s": encoder_noise,
            "bias_m_s": wheel_bias,
        },
        "gyro": {
            "noise_std_rad_s": gyro_noise,
            "bias_rad_s": gyro_bias,
        },
        "lever_arm_m": list(lever_arm),
        "slip": {
            "enabled": add_slip,
            "magnitude": slip_magnitude,
            "num_events": len(slip_intervals) if slip_intervals else 0,
        },
        "performance": {
            "final_error_m": float(final_error),
            "mean_error_m": float(mean_error),
            "max_error_m": float(max_error),
            "drift_rate_percent": float(final_error / total_distance * 100),
        },
        "equations": ["6.11", "6.12", "6.14", "6.15"],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        t,
        pos_true,
        vel_true,
        quat_true,
        wheel_true,
        gyro_true,
        wheel_meas,
        gyro_meas,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch6 Wheel Odometry Dead Reckoning Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline    Clean encoder/gyro, no slip (1-2m drift over 80s)
  noisy       Higher noise, no slip (2-4m drift)
  slip        Moderate noise + wheel slip in turns (4-8m drift)
  poor        Poor sensors + severe slip (10-20m drift)

Examples:
  # Generate baseline dataset
  python scripts/generate_ch6_wheel_odom_dataset.py --preset baseline

  # Generate dataset with custom parameters
  python scripts/generate_ch6_wheel_odom_dataset.py \\
      --output data/sim/my_wheel_odom \\
      --encoder-noise 0.08 \\
      --add-slip \\
      --slip-magnitude 0.4

  # Generate all presets
  python scripts/generate_ch6_wheel_odom_dataset.py --preset baseline
  python scripts/generate_ch6_wheel_odom_dataset.py --preset noisy
  python scripts/generate_ch6_wheel_odom_dataset.py --preset slip
  python scripts/generate_ch6_wheel_odom_dataset.py --preset poor

Learning Focus:
  - Wheel odometry drift is BOUNDED (proportional to distance, not time)
  - Lever arm compensation is critical for accuracy (Eq. 6.11)
  - Wheel slip in turns causes significant errors
  - Combined wheel+IMU outperforms IMU-only strapdown

Book Reference: Chapter 6, Section 6.2 (Wheel Odometry)
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "noisy", "slip", "poor"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/ch6_wheel_odom_square",
        help="Output directory (default: data/sim/ch6_wheel_odom_square)",
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--side-length", type=float, default=20.0, help="Square side length in meters (default: 20.0)"
    )
    traj_group.add_argument(
        "--speed", type=float, default=5.0, help="Forward speed in m/s (default: 5.0)"
    )
    traj_group.add_argument(
        "--num-laps", type=int, default=2, help="Number of square laps (default: 2)"
    )
    traj_group.add_argument(
        "--dt", type=float, default=0.01, help="Time step in seconds (default: 0.01)"
    )

    # Sensor noise parameters
    noise_group = parser.add_argument_group("Sensor Noise Parameters")
    noise_group.add_argument(
        "--encoder-noise", type=float, default=0.05, help="Encoder noise std dev in m/s (default: 0.05)"
    )
    noise_group.add_argument(
        "--gyro-noise", type=float, default=0.001, help="Gyro noise std dev in rad/s (default: 0.001)"
    )
    noise_group.add_argument(
        "--wheel-bias", type=float, default=0.01, help="Wheel speed bias in m/s (default: 0.01)"
    )
    noise_group.add_argument(
        "--gyro-bias", type=float, default=0.0005, help="Gyro bias in rad/s (default: 0.0005)"
    )

    # Lever arm
    lever_group = parser.add_argument_group("Lever Arm Parameters")
    lever_group.add_argument(
        "--lever-arm",
        type=float,
        nargs=3,
        default=[1.5, 0.0, -0.3],
        metavar=("X", "Y", "Z"),
        help="Lever arm from IMU to wheel center [x y z] in meters (default: 1.5 0.0 -0.3)",
    )

    # Slip parameters
    slip_group = parser.add_argument_group("Wheel Slip Parameters")
    slip_group.add_argument(
        "--add-slip", action="store_true", help="Add wheel slip during turns"
    )
    slip_group.add_argument(
        "--slip-magnitude",
        type=float,
        default=0.3,
        help="Slip magnitude as fraction of speed (default: 0.3)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        side_length=args.side_length,
        speed=args.speed,
        num_laps=args.num_laps,
        dt=args.dt,
        encoder_noise=args.encoder_noise,
        gyro_noise=args.gyro_noise,
        wheel_bias=args.wheel_bias,
        gyro_bias=args.gyro_bias,
        lever_arm=tuple(args.lever_arm),
        add_slip=args.add_slip,
        slip_magnitude=args.slip_magnitude,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

