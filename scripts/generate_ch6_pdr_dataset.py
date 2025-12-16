"""
Generate Ch6 Pedestrian Dead Reckoning (PDR) Dataset.

This script generates synthetic PDR datasets for smartphone-based pedestrian
navigation using step detection and heading estimation. Demonstrates the
critical importance of accurate heading for PDR performance.

Key Learning Objectives:
    - Step detection from accelerometer magnitude (Eq. 6.46)
    - Step length models (Weinberg formula, Eq. 6.49)
    - Heading estimation: gyro drift vs. magnetometer absolute
    - 1 degree heading error causes ~1.7% position error per step!

Implements Equations:
    - Eq. (6.46): Total acceleration magnitude
    - Eq. (6.47): Gravity-removed magnitude
    - Eq. (6.48): Step frequency estimation
    - Eq. (6.49): Step length (Weinberg model)
    - Eq. (6.50): 2D position update from step

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

from core.sensors import (
    total_accel_magnitude,
    step_length,
    pdr_step_update,
    integrate_gyro_heading,
    wrap_heading,
    mag_heading,
)


def generate_corridor_walk(
    num_legs: int = 4,
    leg_length: float = 30.0,
    step_freq: float = 2.0,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate corridor walk trajectory with turns.

    Args:
        num_legs: Number of corridor legs (straight segments).
        leg_length: Length of each leg in meters.
        step_freq: Step frequency in Hz.
        dt: Time step in seconds.

    Returns:
        Tuple of (t, pos, accel, gyro, mag, heading, step_times):
            - t: Time array [N]
            - pos: True 2D positions [N, 2] in meters
            - accel: Accelerometer [N, 3] in m/s^2
            - gyro: Gyroscope [N, 3] in rad/s
            - mag: Magnetometer [N, 3] in normalized units
            - heading: True heading [N] in radians
            - step_times: List of step occurrence times
    """
    # Constants
    GRAVITY = 9.81
    HEIGHT = 1.75  # meters
    TURN_DURATION = 2.0  # seconds for 90-degree turn

    # Compute step parameters
    step_period = 1.0 / step_freq
    step_len = step_length(HEIGHT, step_freq)
    steps_per_leg = int(leg_length / step_len)
    leg_duration = steps_per_leg * step_period

    # Total duration
    total_duration = num_legs * (leg_duration + TURN_DURATION)
    t = np.arange(0, total_duration, dt)
    N = len(t)

    # Initialize arrays
    pos = np.zeros((N, 2))
    accel = np.zeros((N, 3))
    gyro = np.zeros((N, 3))
    mag = np.zeros((N, 3))
    heading = np.zeros(N)
    step_times = []

    # Initial state
    x, y = 0.0, 0.0
    yaw = 0.0  # Start heading East

    # Generate walk
    current_leg = 0
    time_in_leg = 0.0
    time_since_last_step = 0.0
    is_turning = False

    for i in range(N):
        current_time = t[i]

        # Determine if in straight walk or turn
        leg_end_time = (current_leg + 1) * (leg_duration + TURN_DURATION)
        leg_start_time = current_leg * (leg_duration + TURN_DURATION)
        time_in_leg = current_time - leg_start_time

        if time_in_leg < leg_duration:
            # Straight walk
            is_turning = False
            omega_z = 0.0

            # Generate steps at regular intervals
            time_since_last_step += dt
            if time_since_last_step >= step_period:
                # Step occurs!
                step_times.append(current_time)
                time_since_last_step = 0.0

                # Move forward
                x += step_len * np.cos(yaw)
                y += step_len * np.sin(yaw)

                # Acceleration spike during step (vertical + forward)
                accel[i, 0] = 1.5 * np.cos(yaw)  # Forward acceleration
                accel[i, 1] = 1.5 * np.sin(yaw)
                accel[i, 2] = GRAVITY + 2.0  # Upward spike
            else:
                # No step: just gravity
                accel[i, 0] = 0.0
                accel[i, 1] = 0.0
                accel[i, 2] = GRAVITY

        elif time_in_leg < leg_duration + TURN_DURATION:
            # Turning
            is_turning = True
            turn_progress = (time_in_leg - leg_duration) / TURN_DURATION
            omega_z = (np.pi / 2) / TURN_DURATION  # 90 degrees over TURN_DURATION

            # Update yaw
            yaw = (current_leg + turn_progress) * (np.pi / 2)

            # Slower steps during turn
            time_since_last_step += dt
            if time_since_last_step >= step_period * 1.5:
                step_times.append(current_time)
                time_since_last_step = 0.0

                # Smaller step during turn
                x += step_len * 0.6 * np.cos(yaw)
                y += step_len * 0.6 * np.sin(yaw)

                accel[i, 0] = 1.0 * np.cos(yaw)
                accel[i, 1] = 1.0 * np.sin(yaw)
                accel[i, 2] = GRAVITY + 1.5
            else:
                accel[i, 0] = 0.0
                accel[i, 1] = 0.0
                accel[i, 2] = GRAVITY

            gyro[i, 2] = omega_z
        else:
            # Move to next leg
            current_leg += 1
            time_in_leg = 0.0
            yaw = current_leg * (np.pi / 2)

        # Store state
        pos[i] = [x, y]
        heading[i] = yaw

        # Magnetometer (points North in map frame)
        # In body frame, this is rotated by -yaw
        mag[i, 0] = np.cos(-yaw)
        mag[i, 1] = np.sin(-yaw)
        mag[i, 2] = 0.0

    return t, pos, accel, gyro, mag, heading, step_times


def add_sensor_noise(
    accel_true: np.ndarray,
    gyro_true: np.ndarray,
    mag_true: np.ndarray,
    accel_noise: float = 0.2,
    gyro_noise: float = 0.01,
    mag_noise: float = 0.1,
    gyro_bias: float = 0.005,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add realistic sensor noise.

    Args:
        accel_true: True accelerometer [N, 3] m/s^2.
        gyro_true: True gyro [N, 3] rad/s.
        mag_true: True magnetometer [N, 3] normalized.
        accel_noise: Accel noise std dev (m/s^2).
        gyro_noise: Gyro noise std dev (rad/s).
        mag_noise: Mag noise std dev (normalized).
        gyro_bias: Gyro bias (rad/s).
        seed: Random seed.

    Returns:
        Tuple of (accel_meas, gyro_meas, mag_meas).
    """
    rng = np.random.default_rng(seed)

    accel_meas = accel_true + rng.normal(0, accel_noise, accel_true.shape)
    gyro_meas = gyro_true + rng.normal(0, gyro_noise, gyro_true.shape) + gyro_bias
    mag_meas = mag_true + rng.normal(0, mag_noise, mag_true.shape)

    # Normalize magnetometer
    mag_norm = np.linalg.norm(mag_meas, axis=1, keepdims=True)
    mag_meas = mag_meas / (mag_norm + 1e-9)

    return accel_meas, gyro_meas, mag_meas


def run_pdr_gyro(
    t: np.ndarray,
    accel_meas: np.ndarray,
    gyro_meas: np.ndarray,
    height: float = 1.75,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run PDR with gyro-integrated heading (drifts over time).

    Args:
        t: Time array [N].
        accel_meas: Accel measurements [N, 3] m/s^2.
        gyro_meas: Gyro measurements [N, 3] rad/s.
        height: Pedestrian height (m).

    Returns:
        Tuple of (pos, heading, num_steps).
    """
    N = len(t)
    dt = t[1] - t[0]

    pos = np.zeros((N, 2))
    heading_est = np.zeros(N)
    step_count = 0

    last_step_time = 0.0
    last_a_mag = 10.0

    for k in range(1, N):
        # Step detection: simple peak crossing at 11 m/s^2
        a_mag = total_accel_magnitude(accel_meas[k])
        is_step = (last_a_mag < 11.0 and a_mag >= 11.0)
        last_a_mag = a_mag

        if is_step and (t[k] - last_step_time) > 0.3:  # Min 0.3s between steps
            step_count += 1
            delta_t = t[k] - last_step_time
            last_step_time = t[k]

            # Step frequency (Eq. 6.48)
            f_step = 1.0 / delta_t if delta_t > 0 else 2.0

            # Step length (Eq. 6.49 - Weinberg model)
            L = step_length(height, f_step)

            # Update position (Eq. 6.50)
            pos[k] = pdr_step_update(pos[k - 1], L, heading_est[k - 1])
        else:
            pos[k] = pos[k - 1]

        # Integrate gyro heading (DRIFTS!)
        heading_est[k] = integrate_gyro_heading(heading_est[k - 1], gyro_meas[k, 2], dt)
        heading_est[k] = wrap_heading(heading_est[k])

    return pos, heading_est, step_count


def run_pdr_mag(
    t: np.ndarray,
    accel_meas: np.ndarray,
    mag_meas: np.ndarray,
    height: float = 1.75,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run PDR with magnetometer heading (absolute but noisy).

    Args:
        t: Time array [N].
        accel_meas: Accel measurements [N, 3] m/s^2.
        mag_meas: Mag measurements [N, 3] normalized.
        height: Pedestrian height (m).

    Returns:
        Tuple of (pos, heading, num_steps).
    """
    N = len(t)

    pos = np.zeros((N, 2))
    heading_est = np.zeros(N)
    step_count = 0

    last_step_time = 0.0
    last_a_mag = 10.0

    for k in range(1, N):
        # Step detection
        a_mag = total_accel_magnitude(accel_meas[k])
        is_step = (last_a_mag < 11.0 and a_mag >= 11.0)
        last_a_mag = a_mag

        if is_step and (t[k] - last_step_time) > 0.3:
            step_count += 1
            delta_t = t[k] - last_step_time
            last_step_time = t[k]

            f_step = 1.0 / delta_t if delta_t > 0 else 2.0
            L = step_length(height, f_step)

            pos[k] = pdr_step_update(pos[k - 1], L, heading_est[k - 1])
        else:
            pos[k] = pos[k - 1]

        # Magnetometer heading (Eqs. 6.51-6.53)
        # Assume level (roll=pitch=0 for simplicity)
        heading_est[k] = mag_heading(mag_meas[k], roll=0.0, pitch=0.0, declination=0.0)

    return pos, heading_est, step_count


def save_dataset(
    output_dir: Path,
    t: np.ndarray,
    pos_true: np.ndarray,
    accel_true: np.ndarray,
    gyro_true: np.ndarray,
    mag_true: np.ndarray,
    heading_true: np.ndarray,
    accel_meas: np.ndarray,
    gyro_meas: np.ndarray,
    mag_meas: np.ndarray,
    step_times: List[float],
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
        header="x (m), y (m)",
    )
    np.savetxt(
        output_dir / "ground_truth_heading.txt",
        heading_true,
        fmt="%.6f",
        header="heading (rad)",
    )

    # Save noisy measurements
    np.savetxt(
        output_dir / "accel.txt",
        accel_meas,
        fmt="%.6f",
        header="ax (m/s^2), ay (m/s^2), az (m/s^2)",
    )
    np.savetxt(
        output_dir / "gyro.txt",
        gyro_meas,
        fmt="%.6f",
        header="omega_x (rad/s), omega_y (rad/s), omega_z (rad/s)",
    )
    np.savetxt(
        output_dir / "magnetometer.txt",
        mag_meas,
        fmt="%.6f",
        header="mx (normalized), my (normalized), mz (normalized)",
    )

    # Save clean signals
    np.savetxt(
        output_dir / "accel_clean.txt",
        accel_true,
        fmt="%.6f",
        header="ax (m/s^2), ay (m/s^2), az (m/s^2)",
    )
    np.savetxt(
        output_dir / "gyro_clean.txt",
        gyro_true,
        fmt="%.6f",
        header="omega_x (rad/s), omega_y (rad/s), omega_z (rad/s)",
    )
    np.savetxt(
        output_dir / "magnetometer_clean.txt",
        mag_true,
        fmt="%.6f",
        header="mx (normalized), my (normalized), mz (normalized)",
    )

    # Save step times
    np.savetxt(
        output_dir / "step_times.txt",
        np.array(step_times),
        fmt="%.6f",
        header="step occurrence times (s)",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print(f"    Files: 10 files (time, GT x2, measurements x3, clean x3, steps, config)")
    print(f"    Samples: {len(t)}")
    print(f"    Steps: {len(step_times)}")


def generate_dataset(
    output_dir: str,
    preset: Optional[str] = None,
    num_legs: int = 4,
    leg_length: float = 30.0,
    step_freq: float = 2.0,
    height: float = 1.75,
    dt: float = 0.01,
    accel_noise: float = 0.2,
    gyro_noise: float = 0.01,
    mag_noise: float = 0.1,
    gyro_bias: float = 0.005,
    seed: int = 42,
) -> None:
    """
    Generate PDR dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        num_legs: Number of corridor legs.
        leg_length: Length of each leg (m).
        step_freq: Step frequency (Hz).
        height: Pedestrian height (m).
        dt: Time step (s).
        accel_noise: Accel noise std dev (m/s^2).
        gyro_noise: Gyro noise std dev (rad/s).
        mag_noise: Mag noise std dev (normalized).
        gyro_bias: Gyro bias (rad/s).
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "baseline":
        # Clean sensors
        accel_noise = 0.15
        gyro_noise = 0.005
        mag_noise = 0.05
        gyro_bias = 0.002
        output_dir = "data/sim/ch6_pdr_corridor_walk"
    elif preset == "noisy":
        # Higher noise
        accel_noise = 0.3
        gyro_noise = 0.015
        mag_noise = 0.15
        gyro_bias = 0.01
        output_dir = "data/sim/ch6_pdr_noisy"
    elif preset == "poor_gyro":
        # Poor gyro (severe heading drift)
        accel_noise = 0.2
        gyro_noise = 0.03
        mag_noise = 0.08
        gyro_bias = 0.02
        output_dir = "data/sim/ch6_pdr_poor_gyro"
    elif preset == "poor_mag":
        # Poor magnetometer (distorted heading)
        accel_noise = 0.2
        gyro_noise = 0.008
        mag_noise = 0.3
        gyro_bias = 0.005
        output_dir = "data/sim/ch6_pdr_poor_mag"

    print("\n" + "=" * 70)
    print(f"Generating Ch6 PDR Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Generate trajectory
    print("\nStep 1: Generating corridor walk...")
    t, pos_true, accel_true, gyro_true, mag_true, heading_true, step_times = generate_corridor_walk(
        num_legs=num_legs,
        leg_length=leg_length,
        step_freq=step_freq,
        dt=dt,
    )

    total_distance = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    duration = t[-1]

    print(f"  Duration: {duration:.1f} s")
    print(f"  Distance: {total_distance:.1f} m")
    print(f"  True steps: {len(step_times)}")
    print(f"  Samples: {len(t)}")

    # Add noise
    print("\nStep 2: Adding sensor noise...")
    print(f"  Accel noise: {accel_noise:.3f} m/s^2")
    print(f"  Gyro noise: {gyro_noise:.6f} rad/s")
    print(f"  Gyro bias: {gyro_bias:.6f} rad/s")
    print(f"  Mag noise: {mag_noise:.3f}")

    accel_meas, gyro_meas, mag_meas = add_sensor_noise(
        accel_true,
        gyro_true,
        mag_true,
        accel_noise=accel_noise,
        gyro_noise=gyro_noise,
        mag_noise=mag_noise,
        gyro_bias=gyro_bias,
        seed=seed,
    )

    # Run PDR with gyro heading
    print("\nStep 3: Running PDR with gyro heading...")
    start = time.time()
    pos_gyro, heading_gyro, steps_gyro = run_pdr_gyro(t, accel_meas, gyro_meas, height)
    elapsed_gyro = time.time() - start

    error_gyro = np.linalg.norm(pos_gyro - pos_true, axis=1)
    final_error_gyro = error_gyro[-1]
    mean_error_gyro = np.mean(error_gyro)

    print(f"  Time: {elapsed_gyro:.3f} s")
    print(f"  Detected steps: {steps_gyro}/{len(step_times)}")
    print(f"  Final error: {final_error_gyro:.3f} m")
    print(f"  Mean error: {mean_error_gyro:.3f} m")

    # Run PDR with mag heading
    print("\nStep 4: Running PDR with magnetometer heading...")
    start = time.time()
    pos_mag, heading_mag, steps_mag = run_pdr_mag(t, accel_meas, mag_meas, height)
    elapsed_mag = time.time() - start

    error_mag = np.linalg.norm(pos_mag - pos_true, axis=1)
    final_error_mag = error_mag[-1]
    mean_error_mag = np.mean(error_mag)

    print(f"  Time: {elapsed_mag:.3f} s")
    print(f"  Detected steps: {steps_mag}/{len(step_times)}")
    print(f"  Final error: {final_error_mag:.3f} m")
    print(f"  Mean error: {mean_error_mag:.3f} m")

    print(f"\nHeading Source Comparison:")
    print(f"  Gyro: {final_error_gyro:.2f}m error (drifts over time)")
    print(f"  Magnetometer: {final_error_mag:.2f}m error (absolute but noisy)")
    print(f"  Improvement: {final_error_gyro / final_error_mag:.1f}x better with magnetometer")

    # Save dataset
    config = {
        "dataset": "ch6_pdr",
        "preset": preset,
        "trajectory": {
            "type": "corridor_walk",
            "num_legs": num_legs,
            "leg_length_m": leg_length,
            "total_distance_m": float(total_distance),
            "duration_s": float(duration),
        },
        "pedestrian": {
            "height_m": height,
            "step_freq_hz": step_freq,
            "num_steps": len(step_times),
        },
        "dt_s": dt,
        "sample_rate_hz": 1.0 / dt,
        "num_samples": len(t),
        "sensors": {
            "accel_noise_std_m_s2": accel_noise,
            "gyro_noise_std_rad_s": gyro_noise,
            "gyro_bias_rad_s": gyro_bias,
            "mag_noise_std": mag_noise,
        },
        "performance": {
            "gyro_heading": {
                "steps_detected": steps_gyro,
                "final_error_m": float(final_error_gyro),
                "mean_error_m": float(mean_error_gyro),
            },
            "mag_heading": {
                "steps_detected": steps_mag,
                "final_error_m": float(final_error_mag),
                "mean_error_m": float(mean_error_mag),
            },
            "improvement_factor": float(final_error_gyro / final_error_mag),
        },
        "equations": ["6.46", "6.47", "6.48", "6.49", "6.50", "6.51", "6.52", "6.53"],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        t,
        pos_true,
        accel_true,
        gyro_true,
        mag_true,
        heading_true,
        accel_meas,
        gyro_meas,
        mag_meas,
        step_times,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch6 Pedestrian Dead Reckoning (PDR) Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline     Clean sensors (gyro: 15-25m drift, mag: 2-4m error)
  noisy        Higher noise (gyro: 40-60m drift, mag: 5-8m error)
  poor_gyro    Severe gyro drift (gyro: 100+m drift, mag: 3-5m error)
  poor_mag     Distorted magnetometer (gyro: 20-30m drift, mag: 10-15m error)

Examples:
  # Generate baseline dataset
  python scripts/generate_ch6_pdr_dataset.py --preset baseline

  # Generate dataset with custom parameters
  python scripts/generate_ch6_pdr_dataset.py \\
      --output data/sim/my_pdr \\
      --num-legs 6 \\
      --leg-length 40 \\
      --step-freq 2.5

  # Generate all presets
  python scripts/generate_ch6_pdr_dataset.py --preset baseline
  python scripts/generate_ch6_pdr_dataset.py --preset noisy
  python scripts/generate_ch6_pdr_dataset.py --preset poor_gyro
  python scripts/generate_ch6_pdr_dataset.py --preset poor_mag

Learning Focus:
  - Heading errors DOMINATE PDR accuracy (1 deg heading = 1.7% position error!)
  - Gyro-integrated heading drifts severely over time
  - Magnetometer provides absolute heading but is noisy
  - Step detection and step length models affect accuracy
  - PDR is lightweight but requires frequent corrections

Book Reference: Chapter 6, Section 6.3 (Pedestrian Dead Reckoning)
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "noisy", "poor_gyro", "poor_mag"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/ch6_pdr_corridor_walk",
        help="Output directory (default: data/sim/ch6_pdr_corridor_walk)",
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--num-legs", type=int, default=4, help="Number of corridor legs (default: 4)"
    )
    traj_group.add_argument(
        "--leg-length", type=float, default=30.0, help="Length of each leg in meters (default: 30.0)"
    )
    traj_group.add_argument(
        "--step-freq", type=float, default=2.0, help="Step frequency in Hz (default: 2.0)"
    )
    traj_group.add_argument(
        "--height", type=float, default=1.75, help="Pedestrian height in meters (default: 1.75)"
    )
    traj_group.add_argument(
        "--dt", type=float, default=0.01, help="Time step in seconds (default: 0.01)"
    )

    # Sensor noise parameters
    noise_group = parser.add_argument_group("Sensor Noise Parameters")
    noise_group.add_argument(
        "--accel-noise", type=float, default=0.2, help="Accel noise std dev in m/s^2 (default: 0.2)"
    )
    noise_group.add_argument(
        "--gyro-noise", type=float, default=0.01, help="Gyro noise std dev in rad/s (default: 0.01)"
    )
    noise_group.add_argument(
        "--gyro-bias", type=float, default=0.005, help="Gyro bias in rad/s (default: 0.005)"
    )
    noise_group.add_argument(
        "--mag-noise", type=float, default=0.1, help="Mag noise std dev normalized (default: 0.1)"
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        num_legs=args.num_legs,
        leg_length=args.leg_length,
        step_freq=args.step_freq,
        height=args.height,
        dt=args.dt,
        accel_noise=args.accel_noise,
        gyro_noise=args.gyro_noise,
        mag_noise=args.mag_noise,
        gyro_bias=args.gyro_bias,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

