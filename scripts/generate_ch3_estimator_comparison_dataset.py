"""
Generate Ch3 Estimator Comparison Dataset.

This script generates synthetic trajectories with measurements demonstrating when
to use KF vs. EKF vs. UKF vs. PF. Shows the critical impact of system nonlinearity
and noise characteristics on estimator performance.

Key Learning Objectives:
    - Understand KF assumptions (linear, Gaussian)
    - Learn when EKF breaks down (high nonlinearity)
    - Study UKF advantage over EKF (better linearization)
    - Explore PF for non-Gaussian scenarios

Implements Equations:
    - Eqs. (3.11-3.19): Linear Kalman Filter
    - Eq. (3.21): Extended Kalman Filter
    - Eqs. (3.24-3.30): Unscented Kalman Filter
    - Eqs. (3.32-3.34): Particle Filter

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_trajectory(
    trajectory_type: str = "circular",
    duration: float = 30.0,
    dt: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground truth trajectory.

    Args:
        trajectory_type: Type of trajectory ('linear', 'circular', 'figure8').
        duration: Duration in seconds.
        dt: Time step in seconds.
        seed: Random seed.

    Returns:
        Tuple of (times, states) where states is [N×4] (x, y, vx, vy).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    N = len(t)
    states = np.zeros((N, 4))  # [x, y, vx, vy]

    if trajectory_type == "linear":
        # Constant velocity motion
        v = 2.0  # m/s
        states[:, 0] = v * t  # x
        states[:, 1] = 0.5 * t  # y
        states[:, 2] = v  # vx
        states[:, 3] = 0.5  # vy

    elif trajectory_type == "circular":
        # Circular motion (nonlinear!)
        omega = 0.3  # rad/s
        radius = 10.0  # m
        states[:, 0] = radius * np.cos(omega * t)  # x
        states[:, 1] = radius * np.sin(omega * t)  # y
        states[:, 2] = -radius * omega * np.sin(omega * t)  # vx
        states[:, 3] = radius * omega * np.cos(omega * t)  # vy

    elif trajectory_type == "figure8":
        # Figure-8 motion (highly nonlinear!)
        omega = 0.4  # rad/s
        states[:, 0] = 10 * np.sin(omega * t)  # x
        states[:, 1] = 5 * np.sin(2 * omega * t)  # y
        states[:, 2] = 10 * omega * np.cos(omega * t)  # vx
        states[:, 3] = 10 * omega * np.cos(2 * omega * t)  # vy

    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    return t, states


def generate_range_measurements(
    states: np.ndarray,
    beacons: np.ndarray,
    noise_std: float = 0.5,
    outlier_rate: float = 0.0,
    outlier_scale: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate range measurements from beacons.

    Args:
        states: True states [N×4] (x, y, vx, vy).
        beacons: Beacon positions [M×2] (x, y).
        noise_std: Measurement noise std dev (m).
        outlier_rate: Fraction of measurements that are outliers.
        outlier_scale: Scale of outlier noise relative to normal noise.
        seed: Random seed.

    Returns:
        Range measurements [N×M] in meters.
    """
    rng = np.random.default_rng(seed)
    N = len(states)
    M = len(beacons)

    ranges = np.zeros((N, M))

    for i in range(N):
        pos = states[i, :2]
        for j in range(M):
            # True range
            true_range = np.linalg.norm(pos - beacons[j])

            # Add noise
            if rng.random() < outlier_rate:
                # Outlier
                noise = rng.normal(0, noise_std * outlier_scale)
            else:
                # Normal measurement
                noise = rng.normal(0, noise_std)

            ranges[i, j] = true_range + noise

    return ranges


def generate_range_bearing_measurements(
    states: np.ndarray,
    beacons: np.ndarray,
    range_noise_std: float = 0.5,
    bearing_noise_std_deg: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate range and bearing measurements.

    Args:
        states: True states [N×4].
        beacons: Beacon positions [M×2].
        range_noise_std: Range noise std dev (m).
        bearing_noise_std_deg: Bearing noise std dev (degrees).
        seed: Random seed.

    Returns:
        Tuple of (ranges [N×M], bearings [N×M]) in meters and radians.
    """
    rng = np.random.default_rng(seed)
    N = len(states)
    M = len(beacons)

    ranges = np.zeros((N, M))
    bearings = np.zeros((N, M))

    bearing_noise_rad = np.deg2rad(bearing_noise_std_deg)

    for i in range(N):
        pos = states[i, :2]
        for j in range(M):
            # True range and bearing
            diff = beacons[j] - pos
            true_range = np.linalg.norm(diff)
            true_bearing = np.arctan2(diff[1], diff[0])

            # Add noise
            ranges[i, j] = true_range + rng.normal(0, range_noise_std)
            bearings[i, j] = true_bearing + rng.normal(0, bearing_noise_rad)

    return ranges, bearings


def save_dataset(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    beacons: np.ndarray,
    ranges: np.ndarray,
    bearings: np.ndarray,
    config: Dict,
) -> None:
    """Save estimator comparison dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save time vector
    np.savetxt(
        output_dir / "time.txt",
        times,
        fmt="%.3f",
        header="time (s)",
    )

    # Save ground truth states
    np.savetxt(
        output_dir / "ground_truth_states.txt",
        states,
        fmt="%.6f",
        header="x (m), y (m), vx (m/s), vy (m/s)",
    )

    # Save beacons
    np.savetxt(
        output_dir / "beacons.txt",
        beacons,
        fmt="%.6f",
        header="x (m), y (m)",
    )

    # Save measurements
    np.savetxt(
        output_dir / "range_measurements.txt",
        ranges,
        fmt="%.6f",
        header=f"ranges to {len(beacons)} beacons (m)",
    )
    np.savetxt(
        output_dir / "bearing_measurements.txt",
        bearings,
        fmt="%.6f",
        header=f"bearings to {len(beacons)} beacons (rad)",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print(f"    Files: 6 files (time, states, beacons, ranges, bearings, config)")
    print(f"    Duration: {times[-1]:.1f}s")
    print(f"    Samples: {len(times)}")
    print(f"    Beacons: {len(beacons)}")


def generate_dataset(
    output_dir: str,
    preset: str = None,
    trajectory: str = "circular",
    duration: float = 30.0,
    dt: float = 0.1,
    n_beacons: int = 4,
    range_noise: float = 0.5,
    bearing_noise_deg: float = 5.0,
    outlier_rate: float = 0.0,
    seed: int = 42,
) -> None:
    """
    Generate estimator comparison dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        trajectory: Trajectory type.
        duration: Duration (s).
        dt: Time step (s).
        n_beacons: Number of beacons.
        range_noise: Range noise std dev (m).
        bearing_noise_deg: Bearing noise std dev (deg).
        outlier_rate: Fraction of outlier measurements.
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "linear":
        trajectory = "linear"
        range_noise = 0.3
        bearing_noise_deg = 3.0
        outlier_rate = 0.0
        output_dir = "data/sim/ch3_estimator_linear"
    elif preset == "nonlinear":
        trajectory = "circular"
        range_noise = 0.5
        bearing_noise_deg = 5.0
        outlier_rate = 0.0
        output_dir = "data/sim/ch3_estimator_nonlinear"
    elif preset == "high_nonlinearity":
        trajectory = "figure8"
        range_noise = 0.5
        bearing_noise_deg = 5.0
        outlier_rate = 0.0
        output_dir = "data/sim/ch3_estimator_high_nonlinear"
    elif preset == "outliers":
        trajectory = "circular"
        range_noise = 0.5
        bearing_noise_deg = 5.0
        outlier_rate = 0.1
        output_dir = "data/sim/ch3_estimator_outliers"

    print("\n" + "=" * 70)
    print(f"Generating Ch3 Estimator Comparison Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Generate trajectory
    print("\nStep 1: Generating trajectory...")
    times, states = generate_trajectory(trajectory, duration, dt, seed)
    print(f"  Trajectory: {trajectory}")
    print(f"  Duration: {duration}s")
    print(f"  Time step: {dt}s")
    print(f"  Samples: {len(times)}")

    # Generate beacons
    print("\nStep 2: Placing beacons...")
    if n_beacons == 4:
        # Square arrangement
        beacons = np.array([
            [-15, -15],
            [15, -15],
            [15, 15],
            [-15, 15]
        ], dtype=float)
    elif n_beacons == 8:
        # Octagon arrangement
        angles = np.linspace(0, 2*np.pi, n_beacons, endpoint=False)
        radius = 20.0
        beacons = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # Random placement
        rng = np.random.default_rng(seed)
        beacons = rng.uniform(-20, 20, (n_beacons, 2))

    print(f"  Beacons: {n_beacons}")
    print(f"  Configuration: {'square' if n_beacons == 4 else 'octagon' if n_beacons == 8 else 'random'}")

    # Generate measurements
    print("\nStep 3: Generating measurements...")
    ranges = generate_range_measurements(
        states, beacons, range_noise, outlier_rate, outlier_scale=5.0, seed=seed
    )
    bearings_ranges, bearings = generate_range_bearing_measurements(
        states, beacons, range_noise, bearing_noise_deg, seed=seed + 1
    )

    print(f"  Range noise: {range_noise}m")
    print(f"  Bearing noise: {bearing_noise_deg}deg")
    print(f"  Outlier rate: {outlier_rate*100:.1f}%")

    # Compute measurement statistics
    true_ranges = np.array([[np.linalg.norm(states[i, :2] - beacons[j]) 
                             for j in range(len(beacons))] 
                            for i in range(len(states))])
    range_errors = ranges - true_ranges
    range_rmse = np.sqrt(np.mean(range_errors**2))

    print(f"\nMeasurement Statistics:")
    print(f"  Range RMSE: {range_rmse:.3f}m")
    print(f"  Range bias: {range_errors.mean():.3f}m")

    # Save dataset
    config = {
        "dataset": "ch3_estimator_comparison",
        "preset": preset,
        "trajectory": {
            "type": trajectory,
            "duration_s": duration,
            "dt_s": dt,
            "num_samples": len(times),
        },
        "beacons": {
            "count": n_beacons,
            "positions": beacons.tolist(),
        },
        "measurements": {
            "range_noise_std_m": range_noise,
            "bearing_noise_std_deg": bearing_noise_deg,
            "outlier_rate": outlier_rate,
        },
        "statistics": {
            "range_rmse_m": float(range_rmse),
            "range_bias_m": float(range_errors.mean()),
        },
        "equations": [
            "3.11-3.19 (KF)",
            "3.21 (EKF)",
            "3.24-3.30 (UKF)",
            "3.32-3.34 (PF)"
        ],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        times,
        states,
        beacons,
        ranges,
        bearings,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch3 Estimator Comparison Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  linear              Linear motion (KF is optimal)
  nonlinear           Circular motion (EKF/UKF/PF needed)
  high_nonlinearity   Figure-8 motion (UKF/PF better than EKF)
  outliers            Circular + 10%% outliers (PF most robust)

Examples:
  # Generate all presets for comparison
  python scripts/generate_ch3_estimator_comparison_dataset.py --preset linear
  python scripts/generate_ch3_estimator_comparison_dataset.py --preset nonlinear
  python scripts/generate_ch3_estimator_comparison_dataset.py --preset high_nonlinearity
  python scripts/generate_ch3_estimator_comparison_dataset.py --preset outliers

  # Custom configuration
  python scripts/generate_ch3_estimator_comparison_dataset.py \\
      --output data/sim/my_estimator \\
      --trajectory figure8 \\
      --range-noise 1.0

Learning Focus:
  - KF optimal for LINEAR systems with Gaussian noise
  - EKF handles MODERATE nonlinearity (circular motion)
  - UKF handles HIGH nonlinearity better than EKF (figure-8)
  - PF handles NON-GAUSSIAN noise and outliers

Book Reference: Chapter 3, Sections 3.2-3.4
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["linear", "nonlinear", "high_nonlinearity", "outliers"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/ch3_estimator_nonlinear",
        help="Output directory (default: data/sim/ch3_estimator_nonlinear)",
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--trajectory",
        type=str,
        choices=["linear", "circular", "figure8"],
        default="circular",
        help="Trajectory type (default: circular)",
    )
    traj_group.add_argument(
        "--duration", type=float, default=30.0, help="Duration in seconds (default: 30.0)"
    )
    traj_group.add_argument(
        "--dt", type=float, default=0.1, help="Time step in seconds (default: 0.1)"
    )

    # Environment parameters
    env_group = parser.add_argument_group("Environment Parameters")
    env_group.add_argument(
        "--n-beacons", type=int, default=4, help="Number of beacons (default: 4)"
    )

    # Noise parameters
    noise_group = parser.add_argument_group("Noise Parameters")
    noise_group.add_argument(
        "--range-noise", type=float, default=0.5, help="Range noise std (m) (default: 0.5)"
    )
    noise_group.add_argument(
        "--bearing-noise", type=float, default=5.0, help="Bearing noise std (deg) (default: 5.0)"
    )
    noise_group.add_argument(
        "--outlier-rate", type=float, default=0.0, help="Outlier rate 0-1 (default: 0.0)"
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        trajectory=args.trajectory,
        duration=args.duration,
        dt=args.dt,
        n_beacons=args.n_beacons,
        range_noise=args.range_noise,
        bearing_noise_deg=args.bearing_noise,
        outlier_rate=args.outlier_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

