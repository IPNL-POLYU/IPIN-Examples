"""
Generate Ch7 SLAM 2D Dataset.

This script generates synthetic 2D LiDAR SLAM datasets demonstrating pose graph
optimization with loop closures. Shows the critical impact of loop closure
detection on trajectory drift correction.

Key Learning Objectives:
    - Understand odometry drift accumulation over time
    - Learn loop closure detection and verification
    - Study pose graph optimization (factor graphs)
    - Compare: dead-reckoning vs SLAM (10× error reduction!)

Implements Equations:
    - Eqs. (7.10-7.11): ICP scan matching
    - Section 7.3: Pose graph construction and optimization
    - Factor graph with odometry + loop closure constraints

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slam import se2_compose, se2_relative


def generate_trajectory(
    trajectory_type: str = "square",
    size: float = 20.0,
    n_poses_per_side: int = 10,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate 2D robot trajectory.

    Args:
        trajectory_type: Type of trajectory ('square', 'figure8', 'random_walk').
        size: Characteristic size in meters.
        n_poses_per_side: Number of poses per segment.
        seed: Random seed.

    Returns:
        List of poses [x, y, yaw] in meters and radians.
    """
    rng = np.random.default_rng(seed)
    poses = []

    if trajectory_type == "square":
        # Closed loop (important for loop closure!)
        poses.append(np.array([0.0, 0.0, 0.0]))

        # Side 1: East
        for i in range(1, n_poses_per_side):
            x = (i / n_poses_per_side) * size
            poses.append(np.array([x, 0.0, 0.0]))

        # Corner 1 & Side 2: North
        poses.append(np.array([size, 0.0, np.pi / 2]))
        for i in range(1, n_poses_per_side):
            y = (i / n_poses_per_side) * size
            poses.append(np.array([size, y, np.pi / 2]))

        # Corner 2 & Side 3: West
        poses.append(np.array([size, size, np.pi]))
        for i in range(1, n_poses_per_side):
            x = size - (i / n_poses_per_side) * size
            poses.append(np.array([x, size, np.pi]))

        # Corner 3 & Side 4: South
        poses.append(np.array([0.0, size, -np.pi / 2]))
        for i in range(1, n_poses_per_side):
            y = size - (i / n_poses_per_side) * size
            poses.append(np.array([0.0, y, -np.pi / 2]))

        # Close loop
        poses.append(np.array([0.0, 0.0, 0.0]))

    elif trajectory_type == "figure8":
        # Figure-8 trajectory (two loops, harder!)
        n_total = n_poses_per_side * 4
        for i in range(n_total):
            t = 2 * np.pi * i / n_total
            x = size/2 * np.sin(t)
            y = size/4 * np.sin(2 * t)
            yaw = np.arctan2(size/2 * np.cos(2 * t), size/2 * np.cos(t))
            poses.append(np.array([x, y, yaw]))

    elif trajectory_type == "random_walk":
        # Random walk (no guaranteed loop closure)
        pose = np.array([0.0, 0.0, 0.0])
        poses.append(pose.copy())

        for _ in range(n_poses_per_side * 4):
            # Random forward/turn
            forward = rng.uniform(0.5, 2.0)
            turn = rng.uniform(-np.pi/6, np.pi/6)

            # Update pose
            pose[2] += turn
            pose[0] += forward * np.cos(pose[2])
            pose[1] += forward * np.sin(pose[2])
            poses.append(pose.copy())

    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    return poses


def generate_landmarks(
    trajectory: List[np.ndarray],
    n_landmarks: int = 50,
    area_margin: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate static landmarks around trajectory.

    Args:
        trajectory: List of poses.
        n_landmarks: Number of landmarks.
        area_margin: Margin around trajectory bounding box (m).
        seed: Random seed.

    Returns:
        Landmark positions [N×2] in meters.
    """
    rng = np.random.default_rng(seed)

    # Find trajectory bounding box
    positions = np.array([[p[0], p[1]] for p in trajectory])
    x_min, y_min = positions.min(axis=0) - area_margin
    x_max, y_max = positions.max(axis=0) + area_margin

    # Generate random landmarks in area
    landmarks = rng.uniform(
        [x_min, y_min],
        [x_max, y_max],
        (n_landmarks, 2)
    )

    return landmarks


def generate_scan_from_pose(
    pose: np.ndarray,
    landmarks: np.ndarray,
    max_range: float = 15.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic 2D LiDAR scan from pose.

    Args:
        pose: Robot pose [x, y, yaw].
        landmarks: Landmark positions [N×2] in global frame.
        max_range: Maximum sensor range (m).
        noise_std: Range noise std dev (m).
        seed: Random seed.

    Returns:
        Point cloud in robot local frame [M×2].
    """
    rng = np.random.default_rng(seed)

    x, y, yaw = pose
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Transform landmarks to robot frame
    diff = landmarks - np.array([x, y])
    x_local = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
    y_local = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]

    # Compute ranges and filter by max_range
    ranges = np.sqrt(x_local**2 + y_local**2)
    visible_mask = ranges < max_range

    # Keep only visible points
    scan = np.column_stack([x_local[visible_mask], y_local[visible_mask]])

    # Add noise
    if noise_std > 0 and len(scan) > 0:
        scan += rng.normal(0, noise_std, scan.shape)

    return scan


def add_odometry_noise(
    true_poses: List[np.ndarray],
    translation_noise: float = 0.1,
    rotation_noise: float = 0.02,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Add cumulative noise to simulate odometry drift.

    Args:
        true_poses: List of true poses.
        translation_noise: Translation error std dev per step (m).
        rotation_noise: Rotation error std dev per step (rad).
        seed: Random seed.

    Returns:
        List of odometry poses (with cumulative drift).
    """
    rng = np.random.default_rng(seed)
    odom_poses = [true_poses[0].copy()]  # First pose is exact

    for i in range(1, len(true_poses)):
        # Relative motion in true trajectory
        rel_true = se2_relative(true_poses[i - 1], true_poses[i])

        # Add noise to relative motion
        rel_noisy = rel_true.copy()
        rel_noisy[0] += rng.normal(0, translation_noise)
        rel_noisy[1] += rng.normal(0, translation_noise)
        rel_noisy[2] += rng.normal(0, rotation_noise)

        # Compose to get odometry pose (accumulates error!)
        odom_pose = se2_compose(odom_poses[-1], rel_noisy)
        odom_poses.append(odom_pose)

    return odom_poses


def detect_loop_closures(
    poses: List[np.ndarray],
    min_index_diff: int = 10,
    max_distance: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Detect potential loop closures based on distance.

    Args:
        poses: List of poses.
        min_index_diff: Minimum index separation for loop closure.
        max_distance: Maximum distance for loop closure candidate (m).

    Returns:
        List of (i, j) index pairs for loop closures.
    """
    loop_closures = []

    for i in range(len(poses)):
        for j in range(i + min_index_diff, len(poses)):
            # Check distance between poses
            dist = np.linalg.norm(poses[i][:2] - poses[j][:2])

            if dist < max_distance:
                loop_closures.append((i, j))

    return loop_closures


def save_dataset(
    output_dir: Path,
    true_poses: List[np.ndarray],
    odom_poses: List[np.ndarray],
    scans: List[np.ndarray],
    landmarks: np.ndarray,
    loop_closures: List[Tuple[int, int]],
    config: Dict,
) -> None:
    """Save SLAM dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert poses to arrays
    true_poses_arr = np.array(true_poses)
    odom_poses_arr = np.array(odom_poses)

    # Save ground truth poses
    np.savetxt(
        output_dir / "ground_truth_poses.txt",
        true_poses_arr,
        fmt="%.6f",
        header="x (m), y (m), yaw (rad)",
    )

    # Save odometry poses
    np.savetxt(
        output_dir / "odometry_poses.txt",
        odom_poses_arr,
        fmt="%.6f",
        header="x (m), y (m), yaw (rad) - with cumulative drift",
    )

    # Save landmarks
    np.savetxt(
        output_dir / "landmarks.txt",
        landmarks,
        fmt="%.6f",
        header="x (m), y (m)",
    )

    # Save scans (as compressed numpy)
    np.savez_compressed(
        output_dir / "scans.npz",
        **{f"scan_{i}": scan for i, scan in enumerate(scans)}
    )

    # Save loop closures
    if loop_closures:
        loop_closures_arr = np.array(loop_closures)
        np.savetxt(
            output_dir / "loop_closures.txt",
            loop_closures_arr,
            fmt="%d",
            header="pose_i, pose_j (index pairs for loop closures)",
        )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Compute statistics
    drift_error = np.linalg.norm(odom_poses_arr[-1, :2] - true_poses_arr[-1, :2])

    print(f"\n  Saved dataset to: {output_dir}")
    print(f"    Files: 6 files (truth, odom, landmarks, scans, loop_closures, config)")
    print(f"    Poses: {len(true_poses)}")
    print(f"    Landmarks: {len(landmarks)}")
    print(f"    Loop closures: {len(loop_closures)}")
    print(f"    Final drift: {drift_error:.2f}m")


def generate_dataset(
    output_dir: str,
    preset: str = None,
    trajectory: str = "square",
    size: float = 20.0,
    n_poses_per_side: int = 10,
    n_landmarks: int = 50,
    max_range: float = 15.0,
    translation_noise: float = 0.1,
    rotation_noise: float = 0.02,
    scan_noise: float = 0.05,
    seed: int = 42,
) -> None:
    """
    Generate SLAM 2D dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        trajectory: Trajectory type.
        size: Trajectory size (m).
        n_poses_per_side: Number of poses per segment.
        n_landmarks: Number of landmarks.
        max_range: Sensor max range (m).
        translation_noise: Odometry translation noise (m).
        rotation_noise: Odometry rotation noise (rad).
        scan_noise: Scan range noise (m).
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "baseline":
        trajectory = "square"
        size = 20.0
        n_poses_per_side = 10
        translation_noise = 0.1
        rotation_noise = 0.02
        output_dir = "data/sim/ch7_slam_2d_square"
    elif preset == "low_drift":
        trajectory = "square"
        size = 20.0
        n_poses_per_side = 10
        translation_noise = 0.02
        rotation_noise = 0.005
        output_dir = "data/sim/ch7_slam_2d_low_drift"
    elif preset == "high_drift":
        trajectory = "square"
        size = 20.0
        n_poses_per_side = 10
        translation_noise = 0.3
        rotation_noise = 0.05
        output_dir = "data/sim/ch7_slam_2d_high_drift"
    elif preset == "figure8":
        trajectory = "figure8"
        size = 15.0
        n_poses_per_side = 15
        translation_noise = 0.1
        rotation_noise = 0.02
        output_dir = "data/sim/ch7_slam_2d_figure8"

    print("\n" + "=" * 70)
    print(f"Generating Ch7 SLAM 2D Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Generate trajectory
    print("\nStep 1: Generating trajectory...")
    true_poses = generate_trajectory(trajectory, size, n_poses_per_side, seed)
    print(f"  Trajectory: {trajectory}")
    print(f"  Poses: {len(true_poses)}")
    print(f"  Size: {size}m")

    # Generate landmarks
    print("\nStep 2: Generating landmarks...")
    landmarks = generate_landmarks(true_poses, n_landmarks, area_margin=5.0, seed=seed)
    print(f"  Landmarks: {len(landmarks)}")

    # Generate scans
    print("\nStep 3: Generating LiDAR scans...")
    scans = []
    for i, pose in enumerate(true_poses):
        scan = generate_scan_from_pose(
            pose, landmarks, max_range, scan_noise, seed=seed + i
        )
        scans.append(scan)
    avg_points = np.mean([len(s) for s in scans])
    print(f"  Max range: {max_range}m")
    print(f"  Scan noise: {scan_noise}m")
    print(f"  Avg points per scan: {avg_points:.1f}")

    # Add odometry noise
    print("\nStep 4: Adding odometry drift...")
    odom_poses = add_odometry_noise(
        true_poses, translation_noise, rotation_noise, seed
    )
    drift_error = np.linalg.norm(
        np.array(odom_poses[-1][:2]) - np.array(true_poses[-1][:2])
    )
    print(f"  Translation noise: {translation_noise}m per step")
    print(f"  Rotation noise: {rotation_noise:.4f}rad per step")
    print(f"  Final drift: {drift_error:.2f}m")

    # Detect loop closures
    print("\nStep 5: Detecting loop closures...")
    loop_closures = detect_loop_closures(true_poses, min_index_diff=15, max_distance=2.0)
    print(f"  Loop closures detected: {len(loop_closures)}")
    if loop_closures:
        print(f"  Examples: {loop_closures[:3]}")

    # Save dataset
    config = {
        "dataset": "ch7_slam_2d",
        "preset": preset,
        "trajectory": {
            "type": trajectory,
            "size_m": size,
            "n_poses_per_side": n_poses_per_side,
            "total_poses": len(true_poses),
        },
        "landmarks": {
            "count": len(landmarks),
        },
        "sensor": {
            "max_range_m": max_range,
            "scan_noise_std_m": scan_noise,
        },
        "odometry": {
            "translation_noise_std_m": translation_noise,
            "rotation_noise_std_rad": rotation_noise,
            "final_drift_m": float(drift_error),
        },
        "loop_closures": {
            "count": len(loop_closures),
            "min_index_diff": 15,
            "max_distance_m": 2.0,
        },
        "equations": ["7.10-7.11 (ICP)", "Section 7.3 (Pose graph)"],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        true_poses,
        odom_poses,
        scans,
        landmarks,
        loop_closures,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch7 SLAM 2D Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline      Square loop, moderate drift (0.1m, 0.02rad)
  low_drift     Square loop, low drift (0.02m, 0.005rad)
  high_drift    Square loop, high drift (0.3m, 0.05rad)
  figure8       Figure-8 trajectory, moderate drift

Examples:
  # Generate baseline dataset
  python scripts/generate_ch7_slam_2d_dataset.py --preset baseline

  # Generate with custom parameters
  python scripts/generate_ch7_slam_2d_dataset.py \\
      --output data/sim/my_slam \\
      --trajectory figure8 \\
      --translation-noise 0.2

  # Generate all presets
  python scripts/generate_ch7_slam_2d_dataset.py --preset baseline
  python scripts/generate_ch7_slam_2d_dataset.py --preset low_drift
  python scripts/generate_ch7_slam_2d_dataset.py --preset high_drift
  python scripts/generate_ch7_slam_2d_dataset.py --preset figure8

Learning Focus:
  - Loop closure CRITICAL for correcting drift (10× error reduction!)
  - Odometry drift accumulates linearly with trajectory length
  - ICP scan matching enables loop closure detection
  - Pose graph optimization distributes error globally

Book Reference: Chapter 7, Sections 7.2-7.3
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "low_drift", "high_drift", "figure8"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sim/ch7_slam_2d_square",
        help="Output directory (default: data/sim/ch7_slam_2d_square)",
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--trajectory",
        type=str,
        choices=["square", "figure8", "random_walk"],
        default="square",
        help="Trajectory type (default: square)",
    )
    traj_group.add_argument(
        "--size", type=float, default=20.0, help="Trajectory size in meters (default: 20.0)"
    )
    traj_group.add_argument(
        "--n-poses-per-side", type=int, default=10, help="Poses per segment (default: 10)"
    )

    # Environment parameters
    env_group = parser.add_argument_group("Environment Parameters")
    env_group.add_argument(
        "--n-landmarks", type=int, default=50, help="Number of landmarks (default: 50)"
    )
    env_group.add_argument(
        "--max-range", type=float, default=15.0, help="Sensor max range in meters (default: 15.0)"
    )

    # Noise parameters
    noise_group = parser.add_argument_group("Noise Parameters")
    noise_group.add_argument(
        "--translation-noise", type=float, default=0.1, help="Odometry translation noise std (m) (default: 0.1)"
    )
    noise_group.add_argument(
        "--rotation-noise", type=float, default=0.02, help="Odometry rotation noise std (rad) (default: 0.02)"
    )
    noise_group.add_argument(
        "--scan-noise", type=float, default=0.05, help="Scan range noise std (m) (default: 0.05)"
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        trajectory=args.trajectory,
        size=args.size,
        n_poses_per_side=args.n_poses_per_side,
        n_landmarks=args.n_landmarks,
        max_range=args.max_range,
        translation_noise=args.translation_noise,
        rotation_noise=args.rotation_noise,
        scan_noise=args.scan_noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

