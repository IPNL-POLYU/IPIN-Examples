"""Complete 2D Pose Graph SLAM Example with ICP/NDT.

This example demonstrates the full SLAM pipeline:
    1. Generate synthetic robot trajectory (square loop)
    2. Simulate LiDAR scans at each pose
    3. Run scan matching (ICP) to estimate relative poses
    4. Build pose graph with odometry and loop closures
    5. Optimize pose graph to correct drift
    6. Visualize results

Can run with:
    - Pre-generated dataset: python example_pose_graph_slam.py --data ch7_slam_2d_square
    - Inline data (default): python example_pose_graph_slam.py
    - High drift scenario: python example_pose_graph_slam.py --data ch7_slam_2d_high_drift

This implements the pose graph SLAM approach from Section 7.3 of Chapter 7.

Usage:
    python -m ch7_slam.example_pose_graph_slam

Author: Navigation Engineer
Date: 2024
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

from core.slam import (
    se2_apply,
    se2_compose,
    se2_relative,
    icp_point_to_point,
    create_pose_graph,
)


def load_slam_dataset(data_dir: str) -> Dict:
    """Load SLAM dataset from directory.
    
    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch7_slam_2d_square')
    
    Returns:
        Dictionary with poses, landmarks, loop closures, and scans
    """
    path = Path(data_dir)
    
    data = {
        'true_poses': np.loadtxt(path / 'ground_truth_poses.txt'),
        'odom_poses': np.loadtxt(path / 'odometry_poses.txt'),
        'landmarks': np.loadtxt(path / 'landmarks.txt'),
        'loop_closures': np.loadtxt(path / 'loop_closures.txt'),
    }
    
    # Load scans from npz
    scans_npz = np.load(path / 'scans.npz')
    data['scans'] = [scans_npz[f'scan_{i}'] for i in range(len(data['true_poses']))]
    
    # Load config
    with open(path / 'config.json') as f:
        data['config'] = json.load(f)
    
    return data


def run_with_dataset(data_dir: str) -> None:
    """Run pose graph SLAM using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
    """
    print("=" * 70)
    print("CHAPTER 7: 2D POSE GRAPH SLAM EXAMPLE")
    print(f"Using dataset: {data_dir}")
    print("=" * 70)
    print()
    
    # Load dataset
    data = load_slam_dataset(data_dir)
    config = data['config']
    
    true_poses = [data['true_poses'][i] for i in range(len(data['true_poses']))]
    odom_poses = [data['odom_poses'][i] for i in range(len(data['odom_poses']))]
    landmarks = data['landmarks']
    scans = data['scans']
    loop_closure_data = data['loop_closures']
    
    n_poses = len(true_poses)
    
    print(f"Dataset Info:")
    print(f"  Trajectory: {config.get('trajectory', {}).get('type', 'unknown')}")
    print(f"  Poses: {n_poses}")
    print(f"  Landmarks: {len(landmarks)}")
    print(f"  Loop closures: {len(loop_closure_data)}")
    
    # Compute initial drift
    initial_drift = np.linalg.norm(odom_poses[0][:2] - true_poses[0][:2])
    final_drift = np.linalg.norm(odom_poses[-1][:2] - true_poses[-1][:2])
    print(f"\n  Initial drift: {initial_drift:.3f} m")
    print(f"  Final drift (without SLAM): {final_drift:.3f} m")
    
    # Prepare loop closures
    print("\n" + "-" * 70)
    print("Loop Closures from Dataset:")
    loop_closures = []
    for lc in loop_closure_data:
        i, j = int(lc[0]), int(lc[1])
        rel_pose = lc[2:5]
        cov = np.diag([0.05, 0.05, 0.01])
        loop_closures.append((i, j, rel_pose, cov))
        print(f"  {i} ↔ {j}: rel_pose=[{rel_pose[0]:.3f}, {rel_pose[1]:.3f}, {np.rad2deg(rel_pose[2]):.1f}°]")
    
    # Build pose graph
    print("\n" + "-" * 70)
    print("Building pose graph...")
    
    # Prepare odometry measurements
    odometry_measurements = []
    for i in range(n_poses - 1):
        rel_pose = se2_relative(np.array(true_poses[i]), np.array(true_poses[i + 1]))
        rel_pose[0] += np.random.normal(0, 0.05)
        rel_pose[1] += np.random.normal(0, 0.05)
        rel_pose[2] += np.random.normal(0, 0.01)
        odometry_measurements.append((i, i + 1, rel_pose))
    
    # Prepare loop closure measurements
    loop_measurements = [(i, j, rel_pose) for i, j, rel_pose, _ in loop_closures]
    
    odom_info = np.linalg.inv(np.diag([0.1, 0.1, 0.02]))
    loop_info = np.linalg.inv(np.diag([0.05, 0.05, 0.01]))
    
    graph = create_pose_graph(
        poses=odom_poses,
        odometry_measurements=odometry_measurements,
        loop_closures=loop_measurements if loop_measurements else None,
        odometry_information=odom_info,
        loop_information=loop_info,
    )
    
    print(f"  Pose graph: {len(graph.variables)} variables, {len(graph.factors)} factors")
    
    # Optimize
    print("\n" + "-" * 70)
    print("Optimizing pose graph...")
    
    initial_error = graph.compute_error()
    print(f"  Initial error: {initial_error:.6f}")
    
    optimized_vars, error_history = graph.optimize(
        method="gauss_newton", max_iterations=50, tol=1e-6
    )
    
    final_error = error_history[-1]
    print(f"  Final error: {final_error:.6f}")
    print(f"  Iterations: {len(error_history) - 1}")
    print(f"  Error reduction: {(1 - final_error / initial_error) * 100:.2f}%")
    
    optimized_poses = [optimized_vars[i] for i in range(n_poses)]
    
    # Evaluate
    print("\n" + "-" * 70)
    print("Results:")
    
    odom_errors = np.array([np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)])
    opt_errors = np.array([np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)])
    
    odom_rmse = np.sqrt(np.mean(odom_errors**2))
    opt_rmse = np.sqrt(np.mean(opt_errors**2))
    
    print(f"  Odometry RMSE: {odom_rmse:.4f} m")
    print(f"  Optimized RMSE: {opt_rmse:.4f} m")
    print(f"  Improvement: {(1 - opt_rmse / odom_rmse) * 100:.2f}%")
    
    final_loop_error = np.linalg.norm(optimized_poses[-1][:2] - optimized_poses[0][:2])
    print(f"  Final loop closure error: {final_loop_error:.4f} m")
    
    # Visualize
    print("\n" + "-" * 70)
    print("Generating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], c="gray", marker="x", s=30, alpha=0.3, label="Landmarks")
    
    true_xy = np.array([[p[0], p[1]] for p in true_poses])
    ax1.plot(true_xy[:, 0], true_xy[:, 1], "g-", linewidth=2, label="Ground Truth", alpha=0.7)
    ax1.scatter(true_xy[0, 0], true_xy[0, 1], c="green", marker="o", s=100, zorder=5)
    
    odom_xy = np.array([[p[0], p[1]] for p in odom_poses])
    ax1.plot(odom_xy[:, 0], odom_xy[:, 1], "r--", linewidth=2, label="Odometry (Drift)", alpha=0.7)
    
    opt_xy = np.array([[p[0], p[1]] for p in optimized_poses])
    ax1.plot(opt_xy[:, 0], opt_xy[:, 1], "b-", linewidth=2, label="Optimized (SLAM)", alpha=0.8)
    ax1.scatter(opt_xy[0, 0], opt_xy[0, 1], c="blue", marker="o", s=100, zorder=5)
    
    for i, j, _, _ in loop_closures:
        ax1.plot([odom_xy[i, 0], odom_xy[j, 0]], [odom_xy[i, 1], odom_xy[j, 1]], "m:", linewidth=1, alpha=0.5)
    
    ax1.set_xlabel("X [m]", fontsize=12)
    ax1.set_ylabel("Y [m]", fontsize=12)
    ax1.set_title("2D Pose Graph SLAM: Trajectories", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    
    ax2 = axes[1]
    timesteps = np.arange(n_poses)
    ax2.plot(timesteps, odom_errors, "r--", linewidth=2, label="Odometry Error", alpha=0.7)
    ax2.plot(timesteps, opt_errors, "b-", linewidth=2, label="Optimized Error", alpha=0.8)
    
    for i, j, _, _ in loop_closures:
        ax2.axvline(j, color="magenta", linestyle=":", alpha=0.5, linewidth=1)
    
    ax2.set_xlabel("Pose Index", fontsize=12)
    ax2.set_ylabel("Position Error [m]", fontsize=12)
    ax2.set_title("Position Error Over Time", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figs_dir = Path("ch7_slam/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_file = figs_dir / "pose_graph_slam_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved figure: {output_file}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("SLAM PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  • Trajectory: {n_poses} poses")
    print(f"  • Loop closures: {len(loop_closures)}")
    print(f"  • Odometry drift: {final_drift:.3f} m")
    print(f"  • SLAM accuracy: {opt_rmse:.4f} m RMSE")
    print(f"  • Improvement: {(1 - opt_rmse / odom_rmse) * 100:.1f}%")
    print()


def generate_square_trajectory(
    side_length: float = 10.0, n_poses_per_side: int = 5
) -> List[np.ndarray]:
    """
    Generate a square trajectory for SLAM demonstration.

    Args:
        side_length: Length of each side in meters.
        n_poses_per_side: Number of poses per side (excluding corners).

    Returns:
        List of poses [x, y, yaw] representing the trajectory.

    Notes:
        - Starts at origin facing East
        - Goes: East → North → West → South → returns to start
        - Creates a closed loop for loop closure detection
    """
    poses = []

    # Starting pose at origin
    poses.append(np.array([0.0, 0.0, 0.0]))

    # Side 1: Move East
    for i in range(1, n_poses_per_side):
        x = (i / n_poses_per_side) * side_length
        poses.append(np.array([x, 0.0, 0.0]))

    # Corner 1: Turn North
    poses.append(np.array([side_length, 0.0, np.pi / 2]))

    # Side 2: Move North
    for i in range(1, n_poses_per_side):
        y = (i / n_poses_per_side) * side_length
        poses.append(np.array([side_length, y, np.pi / 2]))

    # Corner 2: Turn West
    poses.append(np.array([side_length, side_length, np.pi]))

    # Side 3: Move West
    for i in range(1, n_poses_per_side):
        x = side_length - (i / n_poses_per_side) * side_length
        poses.append(np.array([x, side_length, np.pi]))

    # Corner 3: Turn South
    poses.append(np.array([0.0, side_length, -np.pi / 2]))

    # Side 4: Move South
    for i in range(1, n_poses_per_side):
        y = side_length - (i / n_poses_per_side) * side_length
        poses.append(np.array([0.0, y, -np.pi / 2]))

    # Return to start (close loop)
    poses.append(np.array([0.0, 0.0, 0.0]))

    return poses


def generate_scan_from_pose(
    pose: np.ndarray,
    landmarks: np.ndarray,
    max_range: float = 15.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate synthetic 2D LiDAR scan from a robot pose.

    Simulates range-bearing sensor by projecting landmarks into robot frame.

    Args:
        pose: Robot pose [x, y, yaw].
        landmarks: Landmark positions in global frame, shape (N, 2).
        max_range: Maximum sensor range in meters.
        noise_std: Standard deviation of range noise in meters.

    Returns:
        Point cloud in robot local frame, shape (M, 2) where M <= N.
    """
    # Transform landmarks to robot frame
    x, y, yaw = pose
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Translate and rotate landmarks to robot frame
    diff = landmarks - np.array([x, y])
    x_local = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
    y_local = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]

    # Compute ranges
    ranges = np.sqrt(x_local**2 + y_local**2)

    # Filter by max range (only visible landmarks)
    visible_mask = ranges < max_range

    # Keep only visible points
    scan = np.column_stack([x_local[visible_mask], y_local[visible_mask]])

    # Add noise
    if noise_std > 0:
        scan += np.random.normal(0, noise_std, scan.shape)

    return scan


def add_odometry_noise(
    true_poses: List[np.ndarray],
    translation_noise: float = 0.1,
    rotation_noise: float = 0.02,
) -> List[np.ndarray]:
    """
    Add noise to trajectory to simulate odometry drift.

    Args:
        true_poses: List of true poses.
        translation_noise: Standard deviation of translation error per step (m).
        rotation_noise: Standard deviation of rotation error per step (rad).

    Returns:
        List of noisy poses with accumulated drift.
    """
    noisy_poses = [true_poses[0].copy()]  # First pose unchanged

    for i in range(1, len(true_poses)):
        # True relative pose
        true_rel = se2_relative(true_poses[i - 1], true_poses[i])

        # Add noise to relative pose
        noisy_rel = true_rel.copy()
        noisy_rel[0] += np.random.normal(0, translation_noise)
        noisy_rel[1] += np.random.normal(0, translation_noise)
        noisy_rel[2] += np.random.normal(0, rotation_noise)

        # Compose to get noisy absolute pose
        noisy_pose = se2_compose(noisy_poses[-1], noisy_rel)
        noisy_poses.append(noisy_pose)

    return noisy_poses


def detect_loop_closures(
    poses: List[np.ndarray],
    scans: List[np.ndarray],
    distance_threshold: float = 2.0,
    min_time_separation: int = 10,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Detect loop closures using distance threshold and ICP verification.

    Args:
        poses: List of poses (possibly with drift).
        scans: List of scans in local frame.
        distance_threshold: Maximum distance to consider for loop closure (m).
        min_time_separation: Minimum time steps between poses for loop closure.

    Returns:
        List of tuples (pose_i, pose_j, relative_pose, covariance) for each closure.
    """
    loop_closures = []

    n_poses = len(poses)

    for i in range(n_poses):
        for j in range(i + min_time_separation, n_poses):
            # Check distance
            dist = np.linalg.norm(poses[i][:2] - poses[j][:2])

            if dist < distance_threshold:
                # Potential loop closure - verify with ICP
                try:
                    # Initial guess based on current poses
                    initial_guess = se2_relative(poses[i], poses[j])

                    # Run ICP
                    rel_pose, iters, residual, converged = icp_point_to_point(
                        scans[i],
                        scans[j],
                        initial_pose=initial_guess,
                        max_iterations=50,
                        tolerance=1e-4,
                    )

                    if converged and residual < 1.0:
                        # Compute covariance (simplified)
                        cov = np.diag([0.05, 0.05, 0.01])  # Moderate uncertainty

                        loop_closures.append((i, j, rel_pose, cov))

                        print(
                            f"  Loop closure: {i} ↔ {j}, "
                            f"residual={residual:.4f}, "
                            f"iters={iters}"
                        )
                except Exception as e:
                    # ICP failed, skip this pair
                    pass

    return loop_closures


def plot_slam_results(
    true_poses: List[np.ndarray],
    odom_poses: List[np.ndarray],
    optimized_poses: List[np.ndarray],
    landmarks: np.ndarray,
    loop_closures: List[Tuple[int, int, np.ndarray, np.ndarray]],
):
    """
    Visualize SLAM results: ground truth, odometry, and optimized trajectory.

    Args:
        true_poses: Ground truth poses.
        odom_poses: Odometry-only poses (with drift).
        optimized_poses: Optimized poses after pose graph optimization.
        landmarks: Environment landmarks.
        loop_closures: Detected loop closures for visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Trajectories ---
    ax1 = axes[0]

    # Plot landmarks
    ax1.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        c="gray",
        marker="x",
        s=30,
        alpha=0.3,
        label="Landmarks",
    )

    # Plot ground truth
    true_xy = np.array([[p[0], p[1]] for p in true_poses])
    ax1.plot(
        true_xy[:, 0],
        true_xy[:, 1],
        "g-",
        linewidth=2,
        label="Ground Truth",
        alpha=0.7,
    )
    ax1.scatter(true_xy[0, 0], true_xy[0, 1], c="green", marker="o", s=100, zorder=5)

    # Plot odometry (with drift)
    odom_xy = np.array([[p[0], p[1]] for p in odom_poses])
    ax1.plot(
        odom_xy[:, 0],
        odom_xy[:, 1],
        "r--",
        linewidth=2,
        label="Odometry (Drift)",
        alpha=0.7,
    )

    # Plot optimized
    opt_xy = np.array([[p[0], p[1]] for p in optimized_poses])
    ax1.plot(
        opt_xy[:, 0],
        opt_xy[:, 1],
        "b-",
        linewidth=2,
        label="Optimized (SLAM)",
        alpha=0.8,
    )
    ax1.scatter(opt_xy[0, 0], opt_xy[0, 1], c="blue", marker="o", s=100, zorder=5)

    # Plot loop closures
    for i, j, _, _ in loop_closures:
        ax1.plot(
            [odom_xy[i, 0], odom_xy[j, 0]],
            [odom_xy[i, 1], odom_xy[j, 1]],
            "m:",
            linewidth=1,
            alpha=0.5,
        )

    ax1.set_xlabel("X [m]", fontsize=12)
    ax1.set_ylabel("Y [m]", fontsize=12)
    ax1.set_title("2D Pose Graph SLAM: Trajectories", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # --- Plot 2: Position Errors ---
    ax2 = axes[1]

    # Compute position errors
    odom_errors = np.array(
        [np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(len(true_poses))]
    )
    opt_errors = np.array(
        [
            np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2])
            for i in range(len(true_poses))
        ]
    )

    timesteps = np.arange(len(true_poses))

    ax2.plot(
        timesteps, odom_errors, "r--", linewidth=2, label="Odometry Error", alpha=0.7
    )
    ax2.plot(
        timesteps, opt_errors, "b-", linewidth=2, label="Optimized Error", alpha=0.8
    )

    # Mark loop closures
    for i, j, _, _ in loop_closures:
        ax2.axvline(j, color="magenta", linestyle=":", alpha=0.5, linewidth=1)

    ax2.set_xlabel("Pose Index", fontsize=12)
    ax2.set_ylabel("Position Error [m]", fontsize=12)
    ax2.set_title("Position Error Over Time", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ch7_slam/pose_graph_slam_results.png", dpi=150, bbox_inches="tight")
    print("\n[OK] Saved figure: ch7_slam/pose_graph_slam_results.png")
    
    # Only show interactively if not in automated mode
    import os
    if os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") != "Agg":
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass  # Silently skip if display not available


def run_with_inline_data():
    """Run complete pose graph SLAM example with inline data (original behavior)."""
    print("=" * 70)
    print("CHAPTER 7: 2D POSE GRAPH SLAM EXAMPLE")
    print("(Using inline generated data)")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # ------------------------------------------------------------------------
    # 1. Generate Ground Truth Trajectory
    # ------------------------------------------------------------------------
    print("1. Generating square trajectory...")
    true_poses = generate_square_trajectory(side_length=10.0, n_poses_per_side=5)
    n_poses = len(true_poses)
    print(f"   Generated {n_poses} poses in square loop")

    # ------------------------------------------------------------------------
    # 2. Generate Environment (Landmarks for Scan Simulation)
    # ------------------------------------------------------------------------
    print("\n2. Generating environment landmarks...")
    # Create landmarks around the square perimeter
    landmarks = []
    # Landmarks along bottom edge
    landmarks.extend([[i, -1.0] for i in np.linspace(-2, 12, 15)])
    # Landmarks along right edge
    landmarks.extend([[11.0, i] for i in np.linspace(-2, 12, 15)])
    # Landmarks along top edge
    landmarks.extend([[i, 11.0] for i in np.linspace(-2, 12, 15)])
    # Landmarks along left edge
    landmarks.extend([[-1.0, i] for i in np.linspace(-2, 12, 15)])
    landmarks = np.array(landmarks)
    print(f"   Generated {len(landmarks)} landmarks")

    # ------------------------------------------------------------------------
    # 3. Generate Scans at Each Pose
    # ------------------------------------------------------------------------
    print("\n3. Generating LiDAR scans...")
    true_scans = []
    for i, pose in enumerate(true_poses):
        scan = generate_scan_from_pose(pose, landmarks, max_range=8.0, noise_std=0.03)
        true_scans.append(scan)
    print(f"   Generated {n_poses} scans (avg {np.mean([len(s) for s in true_scans]):.1f} points/scan)")

    # ------------------------------------------------------------------------
    # 4. Simulate Odometry with Drift
    # ------------------------------------------------------------------------
    print("\n4. Simulating odometry with drift...")
    odom_poses = add_odometry_noise(
        true_poses, translation_noise=0.08, rotation_noise=0.015
    )

    # Compute initial drift
    initial_drift = np.linalg.norm(odom_poses[0][:2] - true_poses[0][:2])
    final_drift = np.linalg.norm(odom_poses[-1][:2] - true_poses[-1][:2])
    print(f"   Initial drift: {initial_drift:.3f} m")
    print(f"   Final drift (without SLAM): {final_drift:.3f} m")

    # ------------------------------------------------------------------------
    # 5. Detect Loop Closures
    # ------------------------------------------------------------------------
    print("\n5. Detecting loop closures...")
    loop_closures = detect_loop_closures(
        odom_poses, true_scans, distance_threshold=3.0, min_time_separation=10
    )
    print(f"   Detected {len(loop_closures)} loop closures")

    # ------------------------------------------------------------------------
    # 6. Build Pose Graph
    # ------------------------------------------------------------------------
    print("\n6. Building pose graph...")

    # Prepare odometry measurements (consecutive poses)
    odometry_measurements = []
    for i in range(n_poses - 1):
        rel_pose = se2_relative(true_poses[i], true_poses[i + 1])
        # Add odometry noise (simulate sensor noise)
        rel_pose[0] += np.random.normal(0, 0.05)
        rel_pose[1] += np.random.normal(0, 0.05)
        rel_pose[2] += np.random.normal(0, 0.01)
        odometry_measurements.append((i, i + 1, rel_pose))

    # Prepare loop closure measurements
    loop_measurements = []
    for i, j, rel_pose, cov in loop_closures:
        loop_measurements.append((i, j, rel_pose))

    # Create pose graph
    odom_info = np.linalg.inv(np.diag([0.1, 0.1, 0.02]))  # Odometry uncertainty
    loop_info = np.linalg.inv(np.diag([0.05, 0.05, 0.01]))  # ICP uncertainty

    graph = create_pose_graph(
        poses=odom_poses,
        odometry_measurements=odometry_measurements,
        loop_closures=loop_measurements if loop_measurements else None,
        odometry_information=odom_info,
        loop_information=loop_info,
    )

    print(f"   Pose graph: {len(graph.variables)} variables, {len(graph.factors)} factors")
    print(f"   Factors: 1 prior + {len(odometry_measurements)} odometry + {len(loop_measurements)} loop closures")

    # ------------------------------------------------------------------------
    # 7. Optimize Pose Graph
    # ------------------------------------------------------------------------
    print("\n7. Optimizing pose graph...")
    initial_error = graph.compute_error()
    print(f"   Initial error: {initial_error:.6f}")

    optimized_vars, error_history = graph.optimize(
        method="gauss_newton", max_iterations=50, tol=1e-6
    )

    final_error = error_history[-1]
    print(f"   Final error: {final_error:.6f}")
    print(f"   Iterations: {len(error_history) - 1}")
    print(f"   Error reduction: {(1 - final_error / initial_error) * 100:.2f}%")

    # Extract optimized poses
    optimized_poses = [optimized_vars[i] for i in range(n_poses)]

    # ------------------------------------------------------------------------
    # 8. Evaluate Results
    # ------------------------------------------------------------------------
    print("\n8. Evaluating results...")

    # Compute RMSE
    odom_errors = np.array(
        [np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)]
    )
    opt_errors = np.array(
        [np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)]
    )

    odom_rmse = np.sqrt(np.mean(odom_errors**2))
    opt_rmse = np.sqrt(np.mean(opt_errors**2))

    print(f"   Odometry RMSE: {odom_rmse:.4f} m")
    print(f"   Optimized RMSE: {opt_rmse:.4f} m")
    print(f"   Improvement: {(1 - opt_rmse / odom_rmse) * 100:.2f}%")

    # Loop closure error
    final_loop_error = np.linalg.norm(optimized_poses[-1][:2] - optimized_poses[0][:2])
    print(f"   Final loop closure error: {final_loop_error:.4f} m")

    # ------------------------------------------------------------------------
    # 9. Visualize Results
    # ------------------------------------------------------------------------
    print("\n9. Visualizing results...")
    plot_slam_results(true_poses, odom_poses, optimized_poses, landmarks, loop_closures)

    print()
    print("=" * 70)
    print("SLAM PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • Trajectory: {n_poses} poses in square loop")
    print(f"  • Loop closures detected: {len(loop_closures)}")
    print(f"  • Odometry drift: {final_drift:.3f} m")
    print(f"  • SLAM accuracy: {opt_rmse:.4f} m RMSE")
    print(f"  • Improvement: {(1 - opt_rmse / odom_rmse) * 100:.1f}%")
    print()
    print("\nTip: Run with --data ch7_slam_2d_square to use pre-generated dataset")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chapter 7: 2D Pose Graph SLAM Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python example_pose_graph_slam.py
  
  # Run with pre-generated dataset
  python example_pose_graph_slam.py --data ch7_slam_2d_square
  
  # Run with high drift scenario
  python example_pose_graph_slam.py --data ch7_slam_2d_high_drift
        """
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name or path (e.g., 'ch7_slam_2d_square' or full path)"
    )
    
    args = parser.parse_args()
    
    if args.data:
        # Run with dataset
        data_path = Path(args.data)
        if not data_path.exists():
            data_path = Path("data/sim") / args.data
        if not data_path.exists():
            print(f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'")
            print("\nAvailable datasets:")
            sim_dir = Path("data/sim")
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch7"):
                        print(f"  - {d.name}")
            return
        
        run_with_dataset(str(data_path))
    else:
        run_with_inline_data()


if __name__ == "__main__":
    main()

