"""SLAM Front-End Demo: Prediction → Scan-to-Map Alignment → Map Update.

This example demonstrates the explicit SLAM front-end loop:
    1. PREDICTION: Integrate odometry delta to predict pose
    2. CORRECTION: Refine pose via scan-to-map ICP alignment
    3. MAP UPDATE: Add scan to local submap with refined pose

This is a simplified, pedagogical example showing how observation-driven
pose estimation works in SLAM systems.

Usage:
    python -m ch7_slam.example_slam_frontend

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.slam import SlamFrontend2D, se2_relative


def generate_simple_trajectory(n_poses: int = 10) -> list:
    """Generate simple straight-line trajectory.
    
    Args:
        n_poses: Number of poses.
    
    Returns:
        List of poses [x, y, yaw].
    """
    poses = []
    for i in range(n_poses):
        x = i * 0.5  # Move 0.5m per step
        poses.append(np.array([x, 0.0, 0.0]))
    return poses


def generate_wall_scan(pose: np.ndarray, wall_x: float = 5.0) -> np.ndarray:
    """Generate synthetic scan of a wall parallel to Y-axis.
    
    Args:
        pose: Robot pose [x, y, yaw].
        wall_x: X-coordinate of wall in map frame.
    
    Returns:
        Scan points in robot frame.
    """
    # Wall points in map frame
    wall_points = np.array([[wall_x, y] for y in np.linspace(-2, 2, 20)])
    
    # Transform to robot frame
    x, y, yaw = pose
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    diff = wall_points - np.array([x, y])
    x_local = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
    y_local = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
    
    scan = np.column_stack([x_local, y_local])
    
    # Add small noise
    scan += np.random.normal(0, 0.02, scan.shape)
    
    # Filter points behind robot
    scan = scan[scan[:, 0] > 0]
    
    return scan


def main():
    """Run SLAM front-end demo."""
    print("=" * 80)
    print("SLAM FRONT-END DEMO: Prediction -> Scan-to-Map Alignment -> Map Update")
    print("=" * 80)
    print()
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate ground truth trajectory
    print("1. Generating trajectory...")
    true_poses = generate_simple_trajectory(n_poses=10)
    n_poses = len(true_poses)
    print(f"   Generated {n_poses} poses (straight line)")
    
    # Simulate noisy odometry
    print("\n2. Simulating noisy odometry...")
    odom_poses = []
    odom_poses.append(true_poses[0].copy())
    for i in range(1, n_poses):
        true_delta = se2_relative(true_poses[i - 1], true_poses[i])
        # Add drift
        noisy_delta = true_delta + np.array([
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.02),
            np.random.normal(0, 0.01),
        ])
        odom_pose = odom_poses[-1] + noisy_delta  # Simplified composition for straight line
        odom_poses.append(odom_pose)
    
    odom_drift = np.linalg.norm(odom_poses[-1][:2] - true_poses[-1][:2])
    print(f"   Odometry drift: {odom_drift:.3f} m")
    
    # Generate scans
    print("\n3. Generating LiDAR scans...")
    scans = []
    for pose in true_poses:
        scan = generate_wall_scan(pose, wall_x=5.0)
        scans.append(scan)
    print(f"   Generated {n_poses} scans (avg {np.mean([len(s) for s in scans]):.1f} points/scan)")
    
    # Run SLAM front-end
    print("\n4. Running SLAM front-end...")
    print("=" * 80)
    print(f"{'Step':<6} {'Pred X':<10} {'Est X':<10} {'Correction':<12} {'Residual':<10} {'Converged'}")
    print("=" * 80)
    
    frontend = SlamFrontend2D(submap_voxel_size=0.1, max_icp_residual=0.5)
    frontend_poses = []
    
    for i in range(n_poses):
        # Compute odometry delta
        if i == 0:
            odom_delta = np.array([0.0, 0.0, 0.0])
        else:
            odom_delta = se2_relative(odom_poses[i - 1], odom_poses[i])
        
        # Run front-end step
        result = frontend.step(i, odom_delta, scans[i])
        frontend_poses.append(result['pose_est'])
        
        # Log per-step results
        pred = result['pose_pred']
        est = result['pose_est']
        correction = result['correction_magnitude']
        mq = result['match_quality']
        
        print(f"{i:<6} {pred[0]:<10.3f} {est[0]:<10.3f} {correction:<12.4f} "
              f"{mq.residual:<10.4f} {str(mq.converged)}")
    
    print("=" * 80)
    print()
    
    # Evaluate results
    print("5. Evaluating results...")
    
    odom_errors = np.array([
        np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2])
        for i in range(n_poses)
    ])
    frontend_errors = np.array([
        np.linalg.norm(frontend_poses[i][:2] - true_poses[i][:2])
        for i in range(n_poses)
    ])
    
    odom_rmse = np.sqrt(np.mean(odom_errors**2))
    frontend_rmse = np.sqrt(np.mean(frontend_errors**2))
    
    print(f"   Odometry RMSE: {odom_rmse:.4f} m")
    print(f"   Frontend RMSE: {frontend_rmse:.4f} m")
    improvement = (1 - frontend_rmse / odom_rmse) * 100
    print(f"   Improvement: {improvement:.2f}%")
    print()
    
    # Visualize
    print("6. Visualizing results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectories
    ax1 = axes[0]
    
    true_xy = np.array([[p[0], p[1]] for p in true_poses])
    odom_xy = np.array([[p[0], p[1]] for p in odom_poses])
    frontend_xy = np.array([[p[0], p[1]] for p in frontend_poses])
    
    ax1.plot(true_xy[:, 0], true_xy[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(odom_xy[:, 0], odom_xy[:, 1], 'r--', linewidth=2, label='Odometry (Drift)', alpha=0.7)
    ax1.plot(frontend_xy[:, 0], frontend_xy[:, 1], 'b-', linewidth=2, label='Frontend (Scan-to-Map)', alpha=0.8)
    
    ax1.scatter(true_xy[0, 0], true_xy[0, 1], c='green', marker='o', s=100, zorder=5)
    ax1.scatter(frontend_xy[0, 0], frontend_xy[0, 1], c='blue', marker='o', s=100, zorder=5)
    
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_title('SLAM Front-End: Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Errors
    ax2 = axes[1]
    
    timesteps = np.arange(n_poses)
    ax2.plot(timesteps, odom_errors, 'r--', linewidth=2, label='Odometry Error', alpha=0.7)
    ax2.plot(timesteps, frontend_errors, 'b-', linewidth=2, label='Frontend Error', alpha=0.8)
    
    ax2.set_xlabel('Step Index', fontsize=12)
    ax2.set_ylabel('Position Error [m]', fontsize=12)
    ax2.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figs_dir = Path("ch7_slam/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_file = figs_dir / "slam_frontend_demo.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved figure: {output_file}")
    
    print()
    print("=" * 80)
    print("SLAM FRONT-END DEMO COMPLETE!")
    print("=" * 80)
    print()
    print("Key Concepts:")
    print("  1. PREDICTION: pose_pred = se2_compose(prev_pose, odom_delta)")
    print("  2. CORRECTION: pose_est = icp_point_to_point(scan, submap, pose_pred)")
    print("  3. MAP UPDATE: submap.add_scan(pose_est, scan)")
    print()
    print("Note: This demonstrates the front-end loop. For full SLAM,")
    print("      add back-end pose graph optimization with loop closures.")
    print()


if __name__ == "__main__":
    main()
