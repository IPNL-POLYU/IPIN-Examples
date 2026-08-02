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

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from core.eval import plot_error_magnitude_time, plot_trajectory_2d, save_figure
from core.slam import SlamFrontend2D, se2_relative

FIGURE_NAME = "slam_frontend_demo"
DEFAULT_SEED = 42


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


def run_frontend_demo(
    n_poses: int = 10, seed: int = DEFAULT_SEED
) -> Dict[str, object]:
    """Run the front-end loop over a synthetic straight-line walk.

    Kept separate from ``main`` so the figure -- and the tests that pin what
    the figure claims -- can be built without also producing the console
    report.

    Args:
        n_poses: Number of poses in the walk.
        seed: Seed for the odometry noise, so the committed figure regenerates.

    Returns:
        Dictionary holding the three trajectories as (N, 2) arrays under
        ``true_xy``, ``odom_xy`` and ``frontend_xy``; their error magnitudes
        under ``odom_errors`` and ``frontend_errors``; the input ``scans``; and
        the per-step front-end records under ``steps``.
    """
    np.random.seed(seed)

    true_poses = generate_simple_trajectory(n_poses=n_poses)

    # Simulate noisy odometry: integrate each true delta with drift added.
    odom_poses = [true_poses[0].copy()]
    for i in range(1, n_poses):
        true_delta = se2_relative(true_poses[i - 1], true_poses[i])
        noisy_delta = true_delta + np.array([
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.02),
            np.random.normal(0, 0.01),
        ])
        # Simplified composition, valid only because this walk holds yaw at 0.
        odom_poses.append(odom_poses[-1] + noisy_delta)

    scans = [generate_wall_scan(pose, wall_x=5.0) for pose in true_poses]

    frontend = SlamFrontend2D(submap_voxel_size=0.1, max_icp_residual=0.5)
    frontend_poses: List[np.ndarray] = []
    steps: List[dict] = []

    for i in range(n_poses):
        if i == 0:
            odom_delta = np.array([0.0, 0.0, 0.0])
        else:
            odom_delta = se2_relative(odom_poses[i - 1], odom_poses[i])

        result = frontend.step(i, odom_delta, scans[i])
        frontend_poses.append(result['pose_est'])
        steps.append(result)

    true_xy = np.array([pose[:2] for pose in true_poses])
    odom_xy = np.array([pose[:2] for pose in odom_poses])
    frontend_xy = np.array([pose[:2] for pose in frontend_poses])

    return {
        'true_xy': true_xy,
        'odom_xy': odom_xy,
        'frontend_xy': frontend_xy,
        'odom_errors': np.linalg.norm(odom_xy - true_xy, axis=1),
        'frontend_errors': np.linalg.norm(frontend_xy - true_xy, axis=1),
        'scans': scans,
        'steps': steps,
    }


def build_figure(demo: Dict[str, object]) -> plt.Figure:
    """Build the demo figure: trajectories beside error over time.

    Args:
        demo: Result of :func:`run_frontend_demo`.

    Returns:
        The figure, ready for ``core.eval.save_figure``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # equal_aspect=False: this walk covers 4.5 m in x but only 0.13 m in y, and
    # with equal axes matplotlib stretches y out to roughly [-2, 2] to match.
    # All three tracks then share a hairline band under 3% of the panel's
    # height, with the scan-to-map track -- the one this figure is arguing for
    # -- sitting within its own stroke width of the ground truth. The primitive
    # labels the axes as not-to-scale in exchange, which is the right trade
    # here: the reader is being shown a cross-track deviation, not a shape.
    plot_trajectory_2d(
        demo['true_xy'],
        {
            'Odometry (Drift)': demo['odom_xy'],
            'Frontend (Scan-to-Map)': demo['frontend_xy'],
        },
        title='SLAM Front-End: Trajectories',
        axis_labels=('X [m]', 'Y [m]'),
        ax=axes[0],
        equal_aspect=False,
    )

    plot_error_magnitude_time(
        {
            'Odometry Error': demo['odom_errors'],
            'Frontend Error': demo['frontend_errors'],
        },
        t=np.arange(len(demo['odom_errors'])),
        title='Position Error Over Time',
        ax=axes[1],
    )
    # This series is indexed by pose, not seconds, so correct the shared label.
    axes[1].set_xlabel('Step Index', fontsize=12)

    fig.tight_layout()
    return fig


def main():
    """Run SLAM front-end demo."""
    print("=" * 80)
    print("SLAM FRONT-END DEMO: Prediction -> Scan-to-Map Alignment -> Map Update")
    print("=" * 80)
    print()

    n_poses = 10
    demo = run_frontend_demo(n_poses=n_poses)
    true_xy = demo['true_xy']
    odom_xy = demo['odom_xy']
    scans = demo['scans']

    print("1. Generating trajectory...")
    print(f"   Generated {n_poses} poses (straight line)")

    print("\n2. Simulating noisy odometry...")
    odom_drift = np.linalg.norm(odom_xy[-1] - true_xy[-1])
    print(f"   Odometry drift: {odom_drift:.3f} m")

    print("\n3. Generating LiDAR scans...")
    print(f"   Generated {n_poses} scans "
          f"(avg {np.mean([len(s) for s in scans]):.1f} points/scan)")

    print("\n4. Running SLAM front-end...")
    print("=" * 80)
    print(f"{'Step':<6} {'Pred X':<10} {'Est X':<10} {'Correction':<12} {'Residual':<10} {'Converged'}")
    print("=" * 80)

    for i, result in enumerate(demo['steps']):
        pred = result['pose_pred']
        est = result['pose_est']
        mq = result['match_quality']

        print(f"{i:<6} {pred[0]:<10.3f} {est[0]:<10.3f} "
              f"{result['correction_magnitude']:<12.4f} "
              f"{mq.residual:<10.4f} {str(mq.converged)}")

    print("=" * 80)
    print()

    # Evaluate results
    print("5. Evaluating results...")

    odom_rmse = np.sqrt(np.mean(demo['odom_errors'] ** 2))
    frontend_rmse = np.sqrt(np.mean(demo['frontend_errors'] ** 2))

    print(f"   Odometry RMSE: {odom_rmse:.4f} m")
    print(f"   Frontend RMSE: {frontend_rmse:.4f} m")
    improvement = (1 - frontend_rmse / odom_rmse) * 100
    print(f"   Improvement: {improvement:.2f}%")
    print()

    # Visualize
    print("6. Visualizing results...")

    fig = build_figure(demo)
    try:
        paths = save_figure(fig, Path(__file__).parent / "figs", FIGURE_NAME)
    finally:
        plt.close(fig)
    print(f"\n[OK] Saved figure: {paths[0]}")

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
