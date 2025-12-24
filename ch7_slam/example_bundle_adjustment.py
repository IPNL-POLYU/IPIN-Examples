"""Visual Bundle Adjustment Example with Synthetic Camera Observations.

This example demonstrates bundle adjustment for visual SLAM:
    1. Generate synthetic camera trajectory
    2. Generate 3D landmarks (map features)
    3. Simulate camera observations (pixel coordinates)
    4. Add noise to initial estimates
    5. Run bundle adjustment to jointly optimize poses and landmarks
    6. Visualize results

This implements bundle adjustment from Section 7.4 of Chapter 7,
specifically Eqs. (7.68)-(7.70).

Usage:
    python -m ch7_slam.example_bundle_adjustment

Author: Li-Ta Hsu
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from core.slam import (
    CameraIntrinsics,
    se2_compose,
    project_point,
    create_reprojection_factor,
)
from core.estimators.factor_graph import FactorGraph


def generate_camera_trajectory(
    n_poses: int = 10,
    radius: float = 5.0,
) -> List[np.ndarray]:
    """
    Generate circular camera trajectory for bundle adjustment demo.

    Args:
        n_poses: Number of camera poses.
        radius: Radius of circular trajectory.

    Returns:
        List of poses [x, y, yaw] where camera looks toward center.
    """
    poses = []
    
    for i in range(n_poses):
        angle = (i / n_poses) * 2 * np.pi
        
        # Position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Yaw points toward center + 90deg (camera looks inward)
        yaw = angle + np.pi
        
        poses.append(np.array([x, y, yaw]))
    
    return poses


def generate_landmarks(
    n_landmarks: int = 20,
    area_size: float = 6.0,
) -> np.ndarray:
    """
    Generate random 3D landmarks in a volume.

    Args:
        n_landmarks: Number of landmarks.
        area_size: Size of cubic volume centered at origin.

    Returns:
        Landmarks array, shape (n_landmarks, 3) in [x, y, z] format.
    """
    landmarks = np.random.uniform(
        -area_size / 2,
        area_size / 2,
        (n_landmarks, 3)
    )
    # Ensure landmarks are at reasonable height
    landmarks[:, 2] = np.abs(landmarks[:, 2])  # Z should be positive
    
    return landmarks


def generate_observations(
    poses: List[np.ndarray],
    landmarks: np.ndarray,
    intrinsics: CameraIntrinsics,
    observation_noise: float = 1.0,
    min_depth: float = 1.0,
    max_depth: float = 15.0,
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """
    Generate synthetic camera observations (pixel coordinates).

    Args:
        poses: List of camera poses [x, y, yaw] in 2D (assumes constant height).
        landmarks: Landmark positions [x, y, z], shape (N, 3).
        intrinsics: Camera intrinsic parameters.
        observation_noise: Pixel noise standard deviation.
        min_depth: Minimum depth for visibility.
        max_depth: Maximum depth for visibility.

    Returns:
        Dictionary mapping pose_id → [(landmark_id, pixel), ...]
        where pixel is [u, v] coordinates.
    """
    observations = {}
    
    for pose_id, pose_2d in enumerate(poses):
        pose_observations = []
        
        # Assume camera at fixed height (simplified 2D→3D)
        camera_height = 0.0
        x_cam, y_cam, yaw_cam = pose_2d
        
        for landmark_id, landmark in enumerate(landmarks):
            lx, ly, lz = landmark
            
            # Transform landmark to camera frame
            # Camera at (x_cam, y_cam, camera_height), facing yaw_cam
            dx_map = lx - x_cam
            dy_map = ly - y_cam
            dz_map = lz - camera_height
            
            # Rotate to camera frame (X-right, Y-down, Z-forward)
            cos_yaw = np.cos(yaw_cam)
            sin_yaw = np.sin(yaw_cam)
            
            # Transform: camera X points forward (along yaw)
            x_in_cam = cos_yaw * dx_map + sin_yaw * dy_map
            y_in_cam = -sin_yaw * dx_map + cos_yaw * dy_map
            z_in_cam = dz_map
            
            # Camera coordinate convention: [Y-right, Z-down, X-forward]
            # Adjust for projection (Z-forward convention)
            point_camera = np.array([y_in_cam, z_in_cam, x_in_cam])
            
            # Check if landmark is visible (in front of camera, within range)
            depth = point_camera[2]
            if min_depth < depth < max_depth:
                try:
                    # Project to pixel
                    pixel = project_point(intrinsics, point_camera)
                    
                    # Check if within image bounds (640x480 typical)
                    if 0 < pixel[0] < 640 and 0 < pixel[1] < 480:
                        # Add observation noise
                        noisy_pixel = pixel + np.random.normal(0, observation_noise, 2)
                        pose_observations.append((landmark_id, noisy_pixel))
                except ValueError:
                    # Behind camera or projection failed
                    pass
        
        observations[pose_id] = pose_observations
    
    return observations


def add_noise_to_estimates(
    poses: List[np.ndarray],
    landmarks: np.ndarray,
    pose_noise: float = 0.3,
    landmark_noise: float = 0.5,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Add noise to ground truth to create initial estimates for BA.

    Args:
        poses: True camera poses.
        landmarks: True landmark positions.
        pose_noise: Standard deviation of pose noise (m and rad).
        landmark_noise: Standard deviation of landmark noise (m).

    Returns:
        Tuple of (noisy_poses, noisy_landmarks).
    """
    noisy_poses = [
        pose + np.random.normal(0, pose_noise, pose.shape)
        for pose in poses
    ]
    
    noisy_landmarks = landmarks + np.random.normal(0, landmark_noise, landmarks.shape)
    
    return noisy_poses, noisy_landmarks


def plot_bundle_adjustment_results(
    poses_true: List[np.ndarray],
    poses_init: List[np.ndarray],
    poses_opt: List[np.ndarray],
    landmarks_true: np.ndarray,
    landmarks_init: np.ndarray,
    landmarks_opt: np.ndarray,
    error_history: List[float],
):
    """
    Visualize bundle adjustment results.

    Args:
        poses_true: Ground truth poses.
        poses_init: Initial noisy poses.
        poses_opt: Optimized poses.
        landmarks_true: Ground truth landmarks.
        landmarks_init: Initial noisy landmarks.
        landmarks_opt: Optimized landmarks.
        error_history: Optimization error per iteration.
    """
    fig = plt.figure(figsize=(18, 6))
    
    # --- Plot 1: Camera Trajectory (Top View) ---
    ax1 = fig.add_subplot(131)
    
    # Extract XY coordinates
    true_xy = np.array([[p[0], p[1]] for p in poses_true])
    init_xy = np.array([[p[0], p[1]] for p in poses_init])
    opt_xy = np.array([[p[0], p[1]] for p in poses_opt])
    
    # Plot landmarks
    ax1.scatter(
        landmarks_true[:, 0], landmarks_true[:, 1],
        c='gray', marker='x', s=50, alpha=0.5, label='Landmarks (true)'
    )
    ax1.scatter(
        landmarks_init[:, 0], landmarks_init[:, 1],
        c='red', marker='o', s=30, alpha=0.3, label='Landmarks (init)'
    )
    ax1.scatter(
        landmarks_opt[:, 0], landmarks_opt[:, 1],
        c='blue', marker='o', s=30, alpha=0.5, label='Landmarks (opt)'
    )
    
    # Plot trajectories
    ax1.plot(true_xy[:, 0], true_xy[:, 1], 'g-', linewidth=2, label='Poses (true)', alpha=0.7)
    ax1.plot(init_xy[:, 0], init_xy[:, 1], 'r--', linewidth=2, label='Poses (init)', alpha=0.7)
    ax1.plot(opt_xy[:, 0], opt_xy[:, 1], 'b-', linewidth=2, label='Poses (opt)', alpha=0.8)
    
    ax1.set_xlabel('X [m]', fontsize=11)
    ax1.set_ylabel('Y [m]', fontsize=11)
    ax1.set_title('Bundle Adjustment: Top View', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # --- Plot 2: Position Errors ---
    ax2 = fig.add_subplot(132)
    
    pose_errors_init = np.array([
        np.linalg.norm(poses_init[i][:2] - poses_true[i][:2])
        for i in range(len(poses_true))
    ])
    pose_errors_opt = np.array([
        np.linalg.norm(poses_opt[i][:2] - poses_true[i][:2])
        for i in range(len(poses_true))
    ])
    
    landmark_errors_init = np.linalg.norm(landmarks_init - landmarks_true, axis=1)
    landmark_errors_opt = np.linalg.norm(landmarks_opt - landmarks_true, axis=1)
    
    pose_indices = np.arange(len(poses_true))
    landmark_indices = np.arange(len(landmarks_true))
    
    ax2.plot(pose_indices, pose_errors_init, 'r--', linewidth=2, label='Pose Error (init)', alpha=0.7)
    ax2.plot(pose_indices, pose_errors_opt, 'b-', linewidth=2, label='Pose Error (opt)', alpha=0.8)
    
    ax2.set_xlabel('Pose Index', fontsize=11)
    ax2.set_ylabel('Position Error [m]', fontsize=11)
    ax2.set_title('Camera Pose Errors', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Optimization Convergence ---
    ax3 = fig.add_subplot(133)
    
    iterations = np.arange(len(error_history))
    ax3.semilogy(iterations, error_history, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Total Error (log scale)', fontsize=11)
    ax3.set_title('Bundle Adjustment Convergence', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ch7_slam/bundle_adjustment_results.png", dpi=150, bbox_inches="tight")
    print("\n[OK] Saved figure: ch7_slam/bundle_adjustment_results.png")
    
    # Only show interactively if display available
    import os
    if os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") != "Agg":
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass


def main():
    """Run complete bundle adjustment example."""
    print("=" * 80)
    print("CHAPTER 7: VISUAL BUNDLE ADJUSTMENT EXAMPLE")
    print("=" * 80)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # ------------------------------------------------------------------------
    # 1. Setup Camera Parameters
    # ------------------------------------------------------------------------
    print("1. Setting up camera parameters...")
    intrinsics = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        k1=-0.05,   # Slight barrel distortion
        k2=0.01,    # Secondary radial
        p1=0.001,   # Tangential
        p2=0.001,   # Tangential
    )
    print(f"   Camera: fx={intrinsics.fx}, fy={intrinsics.fy}")
    print(f"   Distortion: k1={intrinsics.k1}, k2={intrinsics.k2}")

    # ------------------------------------------------------------------------
    # 2. Generate Ground Truth (Poses + Landmarks)
    # ------------------------------------------------------------------------
    print("\n2. Generating ground truth...")
    
    # Camera trajectory (circular)
    n_poses = 8
    poses_true = generate_camera_trajectory(n_poses=n_poses, radius=5.0)
    print(f"   Generated {n_poses} camera poses (circular trajectory)")
    
    # Landmarks (3D features in the scene)
    n_landmarks = 15
    landmarks_true = generate_landmarks(n_landmarks=n_landmarks, area_size=4.0)
    print(f"   Generated {n_landmarks} 3D landmarks")

    # ------------------------------------------------------------------------
    # 3. Generate Observations (Simulate Camera Measurements)
    # ------------------------------------------------------------------------
    print("\n3. Simulating camera observations...")
    observations = generate_observations(
        poses_true,
        landmarks_true,
        intrinsics,
        observation_noise=0.5,  # 0.5 pixel noise
        min_depth=1.0,
        max_depth=12.0,
    )
    
    total_observations = sum(len(obs) for obs in observations.values())
    avg_obs_per_pose = total_observations / n_poses
    print(f"   Generated {total_observations} observations")
    print(f"   Average {avg_obs_per_pose:.1f} observations per pose")

    # ------------------------------------------------------------------------
    # 4. Create Noisy Initial Estimates
    # ------------------------------------------------------------------------
    print("\n4. Creating noisy initial estimates...")
    poses_init, landmarks_init = add_noise_to_estimates(
        poses_true,
        landmarks_true,
        pose_noise=0.05,      # 5cm position, ~3deg heading (start closer for stability)
        landmark_noise=0.1,   # 10cm landmark position
    )
    
    pose_init_rmse = np.sqrt(np.mean([
        np.linalg.norm(poses_init[i][:2] - poses_true[i][:2])**2
        for i in range(n_poses)
    ]))
    landmark_init_rmse = np.sqrt(np.mean(
        np.linalg.norm(landmarks_init - landmarks_true, axis=1)**2
    ))
    
    print(f"   Initial pose RMSE: {pose_init_rmse:.4f} m")
    print(f"   Initial landmark RMSE: {landmark_init_rmse:.4f} m")

    # ------------------------------------------------------------------------
    # 5. Build Bundle Adjustment Factor Graph
    # ------------------------------------------------------------------------
    print("\n5. Building bundle adjustment factor graph...")
    
    graph = FactorGraph()
    
    # Add camera pose variables (variable IDs 0 to n_poses-1)
    for i, pose in enumerate(poses_init):
        graph.add_variable(i, pose)
    
    # Add landmark variables (variable IDs n_poses to n_poses+n_landmarks-1)
    for i, landmark in enumerate(landmarks_init):
        # For 2D visual SLAM, we use [x, y] landmarks (assuming constant Z)
        # For simplicity in this demo, use full 3D
        graph.add_variable(n_poses + i, landmark)
    
    # Add reprojection factors for all observations
    n_factors = 0
    pixel_info = np.eye(2) / (0.5**2)  # Inverse covariance (0.5 pixel std)
    
    for pose_id, obs_list in observations.items():
        for landmark_id, observed_pixel in obs_list:
            factor = create_reprojection_factor(
                camera_pose_id=pose_id,
                landmark_id=n_poses + landmark_id,
                observed_pixel=observed_pixel,
                camera_intrinsics=intrinsics,
                information=pixel_info,
            )
            graph.add_factor(factor)
            n_factors += 1
    
    # Add weak prior on first pose to prevent gauge freedom
    from core.slam.factors import create_prior_factor
    prior_info = np.diag([10.0, 10.0, 10.0])  # Weak prior
    prior_factor = create_prior_factor(0, poses_true[0], information=prior_info)
    graph.add_factor(prior_factor)
    n_factors += 1
    
    print(f"   Factor graph: {len(graph.variables)} variables")
    print(f"   Variables: {n_poses} poses + {n_landmarks} landmarks")
    print(f"   Factors: {n_factors} ({n_factors-1} reprojection + 1 prior)")

    # ------------------------------------------------------------------------
    # 6. Optimize Bundle Adjustment
    # ------------------------------------------------------------------------
    print("\n6. Running bundle adjustment optimization...")
    
    initial_error = graph.compute_error()
    print(f"   Initial reprojection error: {initial_error:.6f}")
    
    # Run optimization (Gauss-Newton)
    optimized_vars, error_history = graph.optimize(
        method="gauss_newton",
        max_iterations=15,
        tol=1e-3,
    )
    
    final_error = error_history[-1]
    print(f"   Final reprojection error: {final_error:.6f}")
    print(f"   Iterations: {len(error_history) - 1}")
    print(f"   Error reduction: {(1 - final_error / initial_error) * 100:.2f}%")

    # Extract optimized poses and landmarks
    poses_opt = [optimized_vars[i] for i in range(n_poses)]
    landmarks_opt = np.array([optimized_vars[n_poses + i] for i in range(n_landmarks)])

    # ------------------------------------------------------------------------
    # 7. Evaluate Results
    # ------------------------------------------------------------------------
    print("\n7. Evaluating bundle adjustment results...")
    
    # Pose errors
    pose_errors_init = np.array([
        np.linalg.norm(poses_init[i][:2] - poses_true[i][:2])
        for i in range(n_poses)
    ])
    pose_errors_opt = np.array([
        np.linalg.norm(poses_opt[i][:2] - poses_true[i][:2])
        for i in range(n_poses)
    ])
    
    pose_rmse_init = np.sqrt(np.mean(pose_errors_init**2))
    pose_rmse_opt = np.sqrt(np.mean(pose_errors_opt**2))
    
    print(f"   Pose RMSE (initial): {pose_rmse_init:.4f} m")
    print(f"   Pose RMSE (optimized): {pose_rmse_opt:.4f} m")
    print(f"   Pose improvement: {(1 - pose_rmse_opt / pose_rmse_init) * 100:.2f}%")
    
    # Landmark errors
    landmark_errors_init = np.linalg.norm(landmarks_init - landmarks_true, axis=1)
    landmark_errors_opt = np.linalg.norm(landmarks_opt - landmarks_true, axis=1)
    
    landmark_rmse_init = np.sqrt(np.mean(landmark_errors_init**2))
    landmark_rmse_opt = np.sqrt(np.mean(landmark_errors_opt**2))
    
    print(f"   Landmark RMSE (initial): {landmark_rmse_init:.4f} m")
    print(f"   Landmark RMSE (optimized): {landmark_rmse_opt:.4f} m")
    print(f"   Landmark improvement: {(1 - landmark_rmse_opt / landmark_rmse_init) * 100:.2f}%")

    # ------------------------------------------------------------------------
    # 8. Visualize Results
    # ------------------------------------------------------------------------
    print("\n8. Visualizing results...")
    plot_bundle_adjustment_results(
        poses_true, poses_init, poses_opt,
        landmarks_true, landmarks_init, landmarks_opt,
        error_history
    )

    print()
    print("=" * 80)
    print("BUNDLE ADJUSTMENT COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Camera trajectory: {n_poses} poses")
    print(f"  - 3D landmarks: {n_landmarks} features")
    print(f"  - Total observations: {total_observations}")
    print(f"  - Reprojection error reduction: {(1 - final_error / initial_error) * 100:.1f}%")
    print(f"  - Pose accuracy: {pose_rmse_opt:.4f} m RMSE")
    print(f"  - Landmark accuracy: {landmark_rmse_opt:.4f} m RMSE")
    print()
    print("Key Concepts:")
    print("  - Reprojection error: 2D pixel difference (observed vs. projected)")
    print("  - Bundle adjustment: Joint optimization of poses + landmarks")
    print("  - Implements Eqs. (7.68)-(7.70) from Chapter 7, Section 7.4")
    print()


if __name__ == "__main__":
    main()

