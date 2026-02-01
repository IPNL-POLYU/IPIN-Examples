"""Complete 2D Pose Graph SLAM Example.

This example demonstrates a complete observation-driven SLAM pipeline:

**Pipeline Stages:**
    1. FRONT-END: init -> predict (odom) -> scan-to-map (ICP) -> update map
       - SlamFrontend2D corrects odometry drift via scan-to-map alignment
       - Achieves ~70% local RMSE improvement

    2. LOOP CLOSURE: observation-based detection
       - Scan descriptors (range histograms) for candidate selection
       - ICP verification for geometric consistency
       - Finds ~20 loop closures per trajectory

    3. BACK-END: pose graph optimization
       - Prior + odometry + loop closure factors
       - Gauss-Newton solver for global consistency
       - Achieves ~50% total RMSE improvement

**Critical Requirements:**
    - Scans MUST be generated from true_poses (sensor reality)
    - DO NOT generate scans from odom/estimates (that's circular!)
    - frontend_poses MUST differ from odom_poses (ICP must be working)

Can run with:
    python example_pose_graph_slam.py              # Inline mode (full pipeline)
    python example_pose_graph_slam.py --data ch7_slam_2d_square
    python example_pose_graph_slam.py --data ch7_slam_2d_high_drift

Key algorithms used:
    - Section 7.3.1: ICP scan matching (front-end + loop closure verification)
    - Section 7.3.5: Close-loop constraints (Eq. 7.22)
    - Section 7.1.2: GraphSLAM framework (Table 7.2)

Note: NDT (Section 7.3.2) is implemented in core/slam/ndt.py but not used
in this script. ICP is used for all scan matching operations.

Usage:
    python -m ch7_slam.example_pose_graph_slam

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from core.slam import (
    se2_apply,
    se2_compose,
    se2_relative,
    icp_point_to_point,
    create_pose_graph,
    SlamFrontend2D,
    LoopClosureDetector2D,
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


def run_with_dataset(data_dir: str, use_loop_oracle: bool = False) -> None:
    """Run pose graph SLAM using pre-generated dataset.
    
    Args:
        data_dir: Path to dataset directory
        use_loop_oracle: If True, use distance-based oracle instead of observation-based
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
    
    # ------------------------------------------------------------------------
    # Run SLAM Front-End: Prediction -> Scan-to-Map -> Map Update
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Front-end: init -> predict -> scan-to-map -> update map")
    print("-" * 70)
    
    # Initialize front-end (use first odometry pose as initial pose)
    frontend = SlamFrontend2D(
        submap_voxel_size=0.15,
        min_map_points=10,
        max_icp_residual=1.0,  # Slightly more tolerant for dataset mode
        initial_pose=odom_poses[0].copy(),  # Start from odometry's first pose
    )
    
    # Run front-end for each timestep
    frontend_poses = []
    pred_poses = []
    match_qualities = []
    corrections = []
    
    for i in range(n_poses):
        # Compute odometry delta
        if i == 0:
            odom_delta = np.array([0.0, 0.0, 0.0])
        else:
            odom_delta = se2_relative(odom_poses[i - 1], odom_poses[i])
        
        # Run front-end step
        result = frontend.step(i, odom_delta, scans[i])
        
        # Store results
        frontend_poses.append(result['pose_est'])
        pred_poses.append(result['pose_pred'])
        match_qualities.append(result['match_quality'])
        
        # Compute correction magnitude (if not first step)
        if i > 0:
            correction = np.linalg.norm(result['pose_est'][:2] - result['pose_pred'][:2])
            corrections.append(correction)
    
    # Compute front-end statistics
    n_converged = sum(1 for mq in match_qualities if mq.converged)
    converged_qualities = [mq for mq in match_qualities if mq.converged]
    avg_residual = np.mean([mq.residual for mq in converged_qualities]) if converged_qualities else 0.0
    avg_correction = np.mean(corrections) if corrections else 0.0
    
    print(f"\n  Processed {n_poses} steps")
    print(f"  Frontend converged ratio: {100*n_converged/n_poses:.1f}%")
    print(f"  Frontend avg residual: {avg_residual:.4f} m")
    print(f"  Frontend avg correction: {avg_correction:.4f} m")
    
    # Verify frontend_poses differs from odom_poses
    max_trans_diff = 0.0
    max_yaw_diff = 0.0
    for i in range(n_poses):
        trans_diff = np.linalg.norm(frontend_poses[i][:2] - odom_poses[i][:2])
        yaw_diff = abs(frontend_poses[i][2] - odom_poses[i][2])
        max_trans_diff = max(max_trans_diff, trans_diff)
        max_yaw_diff = max(max_yaw_diff, yaw_diff)
    
    print(f"\n  max|frontend - odom| translation: {max_trans_diff:.4f} m")
    print(f"  max|frontend - odom| yaw: {np.degrees(max_yaw_diff):.2f} deg")
    
    if max_trans_diff < 1e-6 and max_yaw_diff < 1e-6:
        print("  [WARNING] Frontend poses identical to odometry - ICP not working!")
    
    # Detect loop closures
    print("\n" + "-" * 70)
    if use_loop_oracle:
        print("Loop Closure Detection (ORACLE MODE - distance-based)...")
    else:
        print("Loop Closure Detection (observation-based)...")
    
    # Default: observation-based detector with no distance gating
    loop_closures = detect_loop_closures(
        poses=frontend_poses,  # Use front-end estimates
        scans=scans,
        use_observation_based=not use_loop_oracle,  # Default: observation-based
        distance_threshold=None,  # No distance gating (observation-based is primary)
        min_time_separation=5  # Lower to find more candidates
    )
    
    mode_str = "oracle" if use_loop_oracle else "observation-based"
    print(f"\n  Detected {len(loop_closures)} loop closures ({mode_str})")
    print()
    
    # Also show what the dataset provided (for reference)
    if loop_closure_data.ndim == 1:
        loop_closure_data = loop_closure_data.reshape(1, -1)
    print(f"  [Reference: Dataset provided {len(loop_closure_data)} ground truth loop closure indices]")
    
    # Build pose graph
    print("\n" + "-" * 70)
    print("Building pose graph...")
    
    # CRITICAL: Initialize graph from front-end poses (NOT odometry, NOT ground truth)
    # This ensures the front-end's scan-to-map corrections are preserved
    initial_poses = frontend_poses  # Must be frontend_poses, not odom_poses!
    print(f"  Graph initial: frontend_poses (scan-to-map corrected trajectory)")
    
    # Prepare odometry measurements from front-end estimates
    odometry_measurements = []
    for i in range(n_poses - 1):
        # Use front-end pose deltas (scan-to-map corrected)
        rel_pose = se2_relative(initial_poses[i], initial_poses[i + 1])
        odometry_measurements.append((i, i + 1, rel_pose))
    
    # Prepare loop closure measurements and information matrices
    loop_measurements = []
    loop_info_matrices = []
    for i, j, rel_pose, cov in loop_closures:
        loop_measurements.append((i, j, rel_pose))
        # Use individual covariance from loop closure (reflects ICP quality)
        loop_info_matrices.append(np.linalg.inv(cov))
    
    # Odometry information (moderate uncertainty)
    odom_info = np.linalg.inv(np.diag([0.1, 0.1, 0.02]))
    
    # Loop closure information (use first one as representative, or average)
    if len(loop_info_matrices) > 0:
        loop_info = loop_info_matrices[0]
    else:
        loop_info = np.linalg.inv(np.diag([0.05, 0.05, 0.01]))
    
    # Create pose graph with front-end estimates as initial values
    # CRITICAL: Set prior to actual first pose (trajectory may not start at origin)
    first_pose = initial_poses[0].copy()
    graph = create_pose_graph(
        poses=initial_poses,  # Initial values from front-end (NOT odometry!)
        odometry_measurements=odometry_measurements,
        loop_closures=loop_measurements if loop_measurements else None,
        prior_pose=first_pose,  # Use actual starting pose as prior
        odometry_information=odom_info,
        loop_information=loop_info,
    )
    
    print(f"  Pose graph: {len(graph.variables)} variables, {len(graph.factors)} factors")
    print(f"  Factors: 1 prior + {len(odometry_measurements)} odometry + {len(loop_measurements)} loop closures")
    
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
    if initial_error > 1e-9:
        print(f"  Error reduction: {(1 - final_error / initial_error) * 100:.2f}%")
    else:
        print(f"  Error reduction: N/A (initial error near zero)")
    
    optimized_poses = [optimized_vars[i] for i in range(n_poses)]
    
    # Evaluate
    print("\n" + "-" * 70)
    print("Results:")
    
    odom_errors = np.array([np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)])
    frontend_errors = np.array([np.linalg.norm(frontend_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)])
    opt_errors = np.array([np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)])
    
    odom_rmse = np.sqrt(np.mean(odom_errors**2))
    frontend_rmse = np.sqrt(np.mean(frontend_errors**2))
    opt_rmse = np.sqrt(np.mean(opt_errors**2))
    
    print(f"  Odometry RMSE: {odom_rmse:.4f} m (baseline)")
    print(f"  Frontend RMSE: {frontend_rmse:.4f} m (scan-to-map corrected)")
    print(f"  Optimized RMSE: {opt_rmse:.4f} m (backend with {len(loop_closures)} loop closures)")
    
    if odom_rmse > 0:
        frontend_improvement = (1 - frontend_rmse / odom_rmse) * 100
        full_improvement = (1 - opt_rmse / odom_rmse) * 100
        print(f"  Frontend improvement: {frontend_improvement:+.2f}%")
        print(f"  Full pipeline improvement: {full_improvement:+.2f}%")
    
    final_loop_error = np.linalg.norm(optimized_poses[-1][:2] - optimized_poses[0][:2])
    print(f"  Final loop closure error: {final_loop_error:.4f} m")
    
    # Visualize
    print("\n" + "-" * 70)
    print("Generating plots...")
    plot_slam_results(
        true_poses, odom_poses, frontend_poses, optimized_poses, 
        landmarks, loop_closures, scans
    )
    
    print("\n" + "=" * 70)
    print("SLAM PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Trajectory: {n_poses} poses")
    print(f"  - Front-end: {n_converged}/{n_poses} converged ({100*n_converged/n_poses:.1f}%)")
    print(f"  - Loop closures: {len(loop_closures)} (observation-based detection)")
    print(f"  - Odometry drift: {final_drift:.3f} m")
    print(f"  - Odometry RMSE: {odom_rmse:.4f} m (baseline)")
    print(f"  - Frontend RMSE: {frontend_rmse:.4f} m (scan-to-map corrected)")
    print(f"  - Optimized RMSE: {opt_rmse:.4f} m (backend)")
    if odom_rmse > 0:
        frontend_improvement = (1 - frontend_rmse / odom_rmse) * 100
        full_improvement = (1 - opt_rmse / odom_rmse) * 100
        print(f"  - Frontend improvement: {frontend_improvement:+.2f}%")
        print(f"  - Full pipeline improvement: {full_improvement:+.2f}%")
    print()
    
    # Machine-readable summary for automated testing
    import json
    summary = {
        "mode": "dataset",
        "frontend_used": True,  # SlamFrontend2D.step() was executed
        "n_scans": len(scans),
        "n_frontend_steps": n_poses,  # Each pose runs frontend.step()
        "n_poses": n_poses,
        "n_loop_closures": len(loop_closures),
        "rmse": {
            "odom": round(odom_rmse, 4),
            "frontend": round(frontend_rmse, 4),
            "optimized": round(opt_rmse, 4),
        }
    }
    print(f"\n[SLAM_SUMMARY] {json.dumps(summary)}")


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
        - Goes: East -> North -> West -> South -> returns to start
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


def generate_corridor_loop_trajectory(
    corridor_length: float = 15.0,
    n_poses_out: int = 30,
    n_poses_back: int = 30,
) -> List[np.ndarray]:
    """
    Generate a trajectory that goes down a corridor, turns, and returns.
    
    The robot returns on the SAME path (same Y, same heading) to ensure
    scans are directly comparable for ICP loop closure detection.
    
    Args:
        corridor_length: How far to travel before turning (meters).
        n_poses_out: Number of poses on outbound leg.
        n_poses_back: Number of poses on return leg.
    
    Returns:
        List of poses [x, y, yaw] representing the trajectory.
    """
    poses = []
    
    # Start/end positions (avoid being too close to end walls)
    x_start = 2.0
    x_end = corridor_length - 2.0  # Stay away from far wall
    travel_dist = x_end - x_start
    
    # Outbound leg: Move East (heading = 0)
    for i in range(n_poses_out):
        x = x_start + (i / n_poses_out) * travel_dist
        y = 0.0  # Center of corridor
        poses.append(np.array([x, y, 0.0]))
    
    # At far end - no turn needed, we'll reverse direction but keep same heading
    # Robot will "back up" - this is artificial but ensures scan consistency
    
    # Return leg: Move West BACKWARDS (still heading = 0, same scans!)
    # This simulates the robot revisiting the same positions
    for i in range(1, n_poses_back + 1):
        x = x_end - (i / n_poses_back) * travel_dist
        y = 0.0  # Same Y as outbound
        poses.append(np.array([x, y, 0.0]))  # Same heading - scans will match!
    
    return poses


def generate_smooth_square_trajectory(
    side_length: float = 8.0,
    n_poses_per_side: int = 15,
) -> List[np.ndarray]:
    """
    Generate a smooth square trajectory with many poses for SLAM front-end.
    
    Uses more poses per side and smooth heading transitions for better
    scan-to-map ICP matching.
    
    Args:
        side_length: Length of each side in meters.
        n_poses_per_side: Number of poses per side (more = smoother).
    
    Returns:
        List of poses [x, y, yaw] representing the trajectory.
    """
    poses = []
    
    # Side 1: Move East (heading = 0)
    for i in range(n_poses_per_side):
        x = (i / n_poses_per_side) * side_length
        poses.append(np.array([x, 0.0, 0.0]))
    
    # Corner 1: Smooth turn to North
    for i in range(5):
        yaw = (i + 1) * (np.pi / 2) / 5
        poses.append(np.array([side_length, 0.0, yaw]))
    
    # Side 2: Move North (heading = pi/2)
    for i in range(1, n_poses_per_side):
        y = (i / n_poses_per_side) * side_length
        poses.append(np.array([side_length, y, np.pi / 2]))
    
    # Corner 2: Smooth turn to West
    for i in range(5):
        yaw = np.pi / 2 + (i + 1) * (np.pi / 2) / 5
        poses.append(np.array([side_length, side_length, yaw]))
    
    # Side 3: Move West (heading = pi)
    for i in range(1, n_poses_per_side):
        x = side_length - (i / n_poses_per_side) * side_length
        poses.append(np.array([x, side_length, np.pi]))
    
    # Corner 3: Smooth turn to South
    for i in range(5):
        yaw = np.pi + (i + 1) * (np.pi / 2) / 5
        # Normalize to [-pi, pi]
        if yaw > np.pi:
            yaw = yaw - 2 * np.pi
        poses.append(np.array([0.0, side_length, yaw]))
    
    # Side 4: Move South (heading = -pi/2)
    for i in range(1, n_poses_per_side):
        y = side_length - (i / n_poses_per_side) * side_length
        poses.append(np.array([0.0, y, -np.pi / 2]))
    
    # Corner 4: Smooth turn back to East (close loop)
    for i in range(5):
        yaw = -np.pi / 2 + (i + 1) * (np.pi / 2) / 5
        poses.append(np.array([0.0, 0.0, yaw]))
    
    return poses


def generate_square_loop_trajectory(
    side_length: float = 8.0,
    n_poses_per_side: int = 12,
    n_laps: int = 2,
) -> List[np.ndarray]:
    """
    Generate a square loop trajectory with multiple laps for SLAM.
    
    This creates a closed-loop trajectory where the robot revisits
    the same locations, enabling loop closure detection.
    
    Args:
        side_length: Length of each side in meters.
        n_poses_per_side: Number of poses per side (more = smoother).
        n_laps: Number of complete laps around the square.
    
    Returns:
        List of poses [x, y, yaw] representing the trajectory.
        Start and end poses are identical (closed loop).
    """
    poses = []
    
    # Generate multiple laps
    for lap in range(n_laps):
        # Side 1: Move East (heading = 0)
        for i in range(n_poses_per_side):
            t = i / n_poses_per_side
            x = t * side_length
            poses.append(np.array([x, 0.0, 0.0]))
        
        # Side 2: Move North (heading = pi/2)
        for i in range(n_poses_per_side):
            t = i / n_poses_per_side
            y = t * side_length
            poses.append(np.array([side_length, y, np.pi / 2]))
        
        # Side 3: Move West (heading = pi)
        for i in range(n_poses_per_side):
            t = i / n_poses_per_side
            x = side_length - t * side_length
            poses.append(np.array([x, side_length, np.pi]))
        
        # Side 4: Move South (heading = -pi/2)
        for i in range(n_poses_per_side):
            t = i / n_poses_per_side
            y = side_length - t * side_length
            poses.append(np.array([0.0, y, -np.pi / 2]))
    
    # Close the loop: add final pose at origin with original heading
    poses.append(np.array([0.0, 0.0, 0.0]))
    
    return poses


def create_room_walls(
    side_length: float = 8.0,
    wall_offset: float = 1.0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Create wall segments forming a room around a square trajectory.
    
    The walls are placed slightly outside the trajectory to ensure
    LiDAR scans hit the walls at reasonable ranges. Internal features
    (pillars, partial walls) break symmetry for reliable ICP and
    meaningful loop closure descriptors.
    
    Args:
        side_length: Size of the square trajectory.
        wall_offset: Distance from trajectory to walls.
    
    Returns:
        Tuple of:
        - List of (start_point, end_point) tuples defining walls.
        - Array of landmark points (wall corners + feature corners).
    """
    # Wall boundaries
    min_coord = -wall_offset
    max_coord = side_length + wall_offset
    
    walls = []
    landmarks_list = []
    
    # Outer walls (room boundary)
    # South wall
    walls.append((np.array([min_coord, min_coord]), np.array([max_coord, min_coord])))
    # North wall
    walls.append((np.array([min_coord, max_coord]), np.array([max_coord, max_coord])))
    # West wall
    walls.append((np.array([min_coord, min_coord]), np.array([min_coord, max_coord])))
    # East wall
    walls.append((np.array([max_coord, min_coord]), np.array([max_coord, max_coord])))
    
    # Room corners as landmarks
    landmarks_list.extend([
        [min_coord, min_coord],  # SW
        [max_coord, min_coord],  # SE
        [max_coord, max_coord],  # NE
        [min_coord, max_coord],  # NW
    ])
    
    # =========================================================================
    # Internal features to break symmetry (CRITICAL for loop closure!)
    # Each corner should look different in range histograms
    # =========================================================================
    
    # Feature 1: Pillar near SW corner at (1.5, 1.5) - makes SW distinctive
    pillar1 = [
        (np.array([1.5, 1.5]), np.array([2.2, 1.5])),  # South side
        (np.array([2.2, 1.5]), np.array([2.2, 2.2])),  # East side
        (np.array([2.2, 2.2]), np.array([1.5, 2.2])),  # North side
        (np.array([1.5, 2.2]), np.array([1.5, 1.5])),  # West side
    ]
    walls.extend(pillar1)
    landmarks_list.extend([[1.5, 1.5], [2.2, 2.2]])  # Pillar corners
    
    # Feature 2: Off-center pillar at (2.0, 5.0) - breaks N-S symmetry
    pillar2 = [
        (np.array([1.8, 4.8]), np.array([2.5, 4.8])),
        (np.array([2.5, 4.8]), np.array([2.5, 5.5])),
        (np.array([2.5, 5.5]), np.array([1.8, 5.5])),
        (np.array([1.8, 5.5]), np.array([1.8, 4.8])),
    ]
    walls.extend(pillar2)
    landmarks_list.extend([[1.8, 4.8], [2.5, 5.5]])  # Pillar corners
    
    # Feature 3: L-shaped obstacle near NE corner - makes NE distinctive
    l_shape = [
        (np.array([5.8, 5.8]), np.array([6.8, 5.8])),  # Horizontal
        (np.array([6.8, 5.8]), np.array([6.8, 7.2])),  # Vertical
    ]
    walls.extend(l_shape)
    landmarks_list.extend([[5.8, 5.8], [6.8, 7.2]])
    
    # Feature 4: Short wall segment on west side - makes west side distinctive
    walls.append((np.array([0.8, 3.5]), np.array([2.0, 3.5])))
    landmarks_list.extend([[0.8, 3.5], [2.0, 3.5]])
    
    # Feature 5: Obstacle near SE corner - makes SE distinctive
    se_obstacle = [
        (np.array([5.5, 0.8]), np.array([7.0, 0.8])),
        (np.array([7.0, 0.8]), np.array([7.0, 2.0])),
    ]
    walls.extend(se_obstacle)
    landmarks_list.extend([[5.5, 0.8], [7.0, 2.0]])
    
    landmarks = np.array(landmarks_list)
    
    return walls, landmarks


def create_corridor_walls(
    length: float = 20.0,
    width: float = 4.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create wall segments forming a closed rectangular corridor.
    
    The corridor runs along the X-axis with:
    - Side walls at y=+width/2 and y=-width/2
    - End walls at x=0 and x=length
    
    The end walls break translational symmetry, making scans at different
    X positions distinguishable. This enables reliable ICP matching.
    
    Args:
        length: Corridor length in meters (X direction).
        width: Corridor width in meters (Y direction).
    
    Returns:
        List of (start_point, end_point) tuples defining walls.
    """
    half_width = width / 2
    
    walls = [
        # Bottom wall (y = -half_width)
        (np.array([0.0, -half_width]), np.array([length, -half_width])),
        # Top wall (y = +half_width)
        (np.array([0.0, half_width]), np.array([length, half_width])),
        # Left end wall (x = 0)
        (np.array([0.0, -half_width]), np.array([0.0, half_width])),
        # Right end wall (x = length)
        (np.array([length, -half_width]), np.array([length, half_width])),
    ]
    
    return walls


def generate_dense_wall_scan(
    pose: np.ndarray,
    walls: List[Tuple[np.ndarray, np.ndarray]],
    max_range: float = 8.0,
    noise_std: float = 0.02,
    points_per_wall: int = 50,
) -> np.ndarray:
    """
    Generate dense LiDAR scan from walls in the environment.
    
    Creates realistic, dense point clouds suitable for ICP scan matching.
    
    Args:
        pose: Robot pose [x, y, yaw].
        walls: List of (start_point, end_point) tuples defining wall segments.
        max_range: Maximum sensor range in meters.
        noise_std: Standard deviation of measurement noise.
        points_per_wall: Number of points to sample per wall segment.
    
    Returns:
        Point cloud in robot local frame, shape (M, 2).
    """
    x, y, yaw = pose
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    all_points = []
    
    for wall_start, wall_end in walls:
        # Generate dense points along the wall
        t = np.linspace(0, 1, points_per_wall)
        wall_points = wall_start + np.outer(t, wall_end - wall_start)
        
        # Transform to robot frame
        diff = wall_points - np.array([x, y])
        x_local = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
        y_local = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
        
        # Filter by range
        ranges = np.sqrt(x_local**2 + y_local**2)
        valid = ranges < max_range
        
        if np.any(valid):
            local_points = np.column_stack([x_local[valid], y_local[valid]])
            all_points.append(local_points)
    
    if not all_points:
        return np.zeros((0, 2))
    
    scan = np.vstack(all_points)
    
    # Add measurement noise
    if noise_std > 0:
        scan += np.random.normal(0, noise_std, scan.shape)
    
    return scan


def generate_scan_from_pose(
    pose: np.ndarray,
    landmarks: np.ndarray,
    max_range: float = 15.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate synthetic 2D LiDAR scan from a robot pose (legacy sparse version).

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
    use_observation_based: bool = True,
    distance_threshold: Optional[float] = None,
    min_time_separation: int = 10,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Detect loop closures using observation-based descriptor matching or distance-based oracle.

    When the robot returns to a previously visited location, loop closures
    enforce the close-loop constraint from Eq. (7.22):
        residual = ln((ΔT_ij')^{-1} T_i^{-1} T_j)^∨
    
    where ΔT_ij' is the scan-matched transform from ICP, and T_i^{-1} T_j
    is the transform implied by the odometry chain.

    Args:
        poses: List of poses (possibly with drift).
        scans: List of scans in local frame.
        use_observation_based: If True, use descriptor-based candidate selection (PRIMARY).
                              If False, use distance-based oracle (LEGACY, for comparison).
        distance_threshold: Optional maximum distance for secondary filtering.
                           If use_observation_based=False, this is the PRIMARY filter.
        min_time_separation: Minimum time steps between poses for loop closure.

    Returns:
        List of tuples (pose_i, pose_j, relative_pose, covariance) for each closure.
        Each tuple contains:
            - pose_i: Earlier pose index (T_i)
            - pose_j: Later pose index (T_j)
            - relative_pose: Scan-matched transform ΔT_ij' from ICP
            - covariance: Uncertainty in the scan matching

    References:
        - Section 7.3.5: Close-loop Constraints
        - Eq. (7.22): Close-loop constraint formulation
    """
    if use_observation_based:
        # Observation-based detection using scan descriptors
        # Use strict ICP residual threshold to reject bad alignments
        # For corridor environment, residuals above 0.1 often indicate wrong local minima
        print("  Loop closure: candidate from descriptor similarity (observation-based)")
        print("  Using frontend_poses for initial guess (scan-to-map corrected trajectory)")
        print()
        
        detector = LoopClosureDetector2D(
            n_bins=32,
            max_range=10.0,
            min_time_separation=min_time_separation,
            min_descriptor_similarity=0.80,  # Stricter to reduce false positives
            max_candidates=10,  # Fewer candidates, higher quality
            max_distance=distance_threshold,  # Optional secondary filter
            max_icp_residual=0.1,  # Strict to reject wrong alignments
            icp_max_iterations=50,
            icp_tolerance=1e-4,
        )
        
        loop_closures_obj = detector.detect(scans, poses)
        
        # Convert to old format
        loop_closures = []
        for lc in loop_closures_obj:
            loop_closures.append((lc.j, lc.i, lc.rel_pose, lc.covariance))
            print(f"  Verified: {lc.j} <-> {lc.i}, "
                  f"desc_sim={lc.descriptor_similarity:.3f}, "
                  f"icp_residual={lc.icp_residual:.4f}, iters={lc.icp_iterations}")
        
        return loop_closures
    
    else:
        # LEGACY: Distance-based oracle detection (for debugging/comparison ONLY)
        # This mode is NOT realistic - in real SLAM, you don't know ground truth positions
        print()
        print("  " + "=" * 60)
        print("  [WARNING] USING DISTANCE-BASED ORACLE (NOT REALISTIC!)")
        print("  " + "=" * 60)
        print("  This mode uses ground truth position knowledge to find loop")
        print("  closure candidates. Real SLAM systems cannot do this!")
        print("  Use observation-based detection (default) for realistic behavior.")
        print("  " + "=" * 60)
        print()
        
        loop_closures = []
        n_poses = len(poses)
        
        if distance_threshold is None:
            distance_threshold = 3.0  # Default for legacy mode
        
        for i in range(n_poses):
            for j in range(i + min_time_separation, n_poses):
                # Check distance (ORACLE - uses position knowledge!)
                dist = np.linalg.norm(poses[i][:2] - poses[j][:2])
                
                if dist < distance_threshold:
                    # Potential loop closure - verify with ICP
                    try:
                        # Initial guess: transform from pose_i to pose_j
                        initial_guess = se2_relative(poses[i], poses[j])
                        
                        # ICP(source=scans[i], target=scans[j]) returns transform from i to j
                        rel_pose, iters, residual, converged = icp_point_to_point(
                            source_scan=scans[i],
                            target_scan=scans[j],
                            initial_pose=initial_guess,
                            max_iterations=50,
                            tolerance=1e-4,
                        )
                        
                        if converged and residual < 1.0:
                            # Compute covariance (simplified)
                            cov = np.diag([0.05, 0.05, 0.01])
                            
                            # Return (from_id=i, to_id=j, rel_pose=i_to_j)
                            loop_closures.append((i, j, rel_pose, cov))
                            
                            print(f"  Loop closure: {i} <-> {j}, residual={residual:.4f}, iters={iters}")
                    
                    except Exception:
                        # ICP failed, skip this pair
                        pass
        
        return loop_closures


def build_map_from_poses(
    poses: List[np.ndarray],
    scans: List[np.ndarray],
    downsample_voxel: float = 0.2,
) -> np.ndarray:
    """
    Build a map point cloud by transforming all scans using given poses.
    
    Args:
        poses: List of SE(2) poses [x, y, theta].
        scans: List of scan point clouds (Nx2 arrays).
        downsample_voxel: Voxel size for downsampling (0 = no downsampling).
        
    Returns:
        Map point cloud as Nx2 array.
    """
    from core.slam import se2_apply
    
    all_points = []
    for pose, scan in zip(poses, scans):
        if len(scan) > 0:
            # Transform scan to global frame
            transformed = se2_apply(pose, scan)
            all_points.append(transformed)
    
    if not all_points:
        return np.zeros((0, 2))
    
    map_points = np.vstack(all_points)
    
    # Optional voxel grid downsampling
    if downsample_voxel > 0:
        # Quantize to voxels
        voxel_indices = np.floor(map_points / downsample_voxel).astype(int)
        # Find unique voxels and compute centroids
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        downsampled = np.zeros((len(unique_voxels), 2))
        for i in range(len(unique_voxels)):
            mask = inverse == i
            downsampled[i] = np.mean(map_points[mask], axis=0)
        return downsampled
    
    return map_points


def create_slam_animation(
    true_poses: List[np.ndarray],
    odom_poses: List[np.ndarray],
    frontend_poses: List[np.ndarray],
    optimized_poses: List[np.ndarray],
    scans: List[np.ndarray],
    loop_closures: List[Tuple],
    trajectory_type: str = "square",
    output_path: str = "ch7_slam/figs/slam_pipeline_square.gif",
    fps: int = 5,
) -> None:
    """Create animated GIF showing SLAM pipeline evolution.
    
    Shows three panels:
    1. Map + Trajectory: map growing, trajectories evolving, current pose
    2. Constraint Graph: odometry edges (gray) + loop closure edges (magenta)
    3. Error: Position error vs time with loop closure markers
    
    After front-end completes, shows optimization correction animation.
    
    Args:
        true_poses: Ground truth trajectory
        odom_poses: Noisy odometry trajectory
        frontend_poses: Front-end corrected trajectory
        optimized_poses: Backend optimized trajectory
        scans: LiDAR scans at each pose
        loop_closures: List of (i, j, rel_pose, cov) tuples
        trajectory_type: "square" or "corridor"
        output_path: Path to save GIF
        fps: Frames per second
    """
    from core.slam import se2_apply
    
    n_poses = len(true_poses)
    
    # Compute errors at each timestep
    odom_errors = [np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) 
                   for i in range(n_poses)]
    frontend_errors = [np.linalg.norm(frontend_poses[i][:2] - true_poses[i][:2]) 
                       for i in range(n_poses)]
    optimized_errors = [np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2]) 
                        for i in range(n_poses)]
    
    # Find when each loop closure is detected (use the larger index)
    loop_closure_times = {max(lc[0], lc[1]): (lc[0], lc[1]) for lc in loop_closures}
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"SLAM Pipeline Animation ({trajectory_type} trajectory)", fontsize=14)
    
    ax1 = fig.add_subplot(1, 3, 1)  # Map + Trajectory
    ax2 = fig.add_subplot(1, 3, 2)  # Constraint graph
    ax3 = fig.add_subplot(1, 3, 3)  # Error plot
    
    # Compute bounds for consistent axes
    all_x = [p[0] for p in true_poses]
    all_y = [p[1] for p in true_poses]
    x_margin = (max(all_x) - min(all_x)) * 0.2
    y_margin = (max(all_y) - min(all_y)) * 0.2
    xlim = (min(all_x) - x_margin - 2, max(all_x) + x_margin + 2)
    ylim = (min(all_y) - y_margin - 2, max(all_y) + y_margin + 2)
    
    # Initialize plot elements
    # Panel 1: Map + Trajectory
    map_scatter = ax1.scatter([], [], s=1, c='blue', alpha=0.3, label='Map')
    current_scan_scatter = ax1.scatter([], [], s=10, c='red', alpha=0.8, label='Current scan')
    true_line, = ax1.plot([], [], 'g-', linewidth=1, alpha=0.5, label='Ground truth')
    odom_line, = ax1.plot([], [], 'r--', linewidth=1, alpha=0.7, label='Odometry')
    frontend_line, = ax1.plot([], [], 'b-', linewidth=2, label='Frontend')
    current_pose_marker, = ax1.plot([], [], 'ko', markersize=8, markerfacecolor='yellow')
    correction_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                                    arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                                    visible=False)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Map + Trajectory')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Constraint graph
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Pose Graph Constraints')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Error plot
    ax3.set_xlim(0, n_poses)
    ax3.set_ylim(0, max(odom_errors) * 1.2)
    ax3.set_xlabel('Pose Index')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error vs Time')
    ax3.grid(True, alpha=0.3)
    
    odom_error_line, = ax3.plot([], [], 'r-', label='Odometry', alpha=0.7)
    frontend_error_line, = ax3.plot([], [], 'b-', label='Frontend', linewidth=2)
    ax3.legend(loc='upper left', fontsize=8)
    
    # Storage for accumulated map points
    accumulated_map = []
    odom_edges = []
    loop_edges = []
    
    # Phase 1: Front-end frames (one per pose)
    # Phase 2: Optimization frames (interpolate frontend -> optimized)
    n_opt_frames = 20  # Frames for optimization animation
    total_frames = n_poses + n_opt_frames
    
    def init():
        return []
    
    def update(frame):
        artists = []
        
        if frame < n_poses:
            # Phase 1: Front-end animation
            i = frame
            
            # Accumulate map points
            if i < len(scans):
                scan_global = se2_apply(frontend_poses[i], scans[i])
                accumulated_map.append(scan_global)
            
            # Update map scatter
            if accumulated_map:
                all_map = np.vstack(accumulated_map)
                # Downsample for performance
                if len(all_map) > 2000:
                    indices = np.random.choice(len(all_map), 2000, replace=False)
                    all_map = all_map[indices]
                map_scatter.set_offsets(all_map)
            
            # Update current scan (highlighted)
            if i < len(scans):
                scan_global = se2_apply(frontend_poses[i], scans[i])
                current_scan_scatter.set_offsets(scan_global)
            
            # Update trajectories (up to current pose)
            true_xy = np.array([[p[0], p[1]] for p in true_poses[:i+1]])
            odom_xy = np.array([[p[0], p[1]] for p in odom_poses[:i+1]])
            frontend_xy = np.array([[p[0], p[1]] for p in frontend_poses[:i+1]])
            
            if len(true_xy) > 0:
                true_line.set_data(true_xy[:, 0], true_xy[:, 1])
                odom_line.set_data(odom_xy[:, 0], odom_xy[:, 1])
                frontend_line.set_data(frontend_xy[:, 0], frontend_xy[:, 1])
            
            # Update current pose marker
            current_pose_marker.set_data([frontend_poses[i][0]], [frontend_poses[i][1]])
            
            # Update error plot
            odom_error_line.set_data(range(i+1), odom_errors[:i+1])
            frontend_error_line.set_data(range(i+1), frontend_errors[:i+1])
            
            # Draw odometry edge (constraint)
            if i > 0:
                ax2.plot([frontend_poses[i-1][0], frontend_poses[i][0]],
                        [frontend_poses[i-1][1], frontend_poses[i][1]],
                        'gray', linewidth=1, alpha=0.5)
            
            # Check for loop closure at this timestep
            if i in loop_closure_times:
                lc_i, lc_j = loop_closure_times[i]
                # Draw loop closure edge (thick magenta)
                line = ax2.plot([frontend_poses[lc_i][0], frontend_poses[lc_j][0]],
                               [frontend_poses[lc_i][1], frontend_poses[lc_j][1]],
                               'magenta', linewidth=3, alpha=0.8)[0]
                loop_edges.append(line)
                
                # Mark on error plot
                ax3.axvline(x=i, color='magenta', linestyle='--', alpha=0.5)
            
            # Update title with progress
            ax1.set_title(f'Map + Trajectory (pose {i+1}/{n_poses})')
            
        else:
            # Phase 2: Optimization animation
            opt_frame = frame - n_poses
            alpha = opt_frame / (n_opt_frames - 1)  # 0 to 1
            
            # Interpolate from frontend to optimized
            interp_poses = []
            for j in range(n_poses):
                interp_pose = (1 - alpha) * frontend_poses[j] + alpha * optimized_poses[j]
                interp_poses.append(interp_pose)
            
            # Rebuild map with interpolated poses
            interp_map = []
            for j in range(n_poses):
                if j < len(scans):
                    scan_global = se2_apply(interp_poses[j], scans[j])
                    interp_map.append(scan_global)
            
            if interp_map:
                all_map = np.vstack(interp_map)
                if len(all_map) > 2000:
                    indices = np.random.choice(len(all_map), 2000, replace=False)
                    all_map = all_map[indices]
                map_scatter.set_offsets(all_map)
            
            # Update trajectories
            interp_xy = np.array([[p[0], p[1]] for p in interp_poses])
            frontend_line.set_data(interp_xy[:, 0], interp_xy[:, 1])
            
            # Hide current scan marker
            current_scan_scatter.set_offsets(np.empty((0, 2)))
            current_pose_marker.set_data([], [])
            
            # Update error (interpolate)
            interp_errors = [(1 - alpha) * frontend_errors[j] + alpha * optimized_errors[j] 
                            for j in range(n_poses)]
            frontend_error_line.set_data(range(n_poses), interp_errors)
            
            # Update title
            if opt_frame == 0:
                ax1.set_title('Optimization: Initial')
                ax2.set_title('Pose Graph (optimizing...)')
            elif opt_frame == n_opt_frames - 1:
                ax1.set_title('Optimization: Complete!')
                ax2.set_title(f'Pose Graph ({len(loop_closures)} loop closures)')
                
                # Add optimized trajectory as dashed line
                opt_xy = np.array([[p[0], p[1]] for p in optimized_poses])
                ax1.plot(opt_xy[:, 0], opt_xy[:, 1], 'c--', linewidth=2, 
                        label='Optimized', alpha=0.8)
                ax1.legend(loc='upper right', fontsize=8)
            else:
                progress = int(alpha * 100)
                ax1.set_title(f'Optimization: {progress}%')
        
        return artists
    
    # Create animation
    print(f"   Creating animation with {total_frames} frames...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init,
                        interval=1000//fps, blit=False)
    
    # Save as GIF
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Saving GIF to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"[OK] Saved animation: {output_path}")


def plot_slam_results(
    true_poses: List[np.ndarray],
    odom_poses: List[np.ndarray],
    frontend_poses: List[np.ndarray],
    optimized_poses: List[np.ndarray],
    landmarks: np.ndarray,
    loop_closures: List[Tuple[int, int, np.ndarray, np.ndarray]],
    scans: Optional[List[np.ndarray]] = None,
):
    """
    Visualize SLAM results: trajectories, maps, and errors.

    Args:
        true_poses: Ground truth poses.
        odom_poses: Odometry-only poses (with drift).
        frontend_poses: Front-end estimates (scan-to-map corrected).
        optimized_poses: Optimized poses after pose graph optimization.
        landmarks: Environment landmarks.
        loop_closures: Detected loop closures for visualization.
        scans: Optional list of LiDAR scans for map visualization.
    """
    # Determine layout based on whether we have scans for map visualization
    if scans is not None and len(scans) > 0:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
        ax_traj = fig.add_subplot(gs[:, 0])  # Trajectories (full height, left)
        ax_map_before = fig.add_subplot(gs[0, 1])  # Map before (top middle)
        ax_map_after = fig.add_subplot(gs[1, 1])   # Map after (bottom middle)
        ax_error = fig.add_subplot(gs[:, 2])       # Errors (full height, right)
        axes = [ax_traj, ax_map_before, ax_map_after, ax_error]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax_traj = axes[0]
        ax_error = axes[1]

    # --- Plot 1: Trajectories ---
    # Plot landmarks
    ax_traj.scatter(
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
    ax_traj.plot(
        true_xy[:, 0],
        true_xy[:, 1],
        "g-",
        linewidth=2,
        label="Ground Truth",
        alpha=0.7,
    )
    ax_traj.scatter(true_xy[0, 0], true_xy[0, 1], c="green", marker="o", s=100, zorder=5)

    # Plot odometry (with drift)
    odom_xy = np.array([[p[0], p[1]] for p in odom_poses])
    ax_traj.plot(
        odom_xy[:, 0],
        odom_xy[:, 1],
        "r--",
        linewidth=2,
        label="Odometry (Drift)",
        alpha=0.7,
    )

    # Plot front-end (scan-to-map corrected)
    frontend_xy = np.array([[p[0], p[1]] for p in frontend_poses])
    ax_traj.plot(
        frontend_xy[:, 0],
        frontend_xy[:, 1],
        "orange",
        linewidth=1.5,
        linestyle="-.",
        label="Front-end (Scan-to-Map)",
        alpha=0.7,
    )

    # Plot optimized
    opt_xy = np.array([[p[0], p[1]] for p in optimized_poses])
    ax_traj.plot(
        opt_xy[:, 0],
        opt_xy[:, 1],
        "b-",
        linewidth=2,
        label="Optimized (Backend)",
        alpha=0.8,
    )
    ax_traj.scatter(opt_xy[0, 0], opt_xy[0, 1], c="blue", marker="o", s=100, zorder=5)

    # Plot loop closures
    for i, j, _, _ in loop_closures:
        ax_traj.plot(
            [odom_xy[i, 0], odom_xy[j, 0]],
            [odom_xy[i, 1], odom_xy[j, 1]],
            "m:",
            linewidth=1,
            alpha=0.5,
        )

    ax_traj.set_xlabel("X [m]", fontsize=12)
    ax_traj.set_ylabel("Y [m]", fontsize=12)
    ax_traj.set_title("4 Trajectories: Truth / Odom / Front-end / Optimized", 
                      fontsize=13, fontweight="bold")
    ax_traj.legend(fontsize=10)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis("equal")

    # --- Plot 2 & 3: Map Point Clouds (if scans provided) ---
    if scans is not None and len(scans) > 0:
        print("   Building map point clouds...")
        
        # Build map from front-end poses (before backend optimization)
        map_before = build_map_from_poses(frontend_poses, scans, downsample_voxel=0.15)
        
        # Build map from optimized poses (after backend optimization)
        map_after = build_map_from_poses(optimized_poses, scans, downsample_voxel=0.15)
        
        print(f"   Map before (front-end): {len(map_before)} points")
        print(f"   Map after (backend):    {len(map_after)} points")
        
        # Plot map before optimization
        ax_map_before.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            c="gray",
            marker="x",
            s=30,
            alpha=0.3,
            label="Landmarks",
        )
        if len(map_before) > 0:
            ax_map_before.scatter(
                map_before[:, 0],
                map_before[:, 1],
                c="orange",
                s=1,
                alpha=0.3,
                label="Map Points (Front-end)",
            )
        ax_map_before.plot(frontend_xy[:, 0], frontend_xy[:, 1], "orange", 
                          linestyle="-.", linewidth=1.5, alpha=0.6)
        ax_map_before.scatter(frontend_xy[0, 0], frontend_xy[0, 1], c="orange", 
                             marker="o", s=80, zorder=5)
        ax_map_before.set_xlabel("X [m]", fontsize=12)
        ax_map_before.set_ylabel("Y [m]", fontsize=12)
        ax_map_before.set_title("Map Before Backend (Front-end)", fontsize=13, fontweight="bold")
        ax_map_before.legend(fontsize=9, loc="upper right")
        ax_map_before.grid(True, alpha=0.3)
        ax_map_before.axis("equal")
        
        # Plot map after optimization
        ax_map_after.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            c="gray",
            marker="x",
            s=30,
            alpha=0.3,
            label="Landmarks",
        )
        if len(map_after) > 0:
            ax_map_after.scatter(
                map_after[:, 0],
                map_after[:, 1],
                c="blue",
                s=1,
                alpha=0.3,
                label="Map Points (Optimized)",
            )
        ax_map_after.plot(opt_xy[:, 0], opt_xy[:, 1], "b-", linewidth=1.5, alpha=0.6)
        ax_map_after.scatter(opt_xy[0, 0], opt_xy[0, 1], c="blue", marker="o", s=80, zorder=5)
        
        # Add loop closure markers
        for i, j, _, _ in loop_closures:
            ax_map_after.plot(
                [opt_xy[i, 0], opt_xy[j, 0]],
                [opt_xy[i, 1], opt_xy[j, 1]],
                "m:",
                linewidth=1.5,
                alpha=0.6,
            )
        
        ax_map_after.set_xlabel("X [m]", fontsize=12)
        ax_map_after.set_ylabel("Y [m]", fontsize=12)
        ax_map_after.set_title("Map After Backend (Optimized)", fontsize=13, fontweight="bold")
        ax_map_after.legend(fontsize=9, loc="upper right")
        ax_map_after.grid(True, alpha=0.3)
        ax_map_after.axis("equal")

    # --- Plot 4 (or 2): Position Errors ---

    # Compute position errors
    odom_errors = np.array(
        [np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(len(true_poses))]
    )
    frontend_errors = np.array(
        [np.linalg.norm(frontend_poses[i][:2] - true_poses[i][:2]) for i in range(len(true_poses))]
    )
    opt_errors = np.array(
        [
            np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2])
            for i in range(len(true_poses))
        ]
    )

    timesteps = np.arange(len(true_poses))

    ax_error.plot(
        timesteps, odom_errors, "r--", linewidth=2, label="Odometry Error", alpha=0.7
    )
    ax_error.plot(
        timesteps, frontend_errors, "orange", linestyle="-.", linewidth=1.5, 
        label="Front-end Error", alpha=0.7
    )
    ax_error.plot(
        timesteps, opt_errors, "b-", linewidth=2, label="Optimized Error", alpha=0.8
    )

    # Mark loop closures
    for i, j, _, _ in loop_closures:
        ax_error.axvline(j, color="magenta", linestyle=":", alpha=0.5, linewidth=1)

    ax_error.set_xlabel("Pose Index", fontsize=12)
    ax_error.set_ylabel("Position Error [m]", fontsize=12)
    ax_error.set_title("Position Error Over Time", fontsize=14, fontweight="bold")
    ax_error.legend(fontsize=10)
    ax_error.grid(True, alpha=0.3)

    # Add main title describing the complete pipeline
    fig.suptitle(
        "Complete SLAM Pipeline: Odometry → Front-end (Scan-to-Map ICP) → "
        "Loop Closure → Backend Optimization",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    # Save to figs directory with deterministic filename
    from pathlib import Path
    figs_dir = Path("ch7_slam/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_file = figs_dir / "slam_with_maps.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved figure: {output_file}")
    
    # Only show interactively if not in automated mode
    import os
    if os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") != "Agg":
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass  # Silently skip if display not available


def run_with_inline_data(
    use_loop_oracle: bool = False,
    trajectory_type: str = "square",
    n_laps: int = 2,
    animate: bool = False,
):
    """Run complete pose graph SLAM example with inline data.
    
    Args:
        use_loop_oracle: If True, use distance-based oracle instead of observation-based.
                        Default is False (observation-based).
        trajectory_type: "square" (default) or "corridor".
        n_laps: Number of laps for square trajectory (default: 2).
        animate: If True, generate animated GIF showing SLAM pipeline.
    
    This mode generates:
    - A square loop trajectory with multiple laps (for loop closure)
    - Dense wall scans (suitable for scan-to-map matching)
    - Moderate odometry drift (correctable by SLAM)
    
    Demonstrates the full SLAM pipeline:
    - Front-end: Prediction -> Scan-to-Map -> Map Update
    - Loop closure: Observation-based detection
    - Back-end: Pose graph optimization
    """
    print("=" * 70)
    print("CHAPTER 7: 2D POSE GRAPH SLAM EXAMPLE")
    print("(Full SLAM pipeline with dense wall scans)")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    side_length = 8.0  # meters

    # ------------------------------------------------------------------------
    # 1. Generate Ground Truth Trajectory
    # ------------------------------------------------------------------------
    if trajectory_type == "square":
        print(f"1. Generating square loop trajectory...")
        print(f"   Trajectory: square, laps: {n_laps}")
        true_poses = generate_square_loop_trajectory(
            side_length=side_length,
            n_poses_per_side=12,
            n_laps=n_laps,
        )
        n_poses = len(true_poses)
        
        # Verify closed loop
        start_pose = true_poses[0]
        end_pose = true_poses[-1]
        closure_dist = np.linalg.norm(end_pose[:2] - start_pose[:2])
        closure_yaw = abs(end_pose[2] - start_pose[2])
        print(f"   Generated {n_poses} poses (square, {n_laps} laps)")
        print(f"   Loop closure check: dist={closure_dist:.3f}m, yaw={np.degrees(closure_yaw):.1f}deg")
        
        # Compute trajectory bounds
        all_x = [p[0] for p in true_poses]
        all_y = [p[1] for p in true_poses]
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        print(f"   Trajectory bounds: x_range={x_range:.2f}m, y_range={y_range:.2f}m")
        
        # Create room walls with asymmetric features
        print("\n2. Creating room environment...")
        walls, landmarks = create_room_walls(side_length=side_length, wall_offset=1.5)
        print(f"   Created {len(walls)} wall segments (room with asymmetric features)")
        print(f"   Landmarks: {len(landmarks)} points (derived from wall endpoints)")
        
    else:  # corridor
        print(f"1. Generating corridor loop trajectory...")
        print(f"   Trajectory: corridor (out-and-back)")
        true_poses = generate_corridor_loop_trajectory(
            corridor_length=15.0, n_poses_out=30, n_poses_back=30
        )
        n_poses = len(true_poses)
        print(f"   Generated {n_poses} poses (corridor out-and-back)")
        
        # Compute trajectory bounds
        all_x = [p[0] for p in true_poses]
        all_y = [p[1] for p in true_poses]
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        print(f"   Trajectory bounds: x_range={x_range:.2f}m, y_range={y_range:.2f}m")
        
        # Create corridor walls
        print("\n2. Creating corridor environment...")
        walls = create_corridor_walls(length=20.0, width=4.0)
        print(f"   Created {len(walls)} wall segments (parallel walls)")
        
        # Landmarks at corners for visualization
        landmarks = np.array([
            [-1.0, -2.0], [20.0, -2.0], [20.0, 2.0], [-1.0, 2.0],
        ])

    # ------------------------------------------------------------------------
    # 3. Simulate Odometry with Moderate Drift
    # ------------------------------------------------------------------------
    print("\n3. Simulating odometry with drift...")
    # Significant noise - creates visible drift that SLAM can correct
    odom_poses = add_odometry_noise(
        true_poses, translation_noise=0.06, rotation_noise=0.015
    )
    
    # Compute drift statistics
    initial_drift = np.linalg.norm(odom_poses[0][:2] - true_poses[0][:2])
    final_drift = np.linalg.norm(odom_poses[-1][:2] - true_poses[-1][:2])
    print(f"   Initial drift: {initial_drift:.3f} m")
    print(f"   Accumulated drift (without SLAM): {final_drift:.3f} m")

    # ------------------------------------------------------------------------
    # 4. Generate Dense LiDAR Scans (from TRUE poses - sensor reality!)
    # ------------------------------------------------------------------------
    # DO NOT generate scans from odom/estimates - that would be circular/invalid!
    # Scans MUST come from true_poses (simulating real sensor observations).
    print("\n4. Generating dense LiDAR scans from true robot positions...")
    scans = []
    for i, pose in enumerate(true_poses):
        scan = generate_dense_wall_scan(
            pose, walls, 
            max_range=6.0,      # Reasonable range
            noise_std=0.02,     # Low noise for reliable ICP
            points_per_wall=20  # Moderate density (room has many walls)
        )
        scans.append(scan)
    avg_points = np.mean([len(s) for s in scans])
    print(f"   Generated {n_poses} scans (avg {avg_points:.0f} points/scan)")
    print(f"   Note: Scans from TRUE poses (sensor reality), NOT from odometry!")

    # ------------------------------------------------------------------------
    # 5. SLAM Front-End: init -> predict -> scan-to-map -> update map
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Front-end: init -> predict -> scan-to-map -> update map")
    print("-" * 70)
    
    # Initialize front-end with submap for scan-to-map alignment
    # Use first odometry pose as initial pose (trajectory starts at origin for square)
    frontend = SlamFrontend2D(
        submap_voxel_size=0.2,   # Voxel size for map downsampling
        min_map_points=5,        # Minimum points needed for ICP
        max_icp_residual=1.5,    # Accept ICP results with reasonable residual
        initial_pose=odom_poses[0].copy(),  # Start from odometry's first pose
    )
    
    # Run front-end for each timestep
    frontend_poses = []
    pred_poses = []
    match_qualities = []
    corrections = []
    
    for i in range(n_poses):
        # Compute odometry delta from noisy odometry poses
        if i == 0:
            odom_delta = np.array([0.0, 0.0, 0.0])
        else:
            odom_delta = se2_relative(odom_poses[i - 1], odom_poses[i])
        
        # Run front-end step: predict -> scan-to-map ICP -> update map
        result = frontend.step(i, odom_delta, scans[i])
        
        # Store results
        frontend_poses.append(result['pose_est'])
        pred_poses.append(result['pose_pred'])
        match_qualities.append(result['match_quality'])
        
        # Compute correction magnitude (difference between prediction and estimate)
        if i > 0:
            correction_xy = np.linalg.norm(
                result['pose_est'][:2] - result['pose_pred'][:2]
            )
            corrections.append(correction_xy)
    
    # Compute front-end statistics
    n_converged = sum(1 for mq in match_qualities if mq.converged)
    converged_qualities = [mq for mq in match_qualities if mq.converged]
    avg_residual = np.mean([mq.residual for mq in converged_qualities]) if converged_qualities else 0.0
    avg_correction = np.mean(corrections) if corrections else 0.0
    
    print(f"\n  Processed {n_poses} steps")
    print(f"  Frontend converged ratio: {100*n_converged/n_poses:.1f}%")
    print(f"  Frontend avg residual: {avg_residual:.4f} m")
    print(f"  Frontend avg correction: {avg_correction:.4f} m")
    
    # Verify frontend_poses differs from odom_poses (NOT identical)
    max_trans_diff = 0.0
    max_yaw_diff = 0.0
    for i in range(n_poses):
        trans_diff = np.linalg.norm(frontend_poses[i][:2] - odom_poses[i][:2])
        yaw_diff = abs(frontend_poses[i][2] - odom_poses[i][2])
        max_trans_diff = max(max_trans_diff, trans_diff)
        max_yaw_diff = max(max_yaw_diff, yaw_diff)
    
    print(f"\n  max|frontend - odom| translation: {max_trans_diff:.4f} m")
    print(f"  max|frontend - odom| yaw: {np.degrees(max_yaw_diff):.2f} deg")
    
    if max_trans_diff < 1e-6 and max_yaw_diff < 1e-6:
        print("  [WARNING] Frontend poses identical to odometry - ICP not working!")

    # ------------------------------------------------------------------------
    # 6. Prepare Odometry Measurements from Front-End
    # ------------------------------------------------------------------------
    print("\n6. Preparing odometry measurements from front-end estimates...")
    odometry_measurements = []
    for i in range(n_poses - 1):
        odom_delta = se2_relative(frontend_poses[i], frontend_poses[i + 1])
        odometry_measurements.append((i, i + 1, odom_delta))
    print(f"   Prepared {len(odometry_measurements)} odometry measurements")

    # ------------------------------------------------------------------------
    # 7. Detect Loop Closures
    # ------------------------------------------------------------------------
    if use_loop_oracle:
        print("\n7. Detecting loop closures (ORACLE MODE - distance-based)...")
    else:
        print("\n7. Detecting loop closures (observation-based)...")
    
    loop_closures = detect_loop_closures(
        frontend_poses,  # Use front-end estimates as initial guess
        scans,
        use_observation_based=not use_loop_oracle,  # Default: observation-based
        distance_threshold=None,  # No distance gating by default
        min_time_separation=10
    )
    print(f"   Detected {len(loop_closures)} loop closures")

    # ------------------------------------------------------------------------
    # 8. Build Pose Graph
    # ------------------------------------------------------------------------
    print("\n8. Building pose graph...")
    
    # CRITICAL: Initialize graph from front-end poses (NOT odometry, NOT ground truth)
    # This ensures the front-end's scan-to-map corrections are preserved
    initial_poses = frontend_poses  # Must be frontend_poses, not odom_poses!
    print(f"   Graph initial: frontend_poses (scan-to-map corrected trajectory)")

    # Prepare loop closure measurements
    loop_measurements = []
    loop_info_matrices = []
    for i, j, rel_pose, cov in loop_closures:
        loop_measurements.append((i, j, rel_pose))
        loop_info_matrices.append(np.linalg.inv(cov))

    # Odometry information (moderate uncertainty)
    odom_info = np.linalg.inv(np.diag([0.1, 0.1, 0.02]))

    # Loop closure information - balanced weight with odometry
    loop_info = np.linalg.inv(np.diag([0.1, 0.1, 0.02]))  # Same weight as odometry

    # Create pose graph with front-end estimates as initial values
    # CRITICAL: Set prior to actual first pose (trajectory may not start at origin)
    first_pose = initial_poses[0].copy()
    graph = create_pose_graph(
        poses=initial_poses,  # Initial values from front-end (NOT odometry!)
        odometry_measurements=odometry_measurements,
        loop_closures=loop_measurements if loop_measurements else None,
        prior_pose=first_pose,  # Use actual starting pose as prior
        odometry_information=odom_info,
        loop_information=loop_info,
    )

    print(f"   Pose graph: {len(graph.variables)} variables, {len(graph.factors)} factors")
    print(f"   Factors: 1 prior + {len(odometry_measurements)} odometry + {len(loop_measurements)} loop closures")

    # ------------------------------------------------------------------------
    # 9. Optimize Pose Graph (Back-End)
    # ------------------------------------------------------------------------
    print("\n9. Optimizing pose graph (backend)...")
    initial_error = graph.compute_error()
    print(f"   Initial error: {initial_error:.6f}")

    optimized_vars, error_history = graph.optimize(
        method="gauss_newton", max_iterations=50, tol=1e-6
    )

    final_error = error_history[-1]
    print(f"   Final error: {final_error:.6f}")
    print(f"   Iterations: {len(error_history) - 1}")
    if initial_error > 1e-9:
        print(f"   Error reduction: {(1 - final_error / initial_error) * 100:.2f}%")
    else:
        print(f"   Error reduction: N/A (initial error near zero)")

    # Extract optimized poses
    optimized_poses = [optimized_vars[i] for i in range(n_poses)]

    # ------------------------------------------------------------------------
    # 10. Evaluate Results
    # ------------------------------------------------------------------------
    print("\n10. Evaluating results...")

    # Compute RMSE for all trajectories
    odom_errors = np.array(
        [np.linalg.norm(odom_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)]
    )
    frontend_errors = np.array(
        [np.linalg.norm(frontend_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)]
    )
    opt_errors = np.array(
        [np.linalg.norm(optimized_poses[i][:2] - true_poses[i][:2]) for i in range(n_poses)]
    )

    odom_rmse = np.sqrt(np.mean(odom_errors**2))
    frontend_rmse = np.sqrt(np.mean(frontend_errors**2))
    opt_rmse = np.sqrt(np.mean(opt_errors**2))

    print(f"   Odometry RMSE: {odom_rmse:.4f} m (baseline)")
    print(f"   Frontend RMSE: {frontend_rmse:.4f} m (scan-to-map corrected)")
    print(f"   Optimized RMSE: {opt_rmse:.4f} m (backend with {len(loop_closures)} loop closures)")
    
    if odom_rmse > 0:
        frontend_improvement = (1 - frontend_rmse / odom_rmse) * 100
        full_improvement = (1 - opt_rmse / odom_rmse) * 100
        print(f"   Frontend improvement: {frontend_improvement:+.2f}%")
        print(f"   Full pipeline improvement: {full_improvement:+.2f}%")

    # Loop closure error
    final_loop_error = np.linalg.norm(optimized_poses[-1][:2] - optimized_poses[0][:2])
    print(f"   Final loop closure error: {final_loop_error:.4f} m")

    # ------------------------------------------------------------------------
    # 11. Visualize Results
    # ------------------------------------------------------------------------
    print("\n11. Visualizing results...")
    plot_slam_results(
        true_poses, odom_poses, frontend_poses, optimized_poses, 
        landmarks, loop_closures, scans
    )
    
    # Generate animation if requested
    if animate:
        print("\n12. Generating SLAM animation...")
        gif_path = f"ch7_slam/figs/slam_pipeline_{trajectory_type}.gif"
        create_slam_animation(
            true_poses, odom_poses, frontend_poses, optimized_poses,
            scans, loop_closures, trajectory_type, gif_path, fps=5
        )

    print()
    print("=" * 70)
    print("SLAM PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Trajectory: {trajectory_type}, {n_poses} poses, {n_laps if trajectory_type == 'square' else 1} lap(s)")
    print(f"  - Loop closures: {len(loop_closures)} (observation-based detection)")
    print(f"  - Odometry drift: {final_drift:.3f} m")
    print(f"  - Odometry RMSE: {odom_rmse:.4f} m (baseline)")
    print(f"  - Frontend RMSE: {frontend_rmse:.4f} m (scan-to-map corrected)")
    print(f"  - Optimized RMSE: {opt_rmse:.4f} m (backend)")
    if odom_rmse > 0:
        frontend_improvement = (1 - frontend_rmse / odom_rmse) * 100
        full_improvement = (1 - opt_rmse / odom_rmse) * 100
        print(f"  - Frontend improvement: {frontend_improvement:+.2f}%")
        print(f"  - Full pipeline improvement: {full_improvement:+.2f}%")
    print()
    print("Pipeline Stages:")
    print("  1. Front-end: init -> predict (odom) -> scan-to-map (ICP) -> update map")
    print("  2. Loop Closure: Observation-based detection via scan descriptors")
    print("  3. Backend: Pose graph optimization with loop constraints")
    print("\nNote: This is a complete SLAM pipeline - all stages executed.")
    
    # Machine-readable summary for automated testing
    import json
    
    # Compute trajectory bounds for validation
    all_x = [p[0] for p in true_poses]
    all_y = [p[1] for p in true_poses]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    
    # Check loop closure (start vs end pose)
    start_pose = true_poses[0]
    end_pose = true_poses[-1]
    closure_dist = np.linalg.norm(end_pose[:2] - start_pose[:2])
    closure_yaw = abs(end_pose[2] - start_pose[2])
    
    summary = {
        "mode": "inline",
        "trajectory": trajectory_type,
        "laps": n_laps if trajectory_type == "square" else 1,
        "frontend_used": True,  # SlamFrontend2D.step() was executed
        "n_scans": len(scans),
        "n_frontend_steps": n_poses,  # Each pose runs frontend.step()
        "n_poses": n_poses,
        "n_loop_closures": len(loop_closures),
        "trajectory_bounds": {
            "x_range": round(x_range, 2),
            "y_range": round(y_range, 2),
        },
        "loop_closure_check": {
            "dist_m": round(closure_dist, 3),
            "yaw_deg": round(np.degrees(closure_yaw), 1),
        },
        "rmse": {
            "odom": round(odom_rmse, 4),
            "frontend": round(frontend_rmse, 4),
            "optimized": round(opt_rmse, 4),
        }
    }
    print(f"\n[SLAM_SUMMARY] {json.dumps(summary)}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Chapter 7: 2D Pose Graph SLAM Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default: square trajectory, 3 laps)
  python example_pose_graph_slam.py
  
  # Run with corridor trajectory
  python example_pose_graph_slam.py --trajectory corridor
  
  # Run with 2 laps (faster)
  python example_pose_graph_slam.py --laps 2
  
  # Run with pre-generated dataset
  python example_pose_graph_slam.py --data ch7_slam_2d_square
        """
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name or path (e.g., 'ch7_slam_2d_square' or full path)"
    )
    parser.add_argument(
        "--trajectory", type=str, default="square",
        choices=["square", "corridor"],
        help="Trajectory type for inline mode (default: square)"
    )
    parser.add_argument(
        "--laps", type=int, default=3,
        help="Number of laps for square trajectory (default: 3)"
    )
    parser.add_argument(
        "--loop_oracle", action="store_true", default=False,
        help="[DEPRECATED] Use distance-based oracle for loop closure instead of "
             "observation-based detection. For comparison/debugging only."
    )
    parser.add_argument(
        "--animate", action="store_true", default=False,
        help="Generate animated GIF showing SLAM pipeline evolution"
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
        
        run_with_dataset(str(data_path), use_loop_oracle=args.loop_oracle)
    else:
        run_with_inline_data(
            use_loop_oracle=args.loop_oracle,
            trajectory_type=args.trajectory,
            n_laps=args.laps,
            animate=args.animate,
        )


if __name__ == "__main__":
    main()

