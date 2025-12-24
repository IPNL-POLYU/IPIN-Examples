"""Integration test: End-to-end pose graph SLAM with loop closure.

Validates complete SLAM pipeline: odometry drift → loop closure detection
→ pose graph optimization → significant RMSE reduction.

This is a high-level smoke test for the full system.

References:
    - Section 7.3: Pose Graph Optimization
    - Example: ch7_slam/example_pose_graph_slam.py

Author: Li-Ta Hsu
Date: 2024
"""

import numpy as np
import pytest

from core.slam import (
    create_pose_graph,
    icp_point_to_point,
    se2_apply,
    se2_compose,
    se2_relative,
)
from core.eval.metrics import compute_position_errors, compute_rmse


class TestPoseGraphSLAMPipeline:
    """End-to-end smoke tests for complete SLAM pipeline."""

    @pytest.fixture
    def square_trajectory_data(self):
        """Generate complete square trajectory scenario with ground truth."""
        np.random.seed(1234)
        
        # Ground truth square trajectory (8 poses total, 2 per side)
        side_length = 10.0
        poses_per_side = 2
        true_poses = []
        
        # Bottom side (moving right)
        for i in range(poses_per_side):
            x = i * (side_length / poses_per_side)
            true_poses.append(np.array([x, 0.0, 0.0]))
        
        # Right side (moving up)
        for i in range(poses_per_side):
            y = i * (side_length / poses_per_side)
            true_poses.append(np.array([side_length, y, np.pi / 2]))
        
        # Top side (moving left)
        for i in range(poses_per_side):
            x = side_length - i * (side_length / poses_per_side)
            true_poses.append(np.array([x, side_length, np.pi]))
        
        # Left side (moving down, closing loop)
        for i in range(poses_per_side):
            y = side_length - i * (side_length / poses_per_side)
            true_poses.append(np.array([0.0, y, -np.pi / 2]))
        
        true_poses = np.array(true_poses)
        
        # Generate environment landmarks (fixed features)
        num_landmarks = 40
        landmarks = np.random.rand(num_landmarks, 2) * (side_length + 2) - 1
        
        # Generate scans at each pose
        scans = []
        sensor_range = 8.0
        for pose in true_poses:
            # Find visible landmarks
            visible_pts = []
            for lm in landmarks:
                # Transform landmark to sensor frame
                lm_sensor = se2_apply(np.array([-pose[0], -pose[1], -pose[2]]), 
                                     lm.reshape(1, -1))[0]
                dist = np.linalg.norm(lm_sensor)
                if dist < sensor_range:
                    visible_pts.append(lm_sensor)
            
            if len(visible_pts) > 5:
                scan = np.array(visible_pts) + np.random.normal(0, 0.02, (len(visible_pts), 2))
                scans.append(scan)
            else:
                # Fallback: generate some points if nothing visible
                scans.append(np.random.rand(10, 2) * 5 - 2.5)
        
        # Simulate odometry with drift
        odometry_poses = [true_poses[0].copy()]
        odometry_measurements = []
        
        for i in range(len(true_poses) - 1):
            # Compute true relative pose
            rel_pose_true = se2_relative(true_poses[i], true_poses[i + 1])
            
            # Add odometry noise (drift accumulates)
            noise = np.array([
                np.random.normal(0, 0.1),   # x noise
                np.random.normal(0, 0.1),   # y noise
                np.random.normal(0, 0.02),  # yaw noise
            ])
            rel_pose_noisy = rel_pose_true + noise
            
            # Accumulate odometry
            odometry_poses.append(se2_compose(odometry_poses[-1], rel_pose_noisy))
            odometry_measurements.append((i, i + 1, rel_pose_noisy))
        
        odometry_poses = np.array(odometry_poses)
        
        return {
            "true_poses": true_poses,
            "odometry_poses": odometry_poses,
            "odometry_measurements": odometry_measurements,
            "scans": scans,
            "landmarks": landmarks,
        }

    def test_full_slam_pipeline_reduces_error(self, square_trajectory_data):
        """Smoke test: Full SLAM pipeline should significantly reduce odometry error."""
        data = square_trajectory_data
        true_poses = data["true_poses"]
        odometry_poses = data["odometry_poses"]
        odometry_meas = data["odometry_measurements"]
        scans = data["scans"]
        
        # 1. Compute initial odometry RMSE
        odom_errors = compute_position_errors(true_poses[:, :2], odometry_poses[:, :2])
        odom_rmse = compute_rmse(odom_errors)
        print(f"\n[BEFORE] Odometry RMSE: {odom_rmse:.4f} m")
        
        # 2. Detect loop closures using ICP
        loop_closures = []
        distance_threshold = 3.0
        min_time_separation = 4
        
        for i in range(len(odometry_poses)):
            for j in range(i + min_time_separation, len(odometry_poses)):
                dist = np.linalg.norm(odometry_poses[i, :2] - odometry_poses[j, :2])
                
                if dist < distance_threshold:
                    # Attempt ICP match
                    initial_guess = se2_relative(odometry_poses[i], odometry_poses[j])
                    
                    rel_pose, iters, residual, converged = icp_point_to_point(
                        scans[i], scans[j],
                        initial_pose=initial_guess,
                        max_iterations=50,
                        tolerance=1e-5,
                    )
                    
                    if converged and residual < 2.0:
                        loop_closures.append((i, j, rel_pose))
                        print(f"  Loop closure: {i} <-> {j}, residual={residual:.3f}")
        
        print(f"[DETECT] Found {len(loop_closures)} loop closures")
        
        # 3. Build and optimize pose graph
        graph = create_pose_graph(
            poses=odometry_poses.tolist(),
            odometry_measurements=odometry_meas,
            loop_closures=loop_closures if loop_closures else None,
            prior_pose=true_poses[0],  # Anchor first pose
            prior_information=np.diag([1e6, 1e6, 1e6]),
        )
        
        # 4. Optimize
        initial_error = graph.compute_error()
        optimized_vars, error_history = graph.optimize(
            method="gauss_newton",
            max_iterations=50,
            tol=1e-6,
        )
        final_error = graph.compute_error()
        
        # Extract optimized poses
        optimized_poses = np.array([optimized_vars[i] for i in range(len(true_poses))])
        
        # 5. Compute optimized RMSE
        slam_errors = compute_position_errors(true_poses[:, :2], optimized_poses[:, :2])
        slam_rmse = compute_rmse(slam_errors)
        print(f"[AFTER] SLAM RMSE: {slam_rmse:.4f} m")
        print(f"[IMPROVE] Error reduction: {(1 - slam_rmse / odom_rmse) * 100:.1f}%")
        
        # 6. Assertions (smoke test thresholds - lenient for synthetic data)
        # Note: Loop closures may not always be detected depending on noise and distance
        assert final_error <= initial_error * 1.1, "Optimization should not significantly increase error"
        
        # If loop closures were found, SLAM should improve
        if len(loop_closures) > 0:
            print(f"  Loop closures found - expecting improvement")
            assert slam_rmse <= odom_rmse, "SLAM with loop closure should reduce error"
            improvement = (odom_rmse - slam_rmse) / odom_rmse if odom_rmse > 0 else 0
            assert improvement >= 0, "SLAM should not worsen accuracy"
        else:
            print(f"  No loop closures - SLAM may not improve much")
            # Without loop closures, just check it doesn't break
            assert slam_rmse < 1.0, f"SLAM RMSE {slam_rmse:.3f}m unexpectedly high even without loop closures"

    def test_slam_without_loop_closure_still_works(self, square_trajectory_data):
        """Test pose graph optimization without loop closures (odometry-only)."""
        data = square_trajectory_data
        true_poses = data["true_poses"]
        odometry_poses = data["odometry_poses"]
        odometry_meas = data["odometry_measurements"]
        
        # Build pose graph with NO loop closures
        graph = create_pose_graph(
            poses=odometry_poses.tolist(),
            odometry_measurements=odometry_meas,
            loop_closures=None,  # NO loop closures
            prior_pose=true_poses[0],
        )
        
        # Optimize (should only smooth odometry, not close loop)
        optimized_vars, _ = graph.optimize(method="gauss_newton", max_iterations=20)
        optimized_poses = np.array([optimized_vars[i] for i in range(len(true_poses))])
        
        # RMSE should be similar to odometry (no loop closure correction)
        odom_errors = compute_position_errors(true_poses[:, :2], odometry_poses[:, :2])
        odom_rmse = compute_rmse(odom_errors)
        no_lc_errors = compute_position_errors(true_poses[:, :2], optimized_poses[:, :2])
        no_lc_rmse = compute_rmse(no_lc_errors)
        
        print(f"\n[NO LOOP CLOSURE] Odometry RMSE: {odom_rmse:.4f} m")
        print(f"[NO LOOP CLOSURE] Smoothed RMSE: {no_lc_rmse:.4f} m")
        
        # Should not improve much without loop closure
        assert np.abs(no_lc_rmse - odom_rmse) < 0.2, \
            "Without loop closure, RMSE should stay similar to odometry"

    def test_loop_closure_impact_quantified(self, square_trajectory_data):
        """Quantify the specific impact of loop closures on accuracy."""
        data = square_trajectory_data
        true_poses = data["true_poses"]
        odometry_poses = data["odometry_poses"]
        odometry_meas = data["odometry_measurements"]
        scans = data["scans"]
        
        # Detect loop closures
        loop_closures = []
        for i in range(len(odometry_poses)):
            for j in range(i + 4, len(odometry_poses)):
                if np.linalg.norm(odometry_poses[i, :2] - odometry_poses[j, :2]) < 3.0:
                    initial_guess = se2_relative(odometry_poses[i], odometry_poses[j])
                    rel_pose, _, residual, converged = icp_point_to_point(
                        scans[i], scans[j], initial_pose=initial_guess, max_iterations=50
                    )
                    if converged and residual < 2.0:
                        loop_closures.append((i, j, rel_pose))
        
        if len(loop_closures) == 0:
            pytest.skip("No loop closures detected in this scenario")
        
        # Optimize WITHOUT loop closures
        graph_no_lc = create_pose_graph(
            poses=odometry_poses.tolist(),
            odometry_measurements=odometry_meas,
            loop_closures=None,
            prior_pose=true_poses[0],
        )
        optimized_no_lc, _ = graph_no_lc.optimize(method="gauss_newton", max_iterations=30)
        poses_no_lc = np.array([optimized_no_lc[i] for i in range(len(true_poses))])
        errors_no_lc = compute_position_errors(true_poses[:, :2], poses_no_lc[:, :2])
        rmse_no_lc = compute_rmse(errors_no_lc)
        
        # Optimize WITH loop closures
        graph_with_lc = create_pose_graph(
            poses=odometry_poses.tolist(),
            odometry_measurements=odometry_meas,
            loop_closures=loop_closures,
            prior_pose=true_poses[0],
        )
        optimized_with_lc, _ = graph_with_lc.optimize(method="gauss_newton", max_iterations=30)
        poses_with_lc = np.array([optimized_with_lc[i] for i in range(len(true_poses))])
        errors_with_lc = compute_position_errors(true_poses[:, :2], poses_with_lc[:, :2])
        rmse_with_lc = compute_rmse(errors_with_lc)
        
        print(f"\n[COMPARISON]")
        print(f"  Without loop closure: {rmse_no_lc:.4f} m")
        print(f"  With loop closure: {rmse_with_lc:.4f} m")
        print(f"  Loop closure benefit: {(rmse_no_lc - rmse_with_lc):.4f} m")
        
        # Loop closures should provide measurable improvement
        assert rmse_with_lc < rmse_no_lc, "Loop closures should improve accuracy"
        improvement = (rmse_no_lc - rmse_with_lc) / rmse_no_lc
        assert improvement > 0.1, \
            f"Loop closure improvement {improvement*100:.1f}% below 10% threshold"


class TestPoseGraphRegressionThresholds:
    """Regression tests with fixed performance thresholds."""

    def test_slam_accuracy_regression_threshold(self):
        """Regression: SLAM on square trajectory should achieve <20cm RMSE."""
        np.random.seed(9999)
        
        # Simplified square trajectory (4 poses, one per corner)
        true_poses = np.array([
            [0, 0, 0],
            [5, 0, np.pi / 2],
            [5, 5, np.pi],
            [0, 5, -np.pi / 2],
        ])
        
        # Noisy odometry
        odometry_poses = true_poses + np.random.normal(0, 0.2, true_poses.shape)
        odometry_poses[0] = true_poses[0]  # Fix first pose
        
        # Generate scans
        landmarks = np.random.rand(30, 2) * 6
        scans = []
        for pose in true_poses:
            visible = []
            for lm in landmarks:
                lm_sensor = se2_apply(np.array([-pose[0], -pose[1], -pose[2]]), 
                                     lm.reshape(1, -1))[0]
                if np.linalg.norm(lm_sensor) < 6.0:
                    visible.append(lm_sensor)
            if len(visible) > 0:
                scans.append(np.array(visible) + np.random.normal(0, 0.01, (len(visible), 2)))
            else:
                # Fallback: add some dummy points if nothing visible
                scans.append(np.random.rand(10, 2) * 3 - 1.5)
        
        # Odometry measurements
        odometry_meas = [
            (i, i + 1, se2_relative(odometry_poses[i], odometry_poses[i + 1]))
            for i in range(len(odometry_poses) - 1)
        ]
        
        # Detect loop closure (0 <-> 3, closing the square)
        rel_pose, _, residual, converged = icp_point_to_point(
            scans[0], scans[3],
            initial_pose=se2_relative(odometry_poses[0], odometry_poses[3]),
            max_iterations=50,
        )
        
        loop_closures = [(0, 3, rel_pose)] if converged and residual < 5.0 else []
        
        # Build and optimize
        graph = create_pose_graph(
            poses=odometry_poses.tolist(),
            odometry_measurements=odometry_meas,
            loop_closures=loop_closures,
            prior_pose=true_poses[0],
        )
        
        optimized_vars, _ = graph.optimize(method="gauss_newton", max_iterations=50)
        optimized_poses = np.array([optimized_vars[i] for i in range(len(true_poses))])
        
        slam_errors = compute_position_errors(true_poses[:, :2], optimized_poses[:, :2])
        slam_rmse = compute_rmse(slam_errors)
        
        # Regression threshold: SLAM should achieve <20cm RMSE
        assert slam_rmse < 0.20, \
            f"SLAM RMSE {slam_rmse:.4f}m exceeds 20cm regression threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

