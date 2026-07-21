"""Unit tests for core.slam.factors module.

Tests SLAM-specific factor graph factors for pose graph optimization
used in Chapter 7.

Author: Li-Ta Hsu
Date: 2024
"""

import numpy as np
import pytest

from core.estimators.factor_graph import FactorGraph
from core.slam import (
    create_landmark_factor,
    create_loop_closure_factor,
    create_odometry_factor,
    create_pose_graph,
    create_prior_factor,
    se2_apply,
    se2_compose,
    se2_inverse,
    se2_relative,
)


class TestOdometryFactor:
    """Test suite for odometry factors."""

    def test_odometry_factor_creation(self):
        """Test creating an odometry factor."""
        rel_pose = np.array([1.0, 0.0, 0.0])
        factor = create_odometry_factor(0, 1, rel_pose)

        assert factor.variable_ids == [0, 1]
        assert factor.information.shape == (3, 3)

    def test_odometry_factor_zero_error(self):
        """Test odometry factor has zero error when poses match measurement."""
        # Create factor graph
        graph = FactorGraph()
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([1.0, 0.0, 0.0]))

        # Add odometry factor: pose 0 → pose 1 moved 1m forward
        rel_pose = np.array([1.0, 0.0, 0.0])
        factor = create_odometry_factor(0, 1, rel_pose)
        graph.add_factor(factor)

        # Error should be zero (poses match measurement)
        error = graph.compute_error()
        assert np.isclose(error, 0.0, atol=1e-10)

    def test_odometry_factor_nonzero_error(self):
        """Test odometry factor has nonzero error when poses don't match."""
        # Create factor graph
        graph = FactorGraph()
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([2.0, 0.0, 0.0]))  # Wrong position

        # Add odometry factor expecting 1m forward
        rel_pose = np.array([1.0, 0.0, 0.0])
        factor = create_odometry_factor(0, 1, rel_pose)
        graph.add_factor(factor)

        # Error should be nonzero
        error = graph.compute_error()
        assert error > 0.0

    def test_odometry_chain_optimization(self):
        """Test optimizing a chain of odometry factors."""
        # Create trajectory: 3 poses connected by odometry
        graph = FactorGraph()
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([0.5, 0.5, 0.0]))  # Bad initial guess
        graph.add_variable(2, np.array([1.5, 0.5, 0.0]))  # Bad initial guess

        # Fix first pose
        prior_factor = create_prior_factor(0, np.array([0.0, 0.0, 0.0]))
        graph.add_factor(prior_factor)

        # Add odometry: each step is 1m forward
        odom_01 = create_odometry_factor(0, 1, np.array([1.0, 0.0, 0.0]))
        odom_12 = create_odometry_factor(1, 2, np.array([1.0, 0.0, 0.0]))
        graph.add_factor(odom_01)
        graph.add_factor(odom_12)

        # Optimize
        optimized, error_history = graph.optimize(max_iterations=10)

        # Check converged poses
        assert np.allclose(optimized[0], [0.0, 0.0, 0.0], atol=1e-4)
        assert np.allclose(optimized[1], [1.0, 0.0, 0.0], atol=1e-4)
        assert np.allclose(optimized[2], [2.0, 0.0, 0.0], atol=1e-4)

        # Final error should be near zero
        assert error_history[-1] < 1e-10

    def test_odometry_with_rotation(self):
        """Test odometry factor with rotation."""
        graph = FactorGraph()
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([1.0, 0.0, np.pi / 2]))  # Initial guess

        # Odometry: 1m forward + 90° left turn
        rel_pose = np.array([1.0, 0.0, np.pi / 2])
        factor = create_odometry_factor(0, 1, rel_pose)
        graph.add_factor(factor)

        # Error should be zero (matches exactly)
        error = factor.compute_error(graph.variables)
        assert np.isclose(error, 0.0, atol=1e-10)


class TestLoopClosureFactor:
    """Test suite for loop closure factors."""

    def test_loop_closure_creation(self):
        """Test creating a loop closure factor."""
        rel_pose = np.array([0.1, 0.1, 0.05])
        factor = create_loop_closure_factor(0, 10, rel_pose)

        assert factor.variable_ids == [0, 10]
        assert factor.information.shape == (3, 3)

    def test_loop_closure_reduces_drift(self):
        """Test that loop closure corrects accumulated drift."""
        # Simulate accumulated drift: straight line with slight drift
        graph = FactorGraph()
        n_poses = 5

        # Initialize poses with drift
        for i in range(n_poses):
            # Drifting trajectory (accumulating y error)
            graph.add_variable(i, np.array([float(i), 0.1 * i, 0.0]))

        # Fix first pose
        prior = create_prior_factor(0, np.array([0.0, 0.0, 0.0]))
        graph.add_factor(prior)

        # Add odometry chain (expects straight line)
        for i in range(n_poses - 1):
            odom = create_odometry_factor(i, i + 1, np.array([1.0, 0.0, 0.0]))
            graph.add_factor(odom)

        # Add loop closure: last pose should be close to first
        # (robot returned to start)
        loop_rel = np.array([4.0, 0.0, 0.0])  # Expected relative pose 0→4
        loop_factor = create_loop_closure_factor(0, n_poses - 1, loop_rel)
        graph.add_factor(loop_factor)

        # Optimize
        initial_error = graph.compute_error()
        optimized, error_history = graph.optimize(max_iterations=20)
        final_error = error_history[-1]

        # Error should decrease significantly
        assert final_error < initial_error

        # Trajectory should be straighter (less y drift)
        for i in range(n_poses):
            assert abs(optimized[i][1]) < 0.1  # Small y deviation


class TestEq722Conformance:
    """Conformance tests for book Eq. (7.22).

    f(T_i, T_j, ΔT'_ij) = ln((ΔT'_ij)⁻¹ T_i⁻¹ T_j)^∨

    The residual must be a *group* operation. A componentwise subtraction
    ``ΔT' - (T_i⁻¹T_j)`` also vanishes at the optimum, so the round-trip test
    alone cannot tell the two apart — the perturbation test below can.
    """

    POSE_I = np.array([1.0, 2.0, 0.3])
    POSE_J = np.array([3.5, -1.0, -0.4])

    def test_consistent_measurement_gives_zero_residual(self):
        """A measurement equal to the true relative pose has no error."""
        rel = se2_relative(self.POSE_I, self.POSE_J)
        factor = create_loop_closure_factor(0, 1, rel)

        residual = factor.residual_func([self.POSE_I, self.POSE_J])

        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)

    def test_residual_equals_book_group_expression(self):
        """Residual matches ln((ΔT')⁻¹ T_i⁻¹ T_j)^∨, not ΔT' - T_i⁻¹T_j."""
        rel_true = se2_relative(self.POSE_I, self.POSE_J)
        perturbation = np.array([0.05, -0.02, 0.01])
        rel_measured = se2_compose(rel_true, perturbation)

        residual = create_loop_closure_factor(0, 1, rel_measured).residual_func(
            [self.POSE_I, self.POSE_J]
        )

        expected = se2_relative(rel_measured, rel_true)
        np.testing.assert_allclose(residual, expected, atol=1e-12)

        # (ΔT'∘d)⁻¹ ∘ ΔT' = d⁻¹: the residual recovers the injected error.
        np.testing.assert_allclose(residual, se2_inverse(perturbation), atol=1e-12)

        # Guard against a regression to componentwise subtraction.
        naive = rel_measured - rel_true
        assert not np.allclose(residual, naive, atol=1e-6)

    def test_residual_translation_lives_in_measurement_frame(self):
        """Translation error is rotated by R(Δθ')ᵀ, per the group inverse."""
        rel_true = se2_relative(self.POSE_I, self.POSE_J)
        offset = np.array([0.1, 0.0, 0.0])
        rel_measured = rel_true + offset  # translation-only discrepancy

        residual = create_loop_closure_factor(0, 1, rel_measured).residual_func(
            [self.POSE_I, self.POSE_J]
        )

        theta_measured = rel_measured[2]
        c, s = np.cos(theta_measured), np.sin(theta_measured)
        rot_transpose = np.array([[c, s], [-s, c]])
        np.testing.assert_allclose(
            residual[:2], rot_transpose @ (-offset[:2]), atol=1e-12
        )
        # Not merely the unrotated difference (Δθ' is far from zero here).
        assert not np.allclose(residual[:2], -offset[:2], atol=1e-6)

    def test_yaw_residual_wraps_across_pi(self):
        """Yaw error stays in [-π, π] across the ±π seam."""
        pose_a = np.array([0.0, 0.0, 3.10])
        pose_b = np.array([1.0, 0.0, -3.10])
        rel = se2_relative(pose_a, pose_b)

        residual = create_loop_closure_factor(0, 1, rel).residual_func([pose_a, pose_b])

        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)


class TestPriorFactor:
    """Test suite for prior factors."""

    def test_prior_factor_creation(self):
        """Test creating a prior factor."""
        prior_pose = np.array([0.0, 0.0, 0.0])
        factor = create_prior_factor(0, prior_pose)

        assert factor.variable_ids == [0]
        assert factor.information.shape == (3, 3)

    def test_strong_prior_anchors_pose(self):
        """Test that strong prior fixes pose near prior value."""
        graph = FactorGraph()
        graph.add_variable(0, np.array([1.0, 1.0, 0.5]))  # Far from prior

        # Strong prior at origin
        prior_pose = np.array([0.0, 0.0, 0.0])
        info = np.diag([1e6, 1e6, 1e6])  # Very strong
        prior = create_prior_factor(0, prior_pose, information=info)
        graph.add_factor(prior)

        # Optimize
        optimized, _ = graph.optimize(max_iterations=10)

        # Should be very close to prior
        np.testing.assert_allclose(optimized[0], prior_pose, atol=1e-4)

    def test_weak_prior_allows_movement(self):
        """Test that weak prior allows pose to move when conflicting with odometry."""
        graph = FactorGraph()
        # Initialize pose 0 far from prior
        graph.add_variable(0, np.array([2.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([3.0, 0.0, 0.0]))

        # Very weak prior on first pose at origin
        weak_info = np.diag([0.01, 0.01, 0.01])  # Very weak
        prior = create_prior_factor(0, np.array([0.0, 0.0, 0.0]), information=weak_info)
        graph.add_factor(prior)

        # Strong odometry factor connecting poses (expects 1m forward)
        strong_odom_info = np.diag([1e4, 1e4, 1e4])  # Very strong
        odom = create_odometry_factor(0, 1, np.array([1.0, 0.0, 0.0]), information=strong_odom_info)
        graph.add_factor(odom)

        # Optimize
        optimized, _ = graph.optimize(max_iterations=50)

        # Pose 0 should stay away from origin (odometry constraint dominates)
        # The solution should maintain the 1m spacing
        spacing = optimized[1] - optimized[0]
        assert np.isclose(spacing[0], 1.0, atol=0.1)  # ~1m spacing maintained


class TestLandmarkFactor:
    """Test suite for landmark observation factors."""

    def test_landmark_factor_creation(self):
        """Test creating a landmark factor."""
        obs = np.array([2.0, 1.0])
        factor = create_landmark_factor(0, 100, obs)

        assert factor.variable_ids == [0, 100]
        assert factor.information.shape == (2, 2)

    def test_landmark_factor_zero_error(self):
        """Test landmark factor with correct pose and landmark."""
        graph = FactorGraph()
        # Pose at origin
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        # Landmark at [2, 1] global
        graph.add_variable(100, np.array([2.0, 1.0]))

        # Observation: landmark at [2, 1] in local frame (same as global at origin)
        obs = np.array([2.0, 1.0])
        factor = create_landmark_factor(0, 100, obs)
        graph.add_factor(factor)

        # Error should be near zero
        error = factor.compute_error(graph.variables)
        assert error < 1e-10

    def test_landmark_optimization(self):
        """Test optimizing pose and landmark together."""
        graph = FactorGraph()
        # Pose (fixed by prior)
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        # Landmark (wrong initial guess)
        graph.add_variable(100, np.array([1.5, 0.5]))

        # Strong prior on pose
        prior = create_prior_factor(0, np.array([0.0, 0.0, 0.0]))
        graph.add_factor(prior)

        # Landmark observation: should be at [2, 1] in local frame
        obs = np.array([2.0, 1.0])
        factor = create_landmark_factor(0, 100, obs)
        graph.add_factor(factor)

        # Optimize
        optimized, _ = graph.optimize(max_iterations=20)

        # Landmark should move to correct position
        np.testing.assert_allclose(optimized[100], [2.0, 1.0], atol=1e-2)


class TestPoseGraphCreation:
    """Test suite for create_pose_graph convenience function."""

    def test_simple_pose_graph(self):
        """Test creating a simple pose graph."""
        poses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]
        odometry = [
            (0, 1, np.array([1.0, 0.0, 0.0])),
            (1, 2, np.array([1.0, 0.0, 0.0])),
        ]

        graph = create_pose_graph(poses, odometry)

        assert len(graph.variables) == 3
        assert len(graph.factors) == 3  # 1 prior + 2 odometry

    def test_pose_graph_with_loop_closure(self):
        """Test pose graph with loop closure."""
        # Square trajectory: 4 poses, should close loop
        poses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, np.pi / 2]),
            np.array([1.0, 1.0, np.pi]),
            np.array([0.0, 1.0, -np.pi / 2]),
        ]

        odometry = [
            (0, 1, np.array([1.0, 0.0, np.pi / 2])),
            (1, 2, np.array([1.0, 0.0, np.pi / 2])),
            (2, 3, np.array([1.0, 0.0, np.pi / 2])),
        ]

        # Loop closure: pose 3 back to pose 0
        loop_closures = [(3, 0, np.array([1.0, 0.0, np.pi / 2]))]

        graph = create_pose_graph(poses, odometry, loop_closures=loop_closures)

        assert len(graph.factors) == 5  # 1 prior + 3 odom + 1 loop

    def test_pose_graph_optimization(self):
        """Test full pose graph optimization."""
        # Trajectory with noise
        poses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.1, 0.05, 0.02]),  # Noisy
            np.array([2.05, 0.1, -0.01]),  # Noisy
        ]

        odometry = [
            (0, 1, np.array([1.0, 0.0, 0.0])),
            (1, 2, np.array([1.0, 0.0, 0.0])),
        ]

        graph = create_pose_graph(poses, odometry)

        # Optimize
        initial_error = graph.compute_error()
        optimized, error_history = graph.optimize(max_iterations=20)
        final_error = error_history[-1]

        # Error should decrease
        assert final_error < initial_error

        # Poses should be closer to straight line
        assert abs(optimized[1][1]) < 0.05  # Less y deviation
        assert abs(optimized[2][1]) < 0.05


class TestIntegration:
    """Integration tests combining multiple factor types."""

    def test_full_slam_scenario(self):
        """Test complete SLAM scenario with odometry and loop closure."""
        # Simulate robot going in square, but with accumulated drift
        n_sides = 4
        n_poses = n_sides + 1  # 5 poses (0, 1, 2, 3, 4 where 4≈0)

        # Initialize with drifting trajectory (not perfect square)
        graph = FactorGraph()
        for i in range(n_poses):
            # Add increasing drift
            drift = 0.05 * i
            graph.add_variable(i, np.array([float(i % 2), float(i // 2), drift]))

        # Fix first pose
        prior = create_prior_factor(0, np.array([0.0, 0.0, 0.0]))
        graph.add_factor(prior)

        # Add odometry for square (each side is 1m)
        odometry_measurements = [
            (0, 1, np.array([1.0, 0.0, np.pi / 2])),  # East + turn left
            (1, 2, np.array([1.0, 0.0, np.pi / 2])),  # North + turn left
            (2, 3, np.array([1.0, 0.0, np.pi / 2])),  # West + turn left
            (3, 4, np.array([1.0, 0.0, np.pi / 2])),  # South + turn left
        ]

        for from_id, to_id, rel in odometry_measurements:
            odom = create_odometry_factor(from_id, to_id, rel)
            graph.add_factor(odom)

        # Add loop closure: pose 4 should match pose 0
        loop_rel = np.array([0.0, 0.0, 0.0])  # Should be at same place
        loop = create_loop_closure_factor(0, 4, loop_rel)
        graph.add_factor(loop)

        # Optimize
        initial_error = graph.compute_error()
        optimized, error_history = graph.optimize(max_iterations=50)
        final_error = error_history[-1]

        # Check optimization improved solution
        assert final_error < initial_error

        # Check loop closure: first and last pose should be close
        pose_0 = optimized[0]
        pose_4 = optimized[4]
        closure_error = np.linalg.norm(pose_0[:2] - pose_4[:2])

        # Should close loop within reasonable tolerance
        # (Note: perfect closure is difficult with noisy initialization)
        assert closure_error < 1.0  # Within 1.0m (relaxed for noisy scenario)

    def test_scan_matching_integration(self):
        """Test integration with scan matching (simulated)."""
        # Scenario: robot moves forward, does ICP, adds factors
        graph = FactorGraph()

        # Two poses
        graph.add_variable(0, np.array([0.0, 0.0, 0.0]))
        graph.add_variable(1, np.array([0.8, 0.1, 0.05]))  # Initial odometry guess

        # Prior on first pose
        prior = create_prior_factor(0, np.array([0.0, 0.0, 0.0]))
        graph.add_factor(prior)

        # Odometry measurement (noisy)
        odom_rel = np.array([0.9, 0.05, 0.02])
        odom_cov = np.diag([0.1, 0.1, 0.01])  # Moderate uncertainty
        odom = create_odometry_factor(0, 1, odom_rel, information=np.linalg.inv(odom_cov))
        graph.add_factor(odom)

        # Simulated ICP measurement (more accurate)
        icp_rel = np.array([1.0, 0.0, 0.0])  # ICP says: exactly 1m forward
        icp_cov = np.diag([0.01, 0.01, 0.001])  # Very certain
        icp_factor = create_odometry_factor(
            0, 1, icp_rel, information=np.linalg.inv(icp_cov)
        )
        graph.add_factor(icp_factor)

        # Optimize
        optimized, _ = graph.optimize(max_iterations=20)

        # Result should favor ICP (higher information)
        # Should be closer to [1.0, 0.0, 0.0] than to odometry
        dist_to_icp = np.linalg.norm(optimized[1] - np.array([1.0, 0.0, 0.0]))
        assert dist_to_icp < 0.05  # Very close to ICP measurement

