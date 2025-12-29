"""Unit tests for observability analysis (Equation 8.3).

Tests the EKF observability matrix computation and rank analysis.

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest

import numpy as np

from ch8_sensor_fusion.observability_demo import (
    compute_observability_matrix,
    analyze_unobservable_states,
)


class TestObservabilityMatrix(unittest.TestCase):
    """Test EKF observability matrix computation (Eq. 8.3)."""
    
    def test_simple_fully_observable_system(self):
        """Test fully observable system (direct state measurement)."""
        # State: [x, y] (2D)
        # Measurement: [x, y] (directly observe both states)
        # F = I (identity dynamics)
        
        n = 2
        F = np.eye(n)
        H = np.eye(n)
        
        F_sequence = [F, F, F]
        H_sequence = [H, H, H]
        
        O_EKF, rank, s = compute_observability_matrix(H_sequence, F_sequence)
        
        # Should be fully observable (rank = 2)
        self.assertEqual(rank, n)
        self.assertEqual(O_EKF.shape, (6, 2))  # 3 measurements x 2 rows each
    
    def test_partially_observable_system(self):
        """Test partially observable system (odometry-like)."""
        # State: [px, py, vx, vy] (4D)
        # Measurement: [vx, vy] (observe velocity only)
        # F = [[1, 0, dt, 0],
        #      [0, 1, 0, dt],
        #      [0, 0, 1, 0],
        #      [0, 0, 0, 1]]
        
        dt = 0.1
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observe velocity only
        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        F_sequence = [F] * 10
        H_sequence = [H] * 10
        
        O_EKF, rank, s = compute_observability_matrix(H_sequence, F_sequence)
        
        # Should have rank 2 (velocity observable, position not)
        self.assertEqual(rank, 2)
        self.assertLess(rank, 4)  # Not fully observable
    
    def test_observability_improves_with_direct_measurement(self):
        """Test that adding direct measurements increases rank."""
        # Same system as above, but with occasional position measurements
        dt = 0.1
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        H_velocity = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        H_position = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Sequence: velocity, velocity, position, velocity, ...
        F_sequence = [F] * 10
        H_sequence = [H_velocity, H_velocity, H_position, H_velocity,
                      H_velocity, H_position, H_velocity, H_velocity,
                      H_position, H_velocity]
        
        O_EKF, rank, s = compute_observability_matrix(H_sequence, F_sequence)
        
        # Should be fully observable now (rank = 4)
        self.assertEqual(rank, 4)
    
    def test_state_transition_accumulation(self):
        """Test that state transition matrix accumulates correctly."""
        # Simple integrator: x[k+1] = x[k] + v[k] * dt
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])  # [position, velocity]
        H = np.array([[1, 0]])  # Observe position only
        
        F_sequence = [F, F, F]
        H_sequence = [H, H, H]
        
        O_EKF, rank, s = compute_observability_matrix(H_sequence, F_sequence)
        
        # Manual computation for verification:
        # O = [H0;
        #      H1 * F1;
        #      H2 * F2 * F1]
        # 
        # H0 = [1, 0]
        # H1 * F1 = [1, 0] * [[1, dt], [0, 1]] = [1, dt]
        # H2 * F2 * F1 = [1, 0] * [[1, dt], [0, 1]] * [[1, dt], [0, 1]]
        #              = [1, dt] * [[1, dt], [0, 1]] = [1, 2*dt]
        
        expected_O = np.array([
            [1, 0],
            [1, dt],
            [1, 2*dt]
        ])
        
        np.testing.assert_allclose(O_EKF, expected_O, atol=1e-10)
        
        # Should be fully observable (rank = 2)
        self.assertEqual(rank, 2)
    
    def test_unobservable_system(self):
        """Test completely unobservable system."""
        # State: [x, y]
        # Measurement: [] (no measurements!)
        # This should fail, but let's test with a null measurement
        
        F = np.eye(2)
        H = np.zeros((1, 2))  # Null measurement
        
        F_sequence = [F, F]
        H_sequence = [H, H]
        
        O_EKF, rank, s = compute_observability_matrix(H_sequence, F_sequence)
        
        # Should have rank 0 (nothing observable)
        self.assertEqual(rank, 0)


class TestUnobservableStateAnalysis(unittest.TestCase):
    """Test unobservable state identification."""
    
    def test_identify_unobservable_directions(self):
        """Test identification of unobservable directions."""
        # Create observability matrix with known null space
        # State: [px, py, vx, vy]
        # Only velocity is observable
        
        # Construct O that only sees velocity subspace
        O_EKF = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        U, s, Vt = np.linalg.svd(O_EKF, full_matrices=False)
        rank = np.sum(s > 1e-10)
        
        analysis = analyze_unobservable_states(
            O_EKF, rank,
            state_names=['px', 'py', 'vx', 'vy']
        )
        
        self.assertEqual(analysis['n_states'], 4)
        self.assertEqual(analysis['n_observable'], 2)
        self.assertEqual(analysis['n_unobservable'], 2)
        
        # Null space should span position subspace
        null_space = analysis['unobservable_modes']
        self.assertEqual(null_space.shape, (4, 2))
        
        # Check that null space vectors have zero velocity components
        for i in range(2):
            mode = null_space[:, i]
            # Velocity components should be near zero
            self.assertLess(np.abs(mode[2]), 0.1)
            self.assertLess(np.abs(mode[3]), 0.1)
    
    def test_fully_observable_has_no_null_space(self):
        """Test that fully observable system has empty null space."""
        O_EKF = np.eye(4)
        rank = 4
        
        analysis = analyze_unobservable_states(O_EKF, rank)
        
        self.assertEqual(analysis['n_unobservable'], 0)
        self.assertEqual(analysis['unobservable_modes'].shape[1], 0)
    
    def test_singular_values_analysis(self):
        """Test that singular values are returned correctly."""
        O_EKF = np.diag([10, 5, 0.1, 0])  # Known singular values
        
        U, s, Vt = np.linalg.svd(O_EKF, full_matrices=False)
        rank = np.sum(s > 1)  # Threshold at 1
        
        analysis = analyze_unobservable_states(O_EKF, rank)
        
        # Check singular values
        self.assertGreater(len(analysis['singular_values']), 0)
        np.testing.assert_allclose(
            analysis['singular_values'],
            [10, 5, 0.1, 0],
            atol=1e-10
        )


class TestObservabilityIntegration(unittest.TestCase):
    """Integration tests for observability analysis."""
    
    def test_odometry_vs_position_fix_scenario(self):
        """Test odometry-only vs odometry+position scenario.
        
        This is the key scenario from the observability demo.
        """
        dt = 0.1
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        H_odometry = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        H_position = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Scenario A: Odometry only
        F_seq_odom = [F] * 20
        H_seq_odom = [H_odometry] * 20
        
        O_odom, rank_odom, _ = compute_observability_matrix(H_seq_odom, F_seq_odom)
        analysis_odom = analyze_unobservable_states(O_odom, rank_odom)
        
        # Scenario B: Odometry + occasional position fixes
        F_seq_fix = [F] * 20
        H_seq_fix = [H_odometry] * 5 + [H_position] + [H_odometry] * 5 + \
                     [H_position] + [H_odometry] * 8
        
        O_fix, rank_fix, _ = compute_observability_matrix(H_seq_fix, F_seq_fix)
        analysis_fix = analyze_unobservable_states(O_fix, rank_fix)
        
        # Verify observations match demo
        self.assertLess(rank_odom, 4)  # Odometry-only not fully observable
        self.assertEqual(rank_fix, 4)  # With fixes, fully observable
        self.assertGreater(
            analysis_odom['n_unobservable'],
            analysis_fix['n_unobservable']
        )


if __name__ == '__main__':
    unittest.main()

