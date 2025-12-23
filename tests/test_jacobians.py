"""
Unit tests for Jacobian correctness.

Tests analytical Jacobians against numerical differentiation to ensure correctness.
This is critical for Extended Kalman Filter performance - incorrect Jacobians
cause filter divergence.

Run with: python -m pytest tests/test_jacobians.py -v
"""

import numpy as np
import pytest
from typing import Callable

from core.models import (
    ConstantVelocity1D,
    ConstantVelocity2D,
    RangeMeasurement2D,
    RangeBearingMeasurement2D,
    PositionMeasurement2D
)


def numerical_jacobian(
    f: Callable,
    x: np.ndarray,
    epsilon: float = 1e-7
) -> np.ndarray:
    """
    Compute Jacobian numerically using central differences.
    
    Args:
        f: Function that takes x and returns y
        x: Point at which to compute Jacobian
        epsilon: Step size for finite differences
    
    Returns:
        Numerical Jacobian, shape (len(y), len(x))
    """
    x = np.asarray(x, dtype=float)
    y0 = f(x)
    
    n_out = len(y0)
    n_in = len(x)
    
    J = np.zeros((n_out, n_in))
    
    for i in range(n_in):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        y_plus = f(x_plus)
        y_minus = f(x_minus)
        
        # Central difference
        J[:, i] = (y_plus - y_minus) / (2 * epsilon)
    
    return J


class TestMotionModelJacobians:
    """Test motion model Jacobians."""
    
    def test_constant_velocity_1d_jacobian(self):
        """Test 1D constant velocity Jacobian."""
        model = ConstantVelocity1D()
        dt = 0.1
        
        # Test at several points
        test_states = [
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
            np.array([-5.0, 3.5]),
        ]
        
        for x in test_states:
            # Analytical Jacobian
            F_analytical = model.F(dt)
            
            # Numerical Jacobian
            def f(x_): return model.f(x_, dt=dt)
            F_numerical = numerical_jacobian(f, x)
            
            # Compare
            np.testing.assert_allclose(
                F_analytical, F_numerical,
                rtol=1e-5, atol=1e-8,
                err_msg=f"Jacobian mismatch at x={x}"
            )
    
    def test_constant_velocity_2d_jacobian(self):
        """Test 2D constant velocity Jacobian."""
        model = ConstantVelocity2D()
        dt = 0.5
        
        test_states = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([5.0, 3.0, 1.0, 0.5]),
            np.array([-2.0, 10.0, -1.5, 2.0]),
        ]
        
        for x in test_states:
            # Analytical
            F_analytical = model.F(dt)
            
            # Numerical
            def f(x_): return model.f(x_, dt=dt)
            F_numerical = numerical_jacobian(f, x)
            
            # Compare
            np.testing.assert_allclose(
                F_analytical, F_numerical,
                rtol=1e-5, atol=1e-8,
                err_msg=f"Jacobian mismatch at x={x}"
            )
    
    def test_motion_jacobian_shapes(self):
        """Test that motion Jacobians have correct shapes."""
        # 1D model
        model_1d = ConstantVelocity1D()
        F_1d = model_1d.F(dt=0.1)
        assert F_1d.shape == (2, 2), f"Expected (2,2), got {F_1d.shape}"
        
        # 2D model
        model_2d = ConstantVelocity2D()
        F_2d = model_2d.F(dt=0.1)
        assert F_2d.shape == (4, 4), f"Expected (4,4), got {F_2d.shape}"


class TestMeasurementModelJacobians:
    """Test measurement model Jacobians."""
    
    def test_range_measurement_jacobian(self):
        """Test range-only measurement Jacobian."""
        anchors = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        model = RangeMeasurement2D(anchors)
        
        test_states = [
            np.array([5.0, 5.0, 1.0, 0.5]),  # Center
            np.array([2.0, 3.0, 0.0, 0.0]),  # Off-center
            np.array([8.0, 7.0, -1.0, 1.5]), # Another point
        ]
        
        for x in test_states:
            # Analytical
            H_analytical = model.H(x)
            
            # Numerical
            def h(x_): return model.h(x_)
            H_numerical = numerical_jacobian(h, x)
            
            # Compare
            np.testing.assert_allclose(
                H_analytical, H_numerical,
                rtol=1e-4, atol=1e-7,
                err_msg=f"Range Jacobian mismatch at x={x}"
            )
    
    def test_range_bearing_measurement_jacobian(self):
        """Test range-bearing measurement Jacobian."""
        landmarks = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0]
        ])
        model = RangeBearingMeasurement2D(landmarks)
        
        test_states = [
            np.array([10.0, 10.0, 1.0, 0.5]),
            np.array([5.0, 5.0, 0.0, 0.0]),
            np.array([15.0, 8.0, -0.5, 1.0]),
        ]
        
        for x in test_states:
            # Analytical
            H_analytical = model.H(x)
            
            # Numerical
            def h(x_): return model.h(x_)
            H_numerical = numerical_jacobian(h, x)
            
            # Compare
            # Note: Bearing Jacobians can be sensitive, so slightly higher tolerance
            np.testing.assert_allclose(
                H_analytical, H_numerical,
                rtol=1e-3, atol=1e-6,
                err_msg=f"Range-Bearing Jacobian mismatch at x={x}"
            )
    
    def test_position_measurement_jacobian(self):
        """Test direct position measurement Jacobian."""
        model = PositionMeasurement2D()
        
        test_states = [
            np.array([0.0, 0.0, 1.0, 0.5]),
            np.array([5.0, 7.0, -1.0, 2.0]),
        ]
        
        for x in test_states:
            # Analytical
            H_analytical = model.H(x)
            
            # Numerical
            def h(x_): return model.h(x_)
            H_numerical = numerical_jacobian(h, x)
            
            # Compare
            np.testing.assert_allclose(
                H_analytical, H_numerical,
                rtol=1e-5, atol=1e-8,
                err_msg=f"Position Jacobian mismatch at x={x}"
            )
    
    def test_measurement_jacobian_shapes(self):
        """Test that measurement Jacobians have correct shapes."""
        # Range-only: 4 anchors
        anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        model_range = RangeMeasurement2D(anchors)
        x = np.array([5, 5, 1, 0.5])
        H_range = model_range.H(x)
        assert H_range.shape == (4, 4), f"Expected (4,4), got {H_range.shape}"
        
        # Range-bearing: 3 landmarks
        landmarks = np.array([[0, 0], [20, 0], [20, 20]])
        model_rb = RangeBearingMeasurement2D(landmarks)
        H_rb = model_rb.H(x)
        assert H_rb.shape == (6, 4), f"Expected (6,4), got {H_rb.shape}"
        
        # Position
        model_pos = PositionMeasurement2D()
        H_pos = model_pos.H(x)
        assert H_pos.shape == (2, 4), f"Expected (2,4), got {H_pos.shape}"


class TestSingularityHandling:
    """Test singularity handling in Jacobians."""
    
    def test_range_at_anchor_singularity(self):
        """Test that range Jacobian handles singularity at anchor position."""
        anchors = np.array([[5.0, 5.0], [10.0, 0.0]])
        model = RangeMeasurement2D(anchors)
        
        # State exactly at first anchor
        x = np.array([5.0, 5.0, 0.0, 0.0])
        
        # Should not crash
        H = model.H(x)
        
        # First row should be zeros (singularity)
        assert np.allclose(H[0, :], 0.0), "Expected zero Jacobian at singularity"
        
        # Second row should be normal
        assert not np.allclose(H[1, :], 0.0), "Non-singular measurement should have non-zero Jacobian"
    
    def test_range_bearing_at_landmark_singularity(self):
        """Test range-bearing at landmark position."""
        landmarks = np.array([[10.0, 10.0]])
        model = RangeBearingMeasurement2D(landmarks)
        
        # State at landmark
        x = np.array([10.0, 10.0, 1.0, 0.5])
        
        # Should not crash
        H = model.H(x)
        
        # Both range and bearing rows should be zeros
        assert np.allclose(H, 0.0), "Expected zero Jacobians at landmark"


class TestInputValidation:
    """Test input validation in models."""
    
    def test_invalid_state_dimension(self):
        """Test that invalid state dimensions are rejected."""
        model = ConstantVelocity2D()
        
        # Wrong dimension
        with pytest.raises(ValueError, match="State must be 4D"):
            model.f(np.array([1.0, 2.0]))  # Only 2D, needs 4D
    
    def test_invalid_anchor_shape(self):
        """Test that invalid anchor shapes are rejected."""
        # 1D anchors (should be 2D)
        with pytest.raises(ValueError, match="must be.*2.*array"):
            RangeMeasurement2D(np.array([1, 2, 3]))
        
        # 3D anchors
        with pytest.raises(ValueError, match="must be.*2.*array"):
            RangeMeasurement2D(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_negative_dt_rejected(self):
        """Test that negative time steps are rejected."""
        from core.models.motion_models import validate_motion_model_inputs
        
        x = np.array([0, 0])
        with pytest.raises(ValueError, match="must be positive"):
            validate_motion_model_inputs(x, 2, dt=-0.1)


class TestProcessNoise:
    """Test process noise covariance matrices."""
    
    def test_process_noise_symmetry(self):
        """Test that Q matrices are symmetric."""
        # 1D
        Q_1d = ConstantVelocity1D.Q(dt=0.1, q=1.0)
        assert np.allclose(Q_1d, Q_1d.T), "Q must be symmetric"
        
        # 2D
        Q_2d = ConstantVelocity2D.Q(dt=0.1, q=1.0)
        assert np.allclose(Q_2d, Q_2d.T), "Q must be symmetric"
    
    def test_process_noise_positive_definite(self):
        """Test that Q matrices are positive definite."""
        # 1D
        Q_1d = ConstantVelocity1D.Q(dt=0.1, q=1.0)
        eigvals_1d = np.linalg.eigvalsh(Q_1d)
        assert np.all(eigvals_1d > 0), "Q must be positive definite"
        
        # 2D
        Q_2d = ConstantVelocity2D.Q(dt=0.1, q=1.0)
        eigvals_2d = np.linalg.eigvalsh(Q_2d)
        assert np.all(eigvals_2d > 0), "Q must be positive definite"
    
    def test_process_noise_scaling(self):
        """Test that Q scales with q and dt correctly."""
        dt = 0.1
        q1 = 1.0
        q2 = 2.0
        
        Q1 = ConstantVelocity1D.Q(dt, q1)
        Q2 = ConstantVelocity1D.Q(dt, q2)
        
        # Q should scale linearly with q
        assert np.allclose(Q2, q2/q1 * Q1), "Q should scale linearly with q"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

