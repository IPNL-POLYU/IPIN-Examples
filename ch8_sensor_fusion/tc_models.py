"""Tightly Coupled IMU + UWB EKF Models for Chapter 8.

Implements the process and measurement models for tightly coupled fusion of:
- IMU: High-rate 2D accelerations and yaw rate
- UWB: Low-rate range measurements to known anchors

State: [x, y, vx, vy, yaw] (5D)
- (x, y): position in map frame (meters)
- (vx, vy): velocity in map frame (m/s)
- yaw: heading angle (radians)

Process model: Simple constant-velocity + yaw integration
Measurement model: Direct range h(x) = ||p - anchor_i|| per anchor

Author: Li-Ta Hsu
References: Chapter 8 - Tightly Coupled Fusion
"""

from typing import Callable, Tuple

import numpy as np


def create_process_model(
    process_noise_std: np.ndarray = None
) -> Tuple[Callable, Callable, Callable]:
    """Create process model functions for 2D IMU-based dead reckoning.
    
    State: x = [px, py, vx, vy, yaw]  (5D)
    Control: u = [ax, ay, gyro_z]  (3D)
    
    Args:
        process_noise_std: [σ_p, σ_v, σ_yaw] (default: [0.01, 0.05, 0.01])
    
    Returns:
        Tuple of (process_model, process_jacobian, process_noise_cov)
    """
    if process_noise_std is None:
        process_noise_std = np.array([0.01, 0.05, 0.01])
    else:
        process_noise_std = np.asarray(process_noise_std)
    
    def process_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear state propagation f(x, u, dt)."""
        px, py, vx, vy, yaw = x
        ax, ay, gyro_z = u
        
        # Rotate accelerations from body to map frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        ax_map = ax * cos_yaw - ay * sin_yaw
        ay_map = ax * sin_yaw + ay * cos_yaw
        
        # Propagate state
        px_next = px + vx * dt
        py_next = py + vy * dt
        vx_next = vx + ax_map * dt
        vy_next = vy + ay_map * dt
        yaw_next = yaw + gyro_z * dt
        
        # Wrap yaw to [-π, π]
        yaw_next = np.arctan2(np.sin(yaw_next), np.cos(yaw_next))
        
        return np.array([px_next, py_next, vx_next, vy_next, yaw_next])
    
    def process_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Compute F = ∂f/∂x."""
        _, _, vx, vy, yaw = x
        ax, ay, _ = u
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Jacobian of accelerations w.r.t. yaw
        dax_map_dyaw = -ax * sin_yaw - ay * cos_yaw
        day_map_dyaw = ax * cos_yaw - ay * sin_yaw
        
        F = np.array([
            [1, 0, dt, 0,  0],
            [0, 1, 0,  dt, 0],
            [0, 0, 1,  0,  dax_map_dyaw * dt],
            [0, 0, 0,  1,  day_map_dyaw * dt],
            [0, 0, 0,  0,  1]
        ])
        
        return F
    
    def process_noise_cov(dt: float) -> np.ndarray:
        """Compute process noise covariance Q(dt)."""
        σ_p, σ_v, σ_yaw = process_noise_std
        
        Q = np.diag([
            (σ_p * dt)**2,
            (σ_p * dt)**2,
            (σ_v * dt)**2,
            (σ_v * dt)**2,
            (σ_yaw * dt)**2
        ])
        
        return Q
    
    return process_model, process_jacobian, process_noise_cov


def create_uwb_range_measurement_model(
    anchor_position: np.ndarray,
    range_noise_std: float = 0.05
) -> Tuple[Callable, Callable, Callable]:
    """Create UWB range measurement model for a single anchor.
    
    Measurement: z = range to anchor
    Model: h(x) = ||p - anchor||
    
    Args:
        anchor_position: Anchor position [x, y] (2,)
        range_noise_std: Range noise std (meters)
    
    Returns:
        Tuple of (measurement_model, measurement_jacobian, measurement_noise_cov)
    """
    anchor = np.asarray(anchor_position)
    
    def measurement_model(x: np.ndarray) -> np.ndarray:
        """Predict range h(x)."""
        px, py = x[0], x[1]
        
        dx = px - anchor[0]
        dy = py - anchor[1]
        range_pred = np.sqrt(dx**2 + dy**2)
        
        return np.array([range_pred])
    
    def measurement_jacobian(x: np.ndarray) -> np.ndarray:
        """Compute H = ∂h/∂x."""
        px, py = x[0], x[1]
        
        dx = px - anchor[0]
        dy = py - anchor[1]
        range_pred = np.sqrt(dx**2 + dy**2)
        
        # Avoid singularity
        if range_pred < 1e-6:
            range_pred = 1e-6
        
        H = np.array([[
            dx / range_pred,  # ∂h/∂px
            dy / range_pred,  # ∂h/∂py
            0.0,              # ∂h/∂vx
            0.0,              # ∂h/∂vy
            0.0               # ∂h/∂yaw
        ]])
        
        return H
    
    def measurement_noise_cov() -> np.ndarray:
        """Return R."""
        return np.array([[range_noise_std**2]])
    
    return measurement_model, measurement_jacobian, measurement_noise_cov


def create_tc_fusion_ekf(
    initial_state: np.ndarray,
    initial_cov: np.ndarray,
    process_noise_std: np.ndarray = None,
) -> any:
    """Create and initialize tightly coupled fusion EKF.
    
    Args:
        initial_state: Initial state [px, py, vx, vy, yaw] (5,)
        initial_cov: Initial covariance (5, 5)
        process_noise_std: Process noise std [σ_p, σ_v, σ_yaw]
    
    Returns:
        Initialized ExtendedKalmanFilter instance
    """
    from core.estimators import ExtendedKalmanFilter
    
    # Create process model functions (but we won't use them directly - EKF will be updated manually)
    # This simplified version just initializes state
    process_f, process_F, process_Q = create_process_model(process_noise_std)
    
    # Dummy measurement model (we'll use per-anchor models during updates)
    def dummy_h(x):
        return np.zeros(1)
    
    def dummy_H(x):
        return np.zeros((1, 5))
    
    def dummy_R():
        return np.eye(1)
    
    ekf = ExtendedKalmanFilter(
        process_model=process_f,
        process_jacobian=process_F,
        measurement_model=dummy_h,
        measurement_jacobian=dummy_H,
        Q=process_Q,
        R=dummy_R,
        x0=initial_state.copy(),
        P0=initial_cov.copy()
    )
    
    return ekf
