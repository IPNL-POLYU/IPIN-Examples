"""Tightly Coupled IMU + UWB EKF Models for Chapter 8.

Implements the process and measurement models for tightly coupled fusion of:
- IMU: High-rate 2D accelerations and yaw rate
- UWB: Low-rate range measurements to known anchors

State Convention: [px, py, vx, vy, yaw] (5D)
- (px, py): position in map frame (meters)
- (vx, vy): velocity in map frame (m/s)
- yaw: heading angle (radians)

Process model: Simple constant-velocity + yaw integration
Measurement model: Direct range h(x) = ||p - anchor_i|| per anchor

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1.2 (Tightly Coupled)
"""

from typing import Callable, Tuple, Optional
from dataclasses import dataclass

import numpy as np


def interpolate_imu_measurements(
    t_query: float,
    t_imu: np.ndarray,
    accel_xy: np.ndarray,
    gyro_z: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Interpolate IMU measurements at a query time (Section 8.5.2 - Direct Interpolation).
    
    Implements direct linear interpolation method from Chapter 8, Section 8.5.2
    for handling asynchronous measurement timestamps. When a measurement arrives
    at time t that doesn't align with IMU samples, we interpolate IMU inputs.
    
    Args:
        t_query: Query time for interpolation (seconds).
        t_imu: IMU timestamps array (N,) - must be sorted.
        accel_xy: Accelerometer measurements (N, 2) in m/s².
        gyro_z: Gyroscope measurements (N,) in rad/s.
    
    Returns:
        Tuple of (u_interp, dt):
            u_interp: Interpolated control input [ax, ay, gyro_z] (3,)
            dt: Time since last IMU sample (for propagation)
    
    Raises:
        ValueError: If t_query is outside the range of t_imu.
    
    Example:
        >>> t_imu = np.array([0.0, 0.01, 0.02])
        >>> accel = np.array([[1.0, 0.5], [1.1, 0.6], [1.2, 0.7]])
        >>> gyro = np.array([0.1, 0.15, 0.2])
        >>> u, dt = interpolate_imu_measurements(0.015, t_imu, accel, gyro)
        >>> # At t=0.015 (halfway between 0.01 and 0.02):
        >>> # u ≈ [1.15, 0.65, 0.175]
    
    Notes:
        This implements the simplest interpolation method (linear).
        More sophisticated methods from Section 8.5.2:
        - Continuous-time propagation (integrate between samples)
        - Physics-based interpolation (use motion model)
        
        Linear interpolation is sufficient for high-rate IMU (≥100 Hz) when
        measurements arrive within ±10ms of IMU samples.
    
    References:
        Chapter 8, Section 8.5.2 (Measurement Timing and Interpolation)
    """
    if t_query < t_imu[0] or t_query > t_imu[-1]:
        raise ValueError(
            f"Query time {t_query:.6f}s is outside IMU range "
            f"[{t_imu[0]:.6f}, {t_imu[-1]:.6f}]s"
        )
    
    # Find the interval [t_imu[idx], t_imu[idx+1]] containing t_query
    idx = np.searchsorted(t_imu, t_query, side='right') - 1
    
    # Handle edge case: t_query exactly equals last IMU timestamp
    if idx >= len(t_imu) - 1:
        idx = len(t_imu) - 2
    
    t0 = t_imu[idx]
    t1 = t_imu[idx + 1]
    
    # Linear interpolation weight
    alpha = (t_query - t0) / (t1 - t0)
    
    # Interpolate accelerometer
    ax_interp = accel_xy[idx, 0] * (1 - alpha) + accel_xy[idx + 1, 0] * alpha
    ay_interp = accel_xy[idx, 1] * (1 - alpha) + accel_xy[idx + 1, 1] * alpha
    
    # Interpolate gyroscope
    gyro_interp = gyro_z[idx] * (1 - alpha) + gyro_z[idx + 1] * alpha
    
    # Control input
    u_interp = np.array([ax_interp, ay_interp, gyro_interp])
    
    # Time step from last IMU sample to query time
    dt = t_query - t0
    
    return u_interp, dt


@dataclass(frozen=True)
class StateIndex:
    """State vector indices for TC fusion.
    
    Enforces single convention: x = [px, py, vx, vy, yaw]
    
    Usage:
        x = np.array([1.0, 2.0, 0.5, 0.3, 0.1])
        position = x[StateIndex.PX:StateIndex.PY+1]  # [1.0, 2.0]
        velocity = x[StateIndex.VX:StateIndex.VY+1]  # [0.5, 0.3]
        yaw = x[StateIndex.YAW]                      # 0.1
    """
    PX: int = 0   # Position X
    PY: int = 1   # Position Y
    VX: int = 2   # Velocity X
    VY: int = 3   # Velocity Y
    YAW: int = 4  # Yaw angle
    
    @staticmethod
    def state_dim() -> int:
        """Return state dimension."""
        return 5


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


# ============================================================================
# Legacy API wrappers for demos
# ============================================================================
# These functions provide a simpler API for the demo scripts while maintaining
# consistency with the create_* factory functions above.


def tc_process_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Legacy wrapper: Process model for demos.
    
    State: x = [px, py, vx, vy, yaw]
    Control: u = [ax, ay, gyro_z]
    """
    f, _, _ = create_process_model()
    return f(x, u, dt)


def tc_process_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Legacy wrapper: Process Jacobian for demos.
    
    State: x = [px, py, vx, vy, yaw]
    Control: u = [ax, ay, gyro_z]
    """
    _, F, _ = create_process_model()
    return F(x, u, dt)


def tc_process_noise_covariance(
    dt: float,
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01
) -> np.ndarray:
    """Legacy wrapper: Process noise covariance for demos.
    
    Args:
        dt: Time step
        accel_noise_std: Acceleration noise std (m/s²)
        gyro_noise_std: Gyro noise std (rad/s)
    
    Returns:
        Q: Process noise covariance (5, 5)
    """
    # Map accel/gyro noise to state noise
    # For simplicity: σ_p ~ σ_accel * dt², σ_v ~ σ_accel * dt, σ_yaw ~ σ_gyro * dt
    process_noise_std = np.array([
        accel_noise_std,  # σ_p
        accel_noise_std,  # σ_v
        gyro_noise_std    # σ_yaw
    ])
    
    _, _, Q = create_process_model(process_noise_std)
    return Q(dt)


def tc_uwb_measurement_model(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Legacy wrapper: Predict ranges to all anchors.
    
    Args:
        x: State [px, py, vx, vy, yaw]
        anchors: Anchor positions (n_anchors, 2)
    
    Returns:
        Predicted ranges (n_anchors,)
    """
    px, py = x[StateIndex.PX], x[StateIndex.PY]
    position = np.array([px, py])
    
    ranges = np.linalg.norm(anchors - position, axis=1)
    return ranges


def tc_uwb_measurement_jacobian(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Legacy wrapper: Measurement Jacobian for all anchors.
    
    Args:
        x: State [px, py, vx, vy, yaw]
        anchors: Anchor positions (n_anchors, 2)
    
    Returns:
        H: Measurement Jacobian (n_anchors, 5)
    """
    px, py = x[StateIndex.PX], x[StateIndex.PY]
    position = np.array([px, py])
    
    n_anchors = anchors.shape[0]
    H = np.zeros((n_anchors, StateIndex.state_dim()))
    
    for i, anchor in enumerate(anchors):
        dx = px - anchor[0]
        dy = py - anchor[1]
        range_pred = np.sqrt(dx**2 + dy**2)
        
        # Avoid singularity
        if range_pred < 1e-6:
            range_pred = 1e-6
        
        H[i, StateIndex.PX] = dx / range_pred
        H[i, StateIndex.PY] = dy / range_pred
        # H[i, VX] = 0
        # H[i, VY] = 0
        # H[i, YAW] = 0
    
    return H
