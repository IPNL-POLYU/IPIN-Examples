"""Loosely Coupled IMU + UWB EKF Models for Chapter 8.

Implements loosely coupled fusion where UWB range measurements are first
solved for a position fix, then the position fix is fused with IMU.

Comparison with Tightly Coupled (TC):
- TC: Fuses raw UWB ranges directly (one update per anchor)
- LC: First solves for position using all ranges, then fuses position

State: [x, y, vx, vy, yaw] (5D) - same as TC
- (x, y): position in map frame (meters)
- (vx, vy): velocity in map frame (m/s)
- yaw: heading angle (radians)

Process model: Same as TC (2D IMU dead-reckoning)
Measurement model: Position fix h(x) = [px, py] (2D)

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1.1 (Loosely Coupled)
"""

from typing import Callable, Optional, Tuple

import numpy as np


def solve_uwb_position_wls(
    ranges: np.ndarray,
    anchor_positions: np.ndarray,
    range_noise_std: float = 0.05,
    anchor_noise_std: Optional[np.ndarray] = None,
    initial_guess: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    tolerance: float = 0.01,
    cov_floor_std: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Solve for 2D position from UWB ranges using Weighted Least Squares.
    
    This implements the iterative WLS position solver from Chapter 4,
    with proper measurement covariance handling for LC fusion.
    
    Key improvements over naive WLS:
    - Uses proper measurement covariance: W = R^{-1} where R = diag(σ_i²)
    - Supports anchor-dependent noise levels (e.g., based on SNR, NLOS flags)
    - Enforces covariance floor to prevent absurd certainty
    
    Args:
        ranges: Range measurements to each anchor (A,), NaN for dropouts
        anchor_positions: Anchor positions (A, 2)
        range_noise_std: Nominal range measurement noise std (meters, default 0.05)
        anchor_noise_std: Per-anchor noise std (A,), overrides range_noise_std if provided
                          Use this for anchor-dependent quality (e.g., NLOS, SNR)
        initial_guess: Initial position guess (2,), default is anchor centroid
        max_iterations: Maximum WLS iterations (default 10)
        tolerance: Convergence tolerance (meters, default 0.01)
        cov_floor_std: Minimum position std (meters, default 0.2)
                       Prevents overconfident covariance estimates
    
    Returns:
        Tuple of (position, covariance, converged):
            position: Estimated 2D position (2,) or None if failed
            covariance: Position covariance (2, 2) or None if failed
            converged: True if solver converged
    
    Example:
        >>> ranges = np.array([5.0, 7.0, 8.5, 6.2])
        >>> anchors = np.array([[0, 0], [20, 0], [20, 15], [0, 15]])
        >>> # Uniform noise
        >>> pos, cov, ok = solve_uwb_position_wls(ranges, anchors, range_noise_std=0.05)
        >>> 
        >>> # Anchor-dependent noise (e.g., anchor 1 has NLOS)
        >>> anchor_stds = np.array([0.05, 0.5, 0.05, 0.05])  # Anchor 1 degraded
        >>> pos, cov, ok = solve_uwb_position_wls(ranges, anchors, anchor_noise_std=anchor_stds)
    
    References:
        Chapter 4, Section 4.2: TOA Positioning with Iterative WLS
        Equations (4.14)-(4.23): Nonlinear TOA I-WLS
    """
    # Filter out NaN ranges
    valid_mask = ~np.isnan(ranges)
    if np.sum(valid_mask) < 3:
        # Need at least 3 ranges for 2D position
        return None, None, False
    
    ranges_valid = ranges[valid_mask]
    anchors_valid = anchor_positions[valid_mask]
    n_anchors = len(ranges_valid)
    
    # Determine per-measurement noise std
    if anchor_noise_std is not None:
        noise_std_valid = anchor_noise_std[valid_mask]
    else:
        noise_std_valid = np.full(n_anchors, range_noise_std)
    
    # Measurement covariance R = diag(σ_i²)
    R = np.diag(noise_std_valid**2)
    
    # Weight matrix W = R^{-1}
    W = np.diag(1.0 / (noise_std_valid**2))
    
    # Initial guess: centroid of valid anchors
    if initial_guess is None:
        pos = np.mean(anchors_valid, axis=0)
    else:
        pos = initial_guess.copy()
    
    # Iterative WLS
    converged = False
    for iteration in range(max_iterations):
        # Compute predicted ranges and residuals
        ranges_pred = np.linalg.norm(anchors_valid - pos, axis=1)
        residuals = ranges_valid - ranges_pred
        
        # Build measurement matrix H (Jacobian)
        # H[i, :] = -(anchor[i] - pos) / range_pred[i]
        H = np.zeros((n_anchors, 2))
        for i in range(n_anchors):
            if ranges_pred[i] > 1e-6:  # Avoid singularity
                diff = anchors_valid[i] - pos
                H[i, :] = -diff / ranges_pred[i]
        
        # WLS update: Δp = (H^T W H)^{-1} H^T W r
        try:
            HTW = H.T @ W
            HTWH = HTW @ H
            delta_pos = np.linalg.solve(HTWH, HTW @ residuals)
        except np.linalg.LinAlgError:
            # Singular matrix, solver failed
            return None, None, False
        
        # Update position
        pos = pos + delta_pos
        
        # Check convergence based on position update magnitude
        # (more robust than residual norm, which is biased by noise)
        if np.linalg.norm(delta_pos) < tolerance:
            converged = True
            break
        
        # Check for reasonable position (within reasonable bounds)
        anchor_min = np.min(anchors_valid, axis=0)
        anchor_max = np.max(anchors_valid, axis=0)
        margin = 50.0  # meters margin around anchors
        if np.any(pos < anchor_min - margin) or np.any(pos > anchor_max + margin):
            # Position diverged outside reasonable bounds
            return None, None, False
    
    # If we reached max iterations without diverging, consider it converged
    # (even if delta_pos was still above tolerance on the last iteration)
    if not converged and iteration == max_iterations - 1:
        # Check that final position is reasonable
        anchor_min = np.min(anchors_valid, axis=0)
        anchor_max = np.max(anchors_valid, axis=0)
        margin = 50.0
        if np.all(pos >= anchor_min - margin) and np.all(pos <= anchor_max + margin):
            converged = True
    
    # Compute covariance: Cov(p) = (H^T W H)^{-1}
    # This is the Cramér-Rao lower bound for unbiased estimators
    ranges_pred_final = np.linalg.norm(anchors_valid - pos, axis=1)
    H_final = np.zeros((n_anchors, 2))
    for i in range(n_anchors):
        if ranges_pred_final[i] > 1e-6:
            diff = anchors_valid[i] - pos
            H_final[i, :] = -diff / ranges_pred_final[i]
    
    try:
        HTW_final = H_final.T @ W
        HTWH_final = HTW_final @ H_final
        cov = np.linalg.inv(HTWH_final)
    except np.linalg.LinAlgError:
        # Default to conservative covariance if inversion fails
        cov = np.eye(2) * 1.0  # 1m std
        converged = False
    
    # Apply covariance floor to prevent overconfidence
    # This accounts for unmodeled errors (multipath, NLOS, anchor position errors, etc.)
    cov_floor = np.eye(2) * (cov_floor_std**2)
    
    # For each diagonal element, enforce minimum variance
    for i in range(2):
        if cov[i, i] < cov_floor[i, i]:
            # Scale covariance to meet floor (preserve correlation structure)
            scale_factor = cov_floor[i, i] / cov[i, i]
            cov[i, :] *= np.sqrt(scale_factor)
            cov[:, i] *= np.sqrt(scale_factor)
    
    return pos, cov, converged


def create_lc_process_model(
    process_noise_std: np.ndarray = None
) -> Tuple[Callable, Callable, Callable]:
    """Create process model for LC fusion (same as TC).
    
    Args:
        process_noise_std: [σ_p, σ_v, σ_yaw] (default: [0.01, 0.05, 0.01])
    
    Returns:
        Tuple of (process_model, process_jacobian, process_noise_cov)
    """
    # Reuse TC process model
    from ch8_sensor_fusion.tc_models import create_process_model
    return create_process_model(process_noise_std)


def create_lc_position_measurement_model(
    position_noise_std: np.ndarray = None
) -> Tuple[Callable, Callable, Callable]:
    """Create position measurement model for LC fusion.
    
    In LC fusion, the UWB position fix is treated as a 2D position measurement.
    
    Measurement: z = [px_meas, py_meas]
    Model: h(x) = [px, py] (direct observation of position state)
    
    Args:
        position_noise_std: Position measurement noise [σ_x, σ_y]
                            Default: [0.5, 0.5] meters
    
    Returns:
        Tuple of (measurement_model, measurement_jacobian, measurement_noise_cov)
    """
    if position_noise_std is None:
        # Conservative estimate for UWB position fix uncertainty
        position_noise_std = np.array([0.5, 0.5])
    else:
        position_noise_std = np.asarray(position_noise_std)
    
    def measurement_model(x: np.ndarray) -> np.ndarray:
        """Predict position measurement h(x) = [px, py]."""
        return x[:2]  # Extract position from state
    
    def measurement_jacobian(x: np.ndarray) -> np.ndarray:
        """Compute H = ∂h/∂x."""
        # h(x) = [px, py] = [x[0], x[1]]
        # H = [[1, 0, 0, 0, 0],
        #      [0, 1, 0, 0, 0]]
        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        return H
    
    def measurement_noise_cov() -> np.ndarray:
        """Return R for position measurement."""
        return np.diag(position_noise_std**2)
    
    return measurement_model, measurement_jacobian, measurement_noise_cov


def create_lc_fusion_ekf(
    initial_state: np.ndarray,
    initial_cov: np.ndarray,
    process_noise_std: np.ndarray = None,
) -> any:
    """Create and initialize loosely coupled fusion EKF.
    
    Args:
        initial_state: Initial state [px, py, vx, vy, yaw] (5,)
        initial_cov: Initial covariance (5, 5)
        process_noise_std: Process noise std [σ_p, σ_v, σ_yaw]
    
    Returns:
        Initialized ExtendedKalmanFilter instance
    """
    from core.estimators import ExtendedKalmanFilter
    
    # Create process model
    process_f, process_F, process_Q = create_lc_process_model(process_noise_std)
    
    # Dummy measurement model (will be replaced during updates)
    def dummy_h(x):
        return np.zeros(2)
    
    def dummy_H(x):
        return np.zeros((2, 5))
    
    def dummy_R():
        return np.eye(2)
    
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

