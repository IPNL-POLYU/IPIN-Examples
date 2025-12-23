"""
Common measurement models for state estimation.

Provides standard measurement models used in positioning and navigation:
- Range measurements (TOA/TDOA)
- Range-bearing measurements
- Position measurements

All models include proper singularity handling and input validation.
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings

from core.utils import normalize_jacobian_singularities, angle_diff


class RangeMeasurement2D:
    """
    Range-only measurement model for 2D positioning.
    
    Measurement: z = ||p - anchor|| + noise
    where p = [px, py] is position from state x
    
    Used in:
    - TOA/TDOA positioning
    - UWB ranging
    - ch3_estimators examples
    - ch8_sensor_fusion (tightly coupled)
    
    Example:
        >>> anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> model = RangeMeasurement2D(anchors)
        >>> x = np.array([5, 5, 1, 0.5])  # [px, py, vx, vy]
        >>> ranges = model.h(x)
        >>> ranges.shape
        (4,)  # One range per anchor
    """
    
    def __init__(self, anchors: np.ndarray, state_position_indices: Tuple[int, int] = (0, 1)):
        """
        Initialize range measurement model.
        
        Args:
            anchors: Anchor positions, shape (N, 2)
            state_position_indices: Indices of [px, py] in state vector (default: (0, 1))
        """
        self.anchors = np.asarray(anchors)
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 2:
            raise ValueError(f"Anchors must be (N, 2) array, got shape {self.anchors.shape}")
        
        self.n_anchors = len(self.anchors)
        self.pos_idx = state_position_indices
        
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: predicted ranges.
        
        Args:
            x: State vector (must contain position at self.pos_idx)
        
        Returns:
            Predicted ranges to all anchors, shape (N,)
        """
        position = x[list(self.pos_idx)]
        ranges = np.linalg.norm(self.anchors - position, axis=1)
        return ranges
    
    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian with singularity handling.
        
        Args:
            x: State vector
        
        Returns:
            Jacobian matrix, shape (N, len(x))
        """
        n_states = len(x)
        position = x[list(self.pos_idx)]
        
        # Compute differences and ranges
        diff = position - self.anchors  # (N, 2)
        ranges = np.linalg.norm(diff, axis=1)  # (N,)
        
        # Normalized Jacobian with singularity protection
        H_pos = normalize_jacobian_singularities(diff, ranges)  # (N, 2)
        
        # Embed in full state Jacobian
        H = np.zeros((self.n_anchors, n_states))
        H[:, self.pos_idx[0]] = H_pos[:, 0]
        H[:, self.pos_idx[1]] = H_pos[:, 1]
        
        return H


class RangeBearingMeasurement2D:
    """
    Range and bearing measurement model for 2D positioning.
    
    Measurements:
    - Range: z_r = ||p - landmark||
    - Bearing: z_θ = atan2(ly - py, lx - px)
    
    Used in:
    - Robot localization
    - Landmark-based navigation
    - ch3_estimators/example_ekf_range_bearing.py
    
    Example:
        >>> landmarks = np.array([[0, 0], [10, 0], [10, 10]])
        >>> model = RangeBearingMeasurement2D(landmarks)
        >>> x = np.array([5, 5, 1, 0.5])  # [px, py, vx, vy]
        >>> z = model.h(x)
        >>> z.shape
        (6,)  # [r0, θ0, r1, θ1, r2, θ2]
    """
    
    def __init__(self, landmarks: np.ndarray, state_position_indices: Tuple[int, int] = (0, 1)):
        """
        Initialize range-bearing measurement model.
        
        Args:
            landmarks: Landmark positions, shape (N, 2)
            state_position_indices: Indices of [px, py] in state (default: (0, 1))
        """
        self.landmarks = np.asarray(landmarks)
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 2:
            raise ValueError(f"Landmarks must be (N, 2) array, got shape {self.landmarks.shape}")
        
        self.n_landmarks = len(self.landmarks)
        self.pos_idx = state_position_indices
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: [range_0, bearing_0, range_1, bearing_1, ...].
        
        Args:
            x: State vector
        
        Returns:
            Measurements [r0, θ0, r1, θ1, ...], shape (2*N,)
        """
        position = x[list(self.pos_idx)]
        measurements = []
        
        for landmark in self.landmarks:
            dx = landmark[0] - position[0]
            dy = landmark[1] - position[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        
        return np.array(measurements)
    
    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian with singularity handling.
        
        Args:
            x: State vector
        
        Returns:
            Jacobian matrix, shape (2*N, len(x))
        """
        n_states = len(x)
        position = x[list(self.pos_idx)]
        
        H_list = []
        
        for landmark in self.landmarks:
            dx = landmark[0] - position[0]
            dy = landmark[1] - position[1]
            r = np.sqrt(dx**2 + dy**2)
            r_sq = max(r**2, 1e-12)  # Prevent division by zero
            
            # Initialize Jacobian rows
            H_range = np.zeros(n_states)
            H_bearing = np.zeros(n_states)
            
            if r < 1e-6:
                # Singularity: at landmark position
                # Set to zero (measurement not informative)
                pass
            else:
                # Range Jacobian: ∂r/∂[px, py] = [-dx/r, -dy/r]
                H_range[self.pos_idx[0]] = -dx / r
                H_range[self.pos_idx[1]] = -dy / r
                
                # Bearing Jacobian: ∂θ/∂[px, py] = [dy/r², -dx/r²]
                H_bearing[self.pos_idx[0]] = dy / r_sq
                H_bearing[self.pos_idx[1]] = -dx / r_sq
            
            H_list.extend([H_range, H_bearing])
        
        return np.array(H_list)
    
    def innovation(self, z_measured: np.ndarray, z_predicted: np.ndarray) -> np.ndarray:
        """
        Compute innovation with proper angle wrapping for bearings.
        
        CRITICAL: Bearing innovations must be wrapped to [-π, π].
        
        Args:
            z_measured: Measured [r0, θ0, r1, θ1, ...]
            z_predicted: Predicted [r0, θ0, r1, θ1, ...]
        
        Returns:
            Innovation vector with wrapped bearing differences
        """
        innovation = z_measured - z_predicted
        
        # Wrap bearing components (odd indices)
        for i in range(1, len(innovation), 2):
            innovation[i] = angle_diff(z_measured[i], z_predicted[i])
        
        return innovation


class PositionMeasurement2D:
    """
    Direct position measurement model (e.g., GPS, UWB position fix).
    
    Measurement: z = [px, py] + noise
    
    Used in:
    - GPS updates
    - UWB position fixes (loosely coupled)
    - Absolute position corrections
    - ch8_sensor_fusion (loosely coupled)
    
    Example:
        >>> model = PositionMeasurement2D()
        >>> x = np.array([5, 7, 1, 0.5])  # [px, py, vx, vy]
        >>> z = model.h(x)
        >>> z
        array([5, 7])  # Just the position
    """
    
    def __init__(self, state_position_indices: Tuple[int, int] = (0, 1)):
        """
        Initialize position measurement model.
        
        Args:
            state_position_indices: Indices of [px, py] in state (default: (0, 1))
        """
        self.pos_idx = state_position_indices
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function: extract position from state.
        
        Args:
            x: State vector
        
        Returns:
            Position [px, py]
        """
        return x[list(self.pos_idx)]
    
    def H(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian (trivial for linear measurement).
        
        Args:
            x: State vector
        
        Returns:
            Jacobian matrix, shape (2, len(x))
        """
        n_states = len(x)
        H = np.zeros((2, n_states))
        H[0, self.pos_idx[0]] = 1.0
        H[1, self.pos_idx[1]] = 1.0
        return H


def validate_measurement_inputs(
    x: np.ndarray,
    z: Optional[np.ndarray] = None,
    expected_x_dim: Optional[int] = None,
    expected_z_dim: Optional[int] = None,
    model_name: str = "measurement model"
) -> None:
    """
    Validate inputs to measurement models.
    
    Args:
        x: State vector
        z: Measurement vector (optional)
        expected_x_dim: Expected state dimension (if known)
        expected_z_dim: Expected measurement dimension (if known)
        model_name: Name of model for error messages
    
    Raises:
        ValueError: If validation fails
        TypeError: If wrong types provided
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{model_name}: state must be numpy array, got {type(x)}")
    
    if x.ndim != 1:
        raise ValueError(f"{model_name}: state must be 1D, got shape {x.shape}")
    
    if expected_x_dim is not None and x.shape[0] != expected_x_dim:
        raise ValueError(
            f"{model_name}: state dimension must be {expected_x_dim}, got {x.shape[0]}"
        )
    
    if z is not None:
        if not isinstance(z, np.ndarray):
            raise TypeError(f"{model_name}: measurement must be numpy array, got {type(z)}")
        
        if z.ndim != 1:
            raise ValueError(f"{model_name}: measurement must be 1D, got shape {z.shape}")
        
        if expected_z_dim is not None and z.shape[0] != expected_z_dim:
            raise ValueError(
                f"{model_name}: measurement dimension must be {expected_z_dim}, got {z.shape[0]}"
            )


def create_measurement_noise_covariance(
    noise_std: np.ndarray,
    correlation: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create measurement noise covariance matrix.
    
    Args:
        noise_std: Standard deviations for each measurement, shape (m,)
        correlation: Optional correlation matrix, shape (m, m)
                    If None, assumes uncorrelated measurements
    
    Returns:
        Measurement noise covariance R, shape (m, m)
    
    Example:
        >>> # Independent measurements
        >>> R = create_measurement_noise_covariance(np.array([0.5, 0.5, 0.05, 0.05]))
        >>> np.diag(R)
        array([0.25, 0.25, 0.0025, 0.0025])
        
        >>> # Correlated measurements
        >>> corr = np.array([[1, 0.5], [0.5, 1]])
        >>> R = create_measurement_noise_covariance(np.array([1.0, 1.0]), corr)
    """
    noise_std = np.asarray(noise_std)
    
    if noise_std.ndim != 1:
        raise ValueError(f"noise_std must be 1D array, got shape {noise_std.shape}")
    
    if np.any(noise_std < 0):
        raise ValueError("noise_std must be non-negative")
    
    m = len(noise_std)
    
    if correlation is None:
        # Uncorrelated: R = diag(σ²)
        return np.diag(noise_std**2)
    else:
        # Correlated: R = Σ C Σ where Σ = diag(σ)
        correlation = np.asarray(correlation)
        if correlation.shape != (m, m):
            raise ValueError(
                f"Correlation matrix must be ({m}, {m}), got {correlation.shape}"
            )
        
        # Check if valid correlation matrix
        if not np.allclose(correlation, correlation.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        if not np.allclose(np.diag(correlation), 1.0):
            warnings.warn(
                "Correlation matrix diagonal should be 1.0",
                RuntimeWarning
            )
        
        # Construct covariance
        Sigma = np.diag(noise_std)
        R = Sigma @ correlation @ Sigma
        
        return R

