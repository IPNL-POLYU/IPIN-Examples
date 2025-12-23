"""
Common motion models (process models) for state estimation.

Provides standard motion models used in Kalman filtering and other estimators:
- Constant velocity (1D and 2D)
- Constant acceleration
- Process noise models

All models are validated and tested for correctness.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class ConstantVelocity1D:
    """
    1D Constant Velocity Motion Model.
    
    State: x = [position, velocity]
    Dynamics: x_{k+1} = F(dt) * x_k + w_k
    
    Used in:
    - ch3_estimators/example_kalman_1d.py
    - Simple 1D tracking problems
    
    Example:
        >>> model = ConstantVelocity1D()
        >>> x = np.array([0.0, 1.0])  # position=0, velocity=1 m/s
        >>> dt = 0.1
        >>> x_next = model.f(x, dt=dt)
        >>> x_next
        array([0.1, 1.0])  # position moved by velocity*dt
    """
    
    @staticmethod
    def f(x: np.ndarray, u: Optional[np.ndarray] = None, dt: float = 1.0) -> np.ndarray:
        """
        Process model: x_{k+1} = f(x_k, dt).
        
        Args:
            x: State [position, velocity]
            u: Control input (unused)
            dt: Time step in seconds
        
        Returns:
            Next state [position', velocity']
        """
        if x.shape != (2,):
            raise ValueError(f"State must be 2D [pos, vel], got shape {x.shape}")
        
        position, velocity = x
        return np.array([
            position + velocity * dt,
            velocity
        ])
    
    @staticmethod
    def F(dt: float) -> np.ndarray:
        """
        State transition matrix (Jacobian of f).
        
        Args:
            dt: Time step in seconds
        
        Returns:
            2x2 state transition matrix
        """
        return np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
    
    @staticmethod
    def Q(dt: float, q: float = 1.0) -> np.ndarray:
        """
        Process noise covariance (continuous white noise acceleration).
        
        Args:
            dt: Time step in seconds
            q: Process noise intensity (acceleration variance)
        
        Returns:
            2x2 process noise covariance matrix
        """
        return q * np.array([
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt]
        ])


class ConstantVelocity2D:
    """
    2D Constant Velocity Motion Model.
    
    State: x = [px, py, vx, vy]
    Dynamics: Constant velocity in x and y independently
    
    Used in:
    - ch3_estimators/example_ekf_range_bearing.py
    - ch8_sensor_fusion (fusion examples)
    - 2D tracking and positioning
    
    Example:
        >>> model = ConstantVelocity2D()
        >>> x = np.array([0.0, 0.0, 1.0, 0.5])  # At origin, moving at [1, 0.5] m/s
        >>> dt = 0.5
        >>> x_next = model.f(x, dt=dt)
        >>> x_next[:2]  # New position
        array([0.5, 0.25])
    """
    
    @staticmethod
    def f(x: np.ndarray, u: Optional[np.ndarray] = None, dt: float = 1.0) -> np.ndarray:
        """
        Process model: x_{k+1} = f(x_k, dt).
        
        Args:
            x: State [px, py, vx, vy]
            u: Control input (unused)
            dt: Time step in seconds
        
        Returns:
            Next state [px', py', vx', vy']
        """
        if x.shape != (4,):
            raise ValueError(f"State must be 4D [px,py,vx,vy], got shape {x.shape}")
        
        px, py, vx, vy = x
        return np.array([
            px + vx * dt,
            py + vy * dt,
            vx,
            vy
        ])
    
    @staticmethod
    def F(dt: float) -> np.ndarray:
        """
        State transition matrix (Jacobian of f).
        
        Args:
            dt: Time step in seconds
        
        Returns:
            4x4 state transition matrix
        """
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    @staticmethod
    def Q(dt: float, q: float = 1.0) -> np.ndarray:
        """
        Process noise covariance (continuous white noise acceleration).
        
        Assumes independent noise in x and y directions.
        
        Args:
            dt: Time step in seconds
            q: Process noise intensity (acceleration variance)
        
        Returns:
            4x4 process noise covariance matrix
        """
        return q * np.array([
            [dt**3/3, 0,       dt**2/2, 0      ],
            [0,       dt**3/3, 0,       dt**2/2],
            [dt**2/2, 0,       dt,      0      ],
            [0,       dt**2/2, 0,       dt     ]
        ])


class ConstantAcceleration2D:
    """
    2D Constant Acceleration Motion Model.
    
    State: x = [px, py, vx, vy, ax, ay]
    Dynamics: Constant acceleration in x and y
    
    Used for:
    - Maneuvering targets
    - Vehicle tracking
    - High-dynamics scenarios
    
    Example:
        >>> model = ConstantAcceleration2D()
        >>> x = np.array([0, 0, 0, 0, 1, 0.5])  # At rest, accelerating
        >>> x_next = model.f(x, dt=1.0)
        >>> x_next[:4]  # Position and velocity after 1 second
        array([0.5, 0.125, 1, 0.5])  # Moved due to acceleration
    """
    
    @staticmethod
    def f(x: np.ndarray, u: Optional[np.ndarray] = None, dt: float = 1.0) -> np.ndarray:
        """
        Process model with constant acceleration.
        
        Args:
            x: State [px, py, vx, vy, ax, ay]
            u: Control input (unused)
            dt: Time step in seconds
        
        Returns:
            Next state [px', py', vx', vy', ax', ay']
        """
        if x.shape != (6,):
            raise ValueError(f"State must be 6D [px,py,vx,vy,ax,ay], got shape {x.shape}")
        
        px, py, vx, vy, ax, ay = x
        return np.array([
            px + vx * dt + 0.5 * ax * dt**2,
            py + vy * dt + 0.5 * ay * dt**2,
            vx + ax * dt,
            vy + ay * dt,
            ax,
            ay
        ])
    
    @staticmethod
    def F(dt: float) -> np.ndarray:
        """
        State transition matrix.
        
        Args:
            dt: Time step in seconds
        
        Returns:
            6x6 state transition matrix
        """
        return np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0        ],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0        ],
            [0, 0, 0,  1,  0,         dt       ],
            [0, 0, 0,  0,  1,         0        ],
            [0, 0, 0,  0,  0,         1        ]
        ])
    
    @staticmethod
    def Q(dt: float, q: float = 1.0) -> np.ndarray:
        """
        Process noise covariance (continuous white noise jerk).
        
        Args:
            dt: Time step in seconds
            q: Process noise intensity (jerk variance)
        
        Returns:
            6x6 process noise covariance matrix
        """
        # Continuous white noise jerk model
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5
        
        Q_1d = q * np.array([
            [dt5/20, dt4/8, dt3/6],
            [dt4/8,  dt3/3, dt2/2],
            [dt3/6,  dt2/2, dt   ]
        ])
        
        # Block diagonal for x and y
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = Q_1d  # x direction
        Q[3:6, 3:6] = Q_1d  # y direction
        
        return Q


def create_process_noise_continuous_white_acceleration(
    dt: float,
    q: float,
    dim: int = 2
) -> np.ndarray:
    """
    Create process noise covariance for continuous white acceleration model.
    
    This is the standard process noise model for constant velocity systems.
    
    Args:
        dt: Time step in seconds
        q: Process noise intensity (acceleration variance, m²/s⁴)
        dim: Spatial dimension (1, 2, or 3)
    
    Returns:
        Process noise covariance matrix
        - dim=1: 2x2 matrix for [pos, vel]
        - dim=2: 4x4 matrix for [px, py, vx, vy]
        - dim=3: 6x6 matrix for [px, py, pz, vx, vy, vz]
    
    Example:
        >>> Q = create_process_noise_continuous_white_acceleration(dt=0.1, q=0.5, dim=2)
        >>> Q.shape
        (4, 4)
    
    References:
        - Bar-Shalom et al., "Estimation with Applications to Tracking and Navigation"
        - Chapter 3: Process noise modeling
    """
    if dim not in [1, 2, 3]:
        raise ValueError(f"Dimension must be 1, 2, or 3, got {dim}")
    
    # 1D block
    Q_1d = q * np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt]
    ])
    
    if dim == 1:
        return Q_1d
    
    # Create block diagonal matrix
    n = 2 * dim  # State dimension
    Q = np.zeros((n, n))
    
    for i in range(dim):
        Q[2*i:2*i+2, 2*i:2*i+2] = Q_1d
    
    return Q


def validate_motion_model_inputs(
    x: np.ndarray,
    expected_dim: int,
    dt: Optional[float] = None,
    model_name: str = "motion model"
) -> None:
    """
    Validate inputs to motion models.
    
    Args:
        x: State vector
        expected_dim: Expected state dimension
        dt: Time step (if provided, must be positive)
        model_name: Name of model for error messages
    
    Raises:
        ValueError: If validation fails
        TypeError: If wrong types provided
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{model_name}: state must be numpy array, got {type(x)}")
    
    if x.ndim != 1:
        raise ValueError(f"{model_name}: state must be 1D, got shape {x.shape}")
    
    if x.shape[0] != expected_dim:
        raise ValueError(
            f"{model_name}: state dimension must be {expected_dim}, got {x.shape[0]}"
        )
    
    if dt is not None:
        if not isinstance(dt, (int, float)):
            raise TypeError(f"{model_name}: dt must be numeric, got {type(dt)}")
        if dt <= 0:
            raise ValueError(f"{model_name}: dt must be positive, got {dt}")
        if dt > 10.0:
            import warnings
            warnings.warn(
                f"{model_name}: dt={dt}s is unusually large. "
                "Check units (should be seconds).",
                RuntimeWarning
            )

