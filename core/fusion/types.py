"""Data types for multi-sensor fusion (Chapter 8).

This module defines the core data structures used across Chapter 8 sensor fusion
examples, including time-stamped measurements and temporal synchronization models.

Author: Navigation Engineer
References: Chapter 8 - Sensor Fusion
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class StampedMeasurement:
    """Generic time-stamped measurement packet used by fusion demos.
    
    This structure provides a unified interface for multi-sensor fusion,
    supporting different sensor types with varying measurement dimensions
    and covariances.
    
    Attributes:
        t: Timestamp in seconds (float, monotonic time).
        sensor: Sensor identifier (e.g., 'imu', 'uwb_range', 'lidar_odom').
        z: Measurement vector as numpy array.
        R: Measurement covariance matrix (m x m where m = len(z)).
        meta: Optional metadata dictionary for sensor-specific information
              (e.g., anchor_id for UWB, frame_id for camera).
    
    Example:
        >>> # UWB range measurement to anchor 3
        >>> uwb_meas = StampedMeasurement(
        ...     t=1.234,
        ...     sensor='uwb_range',
        ...     z=np.array([5.67]),
        ...     R=np.array([[0.01]]),
        ...     meta={'anchor_id': 3}
        ... )
        
        >>> # IMU acceleration measurement
        >>> imu_meas = StampedMeasurement(
        ...     t=1.234,
        ...     sensor='imu_accel',
        ...     z=np.array([0.1, 0.05, 9.81]),
        ...     R=np.diag([0.01, 0.01, 0.01]),
        ...     meta={'frame': 'body'}
        ... )
    """
    
    t: float
    sensor: str
    z: np.ndarray
    R: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate the measurement structure."""
        # Validate timestamp
        if not isinstance(self.t, (float, int)):
            raise TypeError(f"Timestamp must be numeric, got {type(self.t)}")
        if self.t < 0:
            raise ValueError(f"Timestamp must be non-negative, got {self.t}")
        
        # Validate sensor name
        if not isinstance(self.sensor, str) or not self.sensor:
            raise ValueError(f"Sensor must be a non-empty string, got {self.sensor}")
        
        # Validate measurement vector
        if not isinstance(self.z, np.ndarray):
            raise TypeError(f"Measurement z must be numpy array, got {type(self.z)}")
        if self.z.ndim != 1:
            raise ValueError(f"Measurement z must be 1D array, got shape {self.z.shape}")
        
        # Validate covariance matrix
        if not isinstance(self.R, np.ndarray):
            raise TypeError(f"Covariance R must be numpy array, got {type(self.R)}")
        if self.R.ndim != 2:
            raise ValueError(f"Covariance R must be 2D array, got shape {self.R.shape}")
        
        m = len(self.z)
        if self.R.shape != (m, m):
            raise ValueError(
                f"Covariance R shape {self.R.shape} must match "
                f"measurement dimension ({m}, {m})"
            )
        
        # Check symmetry (within tolerance)
        if not np.allclose(self.R, self.R.T):
            raise ValueError("Covariance R must be symmetric")
        
        # Check positive semi-definite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(self.R)
        if np.any(eigvals < -1e-10):  # small negative tolerance for numerical errors
            raise ValueError(f"Covariance R must be positive semi-definite, got eigenvalues {eigvals}")


@dataclass(frozen=True)
class TimeSyncModel:
    """Map sensor-local time to a common fusion time.
    
    This model handles temporal calibration between sensors by accounting for
    constant time offsets and clock drift. Essential for Chapter 8 temporal
    calibration demos (Section 8.5).
    
    The transformation is:
        t_fusion = (1 + drift) * t_sensor + offset
    
    Attributes:
        offset: Constant time offset in seconds. Positive offset means the
                sensor clock is ahead of the fusion clock.
        drift: Clock drift rate in seconds/second (dimensionless). A drift
               of 0.001 means the sensor gains 1 ms per second.
    
    Example:
        >>> # Sensor clock is 0.5 seconds behind fusion clock
        >>> sync = TimeSyncModel(offset=-0.5, drift=0.0)
        >>> sync.to_fusion_time(10.0)  # sensor time
        9.5  # fusion time
        
        >>> # Sensor clock drifts +1 ms per second and is 0.2s ahead
        >>> sync = TimeSyncModel(offset=0.2, drift=0.001)
        >>> sync.to_fusion_time(100.0)
        100.3  # = 100 * 1.001 + 0.2
    
    References:
        Chapter 8, Section 8.5 (Temporal Calibration and Synchronization)
    """
    
    offset: float = 0.0
    drift: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate the time synchronization parameters."""
        if not isinstance(self.offset, (float, int)):
            raise TypeError(f"Offset must be numeric, got {type(self.offset)}")
        
        if not isinstance(self.drift, (float, int)):
            raise TypeError(f"Drift must be numeric, got {type(self.drift)}")
        
        # Warn about unrealistic drift values (typically < 100 ppm = 0.0001)
        if abs(self.drift) > 0.01:
            import warnings
            warnings.warn(
                f"Clock drift of {self.drift} (= {self.drift * 1e6:.0f} ppm) "
                f"is unusually large. Typical values are < 100 ppm (0.0001).",
                UserWarning
            )
    
    def to_fusion_time(self, t_sensor: float) -> float:
        """Convert sensor-local time to fusion time.
        
        Args:
            t_sensor: Timestamp in sensor-local time (seconds).
        
        Returns:
            Timestamp in fusion time (seconds).
        
        Example:
            >>> sync = TimeSyncModel(offset=0.5, drift=0.001)
            >>> sync.to_fusion_time(10.0)
            10.51
        """
        return (1.0 + self.drift) * t_sensor + self.offset
    
    def to_sensor_time(self, t_fusion: float) -> float:
        """Convert fusion time to sensor-local time (inverse operation).
        
        Args:
            t_fusion: Timestamp in fusion time (seconds).
        
        Returns:
            Timestamp in sensor-local time (seconds).
        
        Example:
            >>> sync = TimeSyncModel(offset=0.5, drift=0.001)
            >>> t_fus = sync.to_fusion_time(10.0)
            >>> sync.to_sensor_time(t_fus)  # should recover 10.0
            10.0
        """
        return (t_fusion - self.offset) / (1.0 + self.drift)
    
    def is_synchronized(self, tolerance: float = 1e-6) -> bool:
        """Check if the sensor is already synchronized (identity transform).
        
        Args:
            tolerance: Tolerance for offset and drift (default 1 microsecond).
        
        Returns:
            True if both offset and drift are within tolerance of zero.
        
        Example:
            >>> TimeSyncModel(offset=0.0, drift=0.0).is_synchronized()
            True
            >>> TimeSyncModel(offset=0.5, drift=0.0).is_synchronized()
            False
        """
        return abs(self.offset) < tolerance and abs(self.drift) < tolerance

