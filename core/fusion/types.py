"""Data types for multi-sensor fusion (Chapter 8).

This module defines the core data structures used across Chapter 8 sensor fusion
examples, including time-stamped measurements and temporal synchronization models.

Author: Li-Ta Hsu
References: Chapter 8 - Sensor Fusion
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SENSOR_IMU = "imu"
SENSOR_UWB_RANGE = "uwb_range"
SENSOR_UWB_RANGES_EPOCH = "uwb_ranges_epoch"
SENSOR_UWB_RANGES_BATCH = "uwb_ranges_batch"


@dataclass(frozen=True)
class StampedMeasurement:
    """Generic time-stamped measurement packet used by fusion demos.

    This structure provides a unified interface for multi-sensor fusion,
    supporting different sensor types with varying measurement dimensions
    and covariances.

    Attributes:
        t: Timestamp in seconds (float, monotonic time).
        sensor: Sensor identifier (for example ``SENSOR_IMU``,
                ``SENSOR_UWB_RANGE``, or ``SENSOR_UWB_RANGES_EPOCH``).
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
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_measurement(
        cls,
        *,
        timestamp_s: float,
        sensor_type: str,
        measurement_vector: np.ndarray,
        measurement_covariance: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> "StampedMeasurement":
        """Construct a packet with semantic keyword names.

        The historical ``t/sensor/z/R/meta`` constructor remains supported for
        equation-facing and serialized-data code.
        """
        return cls(
            t=timestamp_s,
            sensor=sensor_type,
            z=measurement_vector,
            R=measurement_covariance,
            meta={} if metadata is None else metadata,
        )

    @property
    def timestamp_s(self) -> float:
        """Measurement timestamp in seconds."""
        return float(self.t)

    @property
    def sensor_type(self) -> str:
        """Descriptive alias for the historical ``sensor`` field."""
        return self.sensor

    @property
    def measurement_vector(self) -> np.ndarray:
        """Sensor measurement vector; shape depends on ``sensor_type``."""
        return self.z

    @property
    def measurement_covariance(self) -> np.ndarray:
        """Measurement covariance, shape ``(m, m)`` for an m-vector."""
        return self.R

    @property
    def metadata(self) -> dict[str, Any]:
        """Sensor-specific open-ended metadata."""
        return self.meta

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
            raise ValueError(
                f"Measurement z must be 1D array, got shape {self.z.shape}"
            )

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
            raise ValueError(
                f"Covariance R must be positive semi-definite, got eigenvalues {eigvals}"
            )


class FusionHistory(dict[str, Any]):
    """Dictionary-compatible result from Chapter 8 fusion runners.

    The fusion examples historically returned plain dictionaries. This subclass
    keeps all dict operations working while adding named properties for fields
    whose string keys are easy for readers to mistype or misread.
    """

    @property
    def t(self) -> list[float]:
        """Backward-compatible alias for ``timestamps_s``."""
        return self.timestamps_s

    @property
    def timestamps_s(self) -> list[float]:
        """Fusion update timestamps in seconds."""
        return self["t"]

    @property
    def x_est(self) -> list[np.ndarray]:
        """Backward-compatible alias for ``estimated_state_vectors``."""
        return self.estimated_state_vectors

    @property
    def estimated_state_vectors(self) -> list[np.ndarray]:
        """Estimated EKF state vectors in the runner's documented ordering."""
        return self["x_est"]

    @property
    def p_trace(self) -> list[float]:
        """Backward-compatible alias for ``state_covariance_trace``."""
        return self.state_covariance_trace

    @property
    def state_covariance_trace(self) -> list[float]:
        """Trace of the EKF state covariance after each stored step."""
        return self["P_trace"]

    @property
    def innovations(self) -> list[np.ndarray]:
        """Backward-compatible alias for ``innovation_vectors``."""
        return self.innovation_vectors

    @property
    def innovation_vectors(self) -> list[np.ndarray]:
        """Measurement residual vectors used for updates."""
        return self["innovations"]

    @property
    def nis(self) -> list[float]:
        """Backward-compatible alias for ``normalized_innovation_squared``."""
        return self.normalized_innovation_squared

    @property
    def normalized_innovation_squared(self) -> list[float]:
        """Normalized innovation squared values."""
        return self["nis"]

    @property
    def measurement_accepted(self) -> list[bool]:
        """True when a UWB update was accepted by gating/update logic."""
        return self["measurement_accepted"]

    @property
    def gated(self) -> list[bool]:
        """Deprecated alias for :attr:`measurement_accepted`."""
        return self["gated"]


@dataclass(frozen=True)
class TimeSyncModel:
    """Map sensor-local time to a common fusion time.

    This model handles temporal calibration between sensors by accounting for
    constant time offsets and clock drift. Essential for Chapter 8 temporal
    calibration demos (Section 8.5).

    The transformation is:
        t_fusion = (1 + drift) * t_sensor + offset

    Read ``offset`` as the sensor-to-fusion offset: the number of seconds to add
    to a sensor timestamp after drift correction to express it on the fusion
    clock. A positive offset maps a sensor timestamp to a later fusion time,
    which means the sensor clock is behind the fusion clock by that offset.
    The explicit ``offset_sensor_to_fusion_sec`` property exposes the same value
    with a name that states the direction.

    Attributes:
        offset: Backward-compatible name for the sensor-to-fusion time offset
                in seconds. Prefer ``offset_sensor_to_fusion_sec`` in new code.
        drift: Clock drift rate in seconds/second (dimensionless). A drift
               of 0.001 means the sensor gains 1 ms per second.

    Example:
        >>> # Sensor clock is 0.5 seconds behind fusion clock
        >>> sync = TimeSyncModel(offset=0.5, drift=0.0)
        >>> sync.to_fusion_time(10.0)
        10.5

        >>> # Sensor clock is 0.2s ahead and drifts +1 ms per second
        >>> sync = TimeSyncModel(offset=-0.2, drift=0.001)
        >>> # 100 * 1.001 - 0.2
        >>> round(sync.to_fusion_time(100.0), 1)
        99.9

    References:
        Chapter 8, Section 8.5 (Temporal Calibration and Synchronization)
    """

    offset: float = 0.0
    drift: float = 0.0

    @classmethod
    def from_sensor_to_fusion_offset(
        cls, offset_sensor_to_fusion_sec: float, drift: float = 0.0
    ) -> "TimeSyncModel":
        """Create a model with an explicit sensor-to-fusion offset name.

        This is equivalent to ``TimeSyncModel(offset=offset_sensor_to_fusion_sec,
        drift=drift)``. It exists so new reader-facing code can state the
        direction without breaking older examples that pass ``offset``.
        """
        return cls(offset=offset_sensor_to_fusion_sec, drift=drift)

    @property
    def offset_sensor_to_fusion_sec(self) -> float:
        """Seconds added to drift-corrected sensor time to get fusion time."""
        return self.offset

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
                UserWarning,
                stacklevel=2,
            )

    def to_fusion_time(self, t_sensor: float) -> float:
        """Convert sensor-local time to fusion time.

        Args:
            t_sensor: Timestamp in sensor-local time (seconds).

        Returns:
            Timestamp in fusion time (seconds).

        Example:
            >>> sync = TimeSyncModel(offset=0.5, drift=0.001)
            >>> round(sync.to_fusion_time(10.0), 6)
            10.51

        Rounded because the exact value is 10.509999999999998: (1.001 * 10.0)
        is not 10.01 in binary floating point. The example used to claim the
        arithmetic answer as though it were the repr, which made it the one
        false doctest in this file and kept the file off the allowlist in
        tests/test_doctests_that_pass_keep_passing.py.
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
