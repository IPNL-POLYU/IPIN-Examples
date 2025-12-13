"""
Data structures for proprioceptive and environmental sensors (Chapter 6).

This module defines the shared data types used across all Chapter 6 algorithms:
    - Sensor time-series packets (IMU, wheel speed, magnetometer, barometer)
    - Navigation state representations (quaternion-velocity-position)
    - Bias-augmented state for realistic INS/EKF implementations

All structures use NumPy arrays for efficient numerical operations and are
designed to work seamlessly with core/estimators and core/sim modules.

Time Base Convention:
    All timestamps are float seconds (monotonic), stored as np.ndarray.

Frame Conventions:
    - B: Body frame (sensor frame)
    - M: Map frame (navigation frame, typically ENU)
    - S: Speed frame (for wheel odometry, defined in Chapter 6)
    - A: Attitude frame (intermediate frame for vehicle odometry)

References:
    Chapter 6: Dead Reckoning and Proprioceptive Sensors
    Section 6.1: IMU strapdown integration
    Section 6.2: Wheel odometry
    Section 6.3: Pedestrian dead reckoning
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class ImuSeries:
    """
    Time-series packet for Inertial Measurement Unit (IMU) data.

    Stores synchronized accelerometer and gyroscope measurements in the
    body frame B, used for strapdown integration (Eqs. (6.2)-(6.10)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        accel: Specific force measurements in body frame B, shape (N, 3).
               Units: m/s². Includes gravity (before correction via Eq. (6.9)).
        gyro: Angular velocity measurements in body frame B, shape (N, 3).
              Units: rad/s. Raw measurements (before correction via Eq. (6.6)).
        meta: Optional metadata dict. May include:
              - 'sample_rate_hz': float, nominal sampling rate
              - 'sensor_id': str, device identifier
              - 'frame': str, typically 'body' or 'B'
              - 'units': dict, e.g. {'accel': 'm/s^2', 'gyro': 'rad/s'}

    Notes:
        - Measurements are assumed to be in the body frame at each timestamp.
        - For error modeling with biases, see NavStateQPVPBias and Eqs. (6.5)-(6.6).
        - frozen=True ensures immutability for safer data pipelines.

    Related Equations:
        - Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
        - Eq. (6.9): Accel correction (f = f̃ - b_a - n_a)
        - Eq. (6.2): Quaternion kinematics using corrected gyro
        - Eq. (6.7): Velocity update using corrected accel
    """

    t: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of IMU data."""
        # Validate t is 1D
        if self.t.ndim != 1:
            raise ValueError(
                f"ImuSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        # Validate accel is (N, 3)
        if self.accel.shape != (n_samples, 3):
            raise ValueError(
                f"ImuSeries.accel must have shape ({n_samples}, 3), "
                f"got {self.accel.shape}"
            )

        # Validate gyro is (N, 3)
        if self.gyro.shape != (n_samples, 3):
            raise ValueError(
                f"ImuSeries.gyro must have shape ({n_samples}, 3), "
                f"got {self.gyro.shape}"
            )


@dataclass(frozen=True)
class WheelSpeedSeries:
    """
    Time-series packet for wheel speed / vehicle odometry data.

    Stores velocity measurements in the speed frame S (defined in Chapter 6
    for wheel odometry systems). Used for vehicle dead reckoning and
    integrated IMU+wheel EKF (Eqs. (6.11)-(6.15), (6.33)-(6.38)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        v_s: Velocity in speed frame S, shape (N, 3). Units: m/s.
             Typically v_s = [v_x, 0, 0] for forward motion in a vehicle.
             See Eq. (6.11) for the speed frame definition.
        meta: Optional metadata dict. May include:
              - 'lever_arm_b': np.ndarray (3,), lever arm in body frame (meters)
              - 'frame': str, typically 'speed' or 'S'
              - 'vehicle_type': str, e.g. 'differential_drive', 'ackermann'

    Notes:
        - Speed frame S is typically aligned with vehicle forward direction.
        - Lever arm compensation (Eq. (6.11)) transforms v_s to attitude frame.
        - For slip/failure mode demos, inject noise or bias in v_s.

    Related Equations:
        - Eq. (6.11): Lever arm compensation (v^A = v^S - [ω_B ×] l^B)
        - Eq. (6.14): Attitude to map velocity transform
        - Eq. (6.15): Position update using wheel speed
        - Eqs. (6.33)-(6.38): Wheel speed measurement model for EKF
    """

    t: np.ndarray
    v_s: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of wheel speed data."""
        if self.t.ndim != 1:
            raise ValueError(
                f"WheelSpeedSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.v_s.shape != (n_samples, 3):
            raise ValueError(
                f"WheelSpeedSeries.v_s must have shape ({n_samples}, 3), "
                f"got {self.v_s.shape}"
            )


@dataclass(frozen=True)
class MagnetometerSeries:
    """
    Time-series packet for magnetometer data.

    Stores magnetic field measurements in the device/body frame, used for
    heading estimation in PDR and environmental sensor fusion
    (Eqs. (6.51)-(6.53)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        mag: Magnetic field vector in body/device frame, shape (N, 3).
             Units: μT (microtesla) or normalized. Typically contains
             Earth's magnetic field plus indoor disturbances.
        meta: Optional metadata dict. May include:
              - 'frame': str, typically 'body' or 'device'
              - 'calibration': dict, hard-iron/soft-iron calibration params
              - 'disturbance_intervals': list of (start_idx, end_idx) tuples
              - 'units': str, e.g. 'uT', 'normalized'

    Notes:
        - Indoor environments often have magnetic disturbances (steel, electronics).
        - Tilt compensation (Eq. (6.52)) requires attitude (roll/pitch) from IMU.
        - Raw measurements must be calibrated for hard-iron and soft-iron effects.

    Related Equations:
        - Eq. (6.51): Magnetometer heading definition
        - Eq. (6.52): Tilt compensation for heading
        - Eq. (6.53): Heading computation from tilt-compensated mag field
    """

    t: np.ndarray
    mag: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency of magnetometer data."""
        if self.t.ndim != 1:
            raise ValueError(
                f"MagnetometerSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.mag.shape != (n_samples, 3):
            raise ValueError(
                f"MagnetometerSeries.mag must have shape ({n_samples}, 3), "
                f"got {self.mag.shape}"
            )


@dataclass(frozen=True)
class BarometerSeries:
    """
    Time-series packet for barometric pressure data.

    Stores atmospheric pressure measurements for altitude estimation and
    floor change detection (Eq. (6.54)).

    Attributes:
        t: Timestamps in seconds, shape (N,). Monotonic time.
        pressure: Atmospheric pressure measurements, shape (N,).
                  Units: Pa (Pascals) or hPa (hectopascals).
                  Must be explicit in meta['units'].
        meta: Optional metadata dict. May include:
              - 'units': str, REQUIRED ('Pa' or 'hPa')
              - 'p0': float, reference pressure at sea level (Pa)
              - 'T': float, temperature in Kelvin for altitude model
              - 'floor_labels': np.ndarray (N,), manual floor annotations

    Notes:
        - Pressure-to-altitude conversion via Eq. (6.54) requires p0 and T.
        - Barometer drift and offset handling are critical for long-term use.
        - Indoor pressure can be affected by HVAC systems and weather changes.

    Related Equations:
        - Eq. (6.54): Pressure to altitude conversion (barometric formula)
        - Eq. (6.55): Generic state/measurement model (for smoothing helper)
    """

    t: np.ndarray
    pressure: np.ndarray
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate shape consistency and units specification."""
        if self.t.ndim != 1:
            raise ValueError(
                f"BarometerSeries.t must be 1D array, got shape {self.t.shape}"
            )

        n_samples = self.t.shape[0]

        if self.pressure.shape != (n_samples,):
            raise ValueError(
                f"BarometerSeries.pressure must have shape ({n_samples},), "
                f"got {self.pressure.shape}"
            )

        # Enforce units specification
        if "units" not in self.meta:
            raise ValueError(
                "BarometerSeries.meta must include 'units' field ('Pa' or 'hPa')"
            )


@dataclass
class NavStateQPVP:
    """
    Minimal navigation state: Quaternion, Velocity, Position (QPVP).

    Represents the core navigation state used in Chapter 6 strapdown
    integration and EKF formulations (base state for Eq. (6.16)).

    Attributes:
        q: Quaternion representing attitude (body to map frame rotation).
           Shape (4,), scalar-first convention: [q0, q1, q2, q3] where
           q0 is the scalar part and [q1, q2, q3] is the vector part.
           Must satisfy ||q|| = 1 (unit quaternion).
        v: Velocity in map frame M, shape (3,). Units: m/s.
           Typically M is ENU (East-North-Up) or NED (North-East-Down).
        p: Position in map frame M, shape (3,). Units: m.
           3D position [x, y, z] or [E, N, U] depending on frame choice.

    Notes:
        - This is a MUTABLE dataclass (frozen=False) to allow state updates.
        - Quaternion normalization is the user's responsibility after updates.
        - Use this for simple examples; use NavStateQPVPBias for realistic EKF.

    Related Equations:
        - Eq. (6.2): Quaternion kinematics (dq/dt = 0.5 * Ω(ω) * q)
        - Eq. (6.7): Velocity update (v_k = v_{k-1} + (C_B^M f + g) Δt)
        - Eq. (6.10): Position update (p_k = p_{k-1} + v_k Δt)
        - Eq. (6.16): State definition for IMU+wheel EKF (with biases)

    Example:
        >>> import numpy as np
        >>> # Initial state: zero velocity, origin position, level attitude
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        >>> v0 = np.zeros(3)
        >>> p0 = np.zeros(3)
        >>> state = NavStateQPVP(q=q0, v=v0, p=p0)
    """

    q: np.ndarray
    v: np.ndarray
    p: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape and basic consistency of navigation state."""
        if self.q.shape != (4,):
            raise ValueError(
                f"NavStateQPVP.q must have shape (4,), got {self.q.shape}"
            )

        if self.v.shape != (3,):
            raise ValueError(
                f"NavStateQPVP.v must have shape (3,), got {self.v.shape}"
            )

        if self.p.shape != (3,):
            raise ValueError(
                f"NavStateQPVP.p must have shape (3,), got {self.p.shape}"
            )

        # Warn if quaternion is not normalized (tolerance 1e-3)
        q_norm = np.linalg.norm(self.q)
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            import warnings

            warnings.warn(
                f"NavStateQPVP initialized with non-unit quaternion "
                f"(||q|| = {q_norm:.6f}). Consider normalizing.",
                UserWarning,
            )


@dataclass
class NavStateQPVPBias:
    """
    Augmented navigation state: Quaternion, Velocity, Position, and Biases.

    Extended state including IMU biases for realistic EKF implementations.
    Recommended for Chapter 6 EKF examples that model sensor errors
    (Eqs. (6.5)-(6.6), (6.9), and state definition in Eq. (6.16)).

    Attributes:
        q: Quaternion (body to map frame), shape (4,). Scalar-first [q0, q1, q2, q3].
        v: Velocity in map frame M, shape (3,). Units: m/s.
        p: Position in map frame M, shape (3,). Units: m.
        b_g: Gyroscope bias in body frame B, shape (3,). Units: rad/s.
             Slow-varying bias modeled as random walk (Eq. (6.5)).
        b_a: Accelerometer bias in body frame B, shape (3,). Units: m/s².
             Slow-varying bias modeled as random walk (Eq. (6.9)).

    Notes:
        - Full state dimension is 13 (4 + 3 + 3 + 3).
        - Bias evolution is typically modeled as random walk in EKF process model.
        - For error-state EKF, biases are estimated as additive corrections.

    Related Equations:
        - Eq. (6.5): Gyro error model (ω̃ = ω + b_g + n_g)
        - Eq. (6.6): Gyro correction (ω = ω̃ - b_g - n_g)
        - Eq. (6.9): Accel error model (f̃ = f + b_a + n_a)
        - Eq. (6.16): State vector for IMU+wheel EKF [q, v, p, b_g, b_a]
        - Eqs. (6.17)-(6.32): Process model and covariance propagation

    Example:
        >>> import numpy as np
        >>> # Initial state with small biases
        >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])
        >>> v0 = np.zeros(3)
        >>> p0 = np.zeros(3)
        >>> b_g0 = np.array([1e-4, -5e-5, 2e-5])  # rad/s
        >>> b_a0 = np.array([0.01, -0.005, 0.02])  # m/s^2
        >>> state = NavStateQPVPBias(q=q0, v=v0, p=p0, b_g=b_g0, b_a=b_a0)
    """

    q: np.ndarray
    v: np.ndarray
    p: np.ndarray
    b_g: np.ndarray
    b_a: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape consistency of augmented state."""
        if self.q.shape != (4,):
            raise ValueError(
                f"NavStateQPVPBias.q must have shape (4,), got {self.q.shape}"
            )

        if self.v.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.v must have shape (3,), got {self.v.shape}"
            )

        if self.p.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.p must have shape (3,), got {self.p.shape}"
            )

        if self.b_g.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.b_g must have shape (3,), got {self.b_g.shape}"
            )

        if self.b_a.shape != (3,):
            raise ValueError(
                f"NavStateQPVPBias.b_a must have shape (3,), got {self.b_a.shape}"
            )

        # Warn if quaternion is not normalized
        q_norm = np.linalg.norm(self.q)
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            import warnings

            warnings.warn(
                f"NavStateQPVPBias initialized with non-unit quaternion "
                f"(||q|| = {q_norm:.6f}). Consider normalizing.",
                UserWarning,
            )

