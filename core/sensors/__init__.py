"""
Proprioceptive and environmental sensor models (Chapter 6).

This package implements the sensor models and dead-reckoning algorithms from
Chapter 6: Dead Reckoning and Proprioceptive Sensors.

Modules:
    types: Sensor data structures and navigation state representations
    imu_models: IMU measurement correction and calibration helpers
    strapdown: Quaternion/velocity/position propagation
    wheel_odometry: Wheel speed DR and lever-arm compensation (future)
    ins_ekf_models: ProcessModel/MeasurementModel for IMU+wheel EKF (future)
    constraints: ZUPT/ZARU/NHC detectors and pseudo-measurements (future)
    pdr: Step detection, step length, PDR propagation (future)
    environment: Magnetometer heading and barometer altitude (future)
    calibration: Allan variance and IMU scale/misalignment model (future)

Primary data structures (from types module):
    ImuSeries: Time-series IMU data (accel, gyro)
    WheelSpeedSeries: Wheel odometry velocity data
    MagnetometerSeries: Magnetic field measurements
    BarometerSeries: Atmospheric pressure data
    NavStateQPVP: Minimal navigation state (quaternion, velocity, position)
    NavStateQPVPBias: Augmented state with IMU biases

IMU correction functions (from imu_models module):
    correct_gyro: Remove bias and noise from gyro (Eq. 6.6)
    correct_accel: Remove bias and noise from accel (Eq. 6.9)
    apply_imu_scale_misalignment: Calibration correction (Eq. 6.59)

Strapdown integration functions (from strapdown module):
    omega_matrix: Ω(ω) matrix for quaternion kinematics (Eq. 6.3)
    quat_integrate: Discrete quaternion update (Eqs. 6.2-6.4)
    quat_to_rotmat: Convert quaternion to rotation matrix
    gravity_vector: Gravity in map frame (Eq. 6.8)
    vel_update: Velocity propagation (Eq. 6.7)
    pos_update: Position propagation (Eq. 6.10)
    strapdown_update: Complete attitude/velocity/position update

Design principles:
    - All sensor models reference Chapter 6 equations in docstrings
    - Dataclasses are frozen (immutable) for sensor packets
    - Navigation states are mutable for in-place updates
    - All algorithms use NumPy for efficiency
    - Frame conventions: B (body), M (map/ENU), S (speed), A (attitude)

Example:
    >>> from core.sensors import (
    ...     ImuSeries, NavStateQPVP,
    ...     correct_gyro, correct_accel,
    ...     strapdown_update
    ... )
    >>> import numpy as np
    >>> 
    >>> # Create IMU data packet
    >>> t = np.linspace(0, 1, 100)
    >>> accel = np.random.randn(100, 3) * 0.1 + [0, 0, -9.81]
    >>> gyro = np.random.randn(100, 3) * 0.01
    >>> imu = ImuSeries(t=t, accel=accel, gyro=gyro, meta={'sample_rate_hz': 100})
    >>> 
    >>> # Create navigation state
    >>> q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    >>> state = NavStateQPVP(q=q0, v=np.zeros(3), p=np.zeros(3))
    >>> 
    >>> # Correct IMU measurements
    >>> bias_g = np.array([0.001, -0.0005, 0.0002])
    >>> bias_a = np.array([0.01, -0.005, 0.02])
    >>> omega = correct_gyro(gyro[0, :], bias_g)
    >>> f_b = correct_accel(accel[0, :], bias_a)
    >>> 
    >>> # Strapdown integration step
    >>> dt = t[1] - t[0]
    >>> q1, v1, p1 = strapdown_update(state.q, state.v, state.p, omega, f_b, dt)
"""

from core.sensors.types import (
    ImuSeries,
    WheelSpeedSeries,
    MagnetometerSeries,
    BarometerSeries,
    NavStateQPVP,
    NavStateQPVPBias,
)

from core.sensors.imu_models import (
    correct_gyro,
    correct_accel,
    apply_imu_scale_misalignment,
    remove_gravity_component,
)

from core.sensors.strapdown import (
    omega_matrix,
    quat_integrate,
    quat_to_rotmat,
    gravity_vector,
    vel_update,
    pos_update,
    strapdown_update,
)

__all__ = [
    # Data types
    "ImuSeries",
    "WheelSpeedSeries",
    "MagnetometerSeries",
    "BarometerSeries",
    "NavStateQPVP",
    "NavStateQPVPBias",
    # IMU correction
    "correct_gyro",
    "correct_accel",
    "apply_imu_scale_misalignment",
    "remove_gravity_component",
    # Strapdown integration
    "omega_matrix",
    "quat_integrate",
    "quat_to_rotmat",
    "gravity_vector",
    "vel_update",
    "pos_update",
    "strapdown_update",
]

__version__ = "0.2.0"

