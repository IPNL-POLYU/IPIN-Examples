"""
Proprioceptive and environmental sensor models (Chapter 6).

This package implements the sensor models and dead-reckoning algorithms from
Chapter 6: Dead Reckoning and Proprioceptive Sensors.

Modules:
    types: Sensor data structures and navigation state representations
    imu_models: IMU measurement correction and calibration helpers
    strapdown: Quaternion/velocity/position propagation
    wheel_odometry: Wheel speed DR and lever-arm compensation
    constraints: ZUPT/ZARU/NHC detectors and pseudo-measurements
    pdr: Step detection, step length, PDR propagation
    environment: Magnetometer heading and barometer altitude
    calibration: Allan variance and IMU noise characterization
    ins_ekf_models: ProcessModel/MeasurementModel for IMU+wheel EKF (future)

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

Wheel odometry functions (from wheel_odometry module):
    skew: Skew-symmetric matrix [v×] (Eq. 6.12)
    wheel_speed_to_attitude_velocity: Lever arm compensation (Eq. 6.11)
    attitude_to_map_velocity: Attitude to map frame transform (Eq. 6.14)
    odom_pos_update: Position update from wheel velocity (Eq. 6.15)
    wheel_odom_update: Complete wheel DR update loop

Drift correction constraints (from constraints module):
    detect_zupt: Stationary detector (Eq. 6.44)
    ZuptMeasurementModel: Zero velocity update (Eq. 6.45)
    ZaruMeasurementModelPlaceholder: Zero angular rate update (INCOMPLETE PLACEHOLDER)
    NhcMeasurementModel: Nonholonomic constraint (Eq. 6.61)

Pedestrian Dead Reckoning functions (from pdr module):
    total_accel_magnitude: Acceleration magnitude (Eq. 6.46)
    remove_gravity_from_magnitude: Gravity removal (Eq. 6.47)
    step_frequency: Step frequency from inter-step time (Eq. 6.48)
    step_length: Weinberg step length model (Eq. 6.49)
    pdr_step_update: 2D position update from step (Eq. 6.50)
    detect_step_simple: Simple peak-based step detector
    integrate_gyro_heading: Gyro-based heading integration
    wrap_heading: Wrap heading to [-π, π]

Environmental sensor functions (from environment module):
    mag_tilt_compensate: Tilt compensation for magnetometer (Eq. 6.52)
    mag_heading: Heading from magnetometer (Eqs. 6.51-6.53)
    pressure_to_altitude: Barometric altitude (Eq. 6.54)
    detect_floor_change: Simple floor change detector
    smooth_measurement_simple: Exponential smoothing (Eq. 6.55 concept)
    compensate_hard_iron: Hard-iron bias correction

Calibration functions (from calibration module):
    allan_variance: Allan variance and deviation computation (Eqs. 6.56-6.58)
    identify_bias_instability: Extract bias instability from Allan curve
    identify_random_walk: Extract angle/velocity random walk coefficient
    identify_rate_random_walk: Extract rate random walk coefficient
    characterize_imu_noise: Complete IMU noise characterization

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
    FrameConvention,
    IMUNoiseParams,
    ImuSeries,
    WheelSpeedSeries,
    MagnetometerSeries,
    BarometerSeries,
    NavStateQPVP,
    NavStateQPVPBias,
)

from core.sensors.gravity import (
    gravity_magnitude_eq6_8,
    gravity_magnitude,
    gravity_magnitude_from_lat_deg,
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

from core.sensors.wheel_odometry import (
    skew,
    wheel_speed_to_attitude_velocity,
    attitude_to_map_velocity,
    odom_pos_update,
    wheel_odom_update,
)

from core.sensors.constraints import (
    zupt_test_statistic,
    detect_zupt_windowed,
    detect_zupt,  # Deprecated, use detect_zupt_windowed instead
    ZuptMeasurementModel,
    ZaruMeasurementModelPlaceholder,
    NhcMeasurementModel,
)

from core.sensors.pdr import (
    total_accel_magnitude,
    remove_gravity_from_magnitude,
    detect_steps_peak_detector,
    step_frequency,
    step_length,
    pdr_step_update,
    detect_step_simple,
    integrate_gyro_heading,
    wrap_heading,
)

from core.sensors.environment import (
    mag_tilt_compensate,
    mag_heading,
    wrap_angle_diff,
    pressure_to_altitude,
    detect_floor_change,
    smooth_measurement_simple,
    compensate_hard_iron,
)

from core.sensors.calibration import (
    allan_variance,
    identify_bias_instability,
    identify_random_walk,
    identify_rate_random_walk,
    characterize_imu_noise,
)

# Import units module (provides explicit unit conversions)
from core.sensors import units

__all__ = [
    # Frame convention
    "FrameConvention",
    # IMU noise parameters
    "IMUNoiseParams",
    # Data types
    "ImuSeries",
    "WheelSpeedSeries",
    "MagnetometerSeries",
    "BarometerSeries",
    "NavStateQPVP",
    "NavStateQPVPBias",
    # Gravity models (Eq. 6.8)
    "gravity_magnitude_eq6_8",
    "gravity_magnitude",
    "gravity_magnitude_from_lat_deg",
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
    # Wheel odometry
    "skew",
    "wheel_speed_to_attitude_velocity",
    "attitude_to_map_velocity",
    "odom_pos_update",
    "wheel_odom_update",
    # Drift correction constraints
    "zupt_test_statistic",
    "detect_zupt_windowed",
    "detect_zupt",  # Deprecated
    "ZuptMeasurementModel",
    "ZaruMeasurementModelPlaceholder",
    "NhcMeasurementModel",
    # Pedestrian Dead Reckoning (PDR)
    "total_accel_magnitude",
    "remove_gravity_from_magnitude",
    "detect_steps_peak_detector",
    "step_frequency",
    "step_length",
    "pdr_step_update",
    "detect_step_simple",  # Deprecated, use detect_steps_peak_detector
    "integrate_gyro_heading",
    "wrap_heading",
    # Environmental sensors (magnetometer + barometer)
    "mag_tilt_compensate",
    "mag_heading",
    "wrap_angle_diff",
    "pressure_to_altitude",
    "detect_floor_change",
    "smooth_measurement_simple",
    "compensate_hard_iron",
    # Calibration utilities (Allan variance)
    "allan_variance",
    "identify_bias_instability",
    "identify_random_walk",
    "identify_rate_random_walk",
    "characterize_imu_noise",
    # Unit conversions
    "units",
]

__version__ = "1.0.0"

