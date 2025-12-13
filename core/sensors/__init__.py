"""
Proprioceptive and environmental sensor models (Chapter 6).

This package implements the sensor models and dead-reckoning algorithms from
Chapter 6: Dead Reckoning and Proprioceptive Sensors.

Modules:
    types: Sensor data structures and navigation state representations
    imu_models: IMU measurement correction and calibration helpers (future)
    strapdown: Quaternion/velocity/position propagation (future)
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

Design principles:
    - All sensor models reference Chapter 6 equations in docstrings
    - Dataclasses are frozen (immutable) for sensor packets
    - Navigation states are mutable for in-place updates
    - All algorithms use NumPy for efficiency
    - Frame conventions: B (body), M (map/ENU), S (speed), A (attitude)

Example:
    >>> from core.sensors import ImuSeries, NavStateQPVP
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
"""

from core.sensors.types import (
    ImuSeries,
    WheelSpeedSeries,
    MagnetometerSeries,
    BarometerSeries,
    NavStateQPVP,
    NavStateQPVPBias,
)

__all__ = [
    "ImuSeries",
    "WheelSpeedSeries",
    "MagnetometerSeries",
    "BarometerSeries",
    "NavStateQPVP",
    "NavStateQPVPBias",
]

__version__ = "0.1.0"

