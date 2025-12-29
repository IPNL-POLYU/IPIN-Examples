"""
Simulation utilities for generating synthetic sensor data from ground truth trajectories.

This package provides forward models that convert ideal trajectories into
realistic sensor measurements, consistent with the sensor models in Chapter 6.

Modules:
    imu_from_trajectory: Generate IMU measurements (accel, gyro) from trajectory
    
The forward models implement the correct physics:
    - Accelerometers measure specific force (reaction force), not acceleration
    - Gyroscopes measure angular velocity in body frame
    - Measurements follow the sensor models in Eqs. (6.5), (6.9)

Author: Li-Ta Hsu
Date: December 2025
"""

from core.sim.imu_from_trajectory import (
    compute_specific_force_body,
    compute_gyro_body,
    generate_imu_from_trajectory,
)

__all__ = [
    "compute_specific_force_body",
    "compute_gyro_body",
    "generate_imu_from_trajectory",
]

__version__ = "1.0.0"








