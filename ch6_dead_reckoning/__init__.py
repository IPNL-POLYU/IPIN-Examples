"""
Chapter 6: Dead Reckoning and Sensor Fusion for Indoor Navigation

This module implements dead reckoning and sensor fusion algorithms from
Chapter 6 of *Principles of Indoor Positioning and Indoor Navigation*.

Provides examples demonstrating:
    - IMU strapdown integration (quaternion, velocity, position)
    - Wheel odometry (vehicle dead reckoning with lever arm)
    - Drift correction constraints (ZUPT, ZARU, NHC)
    - Pedestrian dead reckoning (step-and-heading)
    - Environmental sensors (magnetometer, barometer)
    - Allan variance IMU calibration

Examples:
    - example_imu_strapdown.py: Pure IMU drift demonstration
    - example_zupt.py: Zero-velocity update drift correction
    - example_pdr.py: Pedestrian dead reckoning
    - example_wheel_odometry.py: Vehicle wheel odometry
    - example_environment.py: Magnetometer and barometer
    - example_allan_variance.py: IMU noise characterization
    - example_comparison.py: Compare all methods

Author: Navigation Engineer
Date: December 2024
"""

__version__ = "1.0.0"
__all__ = []

