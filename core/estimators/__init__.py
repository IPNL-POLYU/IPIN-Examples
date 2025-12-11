"""
State estimation algorithms for indoor positioning.

This module provides implementations of various state estimation techniques
described in Chapter 3 of Principles of Indoor Positioning and Indoor Navigation.

Available estimators:
    - Least Squares (LS, WLS, Robust LS)
    - Kalman Filter (KF)
    - Extended Kalman Filter (EKF)
    - Unscented Kalman Filter (UKF)
    - Particle Filter (PF)
"""

from core.estimators.least_squares import (
    linear_least_squares,
    weighted_least_squares,
    iterative_least_squares,
    robust_least_squares,
)

__all__ = [
    "linear_least_squares",
    "weighted_least_squares",
    "iterative_least_squares",
    "robust_least_squares",
]

