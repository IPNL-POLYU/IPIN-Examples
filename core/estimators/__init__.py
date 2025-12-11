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
    - Factor Graph Optimization (FGO)
"""

from core.estimators.least_squares import (
    linear_least_squares,
    weighted_least_squares,
    iterative_least_squares,
    robust_least_squares,
)
from core.estimators.kalman_filter import KalmanFilter
from core.estimators.extended_kalman_filter import ExtendedKalmanFilter
from core.estimators.unscented_kalman_filter import UnscentedKalmanFilter
from core.estimators.particle_filter import ParticleFilter
from core.estimators.factor_graph import Factor, FactorGraph

__all__ = [
    "linear_least_squares",
    "weighted_least_squares",
    "iterative_least_squares",
    "robust_least_squares",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "Factor",
    "FactorGraph",
]

