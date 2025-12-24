"""
State estimation algorithms for indoor positioning.

This module provides implementations of various state estimation techniques
described in Chapter 3 of Principles of Indoor Positioning and Indoor Navigation.

Available estimators:
    - Least Squares (LS, WLS, Robust LS)
    - Nonlinear Least Squares (Gauss-Newton, Levenberg-Marquardt)
    - Kalman Filter (KF)
    - Extended Kalman Filter (EKF)
    - Iterated Extended Kalman Filter (IEKF) - Section 3.2.3
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
from core.estimators.nonlinear_least_squares import (
    gauss_newton,
    levenberg_marquardt,
    robust_gauss_newton,
    solve_nonlinear_ls,
    NonlinearLSResult,
)
from core.estimators.kalman_filter import KalmanFilter
from core.estimators.extended_kalman_filter import ExtendedKalmanFilter
from core.estimators.iterated_extended_kalman_filter import IteratedExtendedKalmanFilter
from core.estimators.unscented_kalman_filter import UnscentedKalmanFilter
from core.estimators.particle_filter import ParticleFilter
from core.estimators.factor_graph import Factor, FactorGraph

__all__ = [
    # Linear LS
    "linear_least_squares",
    "weighted_least_squares",
    "iterative_least_squares",
    "robust_least_squares",
    # Nonlinear LS (Section 3.4.1)
    "gauss_newton",
    "levenberg_marquardt",
    "robust_gauss_newton",
    "solve_nonlinear_ls",
    "NonlinearLSResult",
    # Kalman Filters (Section 3.2)
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "IteratedExtendedKalmanFilter",  # Section 3.2.3
    "UnscentedKalmanFilter",
    # Particle Filter (Section 3.3)
    "ParticleFilter",
    # Factor Graph (Section 3.4)
    "Factor",
    "FactorGraph",
]

