"""
Base classes for state estimators.

This module defines abstract base classes and common interfaces for all
state estimation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class StateEstimator(ABC):
    """Abstract base class for state estimators."""

    def __init__(self, state_dim: int):
        """
        Initialize state estimator.

        Args:
            state_dim: Dimension of the state vector.
        """
        self.state_dim = state_dim
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

    @abstractmethod
    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """
        Perform prediction step (time update).

        Args:
            u: Optional control input vector.
        """
        pass

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """
        Perform measurement update (correction step).

        Args:
            z: Measurement vector.
        """
        pass

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and covariance.

        Returns:
            Tuple of (state_vector, covariance_matrix).
        """
        if self.state is None or self.covariance is None:
            raise RuntimeError("Estimator not initialized. Call predict() first.")
        return self.state.copy(), self.covariance.copy()


class BatchEstimator(ABC):
    """Abstract base class for batch estimation algorithms."""

    @abstractmethod
    def estimate(
        self, measurements: np.ndarray, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute batch estimate from measurements.

        Args:
            measurements: Matrix of measurements (m × n).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (state_estimate, covariance_matrix).
        """
        pass


