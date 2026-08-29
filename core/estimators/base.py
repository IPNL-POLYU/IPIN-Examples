"""
Base classes for state estimators.

This module defines abstract base classes and common interfaces for all
state estimation algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class StateEstimate:
    """Typed snapshot returned by a state estimator.

    Attributes:
        state_vector: Estimated state vector, shape ``(state_dim,)``. Element
            units and ordering are defined by the concrete estimator.
        state_covariance: State covariance matrix, shape
            ``(state_dim, state_dim)``. Row/column ordering matches
            ``state_vector``.
    """

    state_vector: np.ndarray
    state_covariance: np.ndarray


class StateEstimator(ABC):
    """Abstract base class for state estimators."""

    def __init__(self, state_dim: int):
        """
        Initialize state estimator.

        Args:
            state_dim: Dimension of the state vector.
        """
        self.state_dim = state_dim
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    @abstractmethod
    def predict(self, u: np.ndarray | None = None) -> None:
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

    def get_state_estimate(self) -> StateEstimate:
        """Return a typed snapshot of the state and its covariance.

        This is the descriptive alternative to the historical ``get_state()``
        tuple. ``get_state()`` remains available for compatibility.

        Returns:
            Independent copies of the current state vector and state
            covariance in a :class:`StateEstimate`.

        Raises:
            RuntimeError: If the estimator has not been initialized.
        """
        state_vector, state_covariance = self.get_state()
        return StateEstimate(
            state_vector=state_vector,
            state_covariance=state_covariance,
        )


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
