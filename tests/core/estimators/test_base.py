"""Tests for descriptive state-estimator result APIs."""

import numpy as np
import pytest

from core.estimators import StateEstimate
from core.estimators.base import StateEstimator


class _ExampleEstimator(StateEstimator):
    """Minimal concrete estimator used to exercise the base contract."""

    def predict(self, u=None) -> None:
        del u

    def update(self, z: np.ndarray) -> None:
        del z


def test_get_state_estimate_is_typed_and_keeps_legacy_get_state():
    """The descriptive API and historical tuple must contain equal copies."""
    estimator = _ExampleEstimator(state_dim=2)
    estimator.state = np.array([1.0, 2.0])
    estimator.covariance = np.diag([0.25, 0.5])

    typed_result = estimator.get_state_estimate()
    legacy_state, legacy_covariance = estimator.get_state()

    assert isinstance(typed_result, StateEstimate)
    np.testing.assert_array_equal(typed_result.state_vector, legacy_state)
    np.testing.assert_array_equal(typed_result.state_covariance, legacy_covariance)

    typed_result.state_vector[0] = 99.0
    typed_result.state_covariance[0, 0] = 99.0
    assert estimator.state[0] == 1.0
    assert estimator.covariance[0, 0] == 0.25


def test_get_state_estimate_preserves_uninitialized_error():
    """The new API must fail exactly where the legacy snapshot is unavailable."""
    estimator = _ExampleEstimator(state_dim=2)

    with pytest.raises(RuntimeError, match="not initialized"):
        estimator.get_state_estimate()
