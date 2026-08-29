"""Regression tests for the Chapter 6 environment floor demo."""

import numpy as np

from ch6_dead_reckoning.example_environment import (
    DEFAULT_SEED,
    add_env_sensor_noise,
    generate_building_walk,
    run_baro_altitude,
)
from core.sensors import smooth_measurement_simple


def test_environment_demo_detects_floors_from_absolute_altitude() -> None:
    """Default barometer demo should detect floor labels, not only transitions."""
    rng = np.random.default_rng(DEFAULT_SEED)
    t, _, _, mag_true, pressure_true, floor_true = generate_building_walk(rng=rng)
    _, pressure_meas = add_env_sensor_noise(mag_true, pressure_true, t, 0.1, rng=rng)

    altitude = run_baro_altitude(pressure_meas)
    smoothed = np.zeros_like(altitude)
    smoothed[0] = altitude[0]
    for k in range(1, len(altitude)):
        smoothed[k] = smooth_measurement_simple(smoothed[k - 1], altitude[k], alpha=0.1)

    detected = np.rint(smoothed / 3.5).astype(int)
    detected = np.clip(detected, 0, 2)

    assert set(np.unique(detected)) == {0, 1, 2}
    assert np.mean(detected == floor_true) == 0.75
