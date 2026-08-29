"""Regression tests for reader-facing Chapter 8 fusion result names."""

from pathlib import Path

import numpy as np
import pytest

from core.fusion import (
    SENSOR_UWB_RANGE,
    SENSOR_UWB_RANGES_BATCH,
    SENSOR_UWB_RANGES_EPOCH,
    FusionDataset,
    FusionHistory,
    StampedMeasurement,
    load_fusion_dataset,
    run_lc_fusion,
    run_tc_fusion,
)


@pytest.fixture(scope="module")
def fusion_dataset():
    dataset_path = Path("data/sim/ch8_fusion_2d_imu_uwb")
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    return load_fusion_dataset(str(dataset_path))


def test_dataset_keeps_dict_keys_and_adds_semantic_array_accessors(fusion_dataset):
    """Serialized keys stay compatible while roles, units, and shapes are named."""
    assert isinstance(fusion_dataset, FusionDataset)
    assert isinstance(fusion_dataset, dict)
    np.testing.assert_array_equal(
        fusion_dataset.true_positions_xy_m,
        fusion_dataset["truth"]["p_xy"],
    )
    np.testing.assert_array_equal(
        fusion_dataset.measured_uwb_ranges_m,
        fusion_dataset["uwb"]["ranges"],
    )
    assert fusion_dataset.true_positions_xy_m.shape[1] == 2
    assert fusion_dataset.measured_uwb_ranges_m.shape[1] == len(
        fusion_dataset.uwb_anchor_positions_xy_m
    )


def test_lc_history_prefers_measurement_accepted_with_gated_alias(fusion_dataset):
    """LC keeps old ``gated`` callers working while exposing the clearer name."""
    history = run_lc_fusion(fusion_dataset, use_gating=False, verbose=False)

    assert "measurement_accepted" in history
    assert isinstance(history, FusionHistory)
    assert isinstance(history, dict)
    assert history["gated"] is history["measurement_accepted"]
    assert history.gated is history.measurement_accepted
    assert history.timestamps_s is history.t
    assert history.estimated_state_vectors is history.x_est
    assert history.state_covariance_trace is history.p_trace
    assert history.innovation_vectors is history.innovations
    assert history.normalized_innovation_squared is history.nis
    assert len(history["measurement_accepted"]) == history["n_uwb_accepted"]
    assert all(history["measurement_accepted"])


def test_tc_history_prefers_measurement_accepted_with_gated_alias(fusion_dataset):
    """TC keeps old ``gated`` callers working while exposing the clearer name."""
    history = run_tc_fusion(fusion_dataset, use_gating=False, verbose=False)

    assert "measurement_accepted" in history
    assert isinstance(history, FusionHistory)
    assert isinstance(history, dict)
    assert history["gated"] is history["measurement_accepted"]
    assert history.gated is history.measurement_accepted
    assert len(history["measurement_accepted"]) == history["n_uwb_accepted"]
    assert all(history["measurement_accepted"])


def test_uwb_measurement_kind_names_encode_shape():
    """UWB epoch/batch/single-anchor packets no longer share one ambiguous kind."""
    assert SENSOR_UWB_RANGE == "uwb_range"
    assert SENSOR_UWB_RANGES_EPOCH == "uwb_ranges_epoch"
    assert SENSOR_UWB_RANGES_BATCH == "uwb_ranges_batch"

    single = StampedMeasurement(
        t=1.0,
        sensor=SENSOR_UWB_RANGE,
        z=np.array([5.0]),
        R=np.array([[0.01]]),
    )
    assert single.sensor == SENSOR_UWB_RANGE
