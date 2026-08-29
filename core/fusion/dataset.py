"""Loading the Chapter 8 IMU + UWB fusion datasets.

Author: Li-Ta Hsu

This lived in ``ch8_sensor_fusion/example_tc_fusion.py`` and was imported from
there by four of that chapter's demos, which made a demo into a library and
Chapter 8 into the only chapter whose examples are not leaves. The loader is
not tightly-coupled-specific -- both architectures read the same dataset -- so
it belongs beside the fusion utilities every chapter can reach.

References: Chapter 8, Section 8.1 (Sensor Fusion Architectures)
"""

import json
from typing import Any

import numpy as np

from core.utils import resolve_data_path

__all__ = ["FusionDataset", "load_fusion_dataset"]


class FusionDataset(dict[str, Any]):
    """Dictionary-compatible Chapter 8 dataset with semantic accessors.

    Historical examples use the serialized keys directly. The properties below
    make role, units, and shape discoverable without changing those keys.
    """

    @property
    def truth_timestamps_s(self) -> np.ndarray:
        """Ground-truth timestamps, shape ``(N,)``, in seconds."""
        return self["truth"]["t"]

    @property
    def true_positions_xy_m(self) -> np.ndarray:
        """Ground-truth map-frame positions, shape ``(N, 2)``, in metres."""
        return self["truth"]["p_xy"]

    @property
    def true_velocities_xy_mps(self) -> np.ndarray:
        """Ground-truth map-frame velocities, shape ``(N, 2)``, in m/s."""
        return self["truth"]["v_xy"]

    @property
    def true_yaw_rad(self) -> np.ndarray:
        """Ground-truth body-to-map yaw, shape ``(N,)``, in radians."""
        return self["truth"]["yaw"]

    @property
    def imu_timestamps_s(self) -> np.ndarray:
        """IMU sample timestamps, shape ``(K,)``, in seconds."""
        return self["imu"]["t"]

    @property
    def measured_accelerations_body_xy_mps2(self) -> np.ndarray:
        """Measured body-frame planar accelerations, shape ``(K, 2)``."""
        return self["imu"]["accel_xy"]

    @property
    def measured_gyro_z_rad_s(self) -> np.ndarray:
        """Measured body z-axis angular rates, shape ``(K,)``, in rad/s."""
        return self["imu"]["gyro_z"]

    @property
    def uwb_anchor_positions_xy_m(self) -> np.ndarray:
        """Map-frame UWB anchor positions, shape ``(A, 2)``, in metres."""
        return self["uwb_anchors"]

    @property
    def uwb_timestamps_s(self) -> np.ndarray:
        """UWB epoch timestamps, shape ``(M,)``, in seconds."""
        return self["uwb"]["t"]

    @property
    def measured_uwb_ranges_m(self) -> np.ndarray:
        """Measured UWB ranges, shape ``(M, A)``, in metres."""
        return self["uwb"]["ranges"]


def load_fusion_dataset(data_dir: str) -> FusionDataset:
    """Load fusion dataset from directory.

    Args:
        data_dir: Path to dataset directory

    Returns:
        FusionDataset (a dictionary-compatible typed result) with keys:
            - 'truth': dict with t, p_xy, v_xy, yaw
            - 'imu': dict with t, accel_xy, gyro_z
            - 'uwb_anchors': anchor positions (A, 2)
            - 'uwb': dict with t, ranges (M, A)
            - 'config': configuration dict
    """
    data_path = resolve_data_path(data_dir)

    # Load data files
    truth_data = np.load(data_path / "truth.npz")
    imu_data = np.load(data_path / "imu.npz")
    uwb_data = np.load(data_path / "uwb_ranges.npz")
    uwb_anchors = np.load(data_path / "uwb_anchors.npy")

    with open(data_path / "config.json", "r") as f:
        config = json.load(f)

    dataset = FusionDataset(
        {
            "truth": {
                "t": truth_data["t"],
                "p_xy": truth_data["p_xy"],
                "v_xy": truth_data["v_xy"],
                "yaw": truth_data["yaw"],
            },
            "imu": {
                "t": imu_data["t"],
                "accel_xy": imu_data["accel_xy"],
                "gyro_z": imu_data["gyro_z"],
            },
            "uwb_anchors": uwb_anchors,
            "uwb": {"t": uwb_data["t"], "ranges": uwb_data["ranges"]},
            "config": config,
        }
    )

    return dataset
