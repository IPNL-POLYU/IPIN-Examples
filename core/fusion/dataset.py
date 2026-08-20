"""Loading the Chapter 8 IMU + UWB fusion datasets.

Author: Li-Ta Hsu

This lived in ``ch8_sensor_fusion/tc_uwb_imu_ekf.py`` and was imported from
there by four of that chapter's demos, which made a demo into a library and
Chapter 8 into the only chapter whose examples are not leaves. The loader is
not tightly-coupled-specific -- both architectures read the same dataset -- so
it belongs beside the fusion utilities every chapter can reach.

References: Chapter 8, Section 8.1 (Sensor Fusion Architectures)
"""

import json
from typing import Dict

import numpy as np

from core.utils import resolve_data_path

__all__ = ["load_fusion_dataset"]


def load_fusion_dataset(data_dir: str) -> Dict:
    """Load fusion dataset from directory.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with keys:
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

    dataset = {
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

    return dataset
