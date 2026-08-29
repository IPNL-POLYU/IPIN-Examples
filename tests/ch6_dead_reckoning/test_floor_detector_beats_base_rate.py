"""Chapter 6's barometric floor detector used to never fire.

`generate_ch6_env_sensors_dataset.py` used to detect floor changes from the
delta between adjacent *smoothed* altitude samples. The alpha=0.1 exponential
smoother spreads a floor_height (3.5 m) step over dozens of samples, so no
single-sample delta ever approached the 1.5 m detection threshold:
floor_detected stayed 0 for the entire walk, and floor_detection_accuracy
silently scored the base rate of floor 0 -- 50.0%, exactly matching floor 0's
50.0% share of ground_truth_floor. A constant predictor scoring its own base
rate is the bug signature CLAUDE.md calls out under "A number at chance ...
is a bug signature", and an accuracy threshold set anywhere near 50% would
have passed it.

The fix rounds absolute smoothed altitude to the nearest floor instead of
thresholding a delta (see the "Detect floors" comment in
generate_ch6_env_sensors_dataset.py). This guards against the same shape
recurring: assert the detector actually beats a constant predictor, not that
it clears some accuracy threshold.

Regenerates the shipped baseline recipe into a temp directory rather than
reading the committed config.json, so this exercises the generator's real
floor-detection code and fails if that code regresses -- a stored number
would not move if someone reverted the fix.

Author: Li-Ta Hsu
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR = REPO_ROOT / "scripts" / "generate_ch6_env_sensors_dataset.py"

#: How far above the base rate to demand. The shipped baseline dataset scores
#: 63.2% against a 50.0% base rate (13.2 points); the old, broken detector
#: scored exactly the base rate (0.0 points). This sits well below the
#: former and well above the latter.
MARGIN_PERCENTAGE_POINTS = 10.0


def test_floor_detector_beats_a_constant_predictor(tmp_path):
    """A detector stuck on one floor scores exactly that floor's share of samples."""
    proc = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--preset",
            "baseline",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, (
        f"generator failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    accuracy = config["performance"]["barometric_altitude"][
        "floor_detection_accuracy_percent"
    ]

    floor_true = np.loadtxt(tmp_path / "ground_truth_floor.txt")
    _, counts = np.unique(floor_true, return_counts=True)
    base_rate = 100.0 * counts.max() / counts.sum()

    # A detector that never changes its answer reports whichever floor is
    # most common and scores exactly base_rate -- what this generator did
    # before the fix (50.0% == 50.0%, to the printed digit). Require a real
    # margin above it, so a detector that varies but is no better than
    # guessing the mode still fails this.
    assert accuracy > base_rate + MARGIN_PERCENTAGE_POINTS, (
        f"floor_detection_accuracy_percent ({accuracy:.1f}%) is barely above "
        f"the base rate of always guessing the most common floor "
        f"({base_rate:.1f}%) -- this is the signature of a detector that "
        f"never actually changes its answer."
    )
