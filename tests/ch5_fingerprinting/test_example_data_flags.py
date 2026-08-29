"""Ch5 examples expose their input fingerprint database path."""

import os
import subprocess
import sys

import pytest

from tests.example_runner import WORKSPACE_ROOT

CH5_DATA_FLAG_MODULES = [
    "ch5_fingerprinting.example_classification",
    "ch5_fingerprinting.example_comparison",
    "ch5_fingerprinting.example_deterministic",
    "ch5_fingerprinting.example_pattern_recognition",
    "ch5_fingerprinting.example_probabilistic",
    "ch5_fingerprinting.example_walk_posterior",
]


@pytest.mark.parametrize("module", CH5_DATA_FLAG_MODULES)
def test_ch5_examples_advertise_data_flag(module):
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(WORKSPACE_ROOT),
            "PYTHONIOENCODING": "utf-8",
        }
    )

    run = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env=env,
    )

    assert run.returncode == 0, run.stderr[-1000:]
    assert "--data" in run.stdout
