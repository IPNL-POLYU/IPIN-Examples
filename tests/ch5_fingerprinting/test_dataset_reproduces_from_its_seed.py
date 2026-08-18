"""The shipped ch5 databases regenerate from the seed they record.

Every dataset in `data/sim` states how it was made, so a reader can change one
parameter and see what it does. The three ch5 fingerprint databases were the
exception: their `metadata.json` carried the scenario -- AP positions, grid
spacing, path-loss model -- but no seed, so nothing said which of the infinitely
many draws from that scenario the shipped bytes were.

They were reproducible all along. Regenerating with the generator's default seed
of 42 gives `features.npy` and `locations.npy` **byte-identical** to what ships;
only `floor_ids.npy` differed, and only in dtype, because the shipped copy was
written on a platform where the default integer is 32-bit. So this was a
documentation gap rather than a reproducibility one -- but an unrecorded fact
that happens to be true is one nobody can check, which is what this fixes.

The seed is now in `metadata.json`, and this test is what stops it becoming
decorative: it regenerates from the recorded seed and compares against the
shipped arrays. A seed written down and never exercised would drift the first
time the generator's draw order changed, and say nothing when it did.

Marked slow: about 9 s for the three, dominated by the 2,028-reference-point
dense grid.

Author: Li-Ta Hsu
References: Chapter 5, Eqs. (5.1)-(5.6)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: How far a regenerated value may sit from the shipped one.
#:
#: This was np.array_equal -- bit exact -- and that turned out not to be
#: portable. The generator is bit-reproducible on one machine: two local runs
#: and the shipped files agree exactly, max|difference| 0.0. Across CI runners
#: it is not. With identical numpy 2.4.6 and scipy 1.17.1, one run matched
#: exactly and a later one differed by 2.8e-14 on values of order 100 -- one to
#: two ulp, and nothing in that PR could reach this generator, which imports
#: only core.fingerprinting. The likely mechanism is numpy dispatching a
#: different SIMD kernel for np.log10 on a different CPU; that is not proven
#: here, but the last bit clearly does not survive a change of runner.
#:
#: 1e-9 keeps the test able to fail for the reason it exists, and that was
#: measured rather than assumed: regenerating with seed + 1 gives
#: max|difference| = 23.06 dB. So the bound sits ten orders below the defect it
#: must catch and four orders above the noise it tolerates. **A tolerance has to
#: be justified against both** -- against the noise, or it is flaky, and against
#: the defect, or it is decorative.
VALUE_TOL = 1e-9

GENERATOR = "scripts/generate_ch5_wifi_fingerprint_dataset.py"

#: (shipped directory, generator preset)
DATASETS = [
    ("ch5_wifi_fingerprint_grid", "baseline"),
    ("ch5_wifi_fingerprint_dense", "dense"),
    ("ch5_wifi_fingerprint_sparse", "sparse"),
]


@pytest.mark.parametrize("dataset,preset", DATASETS, ids=[d for d, _ in DATASETS])
def test_metadata_records_the_seed(dataset, preset):
    """The scenario alone does not identify a draw; the seed does."""
    meta = json.loads(
        (REPO_ROOT / "data" / "sim" / dataset / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert "seed" in meta, (
        f"{dataset}/metadata.json records no seed, so its arrays cannot be "
        f"regenerated exactly. Every other dataset in data/sim states this."
    )
    assert isinstance(meta["seed"], int)


@pytest.mark.slow
@pytest.mark.parametrize("dataset,preset", DATASETS, ids=[d for d, _ in DATASETS])
def test_regenerating_from_the_seed_reproduces_the_shipped_arrays(
    dataset, preset, tmp_path
):
    """Regenerate into a temporary directory and compare with what ships."""
    shipped = REPO_ROOT / "data" / "sim" / dataset
    seed = json.loads((shipped / "metadata.json").read_text(encoding="utf-8"))["seed"]

    proc = subprocess.run(
        [
            sys.executable, GENERATOR,
            "--preset", preset,
            "--output", str(tmp_path),
            "--seed", str(seed),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"regenerating {dataset} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )
    # --output must win over --preset. It did not always: a preset overwrote it
    # unconditionally, so passing both regenerated the *shipped* dataset in
    # place while appearing to write elsewhere.
    assert (tmp_path / "features.npy").exists(), (
        f"{GENERATOR} wrote nothing to --output. If --preset is overriding it "
        f"again, this test is silently regenerating data/sim instead."
    )

    for name in ("features.npy", "locations.npy"):
        expected = np.load(shipped / name)
        actual = np.load(tmp_path / name)
        assert actual.shape == expected.shape, f"{dataset}/{name}: shape changed"
        difference = np.abs(actual - expected).max()
        assert difference < VALUE_TOL, (
            f"{dataset}/{name} does not reproduce from seed {seed}: "
            f"max|difference| = {difference:.6g}, over the {VALUE_TOL:g} "
            f"tolerance. That is far above last-bit noise, so the generator's "
            f"draw order changed or the shipped file was not made with this "
            f"seed."
        )

    # floor_ids is compared on value, not dtype: the shipped copies were written
    # where the default integer is 32-bit and regenerate as 64-bit.
    expected = np.load(shipped / "floor_ids.npy")
    actual = np.load(tmp_path / "floor_ids.npy")
    assert np.array_equal(actual.astype(np.int64), expected.astype(np.int64)), (
        f"{dataset}/floor_ids.npy does not reproduce from seed {seed}."
    )
