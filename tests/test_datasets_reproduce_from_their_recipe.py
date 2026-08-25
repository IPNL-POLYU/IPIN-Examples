"""Every shipped dataset regenerates from the command that is supposed to make it.

All twenty datasets record ``seed: 42``. Until now exactly three of them --
Chapter 5's -- had anything that checked the seed was real. A seed written down
and never exercised is decorative: it drifts the first time a generator changes,
and nothing says so.

This runs each generator into a temporary directory and compares against the
shipped bytes. The whole sweep is about 30 seconds, because the generators are
fast; the slow part of this repository is the examples, not the data.

Three things this found on its first run, none of which a reading would have:

- **ch6 PDR's ``--preset baseline`` did not produce the shipped baseline.** The
  preset carried cleaner sensors (0.15 / 0.005 / 0.05 / 0.002) than the defaults
  the dataset was actually built from, and wrote to the same directory -- so the
  regeneration command its own README offers replaced the reference dataset with
  different data. Its two sibling ch6 generators do not have this problem: their
  datasets record ``preset=baseline`` and regenerate from it exactly.
- **ch8's presets had no directory of their own**, so ``--preset nlos_severe``
  wrote into whatever ``--output`` defaulted to -- the *baseline* dataset.
- **``nlos_severe`` is not the shipped NLOS dataset.** It is the 1.5 m bias case;
  the shipped one uses 0.8 m, as its README and ``--all-variants`` both say. The
  0.7 m difference is exactly that gap. It now writes somewhere else entirely.

**The comparison has to be NaN-aware, and that is not a detail.** UWB dropouts
are stored as NaN -- 129 of them in each Chapter 8 dataset. ``np.abs(a - b).max()``
returns ``nan`` across a dropout, and ``nan > worst`` is False, so a plain
running-maximum scan silently skips every array that carries one. The first
version of this survey did exactly that and reported all three Chapter 8
datasets as byte-perfect, including the one that differs by 0.7 m.

Author: Li-Ta Hsu
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "sim"

#: How far a regenerated value may sit from the shipped one.
#:
#: Not bit equality. That was tried, in the Chapter 5 version of this test, and
#: it is not portable: with identical numpy and scipy, one CI runner matched
#: exactly and a later one differed by 2.8e-14 on values of order 100. The
#: generator is bit-reproducible on one machine and not across machines.
#:
#: Both sides of the bound were measured rather than assumed. Regenerating ch5
#: with ``seed + 1`` moves an RSS value by 23.06 dB, and the ch8 NLOS bias
#: mismatch this test found is 0.7 m, so the bound sits far below anything that
#: counts as a defect and far above the last-bit noise it has to tolerate.
VALUE_TOL = 1e-9

#: dataset -> extra arguments that regenerate it. ``--output`` is added here.
#:
#: Every entry is verified: each one reproduces its dataset exactly today. An
#: empty list means the generator's own defaults, which is how three of the
#: Chapter 6 datasets were built -- their ``config.json`` records ``preset:
#: null`` and that is not an omission.
RECIPES = {
    "ch2_coords_san_francisco": (
        "generate_ch2_coordinate_transforms_dataset.py",
        ["--preset", "san_francisco"],
    ),
    "ch3_estimator_nonlinear": (
        "generate_ch3_estimator_comparison_dataset.py",
        ["--preset", "nonlinear"],
    ),
    "ch3_estimator_high_nonlinear": (
        "generate_ch3_estimator_comparison_dataset.py",
        ["--preset", "high_nonlinearity"],
    ),
    "ch4_rf_2d_square": (
        "generate_ch4_rf_2d_positioning_dataset.py",
        ["--preset", "baseline"],
    ),
    "ch4_rf_2d_optimal": (
        "generate_ch4_rf_2d_positioning_dataset.py",
        ["--preset", "optimal"],
    ),
    "ch4_rf_2d_linear": (
        "generate_ch4_rf_2d_positioning_dataset.py",
        ["--preset", "poor_geometry"],
    ),
    "ch4_rf_2d_nlos": (
        "generate_ch4_rf_2d_positioning_dataset.py",
        ["--preset", "nlos"],
    ),
    "ch5_wifi_fingerprint_grid": (
        "generate_ch5_wifi_fingerprint_dataset.py",
        ["--preset", "baseline"],
    ),
    "ch5_wifi_fingerprint_dense": (
        "generate_ch5_wifi_fingerprint_dataset.py",
        ["--preset", "dense"],
    ),
    "ch5_wifi_fingerprint_sparse": (
        "generate_ch5_wifi_fingerprint_dataset.py",
        ["--preset", "sparse"],
    ),
    "ch6_env_sensors_heading_altitude": (
        "generate_ch6_env_sensors_dataset.py",
        ["--preset", "baseline"],
    ),
    "ch6_foot_zupt_walk": ("generate_ch6_zupt_dataset.py", []),
    "ch6_pdr_corridor_walk": ("generate_ch6_pdr_dataset.py", ["--preset", "baseline"]),
    "ch6_strapdown_basic": ("generate_ch6_strapdown_dataset.py", []),
    "ch6_wheel_odom_square": (
        "generate_ch6_wheel_odom_dataset.py",
        ["--preset", "baseline"],
    ),
    "ch7_slam_2d_square": ("generate_ch7_slam_2d_dataset.py", ["--preset", "baseline"]),
    "ch7_slam_2d_high_drift": (
        "generate_ch7_slam_2d_dataset.py",
        ["--preset", "high_drift"],
    ),
    "ch8_fusion_2d_imu_uwb": ("generate_ch8_fusion_2d_imu_uwb_dataset.py", []),
    # Explicit flags, not --preset nlos_severe: that preset is the 1.5 m bias
    # case and this dataset is the 0.8 m one. This is the invocation its own
    # README documents, and the one --all-variants uses.
    "ch8_fusion_2d_imu_uwb_nlos": (
        "generate_ch8_fusion_2d_imu_uwb_dataset.py",
        ["--nlos-anchors", "1", "2", "--nlos-bias", "0.8"],
    ),
    "ch8_fusion_2d_imu_uwb_timeoffset": (
        "generate_ch8_fusion_2d_imu_uwb_dataset.py",
        ["--preset", "time_offset_50ms"],
    ),
}


def _shipped_datasets():
    return sorted(d.name for d in DATA.iterdir() if d.is_dir())


def test_every_shipped_dataset_has_a_recipe():
    """A dataset nobody can regenerate is a dataset nobody can check.

    This is the half of the ratchet that keeps the table honest: adding a
    dataset without saying how to rebuild it fails here rather than quietly
    going unexercised, which is the state seventeen of them were in.
    """
    shipped = set(_shipped_datasets())
    listed = set(RECIPES)

    assert shipped == listed, (
        f"RECIPES and data/sim disagree.\n"
        f"  shipped but unlisted: {sorted(shipped - listed)}\n"
        f"  listed but missing:   {sorted(listed - shipped)}\n\n"
        f"Add the command that regenerates the dataset, and check it really "
        f"does before listing it."
    )


@pytest.mark.parametrize("dataset", _shipped_datasets())
def test_the_dataset_records_its_seed(dataset):
    """The scenario alone does not identify a draw; the seed does."""
    directory = DATA / dataset
    seeds = []
    for name in ("metadata.json", "config.json"):
        path = directory / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))

        def find(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "seed" and isinstance(value, int):
                        return value
                    found = find(value)
                    if found is not None:
                        return found
            return None

        seed = find(payload)
        if seed is not None:
            seeds.append(seed)

    assert seeds, (
        f"{dataset} records no seed, so its arrays cannot be tied to a "
        f"particular draw and this file cannot check them."
    )


def _arrays(directory):
    """Every numeric array a dataset ships, keyed by a stable name."""
    out = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".txt":
            try:
                out[path.name] = np.loadtxt(path, ndmin=2)
            except ValueError:
                pass  # a text file that is not a table
        elif path.suffix == ".npy":
            out[path.name] = np.load(path)
        elif path.suffix == ".npz":
            with np.load(path) as bundle:
                for key in bundle.files:
                    out[f"{path.name}:{key}"] = bundle[key]
    return out


@pytest.mark.slow
@pytest.mark.parametrize("dataset", sorted(RECIPES))
def test_regenerating_reproduces_the_shipped_arrays(dataset, tmp_path):
    """Run the recipe and compare against what is committed."""
    generator, extra = RECIPES[dataset]
    shipped = DATA / dataset

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / generator),
            *extra,
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        cwd=REPO_ROOT,
        env={
            **__import__("os").environ,
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
        },
    )
    assert (
        proc.returncode == 0
    ), f"regenerating {dataset} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"

    # --output must win over --preset. It did not always: a preset overwrote it
    # unconditionally, so passing both regenerated the *shipped* dataset in
    # place while appearing to write elsewhere.
    written = list(tmp_path.iterdir())
    assert written, (
        f"{generator} wrote nothing to --output. If a preset is overriding it "
        f"again, this test is silently regenerating data/sim instead."
    )

    want, got = _arrays(shipped), _arrays(tmp_path)
    missing = sorted(set(want) - set(got))
    assert not missing, f"{dataset}: regeneration did not produce {missing}"

    problems = []
    for name, expected in want.items():
        actual = got[name]
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape:
            problems.append(f"{name}: shape {expected.shape} -> {actual.shape}")
            continue
        if not np.issubdtype(expected.dtype, np.number):
            continue

        e, a = expected.astype(float), actual.astype(float)
        # A NaN is a dropped UWB range, not a missing value to skip over. The
        # dropout pattern is part of the draw, so it has to match exactly, and
        # the magnitudes are compared only where both sides are finite --
        # np.abs(...).max() over a NaN returns nan, and `nan > worst` is False,
        # which is how a naive scan reports a mismatched dataset as perfect.
        e_nan, a_nan = np.isnan(e), np.isnan(a)
        if not np.array_equal(e_nan, a_nan):
            problems.append(
                f"{name}: dropout pattern changed "
                f"({int(e_nan.sum())} NaN -> {int(a_nan.sum())} NaN)"
            )
            continue
        if e_nan.all():
            continue
        difference = float(np.abs(e[~e_nan] - a[~a_nan]).max())
        if difference >= VALUE_TOL:
            problems.append(f"{name}: max|difference| = {difference:.6g}")

    assert not problems, (
        f"{dataset} does not reproduce from its recipe "
        f"({generator} {' '.join(extra) or '(defaults)'}):\n  "
        + "\n  ".join(problems)
        + f"\n\nThese are far above the {VALUE_TOL:g} tolerance, so the "
        f"generator's draw order changed, the recipe in RECIPES is not the one "
        f"that built the shipped data, or the shipped data was edited by hand."
    )
