"""The shipped Chapter 4 TDOA measurements are negated, and every reader
of them has been told the result is hyperbolic geometry.

`scripts/generate_ch4_rf_2d_positioning_dataset.py:213` builds each range
difference as

    tdoa_range_difference(beacons[0], beacons[j], pos)      # d_0 - d_j

while `TDOAPositioner(beacons, reference_idx=0)` predicts, and every equation
from (4.34) on is written as,

    dist_i - dist_ref                                       # d_j - d_0

so `tdoa_diffs.txt` carries the negative of what the solver is handed it as.
Solving a negated measurement is not a small perturbation -- it asks for the
branch of the hyperbola on the other side of the array -- and the cost is a
factor of 150:

    ch4_rf_2d_square    as shipped 13.753 m, 11/100 failed
                        negated     0.074 m,  0/100 failed
    ch4_rf_2d_optimal   as shipped 14.009 m, 13/100 failed
                        negated     0.086 m,  0/100 failed

**The negated figure is the one the geometry predicts.** The square array's
TDOA GDOP is 0.87 and the noise is 0.1 m, so `sigma_position = GDOP x
sigma_range` says 0.087 m. That is the CLAUDE.md question -- *what should this
number be?* -- and 13.75 m answers it by a factor of 158.

The collinear variant is unaffected either way (100/100 fail from the beacon
centroid, which sits on the line of symmetry) and so cannot see this at all.

**Three documents had already met the bug and explained it away.**
`data/sim/ch4_rf_2d_square/README.md` calls TDOA "the fragile one" and says
"the hyperbolic geometry costs two orders of magnitude"; its troubleshooting
section carries an entry titled "TDOA Positioning Fails or Returns Large
Errors" whose stated symptom -- ">10m errors while TOA gives <0.5m" -- is a
description of this defect. Same shape as the ch2 dataset README's "Issue 2:
ENU Range Seems Wrong": a troubleshooting entry describing your own output is
a bug report, not documentation.

This file is written to **fail the moment the generator is corrected**, which
is the pattern `tests/ch7_slam/test_frontend_actually_corrects.py` established
for a defect found but deliberately not fixed in the same session. Fixing it
touches shipped bytes in four datasets and the numbers quoted in two dataset
READMEs, so it was left as its own change rather than folded into the geometry
comparison that found it.

When you fix it:

  1. Swap the arguments at `generate_ch4_rf_2d_positioning_dataset.py:213` to
     `tdoa_range_difference(beacons[j], beacons[0], pos)`.
  2. Regenerate all four ch4 datasets (`--preset` per each `config.json`).
  3. Update the TDOA rows in `data/sim/ch4_rf_2d_square/README.md` and
     `data/sim/ch4_rf_2d_linear/README.md`, and the "hyperbolic geometry"
     explanations that were written around the wrong number.
  4. Delete this file and rerun
     `python -m ch4_rf_point_positioning.example_comparison --compare-geometry`,
     whose printed footnote and figure caption both name this defect.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.27)-(4.33) and (4.34)-(4.42)
"""

import json

import numpy as np
import pytest

from core.rf import TDOAPositioner, solve_batch
from tests.example_runner import WORKSPACE_ROOT

#: Datasets whose measurements are dense enough to show the sign, i.e. every
#: ch4 dataset except the collinear one, where both signs stall at the seed.
NEGATED = ("ch4_rf_2d_square", "ch4_rf_2d_optimal", "ch4_rf_2d_nlos")


def _load(name):
    directory = WORKSPACE_ROOT / "data" / "sim" / name
    return {
        "beacons": np.loadtxt(directory / "beacons.txt"),
        "positions": np.loadtxt(directory / "ground_truth_positions.txt"),
        "tdoa": np.loadtxt(directory / "tdoa_diffs.txt"),
        "gdop": np.loadtxt(directory / "gdop_tdoa.txt"),
        "config": json.loads((directory / "config.json").read_text(encoding="utf-8")),
    }


def _predicted(beacons, position):
    """d_j - d_ref for j = 1..K-1, the convention every equation uses."""
    d_ref = np.linalg.norm(position - beacons[0])
    return np.array(
        [np.linalg.norm(position - beacons[j]) - d_ref for j in range(1, len(beacons))]
    )


@pytest.mark.parametrize("dataset", NEGATED)
def test_the_shipped_measurements_carry_the_reference_minus_anchor_sign(dataset):
    """Red when the generator is fixed. Read this file's docstring first.

    Asserting on the *measurements* rather than on the solved error is what
    makes this specific: a solver change, a seed change or a tolerance change
    cannot turn it green or red, because it compares stored bytes against the
    geometry that produced them.
    """
    data = _load(dataset)
    beacons, positions, stored = data["beacons"], data["positions"], data["tdoa"]

    predicted = np.array([_predicted(beacons, p) for p in positions])
    sigma = data["config"]["measurements"]["tdoa_noise_std_m"]

    as_written = np.abs(stored - predicted).mean()
    negated = np.abs(stored + predicted).mean()

    assert negated < 4 * sigma, (
        f"{dataset}/tdoa_diffs.txt no longer matches -(d_j - d_ref): mean "
        f"|stored + predicted| is {negated:.3f} m against a {sigma} m noise "
        f"std. If you have just corrected the generator, this file has done "
        f"its job -- delete it and finish the steps in its docstring."
    )
    assert as_written > negated, (
        f"{dataset}/tdoa_diffs.txt now matches the convention TDOAPositioner "
        f"expects (|stored - predicted| = {as_written:.3f} m, "
        f"|stored + predicted| = {negated:.3f} m). The defect this file pins "
        f"is fixed; delete it and finish the steps in its docstring."
    )


@pytest.mark.parametrize("dataset", NEGATED)
def test_negating_them_recovers_the_accuracy_the_gdop_predicts(dataset):
    """The size of the effect, not just its existence.

    Without this the test above pins a sign nobody can weigh. `sigma_position =
    GDOP x sigma_range` is the prediction Chapter 4 makes and Section 4.5's own
    example validates to 3.5%; the negated solve meets it and the shipped one
    misses it by two orders of magnitude.
    """
    data = _load(dataset)
    beacons, positions, stored = data["beacons"], data["positions"], data["tdoa"]
    guess = beacons.mean(axis=0)
    sigma = data["config"]["measurements"]["tdoa_noise_std_m"]
    solver = TDOAPositioner(beacons, reference_idx=0)

    shipped = solve_batch(solver, stored, guess, positions)
    corrected = solve_batch(solver, -stored, guess, positions)

    bound = data["gdop"].mean() * sigma

    assert corrected.median_m < 2 * bound, (
        f"{dataset}: negating the measurements gives a median of "
        f"{corrected.median_m:.3f} m against a GDOP bound of {bound:.3f} m. "
        f"The sign is not the whole story here -- investigate before deleting "
        f"the check next door."
    )
    assert corrected.n_failed == 0, (
        f"{dataset}: {corrected.n_failed} fixes still fail with the sign "
        f"corrected, so something beyond the convention is wrong."
    )
    assert shipped.median_m > 20 * bound, (
        f"{dataset}: the shipped measurements now give {shipped.median_m:.3f} m "
        f"against a GDOP bound of {bound:.3f} m, which is no longer the "
        f"150x this file was written to record."
    )
