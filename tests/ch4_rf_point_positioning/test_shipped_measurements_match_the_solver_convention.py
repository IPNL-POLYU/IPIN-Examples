"""Every shipped Chapter 4 measurement must carry the sign its solver predicts.

`tdoa_diffs.txt` was negated in all four datasets for as long as they existed.
The generator built each range difference as `d_ref - d_j` while
`TDOAPositioner(reference_idx=0)` predicts `d_j - d_ref`, and Eqs. (4.34)
onward are written the second way. Solving a negated range difference asks for
the branch of the hyperbola on the far side of the array, so the square
geometry reported 13.753 m where its GDOP of 0.87 and 0.1 m of range noise
predict 0.087 m -- a factor of 158, with 11 of 100 fixes failing outright.

**Nothing caught it because nothing read the shipped bytes.** Chapter 4 had two
good accuracy tests either side of this file, `test_toa_attains_the_dop_bound`
and `test_tdoa_error_scales_linearly`, and both build their measurements inline
from the correct convention before solving them. They characterise the
*solvers*, which were never wrong. A dataset defect is invisible to a test that
synthesises its own data, exactly as a figure defect is invisible to a test that
only checks the file was written.

So this file asserts the one thing those cannot: that the bytes in `data/sim`
are the measurements the positioner consuming them expects. It compares each
stored file against the forward model, not against a solved position, which is
what makes it specific -- a solver change, a seed change or a tolerance change
cannot move it, because both sides are geometry.

Deliberately *not* here: whether solving the shipped data attains its GDOP
bound. That is an assertion about the solver, and it is already pinned
elsewhere -- `tests/test_datasets_reproduce_from_their_recipe.py` regenerates
each dataset and byte-compares `config.json`, whose `performance` block records
`median_m` and `failed_count`, so a solver regression surfaces there. Adding it
here would couple this file to the solver it is meant to be independent of.

All three measurement types are checked, not just the one that was broken. TOA
and AOA are correct today and cost nothing to pin; the AOA arm is worth having
because `aoa_azimuth` measures psi *from the agent toward the anchor* and the
opposite reading has already drawn `ch4_aoa_geometry`'s four bearing rays
backwards once.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.1)-(4.3), (4.27)-(4.33), (4.34)-(4.42), (4.63)-(4.66)
"""

import json
from functools import cache

import numpy as np
import pytest

from core.rf import aoa_azimuth, tdoa_range_difference, toa_range
from core.utils import angle_diff
from tests.example_runner import WORKSPACE_ROOT

#: Every dataset the Chapter 4 generator ships.
DATASETS = (
    "ch4_rf_2d_square",
    "ch4_rf_2d_optimal",
    "ch4_rf_2d_linear",
    "ch4_rf_2d_nlos",
)

#: A residual this many noise standard deviations from zero is a convention
#: error, not a draw. Every arm's margin on both sides is measured by
#: `test_the_bound_is_justified_against_both_the_noise_and_the_defect`.
TOLERANCE_SIGMA = 3.0


def _worst_column(residual):
    """Mean |residual| of the worst measurement column, in the units given.

    **The two obvious cheaper statistics are both wrong here, and each was
    written first.** Reviewing this file caught them:

    - A *signed* mean cancels. On the symmetric square array, negating every
      azimuth leaves the signed mean at 0.17 deg against an honest 0.17 deg,
      so a fully sign-inverted AOA file passes; per column the same defect
      reads 90.15 deg. Swapping `atan2`'s arguments -- the exact confusion
      `aoa_azimuth`'s docstring warns about -- is missed the same way. The
      TDOA negation defeats a signed mean *per column* too, at 0.12 sigma
      either way, because `d_j - d_ref` averages to nothing over a symmetric
      grid. Only an absolute residual survives all three.
    - A mean over *all* columns dilutes. A single beacon is one of four, so an
      undeclared +1.1 m bias on one of them -- larger than the 0.8 m the NLOS
      dataset legitimately ships -- reads 0.280 m against a 0.300 m gate and
      passes. Per column it reads 1.102 m.

    So: absolute, and per column. Honest values sit at 0.81-0.85 sigma across
    all three measurement types, which is `sqrt(2/pi)` = 0.798, the mean
    absolute deviation of the noise itself.
    """
    return np.abs(residual).mean(axis=0).max()


@cache
def _load(name):
    """Read one dataset. Cached: 13 tests would otherwise re-read 5 files each.

    Safe to memoise because it only reads and never asserts -- a cached helper
    that asserts inside itself re-runs its whole body on every call once
    anything fails, since `functools.cache` does not cache exceptions.
    """
    directory = WORKSPACE_ROOT / "data" / "sim" / name
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return {
        "beacons": np.loadtxt(directory / "beacons.txt"),
        "positions": np.loadtxt(directory / "ground_truth_positions.txt"),
        "toa": np.loadtxt(directory / "toa_ranges.txt"),
        "tdoa": np.loadtxt(directory / "tdoa_diffs.txt"),
        "aoa": np.loadtxt(directory / "aoa_angles.txt"),
        "config": config,
    }


def _predicted_toa(beacons, positions, config):
    """d_j, plus the NLOS bias the config declares on the beacons it names."""
    ranges = np.array([[toa_range(b, p) for b in beacons] for p in positions])
    nlos = config.get("nlos", {})
    if nlos.get("enabled"):
        for j in nlos["beacon_indices"]:
            ranges[:, j] += nlos["bias_m"]
    return ranges


def _predicted_tdoa(beacons, positions):
    """d_j - d_ref for j = 1..K-1: what `TDOAPositioner` predicts, Eq. (4.34).

    Spelled with `tdoa_range_difference` rather than inline norms so that all
    three helpers here use the same forward models the generator does, and so
    that the argument order this file exists to pin is written down once, in
    the order it belongs: anchor first, reference second.
    """
    return np.array(
        [
            [tdoa_range_difference(beacons[j], beacons[0], p)
             for j in range(1, len(beacons))]
            for p in positions
        ]
    )


def _predicted_aoa(beacons, positions):
    """psi measured from the agent toward each anchor, Eq. (4.64)."""
    return np.array([[aoa_azimuth(b, p) for b in beacons] for p in positions])


@pytest.mark.parametrize("dataset", DATASETS)
def test_toa_ranges_match_the_range_model(dataset):
    data = _load(dataset)
    sigma = data["config"]["measurements"]["toa_noise_std_m"]
    residual = data["toa"] - _predicted_toa(
        data["beacons"], data["positions"], data["config"]
    )
    worst = _worst_column(residual)
    assert worst < TOLERANCE_SIGMA * sigma, (
        f"{dataset}/toa_ranges.txt disagrees with the range model by "
        f"{worst:.3f} m on its worst beacon, against {sigma} m of declared "
        f"noise. Per-beacon means: "
        f"{np.round(np.abs(residual).mean(axis=0), 3).tolist()}. A single "
        f"beacon standing out is a bias the config does not declare; all of "
        f"them is a convention error."
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_tdoa_differences_carry_the_sign_the_positioner_predicts(dataset):
    """The regression guard. Read this file's docstring for what it cost.

    Compares against *both* signs and reports which one the file is in, so a
    failure names the defect rather than only announcing a disagreement.
    """
    data = _load(dataset)
    sigma = data["config"]["measurements"]["tdoa_noise_std_m"]
    predicted = _predicted_tdoa(data["beacons"], data["positions"])

    as_shipped = _worst_column(data["tdoa"] - predicted)
    negated = _worst_column(data["tdoa"] + predicted)
    diagnosis = (
        " The file is negated -- check the argument order of "
        "`tdoa_range_difference` in the generator, which returns d_i - d_j."
        if negated < as_shipped
        else ""
    )

    assert as_shipped < TOLERANCE_SIGMA * sigma, (
        f"{dataset}/tdoa_diffs.txt does not match d_j - d_ref: worst column "
        f"has mean |stored - predicted| of {as_shipped:.3f} m on {sigma} m of "
        f"noise. Against the opposite sign it is {negated:.3f} m.{diagnosis}"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_aoa_angles_are_measured_from_the_agent_toward_the_anchor(dataset):
    """Wrapped, because a raw bearing difference straddles the branch cut."""
    data = _load(dataset)
    sigma_deg = data["config"]["measurements"]["aoa_noise_std_deg"]
    residual = angle_diff(data["aoa"], _predicted_aoa(
        data["beacons"], data["positions"]
    ))
    worst = _worst_column(residual)
    assert worst < TOLERANCE_SIGMA * np.deg2rad(sigma_deg), (
        f"{dataset}/aoa_angles.txt disagrees with `aoa_azimuth` by "
        f"{np.rad2deg(worst):.2f} deg on its worst beacon, against "
        f"{sigma_deg} deg of declared noise. Per-beacon means (deg): "
        f"{np.round(np.rad2deg(np.abs(residual).mean(axis=0)), 2).tolist()}. "
        f"psi is measured from the agent toward the anchor; the reverse "
        f"bearing is 180 deg away, a sign flip is -psi, and swapping "
        f"`atan2`'s arguments reflects about the 45 deg line."
    )


def test_the_bound_is_justified_against_both_the_noise_and_the_defect():
    """A tolerance is decorative unless it is measured against both sides.

    Against the noise, or the guard is flaky; against the defect it must still
    catch, or it is not a guard. Both margins are recomputed on every run
    rather than written down once, so a later change to the datasets' noise
    levels cannot quietly close the gap.

    **This test covered only the TDOA arm when it was written, and that is how
    the other two shipped broken.** It reported a healthy 0.77-vs-90 sigma
    spread that described one third of what the file guards, while the TOA and
    AOA arms were reusing `TOLERANCE_SIGMA` against a statistic nobody had
    measured. Every arm is exercised here now, against the specific corruption
    it exists to reject.
    """
    corruptions = {
        # (arm, what a defect of that kind looks like in the stored file)
        "TOA: +1.1 m undeclared on one beacon": (
            "toa", "toa_noise_std_m",
            lambda a: _bump_column(a, 0, 1.1),
        ),
        "TDOA: sign convention reversed": (
            "tdoa", "tdoa_noise_std_m",
            lambda a: -a,
        ),
        "TDOA: +0.6 m on one column": (
            "tdoa", "tdoa_noise_std_m",
            lambda a: _bump_column(a, 0, 0.6),
        ),
        "AOA: azimuth sign flipped": (
            "aoa", "aoa_noise_std_deg",
            lambda a: -a,
        ),
        "AOA: reverse bearing (anchor to agent)": (
            "aoa", "aoa_noise_std_deg",
            lambda a: np.arctan2(-np.sin(a), -np.cos(a)),
        ),
        "AOA: atan2 arguments swapped": (
            "aoa", "aoa_noise_std_deg",
            lambda a: np.pi / 2 - a,
        ),
    }

    worst_honest = 0.0
    weakest_defect = np.inf
    weakest_name = ""

    for dataset in DATASETS:
        data = _load(dataset)
        for arm in ("toa", "tdoa", "aoa"):
            sigma = _sigma_for(data["config"], arm)
            worst_honest = max(
                worst_honest, _residual(data, arm, data[arm]) / sigma
            )

        for name, (arm, _key, corrupt) in corruptions.items():
            sigma = _sigma_for(data["config"], arm)
            margin = _residual(data, arm, corrupt(data[arm])) / sigma
            if margin < weakest_defect:
                weakest_defect, weakest_name = margin, f"{name} on {dataset}"

    assert worst_honest < TOLERANCE_SIGMA, (
        f"the loosest honest residual across all three arms is "
        f"{worst_honest:.2f} sigma, at or above the {TOLERANCE_SIGMA} sigma "
        f"gate: these tests are now flaky."
    )
    assert weakest_defect > TOLERANCE_SIGMA, (
        f"the weakest corruption this file must reject -- {weakest_name} -- "
        f"registers only {weakest_defect:.2f} sigma, which a "
        f"{TOLERANCE_SIGMA} sigma gate lets through. The guard no longer "
        f"detects a defect it was written for."
    )


def _bump_column(array, column, amount):
    """A defect confined to one beacon, which a mean over all of them hides."""
    corrupted = array.copy()
    corrupted[:, column] += amount
    return corrupted


def _sigma_for(config, arm):
    """Declared noise for one arm, in the units that arm's residual is in."""
    measurements = config["measurements"]
    if arm == "aoa":
        return np.deg2rad(measurements["aoa_noise_std_deg"])
    return measurements[f"{arm}_noise_std_m"]


def _residual(data, arm, observed):
    """Worst-column mean |residual| of `observed` against the forward model."""
    if arm == "toa":
        predicted = _predicted_toa(
            data["beacons"], data["positions"], data["config"]
        )
        return _worst_column(observed - predicted)
    if arm == "tdoa":
        predicted = _predicted_tdoa(data["beacons"], data["positions"])
        return _worst_column(observed - predicted)
    predicted = _predicted_aoa(data["beacons"], data["positions"])
    return _worst_column(angle_diff(observed, predicted))
