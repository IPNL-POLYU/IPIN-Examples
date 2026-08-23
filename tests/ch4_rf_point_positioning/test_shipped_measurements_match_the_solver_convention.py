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

All three measurement types are checked, not just the one that was broken. TOA
and AOA are correct today and cost nothing to pin; the AOA arm is worth having
because `aoa_azimuth` measures psi *from the agent toward the anchor* and the
opposite reading has already drawn `ch4_aoa_geometry`'s four bearing rays
backwards once.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.1)-(4.3), (4.27)-(4.33), (4.34)-(4.42), (4.63)-(4.66)
"""

import json

import numpy as np
import pytest

from core.rf import aoa_azimuth, toa_range
from core.utils import angle_diff
from tests.example_runner import WORKSPACE_ROOT

#: Every dataset the Chapter 4 generator ships.
DATASETS = (
    "ch4_rf_2d_square",
    "ch4_rf_2d_optimal",
    "ch4_rf_2d_linear",
    "ch4_rf_2d_nlos",
)

#: A systematic residual this many noise standard deviations from zero is a
#: convention error, not a draw. Both sides of this number are measured by
#: `test_the_bound_is_justified_against_both_the_noise_and_the_defect`.
TOLERANCE_SIGMA = 3.0


def _load(name):
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
    """d_j - d_ref for j = 1..K-1: what `TDOAPositioner` predicts, Eq. (4.34)."""
    d_ref = np.linalg.norm(positions - beacons[0], axis=1)
    return np.array(
        [
            np.linalg.norm(positions - beacons[j], axis=1) - d_ref
            for j in range(1, len(beacons))
        ]
    ).T


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
    assert abs(residual.mean()) < TOLERANCE_SIGMA * sigma, (
        f"{dataset}/toa_ranges.txt has a systematic residual of "
        f"{residual.mean():+.3f} m against the range model, on {sigma} m of "
        f"declared noise. A mean this far from zero is a convention, or a bias "
        f"the config does not declare, rather than a draw."
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

    as_shipped = np.abs(data["tdoa"] - predicted).mean()
    negated = np.abs(data["tdoa"] + predicted).mean()
    diagnosis = (
        " The file is negated -- check the argument order of "
        "`tdoa_range_difference` in the generator, which returns d_i - d_j."
        if negated < as_shipped
        else ""
    )

    assert as_shipped < TOLERANCE_SIGMA * sigma, (
        f"{dataset}/tdoa_diffs.txt does not match d_j - d_ref: mean "
        f"|stored - predicted| is {as_shipped:.3f} m on {sigma} m of noise. "
        f"Against the opposite sign it is {negated:.3f} m.{diagnosis}"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_aoa_angles_are_measured_from_the_agent_toward_the_anchor(dataset):
    """Wrapped, because a raw bearing difference straddles the branch cut."""
    data = _load(dataset)
    sigma_deg = data["config"]["measurements"]["aoa_noise_std_deg"]
    residual = angle_diff(data["aoa"], _predicted_aoa(
        data["beacons"], data["positions"]
    ))
    assert abs(residual.mean()) < TOLERANCE_SIGMA * np.deg2rad(sigma_deg), (
        f"{dataset}/aoa_angles.txt has a systematic bearing residual of "
        f"{np.rad2deg(residual.mean()):+.2f} deg against `aoa_azimuth`, on "
        f"{sigma_deg} deg of declared noise. psi is measured from the agent "
        f"toward the anchor; the reverse bearing is 180 deg away."
    )


def test_the_bound_is_justified_against_both_the_noise_and_the_defect():
    """A tolerance is decorative unless it is measured against both sides.

    Against the noise, or the guard is flaky; against the defect it must still
    catch, or it is not a guard. Both margins are recomputed here rather than
    asserted once and trusted, so a later change to the datasets' noise levels
    cannot quietly close the gap.
    """
    worst_honest = 0.0
    smallest_defect = np.inf

    for dataset in DATASETS:
        data = _load(dataset)
        sigma = data["config"]["measurements"]["tdoa_noise_std_m"]
        predicted = _predicted_tdoa(data["beacons"], data["positions"])
        worst_honest = max(
            worst_honest, np.abs(data["tdoa"] - predicted).mean() / sigma
        )
        smallest_defect = min(
            smallest_defect, np.abs(-data["tdoa"] - predicted).mean() / sigma
        )

    assert worst_honest < TOLERANCE_SIGMA, (
        f"the loosest honest residual is {worst_honest:.2f} sigma, at or above "
        f"the {TOLERANCE_SIGMA} sigma gate: these tests are now flaky."
    )
    assert smallest_defect > TOLERANCE_SIGMA, (
        f"negating the weakest-signal dataset gives {smallest_defect:.2f} "
        f"sigma, which a {TOLERANCE_SIGMA} sigma gate would not catch: the "
        f"guard no longer detects the defect it was written for."
    )
