"""The shipped TDOA differences must share their reference beacon's error.

A TDOA measurement is not an independent observable. It is built from two
arrival-time measurements, ``(d_j + e_j) - (d_0 + e_0)``, and every difference
in the set is taken against the *same* reference beacon, so ``e_0`` is common
mode (Eq. 4.42)::

    Var(z_j) = 2 sigma^2    Cov(z_j, z_k) = sigma^2    rho = 0.5

The generator drew each difference from ``N(0, sigma)`` instead. That deletes
the common term and hands TDOA information it cannot physically have, and the
consequence was not subtle: mean GDOP on the square array read **0.8730**
against a true 1.0665, and the median error **0.0746 m** against 0.0923 m --
beating TOA's 0.0881 m on the same beacons. Differencing is a projection. It
throws the receiver clock away and cannot add information, so a correctly
simulated TDOA can never beat TOA on the same array; it ties
TOA-with-an-estimated-clock, which is the identity
``test_the_tdoa_dop_is_the_toa_with_a_clock_dop`` pins below.

**Why the guard next door could not see this.**
``test_shipped_measurements_match_the_solver_convention`` compares each stored
measurement against the forward model and gates the worst column's mean
|residual| at 3 sigma. Independent draws make that statistic *smaller*, not
larger -- 0.56 sigma against an honest 0.81 -- so the defect sat comfortably
inside a guard written to catch the opposite kind of error. A magnitude check
cannot see a correlation, and 0.56 looks healthier than 0.81 to anyone reading
the number rather than the model.

So this file asserts the second moment: the spread of each difference *and* the
covariance between them. Both are needed. Halving the reference error's
contribution moves the spread while leaving rho at 0.5; permuting the columns
against each other moves rho while leaving the spread untouched. Neither
statistic alone rejects both, and
``test_the_bounds_are_justified_against_both_the_noise_and_the_defect``
measures each of them against each corruption on every run.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.27)-(4.33), Eq. (4.42), Section 4.5
"""

import json
from functools import cache

import numpy as np
import pytest

from core.rf import (
    build_tdoa_covariance,
    compute_dop,
    compute_geometry_matrix,
    toa_range,
)
from tests.example_runner import WORKSPACE_ROOT

#: Every dataset the Chapter 4 generator ships.
DATASETS = (
    "ch4_rf_2d_square",
    "ch4_rf_2d_optimal",
    "ch4_rf_2d_linear",
    "ch4_rf_2d_nlos",
)

#: Correlation between two differences sharing a reference (Eq. 4.42).
EXPECTED_RHO = 0.5

#: Spread of one difference, in units of the per-arrival-time sigma.
EXPECTED_STD = np.sqrt(2.0)

#: Both bounds are absolute, on statistics of 100 samples. Justified against
#: the sampling noise *and* against the defects they must reject by
#: `test_the_bounds_are_justified_against_both_the_noise_and_the_defect`,
#: which recomputes every margin rather than trusting these numbers.
RHO_TOLERANCE = 0.15
STD_TOLERANCE = 0.20


@cache
def _load(name):
    """One dataset, plus the error each stored difference actually carries.

    Read-only and assertion-free, so memoising it is safe: `functools.cache`
    does not cache exceptions, and a cached helper that asserts inside itself
    re-runs its whole body for every caller once anything fails.
    """
    directory = WORKSPACE_ROOT / "data" / "sim" / name
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    beacons = np.loadtxt(directory / "beacons.txt")
    positions = np.loadtxt(directory / "ground_truth_positions.txt")
    tdoa = np.loadtxt(directory / "tdoa_diffs.txt")

    # The noiseless, bias-carrying prediction. The NLOS bias survives the
    # differencing -- it is not common mode, since beacon 0 is line of sight --
    # so it has to be subtracted here or it reads as a systematic error.
    truth = np.array([[toa_range(b, p) for b in beacons] for p in positions])
    nlos = config.get("nlos", {})
    if nlos.get("enabled"):
        for j in nlos["beacon_indices"]:
            truth[:, j] += nlos["bias_m"]

    residual = tdoa - (truth[:, 1:] - truth[:, [0]])
    return {
        "beacons": beacons,
        "positions": positions,
        "gdop_tdoa": np.loadtxt(directory / "gdop_tdoa.txt"),
        "residual": residual,
        "sigma": config["measurements"]["tdoa_noise_std_m"],
        "config": config,
    }


def _rho_and_std(residual, sigma):
    """Mean off-diagonal correlation, and mean spread in units of sigma."""
    covariance = np.cov(residual.T)
    k = covariance.shape[0]
    off_diagonal = covariance[~np.eye(k, dtype=bool)].mean()
    return off_diagonal / np.diag(covariance).mean(), (
        np.sqrt(np.diag(covariance)).mean() / sigma
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_differences_are_correlated_through_their_reference(dataset):
    """rho = 0.5, because every difference carries the same e_0."""
    data = _load(dataset)
    rho, _ = _rho_and_std(data["residual"], data["sigma"])
    assert abs(rho - EXPECTED_RHO) < RHO_TOLERANCE, (
        f"{dataset}/tdoa_diffs.txt has mean off-diagonal correlation "
        f"{rho:.3f}, against the {EXPECTED_RHO} that sharing a reference "
        f"beacon produces (Eq. 4.42). A value near 0 means each difference "
        f"was drawn independently, which deletes the common reference error "
        f"and makes TDOA look better than the information it carries allows."
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_each_difference_carries_two_arrival_time_errors(dataset):
    """std = sqrt(2) sigma, because a difference has two error terms in it."""
    data = _load(dataset)
    _, std = _rho_and_std(data["residual"], data["sigma"])
    assert abs(std - EXPECTED_STD) < STD_TOLERANCE, (
        f"{dataset}/tdoa_diffs.txt spreads {std:.3f} sigma per difference, "
        f"against sqrt(2) = {EXPECTED_STD:.3f}. `tdoa_noise_std_m` is the "
        f"per-*arrival-time* noise; a difference of two of them has variance "
        f"2 sigma^2. A value near 1.0 means the difference was treated as one "
        f"measurement rather than two."
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_shipped_gdop_is_weighted_for_that_correlation(dataset):
    """`gdop_tdoa.txt` must be the W = C^-1 DOP, not the unweighted one.

    The measurement half and the DOP half of this defect are independent: the
    file could carry correctly correlated differences and still report the DOP
    of an uncorrelated set, which is what it did for as long as it existed.
    """
    data = _load(dataset)
    beacons = data["beacons"]
    weights = np.linalg.inv(build_tdoa_covariance(np.ones(len(beacons))))

    weighted, unweighted = [], []
    for position in data["positions"]:
        H = compute_geometry_matrix(beacons, position, "tdoa")
        weighted.append(compute_dop(H, weights=weights)["GDOP"])
        unweighted.append(compute_dop(H)["GDOP"])

    stored = data["gdop_tdoa"]
    # Files are written at %.6f, so 1e-5 is comfortably above quantisation
    # and ten thousand times below the gap to the unweighted value.
    assert np.allclose(stored, weighted, atol=1e-5), (
        f"{dataset}/gdop_tdoa.txt is not the correlation-weighted DOP: worst "
        f"disagreement {np.abs(stored - np.array(weighted)).max():.4f}. "
        f"Against the *unweighted* DOP it disagrees by "
        f"{np.abs(stored - np.array(unweighted)).max():.4f} -- if that is the "
        f"smaller number, `compute_dop` was called without `weights`."
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_tdoa_dop_is_the_toa_with_a_clock_dop(dataset):
    """The identity the chapter exists to teach, checked position by position.

    TDOA is what is left of TOA once an unknown receiver clock is projected
    out, so the two carry the *same* information and their position DOPs are
    equal -- exactly, not approximately. This is the reason a correct TDOA
    simulation cannot beat TOA, and the reason TDOA's real trade is
    operational rather than statistical: it needs no synchronised clock.
    """
    data = _load(dataset)
    beacons = data["beacons"]
    weights = np.linalg.inv(build_tdoa_covariance(np.ones(len(beacons))))

    worst = 0.0
    for position in data["positions"]:
        H_tdoa = compute_geometry_matrix(beacons, position, "tdoa")
        tdoa_gdop = compute_dop(H_tdoa, weights=weights)["GDOP"]

        # TOA with the clock as a third unknown: a common bias adds 1 to every
        # range, so the design matrix gains a column of ones (Eq. 4.24-4.26).
        H_toa = compute_geometry_matrix(beacons, position, "toa")
        augmented = np.column_stack([H_toa, np.ones(len(beacons))])
        Q = np.linalg.inv(augmented.T @ augmented)
        toa_clock_gdop = np.sqrt(Q[0, 0] + Q[1, 1])

        worst = max(worst, abs(tdoa_gdop - toa_clock_gdop))

    assert worst < 1e-9, (
        f"on {dataset} the correlation-weighted TDOA GDOP and the "
        f"clock-estimating TOA GDOP differ by up to {worst:.2e}. They are the "
        f"same quantity; a gap means one of the two models is wrong."
    )


def test_the_bounds_are_justified_against_both_the_noise_and_the_defect():
    """A tolerance is decorative unless measured against both sides.

    Against the sampling noise, or the guard is flaky; against the defect it
    must still reject, or it is not a guard. Both margins are recomputed here
    on every run rather than written down once, so a later change to the
    datasets cannot quietly close the gap.

    **The two corruptions are chosen to defeat one statistic each**, which is
    how the pair earns its place:

    - *independent draws* -- each column permuted against the others. This is
      the shipped defect: it destroys the correlation and leaves every
      column's marginal distribution exactly as it was, so the spread check
      passes it and only rho rejects it.
    - *one error per difference* -- the residual scaled by 1/sqrt(2), the
      spread an independently drawn difference would have. rho is unchanged
      by a scaling, so only the spread check rejects it.

    A permutation rather than a fresh draw, because a corruption that needs a
    seed is a corruption whose margin moves when the seed does.

    **The first version of this test took a minimum over both corruptions per
    statistic**, which reports the *undersized* file's rho beside the
    *independent* file's spread -- two numbers describing two different
    corruptions, neither of which is the margin either gate actually has. It
    failed on arrival and it was right to: what has to hold is per corruption,
    that *some* statistic rejects it. Same antipattern this repository has
    already recorded twice inside the code under test, arriving in the guard.
    """
    worst_rho_error = 0.0
    worst_std_error = 0.0
    weakest_defect = np.inf
    weakest_name = ""

    for dataset in DATASETS:
        data = _load(dataset)
        sigma, residual = data["sigma"], data["residual"]

        rho, std = _rho_and_std(residual, sigma)
        worst_rho_error = max(worst_rho_error, abs(rho - EXPECTED_RHO))
        worst_std_error = max(worst_std_error, abs(std - EXPECTED_STD))

        rng = np.random.default_rng(0)
        corruptions = {
            "independent draws (columns permuted apart)": np.column_stack(
                [rng.permutation(column) for column in residual.T]
            ),
            "one arrival-time error per difference (/sqrt 2)": (
                residual / np.sqrt(2.0)
            ),
        }
        for name, corrupted in corruptions.items():
            bad_rho, bad_std = _rho_and_std(corrupted, sigma)
            # In units of each gate, so the two are comparable. A corruption
            # is rejected if *either* statistic clears its own gate.
            margin = max(
                abs(bad_rho - EXPECTED_RHO) / RHO_TOLERANCE,
                abs(bad_std - EXPECTED_STD) / STD_TOLERANCE,
            )
            if margin < weakest_defect:
                weakest_defect, weakest_name = margin, f"{name} on {dataset}"

    assert worst_rho_error < RHO_TOLERANCE, (
        f"the honest correlation is already {worst_rho_error:.3f} from "
        f"{EXPECTED_RHO}, against a {RHO_TOLERANCE} gate: this guard is flaky."
    )
    assert worst_std_error < STD_TOLERANCE, (
        f"the honest spread is already {worst_std_error:.3f} from sqrt(2), "
        f"against a {STD_TOLERANCE} gate: this guard is flaky."
    )
    assert weakest_defect > 1.0, (
        f"the weakest corruption these gates must reject -- {weakest_name} -- "
        f"registers only {weakest_defect:.2f}x its gate, so neither statistic "
        f"separates it from an honest file."
    )
