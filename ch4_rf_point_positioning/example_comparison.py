"""
Comparison of RF Positioning Methods.

This script compares TOA, TDOA, AOA, and RSS positioning methods
under various conditions using pre-generated datasets.

Can run with:
    - Inline data (default):
        python -m ch4_rf_point_positioning.example_comparison
    - Pre-generated dataset:
        python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_square
    - Compare geometries, every method on every beacon layout:
        python -m ch4_rf_point_positioning.example_comparison --compare-geometry

Implements:
    - TOA positioning (Eqs. 4.14-4.23)
    - TDOA positioning (Eqs. 4.34-4.42)
    - AOA positioning (Eqs. 4.63-4.67)
    - DOP analysis (Section 4.5)

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import save_figure, show_figures_if_requested
from core.rf import (
    DIVERGENCE_M,
    AOAPositioner,
    SolveOutcome,
    TDOAPositioner,
    TOAPositioner,
    aoa_azimuth,
    rss_to_distance,
    solve_batch,
    toa_range,
    toa_solve_with_clock_bias,
)
from core.utils import resolve_data_path


def load_rf_dataset(data_dir: str) -> Dict:
    """Load RF positioning dataset.

    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch4_rf_2d_square')

    Returns:
        Dictionary with beacons, positions, measurements, and config
    """
    path = Path(data_dir)

    data = {
        "beacons": np.loadtxt(path / "beacons.txt"),
        "positions": np.loadtxt(path / "ground_truth_positions.txt"),
        "toa_ranges": np.loadtxt(path / "toa_ranges.txt"),
        "tdoa_diffs": np.loadtxt(path / "tdoa_diffs.txt"),
        "aoa_angles": np.loadtxt(path / "aoa_angles.txt"),
        "gdop_toa": np.loadtxt(path / "gdop_toa.txt"),
        "gdop_tdoa": np.loadtxt(path / "gdop_tdoa.txt"),
        "gdop_aoa": np.loadtxt(path / "gdop_aoa.txt"),
    }

    with open(path / "config.json") as f:
        data["config"] = json.load(f)

    return data


#: The three methods this comparison reports, in the order Chapter 4
#: introduces them. Every geometry reports all three, including the ones that
#: fail on it -- a method that vanishes from a table because nothing converged
#: is the failure being hidden, not reported.
METHODS = ("TOA", "TDOA", "AOA")


def solve_every_method(data: dict, verbose: bool = True) -> dict[str, SolveOutcome]:
    """Solve a dataset with each of TOA, TDOA and AOA from the beacon centroid.

    The centroid is what a real system starts from, and it is what the dataset
    generator seeds with. Both go through `core.rf.solve_batch`, so the
    **failure counts here equal the `failed_count` in each `config.json`** --
    checked on all four ch4 datasets.

    The TOA *median* equals the one recorded there too, now that the generator
    also asks for `iterative_ls`. It used to ask for `iwls` -- a deprecated
    alias resolving to the 1/d^2 range-weighted solver -- so the same
    measurements were reported at 0.095 m there and 0.088 m here, and this
    docstring existed to explain the gap away.
    """
    beacons = data["beacons"]
    truth = data["positions"]
    guess = np.mean(beacons, axis=0)

    solvers = {
        "TOA": (TOAPositioner(beacons, method="iterative_ls"), data["toa_ranges"]),
        "TDOA": (
            TDOAPositioner(beacons, reference_anchor_index=0),
            data["tdoa_diffs"],
        ),
        "AOA": (AOAPositioner(beacons), data["aoa_angles"]),
    }

    outcomes = {}
    for method in METHODS:
        solver, measurements = solvers[method]
        if verbose:
            print(f"\n--- Running {method} Positioning ---")
        outcomes[method] = solve_batch(
            solver,
            measurements,
            guess,
            truth,
            progress=partial(tqdm, desc=method, disable=not verbose),
        )
    return outcomes


def print_method_table(
    outcomes: dict[str, SolveOutcome], gdop: dict[str, np.ndarray]
) -> None:
    """One row per method, always -- failures included, nothing omitted."""
    print(
        f"A fix has failed if it raised, reported converged=False, never left "
        f"the\ninitial guess, or landed over {DIVERGENCE_M:.0f} m away. "
        f"'median' is over every fix that\nreturned a number; 'mean' and "
        f"'worst' are over the ones that solved."
    )
    header = (
        f"{'Method':<8}{'median(m)':>11}{'mean(m)':>11}{'worst(m)':>11}"
        f"{'failed':>12}{'DOP':>10}"
    )
    print(header)
    print("-" * len(header))
    for method in METHODS:
        out = outcomes[method]
        failed = f"{out.n_failed}/{out.n}"
        print(
            f"{method:<8}{out.median_m:>11.3f}{out.mean_solved_m:>11.3f}"
            f"{out.max_solved_m:>11.3f}{failed:>12}"
            f"{np.mean(gdop[method]):>10.2f}"
        )
    # The AOA entry in that column is not a GDOP and must not be read against
    # the other two. Its geometry rows are [-dy/d^2, dx/d^2], units 1/m, so
    # the number is metres of position error per radian of bearing error.
    # Printed rather than left implicit because the column header cannot say
    # two things at once, and "AOA 15.04 vs TOA 1.02" is the wrong reading.
    print(
        "\nDOP is dimensionless for TOA and TDOA. The AOA entry is a "
        "sensitivity in\nmetres per radian, so it is not comparable to them: "
        "multiply it by the\nangular noise in radians to get a position error. "
        "Divide by a reference\nrange for the dimensionless form -- see "
        "config.json's dop block."
    )


def run_with_dataset(data_dir: str, verbose: bool = True) -> dict:
    """Run RF positioning comparison using pre-generated dataset.

    Args:
        data_dir: Path to dataset directory
        verbose: Print detailed output

    Returns:
        Dictionary with a SolveOutcome per method, plus geometry and GDOP.
    """
    if verbose:
        print("=" * 70)
        print("Chapter 4: RF Positioning Methods Comparison")
        print(f"Using dataset: {data_dir}")
        print("=" * 70)

    data = load_rf_dataset(data_dir)
    config = data["config"]
    beacons = data["beacons"]
    positions = data["positions"]

    if verbose:
        print("\nDataset Info:")
        print(f"  Geometry: {config.get('geometry', {}).get('type', 'unknown')}")
        print(f"  Beacons: {len(beacons)}")
        print(f"  Test points: {len(positions)}")
        print(
            f"  TOA noise: "
            f"{config.get('measurements', {}).get('toa_noise_std_m', 'N/A')} m"
        )
        print(
            f"  AOA noise: "
            f"{config.get('measurements', {}).get('aoa_noise_std_deg', 'N/A')} deg"
        )

    outcomes = solve_every_method(data, verbose=verbose)

    # `gdop_aoa.txt` is a *sensitivity* in metres per radian, not a
    # dimensionless DOP -- the AOA geometry rows are [-dy/d^2, dx/d^2], units
    # 1/m. The printed table reports it as shipped, with a footnote. The
    # figure divides it by the mean beacon range at each point so that its two
    # DOP panels compare one quantity across all three methods instead of
    # putting 15.04 m/rad on the same axis as a dimensionless 1.02. Without
    # that, the scatter reads as "AOA is worse because its geometry is worse",
    # which on the NLOS dataset is exactly backwards: the geometry is
    # byte-identical to the clean square and the error is bias.
    mean_range = np.linalg.norm(
        positions[:, None, :] - beacons[None, :, :], axis=2
    ).mean(axis=1)

    results = {
        "outcomes": outcomes,
        "gdop": {
            "TOA": data["gdop_toa"],
            "TDOA": data["gdop_tdoa"],
            "AOA": data["gdop_aoa"],
        },
        "dop_dimensionless": {
            "TOA": data["gdop_toa"],
            "TDOA": data["gdop_tdoa"],
            "AOA": data["gdop_aoa"] / mean_range,
        },
        "n_points": len(positions),
        "beacons": beacons,
        "positions": positions,
        "config": config,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        print_method_table(outcomes, results["gdop"])

    return results


#: The three geometries this mode compares. The labels are printed and are the
#: figure's x-axis, so they name the layout rather than grade it: "Linear
#: (poor)" prejudges a geometry that is the best of the three for AOA.
GEOMETRIES = (
    ("ch4_rf_2d_square", "Square (4 corners)"),
    ("ch4_rf_2d_optimal", "Optimal (circular)"),
    ("ch4_rf_2d_linear", "Collinear (4 in a row)"),
)


def compare_geometries(verbose: bool = True) -> dict:
    """Compare positioning performance across different beacon geometries.

    Uses ch4_rf_2d_square, ch4_rf_2d_optimal and ch4_rf_2d_linear.

    **Every geometry reports every method**, which is the whole point of the
    mode and used to be the one thing it could not do. Results were aggregated
    as an RMSE over the solves that reported convergence, so a method with no
    converged solves printed no line at all and a method whose "converged"
    solves included three at 1e11 m printed an RMSE of 2.2e10 m. Square and
    Optimal listed TOA and TDOA, the collinear geometry listed only AOA, and no
    method appeared on more than two of the three.
    """
    if verbose:
        print("=" * 70)
        print("Chapter 4: Beacon Geometry Comparison")
        print("=" * 70)

    all_results = {}

    for dataset_name, geometry_label in GEOMETRIES:
        data_path = resolve_data_path(Path("data/sim") / dataset_name)
        if not data_path.exists():
            if verbose:
                print(f"\nSkipping {dataset_name} (not found)")
            continue

        if verbose:
            print(f"\n{'='*70}")
            print(f"Geometry: {geometry_label}   [{dataset_name}]")
            print(f"{'='*70}")

        results = run_with_dataset(str(data_path), verbose=False)
        all_results[geometry_label] = results

        if verbose:
            print_method_table(results["outcomes"], results["gdop"])

    if verbose and all_results:
        print_geometry_insight(all_results)

    return all_results


def print_geometry_insight(all_results: dict) -> None:
    """The comparison this mode exists for: every method on every geometry."""
    print("\n" + "=" * 70)
    print("KEY INSIGHT: a geometry is only good relative to a measurement type")
    print("=" * 70)
    print("Median error in metres over all fixes, failed fixes in [brackets]:\n")

    width = max(len(label) for label in all_results) + 2
    print(f"{'Geometry':<{width}}" + "".join(f"{m:>18}" for m in METHODS))
    for label, results in all_results.items():
        row = f"{label:<{width}}"
        for method in METHODS:
            out = results["outcomes"][method]
            cell = f"{out.median_m:.3f} [{out.n_failed}]"
            row += f"{cell:>18}"
        print(row)

    print(
        "\nThe collinear array is not simply the bad one. It is bad for ranges"
        "\nand better than either alternative for bearings:\n"
        "\n"
        "  - TOA and TDOA fail on all 100 fixes *from the beacon centroid*,\n"
        "    which sits on the line of symmetry. Moving across that line\n"
        "    changes no range to first order, so the Jacobian is rank\n"
        "    deficient there and Gauss-Newton has nowhere to step. Their\n"
        "    median is the distance from the seed to the truth -- a property\n"
        "    of the seed, not of the measurements.\n"
        "  - Seeded off the line they solve, but ranges still cannot separate\n"
        "    a target from its mirror image about the beacon line, so half the\n"
        "    fixes land on the wrong side. See data/sim/ch4_rf_2d_linear.\n"
        "  - AOA is *better* here than on the square array, because reflecting\n"
        "    a position flips every azimuth: bearings carry the side\n"
        "    information that ranges do not. Its remaining failures are the\n"
        "    grid rows within 1 m of the beacon line, where all four bearings\n"
        "    are nearly parallel.\n"
        "\n"
        "DOP sees none of this. TOA GDOP on the collinear array averages 1.43\n"
        "against 1.02 for the square -- a local, first-order measure calling\n"
        "the configuration fine, while the ambiguity that breaks it is global.\n"
        "A healthy DOP is necessary, not sufficient."
    )


def generate_scenario(seed=42):
    """Generate a test scenario with anchors and true positions (inline mode)."""
    np.random.seed(seed)

    # Square anchor layout (10m x 10m area)
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    # Generate test positions
    n_points = 50
    x = np.random.uniform(1, 9, n_points)
    y = np.random.uniform(1, 9, n_points)
    true_positions = np.column_stack([x, y])

    return anchors, true_positions


def toa_positioning_test(
    anchors,
    true_positions,
    range_noise_std_m=0.0,
    clock_bias_m=0.0,
):
    """Test TOA positioning (inline mode).

    Args:
        anchors: Anchor positions, shape (K, 2).
        true_positions: Ground-truth agent positions, shape (N, 2).
        range_noise_std_m: Gaussian range-noise std in metres.
        clock_bias_m: Shared receiver clock bias in metres, added to
            every pseudorange.  TDOA differencing cancels this term; TOA
            has to estimate it, which is what the state below is for.

    **The state is (x, y, c*dt), not (x, y), and that is the whole point.**
    A shared bias is unobservable to a position-only solver: no (x, y)
    makes four pseudoranges each 1.5 m long consistent, so the residual
    never reaches tol and the solve is discarded. Reported as a
    "convergence rate" that read **2-5 of 100** at every noise level,
    which is a property of the harness and not of TOA -- and the few that
    survived were the geometries where the bias could be partly absorbed
    *into the position*, so they were the least-inaccurate rather than the
    accurate ones. The tell was the top row: 0.153 m of median error on
    **noiseless** measurements, where TDOA and AOA both report 0.000.

    `toa_solve_with_clock_bias` is Eqs. (4.24)-(4.26), the chapter's own
    joint position-and-clock estimate, and it was already exported from
    `core.rf`. With it: 100/100 converge at every level, the bias comes
    back as +1.500 m on noiseless data, and the median error is 0.000 m
    there. The figure now shows what Chapter 4 actually teaches -- TOA
    carries the clock as an unknown, TDOA differences it away.
    """
    errors = []

    for true_pos in tqdm(true_positions, desc="  TOA", leave=False, unit="pt"):
        ranges = np.array([toa_range(anchor, true_pos) for anchor in anchors])
        ranges += clock_bias_m
        if range_noise_std_m > 0:
            ranges += np.random.randn(len(ranges)) * range_noise_std_m

        try:
            # (x, y, c*dt): three unknowns from K >= 3 pseudoranges.
            est_pos, _bias, info = toa_solve_with_clock_bias(
                anchors, ranges, initial_guess=np.array([5.0, 5.0, 0.0])
            )
            if info["converged"]:
                error = np.linalg.norm(est_pos[:2] - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def tdoa_positioning_test(
    anchors, true_positions, range_noise_std_m=0.0, clock_bias_m=0.0
):
    """Test TDOA positioning (inline mode).

    **The noise is drawn per anchor and then differenced, not per difference.**
    A TDOA measurement is `(d_j + e_j) - (d_0 + e_0)`: every difference is
    formed against the same reference anchor, so `e_0` is common mode and the
    differences are correlated (Eq. 4.42, `build_tdoa_covariance`) --
    `Var = 2 sigma^2`, `Cov = sigma^2`, `rho = 0.5`.

    This used to add `np.random.randn(len(tdoa)) * noise_std` to the *true*
    differences, which deletes the shared term and hands TDOA information it
    cannot physically have. Differencing is a projection: it throws the
    receiver clock away and cannot add anything. So TDOA below TOA at every
    noise level, as this table used to print, was an artefact of the
    simulation and not a result -- the honest outcome is that TDOA *ties*
    TOA-with-an-estimated-clock, because they carry the same information.

    The clock bias is injected here for the same reason it is injected into
    the TOA pseudo-ranges: it is common to every anchor, so it cancels
    exactly in the differences. That is now demonstrated rather than asserted
    -- and it is what TDOA actually buys, since the transmitters need no
    synchronised clock with the receiver.
    """
    errors = []

    for true_pos in tqdm(true_positions, desc="  TDOA", leave=False, unit="pt"):
        ranges = np.array([np.linalg.norm(true_pos - anchor) for anchor in anchors])
        ranges = ranges + clock_bias_m
        if range_noise_std_m > 0:
            ranges = ranges + np.random.randn(len(ranges)) * range_noise_std_m

        tdoa = ranges[1:] - ranges[0]

        try:
            positioner = TDOAPositioner(anchors, reference_anchor_index=0)
            est_pos, info = positioner.solve(tdoa, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


#: One anchor's bearings are this much noisier than the rest. A real array has
#: a worst element -- an obstructed sector, a miscalibrated element -- and a
#: uniform noise model has nothing for measurement weighting to act on.
DEGRADED_ANCHOR = 3
DEGRADED_ANCHOR_SCALE = 10.0


def aoa_noise_per_anchor(n_anchors, aoa_noise_std_rad):
    """Per-anchor azimuth sigma, with one anchor deliberately degraded."""
    sigma = np.full(n_anchors, aoa_noise_std_rad, dtype=float)
    sigma[DEGRADED_ANCHOR % n_anchors] *= DEGRADED_ANCHOR_SCALE
    return sigma


def aoa_positioning_test(anchors, true_positions, aoa_noise_std_rad=0.0, weighted=True):
    """Test AOA positioning (inline mode).

    Args:
        weighted: Pass the per-anchor sigma to the solver, so it can
            down-weight the degraded anchor (Eq. 4.77's W_a). With `False` the
            solver treats every bearing as equally trustworthy, which is the
            control this comparison needs: a weighting that is never contrasted
            against its absence is an assertion, not a demonstration.

    Note a *scalar* sigma would do nothing here. In angle space the weight
    matrix is diag(1/sigma^2), so a uniform sigma makes W a multiple of the
    identity and it cancels exactly out of (H'WH)^-1 H'W. Only the spread
    between anchors carries information. (Under the old tan parameterisation a
    uniform sigma did change the answer, because var(tan psi) = sec^4(psi)
    var(psi) made the weights angle-dependent -- that was the amplification
    that let near-singular anchors dominate, not a feature.)
    """
    errors = []
    sigma = aoa_noise_per_anchor(len(anchors), aoa_noise_std_rad)

    for true_pos in tqdm(true_positions, desc="  AOA", leave=False, unit="pt"):
        aoa = np.array([aoa_azimuth(anchor, true_pos) for anchor in anchors])
        if aoa_noise_std_rad > 0:
            aoa += np.random.randn(len(aoa)) * sigma

        try:
            positioner = AOAPositioner(anchors)
            kwargs = {"initial_guess": np.array([5.0, 5.0])}
            if weighted and aoa_noise_std_rad > 0:
                kwargs["sigma_psi"] = sigma
            est_pos, info = positioner.solve_angles_rad(aoa, **kwargs)
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def rss_positioning_test(
    anchors,
    true_positions,
    sigma_long_db=0.0,
    sigma_short_linear=0.0,
    n_samples_avg=1,
    short_fading_model="rayleigh",
    path_loss_exp=2.5,
):
    """
    Test RSS positioning with fading noise per book model (Eqs. 4.10-4.13).

    Args:
        anchors: Anchor positions.
        true_positions: True agent positions to test.
        sigma_long_db: Long-term fading std in dB (per Eq. 4.12).
                      Typical indoor values: 4-8 dB.
        sigma_short_linear: Short-term fading parameter (Rayleigh scale sigma).
                           For Rayleigh: typical σ = 0.5-1.0. Defaults to 0.0.
        n_samples_avg: Number of samples to average for short-term fading
                      reduction. Defaults to 1 (no averaging).
        short_fading_model: Short-term fading model ("rayleigh", "gaussian_db", "none").
                           Defaults to "rayleigh".
        path_loss_exp: Path-loss exponent (eta). Defaults to 2.5.

    Returns:
        Array of position errors.
    """
    from core.rf import simulate_rss_measurement

    errors = []
    p_ref_dbm = -40.0  # Reference RSS at d_ref=1m (typical Wi-Fi beacon)

    for true_pos in tqdm(true_positions, desc="  RSS", leave=False, unit="pt"):
        ranges = []
        for anchor in anchors:
            # Use simulate_rss_measurement for full fading model (Eq. 4.12)
            rss_meas, info = simulate_rss_measurement(
                anchor,
                true_pos,
                p_ref_dbm=p_ref_dbm,
                path_loss_exp=path_loss_exp,
                sigma_long_db=sigma_long_db,
                sigma_short_linear=sigma_short_linear,
                n_samples_avg=n_samples_avg,
                short_fading_model=short_fading_model,
            )
            # Invert RSS to range (Eq. 4.11)
            range_est = rss_to_distance(rss_meas, p_ref_dbm, path_loss_exp)
            ranges.append(range_est)

        ranges = np.array(ranges)

        try:
            positioner = TOAPositioner(anchors, method="iterative_ls")
            est_pos, info = positioner.solve(ranges, initial_guess=np.array([5.0, 5.0]))
            if info["converged"]:
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)
        except Exception:
            continue

    return np.array(errors)


def run_inline_comparison():
    """Run comparison with inline generated data (original behavior).

    Each measurement type now uses its own physically-meaningful noise
    schedule so that the comparison is apples-to-apples.  A shared
    receiver clock bias is injected into TOA pseudo-ranges to demonstrate
    that TDOA differencing cancels it.
    """
    print("=" * 70)
    print("RF Positioning Methods Comparison")
    print("(Using inline generated data)")
    print("=" * 70)

    print("\n--- Setting up test scenario ---")
    anchors, true_positions = generate_scenario(seed=42)
    print("Test scenario created:")
    print(f"  Anchors: {len(anchors)}")
    print(f"  Test points: {len(true_positions)}")
    print("  Area: 10m x 10m")

    # ---- Independent noise schedules per method ----
    toa_range_noise_levels_m = [0.0, 0.05, 0.1, 0.2, 0.5]
    tdoa_range_noise_levels_m = [0.0, 0.05, 0.1, 0.2, 0.5]
    aoa_noise_levels_deg = [0.0, 1.0, 3.0, 5.0, 10.0]  # degrees
    rss_fading_noise_levels_db = [0.0, 2.0, 4.0, 6.0, 8.0]

    # Shared clock bias (metres) added to TOA; cancels in TDOA diffs.
    clock_bias_m = 1.5

    n_levels = len(toa_range_noise_levels_m)

    # "AOA_unw" is the unweighted control, reported in the table but not
    # plotted -- the figure compares measurement types, not solver options.
    results = {"TOA": [], "TDOA": [], "AOA": [], "RSS": [], "AOA_unw": []}

    print("\nNoise configuration (independent per method):")
    print(
        f"  TOA : range noise {toa_range_noise_levels_m} m  "
        f"(+ clock bias {clock_bias_m} m)"
    )
    print(f"  TDOA: range noise {tdoa_range_noise_levels_m} m  (clock bias cancels)")
    print(f"  AOA : angle noise {aoa_noise_levels_deg} deg")
    print(
        f"  RSS : long-term fading {rss_fading_noise_levels_db} dB "
        "+ Rayleigh short-term"
    )

    sigma_short_linear = 0.5
    n_samples_avg = 5

    print("\nTesting noise levels...")
    start_time = time.time()

    for i in tqdm(range(n_levels), desc="Overall progress", unit="level"):
        toa_range_noise_m = toa_range_noise_levels_m[i]
        tdoa_range_noise_m = tdoa_range_noise_levels_m[i]
        aoa_noise_rad = np.deg2rad(aoa_noise_levels_deg[i])
        rss_fading_db = rss_fading_noise_levels_db[i]

        print(
            f"\n[{i+1}/{n_levels}] TOA: {toa_range_noise_m:.2f}m "
            f"(+bias {clock_bias_m}m), "
            f"TDOA: {tdoa_range_noise_m:.2f}m, "
            f"AOA: {aoa_noise_levels_deg[i]:.1f}deg, "
            f"RSS: {rss_fading_db:.1f}dB"
        )

        results["TOA"].append(
            toa_positioning_test(
                anchors,
                true_positions,
                toa_range_noise_m,
                clock_bias_m=clock_bias_m,
            )
        )
        results["TDOA"].append(
            tdoa_positioning_test(
                anchors,
                true_positions,
                tdoa_range_noise_m,
                clock_bias_m=clock_bias_m,
            )
        )
        results["AOA"].append(
            aoa_positioning_test(anchors, true_positions, aoa_noise_rad)
        )
        # Same measurements, same seed offset, weighting switched off.
        results["AOA_unw"].append(
            aoa_positioning_test(anchors, true_positions, aoa_noise_rad, weighted=False)
        )
        results["RSS"].append(
            rss_positioning_test(
                anchors,
                true_positions,
                sigma_long_db=rss_fading_db,
                sigma_short_linear=sigma_short_linear,
                n_samples_avg=n_samples_avg,
                short_fading_model="rayleigh",
            )
        )

    elapsed_time = time.time() - start_time
    print(f"\nAll tests completed in {elapsed_time:.2f}s")

    print("\n" + "=" * 70)
    print("Results Summary (median error in metres)")
    print("=" * 70)
    print(f"  Clock bias: {clock_bias_m} m (TOA only; cancels in TDOA)")
    print(
        f"  RSS config: Rayleigh short-term (sigma={sigma_short_linear}), "
        f"{n_samples_avg} samples averaged"
    )
    print(
        f"  AOA anchor {DEGRADED_ANCHOR} is {DEGRADED_ANCHOR_SCALE:.0f}x noisier "
        f"than the others; 'AOA unw' solves the same bearings unweighted"
    )
    header = (
        f"{'Level':<6} {'TOA(m)':<9} {'TDOA(m)':<9} "
        f"{'AOA(deg)':<9} {'RSS(dB)':<9} "
        f"{'TOA':<9} {'TDOA':<9} {'AOA':<9} {'AOA unw':<9} "
        f"{'RSS':<9} {'AOA fail':<9}"
    )
    print(header)
    print("-" * len(header))

    # Median, not RMS. This table used to report an RMS, and the AOA column
    # read 5.3e9 m at *zero* angular noise -- not a comparison of methods but
    # of one method against a handful of its own divergences. See the note
    # printed under the table.
    for i in range(n_levels):

        def _median(arr):
            return np.median(arr) if len(arr) > 0 else np.nan

        def _gross(arr):
            return int(np.sum(np.asarray(arr) > 100.0)) if len(arr) > 0 else 0

        print(
            f"{i+1:<6} "
            f"{toa_range_noise_levels_m[i]:<9.2f} "
            f"{tdoa_range_noise_levels_m[i]:<9.2f} "
            f"{aoa_noise_levels_deg[i]:<9.1f} "
            f"{rss_fading_noise_levels_db[i]:<9.1f} "
            f"{_median(results['TOA'][i]):<9.3f} "
            f"{_median(results['TDOA'][i]):<9.3f} "
            f"{_median(results['AOA'][i]):<9.3f} "
            f"{_median(results['AOA_unw'][i]):<9.3f} "
            f"{_median(results['RSS'][i]):<9.3f} "
            f"{_gross(results['AOA'][i]):<9d}"
        )

    print()
    print("  'AOA fail' counts solves landing over 100 m away. It should now")
    print("  read 0 at every noise level. It did not used to: solving on")
    print("  z = tan(psi) as Eq. (4.64) is written literally, 8 of 39 converged")
    print("  solves from the anchor centroid were wrong, the worst by 3e10 m,")
    print("  while the other 31 were exact to 1e-8 m. That was a")
    print("  basin-of-attraction property of the parameterisation, not a")
    print("  noise-sensitivity result -- tan has period pi, so it cannot tell an")
    print("  anchor ahead from one behind, and its residuals shrink as the")
    print("  estimate runs to infinity, making infinity an attractor the solver")
    print("  reports as convergence.")
    print("  AOAPositioner now forms residuals as wrap(psi - atan2(dE, dN)),")
    print("  which is the same model without the quadrant thrown away. The old")
    print("  behaviour is still reachable as residual='tan'.")
    print()
    print("  'AOA' passes the per-anchor sigma to the solver so it can")
    print("  down-weight the degraded anchor (W_a in Eq. 4.77); 'AOA unw'")
    print("  solves the identical bearings with uniform weights. The gain is")
    print("  largest where the other anchors are still good -- 3-5x at 1 deg")
    print("  across five seeds -- and shrinks to 1.2-1.8x by 10 deg: weighting")
    print("  recovers what a bad sensor costs you only while the rest are")
    print("  worth trusting. By level 5 the degraded anchor is at 100 deg")
    print("  sigma -- near-uniform on the circle, so there is little left to")
    print("  down-weight -- and the other three are themselves at 10 deg.")
    print("  The middle levels are noisy at 50 test points and should be read")
    print("  as a trend, not a curve: this note used to say '4x at 1-3 deg,")
    print("  tapering to nothing by 10 deg', which was one realisation. On")
    print("  five seeds 10 deg gives 1.20-1.84x, and 3 deg spans 1.73-3.87x.")
    print()
    print("  A *scalar* sigma would change nothing at all. In angle space the")
    print("  weight matrix is diag(1/sigma^2), so a uniform sigma makes W a")
    print("  multiple of the identity and it cancels out of (H'WH)^-1 H'W.")
    print("  Only the spread between anchors carries information.")

    return toa_range_noise_levels_m, results


def plot_dataset_results(results: Dict, output_file: str = None):
    """Plot results from dataset-based comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF Positioning: Dataset Analysis", fontsize=16, fontweight="bold")

    beacons = results["beacons"]
    positions = results["positions"]

    # 1. Beacon geometry and test points
    ax1 = axes[0, 0]
    ax1.scatter(
        beacons[:, 0],
        beacons[:, 1],
        s=200,
        c="red",
        marker="^",
        label="Beacons",
        zorder=10,
        edgecolors="black",
        linewidths=2,
    )
    ax1.scatter(
        positions[:, 0], positions[:, 1], s=20, c="blue", alpha=0.5, label="Test Points"
    )
    for i, b in enumerate(beacons):
        ax1.annotate(
            f"B{i}",
            (b[0], b[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Beacon Geometry & Test Points")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # 2. Error CDF over the fixes that solved, labelled with the ones that
    #    did not. A CDF drawn over the successes alone reads as the method's
    #    accuracy; the label is what stops it reading that way.
    ax2 = axes[0, 1]
    colors = {"TOA": "blue", "TDOA": "red", "AOA": "green"}
    for method, color in colors.items():
        out = results["outcomes"][method]
        errors = out.errors[out.solved]
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(
                sorted_errors,
                cdf,
                color=color,
                linewidth=2,
                label=f"{method} ({out.n - out.n_failed}/{out.n} solved)",
            )
        else:
            ax2.plot(
                [], [], color=color, linewidth=2, label=f"{method} (0/{out.n} solved)"
            )
    ax2.set_xlabel("Position Error (m)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Error CDF over solved fixes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    # 3. DOP distribution, dimensionless for all three (see `run_with_dataset`)
    ax3 = axes[1, 0]
    dop = results.get("dop_dimensionless", results["gdop"])
    gdop_data = [dop["TOA"], dop["TDOA"], dop["AOA"]]
    bp = ax3.boxplot(gdop_data, tick_labels=["TOA", "TDOA", "AOA"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["blue", "red", "green"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("GDOP (dimensionless)")
    ax3.set_title("Dilution of precision\n(AOA divided by mean beacon range)")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Error vs GDOP scatter.
    #    Both sides are indexed by the same mask, so point i's error is paired
    #    with point i's GDOP. They used to be `errors[:n]` against `gdop[:n]`
    #    with the errors already compacted to the converged solves, which
    #    silently shifted the pairing whenever anything failed.
    ax4 = axes[1, 1]
    for method, color in colors.items():
        out = results["outcomes"][method]
        gdop = dop[method]
        if out.solved.any():
            ax4.scatter(
                gdop[out.solved],
                out.errors[out.solved],
                alpha=0.5,
                label=method,
                color=color,
                s=20,
            )
    ax4.set_xlabel("GDOP (dimensionless)")
    ax4.set_ylabel("Position Error (m)")
    # Not "lower GDOP = better geometry", which is true of geometry and false
    # of the picture: on the NLOS dataset AOA has the *lowest* DOP of the
    # three and the worst error by 5x, because its corruption is an 18 deg
    # bearing bias and DOP is blind to bias. A title asserting the opposite of
    # what the panel shows is worse than no title.
    ax4.set_title("Error vs GDOP (DOP predicts noise, not bias)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        paths = save_figure(fig, Path(output_file).parent, Path(output_file).stem)
        print(f"\n[OK] Figure saved: {paths[0]}")

    return fig


def plot_geometry_comparison(all_results: dict):
    """Median error and failure rate for every method on every geometry.

    Two panels, because the single-panel version could not carry the result.
    It plotted an RMSE over the converged solves on a linear axis, and on the
    collinear geometry AOA's converged set included three fixes at 1e11 m: one
    bar at 2.2e10 m flattened every other bar in the figure to zero height.
    Methods with nothing converged were drawn at zero, which reads as perfect
    rather than as absent.

    So: a log axis, because the honest numbers span 0.075 m to 6.77 m -- two
    decades even after the negated TDOA measurements were corrected; the median
    rather than an RMSE, because one divergence should not set the height of a
    bar; and the failure rate beside it, since a method can have a fine median
    and still not work -- which is exactly what the RMSE was hiding.
    """
    fig, (ax_err, ax_fail) = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.suptitle("Beacon geometry is method-specific", fontsize=17, fontweight="bold")

    labels = list(all_results)
    x = np.arange(len(labels))
    width = 0.22
    colors = {"TOA": "tab:blue", "TDOA": "tab:red", "AOA": "tab:green"}

    for i, method in enumerate(METHODS):
        medians = [all_results[g]["outcomes"][method].median_m for g in labels]
        failures = [
            100.0
            * all_results[g]["outcomes"][method].n_failed
            / all_results[g]["outcomes"][method].n
            for g in labels
        ]
        offset = x + (i - 1) * width

        bars = ax_err.bar(
            offset,
            medians,
            width,
            label=method,
            color=colors[method],
            edgecolor="white",
            linewidth=0.8,
        )
        for rect, median, failed in zip(bars, medians, failures, strict=True):
            ax_err.annotate(
                f"{median:.2f}",
                (rect.get_x() + rect.get_width() / 2, median),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
            )
            # Hatch the bars whose median is mostly failures, so a tall bar and
            # a broken method are distinguishable at a glance.
            if failed >= 50.0:
                rect.set_hatch("///")
                rect.set_edgecolor("black")

        ax_fail.bar(
            offset,
            failures,
            width,
            label=method,
            color=colors[method],
            edgecolor="white",
            linewidth=0.8,
        )

    ax_err.set_yscale("log")
    ax_err.set_ylabel("Median position error (m), log scale", fontsize=11)
    ax_err.set_title("Median error over all 100 fixes", fontsize=12)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, fontsize=10)
    # Headroom for the legend, on a log axis, so it cannot sit over a bar. The
    # first draft put it at the default "best" location, which matplotlib chose
    # to be on top of the collinear group.
    top = max(all_results[g]["outcomes"][m].median_m for g in labels for m in METHODS)
    ax_err.set_ylim(top=top * 12)
    ax_err.legend(title="Method", loc="upper center", ncols=3, fontsize=10)
    ax_err.grid(True, alpha=0.3, axis="y", which="both")
    ax_err.set_axisbelow(True)

    ax_fail.set_ylabel("Fixes that failed (%)", fontsize=11)
    ax_fail.set_title(
        "Failed = raised, refused, stalled,\nor landed >100 m away",
        fontsize=12,
    )
    ax_fail.set_xticks(x)
    ax_fail.set_xticklabels(labels, fontsize=10)
    ax_fail.set_ylim(0, 122)
    ax_fail.legend(title="Method", loc="upper center", ncols=3, fontsize=10)
    ax_fail.grid(True, alpha=0.3, axis="y")
    ax_fail.set_axisbelow(True)

    fig.text(
        0.5,
        0.055,
        "Hatched bars are medians made mostly of failed fixes: on the collinear "
        "array the beacon centroid lies on the line of symmetry,\nso TOA and "
        "TDOA never leave it, and their 6.77 m is the distance from the seed to "
        "the truth. AOA is the best of the three there.",
        ha="center",
        fontsize=9.5,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_inline_comparison(noise_levels, results):
    """Plot comparison results (inline mode)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF Positioning Methods Comparison", fontsize=16, fontweight="bold")

    methods = ["TOA", "TDOA", "AOA", "RSS"]
    colors = ["blue", "red", "green", "orange"]
    # One dash pattern per method, used by every line panel below. TOA and
    # TDOA now trace each other almost exactly -- they carry the same
    # information once TOA estimates its clock, which is the point of this
    # figure -- so with four solid lines only the last one drawn is visible
    # and the panel reads as "TOA is missing". The success-rate panel already
    # had this problem for the same reason and solved it this way.
    dashes = ["-", "--", "-.", ":"]

    # 1. RMSE vs Noise
    ax1 = axes[0, 0]
    for method, color, dash in zip(methods, colors, dashes, strict=True):
        rmse_values = [
            np.sqrt(np.mean(e**2)) if len(e) > 0 else np.nan for e in results[method]
        ]
        ax1.plot(
            noise_levels,
            rmse_values,
            dash,
            marker="o",
            label=method,
            color=color,
            linewidth=2,
        )
    ax1.set_xlabel("Measurement Noise (m)")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("RMSE vs Measurement Noise")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Error CDF
    ax2 = axes[0, 1]
    noise_idx = 2
    for method, color, dash in zip(methods, colors, dashes, strict=True):
        errors = results[method][noise_idx]
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(sorted_errors, cdf, dash, label=method, color=color, linewidth=2)
    ax2.set_xlabel("Position Error (m)")
    ax2.set_ylabel("CDF")
    ax2.set_title(f"Error CDF (Noise = {noise_levels[noise_idx]:.2f}m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)

    # 3. Boxplot
    ax3 = axes[1, 0]
    data = [results[m][noise_idx] for m in methods if len(results[m][noise_idx]) > 0]
    labels = [m for m in methods if len(results[m][noise_idx]) > 0]
    bp = ax3.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors[: len(data)], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title(f"Error Distribution (Noise = {noise_levels[noise_idx]:.2f}m)")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Success Rate
    #
    # Three of the four methods now sit on 100 % for most of the sweep, so a
    # solid line per method would hide all but the last one drawn -- the same
    # "reads as absent" failure this file warns about elsewhere, arrived at
    # from the opposite direction. Distinct dash patterns keep every method
    # visible where they coincide, without nudging any value off its true
    # position.
    ax4 = axes[1, 1]
    total_points = 50
    for method, color, dash in zip(methods, colors, dashes, strict=True):
        rates = [len(e) / total_points * 100 for e in results[method]]
        ax4.plot(
            noise_levels,
            rates,
            dash,
            marker="o",
            label=method,
            color=color,
            linewidth=2,
        )
    ax4.set_xlabel("Measurement Noise (m)")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Convergence Success Rate")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    return fig


def main():
    """Run RF positioning comparison."""
    parser = argparse.ArgumentParser(
        description="Chapter 4: RF Positioning Methods Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python -m ch4_rf_point_positioning.example_comparison

  # Run with pre-generated dataset
  python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_square

  # Compare different beacon geometries
  python -m ch4_rf_point_positioning.example_comparison --compare-geometry

  # Compare NLOS vs baseline
  python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_nlos
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset name or path (e.g., 'ch4_rf_2d_square' or full path)",
    )
    parser.add_argument(
        "--compare-geometry",
        action="store_true",
        help="Compare positioning across different beacon geometries",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for figure (default: ch4_rf_point_positioning/figs/ch4_rf_comparison.png)",
    )

    args = parser.parse_args()

    overall_start = time.time()

    if args.compare_geometry:
        # Compare different geometries
        all_results = compare_geometries(verbose=True)

        if len(all_results) > 0:
            fig = plot_geometry_comparison(all_results)

            output_file = (
                args.output
                or "ch4_rf_point_positioning/figs/ch4_geometry_comparison.png"
            )
            paths = save_figure(fig, Path(output_file).parent, Path(output_file).stem)
            print(f"\n[OK] Figure saved: {paths[0]}")
            show_figures_if_requested()

    elif args.data:
        # Run with dataset
        data_path = resolve_data_path(args.data)
        if not data_path.exists():
            data_path = resolve_data_path(Path("data/sim") / args.data)
        if not data_path.exists():
            print(
                f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'"
            )
            print("\nAvailable datasets:")
            sim_dir = resolve_data_path(Path("data/sim"))
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch4"):
                        print(f"  - {d.name}")
            return

        results = run_with_dataset(str(data_path), verbose=True)

        output_file = (
            args.output or "ch4_rf_point_positioning/figs/ch4_rf_comparison.png"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plot_dataset_results(results, output_file)
        show_figures_if_requested()

    else:
        # Run with inline data (original behavior)
        print("\n" + "=" * 70)
        print("Chapter 4: RF Positioning Methods Comparison")
        print("=" * 70)
        print("\nTip: Run with --data ch4_rf_2d_square to use pre-generated dataset")
        print("     Run with --compare-geometry to compare beacon layouts\n")

        noise_levels, results = run_inline_comparison()

        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)

        fig = plot_inline_comparison(noise_levels, results)

        output_file = (
            args.output or "ch4_rf_point_positioning/figs/ch4_rf_comparison.png"
        )
        paths = save_figure(fig, Path(output_file).parent, Path(output_file).stem)
        print(f"[OK] Figure saved: {paths[0]}")
        show_figures_if_requested()

    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("Comparison completed successfully!")
    print(f"Total execution time: {overall_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
