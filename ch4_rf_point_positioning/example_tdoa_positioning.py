"""
TDOA Positioning Examples.

This script demonstrates TDOA positioning algorithms from Chapter 4.

Implements:
    - TDOA measurement model (Eqs. 4.27-4.33)
    - TDOA iterative LS and I-WLS (Eqs. 4.34-4.42)
    - Correlated covariance matrix (Eq. 4.42)
    - Fang's TOA closed-form (Eqs. 4.43-4.49)
    - Chan's TDOA closed-form (Eqs. 4.50-4.62)

Author: Li-Ta Hsu
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import save_figure, show_figures_if_requested
from core.rf import (
    TDOAPositioner,
    TOAPositioner,
    build_tdoa_covariance,
    tdoa_chan_solver,
    toa_fang_solver,
)
from core.rf.positioning import STALL_M

# Seed for the Monte Carlo in Demos 2 and 6.
SEED = 42


def demo_tdoa_basic():
    """Demonstrate basic TDOA positioning with iterative LS.

    `TDOAPositioner.solve` without a `covariance` uses W = I, so this is
    iterative *LS*. Demo 4 is where a covariance is supplied and the "W"
    starts meaning something -- see Eq. (4.42).
    """
    print("\n" + "=" * 70)
    print("Demo 1: Basic TDOA Positioning (iterative LS, W = I)")
    print("=" * 70)

    # Setup anchors (5 anchors in a larger area)
    anchors = np.array([[0, 0], [15, 0], [15, 15], [0, 15], [7.5, 7.5]], dtype=float)

    # True position
    true_position = np.array([5.0, 8.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Generate TDOA measurements (reference anchor = 0)
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_measurements = []
    for i in range(1, len(anchors)):
        dist_i = np.linalg.norm(true_position - anchors[i])
        tdoa = dist_i - dist_ref
        tdoa_measurements.append(tdoa)
    tdoa_measurements = np.array(tdoa_measurements)

    print(f"\nTDOA measurements (m): {tdoa_measurements}")

    # Solve with W = I (Eqs. 4.34-4.41, unweighted)
    positioner = TDOAPositioner(anchors, reference_anchor_index=0)
    estimated_position, info = positioner.solve(
        tdoa_measurements, initial_guess=np.array([7.5, 7.5])
    )

    print(f"\nEstimated position: {estimated_position}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Position error: {np.linalg.norm(estimated_position - true_position):.6f} m")

    return anchors, true_position, tdoa_measurements


def demo_tdoa_with_noise():
    """Demonstrate TDOA positioning with measurement noise.

    **The noise is drawn per anchor and then differenced.** A TDOA measurement
    is ``(d_j + e_j) - (d_0 + e_0)``: the reference anchor's error appears in
    every difference, so the differences are correlated -- Eq. (4.42), which
    ``build_tdoa_covariance`` writes as ``Var = 2 sigma^2``, ``Cov = sigma^2``,
    ``rho = 0.5``.

    This used to add ``standard_normal(len(tdoa)) * noise_std`` directly to the
    true differences, which deletes the shared term and hands TDOA information
    it cannot physically have: independent per-difference errors are what you
    would get from four independent clocks, not from one receiver. Every other
    demo in this file (3, 4, 7, 8, 9) already draws per anchor and differences
    afterwards.

    The cost was a third of the error. Over the same 300 draws per level the
    err/noise ratio -- the column this table prints precisely so it can check
    itself -- ran 0.56-0.59 under the independent model and runs 0.80-0.88
    under the correlated one. Note the self-check passes either way: the ratio
    is constant down the column under both, because both are linear in sigma.
    A statistic that only tests *proportionality* cannot see a wrong variance.
    """
    print("\n" + "=" * 70)
    print("Demo 2: TDOA Positioning with Measurement Noise")
    print("=" * 70)

    # Setup
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
    true_position = np.array([7.0, 12.0])

    # Generate noiseless TDOA
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    results = []

    print(f"\nTrue position: {true_position}")
    print(f"Testing {len(noise_levels)} noise levels...")

    # Median over repeated draws rather than one solve per noise level. A
    # single draw per row does not support the trend the table invites you to
    # read: it gave 0.046, 0.136 and 0.857 m for 0.1, 0.5 and 1.0 m of noise,
    # ratios of 3.0x and 6.3x where the noise ratios are 5x and 2x. Demos 3
    # and 4 in this same file already average; this row now matches them.
    trials = 300
    rng = np.random.default_rng(SEED)

    for noise_std in noise_levels:
        errors, n_failed = [], 0
        for _ in range(trials if noise_std > 0 else 1):
            # Per anchor, then differenced: e_j - e_0, not an independent draw
            # per difference. See this demo's docstring.
            range_noise = rng.standard_normal(len(anchors)) * noise_std
            tdoa_noisy = tdoa_true + (range_noise[1:] - range_noise[0])
            positioner = TDOAPositioner(anchors, reference_anchor_index=0)
            est_pos, info = positioner.solve(
                tdoa_noisy, initial_guess=np.array([10.0, 10.0])
            )
            if info["converged"]:
                errors.append(float(np.linalg.norm(est_pos - true_position)))
            else:
                n_failed += 1

        if errors:
            errors = np.asarray(errors)
            results.append(
                {
                    "noise": noise_std,
                    "error": float(np.median(errors)),
                    "n_failed": n_failed,
                    "n_gross": int(np.sum(errors > 100.0)),
                }
            )
        else:
            results.append(
                {
                    "noise": noise_std,
                    "error": np.inf,
                    "n_failed": n_failed,
                    "n_gross": 0,
                }
            )

    # Print results
    print("\n" + "-" * 70)
    print(f"Median position error over {trials} draws per noise level.")
    print("Noise is drawn per anchor at the stated sigma and then differenced,")
    print("so the TDOA errors are correlated (Eq. 4.42: rho = 0.5).")
    print(
        f"{'Noise (m)':<13} {'Median err (m)':<16} {'err/noise':<11} "
        f"{'no-converge':<13} {'>100 m':<8}"
    )
    print("-" * 70)
    for r in results:
        error_str = f"{r['error']:.4f}" if r["error"] != np.inf else "FAILED"
        # Constant if the relationship is linear, which is the check worth
        # running on a table like this.
        slope = (
            f"{r['error'] / r['noise']:.4f}"
            if r["noise"] > 0 and r["error"] != np.inf
            else "-"
        )
        print(
            f"{r['noise']:<13.2f} {error_str:<16} {slope:<11} "
            f"{r['n_failed']:<13d} {r['n_gross']:<8d}"
        )

    return results


def demo_correlated_covariance():
    """
    Demonstrate the impact of correlated covariance for TDOA measurements.

    Implements Eq. (4.42) from Chapter 4:
        The TDOA covariance matrix has off-diagonal terms due to the shared
        reference anchor noise.

    This demo compares:
        - Identity weighting (ignoring correlation)
        - Correlated weighting (proper covariance modeling per Eq. 4.42)
    """
    print("\n" + "=" * 70)
    print("Demo 3: Correlated vs Identity Weighting (Eq. 4.42)")
    print("=" * 70)

    # Setup: 5 anchors with heterogeneous noise levels
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)

    # True position
    true_position = np.array([7.0, 12.0])
    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Per-anchor range noise standard deviations (meters)
    # Reference anchor (idx=0) has larger noise to emphasize correlation effect
    sigmas = np.array([0.5, 0.1, 0.15, 0.12, 0.08])
    print("\nPer-anchor range noise (sigma, meters):")
    print(f"  Reference (anchor 0): {sigmas[0]:.2f} m (higher noise)")
    print(f"  Other anchors: {sigmas[1:]}")

    # Generate noiseless TDOA measurements
    dist_ref = np.linalg.norm(true_position - anchors[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(true_position - anchors[i]) - dist_ref
            for i in range(1, len(anchors))
        ]
    )

    print(f"\nTrue TDOA measurements: {tdoa_true}")

    # Build correlated covariance matrix (Eq. 4.42)
    cov_correlated = build_tdoa_covariance(sigmas, reference_anchor_index=0)
    print("\nCorrelated covariance matrix (Eq. 4.42):")
    print(cov_correlated)
    print("\nDiagonal (var): sigma_k^2 + sigma_ref^2")
    print(f"Off-diagonal (cov): sigma_ref^2 = {sigmas[0]**2:.4f}")

    # Build identity covariance (ignoring correlation)
    cov_identity = np.eye(len(tdoa_true))
    print("\nIdentity covariance matrix:")
    print(cov_identity)

    # Run Monte Carlo simulation
    n_trials = 500
    np.random.seed(42)

    errors_identity = []
    errors_correlated = []

    print(f"\nRunning {n_trials} Monte Carlo trials...")

    for _trial in range(n_trials):
        # Generate noisy range measurements
        # Range noise for each anchor
        range_noise = np.random.randn(len(anchors)) * sigmas

        # Compute noisy ranges
        noisy_ranges = np.array(
            [
                np.linalg.norm(true_position - anchors[i]) + range_noise[i]
                for i in range(len(anchors))
            ]
        )

        # Compute noisy TDOA (range differences relative to reference)
        tdoa_noisy = np.array(
            [noisy_ranges[i] - noisy_ranges[0] for i in range(1, len(anchors))]
        )

        # Solve with identity weighting
        positioner = TDOAPositioner(anchors, reference_anchor_index=0)
        try:
            est_identity, info_id = positioner.solve(
                tdoa_noisy,
                initial_guess=np.array([10.0, 10.0]),
                covariance=cov_identity,
            )
            if info_id["converged"]:
                errors_identity.append(np.linalg.norm(est_identity - true_position))
        except Exception:
            pass

        # Solve with correlated weighting
        try:
            est_correlated, info_corr = positioner.solve(
                tdoa_noisy,
                initial_guess=np.array([10.0, 10.0]),
                covariance=cov_correlated,
            )
            if info_corr["converged"]:
                errors_correlated.append(np.linalg.norm(est_correlated - true_position))
        except Exception:
            pass

    # Compute statistics
    errors_identity = np.array(errors_identity)
    errors_correlated = np.array(errors_correlated)

    print("\n" + "-" * 70)
    print("Results Summary:")
    print("-" * 70)
    print(f"{'Weighting':<20} {'RMSE (m)':<15} {'Mean (m)':<15} {'Std (m)':<15}")
    print("-" * 70)

    rmse_id = np.sqrt(np.mean(errors_identity**2))
    rmse_corr = np.sqrt(np.mean(errors_correlated**2))

    print(
        f"{'Identity':<20} {rmse_id:<15.4f} "
        f"{np.mean(errors_identity):<15.4f} {np.std(errors_identity):<15.4f}"
    )
    print(
        f"{'Correlated (Eq.4.42)':<20} {rmse_corr:<15.4f} "
        f"{np.mean(errors_correlated):<15.4f} {np.std(errors_correlated):<15.4f}"
    )

    improvement = (rmse_id - rmse_corr) / rmse_id * 100
    print(f"\nRMSE improvement with correlated weighting: {improvement:.1f}%")

    print("\nKey Insight (from Eq. 4.42):")
    print("  - When reference anchor has larger noise (sigma_ref = 0.5 m),")
    print("    the off-diagonal correlation terms are significant (0.25).")
    print("  - Ignoring this correlation leads to suboptimal position estimates.")
    print("  - Proper covariance modeling per Eq. 4.42 improves accuracy.")

    return errors_identity, errors_correlated


def demo_covariance_sensitivity():
    """
    Demonstrate sensitivity of positioning accuracy to reference anchor noise.

    Shows how the correlated covariance structure matters more when:
        - Reference anchor has higher noise relative to other anchors
        - Off-diagonal terms become dominant
    """
    print("\n" + "=" * 70)
    print("Demo 4: Sensitivity to Reference Anchor Noise")
    print("=" * 70)

    # Setup: 4 anchors
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    true_position = np.array([8.0, 12.0])

    print(f"\nTrue position: {true_position}")
    print("Other anchor noise (fixed): sigma = 0.1 m")

    # Test different reference anchor noise levels
    ref_noise_levels = [0.05, 0.1, 0.2, 0.5, 1.0]
    n_trials = 200
    np.random.seed(123)

    results = []

    for ref_sigma in ref_noise_levels:
        sigmas = np.array([ref_sigma, 0.1, 0.1, 0.1])

        # Build covariance matrices
        cov_corr = build_tdoa_covariance(sigmas, reference_anchor_index=0)
        cov_id = np.eye(len(anchors) - 1)

        errors_id = []
        errors_corr = []

        for _ in range(n_trials):
            # Generate noisy ranges
            range_noise = np.random.randn(len(anchors)) * sigmas
            noisy_ranges = np.array(
                [
                    np.linalg.norm(true_position - anchors[i]) + range_noise[i]
                    for i in range(len(anchors))
                ]
            )

            # Compute noisy TDOA
            tdoa_noisy = np.array(
                [noisy_ranges[i] - noisy_ranges[0] for i in range(1, len(anchors))]
            )

            positioner = TDOAPositioner(anchors, reference_anchor_index=0)

            # Identity weighting
            try:
                est_id, info = positioner.solve(
                    tdoa_noisy,
                    initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_id,
                )
                if info["converged"]:
                    errors_id.append(np.linalg.norm(est_id - true_position))
            except Exception:
                pass

            # Correlated weighting
            try:
                est_corr, info = positioner.solve(
                    tdoa_noisy,
                    initial_guess=np.array([10.0, 10.0]),
                    covariance=cov_corr,
                )
                if info["converged"]:
                    errors_corr.append(np.linalg.norm(est_corr - true_position))
            except Exception:
                pass

        rmse_id = np.sqrt(np.mean(np.array(errors_id) ** 2))
        rmse_corr = np.sqrt(np.mean(np.array(errors_corr) ** 2))
        improvement = (rmse_id - rmse_corr) / rmse_id * 100 if rmse_id > 0 else 0

        results.append(
            {
                "ref_sigma": ref_sigma,
                "rmse_identity": rmse_id,
                "rmse_correlated": rmse_corr,
                "improvement": improvement,
            }
        )

    # Print results
    print("\n" + "-" * 70)
    print(
        f"{'Ref sigma (m)':<15} {'RMSE Id (m)':<15} "
        f"{'RMSE Corr (m)':<15} {'Improvement':<15}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['ref_sigma']:<15.2f} {r['rmse_identity']:<15.4f} "
            f"{r['rmse_correlated']:<15.4f} {r['improvement']:<15.1f}%"
        )

    print("\nKey Insight:")
    print("  - When reference noise >> other anchor noise, improvement is larger")
    print("  - At sigma_ref = sigma_other, correlation still matters (7.9% here)")
    print("  - The first row is not evidence against that. When the reference")
    print("    anchor is the QUIETEST one there is almost nothing to weight:")
    print("    the true gain there is under 1%, and a ratio of two RMSEs over")
    print("    200 draws carries a standard error of roughly 5%, so a number")
    print("    that small is Monte-Carlo noise rather than a measurement. The")
    print("    same row over eight other seeds runs +0.15% to +0.94%.")
    print("  - So: modelling Eq. (4.42) never costs you, and pays in proportion")
    print("    to how much noisier the reference anchor is than the rest.")

    return results


def demo_visualize_covariance():
    """Visualize the correlated covariance matrix structure."""
    print("\n" + "=" * 70)
    print("Demo 5: Visualizing Covariance Matrix Structure (Eq. 4.42)")
    print("=" * 70)

    # Example with 5 anchors
    sigmas = np.array([0.3, 0.1, 0.15, 0.2, 0.12])
    reference_anchor_index = 0

    print(f"\nPer-anchor sigmas: {sigmas}")
    print(f"Reference anchor index: {reference_anchor_index}")
    print(f"Reference sigma: {sigmas[reference_anchor_index]:.2f} m")

    cov = build_tdoa_covariance(sigmas, reference_anchor_index=reference_anchor_index)

    print("\nCovariance Matrix (4x4 for 4 TDOA measurements):")
    print("-" * 50)
    print("Eq. 4.42 structure:")
    print("  Diagonal[i,i] = sigma_i^2 + sigma_ref^2")
    print("  Off-diag[i,j] = sigma_ref^2 (shared reference noise)")
    print("-" * 50)

    # Print matrix with labels
    non_ref = [i for i in range(len(sigmas)) if i != reference_anchor_index]
    header = "        " + "".join(
        [f"d^{i},{reference_anchor_index}     " for i in non_ref]
    )
    print(header)

    for i, row_idx in enumerate(non_ref):
        row_str = f"d^{row_idx},{reference_anchor_index}  "
        for j in range(len(cov)):
            row_str += f"{cov[i, j]:.4f}    "
        print(row_str)

    print(
        "\nOff-diagonal value (sigma_ref^2): "
        f"{sigmas[reference_anchor_index]**2:.4f}"
    )

    # Show diagonal derivation
    print("\nDiagonal derivation:")
    for anc_idx in non_ref:
        diag_val = sigmas[anc_idx] ** 2 + sigmas[reference_anchor_index] ** 2
        print(
            f"  var(d^{anc_idx},{reference_anchor_index}) = "
            f"{sigmas[anc_idx]:.2f}^2 + "
            f"{sigmas[reference_anchor_index]:.2f}^2 = {diag_val:.4f}"
        )

    # Create figure
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cov, cmap="Blues")
        ax.set_title("TDOA Covariance Matrix (Eq. 4.42)", fontsize=12)
        ax.set_xlabel("TDOA measurement index")
        ax.set_ylabel("TDOA measurement index")

        # One tick per cell, named for the difference it holds. Matplotlib's
        # default locator put ticks at 0.5, 1.5, ... on a 4x4 image -- half of
        # them between cells, labelling nothing, on a figure whose whole
        # subject is which cell holds which covariance.
        tick_labels = [f"d{j},{reference_anchor_index}" for j in non_ref]
        ax.set_xticks(range(len(cov)), tick_labels)
        ax.set_yticks(range(len(cov)), tick_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Covariance (m^2)")

        # Add annotations
        for i in range(len(cov)):
            for j in range(len(cov)):
                ax.text(
                    j,
                    i,
                    f"{cov[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

        plt.tight_layout()
        paths = save_figure(
            fig, Path(__file__).parent / "figs", "tdoa_covariance_matrix"
        )
        print(f"\nFigure saved to: {paths[0]}")
        plt.close()
    except Exception as e:
        print(f"\nCould not save figure: {e}")

    return cov


#: Iteration budget for the iterative arms of Demo 9. The library default of 10
#: leaves RW-LS short of its step tolerance on 41 of 500 draws without changing
#: the answers, so a demo that reports the flag needs the flag to mean
#: something. Measured: at 10, 20 and 50 iterations the RMSE over all 500 draws
#: is 0.3384 m unchanged, while the count reaching tol goes 459, 499, 500.
ITERATIVE_MAX_ITERS = 50

#: Draws per geometry in Demo 6. One draw per row was what this demo used to
#: report, to four decimal places.
GEOMETRY_TRIALS = 500

#: Where Demo 6 starts its solves, as an offset from the anchor centroid. It has
#: to be an offset: the square array's centroid *is* the target, so seeding at
#: the centroid alone would be seeding at the truth for one of the two
#: geometries and not the other -- the comparison silently answering itself.
SEED_OFFSET = np.array([2.0, 2.0])


def _tdoa_geometry_trial(anchors, true_position, seed_point, noise_std, rng, trials):
    """Solve ``trials`` TDOA fixes from one seed, keeping every outcome.

    Returns ``(errors, n_stalled, n_no_converge)``. Nothing is filtered: a
    stall is reported as a stall rather than dropped, because the distance it
    scores is a property of the seed, and averaging it in silently is how a
    solver that never moved comes to look like one with a 5 m error.
    """
    distances = np.linalg.norm(true_position - anchors, axis=1)
    tdoa_true = distances[1:] - distances[0]

    errors, stalled, no_converge = [], 0, 0
    for _ in range(trials):
        range_noise = rng.standard_normal(len(anchors)) * noise_std
        tdoa_noisy = tdoa_true + (range_noise[1:] - range_noise[0])
        positioner = TDOAPositioner(anchors, reference_anchor_index=0)
        est, info = positioner.solve(tdoa_noisy, initial_guess=seed_point)
        errors.append(float(np.linalg.norm(est - true_position)))
        stalled += bool(np.linalg.norm(est - seed_point) < STALL_M)
        no_converge += not info["converged"]
    return np.asarray(errors), stalled, no_converge


def demo_geometry_effect():
    """Demonstrate the effect of anchor geometry on TDOA accuracy.

    **Neither solve starts at the answer, and neither reports one draw.** Both
    used to be seeded with ``initial_guess=[5, 5]``, which is the target
    exactly, and each printed a single realisation to four decimals. Seeding a
    solver with the ground truth converts a fragility into an accuracy; it is
    the rule in ``.cursor/rules/030-figures-and-claims.mdc``, and this demo was
    an instance of it. The collinear array printed 0.9702 m that way; from a
    seed it has to travel from, the median over 500 draws is 1.27 m.

    The seed is the anchor centroid plus ``SEED_OFFSET``, and the offset is
    load-bearing. Seeded *on* the beacon line the collinear array cannot move
    at all: every anchor lies in the same direction from a point on the line,
    the Jacobian loses rank, and all 500 draws stall at the seed and score the
    seed-to-truth distance of 5.3852 m -- one number, identical across draws,
    which is the signature ``030`` names. That case is printed here rather than
    quietly avoided, because the bare centroid is the seed a reader reaches for
    first and 5.3852 m is not an error.

    The noise is drawn per anchor and differenced, as everywhere else in this
    file; see :func:`demo_tdoa_with_noise`.
    """
    print("\n" + "=" * 70)
    print("Demo 6: Effect of Anchor Geometry on TDOA Accuracy")
    print("=" * 70)

    true_position = np.array([5.0, 5.0])

    # Good geometry: anchors surrounding the target
    good_anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    # Poor geometry: anchors on one side
    poor_anchors = np.array([[0, 0], [2, 0], [4, 0], [6, 0]], dtype=float)

    noise_std = 0.2  # meters

    print(f"\nTrue position: {true_position}")
    print(f"Noise std: {noise_std} m (per anchor, then differenced)")
    print(f"Draws per geometry: {GEOMETRY_TRIALS}")
    offset_str = ", ".join(f"{v:g}" for v in SEED_OFFSET)
    print(
        f"Seed: anchor centroid + ({offset_str}) m -- not the truth, "
        "which both solves used to be handed"
    )

    print("\n" + "-" * 78)
    print("Geometry Comparison:")
    print("-" * 78)
    print(
        f"{'Geometry':<26} {'Seed':<14} {'Median err (m)':<16} "
        f"{'stalled':<9} {'no-conv':<9}"
    )
    print("-" * 78)

    medians = {}
    for name, anchors in (
        ("Good (surrounding)", good_anchors),
        ("Poor (collinear)", poor_anchors),
    ):
        seed = anchors.mean(axis=0) + SEED_OFFSET
        errors, stalled, no_converge = _tdoa_geometry_trial(
            anchors,
            true_position,
            seed,
            noise_std,
            np.random.default_rng(SEED),
            GEOMETRY_TRIALS,
        )
        medians[name] = float(np.median(errors))
        print(
            f"{name:<26} {str(np.round(seed, 1)):<14} {medians[name]:<16.4f} "
            f"{stalled:<9d} {no_converge:<9d}"
        )

    # The degenerate seed, shown rather than avoided.
    centroid = poor_anchors.mean(axis=0)
    errors, stalled, no_converge = _tdoa_geometry_trial(
        poor_anchors,
        true_position,
        centroid,
        noise_std,
        np.random.default_rng(SEED),
        GEOMETRY_TRIALS,
    )
    print(
        f"{'Poor, seeded ON the line':<26} {str(np.round(centroid, 1)):<14} "
        f"{np.median(errors):<16.4f} {stalled:<9d} {no_converge:<9d}"
    )

    good, poor = medians["Good (surrounding)"], medians["Poor (collinear)"]
    print(
        f"\nThe collinear array is {poor / good:.1f}x worse than the "
        f"surrounding one ({good:.4f} -> {poor:.4f} m), and the"
    )
    print(
        f"surrounding array's median is {good / noise_std:.2f}x the range noise"
        " -- the same err/noise ratio Demo 2"
    )
    print("reports for its own well-conditioned array.")
    print("\nThe third row is not an error of 5.3852 m. It is the distance from")
    print("the seed to the target, scored 500 times because the solve never")
    print("moved: on the beacon line every anchor lies in the same direction,")
    print("the Jacobian is rank-deficient, and Gauss-Newton has nowhere to")
    print("step. An identical number across every draw is the tell, and it is")
    print("a property of the seed rather than of the measurements.")


def demo_fang_toa_solver():
    """
    Demonstrate Fang's closed-form TOA positioning (Eqs. 4.43-4.49).

    Compares:
        - Fang's closed-form solution (no iteration needed)
        - the range-weighted iterative solution (requires an initial guess)
    """
    print("\n" + "=" * 70)
    print("Demo 7: Fang's TOA Closed-Form vs Range-Weighted LS " "(Eqs. 4.43-4.49)")
    print("=" * 70)

    # Setup: 4 anchors in a square
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    true_position = np.array([7.0, 12.0])

    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")

    # Compute true ranges
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)
    print(f"True ranges: {ranges_true}")

    # Test with perfect measurements
    print("\n--- Perfect Measurements ---")

    # Fang's closed-form
    fang_pos, fang_info = toa_fang_solver(anchors, ranges_true)
    fang_error = np.linalg.norm(fang_pos - true_position)

    # Range-weighted iterative (W_ii = 1/d_i^2)
    positioner = TOAPositioner(anchors, method="range_weighted")
    rw_pos, rw_info = positioner.solve(
        ranges_true, initial_guess=np.array([10.0, 10.0])
    )
    rw_error = np.linalg.norm(rw_pos - true_position)

    print(f"Fang:  position={fang_pos}, error={fang_error:.6f} m")
    print(
        f"RW-LS: position={rw_pos}, error={rw_error:.6f} m, "
        f"iters={rw_info['iterations']}"
    )

    # Test with noisy measurements
    print("\n--- Noisy Measurements (Monte Carlo) ---")
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    n_trials = 200
    np.random.seed(42)

    results = []
    for noise_std in noise_levels:
        fang_errors = []
        rw_errors = []

        for _ in range(n_trials):
            ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std

            # Fang's method (no initial guess needed)
            try:
                f_pos, _ = toa_fang_solver(anchors, ranges_noisy)
                fang_errors.append(np.linalg.norm(f_pos - true_position))
            except Exception:
                pass

            # Range-weighted iterative (requires an initial guess)
            try:
                i_pos, i_info = positioner.solve(
                    ranges_noisy, initial_guess=np.array([10.0, 10.0])
                )
                if i_info["converged"]:
                    rw_errors.append(np.linalg.norm(i_pos - true_position))
            except Exception:
                pass

        fang_rmse = np.sqrt(np.mean(np.array(fang_errors) ** 2))
        rw_rmse = np.sqrt(np.mean(np.array(rw_errors) ** 2))

        results.append(
            {
                "noise": noise_std,
                "fang_rmse": fang_rmse,
                "rw_rmse": rw_rmse,
                "fang_success": len(fang_errors),
                "rw_success": len(rw_errors),
            }
        )

    print(
        f"\n{'Noise (m)':<12} {'Fang RMSE':<15} {'RW-LS RMSE':<15} "
        f"{'Fang Success':<15} {'RW-LS Success':<15}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['noise']:<12.2f} {r['fang_rmse']:<15.4f} {r['rw_rmse']:<15.4f} "
            f"{r['fang_success']:<15} {r['rw_success']:<15}"
        )

    print("\nKey Insights:")
    print("  - Fang's method is non-iterative (no initial guess required)")
    print("  - Range weighting is a heuristic (W_ii = 1/d_i^2), not a book method")
    print("  - Both methods sensitive to noise and geometry (GDOP)")

    return results


def demo_chan_tdoa_solver():
    """
    Demonstrate Chan's closed-form TDOA positioning (Eqs. 4.50-4.62).

    Compares:
        - Chan's closed-form solution (no iteration needed)
        - the iterative TDOA solution: W = I here, W = Sigma^-1 under noise
    """
    print("\n" + "=" * 70)
    print("Demo 8: Chan's TDOA Closed-Form vs Iterative LS/WLS " "(Eqs. 4.50-4.62)")
    print("=" * 70)

    # Setup: 5 anchors for good geometry
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
    true_position = np.array([8.0, 12.0])
    reference_anchor_index = 0

    print(f"\nTrue position: {true_position}")
    print(f"Number of anchors: {len(anchors)}")
    print(f"Reference anchor: {reference_anchor_index}")

    # Compute true ranges and TDOA
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)
    d_ref = ranges_true[reference_anchor_index]
    tdoa_true = np.array(
        [
            ranges_true[i] - d_ref
            for i in range(len(anchors))
            if i != reference_anchor_index
        ]
    )

    print(f"True reference distance: {d_ref:.4f} m")
    print(f"True TDOA measurements: {tdoa_true}")

    # Test with perfect measurements
    print("\n--- Perfect Measurements ---")

    # Chan's closed-form
    chan_pos, chan_info = tdoa_chan_solver(
        anchors, tdoa_true, reference_anchor_index=reference_anchor_index
    )
    chan_error = np.linalg.norm(chan_pos - true_position)

    # Iterative LS: no covariance here, so W = I
    positioner = TDOAPositioner(anchors, reference_anchor_index=reference_anchor_index)
    ils_pos, ils_info = positioner.solve(
        tdoa_true, initial_guess=np.array([10.0, 10.0])
    )
    ils_error = np.linalg.norm(ils_pos - true_position)

    print(f"Chan:  position={chan_pos}, error={chan_error:.6f} m")
    print(f"       reference distance estimate={chan_info['reference_distance']:.4f} m")
    print(
        f"I-LS:  position={ils_pos}, error={ils_error:.6f} m, "
        f"iters={ils_info['iterations']}"
    )

    # Test with noisy measurements (correlated noise)
    print("\n--- Noisy Measurements (Monte Carlo with Correlated Noise) ---")
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    n_trials = 200
    np.random.seed(42)

    # Per-anchor noise (uniform for simplicity)
    results = []
    for noise_std in noise_levels:
        sigmas = np.ones(len(anchors)) * noise_std
        cov = build_tdoa_covariance(
            sigmas, reference_anchor_index=reference_anchor_index
        )

        chan_errors = []
        iwls_errors = []

        for _ in range(n_trials):
            # Generate noisy ranges
            ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std

            # Compute noisy TDOA
            tdoa_noisy = np.array(
                [
                    ranges_noisy[i] - ranges_noisy[reference_anchor_index]
                    for i in range(len(anchors))
                    if i != reference_anchor_index
                ]
            )

            # Chan's method (with WLS using covariance)
            try:
                c_pos, _ = tdoa_chan_solver(
                    anchors,
                    tdoa_noisy,
                    covariance=cov,
                    reference_anchor_index=reference_anchor_index,
                )
                chan_errors.append(np.linalg.norm(c_pos - true_position))
            except Exception:
                pass

            # I-WLS (with covariance)
            try:
                i_pos, i_info = positioner.solve(
                    tdoa_noisy,
                    initial_guess=np.array([10.0, 10.0]),
                    covariance=cov,
                )
                if i_info["converged"]:
                    iwls_errors.append(np.linalg.norm(i_pos - true_position))
            except Exception:
                pass

        chan_rmse = (
            np.sqrt(np.mean(np.array(chan_errors) ** 2)) if chan_errors else np.inf
        )
        iwls_rmse = (
            np.sqrt(np.mean(np.array(iwls_errors) ** 2)) if iwls_errors else np.inf
        )

        results.append(
            {
                "noise": noise_std,
                "chan_rmse": chan_rmse,
                "iwls_rmse": iwls_rmse,
                "chan_success": len(chan_errors),
                "iwls_success": len(iwls_errors),
            }
        )

    print(
        f"\n{'Noise (m)':<12} {'Chan RMSE':<15} {'I-WLS RMSE':<15} "
        f"{'Chan Success':<15} {'I-WLS Success':<15}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['noise']:<12.2f} {r['chan_rmse']:<15.4f} {r['iwls_rmse']:<15.4f} "
            f"{r['chan_success']:<15} {r['iwls_success']:<15}"
        )

    print("\nKey Insights:")
    print("  - Chan's method is non-iterative, estimates position + ref distance")
    print("  - Chan's WLS step uses correlated covariance (Eq. 4.62)")
    print("  - I-WLS requires an initial guess but iterates to a better solution")
    print("  - Both methods benefit from proper covariance modeling")

    return results


def demo_closed_form_comparison():
    """
    Comprehensive comparison of closed-form and iterative solvers.

    Compares:
        - TOA: Fang vs range-weighted LS
        - TDOA: Chan vs I-WLS (a covariance is supplied, Eq. 4.42)
    """
    print("\n" + "=" * 70)
    print("Demo 9: Comprehensive Closed-Form vs Iterative Comparison")
    print("=" * 70)

    # Setup
    anchors = np.array([[0, 0], [20, 0], [20, 20], [0, 20], [10, 10]], dtype=float)
    true_position = np.array([7.5, 11.0])
    reference_anchor_index = 0

    print(f"\nTrue position: {true_position}")
    print(f"Anchors: {len(anchors)} beacons")

    # Compute true measurements
    ranges_true = np.linalg.norm(anchors - true_position, axis=1)

    # Test parameters
    noise_std = 0.3
    n_trials = 500
    np.random.seed(123)

    # Results storage. Every trial is scored for every method: the two
    # iterative arms used to keep only the trials whose step fell below `tol`
    # inside the budget, which put RW-LS's 459 draws beside three columns of
    # 500 and understated its RMSE as 0.3163 against 0.3384 over all of them.
    # Nothing had failed -- the flag is a step-tolerance stop, and the answer
    # at iteration 10 and at iteration 50 agrees to four decimals.
    toa_fang_err = []
    toa_rw_err = []
    tdoa_chan_err = []
    tdoa_iwls_err = []
    toa_rw_reached_tol = 0
    tdoa_iwls_reached_tol = 0

    toa_positioner = TOAPositioner(anchors, method="range_weighted")
    tdoa_positioner = TDOAPositioner(
        anchors, reference_anchor_index=reference_anchor_index
    )
    sigmas = np.ones(len(anchors)) * noise_std
    cov = build_tdoa_covariance(sigmas, reference_anchor_index=reference_anchor_index)

    print(f"\nRunning {n_trials} Monte Carlo trials (noise_std={noise_std} m)...")

    for _ in range(n_trials):
        # Generate noisy ranges
        ranges_noisy = ranges_true + np.random.randn(len(anchors)) * noise_std
        tdoa_noisy = np.array(
            [
                ranges_noisy[i] - ranges_noisy[reference_anchor_index]
                for i in range(len(anchors))
                if i != reference_anchor_index
            ]
        )

        # TOA: Fang
        try:
            pos, _ = toa_fang_solver(anchors, ranges_noisy)
            toa_fang_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

        # TOA: range-weighted iterative
        try:
            pos, info = toa_positioner.solve(
                ranges_noisy,
                initial_guess=np.array([10.0, 10.0]),
                max_iters=ITERATIVE_MAX_ITERS,
            )
            toa_rw_err.append(np.linalg.norm(pos - true_position))
            toa_rw_reached_tol += bool(info["converged"])
        except Exception:
            pass

        # TDOA: Chan
        try:
            pos, _ = tdoa_chan_solver(
                anchors,
                tdoa_noisy,
                covariance=cov,
                reference_anchor_index=reference_anchor_index,
            )
            tdoa_chan_err.append(np.linalg.norm(pos - true_position))
        except Exception:
            pass

        # TDOA: I-WLS
        try:
            pos, info = tdoa_positioner.solve(
                tdoa_noisy,
                initial_guess=np.array([10.0, 10.0]),
                covariance=cov,
                max_iters=ITERATIVE_MAX_ITERS,
            )
            tdoa_iwls_err.append(np.linalg.norm(pos - true_position))
            tdoa_iwls_reached_tol += bool(info["converged"])
        except Exception:
            pass

    # Compute statistics
    def stats(errors):
        if len(errors) == 0:
            return np.inf, np.inf, np.inf, 0
        e = np.array(errors)
        return np.sqrt(np.mean(e**2)), np.mean(e), np.std(e), len(e)

    print("\n" + "-" * 92)
    print(
        f"{'Method':<25} {'RMSE (m)':<12} {'Mean (m)':<12} {'Std (m)':<12} "
        f"{'Scored':<10} {'Step tol reached':<18}"
    )
    print("-" * 92)

    for label, errors, reached in (
        ("TOA Fang (closed-form)", toa_fang_err, None),
        ("TOA RW-LS (iterative)", toa_rw_err, toa_rw_reached_tol),
        ("TDOA Chan (closed-form)", tdoa_chan_err, None),
        ("TDOA I-WLS (iterative)", tdoa_iwls_err, tdoa_iwls_reached_tol),
    ):
        rmse, mean, std, n = stats(errors)
        reached_str = "n/a (direct)" if reached is None else f"{reached}/{n_trials}"
        print(
            f"{label:<25} {rmse:<12.4f} {mean:<12.4f} {std:<12.4f} "
            f"{n:<10} {reached_str:<18}"
        )

    print("\nSummary:")
    print("  - Closed-form methods (Fang, Chan) don't need initial guess")
    print("  - Iterative methods can refine estimates from an initial guess")
    print("  - TOA methods require range measurements; TDOA uses range differences")
    print("  - TDOA eliminates need for clock synchronization between agent & beacons")
    print("  - All methods benefit from good geometry (low GDOP)")
    print("  - 'Scored' is 500 for every method: the RMSEs are over the same")
    print("    500 draws, which is what makes the four columns comparable.")
    print("    'Step tol reached' is how often the iteration's step fell below")
    print("    1e-6 m inside its budget -- a stopping report, not a success")
    print("    rate. Filtering on it used to score RW-LS over 459 draws and")
    print("    quote 0.3163 m where all 500 give 0.3384 m.")

    # Create comparison figure
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # TOA comparison
        ax1 = axes[0]
        data = [toa_fang_err, toa_rw_err]
        tick_labels = ["Fang\n(closed-form)", "RW-LS\n(iterative)"]
        bp = ax1.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightgreen")
        ax1.set_ylabel("Position Error (m)")
        ax1.set_title("TOA Positioning Comparison")
        ax1.grid(True, alpha=0.3)

        # TDOA comparison
        ax2 = axes[1]
        data = [tdoa_chan_err, tdoa_iwls_err]
        tick_labels = ["Chan\n(closed-form)", "I-WLS\n(iterative)"]
        bp = ax2.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightyellow")
        bp["boxes"][1].set_facecolor("lightcoral")
        ax2.set_ylabel("Position Error (m)")
        ax2.set_title("TDOA Positioning Comparison")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Closed-Form vs Iterative Solvers (noise={noise_std}m)", fontsize=12
        )
        plt.tight_layout()
        paths = save_figure(
            fig, Path(__file__).parent / "figs", "closed_form_comparison"
        )
        print(f"\nFigure saved: {paths[0]}")
        plt.close()
    except Exception as e:
        print(f"\nCould not save figure: {e}")

    return {
        "toa_fang": toa_fang_err,
        "toa_rw": toa_rw_err,
        "tdoa_chan": tdoa_chan_err,
        "tdoa_iwls": tdoa_iwls_err,
    }


def main():
    """Run all TDOA positioning examples."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    print("\n" + "=" * 70)
    print("Chapter 4: TDOA Positioning Examples")
    print("=" * 70)

    # Run demos
    demo_tdoa_basic()
    demo_tdoa_with_noise()
    demo_correlated_covariance()  # Demo 3: Correlated vs Identity weighting
    demo_covariance_sensitivity()  # Demo 4: Sensitivity analysis
    demo_visualize_covariance()  # Demo 5: Visualize covariance structure
    demo_geometry_effect()  # Demo 6
    demo_fang_toa_solver()  # Demo 7: Fang's TOA closed-form
    demo_chan_tdoa_solver()  # Demo 8: Chan's TDOA closed-form
    demo_closed_form_comparison()  # Demo 9: Comprehensive comparison

    print("\n" + "=" * 70)
    print("All TDOA examples completed successfully!")
    print("=" * 70)
    show_figures_if_requested()


if __name__ == "__main__":
    main()
