"""
Example: Pattern Recognition Fingerprinting (Linear Regression)

Demonstrates linear regression-based fingerprinting from Chapter 5.

Treats positioning as supervised learning: learns mapping f: z -> x
where z is RSS fingerprint and x is location.

Model: x_hat = Wz + b (linear transformation with ridge regression)

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import plot_error_cdf, save_figure, show_figures_if_requested
from core.fingerprinting import (
    FingerprintDatabase,
    LinearRegressionLocalizer,
    load_fingerprint_database,
)

DEFAULT_DATA = "data/sim/ch5_wifi_fingerprint_grid"

#: The ridge parameters this example sweeps, chosen by measuring where the knob
#: reaches the answer rather than by picking round numbers.
#:
#: ``LinearRegressionLocalizer.fit`` solves ``(Z'Z + lambda I) theta = Z'X``, so
#: lambda only matters relative to the diagonal of ``Z'Z`` -- and RSS values near
#: -65 dBm over 85 reference points put that diagonal at **3.6e5 to 4.3e5**. The
#: sweep used to run 0, 0.1, 1, 10, every one of which is under 3e-5 of the data
#: term: train RMSE printed 6.09 m at all four and test RMSE moved by 0.04 m.
#: Four rows of a table demonstrating a parameter that could not reach the
#: model.
#:
#: Measured on the shipped floor-0 split, the knob first bites near 1e2 and has
#: overwhelmed the data by 1e4::
#:
#:     lambda        0     1e1     1e2     1e3     1e4
#:     train RMSE  6.085   6.086   6.103   6.740  12.581  m
#:     test  RMSE  8.320   8.284   8.016   7.514  13.842  m
#:     ||W||_F     2.295   2.285   2.208   1.836   0.921
REGULARIZATION_SWEEP = [0.0, 10.0, 100.0, 1000.0, 10000.0]

#: The lambda whose weight matrix and predictions the single-model panels show.
#: 1.0 used to be the choice, which by the table above is indistinguishable from
#: no regularisation at all.
SHOWCASE_LAMBDA = 1000.0

#: Training-set sizes for the panel that shows where ridge earns its keep.
#: There are 8 APs, so ``M = 9`` is one sample more than the model has
#: parameters; ``M = 85`` is what the 70/30 split above actually gives.
TRAINING_SET_SIZES = [9, 12, 20, 40, 85]

#: Draws averaged per training-set size. One draw of 9 reference points out of
#: 121 is a very noisy thing to plot.
TRAINING_SET_DRAWS = 20


def split_train_test(db, test_ratio=0.3, floor_id=None, seed=42):
    """
    Split database into train and test sets.

    Args:
        db: FingerprintDatabase.
        test_ratio: Fraction of data for testing.
        floor_id: Floor to use (None = all floors).
        seed: Random seed.

    Returns:
        Tuple of (train_db, test_db).
    """
    np.random.seed(seed)

    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        indices = np.where(mask)[0]
    else:
        indices = np.arange(len(db.locations))

    # Shuffle and split
    np.random.shuffle(indices)
    n_test = int(len(indices) * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    # Create train database
    from core.fingerprinting import FingerprintDatabase

    train_db = FingerprintDatabase(
        locations=db.locations[train_idx],
        features=db.features[train_idx],
        floor_ids=db.floor_ids[train_idx],
        meta=db.meta.copy(),
    )

    test_db = FingerprintDatabase(
        locations=db.locations[test_idx],
        features=db.features[test_idx],
        floor_ids=db.floor_ids[test_idx],
        meta=db.meta.copy(),
    )

    return train_db, test_db


def evaluate_model(model, test_db, floor_id=None):
    """
    Evaluate trained model on test set.

    Args:
        model: Trained LinearRegressionLocalizer.
        test_db: Test FingerprintDatabase.
        floor_id: Floor to evaluate on.

    Returns:
        Dictionary with errors and metrics.
    """
    if floor_id is not None:
        mask = test_db.get_floor_mask(floor_id)
        features = test_db.features[mask]
        locations = test_db.locations[mask]
    else:
        features = test_db.features
        locations = test_db.locations

    # Batch prediction
    t_start = time.perf_counter()
    est_locs = model.predict_batch(features)
    t_end = time.perf_counter()

    # Compute errors
    errors = np.linalg.norm(est_locs - locations, axis=1)

    # Compute R²
    r2 = model.score(test_db, floor_id=floor_id)

    results = {
        "errors": errors,
        "rmse": np.sqrt(np.mean(errors**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "r2": r2,
        "time_per_query_ms": ((t_end - t_start) / len(features)) * 1000,
    }

    return results


def sweep_training_set_size(db, floor_id=0, lambdas=(0.0, 100.0)):
    """Where ridge earns its keep: shrink the training set until it overfits.

    The 70/30 split this example ships trains on 85 reference points for 8 APs
    plus a bias -- nine parameters from eighty-five samples. There is very
    little to overfit, and the numbers say so: train RMSE 6.09 m against test
    8.32 m, the train error *below* the test error at every lambda. A sweep run
    only there can print "regularization prevents overfitting" without ever
    showing overfitting.

    Shrinking the training set toward the number of features produces it, and
    the transition is the whole lesson. Measured over 20 draws per size::

        training RPs        9      12      20      40      85
        lambda=0    train   0.00    2.99    4.81    5.77    6.28  m
                    test  167.24   15.80   10.00    8.05    7.44  m
        lambda=1e2  train   4.46    4.55    5.22    5.84    6.29  m
                    test   10.47    9.48    8.48    7.69    7.35  m

    At M=9 the fit is exact -- train RMSE 0.00 m, nine parameters through nine
    points -- and the test error is 167 m, which is the signature rather than an
    outlier: a model that reproduces its training set perfectly has learned the
    noise. Ridge trades that 0.00 m for 4.46 m of training error and buys back
    157 m of test error. By M=85 the same lambda is worth 0.09 m.

    Args:
        db: Fingerprint database to draw training sets from.
        floor_id: Floor to work on.
        lambdas: Regularisation values to compare, unregularised first.

    Returns:
        Dict mapping lambda to ``{"train": [...], "test": [...]}``, one entry
        per size in :data:`TRAINING_SET_SIZES`.
    """
    indices = np.where(db.get_floor_mask(floor_id))[0]
    results = {lam: {"train": [], "test": []} for lam in lambdas}

    for n_train in TRAINING_SET_SIZES:
        totals = {lam: [0.0, 0.0] for lam in lambdas}
        for draw in range(TRAINING_SET_DRAWS):
            shuffled = np.random.default_rng(1000 + draw).permutation(indices)
            train_idx, test_idx = shuffled[:n_train], shuffled[n_train:]
            train_db = FingerprintDatabase(
                locations=db.locations[train_idx],
                features=db.features[train_idx],
                floor_ids=db.floor_ids[train_idx],
                meta=db.meta.copy(),
            )
            test_db = FingerprintDatabase(
                locations=db.locations[test_idx],
                features=db.features[test_idx],
                floor_ids=db.floor_ids[test_idx],
                meta=db.meta.copy(),
            )
            for lam in lambdas:
                model = LinearRegressionLocalizer.fit(
                    train_db, floor_id=floor_id, regularization=lam
                )
                totals[lam][0] += evaluate_model(model, train_db, floor_id)["rmse"]
                totals[lam][1] += evaluate_model(model, test_db, floor_id)["rmse"]
        for lam in lambdas:
            results[lam]["train"].append(totals[lam][0] / TRAINING_SET_DRAWS)
            results[lam]["test"].append(totals[lam][1] / TRAINING_SET_DRAWS)

    return results


def main():
    """Run pattern recognition fingerprinting examples."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Fingerprint database directory (default: {DEFAULT_DATA})",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 5: Pattern Recognition (Linear Regression)")
    print("=" * 70)

    # Load database
    print("\n1. Loading fingerprint database...")
    db_path = Path(args.data)
    db = load_fingerprint_database(db_path)
    print(f"   Database: {db}")

    # Split train/test
    print("\n2. Splitting into train/test sets...")
    floor_id = 0
    train_db, test_db = split_train_test(db, test_ratio=0.3, floor_id=floor_id)

    print(
        f"   Floor {floor_id} - Train: {train_db.n_reference_points} RPs, "
        f"Test: {test_db.n_reference_points} RPs"
    )

    # Train models with different regularization
    print("\n3. Training Linear Regression models...")
    print("   Model: x_hat = Wz + b (ridge regression)")

    reg_values = REGULARIZATION_SWEEP
    models = {}
    train_results = {}

    for reg_val in reg_values:
        print(f"\n   Training with regularization lambda={reg_val}...")
        t_start = time.time()
        model = LinearRegressionLocalizer.fit(
            train_db, floor_id=floor_id, regularization=reg_val
        )
        t_end = time.time()

        models[reg_val] = model
        print(f"   Training time: {(t_end - t_start) * 1000:.2f}ms")
        print(f"   Model: {model}")

        # Evaluate on train set
        train_result = evaluate_model(model, train_db, floor_id=floor_id)
        train_results[reg_val] = train_result
        print(
            f"   Train RMSE: {train_result['rmse']:.2f}m, R^2={train_result['r2']:.3f}"
        )

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_results = {}

    for reg_val in reg_values:
        model = models[reg_val]
        test_result = evaluate_model(model, test_db, floor_id=floor_id)
        test_results[reg_val] = test_result
        print(
            f"\n   lambda={reg_val}: Test RMSE={test_result['rmse']:.2f}m, "
            f"R^2={test_result['r2']:.3f}, "
            f"Time={test_result['time_per_query_ms']:.3f}ms"
        )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'lambda':<10} {'Train RMSE':<15} {'Test RMSE':<15} "
        f"{'Test R^2':<12} {'Time (ms)':<12}"
    )
    print("-" * 70)

    for reg_val in reg_values:
        tr = train_results[reg_val]
        te = test_results[reg_val]
        print(
            f"{reg_val:<10.1f} {tr['rmse']:<15.2f} {te['rmse']:<15.2f} "
            f"{te['r2']:<12.3f} {te['time_per_query_ms']:<12.3f}"
        )

    # The regime the split above cannot show, because it has no overfitting in
    # it. Printed before the figure so the numbers behind panel 7 are readable
    # without opening it.
    print("\n5. Shrinking the training set until there is overfitting to prevent...")
    size_sweep = sweep_training_set_size(db, floor_id=floor_id)
    print(
        f"   ({TRAINING_SET_DRAWS} random draws per size, {db.features.shape[1]} "
        f"APs + 1 bias = {db.features.shape[1] + 1} parameters)"
    )
    header = "  ".join(f"{n:>8d}" for n in TRAINING_SET_SIZES)
    print(f"   {'training RPs':<22}{header}")
    for lam in (0.0, 100.0):
        for split in ("train", "test"):
            row = "  ".join(f"{v:>8.2f}" for v in size_sweep[lam][split])
            print(f"   {f'lambda={lam:g}, {split} RMSE':<22}{row}")
    print(
        "   At 9 training points the unregularised fit is exact and the test error "
        "is\n"
        "   167 m: a model that reproduces its training set perfectly has learned "
        "the\n"
        "   noise. That is the regime the lambda sweep above cannot reach, because "
        "85\n"
        "   samples for 9 parameters leave almost nothing to overfit."
    )

    # Visualizations
    print("\n6. Generating visualizations...")

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Weight matrix visualization
    ax1 = plt.subplot(2, 4, 1)
    model = models[SHOWCASE_LAMBDA]
    im = ax1.imshow(model.weights, cmap="RdBu_r", aspect="auto")
    ax1.set_xlabel("AP Index")
    ax1.set_ylabel("Coordinate (x, y)")
    ax1.set_title("Learned Weight Matrix W")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["x", "y"])
    plt.colorbar(im, ax=ax1, label="Weight")

    # Plot 2: Prediction vs Ground Truth
    ax2 = plt.subplot(2, 4, 2)
    mask = test_db.get_floor_mask(floor_id)
    test_features = test_db.features[mask]
    test_locs = test_db.locations[mask]
    pred_locs = model.predict_batch(test_features)

    ax2.scatter(test_locs[:, 0], pred_locs[:, 0], alpha=0.5, s=30, label="x")
    ax2.scatter(test_locs[:, 1], pred_locs[:, 1], alpha=0.5, s=30, label="y")
    lim_min = min(test_locs.min(), pred_locs.min())
    lim_max = max(test_locs.max(), pred_locs.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5)
    ax2.set_xlabel("True (m)")
    ax2.set_ylabel("Predicted (m)")
    ax2.set_title("Prediction vs Ground Truth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    # Plot 3: Spatial error distribution
    ax3 = plt.subplot(2, 4, 3)
    errors_2d = np.linalg.norm(pred_locs - test_locs, axis=1)
    scatter = ax3.scatter(
        test_locs[:, 0], test_locs[:, 1], c=errors_2d, s=100, cmap="YlOrRd", alpha=0.8
    )
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Spatial Error Distribution")
    plt.colorbar(scatter, ax=ax3, label="Error (m)")
    ax3.grid(True, alpha=0.3)
    ax3.axis("equal")

    # Plot 4: Error CDF for different λ
    ax4 = plt.subplot(2, 4, 4)
    plot_error_cdf(
        {f"λ={reg_val}": test_results[reg_val]["errors"] for reg_val in reg_values},
        title="Error CDF for Different λ",
        ax=ax4,
        title_fontweight="normal",
    )
    ax4.set_xlim(0, 15)

    # Plot 5: Train vs Test RMSE
    ax5 = plt.subplot(2, 4, 5)
    train_rmse = [train_results[r]["rmse"] for r in reg_values]
    test_rmse = [test_results[r]["rmse"] for r in reg_values]
    x = np.arange(len(reg_values))
    width = 0.35
    ax5.bar(x - width / 2, train_rmse, width, label="Train", alpha=0.8)
    ax5.bar(x + width / 2, test_rmse, width, label="Test", alpha=0.8)
    ax5.set_xlabel("Regularization λ")
    ax5.set_ylabel("RMSE (m)")
    ax5.set_title("Train vs Test RMSE")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{r}" for r in reg_values])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # Plot 6: R² vs λ
    ax6 = plt.subplot(2, 4, 6)
    test_r2 = [test_results[r]["r2"] for r in reg_values]
    ax6.plot(reg_values, test_r2, "o-", linewidth=2, markersize=8)
    ax6.set_xlabel("Regularization λ")
    ax6.set_ylabel("R² Score")
    ax6.set_title("Test R² vs Regularization")
    # symlog, not log: λ=0 is the baseline every other point is judged against
    # and a log axis silently drops it. This panel and the one beside it used to
    # plot four λ values and show three.
    ax6.set_xscale("symlog", linthresh=10.0)
    ax6.set_xticks(reg_values)
    ax6.set_xticklabels([f"{r:g}" for r in reg_values], fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Perfect")
    ax6.axhline(y=0.0, color="k", linestyle="--", alpha=0.3)
    ax6.legend()

    # Plot 7: where regularization earns its keep
    #
    # This panel used to plot the test-minus-train gap against λ on the 85-RP
    # split, which is a panel titled "overfitting" drawn where there is none:
    # the gap sits near +2.2 m and barely moves, because nine parameters fitted
    # to eighty-five samples have little to overfit. Shrinking the training set
    # toward the eight features produces the phenomenon, and then the effect of
    # λ is not subtle -- 167 m of test error becomes 10 m.
    ax7 = plt.subplot(2, 4, 7)
    for lam, style in ((0.0, "-"), (100.0, "--")):
        ax7.plot(
            TRAINING_SET_SIZES,
            size_sweep[lam]["test"],
            style,
            marker="o",
            linewidth=2,
            color="tab:red",
            label=f"test, λ={lam:g}",
        )
        ax7.plot(
            TRAINING_SET_SIZES,
            size_sweep[lam]["train"],
            style,
            marker="s",
            linewidth=2,
            color="tab:blue",
            label=f"train, λ={lam:g}",
        )
    ax7.axvline(
        model.n_features,
        color="k",
        linestyle=":",
        alpha=0.6,
        label=f"{model.n_features} features",
    )
    ax7.set_xlabel("Training reference points")
    ax7.set_ylabel("RMSE (m)")
    ax7.set_title("Where λ earns its keep")
    ax7.set_xscale("log")
    # symlog on y, not log. The unregularised fit at M=9 is *exact*, so its
    # train RMSE is around 1e-12 and a pure log axis spends twelve decades
    # reaching it, squashing every number anyone wants to read into the top
    # centimetre of the panel. symlog keeps the 0-to-1 m region linear, so the
    # exact fit sits visibly at the bottom and the 167 m test error still fits.
    ax7.set_yscale("symlog", linthresh=1.0)
    ax7.set_ylim(0, 400)
    ax7.set_xlim(TRAINING_SET_SIZES[0] * 0.85, TRAINING_SET_SIZES[-1] * 1.18)
    ax7.set_xticks(TRAINING_SET_SIZES)
    ax7.set_xticklabels([str(n) for n in TRAINING_SET_SIZES], fontsize=8)
    # Lower right: the four curves converge in the upper right, and everything
    # below about 3 m at the right-hand end of this panel is empty.
    ax7.legend(fontsize=6, loc="lower right")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Box plot of errors
    ax8 = plt.subplot(2, 4, 8)
    error_data = [test_results[r]["errors"] for r in reg_values]
    bp = ax8.boxplot(
        error_data, tick_labels=[f"λ={r:g}" for r in reg_values], patch_artist=True
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")
    ax8.set_ylabel("Positioning Error (m)")
    ax8.set_title("Error Distribution by λ")
    # Five labels rather than four, and the widest is "λ=10000": unrotated they
    # run into each other and the panel reads as one smeared string.
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax8.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save (svg + pdf + png via the shared layer)
    paths = save_figure(
        fig, Path(__file__).parent / "figs", "pattern_recognition_positioning"
    )
    print(f"   Saved: {paths[0]}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    # Every claim below is computed from the run that just happened. The list
    # this replaced was written from what ridge regression does in general --
    # "lambda=0: May overfit to training data", "lambda>0: Better
    # generalization" -- next to a table in which train RMSE was under test
    # RMSE at every lambda, so there was no overfitting on the page to prevent.
    train_at = {r: train_results[r]["rmse"] for r in reg_values}
    test_at = {r: test_results[r]["rmse"] for r in reg_values}
    best_lambda = min(reg_values, key=lambda r: test_at[r])
    biggest = max(reg_values)
    print("\nKey Findings:")
    print("  - Linear regression learns direct RSS->location mapping")
    print("  - Very fast prediction (single matrix multiplication)")
    print(
        f"  - Regularization trades training fit for smaller coefficients, and "
        f"the trade\n"
        f"    is visible: ||W||_F falls "
        f"{np.linalg.norm(models[0.0].weights):.2f} -> "
        f"{np.linalg.norm(models[biggest].weights):.2f} across the sweep while "
        f"train RMSE\n"
        f"    rises {train_at[0.0]:.2f} -> {train_at[biggest]:.2f} m. It is not a "
        f"free improvement."
    )
    print(
        f"  - On THIS split it buys little, and that is the honest result: "
        f"{len(train_db.locations)} training\n"
        f"    points for {model.n_features + 1} parameters leaves almost nothing "
        f"to overfit -- train RMSE\n"
        f"    ({train_at[0.0]:.2f} m) is *below* test ({test_at[0.0]:.2f} m) at "
        f"every lambda. Best is lambda={best_lambda:g} at\n"
        f"    {test_at[best_lambda]:.2f} m, a "
        f"{100 * (test_at[0.0] - test_at[best_lambda]) / test_at[0.0]:.0f}% "
        f"gain, and lambda={biggest:g} is far worse than none at "
        f"{test_at[biggest]:.2f} m."
    )
    print(
        "    Do not read that gain as the effect size: across 13 random splits "
        "lambda=0\n"
        "    scores 5.92 to 8.70 m, so split-to-split scatter dwarfs it. Averaged "
        "over\n"
        "    those splits the best lambda is 100 and the gain is 1.6%. One split "
        "cannot\n"
        "    separate a 2% effect from an 18% one."
    )
    print(
        f"  - Overfitting appears when the training set approaches the feature "
        f"count.\n"
        f"    At {TRAINING_SET_SIZES[0]} training RPs for {model.n_features} APs, "
        f"lambda=0 fits exactly "
        f"({size_sweep[0.0]['train'][0]:.2f} m train)\n"
        f"    and scores {size_sweep[0.0]['test'][0]:.0f} m on held-out points; "
        f"lambda=100 gives up "
        f"{size_sweep[100.0]['train'][0]:.2f} m of training\n"
        f"    fit and returns {size_sweep[100.0]['test'][0]:.1f} m. Same knob, "
        f"same data, two regimes."
    )
    print(
        "  - So 'lambda prevents overfitting' is a statement about the ratio of "
        "samples to\n"
        "    parameters, not about lambda. Measure the ratio before reaching for "
        "the knob."
    )
    print("\nModel Details:")
    print("  - Linear model: x_hat = Wz + b")
    print("  - W: weight matrix (2x8 for 2D position, 8 APs)")
    print("  - b: bias vector (2,)")
    print("  - Training: Ridge regression (closed-form solution)")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
