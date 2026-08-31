"""Classification-based fingerprinting example for Chapter 5.

This script demonstrates pattern recognition classifiers (Random Forest, SVM)
and hierarchical coarse-to-fine localization, as described in Section 5.2
of the book.

Key demonstrations:
    1. Direct classification: Each RP as a class
    2. Classification accuracy vs deterministic/probabilistic methods
    3. Hierarchical localization: Coarse (floor) -> Fine (k-NN/Bayesian)
    4. Comparison of classifier types (Random Forest vs SVM)

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.eval import plot_error_cdf, save_figure, show_figures_if_requested
from core.fingerprinting import (
    FingerprintDatabase,
    fit_classifier,
    hierarchical_fingerprint_localize,
    k_nearest_neighbor_localize,
    load_fingerprint_database,
)

# Seed for this example's synthetic database and query draws. Fixed so the
# committed figures and reported accuracies can be regenerated exactly.
DEFAULT_SEED = 42
DEFAULT_DATA = "data/sim/ch5_wifi_fingerprint_grid"


def rmse(errors):
    """Return root-mean-square error for a sequence of scalar errors."""
    return float(np.sqrt(np.mean(np.square(errors))))


def load_multifloor_database(data_dir: str = DEFAULT_DATA) -> FingerprintDatabase:
    """Load the shipped three-floor Wi-Fi database, as the other ch5 examples do.

    This example used to build its own database inline, and that database
    could not support either of the things the example measures on it:

    - **No relation between location and RSS.** The features were
      ``base_rss - rng.random((n_rps, n_aps)) * 40``, drawn independently of
      the RP's coordinates. k-NN assumes nearby locations have similar
      fingerprints, so averaging the 5 nearest neighbours in RSS space
      returned 5 spatially unrelated points. Every k-NN number the example
      printed was measuring that, not the method.
    - **Floors 87.5% overlapping.** ``base_rss = -50 - floor * 5`` separates
      the floor means by 5 dB while the uniform spread within a floor is
      40 dB wide (std 11.5 dB), a Fisher ratio of 0.43 -- despite the comment
      claiming the offsets "enable floor classification".

    The shipped database is built from a propagation model, so it has both:
    floor means 15.25 dB apart against a 7.38 dB within-floor std (ratio
    2.07), and fingerprints that vary smoothly with position.
    """
    db_path = Path(data_dir)
    print("\n--- Loading multi-floor fingerprint database ---")
    db = load_fingerprint_database(db_path)

    print(f"  Loaded database: {db_path}")
    print(f"    - {db.n_reference_points} RPs across {db.n_floors} floors")
    print(f"    - {db.n_features} APs")
    print(f"    - {db.n_reference_points // db.n_floors} RPs per floor")

    return db


def evaluate_classification_accuracy(db: FingerprintDatabase, rng=None):
    """Measure classifier accuracy on held-out queries, not on training data.

    Args:
        db: Fingerprint database under test.
        rng: Generator for the held-out query draws; defaults to a fresh
            seeded one.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    print("\n" + "=" * 70)
    print("Test 1: Classification Accuracy")
    print("=" * 70)

    # Fit classifiers
    print("\n--- Training classifiers ---")
    print("  1. Random Forest (n_estimators=100)")
    rf_classifier = fit_classifier(
        db, classifier_type="random_forest", zone_type="rp", n_estimators=100
    )

    print("  2. SVM (RBF kernel)")
    svm_classifier = fit_classifier(
        db, classifier_type="svm", zone_type="rp", kernel="rbf", C=1.0
    )

    # Two measurements, because only the second one is an accuracy.
    #
    # `zone_type="rp"` makes every RP its own class, and the database holds one
    # fingerprint per RP, so the classifiers are trained on exactly one sample
    # of each of their N classes. Feeding the training vectors back in then
    # asks whether the model can memorise N points, and the answer is 100.0%
    # by construction for any classifier that fits its training set. This
    # example used to print that number under the heading "Classification
    # Accuracy" with no held-out comparison beside it.
    features = db.get_mean_features()

    print("\n--- Recall on the training vectors (memorisation check) ---")
    rf_correct = sum(
        np.allclose(rf_classifier.predict(features[i])[0], db.locations[i], atol=0.1)
        for i in range(db.n_reference_points)
    )
    svm_correct = sum(
        np.allclose(svm_classifier.predict(features[i])[0], db.locations[i], atol=0.1)
        for i in range(db.n_reference_points)
    )
    rf_recall = 100 * rf_correct / db.n_reference_points
    svm_recall = 100 * svm_correct / db.n_reference_points

    print(f"  Random Forest: {rf_recall:.1f}% ({rf_correct}/{db.n_reference_points})")
    print(f"  SVM:           {svm_recall:.1f}% ({svm_correct}/{db.n_reference_points})")
    print("  Expected to be 100% -- these are the training vectors themselves,")
    print("  one per class. This says the models fit; it says nothing about")
    print("  how they generalise, and is not a positioning result.")

    # The real measurement: unseen queries, drawn by perturbing an RP's
    # fingerprint the way a live measurement would differ from the survey.
    noise_std = 2.0
    n_queries = 200
    print(f"\n--- Held-out queries (sigma = {noise_std:.1f} dBm, n = {n_queries}) ---")
    rf_hit = svm_hit = 0
    for _ in range(n_queries):
        idx = rng.integers(0, db.n_reference_points)
        query = features[idx] + rng.standard_normal(db.n_features) * noise_std
        rf_hit += np.allclose(
            rf_classifier.predict(query)[0], db.locations[idx], atol=0.1
        )
        svm_hit += np.allclose(
            svm_classifier.predict(query)[0], db.locations[idx], atol=0.1
        )

    rf_accuracy = 100 * rf_hit / n_queries
    svm_accuracy = 100 * svm_hit / n_queries
    print(f"  Random Forest: {rf_accuracy:.1f}% ({rf_hit}/{n_queries}) exact RP")
    print(f"  SVM:           {svm_accuracy:.1f}% ({svm_hit}/{n_queries}) exact RP")
    print("  This is the number to compare against other methods.")

    return rf_classifier, svm_classifier


def evaluate_noisy_queries(
    db: FingerprintDatabase, rf_classifier, svm_classifier, rng=None
):
    """Test classification with noisy queries.

    Args:
        db: Fingerprint database under test.
        rf_classifier: Trained random-forest floor classifier.
        svm_classifier: Trained SVM floor classifier.
        rng: Generator for the query draws; defaults to a fresh seeded one.
            Unseeded, the reported accuracies moved on every run, so the
            committed figure could not be reproduced.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    print("\n" + "=" * 70)
    print("Test 2: Robustness to Noise")
    print("=" * 70)

    # Add noise to database features
    #
    # The sigma = 0 column is exactly 0.00 m for all three methods, and it is
    # not missing data: a zero-noise query *is* the stored fingerprint of the RP
    # it was drawn from, so every method returns that RP's own location. The row
    # is the anchor the rest of the sweep is read against rather than a result,
    # and an uncommented 0.00 in a robustness table invites the opposite
    # reading.
    noise_levels = [0, 2, 4, 6, 8]  # dBm
    n_queries = 50

    rf_errors = []
    svm_errors = []
    knn_errors = []

    for noise_std in noise_levels:
        rf_errs = []
        svm_errs = []
        knn_errs = []

        for _ in range(n_queries):
            # Random RP
            rp_idx = rng.integers(0, db.n_reference_points)
            true_loc = db.locations[rp_idx]
            query = (
                db.get_mean_features()[rp_idx]
                + rng.standard_normal(db.n_features) * noise_std
            )

            # RF classification
            pred_rf, _ = rf_classifier.predict(query)
            rf_errs.append(np.linalg.norm(pred_rf - true_loc))

            # SVM classification
            pred_svm, _ = svm_classifier.predict(query)
            svm_errs.append(np.linalg.norm(pred_svm - true_loc))

            # k-NN for comparison. floor_id=None on purpose: this line used to
            # pass the query's *true* floor, so the baseline was handed a fact
            # the classifiers had to infer. Searching all three floors is the
            # comparison the header claims to be making.
            pred_knn = k_nearest_neighbor_localize(query, db, k=5, floor_id=None)
            knn_errs.append(np.linalg.norm(pred_knn - true_loc))

        # RMSE, as printed. These three lines used to take np.mean of the
        # errors and label the result RMSE, which understates every entry.
        rf_errors.append(rmse(rf_errs))
        svm_errors.append(rmse(svm_errs))
        knn_errors.append(rmse(knn_errs))

        print(f"\n  Noise sigma = {noise_std} dBm:")
        print(f"    Random Forest RMSE: {rf_errors[-1]:.2f} m")
        print(f"    SVM RMSE:          {svm_errors[-1]:.2f} m")
        print(f"    k-NN (k=5) RMSE:   {knn_errors[-1]:.2f} m")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_levels, rf_errors, "o-", label="Random Forest", linewidth=2)
    ax.plot(noise_levels, svm_errors, "s-", label="SVM", linewidth=2)
    ax.plot(noise_levels, knn_errors, "^-", label="k-NN (k=5)", linewidth=2)
    ax.set_xlabel("Noise Standard Deviation (dBm)", fontsize=12)
    ax.set_ylabel("Positioning Error RMSE (m)", fontsize=12)
    ax.set_title("Classification vs k-NN: Robustness to Noise", fontsize=14)
    # Say why all three curves start at zero, on the figure. Three lines meeting
    # at 0.00 m looks like a plotting artefact; it is the definition of the
    # left-hand column.
    ax.annotate(
        "all three are exact at σ=0:\nthe query IS the stored fingerprint",
        xy=(0, 0),
        # Low and close: a leader from the origin to mid-panel crosses the
        # curves it is trying to explain.
        xytext=(0.30, 0.05),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "alpha": 0.6},
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save figure (svg + pdf + png via the shared layer)
    plt.tight_layout()
    paths = save_figure(
        fig, Path(__file__).parent / "figs", "classification_noise_robustness"
    )
    print(f"\n  [OK] Saved figure: {paths[0]}")

    plt.close()


def evaluate_hierarchical_localization(db: FingerprintDatabase, rng=None):
    """Test hierarchical coarse-to-fine localization.

    Args:
        db: Fingerprint database under test.
        rng: Generator for the query draws; defaults to a fresh seeded one.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    print("\n" + "=" * 70)
    print("Test 3: Hierarchical Localization (Coarse -> Fine)")
    print("=" * 70)

    # Generate test queries on each floor
    n_queries = 100
    queries = []
    true_locs = []
    true_floors = []

    for _ in range(n_queries):
        rp_idx = rng.integers(0, db.n_reference_points)
        query = (
            db.get_mean_features()[rp_idx] + rng.standard_normal(db.n_features) * 3.0
        )
        queries.append(query)
        true_locs.append(db.locations[rp_idx])
        true_floors.append(db.floor_ids[rp_idx])

    queries = np.array(queries)
    true_locs = np.array(true_locs)
    true_floors = np.array(true_floors)

    # Method 1: Direct k-NN (no hierarchy)
    print("\n--- Method 1: Direct k-NN (no floor constraint) ---")
    direct_errors = []
    for i, query in enumerate(queries):
        pred = k_nearest_neighbor_localize(query, db, k=5, floor_id=None)
        direct_errors.append(np.linalg.norm(pred - true_locs[i]))
    direct_rmse = np.sqrt(np.mean(np.array(direct_errors) ** 2))
    print(f"  RMSE: {direct_rmse:.2f} m")

    # Method 2: Hierarchical (floor -> k-NN)
    print("\n--- Method 2: Hierarchical (Floor -> k-NN) ---")
    hier_errors = []
    floor_hit = []
    for i, query in enumerate(queries):
        pred, info = hierarchical_fingerprint_localize(
            query, db, coarse_method="floor", fine_method="knn", k=5
        )
        hier_errors.append(np.linalg.norm(pred - true_locs[i]))
        floor_hit.append(info["coarse_floor"] == true_floors[i])

    floor_hit = np.array(floor_hit)
    floor_correct = int(floor_hit.sum())
    hier_rmse = np.sqrt(np.mean(np.array(hier_errors) ** 2))
    floor_accuracy = 100 * floor_correct / n_queries
    print(
        f"  Floor classification accuracy: {floor_accuracy:.1f}% "
        f"({floor_correct}/{n_queries}, chance = {100 / db.n_floors:.1f}%)"
    )

    # Both numbers, because the second one is conditional and the difference
    # matters. The single line here used to read "RMSE (given correct floor)"
    # while reporting `hier_rmse`, which is over *all* queries -- including
    # every one the coarse step sent to the wrong floor. When floor accuracy
    # was 29%, that label was describing a subset of 29 queries and printing
    # the number for 100.
    print(f"  RMSE, all queries:             {hier_rmse:.2f} m")
    if floor_correct:
        subset = np.array(hier_errors)[floor_hit]
        subset_rmse = float(np.sqrt(np.mean(subset**2)))
        print(
            f"  RMSE, correct-floor subset:    {subset_rmse:.2f} m "
            f"(n = {floor_correct})"
        )
        if floor_correct < n_queries:
            print("  The subset figure is the accuracy of the queries the coarse")
            print("  step got right, so it flatters the method whenever that step")
            print("  fails. Quote the all-queries number as the method's error.")

    # Method 3: Hierarchical (RF -> MAP)
    print("\n--- Method 3: Hierarchical (RF -> MAP) ---")
    hier_rf_errors = []
    for i, query in enumerate(queries):
        pred, info = hierarchical_fingerprint_localize(
            query, db, coarse_method="random_forest", fine_method="map"
        )
        hier_rf_errors.append(np.linalg.norm(pred - true_locs[i]))
    hier_rf_rmse = np.sqrt(np.mean(np.array(hier_rf_errors) ** 2))
    print(f"  RMSE: {hier_rf_rmse:.2f} m")

    # Method 4: Hierarchical (floor -> Posterior Mean)
    print("\n--- Method 4: Hierarchical (Floor -> Posterior Mean) ---")
    hier_pm_errors = []
    for i, query in enumerate(queries):
        pred, info = hierarchical_fingerprint_localize(
            query, db, coarse_method="floor", fine_method="posterior_mean", top_k=10
        )
        hier_pm_errors.append(np.linalg.norm(pred - true_locs[i]))
    hier_pm_rmse = np.sqrt(np.mean(np.array(hier_pm_errors) ** 2))
    print(f"  RMSE: {hier_pm_rmse:.2f} m")

    # Summary. The verdict below is computed from the numbers rather than
    # written next to them, so it cannot drift out of agreement with the run.
    #
    # The exact-hit column is why RMSE alone would mislead here. Queries are
    # built by perturbing an RP's fingerprint, so the answer always sits
    # exactly on a reference point. A method that returns one RP can therefore
    # score a true zero, while a method that interpolates between neighbours
    # never can, however close it lands. That is a property of the evaluation,
    # not of the methods, and it is worth several metres of apparent RMSE.
    runs = {
        "Direct k-NN (baseline)": direct_errors,
        "Floor -> kNN": hier_errors,
        "RF -> MAP": hier_rf_errors,
        "Floor -> PM": hier_pm_errors,
    }
    print("\n--- Summary ---")
    print(f"  {'Method':<24}{'RMSE':>8}{'median':>9}{'exact hit':>12}")
    for name, errs in runs.items():
        e = np.asarray(errs)
        print(
            f"  {name:<24}{np.sqrt(np.mean(e**2)):>7.2f}m{np.median(e):>8.2f}m"
            f"{100 * np.mean(e < 1e-9):>11.1f}%"
        )

    candidates = {
        k: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
        for k, v in runs.items()
        if k != "Direct k-NN (baseline)"
    }
    beat = {k: v for k, v in candidates.items() if v < direct_rmse - 1e-9}
    tied = {k: v for k, v in candidates.items() if abs(v - direct_rmse) <= 1e-9}

    print()
    if beat:
        best = min(beat, key=beat.get)
        gain = 100 * (direct_rmse - beat[best]) / direct_rmse
        print(
            f"  Best: hierarchical ({best}) at {beat[best]:.2f} m, "
            f"{gain:.1f}% below the baseline."
        )
        exact = 100 * np.mean(np.asarray(runs[best]) < 1e-9)
        if exact > 20:
            print(f"  Read that with the exact-hit column: {exact:.0f}% of its")
            print("  queries land on the right RP for zero error, which only")
            print("  happens because every query was generated at an RP. Off the")
            print("  survey grid that margin should shrink, and the interpolating")
            print("  methods are the ones with somewhere to move.")
    else:
        print("  No hierarchical variant beats the direct baseline here, so this")
        print("  run does not demonstrate that the hierarchy helps.")
    if tied:
        print(f"  {', '.join(tied)} matches the baseline exactly. That is the")
        print("  expected result once floor classification is reliable: k-NN")
        print("  already draws its neighbours from the right floor, so adding")
        print("  the constraint removes nothing. The hierarchy buys accuracy")
        print("  only where it changes which RPs the fine step can see.")

    # Plot error distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error CDFs
    ax = axes[0, 0]
    plot_error_cdf(
        {
            "Direct k-NN": direct_errors,
            "Hierarchical (Floor -> k-NN)": hier_errors,
            "Hierarchical (RF -> MAP)": hier_rf_errors,
            "Hierarchical (Floor -> PM)": hier_pm_errors,
        },
        title="Error Distribution (CDF)",
        ax=ax,
        title_fontweight="normal",
    )
    ax.legend(fontsize=9)

    # Box plots
    ax = axes[0, 1]
    ax.boxplot(
        [direct_errors, hier_errors, hier_rf_errors, hier_pm_errors],
        tick_labels=[
            "Direct\nk-NN",
            "Hier\nFloor->kNN",
            "Hier\nRF->MAP",
            "Hier\nFloor->PM",
        ],
        showfliers=False,
    )
    ax.set_ylabel("Positioning Error (m)", fontsize=11)
    ax.set_title("Error Distribution (Box Plot)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # RMSE comparison
    ax = axes[1, 0]
    methods = ["Direct\nk-NN", "Hier\nFloor->kNN", "Hier\nRF->MAP", "Hier\nFloor->PM"]
    rmses = [direct_rmse, hier_rmse, hier_rf_rmse, hier_pm_rmse]
    bars = ax.bar(methods, rmses, color=["C0", "C1", "C2", "C3"], alpha=0.7)
    ax.set_ylabel("RMSE (m)", fontsize=11)
    ax.set_title("RMSE Comparison", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    # Add values on bars
    for bar, rmse in zip(bars, rmses, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{rmse:.2f}m",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Floor classification confusion matrix (for hierarchical methods)
    ax = axes[1, 1]
    ax.text(
        0.5,
        0.6,
        f"Floor Classification\nAccuracy: {floor_accuracy:.1f}%",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.4,
        f"Correct: {floor_correct}/{n_queries}",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")

    plt.suptitle("Hierarchical Localization Performance", fontsize=14, y=0.995)
    plt.tight_layout()

    # Save figure (svg + pdf + png via the shared layer)
    paths = save_figure(
        fig, Path(__file__).parent / "figs", "hierarchical_localization"
    )
    print(f"\n  [OK] Saved figure: {paths[0]}")

    plt.close()


def main():
    """Run all classification-based fingerprinting demonstrations."""
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
    print("Chapter 5: Classification-Based Fingerprinting")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Pattern recognition classifiers (Random Forest, SVM)")
    print("  2. Classification accuracy vs deterministic/probabilistic methods")
    print("  3. Hierarchical coarse-to-fine localization")

    # One generator for the whole run, so every query set below is a
    # consecutive draw from a single seeded stream.
    rng = np.random.default_rng(DEFAULT_SEED)
    db = load_multifloor_database(args.data)

    # Test 1: Classification accuracy
    rf_classifier, svm_classifier = evaluate_classification_accuracy(db, rng=rng)

    # Test 2: Robustness to noise
    evaluate_noisy_queries(db, rf_classifier, svm_classifier, rng=rng)

    # Test 3: Hierarchical localization
    evaluate_hierarchical_localization(db, rng=rng)

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - figs/classification_noise_robustness.png")
    print("  - figs/hierarchical_localization.png")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
