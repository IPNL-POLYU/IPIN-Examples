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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fingerprinting import (
    FingerprintDatabase,
    fit_classifier,
    hierarchical_localize,
    knn_localize,
    load_fingerprint_database,
    map_localize,
    fit_gaussian_naive_bayes,
)


def create_multifloor_test_database() -> FingerprintDatabase:
    """Create a synthetic multi-floor database for testing."""
    print("\n--- Creating synthetic multi-floor database ---")
    
    # Parameters
    n_rps_per_floor = 16  # 4x4 grid per floor
    n_floors = 3
    grid_size = 4
    spacing = 5.0  # meters
    n_aps = 6
    
    locations_list = []
    features_list = []
    floor_ids_list = []
    
    for floor in range(n_floors):
        # Grid layout
        x = np.tile(np.arange(grid_size) * spacing, grid_size)
        y = np.repeat(np.arange(grid_size) * spacing, grid_size)
        locs = np.column_stack([x, y])
        
        # RSS features (different per floor to enable floor classification)
        # Floor 0: -50 to -90 dBm range
        # Floor 1: -55 to -95 dBm range (offset)
        # Floor 2: -60 to -100 dBm range (offset)
        base_rss = -50 - floor * 5
        feats = base_rss - np.random.rand(n_rps_per_floor, n_aps) * 40
        
        floors = np.full(n_rps_per_floor, floor, dtype=int)
        
        locations_list.append(locs)
        features_list.append(feats)
        floor_ids_list.append(floors)
    
    locations = np.vstack(locations_list)
    features = np.vstack(features_list)
    floor_ids = np.hstack(floor_ids_list)
    
    db = FingerprintDatabase(
        locations=locations,
        features=features,
        floor_ids=floor_ids,
        meta={
            "ap_ids": [f"AP{i+1}" for i in range(n_aps)],
            "unit": "dBm",
            "environment": "synthetic_multifloor"
        }
    )
    
    print(f"  Created database:")
    print(f"    - {db.n_reference_points} RPs across {db.n_floors} floors")
    print(f"    - {db.n_features} APs")
    print(f"    - {n_rps_per_floor} RPs per floor")
    
    return db


def test_classification_accuracy(db: FingerprintDatabase):
    """Test classification accuracy on database RPs."""
    print("\n" + "=" * 70)
    print("Test 1: Classification Accuracy")
    print("=" * 70)
    
    # Fit classifiers
    print("\n--- Training classifiers ---")
    print("  1. Random Forest (n_estimators=100)")
    rf_classifier = fit_classifier(
        db,
        classifier_type="random_forest",
        zone_type="rp",
        n_estimators=100
    )
    
    print("  2. SVM (RBF kernel)")
    svm_classifier = fit_classifier(
        db,
        classifier_type="svm",
        zone_type="rp",
        kernel="rbf",
        C=1.0
    )
    
    # Test on database RPs (perfect recall test)
    print("\n--- Testing on database RPs (perfect recall) ---")
    features = db.get_mean_features()
    
    rf_correct = 0
    svm_correct = 0
    
    for i in range(db.n_reference_points):
        query = features[i]
        true_loc = db.locations[i]
        
        # RF prediction
        pred_rf, _ = rf_classifier.predict(query)
        if np.allclose(pred_rf, true_loc, atol=0.1):
            rf_correct += 1
        
        # SVM prediction
        pred_svm, _ = svm_classifier.predict(query)
        if np.allclose(pred_svm, true_loc, atol=0.1):
            svm_correct += 1
    
    rf_accuracy = 100 * rf_correct / db.n_reference_points
    svm_accuracy = 100 * svm_correct / db.n_reference_points
    
    print(f"\n  Random Forest accuracy: {rf_accuracy:.1f}% ({rf_correct}/{db.n_reference_points})")
    print(f"  SVM accuracy:          {svm_accuracy:.1f}% ({svm_correct}/{db.n_reference_points})")
    
    return rf_classifier, svm_classifier


def test_noisy_queries(db: FingerprintDatabase, rf_classifier, svm_classifier):
    """Test classification with noisy queries."""
    print("\n" + "=" * 70)
    print("Test 2: Robustness to Noise")
    print("=" * 70)
    
    # Add noise to database features
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
            rp_idx = np.random.randint(0, db.n_reference_points)
            true_loc = db.locations[rp_idx]
            query = db.get_mean_features()[rp_idx] + np.random.randn(db.n_features) * noise_std
            floor_id = db.floor_ids[rp_idx]
            
            # RF classification
            pred_rf, _ = rf_classifier.predict(query)
            rf_errs.append(np.linalg.norm(pred_rf - true_loc))
            
            # SVM classification
            pred_svm, _ = svm_classifier.predict(query)
            svm_errs.append(np.linalg.norm(pred_svm - true_loc))
            
            # k-NN for comparison
            pred_knn = knn_localize(query, db, k=5, floor_id=floor_id)
            knn_errs.append(np.linalg.norm(pred_knn - true_loc))
        
        rf_errors.append(np.mean(rf_errs))
        svm_errors.append(np.mean(svm_errs))
        knn_errors.append(np.mean(knn_errs))
        
        print(f"\n  Noise sigma = {noise_std} dBm:")
        print(f"    Random Forest RMSE: {rf_errors[-1]:.2f} m")
        print(f"    SVM RMSE:          {svm_errors[-1]:.2f} m")
        print(f"    k-NN (k=5) RMSE:   {knn_errors[-1]:.2f} m")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_levels, rf_errors, 'o-', label='Random Forest', linewidth=2)
    ax.plot(noise_levels, svm_errors, 's-', label='SVM', linewidth=2)
    ax.plot(noise_levels, knn_errors, '^-', label='k-NN (k=5)', linewidth=2)
    ax.set_xlabel('Noise Standard Deviation (dBm)', fontsize=12)
    ax.set_ylabel('Mean Positioning Error (m)', fontsize=12)
    ax.set_title('Classification vs k-NN: Robustness to Noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    fig_path = figs_dir / "classification_noise_robustness.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  [OK] Saved figure: {fig_path}")
    
    plt.close()


def test_hierarchical_localization(db: FingerprintDatabase):
    """Test hierarchical coarse-to-fine localization."""
    print("\n" + "=" * 70)
    print("Test 3: Hierarchical Localization (Coarse -> Fine)")
    print("=" * 70)
    
    # Generate test queries on each floor
    n_queries = 100
    queries = []
    true_locs = []
    true_floors = []
    
    for _ in range(n_queries):
        rp_idx = np.random.randint(0, db.n_reference_points)
        query = db.get_mean_features()[rp_idx] + np.random.randn(db.n_features) * 3.0
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
        pred = knn_localize(query, db, k=5, floor_id=None)
        direct_errors.append(np.linalg.norm(pred - true_locs[i]))
    direct_rmse = np.sqrt(np.mean(np.array(direct_errors) ** 2))
    print(f"  RMSE: {direct_rmse:.2f} m")
    
    # Method 2: Hierarchical (floor -> k-NN)
    print("\n--- Method 2: Hierarchical (Floor -> k-NN) ---")
    hier_errors = []
    floor_correct = 0
    for i, query in enumerate(queries):
        pred, info = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="knn",
            k=5
        )
        hier_errors.append(np.linalg.norm(pred - true_locs[i]))
        if info["coarse_floor"] == true_floors[i]:
            floor_correct += 1
    
    hier_rmse = np.sqrt(np.mean(np.array(hier_errors) ** 2))
    floor_accuracy = 100 * floor_correct / n_queries
    print(f"  Floor classification accuracy: {floor_accuracy:.1f}%")
    print(f"  RMSE (given correct floor): {hier_rmse:.2f} m")
    
    # Method 3: Hierarchical (RF -> MAP)
    print("\n--- Method 3: Hierarchical (RF -> MAP) ---")
    hier_rf_errors = []
    for i, query in enumerate(queries):
        pred, info = hierarchical_localize(
            query,
            db,
            coarse_method="random_forest",
            fine_method="map"
        )
        hier_rf_errors.append(np.linalg.norm(pred - true_locs[i]))
    hier_rf_rmse = np.sqrt(np.mean(np.array(hier_rf_errors) ** 2))
    print(f"  RMSE: {hier_rf_rmse:.2f} m")
    
    # Method 4: Hierarchical (floor -> Posterior Mean)
    print("\n--- Method 4: Hierarchical (Floor -> Posterior Mean) ---")
    hier_pm_errors = []
    for i, query in enumerate(queries):
        pred, info = hierarchical_localize(
            query,
            db,
            coarse_method="floor",
            fine_method="posterior_mean",
            top_k=10
        )
        hier_pm_errors.append(np.linalg.norm(pred - true_locs[i]))
    hier_pm_rmse = np.sqrt(np.mean(np.array(hier_pm_errors) ** 2))
    print(f"  RMSE: {hier_pm_rmse:.2f} m")
    
    # Summary
    print("\n--- Summary ---")
    print(f"  Direct k-NN:               {direct_rmse:.2f} m")
    print(f"  Hierarchical (Floor -> kNN): {hier_rmse:.2f} m")
    print(f"  Hierarchical (RF -> MAP):    {hier_rf_rmse:.2f} m")
    print(f"  Hierarchical (Floor -> PM):  {hier_pm_rmse:.2f} m")
    
    # Plot error distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error CDFs
    ax = axes[0, 0]
    for errors, label in [
        (direct_errors, "Direct k-NN"),
        (hier_errors, "Hierarchical (Floor -> k-NN)"),
        (hier_rf_errors, "Hierarchical (RF -> MAP)"),
        (hier_pm_errors, "Hierarchical (Floor -> PM)"),
    ]:
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf, label=label, linewidth=2)
    ax.set_xlabel('Positioning Error (m)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Error Distribution (CDF)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Box plots
    ax = axes[0, 1]
    ax.boxplot(
        [direct_errors, hier_errors, hier_rf_errors, hier_pm_errors],
        labels=['Direct\nk-NN', 'Hier\nFloor->kNN', 'Hier\nRF->MAP', 'Hier\nFloor->PM'],
        showfliers=False
    )
    ax.set_ylabel('Positioning Error (m)', fontsize=11)
    ax.set_title('Error Distribution (Box Plot)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    ax = axes[1, 0]
    methods = ['Direct\nk-NN', 'Hier\nFloor->kNN', 'Hier\nRF->MAP', 'Hier\nFloor->PM']
    rmses = [direct_rmse, hier_rmse, hier_rf_rmse, hier_pm_rmse]
    bars = ax.bar(methods, rmses, color=['C0', 'C1', 'C2', 'C3'], alpha=0.7)
    ax.set_ylabel('RMSE (m)', fontsize=11)
    ax.set_title('RMSE Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    # Add values on bars
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{rmse:.2f}m',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Floor classification confusion matrix (for hierarchical methods)
    ax = axes[1, 1]
    ax.text(
        0.5, 0.6,
        f"Floor Classification\nAccuracy: {floor_accuracy:.1f}%",
        ha='center',
        va='center',
        fontsize=14,
        transform=ax.transAxes
    )
    ax.text(
        0.5, 0.4,
        f"Correct: {floor_correct}/{n_queries}",
        ha='center',
        va='center',
        fontsize=12,
        transform=ax.transAxes
    )
    ax.axis('off')
    
    plt.suptitle('Hierarchical Localization Performance', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    figs_dir = Path(__file__).parent / "figs"
    fig_path = figs_dir / "hierarchical_localization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  [OK] Saved figure: {fig_path}")
    
    plt.close()


def main():
    """Run all classification-based fingerprinting demonstrations."""
    print("=" * 70)
    print("Chapter 5: Classification-Based Fingerprinting")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Pattern recognition classifiers (Random Forest, SVM)")
    print("  2. Classification accuracy vs deterministic/probabilistic methods")
    print("  3. Hierarchical coarse-to-fine localization")
    
    # Create test database
    db = create_multifloor_test_database()
    
    # Test 1: Classification accuracy
    rf_classifier, svm_classifier = test_classification_accuracy(db)
    
    # Test 2: Robustness to noise
    test_noisy_queries(db, rf_classifier, svm_classifier)
    
    # Test 3: Hierarchical localization
    test_hierarchical_localization(db)
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - figs/classification_noise_robustness.png")
    print("  - figs/hierarchical_localization.png")


if __name__ == "__main__":
    main()

