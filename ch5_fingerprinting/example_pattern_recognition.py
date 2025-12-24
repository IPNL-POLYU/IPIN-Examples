"""
Example: Pattern Recognition Fingerprinting (Linear Regression)

Demonstrates linear regression-based fingerprinting from Chapter 5.

Treats positioning as supervised learning: learns mapping f: z → x
where z is RSS fingerprint and x is location.

Model: x̂ = Wz + b (linear transformation with ridge regression)

Author: Li-Ta Hsu
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.fingerprinting import (
    load_fingerprint_database,
    LinearRegressionLocalizer,
)


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


def main():
    """Run pattern recognition fingerprinting examples."""
    print("="*70)
    print("Chapter 5: Pattern Recognition (Linear Regression)")
    print("="*70)
    
    # Load database
    print("\n1. Loading fingerprint database...")
    db_path = Path("data/sim/ch5_wifi_fingerprint_grid")
    db = load_fingerprint_database(db_path)
    print(f"   Database: {db}")
    
    # Split train/test
    print("\n2. Splitting into train/test sets...")
    floor_id = 0
    train_db, test_db = split_train_test(db, test_ratio=0.3, floor_id=floor_id)
    
    print(f"   Floor {floor_id} - Train: {train_db.n_reference_points} RPs, "
          f"Test: {test_db.n_reference_points} RPs")
    
    # Train models with different regularization
    print("\n3. Training Linear Regression models...")
    print("   Model: x̂ = Wz + b (ridge regression)")
    
    reg_values = [0.0, 0.1, 1.0, 10.0]
    models = {}
    train_results = {}
    
    for reg_val in reg_values:
        print(f"\n   Training with regularization λ={reg_val}...")
        t_start = time.time()
        model = LinearRegressionLocalizer.fit(
            train_db, floor_id=floor_id, regularization=reg_val
        )
        t_end = time.time()
        
        models[reg_val] = model
        print(f"   Training time: {(t_end - t_start)*1000:.2f}ms")
        print(f"   Model: {model}")
        
        # Evaluate on train set
        train_result = evaluate_model(model, train_db, floor_id=floor_id)
        train_results[reg_val] = train_result
        print(f"   Train RMSE: {train_result['rmse']:.2f}m, R²={train_result['r2']:.3f}")
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_results = {}
    
    for reg_val in reg_values:
        model = models[reg_val]
        test_result = evaluate_model(model, test_db, floor_id=floor_id)
        test_results[reg_val] = test_result
        print(f"\n   λ={reg_val}: Test RMSE={test_result['rmse']:.2f}m, "
              f"R²={test_result['r2']:.3f}, Time={test_result['time_per_query_ms']:.3f}ms")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'λ':<10} {'Train RMSE':<15} {'Test RMSE':<15} {'Test R²':<12} {'Time (ms)':<12}")
    print("-"*70)
    
    for reg_val in reg_values:
        tr = train_results[reg_val]
        te = test_results[reg_val]
        print(f"{reg_val:<10.1f} {tr['rmse']:<15.2f} {te['rmse']:<15.2f} "
              f"{te['r2']:<12.3f} {te['time_per_query_ms']:<12.3f}")
    
    # Visualizations
    print("\n5. Generating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Weight matrix visualization
    ax1 = plt.subplot(2, 4, 1)
    model = models[1.0]  # Use λ=1.0 model
    im = ax1.imshow(model.weights, cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('AP Index')
    ax1.set_ylabel('Coordinate (x, y)')
    ax1.set_title('Learned Weight Matrix W')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['x', 'y'])
    plt.colorbar(im, ax=ax1, label='Weight')
    
    # Plot 2: Prediction vs Ground Truth
    ax2 = plt.subplot(2, 4, 2)
    mask = test_db.get_floor_mask(floor_id)
    test_features = test_db.features[mask]
    test_locs = test_db.locations[mask]
    pred_locs = model.predict_batch(test_features)
    
    ax2.scatter(test_locs[:, 0], pred_locs[:, 0], alpha=0.5, s=30, label='x')
    ax2.scatter(test_locs[:, 1], pred_locs[:, 1], alpha=0.5, s=30, label='y')
    lim_min = min(test_locs.min(), pred_locs.min())
    lim_max = max(test_locs.max(), pred_locs.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
    ax2.set_xlabel('True (m)')
    ax2.set_ylabel('Predicted (m)')
    ax2.set_title('Prediction vs Ground Truth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Spatial error distribution
    ax3 = plt.subplot(2, 4, 3)
    errors_2d = np.linalg.norm(pred_locs - test_locs, axis=1)
    scatter = ax3.scatter(test_locs[:, 0], test_locs[:, 1],
                         c=errors_2d, s=100, cmap='YlOrRd', alpha=0.8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Spatial Error Distribution')
    plt.colorbar(scatter, ax=ax3, label='Error (m)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Error CDF for different λ
    ax4 = plt.subplot(2, 4, 4)
    for reg_val in reg_values:
        errors = test_results[reg_val]['errors']
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax4.plot(sorted_errors, cdf, label=f'λ={reg_val}', linewidth=2)
    ax4.set_xlabel('Positioning Error (m)')
    ax4.set_ylabel('CDF')
    ax4.set_title('Error CDF for Different λ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 15)
    
    # Plot 5: Train vs Test RMSE
    ax5 = plt.subplot(2, 4, 5)
    train_rmse = [train_results[r]['rmse'] for r in reg_values]
    test_rmse = [test_results[r]['rmse'] for r in reg_values]
    x = np.arange(len(reg_values))
    width = 0.35
    ax5.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
    ax5.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
    ax5.set_xlabel('Regularization λ')
    ax5.set_ylabel('RMSE (m)')
    ax5.set_title('Train vs Test RMSE')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{r}' for r in reg_values])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: R² vs λ
    ax6 = plt.subplot(2, 4, 6)
    test_r2 = [test_results[r]['r2'] for r in reg_values]
    ax6.plot(reg_values, test_r2, 'o-', linewidth=2, markersize=8)
    ax6.set_xlabel('Regularization λ')
    ax6.set_ylabel('R² Score')
    ax6.set_title('Test R² vs Regularization')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Perfect')
    ax6.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
    ax6.legend()
    
    # Plot 7: Overfitting analysis
    ax7 = plt.subplot(2, 4, 7)
    train_rmse = np.array([train_results[r]['rmse'] for r in reg_values])
    test_rmse = np.array([test_results[r]['rmse'] for r in reg_values])
    overfit_gap = test_rmse - train_rmse
    ax7.plot(reg_values, overfit_gap, 'o-', linewidth=2, markersize=8, color='red')
    ax7.set_xlabel('Regularization λ')
    ax7.set_ylabel('Overfitting Gap (m)')
    ax7.set_title('Test RMSE - Train RMSE')
    ax7.set_xscale('log')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 8: Box plot of errors
    ax8 = plt.subplot(2, 4, 8)
    error_data = [test_results[r]['errors'] for r in reg_values]
    bp = ax8.boxplot(error_data, labels=[f'λ={r}' for r in reg_values], 
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax8.set_ylabel('Positioning Error (m)')
    ax8.set_title('Error Distribution by λ')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    output_file = figs_dir / "pattern_recognition_positioning.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    print("\nKey Findings:")
    print("  - Linear regression learns direct RSS→location mapping")
    print("  - Very fast prediction (single matrix multiplication)")
    print("  - Regularization (λ) prevents overfitting")
    print("  - λ=0: May overfit to training data")
    print("  - λ>0: Better generalization, smoother predictions")
    print("  - Optimal λ depends on data size and noise level")
    print("\nModel Details:")
    print("  - Linear model: x̂ = Wz + b")
    print("  - W: weight matrix (2×8 for 2D position, 8 APs)")
    print("  - b: bias vector (2,)")
    print("  - Training: Ridge regression (closed-form solution)")


if __name__ == "__main__":
    main()

