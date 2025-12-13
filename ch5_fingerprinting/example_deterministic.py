"""
Example: Deterministic Fingerprinting (NN and k-NN)

Demonstrates nearest-neighbor (NN) and k-nearest-neighbor (k-NN)
fingerprinting methods from Chapter 5.

Implements:
    - NN positioning (Eq. 5.1): i* = argmin_i D(z, f_i)
    - k-NN positioning (Eq. 5.2): x̂ = Σ w_i x_i / Σ w_i

Author: Navigation Engineer  
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.fingerprinting import (
    load_fingerprint_database,
    nn_localize,
    knn_localize,
)


def generate_test_queries(db, n_queries=100, floor_id=None, noise_std=0.0, seed=42):
    """
    Generate test query fingerprints.
    
    Args:
        db: FingerprintDatabase.
        n_queries: Number of test queries.
        floor_id: Floor to generate queries on (None = random floors).
        noise_std: RSS measurement noise std (dBm).
        seed: Random seed.
    
    Returns:
        Tuple of (query_fingerprints, true_locations, floor_ids).
    """
    np.random.seed(seed)
    
    if floor_id is not None:
        # Single floor
        mask = db.get_floor_mask(floor_id)
        rp_locs = db.locations[mask]
        rp_features = db.features[mask]
        floor_ids_out = np.full(n_queries, floor_id)
    else:
        # All floors
        rp_locs = db.locations
        rp_features = db.features
        floor_ids_out = np.random.choice(db.floor_list, n_queries)
    
    # Generate random locations within convex hull of RPs
    min_x, max_x = rp_locs[:, 0].min(), rp_locs[:, 0].max()
    min_y, max_y = rp_locs[:, 1].min(), rp_locs[:, 1].max()
    
    true_locs = np.column_stack([
        np.random.uniform(min_x, max_x, n_queries),
        np.random.uniform(min_y, max_y, n_queries),
    ])
    
    # Generate fingerprints by interpolating from nearby RPs
    query_fingerprints = []
    
    for i, (true_loc, fid) in enumerate(zip(true_locs, floor_ids_out)):
        # Find k nearest RPs for interpolation
        if floor_id is not None:
            dists = np.linalg.norm(rp_locs - true_loc, axis=1)
        else:
            floor_mask = db.floor_ids == fid
            floor_rps = db.locations[floor_mask]
            floor_features = db.features[floor_mask]
            dists = np.linalg.norm(floor_rps - true_loc, axis=1)
        
        k_nearest = min(4, len(dists))
        nearest_idx = np.argpartition(dists, k_nearest)[:k_nearest]
        
        # Weighted average of nearby RPs' RSS
        weights = 1.0 / (dists[nearest_idx] + 1e-3)
        weights /= weights.sum()
        
        if floor_id is not None:
            query_fp = np.sum(weights[:, None] * rp_features[nearest_idx], axis=0)
        else:
            query_fp = np.sum(weights[:, None] * floor_features[nearest_idx], axis=0)
        
        # Add measurement noise
        if noise_std > 0:
            query_fp += np.random.randn(len(query_fp)) * noise_std
        
        query_fingerprints.append(query_fp)
    
    return np.array(query_fingerprints), true_locs, floor_ids_out


def evaluate_positioning_method(method_name, method_fn, queries, true_locs, **kwargs):
    """
    Evaluate a positioning method.
    
    Args:
        method_name: Name of method.
        method_fn: Positioning function.
        queries: Query fingerprints, shape (N, n_features).
        true_locs: True locations, shape (N, 2).
        **kwargs: Additional arguments for method_fn.
    
    Returns:
        Dictionary with errors, computation time, etc.
    """
    print(f"\n  Evaluating {method_name}...")
    
    errors = []
    times = []
    
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = method_fn(query, **kwargs)
        t_end = time.perf_counter()
        
        error = np.linalg.norm(est_loc - true_loc)
        errors.append(error)
        times.append((t_end - t_start) * 1000)  # ms
    
    errors = np.array(errors)
    times = np.array(times)
    
    results = {
        "method": method_name,
        "errors": errors,
        "times": times,
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "p50": np.percentile(errors, 50),
        "p90": np.percentile(errors, 90),
        "p95": np.percentile(errors, 95),
        "mean_time_ms": np.mean(times),
    }
    
    print(f"    RMSE: {results['rmse']:.2f}m")
    print(f"    Median: {results['median_error']:.2f}m")
    print(f"    90th percentile: {results['p90']:.2f}m")
    print(f"    Avg time: {results['mean_time_ms']:.3f}ms")
    
    return results


def main():
    """Run deterministic fingerprinting examples."""
    print("="*70)
    print("Chapter 5: Deterministic Fingerprinting (NN and k-NN)")
    print("="*70)
    
    # Load database
    print("\n1. Loading fingerprint database...")
    db_path = Path("data/sim/wifi_fingerprint_grid")
    db = load_fingerprint_database(db_path)
    
    print(f"   Database: {db}")
    print(f"   Location range: x=[{db.locations[:, 0].min():.1f}, {db.locations[:, 0].max():.1f}]m, "
          f"y=[{db.locations[:, 1].min():.1f}, {db.locations[:, 1].max():.1f}]m")
    
    # Generate test queries
    print("\n2. Generating test queries...")
    n_queries = 200
    floor_id = 0  # Test on floor 0
    noise_std = 2.0  # 2 dBm measurement noise
    
    queries, true_locs, floor_ids = generate_test_queries(
        db, n_queries=n_queries, floor_id=floor_id, noise_std=noise_std
    )
    
    print(f"   Generated {n_queries} test queries on floor {floor_id}")
    print(f"   RSS noise std: {noise_std} dBm")
    
    # Evaluate methods
    print("\n3. Evaluating positioning methods...")
    print("   (Equations 5.1 and 5.2 from Chapter 5)")
    
    results = []
    
    # NN - Euclidean
    results.append(evaluate_positioning_method(
        "NN (Euclidean)",
        nn_localize,
        queries, true_locs,
        db=db, metric="euclidean", floor_id=floor_id
    ))
    
    # NN - Manhattan
    results.append(evaluate_positioning_method(
        "NN (Manhattan)",
        nn_localize,
        queries, true_locs,
        db=db, metric="manhattan", floor_id=floor_id
    ))
    
    # k-NN with varying k
    for k in [3, 5, 7]:
        results.append(evaluate_positioning_method(
            f"k-NN (k={k}, inv-dist)",
            knn_localize,
            queries, true_locs,
            db=db, k=k, metric="euclidean", weighting="inverse_distance", floor_id=floor_id
        ))
    
    # k-NN uniform weights
    results.append(evaluate_positioning_method(
        "k-NN (k=5, uniform)",
        knn_localize,
        queries, true_locs,
        db=db, k=5, metric="euclidean", weighting="uniform", floor_id=floor_id
    ))
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'RMSE (m)':<12} {'Median (m)':<12} {'90th % (m)':<12} {'Time (ms)':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['method']:<25} {r['rmse']:<12.2f} {r['median_error']:<12.2f} "
              f"{r['p90']:<12.2f} {r['mean_time_ms']:<12.3f}")
    
    # Visualize results
    print("\n4. Generating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Reference points and test queries
    ax1 = plt.subplot(2, 3, 1)
    floor_mask = db.get_floor_mask(floor_id)
    ax1.scatter(db.locations[floor_mask, 0], db.locations[floor_mask, 1],
                c='blue', marker='s', s=50, alpha=0.6, label='Reference Points')
    ax1.scatter(true_locs[:50, 0], true_locs[:50, 1],
                c='red', marker='x', s=30, alpha=0.8, label='Test Queries (sample)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Reference Points & Test Queries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Error CDF
    ax2 = plt.subplot(2, 3, 2)
    for r in results:
        sorted_errors = np.sort(r['errors'])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax2.plot(sorted_errors, cdf, label=r['method'], linewidth=2)
    ax2.set_xlabel('Positioning Error (m)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution of Errors')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(20, sorted_errors.max()))
    
    # Plot 3: Error histogram
    ax3 = plt.subplot(2, 3, 3)
    for i, r in enumerate(results[:3]):  # Show first 3 methods
        ax3.hist(r['errors'], bins=30, alpha=0.5, label=r['method'])
    ax3.set_xlabel('Positioning Error (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution (First 3 Methods)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Box plot comparison
    ax4 = plt.subplot(2, 3, 4)
    error_data = [r['errors'] for r in results]
    method_names = [r['method'] for r in results]
    bp = ax4.boxplot(error_data, labels=method_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_ylabel('Positioning Error (m)')
    ax4.set_title('Error Distribution by Method')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Plot 5: RMSE vs k for k-NN
    ax5 = plt.subplot(2, 3, 5)
    knn_results = [r for r in results if 'k-NN' in r['method'] and 'inv-dist' in r['method']]
    k_values = [int(r['method'].split('k=')[1].split(',')[0]) for r in knn_results]
    rmse_values = [r['rmse'] for r in knn_results]
    ax5.plot(k_values, rmse_values, 'o-', linewidth=2, markersize=8)
    ax5.set_xlabel('k (Number of Neighbors)')
    ax5.set_ylabel('RMSE (m)')
    ax5.set_title('Effect of k on k-NN Performance')
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(k_values)
    
    # Plot 6: Speed vs Accuracy
    ax6 = plt.subplot(2, 3, 6)
    for r in results:
        ax6.scatter(r['mean_time_ms'], r['rmse'], s=100, alpha=0.7)
        ax6.annotate(r['method'], (r['mean_time_ms'], r['rmse']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    ax6.set_xlabel('Computation Time (ms)')
    ax6.set_ylabel('RMSE (m)')
    ax6.set_title('Speed vs Accuracy Trade-off')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path("ch5_fingerprinting/deterministic_positioning.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    print("\nKey Findings:")
    print("  - NN methods are fast but can have discrete jumps")
    print("  - k-NN smooths estimates by averaging k nearest neighbors")
    print("  - Inverse distance weighting performs better than uniform weights")
    print("  - Optimal k depends on RP density and noise level")
    print("  - Manhattan distance can be faster than Euclidean in some cases")
    print("\nReferences:")
    print("  - Equation 5.1: NN decision rule")
    print("  - Equation 5.2: k-NN weighted average")


if __name__ == "__main__":
    main()

