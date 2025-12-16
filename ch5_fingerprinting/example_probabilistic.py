"""
Example: Probabilistic Fingerprinting (Bayesian Methods)

Demonstrates Bayesian fingerprinting using Gaussian Naive Bayes model
from Chapter 5.

Implements:
    - Gaussian Naive Bayes model fitting
    - Log-likelihood computation (Eq. 5.3): log p(z|x_i)
    - MAP estimation (Eq. 5.4): i* = argmax_i p(x_i|z)
    - Posterior mean estimation (Eq. 5.5): x̂ = Σ p(x_i|z) x_i

Author: Navigation Engineer
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.fingerprinting import (
    load_fingerprint_database,
    fit_gaussian_naive_bayes,
    map_localize,
    posterior_mean_localize,
    log_posterior,
)


def generate_test_queries(db, n_queries=100, floor_id=None, noise_std=0.0, seed=42):
    """Generate test query fingerprints."""
    np.random.seed(seed)
    
    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        rp_locs = db.locations[mask]
        rp_features = db.features[mask]
        floor_ids_out = np.full(n_queries, floor_id)
    else:
        rp_locs = db.locations
        rp_features = db.features
        floor_ids_out = np.random.choice(db.floor_list, n_queries)
    
    min_x, max_x = rp_locs[:, 0].min(), rp_locs[:, 0].max()
    min_y, max_y = rp_locs[:, 1].min(), rp_locs[:, 1].max()
    
    true_locs = np.column_stack([
        np.random.uniform(min_x, max_x, n_queries),
        np.random.uniform(min_y, max_y, n_queries),
    ])
    
    query_fingerprints = []
    
    for true_loc, fid in zip(true_locs, floor_ids_out):
        if floor_id is not None:
            dists = np.linalg.norm(rp_locs - true_loc, axis=1)
        else:
            floor_mask = db.floor_ids == fid
            floor_rps = db.locations[floor_mask]
            floor_features = db.features[floor_mask]
            dists = np.linalg.norm(floor_rps - true_loc, axis=1)
        
        k_nearest = min(4, len(dists))
        nearest_idx = np.argpartition(dists, k_nearest)[:k_nearest]
        weights = 1.0 / (dists[nearest_idx] + 1e-3)
        weights /= weights.sum()
        
        if floor_id is not None:
            query_fp = np.sum(weights[:, None] * rp_features[nearest_idx], axis=0)
        else:
            query_fp = np.sum(weights[:, None] * floor_features[nearest_idx], axis=0)
        
        if noise_std > 0:
            query_fp += np.random.randn(len(query_fp)) * noise_std
        
        query_fingerprints.append(query_fp)
    
    return np.array(query_fingerprints), true_locs, floor_ids_out


def evaluate_method(method_name, method_fn, queries, true_locs, **kwargs):
    """Evaluate a positioning method."""
    print(f"\n  Evaluating {method_name}...")
    
    errors = []
    times = []
    
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = method_fn(query, **kwargs)
        t_end = time.perf_counter()
        
        error = np.linalg.norm(est_loc - true_loc)
        errors.append(error)
        times.append((t_end - t_start) * 1000)
    
    errors = np.array(errors)
    times = np.array(times)
    
    results = {
        "method": method_name,
        "errors": errors,
        "times": times,
        "rmse": np.sqrt(np.mean(errors**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    }
    
    print(f"    RMSE: {results['rmse']:.2f}m")
    print(f"    Median: {results['median']:.2f}m")
    print(f"    P90: {results['p90']:.2f}m")
    print(f"    Avg time: {results['mean_time_ms']:.3f}ms")
    
    return results


def visualize_posterior(model, query, true_loc, floor_id, ax, title):
    """Visualize posterior probability map."""
    # Get floor RPs
    mask = model.get_floor_mask(floor_id)
    rp_locs = model.locations[mask]
    
    # Compute posterior at each RP
    log_post = log_posterior(query, model, floor_id=floor_id)
    posteriors = np.exp(log_post[mask])
    
    # Create grid for visualization
    x_min, x_max = rp_locs[:, 0].min(), rp_locs[:, 0].max()
    y_min, y_max = rp_locs[:, 1].min(), rp_locs[:, 1].max()
    
    # Scatter plot with posterior as color
    scatter = ax.scatter(rp_locs[:, 0], rp_locs[:, 1], 
                        c=posteriors, s=100, cmap='hot', alpha=0.8,
                        vmin=0, vmax=posteriors.max())
    
    # Mark estimates
    x_map = map_localize(query, model, floor_id=floor_id)
    x_post_mean = posterior_mean_localize(query, model, floor_id=floor_id)
    
    ax.scatter(*x_map, marker='s', s=200, c='blue', edgecolors='white', 
              linewidth=2, label='MAP', zorder=10)
    ax.scatter(*x_post_mean, marker='^', s=200, c='green', edgecolors='white',
              linewidth=2, label='Post. Mean', zorder=10)
    ax.scatter(*true_loc, marker='*', s=300, c='yellow', edgecolors='black',
              linewidth=2, label='True', zorder=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.colorbar(scatter, ax=ax, label='p(x_i|z)')


def main():
    """Run probabilistic fingerprinting examples."""
    print("="*70)
    print("Chapter 5: Probabilistic Fingerprinting (Bayesian Methods)")
    print("="*70)
    
    # Load database
    print("\n1. Loading fingerprint database...")
    db_path = Path("data/sim/ch5_wifi_fingerprint_grid")
    db = load_fingerprint_database(db_path)
    print(f"   Database: {db}")
    
    # Train Bayesian models with different std values
    print("\n2. Training Gaussian Naive Bayes models...")
    print("   (Fitting Gaussian distributions per RP per AP)")
    
    std_values = [1.0, 2.0, 5.0]
    models = {}
    
    for std_val in std_values:
        print(f"\n   Training model with std={std_val} dBm...")
        t_start = time.time()
        model = fit_gaussian_naive_bayes(db, min_std=std_val)
        t_end = time.time()
        models[std_val] = model
        print(f"   Training time: {(t_end - t_start)*1000:.2f}ms")
        print(f"   Model: {model.n_reference_points} RPs, {model.n_features} features")
    
    # Generate test queries
    print("\n3. Generating test queries...")
    n_queries = 200
    floor_id = 0
    noise_std = 2.0
    
    queries, true_locs, floor_ids = generate_test_queries(
        db, n_queries=n_queries, floor_id=floor_id, noise_std=noise_std
    )
    print(f"   Generated {n_queries} test queries on floor {floor_id}")
    print(f"   RSS noise std: {noise_std} dBm")
    
    # Evaluate methods
    print("\n4. Evaluating probabilistic methods...")
    print("   (Equations 5.3, 5.4, 5.5 from Chapter 5)")
    
    results = []
    
    for std_val in std_values:
        model = models[std_val]
        
        # MAP
        results.append(evaluate_method(
            f"MAP (std={std_val}dBm)",
            map_localize,
            queries, true_locs,
            model=model, floor_id=floor_id
        ))
        
        # Posterior Mean
        results.append(evaluate_method(
            f"Post.Mean (std={std_val}dBm)",
            posterior_mean_localize,
            queries, true_locs,
            model=model, floor_id=floor_id
        ))
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<28} {'RMSE (m)':<12} {'Median (m)':<12} {'P90 (m)':<12} {'Time (ms)':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['method']:<28} {r['rmse']:<12.2f} {r['median']:<12.2f} "
              f"{r['p90']:<12.2f} {r['mean_time_ms']:<12.3f}")
    
    # Visualizations
    print("\n5. Generating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1-3: Posterior probability maps for different std
    for idx, std_val in enumerate(std_values):
        ax = plt.subplot(3, 3, idx + 1)
        visualize_posterior(models[std_val], queries[0], true_locs[0], 
                          floor_id, ax, f'Posterior Map (std={std_val}dBm)')
    
    # Plot 4: Error CDF comparison
    ax4 = plt.subplot(3, 3, 4)
    for r in results:
        sorted_errors = np.sort(r['errors'])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax4.plot(sorted_errors, cdf, label=r['method'], linewidth=2)
    ax4.set_xlabel('Positioning Error (m)')
    ax4.set_ylabel('CDF')
    ax4.set_title('Cumulative Distribution of Errors')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 20)
    
    # Plot 5: Box plot comparison
    ax5 = plt.subplot(3, 3, 5)
    error_data = [r['errors'] for r in results]
    method_names = [r['method'].replace(' (std=', '\n(').replace('dBm)', ')') for r in results]
    bp = ax5.boxplot(error_data, labels=method_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax5.set_ylabel('Positioning Error (m)')
    ax5.set_title('Error Distribution by Method')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: RMSE vs std
    ax6 = plt.subplot(3, 3, 6)
    map_results = [r for r in results if 'MAP' in r['method']]
    pm_results = [r for r in results if 'Post.Mean' in r['method']]
    
    ax6.plot(std_values, [r['rmse'] for r in map_results], 'o-', 
            linewidth=2, markersize=8, label='MAP')
    ax6.plot(std_values, [r['rmse'] for r in pm_results], 's-',
            linewidth=2, markersize=8, label='Posterior Mean')
    ax6.set_xlabel('Model Std (dBm)')
    ax6.set_ylabel('RMSE (m)')
    ax6.set_title('Effect of Model Uncertainty (std)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: MAP vs Posterior Mean scatter
    ax7 = plt.subplot(3, 3, 7)
    map_rmse = [r['rmse'] for r in map_results]
    pm_rmse = [r['rmse'] for r in pm_results]
    ax7.scatter(map_rmse, pm_rmse, s=150, alpha=0.7)
    for i, std in enumerate(std_values):
        ax7.annotate(f'std={std}', (map_rmse[i], pm_rmse[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax7.plot([min(map_rmse), max(map_rmse)], [min(map_rmse), max(map_rmse)],
            'k--', alpha=0.5, label='x=y')
    ax7.set_xlabel('MAP RMSE (m)')
    ax7.set_ylabel('Posterior Mean RMSE (m)')
    ax7.set_title('MAP vs Posterior Mean Accuracy')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')
    
    # Plot 8: Computation time comparison
    ax8 = plt.subplot(3, 3, 8)
    map_times = [r['mean_time_ms'] for r in map_results]
    pm_times = [r['mean_time_ms'] for r in pm_results]
    x = np.arange(len(std_values))
    width = 0.35
    ax8.bar(x - width/2, map_times, width, label='MAP', alpha=0.8)
    ax8.bar(x + width/2, pm_times, width, label='Posterior Mean', alpha=0.8)
    ax8.set_xlabel('Model Std (dBm)')
    ax8.set_ylabel('Computation Time (ms)')
    ax8.set_title('Computation Time Comparison')
    ax8.set_xticks(x)
    ax8.set_xticklabels([f'{s}' for s in std_values])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Example posterior distribution
    ax9 = plt.subplot(3, 3, 9)
    model = models[2.0]  # Use std=2.0 model
    query = queries[0]
    log_post = log_posterior(query, model, floor_id=floor_id)
    mask = model.get_floor_mask(floor_id)
    posteriors = np.exp(log_post[mask])
    
    # Sort and plot top 20 RPs
    sorted_idx = np.argsort(posteriors)[::-1][:20]
    ax9.bar(range(len(sorted_idx)), posteriors[sorted_idx])
    ax9.set_xlabel('RP Index (sorted by posterior)')
    ax9.set_ylabel('Posterior Probability')
    ax9.set_title('Posterior Distribution (Top 20 RPs)')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_file = Path("ch5_fingerprinting/probabilistic_positioning.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    print("\nKey Findings:")
    print("  - MAP provides discrete estimates (selects one RP)")
    print("  - Posterior Mean provides smooth estimates (weighted average)")
    print("  - Model std parameter controls uncertainty/smoothness trade-off")
    print("  - Larger std = more smooth but potentially less accurate")
    print("  - Smaller std = sharper posterior but sensitive to noise")
    print("\nReferences:")
    print("  - Equation 5.3: Log-likelihood log p(z|x_i)")
    print("  - Equation 5.4: MAP estimate")
    print("  - Equation 5.5: Posterior mean estimate")


if __name__ == "__main__":
    main()

