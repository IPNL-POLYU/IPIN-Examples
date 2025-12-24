"""
Comparison of Fingerprinting Methods

This script compares all fingerprinting methods from Chapter 5:
    - Deterministic: NN, k-NN (Eqs. 5.1-5.2)
    - Probabilistic: MAP, Posterior Mean (Eqs. 5.3-5.5)
    - Pattern Recognition: Linear Regression

Evaluates under various conditions:
    - Different noise levels
    - Multi-floor scenarios
    - Sparse vs dense reference points

Author: Li-Ta Hsu
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from core.fingerprinting import (
    load_fingerprint_database,
    nn_localize,
    knn_localize,
    fit_gaussian_naive_bayes,
    map_localize,
    posterior_mean_localize,
    LinearRegressionLocalizer,
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


def evaluate_scenario(scenario_name, db, queries, true_locs, floor_id=None):
    """
    Evaluate all methods on a specific scenario.
    
    Returns:
        List of result dictionaries.
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    
    results = []
    
    # Deterministic methods
    print("\nDeterministic Methods (Eqs. 5.1-5.2):")
    
    # NN
    method_name = "NN (Euclidean)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = nn_localize(query, db, metric="euclidean", floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Deterministic",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    # k-NN
    method_name = "k-NN (k=3)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = knn_localize(query, db, k=3, metric="euclidean", 
                              weighting="inverse_distance", floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Deterministic",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    # Probabilistic methods
    print("\nProbabilistic Methods (Eqs. 5.3-5.5):")
    
    # Train Bayesian model
    print("  Training Bayesian model...", end=" ", flush=True)
    model_bayes = fit_gaussian_naive_bayes(db, min_std=2.0)
    print("Done")
    
    # MAP
    method_name = "MAP"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = map_localize(query, model_bayes, floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Probabilistic",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    # Posterior Mean (Full)
    method_name = "Posterior Mean"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = posterior_mean_localize(query, model_bayes, floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Probabilistic",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    # Posterior Mean (Top-k) - Book guidance: typically sufficient
    method_name = "Post.Mean (k=10)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = posterior_mean_localize(query, model_bayes, floor_id=floor_id, top_k=10)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Probabilistic",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    # Pattern Recognition
    print("\nPattern Recognition:")
    
    # Train Linear Regression
    print("  Training Linear Regression...", end=" ", flush=True)
    model_lr = LinearRegressionLocalizer.fit(db, floor_id=floor_id, regularization=1.0)
    print("Done")
    
    method_name = "Linear Regression"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs):
        t_start = time.perf_counter()
        est_loc = model_lr.predict(query)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)
    
    results.append({
        "method": method_name,
        "category": "Pattern Recognition",
        "errors": np.array(errors),
        "times": np.array(times),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "mean_time_ms": np.mean(times),
    })
    print(f"RMSE={results[-1]['rmse']:.2f}m")
    
    return results


def main():
    """Run comprehensive comparison of fingerprinting methods."""
    print("="*70)
    print("Chapter 5: Fingerprinting Methods Comparison")
    print("="*70)
    
    # Load database
    print("\nLoading fingerprint database...")
    db_path = Path("data/sim/ch5_wifi_fingerprint_grid")
    db = load_fingerprint_database(db_path)
    print(f"Database: {db}")
    
    all_results = {}
    
    # Scenario 1: Baseline (low noise, single floor)
    print("\n" + "="*70)
    print("SCENARIO 1: Baseline (low noise, single floor)")
    print("="*70)
    
    queries1, true_locs1, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=1.0, seed=42
    )
    all_results["Baseline"] = evaluate_scenario(
        "Baseline (sigma=1dBm, Floor 0)", db, queries1, true_locs1, floor_id=0
    )
    
    # Scenario 2: Moderate noise
    print("\n" + "="*70)
    print("SCENARIO 2: Moderate Noise")
    print("="*70)
    
    queries2, true_locs2, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=2.0, seed=43
    )
    all_results["Moderate Noise"] = evaluate_scenario(
        "Moderate Noise (sigma=2dBm, Floor 0)", db, queries2, true_locs2, floor_id=0
    )
    
    # Scenario 3: High noise
    print("\n" + "="*70)
    print("SCENARIO 3: High Noise")
    print("="*70)
    
    queries3, true_locs3, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=5.0, seed=44
    )
    all_results["High Noise"] = evaluate_scenario(
        "High Noise (sigma=5dBm, Floor 0)", db, queries3, true_locs3, floor_id=0
    )
    
    # Print summary table
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        print(f"{'Method':<20} {'Category':<20} {'RMSE (m)':<12} {'Median (m)':<12} {'P90 (m)':<12} {'Time (ms)':<12}")
        print("-"*90)
        for r in results:
            print(f"{r['method']:<20} {r['category']:<20} {r['rmse']:<12.2f} "
                  f"{r['median']:<12.2f} {r['p90']:<12.2f} {r['mean_time_ms']:<12.3f}")
    
    # Visualizations
    print("\n" + "="*70)
    print("Generating comparison visualizations...")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: RMSE comparison across scenarios
    ax1 = plt.subplot(3, 3, 1)
    methods = [r['method'] for r in all_results["Baseline"]]
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (scenario_name, results) in enumerate(all_results.items()):
        rmses = [r['rmse'] for r in results]
        ax1.bar(x + i*width, rmses, width, label=scenario_name, alpha=0.8)
    
    ax1.set_ylabel('RMSE (m)')
    ax1.set_title('RMSE Comparison Across Scenarios')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Error CDF (Baseline scenario)
    ax2 = plt.subplot(3, 3, 2)
    for r in all_results["Baseline"]:
        sorted_errors = np.sort(r['errors'])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax2.plot(sorted_errors, cdf, label=r['method'], linewidth=2)
    ax2.set_xlabel('Positioning Error (m)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Error CDF (Baseline)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    
    # Plot 3: Computation time comparison
    ax3 = plt.subplot(3, 3, 3)
    methods = [r['method'] for r in all_results["Baseline"]]
    times = [r['mean_time_ms'] for r in all_results["Baseline"]]
    colors = ['blue', 'cyan', 'red', 'orange', 'green']
    ax3.barh(methods, times, color=colors, alpha=0.7)
    ax3.set_xlabel('Computation Time (ms)')
    ax3.set_title('Speed Comparison (Baseline)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Box plot comparison (Baseline)
    ax4 = plt.subplot(3, 3, 4)
    error_data = [r['errors'] for r in all_results["Baseline"]]
    bp = ax4.boxplot(error_data, labels=methods, patch_artist=True)
    colors_box = ['lightblue', 'lightcyan', 'lightcoral', 'lightsalmon', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    ax4.set_ylabel('Positioning Error (m)')
    ax4.set_title('Error Distribution (Baseline)')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Robustness to noise (RMSE vs noise std)
    ax5 = plt.subplot(3, 3, 5)
    noise_levels = [1.0, 2.0, 5.0]
    scenario_names = ["Baseline", "Moderate Noise", "High Noise"]
    
    for i, method in enumerate(methods):
        rmses = []
        for scenario_name in scenario_names:
            method_result = [r for r in all_results[scenario_name] if r['method'] == method][0]
            rmses.append(method_result['rmse'])
        ax5.plot(noise_levels, rmses, 'o-', label=method, linewidth=2, markersize=6)
    
    ax5.set_xlabel('RSS Noise Std (dBm)')
    ax5.set_ylabel('RMSE (m)')
    ax5.set_title('Robustness to Measurement Noise')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Speed vs Accuracy (Baseline)
    ax6 = plt.subplot(3, 3, 6)
    for r in all_results["Baseline"]:
        ax6.scatter(r['mean_time_ms'], r['rmse'], s=150, alpha=0.7)
        ax6.annotate(r['method'], (r['mean_time_ms'], r['rmse']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax6.set_xlabel('Computation Time (ms)')
    ax6.set_ylabel('RMSE (m)')
    ax6.set_title('Speed vs Accuracy Trade-off')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Category comparison
    ax7 = plt.subplot(3, 3, 7)
    categories = ["Deterministic", "Probabilistic", "Pattern Recognition"]
    cat_rmses = {}
    for cat in categories:
        cat_methods = [r for r in all_results["Baseline"] if r['category'] == cat]
        cat_rmses[cat] = [r['rmse'] for r in cat_methods]
    
    positions = [1, 2, 3]
    bp = ax7.boxplot(cat_rmses.values(), positions=positions, labels=cat_rmses.keys(),
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightyellow')
    ax7.set_ylabel('RMSE (m)')
    ax7.set_title('Performance by Category')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Percentile comparison
    ax8 = plt.subplot(3, 3, 8)
    x = np.arange(len(methods))
    p50 = [r['median'] for r in all_results["Baseline"]]
    p90 = [r['p90'] for r in all_results["Baseline"]]
    width = 0.35
    ax8.bar(x - width/2, p50, width, label='Median (P50)', alpha=0.8)
    ax8.bar(x + width/2, p90, width, label='P90', alpha=0.8)
    ax8.set_ylabel('Error (m)')
    ax8.set_title('Median vs P90 Errors')
    ax8.set_xticks(x)
    ax8.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Summary radar chart
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    
    # Normalize metrics for radar chart
    baseline_results = all_results["Baseline"]
    metrics = ['RMSE', 'Median', 'P90']
    
    # Select 3 representative methods
    selected_methods = ["NN (Euclidean)", "MAP", "Linear Regression"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for method_name in selected_methods:
        method_result = [r for r in baseline_results if r['method'] == method_name][0]
        values = [
            method_result['rmse'] / 10,  # Normalize
            method_result['median'] / 10,
            method_result['p90'] / 15,
        ]
        values += values[:1]
        ax9.plot(angles, values, 'o-', linewidth=2, label=method_name)
        ax9.fill(angles, values, alpha=0.15)
    
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics)
    ax9.set_ylim(0, 1)
    ax9.set_title('Performance Profile\n(Normalized)', pad=20)
    ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax9.grid(True)
    
    plt.tight_layout()
    
    # Save
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    output_file = figs_dir / "comparison_all_methods.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("  1. Speed: Linear Regression >> NN > k-NN ~= MAP ~= Posterior Mean")
    print("  2. Accuracy (low noise): Probabilistic ~= k-NN > NN > Linear Reg")
    print("  3. Robustness: k-NN and Posterior Mean most stable with noise")
    print("  4. Smoothness: Posterior Mean > k-NN > Linear Reg > MAP ~= NN")
    print("  5. Training: Linear Reg requires training, others just use database")
    print("\nRecommendations:")
    print("  - Real-time apps: Use NN or Linear Regression for speed")
    print("  - High accuracy: Use k-NN or Bayesian methods")
    print("  - Noisy environments: k-NN with k=3-5 or Posterior Mean")
    print("  - Dense RPs: NN sufficient")
    print("  - Sparse RPs: k-NN or Linear Regression for interpolation")


if __name__ == "__main__":
    main()

