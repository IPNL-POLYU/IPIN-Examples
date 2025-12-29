"""Tuning and Robust Loss Function Demo for Chapter 8.

Demonstrates filter tuning and robust measurement handling:
- Under-estimated R causes overconfidence and divergence
- Proper R estimation improves robustness
- Chi-square gating rejects outliers (Eq. 8.9)
- Robust loss functions (Huber, Cauchy) down-weight outliers (Eq. 8.7)

This demo uses the NLOS dataset where some UWB anchors have biased
measurements, demonstrating the need for robust estimation.

Key Equations:
- Eq. (8.5): Innovation y_k = z_k - h(x_k|k-1)
- Eq. (8.6): Innovation covariance S_k = H_k P_k|k-1 H_k^T + R_k
- Eq. (8.7): Robust R scaling: R_k <- w(y_k) * R_k
- Eq. (8.9): Chi-square gating: accept if d^2 < chi2(m, alpha)

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3 (Tuning of Sensor Fusion)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from core.eval import compute_position_errors, compute_rmse
from core.fusion import (
    chi_square_gate,
    huber_R_scale,
    cauchy_R_scale,
    innovation,
    innovation_covariance,
    mahalanobis_distance_squared,
)
from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset
from ch8_sensor_fusion.tc_models import (
    tc_process_model,
    tc_process_jacobian,
    tc_process_noise_covariance,
    tc_uwb_measurement_model,
    tc_uwb_measurement_jacobian,
)
from core.estimators import ExtendedKalmanFilter


def run_fusion_with_strategy(
    dataset: Dict,
    strategy: str = "baseline",
    R_scale: float = 1.0,
    use_gating: bool = False,
    gate_confidence: float = 0.95,
    robust_threshold: float = 2.0,
    verbose: bool = False
) -> Dict:
    """Run TC fusion with different tuning/robust strategies.
    
    Args:
        dataset: Dataset dictionary
        strategy: One of 'baseline', 'gating', 'huber', 'cauchy'
        R_scale: Scale factor for R (e.g., 0.5 = under-estimate, 2.0 = over-estimate)
        use_gating: Enable chi-square gating
        gate_confidence: Gating confidence level (default 0.95 for 95% confidence)
        robust_threshold: Threshold for robust loss (Huber/Cauchy)
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    if verbose:
        print(f"\nRunning fusion with strategy: {strategy.upper()}")
        print(f"  R scale: {R_scale}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
        if strategy in ['huber', 'cauchy']:
            print(f"  Robust threshold: {robust_threshold}")
    
    truth = dataset['truth']
    imu = dataset['imu']
    uwb = dataset['uwb']
    anchors = dataset['uwb_anchors']
    
    # Initial state: [px, py, vx, vy, yaw] (follows StateIndex convention)
    x0 = np.array([
        truth['p_xy'][0, 0],   # px
        truth['p_xy'][0, 1],   # py
        truth['v_xy'][0, 0],   # vx
        truth['v_xy'][0, 1],   # vy
        truth['yaw'][0]        # yaw
    ])
    
    # P0: covariances for [px, py, vx, vy, yaw]
    P0 = np.diag([0.1, 0.1, 0.5, 0.5, 0.1])**2
    
    # Process noise
    accel_noise_std = 0.1
    gyro_noise_std = 0.01
    
    # Measurement noise (scaled)
    uwb_range_noise_std = 0.05 * R_scale
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        process_model=tc_process_model,
        process_jacobian=tc_process_jacobian,
        measurement_model=lambda x: tc_uwb_measurement_model(x, anchors),
        measurement_jacobian=lambda x: tc_uwb_measurement_jacobian(x, anchors),
        Q=lambda dt: tc_process_noise_covariance(dt, accel_noise_std, gyro_noise_std),
        R=lambda: np.eye(4) * uwb_range_noise_std**2,
        x0=x0,
        P0=P0
    )
    
    # Prepare measurements
    from core.fusion import StampedMeasurement
    
    measurements: List[StampedMeasurement] = []
    
    # Add IMU
    for i in range(len(imu['t'])):
        measurements.append(StampedMeasurement(
            t=imu['t'][i],
            sensor='imu',
            z=np.hstack([imu['accel_xy'][i], imu['gyro_z'][i]]),
            R=np.eye(3),
            meta={}
        ))
    
    # Add UWB (per anchor)
    for i in range(len(uwb['t'])):
        for j in range(anchors.shape[0]):
            if not np.isnan(uwb['ranges'][i, j]):
                measurements.append(StampedMeasurement(
                    t=uwb['t'][i],
                    sensor='uwb',
                    z=np.array([uwb['ranges'][i, j]]),
                    R=np.array([[uwb_range_noise_std**2]]),
                    meta={'anchor_id': j, 'anchor_pos': anchors[j]}
                ))
    
    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)
    
    # Run fusion
    history = {
        't': [],
        'x_est': [],
        'P_trace': [],
        'innovations': [],
        'nis': [],
        'gated': [],
        'robust_scales': [],  # Renamed from robust_weights for clarity
    }
    
    n_uwb_accepted = 0
    n_uwb_rejected = 0
    t_prev = measurements[0].t
    
    for meas in measurements:
        dt = meas.t - t_prev
        
        if meas.sensor == 'imu':
            # Propagate
            u = meas.z
            ekf.predict(u=u, dt=dt)
        
        elif meas.sensor == 'uwb':
            # UWB range update
            anchor_id = meas.meta['anchor_id']
            anchor_pos = meas.meta['anchor_pos']
            
            # Predict range to this anchor
            state_pos = ekf.state[:2]
            z_pred = np.array([np.linalg.norm(state_pos - anchor_pos)])
            
            # Innovation
            y = innovation(meas.z, z_pred)
            
            # Jacobian for this anchor
            H_single = tc_uwb_measurement_jacobian(ekf.state, np.array([anchor_pos]))
            
            # Base R
            R_base = np.array([[uwb_range_noise_std**2]])
            
            # Apply robust covariance inflation if requested (Eq. 8.7)
            # Outliers get INFLATED covariance (R_scale >= 1)
            R_scale = 1.0
            if strategy == 'huber':
                R_scale = huber_R_scale(y[0], delta=robust_threshold)
                R_robust = R_scale * R_base  # Eq. 8.7: R <- w_R * R
            elif strategy == 'cauchy':
                R_scale = cauchy_R_scale(y[0], c=robust_threshold)
                R_robust = R_scale * R_base  # Eq. 8.7: R <- w_R * R
            else:
                R_robust = R_base
            
            # Innovation covariance
            S = innovation_covariance(H_single, ekf.covariance, R_robust)
            
            # Gating
            accept = True
            if use_gating or strategy == 'gating':
                accept = chi_square_gate(y, S, confidence=gate_confidence)
            
            if accept:
                # Perform update
                K = ekf.covariance @ H_single.T @ np.linalg.inv(S)
                ekf.state = ekf.state + (K @ y).flatten()
                ekf.covariance = (np.eye(5) - K @ H_single) @ ekf.covariance
                n_uwb_accepted += 1
            else:
                n_uwb_rejected += 1
            
            # Log
            history['innovations'].append(float(np.abs(y[0])))
            history['nis'].append(mahalanobis_distance_squared(y, S))
            history['gated'].append(accept)
            history['robust_scales'].append(R_scale)
        
        # Record state
        history['t'].append(meas.t)
        history['x_est'].append(ekf.state.copy())
        history['P_trace'].append(np.trace(ekf.covariance))
        
        t_prev = meas.t
    
    # Convert to arrays
    history['t'] = np.array(history['t'])
    history['x_est'] = np.array(history['x_est'])
    history['P_trace'] = np.array(history['P_trace'])
    history['n_uwb_accepted'] = n_uwb_accepted
    history['n_uwb_rejected'] = n_uwb_rejected
    
    if verbose:
        print(f"  Accepted: {n_uwb_accepted}")
        print(f"  Rejected: {n_uwb_rejected}")
        print(f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%")
    
    return history


def plot_tuning_comparison(
    dataset: Dict,
    baseline: Dict,
    gating: Dict,
    huber: Dict,
    cauchy: Dict,
    save_path: str = None
) -> None:
    """Generate tuning and robust loss comparison plots.
    
    Args:
        dataset: Dataset dictionary
        baseline: Baseline results (no gating, no robust)
        gating: Chi-square gating results
        huber: Huber robust loss results
        cauchy: Cauchy robust loss results
        save_path: Path to save figure
    """
    truth = dataset['truth']
    anchors = dataset['uwb_anchors']
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    color_truth = 'black'
    color_baseline = 'tab:red'
    color_gating = 'tab:blue'
    color_huber = 'tab:orange'
    color_cauchy = 'tab:green'
    
    # Helper function for errors
    def get_errors(history):
        p_true_interp = np.column_stack([
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
        ])
        errors = history['x_est'][:, :2] - p_true_interp
        return np.linalg.norm(errors, axis=1)
    
    # 1. Baseline trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Truth', zorder=3)
    ax1.plot(baseline['x_est'][:, 0], baseline['x_est'][:, 1],
            color=color_baseline, linewidth=1.5, alpha=0.7,
            label='Baseline (no gating)', zorder=2)
    ax1.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='Anchors', zorder=5)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Baseline (No Gating)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Gating trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Truth', zorder=3)
    ax2.plot(gating['x_est'][:, 0], gating['x_est'][:, 1],
            color=color_gating, linewidth=1.5, alpha=0.7,
            label='Chi-Square Gating', zorder=2)
    ax2.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='Anchors', zorder=5)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Chi-Square Gating')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Robust losses comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Truth', zorder=4)
    ax3.plot(huber['x_est'][:, 0], huber['x_est'][:, 1],
            color=color_huber, linewidth=1.5, alpha=0.7,
            label='Huber Loss', zorder=3)
    ax3.plot(cauchy['x_est'][:, 0], cauchy['x_est'][:, 1],
            color=color_cauchy, linewidth=1.5, alpha=0.7,
            label='Cauchy Loss', zorder=2)
    ax3.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='Anchors', zorder=5)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Robust Loss Functions')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. Position errors comparison
    ax4 = fig.add_subplot(gs[1, 0])
    error_baseline = get_errors(baseline)
    error_gating = get_errors(gating)
    error_huber = get_errors(huber)
    error_cauchy = get_errors(cauchy)
    
    ax4.plot(baseline['t'], error_baseline, color=color_baseline,
            linewidth=1, alpha=0.7, label='Baseline')
    ax4.plot(gating['t'], error_gating, color=color_gating,
            linewidth=1, alpha=0.7, label='Gating')
    ax4.plot(huber['t'], error_huber, color=color_huber,
            linewidth=1, alpha=0.7, label='Huber')
    ax4.plot(cauchy['t'], error_cauchy, color=color_cauchy,
            linewidth=1, alpha=0.7, label='Cauchy')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('Position Error Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. NIS comparison
    ax5 = fig.add_subplot(gs[1, 1])
    if len(baseline['nis']) > 0:
        nis_baseline = np.array(baseline['nis'])
        nis_gating = np.array(gating['nis'])
        
        # Downsample for visibility
        step = max(1, len(nis_baseline) // 500)
        ax5.plot(nis_baseline[::step], color=color_baseline,
                linewidth=0.5, alpha=0.5, label='Baseline')
        ax5.plot(nis_gating[::step], color=color_gating,
                linewidth=0.5, alpha=0.5, label='Gating')
        
        # Chi-square bound
        from core.fusion import chi_square_threshold
        threshold = chi_square_threshold(dof=1, confidence=0.95)
        ax5.axhline(threshold, color='r', linestyle='--',
                   linewidth=1.5, label=f'95% bound (chi^2={threshold:.2f})')
        
        ax5.set_xlabel('UWB Update Index')
        ax5.set_ylabel('NIS (1 DOF)')
        ax5.set_title('NIS: Baseline vs Gating')
        ax5.set_ylim([0, min(20, np.percentile(nis_baseline, 99))])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Robust covariance scales
    ax6 = fig.add_subplot(gs[1, 2])
    if len(huber['robust_scales']) > 0:
        scales_huber = np.array(huber['robust_scales'])
        scales_cauchy = np.array(cauchy['robust_scales'])
        
        step = max(1, len(scales_huber) // 500)
        ax6.plot(scales_huber[::step], color=color_huber,
                linewidth=0.5, alpha=0.7, label='Huber')
        ax6.plot(scales_cauchy[::step], color=color_cauchy,
                linewidth=0.5, alpha=0.7, label='Cauchy')
        
        ax6.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                   label='No inflation (inlier)')
        ax6.set_xlabel('UWB Update Index')
        ax6.set_ylabel('R Scale Factor w_R')
        ax6.set_title('Robust Covariance Inflation (Eq. 8.7): higher = more inflation')
        ax6.set_ylim([0.5, max(10, np.percentile(scales_cauchy, 95))])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. RMSE bar chart
    ax7 = fig.add_subplot(gs[2, 0])
    rmses = [
        compute_rmse(get_errors(baseline)),
        compute_rmse(get_errors(gating)),
        compute_rmse(get_errors(huber)),
        compute_rmse(get_errors(cauchy))
    ]
    methods = ['Baseline', 'Gating', 'Huber', 'Cauchy']
    colors = [color_baseline, color_gating, color_huber, color_cauchy]
    
    bars = ax7.bar(methods, rmses, color=colors, alpha=0.7)
    ax7.set_ylabel('RMSE [m]')
    ax7.set_title('RMSE Comparison')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}m', ha='center', va='bottom', fontsize=9)
    
    # 8. Acceptance rate
    ax8 = fig.add_subplot(gs[2, 1])
    acceptance_rates = [
        100 * baseline['n_uwb_accepted'] / (baseline['n_uwb_accepted'] + baseline['n_uwb_rejected']),
        100 * gating['n_uwb_accepted'] / (gating['n_uwb_accepted'] + gating['n_uwb_rejected']),
        100 * huber['n_uwb_accepted'] / (huber['n_uwb_accepted'] + huber['n_uwb_rejected']),
        100 * cauchy['n_uwb_accepted'] / (cauchy['n_uwb_accepted'] + cauchy['n_uwb_rejected'])
    ]
    
    bars = ax8.bar(methods, acceptance_rates, color=colors, alpha=0.7)
    ax8.set_ylabel('Acceptance Rate [%]')
    ax8.set_title('Measurement Acceptance Rate')
    ax8.set_ylim([0, 105])
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, acceptance_rates):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 9. Innovation distribution
    ax9 = fig.add_subplot(gs[2, 2])
    if len(baseline['innovations']) > 0:
        innov_baseline = np.array(baseline['innovations'])
        innov_gating = np.array(gating['innovations'])[np.array(gating['gated'])]
        
        ax9.hist(innov_baseline, bins=50, alpha=0.5, color=color_baseline,
                label='Baseline (all)', density=True)
        ax9.hist(innov_gating, bins=50, alpha=0.5, color=color_gating,
                label='Gating (accepted)', density=True)
        
        ax9.set_xlabel('Innovation Magnitude [m]')
        ax9.set_ylabel('Density')
        ax9.set_title('Innovation Distribution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    fig.suptitle('Tuning & Robust Loss Comparison (NLOS Dataset)',
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure: {save_path}")
    
    plt.show()


def main():
    """Main entry point for tuning and robust loss demo."""
    parser = argparse.ArgumentParser(
        description="Tuning and Robust Loss Demo (NLOS Dataset)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb_nlos",
        help="Path to NLOS dataset directory"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save results figure"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("\n" + "="*70)
    print("Tuning and Robust Loss Demo (Chapter 8)")
    print("="*70)
    print("\nKey Concepts (Eq. 8.7: R_k <- w_R * R_k):")
    print("  1. Baseline (no gating): Accepts all measurements (including outliers)")
    print("  2. Chi-square gating: Hard rejection based on Mahalanobis distance")
    print("  3. Huber loss: Soft inflation of R for outliers (w_R = |r|/delta for |r|>delta)")
    print("  4. Cauchy loss: Strong inflation of R for outliers (w_R = 1+(r/c)^2)")
    print("\n  Note: Outliers get INFLATED covariance (w_R >= 1), reducing their")
    print("        influence in the Kalman gain without complete rejection.")
    print("")
    
    # Load NLOS dataset
    print(f"Loading NLOS dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)
    
    print("\nDataset info:")
    print(f"  IMU samples: {len(dataset['imu']['t'])}")
    print(f"  UWB epochs: {len(dataset['uwb']['t'])}")
    print(f"  NLOS anchors: {dataset['config']['uwb']['nlos_anchors']}")
    print(f"  NLOS bias: {dataset['config']['uwb']['nlos_bias_m']}m")
    print("")
    
    # Run different strategies
    print("[1/4] Running baseline (no gating, no robust)...")
    baseline = run_fusion_with_strategy(
        dataset, strategy='baseline', use_gating=False, verbose=True
    )
    
    print("[2/4] Running with chi-square gating...")
    gating = run_fusion_with_strategy(
        dataset, strategy='gating', use_gating=True, gate_confidence=0.95, verbose=True
    )
    
    print("[3/4] Running with Huber robust loss...")
    huber = run_fusion_with_strategy(
        dataset, strategy='huber', robust_threshold=1.5, verbose=True
    )
    
    print("[4/4] Running with Cauchy robust loss...")
    cauchy = run_fusion_with_strategy(
        dataset, strategy='cauchy', robust_threshold=2.5, verbose=True
    )
    
    # Compute RMSE
    def compute_final_rmse(history):
        truth = dataset['truth']
        p_true_interp = np.column_stack([
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
        ])
        errors = history['x_est'][:, :2] - p_true_interp
        return compute_rmse(errors)
    
    rmse_baseline = compute_final_rmse(baseline)
    rmse_gating = compute_final_rmse(gating)
    rmse_huber = compute_final_rmse(huber)
    rmse_cauchy = compute_final_rmse(cauchy)
    
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Method':<25} {'RMSE [m]':>12} {'Accepted':>12} {'Rejected':>12}")
    print("-"*70)
    print(f"{'Baseline (no gating)':<25} {rmse_baseline:>12.3f} "
          f"{baseline['n_uwb_accepted']:>12d} {baseline['n_uwb_rejected']:>12d}")
    print(f"{'Chi-Square Gating':<25} {rmse_gating:>12.3f} "
          f"{gating['n_uwb_accepted']:>12d} {gating['n_uwb_rejected']:>12d}")
    print(f"{'Huber Loss':<25} {rmse_huber:>12.3f} "
          f"{huber['n_uwb_accepted']:>12d} {huber['n_uwb_rejected']:>12d}")
    print(f"{'Cauchy Loss':<25} {rmse_cauchy:>12.3f} "
          f"{cauchy['n_uwb_accepted']:>12d} {cauchy['n_uwb_rejected']:>12d}")
    print("="*70)
    
    best_method = min(
        [('Gating', rmse_gating), ('Huber', rmse_huber), ('Cauchy', rmse_cauchy)],
        key=lambda x: x[1]
    )[0]
    improvement = 100 * (rmse_baseline - min(rmse_gating, rmse_huber, rmse_cauchy)) / rmse_baseline
    
    print(f"\nKey Findings:")
    print(f"  * Best method: {best_method}")
    print(f"  * Improvement over baseline: {improvement:.1f}%")
    print(f"  * Gating rejects {gating['n_uwb_rejected']} outliers (hard rejection)")
    print(f"  * Robust losses inflate R for outliers (soft rejection via Eq. 8.7)")
    print(f"  * Huber: linear inflation, Cauchy: quadratic inflation")
    print("")
    
    # Plot
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/tuning_robust_demo.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_tuning_comparison(dataset, baseline, gating, huber, cauchy, save_path=save_path)


if __name__ == "__main__":
    main()

