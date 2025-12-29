"""LC vs TC Fusion Comparison Script (Chapter 8).

Runs both Loosely Coupled (LC) and Tightly Coupled (TC) fusion on the same
dataset and generates comprehensive comparison visualizations and metrics.

This script demonstrates the architectural trade-offs between LC and TC
fusion approaches discussed in Chapter 8.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1 (Loosely Coupled and Tightly Coupled)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from core.eval import compute_position_errors, compute_rmse
from ch8_sensor_fusion.lc_uwb_imu_ekf import load_fusion_dataset, run_lc_fusion
from ch8_sensor_fusion.tc_uwb_imu_ekf import run_tc_fusion


def run_both_fusions(
    dataset: Dict,
    use_gating: bool = True,
    gate_confidence: float = 0.95,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """Run both LC and TC fusion on the same dataset.
    
    Args:
        dataset: Dataset dictionary
        use_gating: Whether to apply chi-square gating
        gate_confidence: Gating confidence level (default 0.95 for 95% confidence)
        verbose: Print progress
    
    Returns:
        Tuple of (lc_results, tc_results)
    """
    if verbose:
        print("\n" + "="*70)
        print("Running LC vs TC Comparison")
        print("="*70)
    
    # Run LC fusion
    if verbose:
        print("\n[1/2] Running Loosely Coupled Fusion...")
    lc_results = run_lc_fusion(
        dataset,
        use_gating=use_gating,
        gate_confidence=gate_confidence,
        verbose=verbose
    )
    
    # Run TC fusion
    if verbose:
        print("\n[2/2] Running Tightly Coupled Fusion...")
    tc_results = run_tc_fusion(
        dataset,
        use_gating=use_gating,
        gate_confidence=gate_confidence,
        verbose=verbose
    )
    
    return lc_results, tc_results


def compute_comparative_metrics(
    dataset: Dict,
    lc_results: Dict,
    tc_results: Dict
) -> Dict:
    """Compute comparison metrics for LC and TC.
    
    Args:
        dataset: Dataset dictionary
        lc_results: LC fusion results
        tc_results: TC fusion results
    
    Returns:
        Dictionary with comparative metrics
    """
    truth = dataset['truth']
    
    # Interpolate truth to estimated timestamps
    def interpolate_truth(t_est):
        return np.column_stack([
            np.interp(t_est, truth['t'], truth['p_xy'][:, 0]),
            np.interp(t_est, truth['t'], truth['p_xy'][:, 1])
        ])
    
    # LC metrics
    p_true_lc = interpolate_truth(lc_results['t'])
    p_est_lc = lc_results['x_est'][:, :2]
    errors_lc = compute_position_errors(p_true_lc, p_est_lc)
    rmse_lc = compute_rmse(errors_lc)
    
    # TC metrics
    p_true_tc = interpolate_truth(tc_results['t'])
    p_est_tc = tc_results['x_est'][:, :2]
    errors_tc = compute_position_errors(p_true_tc, p_est_tc)
    rmse_tc = compute_rmse(errors_tc)
    
    metrics = {
        'lc': {
            'rmse_2d': rmse_lc,
            'rmse_x': np.sqrt(np.mean(errors_lc[:, 0]**2)),
            'rmse_y': np.sqrt(np.mean(errors_lc[:, 1]**2)),
            'max_error': np.max(np.linalg.norm(errors_lc, axis=1)),
            'mean_error': np.mean(np.linalg.norm(errors_lc, axis=1)),
            'final_error': np.linalg.norm(errors_lc[-1]),
            'n_updates': lc_results['n_uwb_accepted'],
            'n_rejected': lc_results['n_uwb_rejected'],
            'n_failed': lc_results['n_uwb_failed'],
            'acceptance_rate': (
                100 * lc_results['n_uwb_accepted'] / 
                (lc_results['n_uwb_accepted'] + lc_results['n_uwb_rejected'])
                if (lc_results['n_uwb_accepted'] + lc_results['n_uwb_rejected']) > 0 
                else 0.0
            ),
        },
        'tc': {
            'rmse_2d': rmse_tc,
            'rmse_x': np.sqrt(np.mean(errors_tc[:, 0]**2)),
            'rmse_y': np.sqrt(np.mean(errors_tc[:, 1]**2)),
            'max_error': np.max(np.linalg.norm(errors_tc, axis=1)),
            'mean_error': np.mean(np.linalg.norm(errors_tc, axis=1)),
            'final_error': np.linalg.norm(errors_tc[-1]),
            'n_updates': tc_results['n_uwb_accepted'],
            'n_rejected': tc_results['n_uwb_rejected'],
            'acceptance_rate': (
                100 * tc_results['n_uwb_accepted'] / 
                (tc_results['n_uwb_accepted'] + tc_results['n_uwb_rejected'])
                if (tc_results['n_uwb_accepted'] + tc_results['n_uwb_rejected']) > 0 
                else 0.0
            ),
        }
    }
    
    return metrics


def print_comparison_table(metrics: Dict) -> None:
    """Print comparison metrics table.
    
    Args:
        metrics: Dictionary with 'lc' and 'tc' metrics
    """
    print("\n" + "="*70)
    print("LC vs TC Performance Comparison")
    print("="*70)
    print(f"{'Metric':<25} {'LC Fusion':>15} {'TC Fusion':>15} {'Difference':>12}")
    print("-"*70)
    
    # Position accuracy
    lc = metrics['lc']
    tc = metrics['tc']
    
    print(f"{'RMSE 2D (m)':<25} {lc['rmse_2d']:>15.3f} {tc['rmse_2d']:>15.3f} "
          f"{lc['rmse_2d'] - tc['rmse_2d']:>+11.3f}")
    print(f"{'RMSE X (m)':<25} {lc['rmse_x']:>15.3f} {tc['rmse_x']:>15.3f} "
          f"{lc['rmse_x'] - tc['rmse_x']:>+11.3f}")
    print(f"{'RMSE Y (m)':<25} {lc['rmse_y']:>15.3f} {tc['rmse_y']:>15.3f} "
          f"{lc['rmse_y'] - tc['rmse_y']:>+11.3f}")
    print(f"{'Max Error (m)':<25} {lc['max_error']:>15.3f} {tc['max_error']:>15.3f} "
          f"{lc['max_error'] - tc['max_error']:>+11.3f}")
    print(f"{'Mean Error (m)':<25} {lc['mean_error']:>15.3f} {tc['mean_error']:>15.3f} "
          f"{lc['mean_error'] - tc['mean_error']:>+11.3f}")
    print(f"{'Final Error (m)':<25} {lc['final_error']:>15.3f} {tc['final_error']:>15.3f} "
          f"{lc['final_error'] - tc['final_error']:>+11.3f}")
    
    print("-"*70)
    
    # Update statistics
    print(f"{'UWB Updates Accepted':<25} {lc['n_updates']:>15d} {tc['n_updates']:>15d} "
          f"{lc['n_updates'] - tc['n_updates']:>+11d}")
    print(f"{'UWB Updates Rejected':<25} {lc['n_rejected']:>15d} {tc['n_rejected']:>15d} "
          f"{lc['n_rejected'] - tc['n_rejected']:>+11d}")
    if 'n_failed' in lc:
        print(f"{'LC Solver Failures':<25} {lc['n_failed']:>15d} {'N/A':>15} {'':>12}")
    print(f"{'Acceptance Rate (%)':<25} {lc['acceptance_rate']:>15.1f} {tc['acceptance_rate']:>15.1f} "
          f"{lc['acceptance_rate'] - tc['acceptance_rate']:>+11.1f}")
    
    print("="*70)
    
    # Summary
    better_rmse = "LC" if lc['rmse_2d'] < tc['rmse_2d'] else "TC"
    better_accept = "LC" if lc['acceptance_rate'] > tc['acceptance_rate'] else "TC"
    
    print(f"\nSummary:")
    print(f"  • {better_rmse} has lower RMSE ({abs(lc['rmse_2d'] - tc['rmse_2d']):.3f}m difference)")
    print(f"  • {better_accept} has higher acceptance rate "
          f"({abs(lc['acceptance_rate'] - tc['acceptance_rate']):.1f}% difference)")
    print(f"  • LC: {lc['n_updates']} updates, TC: {tc['n_updates']} updates "
          f"(TC has {tc['n_updates'] - lc['n_updates']:+d} more)")
    print()


def plot_comparison(
    dataset: Dict,
    lc_results: Dict,
    tc_results: Dict,
    metrics: Dict,
    save_path: str = None
) -> None:
    """Generate comprehensive LC vs TC comparison plots.
    
    Args:
        dataset: Dataset dictionary
        lc_results: LC fusion results
        tc_results: TC fusion results
        metrics: Comparison metrics
        save_path: Path to save figure
    """
    truth = dataset['truth']
    anchors = dataset['uwb_anchors']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    color_truth = 'black'
    color_lc = 'tab:blue'
    color_tc = 'tab:orange'
    
    # ========== Row 1: Trajectories ==========
    
    # 1. LC Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax1.plot(lc_results['x_est'][:, 0], lc_results['x_est'][:, 1],
            color=color_lc, linewidth=1.5, alpha=0.7, label='LC Estimate', zorder=2)
    if len(lc_results.get('uwb_positions', [])) > 0:
        ax1.scatter(lc_results['uwb_positions'][:, 0], lc_results['uwb_positions'][:, 1],
                   s=10, c='cyan', alpha=0.2, label='UWB Fixes', zorder=1)
    ax1.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='UWB Anchors', zorder=5)
    ax1.scatter(truth['p_xy'][0, 0], truth['p_xy'][0, 1], s=200, c='green',
               marker='*', edgecolors='darkgreen', linewidths=2, label='Start', zorder=4)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'LC Trajectory (RMSE: {metrics["lc"]["rmse_2d"]:.2f}m)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. TC Trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax2.plot(tc_results['x_est'][:, 0], tc_results['x_est'][:, 1],
            color=color_tc, linewidth=1.5, alpha=0.7, label='TC Estimate', zorder=2)
    ax2.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='UWB Anchors', zorder=5)
    ax2.scatter(truth['p_xy'][0, 0], truth['p_xy'][0, 1], s=200, c='green',
               marker='*', edgecolors='darkgreen', linewidths=2, label='Start', zorder=4)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title(f'TC Trajectory (RMSE: {metrics["tc"]["rmse_2d"]:.2f}m)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Overlay Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=4)
    ax3.plot(lc_results['x_est'][:, 0], lc_results['x_est'][:, 1],
            color=color_lc, linewidth=1.5, alpha=0.6, label='LC', zorder=3)
    ax3.plot(tc_results['x_est'][:, 0], tc_results['x_est'][:, 1],
            color=color_tc, linewidth=1.5, alpha=0.6, label='TC', zorder=2)
    ax3.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='Anchors', zorder=5)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('LC vs TC Overlay')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # ========== Row 2: Position Errors ==========
    
    # Compute errors
    def interpolate_truth(t_est):
        return np.column_stack([
            np.interp(t_est, truth['t'], truth['p_xy'][:, 0]),
            np.interp(t_est, truth['t'], truth['p_xy'][:, 1])
        ])
    
    p_true_lc = interpolate_truth(lc_results['t'])
    errors_lc = lc_results['x_est'][:, :2] - p_true_lc
    error_norm_lc = np.linalg.norm(errors_lc, axis=1)
    
    p_true_tc = interpolate_truth(tc_results['t'])
    errors_tc = tc_results['x_est'][:, :2] - p_true_tc
    error_norm_tc = np.linalg.norm(errors_tc, axis=1)
    
    # 4. LC Position Error
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(lc_results['t'], error_norm_lc, color=color_lc, linewidth=1)
    ax4.axhline(metrics['lc']['rmse_2d'], color='red', linestyle='--', 
               linewidth=1.5, label=f'RMSE: {metrics["lc"]["rmse_2d"]:.2f}m')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('LC Position Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. TC Position Error
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(tc_results['t'], error_norm_tc, color=color_tc, linewidth=1)
    ax5.axhline(metrics['tc']['rmse_2d'], color='red', linestyle='--',
               linewidth=1.5, label=f'RMSE: {metrics["tc"]["rmse_2d"]:.2f}m')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Position Error [m]')
    ax5.set_title('TC Position Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(lc_results['t'], error_norm_lc, color=color_lc, 
            linewidth=1, alpha=0.7, label='LC')
    ax6.plot(tc_results['t'], error_norm_tc, color=color_tc,
            linewidth=1, alpha=0.7, label='TC')
    ax6.axhline(metrics['lc']['rmse_2d'], color=color_lc, linestyle='--', linewidth=1)
    ax6.axhline(metrics['tc']['rmse_2d'], color=color_tc, linestyle='--', linewidth=1)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Position Error [m]')
    ax6.set_title('Position Error Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ========== Row 3: NIS and Statistics ==========
    
    # 7. LC NIS
    ax7 = fig.add_subplot(gs[2, 0])
    if len(lc_results['nis']) > 0:
        nis_lc = np.array(lc_results['nis'])
        accepted_lc = np.array(lc_results['gated'])
        ax7.plot(np.arange(len(nis_lc))[accepted_lc], nis_lc[accepted_lc],
                'g.', markersize=3, label='Accepted', alpha=0.5)
        if np.any(~accepted_lc):
            ax7.plot(np.arange(len(nis_lc))[~accepted_lc], nis_lc[~accepted_lc],
                    'rx', markersize=4, label='Rejected')
        
        # Chi-square bounds for m=2 DOF (position)
        from core.fusion import chi_square_bounds
        lower, upper = chi_square_bounds(dof=2, confidence=0.95)
        ax7.axhline(upper, color='r', linestyle='--', linewidth=1.5, label='95% bounds')
        ax7.axhline(lower, color='r', linestyle='--', linewidth=1.5)
        
        ax7.set_xlabel('UWB Update Index')
        ax7.set_ylabel('NIS (2 DOF)')
        ax7.set_title(f'LC NIS ({metrics["lc"]["acceptance_rate"]:.1f}% accepted)')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    
    # 8. TC NIS
    ax8 = fig.add_subplot(gs[2, 1])
    if len(tc_results['nis']) > 0:
        nis_tc = np.array(tc_results['nis'])
        accepted_tc = np.array(tc_results['gated'])
        ax8.plot(np.arange(len(nis_tc))[accepted_tc], nis_tc[accepted_tc],
                'g.', markersize=3, label='Accepted', alpha=0.5)
        if np.any(~accepted_tc):
            ax8.plot(np.arange(len(nis_tc))[~accepted_tc], nis_tc[~accepted_tc],
                    'rx', markersize=4, label='Rejected')
        
        # Chi-square bounds for m=1 DOF (range)
        from core.fusion import chi_square_bounds
        lower, upper = chi_square_bounds(dof=1, confidence=0.95)
        ax8.axhline(upper, color='r', linestyle='--', linewidth=1.5, label='95% bounds')
        ax8.axhline(lower, color='r', linestyle='--', linewidth=1.5)
        
        ax8.set_xlabel('UWB Update Index')
        ax8.set_ylabel('NIS (1 DOF)')
        ax8.set_title(f'TC NIS ({metrics["tc"]["acceptance_rate"]:.1f}% accepted)')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # 9. Metrics Comparison Bar Chart
    ax9 = fig.add_subplot(gs[2, 2])
    
    metric_names = ['RMSE\n[m]', 'Max Err\n[m]', 'Updates\n[×100]', 'Accept\n[%]']
    lc_values = [
        metrics['lc']['rmse_2d'],
        metrics['lc']['max_error'],
        metrics['lc']['n_updates'] / 100,
        metrics['lc']['acceptance_rate']
    ]
    tc_values = [
        metrics['tc']['rmse_2d'],
        metrics['tc']['max_error'],
        metrics['tc']['n_updates'] / 100,
        metrics['tc']['acceptance_rate']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax9.bar(x - width/2, lc_values, width, label='LC', color=color_lc, alpha=0.8)
    ax9.bar(x + width/2, tc_values, width, label='TC', color=color_tc, alpha=0.8)
    
    ax9.set_ylabel('Value')
    ax9.set_title('Performance Metrics Comparison')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metric_names, fontsize=9)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (lc_val, tc_val) in enumerate(zip(lc_values, tc_values)):
        ax9.text(i - width/2, lc_val, f'{lc_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax9.text(i + width/2, tc_val, f'{tc_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Loosely Coupled vs Tightly Coupled Fusion Comparison', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison figure: {save_path}")
    
    plt.show()


def save_comparison_report(
    dataset: Dict,
    lc_results: Dict,
    tc_results: Dict,
    metrics: Dict,
    output_path: str
) -> None:
    """Save comparison report to JSON.
    
    Args:
        dataset: Dataset dictionary
        lc_results: LC fusion results
        tc_results: TC fusion results
        metrics: Comparison metrics
        output_path: Path to save JSON report
    """
    report = {
        'dataset': {
            'path': str(dataset.get('path', 'unknown')),
            'config': dataset['config'],
            'n_imu_samples': len(dataset['imu']['t']),
            'n_uwb_epochs': len(dataset['uwb']['t']),
            'duration': float(dataset['truth']['t'][-1]),
        },
        'lc_fusion': metrics['lc'],
        'tc_fusion': metrics['tc'],
        'comparison': {
            'rmse_difference': float(metrics['lc']['rmse_2d'] - metrics['tc']['rmse_2d']),
            'better_rmse': 'LC' if metrics['lc']['rmse_2d'] < metrics['tc']['rmse_2d'] else 'TC',
            'update_ratio': float(metrics['lc']['n_updates'] / metrics['tc']['n_updates'])
                           if metrics['tc']['n_updates'] > 0 else 0.0,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved comparison report: {output_path}")


def main():
    """Main entry point for LC vs TC comparison."""
    parser = argparse.ArgumentParser(
        description="Compare Loosely Coupled vs Tightly Coupled IMU + UWB Fusion"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb",
        help="Path to fusion dataset directory"
    )
    parser.add_argument(
        "--no-gating",
        action="store_true",
        help="Disable chi-square gating"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Gating confidence level (default: 0.95 for 95%% confidence)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save comparison figure"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save comparison report (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)
    dataset['path'] = args.data
    
    # Run both fusions
    lc_results, tc_results = run_both_fusions(
        dataset,
        use_gating=not args.no_gating,
        gate_confidence=args.confidence,
        verbose=True
    )
    
    # Compute metrics
    metrics = compute_comparative_metrics(dataset, lc_results, tc_results)
    
    # Print comparison table
    print_comparison_table(metrics)
    
    # Save report if requested
    if args.report:
        save_comparison_report(dataset, lc_results, tc_results, metrics, args.report)
    
    # Generate comparison plots
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/lc_tc_comparison.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(dataset, lc_results, tc_results, metrics, save_path=save_path)


if __name__ == "__main__":
    main()

