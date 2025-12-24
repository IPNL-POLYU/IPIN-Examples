"""Temporal Calibration Demo for Chapter 8.

Demonstrates the importance of temporal calibration in sensor fusion:
- Time offsets between sensors cause fusion errors
- Clock drift compounds the problem over time
- TimeSyncModel corrects for offset and drift
- Proper temporal alignment restores accuracy

This demo uses the time-offset dataset where IMU and UWB have a 50ms
time offset and 100ppm clock drift.

Key Concepts:
- Temporal misalignment: Sensors have different time bases
- Time offset: Constant shift between sensor clocks
- Clock drift: Relative rate difference between clocks
- Time synchronization: t_fusion = (1 + drift) * t_sensor + offset

Author: Li-Ta Hsu
References: Chapter 8, Section on Temporal Calibration
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from core.eval import compute_position_errors, compute_rmse
from core.fusion import StampedMeasurement, TimeSyncModel
from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset
from ch8_sensor_fusion.tc_models import (
    tc_process_model,
    tc_process_jacobian,
    tc_process_noise_covariance,
    tc_uwb_measurement_model,
    tc_uwb_measurement_jacobian,
)
from core.estimators import ExtendedKalmanFilter


def run_fusion_with_time_sync(
    dataset: Dict,
    apply_correction: bool = False,
    use_gating: bool = True,
    gate_alpha: float = 0.05,
    verbose: bool = False
) -> Dict:
    """Run TC fusion with or without temporal calibration.
    
    Args:
        dataset: Dataset dictionary
        apply_correction: Whether to apply TimeSyncModel correction
        use_gating: Enable chi-square gating
        gate_alpha: Gating significance level
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    if verbose:
        print(f"\nRunning fusion:")
        print(f"  Temporal correction: {'ENABLED' if apply_correction else 'DISABLED'}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
    
    truth = dataset['truth']
    imu = dataset['imu']
    uwb = dataset['uwb']
    anchors = dataset['uwb_anchors']
    config = dataset['config']
    
    # Get time offset and drift from config
    time_offset = config['temporal_calibration']['time_offset_sec']
    clock_drift = config['temporal_calibration']['clock_drift']
    
    if verbose and apply_correction:
        print(f"  Time offset: {time_offset*1000:.1f} ms")
        print(f"  Clock drift: {clock_drift*1e6:.1f} ppm")
    
    # Create TimeSyncModel for UWB (IMU is reference)
    if apply_correction:
        uwb_time_sync = TimeSyncModel(offset=time_offset, drift=clock_drift)
    else:
        uwb_time_sync = TimeSyncModel(offset=0.0, drift=0.0)
    
    # Initial state
    x0 = np.array([
        truth['p_xy'][0, 0],
        truth['p_xy'][0, 1],
        truth['yaw'][0],
        truth['v_xy'][0, 0],
        truth['v_xy'][0, 1]
    ])
    
    P0 = np.diag([0.1, 0.1, 0.1, 0.5, 0.5])**2
    
    # Process and measurement noise
    accel_noise_std = 0.1
    gyro_noise_std = 0.01
    uwb_range_noise_std = 0.05
    
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
    measurements: List[StampedMeasurement] = []
    
    # Add IMU (reference time)
    for i in range(len(imu['t'])):
        measurements.append(StampedMeasurement(
            t=imu['t'][i],
            sensor='imu',
            z=np.hstack([imu['accel_xy'][i], imu['gyro_z'][i]]),
            R=np.eye(3),
            meta={}
        ))
    
    # Add UWB (apply time sync correction if requested)
    for i in range(len(uwb['t'])):
        # Convert UWB sensor time to fusion time
        if apply_correction:
            t_fusion = uwb_time_sync.to_fusion_time(uwb['t'][i])
        else:
            t_fusion = uwb['t'][i]  # Use raw (incorrect) time
        
        for j in range(anchors.shape[0]):
            if not np.isnan(uwb['ranges'][i, j]):
                measurements.append(StampedMeasurement(
                    t=t_fusion,
                    sensor='uwb',
                    z=np.array([uwb['ranges'][i, j]]),
                    R=np.array([[uwb_range_noise_std**2]]),
                    meta={'anchor_id': j, 'anchor_pos': anchors[j]}
                ))
    
    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)
    
    # Run fusion
    from core.fusion import chi_square_gate, innovation, innovation_covariance
    
    history = {
        't': [],
        'x_est': [],
        'P_trace': [],
        'innovations': [],
        'nis': [],
        'gated': [],
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
            
            # Predict range
            state_pos = ekf.state[:2]
            z_pred = np.array([np.linalg.norm(state_pos - anchor_pos)])
            
            # Innovation
            y = innovation(meas.z, z_pred)
            
            # Jacobian
            H_single = tc_uwb_measurement_jacobian(ekf.state, np.array([anchor_pos]))
            
            # Innovation covariance
            R = np.array([[uwb_range_noise_std**2]])
            S = innovation_covariance(H_single, ekf.covariance, R)
            
            # Gating
            accept = True
            if use_gating:
                accept = chi_square_gate(y, S, alpha=gate_alpha)
            
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
            history['nis'].append(float(y @ np.linalg.inv(S) @ y))
            history['gated'].append(accept)
        
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
        if n_uwb_accepted + n_uwb_rejected > 0:
            print(f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%")
    
    return history


def plot_temporal_calibration(
    dataset: Dict,
    no_correction: Dict,
    with_correction: Dict,
    save_path: str = None
) -> None:
    """Generate temporal calibration comparison plots.
    
    Args:
        dataset: Dataset dictionary
        no_correction: Results without time sync correction
        with_correction: Results with time sync correction
        save_path: Path to save figure
    """
    truth = dataset['truth']
    anchors = dataset['uwb_anchors']
    config = dataset['config']
    
    time_offset_ms = config['temporal_calibration']['time_offset_sec'] * 1000
    clock_drift_ppm = config['temporal_calibration']['clock_drift'] * 1e6
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    color_truth = 'black'
    color_no_corr = 'tab:red'
    color_with_corr = 'tab:green'
    
    # Helper function for errors
    def get_errors(history):
        p_true_interp = np.column_stack([
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
            np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
        ])
        errors = history['x_est'][:, :2] - p_true_interp
        return np.linalg.norm(errors, axis=1)
    
    # 1. Trajectory without correction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax1.plot(no_correction['x_est'][:, 0], no_correction['x_est'][:, 1],
            color=color_no_corr, linewidth=1.5, alpha=0.7,
            label='No Time Correction', zorder=2)
    ax1.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='UWB Anchors', zorder=5)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Without Correction (offset={time_offset_ms:.0f}ms, drift={clock_drift_ppm:.0f}ppm)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Trajectory with correction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax2.plot(with_correction['x_est'][:, 0], with_correction['x_est'][:, 1],
            color=color_with_corr, linewidth=1.5, alpha=0.7,
            label='With Time Correction', zorder=2)
    ax2.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, label='UWB Anchors', zorder=5)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('With TimeSyncModel Correction')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Overlay comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Truth', zorder=4)
    ax3.plot(no_correction['x_est'][:, 0], no_correction['x_est'][:, 1],
            color=color_no_corr, linewidth=1.5, alpha=0.6,
            label='No Correction', zorder=2)
    ax3.plot(with_correction['x_est'][:, 0], with_correction['x_est'][:, 1],
            color=color_with_corr, linewidth=1.5, alpha=0.8,
            label='With Correction', zorder=3)
    ax3.scatter(anchors[:, 0], anchors[:, 1], s=150, c='red', marker='^',
               edgecolors='darkred', linewidths=2, zorder=5)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Direct Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. Position error comparison
    ax4 = fig.add_subplot(gs[1, 0])
    error_no_corr = get_errors(no_correction)
    error_with_corr = get_errors(with_correction)
    
    ax4.plot(no_correction['t'], error_no_corr, color=color_no_corr,
            linewidth=1.5, label='No Correction')
    ax4.plot(with_correction['t'], error_with_corr, color=color_with_corr,
            linewidth=1.5, label='With Correction')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('Position Error vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. NIS comparison
    ax5 = fig.add_subplot(gs[1, 1])
    if len(no_correction['nis']) > 0:
        nis_no_corr = np.array(no_correction['nis'])
        nis_with_corr = np.array(with_correction['nis'])
        
        # Downsample for visibility
        step = max(1, len(nis_no_corr) // 500)
        ax5.plot(nis_no_corr[::step], color=color_no_corr,
                linewidth=0.5, alpha=0.5, label='No Correction')
        ax5.plot(nis_with_corr[::step], color=color_with_corr,
                linewidth=0.5, alpha=0.5, label='With Correction')
        
        # Chi-square bound
        from core.fusion import chi_square_threshold
        threshold = chi_square_threshold(dof=1, alpha=0.05)
        ax5.axhline(threshold, color='r', linestyle='--',
                   linewidth=1.5, label=f'95% bound (χ²={threshold:.2f})')
        
        ax5.set_xlabel('UWB Update Index')
        ax5.set_ylabel('NIS (1 DOF)')
        ax5.set_title('Innovation Consistency (NIS)')
        ax5.set_ylim([0, min(20, np.percentile(nis_no_corr, 99))])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
    
    rmse_no_corr = compute_rmse(error_no_corr)
    rmse_with_corr = compute_rmse(error_with_corr)
    
    metrics = ['RMSE\n[m]', 'Max Error\n[m]', 'Accept\nRate [%]']
    no_corr_vals = [
        rmse_no_corr,
        np.max(error_no_corr),
        100 * no_correction['n_uwb_accepted'] / (no_correction['n_uwb_accepted'] + no_correction['n_uwb_rejected'])
    ]
    with_corr_vals = [
        rmse_with_corr,
        np.max(error_with_corr),
        100 * with_correction['n_uwb_accepted'] / (with_correction['n_uwb_accepted'] + with_correction['n_uwb_rejected'])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, no_corr_vals, width,
                    label='No Correction', color=color_no_corr, alpha=0.7)
    bars2 = ax6.bar(x + width/2, with_corr_vals, width,
                    label='With Correction', color=color_with_corr, alpha=0.7)
    
    ax6.set_ylabel('Value')
    ax6.set_title('Metrics Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics, fontsize=9)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, no_corr_vals):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, with_corr_vals):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(f'Temporal Calibration Demo (offset={time_offset_ms:.0f}ms, drift={clock_drift_ppm:.0f}ppm)',
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure: {save_path}")
    
    plt.show()


def main():
    """Main entry point for temporal calibration demo."""
    parser = argparse.ArgumentParser(
        description="Temporal Calibration Demo (Time-Offset Dataset)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb_timeoffset",
        help="Path to time-offset dataset directory"
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
    print("Temporal Calibration Demo (Chapter 8)")
    print("="*70)
    print("\nKey Concept:")
    print("  Sensor clocks are not perfectly synchronized.")
    print("  -> Time offsets and clock drift cause fusion errors.")
    print("  -> TimeSyncModel corrects for offset and drift.")
    print("  -> Formula: t_fusion = (1 + drift) * t_sensor + offset")
    print("")
    
    # Load time-offset dataset
    print(f"Loading time-offset dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)
    
    time_offset_ms = dataset['config']['temporal_calibration']['time_offset_sec'] * 1000
    clock_drift_ppm = dataset['config']['temporal_calibration']['clock_drift'] * 1e6
    
    print("\nDataset info:")
    print(f"  IMU samples: {len(dataset['imu']['t'])}")
    print(f"  UWB epochs: {len(dataset['uwb']['t'])}")
    print(f"  Time offset: {time_offset_ms:.1f} ms")
    print(f"  Clock drift: {clock_drift_ppm:.1f} ppm")
    print("")
    
    # Run without correction
    print("[1/2] Running fusion WITHOUT time sync correction...")
    no_correction = run_fusion_with_time_sync(
        dataset, apply_correction=False, use_gating=True, verbose=True
    )
    
    # Run with correction
    print("[2/2] Running fusion WITH time sync correction...")
    with_correction = run_fusion_with_time_sync(
        dataset, apply_correction=True, use_gating=True, verbose=True
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
    
    rmse_no_corr = compute_final_rmse(no_correction)
    rmse_with_corr = compute_final_rmse(with_correction)
    
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Method':<30} {'RMSE [m]':>12} {'Improvement':>15}")
    print("-"*70)
    print(f"{'Without Time Correction':<30} {rmse_no_corr:>12.3f} {'(baseline)':>15}")
    print(f"{'With TimeSyncModel':<30} {rmse_with_corr:>12.3f} "
          f"{100*(rmse_no_corr-rmse_with_corr)/rmse_no_corr:>14.1f}%")
    print("="*70)
    
    improvement = 100 * (rmse_no_corr - rmse_with_corr) / rmse_no_corr
    
    print(f"\nKey Findings:")
    print(f"  * Time offset: {time_offset_ms:.1f} ms causes {rmse_no_corr:.2f}m RMSE")
    print(f"  * TimeSyncModel correction improves RMSE by {improvement:.1f}%")
    print(f"  * Proper temporal alignment is critical for fusion accuracy")
    print("")
    
    # Plot
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/temporal_calibration_demo.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_temporal_calibration(dataset, no_correction, with_correction, save_path=save_path)


if __name__ == "__main__":
    main()

