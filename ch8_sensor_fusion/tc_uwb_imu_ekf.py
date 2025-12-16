"""Tightly Coupled IMU + UWB EKF Fusion Demo (Chapter 8).

Demonstrates tightly coupled fusion where raw UWB range measurements
are fused directly in the EKF, rather than first computing position fixes.

Features:
- High-rate IMU propagation (100 Hz)
- Low-rate UWB range updates (10 Hz per anchor)
- Chi-square innovation gating (Eq. 8.9)
- Innovation monitoring (Eqs. 8.5-8.6)
- Comparison with IMU-only dead reckoning

Author: Navigation Engineer
References: Chapter 8, Section 8.2 (Tightly Coupled Fusion)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from core.eval import compute_position_errors, compute_rmse
from core.fusion import (
    StampedMeasurement,
    chi_square_gate,
    innovation,
    innovation_covariance,
)

from ch8_sensor_fusion.tc_models import (
    create_process_model,
    create_uwb_range_measurement_model,
    create_tc_fusion_ekf,
)


def load_fusion_dataset(data_dir: str) -> Dict:
    """Load fusion dataset from directory.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Dictionary with keys:
            - 'truth': dict with t, p_xy, v_xy, yaw
            - 'imu': dict with t, accel_xy, gyro_z
            - 'uwb_anchors': anchor positions (A, 2)
            - 'uwb': dict with t, ranges (M, A)
            - 'config': configuration dict
    """
    data_path = Path(data_dir)
    
    # Load data files
    truth_data = np.load(data_path / "truth.npz")
    imu_data = np.load(data_path / "imu.npz")
    uwb_data = np.load(data_path / "uwb_ranges.npz")
    uwb_anchors = np.load(data_path / "uwb_anchors.npy")
    
    with open(data_path / "config.json", "r") as f:
        config = json.load(f)
    
    dataset = {
        'truth': {
            't': truth_data['t'],
            'p_xy': truth_data['p_xy'],
            'v_xy': truth_data['v_xy'],
            'yaw': truth_data['yaw']
        },
        'imu': {
            't': imu_data['t'],
            'accel_xy': imu_data['accel_xy'],
            'gyro_z': imu_data['gyro_z']
        },
        'uwb_anchors': uwb_anchors,
        'uwb': {
            't': uwb_data['t'],
            'ranges': uwb_data['ranges']
        },
        'config': config
    }
    
    return dataset


def run_tc_fusion(
    dataset: Dict,
    use_gating: bool = True,
    gate_alpha: float = 0.05,
    verbose: bool = True
) -> Dict:
    """Run tightly coupled IMU + UWB fusion.
    
    Args:
        dataset: Dataset dictionary from load_fusion_dataset
        use_gating: Whether to apply chi-square gating
        gate_alpha: Gating significance level (default 0.05 for 95% confidence)
        verbose: Print progress
    
    Returns:
        Results dictionary with:
            - 't': timestamps (N,)
            - 'x_est': estimated states (N, 5)
            - 'P_trace': trace of covariance (N,)
            - 'innovations': list of innovations
            - 'nis': list of NIS values
            - 'gated': list of booleans (accepted/rejected)
            - 'n_uwb_accepted': number of UWB updates accepted
            - 'n_uwb_rejected': number of UWB updates rejected
    """
    if verbose:
        print("="*70)
        print("Tightly Coupled IMU + UWB EKF Fusion")
        print("="*70)
    
    # Extract data
    truth = dataset['truth']
    imu = dataset['imu']
    uwb = dataset['uwb']
    anchors = dataset['uwb_anchors']
    config = dataset['config']
    
    # Initialize EKF at true starting position
    x0 = np.array([
        truth['p_xy'][0, 0],  # px
        truth['p_xy'][0, 1],  # py
        truth['v_xy'][0, 0],  # vx
        truth['v_xy'][0, 1],  # vy
        truth['yaw'][0]        # yaw
    ])
    
    P0 = np.diag([0.1, 0.1, 0.5, 0.5, 0.1])**2  # Initial uncertainty
    
    ekf = create_tc_fusion_ekf(
        initial_state=x0,
        initial_cov=P0
    )
    
    if verbose:
        print(f"\nInitialization:")
        print(f"  State: {x0}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
        if use_gating:
            print(f"  Alpha: {gate_alpha} ({(1-gate_alpha)*100:.0f}% confidence)")
    
    # Create measurement model functions for each anchor
    meas_models = [
        create_uwb_range_measurement_model(
            anchor_position=anchors[i],
            range_noise_std=config['uwb']['range_noise_std_m']
        )
        for i in range(anchors.shape[0])
    ]
    
    # Prepare timestamped measurements
    measurements: List[StampedMeasurement] = []
    
    # Add IMU measurements
    for i in range(len(imu['t'])):
        measurements.append(StampedMeasurement(
            t=imu['t'][i],
            sensor='imu',
            z=np.hstack([imu['accel_xy'][i], imu['gyro_z'][i]]),  # [ax, ay, gz]
            R=np.eye(3),  # Not used for propagation
            meta={}
        ))
    
    # Add UWB measurements (per anchor)
    for i in range(len(uwb['t'])):
        for anchor_idx in range(anchors.shape[0]):
            range_meas = uwb['ranges'][i, anchor_idx]
            if not np.isnan(range_meas):  # Skip dropouts
                measurements.append(StampedMeasurement(
                    t=uwb['t'][i],
                    sensor='uwb',
                    z=np.array([range_meas]),
                    R=np.array([[config['uwb']['range_noise_std_m']**2]]),
                    meta={'anchor_idx': anchor_idx}
                ))
    
    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)
    
    if verbose:
        print(f"\nMeasurements:")
        print(f"  IMU samples: {len(imu['t'])}")
        print(f"  UWB samples: {len([m for m in measurements if m.sensor == 'uwb'])}")
        print(f"  Total: {len(measurements)}")
    
    # Run fusion
    history = {
        't': [],
        'x_est': [],
        'P_trace': [],
        'innovations': [],
        'nis': [],
        'gated': []
    }
    
    n_uwb_accepted = 0
    n_uwb_rejected = 0
    t_prev = measurements[0].t
    
    for idx, meas in enumerate(measurements):
        dt = meas.t - t_prev
        
        if meas.sensor == 'imu':
            # Propagate with IMU
            u = meas.z  # [ax, ay, gyro_z]
            ekf.predict(u=u, dt=dt)
        
        elif meas.sensor == 'uwb':
            # UWB range update
            anchor_idx = meas.meta['anchor_idx']
            h, H_func, R_func = meas_models[anchor_idx]
            
            # Compute innovation
            z_pred = h(ekf.state)
            y = innovation(meas.z, z_pred)
            
            # Compute innovation covariance
            H = H_func(ekf.state)
            R = R_func()
            S = innovation_covariance(H, ekf.covariance, R)
            
            # Gating
            accept = True
            if use_gating:
                accept = chi_square_gate(y, S, alpha=gate_alpha)
            
            if accept:
                # Manually perform EKF update
                K = ekf.covariance @ H.T @ np.linalg.inv(S)
                ekf.state = ekf.state + (K @ y).flatten()
                ekf.covariance = (np.eye(5) - K @ H) @ ekf.covariance
                n_uwb_accepted += 1
            else:
                n_uwb_rejected += 1
            
            # Log
            history['innovations'].append(y[0])
            nis_value = float(y @ np.linalg.inv(S) @ y)
            history['nis'].append(nis_value)
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
        print(f"\nFusion complete:")
        print(f"  UWB accepted: {n_uwb_accepted}")
        print(f"  UWB rejected: {n_uwb_rejected}")
        if n_uwb_accepted + n_uwb_rejected > 0:
            print(f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%")
    
    return history


def evaluate_results(dataset: Dict, history: Dict) -> Dict:
    """Evaluate fusion results against ground truth.
    
    Args:
        dataset: Dataset dictionary
        history: Fusion results from run_tc_fusion
    
    Returns:
        Metrics dictionary
    """
    truth = dataset['truth']
    
    # Interpolate truth to estimated timestamps
    p_true_interp = np.column_stack([
        np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
        np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
    ])
    
    # Extract estimated positions
    p_est = history['x_est'][:, :2]
    
    # Compute errors
    errors = compute_position_errors(p_true_interp, p_est)
    rmse = compute_rmse(errors)
    
    metrics = {
        'rmse_2d': rmse,
        'rmse_x': np.sqrt(np.mean(errors[:, 0]**2)),
        'rmse_y': np.sqrt(np.mean(errors[:, 1]**2)),
        'max_error': np.max(np.linalg.norm(errors, axis=1)),
        'final_error': np.linalg.norm(errors[-1])
    }
    
    return metrics


def plot_results(dataset: Dict, history: Dict, save_path: str = None) -> None:
    """Generate fusion results plots.
    
    Args:
        dataset: Dataset dictionary
        history: Fusion results
        save_path: Optional path to save figure
    """
    truth = dataset['truth']
    anchors = dataset['uwb_anchors']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trajectory plot
    ax = axes[0, 0]
    ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 'k-', label='Truth', linewidth=2)
    ax.plot(history['x_est'][:, 0], history['x_est'][:, 1], 'b-', label='TC EKF', alpha=0.7)
    ax.scatter(anchors[:, 0], anchors[:, 1], s=100, c='red', marker='^', label='UWB Anchors', zorder=5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory: TC IMU + UWB Fusion')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # 2. Position error
    ax = axes[0, 1]
    p_true_interp = np.column_stack([
        np.interp(history['t'], truth['t'], truth['p_xy'][:, 0]),
        np.interp(history['t'], truth['t'], truth['p_xy'][:, 1])
    ])
    errors = history['x_est'][:, :2] - p_true_interp
    error_norm = np.linalg.norm(errors, axis=1)
    ax.plot(history['t'], error_norm, 'b-')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Position Error vs Time')
    ax.grid(True)
    
    # 3. NIS plot
    ax = axes[1, 0]
    if len(history['nis']) > 0:
        nis = np.array(history['nis'])
        accepted = np.array(history['gated'])
        
        ax.plot(nis[accepted], 'g.', label='Accepted', markersize=4)
        if np.any(~accepted):
            ax.plot(np.where(~accepted)[0], nis[~accepted], 'rx', label='Rejected', markersize=6)
        
        # Chi-square bounds for m=1 DOF
        from core.fusion import chi_square_bounds
        lower, upper = chi_square_bounds(dof=1, alpha=0.05)
        ax.axhline(upper, color='r', linestyle='--', label=f'95% bounds')
        ax.axhline(lower, color='r', linestyle='--')
        
        ax.set_xlabel('UWB Update Index')
        ax.set_ylabel('NIS (Normalized Innovation Squared)')
        ax.set_title('Innovation Consistency (NIS)')
        ax.legend()
        ax.grid(True)
    
    # 4. Covariance trace
    ax = axes[1, 1]
    ax.plot(history['t'], history['P_trace'], 'b-')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Trace(P)')
    ax.set_title('Covariance Trace')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure: {save_path}")
    
    plt.show()


def main():
    """Main entry point for TC fusion demo."""
    parser = argparse.ArgumentParser(
        description="Tightly Coupled IMU + UWB EKF Fusion Demo"
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
        "--alpha",
        type=float,
        default=0.05,
        help="Gating significance level (default: 0.05)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save results figure"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)
    
    # Run fusion
    history = run_tc_fusion(
        dataset,
        use_gating=not args.no_gating,
        gate_alpha=args.alpha,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation Metrics")
    print("="*70)
    metrics = evaluate_results(dataset, history)
    print(f"  RMSE (2D)    : {metrics['rmse_2d']:.3f} m")
    print(f"  RMSE (X)     : {metrics['rmse_x']:.3f} m")
    print(f"  RMSE (Y)     : {metrics['rmse_y']:.3f} m")
    print(f"  Max Error    : {metrics['max_error']:.3f} m")
    print(f"  Final Error  : {metrics['final_error']:.3f} m")
    print("")
    
    # Plot
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/tc_uwb_imu_results.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_results(dataset, history, save_path=save_path)


if __name__ == "__main__":
    main()

