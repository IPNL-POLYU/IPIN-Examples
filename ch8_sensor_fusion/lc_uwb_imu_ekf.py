"""Loosely Coupled IMU + UWB EKF Fusion Demo (Chapter 8).

Demonstrates loosely coupled fusion where UWB ranges are first solved
for a position fix, then the position fix is fused with IMU propagation.

Comparison with Tightly Coupled (TC):
- TC: Fuses raw UWB range measurements directly (one update per anchor)
- LC: First solves for position from all ranges, then fuses position

Features:
- High-rate IMU propagation (100 Hz)
- Low-rate UWB position fix updates (10 Hz)
- WLS position solver from Chapter 4
- Chi-square innovation gating (Eq. 8.9)
- Innovation monitoring (Eqs. 8.5-8.6)

Author: Navigation Engineer
References: Chapter 8, Section 8.2 (Loosely vs Tightly Coupled Fusion)
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

from ch8_sensor_fusion.lc_models import (
    create_lc_fusion_ekf,
    create_lc_position_measurement_model,
    solve_uwb_position_wls,
)


def load_fusion_dataset(data_dir: str) -> Dict:
    """Load fusion dataset from directory (reuse from TC)."""
    from ch8_sensor_fusion.tc_uwb_imu_ekf import load_fusion_dataset as tc_load
    return tc_load(data_dir)


def run_lc_fusion(
    dataset: Dict,
    use_gating: bool = True,
    gate_alpha: float = 0.05,
    verbose: bool = True
) -> Dict:
    """Run loosely coupled IMU + UWB fusion.
    
    Args:
        dataset: Dataset dictionary from load_fusion_dataset
        use_gating: Whether to apply chi-square gating
        gate_alpha: Gating significance level (default 0.05)
        verbose: Print progress
    
    Returns:
        Results dictionary with:
            - 't': timestamps (N,)
            - 'x_est': estimated states (N, 5)
            - 'P_trace': trace of covariance (N,)
            - 'innovations': list of innovations
            - 'nis': list of NIS values
            - 'gated': list of booleans
            - 'n_uwb_accepted': number of UWB position fixes accepted
            - 'n_uwb_rejected': number of UWB position fixes rejected
            - 'n_uwb_failed': number of UWB position solves that failed
    """
    if verbose:
        print("="*70)
        print("Loosely Coupled IMU + UWB EKF Fusion")
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
    
    ekf = create_lc_fusion_ekf(
        initial_state=x0,
        initial_cov=P0
    )
    
    if verbose:
        print(f"\nInitialization:")
        print(f"  State: {x0}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
        if use_gating:
            print(f"  Alpha: {gate_alpha} ({(1-gate_alpha)*100:.0f}% confidence)")
    
    # Create position measurement model
    h, H_func, R_func = create_lc_position_measurement_model()
    
    # Prepare timestamped measurements
    measurements: List[StampedMeasurement] = []
    
    # Add IMU measurements
    for i in range(len(imu['t'])):
        measurements.append(StampedMeasurement(
            t=imu['t'][i],
            sensor='imu',
            z=np.hstack([imu['accel_xy'][i], imu['gyro_z'][i]]),  # [ax, ay, gz]
            R=np.eye(3),  # Not used
            meta={}
        ))
    
    # Add UWB measurements (aggregate by timestamp)
    for i in range(len(uwb['t'])):
        measurements.append(StampedMeasurement(
            t=uwb['t'][i],
            sensor='uwb',
            z=uwb['ranges'][i, :],  # All ranges at this timestamp
            R=np.eye(anchors.shape[0]),  # Not used (WLS computes own cov)
            meta={'epoch_idx': i}
        ))
    
    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)
    
    if verbose:
        print(f"\nMeasurements:")
        print(f"  IMU samples: {len(imu['t'])}")
        print(f"  UWB epochs: {len([m for m in measurements if m.sensor == 'uwb'])}")
        print(f"  Total: {len(measurements)}")
    
    # Run fusion
    history = {
        't': [],
        'x_est': [],
        'P_trace': [],
        'innovations': [],
        'nis': [],
        'gated': [],
        'uwb_positions': [],  # Store solved UWB positions for analysis
    }
    
    n_uwb_accepted = 0
    n_uwb_rejected = 0
    n_uwb_failed = 0
    t_prev = measurements[0].t
    
    for idx, meas in enumerate(measurements):
        dt = meas.t - t_prev
        
        if meas.sensor == 'imu':
            # Propagate with IMU
            u = meas.z  # [ax, ay, gyro_z]
            ekf.predict(u=u, dt=dt)
        
        elif meas.sensor == 'uwb':
            # Solve for UWB position fix
            ranges = meas.z  # All ranges at this epoch
            
            # Use current EKF position as initial guess
            initial_guess = ekf.state[:2]
            
            pos_uwb, cov_uwb, converged = solve_uwb_position_wls(
                ranges=ranges,
                anchor_positions=anchors,
                initial_guess=initial_guess
            )
            
            if pos_uwb is None or not converged:
                # WLS solver failed (too few valid ranges or diverged)
                n_uwb_failed += 1
                continue
            
            # Store solved position
            history['uwb_positions'].append(pos_uwb)
            
            # Compute innovation (position residual)
            z_pred = h(ekf.state)  # Predicted position [px, py]
            y = innovation(pos_uwb, z_pred)
            
            # Compute innovation covariance
            # Use WLS covariance + state covariance
            H = H_func(ekf.state)
            R = cov_uwb  # Use WLS-computed covariance
            S = innovation_covariance(H, ekf.covariance, R)
            
            # Gating
            accept = True
            if use_gating:
                accept = chi_square_gate(y, S, alpha=gate_alpha)
            
            if accept:
                # Perform EKF update with position fix
                K = ekf.covariance @ H.T @ np.linalg.inv(S)
                ekf.state = ekf.state + (K @ y).flatten()
                ekf.covariance = (np.eye(5) - K @ H) @ ekf.covariance
                n_uwb_accepted += 1
            else:
                n_uwb_rejected += 1
            
            # Log
            history['innovations'].append(np.linalg.norm(y))  # 2D innovation norm
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
    if history['uwb_positions']:
        history['uwb_positions'] = np.array(history['uwb_positions'])
    history['n_uwb_accepted'] = n_uwb_accepted
    history['n_uwb_rejected'] = n_uwb_rejected
    history['n_uwb_failed'] = n_uwb_failed
    
    if verbose:
        print(f"\nFusion complete:")
        print(f"  UWB position fixes solved: {n_uwb_accepted + n_uwb_rejected}")
        print(f"  UWB fixes accepted: {n_uwb_accepted}")
        print(f"  UWB fixes rejected: {n_uwb_rejected}")
        print(f"  UWB solver failures: {n_uwb_failed}")
        if n_uwb_accepted + n_uwb_rejected > 0:
            print(f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%")
    
    return history


def evaluate_results(dataset: Dict, history: Dict) -> Dict:
    """Evaluate fusion results against ground truth."""
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
    """Generate LC fusion results plots."""
    truth = dataset['truth']
    anchors = dataset['uwb_anchors']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trajectory plot
    ax = axes[0, 0]
    ax.plot(truth['p_xy'][:, 0], truth['p_xy'][:, 1], 'k-', label='Truth', linewidth=2)
    ax.plot(history['x_est'][:, 0], history['x_est'][:, 1], 'b-', label='LC EKF', alpha=0.7)
    
    # Plot UWB position fixes
    if len(history['uwb_positions']) > 0:
        ax.scatter(history['uwb_positions'][:, 0], history['uwb_positions'][:, 1],
                  s=20, c='orange', alpha=0.3, label='UWB Fixes', zorder=2)
    
    ax.scatter(anchors[:, 0], anchors[:, 1], s=100, c='red', marker='^',
              label='UWB Anchors', zorder=5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory: LC IMU + UWB Fusion')
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
            ax.plot(np.where(~accepted)[0], nis[~accepted], 'rx',
                   label='Rejected', markersize=6)
        
        # Chi-square bounds for m=2 DOF (position is 2D)
        from core.fusion import chi_square_bounds
        lower, upper = chi_square_bounds(dof=2, alpha=0.05)
        ax.axhline(upper, color='r', linestyle='--', label=f'95% bounds')
        ax.axhline(lower, color='r', linestyle='--')
        
        ax.set_xlabel('UWB Update Index')
        ax.set_ylabel('NIS (Normalized Innovation Squared)')
        ax.set_title('Innovation Consistency (NIS) - 2 DOF')
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
    """Main entry point for LC fusion demo."""
    parser = argparse.ArgumentParser(
        description="Loosely Coupled IMU + UWB EKF Fusion Demo"
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
    history = run_lc_fusion(
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
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/lc_uwb_imu_results.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_results(dataset, history, save_path=save_path)


if __name__ == "__main__":
    main()

