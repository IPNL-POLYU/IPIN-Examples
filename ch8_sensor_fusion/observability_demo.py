"""Observability Demonstration for Chapter 8.

Demonstrates the observability concept from Chapter 8, Equations (8.1)-(8.2):
- **Odometry-only**: Cannot determine global translation (unobservable mode)
- **Odometry + Absolute fixes**: Translation becomes observable

This demo shows that two trajectories differing only by a constant translation
produce identical odometry measurements, making absolute position unobservable.
Adding occasional absolute position fixes (e.g., UWB, GPS) restores observability.

Key Concepts:
- Unobservable modes: States that cannot be determined from measurements
- Observability: A system is observable if the state can be uniquely determined
- Odometry measures relative displacement, not absolute position
- Absolute measurements (position fixes) make translation observable

Author: Li-Ta Hsu
References: Chapter 8, Equations (8.1)-(8.2), Section on Observability
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from core.estimators import ExtendedKalmanFilter


def generate_trajectory(
    duration: float = 30.0,
    speed: float = 1.0,
    dt: float = 0.1
) -> Dict:
    """Generate a simple 2D trajectory for observability demo.
    
    Args:
        duration: Trajectory duration in seconds
        speed: Constant speed in m/s
        dt: Time step in seconds
    
    Returns:
        Dictionary with 't', 'p_xy', 'v_xy'
    """
    t = np.arange(0, duration, dt)
    n = len(t)
    
    # Create a figure-8 trajectory
    p_xy = np.zeros((n, 2))
    v_xy = np.zeros((n, 2))
    
    for i, ti in enumerate(t):
        # Figure-8 parametric curve
        angle = 2 * np.pi * ti / duration
        p_xy[i, 0] = 5 * np.sin(angle)
        p_xy[i, 1] = 5 * np.sin(2 * angle)
        
        # Velocity (derivative)
        v_xy[i, 0] = 5 * (2 * np.pi / duration) * np.cos(angle)
        v_xy[i, 1] = 5 * 2 * (2 * np.pi / duration) * np.cos(2 * angle)
    
    return {'t': t, 'p_xy': p_xy, 'v_xy': v_xy}


def generate_odometry_measurements(
    trajectory: Dict,
    noise_std: float = 0.05
) -> Dict:
    """Generate odometry measurements (relative displacements).
    
    Odometry measures INCREMENTS, not absolute position.
    This is why absolute translation is unobservable from odometry alone.
    
    Args:
        trajectory: Trajectory dictionary with 'p_xy'
        noise_std: Odometry noise standard deviation (m)
    
    Returns:
        Dictionary with 't', 'delta_p' (incremental displacement)
    """
    p_xy = trajectory['p_xy']
    t = trajectory['t']
    n = len(t)
    
    # Compute increments (odometry measures displacement between steps)
    delta_p = np.diff(p_xy, axis=0)
    
    # Add noise
    noise = np.random.randn(*delta_p.shape) * noise_std
    delta_p_noisy = delta_p + noise
    
    return {
        't': t[1:],  # Odometry starts at t[1]
        'delta_p': delta_p_noisy,
        'noise_std': noise_std
    }


def generate_absolute_fixes(
    trajectory: Dict,
    fix_rate: float = 1.0,  # Hz
    noise_std: float = 0.5  # meters
) -> Dict:
    """Generate occasional absolute position fixes (e.g., UWB, GPS).
    
    These measurements observe ABSOLUTE position, making translation observable.
    
    Args:
        trajectory: Trajectory dictionary
        fix_rate: Fix rate in Hz
        noise_std: Position fix noise standard deviation (m)
    
    Returns:
        Dictionary with 't_fix', 'p_fix'
    """
    t = trajectory['t']
    p_xy = trajectory['p_xy']
    dt = t[1] - t[0]
    
    # Sample at fix_rate
    fix_interval = int(1.0 / (fix_rate * dt))
    fix_indices = np.arange(0, len(t), fix_interval)
    
    t_fix = t[fix_indices]
    p_fix = p_xy[fix_indices] + np.random.randn(len(fix_indices), 2) * noise_std
    
    return {
        't_fix': t_fix,
        'p_fix': p_fix,
        'noise_std': noise_std
    }


def run_odometry_only_fusion(
    trajectory: Dict,
    odometry: Dict,
    translation_offset: np.ndarray = np.array([0.0, 0.0])
) -> Dict:
    """Run fusion with odometry only (no absolute position fixes).
    
    State: [px, py, vx, vy] (4D)
    Measurement: [delta_px, delta_py] (odometry increment)
    
    This demonstrates that absolute translation is UNOBSERVABLE.
    
    Args:
        trajectory: Ground truth trajectory
        odometry: Odometry measurements
        translation_offset: Initial position offset (unobservable!)
    
    Returns:
        Fusion results dictionary
    """
    # Initial state (with translation offset)
    true_p0 = trajectory['p_xy'][0]
    x0 = np.array([
        true_p0[0] + translation_offset[0],
        true_p0[1] + translation_offset[1],
        trajectory['v_xy'][0, 0],
        trajectory['v_xy'][0, 1]
    ])
    
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])**2  # Initial covariance
    
    # Process model: constant velocity
    def process_model(x, u, dt):
        px, py, vx, vy = x
        return np.array([px + vx * dt, py + vy * dt, vx, vy])
    
    def process_jacobian(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F
    
    def process_noise_cov(dt):
        # Process noise (small velocity perturbations)
        q_v = 0.1**2
        Q = np.array([
            [0.25 * q_v * dt**4, 0, 0.5 * q_v * dt**3, 0],
            [0, 0.25 * q_v * dt**4, 0, 0.5 * q_v * dt**3],
            [0.5 * q_v * dt**3, 0, q_v * dt**2, 0],
            [0, 0.5 * q_v * dt**3, 0, q_v * dt**2]
        ])
        return Q
    
    # Measurement model: odometry increment
    # z = [delta_px, delta_py] = [vx * dt, vy * dt] (approximately)
    def measurement_model(x):
        # Odometry observes incremental displacement
        # This is velocity * dt, but we'll use the actual delta from measurements
        return np.array([x[2], x[3]])  # Return velocity as proxy
    
    def measurement_jacobian(x):
        # H observes velocity only (position is unobservable!)
        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return H
    
    def measurement_noise_cov():
        R = np.eye(2) * odometry['noise_std']**2
        return R
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        process_model=process_model,
        process_jacobian=process_jacobian,
        measurement_model=measurement_model,
        measurement_jacobian=measurement_jacobian,
        Q=process_noise_cov,
        R=measurement_noise_cov,
        x0=x0,
        P0=P0
    )
    
    # Run fusion
    dt = trajectory['t'][1] - trajectory['t'][0]
    history = {'t': [], 'x_est': [], 'P_trace': []}
    
    # Initial state
    history['t'].append(trajectory['t'][0])
    history['x_est'].append(ekf.state.copy())
    history['P_trace'].append(np.trace(ekf.covariance))
    
    for i, delta_p in enumerate(odometry['delta_p']):
        # Predict
        ekf.predict(u=None, dt=dt)
        
        # Update with odometry (velocity measurement as proxy for increment)
        z = delta_p / dt  # Convert increment to velocity
        ekf.update(z)
        
        # Log
        history['t'].append(odometry['t'][i])
        history['x_est'].append(ekf.state.copy())
        history['P_trace'].append(np.trace(ekf.covariance))
    
    # Convert to arrays
    history['t'] = np.array(history['t'])
    history['x_est'] = np.array(history['x_est'])
    history['P_trace'] = np.array(history['P_trace'])
    
    return history


def run_odometry_with_fixes_fusion(
    trajectory: Dict,
    odometry: Dict,
    absolute_fixes: Dict,
    translation_offset: np.ndarray = np.array([0.0, 0.0])
) -> Dict:
    """Run fusion with odometry + absolute position fixes.
    
    This demonstrates that absolute translation becomes OBSERVABLE
    when absolute position measurements are available.
    
    Args:
        trajectory: Ground truth trajectory
        odometry: Odometry measurements
        absolute_fixes: Absolute position fix measurements
        translation_offset: Initial position offset (will be corrected!)
    
    Returns:
        Fusion results dictionary
    """
    # Same initialization as odometry-only
    true_p0 = trajectory['p_xy'][0]
    x0 = np.array([
        true_p0[0] + translation_offset[0],
        true_p0[1] + translation_offset[1],
        trajectory['v_xy'][0, 0],
        trajectory['v_xy'][0, 1]
    ])
    
    P0 = np.diag([1.0, 1.0, 0.5, 0.5])**2
    
    # Process model (same as before)
    def process_model(x, u, dt):
        px, py, vx, vy = x
        return np.array([px + vx * dt, py + vy * dt, vx, vy])
    
    def process_jacobian(x, u, dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def process_noise_cov(dt):
        q_v = 0.1**2
        Q = np.array([
            [0.25 * q_v * dt**4, 0, 0.5 * q_v * dt**3, 0],
            [0, 0.25 * q_v * dt**4, 0, 0.5 * q_v * dt**3],
            [0.5 * q_v * dt**3, 0, q_v * dt**2, 0],
            [0, 0.5 * q_v * dt**3, 0, q_v * dt**2]
        ])
        return Q
    
    # Odometry measurement model
    def odom_measurement_model(x):
        return np.array([x[2], x[3]])
    
    def odom_measurement_jacobian(x):
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def odom_measurement_noise_cov():
        return np.eye(2) * odometry['noise_std']**2
    
    # Absolute position measurement model
    def pos_measurement_model(x):
        return np.array([x[0], x[1]])  # Observe ABSOLUTE position
    
    def pos_measurement_jacobian(x):
        # H observes position directly (makes translation observable!)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    
    def pos_measurement_noise_cov():
        return np.eye(2) * absolute_fixes['noise_std']**2
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        process_model=process_model,
        process_jacobian=process_jacobian,
        measurement_model=odom_measurement_model,  # Default to odometry
        measurement_jacobian=odom_measurement_jacobian,
        Q=process_noise_cov,
        R=odom_measurement_noise_cov,
        x0=x0,
        P0=P0
    )
    
    # Merge odometry and fixes by timestamp
    from core.fusion import StampedMeasurement
    
    measurements = []
    
    # Add odometry
    for i in range(len(odometry['t'])):
        measurements.append(StampedMeasurement(
            t=odometry['t'][i],
            sensor='odometry',
            z=odometry['delta_p'][i],
            R=np.eye(2) * odometry['noise_std']**2,
            meta={}
        ))
    
    # Add absolute fixes
    for i in range(len(absolute_fixes['t_fix'])):
        measurements.append(StampedMeasurement(
            t=absolute_fixes['t_fix'][i],
            sensor='position_fix',
            z=absolute_fixes['p_fix'][i],
            R=np.eye(2) * absolute_fixes['noise_std']**2,
            meta={}
        ))
    
    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)
    
    # Run fusion
    dt = trajectory['t'][1] - trajectory['t'][0]
    history = {'t': [], 'x_est': [], 'P_trace': [], 'fix_times': []}
    
    # Initial state
    history['t'].append(trajectory['t'][0])
    history['x_est'].append(ekf.state.copy())
    history['P_trace'].append(np.trace(ekf.covariance))
    
    t_prev = trajectory['t'][0]
    
    for meas in measurements:
        dt_step = meas.t - t_prev
        
        # Predict
        ekf.predict(u=None, dt=dt_step)
        
        # Update based on sensor type
        if meas.sensor == 'odometry':
            # Odometry update (velocity)
            z_odom = meas.z / dt_step
            H_odom = odom_measurement_jacobian(ekf.state)
            R_odom = odom_measurement_noise_cov()
            
            # Manual update (simpler than switching models)
            y = z_odom - odom_measurement_model(ekf.state)
            S = H_odom @ ekf.covariance @ H_odom.T + R_odom
            K = ekf.covariance @ H_odom.T @ np.linalg.inv(S)
            ekf.state = ekf.state + K @ y
            ekf.covariance = (np.eye(4) - K @ H_odom) @ ekf.covariance
        
        elif meas.sensor == 'position_fix':
            # Position fix update (absolute position)
            z_pos = meas.z
            H_pos = pos_measurement_jacobian(ekf.state)
            R_pos = pos_measurement_noise_cov()
            
            y = z_pos - pos_measurement_model(ekf.state)
            S = H_pos @ ekf.covariance @ H_pos.T + R_pos
            K = ekf.covariance @ H_pos.T @ np.linalg.inv(S)
            ekf.state = ekf.state + K @ y
            ekf.covariance = (np.eye(4) - K @ H_pos) @ ekf.covariance
            
            history['fix_times'].append(meas.t)
        
        # Log
        history['t'].append(meas.t)
        history['x_est'].append(ekf.state.copy())
        history['P_trace'].append(np.trace(ekf.covariance))
        
        t_prev = meas.t
    
    # Convert to arrays
    history['t'] = np.array(history['t'])
    history['x_est'] = np.array(history['x_est'])
    history['P_trace'] = np.array(history['P_trace'])
    history['fix_times'] = np.array(history['fix_times'])
    
    return history


def plot_observability_demo(
    trajectory: Dict,
    odom_only_1: Dict,
    odom_only_2: Dict,
    odom_with_fixes: Dict,
    translation_offset: np.ndarray,
    save_path: str = None
) -> None:
    """Generate observability demonstration plots.
    
    Args:
        trajectory: Ground truth
        odom_only_1: Odometry-only fusion (offset 1)
        odom_only_2: Odometry-only fusion (offset 2)
        odom_with_fixes: Odometry + absolute fixes fusion
        translation_offset: Translation offset used
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    color_truth = 'black'
    color_odom1 = 'tab:red'
    color_odom2 = 'tab:purple'
    color_fixes = 'tab:green'
    
    # 1. Odometry-only: Two translations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trajectory['p_xy'][:, 0], trajectory['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax1.plot(odom_only_1['x_est'][:, 0], odom_only_1['x_est'][:, 1],
            color=color_odom1, linewidth=1.5, alpha=0.7,
            label=f'Odom-only (offset=[0,0])', zorder=2)
    ax1.plot(odom_only_2['x_est'][:, 0], odom_only_2['x_est'][:, 1],
            color=color_odom2, linewidth=1.5, alpha=0.7,
            label=f'Odom-only (offset={translation_offset})', zorder=1)
    ax1.scatter(trajectory['p_xy'][0, 0], trajectory['p_xy'][0, 1],
               s=200, c='green', marker='*', edgecolors='darkgreen',
               linewidths=2, label='True Start', zorder=5)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Odometry-Only: Translation is Unobservable')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Odometry + Fixes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(trajectory['p_xy'][:, 0], trajectory['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Ground Truth', zorder=3)
    ax2.plot(odom_with_fixes['x_est'][:, 0], odom_with_fixes['x_est'][:, 1],
            color=color_fixes, linewidth=1.5, alpha=0.7,
            label='Odom + Absolute Fixes', zorder=2)
    # Mark position fix times
    fix_indices = [np.argmin(np.abs(odom_with_fixes['t'] - t_fix))
                  for t_fix in odom_with_fixes['fix_times']]
    ax2.scatter(odom_with_fixes['x_est'][fix_indices, 0],
               odom_with_fixes['x_est'][fix_indices, 1],
               s=50, c='orange', marker='o', alpha=0.5,
               label='Position Fixes', zorder=4)
    ax2.scatter(trajectory['p_xy'][0, 0], trajectory['p_xy'][0, 1],
               s=200, c='green', marker='*', edgecolors='darkgreen',
               linewidths=2, label='True Start', zorder=5)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Odometry + Fixes: Translation is Observable')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(trajectory['p_xy'][:, 0], trajectory['p_xy'][:, 1],
            color=color_truth, linewidth=2, label='Truth', zorder=4)
    ax3.plot(odom_only_2['x_est'][:, 0], odom_only_2['x_est'][:, 1],
            color=color_odom2, linewidth=1.5, alpha=0.6,
            label='Odom-only (drifted)', zorder=2)
    ax3.plot(odom_with_fixes['x_est'][:, 0], odom_with_fixes['x_est'][:, 1],
            color=color_fixes, linewidth=1.5, alpha=0.8,
            label='Odom + Fixes (corrected)', zorder=3)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('Direct Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. Position errors (Odom-only)
    ax4 = fig.add_subplot(gs[1, 0])
    # Interpolate truth
    def get_errors(history):
        p_true_interp = np.column_stack([
            np.interp(history['t'], trajectory['t'], trajectory['p_xy'][:, 0]),
            np.interp(history['t'], trajectory['t'], trajectory['p_xy'][:, 1])
        ])
        errors = history['x_est'][:, :2] - p_true_interp
        return np.linalg.norm(errors, axis=1)
    
    error_odom1 = get_errors(odom_only_1)
    error_odom2 = get_errors(odom_only_2)
    
    ax4.plot(odom_only_1['t'], error_odom1, color=color_odom1,
            linewidth=1.5, label='Offset=[0,0]')
    ax4.plot(odom_only_2['t'], error_odom2, color=color_odom2,
            linewidth=1.5, label=f'Offset={translation_offset}')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Position Error [m]')
    ax4.set_title('Odometry-Only: Constant Translation Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Position error (Odom + Fixes)
    ax5 = fig.add_subplot(gs[1, 1])
    error_fixes = get_errors(odom_with_fixes)
    ax5.plot(odom_with_fixes['t'], error_fixes, color=color_fixes, linewidth=1.5)
    # Mark position fix times
    for t_fix in odom_with_fixes['fix_times']:
        ax5.axvline(t_fix, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Position Error [m]')
    ax5.set_title('Odom + Fixes: Error Corrected at Fixes')
    ax5.grid(True, alpha=0.3)
    
    # 6. Covariance trace
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(odom_only_1['t'], odom_only_1['P_trace'],
            color=color_odom1, linewidth=1.5, label='Odom-only', alpha=0.7)
    ax6.plot(odom_with_fixes['t'], odom_with_fixes['P_trace'],
            color=color_fixes, linewidth=1.5, label='Odom + Fixes')
    # Mark fix times
    for t_fix in odom_with_fixes['fix_times']:
        ax6.axvline(t_fix, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Trace(P) [m²]')
    ax6.set_title('Covariance: Fixes Reduce Uncertainty')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Observability Demo: Odometry-Only vs Odometry + Absolute Fixes',
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure: {save_path}")
    
    plt.show()


def main():
    """Main entry point for observability demo."""
    parser = argparse.ArgumentParser(
        description="Observability Demo: Odometry-Only vs Odometry + Absolute Fixes"
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
    print("Observability Demonstration (Chapter 8)")
    print("="*70)
    print("\nKey Concept:")
    print("  Odometry measures INCREMENTS, not absolute position.")
    print("  -> Translation is UNOBSERVABLE from odometry alone.")
    print("  -> Adding absolute position fixes makes translation OBSERVABLE.")
    print("")
    
    # Generate trajectory
    print("[1/6] Generating trajectory...")
    trajectory = generate_trajectory(duration=30.0, speed=1.0, dt=0.1)
    
    # Generate measurements
    print("[2/6] Generating odometry measurements...")
    odometry = generate_odometry_measurements(trajectory, noise_std=0.05)
    
    print("[3/6] Generating absolute position fixes (1 Hz)...")
    absolute_fixes = generate_absolute_fixes(trajectory, fix_rate=1.0, noise_std=0.5)
    
    # Run fusions
    translation_offset = np.array([3.0, 2.0])
    
    print(f"[4/6] Running odometry-only fusion (offset [0, 0])...")
    odom_only_1 = run_odometry_only_fusion(
        trajectory, odometry, translation_offset=np.array([0.0, 0.0])
    )
    
    print(f"[5/6] Running odometry-only fusion (offset {translation_offset})...")
    odom_only_2 = run_odometry_only_fusion(
        trajectory, odometry, translation_offset=translation_offset
    )
    
    print(f"[6/6] Running odometry + absolute fixes fusion...")
    odom_with_fixes = run_odometry_with_fixes_fusion(
        trajectory, odometry, absolute_fixes, translation_offset=translation_offset
    )
    
    # Compute final errors
    def final_error(history):
        p_true_final = trajectory['p_xy'][-1]
        p_est_final = history['x_est'][-1, :2]
        return np.linalg.norm(p_true_final - p_est_final)
    
    error_odom1 = final_error(odom_only_1)
    error_odom2 = final_error(odom_only_2)
    error_fixes = final_error(odom_with_fixes)
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"{'Method':<35} {'Final Error [m]':>15}")
    print("-"*70)
    print(f"{'Odometry-only (offset [0, 0])':<35} {error_odom1:>15.3f}")
    print(f"{'Odometry-only (offset [3, 2])':<35} {error_odom2:>15.3f}")
    print(f"{'Odometry + Absolute Fixes':<35} {error_fixes:>15.3f}")
    print("="*70)
    
    print("\nInterpretation:")
    print(f"  * Odometry-only error: ~{np.linalg.norm(translation_offset):.1f}m")
    print(f"    (constant offset = unobservable translation)")
    print(f"  * Odometry + Fixes error: {error_fixes:.2f}m")
    print(f"    (corrected by absolute measurements)")
    print("")
    
    # Plot
    save_path = args.save if args.save else "ch8_sensor_fusion/figs/observability_demo.svg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_observability_demo(
        trajectory, odom_only_1, odom_only_2, odom_with_fixes,
        translation_offset, save_path=save_path
    )


if __name__ == "__main__":
    main()

