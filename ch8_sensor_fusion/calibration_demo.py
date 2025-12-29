"""Calibration Demonstration for Chapter 8, Section 8.4.

Demonstrates intrinsic and extrinsic calibration techniques essential
for multi-sensor fusion systems.

Key Concepts:
- **Intrinsic Calibration**: Sensor-specific parameters (biases, scale factors)
- **Extrinsic Calibration**: Relative poses between sensors (lever-arm, rotation)

This demo covers:
1. IMU intrinsic calibration (accelerometer/gyroscope bias estimation)
2. 2D extrinsic calibration (lever-arm and rotation between sensors)

Author: Li-Ta Hsu
Date: December 2025
References: Chapter 8, Section 8.4 (Calibration Techniques)
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def estimate_imu_bias_stationary(
    accel_samples: np.ndarray,
    gyro_samples: np.ndarray,
    gravity_magnitude: float = 9.81
) -> Dict:
    """Estimate IMU biases from stationary measurements.
    
    During a stationary period:
    - Gyroscope should read zero (any reading is bias)
    - Accelerometer should read gravity vector (deviation is bias)
    
    This is the simplest intrinsic IMU calibration method mentioned
    in the book (Section 8.4.1.3).
    
    Args:
        accel_samples: Accelerometer samples (N, 3) in m/s²
        gyro_samples: Gyroscope samples (N, 3) in rad/s
        gravity_magnitude: Expected gravity magnitude (default 9.81 m/s²)
    
    Returns:
        Dictionary with:
            - 'accel_bias': Estimated accelerometer bias (3,) in m/s²
            - 'gyro_bias': Estimated gyroscope bias (3,) in rad/s
            - 'accel_std': Standard deviation of accel samples
            - 'gyro_std': Standard deviation of gyro samples
            - 'gravity_axis': Identified gravity axis (0=x, 1=y, 2=z)
    
    References:
        Chapter 8, Section 8.4.1.3: IMU Intrinsic Calibration
    """
    # Gyroscope bias: mean of stationary samples (should be zero)
    gyro_bias = np.mean(gyro_samples, axis=0)
    gyro_std = np.std(gyro_samples, axis=0)
    
    # Accelerometer bias: deviation from gravity
    accel_mean = np.mean(accel_samples, axis=0)
    accel_std = np.std(accel_samples, axis=0)
    
    # Identify gravity axis (largest magnitude component)
    gravity_axis = np.argmax(np.abs(accel_mean))
    
    # Accelerometer bias: measured - expected
    # Expected: gravity along identified axis, zero on others
    expected_accel = np.zeros(3)
    expected_accel[gravity_axis] = gravity_magnitude * np.sign(accel_mean[gravity_axis])
    
    accel_bias = accel_mean - expected_accel
    
    return {
        'accel_bias': accel_bias,
        'gyro_bias': gyro_bias,
        'accel_std': accel_std,
        'gyro_std': gyro_std,
        'gravity_axis': gravity_axis,
        'accel_mean': accel_mean,
        'n_samples': len(accel_samples)
    }


def calibrate_extrinsic_2d_least_squares(
    p_sensor1: np.ndarray,
    p_sensor2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate 2D extrinsic calibration between two sensors.
    
    Estimates the relative pose (translation + rotation) between two sensors
    observing the same motion or scene.
    
    Model: p_sensor2 = R @ p_sensor1 + t
    
    This uses least-squares fitting to estimate R (2x2 rotation) and
    t (2D translation/lever-arm).
    
    Args:
        p_sensor1: Positions from sensor 1 (N, 2)
        p_sensor2: Positions from sensor 2 (N, 2) at same timestamps
    
    Returns:
        Tuple of (R, t):
            R: 2D rotation matrix (2, 2)
            t: 2D translation vector (2,) - lever-arm
    
    References:
        Chapter 8, Section 8.4.2: Extrinsic Calibration
    """
    # Center the data
    p1_mean = np.mean(p_sensor1, axis=0)
    p2_mean = np.mean(p_sensor2, axis=0)
    
    p1_centered = p_sensor1 - p1_mean
    p2_centered = p_sensor2 - p2_mean
    
    # Estimate rotation using SVD (Procrustes problem)
    # R = arg min ||p2_centered - R @ p1_centered.T||^2
    H = p1_centered.T @ p2_centered  # 2x2 correlation matrix
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Estimate translation
    t = p2_mean - R @ p1_mean
    
    return R, t


def generate_synthetic_imu_stationary(
    duration: float = 10.0,
    rate: float = 100.0,
    accel_bias: np.ndarray = None,
    gyro_bias: np.ndarray = None,
    accel_noise_std: float = 0.01,
    gyro_noise_std: float = 0.001
) -> Dict:
    """Generate synthetic stationary IMU data for calibration testing.
    
    Args:
        duration: Duration in seconds
        rate: Sampling rate in Hz
        accel_bias: True accelerometer bias (3,) in m/s²
        gyro_bias: True gyroscope bias (3,) in rad/s
        accel_noise_std: Accelerometer noise std
        gyro_noise_std: Gyroscope noise std
    
    Returns:
        Dictionary with 't', 'accel', 'gyro', 'true_accel_bias', 'true_gyro_bias'
    """
    if accel_bias is None:
        accel_bias = np.array([0.05, -0.03, 0.02])  # m/s²
    
    if gyro_bias is None:
        gyro_bias = np.array([0.01, -0.005, 0.008])  # rad/s
    
    n_samples = int(duration * rate)
    t = np.arange(n_samples) / rate
    
    # Stationary: accelerometer reads gravity + bias + noise
    # Assume gravity along -Z axis (sensor facing up)
    gravity = np.array([0, 0, -9.81])
    
    accel = np.zeros((n_samples, 3))
    gyro = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        accel[i] = gravity + accel_bias + np.random.randn(3) * accel_noise_std
        gyro[i] = gyro_bias + np.random.randn(3) * gyro_noise_std
    
    return {
        't': t,
        'accel': accel,
        'gyro': gyro,
        'true_accel_bias': accel_bias,
        'true_gyro_bias': gyro_bias
    }


def generate_synthetic_extrinsic_data(
    duration: float = 30.0,
    rate: float = 10.0,
    lever_arm: np.ndarray = None,
    rotation_angle: float = np.pi / 6  # 30 degrees
) -> Dict:
    """Generate synthetic data for 2D extrinsic calibration.
    
    Simulates two sensors observing the same trajectory with known
    relative pose (lever-arm + rotation).
    
    Args:
        duration: Duration in seconds
        rate: Sampling rate in Hz
        lever_arm: True lever-arm (2,) in meters
        rotation_angle: Rotation angle in radians
    
    Returns:
        Dictionary with positions from both sensors and true calibration
    """
    if lever_arm is None:
        lever_arm = np.array([0.5, 0.3])  # meters
    
    n_samples = int(duration * rate)
    t = np.arange(n_samples) / rate
    
    # Generate base trajectory (circular motion)
    radius = 5.0
    angular_freq = 2 * np.pi / duration
    
    p_sensor1 = np.zeros((n_samples, 2))
    for i, ti in enumerate(t):
        angle = angular_freq * ti
        p_sensor1[i, 0] = radius * np.cos(angle)
        p_sensor1[i, 1] = radius * np.sin(angle)
    
    # Apply transformation to get sensor 2 positions
    R_true = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    p_sensor2 = (R_true @ p_sensor1.T).T + lever_arm
    
    # Add noise
    noise_std = 0.05  # meters
    p_sensor1 += np.random.randn(n_samples, 2) * noise_std
    p_sensor2 += np.random.randn(n_samples, 2) * noise_std
    
    return {
        't': t,
        'p_sensor1': p_sensor1,
        'p_sensor2': p_sensor2,
        'true_R': R_true,
        'true_t': lever_arm,
        'true_rotation_angle': rotation_angle
    }


def plot_imu_calibration(
    data: Dict,
    calibration: Dict,
    save_path: str = None
):
    """Plot IMU calibration results.
    
    Args:
        data: IMU data dictionary
        calibration: Calibration results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Accelerometer data
    ax1 = fig.add_subplot(gs[0, :])
    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax1.plot(data['t'], data['accel'][:, i], label=f'Accel {axis}',
                alpha=0.7, linewidth=0.5)
    
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(-9.81, color='r', linestyle='--', alpha=0.3, label='Gravity')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Acceleration [m/s²]')
    ax1.set_title('Accelerometer Raw Data (Stationary)')
    ax1.legend(ncol=4)
    ax1.grid(True, alpha=0.3)
    
    # Gyroscope data
    ax2 = fig.add_subplot(gs[1, :])
    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax2.plot(data['t'], data['gyro'][:, i] * 180/np.pi,
                label=f'Gyro {axis}', alpha=0.7, linewidth=0.5)
    
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Rate [deg/s]')
    ax2.set_title('Gyroscope Raw Data (Stationary)')
    ax2.legend(ncol=3)
    ax2.grid(True, alpha=0.3)
    
    # Bias estimation results
    ax3 = fig.add_subplot(gs[2, 0])
    
    axes = ['X', 'Y', 'Z']
    x_pos = np.arange(3)
    
    true_bias = data['true_accel_bias']
    est_bias = calibration['accel_bias']
    
    width = 0.35
    ax3.bar(x_pos - width/2, true_bias, width, label='True Bias', alpha=0.7)
    ax3.bar(x_pos + width/2, est_bias, width, label='Estimated Bias', alpha=0.7)
    
    ax3.set_xlabel('Axis')
    ax3.set_ylabel('Bias [m/s²]')
    ax3.set_title('Accelerometer Bias Estimation')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(axes)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(gs[2, 1])
    
    true_bias_gyro = data['true_gyro_bias'] * 180/np.pi
    est_bias_gyro = calibration['gyro_bias'] * 180/np.pi
    
    ax4.bar(x_pos - width/2, true_bias_gyro, width, label='True Bias', alpha=0.7)
    ax4.bar(x_pos + width/2, est_bias_gyro, width, label='Estimated Bias', alpha=0.7)
    
    ax4.set_xlabel('Axis')
    ax4.set_ylabel('Bias [deg/s]')
    ax4.set_title('Gyroscope Bias Estimation')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(axes)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('IMU Intrinsic Calibration (Section 8.4.1.3)', fontsize=14, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()


def plot_extrinsic_calibration(
    data: Dict,
    R_est: np.ndarray,
    t_est: np.ndarray,
    save_path: str = None
):
    """Plot extrinsic calibration results.
    
    Args:
        data: Extrinsic calibration data
        R_est: Estimated rotation matrix
        t_est: Estimated translation
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trajectories
    ax = axes[0, 0]
    ax.plot(data['p_sensor1'][:, 0], data['p_sensor1'][:, 1],
           'b-', label='Sensor 1', alpha=0.7)
    ax.plot(data['p_sensor2'][:, 0], data['p_sensor2'][:, 1],
           'r-', label='Sensor 2', alpha=0.7)
    ax.scatter(0, 0, c='black', s=100, marker='x', label='Origin', zorder=5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Sensor Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Aligned trajectories
    ax = axes[0, 1]
    
    # Transform sensor 1 to sensor 2 frame using estimated calibration
    p1_transformed = (R_est @ data['p_sensor1'].T).T + t_est
    
    ax.plot(data['p_sensor2'][:, 0], data['p_sensor2'][:, 1],
           'r-', label='Sensor 2 (reference)', alpha=0.7, linewidth=2)
    ax.plot(p1_transformed[:, 0], p1_transformed[:, 1],
           'b--', label='Sensor 1 (transformed)', alpha=0.7, linewidth=2)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('After Calibration (Aligned)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Calibration parameters
    ax = axes[1, 0]
    ax.axis('off')
    
    R_true = data['true_R']
    t_true = data['true_t']
    angle_true = data['true_rotation_angle'] * 180/np.pi
    angle_est = np.arctan2(R_est[1, 0], R_est[0, 0]) * 180/np.pi
    
    info_text = f"""
Extrinsic Calibration Results:

True Rotation Angle:    {angle_true:>8.2f} deg
Estimated Rotation:     {angle_est:>8.2f} deg
Error:                  {abs(angle_true - angle_est):>8.4f} deg

True Lever-arm:         [{t_true[0]:>6.3f}, {t_true[1]:>6.3f}] m
Estimated Lever-arm:    [{t_est[0]:>6.3f}, {t_est[1]:>6.3f}] m
Error:                  {np.linalg.norm(t_true - t_est):>8.4f} m

Rotation Matrix (Estimated):
  [{R_est[0,0]:>7.4f}, {R_est[0,1]:>7.4f}]
  [{R_est[1,0]:>7.4f}, {R_est[1,1]:>7.4f}]
    """
    
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
           verticalalignment='center')
    
    # Residual errors
    ax = axes[1, 1]
    
    residuals = data['p_sensor2'] - p1_transformed
    residual_norms = np.linalg.norm(residuals, axis=1)
    
    ax.plot(data['t'], residual_norms, 'g-', linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Alignment Error [m]')
    ax.set_title(f'Calibration Residuals (RMSE: {np.sqrt(np.mean(residual_norms**2)):.4f} m)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('2D Extrinsic Calibration (Section 8.4.2)', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()


def main():
    """Main entry point for calibration demo."""
    parser = argparse.ArgumentParser(
        description="Calibration Demo: Intrinsic and Extrinsic Calibration"
    )
    parser.add_argument(
        "--skip-intrinsic",
        action="store_true",
        help="Skip intrinsic calibration demo"
    )
    parser.add_argument(
        "--skip-extrinsic",
        action="store_true",
        help="Skip extrinsic calibration demo"
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
    print("Calibration Demonstration (Chapter 8, Section 8.4)")
    print("="*70)
    
    # =====================================================================
    # Part 1: Intrinsic IMU Calibration
    # =====================================================================
    
    if not args.skip_intrinsic:
        print("\n[PART 1] IMU Intrinsic Calibration (Bias Estimation)")
        print("-" * 70)
        print("Concept: During stationary period:")
        print("  - Gyroscope should read zero -> any reading is bias")
        print("  - Accelerometer should read gravity -> deviation is bias")
        print("")
        
        print("Generating synthetic stationary IMU data...")
        imu_data = generate_synthetic_imu_stationary(
            duration=10.0,
            rate=100.0
        )
        
        print(f"  Duration: {imu_data['t'][-1]:.1f}s")
        print(f"  Samples: {len(imu_data['t'])}")
        print(f"  True Accel Bias: {imu_data['true_accel_bias']}")
        print(f"  True Gyro Bias: {imu_data['true_gyro_bias'] * 180/np.pi} deg/s")
        print("")
        
        print("Estimating biases from stationary window...")
        calibration = estimate_imu_bias_stationary(
            imu_data['accel'],
            imu_data['gyro']
        )
        
        print("\n" + "="*70)
        print("IMU Calibration Results")
        print("="*70)
        print(f"{'Parameter':<30} {'Estimated':>15} {'True':>15} {'Error':>10}")
        print("-" * 70)
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            true_val = imu_data['true_accel_bias'][i]
            est_val = calibration['accel_bias'][i]
            error = abs(est_val - true_val)
            print(f"Accel Bias {axis} [m/s²]       {est_val:>15.4f} {true_val:>15.4f} {error:>10.5f}")
        
        print("")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            true_val = imu_data['true_gyro_bias'][i] * 180/np.pi
            est_val = calibration['gyro_bias'][i] * 180/np.pi
            error = abs(est_val - true_val)
            print(f"Gyro Bias {axis} [deg/s]       {est_val:>15.4f} {true_val:>15.4f} {error:>10.5f}")
        
        print("="*70)
        print(f"\nGravity identified along axis: {['X', 'Y', 'Z'][calibration['gravity_axis']]}")
        print(f"Number of samples used: {calibration['n_samples']}")
        
        # Plot
        save_path = "ch8_sensor_fusion/figs/imu_calibration.svg"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plot_imu_calibration(imu_data, calibration, save_path=save_path)
    
    # =====================================================================
    # Part 2: Extrinsic Calibration (2D)
    # =====================================================================
    
    if not args.skip_extrinsic:
        print("\n[PART 2] 2D Extrinsic Calibration (Lever-arm + Rotation)")
        print("-" * 70)
        print("Concept: Estimate relative pose between two sensors")
        print("  - Rotation matrix R (2x2)")
        print("  - Translation vector t (lever-arm)")
        print("  Model: p_sensor2 = R @ p_sensor1 + t")
        print("")
        
        print("Generating synthetic dual-sensor data...")
        ext_data = generate_synthetic_extrinsic_data(
            duration=30.0,
            rate=10.0
        )
        
        print(f"  Duration: {ext_data['t'][-1]:.1f}s")
        print(f"  Samples: {len(ext_data['t'])}")
        print(f"  True Rotation: {ext_data['true_rotation_angle'] * 180/np.pi:.2f} deg")
        print(f"  True Lever-arm: {ext_data['true_t']}")
        print("")
        
        print("Estimating extrinsic calibration (least-squares)...")
        R_est, t_est = calibrate_extrinsic_2d_least_squares(
            ext_data['p_sensor1'],
            ext_data['p_sensor2']
        )
        
        angle_est = np.arctan2(R_est[1, 0], R_est[0, 0]) * 180/np.pi
        angle_true = ext_data['true_rotation_angle'] * 180/np.pi
        
        print("\n" + "="*70)
        print("Extrinsic Calibration Results")
        print("="*70)
        print(f"{'Parameter':<30} {'Estimated':>15} {'True':>15} {'Error':>10}")
        print("-" * 70)
        print(f"Rotation Angle [deg]          {angle_est:>15.2f} {angle_true:>15.2f} {abs(angle_est - angle_true):>10.4f}")
        print(f"Lever-arm X [m]               {t_est[0]:>15.4f} {ext_data['true_t'][0]:>15.4f} {abs(t_est[0] - ext_data['true_t'][0]):>10.5f}")
        print(f"Lever-arm Y [m]               {t_est[1]:>15.4f} {ext_data['true_t'][1]:>15.4f} {abs(t_est[1] - ext_data['true_t'][1]):>10.5f}")
        print("="*70)
        
        # Compute RMSE after calibration
        p1_transformed = (R_est @ ext_data['p_sensor1'].T).T + t_est
        residuals = ext_data['p_sensor2'] - p1_transformed
        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        
        print(f"\nAlignment RMSE after calibration: {rmse:.4f} m")
        print("(Should be close to measurement noise: ~0.05 m)")
        
        # Plot
        save_path = "ch8_sensor_fusion/figs/extrinsic_calibration.svg"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plot_extrinsic_calibration(ext_data, R_est, t_est, save_path=save_path)
    
    print("\n" + "="*70)
    print("Calibration Demo Complete")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Intrinsic calibration corrects sensor-specific errors (biases)")
    print("  2. Extrinsic calibration aligns multi-sensor coordinate frames")
    print("  3. Both are prerequisites for accurate sensor fusion")
    print("  4. Real-world calibration requires careful data collection procedures")
    print("")


if __name__ == '__main__':
    main()

