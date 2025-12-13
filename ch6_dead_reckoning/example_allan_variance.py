"""
Example: Allan Variance for IMU Noise Characterization

Demonstrates Allan variance computation and noise parameter extraction.
Critical for IMU selection and Kalman filter tuning.

Implements:
    - Allan variance computation (Eqs. 6.56-6.58)
    - Noise parameter identification (ARW, bias instability, RRW)

Key Insight: Allan variance reveals ALL IMU noise characteristics on
            a single log-log plot. Essential for system design!

Author: Navigation Engineer
Date: December 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    allan_variance,
    characterize_imu_noise,
)


def generate_imu_stationary_data(duration=3600.0, fs=100.0, imu_grade='consumer'):
    """
    Generate synthetic stationary IMU data with realistic noise.
    
    Args:
        duration: Duration [s] (recommend 1-24 hours).
        fs: Sampling frequency [Hz].
        imu_grade: 'consumer', 'tactical', or 'navigation'.
    
    Returns:
        Tuple of (t, gyro_data, accel_data).
    """
    N = int(duration * fs)
    t = np.arange(N) / fs
    dt = 1.0 / fs
    
    # IMU noise specifications
    specs = {
        'consumer': {
            'gyro_arw': np.deg2rad(0.5),  # deg/sqrt(hr) → rad/sqrt(s)
            'gyro_bias_instability': np.deg2rad(10.0),  # deg/hr → rad/s
            'gyro_rrw': np.deg2rad(0.01),  # deg/s/sqrt(hr)
            'accel_vrw': 0.01,  # m/s/sqrt(s)
            'accel_bias_instability': 0.0001,  # m/s²
        },
        'tactical': {
            'gyro_arw': np.deg2rad(0.05),
            'gyro_bias_instability': np.deg2rad(1.0),
            'gyro_rrw': np.deg2rad(0.001),
            'accel_vrw': 0.001,
            'accel_bias_instability': 0.00001,
        },
    }
    
    spec = specs.get(imu_grade, specs['consumer'])
    
    # Convert ARW to noise density
    gyro_noise_density = spec['gyro_arw'] / np.sqrt(3600)  # rad/sqrt(s)
    accel_noise_density = spec['accel_vrw']  # m/s/sqrt(s)
    
    # Generate noise components
    gyro_data = np.zeros((N, 3))
    accel_data = np.zeros((N, 3))
    
    for axis in range(3):
        # Gyro: Angle Random Walk (white noise on angular rate)
        arw_noise = np.random.randn(N) * gyro_noise_density * np.sqrt(fs)
        
        # Bias instability (1/f noise, approximated by random walk of bias)
        bias_rw_std = spec['gyro_bias_instability'] * np.sqrt(dt)
        bias = np.cumsum(np.random.randn(N) * bias_rw_std)
        bias -= np.mean(bias)  # Remove DC
        bias *= 0.1  # Scale down (bias instability is slow)
        
        # Rate random walk (diffusion of bias)
        rrw_noise = np.cumsum(np.random.randn(N)) * spec['gyro_rrw'] * np.sqrt(dt/3600)
        
        gyro_data[:, axis] = arw_noise + bias + rrw_noise
        
        # Accel: Similar structure
        vrw_noise = np.random.randn(N) * accel_noise_density * np.sqrt(fs)
        accel_bias_rw = np.cumsum(np.random.randn(N) * spec['accel_bias_instability'] * np.sqrt(dt))
        accel_bias_rw -= np.mean(accel_bias_rw)
        accel_bias_rw *= 0.1
        
        accel_data[:, axis] = vrw_noise + accel_bias_rw
    
    return t, gyro_data, accel_data


def plot_allan_deviation(taus, adev, noise_params, sensor_type, grade, figs_dir):
    """Plot Allan deviation with identified noise parameters."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Allan deviation
    ax.loglog(taus, adev, 'b-', linewidth=2, label=f'{sensor_type} Allan Deviation')
    
    # Mark identified parameters
    if 'angle_random_walk' in noise_params:
        # ARW: read at tau=1s, slope -1/2
        arw_value = noise_params['angle_random_walk']
        ax.loglog(1.0, arw_value, 'ro', markersize=10, label=f'ARW = {np.rad2deg(arw_value):.3f} °/√hr')
        
        # Draw reference line
        tau_ref = np.array([0.1, 10])
        arw_line = arw_value * (tau_ref / 1.0)**(-0.5)
        ax.loglog(tau_ref, arw_line, 'r--', alpha=0.5, linewidth=1)
        ax.text(0.15, arw_value*1.5, 'Slope = -1/2\n(ARW)', fontsize=9, color='red')
    
    if 'bias_instability' in noise_params:
        # Bias instability: minimum of curve
        bi_value = noise_params['bias_instability']
        bi_tau = noise_params.get('bi_tau', 100.0)
        ax.loglog(bi_tau, bi_value, 'gs', markersize=10, label=f'BI = {np.rad2deg(bi_value)*3600:.2f} °/hr')
        ax.text(bi_tau*1.5, bi_value, 'Slope = 0\n(Bias Instability)', fontsize=9, color='green')
    
    if 'rate_random_walk' in noise_params:
        # RRW: slope +1/2 at long tau
        rrw_value = noise_params['rate_random_walk']
        rrw_tau = taus[-10] if len(taus) > 10 else taus[-1]
        rrw_adev = rrw_value * np.sqrt(rrw_tau / 3600)
        ax.loglog(rrw_tau, rrw_adev, 'md', markersize=10, label=f'RRW = {np.rad2deg(rrw_value)*3600:.4f} °/s/√hr')
    
    ax.set_xlabel('Averaging Time τ [s]', fontsize=12)
    ax.set_ylabel('Allan Deviation [rad/s] or [m/s²]', fontsize=12)
    ax.set_title(f'Allan Variance: {grade.capitalize()} {sensor_type}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([taus[0], taus[-1]])
    
    plt.tight_layout()
    filename = f'allan_{sensor_type.lower()}_{grade}'
    fig.savefig(figs_dir / f'{filename}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(figs_dir / f'{filename}.pdf', bbox_inches='tight')
    print(f"  [OK] Saved: {figs_dir / filename}.svg")
    
    plt.close(fig)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("Chapter 6: Allan Variance for IMU Noise Characterization")
    print("="*70)
    print("\nDemonstrates IMU noise identification using Allan variance.")
    print("Key equations: 6.56-6.58 (Allan variance and deviation)\n")
    
    # Configuration
    duration = 3600.0  # 1 hour (recommend 1-24 hours for real data)
    fs = 100.0  # Hz
    grade = 'consumer'
    
    print(f"Configuration:")
    print(f"  Duration:        {duration/3600:.1f} hours")
    print(f"  Sampling Rate:   {fs} Hz")
    print(f"  IMU Grade:       {grade}")
    print(f"  (Note: Real calibration requires 1-24 hours of stationary data)\n")
    
    # Generate synthetic IMU data
    print("Generating synthetic stationary IMU data...")
    start = time.time()
    t, gyro_data, accel_data = generate_imu_stationary_data(duration, fs, grade)
    print(f"  Time: {time.time()-start:.2f} s")
    print(f"  Samples: {len(t):,}")
    
    # Compute Allan variance for gyro (all 3 axes)
    print("\nComputing Allan variance (Gyro X-axis)...")
    start = time.time()
    taus, adev = allan_variance(gyro_data[:, 0], fs, taus=None)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f} s")
    print(f"  Tau range: {taus[0]:.2f} to {taus[-1]:.1f} s")
    
    # Characterize noise
    print("\nIdentifying noise parameters...")
    start = time.time()
    noise_char = characterize_imu_noise(gyro_data, accel_data, fs)
    print(f"  Time: {time.time()-start:.2f} s")
    
    # Create output directory
    figs_dir = Path(__file__).parent / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    # Plot gyro
    print("\nGenerating plots...")
    plot_allan_deviation(taus, adev, noise_char['gyro'], 'Gyroscope', grade, figs_dir)
    
    # Plot accel
    taus_a, adev_a = allan_variance(accel_data[:, 0], fs, taus=None)
    plot_allan_deviation(taus_a, adev_a, noise_char['accel'], 'Accelerometer', grade, figs_dir)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS - IMU Noise Characterization")
    print("="*70)
    print(f"\nGyroscope ({grade}):")
    print(f"  Angle Random Walk (ARW):     {np.rad2deg(noise_char['gyro']['angle_random_walk']):.4f} °/√hr")
    print(f"  Bias Instability (BI):       {np.rad2deg(noise_char['gyro']['bias_instability'])*3600:.2f} °/hr")
    print(f"  Rate Random Walk (RRW):      {np.rad2deg(noise_char['gyro']['rate_random_walk'])*3600:.5f} °/s/√hr")
    
    print(f"\nAccelerometer ({grade}):")
    print(f"  Velocity Random Walk (VRW):  {noise_char['accel']['velocity_random_walk']:.5f} m/s/√s")
    print(f"  Bias Instability:            {noise_char['accel']['bias_instability']:.6f} m/s²")
    
    print("\n" + "-"*70)
    print("Reference IMU Grades:")
    print("-"*70)
    print("  Grade      | ARW [°/√hr]  | BI [°/hr]    | Cost")
    print("  -----------|--------------|--------------|--------")
    print("  Consumer   | 0.1 - 1.0    | 10 - 100     | $1-10")
    print("  Tactical   | 0.01 - 0.1   | 1 - 10       | $100-1k")
    print("  Navigation | < 0.01       | < 1          | $10k-100k")
    
    print(f"\nFigures saved to: {figs_dir}/")
    print()
    print("="*70)
    print("KEY INSIGHT: Allan variance reveals ALL noise sources!")
    print("             - Slope -1/2: Angle/Velocity Random Walk")
    print("             - Slope 0:    Bias Instability (minimum)")
    print("             - Slope +1/2: Rate Random Walk")
    print("             Essential for IMU selection and filter tuning!")
    print("="*70)
    print()


if __name__ == "__main__":
    main()

