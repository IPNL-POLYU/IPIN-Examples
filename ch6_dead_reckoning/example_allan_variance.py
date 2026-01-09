"""
Example: Allan Variance for IMU Noise Characterization

Demonstrates Allan variance computation and noise parameter extraction.
Critical for IMU selection and Kalman filter tuning.

Implements:
    - Allan variance computation (Eqs. 6.56-6.58)
    - Noise parameter identification (ARW, bias instability, RRW)
    - Pink noise (1/f) generation for bias instability
    - Debug mode for component-wise Allan deviation analysis

Key Insight: Allan variance reveals ALL IMU noise characteristics on
            a single log-log plot. Essential for system design!

Author: Li-Ta Hsu
Date: December 2025
"""

import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.sensors import (
    allan_variance,
    characterize_imu_noise,
)
from core.sim import (
    pink_noise_1f_fft,
    scale_to_bias_instability,
)


def generate_imu_stationary_data(
    duration=3600.0, fs=100.0, imu_grade='consumer', return_components=False
):
    """
    Generate synthetic stationary IMU data with realistic noise.
    
    Args:
        duration: Duration [s] (recommend 1-24 hours).
        fs: Sampling frequency [Hz].
        imu_grade: 'consumer', 'tactical', or 'navigation'.
        return_components: If True, return individual noise components
                          for debug analysis. Default: False.
    
    Returns:
        If return_components=False:
            Tuple of (t, gyro_data, accel_data).
        If return_components=True:
            Tuple of (t, gyro_data, accel_data, gyro_components, accel_components).
            where gyro_components = {'arw': ..., 'bi': ..., 'rrw': ...}
    """
    N = int(duration * fs)
    t = np.arange(N) / fs
    dt = 1.0 / fs
    
    # IMU noise specifications
    specs = {
        'consumer': {
            'gyro_arw': np.deg2rad(0.5),  # deg/sqrt(hr) → rad/sqrt(s)
            'gyro_bias_instability': np.deg2rad(10.0) / 3600.0,  # deg/hr → rad/s
            'gyro_rrw': np.deg2rad(0.01),  # deg/s/sqrt(hr)
            'accel_vrw': 0.01,  # m/s/sqrt(s)
            'accel_bias_instability': 0.0001 / 3600.0,  # m/s² (converted from per-hr)
        },
        'tactical': {
            'gyro_arw': np.deg2rad(0.05),
            'gyro_bias_instability': np.deg2rad(1.0) / 3600.0,  # deg/hr → rad/s
            'gyro_rrw': np.deg2rad(0.001),
            'accel_vrw': 0.001,
            'accel_bias_instability': 0.00001,  # m/s²
        },
    }
    
    spec = specs.get(imu_grade, specs['consumer'])
    
    # Convert ARW to noise density
    gyro_noise_density = spec['gyro_arw'] / np.sqrt(3600)  # rad/sqrt(s)
    accel_noise_density = spec['accel_vrw']  # m/s/sqrt(s)
    
    # Create RNG for reproducibility
    rng = np.random.default_rng()
    
    # Create tau grid for BI scaling (used by scale_to_bias_instability)
    tau_grid = np.logspace(0, 3, 50)  # 1s to 1000s
    
    # Generate noise components
    gyro_data = np.zeros((N, 3))
    accel_data = np.zeros((N, 3))
    
    # For debug mode: store individual components (first axis only)
    gyro_components = {}
    accel_components = {}
    
    for axis in range(3):
        # === GYRO: ARW + BI + RRW ===
        
        # 1) Angle Random Walk (white noise on angular rate, slope -1/2)
        arw_noise = rng.standard_normal(N) * gyro_noise_density * np.sqrt(fs)
        
        # 2) Bias Instability (1/f pink noise, slope ~0)
        # Generate unit pink noise
        pink_unit = pink_noise_1f_fft(N, fs, rng=rng)
        # Scale to match target BI (using Allan deviation convention)
        bi_noise = scale_to_bias_instability(
            pink_unit=pink_unit,
            target_bi_rad_s=spec['gyro_bias_instability'],
            allan_sigma_func=allan_variance,
            tau_grid_s=tau_grid,
            fs=fs,
            bi_factor=0.664,
        )
        
        # 3) Rate Random Walk (diffusion of bias, slope +1/2)
        # Single random walk term (NOT double cumsum)
        rrw_coeff = spec['gyro_rrw'] / np.sqrt(3600)  # Convert to rad/s/sqrt(s)
        rrw_bias = np.cumsum(rng.standard_normal(N)) * rrw_coeff * np.sqrt(dt)
        
        # Combine all three components
        gyro_data[:, axis] = arw_noise + bi_noise + rrw_bias
        
        # Store components for first axis (debug mode)
        if axis == 0 and return_components:
            gyro_components['arw'] = arw_noise
            gyro_components['bi'] = bi_noise
            gyro_components['rrw'] = rrw_bias
        
        # === ACCEL: VRW + BI ===
        
        # 1) Velocity Random Walk (white noise, slope -1/2)
        vrw_noise = rng.standard_normal(N) * accel_noise_density * np.sqrt(fs)
        
        # 2) Bias Instability (1/f pink noise, slope ~0)
        pink_unit_accel = pink_noise_1f_fft(N, fs, rng=rng)
        accel_bi_noise = scale_to_bias_instability(
            pink_unit=pink_unit_accel,
            target_bi_rad_s=spec['accel_bias_instability'],
            allan_sigma_func=allan_variance,
            tau_grid_s=tau_grid,
            fs=fs,
            bi_factor=0.664,
        )
        
        # Combine components
        accel_data[:, axis] = vrw_noise + accel_bi_noise
        
        # Store components for first axis (debug mode)
        if axis == 0 and return_components:
            accel_components['vrw'] = vrw_noise
            accel_components['bi'] = accel_bi_noise
    
    if return_components:
        return t, gyro_data, accel_data, gyro_components, accel_components
    else:
        return t, gyro_data, accel_data


def plot_allan_deviation_components(
    fs, components, sensor_type, grade, figs_dir
):
    """
    Plot Allan deviation for individual noise components (debug mode).
    
    This helps verify that each component produces the expected slope:
        - ARW (white noise): slope -1/2
        - BI (pink noise): slope ~0 (flat region)
        - RRW (random walk): slope +1/2
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = {'arw': 'blue', 'bi': 'green', 'rrw': 'red', 'vrw': 'blue'}
    labels = {
        'arw': 'ARW (Angle Random Walk)',
        'bi': 'BI (Bias Instability)',
        'rrw': 'RRW (Rate Random Walk)',
        'vrw': 'VRW (Velocity Random Walk)',
    }
    expected_slopes = {'arw': -0.5, 'bi': 0.0, 'rrw': 0.5, 'vrw': -0.5}
    
    tau_grid = np.logspace(0, 3, 50)  # 1s to 1000s
    
    for key, component_data in components.items():
        # Compute Allan deviation
        taus, sigma = allan_variance(component_data, fs, tau_grid)
        
        # Plot
        color = colors.get(key, 'black')
        label = labels.get(key, key.upper())
        ax.loglog(
            taus, sigma, '-', color=color, linewidth=2, label=label, alpha=0.8
        )
        
        # Add expected slope indicator
        slope = expected_slopes.get(key, 0.0)
        # Draw reference line at mid-range
        tau_mid = 10 ** ((np.log10(taus[0]) + np.log10(taus[-1])) / 2)
        idx_mid = np.argmin(np.abs(taus - tau_mid))
        sigma_mid = sigma[idx_mid]
        
        tau_ref = np.array([tau_mid / 3, tau_mid * 3])
        sigma_ref = sigma_mid * (tau_ref / tau_mid) ** slope
        ax.loglog(tau_ref, sigma_ref, '--', color=color, alpha=0.4, linewidth=1)
        
        # Add slope annotation
        slope_text = f'slope = {slope:+.1f}'
        ax.text(
            tau_mid * 1.5,
            sigma_mid * 1.2,
            slope_text,
            fontsize=9,
            color=color,
            style='italic',
        )
    
    ax.set_xlabel('Averaging Time τ [s]', fontsize=13, fontweight='bold')
    ax.set_ylabel(
        'Allan Deviation [rad/s] or [m/s²]', fontsize=13, fontweight='bold'
    )
    ax.set_title(
        f'Allan Variance Component Analysis: {grade.capitalize()} {sensor_type}\n'
        'Debug Mode: Individual Noise Components',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_xlim([taus[0], taus[-1]])
    
    plt.tight_layout()
    filename = f'allan_{sensor_type.lower()}_{grade}_debug_components'
    fig.savefig(figs_dir / f'{filename}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(figs_dir / f'{filename}.pdf', bbox_inches='tight')
    print(f"  [DEBUG] Saved: {figs_dir / filename}.svg")
    
    plt.close(fig)


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
    # Check for debug mode
    debug_mode = '--debug' in sys.argv
    
    print("\n" + "="*70)
    print("Chapter 6: Allan Variance for IMU Noise Characterization")
    print("="*70)
    print("\nDemonstrates IMU noise identification using Allan variance.")
    print("Key equations: 6.56-6.58 (Allan variance and deviation)")
    if debug_mode:
        print("\n[DEBUG MODE] Will plot individual noise components.")
    print()
    
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
    if debug_mode:
        result = generate_imu_stationary_data(
            duration, fs, grade, return_components=True
        )
        t, gyro_data, accel_data, gyro_components, accel_components = result
    else:
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
    
    # Debug mode: plot individual components
    if debug_mode:
        print("\n[DEBUG MODE] Plotting individual noise components...")
        plot_allan_deviation_components(
            fs, gyro_components, 'Gyroscope', grade, figs_dir
        )
        plot_allan_deviation_components(
            fs, accel_components, 'Accelerometer', grade, figs_dir
        )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS - IMU Noise Characterization")
    print("="*70)
    print(f"\nGyroscope ({grade}):")
    print(f"  Angle Random Walk (ARW):     {np.rad2deg(noise_char['gyro']['angle_random_walk']):.4f} deg/sqrt(hr)")
    print(f"  Bias Instability (BI):       {np.rad2deg(noise_char['gyro']['bias_instability'])*3600:.2f} deg/hr")
    print(f"  Rate Random Walk (RRW):      {np.rad2deg(noise_char['gyro']['rate_random_walk'])*3600:.5f} deg/s/sqrt(hr)")
    
    print(f"\nAccelerometer ({grade}):")
    print(f"  Velocity Random Walk (VRW):  {noise_char['accel']['velocity_random_walk']:.5f} m/s/sqrt(s)")
    print(f"  Bias Instability:            {noise_char['accel']['bias_instability']:.6f} m/s^2")
    
    print("\n" + "-"*70)
    print("Reference IMU Grades:")
    print("-"*70)
    print("  Grade      | ARW [deg/sqrt(hr)] | BI [deg/hr]  | Cost")
    print("  -----------|--------------------|--------------|--------")
    print("  Consumer   | 0.1 - 1.0          | 10 - 100     | $1-10")
    print("  Tactical   | 0.01 - 0.1         | 1 - 10       | $100-1k")
    print("  Navigation | < 0.01             | < 1          | $10k-100k")
    
    print(f"\nFigures saved to: {figs_dir}/")
    if debug_mode:
        print("\n[DEBUG MODE] Component-wise plots show expected slopes:")
        print("  ARW (white):       -1/2 slope (short tau)")
        print("  BI (pink):         ~0 slope (flat region at mid tau)")
        print("  RRW (random walk): +1/2 slope (long tau)")
    print()
    print("="*70)
    print("KEY INSIGHT: Allan variance reveals ALL noise sources!")
    print("             - Slope -1/2: Angle/Velocity Random Walk")
    print("             - Slope 0:    Bias Instability (minimum)")
    print("             - Slope +1/2: Rate Random Walk")
    print("             Essential for IMU selection and filter tuning!")
    if debug_mode:
        print("\nTo run without debug mode: python example_allan_variance.py")
    else:
        print("\nTo see component breakdown: python example_allan_variance.py --debug")
    print("="*70)
    print()


if __name__ == "__main__":
    main()

