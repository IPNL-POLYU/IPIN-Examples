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

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import resolve_figs_dir, save_figure, show_figures_if_requested
from core.sensors import (
    allan_variance,
    characterize_imu_noise,
)
from core.sim import (
    pink_noise_1f_fft,
    scale_to_bias_instability,
)

#: Seed for the synthetic record. The generator used a bare
#: ``np.random.default_rng()``, so every run drew a different realisation:
#: bias instability came out 11.15 deg/hr on one run and 7.85 on the next, a
#: 42% swing on a figure committed to the book. Allan variance estimates are
#: noisy by nature at long tau, which is exactly why the draw has to be pinned.
DEFAULT_SEED = 42

#: Noise the synthetic record is built from, in SI: gyro ARW rad/sqrt(s) after
#: the sqrt(3600) below, bias instability rad/s, RRW rad/s^(3/2), accel VRW
#: m/s^(3/2), accel bias instability m/s^2.
#:
#: Module-level rather than buried in the generator so that the example -- and
#: its test -- can compare what the Allan analysis recovers against what went
#: in. That comparison is the whole point of running this on synthetic data,
#: and its absence is why three unit errors survived here: nothing ever put the
#: answer next to the question.
IMU_SPECS = {
    'consumer': {
        'gyro_arw': np.deg2rad(0.5),  # deg/sqrt(hr) → rad/sqrt(s)
        'gyro_bias_instability': np.deg2rad(10.0) / 3600.0,  # deg/hr → rad/s
        'gyro_rrw': np.deg2rad(0.01),  # deg/s/sqrt(hr)
        'accel_vrw': 0.01,  # m/s/sqrt(s)
        # The "/ 3600.0" that used to be here was spurious -- a bias
        # instability in m/s^2 is already a rate, so there is nothing to
        # convert from per-hour. It made the consumer accelerometer
        # 2.8e-8 m/s^2, which is 360x *better* than the tactical entry below,
        # and small enough that the Allan curve had no flat region at all: the
        # estimator returned the white-noise floor, 5.4e-4, and the example
        # printed that as a bias instability. The tactical spec had it right.
        'accel_bias_instability': 0.0001,  # m/s²
    },
    'tactical': {
        'gyro_arw': np.deg2rad(0.05),
        'gyro_bias_instability': np.deg2rad(1.0) / 3600.0,  # deg/hr → rad/s
        'gyro_rrw': np.deg2rad(0.001),
        'accel_vrw': 0.001,
        'accel_bias_instability': 0.00001,  # m/s²
    },
}


def injected_si(grade):
    """What the generator actually puts into the record, in SI units.

    The spec table stores the two "per sqrt(hour)" quantities in their
    per-sqrt-hour form and the generator divides by sqrt(3600); this returns
    the values as ``characterize_imu_noise`` reports them, so recovered and
    injected can be compared without a conversion in between -- which is where
    the errors were.

    Args:
        grade: Key into :data:`IMU_SPECS`.

    Returns:
        Dict with the same keys as the spec, in rad/sqrt(s), rad/s,
        rad/s^(3/2), m/s^(3/2) and m/s^2.
    """
    spec = IMU_SPECS.get(grade, IMU_SPECS['consumer'])
    return {
        'gyro_arw': spec['gyro_arw'] / np.sqrt(3600),
        'gyro_bias_instability': spec['gyro_bias_instability'],
        'gyro_rrw': spec['gyro_rrw'] / np.sqrt(3600),
        'accel_vrw': spec['accel_vrw'],
        'accel_bias_instability': spec['accel_bias_instability'],
    }


def generate_imu_stationary_data(
    duration=3600.0, fs=100.0, imu_grade='consumer', return_components=False,
    seed=DEFAULT_SEED,
):
    """
    Generate synthetic stationary IMU data with realistic noise.
    
    Args:
        duration: Duration [s] (recommend 1-24 hours).
        fs: Sampling frequency [Hz].
        imu_grade: 'consumer', 'tactical', or 'navigation'.
        return_components: If True, return individual noise components
                          for debug analysis. Default: False.
        seed: Seed for the noise draws. Pinned so the committed figure can be
              regenerated; pass a different value to see another realisation.

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

    # Go through injected_si rather than converting here. The sqrt(3600) used
    # to be written out separately in this function and again where the record
    # is described, and two copies of one conversion that must agree is exactly
    # how this example's unit errors happened -- the console and the figure
    # each carried their own, and agreeing with each other made them look
    # right. One definition, consumed everywhere.
    spec = injected_si(imu_grade)

    gyro_noise_density = spec['gyro_arw']  # rad/sqrt(s)
    accel_noise_density = spec['accel_vrw']  # m/s/sqrt(s)

    # Create RNG for reproducibility
    rng = np.random.default_rng(seed)

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
        rrw_coeff = spec['gyro_rrw']  # rad/s/sqrt(s), already converted
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
    paths = save_figure(fig, figs_dir, filename)
    print(f"  [DEBUG] Saved: {paths[0]}")

    plt.close(fig)


def plot_allan_deviation(taus, adev, noise_params, sensor_type, grade, figs_dir):
    """Plot Allan deviation with identified noise parameters."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Allan deviation
    ax.loglog(taus, adev, 'b-', linewidth=2, label=f'{sensor_type} Allan Deviation')

    # This function is called for both sensors, and used to label both with the
    # gyroscope's units: the accelerometer figure reported its bias instability
    # as "122.26 °/hr", degrees per hour, for a quantity in m/s^2. It also drew
    # no white-noise marker on the accelerometer at all, because it looked only
    # for 'angle_random_walk' -- so the one parameter an hour of data recovers
    # correctly was the one the figure omitted.
    is_accel = sensor_type.lower().startswith('accel')
    if is_accel:
        white_key, white_name = 'velocity_random_walk', 'VRW'
        fmt_white = lambda v: f'{v:.5f} m/s/√s'
        fmt_bi = lambda v: f'{v:.2e} m/s²'
        fmt_rrw = lambda v: f'{v:.2e} m/s²/√s'
    else:
        white_key, white_name = 'angle_random_walk', 'ARW'
        # rad/sqrt(s) -> deg/sqrt(hr) is rad2deg then x60, because
        # sqrt(3600) = 60. The x60 was missing here as well as in the console
        # output, so the legend and the printed table were wrong in the same
        # way -- and agreeing with each other is what made it look right.
        fmt_white = lambda v: f'{np.rad2deg(v)*60:.3f} °/√hr'
        fmt_bi = lambda v: f'{np.rad2deg(v)*3600:.2f} °/hr'
        fmt_rrw = lambda v: f'{np.rad2deg(v)*60:.4f} °/s/√hr'

    # Mark identified parameters
    if white_key in noise_params:
        # White noise: read at tau=1s on the -1/2 slope.
        white_value = noise_params[white_key]
        ax.loglog(1.0, white_value, 'ro', markersize=10,
                  label=f'{white_name} = {fmt_white(white_value)}')

        # Draw reference line
        tau_ref = np.array([0.1, 10])
        arw_line = white_value * (tau_ref / 1.0)**(-0.5)
        ax.loglog(tau_ref, arw_line, 'r--', alpha=0.5, linewidth=1)
        ax.text(0.15, white_value*1.5, f'Slope = -1/2\n({white_name})',
                fontsize=9, color='red')

    if 'bias_instability' in noise_params:
        # Bias instability is read off the minimum of the curve, so put the
        # marker there. It used to be drawn at (100 s, B): the tau came from
        # `noise_params.get('bi_tau', 100.0)` and characterize_imu_noise
        # returns no 'bi_tau', so the fallback was always taken, and B itself
        # is 0.664x the minimum rather than a point on the curve. The marker
        # therefore sat off the line in both axes -- at a tau that was not the
        # minimum and a height the curve never reaches.
        bi_index = int(np.argmin(adev))
        bi_value = noise_params['bias_instability']
        ax.loglog(taus[bi_index], adev[bi_index], 'gs', markersize=10,
                  label=f'BI = {fmt_bi(bi_value)}')
        # The minimum can fall on the last tau -- with few clusters left the
        # tail is noisy and often dips, and on a curve that never flattens it
        # is simply the last point -- so label up and to the left when it does,
        # or the text runs off the bottom-right corner.
        near_right_edge = bi_index > 0.75 * len(taus)

        # Only call it a bias-instability plateau if the curve is actually flat
        # there. The accelerometer's is still falling as white noise at -0.48,
        # and labelling that "Slope = 0" made the figure assert something the
        # example's own analysis prints as NOT REACHED.
        tail = taus >= taus[-1] / 10.0
        tail_slope = np.polyfit(np.log10(taus[tail]), np.log10(adev[tail]), 1)[0]
        if abs(tail_slope) < 0.2:
            bi_text = 'Slope = 0\n(Bias Instability)'
        else:
            bi_text = f'curve minimum\n(slope {tail_slope:+.2f}, no plateau)'

        ax.annotate(
            bi_text,
            xy=(taus[bi_index], adev[bi_index]),
            xytext=(-10, 18) if near_right_edge else (8, 0),
            textcoords='offset points',
            ha='right' if near_right_edge else 'left',
            va='center', fontsize=9, color='green',
        )

    if 'rate_random_walk' in noise_params:
        # RRW: slope +1/2 at long tau. The Allan deviation of a rate random
        # walk is sigma(tau) = K*sqrt(tau/3) -- the 3 comes from the Allan
        # variance of a Wiener process, and is not a unit conversion. Dividing
        # by 3600 instead put the marker 35x below the curve it annotates.
        rrw_value = noise_params['rate_random_walk']
        rrw_tau = taus[-10] if len(taus) > 10 else taus[-1]
        rrw_adev = rrw_value * np.sqrt(rrw_tau / 3.0)
        ax.loglog(rrw_tau, rrw_adev, 'md', markersize=10,
                  label=f'RRW = {fmt_rrw(rrw_value)}')

    ax.set_xlabel('Averaging Time τ [s]', fontsize=12)
    # One sensor per figure, so name its unit rather than offering both.
    unit = '[m/s²]' if sensor_type.lower().startswith('accel') else '[rad/s]'
    ax.set_ylabel(f'Allan Deviation {unit}', fontsize=12)
    ax.set_title(f'Allan Variance: {grade.capitalize()} {sensor_type}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([taus[0], taus[-1]])

    plt.tight_layout()
    filename = f'allan_{sensor_type.lower()}_{grade}'
    paths = save_figure(fig, figs_dir, filename)
    print(f"  [OK] Saved: {paths[0]}")

    plt.close(fig)


def main():
    """Main execution."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also plot the individual noise components behind each curve.",
    )
    args = parser.parse_args()

    # Check for debug mode
    debug_mode = args.debug

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

    print("Configuration:")
    print(f"  Duration:        {duration/3600:.1f} hours")
    print(f"  Sampling Rate:   {fs} Hz")
    print(f"  IMU Grade:       {grade}")
    print("  (Note: Real calibration requires 1-24 hours of stationary data)\n")

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
    # characterize_imu_noise returns SI: ARW in rad/sqrt(s), RRW in
    # rad/s^(3/2). Both convert to a "per sqrt(hour)" unit with rad2deg then
    # x60, because sqrt(3600) = 60 -- the factor its own docstring uses.
    #
    # ARW was printed without the x60, so a 0.5 deg/sqrt(hr) gyro was reported
    # as 0.0090, which the reference table printed twelve lines below calls
    # better than navigation grade. RRW was multiplied by 3600 instead of 60,
    # sixty times too large in the other direction.
    print(f"\nGyroscope ({grade}):")
    print(f"  Angle Random Walk (ARW):     {np.rad2deg(noise_char['gyro']['angle_random_walk'])*60:.4f} deg/sqrt(hr)")
    print(f"  Bias Instability (BI):       {np.rad2deg(noise_char['gyro']['bias_instability'])*3600:.2f} deg/hr")
    print(f"  Rate Random Walk (RRW):      {np.rad2deg(noise_char['gyro']['rate_random_walk'])*60:.5f} deg/s/sqrt(hr)")

    print(f"\nAccelerometer ({grade}):")
    print(f"  Velocity Random Walk (VRW):  {noise_char['accel']['velocity_random_walk']:.5f} m/s/sqrt(s)")
    print(f"  Bias Instability:            {noise_char['accel']['bias_instability']:.6f} m/s^2")

    # The check this example never made. It runs on synthetic data, so the
    # right answer is known exactly -- and printing the recovered value next to
    # it is the only thing that would have caught the three unit errors above,
    # each of which looked entirely plausible on its own.
    injected = injected_si(grade)
    g, a = noise_char['gyro'], noise_char['accel']
    print("\n" + "-"*70)
    print("Recovered vs injected (this is synthetic data: the answer is known)")
    print("-"*70)
    print(f"  {'quantity':<28} {'injected':>11} {'recovered':>11} {'ratio':>7}")
    for label, key_in, value_out in (
        ("gyro ARW  [rad/sqrt(s)]", 'gyro_arw', g['angle_random_walk']),
        ("gyro BI   [rad/s]", 'gyro_bias_instability', g['bias_instability']),
        ("gyro RRW  [rad/s^1.5]", 'gyro_rrw', g['rate_random_walk']),
        ("accel VRW [m/s^1.5]", 'accel_vrw', a['velocity_random_walk']),
        ("accel BI  [m/s^2]", 'accel_bias_instability', a['bias_instability']),
    ):
        value_in = injected[key_in]
        print(f"  {label:<28} {value_in:11.3e} {value_out:11.3e} "
              f"{value_out / value_in:6.1f}x")

    # Two of those ratios are not estimator error -- they are parameters this
    # record cannot identify, and saying which is more useful than the numbers.
    # A parameter is only readable where the Allan curve actually shows its
    # slope, so measure the slope at long tau rather than assuming the region
    # is there. characterize_imu_noise already warns about this; putting the
    # evidence in the output means the reader does not have to catch a warning.
    print()
    for sensor, key, wanted, what in (
        ('gyro', 'gyro', +0.5, 'rate random walk'),
        ('accel', 'accel', 0.0, 'bias instability'),
    ):
        taus_s = np.asarray(noise_char[key]['taus'])
        adev_s = np.asarray(noise_char[key]['adev'])
        long_tau = taus_s >= taus_s[-1] / 10.0
        slope = np.polyfit(
            np.log10(taus_s[long_tau]), np.log10(adev_s[long_tau]), 1
        )[0]
        verdict = "as expected" if abs(slope - wanted) < 0.2 else "NOT REACHED"
        print(f"  {sensor:<5} long-tau slope {slope:+.2f} "
              f"(needs {wanted:+.2f} for {what}) -- {verdict}")

    print()
    print("  The gyro curve is still on its bias-instability shoulder and the")
    print("  accelerometer's is still falling as white noise, so the RRW and")
    print("  accel-BI figures above are read off regions that do not exist in a")
    print("  1-hour record. Lengthening it barely helps: at 4 hours the gyro")
    print("  slope is 0.02 and the RRW still 2.5x high, for 20x the runtime,")
    print("  and the accelerometer's white noise does not fall to its BI floor")
    print("  until tau = 22681 s, which needs a 63-hour record. That is the")
    print("  real lesson of the 1-24 hour guidance above, and it is why ARW and")
    print("  VRW -- both read at short tau -- come back within 10%.")

    print("\n" + "-"*70)
    print("Reference IMU Grades:")
    print("-"*70)
    print("  Grade      | ARW [deg/sqrt(hr)] | BI [deg/hr]  | Cost")
    print("  -----------|--------------------|--------------|--------")
    print("  Consumer   | 0.1 - 1.0          | 10 - 100     | $1-10")
    print("  Tactical   | 0.01 - 0.1         | 1 - 10       | $100-1k")
    print("  Navigation | < 0.01             | < 1          | $10k-100k")

    print(f"\nFigures saved to: {resolve_figs_dir(figs_dir)}/")
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
    show_figures_if_requested()


if __name__ == "__main__":
    main()

