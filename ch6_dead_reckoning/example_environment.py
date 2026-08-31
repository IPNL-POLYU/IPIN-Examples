"""
Example: Environmental Sensors (Magnetometer + Barometer)

Demonstrates magnetometer heading and barometric altitude for indoor
navigation. Shows absolute reference capability and failure modes.

Implements:
    - Magnetometer tilt compensation (Eq. 6.52)
    - Magnetometer heading (Eqs. 6.51-6.53)
    - Barometric altitude (Eq. 6.54)
    - Floor change detection

Key Insight: Environmental sensors provide absolute references to bound
            drift, but suffer from indoor disturbances (magnetic, weather).

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

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
    earth_field_body,
    earth_field_map,
    mag_heading,
    pressure_to_altitude,
    smooth_measurement_simple,
    wrap_angle_diff,
)

# Seed for this example's random walk and sensor noise. Fixed so the
# committed figures can be regenerated exactly.
DEFAULT_SEED = 42


class BuildingWalk(NamedTuple):
    """Indoor building walk truth with ideal environmental sensor signals."""

    timestamps_s: np.ndarray
    true_positions_map_m: np.ndarray
    true_attitudes_rpy_rad: np.ndarray
    true_magnetic_field_body: np.ndarray
    true_pressure_pa: np.ndarray
    true_floor_ids: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return self.timestamps_s

    @property
    def pos_true(self) -> np.ndarray:
        return self.true_positions_map_m

    @property
    def att_true(self) -> np.ndarray:
        return self.true_attitudes_rpy_rad

    @property
    def mag_true(self) -> np.ndarray:
        return self.true_magnetic_field_body

    @property
    def pressure_true(self) -> np.ndarray:
        return self.true_pressure_pa

    @property
    def floor_true(self) -> np.ndarray:
        return self.true_floor_ids


def generate_building_walk(duration=180.0, dt=0.1, rng=None):
    """
    Generate multi-floor building walk with floor changes.

    Args:
        duration: Total duration [s].
        dt: Sample interval [s].
        rng: Generator for the random-walk jitter. Pass the same one used for
            the sensor noise so the two draw from a single seeded stream rather
            than two identical ones; defaults to a fresh seeded generator.

    Returns:
        BuildingWalk with semantic fields and tuple-compatible order:
        timestamps_s, true_positions_map_m, true_attitudes_rpy_rad,
        true_magnetic_field_body, true_pressure_pa, true_floor_ids.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    t = np.arange(0, duration, dt)
    N = len(t)

    pos_true = np.zeros((N, 3))
    att_true = np.zeros((N, 3))  # roll, pitch, yaw
    mag_true = np.zeros((N, 3))
    pressure_true = np.zeros(N)
    floor_true = np.zeros(N, dtype=int)

    # Building: 3 floors, 3.5m per floor
    floor_height = 3.5  # meters
    p0 = 101325.0  # Sea level pressure [Pa]
    T = 288.15  # Temperature [K]

    # Walk pattern: ground floor → 2nd floor → 3rd floor → ground
    floor_schedule = [
        (0, 60, 0),  # Ground floor for 60s
        (60, 70, 1),  # Climb to floor 1
        (70, 110, 1),  # Floor 1 for 40s
        (110, 120, 2),  # Climb to floor 2
        (120, 150, 2),  # Floor 2 for 30s
        (150, 160, 1),  # Descend to floor 1
        (160, 170, 0),  # Descend to ground
        (170, 180, 0),  # Ground floor
    ]

    # Heading: continuously rotating while walking
    # NOTE FOR STUDENTS: This trajectory has CONTINUOUSLY CHANGING heading
    # (20°/s rotation) to test magnetometer heading estimation under dynamic
    # conditions. This is different from fixed-heading examples because we're
    # specifically demonstrating magnetometer performance with magnetic disturbances.
    heading_rate = np.deg2rad(20.0)  # 20 deg/s continuous rotation

    # One fixed Earth field for the whole walk, in the map frame.
    field_map = earth_field_map()

    for k in range(N):
        # Determine current floor
        current_floor = 0
        for start, end, floor in floor_schedule:
            if start <= t[k] < end:
                current_floor = floor
                break
        floor_true[k] = current_floor

        # Position (x,y random walk, z from floor)
        if k > 0:
            pos_true[k, 0] = pos_true[k - 1, 0] + 0.1 * rng.standard_normal()
            pos_true[k, 1] = pos_true[k - 1, 1] + 0.1 * rng.standard_normal()
        pos_true[k, 2] = current_floor * floor_height

        # Attitude (device orientation changes)
        att_true[k, 0] = 0.1 * np.sin(2 * np.pi * t[k] / 10)  # Roll oscillation
        att_true[k, 1] = 0.05 * np.sin(2 * np.pi * t[k] / 15)  # Pitch oscillation
        att_true[k, 2] = heading_rate * t[k]  # Continuous rotation

        # Magnetometer: the FIXED Earth field of `earth_field_map`, resolved
        # into whatever attitude the device happens to hold. That is the only
        # honest forward model, and it is one call now
        # (core.sensors.earth_field_body) rather than a hand-rolled rotation
        # per file.
        #
        # This used to build `mag_level = [cos(yaw), sin(yaw), 0]`: a field
        # that turned with the device, invented so that the old
        # `atan2(m_y, m_x)` readout would return yaw. It made a compass
        # bearing look like an ENU heading, and the error cancelled inside
        # this file so nothing downstream could see it.
        roll, pitch, yaw = att_true[k]
        mag_true[k] = earth_field_body(roll, pitch, yaw, field_map=field_map)

        # Barometric pressure from altitude
        h = pos_true[k, 2]
        # Standard atmosphere model (Eq. 6.54 inverted)
        L = 0.0065  # Temperature lapse rate [K/m]
        g = 9.81
        M = 0.029  # Molar mass of air [kg/mol]
        R_gas = 8.314  # Gas constant [J/(mol·K)]
        pressure_true[k] = p0 * (1 - L * h / T) ** (g * M / (R_gas * L))

    return BuildingWalk(
        timestamps_s=t,
        true_positions_map_m=pos_true,
        true_attitudes_rpy_rad=att_true,
        true_magnetic_field_body=mag_true,
        true_pressure_pa=pressure_true,
        true_floor_ids=floor_true,
    )


def add_env_sensor_noise(mag_true, pressure_true, t, dt, rng=None):
    """Add realistic environmental sensor noise + disturbances.

    Args:
        mag_true: Noise-free magnetic field, shape (N, 3).
        pressure_true: Noise-free pressure [Pa], shape (N,).
        t: Time vector [s], shape (N,).
        dt: Sample interval [s].
        rng: Generator for the noise draws; defaults to a fresh seeded one.
            The magnetic disturbances below are what the heading estimate has
            to survive, so redrawing them each run changed the figure without
            any change to the code.

    Returns:
        Tuple of (mag_meas, pressure_meas).
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    N = len(mag_true)
    duration = t[-1]

    # Every magnetic perturbation below is quoted as a FRACTION of the
    # horizontal field and scaled here. The field used to be a unit vector,
    # so the bare numbers were already fractions; keeping them fractions
    # means the heading errors this figure exists to show are unchanged by
    # the switch to a physical 45 uT field, and the constants stay readable
    # (0.5 is a *severe* disturbance, half the field, whatever the units).
    horizontal_ut = float(np.hypot(*earth_field_map()[:2]))

    # Magnetometer noise
    mag_noise = rng.standard_normal((N, 3)) * 0.05 * horizontal_ut

    # Magnetic disturbances (steel structures)
    mag_disturbance = np.zeros((N, 3))
    disturbance_times = [(30, 50), (100, 120)]  # Near steel at these times
    for start, end in disturbance_times:
        mask = (t >= start) & (t < end)
        # Strong disturbance: shifts mag field direction
        mag_disturbance[mask] = (
            rng.standard_normal((np.sum(mask), 3)) * 0.5 * horizontal_ut
        )

    # Hard-iron offset (constant bias from device)
    hard_iron = np.array([0.1, -0.15, 0.05]) * horizontal_ut

    mag_meas = mag_true + mag_noise + mag_disturbance + hard_iron

    # Barometer noise
    pressure_noise = rng.standard_normal(N) * 10.0  # Pa (typical noise)

    # Slow weather drift (pressure changes over time)
    weather_drift = 50.0 * np.sin(2 * np.pi * t / (duration))  # ±50 Pa drift

    pressure_meas = pressure_true + pressure_noise + weather_drift

    return mag_meas, pressure_meas


def run_mag_heading(t, mag_meas, att_true):
    """Compute heading from magnetometer."""
    N = len(t)
    heading_est = np.zeros(N)

    for k in range(N):
        roll, pitch, _ = att_true[k]
        heading_est[k] = mag_heading(mag_meas[k], roll, pitch, declination=0.0)

    return heading_est


def run_baro_altitude(pressure_meas, p0=101325.0, T=288.15):
    """Compute altitude from barometer."""
    alt_est = np.array([pressure_to_altitude(p, p0, T) for p in pressure_meas])
    return alt_est


def plot_results(
    t,
    att_true,
    mag_meas,
    heading_est,
    pressure_meas,
    alt_est,
    floor_true,
    floor_detected,
    figs_dir,
):
    """Generate plots."""

    heading_true = att_true[:, 2]  # yaw

    # Figure 1: Magnetometer heading
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Wrap the truth for display. A magnetometer reports a wrapped angle, so
    # plotting it against an unwrapped truth that climbs to 3600 deg puts the
    # two traces in different parts of the axes and makes the comparison the
    # panel invites impossible -- they never overlap even when they agree.
    # The error below still uses the unwrapped truth via wrap_angle_diff.
    heading_true_wrapped = np.arctan2(np.sin(heading_true), np.cos(heading_true))
    ax1.plot(
        t,
        np.rad2deg(heading_true_wrapped),
        "k-",
        linewidth=2,
        label="True Heading (wrapped)",
    )
    ax1.plot(
        t, np.rad2deg(heading_est), "b-", linewidth=2, alpha=0.7, label="Mag Heading"
    )
    ax1.set_ylabel("Heading [deg]", fontsize=12)
    ax1.set_title(
        "Magnetometer Example: Heading with Disturbances",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Compute heading error with proper angle wrapping
    heading_error_rad = np.array(
        [
            wrap_angle_diff(heading_est[i], heading_true[i])
            for i in range(len(heading_est))
        ]
    )
    heading_error = np.abs(np.rad2deg(heading_error_rad))  # Absolute error in degrees
    # Note: By using wrap_angle_diff, heading_error is guaranteed to be <= 180°
    ax2.plot(t, heading_error, "r-", linewidth=2)
    ax2.axhline(10, color="orange", linestyle="--", label="10° threshold")
    ax2.fill_between(
        t,
        0,
        100,
        where=(t >= 30) & (t < 50),
        alpha=0.2,
        color="red",
        label="Disturbance zone",
    )
    ax2.fill_between(t, 0, 100, where=(t >= 100) & (t < 120), alpha=0.2, color="red")
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Heading Error [deg]", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])

    plt.tight_layout()
    paths = save_figure(fig1, figs_dir, "environment_mag_heading")
    print(f"  [OK] Saved: {paths[0]}")

    # Figure 2: Barometric altitude and floor detection
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    floor_heights = floor_true * 3.5
    ax1.plot(
        t,
        floor_heights,
        "k-",
        linewidth=3,
        label="True Altitude",
        drawstyle="steps-post",
    )
    ax1.plot(t, alt_est, "b-", linewidth=1.5, alpha=0.7, label="Baro Altitude")
    ax1.set_ylabel("Altitude [m]", fontsize=12)
    ax1.set_title(
        "Barometer Example: Altitude and Floor Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        t, floor_true, "k-", linewidth=3, label="True Floor", drawstyle="steps-post"
    )
    ax2.plot(
        t,
        floor_detected,
        "b--",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Detected Floor",
    )
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Floor Number", fontsize=12)
    ax2.set_yticks([0, 1, 2])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[-1]])

    plt.tight_layout()
    paths = save_figure(fig2, figs_dir, "environment_baro_altitude")
    print(f"  [OK] Saved: {paths[0]}")

    plt.close("all")

    return heading_error


def main():
    """Main execution."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    print("\n" + "=" * 70)
    print("Chapter 6: Environmental Sensors (Magnetometer + Barometer)")
    print("=" * 70)
    print("\nDemonstrates absolute references with indoor challenges.")
    print("Key equations: 6.51-6.54 (mag heading, tilt comp, baro altitude)\n")

    duration = 180.0
    dt = 0.1

    print("Configuration:")
    print(f"  Duration:        {duration} s")
    print("  Scenario:        Multi-floor building walk")
    print("  Floor Height:    3.5 m\n")

    print("Generating trajectory...")
    # One generator for the whole run, so the walk jitter and the sensor noise
    # are consecutive draws from a single seeded stream rather than two copies
    # of the same one.
    rng = np.random.default_rng(DEFAULT_SEED)
    t, pos_true, att_true, mag_true, pressure_true, floor_true = generate_building_walk(
        duration, dt, rng=rng
    )
    print(f"  Floors visited:  {np.unique(floor_true)}")

    print("\nAdding sensor noise + disturbances...")
    mag_meas, pressure_meas = add_env_sensor_noise(
        mag_true, pressure_true, t, dt, rng=rng
    )

    print("\nComputing magnetometer heading...")
    start = time.time()
    heading_est = run_mag_heading(t, mag_meas, att_true)
    print(f"  Time: {time.time() - start:.3f} s")

    print("\nComputing barometric altitude...")
    start = time.time()
    alt_est = run_baro_altitude(pressure_meas)

    # Smooth altitude
    alt_smooth = np.zeros_like(alt_est)
    alt_smooth[0] = alt_est[0]
    for k in range(1, len(alt_est)):
        alt_smooth[k] = smooth_measurement_simple(
            alt_smooth[k - 1], alt_est[k], alpha=0.1
        )

    # Detect floors from absolute altitude relative to the known start floor.
    # The simple detect_floor_change() helper is still useful for two-sample
    # step changes, but applying it to adjacent smoothed samples here misses
    # most floor transitions: smoothing deliberately spreads a 3.5 m step over
    # many samples, so no individual sample crosses the 1.5 m threshold. For a
    # barometer demo with a known starting floor, the readable baseline is:
    # altitude / floor_height -> nearest integer floor, then clamp to building.
    floor_height = 3.5
    floor_detected = np.rint(alt_smooth / floor_height).astype(int)
    floor_detected = np.clip(floor_detected, 0, 2)

    print(f"  Time: {time.time() - start:.3f} s")

    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")
    heading_error = plot_results(
        t,
        att_true,
        mag_meas,
        heading_est,
        pressure_meas,
        alt_smooth,
        floor_true,
        floor_detected,
        figs_dir,
    )

    # Metrics
    heading_rmse = np.sqrt(np.mean(heading_error**2))
    alt_error = np.abs(alt_smooth - floor_true * 3.5)
    alt_rmse = np.sqrt(np.mean(alt_error**2))
    floor_accuracy = np.sum(floor_detected == floor_true) / len(floor_true) * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Magnetometer Heading:")
    print(f"  RMSE:             {heading_rmse:.1f}°")
    print(f"  Max error:        {np.max(heading_error):.1f}°")
    print("  (Note: Large errors during disturbances at 30-50s, 100-120s)")
    print()
    print("Barometric Altitude:")
    print(f"  RMSE:             {alt_rmse:.2f} m")
    print(f"  Floor Accuracy:   {floor_accuracy:.1f}%")
    print()
    print(f"Figures saved to: {resolve_figs_dir(figs_dir)}/")
    print()
    print("=" * 70)
    print("KEY INSIGHT: Environmental sensors provide absolute references!")
    print("             Magnetometer: bounds heading drift (when clean).")
    print("             Barometer: provides floor-level positioning.")
    print("             BUT sensitive to indoor disturbances (steel, weather).")
    print("=" * 70)
    print()
    show_figures_if_requested()


if __name__ == "__main__":
    main()
