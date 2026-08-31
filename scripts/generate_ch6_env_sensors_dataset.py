"""
Generate Ch6 Environmental Sensors Dataset (Magnetometer + Barometer).

This script generates synthetic environmental sensor datasets for indoor
navigation using magnetometer (heading) and barometer (altitude). Demonstrates
absolute measurement capability and indoor disturbance challenges.

Key Learning Objectives:
    - Magnetometer provides absolute heading (no drift!)
    - Barometer provides altitude for floor detection
    - Indoor magnetic disturbances corrupt heading
    - Weather pressure changes affect barometric altitude
    - Tilt compensation is essential for magnetometer accuracy

Implements Equations:
    - Eq. (6.51): Magnetometer heading definition
    - Eq. (6.52): Tilt compensation for magnetometer
    - Eq. (6.53): Heading computation from field
    - Eq. (6.54): Barometric altitude from pressure
    - Eq. (6.55): Exponential smoothing filter

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensors import (
    earth_field_body,
    earth_field_map,
    mag_heading,
    pressure_to_altitude,
    smooth_measurement_simple,
    wrap_angle_diff,
)

#: The Earth field these datasets are synthesised in: fixed in the map
#: frame, so the reading changes only because the walker turns. Declination
#: is left at zero here, which makes magnetic north true north and the
#: correction a no-op -- the dataset is a heading demonstration, and a
#: constant offset between two norths would only obscure it.
EARTH_FIELD_MAP = earth_field_map()


def _runs_of(flag):
    """Yield (start, stop) inclusive index pairs for each True run in *flag*."""
    runs = []
    start = None
    for i, value in enumerate(flag):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flag) - 1))
    return runs


def generate_building_walk(
    duration: float = 180.0,
    floor_height: float = 3.5,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate multi-floor building walk trajectory.

    Args:
        duration: Total duration in seconds.
        floor_height: Height of each floor in meters.
        dt: Time step in seconds.

    Returns:
        Tuple of (t, pos, att, mag, pressure, floor):
            - t: Time array [N]
            - pos: 3D positions [N, 3] in meters
            - att: Attitude (roll, pitch, yaw) [N, 3] in radians
            - mag: Magnetometer [N, 3] in microTesla
            - pressure: Pressure [N] in Pascals
            - floor: Floor number [N] (0 = ground, 1 = first, etc.)
    """
    t = np.arange(0, duration, dt)
    N = len(t)

    # Constants
    # The Earth field is EARTH_FIELD_MAP, module level. It used to be two
    # local constants here, MAG_NORTH = 20 and MAG_DOWN = 40 uT, spent
    # building the field by hand a few lines below.
    P0 = 101325.0  # Sea level pressure (Pa)
    GRAVITY = 9.81

    # Initialize arrays
    pos = np.zeros((N, 3))
    att = np.zeros((N, 3))  # [roll, pitch, yaw]
    mag = np.zeros((N, 3))
    pressure = np.zeros(N)
    floor_num = np.zeros(N, dtype=int)

    # Trajectory phases
    # Phase 1: Ground floor walk (0-60s)
    # Phase 2: Stairs up to floor 1 (60-90s)
    # Phase 3: Floor 1 walk (90-120s)
    # Phase 4: Stairs up to floor 2 (120-150s)
    # Phase 5: Floor 2 walk (150-180s)

    # Every phase starts where the previous one ended. The stairwell is the
    # hinge: the ground-floor circle begins and ends at (10, 0), so that is
    # where the stairs go, and the figure-8 above is centred on the same point
    # so it both leaves from and returns to the stairhead.
    STAIR_XY = (10.0, 0.0)
    CIRCLE_RADIUS = 10.0

    for i in range(N):
        time_now = t[i]

        if time_now < 60:
            # Phase 1: ground floor, one circle anticlockwise from (10, 0).
            current_floor = 0
            omega = 2 * np.pi / 60.0
            x = CIRCLE_RADIUS * np.cos(omega * time_now)
            y = CIRCLE_RADIUS * np.sin(omega * time_now)
            z = 0.0

        elif time_now < 90:
            # Phase 2: stairs to floor 1, standing at the stairhead.
            progress = (time_now - 60) / 30.0
            current_floor = 0  # still climbing
            x, y = STAIR_XY
            z = progress * floor_height

        elif time_now < 120:
            # Phase 3: floor 1, figure-8 centred on the stairhead, so it starts
            # and finishes there.
            current_floor = 1
            phase = (time_now - 90) / 30.0 * 2 * np.pi
            x = STAIR_XY[0] + 8.0 * np.sin(phase)
            y = STAIR_XY[1] + 4.0 * np.sin(2 * phase)
            z = floor_height

        elif time_now < 150:
            # Phase 4: stairs to floor 2, same stairhead.
            progress = (time_now - 120) / 30.0
            current_floor = 1  # still climbing
            x, y = STAIR_XY
            z = floor_height + progress * floor_height

        else:
            # Phase 5: floor 2, a flattened loop rather than a line walked back
            # and forth. A line forces the walker to reverse heading in a single
            # sample; a loop turns through the same 180 degrees continuously,
            # which is both what a person does and something a gyro can follow.
            current_floor = 2
            phase = ((time_now - 150) % 20.0) / 20.0
            x = STAIR_XY[0] + 5.0 * (1.0 - np.cos(2 * np.pi * phase))
            y = STAIR_XY[1] + 1.5 * np.sin(2 * np.pi * phase)
            z = 2 * floor_height

        # Store position
        pos[i] = [x, y, z]
        floor_num[i] = current_floor

    # Heading follows the horizontal velocity rather than being asserted per
    # phase. Asserting it is what let the figure-8 ship a swapped atan2 for a
    # sixth of the run: the trajectory said one thing and the heading another,
    # and nothing compared them. Deriving it cannot disagree with the motion.
    #
    # On the stairs the walker is stationary in x-y, so there is no direction to
    # read; hold the heading from the last moving sample, which is what someone
    # standing on a landing facing the way they arrived actually does.
    vx = np.gradient(pos[:, 0], t)
    vy = np.gradient(pos[:, 1], t)
    speed_xy = np.hypot(vx, vy)
    MOVING = 0.05  # m/s

    moving = speed_xy > MOVING
    yaw_series = np.where(moving, np.arctan2(vy, vx), np.nan)

    # On a landing the walker is stationary in x-y, so there is no direction to
    # read. Holding the arrival heading is wrong at the far end: they arrive at
    # the stairs heading north and leave along the figure-8 at 45 degrees, so
    # the whole turn lands in one sample -- 447 deg/s, which is still a step, a
    # smaller one. Turn through it instead, over the climb, which is what a
    # person on a staircase does.
    index = np.arange(N)
    known = index[moving]
    for start, stop in _runs_of(~moving):
        before = known[known < start]
        after = known[known > stop]
        if not len(before) or not len(after):
            # A stationary run at either end has only one heading to take.
            fill = yaw_series[before[-1]] if len(before) else yaw_series[after[0]]
            yaw_series[start : stop + 1] = fill
            continue
        a, b = yaw_series[before[-1]], yaw_series[after[0]]
        # Turn the short way round: the difference is wrapped before it is
        # spread, so a turn from +170 to -170 is 20 degrees and not 340.
        sweep = (b - a + np.pi) % (2 * np.pi) - np.pi
        steps = np.linspace(0.0, 1.0, stop - start + 3)[1:-1]
        yaw_series[start : stop + 1] = a + sweep * steps

    # Accumulate rather than wrap. atan2 returns (-pi, pi], and storing that is
    # what tests/ch6_dead_reckoning/test_heading_error_is_wrapped.py exists to
    # forbid: the true yaw has to leave (-pi, pi] or the naive
    # min(|d|, 2pi - |d|) reduction never goes negative here, and the file stops
    # discriminating between the broken form and the correct one. One circuit of
    # the ground floor is a full turn, so unwrapping carries the heading past pi
    # exactly as the original closed form did.
    yaw_series = np.unwrap(yaw_series)

    for i in range(N):
        time_now = t[i]
        yaw = yaw_series[i]

        # Attitude (slight device tilt during walking)
        roll = 0.1 * np.sin(2 * np.pi * 2.0 * time_now)  # 0.1 rad oscillation
        pitch = 0.05 * np.cos(2 * np.pi * 2.0 * time_now)
        att[i] = [roll, pitch, yaw]

        # Magnetometer: one FIXED Earth field in the map frame, resolved into
        # the device's attitude. The field does not turn with the walker --
        # that is the whole mechanism the chapter's heading estimate inverts,
        # and `earth_field_body` applies exactly the transpose of the attitude
        # matrix strapdown integration uses, so the two cannot disagree.
        #
        # This used to build a level-frame field along `[cos(yaw), sin(yaw)]`,
        # which turns with the device, chosen so that the old compass-style
        # `atan2(m_y, m_x)` readout would return yaw. The dataset and the
        # library were wrong in opposite directions and every check between
        # them passed.
        mag[i] = earth_field_body(roll, pitch, yaw, field_map=EARTH_FIELD_MAP)

        # Barometric pressure (decreases with altitude)
        # International barometric formula (Eq. 6.54)
        T0 = 288.15  # K (15°C at sea level)
        L = 0.0065  # K/m (temperature lapse rate)
        M = 0.0289644  # kg/mol (molar mass of air)
        R = 8.31447  # J/(mol·K) (gas constant)
        g = GRAVITY
        exponent = (g * M) / (R * L)

        # Read the altitude back from the stored position. Using the loop
        # variable z here would silently pick up whatever the *position* loop
        # left behind -- it did, and pinned every sample at the final 7.0 m,
        # for an 84 Pa constant error that read as a barometer three times
        # worse than it is.
        altitude = pos[i, 2]
        pressure[i] = P0 * (1 - L * altitude / T0) ** exponent

    return t, pos, att, mag, pressure, floor_num


def add_env_sensor_noise(
    mag_true: np.ndarray,
    pressure_true: np.ndarray,
    t: np.ndarray,
    mag_noise: float = 2.0,
    mag_disturbance: bool = False,
    disturbance_locations: Optional[List[Tuple[float, float]]] = None,
    pressure_noise: float = 10.0,
    weather_drift: float = 50.0,
    dt: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic noise and disturbances to environmental sensors.

    Args:
        mag_true: True magnetometer [N, 3] microTesla.
        pressure_true: True pressure [N] Pascals.
        t: Time array [N] seconds.
        mag_noise: Magnetometer noise std dev (microTesla).
        mag_disturbance: Whether to add magnetic disturbances.
        disturbance_locations: List of (start_time, end_time) for disturbances.
        pressure_noise: Pressure noise std dev (Pa).
        weather_drift: Weather-induced pressure drift (Pa).
        dt: Time step.
        seed: Random seed.

    Returns:
        Tuple of (mag_meas, pressure_meas).
    """
    rng = np.random.default_rng(seed)

    # Magnetometer noise
    mag_meas = mag_true + rng.normal(0, mag_noise, mag_true.shape)

    # Add magnetic disturbances (indoor steel structures, electronics)
    if mag_disturbance and disturbance_locations is not None:
        for start_t, end_t in disturbance_locations:
            mask = (t >= start_t) & (t <= end_t)
            # Add anomalous field (simulates nearby steel/electronics)
            disturbance_field = np.array([15.0, 10.0, 5.0])  # microTesla
            mag_meas[mask] += disturbance_field

    # Pressure noise
    pressure_meas = pressure_true + rng.normal(0, pressure_noise, pressure_true.shape)

    # Weather-induced pressure drift (slow variation)
    if weather_drift > 0:
        drift = weather_drift * np.sin(2 * np.pi * t / (t[-1] * 0.5))
        pressure_meas += drift

    return mag_meas, pressure_meas


def run_mag_heading_estimation(
    t: np.ndarray,
    mag_meas: np.ndarray,
    att: np.ndarray,
) -> np.ndarray:
    """
    Compute heading from magnetometer with tilt compensation.

    Args:
        t: Time array [N].
        mag_meas: Magnetometer measurements [N, 3] microTesla.
        att: Attitude (roll, pitch, yaw) [N, 3] radians.

    Returns:
        Estimated heading [N] radians.
    """
    N = len(t)
    heading = np.zeros(N)

    for k in range(N):
        roll = att[k, 0]
        pitch = att[k, 1]
        heading[k] = mag_heading(mag_meas[k], roll, pitch, declination=0.0)

    return heading


def run_baro_altitude_estimation(
    pressure_meas: np.ndarray,
    p0: float = 101325.0,
) -> np.ndarray:
    """
    Compute altitude from barometric pressure.

    Args:
        pressure_meas: Pressure measurements [N] Pa.
        p0: Reference sea-level pressure (Pa).

    Returns:
        Estimated altitude [N] meters.
    """
    altitude = np.array([pressure_to_altitude(p, p0) for p in pressure_meas])
    return altitude


def save_dataset(
    output_dir: Path,
    t: np.ndarray,
    pos_true: np.ndarray,
    att_true: np.ndarray,
    mag_true: np.ndarray,
    pressure_true: np.ndarray,
    floor_true: np.ndarray,
    mag_meas: np.ndarray,
    pressure_meas: np.ndarray,
    config: Dict,
) -> None:
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth
    np.savetxt(output_dir / "time.txt", t, fmt="%.6f", header="time (s)")
    np.savetxt(
        output_dir / "ground_truth_position.txt",
        pos_true,
        fmt="%.6f",
        header="x (m), y (m), z (m)",
    )
    np.savetxt(
        output_dir / "ground_truth_attitude.txt",
        att_true,
        fmt="%.6f",
        header="roll (rad), pitch (rad), yaw (rad)",
    )
    np.savetxt(
        output_dir / "ground_truth_floor.txt",
        floor_true,
        fmt="%d",
        header="floor number (0=ground, 1=first, 2=second)",
    )

    # Save noisy measurements
    np.savetxt(
        output_dir / "magnetometer.txt",
        mag_meas,
        fmt="%.6f",
        header="mx (microTesla), my (microTesla), mz (microTesla)",
    )
    np.savetxt(
        output_dir / "barometer.txt",
        pressure_meas,
        fmt="%.6f",
        header="pressure (Pa)",
    )

    # Save clean signals
    np.savetxt(
        output_dir / "magnetometer_clean.txt",
        mag_true,
        fmt="%.6f",
        header="mx (microTesla), my (microTesla), mz (microTesla)",
    )
    np.savetxt(
        output_dir / "barometer_clean.txt",
        pressure_true,
        fmt="%.6f",
        header="pressure (Pa)",
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved dataset to: {output_dir}")
    print("    Files: 9 files (time, GT x3, measurements x2, clean x2, config)")
    print(f"    Samples: {len(t)}")


def generate_dataset(
    output_dir: str,
    preset: Optional[str] = None,
    duration: float = 180.0,
    floor_height: float = 3.5,
    dt: float = 0.1,
    mag_noise: float = 2.0,
    mag_disturbance: bool = False,
    pressure_noise: float = 10.0,
    weather_drift: float = 50.0,
    seed: int = 42,
) -> None:
    """
    Generate environmental sensors dataset.

    Args:
        output_dir: Output directory path.
        preset: Preset configuration name.
        duration: Total duration (s).
        floor_height: Height of each floor (m).
        dt: Time step (s).
        mag_noise: Magnetometer noise (microTesla).
        mag_disturbance: Add magnetic disturbances.
        pressure_noise: Pressure noise (Pa).
        weather_drift: Weather pressure drift (Pa).
        seed: Random seed.
    """
    # Apply preset if specified
    if preset == "baseline":
        # Clean sensors
        mag_noise = 1.5
        mag_disturbance = False
        pressure_noise = 8.0
        weather_drift = 30.0
        output_dir = output_dir or "data/sim/ch6_env_sensors_heading_altitude"
    elif preset == "noisy":
        # Higher noise
        mag_noise = 4.0
        mag_disturbance = False
        pressure_noise = 20.0
        weather_drift = 80.0
        output_dir = output_dir or "data/sim/ch6_env_sensors_noisy"
    elif preset == "disturbances":
        # Indoor magnetic disturbances
        mag_noise = 2.5
        mag_disturbance = True
        pressure_noise = 12.0
        weather_drift = 50.0
        output_dir = output_dir or "data/sim/ch6_env_sensors_disturbances"
    elif preset == "poor":
        # Poor quality + disturbances
        mag_noise = 6.0
        mag_disturbance = True
        pressure_noise = 30.0
        weather_drift = 120.0
        output_dir = output_dir or "data/sim/ch6_env_sensors_poor"

    # No preset and no --output: the module's own dataset.
    output_dir = output_dir or "data/sim/ch6_env_sensors_heading_altitude"

    print("\n" + "=" * 70)
    print(f"Generating Ch6 Environmental Sensors Dataset: {Path(output_dir).name}")
    print("=" * 70)

    # Generate trajectory
    print("\nStep 1: Generating building walk...")
    t, pos_true, att_true, mag_true, pressure_true, floor_true = generate_building_walk(
        duration=duration,
        floor_height=floor_height,
        dt=dt,
    )

    num_floors = len(np.unique(floor_true))
    max_altitude = pos_true[:, 2].max()

    print(f"  Duration: {duration:.1f} s")
    print(f"  Floors: {num_floors}")
    print(f"  Max altitude: {max_altitude:.1f} m")
    print(f"  Samples: {len(t)}")

    # Determine disturbance locations (in time)
    disturbance_locations = None
    if mag_disturbance:
        # Add disturbances during floor 1 and floor 2 walks
        disturbance_locations = [(90, 95), (110, 115), (160, 170)]

    # Add noise and disturbances
    print("\nStep 2: Adding sensor noise...")
    print(f"  Mag noise: {mag_noise:.2f} microTesla")
    print(f"  Mag disturbances: {'YES' if mag_disturbance else 'NO'}")
    if mag_disturbance:
        print(f"  Disturbance events: {len(disturbance_locations)}")
    print(f"  Pressure noise: {pressure_noise:.2f} Pa")
    print(f"  Weather drift: {weather_drift:.2f} Pa")

    mag_meas, pressure_meas = add_env_sensor_noise(
        mag_true,
        pressure_true,
        t,
        mag_noise=mag_noise,
        mag_disturbance=mag_disturbance,
        disturbance_locations=disturbance_locations,
        pressure_noise=pressure_noise,
        weather_drift=weather_drift,
        dt=dt,
        seed=seed,
    )

    # Run magnetometer heading estimation
    print("\nStep 3: Computing magnetometer heading...")
    start = time.time()
    heading_est = run_mag_heading_estimation(t, mag_meas, att_true)
    elapsed_mag = time.time() - start

    # Compute heading error.
    #
    # Wrap the *difference* to [-pi, pi]. The previous form was
    # `min(|d|, 2pi - |d|)`, which is only the shorter arc when |d| <= 2pi --
    # and it is not: the walk's first phase runs yaw up to 7.84 rad while
    # `mag_heading` returns (-pi, pi], so |d| reached 11 rad and `2pi - |d|`
    # went negative. 221 of 1800 samples were handed a negative "error", which
    # dragged the reported mean down to 2.66 deg from a true 3.51 deg.
    #
    # `wrap_angle_diff` is the repo's primitive for exactly this, and
    # `ch6_dead_reckoning/example_environment.py` already computes this same
    # heading error with it. The generator hand-rolled a fourth variant instead
    # and got it wrong -- duplicated policy only has to be forgotten once.
    heading_true = att_true[:, 2]
    heading_error = np.abs(
        np.array(
            [
                wrap_angle_diff(e, t_)
                for e, t_ in zip(heading_est, heading_true, strict=True)
            ]
        )
    )
    heading_error_deg = np.rad2deg(heading_error)
    mean_heading_error = np.mean(heading_error_deg)
    max_heading_error = np.max(heading_error_deg)

    print(f"  Time: {elapsed_mag:.3f} s")
    print(f"  Mean heading error: {mean_heading_error:.2f} deg")
    print(f"  Max heading error: {max_heading_error:.2f} deg")

    # Run barometric altitude estimation
    print("\nStep 4: Computing barometric altitude...")
    start = time.time()
    altitude_est = run_baro_altitude_estimation(pressure_meas)
    elapsed_baro = time.time() - start

    # Smooth altitude
    altitude_smooth = np.zeros_like(altitude_est)
    altitude_smooth[0] = altitude_est[0]
    for k in range(1, len(altitude_est)):
        altitude_smooth[k] = smooth_measurement_simple(
            altitude_smooth[k - 1], altitude_est[k], alpha=0.1
        )

    # Compute altitude error
    altitude_true = pos_true[:, 2]
    altitude_error = np.abs(altitude_smooth - altitude_true)
    mean_altitude_error = np.mean(altitude_error)
    max_altitude_error = np.max(altitude_error)

    # Detect floors from absolute smoothed altitude, not from a two-sample
    # delta. The previous form called detect_floor_change() on adjacent
    # altitude_smooth samples, but the alpha=0.1 exponential smoother spreads
    # a floor_height step over dozens of samples, so no single-sample delta
    # ever approached the 1.5 m threshold: floor_detected stayed 0 for the
    # entire walk, and floor_detection_accuracy silently scored the base rate
    # of floor 0 (50.0%, coincidentally matching floor 0's 50.0% share of
    # ground_truth_floor exactly). ch6_dead_reckoning/example_environment.py
    # fixes the same bug the same way: round absolute altitude to the nearest
    # floor -- altitude 0 is the known starting floor -- then clip to the
    # building's floor range. This generator already tracks that range as
    # num_floors, so the clip uses it instead of a hardcoded upper bound.
    floor_detected = np.rint(altitude_smooth / floor_height).astype(int)
    floor_detected = np.clip(floor_detected, 0, num_floors - 1)

    floor_detection_accuracy = np.mean(floor_detected == floor_true) * 100

    print(f"  Time: {elapsed_baro:.3f} s")
    print(f"  Mean altitude error: {mean_altitude_error:.2f} m")
    print(f"  Max altitude error: {max_altitude_error:.2f} m")
    print(f"  Floor detection accuracy: {floor_detection_accuracy:.1f}%")

    # Save dataset
    config = {
        "dataset": "ch6_env_sensors",
        "preset": preset,
        "trajectory": {
            "type": "building_walk",
            "duration_s": float(duration),
            "num_floors": int(num_floors),
            "floor_height_m": floor_height,
            "max_altitude_m": float(max_altitude),
        },
        "dt_s": dt,
        "sample_rate_hz": 1.0 / dt,
        "num_samples": len(t),
        "sensors": {
            "magnetometer": {
                "noise_std_uT": mag_noise,
                "disturbances_enabled": mag_disturbance,
                "num_disturbance_events": (
                    len(disturbance_locations) if disturbance_locations else 0
                ),
            },
            "barometer": {
                "noise_std_Pa": pressure_noise,
                "weather_drift_Pa": weather_drift,
            },
        },
        "performance": {
            "magnetometer_heading": {
                "mean_error_deg": float(mean_heading_error),
                "max_error_deg": float(max_heading_error),
            },
            "barometric_altitude": {
                "mean_error_m": float(mean_altitude_error),
                "max_error_m": float(max_altitude_error),
                "floor_detection_accuracy_percent": float(floor_detection_accuracy),
            },
        },
        "equations": ["6.51", "6.52", "6.53", "6.54", "6.55"],
        "seed": seed,
    }

    save_dataset(
        Path(output_dir),
        t,
        pos_true,
        att_true,
        mag_true,
        pressure_true,
        floor_true,
        mag_meas,
        pressure_meas,
        config,
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ch6 Environmental Sensors Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  baseline        Clean sensors (heading: 2-4 deg, altitude: 0.3-0.5m)
  noisy           Higher noise (heading: 5-8 deg, altitude: 0.8-1.2m)
  disturbances    Indoor magnetic anomalies (heading: 10-30 deg spikes)
  poor            Poor quality + disturbances (heading: 15-40 deg, altitude: 1.5-2.5m)

Examples:
  # Generate baseline dataset
  python scripts/generate_ch6_env_sensors_dataset.py --preset baseline

  # Generate dataset with custom parameters
  python scripts/generate_ch6_env_sensors_dataset.py \\
      --output data/sim/my_env_sensors \\
      --mag-noise 3.0 \\
      --add-disturbances \\
      --weather-drift 100

  # Generate all presets
  python scripts/generate_ch6_env_sensors_dataset.py --preset baseline
  python scripts/generate_ch6_env_sensors_dataset.py --preset noisy
  python scripts/generate_ch6_env_sensors_dataset.py --preset disturbances
  python scripts/generate_ch6_env_sensors_dataset.py --preset poor

Learning Focus:
  - Magnetometer provides absolute heading (no gyro drift!)
  - Barometer enables floor detection (crucial for indoor navigation)
  - Indoor magnetic disturbances corrupt magnetometer heading
  - Weather pressure changes affect barometric altitude
  - Tilt compensation is ESSENTIAL for accurate magnetometer heading

Book Reference: Chapter 6, Section 6.4 (Environmental Sensors)
        """,
    )

    # Preset or custom
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "noisy", "disturbances", "poor"],
        help="Use preset configuration (overrides other parameters)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/sim/ch6_env_sensors_heading_altitude)",
    )

    # Trajectory parameters
    traj_group = parser.add_argument_group("Trajectory Parameters")
    traj_group.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Total duration in seconds (default: 180.0)",
    )
    traj_group.add_argument(
        "--floor-height",
        type=float,
        default=3.5,
        help="Height of each floor in meters (default: 3.5)",
    )
    traj_group.add_argument(
        "--dt", type=float, default=0.1, help="Time step in seconds (default: 0.1)"
    )

    # Sensor noise parameters
    noise_group = parser.add_argument_group("Sensor Noise Parameters")
    noise_group.add_argument(
        "--mag-noise",
        type=float,
        default=2.0,
        help="Magnetometer noise in microTesla (default: 2.0)",
    )
    noise_group.add_argument(
        "--add-disturbances",
        action="store_true",
        help="Add indoor magnetic disturbances",
    )
    noise_group.add_argument(
        "--pressure-noise",
        type=float,
        default=10.0,
        help="Pressure noise in Pa (default: 10.0)",
    )
    noise_group.add_argument(
        "--weather-drift",
        type=float,
        default=50.0,
        help="Weather pressure drift in Pa (default: 50.0)",
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        preset=args.preset,
        duration=args.duration,
        floor_height=args.floor_height,
        dt=args.dt,
        mag_noise=args.mag_noise,
        mag_disturbance=args.add_disturbances,
        pressure_noise=args.pressure_noise,
        weather_drift=args.weather_drift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
