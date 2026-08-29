"""
Example: Comprehensive Comparison of Dead Reckoning Methods

Compares all Chapter 6 dead reckoning approaches on a common trajectory:
    1. IMU Strapdown (pure, no corrections)
    2. IMU + ZUPT (foot-mounted with stance detection)
    3. Wheel Odometry (vehicle)
    4. Pedestrian DR (step-and-heading with magnetometer)

Demonstrates the trade-offs between different approaches and the critical
importance of drift correction.

Author: Li-Ta Hsu
Date: December 2025
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

from core.eval import (
    plot_error_cdf,
    plot_error_magnitude_time,
    plot_trajectory_2d,
    resolve_figs_dir,
    save_figure,
    show_figures_if_requested,
)
from core.sensors import (
    FrameConvention,
    IMUNoiseParams,
    NavStateQPVP,
    detect_steps_peak_detector,
    detect_zupt_windowed,
    mag_heading,
    pdr_step_update,
    random_walk_to_rate_sample_std,
    step_frequency,
    step_length_book_eq6_49,
    strapdown_update,
    wheel_odom_update,
)
from core.sim import generate_imu_from_trajectory

# Seed for the sensor-noise draws. Fixed so the committed figures can be
# regenerated exactly; see add_sensor_noise. A single seed is a single noise
# realisation, not a characterisation of any method -- see run_seed_sweep and
# --seed-sweep for the distribution across seeds.
DEFAULT_SEED = 42

# (min, median, max) over a 12-seed sweep (seed 42, then 0-10), reusing this
# trajectory, IMUNoiseParams.consumer_grade() and add_sensor_noise's model --
# see run_seed_sweep(). Backs the "one draw, not a property of the method"
# caveats in the KEY INSIGHTS text below.
#
# A documented snapshot, not a live computation: recomputing it on every run
# would multiply main()'s cost ~12x, and the default run must not get
# slower. Reproduce with `--seed-sweep 12` and update these tuples if the
# trajectory, IMU params or noise model change.
SWEEP_STATS_12SEED = {
    "imu_only_final_m": (52.58, 141.70, 311.04),
    "imu_only_rmse_m": (53.78, 116.42, 247.93),
    "zupt_rmse_m": (8.82, 22.11, 45.05),
    "zupt_reduction_pct": (74.4, 81.6, 83.6),
    "wheel_odom_rmse_m": (0.39, 0.42, 0.48),
    "pdr_rmse_m": (0.43, 0.50, 0.61),
}

# Lever arm from the IMU/navigation centre to the wheel-speed sensor, in the
# attitude frame A (Eq. 6.11): 1 m forward and 0.2 m below.
LEVER_ARM_A = np.array([1.0, 0.0, -0.2])

# Rotation C_S^A from the speed frame S (x=right, y=forward, z=up, Section 6.2)
# to the attitude frame A. A is the IMU body frame, whose x-axis points forward
# because the attitude quaternion is a yaw about the ENU heading -- so the two
# frames differ by -90 deg about z and are NOT aligned. Leaving C_S^A at its
# identity default feeds a forward speed of [0, v, 0] straight through, which
# lands on the body y-axis and reports the entire track rotated 90 deg.
C_SPEED_TO_BODY = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


class MixedTrajectory(NamedTuple):
    """Shared trajectory and ideal sensor signals for the method comparison."""

    timestamps_s: np.ndarray
    true_positions_map_m: np.ndarray
    true_velocities_map_mps: np.ndarray
    specific_force_body_mps2: np.ndarray
    angular_rates_body_rad_s: np.ndarray
    true_headings_rad: np.ndarray
    true_magnetic_field_body: np.ndarray
    stance_mask_stationary: np.ndarray
    true_wheel_speed_mps: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return self.timestamps_s

    @property
    def pos_true(self) -> np.ndarray:
        return self.true_positions_map_m

    @property
    def vel_true(self) -> np.ndarray:
        return self.true_velocities_map_mps

    @property
    def accel_body(self) -> np.ndarray:
        return self.specific_force_body_mps2

    @property
    def gyro_body(self) -> np.ndarray:
        return self.angular_rates_body_rad_s

    @property
    def heading_true(self) -> np.ndarray:
        return self.true_headings_rad

    @property
    def mag_body(self) -> np.ndarray:
        return self.true_magnetic_field_body

    @property
    def stance_mask(self) -> np.ndarray:
        return self.stance_mask_stationary

    @property
    def wheel_speed_true(self) -> np.ndarray:
        return self.true_wheel_speed_mps


def _speed_envelope(tau: np.ndarray, duration: float, ramp: float) -> np.ndarray:
    """Raised-cosine trapezoid rising 0->1, holding, then falling 1->0.

    A walker does not reach 1.2 m/s in one 10 ms sample. Stepping the speed
    instead of ramping it puts a 120 m/s^2 spike into the synthesised
    accelerometer at every start and stop -- two orders of magnitude above real
    gait -- and those spikes were the only thing the old PDR step detector ever
    fired on. The raised cosine keeps the acceleration continuous.

    Args:
        tau: Time since the start of the phase, shape (M,). Units: s.
        duration: Total phase duration. Units: s.
        ramp: Rise and fall time. Clipped to half the duration. Units: s.

    Returns:
        Envelope in [0, 1], shape (M,). Integrates to ``duration - ramp``.
    """
    ramp = min(ramp, duration / 2.0)
    env = np.ones_like(tau)
    if ramp <= 0.0:
        return env

    rising = tau < ramp
    env[rising] = 0.5 * (1.0 - np.cos(np.pi * tau[rising] / ramp))
    falling = tau > duration - ramp
    env[falling] = 0.5 * (1.0 - np.cos(np.pi * (duration - tau[falling]) / ramp))
    return np.clip(env, 0.0, 1.0)


def _turn_profile(tau: np.ndarray, duration: float) -> np.ndarray:
    """Smoothstep 0->1 used to rotate in place, with zero rate at both ends.

    This is the integral of the raised cosine used for the speed ramps, so the
    yaw rate starts and ends at zero. The alternative -- switching heading
    between samples, as the trajectory used to -- synthesises a 200 rad/s gyro
    spike that no first-order quaternion integrator can absorb: the wheel
    odometry lost ~14 deg of heading at every corner.

    Args:
        tau: Time since the start of the turn, shape (M,). Units: s.
        duration: Turn duration. Units: s.

    Returns:
        Fraction of the turn completed, in [0, 1], shape (M,).
    """
    u = np.clip(tau / duration, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def generate_mixed_trajectory(
    duration: float = 120.0,
    dt: float = 0.01,
    frame: FrameConvention | None = None,
    v_walk: float = 1.2,
    step_freq: float = 1.75,
    bob_accel_mps2: float = 2.5,
    ramp_time: float = 0.6,
    stop_duration: float = 3.0,
    turn_time: float = 1.5,
    lever_arm_a: np.ndarray | None = None,
) -> MixedTrajectory:
    """Generate one trajectory that all four DR methods can be run against.

    A 30 m x 20 m rectangular walk with a pause and an in-place turn at each
    corner. The whole point of the comparison is that every method sees the
    *same* motion, so the trajectory has to carry every signal the methods
    depend on:

    - **Gait dynamics.** The vertical bob at ``step_freq`` is what the PDR peak
      detector (Eqs. 6.46-6.47) counts and what lets the ZUPT test statistic
      (Eq. 6.44) tell walking from standing. Without it this trajectory is
      piecewise constant velocity, the specific force while walking is
      indistinguishable from the specific force while standing, and both
      detectors are blind: measured on the old trajectory, T_k had a median of
      20.83 in motion against 20.85 at rest.
    - **Bounded accelerations.** Speed ramps in and out of each segment rather
      than stepping (see ``_speed_envelope``).
    - **Bounded turn rates.** Heading rotates over ``turn_time`` in the middle
      of each pause rather than jumping between samples (see
      ``_turn_profile``).

    Gait self-consistency: ``step_freq`` is chosen so the book's step-length
    model (Eq. 6.49) reproduces the simulated walking speed. At h = 1.75 m,
    Eq. (6.49) gives SL = 0.691 m at 1.75 Hz, hence 1.209 m/s against the
    simulated 1.2 m/s -- a 0.75% step-length bias, which leaves PDR error
    dominated by heading, as Section 6.3 argues it should be. Pairing an
    arbitrary speed with an arbitrary step rate instead builds in a constant
    scale error that teaches nothing about PDR.

    Args:
        duration: Total duration. Units: s.
        dt: Sample interval. Units: s.
        frame: Frame convention. Default: None (creates ENU).
        v_walk: Cruise walking speed. Units: m/s.
        step_freq: Step frequency; drives the vertical bob. Units: Hz.
        bob_accel_mps2: Amplitude of the vertical gait acceleration.
            Units: m/s^2. 2.5 m/s^2 matches ``example_pdr.generate_corridor_walk``.
        ramp_time: Speed rise/fall time at each segment end. Units: s.
        stop_duration: Pause at each corner. Units: s.
        turn_time: In-place rotation time, centred in the pause. Units: s.
        lever_arm_a: Lever arm to the wheel-speed sensor in frame A, shape (3,).
            Default: None (uses ``LEVER_ARM_A``).

    Returns:
        MixedTrajectory with semantic fields and tuple-compatible order:
            timestamps_s: Time vector, shape (N,), units s.
            true_positions_map_m: Ground-truth position in map frame, shape
                (N, 3), units m.
            true_velocities_map_mps: Ground-truth velocity in map frame, shape
                (N, 3), units m/s.
            specific_force_body_mps2: Ideal specific force in body frame,
                shape (N, 3), units m/s^2.
            angular_rates_body_rad_s: Ideal angular rate in body frame, shape
                (N, 3), units rad/s.
            true_headings_rad: Ground-truth heading in map frame, shape (N,),
                units rad (ENU: 0 = East).
            true_magnetic_field_body: Ideal magnetometer reading in body frame,
                shape (N, 3).
            stance_mask_stationary: True where the walker is not translating,
                shape (N,).
            true_wheel_speed_mps: Ideal speed-frame velocity at the wheel-speed
                sensor, shape (N, 3), units m/s.

    References:
        Chapter 6, Sections 6.1-6.3; Eqs. (6.11), (6.44), (6.46)-(6.50).
    """
    if frame is None:
        frame = FrameConvention.create_enu()
    if lever_arm_a is None:
        lever_arm_a = LEVER_ARM_A

    t = np.arange(0.0, duration, dt)
    n_samples = len(t)

    # Rectangular path: 30 m x 20 m, closing back on the start point.
    waypoints = np.array(
        [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0], [0.0, 0.0]]
    )
    deltas = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    # ENU: heading is measured from East, so atan2(North, East).
    segment_headings = np.arctan2(deltas[:, 1], deltas[:, 0])

    speed = np.zeros(n_samples)
    # Heading is accumulated UNWRAPPED. Wrapping it would flip the sign of the
    # quaternion's scalar part halfway round the rectangle, and compute_gyro_body
    # differences the raw components -- a sign flip there reads as an enormous
    # angular rate, which is a subtler version of the same defect the smooth
    # turns fix. cos/sin are indifferent to the unwrapping.
    heading_true = np.zeros(n_samples)
    current_heading = float(segment_headings[0])

    k = 0
    for seg in range(len(segment_lengths)):
        # --- Walking phase -----------------------------------------------
        # A raised-cosine trapezoid of duration T covers v * (T - ramp), so
        # solving for T gives the segment length exactly.
        walk_samples = int(round((segment_lengths[seg] / v_walk + ramp_time) / dt))
        end = min(n_samples, k + walk_samples)
        if end > k:
            tau = np.arange(end - k) * dt
            speed[k:end] = v_walk * _speed_envelope(tau, walk_samples * dt, ramp_time)
            heading_true[k:end] = current_heading
        k = end
        if k >= n_samples:
            break

        # --- Pause, with an in-place turn to the next segment -------------
        stop_samples = int(round(stop_duration / dt))
        end = min(n_samples, k + stop_samples)
        heading_true[k:end] = current_heading
        if seg + 1 < len(segment_headings):
            delta_psi = _wrap_to_pi(segment_headings[seg + 1] - segment_headings[seg])
            turn_start = k + int(round((stop_duration - turn_time) / 2.0 / dt))
            turn_end = min(n_samples, turn_start + int(round(turn_time / dt)))
            if turn_end > turn_start:
                tau = np.arange(turn_end - turn_start) * dt
                heading_true[turn_start:turn_end] = (
                    current_heading + delta_psi * _turn_profile(tau, turn_time)
                )
            current_heading += delta_psi
            heading_true[turn_end:end] = current_heading
        k = end
        if k >= n_samples:
            break

    # Whatever time is left over is spent standing at the final waypoint --
    # useful in its own right, since it is where unaided strapdown runs away
    # while the corrected solutions hold station.
    if k < n_samples:
        heading_true[k:] = current_heading

    # Stance = not translating. Note this is looser than what the Eq. (6.44)
    # detector will report: it also sees the yaw rate, so it correctly refuses
    # to call the in-place turns stationary.
    stance_mask = speed <= 1e-9

    # Vertical gait. The envelope is the normalised speed, so the bob fades in
    # and out with the walk instead of switching on at a stance boundary.
    bob_vel_amplitude = bob_accel_mps2 / (2.0 * np.pi * step_freq)
    vel_up = (speed / v_walk) * bob_vel_amplitude * np.cos(2.0 * np.pi * step_freq * t)

    vel_true = np.column_stack(
        [speed * np.cos(heading_true), speed * np.sin(heading_true), vel_up]
    )
    # Integrate with the same backward-Euler rule the estimators use
    # (core.sensors.strapdown.pos_update), so truth and estimate share a
    # discretisation and the residual is algorithm error, not quadrature error.
    pos_true = np.cumsum(vel_true * dt, axis=0)

    quat_true = np.column_stack(
        [
            np.cos(heading_true / 2.0),
            np.zeros(n_samples),
            np.zeros(n_samples),
            np.sin(heading_true / 2.0),
        ]
    )

    accel_body, gyro_body = generate_imu_from_trajectory(
        pos_map=pos_true,
        vel_map=vel_true,
        quat_b_to_m=quat_true,
        dt=dt,
        frame=frame,
        g=9.81,
    )

    # Speed-frame velocity at the wheel sensor, Eq. (6.11) read forwards:
    #   v^S = C_A^S (v_nav^A + [omega^A x] l^A)
    # The estimator subtracts the lever-arm term, so the truth has to contain
    # it; generating a bare forward speed instead leaves the compensation with
    # nothing to cancel and injects a spurious sideways sweep at every turn.
    # Yaw rate is differenced from the heading itself, which is what
    # ``compute_gyro_body`` recovers from the quaternion series.
    yaw_rate = np.zeros(n_samples)
    if n_samples > 1:
        yaw_rate[1:] = np.diff(heading_true) / dt
    omega_cross_lever = np.column_stack(
        [
            -yaw_rate * lever_arm_a[1],
            yaw_rate * lever_arm_a[0],
            np.zeros(n_samples),
        ]
    )
    vel_wheel_body = np.column_stack([speed, np.zeros(n_samples), np.zeros(n_samples)])
    vel_wheel_body += omega_cross_lever
    wheel_speed_true = vel_wheel_body @ C_SPEED_TO_BODY

    # Magnetometer: a fixed map-frame reference direction resolved into the
    # body frame, so that mag_heading (Eqs. 6.51-6.53) recovers the heading the
    # walker actually holds. The reference is the map +x axis, which in ENU is
    # East -- the same direction heading is measured from.
    mag_body = np.zeros((n_samples, 3))
    mag_reference_map = np.array([1.0, 0.0, 0.0])
    for sample in range(n_samples):
        yaw = heading_true[sample]
        c_yaw = np.array(
            [
                [np.cos(yaw), np.sin(yaw), 0.0],
                [-np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mag_body[sample] = c_yaw.T @ mag_reference_map

    return MixedTrajectory(
        timestamps_s=t,
        true_positions_map_m=pos_true,
        true_velocities_map_mps=vel_true,
        specific_force_body_mps2=accel_body,
        angular_rates_body_rad_s=gyro_body,
        true_headings_rad=heading_true,
        true_magnetic_field_body=mag_body,
        stance_mask_stationary=stance_mask,
        true_wheel_speed_mps=wheel_speed_true,
    )


def add_sensor_noise(
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    mag_body: np.ndarray,
    wheel_true: np.ndarray,
    dt: float,
    imu_params: IMUNoiseParams,
    seed: int = DEFAULT_SEED,
    wheel_scale_error: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add noise to all sensors with explicit units.

    Args:
        accel_body: Ideal specific force, shape (N, 3), units m/s^2.
        gyro_body: Ideal angular rate, shape (N, 3), units rad/s.
        mag_body: Ideal magnetometer reading, shape (N, 3), normalised.
        wheel_true: Ideal speed-frame velocity, shape (N, 3), units m/s.
        dt: Sample interval. Units: s.
        imu_params: IMU noise specification.
        seed: Seed for this example's random draws. This is the only source of
            randomness in the script, so fixing it makes the committed figures
            reproducible -- and since core.eval.save_figure already writes
            byte-reproducible output, a figure diff then means the picture
            genuinely changed rather than that the biases were redrawn.
            Unseeded, each run picked a new bias vector and the IMU-only track
            drifted off in a different direction every time.
        wheel_scale_error: Encoder scale-factor error, i.e. a wrong assumed
            wheel radius. Dimensionless; 0.02 (2%) is typical. This is the
            error that decides odometry accuracy, because it is systematic and
            therefore integrates into a drift proportional to distance -- the
            bound Section 6.2 claims for the method. Zero-mean encoder noise on
            its own averages out over a 100 m walk and leaves odometry looking
            exact, which is a simulation artefact rather than a result.

    Returns:
        Tuple ``(accel_meas, gyro_meas, mag_meas, wheel_meas)`` with the same
        shapes and units as the corresponding inputs.
    """
    n_samples = len(accel_body)

    # Own the generator rather than seeding the global RNG: nothing else in
    # this script draws random numbers, so there is no hidden consumer that
    # needs the global state set.
    rng = np.random.default_rng(seed)

    # IMU noise and biases
    gyro_bias = rng.standard_normal(3) * imu_params.gyro_bias_rad_s
    gyro_noise_std = random_walk_to_rate_sample_std(imu_params.gyro_arw_rad_sqrt_s, dt)
    gyro_noise = rng.standard_normal((n_samples, 3)) * gyro_noise_std

    accel_bias = rng.standard_normal(3) * imu_params.accel_bias_mps2
    accel_noise_std = random_walk_to_rate_sample_std(
        imu_params.accel_vrw_mps_sqrt_s, dt
    )
    accel_noise = rng.standard_normal((n_samples, 3)) * accel_noise_std

    gyro_meas = gyro_body + gyro_bias + gyro_noise
    accel_meas = accel_body + accel_bias + accel_noise

    # Magnetometer noise
    mag_noise = rng.standard_normal((n_samples, 3)) * 0.05
    mag_meas = mag_body + mag_noise

    # Wheel encoder scale-factor error and noise
    wheel_noise = rng.standard_normal((n_samples, 3)) * 0.05
    wheel_meas = wheel_true * (1.0 + wheel_scale_error) + wheel_noise

    return accel_meas, gyro_meas, mag_meas, wheel_meas


def run_imu_only(
    t: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    initial: NavStateQPVP,
    frame: FrameConvention,
) -> np.ndarray:
    """Method 1: pure IMU strapdown, Eqs. (6.2)-(6.10).

    Args:
        t: Time vector, shape (N,). Units: s.
        accel: Measured specific force, shape (N, 3). Units: m/s^2.
        gyro: Measured angular rate, shape (N, 3). Units: rad/s.
        initial: Initial navigation state.
        frame: Frame convention.

    Returns:
        Estimated position in the map frame, shape (N, 3). Units: m.
    """
    n_samples, dt = len(t), t[1] - t[0]
    q, v, p = initial.q.copy(), initial.v.copy(), initial.p.copy()
    pos = np.zeros((n_samples, 3))
    pos[0] = p

    for k in range(1, n_samples):
        q, v, p = strapdown_update(q, v, p, gyro[k - 1], accel[k - 1], dt, frame=frame)
        pos[k] = p
    return pos


def run_imu_zupt(
    t: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    initial: NavStateQPVP,
    frame: FrameConvention,
    imu_params: IMUNoiseParams,
    window_size: int = 10,
    gamma: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Method 2: IMU + ZUPT, windowed detector (Eq. 6.44) and reset (Eq. 6.45).

    The threshold is the whole algorithm here. On this trajectory the test
    statistic sits near 21 while standing and near 900 while walking, so gamma
    has an order of magnitude of headroom on either side; the previous 1e6
    accepted every sample, zeroed the velocity at every step and pinned the
    solution to its start point.

    Args:
        t: Time vector, shape (N,). Units: s.
        accel: Measured specific force, shape (N, 3). Units: m/s^2.
        gyro: Measured angular rate, shape (N, 3). Units: rad/s.
        initial: Initial navigation state.
        frame: Frame convention.
        imu_params: IMU noise specification, used to scale the detector.
        window_size: Detector window length. Units: samples.
        gamma: Detection threshold, Eq. (6.44). Dimensionless.

    Returns:
        Tuple ``(pos, zupt_detected)``:
            pos: Estimated position, shape (N, 3). Units: m.
            zupt_detected: True where the detector fired, shape (N,).
    """
    n_samples, dt = len(t), t[1] - t[0]
    q, v, p = initial.q.copy(), initial.v.copy(), initial.p.copy()
    pos = np.zeros((n_samples, 3))
    pos[0] = p
    zupt_detected = np.zeros(n_samples, dtype=bool)

    # Compute noise std devs for ZUPT detector
    sigma_a = random_walk_to_rate_sample_std(imu_params.accel_vrw_mps_sqrt_s, dt)
    sigma_g = random_walk_to_rate_sample_std(imu_params.gyro_arw_rad_sqrt_s, dt)

    for k in range(1, n_samples):
        q, v, p = strapdown_update(q, v, p, gyro[k - 1], accel[k - 1], dt, frame=frame)

        # Windowed ZUPT detection (OFFLINE/POST-PROCESSING)
        # Uses centered window (includes future samples) - appropriate for batch
        # processing. For real-time: use a trailing window
        # (k-window_size+1:k+1) or accept the latency.
        window_start = max(0, k - window_size // 2)
        window_end = min(n_samples, k + window_size // 2 + 1)
        accel_window = accel[window_start:window_end]
        gyro_window = gyro[window_start:window_end]

        if len(accel_window) >= window_size // 2:
            if detect_zupt_windowed(accel_window, gyro_window, sigma_a, sigma_g, gamma):
                v = np.zeros(3)
                zupt_detected[k] = True

        pos[k] = p
    return pos, zupt_detected


def run_wheel_odom(
    t: np.ndarray,
    wheel: np.ndarray,
    gyro: np.ndarray,
    initial: NavStateQPVP,
    lever_arm: np.ndarray,
) -> np.ndarray:
    """Method 3: wheel odometry, Eqs. (6.11)-(6.15).

    Args:
        t: Time vector, shape (N,). Units: s.
        wheel: Measured speed-frame velocity, shape (N, 3). Units: m/s.
        gyro: Measured angular rate, shape (N, 3). Units: rad/s.
        initial: Initial navigation state.
        lever_arm: Lever arm to the speed sensor in frame A, shape (3,), m.

    Returns:
        Estimated position in the map frame, shape (N, 3). Units: m.
    """
    n_samples, dt = len(t), t[1] - t[0]
    p = initial.p.copy()
    q = initial.q.copy()
    pos = np.zeros((n_samples, 3))
    pos[0] = p

    for k in range(1, n_samples):
        p = wheel_odom_update(
            p,
            q,
            wheel[k - 1],
            gyro[k - 1],
            lever_arm,
            dt,
            C_S_A=C_SPEED_TO_BODY,
        )
        # Update quaternion from gyro
        q_new = q.copy()
        dq = (
            0.5
            * dt
            * np.array(
                [
                    -q[1] * gyro[k - 1, 0]
                    - q[2] * gyro[k - 1, 1]
                    - q[3] * gyro[k - 1, 2],
                    q[0] * gyro[k - 1, 0]
                    + q[2] * gyro[k - 1, 2]
                    - q[3] * gyro[k - 1, 1],
                    q[0] * gyro[k - 1, 1]
                    - q[1] * gyro[k - 1, 2]
                    + q[3] * gyro[k - 1, 0],
                    q[0] * gyro[k - 1, 2]
                    + q[1] * gyro[k - 1, 1]
                    - q[2] * gyro[k - 1, 0],
                ]
            )
        )
        q = q_new + dq
        q = q / np.linalg.norm(q)
        pos[k] = p
    return pos


def run_pdr(
    t: np.ndarray,
    accel: np.ndarray,
    mag: np.ndarray,
    height: float = 1.75,
) -> tuple[np.ndarray, int]:
    """Method 4: pedestrian DR with magnetometer heading, Eqs. (6.46)-(6.53).

    Steps come from the book's peak detector rather than a bare threshold
    crossing on the raw magnitude. The threshold version tested
    ``|a| >= 11.0`` against a signal whose mean is 9.81 and whose only
    excursions were the numerical spikes at the old trajectory's instantaneous
    starts and stops: it found three "steps" in a 100 m walk. Removing gravity
    first (Eq. 6.47) and requiring a prominent, refractory peak is both the
    method Section 6.3.2 describes and what ``example_pdr`` already uses.

    Args:
        t: Time vector, shape (N,). Units: s.
        accel: Measured specific force, shape (N, 3). Units: m/s^2.
        mag: Measured magnetic field in the body frame, shape (N, 3).
        height: Pedestrian height, used by Eq. (6.49). Units: m.

    Returns:
        Tuple ``(pos, step_count)``:
            pos: Estimated position, shape (N, 3) with z = 0. Units: m.
            step_count: Number of detected steps.
    """
    n_samples, dt = len(t), t[1] - t[0]

    step_indices, _ = detect_steps_peak_detector(
        accel,
        dt=dt,
        g=9.81,
        min_peak_height=1.0,  # m/s^2 above gravity
        min_peak_distance=0.3,  # s between steps (max ~3.3 steps/s)
        lowpass_cutoff=5.0,  # Hz
    )
    is_step = np.zeros(n_samples, dtype=bool)
    is_step[step_indices] = True

    pos = np.zeros((n_samples, 2))
    last_step_time = t[0]

    for k in range(1, n_samples):
        # Magnetometer heading (Eqs. 6.51-6.53); the walk is level, so roll and
        # pitch are zero and tilt compensation is a no-op.
        heading = mag_heading(mag[k], roll=0.0, pitch=0.0, declination=0.0)

        if is_step[k]:
            delta_t = t[k] - last_step_time
            last_step_time = t[k]
            f_step = step_frequency(delta_t) if delta_t > 0 else 2.0
            # Book Eq. (6.49) for step length.
            step_len = step_length_book_eq6_49(height, f_step)
            pos[k] = pdr_step_update(pos[k - 1], step_len, heading)
        else:
            pos[k] = pos[k - 1]

    pos_3d = np.column_stack([pos, np.zeros(n_samples)])
    return pos_3d, int(len(step_indices))


def plot_comparison(
    t: np.ndarray,
    pos_true: np.ndarray,
    results: dict[str, np.ndarray],
    figs_dir: Path,
) -> dict[str, dict[str, float]]:
    """Generate comprehensive comparison plots.

    All three figures come from core.eval primitives rather than hand-rolled
    axes: the trajectory overlay, the error-magnitude history and the error CDF
    are the same plots every chapter needs, and keeping them shared means a
    styling or correctness fix lands everywhere at once.

    Args:
        t: Time vector, shape (N,). Units: s.
        pos_true: Ground-truth position, shape (N, 3). Units: m.
        results: {method name: estimated position, shape (N, 3)}.
        figs_dir: Output directory for the figures.

    Returns:
        {method name: {'rmse', 'final', 'median', 'p90', 'path'}}, all in m.
        Errors are horizontal: the walk is planar, every method reports its own
        vertical channel (PDR has none at all), and the figures plot the
        horizontal error, so the table has to agree with them.
    """
    # Errors are shared by figures 2 and 3, so compute them once.
    errors = {name: pos[:, :2] - pos_true[:, :2] for name, pos in results.items()}

    # Figure 1: all trajectories, in the local-level frame.
    #
    # Two panels, for the same reason the Section 6.5 ZUPT animation uses them:
    # unaided IMU strapdown drifts several hundred metres against a 30 x 20 m
    # ground truth, and a trajectory plot needs equal axes, so on one pair of
    # limits everything that stayed near the truth collapses into a single blob
    # at the origin. The left panel keeps the full extent, because the size of
    # the unbounded drift IS the lesson of Section 6.1; the right panel
    # resolves what happens at the scale of the walk itself.
    #
    # The zoom panel is also what made the frozen ZUPT and PDR tracks visible
    # in the first place -- at the full scale they were inside the blob. Both
    # now track; the 'path' metric below is the non-visual version of the same
    # check, so a regression does not depend on someone opening the figure.
    fig1 = plot_trajectory_2d(
        pos_true[:, :2],
        {name: pos[:, :2] for name, pos in results.items()},
        title="IMU alone drifts to 54 m RMSE; ZUPT cuts it to 8.8, odometry and PDR to under a metre",
        axis_labels=("East [m]", "North [m]"),
        zoom_to_truth=True,
    )
    paths = save_figure(fig1, figs_dir, "comparison_trajectories")
    print(f"  [OK] Saved: {paths[0]}")

    # Figure 2: error magnitude over time. Log scale because unaided IMU drift
    # and a ZUPT-corrected solution differ by orders of magnitude, and a linear
    # axis flattens the corrected one onto the baseline.
    fig2 = plot_error_magnitude_time(
        errors,
        t=t,
        title="Chapter 6 Comparison: Position Error vs Time",
        log_scale=True,
    )
    paths = save_figure(fig2, figs_dir, "comparison_error_time")
    print(f"  [OK] Saved: {paths[0]}")

    # Figure 3: error CDF.
    fig3 = plot_error_cdf(errors, title="Chapter 6 Comparison: Error CDF")
    paths = save_figure(fig3, figs_dir, "comparison_error_cdf")
    print(f"  [OK] Saved: {paths[0]}")

    plt.close("all")

    return _compute_metrics(pos_true, results)


def _compute_metrics(
    pos_true: np.ndarray, results: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Horizontal error statistics for each method's estimated position.

    Split out of plot_comparison so run_seed_sweep can compute the same
    statistics for many noise draws without generating (and discarding)
    figures for each one.

    Args:
        pos_true: Ground-truth position, shape (N, 3). Units: m.
        results: {method name: estimated position, shape (N, 3)}.

    Returns:
        {method name: {'rmse', 'final', 'median', 'p90', 'path'}}, all in m.
        Errors are horizontal -- see plot_comparison, the other caller, for
        why: the walk is planar, every method reports its own vertical
        channel (PDR has none at all), and the figures plot the horizontal
        error, so this has to agree with them.
    """
    metrics = {}
    for name, pos in results.items():
        error = np.linalg.norm(pos[:, :2] - pos_true[:, :2], axis=1)
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "final": float(error[-1]),
            "median": float(np.median(error)),
            "p90": float(np.percentile(error, 90)),
            # Horizontal path length. A method that has silently stopped
            # tracking still scores well on error alone, because this ground
            # truth returns to its own start point -- so report the distance
            # actually traced next to it.
            "path": float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))),
        }
    return metrics


def run_seed_sweep(
    t: np.ndarray,
    pos_true: np.ndarray,
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    mag_body: np.ndarray,
    wheel_true: np.ndarray,
    dt: float,
    imu_params: IMUNoiseParams,
    initial: NavStateQPVP,
    frame: FrameConvention,
    height: float,
    seeds: tuple[int, ...],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Re-run every method over several noise seeds and summarise.

    A single seed is one noise realisation; SWEEP_STATS_12SEED and the
    KEY INSIGHTS caveats it backs exist because seed 42 turns out to be an
    unusually kind draw for the unaided IMU. This is how those constants
    were produced, and --seed-sweep lets a reader reproduce or refresh them.

    Reuses add_sensor_noise -- the sole noise-model entry point -- and the
    four run_* estimators; only the seed passed to add_sensor_noise changes
    between iterations, so this characterises seed sensitivity rather than
    introducing a second noise model that could drift from the first.

    Args:
        t, pos_true, accel_body, gyro_body, mag_body, wheel_true, dt: the
            trajectory and ideal sensor streams from generate_mixed_trajectory.
            None of these depend on the noise seed, so the caller generates
            them once and this function reuses them for every seed.
        imu_params: IMU noise specification.
        initial: Initial navigation state.
        frame: Frame convention.
        height: Pedestrian height for PDR, Eq. (6.49).
        seeds: Seeds to draw sensor noise with; one full run per seed.

    Returns:
        {method name: {'rmse': (min, median, max), 'final': (min, median, max)}}
        in metres, plus a synthetic "ZUPT Reduction" entry whose 'pct' is the
        (min, median, max) of the *per-seed* RMSE reduction -- not derived
        from the min/median/max RMSEs above, which would pair each method's
        best seed with the other's best seed even when those are different
        seeds.
    """
    per_seed: dict[str, dict[str, list]] = {
        name: {"rmse": [], "final": []}
        for name in ("IMU Only", "IMU + ZUPT", "Wheel Odom", "PDR (Mag)")
    }
    zupt_reduction_pct = []

    for seed in seeds:
        accel_meas, gyro_meas, mag_meas, wheel_meas = add_sensor_noise(
            accel_body, gyro_body, mag_body, wheel_true, dt, imu_params, seed=seed
        )
        results = {
            "IMU Only": run_imu_only(t, accel_meas, gyro_meas, initial, frame),
            "IMU + ZUPT": run_imu_zupt(
                t, accel_meas, gyro_meas, initial, frame, imu_params
            )[0],
            "Wheel Odom": run_wheel_odom(
                t, wheel_meas, gyro_meas, initial, LEVER_ARM_A
            ),
            "PDR (Mag)": run_pdr(t, accel_meas, mag_meas, height)[0],
        }
        seed_metrics = _compute_metrics(pos_true, results)
        for name in per_seed:
            per_seed[name]["rmse"].append(seed_metrics[name]["rmse"])
            per_seed[name]["final"].append(seed_metrics[name]["final"])
        zupt_reduction_pct.append(
            100
            * (
                1
                - seed_metrics["IMU + ZUPT"]["rmse"] / seed_metrics["IMU Only"]["rmse"]
            )
        )

    summary = {
        name: {
            metric: (min(vals), float(np.median(vals)), max(vals))
            for metric, vals in by_metric.items()
        }
        for name, by_metric in per_seed.items()
    }
    summary["ZUPT Reduction"] = {
        "pct": (
            min(zupt_reduction_pct),
            float(np.median(zupt_reduction_pct)),
            max(zupt_reduction_pct),
        )
    }
    return summary


def _print_seed_sweep(
    summary: dict[str, dict[str, tuple[float, float, float]]],
    seeds: tuple[int, ...],
    elapsed_s: float,
) -> None:
    """Print the run_seed_sweep summary as a min/median/max table."""
    print("\n" + "=" * 75)
    print(
        f"SEED SWEEP: N={len(seeds)} seeds ({', '.join(str(s) for s in seeds)}); "
        f"{elapsed_s:.0f} s"
    )
    print("=" * 75)
    for metric_key, label in (("rmse", "RMSE [m]"), ("final", "Final [m]")):
        print(f"\n{label:<20} {'min':>10} {'median':>10} {'max':>10}")
        print("-" * 52)
        for name in ("IMU Only", "IMU + ZUPT", "Wheel Odom", "PDR (Mag)"):
            lo, med, hi = summary[name][metric_key]
            print(f"{name:<20} {lo:>10.2f} {med:>10.2f} {hi:>10.2f}")

    red_lo, red_med, red_hi = summary["ZUPT Reduction"]["pct"]
    print(
        f"\nZUPT RMSE reduction, per seed: min {red_lo:.0f}%, median "
        f"{red_med:.0f}%, max {red_hi:.0f}%."
    )


def main() -> None:
    """Main execution."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed-sweep",
        type=int,
        default=0,
        metavar="N",
        help=(
            f"Re-run every method over N noise seeds (seed {DEFAULT_SEED} "
            "plus 0..N-2) and print the median and min-max range per "
            "method, instead of trusting the single seed used for the "
            "figures and the RESULTS table above. Off by default: it does "
            "not change the figures, and each extra seed costs roughly as "
            "long as the base run (~10 s here), so N=12 takes ~2 minutes."
        ),
    )
    args = parser.parse_args()
    if args.seed_sweep < 0:
        parser.error("--seed-sweep must be >= 0")

    print("\n" + "=" * 75)
    print("Chapter 6: COMPREHENSIVE COMPARISON of Dead Reckoning Methods")
    print("=" * 75)
    print("\nCompares all major DR approaches on a common trajectory.")
    print("Demonstrates trade-offs and the critical need for drift correction.\n")

    duration = 120.0
    dt = 0.01
    height = 1.75
    frame = FrameConvention.create_enu()  # Use ENU frame
    imu_params = IMUNoiseParams.consumer_grade()  # Consumer-grade IMU

    print("Configuration:")
    print(f"  Duration:        {duration} s")
    print("  Trajectory:      30m x 20m rectangular path with stops")
    print(f"  IMU Rate:        {1 / dt:.0f} Hz")
    print(f"  Frame:           {frame.map_frame}")
    print(f"  Noise seed:      {DEFAULT_SEED} (figures are reproducible)\n")

    # Print IMU specifications
    print(imu_params.format_specs())
    print()

    print("Generating trajectory with correct IMU forward model...")
    (
        t,
        pos_true,
        vel_true,
        accel_body,
        gyro_body,
        heading_true,
        mag_body,
        stance,
        wheel_true,
    ) = generate_mixed_trajectory(duration, dt, frame, lever_arm_a=LEVER_ARM_A)

    total_dist = np.sum(np.linalg.norm(np.diff(pos_true[:, :2], axis=0), axis=1))
    print(f"  Total distance:  {total_dist:.1f} m (horizontal)")
    print(f"  Standing:        {100 * stance.mean():.1f}% of samples")

    print("\nAdding sensor noise...")
    accel_meas, gyro_meas, mag_meas, wheel_meas = add_sensor_noise(
        accel_body, gyro_body, mag_body, wheel_true, dt, imu_params, seed=DEFAULT_SEED
    )

    initial = NavStateQPVP(q=np.array([1, 0, 0, 0]), v=vel_true[0], p=pos_true[0])

    # Run all methods
    print("\nRunning all methods...")
    methods = {}

    print("  1. IMU only (pure strapdown)...")
    start = time.time()
    methods["IMU Only"] = run_imu_only(t, accel_meas, gyro_meas, initial, frame)
    print(f"     Time: {time.time() - start:.3f} s")

    print("  2. IMU + ZUPT (windowed, Eq. 6.44)...")
    start = time.time()
    methods["IMU + ZUPT"], zupt_detected = run_imu_zupt(
        t, accel_meas, gyro_meas, initial, frame, imu_params
    )
    print(f"     Time: {time.time() - start:.3f} s")
    print(
        f"     ZUPT fired on {100 * zupt_detected.mean():.1f}% of samples "
        f"({100 * stance.mean():.1f}% truly stationary)"
    )

    print("  3. Wheel Odometry...")
    start = time.time()
    methods["Wheel Odom"] = run_wheel_odom(
        t, wheel_meas, gyro_meas, initial, LEVER_ARM_A
    )
    print(f"     Time: {time.time() - start:.3f} s")

    print("  4. PDR (step-and-heading)...")
    start = time.time()
    methods["PDR (Mag)"], step_count = run_pdr(t, accel_meas, mag_meas, height)
    print(f"     Time: {time.time() - start:.3f} s")
    print(f"     Steps detected:  {step_count}")

    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)

    print("\nGenerating comparison plots...")
    metrics = plot_comparison(t, pos_true, methods, figs_dir)

    # Print comparison table
    print("\n" + "=" * 75)
    print("RESULTS - Performance Comparison (horizontal error)")
    print("=" * 75)
    print(
        f"\n{'Method':<20} {'RMSE [m]':>10} {'Final [m]':>10} {'Median [m]':>10} "
        f"{'90% [m]':>10} {'Path [m]':>10}"
    )
    print("-" * 75)
    print(
        f"{'(ground truth)':<20} {'-':>10} {'-':>10} {'-':>10} {'-':>10} "
        f"{total_dist:>10.1f}"
    )

    for name in ["IMU Only", "IMU + ZUPT", "Wheel Odom", "PDR (Mag)"]:
        m = metrics[name]
        print(
            f"{name:<20} {m['rmse']:>10.2f} {m['final']:>10.2f} "
            f"{m['median']:>10.2f} {m['p90']:>10.2f} {m['path']:>10.1f}"
        )

    print(f"\nFigures saved to: {resolve_figs_dir(figs_dir)}/")
    print()
    print("=" * 75)
    print("KEY INSIGHTS:")
    zupt_reduction = 100 * (
        1 - metrics["IMU + ZUPT"]["rmse"] / metrics["IMU Only"]["rmse"]
    )
    pdr_overrun = 100 * (metrics["PDR (Mag)"]["path"] / total_dist - 1)

    # Across-seed ranges for the caveats below -- see SWEEP_STATS_12SEED.
    imu_final_lo, imu_final_med, imu_final_hi = SWEEP_STATS_12SEED["imu_only_final_m"]
    zupt_red_lo, zupt_red_med, zupt_red_hi = SWEEP_STATS_12SEED["zupt_reduction_pct"]
    wheel_rmse_lo, _, wheel_rmse_hi = SWEEP_STATS_12SEED["wheel_odom_rmse_m"]
    pdr_rmse_lo, _, pdr_rmse_hi = SWEEP_STATS_12SEED["pdr_rmse_m"]

    print(
        f"  1. IMU-only: UNBOUNDED. {metrics['IMU Only']['final']:.0f} m off "
        f"after {duration:.0f} s on seed {DEFAULT_SEED}, tracing "
        f"{metrics['IMU Only']['path']:.0f} m for a {total_dist:.0f} m walk."
    )
    print(
        f"     One noise draw, not a property of the method: a 12-seed "
        f"sweep (--seed-sweep 12) puts the final error at "
        f"{imu_final_lo:.0f}-{imu_final_hi:.0f} m, median "
        f"{imu_final_med:.0f} m."
    )
    print("     Unusable without corrections either way.")
    print(
        f"  2. IMU+ZUPT: {zupt_reduction:.0f}% RMSE reduction on this seed "
        f"({metrics['IMU Only']['rmse']:.0f} m -> "
        f"{metrics['IMU + ZUPT']['rmse']:.1f} m), detector active on "
        f"{100 * zupt_detected.mean():.0f}% of samples."
    )
    print(
        f"     The reduction is seed-dependent too: {zupt_red_lo:.0f}-"
        f"{zupt_red_hi:.0f}% over the same 12-seed sweep, median "
        f"{zupt_red_med:.0f}%."
    )
    print(
        "     Velocity is reset while standing but attitude is never "
        "corrected, so error still grows -- far more slowly."
    )
    print(
        f"  3. Wheel Odom: BOUNDED. Error follows distance, not time: RMSE "
        f"{metrics['Wheel Odom']['rmse']:.2f} m over {total_dist:.0f} m, set "
        f"by the 2% encoder scale error -- a systematic term, so this stays "
        f"{wheel_rmse_lo:.2f}-{wheel_rmse_hi:.2f} m across the same sweep."
    )
    print(
        "     'Final' is near zero only because the loop closes on its own "
        "start point; read 'Path' instead."
    )
    print(
        f"  4. PDR: BOUNDED, heading-limited. {step_count} detected steps "
        f"cover {metrics['PDR (Mag)']['path']:.1f} m against "
        f"{total_dist:.1f} m ({pdr_overrun:+.1f}%), RMSE "
        f"{metrics['PDR (Mag)']['rmse']:.2f} m ({pdr_rmse_lo:.2f}-"
        f"{pdr_rmse_hi:.2f} m across the sweep; the step count itself does "
        f"not move)."
    )
    print()
    print("  The 'Path' column is the check that makes the rest meaningful: a")
    print("  method frozen at the origin scores well on error alone, because this")
    print("  ground truth ends where it started.")
    print()
    print("  Conclusion: Dead reckoning REQUIRES corrections or fusion!")
    print("             - Use ZUPT for foot-mounted IMU")
    print("             - Use wheel encoders for vehicles")
    print("             - Use magnetometer for heading reference")
    print("             - Best: Multi-sensor fusion (Chapter 8)")
    print("=" * 75)
    print()

    if args.seed_sweep > 0:
        seeds = (DEFAULT_SEED,) + tuple(range(args.seed_sweep - 1))
        print(
            f"Running --seed-sweep {args.seed_sweep}: all four methods, "
            f"once per seed..."
        )
        sweep_start = time.time()
        summary = run_seed_sweep(
            t,
            pos_true,
            accel_body,
            gyro_body,
            mag_body,
            wheel_true,
            dt,
            imu_params,
            initial,
            frame,
            height,
            seeds,
        )
        _print_seed_sweep(summary, seeds, time.time() - sweep_start)
        print()

    show_figures_if_requested()


if __name__ == "__main__":
    main()
