"""
Example: Extended Kalman Filter for 2D Range-Bearing Positioning

This script demonstrates the Extended Kalman Filter applied to 2D positioning
using nonlinear range and bearing measurements from known landmarks.

Run from repository root:
    python -m ch3_estimators.example_ekf_range_bearing
    python -m ch3_estimators.example_ekf_range_bearing --data ch3_estimator_nonlinear

Demonstrates:
    - Extended Kalman Filter (EKF) for nonlinear systems (Section 3.2.2)
    - 2D positioning from range-bearing measurements
    - Proper angle wrapping for bearing innovations (handles pi <-> -pi crossing)
    - Comparison with true trajectory

Book Reference (Chapter 3):
    - Eq. (3.21): Nonlinear models x_k = f(x_{k-1}, u_k) + w_k, z_k = h(x_k) + v_k
    - Eq. (3.22): EKF prediction x_k^- = f(x_{k-1}), P_k^- = F P_{k-1} F^T + Q
    - Eq. (3.23): EKF update with Jacobian H_k = dh/dx|_{x_k^-}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.estimators import ExtendedKalmanFilter
from core.eval import save_figure, show_figures_if_requested
from core.utils import angle_diff, resolve_data_path
from core.utils.geometry import check_anchor_geometry
from core.utils.observability import check_range_only_observability_2d


def create_range_bearing_innovation_func(n_landmarks: int):
    """
    Create innovation function that wraps bearing angles properly.

    For range-bearing measurements, the format is:
        z = [range1, bearing1, range2, bearing2, ...]

    Range innovations use simple subtraction, but bearing innovations
    must be wrapped to [-pi, pi] to handle the pi <-> -pi discontinuity.

    Args:
        n_landmarks: Number of landmarks (determines measurement size).

    Returns:
        innovation_func(z, z_pred) -> innovation vector with wrapped bearings.
    """

    def innovation_func(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        innovation = np.zeros_like(z)
        for i in range(n_landmarks):
            # Range innovation (index 2*i): simple subtraction
            innovation[2 * i] = z[2 * i] - z_pred[2 * i]
            # Bearing innovation (index 2*i + 1): angle wrapping
            innovation[2 * i + 1] = angle_diff(z[2 * i + 1], z_pred[2 * i + 1])
        return innovation

    return innovation_func


#: Keys a Chapter 3 dataset's `config.json` must carry before the `--data` path
#: can build `R` from the noise its measurements were actually drawn with. Both
#: name their unit in the key, and both are converted accordingly below.
REQUIRED_MEASUREMENT_KEYS = ("range_noise_std_m", "bearing_noise_std_deg")


def read_measurement_noise(config: Dict, data_dir: str) -> Tuple[float, float]:
    """Read the measurement noise a dataset records, in SI units for the filter.

    Args:
        config: The dataset's parsed `config.json`.
        data_dir: Where it came from, for the error message.

    Returns:
        `(range_std_m, bearing_std_rad)`.

    Raises:
        KeyError: if either required key is absent, naming both.

    A default here is worse than a crash, and this example is why. It used to
    ask for `range_noise_std` and `bearing_noise_std` -- names the generator
    has never written -- so both `.get()` calls fell through to hardcoded
    values and the filter ran `sigma_bearing = 0.05 rad` against a shipped
    5.0 deg = 0.0873 rad, making `R` too small by 3.0x in variance (2.9x
    against the 0.0848 rad the shipped bearings actually scatter by). The
    console printed its own symptom two lines earlier as "Range noise: N/A m"
    and "Bearing noise: 0.00 deg", and nothing failed.
    """
    measurements = config.get("measurements", {})
    missing = [key for key in REQUIRED_MEASUREMENT_KEYS if key not in measurements]
    if missing:
        raise KeyError(
            f"{data_dir}/config.json is missing {missing} under 'measurements'. "
            f"This example builds R from {list(REQUIRED_MEASUREMENT_KEYS)} and "
            f"will not substitute a default for a noise level it cannot read. "
            f"Present under 'measurements': {sorted(measurements)}. Regenerate "
            f"with scripts/generate_ch3_estimator_comparison_dataset.py."
        )
    return (
        float(measurements["range_noise_std_m"]),
        float(np.deg2rad(measurements["bearing_noise_std_deg"])),
    )


def load_estimator_dataset(data_dir: str) -> Dict:
    """Load estimator dataset from directory.

    Args:
        data_dir: Path to dataset directory (e.g., 'data/sim/ch3_estimator_nonlinear')

    Returns:
        Dictionary with time, ground truth, and measurements
    """
    path = Path(data_dir)

    data = {
        "t": np.loadtxt(path / "time.txt"),
        "beacons": np.loadtxt(path / "beacons.txt"),
        "true_states": np.loadtxt(path / "ground_truth_states.txt"),
        "range_meas": np.loadtxt(path / "range_measurements.txt"),
        "bearing_meas": np.loadtxt(path / "bearing_measurements.txt"),
    }

    # Load config
    with open(path / "config.json") as f:
        data["config"] = json.load(f)

    return data


def run_with_dataset(data_dir: str) -> None:
    """Run EKF example using pre-generated dataset.

    Args:
        data_dir: Path to dataset directory
    """
    print("\n" + "=" * 70)
    print("CHAPTER 3: EXTENDED KALMAN FILTER EXAMPLE")
    print(f"Using dataset: {data_dir}")
    print("=" * 70)

    # Load dataset
    data = load_estimator_dataset(data_dir)
    config = data["config"]

    t = data["t"]
    landmarks = data["beacons"]
    true_states = data["true_states"]
    range_meas = data["range_meas"]
    bearing_meas = data["bearing_meas"]

    dt = t[1] - t[0] if len(t) > 1 else 0.5
    n_steps = len(t) - 1

    # Check observability and geometry
    print("\nGeometry Check:")
    initial_pos = true_states[0, :2]
    is_valid, msg = check_anchor_geometry(landmarks, position=initial_pos)
    if not is_valid:
        print(f"  WARNING: {msg}")
    else:
        print("  [OK] Landmark geometry is valid")

    is_obs, obs_msg = check_range_only_observability_2d(
        landmarks, initial_pos, warn=False
    )
    if is_obs:
        print("  [OK] Position is observable from range measurements")
    else:
        print(f"  WARNING: {obs_msg}")

    # Read before printing, so a dataset this example cannot interpret stops
    # here rather than being filtered with someone else's noise model.
    range_std, bearing_std = read_measurement_noise(config, data_dir)

    print("\nDataset Info:")
    print(f"  Duration: {t[-1]:.1f} s ({n_steps} steps)")
    print(f"  Time step: {dt:.2f} s")
    print(f"  Landmarks: {len(landmarks)}")
    print(f"  Range noise: {range_std:.2f} m")
    print(f"  Bearing noise: {np.rad2deg(bearing_std):.2f}° ({bearing_std:.4f} rad)")

    # Process model: constant velocity
    def process_model(x, u, dt_val):
        F = np.array([[1, 0, dt_val, 0], [0, 1, 0, dt_val], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x

    def process_jacobian(x, u, dt_val):
        return np.array(
            [[1, 0, dt_val, 0], [0, 1, 0, dt_val], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    # Measurement model: range and bearing to landmarks
    def measurement_model(x):
        meas = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            meas.extend([r, theta])
        return np.array(meas)

    def measurement_jacobian(x):
        H = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            r_sq = max(r**2, 1e-12)  # Prevent division by zero

            # Use standardized singularity handling
            if r < 1e-6:
                # Singularity: at landmark position
                H.extend([[0, 0, 0, 0], [0, 0, 0, 0]])
            else:
                # Range Jacobian: ∂r/∂[x,y] = [-dx/r, -dy/r]
                H.append([-dx / r, -dy / r, 0, 0])
                # Bearing Jacobian: ∂θ/∂[x,y] = [dy/r², -dx/r²]
                H.append([dy / r_sq, -dx / r_sq, 0, 0])
        return np.array(H)

    # Process noise. This one is genuinely a filter tuning choice and not a
    # dataset property: every trajectory the generator writes is a closed-form
    # curve with no random component (see its `generate_trajectory` docstring),
    # so no `process` block exists to read and asking for one with a default
    # would be the same silent fallback that hid the measurement keys. `R` is
    # read from the dataset above; `Q` is stated here.
    q = 0.5

    def Q_func(dt_val):
        return q * np.array(
            [
                [dt_val**3 / 3, 0, dt_val**2 / 2, 0],
                [0, dt_val**3 / 3, 0, dt_val**2 / 2],
                [dt_val**2 / 2, 0, dt_val, 0],
                [0, dt_val**2 / 2, 0, dt_val],
            ]
        )

    def R_func():
        R_diag = []
        for _ in landmarks:
            R_diag.extend([range_std**2, bearing_std**2])
        return np.diag(R_diag)

    # Initial estimate
    x0_est = np.array([true_states[0, 0], true_states[0, 1], 0.0, 0.0])
    P0 = np.diag([2.0, 2.0, 2.0, 2.0])

    # Create innovation function with angle wrapping for bearings
    innovation_func = create_range_bearing_innovation_func(len(landmarks))

    # Run EKF
    print("\nRunning Extended Kalman Filter...")
    print("  (Using angle wrapping for bearing innovations)")
    ekf = ExtendedKalmanFilter(
        process_model,
        process_jacobian,
        measurement_model,
        measurement_jacobian,
        Q_func,
        R_func,
        x0_est,
        P0,
        innovation_func=innovation_func,
    )

    estimates = [x0_est.copy()]
    covariances = [P0.copy()]
    nis_values = []

    for k in range(n_steps):
        # Form measurement from dataset
        z = []
        for i in range(len(landmarks)):
            z.extend([range_meas[k + 1, i], bearing_meas[k + 1, i]])
        z = np.array(z)

        ekf.predict(dt=dt)

        # Normalised innovation squared, before the update consumes it.
        # Computed here rather than through `ekf.get_innovation`, which
        # subtracts the angles raw and so cannot be used on bearings.
        x_pred, P_pred = ekf.get_state()
        H_pred = measurement_jacobian(x_pred)
        S = H_pred @ P_pred @ H_pred.T + R_func()
        nu = innovation_func(z, measurement_model(x_pred))
        nis_values.append(float(nu @ np.linalg.solve(S, nu)))

        ekf.update(z)
        x_est, P_est = ekf.get_state()
        estimates.append(x_est.copy())
        covariances.append(P_est.copy())

    estimates = np.array(estimates)

    # Compute errors
    position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
    velocity_errors = np.linalg.norm(estimates[:, 2:] - true_states[:, 2:], axis=1)

    print("\nResults:")
    print(f"  Final position error: {position_errors[-1]:.4f} m")
    print(f"  Mean position error: {np.mean(position_errors[5:]):.4f} m")
    print(f"  RMSE position: {np.sqrt(np.mean(position_errors**2)):.4f} m")
    print(f"  Final velocity error: {velocity_errors[-1]:.4f} m/s")

    # Consistency, which is what reading R from the dataset actually buys.
    # The mean NIS should sit near the measurement dimension; well above it
    # means the filter believes its measurements more than the data supports.
    # Building R from a hardcoded sigma_bearing = 0.05 rad instead of the
    # 0.0873 rad this dataset records puts it at 15.6 against an ideal 8 here
    # (16.6 on the high-nonlinearity set) -- two times overconfident -- while
    # moving the position RMSE by only a few percent. Accuracy is the wrong
    # place to look for a wrong R; this line is the right one.
    n_meas = 2 * len(landmarks)
    print(
        f"  Mean NIS (steps 5+): {np.mean(nis_values[5:]):.2f} "
        f"(measurement dimension {n_meas}; equal is consistent)"
    )

    # Visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"EKF Range-Bearing Positioning -- {Path(data_dir).name}\n"
        f"R from the dataset: {range_std:.2f} m range, "
        f"{np.rad2deg(bearing_std):.1f}° bearing",
        fontsize=13,
        fontweight="bold",
    )

    # Trajectory
    ax = axes[0, 0]
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        s=200,
        c="red",
        marker="^",
        label="Landmarks",
        zorder=3,
        edgecolors="black",
        linewidths=2,
    )
    ax.plot(
        true_states[:, 0], true_states[:, 1], "g-", linewidth=2, label="True Trajectory"
    )
    ax.plot(estimates[:, 0], estimates[:, 1], "b--", linewidth=2, label="EKF Estimate")
    ax.scatter(
        true_states[0, 0],
        true_states[0, 1],
        s=150,
        c="green",
        marker="o",
        label="Start",
        zorder=3,
    )
    ax.scatter(
        true_states[-1, 0],
        true_states[-1, 1],
        s=150,
        c="orange",
        marker="s",
        label="End",
        zorder=3,
    )
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    ax.set_title("2D Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Position error
    ax = axes[0, 1]
    ax.plot(t, position_errors, "r-", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Estimation Error")
    ax.grid(True, alpha=0.3)

    # X/Y positions
    ax = axes[1, 0]
    ax.plot(t, true_states[:, 0], "g-", linewidth=2, label="True X")
    ax.plot(t, estimates[:, 0], "b--", linewidth=2, label="Est X")
    ax.plot(t, true_states[:, 1], "g:", linewidth=2, label="True Y")
    ax.plot(t, estimates[:, 1], "b:", linewidth=2, label="Est Y")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("X and Y Positions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity error
    ax = axes[1, 1]
    ax.plot(t, velocity_errors, "r-", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity Error [m/s]")
    ax.set_title("Velocity Estimation Error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # A different basename from the inline run's. Both used to be
    # `ch3_ekf_range_bearing`, so `--data` silently overwrote the committed
    # figure the chapter README displays with a picture of a different
    # trajectory. Same fix as Chapter 4's `ch4_rf_comparison_dataset`.
    paths = save_figure(
        fig, Path(__file__).parent / "figs", "ch3_ekf_range_bearing_dataset"
    )
    print(f"Plot saved: {paths[0]}")

    show_figures_if_requested()

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70)


def example_2d_range_bearing_positioning():
    """
    Example: 2D positioning with range-bearing measurements using EKF (inline data).

    Demonstrates Eqs. (3.21)-(3.22): EKF for tracking a target in 2D
    using nonlinear range and bearing measurements from landmarks.
    """
    print("=" * 70)
    print("EXAMPLE: 2D Range-Bearing Positioning with EKF")
    print("(Using inline generated data)")
    print("=" * 70)

    # Simulation parameters
    dt = 0.5  # Time step (seconds)
    t_max = 20.0  # Total time (seconds)
    n_steps = int(t_max / dt)

    # Landmark positions (known)
    landmarks = np.array(
        [
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0],
            [0.0, 20.0],
        ]
    )

    print("\nSimulation Parameters:")
    print(f"  Time step: {dt} s")
    print(f"  Duration: {t_max} s ({n_steps} steps)")
    print(f"  Number of landmarks: {len(landmarks)}")

    # True initial state: [x, y, vx, vy]
    true_x0 = np.array([5.0, 5.0, 1.0, 0.5])

    # Check observability and geometry
    print("\nGeometry Check:")
    is_valid, msg = check_anchor_geometry(landmarks, position=true_x0[:2])
    if not is_valid:
        print(f"  WARNING: {msg}")
    else:
        print("  [OK] Landmark geometry is valid")

    is_obs, obs_msg = check_range_only_observability_2d(
        landmarks, true_x0[:2], warn=False
    )
    if is_obs:
        print("  [OK] Position is observable from range measurements")
    else:
        print(f"  WARNING: {obs_msg}")

    # Process model: constant velocity in 2D
    def process_model(x, u, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x

    def process_jacobian(x, u, dt):
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Measurement model: range and bearing to all landmarks
    def measurement_model(x):
        measurements = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        return np.array(measurements)

    def measurement_jacobian(x):
        H = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            r_sq = max(r**2, 1e-12)  # Prevent division by zero

            # Standardized singularity handling
            if r < 1e-6:
                # Singularity: at landmark position
                H.extend([[0, 0, 0, 0], [0, 0, 0, 0]])
            else:
                # Range Jacobian: ∂r/∂[x,y] = [-dx/r, -dy/r]
                dr_dx = -dx / r
                dr_dy = -dy / r
                H.append([dr_dx, dr_dy, 0, 0])
                # Bearing Jacobian: ∂θ/∂[x,y] = [dy/r², -dx/r²]
                dtheta_dx = dy / r_sq
                dtheta_dy = -dx / r_sq
                H.append([dtheta_dx, dtheta_dy, 0, 0])

        return np.array(H)

    # Process noise covariance
    q = 0.5

    def Q_func(dt):
        return q * np.array(
            [
                [dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**3 / 3, 0, dt**2 / 2],
                [dt**2 / 2, 0, dt, 0],
                [0, dt**2 / 2, 0, dt],
            ]
        )

    # Measurement noise covariance
    range_std = 0.5
    bearing_std = 0.05

    def R_func():
        R_diag = []
        for _ in landmarks:
            R_diag.extend([range_std**2, bearing_std**2])
        return np.diag(R_diag)

    print(f"  Range measurement noise: {range_std:.2f} m")
    print(f"  Bearing measurement noise: {np.rad2deg(bearing_std):.2f} deg")

    # Initial estimate and covariance
    x0_est = np.array([5.0, 5.0, 0.0, 0.0])
    P0 = np.diag([2.0, 2.0, 2.0, 2.0])

    # Generate true trajectory
    print("\nGenerating true trajectory...")
    true_states = [true_x0.copy()]
    true_state = true_x0.copy()

    np.random.seed(42)
    for _ in range(n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(4), Q_func(dt))
        true_state = process_model(true_state, None, dt) + process_noise
        true_states.append(true_state.copy())

    # Generate measurements
    print("Generating measurements...")
    measurements = []
    for state in true_states[1:]:
        true_meas = measurement_model(state)
        noise = np.random.multivariate_normal(np.zeros(len(true_meas)), R_func())
        measurements.append(true_meas + noise)

    # Create innovation function with angle wrapping for bearings
    innovation_func = create_range_bearing_innovation_func(len(landmarks))

    # Run EKF
    print("\nRunning Extended Kalman Filter...")
    print("  (Using angle wrapping for bearing innovations)")
    ekf = ExtendedKalmanFilter(
        process_model,
        process_jacobian,
        measurement_model,
        measurement_jacobian,
        Q_func,
        R_func,
        x0_est,
        P0,
        innovation_func=innovation_func,
    )

    estimates = [x0_est.copy()]
    covariances = [P0.copy()]

    for z in measurements:
        ekf.predict(dt=dt)
        ekf.update(z)
        x_est, P_est = ekf.get_state()
        estimates.append(x_est.copy())
        covariances.append(P_est.copy())

    # Convert to arrays
    true_states = np.array(true_states)
    estimates = np.array(estimates)
    time = np.arange(n_steps + 1) * dt

    # Compute errors
    position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
    velocity_errors = np.linalg.norm(estimates[:, 2:] - true_states[:, 2:], axis=1)

    print("\nResults:")
    print(
        f"  Final true position: ({true_states[-1, 0]:.2f}, {true_states[-1, 1]:.2f}) m"
    )
    print(
        f"  Final estimated position: ({estimates[-1, 0]:.2f}, {estimates[-1, 1]:.2f}) m"
    )
    print(f"  Final position error: {position_errors[-1]:.4f} m")
    print(f"  Mean position error: {np.mean(position_errors[5:]):.4f} m")
    print(f"  Final velocity error: {velocity_errors[-1]:.4f} m/s")

    # Visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        s=200,
        c="red",
        marker="^",
        label="Landmarks",
        zorder=3,
        edgecolors="black",
        linewidths=2,
    )
    ax.plot(
        true_states[:, 0], true_states[:, 1], "g-", linewidth=2, label="True Trajectory"
    )
    ax.plot(estimates[:, 0], estimates[:, 1], "b--", linewidth=2, label="EKF Estimate")
    ax.scatter(
        true_states[0, 0],
        true_states[0, 1],
        s=150,
        c="green",
        marker="o",
        label="Start",
        zorder=3,
        edgecolors="black",
    )
    ax.scatter(
        true_states[-1, 0],
        true_states[-1, 1],
        s=150,
        c="orange",
        marker="s",
        label="End",
        zorder=3,
        edgecolors="black",
    )

    P_final = covariances[-1]
    pos_cov = P_final[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        estimates[-1, :2],
        width,
        height,
        angle=np.rad2deg(angle),
        facecolor="blue",
        alpha=0.2,
        edgecolor="blue",
        linewidth=2,
    )
    ax.add_patch(ellipse)

    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("2D Trajectory", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    ax = axes[0, 1]
    ax.plot(time, position_errors, "r-", linewidth=2, label="Position Error")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time, true_states[:, 0], "g-", linewidth=2, label="True X")
    ax.plot(time, estimates[:, 0], "b--", linewidth=2, label="Estimated X")
    ax.plot(time, true_states[:, 1], "g:", linewidth=2, label="True Y")
    ax.plot(time, estimates[:, 1], "b:", linewidth=2, label="Estimated Y")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position [m]", fontsize=12)
    ax.set_title("X and Y Positions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time, velocity_errors, "r-", linewidth=2, label="Velocity Error")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity Error [m/s]", fontsize=12)
    ax.set_title("Velocity Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    paths = save_figure(fig, Path(__file__).parent / "figs", "ch3_ekf_range_bearing")
    print(f"Plot saved as: {paths[0]}")
    show_figures_if_requested()

    print("\nTip: Run with --data ch3_estimator_nonlinear to use pre-generated dataset")


def main():
    """Run the EKF range-bearing positioning example."""
    parser = argparse.ArgumentParser(
        description="Chapter 3: Extended Kalman Filter Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with inline generated data (default)
  python example_ekf_range_bearing.py
  
  # Run with pre-generated dataset
  python example_ekf_range_bearing.py --data ch3_estimator_nonlinear
  
  # Run with high nonlinearity scenario
  python example_ekf_range_bearing.py --data ch3_estimator_high_nonlinear
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset name or path (e.g., 'ch3_estimator_nonlinear' or full path)",
    )

    args = parser.parse_args()

    if args.data:
        # Run with dataset
        data_path = resolve_data_path(args.data)
        if not data_path.exists():
            data_path = resolve_data_path(Path("data/sim") / args.data)
        if not data_path.exists():
            print(
                f"Error: Dataset not found at '{args.data}' or 'data/sim/{args.data}'"
            )
            print("\nAvailable datasets:")
            sim_dir = resolve_data_path(Path("data/sim"))
            if sim_dir.exists():
                for d in sorted(sim_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("ch3"):
                        print(f"  - {d.name}")
            return

        run_with_dataset(str(data_path))
    else:
        print("\n" + "=" * 70)
        print("CHAPTER 3: EXTENDED KALMAN FILTER EXAMPLE")
        print("=" * 70)
        print("\nDemonstrates Equations (3.21)-(3.22) from Chapter 3")
        print("Application: 2D positioning from range-bearing measurements")

        example_2d_range_bearing_positioning()

        print("\n" + "=" * 70)
        print("EXAMPLE COMPLETED")
        print("=" * 70)


if __name__ == "__main__":
    main()
