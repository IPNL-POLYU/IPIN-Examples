"""
Example: 1D Kalman Filter Tracking

This script demonstrates the linear Kalman filter applied to 1D constant
velocity tracking, comparing it with simple measurements.

Demonstrates:
    - Linear Kalman Filter (KF) for 1D tracking
    - Comparison with raw measurements
    - Visualization of estimation performance

Implements equations (3.8)-(3.20) from Chapter 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.estimators import KalmanFilter


def example_1d_constant_velocity():
    """
    Example: 1D constant velocity tracking with Kalman Filter.

    Demonstrates Eqs. (3.8)-(3.20): Linear KF for tracking a target
    moving with constant velocity in 1D.
    """
    print("=" * 70)
    print("EXAMPLE: 1D Constant Velocity Tracking with Kalman Filter")
    print("=" * 70)

    # Simulation parameters
    dt = 0.1  # Time step (seconds)
    t_max = 10.0  # Total time (seconds)
    n_steps = int(t_max / dt)

    # True initial state: [position, velocity]
    true_x0 = np.array([0.0, 2.0])  # Start at x=0, velocity=2 m/s

    # State transition matrix (constant velocity model)
    # Eq. (3.11): x_k = F x_{k-1}
    F = np.array([[1.0, dt], [0.0, 1.0]])

    # Process noise covariance (small uncertainty in velocity)
    # Eq. (3.12): Q represents uncertainty in the motion model
    q = 0.1  # Process noise intensity
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    # Measurement matrix (observe position only)
    # Eq. (3.8): z_k = H x_k + w_k
    H = np.array([[1.0, 0.0]])

    # Measurement noise covariance
    # Eq. (3.9): R represents sensor noise
    measurement_std = 0.5  # meters
    R = np.array([[measurement_std**2]])

    # Initial estimate and covariance
    x0_est = np.array([0.0, 0.0])  # Start with poor velocity estimate
    P0 = np.diag([1.0, 5.0])  # High uncertainty in velocity

    print(f"\nSimulation Parameters:")
    print(f"  Time step: {dt} s")
    print(f"  Duration: {t_max} s ({n_steps} steps)")
    print(f"  True initial state: position={true_x0[0]:.1f} m, velocity={true_x0[1]:.1f} m/s")
    print(f"  Measurement noise: {measurement_std:.2f} m (std dev)")
    print(f"  Process noise intensity: {q:.2f}")

    # Generate true trajectory
    print(f"\nGenerating true trajectory...")
    true_states = [true_x0.copy()]
    true_state = true_x0.copy()

    np.random.seed(42)
    for _ in range(n_steps):
        # Propagate with process noise
        process_noise = np.random.multivariate_normal(np.zeros(2), Q)
        true_state = F @ true_state + process_noise
        true_states.append(true_state.copy())

    # Generate noisy measurements
    print(f"Generating noisy measurements...")
    measurements = []
    for state in true_states[1:]:  # Skip initial state
        true_measurement = H @ state
        noise = np.random.normal(0, measurement_std)
        measurements.append(true_measurement[0] + noise)

    # Run Kalman Filter
    print(f"\nRunning Kalman Filter...")
    kf = KalmanFilter(F, Q, H, R, x0_est, P0)

    estimates = [x0_est.copy()]
    covariances = [P0.copy()]

    for z in measurements:
        # Eq. (3.11)-(3.12): Prediction step
        kf.predict(dt=dt)

        # Eq. (3.17)-(3.19): Update step
        kf.update(np.array([z]))

        x_est, P_est = kf.get_state()
        estimates.append(x_est.copy())
        covariances.append(P_est.copy())

    # Convert to arrays
    true_states = np.array(true_states)
    estimates = np.array(estimates)
    measurements = np.array(measurements)
    time = np.arange(n_steps + 1) * dt

    # Compute errors
    position_errors = np.abs(estimates[:, 0] - true_states[:, 0])
    velocity_errors = np.abs(estimates[:, 1] - true_states[:, 1])

    # Statistics
    print(f"\nResults:")
    print(f"  Final true position: {true_states[-1, 0]:.2f} m")
    print(f"  Final estimated position: {estimates[-1, 0]:.2f} m")
    print(f"  Final position error: {position_errors[-1]:.4f} m")
    print(f"  Mean position error: {np.mean(position_errors[10:]):.4f} m")
    print(f"  Final velocity error: {velocity_errors[-1]:.4f} m/s")
    print(f"  Mean velocity error: {np.mean(velocity_errors[10:]):.4f} m/s")

    # Visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Position vs Time
    ax = axes[0, 0]
    ax.plot(time, true_states[:, 0], "g-", linewidth=2, label="True Position")
    ax.plot(time[1:], measurements, "r.", alpha=0.5, markersize=4, label="Measurements")
    ax.plot(time, estimates[:, 0], "b-", linewidth=2, label="KF Estimate")

    # Plot uncertainty bounds (±2σ)
    pos_std = np.array([np.sqrt(P[0, 0]) for P in covariances])
    ax.fill_between(
        time,
        estimates[:, 0] - 2 * pos_std,
        estimates[:, 0] + 2 * pos_std,
        alpha=0.2,
        color="blue",
        label="±2σ Confidence",
    )

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position [m]", fontsize=12)
    ax.set_title("Position Tracking", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity vs Time
    ax = axes[0, 1]
    ax.plot(time, true_states[:, 1], "g-", linewidth=2, label="True Velocity")
    ax.plot(time, estimates[:, 1], "b-", linewidth=2, label="KF Estimate")

    vel_std = np.array([np.sqrt(P[1, 1]) for P in covariances])
    ax.fill_between(
        time,
        estimates[:, 1] - 2 * vel_std,
        estimates[:, 1] + 2 * vel_std,
        alpha=0.2,
        color="blue",
        label="±2σ Confidence",
    )

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity [m/s]", fontsize=12)
    ax.set_title("Velocity Estimation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Position Error
    ax = axes[1, 0]
    ax.plot(time, position_errors, "r-", linewidth=2, label="Position Error")
    ax.axhline(y=measurement_std, color="k", linestyle="--", label=f"Measurement Noise ({measurement_std} m)")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Velocity Error
    ax = axes[1, 1]
    ax.plot(time, velocity_errors, "r-", linewidth=2, label="Velocity Error")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity Error [m/s]", fontsize=12)
    ax.set_title("Velocity Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ch3_kalman_1d_tracking.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved as: ch3_kalman_1d_tracking.png")
    plt.show()


def main():
    """Run the 1D Kalman Filter example."""
    print("\n" + "=" * 70)
    print("CHAPTER 3: KALMAN FILTER EXAMPLE")
    print("=" * 70)
    print("\nDemonstrates Equations (3.8)-(3.20) from Chapter 3")
    print("Application: 1D constant velocity tracking")

    example_1d_constant_velocity()

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

