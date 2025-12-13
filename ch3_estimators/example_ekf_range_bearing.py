"""
Example: Extended Kalman Filter for 2D Range-Bearing Positioning

This script demonstrates the Extended Kalman Filter applied to 2D positioning
using nonlinear range and bearing measurements from known landmarks.

Demonstrates:
    - Extended Kalman Filter (EKF) for nonlinear systems
    - 2D positioning from range-bearing measurements
    - Comparison with true trajectory

Implements equations (3.21)-(3.22) from Chapter 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.estimators import ExtendedKalmanFilter


def example_2d_range_bearing_positioning():
    """
    Example: 2D positioning with range-bearing measurements using EKF.

    Demonstrates Eqs. (3.21)-(3.22): EKF for tracking a target in 2D
    using nonlinear range and bearing measurements from landmarks.
    """
    print("=" * 70)
    print("EXAMPLE: 2D Range-Bearing Positioning with EKF")
    print("=" * 70)

    # Simulation parameters
    dt = 0.5  # Time step (seconds)
    t_max = 20.0  # Total time (seconds)
    n_steps = int(t_max / dt)

    # Landmark positions (known)
    landmarks = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 20.0],
        [0.0, 20.0],
    ])

    print(f"\nSimulation Parameters:")
    print(f"  Time step: {dt} s")
    print(f"  Duration: {t_max} s ({n_steps} steps)")
    print(f"  Number of landmarks: {len(landmarks)}")

    # True initial state: [x, y, vx, vy]
    true_x0 = np.array([5.0, 5.0, 1.0, 0.5])

    # Process model: constant velocity in 2D
    # Eq. (3.21): x_k = f(x_{k-1}, u_k) + w_k
    def process_model(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    # Eq. (3.22): Jacobian F_k = ∂f/∂x
    def process_jacobian(x, u, dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    # Measurement model: range and bearing to all landmarks
    # Eq. (3.21): z_k = h(x_k) + v_k
    def measurement_model(x):
        measurements = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        return np.array(measurements)

    # Measurement Jacobian: H_k = ∂h/∂x
    def measurement_jacobian(x):
        H = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            r_sq = r**2

            if r < 1e-6:
                # Avoid division by zero
                H.extend([[0, 0, 0, 0], [0, 0, 0, 0]])
            else:
                # Range Jacobian
                dr_dx = -dx / r
                dr_dy = -dy / r
                H.append([dr_dx, dr_dy, 0, 0])

                # Bearing Jacobian
                dtheta_dx = dy / r_sq
                dtheta_dy = -dx / r_sq
                H.append([dtheta_dx, dtheta_dy, 0, 0])

        return np.array(H)

    # Process noise covariance
    q = 0.5
    def Q_func(dt):
        return q * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

    # Measurement noise covariance
    range_std = 0.5  # meters
    bearing_std = 0.05  # radians (~3 degrees)

    def R_func():
        # Diagonal covariance for all range-bearing pairs
        R_diag = []
        for _ in landmarks:
            R_diag.extend([range_std**2, bearing_std**2])
        return np.diag(R_diag)

    print(f"  Range measurement noise: {range_std:.2f} m")
    print(f"  Bearing measurement noise: {np.rad2deg(bearing_std):.2f} deg")

    # Initial estimate and covariance
    x0_est = np.array([5.0, 5.0, 0.0, 0.0])  # Poor velocity estimate
    P0 = np.diag([2.0, 2.0, 2.0, 2.0])

    # Generate true trajectory
    print(f"\nGenerating true trajectory...")
    true_states = [true_x0.copy()]
    true_state = true_x0.copy()

    np.random.seed(42)
    for _ in range(n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(4), Q_func(dt))
        true_state = process_model(true_state, None, dt) + process_noise
        true_states.append(true_state.copy())

    # Generate measurements
    print(f"Generating measurements...")
    measurements = []
    for state in true_states[1:]:
        true_meas = measurement_model(state)
        noise = np.random.multivariate_normal(np.zeros(len(true_meas)), R_func())
        measurements.append(true_meas + noise)

    # Run EKF
    print(f"\nRunning Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0_est, P0
    )

    estimates = [x0_est.copy()]
    covariances = [P0.copy()]

    for z in measurements:
        # Eq. (3.21)-(3.22): EKF prediction
        ekf.predict(dt=dt)

        # EKF update
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

    # Statistics
    print(f"\nResults:")
    print(f"  Final true position: ({true_states[-1, 0]:.2f}, {true_states[-1, 1]:.2f}) m")
    print(f"  Final estimated position: ({estimates[-1, 0]:.2f}, {estimates[-1, 1]:.2f}) m")
    print(f"  Final position error: {position_errors[-1]:.4f} m")
    print(f"  Mean position error: {np.mean(position_errors[5:]):.4f} m")
    print(f"  Final velocity error: {velocity_errors[-1]:.4f} m/s")

    # Visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: 2D Trajectory
    ax = axes[0, 0]
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=200, c="red", marker="^",
               label="Landmarks", zorder=3, edgecolors="black", linewidths=2)
    ax.plot(true_states[:, 0], true_states[:, 1], "g-", linewidth=2, label="True Trajectory")
    ax.plot(estimates[:, 0], estimates[:, 1], "b--", linewidth=2, label="EKF Estimate")
    ax.scatter(true_states[0, 0], true_states[0, 1], s=150, c="green", marker="o",
               label="Start", zorder=3, edgecolors="black")
    ax.scatter(true_states[-1, 0], true_states[-1, 1], s=150, c="orange", marker="s",
               label="End", zorder=3, edgecolors="black")

    # Plot uncertainty ellipse at final position
    P_final = covariances[-1]
    pos_cov = P_final[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2σ ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(estimates[-1, :2], width, height, angle=np.rad2deg(angle),
                      facecolor="blue", alpha=0.2, edgecolor="blue", linewidth=2)
    ax.add_patch(ellipse)

    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("2D Trajectory", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position Error vs Time
    ax = axes[0, 1]
    ax.plot(time, position_errors, "r-", linewidth=2, label="Position Error")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: X and Y positions
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

    # Plot 4: Velocity Error
    ax = axes[1, 1]
    ax.plot(time, velocity_errors, "r-", linewidth=2, label="Velocity Error")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity Error [m/s]", fontsize=12)
    ax.set_title("Velocity Estimation Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ch3_ekf_range_bearing.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved as: ch3_ekf_range_bearing.png")
    plt.show()


def main():
    """Run the EKF range-bearing positioning example."""
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


