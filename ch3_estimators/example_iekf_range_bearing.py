"""
Example: Iterated Extended Kalman Filter for 2D Range-Bearing Positioning.

This script demonstrates the IEKF applied to 2D positioning using nonlinear
range and bearing measurements, comparing it with standard EKF to show
improved stability with high nonlinearity.

Run from repository root:
    python ch3_estimators/example_iekf_range_bearing.py

Demonstrates:
    - Iterated Extended Kalman Filter (IEKF) from Section 3.2.3
    - Comparison with EKF showing IEKF's advantage in high nonlinearity
    - 2D positioning from range-bearing measurements
    - Proper angle wrapping for bearing innovations (handles pi <-> -pi crossing)

Book Reference (Chapter 3, Section 3.2.3):
    "When measurement nonlinearities are severe, one way to improve the EKF's
    accuracy is to iterate the update step multiple times per measurement."

IEKF Algorithm:
    1. Initialize: x_k^(0) = x_k^- (predicted state)
    2. Iterate j = 0, 1, ..., N:
       - H_k^(j) = dh/dx|_{x_k^(j)}
       - y_k^(j) = z_k - h(x_k^(j)) + H_k^(j) (x_k^(j) - x_k^-)
       - K_k^(j) = P_k^- H^T (H P_k^- H^T + R)^{-1}
       - x_k^(j+1) = x_k^- + K_k^(j) y_k^(j)
    3. Finalize: P_k = (I - K H) P_k^-
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.estimators import ExtendedKalmanFilter, IteratedExtendedKalmanFilter
from core.utils import angle_diff


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


def setup_high_nonlinearity_scenario():
    """
    Setup a scenario with high measurement nonlinearity.

    Returns a configuration where range-bearing measurements become
    highly nonlinear due to close proximity to landmarks.
    """
    # Landmarks positioned to create high nonlinearity when close
    landmarks = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
        [15.0, 15.0],
        [0.0, 15.0],
    ])

    # Start position close to a landmark (high nonlinearity)
    true_x0 = np.array([2.0, 2.0, 0.8, 0.6])

    return landmarks, true_x0


def create_models(landmarks):
    """Create process and measurement models for range-bearing positioning."""

    # Process model: constant velocity in 2D
    def process_model(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    def process_jacobian(x, u, dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    # Measurement model: range and bearing to all landmarks
    def measurement_model(x):
        measurements = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        return np.array(measurements)

    def measurement_jacobian(x):
        H = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx ** 2 + dy ** 2)
            r_sq = max(r ** 2, 1e-12)

            if r < 1e-6:
                # At landmark - singularity
                H.extend([[0, 0, 0, 0], [0, 0, 0, 0]])
            else:
                # Range Jacobian
                H.append([-dx / r, -dy / r, 0, 0])
                # Bearing Jacobian
                H.append([dy / r_sq, -dx / r_sq, 0, 0])
        return np.array(H)

    return process_model, process_jacobian, measurement_model, measurement_jacobian


def example_iekf_vs_ekf_comparison():
    """
    Compare IEKF vs EKF on high nonlinearity scenario.

    Demonstrates Section 3.2.3: "IEKF iteratively refines EKF updates for
    improved linearization accuracy."
    """
    print("=" * 70)
    print("EXAMPLE: IEKF vs EKF Comparison")
    print("Section 3.2.3: Iterated Extended Kalman Filter")
    print("=" * 70)

    # Setup scenario
    landmarks, true_x0 = setup_high_nonlinearity_scenario()
    process_model, process_jacobian, measurement_model, measurement_jacobian = \
        create_models(landmarks)

    # Simulation parameters
    dt = 0.5
    t_max = 25.0
    n_steps = int(t_max / dt)

    print(f"\nScenario: High Nonlinearity (close to landmarks)")
    print(f"  Duration: {t_max} s ({n_steps} steps)")
    print(f"  Time step: {dt} s")
    print(f"  Landmarks: {len(landmarks)}")
    print(f"  True initial position: ({true_x0[0]:.1f}, {true_x0[1]:.1f}) m")

    # Noise covariances
    q = 0.3  # Process noise

    def Q_func(dt):
        return q * np.array([
            [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
            [0, dt ** 3 / 3, 0, dt ** 2 / 2],
            [dt ** 2 / 2, 0, dt, 0],
            [0, dt ** 2 / 2, 0, dt]
        ])

    range_std = 0.3
    bearing_std = 0.08  # Higher bearing noise for more challenge

    def R_func():
        R_diag = []
        for _ in landmarks:
            R_diag.extend([range_std ** 2, bearing_std ** 2])
        return np.diag(R_diag)

    print(f"  Range noise: {range_std:.2f} m")
    print(f"  Bearing noise: {np.rad2deg(bearing_std):.2f} deg")

    # Initial estimate with significant error (to challenge filters)
    x0_est = np.array([4.0, 4.0, 0.0, 0.0])  # Wrong initial position
    P0 = np.diag([3.0, 3.0, 2.0, 2.0])

    print(f"  Initial estimate: ({x0_est[0]:.1f}, {x0_est[1]:.1f}) m")
    print(f"  Initial error: {np.linalg.norm(x0_est[:2] - true_x0[:2]):.2f} m")

    # Create innovation function with angle wrapping for bearings
    innovation_func = create_range_bearing_innovation_func(len(landmarks))

    # Create both filters
    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0_est.copy(), P0.copy(),
        innovation_func=innovation_func
    )

    iekf = IteratedExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0_est.copy(), P0.copy(),
        max_iterations=5,
        convergence_tol=1e-6,
        innovation_func=innovation_func
    )

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

    # Run both filters
    print(f"\nRunning EKF and IEKF...")

    ekf_estimates = [x0_est.copy()]
    iekf_estimates = [x0_est.copy()]
    iekf_iterations = []

    for z in measurements:
        # EKF
        ekf.predict(dt=dt)
        ekf.update(z)
        ekf_est, _ = ekf.get_state()
        ekf_estimates.append(ekf_est.copy())

        # IEKF
        iekf.predict(dt=dt)
        n_iters = iekf.update(z)
        iekf_est, _ = iekf.get_state()
        iekf_estimates.append(iekf_est.copy())
        iekf_iterations.append(n_iters)

    # Convert to arrays
    true_states = np.array(true_states)
    ekf_estimates = np.array(ekf_estimates)
    iekf_estimates = np.array(iekf_estimates)
    time = np.arange(n_steps + 1) * dt

    # Compute errors
    ekf_pos_errors = np.linalg.norm(ekf_estimates[:, :2] - true_states[:, :2], axis=1)
    iekf_pos_errors = np.linalg.norm(iekf_estimates[:, :2] - true_states[:, :2], axis=1)

    ekf_vel_errors = np.linalg.norm(ekf_estimates[:, 2:] - true_states[:, 2:], axis=1)
    iekf_vel_errors = np.linalg.norm(iekf_estimates[:, 2:] - true_states[:, 2:], axis=1)

    # Results
    print(f"\n" + "=" * 50)
    print(f"RESULTS COMPARISON")
    print(f"=" * 50)
    print(f"\n{'Metric':<30} {'EKF':<15} {'IEKF':<15}")
    print("-" * 60)
    print(f"{'Mean position error (m)':<30} {np.mean(ekf_pos_errors[5:]):<15.4f} "
          f"{np.mean(iekf_pos_errors[5:]):<15.4f}")
    print(f"{'RMSE position (m)':<30} {np.sqrt(np.mean(ekf_pos_errors**2)):<15.4f} "
          f"{np.sqrt(np.mean(iekf_pos_errors**2)):<15.4f}")
    print(f"{'Max position error (m)':<30} {np.max(ekf_pos_errors):<15.4f} "
          f"{np.max(iekf_pos_errors):<15.4f}")
    print(f"{'Final position error (m)':<30} {ekf_pos_errors[-1]:<15.4f} "
          f"{iekf_pos_errors[-1]:<15.4f}")
    print(f"{'Mean velocity error (m/s)':<30} {np.mean(ekf_vel_errors[5:]):<15.4f} "
          f"{np.mean(iekf_vel_errors[5:]):<15.4f}")

    improvement = (np.mean(ekf_pos_errors[5:]) - np.mean(iekf_pos_errors[5:])) / \
                  np.mean(ekf_pos_errors[5:]) * 100
    print(f"\nIEKF improvement: {improvement:.1f}%")
    print(f"Mean IEKF iterations per update: {np.mean(iekf_iterations):.1f}")

    # Visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IEKF vs EKF: High Nonlinearity Comparison (Section 3.2.3)',
                 fontsize=14, fontweight='bold')

    # Trajectory comparison
    ax = axes[0, 0]
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=200, c="red", marker="^",
               label="Landmarks", zorder=5, edgecolors="black", linewidths=2)
    ax.plot(true_states[:, 0], true_states[:, 1], "g-", linewidth=2,
            label="True Trajectory")
    ax.plot(ekf_estimates[:, 0], ekf_estimates[:, 1], "b--", linewidth=2,
            label="EKF Estimate", alpha=0.7)
    ax.plot(iekf_estimates[:, 0], iekf_estimates[:, 1], "m-.", linewidth=2,
            label="IEKF Estimate")
    ax.scatter(true_states[0, 0], true_states[0, 1], s=150, c="green", marker="o",
               label="Start", zorder=5, edgecolors="black")
    ax.scatter(true_states[-1, 0], true_states[-1, 1], s=150, c="orange", marker="s",
               label="End", zorder=5, edgecolors="black")
    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("2D Trajectory Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Position error comparison
    ax = axes[0, 1]
    ax.plot(time, ekf_pos_errors, "b-", linewidth=2, label="EKF", alpha=0.7)
    ax.plot(time, iekf_pos_errors, "m-", linewidth=2, label="IEKF")
    ax.axhline(y=np.mean(ekf_pos_errors[5:]), color="b", linestyle="--", alpha=0.5,
               label=f"EKF mean: {np.mean(ekf_pos_errors[5:]):.2f} m")
    ax.axhline(y=np.mean(iekf_pos_errors[5:]), color="m", linestyle="--", alpha=0.5,
               label=f"IEKF mean: {np.mean(iekf_pos_errors[5:]):.2f} m")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Error Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # IEKF iterations
    ax = axes[1, 0]
    ax.bar(time[1:], iekf_iterations, width=dt * 0.8, color="purple", alpha=0.7)
    ax.axhline(y=np.mean(iekf_iterations), color="r", linestyle="--",
               label=f"Mean: {np.mean(iekf_iterations):.1f}")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("IEKF Iterations", fontsize=12)
    ax.set_title("IEKF Iterations per Update", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(iekf_iterations) + 1)

    # Cumulative error
    ax = axes[1, 1]
    ekf_cumulative = np.cumsum(ekf_pos_errors ** 2)
    iekf_cumulative = np.cumsum(iekf_pos_errors ** 2)
    ax.plot(time, ekf_cumulative, "b-", linewidth=2, label="EKF")
    ax.plot(time, iekf_cumulative, "m-", linewidth=2, label="IEKF")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Cumulative Squared Error [m^2]", fontsize=12)
    ax.set_title("Cumulative Position Error", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    figs_dir = Path("ch3_estimators/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_file = figs_dir / "ch3_iekf_vs_ekf_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("Key Insight from Section 3.2.3:")
    print("  'IEKF iteratively refines EKF updates for improved linearization")
    print("  accuracy at the cost of additional computation. Typically, 2-5")
    print("  iterations are sufficient.'")
    print("=" * 70)

    return ekf_estimates, iekf_estimates, true_states


def example_iekf_convergence_demo():
    """
    Demonstrate IEKF convergence behavior within a single update step.

    Shows how the modified residual y_k^(j) converges across iterations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: IEKF Convergence Within Single Update")
    print("=" * 70)

    landmarks = np.array([[0.0, 0.0], [10.0, 0.0]])

    # Measurement model
    def measurement_model(x):
        measurements = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        return np.array(measurements)

    def measurement_jacobian(x):
        H = []
        for lm in landmarks:
            dx = lm[0] - x[0]
            dy = lm[1] - x[1]
            r = np.sqrt(dx ** 2 + dy ** 2)
            r_sq = max(r ** 2, 1e-12)
            if r < 1e-6:
                H.extend([[0, 0, 0, 0], [0, 0, 0, 0]])
            else:
                H.append([-dx / r, -dy / r, 0, 0])
                H.append([dy / r_sq, -dx / r_sq, 0, 0])
        return np.array(H)

    # True state
    true_state = np.array([2.0, 3.0, 0.5, 0.3])
    z_true = measurement_model(true_state)

    # Predicted state with error
    x_pred = np.array([4.0, 5.0, 0.0, 0.0])
    P_pred = np.diag([2.0, 2.0, 1.0, 1.0])
    R = np.diag([0.3 ** 2, 0.05 ** 2, 0.3 ** 2, 0.05 ** 2])

    print(f"\nTrue state:      {true_state[:2]}")
    print(f"Predicted state: {x_pred[:2]}")
    print(f"Initial error:   {np.linalg.norm(x_pred[:2] - true_state[:2]):.3f} m")

    # Manual IEKF iteration to show convergence
    x_iter = x_pred.copy()
    print(f"\nIEKF Iteration Progress:")
    print(f"{'Iter':<6} {'x':<8} {'y':<8} {'||delta_x||':<12} {'||residual||':<12}")
    print("-" * 50)

    for j in range(10):
        H = measurement_jacobian(x_iter)
        z_pred = measurement_model(x_iter)

        # Modified residual
        delta_x_from_pred = x_iter - x_pred
        y_modified = z_true - z_pred + H @ delta_x_from_pred

        # Kalman gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        x_new = x_pred + K @ y_modified
        step_size = np.linalg.norm(x_new - x_iter)
        residual_norm = np.linalg.norm(z_true - measurement_model(x_new))

        print(f"{j:<6} {x_iter[0]:<8.4f} {x_iter[1]:<8.4f} "
              f"{step_size:<12.6f} {residual_norm:<12.6f}")

        if step_size < 1e-6:
            print(f"\nConverged at iteration {j + 1}")
            break

        x_iter = x_new

    final_error = np.linalg.norm(x_iter[:2] - true_state[:2])
    print(f"\nFinal state:     {x_iter[:2]}")
    print(f"Final error:     {final_error:.4f} m")

    return x_iter


def main():
    """Run the IEKF examples."""
    parser = argparse.ArgumentParser(
        description="Chapter 3: Iterated Extended Kalman Filter Example (Section 3.2.3)"
    )
    parser.add_argument(
        "--demo", choices=["comparison", "convergence", "both"],
        default="both",
        help="Which demo to run"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CHAPTER 3: ITERATED EXTENDED KALMAN FILTER (IEKF)")
    print("Section 3.2.3 Implementation")
    print("=" * 70)
    print("\nBook Reference:")
    print("  'When measurement nonlinearities are severe, one way to improve")
    print("  the EKF's accuracy is to iterate the update step multiple times")
    print("  per measurement.'")

    if args.demo in ["comparison", "both"]:
        example_iekf_vs_ekf_comparison()

    if args.demo in ["convergence", "both"]:
        example_iekf_convergence_demo()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

