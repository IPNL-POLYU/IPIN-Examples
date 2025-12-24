"""
Example: Comparison of State Estimation Methods (Chapter 3)

This script compares all state estimation methods from Chapter 3 on the same
2D tracking problem with nonlinear range measurements from multiple anchors.

Run from repository root:
    python ch3_estimators/example_comparison.py

Compares (Section 3.5, Table 3.4):
    - Extended Kalman Filter (EKF) - Section 3.2.2, Eqs. (3.21)-(3.23)
    - Unscented Kalman Filter (UKF) - Section 3.2.4, Eqs. (3.24)-(3.30)
    - Particle Filter (PF) - Section 3.3, Eqs. (3.32)-(3.34) SIR algorithm
    - Factor Graph Optimization (FGO) - Section 3.4, Eqs. (3.35)-(3.41)

Demonstrates the relative performance, accuracy, and computational cost of each.

Particle Filter Algorithm (SIR - Sequential Importance Resampling):
    1. PROPAGATE: x_k^(i) ~ p(x_k | x_{k-1}^(i)) [Eq. 3.33]
    2. WEIGHT: w_k^(i) = w_{k-1}^(i) * p(z_k | x_k^(i)) [Eq. 3.34]
    3. NORMALIZE: w_k^(i) = w_k^(i) / sum(w_k)
    4. RESAMPLE: If N_eff < threshold
    5. ESTIMATE: x_hat = sum(w_k^(i) * x_k^(i)) [weighted mean]

Book Reference: Section 3.5 and Table 3.4 provide comparison criteria.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.estimators import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    Factor,
    FactorGraph,
)


def setup_scenario():
    """
    Set up 2D tracking scenario with range measurements.

    Returns:
        Tuple of (dt, n_steps, anchors, true_trajectory, measurements).
    """
    dt = 0.5
    n_steps = 30

    # Landmark/anchor positions
    anchors = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 20.0],
        [0.0, 20.0],
    ])

    # Generate true trajectory (constant velocity with process noise)
    print("\n--- Setting up scenario ---")
    true_x0 = np.array([10.0, 10.0, 1.0, 0.5])  # [x, y, vx, vy]

    def process_model_true(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    q = 0.5
    Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt]
    ])

    true_states = [true_x0.copy()]
    true_state = true_x0.copy()

    np.random.seed(42)
    for _ in tqdm(range(n_steps), desc="Generating trajectory", unit="step"):
        process_noise = np.random.multivariate_normal(np.zeros(4), Q)
        true_state = process_model_true(true_state, None, dt) + process_noise
        true_states.append(true_state.copy())

    # Generate range measurements from all anchors
    measurements = []
    range_std = 0.5

    for state in tqdm(true_states[1:], desc="Generating measurements", unit="meas"):
        ranges = []
        for anchor in anchors:
            true_range = np.linalg.norm(state[:2] - anchor)
            noisy_range = true_range + np.random.normal(0, range_std)
            ranges.append(noisy_range)
        measurements.append(np.array(ranges))

    return dt, n_steps, anchors, np.array(true_states), measurements, Q, range_std


def run_ekf(dt, n_steps, anchors, measurements, Q, range_std):
    """Run Extended Kalman Filter."""
    print("\nRunning EKF...")

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

    def measurement_model(x):
        ranges = []
        for anchor in anchors:
            r = np.linalg.norm(x[:2] - anchor)
            ranges.append(r)
        return np.array(ranges)

    def measurement_jacobian(x):
        H = []
        for anchor in anchors:
            dx = x[0] - anchor[0]
            dy = x[1] - anchor[1]
            r = np.sqrt(dx**2 + dy**2)
            if r < 1e-6:
                H.append([0, 0, 0, 0])
            else:
                H.append([dx/r, dy/r, 0, 0])
        return np.array(H)

    def Q_func(dt):
        return Q

    def R_func():
        return np.diag([range_std**2] * len(anchors))

    x0 = np.array([10.0, 10.0, 0.0, 0.0])
    P0 = np.diag([4.0, 4.0, 2.0, 2.0])

    ekf = ExtendedKalmanFilter(
        process_model, process_jacobian,
        measurement_model, measurement_jacobian,
        Q_func, R_func, x0, P0
    )

    estimates = [x0.copy()]
    start_time = time.time()

    for z in tqdm(measurements, desc="EKF filtering", unit="step"):
        ekf.predict(dt=dt)
        ekf.update(z)
        x_est, _ = ekf.get_state()
        estimates.append(x_est.copy())

    elapsed_time = time.time() - start_time
    print(f"  [OK] EKF completed in {elapsed_time:.4f}s")

    return np.array(estimates), elapsed_time


def run_ukf(dt, n_steps, anchors, measurements, Q, range_std):
    """Run Unscented Kalman Filter."""
    print("Running UKF...")

    def process_model(x, u, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    def measurement_model(x):
        ranges = []
        for anchor in anchors:
            r = np.linalg.norm(x[:2] - anchor)
            ranges.append(r)
        return np.array(ranges)

    def Q_func(dt):
        return Q

    def R_func():
        return np.diag([range_std**2] * len(anchors))

    x0 = np.array([10.0, 10.0, 0.0, 0.0])
    P0 = np.diag([4.0, 4.0, 2.0, 2.0])

    ukf = UnscentedKalmanFilter(
        process_model, measurement_model,
        Q_func, R_func, x0, P0
    )

    estimates = [x0.copy()]
    start_time = time.time()

    for z in tqdm(measurements, desc="UKF filtering", unit="step"):
        ukf.predict(dt=dt)
        ukf.update(z)
        x_est, _ = ukf.get_state()
        estimates.append(x_est.copy())

    elapsed_time = time.time() - start_time
    print(f"  [OK] UKF completed in {elapsed_time:.4f}s")

    return np.array(estimates), elapsed_time


def run_pf(dt, n_steps, anchors, measurements, Q, range_std):
    """
    Run Particle Filter (Sequential Importance Resampling - SIR).

    Implements the Particle Filter algorithm from Section 3.3 (Eqs. 3.32-3.34):

    SIR Algorithm Steps (each time step):
        1. PROPAGATE: Sample x_k^(i) ~ p(x_k | x_{k-1}^(i)) [Eq. 3.33]
           - Each particle is propagated through process model with noise

        2. WEIGHT: Update weights w_k^(i) = w_{k-1}^(i) * p(z_k | x_k^(i)) [Eq. 3.34]
           - Compute likelihood of measurement given each particle's state

        3. NORMALIZE: w_k^(i) = w_k^(i) / sum(w_k)
           - Ensure weights sum to 1

        4. RESAMPLE: If N_eff < threshold, resample particles
           - Prevents weight degeneracy by duplicating high-weight particles

        5. ESTIMATE: Compute weighted mean (or MAP particle)
           - x_hat = sum(w_k^(i) * x_k^(i))

    Book Reference: Eq. (3.32) - Recursive Bayes update
        p(x_k | z_{1:k}) proportional to p(z_k | x_k) * p(x_k | z_{1:k-1})
    """
    print("Running PF (SIR algorithm, Book Eqs. 3.32-3.34)...")
    n_particles = 300

    # Eq. (3.33): Process model with noise for particle propagation
    # Each particle samples from p(x_k | x_{k-1}^(i))
    def process_model_with_noise(x, u, dt):
        """
        Particle propagation: x_k^(i) ~ p(x_k | x_{k-1}^(i))

        Implements Eq. (3.33): Sample from transition prior.
        """
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        process_noise = np.random.multivariate_normal(np.zeros(4), Q)
        return F @ x + process_noise

    # Eq. (3.34): Likelihood function for weight update
    # w_k^(i) = w_{k-1}^(i) * p(z_k | x_k^(i))
    def likelihood_func(z, x):
        """
        Weight update: w_k^(i) proportional to p(z_k | x_k^(i))

        Implements Eq. (3.34): Compute measurement likelihood.
        Uses Gaussian likelihood for range measurements.
        """
        # Predicted ranges from particle state
        predicted_ranges = np.array([
            np.linalg.norm(x[:2] - anchor) for anchor in anchors
        ])

        # Gaussian likelihood: p(z | x) = N(z; h(x), R)
        residual = z - predicted_ranges
        mahalanobis_sq = np.sum((residual / range_std)**2)
        likelihood = np.exp(-0.5 * mahalanobis_sq)
        # Normalize by Gaussian constant (optional for relative weights)
        likelihood /= (range_std * np.sqrt(2 * np.pi))**len(anchors)
        return likelihood

    x0 = np.array([10.0, 10.0, 0.0, 0.0])
    P0 = np.diag([4.0, 4.0, 2.0, 2.0])

    # Initialize Particle Filter with N particles
    # Particles are drawn from N(x0, P0)
    pf = ParticleFilter(
        process_model_with_noise, likelihood_func,
        n_particles, x0, P0,
        resample_threshold=0.5  # Resample when N_eff < 0.5 * N
    )

    estimates = [x0.copy()]
    start_time = time.time()

    for z in tqdm(measurements, desc=f"PF filtering ({n_particles} particles)", unit="step"):
        # SIR Algorithm per time step:
        # Step 1: PROPAGATE - pf.predict() propagates all particles through
        #         process model with noise [Eq. 3.33]
        pf.predict(dt=dt)

        # Steps 2-5: WEIGHT -> NORMALIZE -> RESAMPLE -> ESTIMATE
        # pf.update() computes likelihoods [Eq. 3.34], normalizes weights,
        # resamples if needed, and computes weighted mean estimate
        pf.update(z)

        # Get weighted mean estimate
        x_est, _ = pf.get_state()
        estimates.append(x_est.copy())

    elapsed_time = time.time() - start_time
    print(f"  [OK] PF completed in {elapsed_time:.4f}s")

    return np.array(estimates), elapsed_time


def run_fgo(dt, n_steps, anchors, measurements, Q, range_std):
    """Run Factor Graph Optimization (batch smoother)."""
    print("Running FGO...")

    # Create factor graph with all variables
    graph = FactorGraph()

    # Add all state variables
    x_init = np.array([10.0, 10.0, 0.0, 0.0])
    for i in range(n_steps + 1):
        graph.add_variable(i, x_init.copy())

    # Add prior factor for first state
    def prior_residual(x_vars):
        return x_vars[0] - x_init

    def prior_jacobian(x_vars):
        return [np.eye(4)]

    prior_info = np.linalg.inv(np.diag([4.0, 4.0, 2.0, 2.0]))
    prior_factor = Factor([0], prior_residual, prior_jacobian, prior_info)
    graph.add_factor(prior_factor)

    # Add process model factors
    Q_inv = np.linalg.inv(Q)

    for i in tqdm(range(n_steps), desc="Adding process factors", unit="factor"):
        def process_residual(x_vars, i=i, dt=dt):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return x_vars[1] - F @ x_vars[0]

        def process_jacobian(x_vars, dt=dt):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return [-F, np.eye(4)]

        process_factor = Factor([i, i + 1], process_residual, process_jacobian, Q_inv)
        graph.add_factor(process_factor)

    # Add measurement factors
    R_inv = np.linalg.inv(np.diag([range_std**2] * len(anchors)))

    for i, z in tqdm(enumerate(measurements), desc="Adding measurement factors", unit="factor", total=len(measurements)):
        def meas_residual(x_vars, z=z):
            x = x_vars[0]
            predicted_ranges = []
            for anchor in anchors:
                r = np.linalg.norm(x[:2] - anchor)
                predicted_ranges.append(r)
            return np.array(predicted_ranges) - z

        def meas_jacobian(x_vars):
            x = x_vars[0]
            H = []
            for anchor in anchors:
                dx = x[0] - anchor[0]
                dy = x[1] - anchor[1]
                r = np.sqrt(dx**2 + dy**2)
                if r < 1e-6:
                    H.append([0, 0, 0, 0])
                else:
                    H.append([dx/r, dy/r, 0, 0])
            return [np.array(H)]

        meas_factor = Factor([i + 1], meas_residual, meas_jacobian, R_inv)
        graph.add_factor(meas_factor)

    # Optimize
    print(f"  Optimizing factor graph (10 Gauss-Newton iterations)...")
    start_time = time.time()
    optimized_vars, _ = graph.optimize(method="gauss_newton", max_iterations=10)
    elapsed_time = time.time() - start_time
    print(f"  [OK] FGO completed in {elapsed_time:.4f}s")

    # Extract estimates
    estimates = []
    for i in range(n_steps + 1):
        estimates.append(optimized_vars[i])

    return np.array(estimates), elapsed_time


def main():
    """Run comparison of all estimators."""
    overall_start = time.time()
    
    print("=" * 70)
    print("CHAPTER 3: COMPARISON OF STATE ESTIMATORS")
    print("=" * 70)
    print("\nScenario: 2D tracking with range measurements from 4 anchors")

    # Set up scenario
    dt, n_steps, anchors, true_states, measurements, Q, range_std = setup_scenario()

    print(f"\nParameters:")
    print(f"  Time step: {dt} s")
    print(f"  Duration: {n_steps * dt} s ({n_steps} steps)")
    print(f"  Number of anchors: {len(anchors)}")
    print(f"  Range measurement std: {range_std} m")

    # Run all estimators
    print("\n" + "=" * 70)
    print("RUNNING ESTIMATORS (1/4 to 4/4)")
    print("=" * 70)

    results = {}
    
    print("\n[1/4] Extended Kalman Filter (EKF)")
    results['EKF'], results['EKF_time'] = run_ekf(dt, n_steps, anchors, measurements, Q, range_std)
    
    print("\n[2/4] Unscented Kalman Filter (UKF)")
    results['UKF'], results['UKF_time'] = run_ukf(dt, n_steps, anchors, measurements, Q, range_std)
    
    print("\n[3/4] Particle Filter (PF)")
    results['PF'], results['PF_time'] = run_pf(dt, n_steps, anchors, measurements, Q, range_std)
    
    print("\n[4/4] Factor Graph Optimization (FGO)")
    results['FGO'], results['FGO_time'] = run_fgo(dt, n_steps, anchors, measurements, Q, range_std)

    # Compute errors
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method in ['EKF', 'UKF', 'PF', 'FGO']:
        estimates = results[method]
        position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
        rmse = np.sqrt(np.mean(position_errors**2))
        mean_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        comp_time = results[f'{method}_time']

        print(f"\n{method}:")
        print(f"  RMSE: {rmse:.4f} m")
        print(f"  Mean error: {mean_error:.4f} m")
        print(f"  Max error: {max_error:.4f} m")
        print(f"  Computation time: {comp_time:.4f} s")

    # Visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    print("Generating plots (this may take a moment)...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.scatter(anchors[:, 0], anchors[:, 1], s=200, c="red", marker="^",
               label="Anchors", zorder=3, edgecolors="black", linewidths=2)
    ax.plot(true_states[:, 0], true_states[:, 1], "k-", linewidth=3, label="True", zorder=2)
    ax.plot(results['EKF'][:, 0], results['EKF'][:, 1], "b--", linewidth=2, label="EKF", alpha=0.7)
    ax.plot(results['UKF'][:, 0], results['UKF'][:, 1], "g--", linewidth=2, label="UKF", alpha=0.7)
    ax.plot(results['PF'][:, 0], results['PF'][:, 1], "m--", linewidth=2, label="PF", alpha=0.7)
    ax.plot(results['FGO'][:, 0], results['FGO'][:, 1], "r:", linewidth=2, label="FGO", alpha=0.7)

    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("Trajectory Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position Errors vs Time
    ax = axes[0, 1]
    time_steps = np.arange(n_steps + 1) * dt
    for method, color, style in [('EKF', 'b', '-'), ('UKF', 'g', '--'), ('PF', 'm', '-.'), ('FGO', 'r', ':')]:
        estimates = results[method]
        position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
        ax.plot(time_steps, position_errors, color + style, linewidth=2, label=method)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Error vs Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: CDF of Errors
    ax = axes[1, 0]
    for method, color in [('EKF', 'b'), ('UKF', 'g'), ('PF', 'm'), ('FGO', 'r')]:
        estimates = results[method]
        position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
        sorted_errors = np.sort(position_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf, color, linewidth=2, label=method)

    ax.set_xlabel("Position Error [m]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Cumulative Distribution of Errors", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Computation Time Comparison
    ax = axes[1, 1]
    methods = ['EKF', 'UKF', 'PF', 'FGO']
    times = [results[f'{m}_time'] for m in methods]
    colors = ['b', 'g', 'm', 'r']
    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel("Computation Time [s]", fontsize=12)
    ax.set_title("Computational Cost", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("ch3_estimator_comparison.png", dpi=150, bbox_inches="tight")
    print("[OK] Plot saved as: ch3_estimator_comparison.png")
    plt.show()

    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()



