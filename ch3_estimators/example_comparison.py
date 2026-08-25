"""
Example: Comparison of State Estimation Methods (Chapter 3)

This script compares all state estimation methods from Chapter 3 on the same
2D tracking problem with nonlinear range measurements from multiple anchors.

Run from repository root:
    python -m ch3_estimators.example_comparison

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

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.estimators import (
    ExtendedKalmanFilter,
    Factor,
    FactorGraph,
    ParticleFilter,
    UnscentedKalmanFilter,
)
from core.eval import (
    plot_error_cdf,
    plot_error_magnitude_time,
    plot_trajectory_2d,
    save_figure,
    show_figures_if_requested,
)


def setup_scenario(seed=42):
    """
    Set up 2D tracking scenario with range measurements.

    Args:
        seed: Seed for the trajectory and measurement noise. Parameterised so
            the comparison can be repeated over realisations; a single one
            cannot separate a real ordering from a coincidence. The default
            keeps the committed figures unchanged.

    Returns:
        Tuple of (dt, n_steps, anchors, true_trajectory, measurements).
    """
    dt = 0.5
    n_steps = 30

    # Landmark/anchor positions
    anchors = np.array(
        [
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0],
            [0.0, 20.0],
        ]
    )

    # Generate true trajectory (constant velocity with process noise)
    print("\n--- Setting up scenario ---")
    true_x0 = np.array([10.0, 10.0, 1.0, 0.5])  # [x, y, vx, vy]

    def process_model_true(x, u, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x

    q = 0.5
    Q = q * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ]
    )

    true_states = [true_x0.copy()]
    true_state = true_x0.copy()

    np.random.seed(seed)
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
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x

    def process_jacobian(x, u, dt):
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

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
                H.append([dx / r, dy / r, 0, 0])
        return np.array(H)

    def Q_func(dt):
        return Q

    def R_func():
        return np.diag([range_std**2] * len(anchors))

    x0 = np.array([10.0, 10.0, 0.0, 0.0])
    P0 = np.diag([4.0, 4.0, 2.0, 2.0])

    ekf = ExtendedKalmanFilter(
        process_model,
        process_jacobian,
        measurement_model,
        measurement_jacobian,
        Q_func,
        R_func,
        x0,
        P0,
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
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
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
        process_model, measurement_model, Q_func, R_func, x0, P0
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
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
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
        predicted_ranges = np.array(
            [np.linalg.norm(x[:2] - anchor) for anchor in anchors]
        )

        # Gaussian likelihood: p(z | x) = N(z; h(x), R)
        residual = z - predicted_ranges
        mahalanobis_sq = np.sum((residual / range_std) ** 2)
        likelihood = np.exp(-0.5 * mahalanobis_sq)
        # Normalize by Gaussian constant (optional for relative weights)
        likelihood /= (range_std * np.sqrt(2 * np.pi)) ** len(anchors)
        return likelihood

    x0 = np.array([10.0, 10.0, 0.0, 0.0])
    P0 = np.diag([4.0, 4.0, 2.0, 2.0])

    # Initialize Particle Filter with N particles
    # Particles are drawn from N(x0, P0)
    pf = ParticleFilter(
        process_model_with_noise,
        likelihood_func,
        n_particles,
        x0,
        P0,
        resample_threshold=0.5,  # Resample when N_eff < 0.5 * N
    )

    estimates = [x0.copy()]
    start_time = time.time()

    for z in tqdm(
        measurements, desc=f"PF filtering ({n_particles} particles)", unit="step"
    ):
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

    return np.array(estimates), elapsed_time, n_particles


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
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            return x_vars[1] - F @ x_vars[0]

        def process_jacobian(x_vars, dt=dt):
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            return [-F, np.eye(4)]

        process_factor = Factor([i, i + 1], process_residual, process_jacobian, Q_inv)
        graph.add_factor(process_factor)

    # Add measurement factors
    R_inv = np.linalg.inv(np.diag([range_std**2] * len(anchors)))

    for i, z in tqdm(
        enumerate(measurements),
        desc="Adding measurement factors",
        unit="factor",
        total=len(measurements),
    ):

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
                    H.append([dx / r, dy / r, 0, 0])
            return [np.array(H)]

        meas_factor = Factor([i + 1], meas_residual, meas_jacobian, R_inv)
        graph.add_factor(meas_factor)

    # Optimize
    print("  Optimizing factor graph (up to 10 Gauss-Newton iterations)...")
    start_time = time.time()
    optimized_vars, costs = graph.optimize(method="gauss_newton", max_iterations=10)
    elapsed_time = time.time() - start_time
    # optimize() returns the cost at each iteration, so this is the number of
    # iterations actually taken rather than the 10 it was allowed.
    n_iterations = len(costs)
    print(
        f"  [OK] FGO completed in {elapsed_time:.4f}s " f"({n_iterations} iterations)"
    )

    # Extract estimates
    estimates = []
    for i in range(n_steps + 1):
        estimates.append(optimized_vars[i])

    return np.array(estimates), elapsed_time, n_iterations


def model_evaluation_counts(n_steps, n_states, n_particles, fgo_iterations):
    """Count how many model evaluations each estimator spends on the run.

    This is what the cost panel plots, in place of the wall-clock seconds it
    used to. A measured second cannot be committed to a figure: it differs on
    every run and every machine, so the figure churned on regeneration while
    telling a reader nothing about their own hardware. What separates these four
    estimators is not clock speed but how many times each pushes a state vector
    through the process and measurement models, and that is exact.

    One evaluation is one state vector passed through one model:

        - EKF propagates the mean and predicts the measurement once per step.
        - UKF does both for each of its 2n+1 sigma points (Section 3.4).
        - PF does both for every particle -- this is why it dominates.
        - FGO is a batch smoother: each Gauss-Newton iteration re-evaluates the
          motion and measurement residual at every step, so its cost scales with
          the iterations it actually takes, not with the limit it was given.

    Jacobians and the linear algebra are deliberately not counted. They differ
    per estimator too, but including them would mean guessing at BLAS internals
    and would trade an exact number for a fabricated one.

    Args:
        n_steps: Number of filter steps in the run.
        n_states: State dimension, which sets the UKF's sigma-point count.
        n_particles: Particle count used by the PF.
        fgo_iterations: Gauss-Newton iterations the FGO actually took.

    Returns:
        Dict mapping estimator name to its model-evaluation count (int).
    """
    n_sigma = 2 * n_states + 1

    return {
        "EKF": n_steps * 2,
        "UKF": n_steps * 2 * n_sigma,
        "PF": n_steps * 2 * n_particles,
        "FGO": fgo_iterations * 2 * n_steps,
    }


def main():
    """Run comparison of all estimators."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    overall_start = time.time()

    print("=" * 70)
    print("CHAPTER 3: COMPARISON OF STATE ESTIMATORS")
    print("=" * 70)
    print("\nScenario: 2D tracking with range measurements from 4 anchors")

    # Set up scenario
    dt, n_steps, anchors, true_states, measurements, Q, range_std = setup_scenario()

    print("\nParameters:")
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
    results["EKF"], results["EKF_time"] = run_ekf(
        dt, n_steps, anchors, measurements, Q, range_std
    )

    print("\n[2/4] Unscented Kalman Filter (UKF)")
    results["UKF"], results["UKF_time"] = run_ukf(
        dt, n_steps, anchors, measurements, Q, range_std
    )

    print("\n[3/4] Particle Filter (PF)")
    results["PF"], results["PF_time"], n_particles = run_pf(
        dt, n_steps, anchors, measurements, Q, range_std
    )

    print("\n[4/4] Factor Graph Optimization (FGO)")
    results["FGO"], results["FGO_time"], fgo_iterations = run_fgo(
        dt, n_steps, anchors, measurements, Q, range_std
    )

    results["model_evaluations"] = model_evaluation_counts(
        n_steps=n_steps,
        n_states=true_states.shape[1],
        n_particles=n_particles,
        fgo_iterations=fgo_iterations,
    )

    # Compute errors
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method in ["EKF", "UKF", "PF", "FGO"]:
        estimates = results[method]
        position_errors = np.linalg.norm(estimates[:, :2] - true_states[:, :2], axis=1)
        rmse = np.sqrt(np.mean(position_errors**2))
        mean_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        comp_time = results[f"{method}_time"]

        print(f"\n{method}:")
        print(f"  RMSE: {rmse:.4f} m")
        print(f"  Mean error: {mean_error:.4f} m")
        print(f"  Max error: {max_error:.4f} m")
        print(f"  Computation time: {comp_time:.4f} s")

    # One realisation cannot separate an ordering from a coincidence, and
    # these four are close enough that it matters: on this seed PF (0.4445)
    # beats EKF (0.4716) and UKF beats EKF by 0.25%. Neither survives
    # repetition. Repeating the whole scenario is the only honest way to say
    # which differences are real.
    n_seeds = 8
    print("\n" + "-" * 70)
    print(f"Same comparison repeated over {n_seeds} scenario seeds")
    print("-" * 70)

    # Each run prints a header and four progress bars; 32 of those would bury
    # the table they exist to produce.
    repeated = {name: [] for name in ("EKF", "UKF", "PF", "FGO")}
    quiet = io.StringIO()
    # setup_scenario reseeds the global RNG and the particle filter draws from
    # it, so this loop would otherwise leave the generator somewhere else and
    # change the figure plotted below -- which it did, until this was added.
    rng_state = np.random.get_state()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        for seed in range(n_seeds):
            scenario = setup_scenario(seed=seed)
            dt_r, n_r, anchors_r, truth_r, meas_r, Q_r, rstd_r = scenario
            for name, runner in (
                ("EKF", run_ekf),
                ("UKF", run_ukf),
                ("PF", run_pf),
                ("FGO", run_fgo),
            ):
                # Index rather than unpack: run_pf returns a third value
                # (n_particles) that the others do not, and only the estimates
                # are wanted here.
                est = runner(dt_r, n_r, anchors_r, meas_r, Q_r, rstd_r)[0]
                errors = np.linalg.norm(est[:, :2] - truth_r[:, :2], axis=1)
                repeated[name].append(float(np.sqrt(np.mean(errors**2))))
    np.random.set_state(rng_state)

    best_counts = {name: 0 for name in repeated}
    for i in range(n_seeds):
        best_counts[min(repeated, key=lambda k: repeated[k][i])] += 1

    print(f"{'Method':<8} {'mean RMSE':<12} {'min':<9} {'max':<9} {'best of 4':<10}")
    for name, values in repeated.items():
        values = np.asarray(values)
        print(
            f"{name:<8} {values.mean():<12.3f} {values.min():<9.3f} "
            f"{values.max():<9.3f} {best_counts[name]}/{n_seeds}"
        )

    print()
    print("  FGO wins every seed, which is what batch smoothing should do: it")
    print("  uses every measurement to estimate every state, while the filters")
    print("  only ever look backwards. The three filters are within a couple of")
    print("  percent of each other and their ordering changes with the seed, so")
    print("  the single run above should not be read as ranking them.")

    # Visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    print("Generating plots (this may take a moment)...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panels 1-3 are the shared primitives drawn into this grid; only panel 4
    # (timing bars) is specific to this comparison.
    methods = ["EKF", "UKF", "PF", "FGO"]
    trajectories = {m: results[m][:, :2] for m in methods}
    errors = {m: results[m][:, :2] - true_states[:, :2] for m in methods}
    time_steps = np.arange(n_steps + 1) * dt

    # Plot 1: Trajectories
    plot_trajectory_2d(
        true_states[:, :2],
        trajectories,
        anchors_xy=anchors[:, :2],
        title="Trajectory Comparison",
        axis_labels=("X Position [m]", "Y Position [m]"),
        ax=axes[0, 0],
    )

    # Plot 2: Position Errors vs Time
    plot_error_magnitude_time(
        errors, t=time_steps, title="Position Error vs Time", ax=axes[0, 1]
    )

    # Plot 3: CDF of Errors
    plot_error_cdf(errors, title="Cumulative Distribution of Errors", ax=axes[1, 0])

    # Plot 4: Computational cost, counted rather than timed
    #
    # This panel used to plot measured seconds, which made the committed figure
    # churn on every regeneration and told a reader nothing about their own
    # machine. Counting model evaluations is exact and reproducible, and it is
    # the quantity that actually separates these estimators.
    ax = axes[1, 1]
    methods = ["EKF", "UKF", "PF", "FGO"]
    evaluations = [results["model_evaluations"][m] for m in methods]
    colors = ["b", "g", "m", "r"]
    bars = ax.bar(methods, evaluations, color=colors, alpha=0.7, edgecolor="black")

    ax.set_yscale("log")
    ax.set_ylabel("Model evaluations", fontsize=12)
    ax.set_title("Computational Cost", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, n_evals in zip(bars, evaluations, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{n_evals:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    paths = save_figure(fig, Path(__file__).parent / "figs", "ch3_estimator_comparison")
    print(f"[OK] Plot saved as: {paths[0]}")
    show_figures_if_requested()

    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)
    print(
        f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.1f} minutes)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
