"""Tuning and Robust Loss Functions for Chapter 8.

Runs four measurement-handling strategies -- no gating, chi-square gating
(Eq. 8.9), Huber and Cauchy covariance inflation (Eq. 8.7) -- across three
scenarios, because a robust loss can only be judged against the outlier
distribution it was designed for:

  1. LOS          the clean dataset. Nothing to reject, so the question is
                  what robustness *costs* when it is not needed.
  2. Sporadic     the clean dataset with a +3 m bias added to a random 5% of
                  ranges. An inlier majority at almost every epoch, which is
                  the assumption every M-estimator is built on.
  3. NLOS         the shipped NLOS dataset: 2 of 4 anchors biased +0.8 m for
                  the whole run. No inlier majority in 2D, and the section
                  below says what that costs.

The losses take a NORMALIZED residual, and the scale matters
--------------------------------------------------------------
`huber_R_scale` and `cauchy_R_scale` take r = innovation / sigma, not an
innovation in metres. Handed metres directly, with the delta = 1.5 this demo
used to pass, Huber fired **once in 2271 updates** and its largest inflation
was 1.020: the results table reported "Huber 0.722" against "Baseline 0.722"
and the reader was invited to conclude that robustness did not help, when in
fact nothing had happened at all.

The obvious repair -- divide by sqrt(S), the innovation std, which is what a
normalized residual usually means -- makes every configuration worse than
baseline, on clean data included (0.190 m becomes 1.043 m at delta = 5). S
tracks the filter's own confidence, and this filter is over-confident: on the
clean dataset sqrt(S) averages 0.0524 m against sqrt(R) = 0.0500, so H P H'
contributes almost nothing and the normalized residual inherits the filter's
collapsed P. Inflating R then weakens the correction, the state drifts, the
next residual is larger, and the loop feeds itself.

So the residual is normalized by **sqrt(R)**, a fixed scale. It cannot chase
the filter's confidence, so it cannot form that loop, and a threshold is then
in units of the measurement noise -- something a sensor datasheet can supply.

Why delta = 20 sigma and not the textbook 1.345
-----------------------------------------------
1.345 is the value for 95% efficiency on *Gaussian* residuals, and these are
not Gaussian. Measured on the clean dataset, with no outlier anywhere in it,
|innovation| / sigma_R has median 0.90, 90th percentile 3.53, 99th percentile
14.51 and maximum 17.23; 32.4% of measurements already lie beyond 1.345. A
standard threshold therefore fires on a third of a clean run and destroys it:
delta = 1.345 scores 2.591 m against a 0.190 m baseline.

delta = 20 sigma is chosen to sit just above that measured maximum, so the
dead zone covers every clean-data residual and Huber is exactly neutral when
there is nothing to reject. The choice is a plateau rather than a knife edge:
15 / 20 / 25 / 30 give -40.6% / -37.9% / -34.1% / -29.7% on the sporadic
scenario, and 15 is the only one that costs anything on clean data (+1.0%).

Cauchy has no dead zone -- w_R = 1 + (r/c)^2 exceeds 1 for every residual --
so it cannot be made neutral, only cheap. c = 50 sigma holds the inflation at
the clean 99th percentile to 1.084, and costs 2.9% on the clean case.

What robust weighting cannot do
-------------------------------
An M-estimator down-weights the minority that disagrees with the majority. On
the NLOS dataset the "minority" is half the anchors, biased in a fixed
direction for the entire run: measured per anchor, the mean range residual is
+0.001, +0.798, +0.799, +0.001 m. With four anchors in 2D, two consistent
biased ranges define a position as firmly as the two honest ones, so there is
no majority to side with and no residual pattern to key on. Huber recovers
2.0% here and Cauchy 1.1%, and that is the truthful result rather than a
disappointing one. Persistent bias needs a method that can *represent* it:
state augmentation with a per-anchor bias term, or NLOS identification from
signal features, not a reweighting of the residual.

Chi-square gating collapses, and NLOS is not why
------------------------------------------------
Gating scores 24-26 m in all three scenarios, including the clean one, where
the ungated NIS median is 0.74 against 0.45 for a consistent filter and 81% of
samples already sit inside the 3.84 gate. So the collapse is not caused by the
NLOS bias -- it is the feedback: a 95% gate keyed to a Gaussian assumption
rejects roughly four times too many of these heavy-tailed innovations, the
starved state drifts, drift inflates the next innovation, and the rejection
rate runs away to 67-81%. A hard gate inherits every error in the covariance
it tests against; a robust loss scales an outlier's influence down instead of
removing it, and survives the same mis-specified R.

Reading the results table
-------------------------
Both RMSE and median are reported, and they disagree on purpose. RMSE here is
dominated by the transient after the 57 deg/s turn at t = 52-54 s: for both
robust methods 100% of the worst 1% of samples fall in that window. A robust
loss inflates R there too, because a manoeuvre the process model does not
predict looks exactly like an outlier -- so the RMSE gain is smaller than the
median gain and less stable across outlier draws. Measured over five draws of
the sporadic scenario the median error improves by 57-62% every time, while
the RMSE gain ranges from +38% to -44%. The median is the statistic that
describes the method; the RMSE describes the trajectory.

Key Equations:
- Eq. (8.5): Innovation y_k = z_k - h(x_k|k-1)
- Eq. (8.6): Innovation covariance S_k = H_k P_k|k-1 H_k^T + R_k
- Eq. (8.7): Robust R scaling: R_k <- w_R(r_k) * R_k, r_k = y_k / sqrt(R_k)
- Eq. (8.9): Chi-square gating: accept if d^2 < chi2(m, alpha)

Author: Li-Ta Hsu
References: Chapter 8, Section 8.3 (Tuning of Sensor Fusion)
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.estimators import ExtendedKalmanFilter
from core.eval import (
    compute_rmse,
    plot_trajectory_2d,
    save_figure,
    show_figures_if_requested,
)
from core.fusion import (
    SENSOR_IMU,
    SENSOR_UWB_RANGE,
    cauchy_R_scale,
    chi_square_gate,
    huber_R_scale,
    innovation,
    innovation_covariance,
    load_fusion_dataset,
    mahalanobis_distance_squared,
)
from core.fusion.tc_models import (
    tc_process_jacobian,
    tc_process_model,
    tc_process_noise_covariance,
    tc_uwb_measurement_jacobian,
    tc_uwb_measurement_model,
)

#: Range noise the filter is told to expect, and the scale the robust
#: residual is normalized by. The datasets record 0.05 m.
ASSUMED_RANGE_NOISE_STD = 0.05

#: Huber threshold, in units of sigma_R. Twenty, not the textbook 1.345,
#: because the clean-data residual reaches 17.23 sigma with no outlier in the
#: data at all -- see the module docstring, which carries the measurement.
HUBER_DELTA_SIGMA = 20.0

#: Cauchy scale, in units of sigma_R. Cauchy inflates every residual, so this
#: is chosen to keep the inflation at the clean 99th percentile (14.51 sigma)
#: down to 1 + (14.51 / 50)^2 = 1.084 rather than to sit above a maximum.
CAUCHY_C_SIGMA = 50.0

#: The sporadic-outlier scenario: a fixed bias on a random fraction of ranges.
#: Large enough to matter against 0.05 m of noise, rare enough that the
#: inlier-majority assumption every M-estimator rests on actually holds.
SPORADIC_OUTLIER_BIAS_M = 3.0
SPORADIC_OUTLIER_RATE = 0.05

#: Strategy -> default robust threshold. One default cannot serve both losses:
#: Huber's delta is a dead-zone edge and Cauchy's c is a curvature scale, so
#: the same number means different things to them (Cauchy at 20 sigma scores
#: 2.484 m on the clean data that Huber at 20 sigma leaves untouched).
DEFAULT_ROBUST_THRESHOLD = {"huber": HUBER_DELTA_SIGMA, "cauchy": CAUCHY_C_SIGMA}


def make_sporadic_outlier_dataset(
    dataset: dict,
    bias_m: float = SPORADIC_OUTLIER_BIAS_M,
    rate: float = SPORADIC_OUTLIER_RATE,
    seed: int = 1,
) -> dict:
    """Copy a dataset and add a fixed bias to a random fraction of its ranges.

    The shipped NLOS dataset biases two of four anchors for the whole run,
    which leaves no inlier majority for an M-estimator to side with. This
    builds the case M-estimation is actually designed for: occasional gross
    errors against a clean background.

    Args:
        dataset: A loaded fusion dataset. Not modified.
        bias_m: Bias added to each selected range, in metres.
        rate: Fraction of range measurements to corrupt.
        seed: Seed for the outlier placement.

    Returns:
        A deep copy with corrupted ranges and an ``n_outliers`` count added.
    """
    corrupted = copy.deepcopy(dataset)
    ranges = np.array(corrupted["uwb"]["ranges"], dtype=float)

    rng = np.random.default_rng(seed)
    selected = (rng.random(ranges.shape) < rate) & np.isfinite(ranges)

    corrupted["uwb"]["ranges"] = np.where(selected, ranges + bias_m, ranges)
    corrupted["n_outliers"] = int(np.sum(selected))
    return corrupted


def run_fusion_with_strategy(
    dataset: dict,
    strategy: str = "baseline",
    R_scale: float = 1.0,
    use_gating: bool = False,
    gate_confidence: float = 0.95,
    robust_threshold: float | None = None,
    verbose: bool = False,
) -> dict:
    """Run TC fusion with different tuning/robust strategies.

    Args:
        dataset: Dataset dictionary
        strategy: One of 'baseline', 'gating', 'huber', 'cauchy'
        R_scale: Scale factor for R (e.g., 0.5 = under-estimate, 2.0 = over-estimate)
        use_gating: Enable chi-square gating
        gate_confidence: Gating confidence level (default 0.95 for 95% confidence)
        robust_threshold: Huber delta or Cauchy c, **in units of sigma_R**, since
            that is what the loss is handed. Defaults per strategy from
            DEFAULT_ROBUST_THRESHOLD; see the module docstring for why 20 and
            50 rather than the Gaussian-efficiency values 1.345 and 2.385.
        verbose: Print progress

    Returns:
        Results dictionary
    """
    if robust_threshold is None:
        robust_threshold = DEFAULT_ROBUST_THRESHOLD.get(strategy, HUBER_DELTA_SIGMA)

    if verbose:
        print(f"\nRunning fusion with strategy: {strategy.upper()}")
        print(f"  R scale: {R_scale}")
        print(f"  Gating: {'Enabled' if use_gating else 'Disabled'}")
        if strategy in ["huber", "cauchy"]:
            print(f"  Robust threshold: {robust_threshold} sigma_R")

    truth = dataset["truth"]
    imu = dataset["imu"]
    uwb = dataset["uwb"]
    anchors = dataset["uwb_anchors"]

    # Initial state: [px, py, vx, vy, yaw] (follows StateIndex convention)
    x0 = np.array(
        [
            truth["p_xy"][0, 0],  # px
            truth["p_xy"][0, 1],  # py
            truth["v_xy"][0, 0],  # vx
            truth["v_xy"][0, 1],  # vy
            truth["yaw"][0],  # yaw
        ]
    )

    # P0: covariances for [px, py, vx, vy, yaw]
    P0 = np.diag([0.1, 0.1, 0.5, 0.5, 0.1]) ** 2

    # Process noise
    accel_noise_std = 0.1
    gyro_noise_std = 0.01

    # Measurement noise (scaled)
    uwb_range_noise_std = ASSUMED_RANGE_NOISE_STD * R_scale

    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        process_model=tc_process_model,
        process_jacobian=tc_process_jacobian,
        measurement_model=lambda x: tc_uwb_measurement_model(x, anchors),
        measurement_jacobian=lambda x: tc_uwb_measurement_jacobian(x, anchors),
        Q=lambda dt: tc_process_noise_covariance(dt, accel_noise_std, gyro_noise_std),
        R=lambda: np.eye(4) * uwb_range_noise_std**2,
        x0=x0,
        P0=P0,
    )

    # Prepare measurements
    from core.fusion import StampedMeasurement

    measurements: list[StampedMeasurement] = []

    # Add IMU
    for i in range(len(imu["t"])):
        measurements.append(
            StampedMeasurement(
                t=imu["t"][i],
                sensor=SENSOR_IMU,
                z=np.hstack([imu["accel_xy"][i], imu["gyro_z"][i]]),
                R=np.eye(3),
                meta={},
            )
        )

    # Add UWB (per anchor)
    for i in range(len(uwb["t"])):
        for j in range(anchors.shape[0]):
            if not np.isnan(uwb["ranges"][i, j]):
                measurements.append(
                    StampedMeasurement(
                        t=uwb["t"][i],
                        sensor=SENSOR_UWB_RANGE,
                        z=np.array([uwb["ranges"][i, j]]),
                        R=np.array([[uwb_range_noise_std**2]]),
                        meta={"anchor_id": j, "anchor_pos": anchors[j]},
                    )
                )

    # Sort by timestamp
    measurements.sort(key=lambda m: m.t)

    # Run fusion
    accepted_history = []
    history = {
        "t": [],
        "x_est": [],
        "P_trace": [],
        "innovations": [],
        "normalized_residuals": [],  # innovation / sqrt(R), what the loss sees
        "nis": [],
        "measurement_accepted": accepted_history,
        "gated": accepted_history,  # Backward-compatible alias.
        "robust_scales": [],  # Renamed from robust_weights for clarity
    }

    n_uwb_accepted = 0
    n_uwb_rejected = 0
    t_prev = measurements[0].t

    for meas in measurements:
        dt = meas.t - t_prev

        if meas.sensor == SENSOR_IMU:
            # Propagate
            u = meas.z
            ekf.predict(u=u, dt=dt)

        elif meas.sensor in {SENSOR_UWB_RANGE, "uwb"}:
            # UWB range update
            anchor_pos = meas.meta["anchor_pos"]

            # Predict range to this anchor
            state_pos = ekf.state[:2]
            z_pred = np.array([np.linalg.norm(state_pos - anchor_pos)])

            # Innovation
            y = innovation(meas.z, z_pred)

            # Jacobian for this anchor
            H_single = tc_uwb_measurement_jacobian(ekf.state, np.array([anchor_pos]))

            # Base R
            R_base = np.array([[uwb_range_noise_std**2]])

            # The robust losses take a *normalized* residual. Normalizing by
            # sqrt(R) rather than by sqrt(S) is the whole point: sqrt(R) is
            # fixed, so it cannot chase the filter's own collapsing
            # confidence, and the thresholds above are then in sensor units.
            r_normalized = y[0] / uwb_range_noise_std

            # Apply robust covariance inflation if requested (Eq. 8.7)
            # Outliers get INFLATED covariance (w_R >= 1)
            w_R = 1.0
            if strategy == "huber":
                w_R = huber_R_scale(r_normalized, delta=robust_threshold)
                R_robust = w_R * R_base  # Eq. 8.7: R <- w_R * R
            elif strategy == "cauchy":
                w_R = cauchy_R_scale(r_normalized, c=robust_threshold)
                R_robust = w_R * R_base  # Eq. 8.7: R <- w_R * R
            else:
                R_robust = R_base

            # Innovation covariance
            S = innovation_covariance(H_single, ekf.covariance, R_robust)

            # Gating
            accept = True
            if use_gating or strategy == "gating":
                accept = chi_square_gate(y, S, confidence=gate_confidence)

            if accept:
                # Perform update
                K = ekf.covariance @ H_single.T @ np.linalg.inv(S)
                ekf.state = ekf.state + (K @ y).flatten()
                ekf.covariance = (np.eye(5) - K @ H_single) @ ekf.covariance
                n_uwb_accepted += 1
            else:
                n_uwb_rejected += 1

            # Log
            history["innovations"].append(float(np.abs(y[0])))
            history["normalized_residuals"].append(float(np.abs(r_normalized)))
            history["nis"].append(mahalanobis_distance_squared(y, S))
            history["measurement_accepted"].append(accept)
            history["robust_scales"].append(w_R)

        # Record state
        history["t"].append(meas.t)
        history["x_est"].append(ekf.state.copy())
        history["P_trace"].append(np.trace(ekf.covariance))

        t_prev = meas.t

    # Convert to arrays
    history["t"] = np.array(history["t"])
    history["x_est"] = np.array(history["x_est"])
    history["P_trace"] = np.array(history["P_trace"])
    history["n_uwb_accepted"] = n_uwb_accepted
    history["n_uwb_rejected"] = n_uwb_rejected

    if verbose:
        print(f"  Accepted: {n_uwb_accepted}")
        print(f"  Rejected: {n_uwb_rejected}")
        print(
            f"  Acceptance rate: {100*n_uwb_accepted/(n_uwb_accepted+n_uwb_rejected):.1f}%"
        )

    return history


def position_errors(history: dict, truth: dict) -> np.ndarray:
    """Horizontal position error at every logged step.

    Norm first, then RMS downstream. Handing the (N, 2) error vectors to
    ``compute_rmse`` averages over 2N components instead of N positions, which
    is the per-axis RMS and is smaller by exactly sqrt(2) -- this table used to
    report 12.74 m for a gating run the figure's own bar chart labelled
    18.02 m.
    """
    p_true = np.column_stack(
        [
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 0]),
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 1]),
        ]
    )
    return np.linalg.norm(history["x_est"][:, :2] - p_true, axis=1)


def per_anchor_range_bias(dataset: dict) -> np.ndarray:
    """Mean signed range residual per anchor, against the ground truth.

    This is the statistic that separates the two outlier scenarios: sporadic
    corruption leaves every anchor near zero on average, while a persistent
    NLOS bias parks half of them at +0.8 m. A signed mean is right here
    precisely because the defect being looked for is a systematic offset.
    """
    truth = dataset["truth"]
    anchors = np.asarray(dataset["uwb_anchors"])
    idx = np.searchsorted(truth["t"], dataset["uwb"]["t"]).clip(0, len(truth["t"]) - 1)
    true_range = np.linalg.norm(
        truth["p_xy"][idx][:, None, :] - anchors[None, :, :2], axis=2
    )
    residual = np.asarray(dataset["uwb"]["ranges"]) - true_range
    return np.array(
        [float(np.nanmean(residual[:, j])) for j in range(anchors.shape[0])]
    )


def turn_windows(truth: dict, rate_deg_s: float = 20.0) -> list[tuple]:
    """Time spans where the ground truth turns faster than ``rate_deg_s``.

    The RMSE in this chapter is dominated by the transient after each corner,
    so the figure marks them rather than leaving the reader to guess which
    excursions are the sensor and which are the trajectory.
    """
    t = np.asarray(truth["t"])
    yaw_rate = np.degrees(np.diff(np.unwrap(np.asarray(truth["yaw"]))) / np.diff(t))
    turning = np.abs(yaw_rate) > rate_deg_s
    windows = []
    start = None
    for i, is_turn in enumerate(turning):
        if is_turn and start is None:
            start = i
        elif not is_turn and start is not None:
            windows.append((float(t[start]), float(t[i])))
            start = None
    if start is not None:
        windows.append((float(t[start]), float(t[-1])))
    return windows


def plot_robust_comparison(
    scenarios: dict[str, dict],
    results: dict[str, dict[str, dict]],
    save_path: str = None,
) -> None:
    """Nine panels: what each strategy does, and why the thresholds are what they are.

    Args:
        scenarios: name -> dataset dictionary, in display order.
        results: scenario name -> strategy name -> fusion history.
        save_path: Path to save figure.
    """
    method_colors = {
        "baseline": "tab:red",
        "gating": "tab:blue",
        "huber": "tab:orange",
        "cauchy": "tab:green",
    }
    method_labels = {
        "baseline": "Baseline",
        "gating": "Chi-square gating",
        "huber": f"Huber (delta={HUBER_DELTA_SIGMA:.0f}$\\sigma$)",
        "cauchy": f"Cauchy (c={CAUCHY_C_SIGMA:.0f}$\\sigma$)",
    }
    scenario_colors = {
        "LOS": "tab:green",
        "Sporadic": "tab:purple",
        "NLOS": "tab:brown",
    }
    methods = ["baseline", "gating", "huber", "cauchy"]
    names = list(scenarios)

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.26)

    # --- Row 0: one trajectory panel per scenario -------------------------
    for col, name in enumerate(names):
        dataset = scenarios[name]
        truth = dataset["truth"]
        ax = fig.add_subplot(gs[0, col])
        estimates = {
            method_labels[m]: results[name][m]["x_est"][:, :2]
            for m in ("baseline", "huber", "cauchy")
        }
        plot_trajectory_2d(
            truth["p_xy"],
            estimates,
            anchors_xy=np.asarray(dataset["uwb_anchors"])[:, :2],
            title=f"{name}: median error "
            f"{np.median(position_errors(results[name]['baseline'], truth)):.3f} m "
            f"-> {np.median(position_errors(results[name]['huber'], truth)):.3f} m",
            axis_labels=("East [m]", "North [m]"),
            ax=ax,
            title_fontweight="normal",
        )
        ax.legend(fontsize=7, loc="best")
        # Gating is deliberately not drawn here. It wanders ~25 m from a 20 m
        # building, so equal axes would put every other trace inside one
        # stroke width -- but an absent trace reads as an untested method, so
        # say where it went rather than letting the panel imply three.
        ax.text(
            0.5,
            0.02,
            "gating omitted: off-frame (see RMSE panel)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            style="italic",
            color="tab:blue",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )

    # --- (1, 0) RMSE and median, every scenario x every method -----------
    ax = fig.add_subplot(gs[1, 0])
    width = 0.2
    x = np.arange(len(names))
    for k, method in enumerate(methods):
        rmses = [
            compute_rmse(position_errors(results[n][method], scenarios[n]["truth"]))
            for n in names
        ]
        medians = [
            float(np.median(position_errors(results[n][method], scenarios[n]["truth"])))
            for n in names
        ]
        offset = (k - 1.5) * width
        ax.bar(
            x + offset,
            rmses,
            width,
            color=method_colors[method],
            alpha=0.75,
            label=method_labels[method],
        )
        # The median sits inside its own RMSE bar, so the gap between the two
        # is visible per method -- that gap is the post-corner transient.
        ax.plot(
            x + offset,
            medians,
            "_",
            color="black",
            markersize=9,
            markeredgewidth=1.6,
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Position error [m], log scale")
    ax.set_title("Bar = RMSE, black tick = median")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1, 1) the residual tail the thresholds have to clear ------------
    ax = fig.add_subplot(gs[1, 1])
    for name in names:
        r = np.sort(np.asarray(results[name]["baseline"]["normalized_residuals"]))
        survival = 1.0 - np.arange(len(r)) / len(r)
        ax.loglog(
            np.maximum(r, 1e-3),
            survival,
            color=scenario_colors[name],
            linewidth=1.4,
            label=name,
        )
    for value, style, text in (
        (1.345, ":", "textbook 1.345"),
        (HUBER_DELTA_SIGMA, "--", f"Huber delta = {HUBER_DELTA_SIGMA:.0f}"),
        (CAUCHY_C_SIGMA, "-.", f"Cauchy c = {CAUCHY_C_SIGMA:.0f}"),
    ):
        ax.axvline(value, color="0.35", linestyle=style, linewidth=1.2, label=text)
    ax.set_xlabel(r"$|r| = |y| / \sigma_R$")
    ax.set_ylabel(r"fraction of updates above $|r|$")
    ax.set_title("Why 1.345 destroys this filter: the tail is not Gaussian")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    # --- (1, 2) the two loss shapes, on the same axis ---------------------
    ax = fig.add_subplot(gs[1, 2])
    r_grid = np.linspace(0.0, 80.0, 800)
    ax.plot(
        r_grid,
        [huber_R_scale(v, delta=1.345) for v in r_grid],
        color="0.35",
        linestyle=":",
        linewidth=1.4,
        label="Huber, delta = 1.345 (textbook)",
    )
    ax.plot(
        r_grid,
        [huber_R_scale(v, delta=HUBER_DELTA_SIGMA) for v in r_grid],
        color=method_colors["huber"],
        linewidth=1.8,
        label=method_labels["huber"],
    )
    ax.plot(
        r_grid,
        [cauchy_R_scale(v, c=CAUCHY_C_SIGMA) for v in r_grid],
        color=method_colors["cauchy"],
        linewidth=1.8,
        label=method_labels["cauchy"],
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvspan(
        0.0,
        HUBER_DELTA_SIGMA,
        color=method_colors["huber"],
        alpha=0.08,
        label="Huber dead zone (no inflation)",
    )
    # The difference between the two at an ordinary clean residual is real and
    # far too small to see on a decade axis (1.00 against 1.08), so state it.
    # It is the reason Cauchy costs a few percent on LOS and Huber costs zero.
    clean_p99 = float(
        np.percentile(results["LOS"]["baseline"]["normalized_residuals"], 99)
    )
    ax.axvline(clean_p99, color=scenario_colors["LOS"], linewidth=1.2, alpha=0.8)
    ax.annotate(
        f"clean 99th pct = {clean_p99:.1f}$\\sigma$\n"
        f"Huber {huber_R_scale(clean_p99, delta=HUBER_DELTA_SIGMA):.2f}, "
        f"Cauchy {cauchy_R_scale(clean_p99, c=CAUCHY_C_SIGMA):.2f}, "
        f"textbook {huber_R_scale(clean_p99, delta=1.345):.1f}",
        xy=(clean_p99, 1.0),
        xytext=(clean_p99 + 4, 1.6),
        fontsize=7,
        color="0.2",
        arrowprops={"arrowstyle": "->", "color": "0.4", "linewidth": 0.8},
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$|r| = |y| / \sigma_R$")
    ax.set_ylabel(r"R inflation $w_R$ (log)")
    ax.set_title("Eq. (8.7) scale factors: the textbook delta inflates a clean run")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- (2, 0) error against time on the scenario robustness can fix -----
    ax = fig.add_subplot(gs[2, 0])
    sporadic = scenarios["Sporadic"]
    for method in ("baseline", "huber", "cauchy"):
        history = results["Sporadic"][method]
        ax.semilogy(
            history["t"],
            np.maximum(position_errors(history, sporadic["truth"]), 1e-4),
            color=method_colors[method],
            linewidth=0.9,
            alpha=0.8,
            label=method_labels[method],
        )
    for start, end in turn_windows(sporadic["truth"]):
        ax.axvspan(start, end, color="0.6", alpha=0.25)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m], log scale")
    ax.set_title("Sporadic outliers: the win is steady state, not the turns (shaded)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- (2, 1) NIS, and the gate that cannot survive it ------------------
    ax = fig.add_subplot(gs[2, 1])
    from core.fusion import chi_square_threshold

    threshold = chi_square_threshold(dof=1, confidence=0.95)
    for name in names:
        nis = np.asarray(results[name]["baseline"]["nis"])
        step = max(1, len(nis) // 400)
        ax.semilogy(
            np.maximum(nis[::step], 1e-3),
            ".",
            color=scenario_colors[name],
            markersize=1.6,
            alpha=0.5,
            label=f"{name} (median {np.median(nis):.2f})",
        )
    ax.axhline(
        threshold,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"95% bound (chi^2={threshold:.2f})",
    )
    ax.axhline(
        0.4549,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="median if consistent (0.45)",
    )
    ax.set_xlabel("UWB update index (subsampled)")
    ax.set_ylabel("NIS, 1 DOF (log)")
    ax.set_title("Ungated NIS: even LOS sits above a consistent filter")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- (2, 2) the panel that says why NLOS is not fixable here ----------
    ax = fig.add_subplot(gs[2, 2])
    n_anchors = len(per_anchor_range_bias(scenarios[names[0]]))
    x = np.arange(n_anchors)
    width = 0.26
    for k, name in enumerate(names):
        biases = per_anchor_range_bias(scenarios[name])
        ax.bar(
            x + (k - 1) * width,
            biases,
            width,
            color=scenario_colors[name],
            alpha=0.8,
            label=name,
        )
        # A 0.001 m bar beside a 0.8 m one has no height at all, and an
        # invisible bar reads as a missing series rather than as a small
        # number -- the same failure as a diverged method plotting as zero.
        # symlog keeps both visible; the labels keep both readable.
        for anchor, bias in enumerate(biases):
            ax.text(
                anchor + (k - 1) * width,
                bias,
                f"{bias:+.3f}",
                ha="center",
                va="bottom" if bias >= 0 else "top",
                fontsize=5.5,
                rotation=90,
                color="0.2",
            )
    ax.set_yscale("symlog", linthresh=1e-3)
    # Headroom for the rotated value labels, which otherwise run into the title.
    ax.set_ylim(-0.02, 8.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"anchor {j}" for j in range(n_anchors)])
    ax.set_ylabel("Mean signed range residual [m], symlog")
    ax.set_title("Half the NLOS anchors are biased all run: no inlier majority")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Robust Loss Functions Against Three Outlier Distributions (Eq. 8.7)",
        fontsize=16,
        fontweight="bold",
    )

    if save_path:
        # save_figure takes a directory and a stem, and writes svg/pdf/png
        # together; callers still pass a single path, so split it here.
        save_path = Path(save_path)
        written = save_figure(fig, save_path.parent, save_path.stem)
        print(f"\nSaved figure: {written[0]}")

    show_figures_if_requested()


def main():
    """Main entry point for tuning and robust loss demo."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb_nlos",
        help="Path to the persistent-NLOS dataset directory",
    )
    parser.add_argument(
        "--los-data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb",
        help="Path to the clean LOS dataset the sporadic scenario is built from",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save results figure"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed placing the sporadic outliers",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 78)
    print("Tuning and Robust Loss Demo (Chapter 8)")
    print("=" * 78)
    print("\nStrategies (Eq. 8.7: R_k <- w_R(r_k) * R_k, r_k = y_k / sqrt(R_k)):")
    print("  1. Baseline:          accept every measurement unchanged")
    print("  2. Chi-square gating: hard rejection on Mahalanobis distance (Eq. 8.9)")
    print(
        f"  3. Huber loss:        w_R = |r|/delta beyond delta = "
        f"{HUBER_DELTA_SIGMA:.0f} sigma_R, else 1"
    )
    print(
        f"  4. Cauchy loss:       w_R = 1 + (r/c)^2 with c = "
        f"{CAUCHY_C_SIGMA:.0f} sigma_R, always above 1"
    )
    print("\n  The losses take the innovation divided by sigma_R, not the innovation")
    print("  in metres. Run --help for why the thresholds are 20 and 50 rather")
    print("  than the textbook 1.345 and 2.385.")
    print("")

    print(f"Loading LOS dataset from:  {args.los_data}")
    los = load_fusion_dataset(args.los_data)
    print(f"Loading NLOS dataset from: {args.data}")
    nlos = load_fusion_dataset(args.data)
    sporadic = make_sporadic_outlier_dataset(los, seed=args.seed)

    print("\nScenarios:")
    print(
        f"  LOS       {len(los['uwb']['t'])} UWB epochs, {los['uwb']['ranges'].shape[1]} "
        f"anchors, range noise {los['config']['uwb']['range_noise_std_m']} m, no bias"
    )
    print(
        f"  Sporadic  LOS plus +{SPORADIC_OUTLIER_BIAS_M:.0f} m on "
        f"{sporadic['n_outliers']} of "
        f"{int(np.sum(np.isfinite(los['uwb']['ranges'])))} ranges "
        f"({100*SPORADIC_OUTLIER_RATE:.0f}% target rate, seed {args.seed})"
    )
    print(
        f"  NLOS      anchors {nlos['config']['uwb']['nlos_anchors']} biased "
        f"+{nlos['config']['uwb']['nlos_bias_m']} m for the whole run"
    )
    print("")

    scenarios = {"LOS": los, "Sporadic": sporadic, "NLOS": nlos}
    methods = ["baseline", "gating", "huber", "cauchy"]

    results: dict[str, dict[str, dict]] = {}
    for name, dataset in scenarios.items():
        results[name] = {}
        for method in methods:
            print(f"[{name}] running {method}...")
            results[name][method] = run_fusion_with_strategy(
                dataset,
                strategy=method,
                use_gating=(method == "gating"),
            )

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 78)
    print("Results Summary")
    print("=" * 78)
    print(
        f"{'Scenario':<10} {'Method':<18} {'RMSE [m]':>10} {'Median [m]':>11} "
        f"{'Accepted':>9} {'Rejected':>9}"
    )
    print("-" * 78)

    errors = {
        name: {
            method: position_errors(results[name][method], scenarios[name]["truth"])
            for method in methods
        }
        for name in scenarios
    }
    labels = {
        "baseline": "Baseline",
        "gating": "Chi-square gating",
        "huber": "Huber loss",
        "cauchy": "Cauchy loss",
    }
    for name in scenarios:
        for method in methods:
            history = results[name][method]
            print(
                f"{name:<10} {labels[method]:<18} "
                f"{compute_rmse(errors[name][method]):>10.3f} "
                f"{np.median(errors[name][method]):>11.3f} "
                f"{history['n_uwb_accepted']:>9d} {history['n_uwb_rejected']:>9d}"
            )
        print("-" * 78)

    # ------------------------------------------------------------- findings
    def gain(name, method, statistic):
        base = statistic(errors[name]["baseline"])
        return 100.0 * (base - statistic(errors[name][method])) / base

    best = min(
        ("huber", "cauchy"),
        key=lambda m: float(np.median(errors["Sporadic"][m])),
    )

    print("\nKey Findings:")
    print(
        f"  * Sporadic outliers are what an M-estimator is for. Best method: "
        f"{labels[best]}, median"
    )
    print(
        f"    error {np.median(errors['Sporadic']['baseline']):.3f} m -> "
        f"{np.median(errors['Sporadic'][best]):.3f} m, "
        f"{gain('Sporadic', best, np.median):.0f}% better; RMSE "
        f"{gain('Sporadic', best, compute_rmse):.0f}% better."
    )
    print(
        f"  * Robustness is close to free when it is not needed. On LOS the "
        f"RMSE goes "
        f"{compute_rmse(errors['LOS']['baseline']):.3f} -> "
        f"{compute_rmse(errors['LOS']['huber']):.3f} m under Huber, whose dead "
        f"zone covers"
    )
    print(
        f"    every clean residual, and "
        f"{compute_rmse(errors['LOS']['baseline']):.3f} -> "
        f"{compute_rmse(errors['LOS']['cauchy']):.3f} m under Cauchy, which "
        f"has no dead zone and so cannot be exactly free."
    )
    print(
        f"  * Persistent NLOS is not an outlier problem. Huber recovers "
        f"{gain('NLOS', 'huber', compute_rmse):.1f}% and Cauchy "
        f"{gain('NLOS', 'cauchy', compute_rmse):.1f}%,"
    )
    bias = per_anchor_range_bias(nlos)
    print(
        "    and that is the correct answer rather than a disappointing one. "
        "Mean range residual per"
    )
    print(
        f"    anchor is {', '.join(f'{b:+.3f}' for b in bias)} m: with 4 anchors "
        f"in 2D, two consistently"
    )
    print(
        "    biased ranges fix a position as firmly as two honest ones, so "
        "there is no inlier"
    )
    print(
        "    majority to side with. Fix it by estimating a per-anchor bias "
        "(state augmentation)"
    )
    print("    or by identifying NLOS links, not by reweighting the residual.")
    print("")

    nis_los = np.asarray(results["LOS"]["baseline"]["nis"])
    for name in scenarios:
        gating_history = results[name]["gating"]
        total = gating_history["n_uwb_accepted"] + gating_history["n_uwb_rejected"]
        rate = 100.0 * gating_history["n_uwb_accepted"] / total
        print(
            f"  Chi-square gating on {name:<9} {compute_rmse(errors[name]['gating']):>7.2f} m"
            f"   accepted {rate:.0f}%"
        )
    print(
        f"    - It collapses on LOS data too, where the ungated NIS median is "
        f"{np.median(nis_los):.2f} against 0.45 for a"
    )
    print(
        f"      consistent filter and {100*np.mean(nis_los < 3.841):.0f}% of "
        f"samples already sit inside the 3.84 gate. So the"
    )
    print(
        "      NLOS bias is not the cause: a Gaussian gate over heavy-tailed "
        "innovations rejects several"
    )
    print(
        "      times too many, the starved state drifts, drift inflates the "
        "next innovation, and the"
    )
    print("      rejection rate runs away.")
    print(
        "    - A hard gate inherits every error in the covariance it tests "
        "against. A robust loss scales"
    )
    print(
        "      an outlier's influence down instead of removing it, and "
        "survives the same mis-specified R."
    )
    print("")
    print(
        "  Note RMSE and median disagree. The worst samples are the transient "
        "after the 57 deg/s turn"
    )
    print(
        "  at t = 52-54 s, where a manoeuvre the process model does not "
        "predict looks exactly like an"
    )
    print(
        "  outlier and the robust losses inflate R too. The median describes "
        "the method; the RMSE"
    )
    print("  describes the trajectory.")
    print("")

    # Plot
    save_path = (
        args.save if args.save else "ch8_sensor_fusion/figs/tuning_robust_demo.svg"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_robust_comparison(scenarios, results, save_path=save_path)


if __name__ == "__main__":
    main()
