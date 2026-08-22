"""Tightly Coupled IMU + UWB EKF Fusion Demo (Chapter 8).

Demonstrates tightly coupled fusion where raw UWB range measurements
are fused directly in the EKF, rather than first computing position fixes.

Features:
- High-rate IMU propagation (100 Hz)
- Low-rate UWB range updates (10 Hz per anchor)
- Chi-square innovation gating (Eq. 8.9)
- Innovation monitoring (Eqs. 8.5-8.6)
- Comparison with IMU-only dead reckoning

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1.2 (Tightly Coupled)
"""

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from core.eval import (
    compute_position_errors,
    compute_position_rmse,
    save_figure,
    show_figures_if_requested,
)
from core.fusion import load_fusion_dataset, run_tc_fusion


def evaluate_results(dataset: Dict, history: Dict) -> Dict:
    """Evaluate fusion results against ground truth.

    Args:
        dataset: Dataset dictionary
        history: Fusion results from run_tc_fusion

    Returns:
        Metrics dictionary
    """
    truth = dataset["truth"]

    # Interpolate truth to estimated timestamps
    p_true_interp = np.column_stack(
        [
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 0]),
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 1]),
        ]
    )

    # Extract estimated positions
    p_est = history["x_est"][:, :2]

    # Compute errors
    errors = compute_position_errors(p_true_interp, p_est)
    rmse = compute_position_rmse(errors)

    # The error distribution here is bimodal, so the RMS alone misleads in both
    # directions -- see 'median_error' and 'transient_fraction' below.
    magnitudes = np.linalg.norm(errors, axis=1)

    metrics = {
        "rmse_2d": rmse,
        "rmse_x": np.sqrt(np.mean(errors[:, 0] ** 2)),
        "rmse_y": np.sqrt(np.mean(errors[:, 1] ** 2)),
        "max_error": np.max(magnitudes),
        "final_error": np.linalg.norm(errors[-1]),
        "median_error": float(np.median(magnitudes)),
        "transient_fraction": float(np.mean(magnitudes > 0.5)),
    }

    return metrics


def plot_results(dataset: Dict, history: Dict, save_path: str = None) -> None:
    """Generate fusion results plots.

    Args:
        dataset: Dataset dictionary
        history: Fusion results
        save_path: Optional path to save figure
    """
    truth = dataset["truth"]
    anchors = dataset["uwb_anchors"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trajectory plot
    ax = axes[0, 0]
    ax.plot(truth["p_xy"][:, 0], truth["p_xy"][:, 1], "k-", label="Truth", linewidth=2)
    ax.plot(
        history["x_est"][:, 0], history["x_est"][:, 1], "b-", label="TC EKF", alpha=0.7
    )
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=100,
        c="red",
        marker="^",
        label="UWB Anchors",
        zorder=5,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory: TC IMU + UWB Fusion")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    # 2. Position error
    ax = axes[0, 1]
    p_true_interp = np.column_stack(
        [
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 0]),
            np.interp(history["t"], truth["t"], truth["p_xy"][:, 1]),
        ]
    )
    errors = history["x_est"][:, :2] - p_true_interp
    error_norm = np.linalg.norm(errors, axis=1)
    ax.plot(history["t"], error_norm, "b-")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Error vs Time")
    ax.grid(True)

    # 3. NIS plot
    ax = axes[1, 0]
    if len(history["nis"]) > 0:
        nis = np.array(history["nis"])
        accepted = np.array(history["gated"])

        ax.plot(nis[accepted], "g.", label="Accepted", markersize=4)
        if np.any(~accepted):
            ax.plot(
                np.where(~accepted)[0],
                nis[~accepted],
                "rx",
                label="Rejected",
                markersize=6,
            )

        # Chi-square bounds for m=1 DOF
        from core.fusion import chi_square_bounds

        lower, upper = chi_square_bounds(dof=1, confidence=0.95)
        ax.axhline(upper, color="r", linestyle="--", label="95% bounds")
        ax.axhline(lower, color="r", linestyle="--")

        ax.set_xlabel("UWB Update Index")
        ax.set_ylabel("NIS (Normalized Innovation Squared)")
        ax.set_title("Innovation Consistency (NIS)")
        ax.legend()
        ax.grid(True)

    # 4. Covariance trace
    ax = axes[1, 1]
    ax.plot(history["t"], history["P_trace"], "b-")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Trace(P)")
    ax.set_title("Covariance Trace")
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        # save_figure takes a directory and a stem, and writes svg/pdf/png
        # together; callers still pass a single path, so split it here.
        save_path = Path(save_path)
        written = save_figure(fig, save_path.parent, save_path.stem)
        print(f"\nSaved figure: {written[0]}")

    show_figures_if_requested()


def main():
    """Main entry point for TC fusion demo."""
    parser = argparse.ArgumentParser(
        description="Tightly Coupled IMU + UWB EKF Fusion Demo"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb",
        help="Path to fusion dataset directory",
    )
    parser.add_argument(
        "--no-gating", action="store_true", help="Disable chi-square gating"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Gating confidence level (default: 0.95 for 95%% confidence)",
    )
    parser.add_argument(
        "--batch-update",
        action="store_true",
        help="Use batch update mode (all ranges at same timestamp together)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save results figure"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)

    # Run fusion
    history = run_tc_fusion(
        dataset,
        use_gating=not args.no_gating,
        gate_confidence=args.confidence,
        batch_update=args.batch_update,
        verbose=True,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Metrics")
    print("=" * 70)
    metrics = evaluate_results(dataset, history)
    print(f"  RMSE (2D)    : {metrics['rmse_2d']:.3f} m")
    print(f"  RMSE (X)     : {metrics['rmse_x']:.3f} m")
    print(f"  RMSE (Y)     : {metrics['rmse_y']:.3f} m")
    print(f"  Max Error    : {metrics['max_error']:.3f} m")
    print(f"  Final Error  : {metrics['final_error']:.3f} m")
    print(f"  Median Error : {metrics['median_error']:.3f} m  <- typical tracking")

    # Median as well as RMS, because the distribution is not symmetric: the
    # filter tracks to a few centimetres, matching the ~3.5 cm ranging, and
    # lags briefly after each corner. Reporting only the RMS used to hide both.
    #
    # It used to hide much more. Until the trajectory generator was given a
    # finite turn rate, the corners were instantaneous -- 9000 deg/s, 5.1 g --
    # and the resulting transients alone took this RMSE from 0.167 m to
    # 0.739 m. What remains is ordinary manoeuvre lag at 57 deg/s.
    print(
        f"    {100 * metrics['transient_fraction']:.1f}% of samples exceed 0.5 m, "
        f"in the moments after a corner; the rest track at the median above."
    )
    print("")

    # Plot
    save_path = (
        args.save if args.save else "ch8_sensor_fusion/figs/tc_uwb_imu_results.svg"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_results(dataset, history, save_path=save_path)


if __name__ == "__main__":
    main()
