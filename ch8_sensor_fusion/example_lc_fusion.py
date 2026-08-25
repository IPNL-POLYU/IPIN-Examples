"""Loosely Coupled IMU + UWB EKF Fusion Demo (Chapter 8).

Demonstrates loosely coupled fusion where UWB ranges are first solved
for a position fix, then the position fix is fused with IMU propagation.

Comparison with Tightly Coupled (TC):
- TC: Fuses raw UWB range measurements directly (one update per anchor)
- LC: First solves for position from all ranges, then fuses position

Features:
- High-rate IMU propagation (100 Hz)
- Low-rate UWB position fix updates (10 Hz)
- WLS position solver from Chapter 4
- Chi-square innovation gating (Eq. 8.9)
- Innovation monitoring (Eqs. 8.5-8.6)

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1.1 (Loosely Coupled)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import (
    compute_position_errors,
    compute_position_rmse,
    save_figure,
    show_figures_if_requested,
)
from core.fusion import load_fusion_dataset, run_lc_fusion


def evaluate_results(dataset: Dict, history: Dict) -> Dict:
    """Evaluate fusion results against ground truth."""
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

    metrics = {
        "rmse_2d": rmse,
        "rmse_x": np.sqrt(np.mean(errors[:, 0] ** 2)),
        "rmse_y": np.sqrt(np.mean(errors[:, 1] ** 2)),
        "max_error": np.max(np.linalg.norm(errors, axis=1)),
        "final_error": np.linalg.norm(errors[-1]),
    }

    return metrics


def plot_results(dataset: Dict, history: Dict, save_path: str = None) -> None:
    """Generate LC fusion results plots."""
    truth = dataset["truth"]
    anchors = dataset["uwb_anchors"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trajectory plot
    ax = axes[0, 0]
    ax.plot(truth["p_xy"][:, 0], truth["p_xy"][:, 1], "k-", label="Truth", linewidth=2)
    ax.plot(
        history["x_est"][:, 0], history["x_est"][:, 1], "b-", label="LC EKF", alpha=0.7
    )

    # Plot UWB position fixes
    if len(history["uwb_positions"]) > 0:
        ax.scatter(
            history["uwb_positions"][:, 0],
            history["uwb_positions"][:, 1],
            s=20,
            c="orange",
            alpha=0.3,
            label="UWB Fixes",
            zorder=2,
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
    ax.set_title("Trajectory: LC IMU + UWB Fusion")
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

        # Chi-square bounds for m=2 DOF (position is 2D)
        from core.fusion import chi_square_bounds

        lower, upper = chi_square_bounds(dof=2, confidence=0.95)
        ax.axhline(upper, color="r", linestyle="--", label="95% bounds")
        ax.axhline(lower, color="r", linestyle="--")

        ax.set_xlabel("UWB Update Index")
        ax.set_ylabel("NIS (Normalized Innovation Squared)")
        ax.set_title("Innovation Consistency (NIS) - 2 DOF")
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
    """Main entry point for LC fusion demo."""
    parser = argparse.ArgumentParser(
        description="Loosely Coupled IMU + UWB EKF Fusion Demo"
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
        "--save", type=str, default=None, help="Path to save results figure"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    dataset = load_fusion_dataset(args.data)

    # Run fusion
    history = run_lc_fusion(
        dataset,
        use_gating=not args.no_gating,
        gate_confidence=args.confidence,
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
    print("")

    # Plot
    save_path = (
        args.save if args.save else "ch8_sensor_fusion/figs/lc_uwb_imu_results.svg"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_results(dataset, history, save_path=save_path)


if __name__ == "__main__":
    main()
