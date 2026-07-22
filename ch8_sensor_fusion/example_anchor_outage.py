"""Anchor Outage: why tight coupling exists, Chapter 8 Section 8.1.

The difference between loose and tight coupling is easy to state and hard to
feel from summary statistics. On the shipped dataset the two are close --
RMSE 0.95 m for LC against 0.74 m for TC -- and the natural anchor dropouts
are single isolated epochs, so the loosely coupled filter simply coasts for a
fraction of a second and nothing visible happens.

The distinction only bites when anchor visibility drops *and stays down*:

- **Loosely coupled** must solve for a position before it can fuse anything.
  Two ranges do not determine a 2-D position, so its front end returns nothing
  and the filter dead-reckons on IMU alone for the whole outage.
- **Tightly coupled** fuses raw ranges, so two ranges are simply two
  measurement updates. It never stops correcting.

This example constructs that outage deliberately -- 8 seconds with at most two
of four anchors visible -- because the shipped dataset does not contain one.
(At most two, not exactly two: the dataset's own dropouts stack on top of the
mask, leaving a single anchor for 6 of the 81 epochs.) With the outage in place
the gap is no longer subtle. LC's error ramps linearly to 4.5 m -- the shape of
pure dead reckoning -- and snaps back the instant anchors return, while TC
holds 0.3 m throughout. That is a factor of 15, and 93 LC position fixes fail
outright.

Two caveats keep this honest, because tight coupling is a trade, not a free
win:

- **TC is more exposed to a bad range.** Away from the outage this dataset has
  two outlier events, and at both of them TC peaks *higher* than LC: 4.4 m
  against 3.4 m near t = 37 s, and 5.2 m against 3.4 m near t = 57 s. A
  corrupted range enters the filter directly, whereas LC's front end solves a
  position from all ranges at once and can absorb or reject the bad one there.
  This is exactly what the chi-square gating of Section 8.3.2 is for.
- **TC needs usable geometry.** Repeating the outage with a *single* visible
  anchor leaves TC updating from a degenerate configuration and its overall
  RMSE degrades to 1.8 m, worse than LC's 1.3 m. One range constrains a circle,
  not a point.

Run:
    python -m ch8_sensor_fusion.example_anchor_outage
    python -m ch8_sensor_fusion.example_anchor_outage --animate

Author: Li-Ta Hsu
References: Chapter 8, Section 8.1 (loose vs tight coupling)
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch8_sensor_fusion.lc_uwb_imu_ekf import load_fusion_dataset, run_lc_fusion
from ch8_sensor_fusion.tc_uwb_imu_ekf import run_tc_fusion
from core.eval import save_animation, save_figure

FIGS_DIR = Path(__file__).parent / "figs"
DEFAULT_DATA = "data/sim/ch8_fusion_2d_imu_uwb"

# The constructed outage: keep only this many anchors between these times.
OUTAGE_WINDOW = (20.0, 28.0)
ANCHORS_KEPT = 2

COLOR_TRUTH = "black"
COLOR_LC = "tab:blue"
COLOR_TC = "tab:orange"


def apply_anchor_outage(dataset, window=OUTAGE_WINDOW, keep=ANCHORS_KEPT):
    """Blank all but ``keep`` anchors inside ``window``.

    Args:
        dataset: Fusion dataset dictionary.
        window: (start, end) seconds of the outage.
        keep: Number of anchors left visible during the outage.

    Returns:
        A deep copy of the dataset with the outage applied.
    """
    outaged = copy.deepcopy(dataset)
    ranges = np.asarray(outaged["uwb"]["ranges"]).copy()
    times = np.asarray(outaged["uwb"]["t"])

    selected = (times >= window[0]) & (times <= window[1])
    ranges[np.ix_(selected, np.arange(keep, ranges.shape[1]))] = np.nan
    outaged["uwb"]["ranges"] = ranges
    return outaged


def _position_error(result, truth):
    """Horizontal error of an estimate against truth, on the estimate's clock.

    LC and TC run on different time bases, so each is interpolated separately
    rather than compared sample-for-sample.
    """
    t = np.asarray(result["t"])
    p = np.asarray(result["x_est"])[:, :2]
    t_truth = np.asarray(truth["t"])
    p_truth = np.asarray(truth["p_xy"])
    east = np.interp(t, t_truth, p_truth[:, 0])
    north = np.interp(t, t_truth, p_truth[:, 1])
    return t, np.hypot(p[:, 0] - east, p[:, 1] - north)


def run_outage_scenario(data_dir=DEFAULT_DATA, window=OUTAGE_WINDOW,
                        keep=ANCHORS_KEPT, verbose=True):
    """Run LC and TC over a dataset with a constructed anchor outage.

    Args:
        data_dir: Fusion dataset directory.
        window: (start, end) seconds of the outage.
        keep: Anchors left visible during the outage.
        verbose: Print progress from the fusion runs.

    Returns:
        Dictionary with the dataset, both results, errors and visibility.
    """
    dataset = apply_anchor_outage(load_fusion_dataset(data_dir), window, keep)

    lc_results = run_lc_fusion(dataset, verbose=verbose)
    tc_results = run_tc_fusion(dataset, verbose=verbose)

    truth = dataset["truth"]
    t_lc, error_lc = _position_error(lc_results, truth)
    t_tc, error_tc = _position_error(tc_results, truth)

    ranges = np.asarray(dataset["uwb"]["ranges"])
    visibility = np.sum(~np.isnan(ranges), axis=1)

    return {
        "dataset": dataset,
        "lc": lc_results,
        "tc": tc_results,
        "t_lc": t_lc,
        "error_lc": error_lc,
        "t_tc": t_tc,
        "error_tc": error_tc,
        "t_uwb": np.asarray(dataset["uwb"]["t"]),
        "visibility": visibility,
        "window": window,
    }


def animate_anchor_outage(scenario, n_frames: int = 40):
    """Build the outage animation.

    Args:
        scenario: Output of :func:`run_outage_scenario`.
        n_frames: Number of animation frames.

    Returns:
        Tuple of (figure, update callback, frame count) for save_animation.
    """
    dataset = scenario["dataset"]
    truth = dataset["truth"]
    anchors = np.asarray(dataset["uwb_anchors"])
    p_truth = np.asarray(truth["p_xy"])
    t_truth = np.asarray(truth["t"])

    t_lc, error_lc = scenario["t_lc"], scenario["error_lc"]
    t_tc, error_tc = scenario["t_tc"], scenario["error_tc"]
    p_lc = np.asarray(scenario["lc"]["x_est"])[:, :2]
    p_tc = np.asarray(scenario["tc"]["x_est"])[:, :2]
    t_uwb, visibility = scenario["t_uwb"], scenario["visibility"]
    window = scenario["window"]

    t_end = min(t_lc[-1], t_tc[-1])
    frame_times = np.linspace(t_truth[0], t_end, n_frames)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    max_error = max(error_lc.max(), error_tc.max()) * 1.1

    def update(frame: int):
        """Draw everything up to ``frame_times[frame]``."""
        now = frame_times[frame]
        for ax in axes:
            ax.clear()

        in_outage = window[0] <= now <= window[1]
        n_visible = int(
            visibility[max(np.searchsorted(t_uwb, now) - 1, 0)]
        )

        # --- trajectory, with anchors switching off during the outage
        k_truth = np.searchsorted(t_truth, now) + 1
        k_lc = np.searchsorted(t_lc, now) + 1
        k_tc = np.searchsorted(t_tc, now) + 1
        axes[0].plot(p_truth[:k_truth, 0], p_truth[:k_truth, 1],
                     color=COLOR_TRUTH, linewidth=2.0, label="ground truth")
        axes[0].plot(p_lc[:k_lc, 0], p_lc[:k_lc, 1], color=COLOR_LC,
                     linewidth=1.5, label="loosely coupled")
        axes[0].plot(p_tc[:k_tc, 0], p_tc[:k_tc, 1], color=COLOR_TC,
                     linewidth=1.5, label="tightly coupled")

        for index, anchor in enumerate(anchors):
            visible = (not in_outage) or index < ANCHORS_KEPT
            axes[0].scatter(
                anchor[0], anchor[1], s=140, marker="^",
                c="red" if visible else "none",
                edgecolors="darkred", linewidths=2, zorder=5,
            )
        axes[0].set_aspect("equal")
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel("X [m]")
        axes[0].set_ylabel("Y [m]")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_title(
            f"t = {now:5.1f} s   -   {n_visible} of {len(anchors)} anchors "
            f"visible\n(hollow triangles are blacked out)",
            fontsize=10,
        )

        # --- anchor visibility over time
        shown = t_uwb <= now
        axes[1].step(t_uwb[shown], visibility[shown], where="post",
                     color="0.25", linewidth=1.6)
        axes[1].axhline(3, color="red", linestyle="--", linewidth=1.4,
                        label="LC needs 3 for a fix")
        axes[1].axvspan(window[0], min(now, window[1]), color="0.85", zorder=0)
        axes[1].set_xlim(t_truth[0], t_end)
        axes[1].set_ylim(0, len(anchors) + 0.5)
        axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("anchors visible")
        axes[1].legend(fontsize=8, loc="lower left")
        axes[1].set_title("anchor visibility", fontsize=10)

        # --- error, the payoff
        axes[2].plot(t_lc[:k_lc], error_lc[:k_lc], color=COLOR_LC,
                     linewidth=1.6, label="loosely coupled")
        axes[2].plot(t_tc[:k_tc], error_tc[:k_tc], color=COLOR_TC,
                     linewidth=1.6, label="tightly coupled")
        axes[2].axvspan(window[0], min(now, window[1]), color="0.85", zorder=0)
        axes[2].set_xlim(t_truth[0], t_end)
        axes[2].set_ylim(0, max_error)
        axes[2].grid(alpha=0.3)
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("horizontal position error [m]")
        axes[2].legend(fontsize=8, loc="upper left")
        current_lc = error_lc[min(k_lc, len(error_lc)) - 1]
        current_tc = error_tc[min(k_tc, len(error_tc)) - 1]
        axes[2].set_title(
            f"error: LC {current_lc:.2f} m   |   TC {current_tc:.2f} m",
            fontsize=10,
        )

        state = (f"OUTAGE: {n_visible} anchors -- LC cannot solve a fix at all"
                 if in_outage else "anchors nominal")
        fig.suptitle(
            f"Loose vs tight coupling under anchor outage   -   {state}",
            fontsize=11,
        )
        fig.tight_layout()
        return axes

    return fig, update, len(frame_times)


def plot_outage_summary(scenario) -> plt.Figure:
    """Static counterpart to the animation: the whole run at a glance."""
    window = scenario["window"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    axes[0].step(scenario["t_uwb"], scenario["visibility"], where="post",
                 color="0.25", linewidth=1.6)
    axes[0].axhline(3, color="red", linestyle="--", linewidth=1.4,
                    label="LC needs 3 for a fix")
    axes[0].axvspan(*window, color="0.85", zorder=0)
    axes[0].set_ylabel("anchors visible")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        f"Constructed outage: at most {ANCHORS_KEPT} of 4 anchors between "
        f"t = {window[0]:.0f} s and {window[1]:.0f} s",
        fontsize=11,
    )

    axes[1].plot(scenario["t_lc"], scenario["error_lc"], color=COLOR_LC,
                 linewidth=1.6, label="loosely coupled")
    axes[1].plot(scenario["t_tc"], scenario["error_tc"], color=COLOR_TC,
                 linewidth=1.6, label="tightly coupled")
    axes[1].axvspan(*window, color="0.85", zorder=0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("horizontal position error [m]")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(
        "In the outage LC dead-reckons (linear ramp) while TC keeps fusing "
        "the surviving ranges.\n"
        "Outside it, at the two outlier events, TC peaks HIGHER: "
        "a bad range hits the filter directly.",
        fontsize=11,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    """Run the outage scenario and write its figures."""
    parser = argparse.ArgumentParser(
        description="Anchor outage: loose vs tight coupling (Chapter 8)"
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Fusion dataset directory")
    parser.add_argument("--out-dir", default=str(FIGS_DIR),
                        help="Output directory for figures")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Also render the outage animation GIF (slower)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 8: Anchor Outage -- Loose vs Tight Coupling")
    print("=" * 70)
    print(f"Constructed outage: at most {ANCHORS_KEPT} of 4 anchors between "
          f"t = {OUTAGE_WINDOW[0]:.0f} s and {OUTAGE_WINDOW[1]:.0f} s")
    print("(the shipped dataset's own dropouts are single isolated epochs "
          "and do not stress the difference)\n")

    scenario = run_outage_scenario(args.data, verbose=False)

    def in_window(times):
        """Mask covering the outage plus a short recovery tail."""
        return (times >= OUTAGE_WINDOW[0]) & (times <= OUTAGE_WINDOW[1] + 3.0)

    peak_lc = scenario["error_lc"][in_window(scenario["t_lc"])].max()
    peak_tc = scenario["error_tc"][in_window(scenario["t_tc"])].max()
    rmse_lc = np.sqrt(np.mean(scenario["error_lc"] ** 2))
    rmse_tc = np.sqrt(np.mean(scenario["error_tc"] ** 2))

    print(f"  LC position fixes that failed outright: "
          f"{scenario['lc']['n_uwb_failed']}")
    print(f"  RMSE over the run:      LC {rmse_lc:.3f} m   TC {rmse_tc:.3f} m")
    print(f"  Peak error in outage:   LC {peak_lc:.2f} m   TC {peak_tc:.2f} m "
          f"({peak_lc / peak_tc:.0f}x)")

    # The other half of the story: TC is more exposed to a corrupted range.
    for lo, hi in ((35.0, 40.0), (55.0, 60.0)):
        span_lc = scenario["error_lc"][
            (scenario["t_lc"] >= lo) & (scenario["t_lc"] <= hi)
        ].max()
        span_tc = scenario["error_tc"][
            (scenario["t_tc"] >= lo) & (scenario["t_tc"] <= hi)
        ].max()
        verdict = "TC worse" if span_tc > span_lc else "TC better"
        print(f"  Outlier event t={lo:.0f}-{hi:.0f}s:  LC {span_lc:.2f} m   "
              f"TC {span_tc:.2f} m   ({verdict})")
    print()

    paths = save_figure(plot_outage_summary(scenario), args.out_dir,
                        "ch8_anchor_outage")
    print(f"  saved ch8_anchor_outage: "
          f"{', '.join(p.suffix.lstrip('.') for p in paths)}")

    if args.animate:
        fig, update, n_frames = animate_anchor_outage(scenario)
        path = save_animation(fig, update, n_frames, args.out_dir,
                              "ch8_anchor_outage", fps=5)
        plt.close(fig)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  saved {path.name}: {n_frames} frames, {size_mb:.2f} MB")

    plt.close("all")
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
