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
the gap is no longer subtle. LC's error ramps to 5.7 m by the end of the outage
-- the shape of pure dead reckoning -- and snaps back the instant anchors
return. TC keeps being corrected the whole way through: its median error inside
the outage is 0.71 m against LC's 2.98 m, and at the moment anchors return LC
sits at 5.65 m while TC is at 0.04 m. 93 LC position fixes fail outright.

The comparison is between an estimator that stops updating and one that does
not, so read the *end* of the outage rather than the average: LC's error grows
without bound for as long as the outage lasts, while TC's does not.

Two caveats keep this honest, because tight coupling is a trade, not a free
win:

- **Two collinear anchors leave a mirror ambiguity, and TC can latch onto the
  wrong branch.** The outage keeps anchors 0 and 1, at (0, 0) and (20, 0), both
  on the line y = 0, while the platform travels the x = 20 leg. Two ranges to
  two anchors are satisfied equally well by the true position and by its
  reflection across the anchor baseline, and for 0.2 s around t = 25.8 s this
  filter takes the wrong one: the estimate jumps to (30.1, -35.5) against a
  truth of (20.0, 7.2), a 43.9 m peak, before the returning anchors snap it
  back. LC never does this because its front end solves a position from all
  available ranges and simply fails when it cannot.

  That excursion is brief but it is what puts TC's whole-run RMSE (2.34 m)
  above LC's (1.57 m) *at this particular window*, which is worth stating
  because it is the one number in this demo where LC wins. It is a knife edge,
  not the general case: moving the outage to (18, 26), (22, 30), (24, 32) or
  (40, 48) gives TC peak errors of 0.36, 0.12, 0.79 and 0.05 m against LC's
  7.19, 7.25, 1.82 and 5.98 m, and TC RMSE around 0.17 m against LC's 1.1 to
  1.9 m. The window has deliberately not been moved to a flattering one.
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

from core.fusion import load_fusion_dataset, run_lc_fusion, run_tc_fusion
from core.eval import resolve_figs_dir, save_animation, save_figure, show_figures_if_requested

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

    # Log axis. The mirror-branch flip reaches 43.9 m while the claim this
    # panel exists to make -- LC's dead-reckoning ramp -- tops out at 5.9 m, so
    # a linear axis renders the ramp as a low squiggle beneath one narrow
    # spike. The error is positive and spans two decades, which is what a log
    # axis is for.
    floor = 1e-3  # a log axis cannot show an exact zero
    axes[1].semilogy(scenario["t_lc"],
                     np.maximum(scenario["error_lc"], floor),
                     color=COLOR_LC, linewidth=1.6, label="loosely coupled")
    axes[1].semilogy(scenario["t_tc"],
                     np.maximum(scenario["error_tc"], floor),
                     color=COLOR_TC, linewidth=1.6, label="tightly coupled")
    axes[1].axvspan(*window, color="0.85", zorder=0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("horizontal position error [m], log")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(
        "In the outage LC dead-reckons (linear ramp) while TC keeps fusing "
        "the surviving ranges.\n"
        "TC's one spike is a mirror-branch flip: the two surviving anchors "
        "are collinear with the leg being walked.",
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

    # The other half of the story, and it is a geometry problem rather than a
    # bad measurement. The surviving anchors are collinear with the leg being
    # walked, so two ranges fit the truth and its mirror image equally well.
    worst = int(np.argmax(scenario["error_tc"]))
    if scenario["error_tc"][worst] > 5.0:
        est = np.asarray(scenario["tc"]["x_est"])[worst, :2]
        truth = scenario["dataset"]["truth"]
        t_worst = scenario["t_tc"][worst]
        truth_xy = np.array([
            np.interp(t_worst, truth["t"], truth["p_xy"][:, 0]),
            np.interp(t_worst, truth["t"], truth["p_xy"][:, 1]),
        ])
        print(f"  TC's peak is a mirror-branch flip at t="
              f"{scenario['t_tc'][worst]:.1f} s: estimate "
              f"({est[0]:.1f}, {est[1]:.1f}) against truth "
              f"({truth_xy[0]:.1f}, {truth_xy[1]:.1f}),")
        print("    reflected across the y = 0 baseline joining the two "
              "surviving anchors. It lasts under a second, and it is what "
              "puts TC's")
        print("    whole-run RMSE above LC's at this window. Other outage "
              "windows do not trigger it -- see the module docstring.")
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
    print(f"\nFigures written to {resolve_figs_dir(args.out_dir)}")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
