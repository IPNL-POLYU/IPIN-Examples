"""Fingerprint Posterior Along a Walk, Chapter 5 Section 5.2.

The natural animation to reach for here is "watch the posterior sharpen as the
user walks." Measurement says there is nothing to sharpen. With eight access
points over this grid the Gaussian Naive-Bayes posterior of Eq. (5.3) is
essentially a delta everywhere -- its entropy averages ~0.1 against a maximum
of 4.8, and its peak probability sits between 0.92 and 1.00. The posterior does
not gradually tighten; it is already tight.

What is genuinely dynamic, and specific to fingerprinting, is how that sharp
posterior *fails*. It does not spread out and hedge when the measurement is
noisy. It stays a confident spike and occasionally puts that spike in the wrong
place -- a distant reference point whose stored radio signature happens to
resemble the current one. This is RSS aliasing, and it is the characteristic
failure mode of fingerprinting: not graceful uncertainty, but a confident jump.

This example walks a user along an L-shaped path and shows the posterior over
the floor as a heat map at each step, with the MAP estimate (Eq. 5.4) on top.
At a modest measurement noise the hot spot tracks the user faithfully. Raise
the noise and the estimate begins to *teleport*: measured over a 21-step walk,
aliasing jumps beyond 10 m occur 0 times at 1 dB, 0 times at 3 dB, and 6 times
at 6 dB. The median error stays 0 throughout -- the estimate is usually exactly
right -- while the mean is dragged to 6.6 m entirely by those few jumps. A mean
error alone would hide the failure completely; the walk makes it visible.

Run:
    python -m ch5_fingerprinting.example_walk_posterior
    python -m ch5_fingerprinting.example_walk_posterior --animate

Author: Li-Ta Hsu
References: Chapter 5, Sections 5.1-5.2, Eqs. (5.3)-(5.4), (5.6)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from core.eval import save_animation, save_figure
from core.fingerprinting import (
    fit_gaussian_naive_bayes,
    load_fingerprint_database,
)
from core.fingerprinting.probabilistic import log_posterior, map_localize

FIGS_DIR = Path(__file__).parent / "figs"
DEFAULT_DATA = "data/sim/ch5_wifi_fingerprint_grid"

FLOOR_ID = 0
MIN_STD = 2.0
NOISE_STD = 6.0  # dB; high enough to expose aliasing (0 jumps at 1-3 dB)
SEED = 11
ALIASING_THRESHOLD = 10.0  # m; an error beyond this is a teleport, not drift

# A fixed [0, 1] probability scale, shared by every frame so the colorbar can
# be built once. Gamma compression lifts the runner-up cells of a near-delta
# posterior into view; "viridis" keeps a dark-but-not-black floor and a vivid
# yellow peak.
POSTERIOR_CMAP = "viridis"
POSTERIOR_NORM = PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)


def _build_l_walk(locations):
    """Reference-point indices along an L: the bottom edge then the right edge.

    Args:
        locations: Reference-point positions on the floor, shape (N, 2).

    Returns:
        List of indices into ``locations`` tracing the walk.
    """
    key = {
        (round(float(x)), round(float(y))): i
        for i, (x, y) in enumerate(locations)
    }
    x_max = int(round(locations[:, 0].max()))
    y_max = int(round(locations[:, 1].max()))
    step = int(round(np.min(np.diff(np.unique(locations[:, 0])))))

    path = [(x, 0) for x in range(0, x_max + 1, step)]
    path += [(x_max, y) for y in range(step, y_max + 1, step)]
    return [key[p] for p in path if p in key]


def run_walk(data_dir=DEFAULT_DATA, noise_std=NOISE_STD, seed=SEED):
    """Walk a user along the L-path and record the posterior at each step.

    Returns:
        Dictionary with the model, floor geometry, per-step posteriors,
        estimates, true positions and errors.
    """
    model = fit_gaussian_naive_bayes(
        load_fingerprint_database(data_dir), min_std=MIN_STD
    )
    mask = model.get_floor_mask(FLOOR_ID)
    locations = model.locations[mask]
    means = model.means[mask]

    walk = _build_l_walk(locations)

    rng = np.random.default_rng(seed)
    posteriors, estimates, true_xy, errors, hot_counts = [], [], [], [], []
    for rp_index in walk:
        query = means[rp_index] + rng.normal(0.0, noise_std, means.shape[1])

        log_post = log_posterior(query, model, floor_id=FLOOR_ID)[mask]
        posterior = np.exp(log_post - log_post.max())
        posterior /= posterior.sum()

        estimate = map_localize(query, model, floor_id=FLOOR_ID)
        truth = locations[rp_index]

        posteriors.append(posterior)
        estimates.append(estimate)
        true_xy.append(truth)
        errors.append(float(np.linalg.norm(estimate - truth)))
        hot_counts.append(int(np.sum(posterior > 0.15 * posterior.max())))

    return {
        "locations": locations,
        "posteriors": posteriors,
        "estimates": np.array(estimates),
        "true_xy": np.array(true_xy),
        "errors": np.array(errors),
        "hot_counts": np.array(hot_counts),
        "noise_std": noise_std,
    }


def _draw_frame(axes, walk, index):
    """Render the heat map and error trace up to step ``index``.

    Returns:
        The scatter mappable, so the caller can attach a single colorbar.
        Does not create a colorbar itself -- doing so per frame stacks a new
        one every step, since ax.clear() does not remove the colorbar axes.
    """
    locations = walk["locations"]
    posterior = walk["posteriors"][index]
    truth = walk["true_xy"][index]
    estimate = walk["estimates"][index]
    error = walk["errors"][index]

    for ax in axes:
        ax.clear()

    # --- posterior heat map over the floor, on a fixed [0, 1] probability
    # scale (see POSTERIOR_NORM). The posterior is a near-delta -- one cell
    # near 0.9, the rest near 1e-4 -- so the gamma-compressed norm is what
    # lifts the runner-up cells enough to see where the mass sits.
    scatter = axes[0].scatter(
        locations[:, 0], locations[:, 1], c=posterior, s=160,
        cmap=POSTERIOR_CMAP, norm=POSTERIOR_NORM, marker="s",
    )
    axes[0].plot(walk["true_xy"][: index + 1, 0],
                 walk["true_xy"][: index + 1, 1],
                 "-", color="white", linewidth=1.2, alpha=0.6)
    axes[0].scatter(*truth, marker="*", s=280, c="red",
                    edgecolors="white", linewidths=1.5, zorder=6,
                    label="true position")
    # Hollow ring, so the bright peak cell it sits on stays visible -- a solid
    # marker would hide the very cell it reports and make the runner-up look
    # like the answer.
    axes[0].scatter(*estimate, marker="o", s=260, facecolors="none",
                    edgecolors="white", linewidths=2.5, zorder=6,
                    label="MAP estimate (peak cell)")
    if error > ALIASING_THRESHOLD:
        axes[0].plot([truth[0], estimate[0]], [truth[1], estimate[1]],
                     "--", color="red", linewidth=1.8, zorder=5)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    verdict = "ALIASED: MAP jumped to a radio-similar spot" if (
        error > ALIASING_THRESHOLD) else "MAP on target"
    axes[0].set_title(
        f"step {index + 1}/{len(walk['errors'])}   -   {verdict}", fontsize=10
    )

    # --- error trace, with the aliasing threshold marked
    steps = np.arange(1, index + 2)
    axes[1].plot(steps, walk["errors"][: index + 1], "-o", color="tab:blue",
                 markersize=4, linewidth=1.5)
    axes[1].axhline(ALIASING_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.3, label=f"aliasing (> {ALIASING_THRESHOLD:.0f} m)")
    axes[1].set_xlim(0.5, len(walk["errors"]) + 0.5)
    axes[1].set_ylim(-1, max(walk["errors"].max() * 1.1, 12))
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MAP position error [m]")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(
        f"error {error:.1f} m   |   posterior peaked on "
        f"{walk['hot_counts'][index]} cell(s)",
        fontsize=10,
    )

    return scatter


def _add_posterior_colorbar(fig, axes):
    """Attach the single shared p(x|z) colorbar to the heat-map axes."""
    mappable = plt.cm.ScalarMappable(norm=POSTERIOR_NORM, cmap=POSTERIOR_CMAP)
    fig.colorbar(mappable, ax=axes[0], fraction=0.046, pad=0.04,
                 label="p(x | z), Eq. (5.3)")


def animate_walk(walk):
    """Build the walking-posterior animation.

    Args:
        walk: Output of :func:`run_walk`.

    Returns:
        Tuple of (figure, update callback, frame count) for save_animation.
    """
    n_frames = len(walk["errors"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # One colorbar for the whole animation; the scale is fixed at [0, 1].
    _add_posterior_colorbar(fig, axes)

    def update(frame: int):
        _draw_frame(axes, walk, frame)
        fig.suptitle(
            "Fingerprint posterior along a walk, Section 5.2: the hot spot "
            "tracks the user, then teleports when RSS aliases",
            fontsize=11,
        )
        fig.tight_layout()
        return axes

    return fig, update, n_frames


def plot_walk_summary(walk) -> plt.Figure:
    """Static counterpart: an aliased step beside the error trace."""
    aliased = np.where(walk["errors"] > ALIASING_THRESHOLD)[0]
    highlight = int(aliased[0]) if len(aliased) else int(
        np.argmax(walk["errors"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    _draw_frame(axes, walk, highlight)
    _add_posterior_colorbar(fig, axes)

    # Redraw the error panel over the whole walk, not just up to the step.
    steps = np.arange(1, len(walk["errors"]) + 1)
    axes[1].clear()
    axes[1].plot(steps, walk["errors"], "-o", color="tab:blue", markersize=4,
                 linewidth=1.5)
    axes[1].axhline(ALIASING_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.3, label=f"aliasing (> {ALIASING_THRESHOLD:.0f} m)")
    for step in aliased:
        axes[1].scatter(step + 1, walk["errors"][step], s=90, c="red",
                        zorder=5)
    axes[1].set_xlim(0.5, len(walk["errors"]) + 0.5)
    axes[1].set_ylim(-1, max(walk["errors"].max() * 1.1, 12))
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MAP position error [m]")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(
        f"median {np.median(walk['errors']):.1f} m, "
        f"mean {walk['errors'].mean():.1f} m -- "
        f"{len(aliased)} of {len(steps)} steps alias",
        fontsize=10,
    )

    fig.suptitle(
        f"Fingerprint posterior, Section 5.2 (noise {walk['noise_std']:.0f} dB): "
        "the median is 0 m, the mean is not -- aliasing hides in the mean",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the walk and write its figures."""
    parser = argparse.ArgumentParser(
        description="Fingerprint posterior along a walk (Chapter 5)"
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Fingerprint database directory")
    parser.add_argument("--out-dir", default=str(FIGS_DIR),
                        help="Output directory for figures")
    parser.add_argument("--noise", type=float, default=NOISE_STD,
                        help="Measurement noise std in dB")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Also render the walking-posterior GIF (slower)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 5, Section 5.2: a sharp posterior that fails by aliasing")
    print("=" * 70)

    walk = run_walk(args.data, noise_std=args.noise)
    errors = walk["errors"]
    aliased = int(np.sum(errors > ALIASING_THRESHOLD))

    print(f"  L-walk of {len(errors)} reference points, "
          f"noise {args.noise:.0f} dB")
    print(f"  posterior peaks on a median of "
          f"{int(np.median(walk['hot_counts']))} cell(s) -- it stays sharp")
    print(f"  MAP error:  median {np.median(errors):.1f} m   "
          f"mean {errors.mean():.1f} m   max {errors.max():.1f} m")
    print(f"  aliasing jumps (> {ALIASING_THRESHOLD:.0f} m): "
          f"{aliased} of {len(errors)} steps")
    print("  -> the median says 'perfect', the mean says otherwise\n")

    paths = save_figure(plot_walk_summary(walk), args.out_dir,
                        "ch5_walk_posterior")
    print(f"  saved ch5_walk_posterior: "
          f"{', '.join(p.suffix.lstrip('.') for p in paths)}")

    if args.animate:
        fig, update, n_frames = animate_walk(walk)
        path = save_animation(fig, update, n_frames, args.out_dir,
                              "ch5_walk_posterior", fps=3)
        plt.close(fig)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  saved {path.name}: {n_frames} frames, {size_mb:.2f} MB")

    plt.close("all")
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
