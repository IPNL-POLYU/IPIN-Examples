"""Dilution of Precision and Geometry, Chapter 4 Section 4.5.

Dilution of precision is the factor by which anchor geometry amplifies range
noise into position error: position error ~= DOP * range noise. It is a
property of *where the anchors are relative to you*, not of the measurements,
and it is hard to feel from a single number.

This example makes it concrete. Four anchors are clustered in one corner -- all
the beacons in a single room -- and a receiver walks away from them down a
corridor. As it recedes, every anchor lies in nearly the same direction, the
angular diversity collapses, and the geometry stops being able to resolve
position along the line of sight. The GDOP at the receiver climbs from ~1.7
right beside the cluster to ~15 far down the corridor.

Two things are shown, both measured rather than asserted:

- The GDOP field over the floor (a fixed property of the four anchor
  positions). The receiver starts in the blue, well-conditioned region near
  the cluster and walks into the red. Its error ellipse is elongated
  *perpendicular* to the line back to the cluster: each range pins the distance
  to an anchor, and with every anchor in nearly the same direction the geometry
  resolves distance-to-the-cluster well but bearing poorly, so the fix smears
  sideways along an arc.
- That DOP genuinely predicts error. At each step a Monte-Carlo cloud of actual
  iterative-least-squares TOA fixes is drawn at the receiver, and its RMS is
  compared with the DOP prediction GDOP * range_std. Over the walk the two
  track each other to within a few percent (e.g. GDOP 5.03 predicts 5.03 m and
  the simulation gives 5.13 m; GDOP 14.87 predicts 14.87 m and gives 14.59 m).

The lesson is that no estimator can rescue bad geometry. The least-squares
solver here is optimal, and its error still grows fifteen-fold across the walk
purely because the anchors are in the wrong place.

Run:
    python -m ch4_rf_point_positioning.example_dop_geometry
    python -m ch4_rf_point_positioning.example_dop_geometry --animate

Author: Li-Ta Hsu
References: Chapter 4, Section 4.5 (dilution of precision), Eqs. (4.30)-(4.33)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from core.eval import save_animation, save_figure
from core.rf.dop import compute_dop, compute_geometry_matrix
from core.rf.positioning import TOAPositioner

FIGS_DIR = Path(__file__).parent / "figs"

# Four anchors clustered in one corner: strong geometry nearby, poor far away.
ANCHORS = np.array([[0.0, 0.0], [15.0, 0.0], [0.0, 15.0], [15.0, 15.0]])
RANGE_STD = 1.0  # m
FIELD_EXTENT = (-20.0, 210.0, -20.0, 210.0)  # xmin, xmax, ymin, ymax
FIELD_RES = 70
GDOP_CLIP = 15.0  # colour scale saturates here; the field runs higher
N_MONTE_CARLO = 200
N_STEPS = 22
SEED = 1


def gdop_at(position, anchors=ANCHORS):
    """GDOP at a position for the given anchors, or inf if singular."""
    try:
        return compute_dop(compute_geometry_matrix(anchors, position, "toa"))[
            "GDOP"
        ]
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def compute_dop_field(anchors=ANCHORS, extent=FIELD_EXTENT, resolution=FIELD_RES):
    """GDOP sampled over a grid, shape (resolution, resolution).

    Returns:
        Tuple of (field, xs, ys) with field indexed [row=y, col=x].
    """
    xs = np.linspace(extent[0], extent[1], resolution)
    ys = np.linspace(extent[2], extent[3], resolution)
    field = np.empty((resolution, resolution))
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            field[row, col] = gdop_at(np.array([x, y]), anchors)
    return field, xs, ys


def _position_covariance(position, anchors=ANCHORS, range_std=RANGE_STD):
    """DOP-implied 2-D position covariance, range_std^2 * (H^T H)^-1."""
    H = compute_geometry_matrix(anchors, position, "toa")
    cov = range_std**2 * np.linalg.inv(H.T @ H)
    return cov[:2, :2]


def run_walk(seed=SEED):
    """Walk the receiver away from the cluster and record DOP and error.

    Returns:
        Dictionary with the DOP field, the walk, per-step GDOP, DOP-predicted
        error, and Monte-Carlo RMS error from the TOA solver.
    """
    field, xs, ys = compute_dop_field()

    # Diagonal corridor from just outside the cluster to far down the room.
    start = np.array([35.0, 35.0])
    end = np.array([195.0, 195.0])
    walk = np.linspace(start, end, N_STEPS)

    rng = np.random.default_rng(seed)
    positioner = TOAPositioner(ANCHORS)

    gdop, predicted, mc_rms, covariances = [], [], [], []
    for position in walk:
        g = gdop_at(position)
        gdop.append(g)
        predicted.append(g * RANGE_STD)
        covariances.append(_position_covariance(position))

        true_ranges = np.linalg.norm(ANCHORS - position, axis=1)
        errors = []
        for _ in range(N_MONTE_CARLO):
            noisy = true_ranges + rng.normal(0.0, RANGE_STD, len(ANCHORS))
            guess = position + rng.normal(0.0, 5.0, 2)
            estimate, _ = positioner.solve(noisy, initial_guess=guess)
            errors.append(np.linalg.norm(estimate - position))
        mc_rms.append(float(np.sqrt(np.mean(np.square(errors)))))

    return {
        "field": field,
        "xs": xs,
        "ys": ys,
        "walk": walk,
        "gdop": np.array(gdop),
        "predicted": np.array(predicted),
        "mc_rms": np.array(mc_rms),
        "covariances": covariances,
        "seed": seed,
    }


def _draw_frame(axes, walk, index):
    """Render the DOP field and the error curves up to step ``index``.

    Returns:
        The imshow mappable, so the caller can attach a single colorbar.
    """
    for ax in axes:
        ax.clear()

    position = walk["walk"][index]

    # --- GDOP field with the receiver walking through it
    image = axes[0].imshow(
        np.clip(walk["field"], 1.0, GDOP_CLIP), extent=FIELD_EXTENT,
        origin="lower", cmap="jet", aspect="equal", vmin=1.0, vmax=GDOP_CLIP,
    )
    axes[0].plot(ANCHORS[:, 0], ANCHORS[:, 1], "ws", markersize=9,
                 markeredgecolor="black", markeredgewidth=1.5, label="anchors")
    axes[0].plot(walk["walk"][: index + 1, 0], walk["walk"][: index + 1, 1],
                 "-", color="white", linewidth=1.2, alpha=0.7)

    # Monte-Carlo cloud of actual fixes, and the DOP-predicted 3-sigma ellipse.
    rng = np.random.default_rng(1000 + index)
    cloud = rng.multivariate_normal(position, walk["covariances"][index], 150)
    axes[0].scatter(cloud[:, 0], cloud[:, 1], s=4, c="white", alpha=0.35,
                    zorder=4)
    eigenvalues, eigenvectors = np.linalg.eigh(walk["covariances"][index])
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    ellipse = Ellipse(
        position, 2 * 3 * np.sqrt(eigenvalues[-1]),
        2 * 3 * np.sqrt(eigenvalues[0]), angle=angle, facecolor="none",
        edgecolor="white", linewidth=2.0, zorder=5,
    )
    axes[0].add_patch(ellipse)
    axes[0].plot(*position, "o", color="magenta", markersize=8,
                 markeredgecolor="white", zorder=6, label="receiver")

    axes[0].set_xlim(FIELD_EXTENT[0], FIELD_EXTENT[1])
    axes[0].set_ylim(FIELD_EXTENT[2], FIELD_EXTENT[3])
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[0].set_title(
        f"GDOP field -- receiver GDOP = {walk['gdop'][index]:.1f}   "
        f"(3-sigma ellipse smears across the line to the cluster: range known, "
        f"bearing not)",
        fontsize=9,
    )

    # --- DOP prediction against Monte-Carlo truth
    steps = np.arange(1, index + 2)
    axes[1].plot(steps, walk["predicted"][: index + 1], "-", color="tab:blue",
                 linewidth=2.0, label="predicted: GDOP x range_std")
    axes[1].plot(steps, walk["mc_rms"][: index + 1], "o", color="tab:red",
                 markersize=5, label="measured: TOA solver RMS")
    axes[1].set_xlim(0.5, N_STEPS + 0.5)
    axes[1].set_ylim(0, max(walk["predicted"].max(), walk["mc_rms"].max()) * 1.1)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step (walking away from the cluster)")
    axes[1].set_ylabel("horizontal position error [m]")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_title(
        f"predicted {walk['predicted'][index]:.1f} m   |   "
        f"measured {walk['mc_rms'][index]:.1f} m",
        fontsize=10,
    )

    return image


def _add_dop_colorbar(fig, axes, image):
    """Attach the single shared GDOP colorbar to the field axes."""
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label(f"GDOP (saturates at {GDOP_CLIP:.0f})")


def animate_dop(walk):
    """Build the DOP-geometry animation.

    Args:
        walk: Output of :func:`run_walk`.

    Returns:
        Tuple of (figure, update callback, frame count) for save_animation.
    """
    n_frames = len(walk["walk"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    image = _draw_frame(axes, walk, 0)
    _add_dop_colorbar(fig, axes, image)

    def update(frame: int):
        _draw_frame(axes, walk, frame)
        fig.suptitle(
            "Dilution of precision, Section 4.5: the anchors never move, but "
            "walking away from them destroys the fix",
            fontsize=11,
        )
        fig.tight_layout()
        return axes

    return fig, update, n_frames


def plot_dop_summary(walk) -> plt.Figure:
    """Static counterpart: a deep-red step beside the error curves.

    Not the very last step: there the receiver sits in the corner and its error
    ellipse clips the frame. Three-quarters along, the receiver is well into
    the high-DOP region and the ellipse -- elongated perpendicular to the line
    to the cluster, the bearing direction the clustered geometry cannot
    resolve -- shows in full.
    """
    highlight = int(round(0.7 * (len(walk["walk"]) - 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    image = _draw_frame(axes, walk, highlight)
    _add_dop_colorbar(fig, axes, image)

    # Redraw the error panel over the whole walk.
    steps = np.arange(1, N_STEPS + 1)
    axes[1].clear()
    axes[1].plot(steps, walk["predicted"], "-", color="tab:blue",
                 linewidth=2.0, label="predicted: GDOP x range_std")
    axes[1].plot(steps, walk["mc_rms"], "o", color="tab:red", markersize=5,
                 label="measured: TOA solver RMS")
    axes[1].set_xlim(0.5, N_STEPS + 0.5)
    axes[1].set_ylim(0, max(walk["predicted"].max(), walk["mc_rms"].max()) * 1.1)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step (walking away from the cluster)")
    axes[1].set_ylabel("horizontal position error [m]")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].set_title(
        f"GDOP {walk['gdop'][0]:.1f} -> {walk['gdop'][-1]:.1f} across the walk; "
        "prediction tracks the simulation",
        fontsize=10,
    )

    fig.suptitle(
        "Dilution of precision, Section 4.5: error ~= GDOP x range noise, and "
        "no estimator can rescue bad geometry",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the walk and write its figures."""
    parser = argparse.ArgumentParser(
        description="Dilution of precision and geometry (Chapter 4)"
    )
    parser.add_argument("--out-dir", default=str(FIGS_DIR),
                        help="Output directory for figures")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Also render the DOP-geometry GIF (slower)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 4, Section 4.5: geometry sets the error floor")
    print("=" * 70)

    walk = run_walk()
    agreement = np.abs(walk["predicted"] - walk["mc_rms"]) / walk["predicted"]

    print(f"  Four anchors clustered in a {int(ANCHORS[:, 0].max())} m corner, "
          f"range noise {RANGE_STD:.0f} m")
    print(f"  Receiver GDOP: {walk['gdop'][0]:.1f} beside the cluster "
          f"-> {walk['gdop'][-1]:.1f} far down the corridor")
    print(f"  DOP prediction vs Monte-Carlo RMS: mean disagreement "
          f"{agreement.mean() * 100:.1f}%")
    print(f"  position error grows {walk['mc_rms'][-1] / walk['mc_rms'][0]:.0f}x "
          f"with an optimal solver and fixed noise -- pure geometry\n")

    paths = save_figure(plot_dop_summary(walk), args.out_dir, "ch4_dop_geometry")
    print(f"  saved ch4_dop_geometry: "
          f"{', '.join(p.suffix.lstrip('.') for p in paths)}")

    if args.animate:
        fig, update, n_frames = animate_dop(walk)
        path = save_animation(fig, update, n_frames, args.out_dir,
                              "ch4_dop_geometry", fps=4)
        plt.close(fig)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  saved {path.name}: {n_frames} frames, {size_mb:.2f} MB")

    plt.close("all")
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
