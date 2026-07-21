"""Scan Matching Visualization for Chapter 7, Section 7.3.

ICP and NDT are the two scan-matching algorithms of Section 7.3, and both were
previously visible only through their final numbers -- a pose and an RMSE. What
the numbers cannot show is *how* each one gets there, which is exactly where
they differ and where NDT's implementation had been broken.

Four figures:

1. ``ch7_icp_correspondences`` -- the correspondence set of Eq. (7.11) drawn as
   lines between the scans, before and after alignment, with the residual of
   Eq. (7.10) falling across iterations.
2. ``ch7_ndt_voxels``          -- the per-voxel Gaussians of Eqs. (7.12)-(7.13)
   as covariance ellipses over the target scan. This is the map ICP does not
   build: NDT replaces point correspondences with a piecewise-Gaussian surface.
3. ``ch7_ndt_score_surface``   -- the Eq. (7.16) objective over a grid of
   translations, sliced at the true yaw, with two ndt_align runs marked. The
   surface *steps* at voxel boundaries, which is why one-sided 1e-6 finite
   differences reported gradients of ~3.6e4 at the optimum and sent the old
   ndt_align hundreds of metres away.

   Those steps still bite. Steepest descent with a backtracking line search
   reaches the optimum at step_size 0.05 and 0.5, but stalls after three
   iterations at the 0.1 default and at 0.3 -- the outcome is not monotonic
   in step length, and every run reports converged=True regardless, because
   "converged" means the line search stopped improving, not that the answer
   is right. ICP recovers this same displacement to 0.004 m.

   The slice is taken at the true yaw deliberately. A yaw = 0 slice looks
   perfectly plausible but scores ~255 where the true pose scores ~32, so it
   would put the visible minimum in the wrong place.
4. ``ch7_convergence_basin``   -- how far the initial guess can be displaced
   before each method fails, sampled over a grid of starting offsets, plus
   NDT's capture range as a function of voxel size. ICP with a generous
   correspondence gate recovers from nearly anywhere in the sampled range;
   NDT's basin is roughly one voxel wide, and widens monotonically as the
   voxels grow (measured: 4, 17, 40, 45 of 81 starts for 0.5, 1, 2, 3 m).

Run:
    python -m ch7_slam.example_scan_matching_visualization --no-show

Author: Li-Ta Hsu
References: Chapter 7, Sections 7.3.1-7.3.2, Eqs. (7.10)-(7.16)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from core.eval import save_figure
from core.slam.ndt import build_ndt_map, ndt_align, ndt_score
from core.slam.scan_generation import generate_scan_with_occlusion
from core.slam.scan_matching import (
    align_svd,
    compute_icp_residual,
    find_correspondences,
    icp_point_to_point,
)
from core.slam.se2 import se2_apply, se2_compose

FIGS_DIR = Path(__file__).parent / "figs"

# A simple room: four walls plus an interior partition, so scans have enough
# structure to constrain both translation and rotation.
WALLS = [
    (np.array([0.0, 0.0]), np.array([10.0, 0.0])),
    (np.array([10.0, 0.0]), np.array([10.0, 8.0])),
    (np.array([10.0, 8.0]), np.array([0.0, 8.0])),
    (np.array([0.0, 8.0]), np.array([0.0, 0.0])),
    (np.array([4.0, 0.0]), np.array([4.0, 3.5])),
]

TRUE_POSE = np.array([5.0, 4.0, 0.0])
# The displacement the matcher has to recover.
TRUE_MOTION = np.array([0.6, -0.4, np.deg2rad(8.0)])


def _make_scan_pair(seed: int = 0):
    """Build a target scan and a source scan displaced by TRUE_MOTION.

    Returns:
        Tuple of (target_scan, source_scan) in the sensor frame, shape (N, 2).
    """
    # generate_scan_with_occlusion draws its range noise from the global numpy
    # RNG and takes no seed argument, so seed globally to keep figures
    # reproducible between runs.
    np.random.seed(seed)
    target = generate_scan_with_occlusion(
        TRUE_POSE, WALLS, num_rays=240, noise_std=0.015
    )
    moved_pose = np.array(
        [
            TRUE_POSE[0] + TRUE_MOTION[0],
            TRUE_POSE[1] + TRUE_MOTION[1],
            TRUE_POSE[2] + TRUE_MOTION[2],
        ]
    )
    source = generate_scan_with_occlusion(
        moved_pose, WALLS, num_rays=240, noise_std=0.015
    )
    return target, source


def plot_icp_correspondences(max_pairs: int = 45) -> plt.Figure:
    """Figure 1: the correspondence set of (7.11) and the residual of (7.10).

    Args:
        max_pairs: How many correspondence lines to draw, for legibility.

    Returns:
        The matplotlib figure.
    """
    target, source = _make_scan_pair()

    # Re-run ICP one iteration at a time so the residual history is available.
    # Each step is composed on the SE(2) group, not added componentwise.
    pose = np.zeros(3)
    residuals = []
    for _ in range(25):
        moved = se2_apply(pose, source)
        matched_source, matched_target, _ = find_correspondences(
            moved, target, max_distance=1.0
        )
        if len(matched_source) < 3:
            break
        residuals.append(compute_icp_residual(matched_source, matched_target))
        step = align_svd(matched_source, matched_target)
        pose = se2_compose(step, pose)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, applied, title in (
        (axes[0], np.zeros(3), "Before: initial guess = identity"),
        (axes[1], pose, "After: ICP converged"),
    ):
        moved = se2_apply(applied, source)
        matched_source, matched_target, _ = find_correspondences(
            moved, target, max_distance=1.0
        )
        ax.plot(target[:, 0], target[:, 1], ".", color="#1f77b4",
                markersize=3, label="target scan")
        ax.plot(moved[:, 0], moved[:, 1], ".", color="#d62728",
                markersize=3, label="source scan")
        stride = max(1, len(matched_source) // max_pairs)
        for src_pt, tgt_pt in zip(matched_source[::stride],
                                  matched_target[::stride]):
            ax.plot([src_pt[0], tgt_pt[0]], [src_pt[1], tgt_pt[1]],
                    "-", color="0.4", linewidth=0.7, alpha=0.8)
        ax.set_title(
            f"{title}\n{len(matched_source)} correspondences, Eq. (7.11)",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("y [m]")

    axes[2].semilogy(np.arange(1, len(residuals) + 1), residuals,
                     "o-", color="#2ca02c", markersize=4)
    axes[2].set_title("Eq. (7.10) objective per iteration", fontsize=10)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("sum of squared distances")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        "ICP, Section 7.3.1: grey lines are the correspondences of Eq. (7.11), "
        "re-associated every iteration",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_ndt_voxels(voxel_size: float = 1.0) -> plt.Figure:
    """Figure 2: the voxel Gaussians of (7.12)-(7.13).

    Args:
        voxel_size: Voxel edge length in metres.

    Returns:
        The matplotlib figure.
    """
    target, _ = _make_scan_pair()
    ndt_map = build_ndt_map(target, voxel_size=voxel_size)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(target[:, 0], target[:, 1], ".", color="0.55", markersize=3,
            label="target scan", zorder=1)

    for cell in ndt_map.values():
        mean = np.asarray(cell["mean"], dtype=float)
        cov = np.asarray(cell["cov"], dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-9)
        # 2-sigma ellipse: width/height are full axes, hence the factor 2.
        angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
        ellipse = Ellipse(
            xy=mean,
            width=2.0 * 2.0 * np.sqrt(eigenvalues[-1]),
            height=2.0 * 2.0 * np.sqrt(eigenvalues[0]),
            angle=angle,
            facecolor="#1f77b4",
            alpha=0.28,
            edgecolor="#1f77b4",
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(ellipse)
        ax.plot(mean[0], mean[1], "+", color="#d62728", markersize=7, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.set_title(
        f"NDT map, Eqs. (7.12)-(7.13): {len(ndt_map)} voxels of "
        f"{voxel_size:g} m, 2-sigma ellipses\n"
        "crosses are the voxel means. Thin ellipses hug a wall -- almost no "
        "variance across it,\nplenty along it -- so NDT constrains motion "
        "perpendicular to a wall far better than\nmotion along it. The fatter, "
        "tilted cells are where one voxel straddles a corner.",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_ndt_score_surface(voxel_size: float = 1.0,
                           half_width: float = 1.6,
                           resolution: int = 90) -> plt.Figure:
    """Figure 3: the (7.16) objective surface and the descent path.

    Args:
        voxel_size: Voxel edge length in metres.
        half_width: Half-extent of the translation grid, in metres.
        resolution: Grid samples per axis.

    Returns:
        The matplotlib figure.
    """
    target, source = _make_scan_pair()
    ndt_map = build_ndt_map(target, voxel_size=voxel_size)

    # Sweep translation at the true yaw. A yaw=0 slice looks reasonable but
    # never contains the optimum -- its best score is ~300 against the true
    # pose's -4, so the picture would imply ndt_align had missed.
    true_yaw = TRUE_MOTION[2]
    offsets = np.linspace(-half_width, half_width, resolution)
    scores = np.empty((resolution, resolution))
    for row, dy in enumerate(offsets):
        for col, dx in enumerate(offsets):
            scores[row, col] = ndt_score(
                source, ndt_map, np.array([dx, dy, true_yaw]), voxel_size
            )

    # Two runs that differ only in step size. On a stepped objective the
    # outcome is not monotonic in step length: 0.05 and 0.5 both reach the
    # optimum, while the 0.1 default stalls after three iterations.
    stalled_pose, stalled_iters, stalled_score, _ = ndt_align(
        source, target, voxel_size=voxel_size, step_size=0.1
    )
    aligned_pose, iterations, final_score, converged = ndt_align(
        source, target, voxel_size=voxel_size, step_size=0.5
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    mesh = axes[0].pcolormesh(
        offsets, offsets, scores, shading="auto", cmap="viridis"
    )
    fig.colorbar(mesh, ax=axes[0], label="NDT score (lower is better)")
    axes[0].plot(TRUE_MOTION[0], TRUE_MOTION[1], "*", color="white",
                 markersize=16, markeredgecolor="k",
                 label="true alignment")
    axes[0].plot(aligned_pose[0], aligned_pose[1], "o", color="#d62728",
                 markersize=8, markeredgecolor="k",
                 label="ndt_align, step 0.5 (converges)")
    axes[0].plot(stalled_pose[0], stalled_pose[1], "X", color="#ff00ff",
                 markersize=10, markeredgecolor="k",
                 label="ndt_align, step 0.1 (stalls)")
    axes[0].plot(0.0, 0.0, "s", color="#ff7f0e", markersize=8,
                 markeredgecolor="k", label="initial guess")
    axes[0].set_xlabel("translation x [m]")
    axes[0].set_ylabel("translation y [m]")
    axes[0].set_title(
        f"Eq. (7.16) objective over translation, sliced at the true yaw "
        f"({np.rad2deg(true_yaw):.0f} deg)\n"
        f"step 0.5: {iterations} iters, score {final_score:.1f}   |   "
        f"step 0.1: {stalled_iters} iters, score {stalled_score:.0f} "
        f"(both report converged)",
        fontsize=10,
    )
    axes[0].legend(fontsize=8, loc="upper right")

    # Slice through the optimum so the basin and the roughness are both visible.
    row_at_truth = int(np.argmin(np.abs(offsets - TRUE_MOTION[1])))
    axes[1].plot(offsets, scores[row_at_truth, :], "-", color="#1f77b4",
                 linewidth=1.6)
    axes[1].axvline(TRUE_MOTION[0], color="#d62728", linestyle="--",
                    linewidth=1.2, label="true alignment")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel(
        f"translation x [m]  (y = {offsets[row_at_truth]:.2f} m, true yaw)"
    )
    axes[1].set_ylabel("NDT score")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(
        "Slice through the optimum. The basin is deep and clean, but the\n"
        "surface steps: points cross voxel boundaries and ndt_score divides\n"
        "by the number matched, so the value jumps (note x = -0.45).\n"
        "Finite differences have to straddle those steps to mean anything.",
        fontsize=9.5,
    )

    fig.suptitle(
        "NDT, Section 7.3.2: on a stepped objective the step size decides the "
        "answer, and not monotonically -- 0.05 and 0.5 reach the optimum, 0.1 "
        "and 0.3 stall",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_convergence_basin(grid: int = 9, span: float = 1.6,
                           tolerance: float = 0.25) -> plt.Figure:
    """Figure 4: which initial guesses each method recovers from.

    Args:
        grid: Samples per axis over the initial-offset grid.
        span: Half-extent of the initial-offset grid, in metres.
        tolerance: Position error below which a run counts as converged.

    Returns:
        The matplotlib figure.
    """
    target, source = _make_scan_pair()
    truth = TRUE_MOTION[:2].copy()

    offsets = np.linspace(-span, span, grid)
    icp_ok = np.zeros((grid, grid), dtype=bool)
    ndt_ok = np.zeros((grid, grid), dtype=bool)

    for row, dy in enumerate(offsets):
        for col, dx in enumerate(offsets):
            guess = np.array([dx, dy, 0.0])

            pose, _, _, _ = icp_point_to_point(
                source, target, initial_pose=guess.copy(),
                max_correspondence_distance=1.0
            )
            icp_ok[row, col] = np.linalg.norm(pose[:2] - truth) < tolerance

            pose, _, _, _ = ndt_align(
                source, target, initial_pose=guess.copy(), voxel_size=1.0
            )
            ndt_ok[row, col] = np.linalg.norm(pose[:2] - truth) < tolerance

    # NDT's capture range is set by the voxel size: coarse cells reach further
    # but localise less precisely. Measure it rather than assert it.
    voxel_sizes = (0.5, 1.0, 2.0, 3.0)
    ndt_counts = []
    for voxel_size in voxel_sizes:
        count = 0
        for dy in offsets:
            for dx in offsets:
                pose, _, _, _ = ndt_align(
                    source, target, initial_pose=np.array([dx, dy, 0.0]),
                    voxel_size=voxel_size
                )
                count += np.linalg.norm(pose[:2] - truth) < tolerance
        ndt_counts.append(count)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    for ax, mask, name in (
        (axes[0], icp_ok, "ICP, max correspondence 1.0 m"),
        (axes[1], ndt_ok, "NDT, voxel 1.0 m"),
    ):
        ax.pcolormesh(offsets, offsets, mask.astype(float), shading="auto",
                      cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.plot(truth[0], truth[1], "*", color="white", markersize=16,
                markeredgecolor="k")
        ax.set_aspect("equal")
        ax.set_xlabel("initial x offset [m]")
        ax.set_title(
            f"{name}\n{mask.sum()} of {mask.size} starts converge "
            f"(within {tolerance:g} m)",
            fontsize=10,
        )
    axes[0].set_ylabel("initial y offset [m]")

    axes[2].plot(voxel_sizes, ndt_counts, "o-", color="#1f77b4",
                 markersize=6, label="NDT")
    axes[2].axhline(icp_ok.sum(), color="#d62728", linestyle="--",
                    linewidth=1.4, label="ICP (for reference)")
    axes[2].set_xlabel("NDT voxel size [m]")
    axes[2].set_ylabel(f"starts converging (of {icp_ok.size})")
    axes[2].set_ylim(0, icp_ok.size)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=9, loc="upper left")
    axes[2].set_title(
        "NDT's capture range grows with voxel size:\n"
        "coarse cells reach further but localise less precisely",
        fontsize=10,
    )

    fig.suptitle(
        "Convergence basin: green recovers the true alignment, red does not. "
        "The star marks the answer.",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Generate and save the Chapter 7 scan-matching figures."""
    parser = argparse.ArgumentParser(
        description="Chapter 7 scan matching visualizations"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save figures without displaying"
    )
    parser.add_argument(
        "--out-dir", default=str(FIGS_DIR), help="Output directory for figures"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 7, Section 7.3: Scan Matching Visualization")
    print("=" * 70)
    print(f"True motion between scans: dx={TRUE_MOTION[0]:.2f} m, "
          f"dy={TRUE_MOTION[1]:.2f} m, dyaw={np.rad2deg(TRUE_MOTION[2]):.1f} deg")
    print()

    figures = [
        ("ch7_icp_correspondences", plot_icp_correspondences()),
        ("ch7_ndt_voxels", plot_ndt_voxels()),
        ("ch7_ndt_score_surface", plot_ndt_score_surface()),
        ("ch7_convergence_basin", plot_convergence_basin()),
    ]

    for name, fig in figures:
        paths = save_figure(fig, args.out_dir, name)
        print(f"  saved {name}: {', '.join(p.suffix.lstrip('.') for p in paths)}")

    print()
    print(f"Figures written to {args.out_dir}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
