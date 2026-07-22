"""Bimodal posteriors: what a particle filter can represent, Chapter 3 Section 3.3.

Section 3.3 motivates the particle filter by its ability to carry an arbitrary
posterior, not just a Gaussian one. The chapter's existing examples never show
that, because their scenarios are strongly over-determined: with four anchors
and tight ranges every estimator here converges in a *single* update. Measured
on the shipped comparison scenario, the particle cloud's spread collapses 37x
between step 0 and step 1 and then barely moves again -- and the EKF
range-bearing example collapses 19x in one step likewise. There is nothing to
watch.

The interesting regime is the under-determined one. With only **two**
range-only anchors, the two range circles intersect twice, so the posterior is
genuinely **bimodal**: the target is either above or below the anchor baseline
and the measurements cannot say which. A Kalman filter cannot represent that
state of knowledge at all -- a single Gaussian has one peak. A particle filter
simply keeps particles in both places.

This example runs exactly that, then adds a third anchor half way through:

- **Steps 1-14, two anchors.** The cloud splits either side of the baseline and
  stays split: both modes hold particles on 13 of those 14 steps, with the
  share above the baseline sloshing between 8% and 99% as successive ranges
  happen to favour one side or the other. Neither hypothesis dies, because
  neither is contradicted. Cloud spread stays in the range 0.4 to 2.8.
- **Step 15, a third anchor appears.** The mirror hypothesis becomes instantly
  inconsistent, every particle moves to the correct side, and the spread drops
  by roughly 50x to 0.05-0.11.

The most useful lesson is what happens to the *mean*. Averaged over the bimodal
phase the weighted mean sits 4.60 m from truth while the nearest mode is 0.38 m
away -- it is parked in the empty space between two clouds, a position the
target is not merely unlikely to occupy but cannot occupy. Reporting a mean and
a covariance, which is all a Kalman filter can do, is actively misleading here.
Once the third anchor resolves the ambiguity the distribution is unimodal and
mean and mode agree exactly (0.26 m each).

Run:
    python -m ch3_estimators.example_particle_bimodal
    python -m ch3_estimators.example_particle_bimodal --animate

Author: Li-Ta Hsu
References: Chapter 3, Section 3.3 (particle filter), Eqs. (3.32)-(3.34)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.estimators import ParticleFilter
from core.eval import save_animation, save_figure

FIGS_DIR = Path(__file__).parent / "figs"

DT = 0.5
N_STEPS = 30
RANGE_STD = 0.4
N_PARTICLES = 800
SEED = 7

# Two anchors on a baseline along y = 0: their range circles intersect twice,
# mirrored about that baseline. The third breaks the symmetry.
ANCHORS_TWO = np.array([[0.0, 0.0], [20.0, 0.0]])
ANCHORS_THREE = np.array([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]])
THIRD_ANCHOR_STEP = 15

TRUE_X0 = np.array([6.0, 8.0, 0.8, 0.2])
PROCESS_NOISE = np.diag([0.02, 0.02, 0.01, 0.01])

TRANSITION = np.array(
    [[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
)


def _anchors_at(step):
    """Anchors visible at a given step."""
    return ANCHORS_TWO if step < THIRD_ANCHOR_STEP else ANCHORS_THREE


def _make_likelihood(anchors):
    """Gaussian range likelihood, Eq. (3.34), for a given anchor set."""

    def likelihood(z, x):
        predicted = np.array([np.linalg.norm(x[:2] - a) for a in anchors])
        return np.exp(-0.5 * np.sum(((z - predicted) / RANGE_STD) ** 2))

    return likelihood


def run_bimodal_scenario(seed: int = SEED):
    """Run the two-anchor then three-anchor particle filter scenario.

    Returns:
        Dictionary with truth, per-step particle clouds, means, mode means,
        errors and the anchor count at each step.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        truth = [TRUE_X0.copy()]
        for _ in range(N_STEPS):
            truth.append(TRANSITION @ truth[-1])
        truth = np.array(truth)

        def process_model(x, u, dt):
            """Eq. (3.33): sample from the transition prior."""
            return TRANSITION @ x + np.random.multivariate_normal(
                np.zeros(4), PROCESS_NOISE
            )

        # A deliberately vague prior: the filter is not told which side it is on.
        particle_filter = ParticleFilter(
            process_model,
            _make_likelihood(ANCHORS_TWO),
            N_PARTICLES,
            np.array([10.0, 0.0, 0.0, 0.0]),
            np.diag([80.0, 80.0, 1.0, 1.0]),
        )

        clouds, means, upper_means, lower_means = [], [], [], []
        fraction_above, n_anchors = [], []
        error_mean, error_best_mode = [], []

        for step in range(1, N_STEPS + 1):
            anchors = _anchors_at(step)
            particle_filter.likelihood_func = _make_likelihood(anchors)

            ranges = np.array(
                [np.linalg.norm(truth[step][:2] - a) for a in anchors]
            )
            ranges = ranges + np.random.normal(0.0, RANGE_STD, len(anchors))

            particle_filter.predict(dt=DT)
            particle_filter.update(ranges)

            particles = particle_filter.get_particles()
            if isinstance(particles, tuple):
                particles = particles[0]
            particles = np.asarray(particles)

            above = particles[particles[:, 1] > 0][:, :2]
            below = particles[particles[:, 1] <= 0][:, :2]
            mean_xy = particles[:, :2].mean(axis=0)
            modes = [m.mean(axis=0) for m in (above, below) if len(m) > 5]

            clouds.append(particles[:, :2].copy())
            means.append(mean_xy)
            upper_means.append(above.mean(axis=0) if len(above) > 5 else None)
            lower_means.append(below.mean(axis=0) if len(below) > 5 else None)
            fraction_above.append(len(above) / len(particles))
            n_anchors.append(len(anchors))

            truth_xy = truth[step][:2]
            error_mean.append(np.linalg.norm(mean_xy - truth_xy))
            error_best_mode.append(
                min(np.linalg.norm(m - truth_xy) for m in modes)
                if modes else np.linalg.norm(mean_xy - truth_xy)
            )
    finally:
        np.random.set_state(rng_state)

    return {
        "truth": truth,
        "clouds": clouds,
        "means": np.array(means),
        "upper_means": upper_means,
        "lower_means": lower_means,
        "fraction_above": np.array(fraction_above),
        "n_anchors": np.array(n_anchors),
        "error_mean": np.array(error_mean),
        "error_best_mode": np.array(error_best_mode),
        "steps": np.arange(1, N_STEPS + 1),
    }


def _draw_frame(axes, scenario, index):
    """Render the three panels up to ``index`` (0-based step index)."""
    truth = scenario["truth"]
    steps = scenario["steps"]
    step = steps[index]
    anchors = _anchors_at(step)
    cloud = scenario["clouds"][index]

    for ax in axes:
        ax.clear()

    # --- particle cloud
    above = cloud[cloud[:, 1] > 0]
    below = cloud[cloud[:, 1] <= 0]
    axes[0].scatter(above[:, 0], above[:, 1], s=5, c="tab:blue", alpha=0.35,
                    label=f"particles above baseline ({len(above)})")
    axes[0].scatter(below[:, 0], below[:, 1], s=5, c="tab:purple", alpha=0.35,
                    label=f"particles below ({len(below)})")
    axes[0].plot(truth[: step + 1, 0], truth[: step + 1, 1], "k-",
                 linewidth=2.0, label="true trajectory")
    axes[0].scatter(*truth[step][:2], s=90, marker="*", c="lime",
                    edgecolors="black", zorder=6, label="true position")
    axes[0].scatter(*scenario["means"][index], s=70, marker="X", c="red",
                    edgecolors="black", zorder=6, label="posterior mean")
    axes[0].scatter(anchors[:, 0], anchors[:, 1], s=150, marker="^", c="orange",
                    edgecolors="black", linewidths=1.5, zorder=5,
                    label=f"{len(anchors)} anchors")
    axes[0].axhline(0.0, color="0.6", linestyle=":", linewidth=1.2)
    axes[0].set_xlim(-6, 26)
    axes[0].set_ylim(-16, 22)
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.25)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    # Upper left, deliberately: at 'lower right' the legend box lands squarely
    # on the mirror mode near (11, -9) and hides the figure's whole point.
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    axes[0].set_title(
        f"step {step} -- {len(anchors)} anchors"
        + ("   BIMODAL" if len(anchors) == 2 else "   resolved"),
        fontsize=10,
    )

    # --- how the two modes share the particles
    axes[1].plot(steps[: index + 1], scenario["fraction_above"][: index + 1],
                 "-", color="tab:blue", linewidth=1.8)
    axes[1].axhline(0.5, color="0.6", linestyle=":", linewidth=1.2)
    axes[1].axvline(THIRD_ANCHOR_STEP, color="tab:green", linestyle="--",
                    linewidth=1.4, label="third anchor appears")
    axes[1].set_xlim(steps[0], steps[-1])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("fraction of particles above baseline")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title(
        "two live hypotheses, then one", fontsize=10
    )

    # --- the punchline: the mean is not where the target is
    axes[2].plot(steps[: index + 1], scenario["error_mean"][: index + 1],
                 "-", color="red", linewidth=1.8, label="posterior mean")
    axes[2].plot(steps[: index + 1], scenario["error_best_mode"][: index + 1],
                 "-", color="tab:blue", linewidth=1.8, label="nearest mode")
    axes[2].axvline(THIRD_ANCHOR_STEP, color="tab:green", linestyle="--",
                    linewidth=1.4)
    axes[2].set_xlim(steps[0], steps[-1])
    axes[2].set_ylim(0, max(scenario["error_mean"].max() * 1.15, 1.0))
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("position error [m]")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_title(
        f"mean {scenario['error_mean'][index]:.2f} m   |   "
        f"mode {scenario['error_best_mode'][index]:.2f} m",
        fontsize=10,
    )


def animate_bimodal(scenario, n_frames: int = None):
    """Build the bimodal-posterior animation.

    Args:
        scenario: Output of :func:`run_bimodal_scenario`.
        n_frames: Frames to render; defaults to one per step.

    Returns:
        Tuple of (figure, update callback, frame count) for save_animation.
    """
    total = len(scenario["steps"])
    frames = total if n_frames is None else min(n_frames, total)
    indices = np.linspace(0, total - 1, frames, dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    def update(frame: int):
        _draw_frame(axes, scenario, indices[frame])
        fig.suptitle(
            "Particle filter, Section 3.3: two range-only anchors leave the "
            "posterior genuinely bimodal -- a shape no single Gaussian can hold",
            fontsize=11,
        )
        fig.tight_layout()
        return axes

    return fig, update, frames


def plot_bimodal_summary(scenario) -> plt.Figure:
    """Static counterpart: the bimodal phase beside the resolved one."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Draw the last bimodal step; panels 2 and 3 carry the full history.
    _draw_frame(axes, scenario, THIRD_ANCHOR_STEP - 2)

    # _draw_frame truncated panels 2 and 3 to that step; redraw them over the
    # complete run so the static figure shows the whole story at once.
    steps = scenario["steps"]
    axes[1].clear()
    axes[1].plot(steps, scenario["fraction_above"], "-", color="tab:blue",
                 linewidth=1.8)
    axes[1].axhline(0.5, color="0.6", linestyle=":", linewidth=1.2)
    axes[1].axvline(THIRD_ANCHOR_STEP, color="tab:green", linestyle="--",
                    linewidth=1.4, label="third anchor appears")
    axes[1].set_xlim(steps[0], steps[-1])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("fraction of particles above baseline")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("two live hypotheses, then one", fontsize=10)

    axes[2].clear()
    axes[2].plot(steps, scenario["error_mean"], "-", color="red",
                 linewidth=1.8, label="posterior mean")
    axes[2].plot(steps, scenario["error_best_mode"], "-", color="tab:blue",
                 linewidth=1.8, label="nearest mode")
    axes[2].axvline(THIRD_ANCHOR_STEP, color="tab:green", linestyle="--",
                    linewidth=1.4)
    axes[2].set_xlim(steps[0], steps[-1])
    axes[2].set_ylim(0, scenario["error_mean"].max() * 1.15)
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("position error [m]")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_title(
        "while bimodal, the mean sits between the modes -- in empty space",
        fontsize=10,
    )

    fig.suptitle(
        "Particle filter, Section 3.3: the left panel is the last bimodal step; "
        "the third anchor arrives at step "
        f"{THIRD_ANCHOR_STEP}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the scenario and write its figures."""
    parser = argparse.ArgumentParser(
        description="Bimodal particle-filter posterior (Chapter 3)"
    )
    parser.add_argument("--out-dir", default=str(FIGS_DIR),
                        help="Output directory for figures")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Also render the animation GIF (slower)")
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 3, Section 3.3: a posterior that is not Gaussian")
    print("=" * 70)
    print(f"Two range-only anchors for steps 1-{THIRD_ANCHOR_STEP - 1}, "
          f"a third from step {THIRD_ANCHOR_STEP}.\n")

    scenario = run_bimodal_scenario()

    bimodal = scenario["steps"] < THIRD_ANCHOR_STEP
    resolved = ~bimodal
    print(f"  While bimodal ({bimodal.sum()} steps):")
    print(f"    particles above baseline: "
          f"{scenario['fraction_above'][bimodal].min():.2f} to "
          f"{scenario['fraction_above'][bimodal].max():.2f}")
    print(f"    mean error      {scenario['error_mean'][bimodal].mean():.2f} m")
    print(f"    nearest-mode    "
          f"{scenario['error_best_mode'][bimodal].mean():.2f} m"
          f"   <- the mean is the misleading one")
    print(f"  After the third anchor ({resolved.sum()} steps):")
    print(f"    mean error      {scenario['error_mean'][resolved].mean():.2f} m")
    print(f"    nearest-mode    "
          f"{scenario['error_best_mode'][resolved].mean():.2f} m\n")

    paths = save_figure(plot_bimodal_summary(scenario), args.out_dir,
                        "ch3_particle_bimodal")
    print(f"  saved ch3_particle_bimodal: "
          f"{', '.join(p.suffix.lstrip('.') for p in paths)}")

    if args.animate:
        fig, update, n_frames = animate_bimodal(scenario)
        path = save_animation(fig, update, n_frames, args.out_dir,
                              "ch3_particle_bimodal", fps=4)
        plt.close(fig)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  saved {path.name}: {n_frames} frames, {size_mb:.2f} MB")

    plt.close("all")
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
