"""Where an iterative solve starts, and why that is usually the wrong thing to blame.

Chapter 4, Section 4.4. A Gauss-Newton positioning solve needs an initial guess, and when it
fails the reflex is to blame the guess. This example sweeps the guess over the whole floor,
twice, changing nothing but the space the residual is formed in.

The measurements here are NOISELESS, so every error on this page is the solver.

Two parameterisations of one measurement model:

  residual="tan"     z = tan(psi), Eq. (4.64) written literally
  residual="angle"   wrap(psi_measured - atan2(dE, dN))   [the default]

The tan form carries two defects that no starting point repairs. `tan` has period pi, so an
anchor ahead and an anchor behind produce the same measurement and the residual cannot tell
them apart. And as the estimate runs to infinity every anchor tends to the same bearing, so
the tan residuals *shrink* on the way out -- infinity is an attractor, and Gauss-Newton
arrives there reporting success. The fourth panel traces one such run.

Both questions to ask of any residual are answered by the wrapped-angle form and failed by
the tan form: **is it bounded?** and **does the cost stay large when the estimate is far
wrong?**

WHAT THE SWEEP ACTUALLY SHOWS -- measured, and not what this example was written expecting.
Over 1681 seeds the wrapped-angle form fails 341 times against tan's 785, so the honest
headline is 2.3x, not "the basin disappears". What it removes is the QUIET class, and that it
removes completely: seeds that stall at the guess (82 -> 0) and seeds that stop somewhere
plausible but wrong (181 -> 0). What survives is the loud one -- a seed far outside the room
still walks off under either parameterisation, and 196 of those still set converged=True.

So the parameterisation makes the solver *honest*, not *safe*. Fixing the residual is worth
doing and is not a substitute for the four-condition failure test in `core.rf.solve_batch`:
the convergence flag is not a check, whichever residual you form.

Run:
    python -m ch4_rf_point_positioning.example_initial_guess_basin

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.63)-(4.65), (4.66)-(4.70). The behaviour pinned here is the
            same one asserted in tests/ch4_rf_point_positioning/test_aoa_initialisation_basin.py;
            this example is its picture.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Run as a script, sys.path[0] is THIS directory, so `core` resolves to whatever is
# installed -- another clone, a stale editable install -- or fails outright on a fresh
# one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import resolve_figs_dir, save_figure, show_figures_if_requested
from core.rf import AOAPositioner, aoa_azimuth, solve_batch
from core.rf.positioning import STALL_M

FIGS_DIR = Path(__file__).parent / "figs"

#: Four anchors on a square room -- a geometry that is not the problem here. The collinear
#: array in `example_comparison` fails for reasons of geometry; this one does not.
ANCHORS = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

#: One target, off-centre AND off the seed lattice. At (3.0, 7.0) one seed of the 1681 was
#: the answer exactly, so the solver never moved and `solve_batch` scored it -- correctly, by
#: its own definition -- as a stall. That is a classification artifact rather than a failure,
#: and it is also the rule against seeding a solver with the ground truth, arrived at by
#: accident. Off-lattice makes the collision impossible instead of subtracting it later.
TRUTH = np.array([3.2, 6.8])

#: The seed grid extends well outside the room: a cold start does not know where it is.
GRID_MIN, GRID_MAX, GRID_STEP = -5.0, 15.0, 0.5

#: Noiseless measurements converge to ~1e-8 m, so a millimetre is a generous "solved".
SOLVED_M = 1e-3

#: Outcome codes, in the order they are drawn.
SOLVED, STALLED, WRONG, DIVERGED, RAISED = 0, 1, 2, 3, 4
LABELS = {
    SOLVED: "solved",
    STALLED: "stalled at the seed",
    WRONG: "converged, wrong place",
    DIVERGED: "diverged (>100 m)",
    RAISED: "solver raised",
}
COLOURS = ["#2E7D32", "#F9A825", "#C62828", "#6A1B9A", "#455A64"]

#: Separate palette for the two PARAMETERISATIONS. Reusing the outcome colours for the bars
#: put "tan" in the green of "solved" and "angle" in the purple of "diverged" on the same
#: figure -- two meanings for one colour, caught by looking at the render.
SERIES = ["#37474F", "#0277BD"]


def measurements(truth=TRUTH):
    """Noiseless AOA azimuths from every anchor to the target."""
    return np.array([aoa_azimuth(a, truth) for a in ANCHORS])


def seed_grid():
    """The initial guesses to try, as a meshgrid and as a flat (N, 2) list."""
    axis = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP)
    xx, yy = np.meshgrid(axis, axis)
    seeds = np.column_stack([xx.ravel(), yy.ravel()])
    assert (
        np.min(np.linalg.norm(seeds - TRUTH, axis=1)) > STALL_M
    ), "a seed coincides with the target: it cannot move, and would be scored a stall"
    return axis, xx, yy, seeds


def sweep(residual, verbose=True):
    """Solve the same fix from every seed and classify each outcome.

    Classification goes through `core.rf.solve_batch` rather than being re-derived here:
    it is the four failure conditions applied once (raised / converged=False / never left
    the seed / landed beyond the divergence threshold), and re-implementing them is the
    recurring bug this repository has already paid for.
    """
    axis, xx, yy, seeds = seed_grid()
    meas = measurements()
    truth = TRUTH[None, :]
    solver = AOAPositioner(ANCHORS)

    codes = np.empty(len(seeds), dtype=int)
    errors = np.empty(len(seeds))
    claimed = np.zeros(len(seeds), dtype=bool)
    for i, seed in enumerate(seeds):
        out = solve_batch(solver, meas[None, :], seed, truth, residual=residual)
        err = float(out.errors[0])
        errors[i] = err
        claimed[i] = bool(out.converged[0])
        if np.isnan(err):
            codes[i] = RAISED
        elif bool(out.stalled[0]):
            codes[i] = STALLED
        elif err > out.divergence_m:
            codes[i] = DIVERGED
        elif err > SOLVED_M:
            codes[i] = WRONG
        else:
            codes[i] = SOLVED

    result = {
        "residual": residual,
        "axis": axis,
        "xx": xx,
        "yy": yy,
        "seeds": seeds,
        "codes": codes.reshape(xx.shape),
        "errors": errors.reshape(xx.shape),
        "claimed": claimed.reshape(xx.shape),
        # the failures that lied: wrong answer, convergence flag set
        "silent": int(np.sum((codes != SOLVED) & (codes != RAISED) & claimed)),
        "counts": {c: int(np.sum(codes == c)) for c in LABELS},
        "n": len(seeds),
    }
    if verbose:
        n_bad = result["n"] - result["counts"][SOLVED]
        print(
            f"\n  residual={residual!r}: {n_bad}/{result['n']} seeds failed to reach "
            f"the target"
        )
        for c, name in LABELS.items():
            if result["counts"][c]:
                print(f"      {name:<24} {result['counts'][c]:>5}")
        finite = errors[np.isfinite(errors)]
        print(f"      median error            {np.median(finite):>10.2e} m")
        print(f"      worst error             {finite.max():>10.2e} m")
        print(f"      of those failures, {result['silent']} reported converged=True")
    return result


def trace_worst(result):
    """Re-solve the worst SILENT failure and return its iterate path.

    Deliberately not the worst failure outright: the largest error in this sweep reports
    `converged=False`, which is the solver behaving correctly and is not the thing worth a
    panel. The interesting run is the furthest one that still set the flag.

    `AOAPositioner.solve` returns the iterates in `info["history"]`, so the walk out to
    infinity can be drawn rather than described.
    """
    flat = result["errors"].ravel()
    finite = np.where(np.isfinite(flat), flat, -np.inf)
    lying = result["claimed"].ravel() & (result["codes"].ravel() == DIVERGED)
    ranked = np.where(lying, finite, -np.inf)
    if not np.isfinite(ranked).any():  # no silent failure: fall back to the worst
        ranked = finite
    seed = result["seeds"][int(np.argmax(ranked))]
    _, info = AOAPositioner(ANCHORS).solve(
        measurements(), initial_guess=seed, residual=result["residual"]
    )
    return seed, np.asarray(info["history"]), bool(info["converged"])


def plot_basin(ax, result):
    """One basin map: every seed coloured by what the solve did from there."""
    cmap = ListedColormap(COLOURS)
    ax.pcolormesh(
        result["xx"],
        result["yy"],
        result["codes"],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(COLOURS) - 0.5,
        shading="auto",
    )
    ax.plot(ANCHORS[:, 0], ANCHORS[:, 1], "k^", ms=9, label="anchors")
    ax.plot(*TRUTH, "w*", ms=18, mec="k", mew=1.2, label="target")
    solved = result["counts"][SOLVED]
    ax.set_title(
        f'residual="{result["residual"]}"   '
        f'{result["n"] - solved}/{result["n"]} seeds fail',
        fontsize=11,
    )
    ax.set_xlabel("initial guess x (m)")
    ax.set_ylabel("initial guess y (m)")
    ax.set_aspect("equal")


def plot_failure_rates(ax, results):
    """Failure modes side by side.

    Its own panel, because an accuracy figure cannot say "this did not work" -- and a
    method that failed everywhere must never be drawn as a zero-height bar.
    """
    codes = [c for c in LABELS if any(r["counts"][c] for r in results)]
    width = 0.8 / len(results)
    for k, r in enumerate(results):
        pos = np.arange(len(codes)) + k * width - 0.4 + width / 2
        pct = [100 * r["counts"][c] / r["n"] for c in codes]
        bars = ax.bar(
            pos,
            pct,
            width,
            label=f'residual="{r["residual"]}"',
            color=SERIES[k],
            edgecolor="black",
            linewidth=0.5,
        )
        for b, v, c in zip(bars, pct, codes, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                max(v, 0) + 1.5,
                f"{r['counts'][c]}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels([LABELS[c].replace(" ", "\n", 1) for c in codes], fontsize=9)
    ax.set_ylabel("% of seeds")
    ax.set_ylim(0, 108)
    ax.set_title("Same measurements, same seeds, one line of difference", fontsize=11)
    ax.legend(fontsize=9)


def plot_trace(ax, seed, history, converged):
    """The walk to infinity, with the flag it set on arrival."""
    d = np.linalg.norm(history - TRUTH, axis=1)
    ax.semilogy(np.arange(len(d)), np.maximum(d, 1e-12), "o-", color="#6A1B9A", ms=4)
    ax.axhline(
        100.0, color="#C62828", ls="--", lw=1, label="divergence threshold, 100 m"
    )
    ax.set_xlabel("Gauss-Newton iteration")
    ax.set_ylabel("distance from the target (m)")
    ax.set_title(
        f"seed ({seed[0]:.1f}, {seed[1]:.1f}) -> {d[-1]:.1e} m, "
        f"converged={converged}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")


def plot_summary(results, trace):
    """The whole story on one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.4))
    plot_basin(axes[0, 0], results[0])
    plot_basin(axes[0, 1], results[1])
    plot_failure_rates(axes[1, 0], results)
    plot_trace(axes[1, 1], *trace)

    # only the outcomes that actually occurred: an unused "solver raised" patch sat in a
    # grey almost identical to the tan bars, which is a second meaning for one colour
    seen = [c for c in LABELS if any(r["counts"][c] for r in results)]
    handles = [
        Patch(facecolor=COLOURS[c], edgecolor="black", label=LABELS[c]) for c in seen
    ]
    handles += [
        plt.Line2D([], [], color="k", marker="^", ls="", label="anchor"),
        plt.Line2D(
            [], [], color="w", marker="*", mec="k", ls="", ms=12, label="target"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "An initial-guess problem that is not about the initial guess\n"
        "AOA, four anchors, zero measurement noise",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    return fig


def main() -> None:
    """Sweep both parameterisations and write the figure."""
    parser = argparse.ArgumentParser(
        description="Initial-guess basin for AOA positioning (Chapter 4)"
    )
    parser.add_argument(
        "--out-dir", default=str(FIGS_DIR), help="Output directory for figures"
    )
    args = parser.parse_args()

    axis, *_ = seed_grid()
    print("=" * 70)
    print("Chapter 4: the initial guess is not usually the problem")
    print("=" * 70)
    print(
        f"  {len(axis)}x{len(axis)} seeds over [{GRID_MIN:.0f}, {GRID_MAX:.0f}] m, "
        f"target at ({TRUTH[0]:.1f}, {TRUTH[1]:.1f}), zero measurement noise"
    )

    tan_r = sweep("tan")
    ang_r = sweep("angle")

    bad_tan = tan_r["n"] - tan_r["counts"][SOLVED]
    bad_ang = ang_r["n"] - ang_r["counts"][SOLVED]
    quiet_tan = tan_r["counts"][STALLED] + tan_r["counts"][WRONG]
    quiet_ang = ang_r["counts"][STALLED] + ang_r["counts"][WRONG]

    print("\n  " + "-" * 66)
    print(f"  {'':30}{'tan(psi)':>12}{'wrap(angle)':>14}")
    print(f"  {'seeds that fail':30}{bad_tan:>12}{bad_ang:>14}")
    print(f"  {'  quiet: stalled or plausible':30}{quiet_tan:>12}{quiet_ang:>14}")
    print(
        f"  {'  loud: walked off past 100 m':30}"
        f"{tan_r['counts'][DIVERGED]:>12}{ang_r['counts'][DIVERGED]:>14}"
    )
    print(
        f"  {'failures claiming converged':30}"
        f"{tan_r['silent']:>12}{ang_r['silent']:>14}"
    )
    print(
        f"\n  Overall {bad_tan / max(bad_ang, 1):.1f}x fewer failures -- but the honest "
        f"statement is narrower:"
    )
    print(
        f"  the wrapped-angle form removes the QUIET failures ({quiet_tan} -> "
        f"{quiet_ang}), the ones"
    )
    print(
        "  that look like answers. Seeds far outside the room still walk off under both,"
    )
    print(
        f"  and {ang_r['silent']} of those still set converged=True. The residual fix "
        f"makes the"
    )
    print(
        "  solver honest, not safe -- the four-condition test is still what catches it."
    )

    seed, history, converged = trace_worst(tan_r)
    print(
        f"\n  Worst SILENT tan failure: seed ({seed[0]:.1f}, {seed[1]:.1f}) walked to "
        f"{np.linalg.norm(history[-1] - TRUTH):.2e} m"
    )
    print(f"  in {len(history) - 1} iterations and reported converged={converged}.")

    paths = save_figure(
        plot_summary([tan_r, ang_r], (seed, history, converged)),
        args.out_dir,
        "ch4_initial_guess_basin",
    )
    print(
        f"\n  saved ch4_initial_guess_basin: "
        f"{', '.join(p.suffix.lstrip('.') for p in paths)}"
    )
    plt.close("all")
    print(f"Figures written to {resolve_figs_dir(args.out_dir)}")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
