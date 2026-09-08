"""A collinear array leaves an exact mirror twin of every position -- and DOP cannot see it.

Chapter 4, Sections 4.2-4.4. `example_initial_guess_basin` sweeps the initial guess over a
SQUARE room and finds that the basin belongs to the residual, not the geometry (Eqs.
(4.63)-(4.70)). This example asks the geometry question directly: put the four beacons on
one line -- `data/sim/ch4_rf_2d_linear`, this repository's ``poor_geometry`` preset -- and
sweep the same lattice again.

**Ranges have a mirror twin.** A target at `(x, 10 + h)` and its reflection `(x, 10 - h)`
about the beacon line `y = 10` are the same distance from every beacon on that line
(TOA, Eqs. (4.1)-(4.3); TDOA differences the same ranges, Eqs. (4.27)-(4.33)), so an
iterative solve that reaches either one has, by every distance-based test, solved the
problem. `data/sim/ch4_rf_2d_linear/README.md`'s own "reflection ambiguity" section
documents this from the shipped dataset; this example measures where it lands on the
initial-guess lattice specifically, and shows that AOA -- which measures a bearing, not a
distance, Eqs. (4.63)-(4.66) -- is not exposed to it at all: reflecting a position flips the
sign of every azimuth.

**A binary solved/failed count would misreport this.** A seed that converges to the mirror
is not a failure by `core.rf.solve_batch`'s four-condition test (Section 4.5's DOP tools
cannot distinguish it from the truth either -- see the dataset README). So this sweep
scores every seed TRUTH / MIRROR / FAIL rather than solved / failed, and the beacon
centroid -- which sits exactly on the line of symmetry, where the range Jacobian has no
across-line column -- is measured, not assumed, to be the one seed shared by every method
that cannot move at all.

Run:
    python -m ch4_rf_point_positioning.example_reflection_ambiguity

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.1)-(4.3) TOA ranging, (4.27)-(4.33) TDOA, (4.63)-(4.66) AOA
            azimuth, Section 4.5 DOP. Companion to `example_initial_guess_basin.py`
            (Eqs. (4.63)-(4.70), the residual-space basin on the square room) and to
            `data/sim/ch4_rf_2d_linear/README.md`, which names this ambiguity from the
            shipped dataset rather than from a sweep.
"""

from __future__ import annotations

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
# one. Same fix as example_initial_guess_basin.py (see its comment, issue #86).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import resolve_figs_dir, save_figure, show_figures_if_requested
from core.rf import (
    AOAPositioner,
    TDOAPositioner,
    aoa_angle_vector,
    solve_batch,
    tdoa_measurement_vector,
    toa_range,
    toa_solve_with_clock_bias,
)
from core.rf.positioning import STALL_M

FIGS_DIR = Path(__file__).parent / "figs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sim" / "ch4_rf_2d_linear"

#: The corridor array: four beacons, all on one line. Loaded, not hard-coded, so the
#: figure cannot drift from the shipped dataset.
BEACONS = np.loadtxt(DATA_DIR / "beacons.txt")

#: Derived from BEACONS, not a separate hard-coded 10.0: a future regeneration of the
#: dataset with the beacons on a different line must move this line too, automatically,
#: rather than silently mirroring every position about the wrong one. The assert makes
#: "all on one line" a checked fact rather than a docstring claim.
assert np.allclose(
    BEACONS[:, 1], BEACONS[0, 1]
), f"beacons are not collinear in y: {BEACONS[:, 1].tolist()}"
LINE_Y = float(BEACONS[0, 1])

#: The dataset's 100-point ground truth grid; `choose_truth` below picks one point from it.
GROUND_TRUTH = np.loadtxt(DATA_DIR / "ground_truth_positions.txt")

#: Same convention as example_initial_guess_basin.py: seeds run well outside the room
#: (there, [-5,15] over a 10 m room; here, the room is 20 m, so the lattice keeps the
#: same relative extent and the same 41x41 = 1681 density).
GRID_MIN, GRID_MAX, GRID_STEP = -10.0, 30.0, 1.0

#: Same tolerance as example_initial_guess_basin.py's SOLVED_M, reused for the distance
#: to the mirror point too, so both states are judged on one yardstick.
SOLVED_M = 1e-3

#: Tri-state outcome codes, in the order they are drawn.
TRUTH_S, MIRROR_S, FAIL_S = 0, 1, 2
STATE_LABELS = {
    TRUTH_S: "converged to the TRUTH",
    MIRROR_S: "converged to the MIRROR (reflection about the beacon line)",
    FAIL_S: "FAIL: raised, refused, stalled, diverged, or solved elsewhere",
}
#: Okabe-Ito colours, separated in hue AND lightness so the map survives grayscale and a
#: covered legend. MIRROR also carries a hatch: it is the state a reader must not miss.
STATE_COLOURS = {TRUTH_S: "#009E73", MIRROR_S: "#F0E442", FAIL_S: "#4D4D4D"}
STATE_HATCH = {TRUTH_S: None, MIRROR_S: "///", FAIL_S: None}

#: Where the two validation seeds sit: the beacon centroid (on the line of symmetry) and
#: an off-line point the dataset README also uses.
CENTROID_SEED = BEACONS.mean(axis=0)
OFFLINE_SEED = np.array([10.0, 3.0])


class _ClockStateSolver:
    """`toa_solve_with_clock_bias` behind the `solve(m, initial_guess=...)` interface
    `core.rf.solve_batch` expects.

    Same small adapter as `example_initial_guess_basin.py`'s `_ClockStateSolver`, defined
    again here rather than imported: a chapter example is a leaf
    (tests/test_repo_conventions.py::test_chapter_module_does_not_import_its_sibling), so
    nothing under `ch4_rf_point_positioning/` imports another file in its own chapter.
    """

    def __init__(self, anchors: np.ndarray) -> None:
        self.anchors = np.asarray(anchors, dtype=float)

    def solve(self, ranges, initial_guess, **kwargs):
        state0 = np.concatenate([np.asarray(initial_guess, dtype=float), [0.0]])
        position, _bias_m, info = toa_solve_with_clock_bias(
            self.anchors, ranges, state0, **kwargs
        )
        return position, info


def choose_truth() -> tuple[np.ndarray, int]:
    """One off-line, off-lattice truth from the dataset's own ground truth.

    The dataset's first point, `(2.0, 2.0)`, sits exactly on the seed lattice (step 1.0),
    and so does its mirror `(2.0, 18.0)` -- the same collision
    `example_initial_guess_basin.py:71-76` documents and avoids by picking a target off
    the lattice. This is that rule, applied to both the point and its reflection: the
    first ground-truth row, in file order, whose position AND whose mirror both miss
    every lattice line.
    """
    axis = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP)
    for idx, point in enumerate(GROUND_TRUTH):
        mirror = np.array([point[0], 2 * LINE_Y - point[1]])
        off = np.min(np.abs(point[:, None] - axis[None, :]), axis=1) > STALL_M
        off_mirror = np.min(np.abs(mirror[:, None] - axis[None, :]), axis=1) > STALL_M
        if off.all() and off_mirror.all():
            return point.copy(), idx
    raise RuntimeError("no off-lattice ground-truth point found")


TRUTH, TRUTH_ROW = choose_truth()
MIRROR = np.array([TRUTH[0], 2 * LINE_Y - TRUTH[1]])


def seed_grid():
    """The initial guesses to try, as a meshgrid and as a flat (N, 2) list."""
    axis = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP)
    xx, yy = np.meshgrid(axis, axis)
    seeds = np.column_stack([xx.ravel(), yy.ravel()])
    assert (
        np.min(np.linalg.norm(seeds - TRUTH, axis=1)) > STALL_M
    ), "a seed coincides with the target: it cannot move, and would be scored a stall"
    assert (
        np.min(np.linalg.norm(seeds - MIRROR, axis=1)) > STALL_M
    ), "a seed coincides with the mirror solution: same artifact, MIRROR class"
    return axis, xx, yy, seeds


def measurements(truth=None):
    """Noiseless TOA / TDOA / AOA measurements from the corridor array to `truth`.

    Every one is the repository's own forward model, so the residual a solver forms is
    the exact model it inverts, and zero noise really is zero.
    """
    truth = TRUTH if truth is None else truth
    return {
        "toa": np.array([toa_range(b, truth) for b in BEACONS]),
        "tdoa": tdoa_measurement_vector(BEACONS, truth, reference_anchor_index=0),
        "aoa": aoa_angle_vector(BEACONS, truth),
    }


#: (label, measurement key, solver factory, extra solve_batch kwargs)
ARMS = [
    ("TOA + clock state", "toa", lambda: _ClockStateSolver(BEACONS), {}),
    ("TDOA", "tdoa", lambda: TDOAPositioner(BEACONS, reference_anchor_index=0), {}),
    (
        'AOA, residual="angle"',
        "aoa",
        lambda: AOAPositioner(BEACONS),
        {"residual": "angle"},
    ),
]


def sweep(label, key, make_solver, kwargs, verbose=True):
    """Solve the same fix from every seed and classify it TRUTH / MIRROR / FAIL.

    One `solve_batch` call per seed with a one-row measurement matrix, the same shape
    `example_initial_guess_basin.py`'s `sweep`/`sweep_arm` use. `SolveOutcome.solved`
    (the four-condition test) is used verbatim; the only addition is a second distance,
    to the mirror, which a symmetric geometry needs and the repo's own tools have no
    reason to compute.
    """
    axis, xx, yy, seeds = seed_grid()
    meas = measurements()[key]
    solver = make_solver()

    codes = np.empty(len(seeds), dtype=int)
    err_truth = np.empty(len(seeds))
    err_mirror = np.empty(len(seeds))
    stalled = np.zeros(len(seeds), dtype=bool)
    claimed = np.zeros(len(seeds), dtype=bool)

    for i, seed in enumerate(seeds):
        out = solve_batch(solver, meas[None, :], seed, TRUTH[None, :], **kwargs)
        est = out.estimates[0]
        e_t = float(out.errors[0])
        e_m = (
            float(np.linalg.norm(est - MIRROR)) if np.all(np.isfinite(est)) else np.nan
        )
        err_truth[i] = e_t
        err_mirror[i] = e_m
        claimed[i] = bool(out.converged[0])
        stalled[i] = bool(out.stalled[0])

        if not bool(out.solved[0]):
            codes[i] = FAIL_S
        elif e_t <= SOLVED_M:
            codes[i] = TRUTH_S
        elif np.isfinite(e_m) and e_m <= SOLVED_M:
            codes[i] = MIRROR_S
        else:
            codes[i] = FAIL_S  # solved by the four-condition test, but at neither point

    result = {
        "label": label,
        "key": key,
        "axis": axis,
        "xx": xx,
        "yy": yy,
        "seeds": seeds,
        "codes": codes.reshape(xx.shape),
        "counts": {s: int(np.sum(codes == s)) for s in STATE_LABELS},
        "n": len(seeds),
        "n_stalled": int(stalled.sum()),
        "stalled_on_line": int(np.sum(stalled & (seeds[:, 1] == LINE_Y))),
        "n_seeds_on_line": int(np.sum(seeds[:, 1] == LINE_Y)),
        "n_mirror_claimed": int(np.sum(claimed & (codes == MIRROR_S))),
    }
    if verbose:
        c = result["counts"]
        print(f"\n  {label}")
        print(f"      TRUTH  {c[TRUTH_S]:>5} / {result['n']}")
        print(f"      MIRROR {c[MIRROR_S]:>5} / {result['n']}")
        print(f"      FAIL   {c[FAIL_S]:>5} / {result['n']}")
        print(
            f"      stalled at the seed: {result['n_stalled']} "
            f"({result['stalled_on_line']} of them on the beacon line y = {LINE_Y:.0f}, "
            f"which holds {result['n_seeds_on_line']} seeds)"
        )
        if c[MIRROR_S]:
            print(
                f"      of the {c[MIRROR_S]} MIRROR seeds, {result['n_mirror_claimed']} "
                f"reported converged=True"
            )
    return result


def state_at(result, seed):
    """The tri-state code the sweep assigned to a given lattice seed."""
    axis = result["axis"]
    ix = int(np.argmin(np.abs(axis - seed[0])))
    iy = int(np.argmin(np.abs(axis - seed[1])))
    assert (
        abs(axis[ix] - seed[0]) < 1e-9 and abs(axis[iy] - seed[1]) < 1e-9
    ), f"seed {seed} is not on the lattice"
    return int(result["codes"][iy, ix])


def plot_basin(ax, result, show_ylabel):
    """One tri-state basin map. Every seed cell is painted; nothing is left uncovered
    (an uncovered region under the legend has been misread as data the legend hides)."""
    cmap = ListedColormap([STATE_COLOURS[s] for s in (TRUTH_S, MIRROR_S, FAIL_S)])
    ax.pcolormesh(
        result["xx"],
        result["yy"],
        result["codes"],
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        shading="nearest",
    )
    for s, h in STATE_HATCH.items():
        if h is None or result["counts"][s] == 0:
            continue
        ax.contourf(
            result["xx"],
            result["yy"],
            (result["codes"] == s).astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[h],
        )

    ax.plot(
        [BEACONS[:, 0].min(), BEACONS[:, 0].max()],
        [LINE_Y, LINE_Y],
        "-",
        color="white",
        lw=3.0,
        solid_capstyle="butt",
        zorder=4,
    )
    ax.plot(
        [BEACONS[:, 0].min(), BEACONS[:, 0].max()],
        [LINE_Y, LINE_Y],
        "-",
        color="black",
        lw=1.4,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.plot(
        BEACONS[:, 0],
        BEACONS[:, 1],
        "^",
        color="white",
        mec="black",
        mew=1.1,
        ms=8,
        ls="",
        zorder=6,
        label="beacon",
    )
    ax.plot(
        *TRUTH, "*", color="white", mec="black", mew=1.2, ms=15, zorder=7, label="truth"
    )
    ax.plot(
        *MIRROR,
        "D",
        color="white",
        mec="black",
        mew=1.2,
        ms=7,
        zorder=7,
        label="mirror",
    )

    c = result["counts"]
    ax.set_title(
        f"{result['label']}\ntruth {c[TRUTH_S]}  ·  mirror {c[MIRROR_S]}  ·  "
        f"fail {c[FAIL_S]}",
        fontsize=11,
    )
    ax.set_xlabel("initial guess x (m)")
    if show_ylabel:
        ax.set_ylabel("initial guess y (m)")
    ax.set_aspect("equal")


def state_handles():
    return [
        Patch(
            facecolor=STATE_COLOURS[TRUTH_S],
            edgecolor="black",
            label="TRUTH -- the real target",
        ),
        Patch(
            facecolor=STATE_COLOURS[MIRROR_S],
            edgecolor="black",
            hatch="///",
            label=f"MIRROR -- reflected in y = {LINE_Y:.0f} m",
        ),
        Patch(
            facecolor=STATE_COLOURS[FAIL_S],
            edgecolor="black",
            label="FAIL -- stalled, diverged, refused",
        ),
    ]


def mark_handles():
    return [
        plt.Line2D(
            [],
            [],
            color="white",
            mec="black",
            mew=1.1,
            marker="^",
            ls="",
            ms=8,
            label="beacon",
        ),
        plt.Line2D(
            [],
            [],
            color="white",
            mec="black",
            mew=1.2,
            marker="*",
            ls="",
            ms=14,
            label="truth",
        ),
        plt.Line2D(
            [],
            [],
            color="white",
            mec="black",
            mew=1.2,
            marker="D",
            ls="",
            ms=7,
            label="mirror of the truth",
        ),
    ]


def plot_summary(results):
    """Three tri-state basin maps, sharing one legend below them."""
    with plt.rc_context({"hatch.linewidth": 0.8, "hatch.color": "#3A3A00"}):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8), sharey=True)
        for k, (ax, r) in enumerate(zip(axes, results, strict=True)):
            plot_basin(ax, r, show_ylabel=(k == 0))

        handles = state_handles() + mark_handles()
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=9.5,
            handletextpad=0.5,
            columnspacing=1.4,
        )
        fig.suptitle(
            "Where you start decides which answer you get -- corridor array, zero noise\n"
            f"4 beacons on y = {LINE_Y:.0f} m; target ({TRUTH[0]:.2f}, {TRUTH[1]:.2f}); "
            f"{len(results[0]['seeds'])} seeds",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial-guess basin on a collinear array: truth vs. mirror (Chapter 4)"
    )
    parser.add_argument(
        "--out-dir", default=str(FIGS_DIR), help="Output directory for figures"
    )
    args = parser.parse_args()

    axis, *_ = seed_grid()
    print("=" * 74)
    print("Chapter 4: the initial-guess basin on a collinear array (truth vs. mirror)")
    print("=" * 74)
    print(f"  beacons  {BEACONS.tolist()}  (all on y = {LINE_Y:.0f} m)")
    print(
        f"  truth    ({TRUTH[0]:.4f}, {TRUTH[1]:.4f})  [ground_truth_positions.txt row {TRUTH_ROW}]"
    )
    print(f"  mirror   ({MIRROR[0]:.4f}, {MIRROR[1]:.4f})")
    print(
        f"  seeds    {len(axis)}x{len(axis)} = {len(axis) ** 2} over "
        f"[{GRID_MIN:.0f}, {GRID_MAX:.0f}] m, step {GRID_STEP}, zero measurement noise"
    )

    m_truth = measurements(TRUTH)
    m_mirror = measurements(MIRROR)
    print(
        f"\n  |TOA(truth) - TOA(mirror)|   max = "
        f"{np.max(np.abs(m_truth['toa'] - m_mirror['toa'])):.3e} m  <- the ambiguity, measured"
    )
    print(
        f"  |TDOA(truth) - TDOA(mirror)| max = "
        f"{np.max(np.abs(m_truth['tdoa'] - m_mirror['tdoa'])):.3e} m"
    )
    print(
        f"  |AOA(truth) - AOA(mirror)|   max = "
        f"{np.degrees(np.max(np.abs(m_truth['aoa'] - m_mirror['aoa']))):.3f} deg  "
        f"<- bearings tell truth from mirror"
    )

    results = [sweep(*arm) for arm in ARMS]

    print("\n" + "-" * 74)
    print("  validation: the beacon centroid sits ON the line of symmetry")
    names = {TRUTH_S: "TRUTH", MIRROR_S: "MIRROR", FAIL_S: "FAIL"}
    for r in results:
        s_c = state_at(r, CENTROID_SEED)
        s_o = state_at(r, OFFLINE_SEED)
        print(
            f"      {r['label']:<24} centroid seed -> {names[s_c]:<6}   "
            f"off-line seed -> {names[s_o]}"
        )

    fig = plot_summary(results)
    paths = save_figure(fig, args.out_dir, "ch4_reflection_ambiguity_basin")
    print(
        f"\n  saved ch4_reflection_ambiguity_basin: "
        f"{', '.join(p.suffix.lstrip('.') for p in paths)}"
    )
    plt.close("all")
    print(f"Figures written to {resolve_figs_dir(args.out_dir)}")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
