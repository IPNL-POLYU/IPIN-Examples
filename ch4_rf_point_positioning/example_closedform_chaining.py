"""Closed-form breaks the chicken-and-egg: no seed anywhere, then refine.

Chapter 4, Sections 4.3.3-4.3.5. An iterative solve needs an initial guess (Eqs.
(4.14)-(4.23)) -- and Section 4.4's basins (also
`example_initial_guess_basin.py`, `example_reflection_ambiguity.py`) are two different
ways that guess can go wrong. Closed-form solvers sidestep the chicken-and-egg entirely:
Fang's algorithm (Eqs. (4.43)-(4.49), `core.rf.toa_fang_solver`) turns squared TOA range
differences into one linear solve, and Chan's algorithm (Eqs. (4.50)-(4.62),
`core.rf.tdoa_chan_solver`) does the same for TDOA -- neither one takes an
``initial_guess`` argument, because neither iterates.

**This example measures whether that closed-form output is worth anything**, on the two
datasets `example_comparison.py --compare-geometry` already contrasts:

- `data/sim/ch4_rf_2d_square` (control): closed-form alone (Fang) already lands within
  0.011 m of the fully-refined answer, and CHAINING it into the iterative refiner --
  feeding its output in as that solver's own ``initial_guess``, per row -- reproduces the
  good-seed iterative result exactly. Nothing here needed a seed from anywhere.
- `data/sim/ch4_rf_2d_linear` (corridor): the beacons are collinear, so every non-reference
  row of Fang's linear system (the ``h_n^i = -2*(x_n^i - x_n^ref)`` term of
  Eqs. (4.43)-(4.49)) is exactly zero -- measured here as ``cond(H_a^T H_a) = inf`` and
  every one of the 100 Fang y-estimates landing at exactly 0.0, never near the true y in
  [2, 18]. The closed form
  recovers x and gives up on y outright; it is not merely inaccurate. Chaining that
  (x, 0.0) seed into the iterative refiner still converges -- but the corridor's mirror
  ambiguity (`example_reflection_ambiguity.py`; a target and its reflection about the
  beacon line are the same distance from every beacon) resurfaces exactly because the
  seed's y is always on ONE side of the line, so a converged fix lands on the truth or on
  its mirror depending only on which side the true point started on.

Run:
    python -m ch4_rf_point_positioning.example_closedform_chaining

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.43)-(4.49) Fang closed-form TOA (Section 4.3.4),
            (4.50)-(4.62) Chan closed-form TDOA (Section 4.3.5), (4.14)-(4.23) iterative
            TOA/I-WLS. Companion to `example_reflection_ambiguity.py`, whose tri-state
            TRUTH/MIRROR/FAIL classification this example reuses for the corridor's
            chained fixes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Run as a script, sys.path[0] is THIS directory, so `core` resolves to whatever is
# installed. Same fix as the chapter's other examples (issue #86).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import resolve_figs_dir, save_figure, show_figures_if_requested
from core.rf import (
    TDOAPositioner,
    TOAPositioner,
    solve_batch,
    tdoa_chan_solver,
    toa_fang_solver,
)
from core.rf.positioning import DIVERGENCE_M, STALL_M

FIGS_DIR = Path(__file__).parent / "figs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "sim"

DATASETS = {
    "square (control)": DATA_ROOT / "ch4_rf_2d_square",
    "corridor (collinear)": DATA_ROOT / "ch4_rf_2d_linear",
}

#: The corridor beacons all sit on this line (data/sim/ch4_rf_2d_linear/beacons.txt);
#: a target and its reflection about it are equidistant from every beacon on it.
LINE_Y = 10.0


def load_dataset(ds_dir: Path):
    """Beacons, ground truth, TOA ranges, TDOA diffs, and the room's own failure floor."""
    beacons = np.loadtxt(ds_dir / "beacons.txt")
    truth = np.loadtxt(ds_dir / "ground_truth_positions.txt")
    toa = np.loadtxt(ds_dir / "toa_ranges.txt")
    tdoa = np.loadtxt(ds_dir / "tdoa_diffs.txt")
    config = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    room_diagonal_m = float(config["geometry"]["area_size_m"]) * np.sqrt(2.0)
    return beacons, truth, toa, tdoa, room_diagonal_m


def row_stats(errors: np.ndarray, failed: np.ndarray) -> dict:
    """Median over every finite error, failures included (`SolveOutcome.median_m`'s own
    convention) -- a bare mean over solves that include divergences is not a statistic
    worth reporting; see `.cursor/rules/030-figures-and-claims.mdc`."""
    errors = np.asarray(errors, dtype=float)
    finite = np.isfinite(errors)
    ok = finite & ~failed
    return {
        "median_m": float(np.median(errors[finite])) if finite.any() else float("nan"),
        "mean_m": float(errors[ok].mean()) if ok.any() else float("nan"),
        "fails": int(np.sum(failed)),
        "n": int(len(errors)),
    }


def closed_form_arm(solver_fn, beacons, meas, truth, room_diagonal_m, label):
    """A closed-form solver (no initial guess) over every row.

    Failure = raised, non-finite/complex output, or an error beyond the room's own
    diagonal (`room_diagonal_m`, from the dataset's `config.json`, not a fixed constant).
    Tracks the linear system's condition number so a structural singularity -- not just
    noise -- is visible in the printed report.
    """
    n = len(truth)
    estimates = np.full((n, 2), np.nan)
    cond_numbers = np.full(n, np.nan)
    failed = np.zeros(n, dtype=bool)

    for i in range(n):
        try:
            pos, info = solver_fn(beacons, meas[i])
        except Exception:  # noqa: BLE001 - a raise IS one of the failure modes
            failed[i] = True
            continue
        pos = np.asarray(pos)
        # Recorded regardless of what `pos` turns out to be below: the condition number
        # is a property of the solver's linear system, not of whether that system's
        # solution happens to be finite -- see `cond_is_ever_infinite` below, which reads
        # this array unfiltered by `errors` for exactly that reason.
        cond_numbers[i] = info.get("condition_number", np.nan)
        if not np.all(np.isfinite(pos)) or np.iscomplexobj(pos):
            failed[i] = True
            continue
        estimates[i] = pos

    errors = np.linalg.norm(estimates - truth, axis=1)
    failed = (
        failed
        | ~np.isfinite(errors)
        | (np.nan_to_num(errors, nan=np.inf) > room_diagonal_m)
    )
    stats = row_stats(errors, failed)
    stats.update(
        label=label,
        estimates=estimates,
        cond_median=float(np.nanmedian(cond_numbers)),
        # `np.isinf` (not `~np.isfinite`) deliberately: a NaN entry means the row raised
        # before a condition number was ever computed, which is not the same claim as an
        # infinite one. Unfiltered by `errors`' finiteness -- a structural singularity in
        # H_a^T H_a is a fact about the linear system whether or not this solver's
        # particular row happened to land on a finite position.
        cond_is_ever_infinite=bool(np.any(np.isinf(cond_numbers))),
    )
    return stats


def iterative_reference_arm(positioner, meas, seed, truth, label):
    """The naive baseline: `core.rf.solve_batch`, one shared seed for every row."""
    out = solve_batch(positioner, meas, seed, truth)
    return {
        "label": label,
        "median_m": out.median_m,
        "mean_m": out.mean_solved_m,
        "fails": out.n_failed,
        "n": out.n,
        "estimates": out.estimates,
    }


def chained_arm(positioner, meas, seed_estimates, truth, label):
    """Closed-form output fed in as a PER-ROW `initial_guess` to the iterative solver.

    `solve_batch` only takes one shared seed, so this replicates its exact failure
    semantics (raised / refused / stalled / diverged) with a per-row one instead.
    """
    n = len(truth)
    estimates = np.full((n, 2), np.nan)
    converged = np.zeros(n, dtype=bool)
    stalled = np.zeros(n, dtype=bool)
    for i in range(n):
        guess = seed_estimates[i]
        if not np.all(np.isfinite(guess)):
            continue  # closed-form itself failed on this row; nothing to chain
        try:
            pos, info = positioner.solve(meas[i], initial_guess=guess)
        except Exception:  # noqa: BLE001
            continue
        estimates[i] = pos
        converged[i] = bool(info.get("converged", True))
        stalled[i] = bool(np.linalg.norm(pos - guess) < STALL_M)
    errors = np.linalg.norm(estimates - truth, axis=1)
    solved = (
        converged
        & ~stalled
        & np.isfinite(errors)
        & (np.nan_to_num(errors, nan=np.inf) < DIVERGENCE_M)
    )
    stats = row_stats(errors, ~solved)
    stats.update(label=label, estimates=estimates, solved=solved)
    return stats


def mirror_split(chained: dict, truth: np.ndarray, label: str, verbose=True) -> dict:
    """Tri-state TRUTH / MIRROR / FAIL for a corridor chained arm, mutually exclusive
    and summing to `n` -- same convention as `example_reflection_ambiguity.py`'s `sweep`.

    Gated on `chained["solved"]` (the same converged & ~stalled & finite & <divergence
    test `core.rf.solve_batch` applies) rather than on finiteness alone: an EARLIER
    version of this function classified truth-vs-mirror over every row with a finite
    estimate, independent of whether it had actually solved, and one dataset's Chan arm
    (85/100 solved, 100/100 finite) then double-counted 15 unconverged-but-finite rows
    into truth/mirror on top of the fail bucket -- caught by looking at the rendered
    figure (`.cursor/rules/030-figures-and-claims.mdc`: "open the rendered PNG"; a bar
    reaching past its own 100-row total was the tell), not by a test. A median over
    these estimates would hide the split just as badly:
    `data/sim/ch4_rf_2d_linear/README.md` calls that out by name ("a median can hide a
    bimodal result").
    """
    est = chained["estimates"]
    solved = chained["solved"]
    err_truth = np.linalg.norm(est - truth, axis=1)
    mirrored = truth.copy()
    mirrored[:, 1] = 2 * LINE_Y - truth[:, 1]
    err_mirror = np.linalg.norm(est - mirrored, axis=1)
    mirror_side = solved & (err_mirror < err_truth)
    truth_side = solved & ~mirror_side

    n = chained["n"]
    result = {
        "n": int(n),
        "converged": int(solved.sum()),
        "truth_side": int(truth_side.sum()),
        "mirror_side": int(mirror_side.sum()),
        "fail": int((~solved).sum()),
    }
    assert (
        result["truth_side"] + result["mirror_side"] + result["fail"] == n
    ), "truth/mirror/fail must partition every row exactly once"
    if verbose:
        print(f"\n  [{label}] tri-state split of the chained fixes")
        print(f"      truth-side : {result['truth_side']}/{n}")
        print(f"      mirror-side: {result['mirror_side']}/{n}")
        print(f"      fail       : {result['fail']}/{n}")
    return result


def run_square(beacons, truth, toa, tdoa, room_diagonal_m):
    """Control geometry: does chaining a closed-form seed into the refiner reproduce the
    good-seed iterative answer, with NO seed anywhere in the pipeline?"""
    centroid = beacons.mean(axis=0)
    results = {}
    results["iter_toa"] = iterative_reference_arm(
        TOAPositioner(beacons, method="iterative_ls"),
        toa,
        centroid,
        truth,
        "Iterative TOA (I-LS) @ centroid seed",
    )
    results["iter_tdoa"] = iterative_reference_arm(
        TDOAPositioner(beacons, reference_anchor_index=0),
        tdoa,
        centroid,
        truth,
        "Iterative TDOA (LS) @ centroid seed",
    )
    results["fang"] = closed_form_arm(
        toa_fang_solver,
        beacons,
        toa,
        truth,
        room_diagonal_m,
        "Fang TOA closed-form (no seed)",
    )
    results["chan"] = closed_form_arm(
        tdoa_chan_solver,
        beacons,
        tdoa,
        truth,
        room_diagonal_m,
        "Chan TDOA closed-form (no seed)",
    )
    results["fang_to_iter"] = chained_arm(
        TOAPositioner(beacons, method="iterative_ls"),
        toa,
        results["fang"]["estimates"],
        truth,
        "Fang -> iterative TOA refine",
    )
    results["chan_to_iter"] = chained_arm(
        TDOAPositioner(beacons, reference_anchor_index=0),
        tdoa,
        results["chan"]["estimates"],
        truth,
        "Chan -> iterative TDOA refine",
    )
    return results


def run_corridor(beacons, truth, toa, tdoa, room_diagonal_m):
    """Collinear geometry: the naive centroid seed is degenerate, and closed-form's own
    y-column is exactly singular. Chaining still converges; the tri-state split shows
    where."""
    # LINE_Y is a module-level constant (mirror_split's own reflection formula uses it),
    # not derived from `beacons` -- assert it still matches the loaded dataset rather
    # than let a future regeneration of data/sim/ch4_rf_2d_linear silently mirror
    # everything about the wrong line.
    assert np.allclose(beacons[:, 1], LINE_Y), (
        f"LINE_Y={LINE_Y} no longer matches the corridor beacons' y-coordinates "
        f"{beacons[:, 1].tolist()} -- update the constant"
    )
    centroid = beacons.mean(axis=0)
    results = {}
    results["iter_toa"] = iterative_reference_arm(
        TOAPositioner(beacons, method="iterative_ls"),
        toa,
        centroid,
        truth,
        "Iterative TOA (I-LS) @ centroid seed",
    )
    results["iter_tdoa"] = iterative_reference_arm(
        TDOAPositioner(beacons, reference_anchor_index=0),
        tdoa,
        centroid,
        truth,
        "Iterative TDOA (LS) @ centroid seed",
    )
    results["fang"] = closed_form_arm(
        toa_fang_solver,
        beacons,
        toa,
        truth,
        room_diagonal_m,
        "Fang TOA closed-form (no seed)",
    )
    results["chan"] = closed_form_arm(
        tdoa_chan_solver,
        beacons,
        tdoa,
        truth,
        room_diagonal_m,
        "Chan TDOA closed-form (no seed)",
    )
    results["fang_to_iter"] = chained_arm(
        TOAPositioner(beacons, method="iterative_ls"),
        toa,
        results["fang"]["estimates"],
        truth,
        "Fang -> iterative TOA refine",
    )
    results["chan_to_iter"] = chained_arm(
        TDOAPositioner(beacons, reference_anchor_index=0),
        tdoa,
        results["chan"]["estimates"],
        truth,
        "Chan -> iterative TDOA refine",
    )
    results["fang_mirror"] = mirror_split(
        results["fang_to_iter"], truth, "Fang -> iterative TOA"
    )
    results["chan_mirror"] = mirror_split(
        results["chan_to_iter"], truth, "Chan -> iterative TDOA"
    )
    return results


def fmt(s: dict) -> str:
    def f(x):
        return "nan" if not np.isfinite(x) else f"{x:.3f}"

    return (
        f"{s['label']:<38s} median={f(s['median_m']):>8s}  mean={f(s['mean_m']):>8s}  "
        f"fails={s['fails']:3d}/{s['n']}"
    )


def plot_chicken_and_egg(square, corridor):
    """Left: median error, square dataset -- closed-form alone is nearly the refined
    answer, chaining reproduces it exactly, with no seed anywhere. Right: the corridor's
    chained fixes, split truth vs. mirror -- a median would hide exactly this."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.2))

    bars = [
        (
            "Iterative,\ncentroid seed",
            square["iter_toa"]["median_m"],
            "tab:blue",
            False,
        ),
        ("Fang closed-form,\nNO seed", square["fang"]["median_m"], "tab:blue", True),
        ("Fang ->\nrefine", square["fang_to_iter"]["median_m"], "tab:blue", False),
        ("Chan ->\nrefine", square["chan_to_iter"]["median_m"], "tab:red", False),
    ]
    for xi, (_label, val, color, hatch) in enumerate(bars):
        ax_l.bar(
            xi,
            val,
            width=0.6,
            color=color,
            edgecolor="black" if hatch else "white",
            linewidth=0.8,
            hatch="///" if hatch else None,
        )
        ax_l.annotate(
            f"{val:.3f}",
            (xi, val),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax_l.set_xticks(range(len(bars)))
    ax_l.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax_l.set_ylabel("median error [m]")
    ax_l.set_ylim(0, max(v for _, v, _, _ in bars) * 1.5)
    ax_l.set_title(
        "Square (control): closed-form needs no seed\nand chaining matches the good-seed answer"
    )
    ax_l.grid(True, axis="y", alpha=0.3)
    ax_l.set_axisbelow(True)

    groups = [
        (
            "Iterative,\ncentroid seed",
            corridor["iter_toa"]["n"] - corridor["iter_toa"]["fails"],
            0,
            corridor["iter_toa"]["fails"],
        ),
        (
            "Fang ->\nrefine",
            corridor["fang_mirror"]["truth_side"],
            corridor["fang_mirror"]["mirror_side"],
            corridor["fang_mirror"]["fail"],
        ),
        (
            "Chan ->\nrefine",
            corridor["chan_mirror"]["truth_side"],
            corridor["chan_mirror"]["mirror_side"],
            corridor["chan_mirror"]["fail"],
        ),
    ]
    assert all(
        sum(g[1:]) == 100 for g in groups
    ), "each stacked bar must total 100 fixes"
    x = np.arange(len(groups))
    truth_c = [g[1] for g in groups]
    mirror_c = [g[2] for g in groups]
    fail_c = [g[3] for g in groups]
    ax_r.bar(
        x, truth_c, width=0.6, color="#009E73", edgecolor="black", label="truth-side"
    )
    ax_r.bar(
        x,
        mirror_c,
        width=0.6,
        bottom=truth_c,
        color="#F0E442",
        edgecolor="black",
        hatch="///",
        label="mirror-side",
    )
    ax_r.bar(
        x,
        fail_c,
        width=0.6,
        bottom=np.array(truth_c) + np.array(mirror_c),
        color="#4D4D4D",
        edgecolor="black",
        label="fail",
    )
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax_r.set_ylabel("fixes out of 100")
    ax_r.set_title(
        "Corridor: chaining converges, but the\nmirror ambiguity survives it"
    )
    ax_r.legend(fontsize=9, loc="upper right")
    ax_r.grid(True, axis="y", alpha=0.3)
    ax_r.set_axisbelow(True)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=str(FIGS_DIR), help="Output directory for figures"
    )
    args = parser.parse_args()

    all_results = {}
    for name, ds_dir in DATASETS.items():
        beacons, truth, toa, tdoa, room_diagonal_m = load_dataset(ds_dir)
        print("=" * 88)
        print(f"DATASET: {name}  ({ds_dir.name})")
        print("=" * 88)
        print(f"  beacons = {beacons.tolist()}")
        runner = run_square if name == "square (control)" else run_corridor
        results = runner(beacons, truth, toa, tdoa, room_diagonal_m)
        for key in (
            "iter_toa",
            "iter_tdoa",
            "fang",
            "chan",
            "fang_to_iter",
            "chan_to_iter",
        ):
            print(fmt(results[key]))
            if key in ("fang", "chan"):
                print(
                    f"      cond(H^T H) median={results[key]['cond_median']:.3e}; "
                    f"ever non-finite over the solved rows: "
                    f"{results[key]['cond_is_ever_infinite']}"
                )
        all_results[name] = results

    print("\n" + "=" * 88)
    print(
        "Nothing in either dataset was written back; this reads data/sim and prints/plots."
    )

    fig = plot_chicken_and_egg(
        all_results["square (control)"], all_results["corridor (collinear)"]
    )
    paths = save_figure(fig, args.out_dir, "ch4_closedform_chaining")
    print(
        f"\n  saved ch4_closedform_chaining: "
        f"{', '.join(p.suffix.lstrip('.') for p in paths)}"
    )
    plt.close("all")
    print(f"Figures written to {resolve_figs_dir(args.out_dir)}")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
