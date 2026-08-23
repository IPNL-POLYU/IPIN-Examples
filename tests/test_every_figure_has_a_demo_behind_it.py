"""Every committed figure has a demo behind it, and every demo's figure is committed.

The figures are this repository's deliverable, and until now nothing checked
that the set on disk is the set the code produces. Drift is silent in both
directions: a committed figure that no code writes still renders in a README, so
readers are shown a picture of an older version of the code; and a figure a demo
writes but nobody commits simply is not there when someone reads the chapter.

This began as a Chapter 8 check and found four orphans there --
`temporal_calibration_test.png`, `temporal_calibration_corrected.png`,
`imu_interpolation_test.png`, `robust_loss_comparison.png`, all dated December
against July for every live figure. Widened to every chapter it found three
more, and the same tell names them without running anything: `save_figure`
writes svg, pdf and png together, so **an incomplete set is by itself evidence
that something else made it**.

    ch6_dead_reckoning/figs/strapdown_trajectory.svg     svg only, Dec 2025
    ch6_dead_reckoning/figs/zupt_trajectory_stance.svg   svg only, Dec 2025
    ch7_slam/figs/pose_graph_slam_results.png            png only, Feb 2025

The Chapter 6 pair is the worse kind: both were still displayed in the chapter
README, so the picture a reader saw was one the code had stopped producing. The
caption on one of them read "Alternative trajectory view", which is the same
shape as Chapter 8's "Test output" -- what gets written when nobody knows.

**Flags rather than exemptions, where the flag is cheap.** Fifteen further
figures are not written by a plain run but are perfectly real: Chapter 6's Allan
component plots need `--debug` and its PDR dataset panel needs `--data`. Those
are run here rather than excused, at a cost of about 21 s, which leaves the
exemption list holding one thing -- animations -- for one reason.

**The reverse direction sees the whole session's invocations, not just this
test's.** `run_example` writes every run of a chapter into one scratch
`figs/` per chapter, so if any test anywhere in the suite invokes a demo with a
flag that writes an extra figure, this check sees it. That is why running this
file alone can pass where the full suite fails -- and it is a feature, because
it catches figures produced by flags nobody thought to list here. It found
`ch4_geometry_comparison` that way.

**It does not compare the bytes.** `save_figure` is byte-reproducible on one
machine and not across two: measured on the Ubuntu runner against Chapter 8's
set, all 27 files differ, svg by ~0.1%, pdf by 2-4%, png by 5-27%. That is font
metrics and rasterisation, not a changed picture. See CLAUDE.md; the byte
version of this check also hung CI three times, because pytest diffs
`bytes == bytes` element by element.

Author: Li-Ta Hsu
"""

import functools

import pytest

from tests.example_runner import WORKSPACE_ROOT, run_example

#: Invocations beyond the plain run that a chapter needs to write all of its
#: committed figures. Each is here rather than in the exemption list below
#: because running it is cheap and an exemption is a hole.
EXTRA_RUNS = {
    "ch4_rf_point_positioning": (
        # The geometry comparison is a documented Quick Start mode and writes
        # its own figure. It was found by the reverse direction below rather
        # than listed here, and sat in KNOWN_UNCOMMITTED while the comparison
        # it drew was wrong; naming the invocation is what stops the figure
        # depending on some other test in the suite happening to run it.
        ("example_comparison", ("--compare-geometry",)),
    ),
    "ch6_dead_reckoning": (
        # Allan variance draws the per-component breakdown only in debug mode.
        ("example_allan_variance", ("--debug",)),
        # PDR's dataset panel exists only on the --data path.
        ("example_pdr", ("--data", "ch6_pdr_corridor_walk")),
    ),
}

#: Committed figures no run here produces, with the reason.
#:
#: All eight are animations behind `--animate`. Building them is slow -- the
#: chapter READMEs say so on the lines that document the flag -- and each has a
#: still figure from the same example, drawn from the same data, that *is*
#: checked. If you add a ninth entry for any other reason, it needs its own
#: sentence saying why running it is not the better answer.
EXPECTED_WITHOUT_A_RUN = {
    "ch3_estimators/figs/ch3_particle_bimodal.gif",
    "ch4_rf_point_positioning/figs/ch4_dop_geometry.gif",
    "ch5_fingerprinting/figs/ch5_walk_posterior.gif",
    "ch6_dead_reckoning/figs/ch6_zupt_drift.gif",
    "ch7_slam/figs/bundle_adjustment.gif",
    "ch7_slam/figs/ch7_icp_convergence.gif",
    "ch7_slam/figs/slam_pipeline_square.gif",
    "ch8_sensor_fusion/figs/ch8_anchor_outage.gif",
}


#: Figures a demo writes that are deliberately not committed, predating this
#: check. Same ratchet as the rest of the repository: only shrink it.
#:
#: **It is empty.** Its one entry was the three files of
#: `ch4_geometry_comparison`, held open because the figure was *wrong* rather
#: than merely absent: `example_comparison --compare-geometry` aggregated each
#: method as an RMSE over the solves that reported convergence, so AOA on the
#: collinear dataset reported 2.2e10 m -- three fixes at 1e11 m among 95
#: "converged" ones -- which on a linear axis flattened every other bar to zero
#: height. Methods with nothing converged printed no line and drew no bar,
#: which is why Square and Optimal showed TOA and TDOA, the collinear geometry
#: showed only AOA, and no method appeared on all three.
#:
#: It now reports the median over every fix with the failure count beside it,
#: on a log axis, and every geometry carries every method. The invocation is in
#: EXTRA_RUNS above so this file produces the figure itself.
KNOWN_UNCOMMITTED = set()


def _chapters():
    return sorted(p.name for p in WORKSPACE_ROOT.glob("ch*_*") if p.is_dir())


@functools.lru_cache(maxsize=None)
def _regenerated(chapter):
    """Run a chapter's demos once. Returns (figs_dir, failures).

    **Failures are returned rather than raised.** `functools.lru_cache` does not
    cache exceptions, so asserting here would re-run every demo for every one of
    this chapter's parametrised comparisons the moment one misbehaved. That
    turned a single failure into a cancelled 45-minute CI job once already.
    """
    figs_dir = None
    failures = []
    invocations = [
        (p.stem, ()) for p in sorted((WORKSPACE_ROOT / chapter).glob("example_*.py"))
    ]
    invocations += list(EXTRA_RUNS.get(chapter, ()))
    for module, args in invocations:
        spelled = f"{chapter}.{module}"
        try:
            run = run_example(spelled, *args)
        except Exception as exc:  # noqa: BLE001 - reported by test_every_demo_runs
            failures.append(f"{spelled} {' '.join(args)}: {type(exc).__name__}: {exc}")
            continue
        if run.process.returncode != 0:
            failures.append(
                f"{spelled} {' '.join(args)}: exited {run.process.returncode}\n"
                f"{run.process.stderr[-1200:]}"
            )
        figs_dir = run.figs_dir
    return figs_dir, tuple(failures)


def _figs_dir(chapter):
    figs_dir, failures = _regenerated(chapter)
    if failures or figs_dir is None:
        pytest.skip(f"a {chapter} demo did not run; see test_every_demo_runs")
    return figs_dir


def _committed(chapter):
    figs = WORKSPACE_ROOT / chapter / "figs"
    return sorted(p for p in figs.iterdir() if p.is_file()) if figs.is_dir() else []


def _all_committed():
    return [(c, p) for c in _chapters() for p in _committed(c)]


def _figure_id(case):
    chapter, path = case
    return f"{chapter}/{path.name}"


@pytest.mark.parametrize("chapter", _chapters())
def test_every_demo_runs(chapter):
    """Guard the guard: the comparisons below prove nothing if nothing ran."""
    figs_dir, failures = _regenerated(chapter)
    assert not failures, (
        f"{chapter} demos that did not complete, so their figures were never "
        "regenerated and every comparison for this chapter is vacuous:\n\n"
        + "\n\n".join(failures)
    )
    assert (
        figs_dir is not None and figs_dir.is_dir()
    ), f"{chapter} produced no figure directory ({figs_dir})."


@pytest.mark.parametrize("case", _all_committed(), ids=_figure_id)
def test_committed_figure_still_has_a_demo_behind_it(case):
    """Something in the chapter must still write every file in its figs/."""
    chapter, committed = case
    relative = f"{chapter}/figs/{committed.name}"
    if relative in EXPECTED_WITHOUT_A_RUN:
        pytest.skip(f"{relative} needs --animate; see the set's comment")

    fresh = _figs_dir(chapter) / committed.name
    assert fresh.is_file(), (
        f"{relative} is committed but no {chapter} demo writes it.\n\n"
        "Usually a leftover from a version of the code that no longer exists -- "
        "seven were, when this check was written, and an incomplete svg/pdf/png "
        "set is the tell, since save_figure always writes all three. Delete it, "
        "and check whether a README is still displaying it. If a real figure "
        "needs a flag to appear, add the invocation to EXTRA_RUNS rather than "
        "excusing the file."
    )


@pytest.mark.parametrize("chapter", _chapters())
def test_no_demo_writes_a_figure_that_is_not_committed(chapter):
    """The other direction: a demo's output must be in the repository."""
    fresh = _figs_dir(chapter)
    have = {p.name for p in _committed(chapter)}
    uncommitted = sorted(
        p.name
        for p in fresh.iterdir()
        if p.is_file()
        and p.name not in have
        and f"{chapter}/figs/{p.name}" not in KNOWN_UNCOMMITTED
    )
    assert not uncommitted, (
        f"{chapter} demos write figures that are not committed:\n  "
        + "\n  ".join(uncommitted)
        + "\n\nThe figures are the deliverable; commit them, and open the PNGs "
        "before you do."
    )
