"""Every Chapter 8 figure has a demo behind it, and every demo's figure is committed.

Four PNGs sat in `ch8_sensor_fusion/figs/` that **no code produces**:
`temporal_calibration_test.png`, `temporal_calibration_corrected.png`,
`imu_interpolation_test.png` and `robust_loss_comparison.png`. All four were
dated December while every live figure was dated July, and all four were
PNG-only -- `save_figure` writes svg, pdf and png together, so a lone PNG is by
itself evidence that something else made it. The demos had been rewritten to
emit one combined figure each; the old outputs were never deleted, and the
README's Figure Gallery went on describing them in four numbered sections,
sourcing them to "Test output".

An extra file in `figs/` breaks nothing at run time. It only misleads readers,
which is why it needed a check rather than a bug report. The check runs both
ways: nothing committed that no demo writes, and nothing written that is not
committed. It has earned itself twice -- on those four, and again when a
mechanical rename moved three demos' default figure paths and this named the
files.

**It deliberately does not compare the bytes, and that was measured rather than
assumed.** The first version asserted
`fresh.read_bytes() == committed.read_bytes()`, which passes locally, because
`save_figure` really is byte-reproducible on one machine. On the Ubuntu runner
**all 27 files differ**: svg by about 0.1%, pdf by 2-4%, png by 5-27%
(252546 B against 321177 B for one of them). That is font metrics and
rasterisation, not a changed picture. It is the figure analogue of the rule this
repository already learned about float equality and CI runners -- assert bit
equality on stored bytes, never on a fresh computation compared against what
some other machine produced.

That version also **hung CI three times rather than failing**. pytest's
assertion introspection on `bytes == bytes` builds the diff element by element,
so a mismatch on a 300 KB PNG is not a failure report but an astronomical one:
the job was cancelled at the 45-minute limit having printed `.s` and nothing
further. If you ever assert equality on large binary blobs, compare digests.

To check that a change left the pictures alone, do it the way CLAUDE.md already
says: regenerate on your own machine, read `git status`, and open the PNGs.
That works precisely because the bytes are reproducible per-machine.

Author: Li-Ta Hsu
"""

import functools
from pathlib import Path

import pytest

from tests.example_runner import WORKSPACE_ROOT, run_example

COMMITTED = WORKSPACE_ROOT / "ch8_sensor_fusion" / "figs"

#: Every runnable demo in the chapter, spelled as the README's transcript
#: markers spell them so the memoised runs are shared rather than repeated.
DEMOS = (
    "ch8_sensor_fusion.tc_uwb_imu_ekf",
    "ch8_sensor_fusion.lc_uwb_imu_ekf",
    "ch8_sensor_fusion.compare_lc_tc",
    "ch8_sensor_fusion.observability_demo",
    "ch8_sensor_fusion.tuning_robust_demo",
    "ch8_sensor_fusion.temporal_calibration_demo",
    "ch8_sensor_fusion.calibration_demo",
    "ch8_sensor_fusion.example_anchor_outage",
)

#: Committed figures no plain run produces, with the reason.
#:
#: The animation is behind --animate because building it is slow and the
#: chapter README says so on the line that documents the flag. Regenerating it
#: to compare bytes would cost more than the check is worth; the still figure
#: from the same example is compared, and it is drawn from the same data.
EXPECTED_WITHOUT_A_PLAIN_RUN = {"ch8_anchor_outage.gif"}


@functools.lru_cache(maxsize=1)
def _regenerated():
    """Run every Chapter 8 demo once. Returns (figs_dir, failures).

    **The failures are returned rather than raised, and that is the whole
    point.** `functools.lru_cache` does not cache exceptions, so an earlier
    version of this that asserted here re-ran all eight demos for *every* one
    of the 28 parametrised tests the moment one demo misbehaved. On a CI runner
    where a demo hit `run_example`'s 600 s deadlock guard, that turned one
    failure into hours: the job was cancelled at the workflow's 45-minute limit
    having printed nothing since the previous test file.

    A cached failure reports once, clearly, and lets the byte comparisons fail
    fast instead of each paying for the whole set again.
    """
    figs_dir = None
    failures = []
    for module in DEMOS:
        try:
            run = run_example(module)
        except Exception as exc:                     # noqa: BLE001 - reported below
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
            continue
        if run.process.returncode != 0:
            failures.append(
                f"{module}: exited {run.process.returncode}\n"
                f"{run.process.stderr[-1500:]}"
            )
        figs_dir = run.figs_dir
    return figs_dir, tuple(failures)


def _figs_dir() -> Path:
    """Where this session's regenerated figures landed, or skip the comparison."""
    figs_dir, failures = _regenerated()
    if failures or figs_dir is None:
        pytest.skip("a Chapter 8 demo did not run; see test_every_demo_runs")
    return figs_dir


def test_every_demo_runs():
    """Guard the guard: a byte comparison proves nothing if nothing ran."""
    figs_dir, failures = _regenerated()
    assert not failures, (
        "Chapter 8 demos that did not complete, so their figures were never "
        "regenerated and every comparison below is vacuous:\n\n"
        + "\n\n".join(failures)
    )
    assert figs_dir is not None and figs_dir.is_dir(), (
        f"No figure directory was produced ({figs_dir})."
    )


def _committed_figures():
    if not COMMITTED.is_dir():
        return []
    return sorted(p for p in COMMITTED.iterdir() if p.is_file())


@pytest.mark.parametrize("committed", _committed_figures(), ids=lambda p: p.name)
def test_committed_figure_still_has_a_demo_behind_it(committed):
    """Something in the chapter must still write every file in figs/."""
    if committed.name in EXPECTED_WITHOUT_A_PLAIN_RUN:
        pytest.skip(f"{committed.name} needs --animate; see the set's comment")

    fresh = _figs_dir() / committed.name
    assert fresh.is_file(), (
        f"{committed.name} is committed under ch8_sensor_fusion/figs/ but no "
        "Chapter 8 demo wrote it. Either it is a leftover from a version of "
        "the code that no longer exists -- four PNGs were, when this check was "
        "written -- or a demo that should produce it has been renamed or "
        "stopped. Delete it, or point the demo back at the committed name."
    )


def test_no_demo_writes_a_figure_that_is_not_committed():
    """The reverse direction: a demo's output must be in the repository."""
    fresh = _figs_dir()
    uncommitted = sorted(
        path.name
        for path in fresh.iterdir()
        if path.is_file() and not (COMMITTED / path.name).is_file()
    )
    assert not uncommitted, (
        "Chapter 8 demos write figures that are not committed:\n  "
        + "\n  ".join(uncommitted)
        + "\n\nThe chapter's figures are the deliverable; commit them, and open "
        "the PNGs before you do."
    )
