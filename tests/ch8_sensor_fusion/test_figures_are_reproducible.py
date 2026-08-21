"""Chapter 8's committed figures must be what its demos produce, both ways.

`core.eval.save_figure` writes bytes and is byte-reproducible, so a committed
figure that differs from a fresh run means the picture changed. CLAUDE.md says
so and treats a figure diff as a real signal -- but nothing checked it, and the
gap showed in the direction nobody watches.

Four PNGs sat in `ch8_sensor_fusion/figs/` that **no code produces**:
`temporal_calibration_test.png`, `temporal_calibration_corrected.png`,
`imu_interpolation_test.png` and `robust_loss_comparison.png`. All four are
dated December while every live figure is dated July, and all four are PNG-only
-- `save_figure` writes svg, pdf and png together, so a lone PNG is by itself
evidence that something else made it. The demos had been rewritten to emit one
combined figure each; the old outputs were never deleted, and the README's
Figure Gallery went on describing them in four numbered sections, sourcing them
to "Test output".

So the check runs in both directions:

- every committed figure regenerates byte-for-byte, and
- nothing sits in `figs/` that no demo writes.

The second is the half that found the orphans, and it is the half a
regenerate-and-compare check normally omits, because an extra file breaks
nothing at run time. It only misleads readers.

Cost is near zero: every marked transcript in the chapter README already runs
these eight modules with no arguments, and `run_example` is memoised on
(module, args), so this test reads the output of subprocesses
`tests/docs/test_readme_example_output.py` has already paid for.

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
def test_committed_figure_regenerates_byte_for_byte(committed):
    """A committed figure must be exactly what a fresh run writes."""
    if committed.name in EXPECTED_WITHOUT_A_PLAIN_RUN:
        pytest.skip(f"{committed.name} needs --animate; see the set's comment")

    fresh = _figs_dir() / committed.name
    assert fresh.is_file(), (
        f"{committed.name} is committed under ch8_sensor_fusion/figs/ but no "
        "Chapter 8 demo wrote it. Either it is a leftover from a version of "
        "the code that no longer exists -- four PNGs were, when this check was "
        "written -- or a demo that should produce it has stopped. Delete it, or "
        "restore whatever wrote it."
    )
    assert fresh.read_bytes() == committed.read_bytes(), (
        f"{committed.name} differs from what the demos now produce "
        f"({fresh.stat().st_size} bytes fresh against "
        f"{committed.stat().st_size} committed).\n\n"
        "save_figure is byte-reproducible, so this means the picture changed. "
        "If that was intended, regenerate and commit the new figure -- and open "
        "the PNG first, because this check compares bytes and cannot tell an "
        "improvement from a defect."
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
