"""An example must find its dataset from any working directory.

Every example cites its data the way the book does, ``data/sim/<name>``, and
that spelling used to be resolved against the working directory alone. So the
most natural thing a reader does after opening a chapter folder --

    cd ch5_fingerprinting
    python example_deterministic.py

-- failed with

    FileNotFoundError: Required file not found: data/sim/.../locations.npy

which names the dataset and never mentions the working directory. It reads as
"the data is missing", so the reader goes looking for a download link or tries
to regenerate a dataset that was sitting there the whole time. Worse, the rule
was not learnable: ch2 and ch4's inline-data examples run fine from that same
directory, so nothing distinguishes the twelve examples that break from the
nineteen that do not. Ten of those twelve said nothing about it in their
docstrings.

`core.utils.resolve_data_path` now resolves against the working directory first
and the repository root second, and this holds it.

**Exit status alone would not catch a regression here, and nearly did not.**
Chapter 4's geometry comparison prints ``Skipping <name> (not found)`` and
carries on, so a broken resolver still exits 0 -- it just compares nothing.
Chapter 2, 3 and 6's ``--data`` handling has the same shape: a message and
``return``. The check is therefore on the output, not the return code, and
Chapter 4 is in the list precisely because it is the one that fails quietly.

Chapter 7 is deliberately absent. CLAUDE.md routes every Chapter 7 example
subprocess through the shared runner's memoised invocations, and adding a
second working directory would double the slowest example in the suite to hold
an invariant the other six already demonstrate.

Author: Li-Ta Hsu
"""

import pytest

from tests.example_runner import WORKSPACE_ROOT, run_example

#: One dataset-reading example per chapter that has one, with arguments that
#: force the dataset path to be exercised rather than the inline-data branch.
DATASET_EXAMPLES = [
    (
        "ch2_coords.example_coordinate_transforms",
        ("--data", "ch2_coords_san_francisco"),
    ),
    ("ch3_estimators.example_ekf_range_bearing", ("--data", "ch3_estimator_nonlinear")),
    # --compare-geometry is the silent-skip path: three hardcoded dataset names,
    # each `continue`d past with a message if the path does not resolve.
    ("ch4_rf_point_positioning.example_comparison", ("--compare-geometry",)),
    ("ch5_fingerprinting.example_deterministic", ()),
    ("ch6_dead_reckoning.example_pdr", ("--data", "ch6_pdr_corridor_walk")),
    ("ch8_sensor_fusion.example_anchor_outage", ()),
]

#: What a dataset that failed to resolve looks like in the output. The first two
#: are the loud forms; the last two are the quiet ones that still exit 0.
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "FileNotFoundError",
    "Dataset not found",
    "(not found)",
)


def _chapter_dir(module):
    return str(WORKSPACE_ROOT / module.split(".")[0])


@pytest.mark.parametrize(
    "module,args", DATASET_EXAMPLES, ids=[m for m, _ in DATASET_EXAMPLES]
)
def test_example_finds_its_dataset_from_its_chapter_directory(module, args):
    """Running from the chapter folder must load the same data as from root."""
    run = run_example(module, *args, cwd=_chapter_dir(module))
    output = run.process.stdout + run.process.stderr

    assert run.process.returncode == 0, (
        f"{module} exited {run.process.returncode} when run from its own "
        f"chapter directory.\n\n{output[-2000:]}"
    )

    found = [marker for marker in FAILURE_MARKERS if marker in output]
    assert not found, (
        f"{module} ran from its chapter directory and reported {found} -- its "
        "dataset did not resolve. Dataset paths must go through "
        "core.utils.resolve_data_path, which tries the working directory and "
        'then the repository root; a bare Path("data/sim") / name only works '
        "when the reader happens to be standing at the root.\n\n" + output[-2000:]
    )


@pytest.mark.parametrize(
    "module,args", DATASET_EXAMPLES, ids=[m for m, _ in DATASET_EXAMPLES]
)
def test_the_same_example_is_clean_from_the_repository_root(module, args):
    """Guard the guard: the run above is only evidence if the root run is clean.

    Without this, a marker that stopped appearing for an unrelated reason -- a
    reworded message, a stage that no longer runs -- would leave the check above
    green while testing nothing.
    """
    run = run_example(module, *args)
    output = run.process.stdout + run.process.stderr
    assert run.process.returncode == 0, output[-2000:]
    assert not [m for m in FAILURE_MARKERS if m in output], (
        f"{module} does not resolve its dataset even from the repository root, "
        "so the chapter-directory check next door proves nothing.\n\n" + output[-2000:]
    )
