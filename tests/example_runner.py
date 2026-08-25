"""Run a chapter example once per distinct invocation, and share the result.

This began life as ``tests/ch7_slam/slam_example_runner.py``, written because
five Chapter 7 tests were each shelling out to the same example at a timeout
thin enough that the suite's answer depended on machine load. Its docstring
records that history and is still worth reading.

Nothing in the mechanism was ever Chapter 7 specific -- it takes a module path
and arguments -- and CLAUDE.md tells anyone adding a test that shells out to an
example to go through it. A second caller arrived (``tests/docs``, checking
that the transcripts in the chapter READMEs still match what the examples
print), so the generic half lives here and the SLAM module re-exports it.

Sharing the *function object* is the point, not sharing the code: the
memoisation is an ``lru_cache`` on this function, so a Chapter 7 test and a
docs test asking for the same invocation get one subprocess between them. Two
modules with their own copies would silently run everything twice.

Author: Li-Ta Hsu
"""

import atexit
import functools
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Generous on purpose. The timeout is a deadlock guard, not a performance
# budget; a tight one buys nothing and costs false failures. The slowest
# example measured here is the scan-matching visualisation at ~166 s
# standalone, so this is ~3.6x its observed runtime.
EXAMPLE_TIMEOUT_S = 600


@functools.lru_cache(maxsize=1)
def _figs_root() -> Path:
    """Scratch directory that IPIN_FIGS_DIR points the examples at.

    One per pytest session, matching the lifetime of the memoised runs whose
    output tests read back. Removed at interpreter exit.
    """
    root = Path(tempfile.mkdtemp(prefix="ipin-example-figs-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


class ExampleRun(NamedTuple):
    """One completed example run, shared between tests.

    Attributes:
        process: The finished subprocess. Treat as read-only -- every test
            asserting on this invocation sees the same object.
        started_at: Wall-clock epoch seconds from just before launch. Lets a
            test tell a file this run wrote from a stale committed one.
        figs_dir: Where this run's ``chX/figs`` output actually landed. Assert
            against this, never against the in-repo path.
    """

    process: subprocess.CompletedProcess
    started_at: float
    figs_dir: Path


@functools.lru_cache(maxsize=None)
def run_example(module: str, *args: str, cwd: str = None) -> ExampleRun:
    """Run a chapter example as a subprocess, once per argument tuple.

    Args:
        module: Importable module path, e.g. "ch7_slam.example_pose_graph_slam".
        *args: Command-line arguments passed to the module.
        cwd: Working directory to run from, defaulting to the repository root.
            Part of the memoisation key, so a run from a chapter directory does
            not collide with the same invocation from the root. Only
            `tests/docs/test_examples_run_from_chapter_dir.py` passes it; every
            other caller wants the default.

    Returns:
        The shared ExampleRun for this invocation.
    """
    figs_root = _figs_root()

    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(WORKSPACE_ROOT),
            # Send the figures to scratch. Without this these tests rewrite the
            # tracked chX_*/figs/*, so every run manufactured a diff -- which is
            # why slam_with_maps.png kept turning up in git status. The examples
            # still name their own chX/figs; core.eval mirrors in-repo writes under
            # this root, preserving the repo-relative path.
            "IPIN_FIGS_DIR": str(figs_root),
            # Decode the child's stdout the same way on every machine. The examples
            # legitimately print U+00B0 (tests/test_example_console_encoding.py
            # keeps it legal on a legacy console), but on a CJK Windows host the
            # child encodes it as a cp950 byte, which arrives here as U+FFFD --
            # while on the Ubuntu runner it arrives as the degree sign. Any test
            # comparing this text against a file in the repo would then pass on CI
            # and fail locally, for a reason that has nothing to do with the code.
            "PYTHONIOENCODING": "utf-8",
        }
    )

    started_at = time.time()
    process = subprocess.run(
        [sys.executable, "-m", module, *args],
        # The repo root by default. Dataset paths now resolve against the
        # working directory *and then* the repository root
        # (core.utils.resolve_data_path), so this is no longer load-bearing for
        # dataset loading -- but it is what every existing caller expects, and
        # test_examples_run_from_chapter_dir exists to hold the other case.
        # Only the figure output is diverted, and by IPIN_FIGS_DIR above.
        cwd=cwd or WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=EXAMPLE_TIMEOUT_S,
        env=env,
    )
    return ExampleRun(
        process=process,
        started_at=started_at,
        figs_dir=figs_root / module.split(".")[0] / "figs",
    )
