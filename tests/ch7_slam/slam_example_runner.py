"""Run a Chapter 7 SLAM example once per distinct invocation, and share it.

Five tests across two files were each shelling out to
``ch7_slam.example_pose_graph_slam``, three of them with identical arguments,
at ``timeout=180``. The margin was far thinner than it looked: the script
takes ~135 s standalone, but measured inside the suite the same run took
170.8 s, and after consolidation a single run was observed at 207.0 s -- past
the old limit on its own. So the suite was a coin flip. Identical content
produced 2 failed / 1351 passed in 954 s once, then 1353 passed / 0 failed in
647 s when the machine was quieter. A test suite whose answer depends on what
else is running is not a signal, and every contributor who hit it paid for the
same investigation.

Three changes, all here rather than in the tests:

  - Memoise on the argument tuple, so each distinct invocation runs once per
    pytest session and the assertions share the result. Five distinct
    invocations remain, because they exercise genuinely different paths: the
    pose graph inline, on the square dataset and on the high-drift dataset,
    plus the front-end and the scan-matching visualisation.
  - Raise the timeout well clear of the runtime. The timeout is a deadlock
    guard, not a performance budget; a tight one buys nothing and costs false
    failures. If the example ever gets slow enough to trip this, that is worth
    a real investigation rather than a flaky red.
  - Point ``IPIN_FIGS_DIR`` at scratch, so a run cannot rewrite the committed
    figures. Assert against ``ExampleRun.figs_dir``, not the in-repo path.

Every Chapter 7 test that shells out to an example goes through here. The
scan-matching visualisation was the last holdout, and it is worth saying why
it was folded in even though it was already correct: it had its own copy of
the env setup, the timeout and the figure diversion, and duplicated policy
only has to be forgotten once. Its ``TemporaryDirectory`` also died with the
class that owned it, where this scratch root lives for the session, so another
file can assert on the same run without provoking a second one.

Author: Li-Ta Hsu
"""

import json
import re
from typing import Any, Dict, Optional

# The generic half of this module now lives in tests/example_runner.py, because
# a second caller outside Chapter 7 needed it (tests/docs, which checks the
# README transcripts against what the examples actually print). Re-exported
# rather than reimplemented: run_example carries the lru_cache, so both callers
# must share this one function object or the same invocation runs twice.
from tests.example_runner import WORKSPACE_ROOT, ExampleRun, run_example

# WORKSPACE_ROOT is re-exported: two Chapter 7 test modules import it from here
# and predate the split. __all__ rather than a noqa, because the pyflakes
# ratchet in test_repo_conventions runs pyflakes directly and does not read
# ruff's suppressions -- which is how the first attempt at this import failed.
__all__ = [
    "ExampleRun",
    "run_example",
    "WORKSPACE_ROOT",
    "POSE_GRAPH_MODULE",
    "FRONTEND_MODULE",
    "SCAN_MATCHING_MODULE",
    "run_pose_graph_example",
    "run_frontend_example",
    "run_scan_matching_example",
    "parse_slam_summary",
]

POSE_GRAPH_MODULE = "ch7_slam.example_pose_graph_slam"
FRONTEND_MODULE = "ch7_slam.example_slam_frontend"
SCAN_MATCHING_MODULE = "ch7_slam.example_scan_matching_visualization"


def run_pose_graph_example(*args: str) -> ExampleRun:
    """Run the pose-graph SLAM example; shared across tests and files."""
    return run_example(POSE_GRAPH_MODULE, *args)


def run_frontend_example(*args: str) -> ExampleRun:
    """Run the SLAM front-end example; shared across tests and files."""
    return run_example(FRONTEND_MODULE, *args)


def run_scan_matching_example(*args: str) -> ExampleRun:
    """Run the scan-matching visualisation example; shared across tests."""
    return run_example(SCAN_MATCHING_MODULE, *args)


def parse_slam_summary(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse the [SLAM_SUMMARY] JSON line from script output.

    Args:
        stdout: Standard output from the SLAM script.

    Returns:
        Parsed JSON dictionary, or None if the line is absent.

    Raises:
        ValueError: If the summary line is present but malformed.
    """
    match = re.search(r"\[SLAM_SUMMARY\]\s*(\{.*\})", stdout)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed SLAM_SUMMARY JSON: {e}")
