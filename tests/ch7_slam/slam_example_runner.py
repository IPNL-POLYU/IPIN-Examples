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

Two changes, both here rather than in the tests:

  - Memoise on the argument tuple, so each distinct invocation runs once per
    pytest session and the assertions share the result. Three distinct
    invocations remain (inline, square dataset, high-drift dataset) because
    they exercise genuinely different paths.
  - Raise the timeout well clear of the runtime. The timeout is a deadlock
    guard, not a performance budget; a tight one buys nothing and costs false
    failures. If the example ever gets slow enough to trip this, that is worth
    a real investigation rather than a flaky red.

Author: Li-Ta Hsu
"""

import functools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

POSE_GRAPH_MODULE = "ch7_slam.example_pose_graph_slam"
FRONTEND_MODULE = "ch7_slam.example_slam_frontend"

# Generous on purpose: see the module docstring. ~4x the observed 135 s.
EXAMPLE_TIMEOUT_S = 600


class ExampleRun(NamedTuple):
    """One completed example run, shared between tests.

    Attributes:
        process: The finished subprocess. Treat as read-only -- every test
            asserting on this invocation sees the same object.
        started_at: Wall-clock epoch seconds from just before launch. Lets a
            test tell a file this run wrote from a stale committed one.
    """

    process: subprocess.CompletedProcess
    started_at: float


@functools.lru_cache(maxsize=None)
def run_example(module: str, *args: str) -> ExampleRun:
    """Run a chapter example as a subprocess, once per argument tuple.

    Args:
        module: Importable module path, e.g. "ch7_slam.example_pose_graph_slam".
        *args: Command-line arguments passed to the module.

    Returns:
        The shared ExampleRun for this invocation.
    """
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONPATH": str(WORKSPACE_ROOT)})

    started_at = time.time()
    process = subprocess.run(
        [sys.executable, "-m", module, *args],
        # The example resolves both "data/sim" and "ch7_slam/figs" relative to
        # the working directory, so this cannot be redirected to a tmpdir
        # without breaking dataset loading. A consequence worth knowing: these
        # tests rewrite the tracked ch7_slam/figs/slam_with_maps.png, which is
        # why it keeps appearing in git status.
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=EXAMPLE_TIMEOUT_S,
        env=env,
    )
    return ExampleRun(process=process, started_at=started_at)


def run_pose_graph_example(*args: str) -> ExampleRun:
    """Run the pose-graph SLAM example; shared across tests and files."""
    return run_example(POSE_GRAPH_MODULE, *args)


def run_frontend_example(*args: str) -> ExampleRun:
    """Run the SLAM front-end example; shared across tests and files."""
    return run_example(FRONTEND_MODULE, *args)


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
