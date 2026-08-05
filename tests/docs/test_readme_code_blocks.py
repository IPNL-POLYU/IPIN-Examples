"""The Python blocks in the dataset READMEs must run.

Replaces `tests/docs/test_ch6_examples.py`, which did this and had never run:
it held a `main()` and no `def test_`, so pytest collected nothing from it.

Running it changed the answer twice over.

It executed every block in a *fresh* namespace and reported 13 of 32 failing.
That is not the reader's experience -- a README is read top to bottom, and its
later blocks legitimately use names the earlier ones introduced. Under a
shared namespace, which is what a reader actually has, 8 fail rather than 13.

Those 8 are real: `alt_prev`, `mag`, `heading_gyro`, `mean_step_length` and
`quat` are never defined anywhere in their README, in any order. A reader
copying those blocks gets a NameError.

The old file also looked only at Chapter 6. Widening it to every dataset
README takes the count from 8 to **30 broken blocks across 10 datasets**,
Chapters 2 through 8. Chapter 3's README fails 5 of its 8 blocks.

Listed rather than fixed here, on the same ratchet as everywhere else in this
repo: the count may only go down, and repairing eight documentation snippets
is its own change with its own review.

Author: Li-Ta Hsu
"""

import contextlib
import io
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- after use()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Blocks that fail even when the README is read in order, as {dataset: count}.
#:
#: THIS MAY ONLY SHRINK. Each entry is a snippet a reader cannot run.
KNOWN_BROKEN_BLOCKS = {
    "ch2_coords_san_francisco": 2,
    "ch3_estimator_nonlinear": 5,
    "ch4_rf_2d_square": 2,
    "ch5_wifi_fingerprint_grid": 4,
    "ch6_env_sensors_heading_altitude": 4,
    "ch6_pdr_corridor_walk": 3,
    "ch6_wheel_odom_square": 1,
    "ch7_slam_2d_square": 5,
    "ch8_fusion_2d_imu_uwb": 1,
    "ch8_fusion_2d_imu_uwb_timeoffset": 3,
}


def _readmes():
    """Dataset READMEs carrying at least one Python block."""
    found = []
    for path in sorted(REPO_ROOT.glob("data/sim/*/README.md")):
        if "```python" in path.read_text(encoding="utf-8"):
            found.append(path)
    return found


def _dataset_name(path):
    return path.parent.name


@pytest.mark.parametrize("readme", _readmes(), ids=_dataset_name)
def test_readme_blocks_run_in_order(readme):
    """Execute every block into one namespace, as a reader would."""
    name = _dataset_name(readme)
    blocks = re.findall(
        r"```python\n(.*?)\n```", readme.read_text(encoding="utf-8"), re.DOTALL
    )
    assert blocks, f"{name}: no Python blocks found"

    # Neutralise figure writing for the duration. These blocks are run to
    # prove they execute, not to produce pictures, and several of them call
    # savefig with a bare filename -- which, with the working directory at the
    # repo root so their `data/sim/...` paths resolve, drops eight SVGs there
    # and trips test_no_figures_written_to_the_repo_root.
    #
    # Worth knowing that a `git status` check does not catch this: the repo
    # ignores stray figures, so they are invisible to porcelain and visible
    # only to a filesystem glob. CI found it; the local check had not.
    namespace = {}
    failures = []
    saved = (plt.savefig, plt.show, plt.Figure.savefig)
    plt.savefig = lambda *a, **k: None
    plt.show = lambda *a, **k: None
    plt.Figure.savefig = lambda *a, **k: None
    with contextlib.chdir(REPO_ROOT):
        for number, block in enumerate(blocks, start=1):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(block, f"{name}#{number}", "exec"), namespace)
            except Exception as exc:  # noqa: BLE001 -- reporting, not handling
                failures.append(f"block {number}: {type(exc).__name__}: {exc}")
            finally:
                plt.close("all")
    plt.savefig, plt.show, plt.Figure.savefig = saved

    allowed = KNOWN_BROKEN_BLOCKS.get(name, 0)
    assert len(failures) <= allowed, (
        f"{name}: {len(failures)} of {len(blocks)} blocks fail, above the "
        f"{allowed} on record.\n  " + "\n  ".join(failures)
    )
    assert len(failures) == allowed or allowed == 0, (
        f"{name}: only {len(failures)} blocks fail now, against {allowed} on "
        f"record. Lower the entry in KNOWN_BROKEN_BLOCKS."
    )
