"""The Python blocks in the dataset READMEs must run.

Replaces `tests/docs/test_ch6_examples.py`, which did this and had never run:
it held a `main()` and no `def test_`, so pytest collected nothing from it.

Running it changed the answer three times over.

It executed every block in a *fresh* namespace and reported 13 of 32 failing.
That is not the reader's experience -- a README is read top to bottom, and its
later blocks legitimately use names the earlier ones introduced. Under a
shared namespace, which is what a reader actually has, 8 fail rather than 13.

Widening it from Chapter 6 to every dataset README took that to 30 failures
across 10 datasets. **But most of that 30 was not broken code.** Twenty-one of
the blocks are illustrative fragments never meant to run standalone -- a bare
`quat = quat / np.linalg.norm(quat)` showing a normalisation, a block opening
`# In tc_uwb_imu_ekf.py, add parameter:`, or `Δt_{k+1} = Δt_k + w_Δt` which is
maths rather than Python. Executing those is the wrong check, and counting them
made the register mean "failed to execute" when it should mean "is wrong".

So fragments are now fenced ```py and runnable examples ```python. Both still
highlight as Python in GitHub, VS Code and mkdocs; only ```python is collected
here. **A block you expect a reader to run must be fenced ```python.**

`FRAGMENT_BLOCKS` pins how many ```py fences each README carries, because
otherwise the fence is an unreviewed escape hatch: demoting a genuinely broken
example to ```py would silence this test with a one-word diff. Adding a
fragment is fine, it just has to be a deliberate line in the diff.

The eight genuine defects found this way are the reason the check earns its
keep -- reading the prose reveals none of them:

- `ch5_wifi_fingerprint_grid` #1 used `db.n_samples`; it is now
  `db.n_reference_points`.
- `ch7_slam_2d_square` #4 and #9 imported `optimize_factor_graph` from
  `core.estimators`. No such function: optimisation is `FactorGraph.optimize`,
  a method, and `create_pose_graph` wants (from, to, relative_pose) triples
  rather than the bare index pairs the README passed.
- `ch7_slam_2d_square` #5 called `se2_apply` once per point; it takes the whole
  (N, 2) cloud and rejects a bare (2,) row.
- `ch3_estimator_nonlinear` #7 documented `f_jac=` / `h_jac=` keywords that do
  not exist -- the Jacobians are positional, next to the model each
  differentiates.
- `ch6_pdr_corridor_walk` #4 elided the computation it then plotted.
- `ch2_coords_san_francisco` #5 unpacked `ref_llh[0]`, a scalar; the file is a
  single row, so the (3,) vector unpacks directly.
- `ch4_rf_2d_square` #6 called `toa_solver.solve(ranges)`; `initial_guess` is
  required, not optional.
- `ch6_env_sensors_heading_altitude` #9 passed the whole pressure series to
  `pressure_to_altitude`, which takes one sample at a time. The same README
  maps it correctly 350 lines earlier.

Worth knowing how the last three were separated from the fragments, because
the rule generalises: **the exception type tells you which kind you have.**
A `NameError` on an undefined placeholder means the block was never meant to
run alone. A `TypeError`, `ValueError`, `ImportError` or `AttributeError` means
every name resolved and the *call* was wrong -- that is API drift, and it is
always real.

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

#: ```py fences per README: illustrative fragments, deliberately not executed.
#:
#: A new entry here must be a fragment, not a broken example. See the module
#: docstring: if the block fails with anything other than a NameError on a
#: placeholder, it is API drift and belongs in ```python, fixed.
FRAGMENT_BLOCKS = {
    "ch2_coords_san_francisco": 1,
    "ch3_estimator_nonlinear": 4,
    "ch4_rf_2d_square": 1,
    "ch5_wifi_fingerprint_grid": 3,
    "ch6_env_sensors_heading_altitude": 3,
    "ch6_pdr_corridor_walk": 2,
    "ch6_wheel_odom_square": 1,
    "ch7_slam_2d_square": 2,
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

    assert not failures, (
        f"{name}: {len(failures)} of {len(blocks)} ```python blocks fail.\n  "
        + "\n  ".join(failures)
        + "\n\nA NameError on a placeholder means the block is a fragment: "
        "fence it ```py and add it to FRAGMENT_BLOCKS. Anything else means "
        "the names resolved and the call is wrong -- fix the block."
    )


@pytest.mark.parametrize("readme", _readmes(), ids=_dataset_name)
def test_fragment_fence_count_is_pinned(readme):
    """```py is an opt-out from the check above, so it is counted."""
    name = _dataset_name(readme)
    found = len(re.findall(r"```py\n", readme.read_text(encoding="utf-8")))
    expected = FRAGMENT_BLOCKS.get(name, 0)
    assert found == expected, (
        f"{name}: {found} ```py fragment fences, against {expected} on record. "
        "Update FRAGMENT_BLOCKS if the change is deliberate -- but a block that "
        "a reader is meant to run belongs in ```python."
    )
