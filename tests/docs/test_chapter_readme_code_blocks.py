"""The Python blocks in the chapter READMEs must run.

`test_readme_code_blocks.py` next door does this for `data/sim/*/README.md` and
has done since a sweep found eight genuine API drifts there. It never covered the
chapter READMEs, which are the more read of the two -- and the transcript checker
added later reads their *output* blocks, not their code. So `ch*/README.md` was
the one class of executable documentation in this repository that nothing ran.

It had drifted, as unexecuted code does. Of 42 blocks, 16 failed. Applying the
rule from the sibling module -- **the exception type tells you which kind you
have** -- three were real:

- `ch5` built 3-, 2- and 6-element query vectors against a database with 8 APs,
  so `normalize_fingerprint`, `preprocess_query` and `nn_localize` all raised
  ValueError on the shape. Widened to eight values, which a reader can now run.
- `ch4`'s clock-bias block passed a 2-element `initial_guess` to
  `toa_solve_with_clock_bias`, which estimates `[x, y, bias_m]` and needs three.
- `ch4`'s TDOA covariance block inherited a *five*-anchor `anchors` from an
  earlier block on the same page and then passed three measurements, so it wanted
  four. It now derives its own inputs. Worth knowing that this is exactly what a
  reader working top to bottom would hit: the namespace is shared, and an earlier
  example's leftovers are in scope.

The other thirteen were NameErrors on placeholders, which means fragments. Ten
are fenced ```py and pinned below; three (`ch4`) were made runnable instead,
because the data they needed was already loaded a few blocks above.

**Fencing a fragment freezes whatever it says**, so each of the ten was checked
against the real signature even though none can execute as written. The two ch7
corrections below were then verified by *running* them: a scratch harness built
the odometry deltas from the shipped square dataset, and confirmed all four dict
keys are present over 40 steps and that `detect(scans, poses)` returns one
`LoopClosure` with every field the README names readable. Signature inspection
would have been enough to find the bugs; execution is what proves the fix.

That found two defects, both in `ch7_slam/README.md`, and both cases of two
documents describing one API differently:

- `SlamFrontend2D.step()` returns a **dict**; the README unpacked it as a
  three-tuple. `QUICK_START.md` had it right.
- `LoopClosureDetector2D.detect()` takes **scans first**; the README passed
  poses first. `QUICK_START.md` had that right too, and the README also
  described the return as tuples where it is a list of `LoopClosure`.

Conventions, matching the sibling module exactly:

- ```python -- a reader is expected to run this, and this file executes it.
- ```py     -- an illustrative fragment using placeholders the reader supplies.

Both still highlight as Python everywhere that matters. `FRAGMENT_FENCES` pins
the ```py count per file, because otherwise the fence is an unreviewed escape
hatch: demoting a broken example to ```py would silence this check in a one-word
diff.

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

#: ```py fences per chapter document: fragments, deliberately not executed.
#:
#: A new entry must be a fragment, not a broken example. If a block fails with
#: anything other than a NameError on a placeholder, the names resolved and the
#: call is wrong -- fix it rather than demoting it.
#:
#: And check the fragment against the real signature before fencing it. Fencing
#: is what froze `detect(poses, scans)` and a three-tuple unpack of a dict in
#: ch7 for as long as nothing ran these.
FRAGMENT_FENCES = {
    # strapdown_update(q, v, p, ...) shown with the state the reader supplies.
    "ch6_dead_reckoning/README.md": 1,
    # The front-end loop, the loop-closure detector, graph construction and
    # map building, each shown as a call shape over a reader's own data.
    "ch7_slam/README.md": 4,
    # Same three stages, as a quick-start walkthrough.
    "ch7_slam/QUICK_START.md": 3,
    # Calibration from IMU samples and sensor-pair positions the reader collects.
    "ch8_sensor_fusion/README.md": 2,
}


def _documents():
    """Every chapter README and quick-start guide, in a stable order."""
    found = sorted(REPO_ROOT.glob("ch*_*/README.md"))
    found += sorted(REPO_ROOT.glob("ch*_*/QUICK_START.md"))
    return found


def _name(path):
    return f"{path.parent.name}/{path.name}"


@pytest.mark.slow
@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_chapter_blocks_run_in_order(document):
    """Execute every ```python block into one namespace, as a reader would."""
    name = _name(document)
    blocks = re.findall(
        r"```python\n(.*?)\n```", document.read_text(encoding="utf-8"), re.DOTALL
    )
    if not blocks:
        pytest.skip(f"{name}: only fragments, all pinned in FRAGMENT_FENCES")

    # Neutralise figure writing, as the sibling module does and for the same
    # reason: several blocks call savefig with a bare filename, and the working
    # directory has to be the repo root for their data/sim paths to resolve.
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
        "fence it ```py and raise its FRAGMENT_FENCES count. Anything else "
        "means the names resolved and the call is wrong -- fix the block. "
        "Note the namespace is shared down the page, so an earlier block's "
        "leftovers are in scope; a block that reuses a name like `anchors` "
        "should re-derive it."
    )


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_fragment_fence_count_is_pinned(document):
    """No new ```py fences without a deliberate line in the diff."""
    name = _name(document)
    text = document.read_text(encoding="utf-8")
    found = len(re.findall(r"^```py\s*$", text, re.M))
    expected = FRAGMENT_FENCES.get(name, 0)

    assert found == expected, (
        f"{name} has {found} ```py fences, expected {expected}.\n\n"
        f"A ```py fence exempts a block from execution. Adding one is fine when "
        f"the block really is a fragment, but it has to be a deliberate change "
        f"to FRAGMENT_FENCES -- and check the fragment against the real "
        f"signature first, since nothing will run it again."
    )
