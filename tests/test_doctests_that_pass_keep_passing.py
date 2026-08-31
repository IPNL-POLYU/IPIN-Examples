"""The doctests that work today are held; the rest are measured, not gated.

54 files under `core/` carry `>>>` examples and nothing has ever run any of
them. Turning them all on is not the answer -- most were never written to be
run. Classified per file (the full table is in the pull request that added
this):

    PASS            12   every example runs and matches
    NO-EXPECTATION  13   `>>> print(p1)  # [0.01, 0, 0]` -- the expected value
                         is a trailing *comment*, so doctest's `want` is empty
                         and any output is a "mismatch"
    FRAGMENT         8   NameError on a name the reader supplies (`db`, `scans`)
    DRIFT           20   a written expectation that is false
    PARSE-ERROR      1   core/fingerprinting/classification.py, whose docstring
                         doctest cannot even parse

The middle two categories are the same one `docs/` is deliberately not
executed for: a ` ```python ` fence there has never meant "runnable", and a
`>>>` in an Args-and-Returns docstring has never meant it either. Gating them
would mean rewriting ~90 illustrative examples to protect the 139 that work,
which is the trade CLAUDE.md already measured and declined once.

So this is an **allowlist, not a blocklist**. A new file with doctests joins it
deliberately, by someone who checked the examples are claims rather than
illustrations. The 20 DRIFT files are real defects and are listed in the pull
request for follow-up; when one is repaired it belongs here.

**Bare optionflags, honouring only inline directives.** Enabling ELLIPSIS
globally would admit one more file (`core/fusion/gating.py`, whose `9.21...`
becomes a wildcard) -- which is admitting it on the strength of a loosened
comparison rather than a true claim. `core/coords/rotations.py` shows the
inline `# doctest: +ELLIPSIS` idiom for the cases that genuinely need it.

Author: Li-Ta Hsu
"""

import contextlib
import doctest
import importlib
import io

import pytest

from tests.example_runner import WORKSPACE_ROOT

#: Files whose doctests all pass, and the number of examples each held when it
#: was added. Only ever add a file here after running its doctests.
#:
#: The count is the second half of the guard. `doctest.testmod` on a module
#: with no examples left reports zero failures, so deleting the examples is a
#: way to make a file pass -- indistinguishable, from the assertion's side,
#: from fixing them. Same shape as the uncollected `test_`-named functions in
#: `core/estimators/`: it fails by never speaking at all.
#:
#: core/fusion/types.py joined this list in the same change. Its
#: `TimeSyncModel.to_fusion_time` example claimed `10.51` where the value is
#: 10.509999999999998 -- the code was right and the docstring had written the
#: arithmetic answer as though it were the repr.
#:
#: Two more DRIFT files repaired and admitted since. Both are worth the
#: sentence, because neither defect was the kind the category name suggests:
#:
#: - core/fusion/tuning.py: the arithmetic in all nine examples was correct.
#:   `np.isclose` returns a `numpy.bool_`, whose repr became `np.True_` in
#:   numpy 2, so two examples claiming `True` stopped matching without anything
#:   about the code or the claim changing. `bool(...)` around the comparison
#:   says what was meant. Same family as the matplotlib `labels=` removal: a
#:   library changed a repr under a docstring that was right.
#: - core/rf/dop.py: `compute_dop` claimed HDOP 1.41 and sigma_horizontal
#:   0.42 m for four anchors on a square seen from its centre. H^T H is 2I
#:   there, so HDOP is exactly 1.00 -- the optimal-geometry case, quoted at the
#:   one value it cannot take. `compute_dop_map` alongside it was a
#:   NO-EXPECTATION case (a `print` with no `want` line) and now has one.
ALLOWLIST = {
    "core/eval/plots.py": 2,
    "core/fingerprinting/shadowing.py": 4,
    "core/fusion/lc_models.py": 5,
    "core/fusion/tc_models.py": 4,
    "core/fusion/tuning.py": 50,
    "core/fusion/types.py": 13,
    "core/rf/dop.py": 23,
    "core/sensors/__init__.py": 14,
    "core/slam/__init__.py": 7,
    "core/slam/camera.py": 13,
    "core/slam/factors.py": 32,
    "core/slam/scan_descriptor_2d.py": 11,
    "core/slam/submap_2d.py": 31,
    "core/utils/paths.py": 3,
}


def _module_for(relative_path):
    """Import the module at a repo-relative path and return it."""
    return importlib.import_module(relative_path[: -len(".py")].replace("/", "."))


def _example_count(module, relative_path):
    """How many doctest examples this module holds, skipped ones included."""
    name = relative_path[: -len(".py")].replace("/", ".")
    return sum(
        len(test.examples) for test in doctest.DocTestFinder().find(module, name)
    )


@pytest.mark.parametrize("relative_path", sorted(ALLOWLIST))
def test_the_allowlisted_doctests_still_pass(relative_path):
    """Every example in an allowlisted file runs and matches what it claims."""
    assert (WORKSPACE_ROOT / relative_path).is_file(), (
        f"{relative_path} is in ALLOWLIST but does not exist. If it moved, "
        "move the entry; if its doctests went away, delete the entry."
    )

    module = _module_for(relative_path)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        result = doctest.testmod(module, verbose=False, report=False, optionflags=0)

    assert not result.failed, (
        f"{result.failed} of {result.attempted} doctests in {relative_path} "
        "no longer match what the docstring claims:\n\n" + report.getvalue()
    )


@pytest.mark.parametrize("relative_path", sorted(ALLOWLIST))
def test_no_allowlisted_file_has_quietly_lost_its_examples(relative_path):
    """Deleting the examples must not be a way to make a file pass."""
    module = _module_for(relative_path)
    found = _example_count(module, relative_path)
    recorded = ALLOWLIST[relative_path]

    assert found >= recorded, (
        f"{relative_path} holds {found} doctest examples, down from the "
        f"{recorded} recorded when it was allowlisted. Removing an example is "
        "how a file passes this gate without its claims being true. If the "
        "removal is deliberate, lower the number and say why in the commit."
    )
