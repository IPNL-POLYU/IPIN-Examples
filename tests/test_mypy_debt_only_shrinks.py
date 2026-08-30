"""Any mypy error in `core/` is a failure. There is no debt left to ratchet.

`pyproject.toml` has configured mypy strictly for as long as it has existed --
`disallow_untyped_defs`, `warn_return_any`, `check_untyped_defs`,
`strict_equality` -- and the README used to tell readers to run it. CI runs a
single pytest job and had never invoked mypy once. A configured checker that is
not run is indistinguishable from one that passes, which is how 408 errors
accumulated without anyone deciding to accept them.

**That number is now zero, and this file changed meaning when it got there.**
It began as a ratchet: a count per error code that could only be edited
downwards, exactly as `test_lint_debt_only_shrinks.py` still is for ruff, and
for the same reason -- a single total would have let ten fixed
`no-untyped-def` pay for ten new `arg-type`. The count started at 408 under
mypy 1.19 and was re-measured at 413 when the pin moved to 2.3.1, then fell in
four waves: B took `core/fusion`, `core/utils`, `core/models` and `core/sim`
(413 -> 331), C took `core/estimators` (331 -> 232), D took `core/rf` and
`core/sensors` (232 -> 125), and E took `core/coords`, `core/eval`,
`core/fingerprinting` and `core/slam`, which was the rest. The ratchet's own
instruction was to delete each code's entry at zero, because an absent code
already failed the `appeared` check; carried to its conclusion, deleting the
last entry leaves the assertion below, which is that mypy reports nothing at
all. So this is a tripwire now, not a budget, and it prints the mypy output on
failure so the red names the lines rather than a number.

**Scope is `core/` only, and that is a decision rather than an oversight.**
`core/` is the shared library every chapter imports, so a wrong annotation
there is wrong in seven places. The chapters are worked examples -- 31 scripts
whose job is to be read alongside the book -- and whether they should carry
full annotations is a separate question with a different answer. Widening the
scope means measuring the chapters first and writing down whatever that finds;
it is not a one-word change to this file.

**Runtime: 1.5-3.5 s settled, but the first run on a clean checkout is
33-63 s, and the run after that is still ~20 s.** mypy's cache takes two passes
to settle. Measured both through `mypy.api.run` and through
`python -m mypy core`, which behave the same -- an early comparison here looked
like the API was 16x faster and was really just comparing a settled cache
against a fresh one. CI starts with no `.mypy_cache` (only pip is cached), so
expect the slow number there every time. It is not a hung test.

**The numpy-stub sensitivity did not go away, and at zero it bites harder.**
numpy is NOT pinned, and its stubs are what most of the old `no-any-return` and
`arg-type` entries were measured against; installing `scipy-stubs` would
likewise change what `import-untyped` sees. Under the ratchet a stub change
moved a number. Under this assertion it turns the suite red, which is the
intended trade -- new errors now have to be read and annotated (or the change
argued for in `pyproject.toml`) rather than absorbed by raising a baseline.
Expect that on a numpy bump, and fix it in the same commit as the bump.

The stale-baseline companion test that used to sit here is gone rather than
kept trivially green: with `BASELINE` deleted it iterated an empty dict and
could not fail, and a test that cannot fail reads as coverage from every angle
except the one that decides whether it means anything.

Author: Li-Ta Hsu
"""

import functools

import pytest

from tests.example_runner import WORKSPACE_ROOT


@functools.lru_cache(maxsize=1)
def _mypy_output() -> str:
    """mypy's stdout over `core/`, from one in-process run.

    Cached because a run costs 20-60 s on a cold cache. Nothing in here
    asserts, so the `lru_cache` does not have the "exceptions are not cached,
    so a failing memoised helper re-runs its whole body per caller" problem.

    Every path is absolute and the config file is named explicitly, so the
    result does not depend on pytest's working directory. `.mypy_cache` is
    gitignored, so writing it does not disturb the clean-tree check in CI.
    """
    from mypy import api

    stdout, _stderr, _status = api.run(
        [
            "--config-file",
            str(WORKSPACE_ROOT / "pyproject.toml"),
            "--cache-dir",
            str(WORKSPACE_ROOT / ".mypy_cache"),
            str(WORKSPACE_ROOT / "core"),
        ]
    )
    return stdout


def test_mypy_reports_no_errors_in_core():
    """`core/` type-checks clean, and a single new error fails this."""
    pytest.importorskip("mypy", reason="mypy is declared in the dev extra")

    stdout = _mypy_output()
    errors = [line for line in stdout.splitlines() if ": error: " in line]

    assert not errors, (
        f"mypy reports {len(errors)} error(s) in core/, which has been at zero "
        "since Wave E. There is no baseline to raise: annotate the new code, "
        "or -- if a check is genuinely not worth following here -- turn it off "
        "in pyproject.toml with a reason.\n\n"
        + "\n".join(errors)
        + "\n\nIf this appeared after a numpy or scipy upgrade, the stubs "
        "moved rather than this repository's code; that still has to be "
        "annotated, in the same commit as the upgrade."
    )


def test_the_installed_mypy_is_the_pinned_one():
    """The guard is an assertion about a SPECIFIC mypy version.

    This exact failure shipped once already: the dev machine ran mypy 1.19.0
    while CI freshly installed 2.3.1, and the two counted different error
    totals over the same tree. The pin in pyproject.toml is the single source
    of truth; this test reads it rather than repeating it, and fails with the
    upgrade command instead of letting version skew masquerade as new debt.
    """
    from pathlib import Path

    import tomllib
    from mypy.version import __version__ as installed

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    pin = next(d.split("==")[1] for d in dev if d.startswith("mypy=="))
    assert installed == pin, (
        f"mypy {installed} is installed but this file's zero-error assertion "
        f"was measured with {pin}. Run: pip install -e .[dev]"
    )
