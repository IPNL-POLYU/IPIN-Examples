"""The mypy debt is written down, and it may only get smaller.

`pyproject.toml` has configured mypy strictly for as long as it has existed --
`disallow_untyped_defs`, `warn_return_any`, `check_untyped_defs`,
`strict_equality` -- and the README used to tell readers to run it. CI runs a
single pytest job and has never invoked mypy once. A configured checker that is
not run is indistinguishable from one that passes, which is how 408 errors
accumulated without anyone deciding to accept them.

This does not fix them. It writes them down per error code and refuses to let
the number grow, exactly as `test_lint_debt_only_shrinks.py` does for ruff, and
for the same reason: a single total would let ten fixed `no-untyped-def` pay
for ten new `arg-type`.

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

**Two things move these counts that are not this repository's code**, and both
should be fixed by re-recording the baseline rather than by loosening anything:
installing `scipy-stubs` would resolve four of the six `import-untyped`, and a
numpy release ships new stubs, which is what most of `no-any-return` and
`arg-type` are measured against.

Author: Li-Ta Hsu
"""

import functools
import re

import pytest

from tests.example_runner import WORKSPACE_ROOT

#: Errors per mypy code today. Only ever edit these downwards, and delete the
#: entry entirely at zero -- an absent code is the guard, because the
#: `appeared` check below fails on any code not listed here.
#:
#: 125 in 21 files, of 76 checked. `no-any-return` (42) is still the large one
#: and it is shallow: returning `Any` from a function declared to return
#: something narrower, overwhelmingly numpy operations whose stubs give back
#: `Any`. The one with content is `union-attr` (23), the shape that becomes
#: AttributeError at runtime, and it is now the second-largest entry.
#:
#: Wave B took `core/fusion`, `core/utils`, `core/models` and `core/sim` to
#: zero, which is the 82 errors between the 413 measured under mypy 2.3.1 and
#: the 331 that stood before Wave C. `unreachable` fell from 10 to 2 on
#: annotations alone: every one of the eight was a parameter defaulting to
#: None while annotated as non-optional, so mypy called the `if x is None:`
#: branch dead. The branches were live and load-bearing -- deleting them would
#: have removed the `alpha`/`confidence` deprecation path and two noise-std
#: defaults.
#:
#: Wave C took `core/estimators` to zero: the 99 errors between 331 and the
#: 232 here, and `override` with them, which is why that entry is gone. 86 of
#: the 99 were `no-untyped-def` on the `check_*` self-checks and the process
#: and measurement closures inside them -- the demonstration code at the foot
#: of each filter, which nothing had ever annotated. The whole wave is
#: annotations, `cast()` and one `assert`; all eight modules print
#: byte-identical self-check output on both sides of it.
#:
#: Two errors are silenced rather than fixed, both in
#: `iterated_extended_kalman_filter.py` and both carrying a comment saying
#: why. The `[override]` on `update` is deliberate -- it returns the iteration
#: count where the base class returns None. The `[operator]`/`[union-attr]`
#: pair on the covariance update is a real latent crash at
#: `max_iterations <= 0`, left for a behaviour change to fix.
#:
#: Wave D took `core/rf` and `core/sensors` to zero: the 107 errors between
#: 232 and the 125 here, and `operator` with them, which is why that entry is
#: gone. 30 of the 107 sat in one function, `AOAPositioner._compute_weight_
#: matrix`, where `np.isscalar(sigma)` guards each branch -- it is not a
#: TypeGuard, so mypy keeps the whole `float | np.ndarray` union in both arms
#: and every `sigma**2` and `sigma[i]` in the body reads as an error.
#:
#: The one error worth reading is the last `unreachable`'s twin. Wave B's
#: `wrap_angle` lesson -- that `float -> float` on a wrapping helper is a lie
#: the suite already disproves -- had a second live instance here in
#: `core.sensors.pdr.wrap_heading`, and the sanctioned scalar repair
#: (`return float(wrapped)`) raises "only 0-dimensional arrays can be
#: converted to Python scalars" against `test_the_helpers_vectorise`. Verified
#: by applying it and watching that test go red, not by reading. Both it and
#: its body-identical sibling `core.sensors.wrap_angle_diff` are now
#: `float | np.ndarray`, which is what `core.utils.angle_diff` beside them
#: already said.
# Measured with mypy==2.3.1 (pinned in the dev extra -- see pyproject.toml for
# why the pin is exact) under numpy 2.4.6. numpy is NOT pinned, and its stubs
# move these counts: a numpy release can turn this red in either direction, at
# which point re-measure and update in the same commit as the numpy bump.
BASELINE = {
    "no-any-return": 42,
    "union-attr": 23,
    "no-untyped-def": 17,
    "assignment": 16,
    "arg-type": 14,
    "import-untyped": 4,
    "attr-defined": 3,
    "return-value": 2,
    "dict-item": 1,
    "index": 1,
    "unreachable": 1,
    "var-annotated": 1,
}

#: mypy writes one error per line and puts the code last, in brackets.
_ERROR_CODE = re.compile(r"\[([a-z-]+)\]\s*$", re.MULTILINE)


@functools.lru_cache(maxsize=1)
def _mypy_counts():
    """Errors per code, from one in-process mypy run shared by both tests.

    Cached because a second run costs another 20-60 s on a cold cache and both
    tests below need the same numbers. Nothing in here asserts, so the
    `lru_cache` does not have the "exceptions are not cached, so a failing
    memoised helper re-runs its whole body per caller" problem.

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

    counts: dict[str, int] = {}
    for match in _ERROR_CODE.finditer(stdout):
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


def test_no_error_code_has_more_errors_than_it_did():
    """Every code's count must be at or below its recorded number."""
    pytest.importorskip("mypy", reason="mypy is declared in the dev extra")
    counts = _mypy_counts()

    grown = {
        code: (count, BASELINE[code])
        for code, count in counts.items()
        if code in BASELINE and count > BASELINE[code]
    }
    appeared = {code: count for code, count in counts.items() if code not in BASELINE}

    assert not grown and not appeared, (
        "The mypy debt grew.\n\n"
        + "".join(
            f"  {code}: {now} errors, was {then}\n"
            for code, (now, then) in sorted(grown.items())
        )
        + "".join(
            f"  {code}: {now} errors, new\n" for code, now in sorted(appeared.items())
        )
        + "\nRun `python -m mypy core` to see them. Annotate the new code, or "
        "-- if a check is genuinely not worth following here -- turn it off in "
        "pyproject.toml with a reason, rather than raising the number."
    )


def test_the_recorded_numbers_are_not_stale():
    """A baseline above the real count hides the debt it is meant to expose."""
    pytest.importorskip("mypy", reason="mypy is declared in the dev extra")
    counts = _mypy_counts()

    stale = {
        code: (counts.get(code, 0), recorded)
        for code, recorded in BASELINE.items()
        if counts.get(code, 0) < recorded
    }

    assert not stale, (
        "These codes have fewer errors than recorded, which is good news that "
        "has to be written down or the ratchet stops ratcheting:\n\n"
        + "".join(
            f"  {code}: {now} errors, recorded {then}\n"
            for code, (now, then) in sorted(stale.items())
        )
        + "\nLower the numbers in BASELINE, and delete the entry entirely at "
        "zero."
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
        f"mypy {installed} is installed but BASELINE was measured with {pin}. "
        f"Run: pip install -e .[dev]"
    )
