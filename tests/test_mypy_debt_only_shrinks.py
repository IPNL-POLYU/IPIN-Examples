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
#: 408 in 56 files, of 76 checked. The two large ones are shallow: 119
#: `no-untyped-def` is missing annotations, and 105 `no-any-return` is
#: returning `Any` from a function declared to return something narrower --
#: overwhelmingly numpy operations whose stubs give back `Any`. The ones with
#: content are further down: `union-attr` (32) and `index` (8) are the shapes
#: that become AttributeError and IndexError at runtime.
BASELINE = {
    "no-untyped-def": 119,
    "no-any-return": 105,
    "arg-type": 40,
    "assignment": 35,
    "union-attr": 32,
    "operator": 27,
    "unreachable": 10,
    "return-value": 9,
    "dict-item": 8,
    "index": 8,
    "import-untyped": 6,
    "attr-defined": 4,
    "var-annotated": 4,
    "override": 1,
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
