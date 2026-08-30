"""The lint debt is written down, and it may only get smaller.

The README used to tell readers to run four tools -- black, ruff, mypy, pylint --
as though the repository passed them. Measured against the 304 tracked Python
files, it did not, and not narrowly:

    ruff    5836 findings
    black   237 of 288 files would be reformatted
    mypy    404 errors in core/ alone

Black passes now, and ruff is at 951. mypy is untouched at 406 errors in
core/, and is the honest remaining gap.

A reader who followed that section got thousands of complaints and reasonably
concluded they had broken something. The README says what is true now; this
holds the ruff half of it so the number cannot quietly grow back.

**Whitespace was 83% of the original number and is gone**, in two passes that
are worth telling apart. `ruff --fix` cleared 3961 of the 4868 W291/W293 and
refused the rest, because they sat inside string literals where whitespace is
content rather than layout. Running black then cleared 889 of the remaining
907 -- **black knows which triple-quoted strings are docstrings**, and
normalises those, where ruff could only see a string. A tool declining an
unsafe fix was right; the answer was a tool that could tell the difference, not
`--unsafe-fixes`.

The twelve W293 still here sit in argparse `epilog=` strings, which are not
docstrings and whose blank lines are printed. Black leaves them, correctly.

**What is left is mostly not lint at all.** 727 of the remaining findings are
UP006/UP045/UP035/UP007 -- `List[int]` for `list[int]`, `Optional[X]` for
`X | None`. Those became legal only when the floor moved to 3.10, they are
mechanical, and they are worth doing in their own change. The ~100 after that
are the ones with actual content -- B905 was the largest of them and is now
gone, audited rather than swept: see the comment on BASELINE below.

Per-rule rather than a single total on purpose: a total lets ten fixed W293
pay for ten new B905, which is the opposite of what a ratchet is for.

Author: Li-Ta Hsu
"""

import subprocess
import sys

import pytest

from tests.example_runner import WORKSPACE_ROOT

#: The directories that are this repository's own Python.
SOURCE_DIRS = (
    "core",
    "scripts",
    "tools",
    "tests",
    "ch2_coords",
    "ch3_estimators",
    "ch4_rf_point_positioning",
    "ch5_fingerprinting",
    "ch6_dead_reckoning",
    "ch7_slam",
    "ch8_sensor_fusion",
)

#: Findings per rule today. Only ever edit these downwards.
#:
#: UP0xx are the annotation modernisations that the 3.10 floor made available,
#: and are the bulk of what is left.
#:
#: B905 is absent, and its absence is the guard: the `appeared` check below
#: fails on any rule not listed here, so one new `zip()` without `strict=`
#: turns this red.
#: The 41 it used to record were read one at a time rather than swept, which is
#: how the two that mattered were found -- a boxplot palette one shade short of
#: its methods, and one site where `strict=True` is simply wrong, because
#: `zip(xs, xs[1:])` relies on the truncation. The twelve remaining W293 sit
#: inside argparse `epilog` strings,
#: where the whitespace is content that gets printed rather than layout --
#: black leaves those alone, correctly, and so should you.
#:
#: UP007 fell 38 -> 34 in the mypy Wave C change, which is not an annotation
#: migration: `KalmanFilter` spelled `Union[np.ndarray, Callable]` at five
#: separate sites, and they became one `MatrixOrCallable` alias. Naming a
#: repeated union once is what removed four findings; the wave otherwise
#: writes its new annotations in PEP 585/604 form so it adds none.
#:
#: UP045 fell 137 -> 133 in mypy Wave E, for the same reason and not as a
#: migration either: `normalize_fingerprint` and `preprocess_query` each had
#: two reference statistics annotated `Optional[np.ndarray]` while their own
#: bodies assign a float into them. Widening those four is what the wave
#: needed, and `float | np.ndarray | None` is the PEP 604 spelling of the
#: result, so four `Optional[...]` left the tree as a side effect. Every other
#: rule is unchanged: the wave's new annotations are PEP 585/604 throughout.
BASELINE = {
    "UP006": 283,
    "UP045": 133,
    "UP035": 87,
    "I001": 59,
    "UP007": 34,
    "B007": 17,
    "B028": 9,
    "E712": 4,
    "W293": 10,
    "E731": 6,
    "B904": 6,
    "E741": 4,
    "UP015": 4,
    "C408": 2,
    "B017": 2,
    "C420": 2,
    "UP033": 2,
    "C401": 1,
    "C416": 1,
    "UP032": 1,
}


def _ruff_counts():
    """Findings per rule code, from ruff's own statistics output."""
    process = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--statistics", *SOURCE_DIRS],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    counts = {}
    for line in process.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        counts[parts[1].strip()] = int(parts[0].strip())
    return counts


def test_no_rule_has_more_findings_than_it_did():
    """Every rule's count must be at or below its recorded number."""
    pytest.importorskip("ruff", reason="ruff is declared in the dev extra")
    counts = _ruff_counts()

    grown = {
        code: (count, BASELINE[code])
        for code, count in counts.items()
        if code in BASELINE and count > BASELINE[code]
    }
    appeared = {code: count for code, count in counts.items() if code not in BASELINE}

    assert not grown and not appeared, (
        "The lint debt grew.\n\n"
        + "".join(
            f"  {code}: {now} findings, was {then}\n"
            for code, (now, then) in sorted(grown.items())
        )
        + "".join(
            f"  {code}: {now} findings, new\n" for code, now in sorted(appeared.items())
        )
        + "\nFix them, or -- if a rule is genuinely not worth following here -- "
        "add it to the ignore list in pyproject.toml with a reason, rather than "
        "raising the number."
    )


def test_the_recorded_numbers_are_not_stale():
    """A baseline above the real count hides the debt it is meant to expose."""
    pytest.importorskip("ruff", reason="ruff is declared in the dev extra")
    counts = _ruff_counts()

    stale = {
        code: (counts.get(code, 0), recorded)
        for code, recorded in BASELINE.items()
        if counts.get(code, 0) < recorded
    }

    assert not stale, (
        "These rules have fewer findings than recorded, which is good news that "
        "has to be written down or the ratchet stops ratcheting:\n\n"
        + "".join(
            f"  {code}: {now} findings, recorded {then}\n"
            for code, (now, then) in sorted(stale.items())
        )
        + "\nLower the numbers in BASELINE, and delete the entry entirely at zero."
    )
