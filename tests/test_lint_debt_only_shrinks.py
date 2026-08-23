"""The lint debt is written down, and it may only get smaller.

The README used to tell readers to run four tools -- black, ruff, mypy, pylint --
as though the repository passed them. Measured against the 304 tracked Python
files, it did not, and not narrowly:

    ruff    5836 findings
    black   237 of 288 files would be reformatted
    mypy    404 errors in core/ alone

A reader who followed that section got thousands of complaints and reasonably
concluded they had broken something. The README says what is true now; this
holds the ruff half of it so the number cannot quietly grow back.

**Most of the original number was whitespace and is gone.** 4737 W293 (a blank
line containing spaces) and 131 W291 accounted for 83% of it. Ruff fixed 3961
of them, and every one of the 3961 changed lines was verified to differ only by
trailing whitespace before the change was believed.

**Ruff declined to fix the rest, and it was right to.** The 907 that remain sit
inside docstrings, where the whitespace is string content rather than layout --
and in this repository that content is *printed*, because every example now
passes `description=__doc__` to argparse. A tool refusing an unsafe fix is not
an obstacle to work around with `--unsafe-fixes`.

**What is left is mostly not lint at all.** 727 of the remaining findings are
UP006/UP045/UP035/UP007 -- `List[int]` for `list[int]`, `Optional[X]` for
`X | None`. Those became legal only when the floor moved to 3.10, they are
mechanical, and they are worth doing in their own change rather than smuggled
into this one. The ~200 after that are the ones with actual content: 41 `zip()`
calls without `strict=`, which truncate silently to the shorter argument, are
the interesting ones.

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
#: W293/W291 are docstring whitespace ruff will not safely fix, see the module
#: docstring. UP0xx are the annotation modernisations that the 3.10 floor made
#: available. B905 is the one group worth reading before fixing: `zip()` without
#: `strict=` truncates to the shorter argument without saying so.
BASELINE = {
    "W293": 831,
    "UP006": 386,
    "UP045": 178,
    "UP035": 123,
    "I001": 113,
    "W291": 70,
    "B905": 41,
    "UP007": 38,
    "B007": 28,
    "B028": 14,
    "E712": 13,
    "E731": 7,
    "B904": 6,
    "E741": 4,
    "UP015": 4,
    "C408": 3,
    "B017": 2,
    "C420": 2,
    "UP033": 2,
    "C401": 1,
    "C416": 1,
    "E402": 1,
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
