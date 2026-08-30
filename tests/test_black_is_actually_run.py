"""black is configured, documented, and until now nothing ran it.

`pyproject.toml` carries a `[tool.black]` section and the README tells readers
to run the formatter. CI runs a single pytest job and invokes black nowhere, so
for a contributor the only difference between "the repository is formatted" and
"nobody checked" was whether they happened to run it by hand.

That is not hypothetical. The whole tree was brought to `black --check` clean in
an earlier sweep, and it has drifted back twice since: once fixed by #109, and
again by `tests/ch6_dead_reckoning/test_floor_detector_beats_base_rate.py`,
which arrived with #115 and is the file this test first went red on. Two
regressions through the same gap is the argument for a gate rather than for a
third manual fix.

**The scope is imported, not copied.** `SOURCE_DIRS` lives in
`test_lint_debt_only_shrinks.py` and is shared with it deliberately: a
second, hand-maintained copy is how `notebooks/` escaped the `zip(strict=)`
sweep for as long as it did -- a file outside the sweep's scope is a file
nothing sweeps, and that scope was a tuple somebody wrote once.

**Why a process pool, when the pyflakes check next door argues for the API.**
`test_no_pyflakes_warnings` uses the pyflakes API rather than a subprocess per
file because a spawn costs ~0.85 s and the analysis costs almost nothing --
four minutes against six seconds. That reasoning does not transfer to black,
and assuming it did would have produced a slow test. Measured over these 318
files on this machine:

    serial, in-process             57 s
    ThreadPoolExecutor(8)          64 s   -- slower than serial
    ProcessPoolExecutor            5.8 s
    `black --check` CLI, cold      6.9 s

black's cost is the reformatting itself, not the spawn, so the API alone buys
nothing; and black is compiled with mypyc, which does not release the GIL, so
threads only add contention. Processes are what black's own CLI uses, and they
land at the CLI's own number. Expect roughly `57 / cores` seconds on a runner
with fewer cores than this machine.

**A red here is fixed by running black, never by narrowing this test.** The one
case where that is not a contributor's fault: black's stable style is versioned
by calendar year and `pyproject.toml` floors the dependency at `>=22.0.0`
rather than pinning it, so a new black release can legitimately reformat code
nobody touched. The fix is still to run black.

Author: Li-Ta Hsu
"""

import functools
from concurrent.futures import ProcessPoolExecutor

import pytest

from tests.example_runner import WORKSPACE_ROOT
from tests.test_lint_debt_only_shrinks import SOURCE_DIRS

black = pytest.importorskip("black", reason="black is declared in the dev extra")
black_files = pytest.importorskip(
    "black.files", reason="black is declared in the dev extra"
)


@functools.lru_cache(maxsize=1)
def _mode():
    """The `[tool.black]` settings from pyproject.toml, as a black Mode.

    Read from the file rather than restated here, so this test cannot come to
    disagree with the configuration it exists to enforce.
    """
    config = black_files.parse_pyproject_toml(str(WORKSPACE_ROOT / "pyproject.toml"))
    return black.Mode(
        target_versions={
            black.TargetVersion[name.upper()]
            for name in config.get("target_version", [])
        },
        line_length=config.get("line_length", black.DEFAULT_LINE_LENGTH),
    )


def _would_reformat(path_str):
    """True if black would rewrite this file. Module level so it can be pickled.

    `write_back=WriteBack.NO` is what `--check` passes: black reads the file,
    formats in memory and reports, without touching the tree.

    `fast=True` skips the AST-equivalence assertion, which black only runs on
    files it would change anyway. Here that path ends in a failure report
    regardless, and leaving it on would turn an offending file into a raised
    exception instead of a named filename.
    """
    from pathlib import Path

    return black.format_file_in_place(
        Path(path_str),
        fast=True,
        mode=_mode(),
        write_back=black.WriteBack.NO,
    )


def _python_files():
    """Every `.py` file black would pick up from SOURCE_DIRS.

    Not `black .` from the root: that walks `.claude/worktrees/`, where the
    parallel sessions this repo runs keep their own checkouts, and reports
    around 971 files that are not this branch's problem.
    """
    return [
        str(path)
        for directory in SOURCE_DIRS
        for path in sorted((WORKSPACE_ROOT / directory).rglob("*.py"))
    ]


def test_every_source_file_is_black_formatted():
    """`black --check` over SOURCE_DIRS, in process and in parallel."""
    files = _python_files()
    assert files, "SOURCE_DIRS matched no Python files, which cannot be right."

    with ProcessPoolExecutor() as pool:
        results = pool.map(_would_reformat, files, chunksize=8)
        offenders = [
            path
            for path, would_change in zip(files, results, strict=True)
            if would_change
        ]

    assert not offenders, (
        f"{len(offenders)} of {len(files)} files are not black-formatted:\n\n"
        + "".join(f"  {path}\n" for path in offenders)
        + "\nRun `python -m black "
        + " ".join(SOURCE_DIRS)
        + "`.\n"
        "The diff is deliberately not printed here -- black prints a better "
        "one, and a formatting diff in a pytest failure report is unreadable."
    )


def test_the_installed_black_is_the_pinned_one():
    """The guard is an assertion about a SPECIFIC black version.

    This exact failure shipped once already: the dev machine ran black
    25.12.0 while CI freshly installed 26.5.1, and the two reported
    different results over the same tree -- red on one side, green on the
    other, and the disagreement was invisible until the merge. The pin in
    pyproject.toml is the single source of truth; this test reads it rather
    than repeating it, and fails with the upgrade command instead of letting
    a version skew masquerade as a formatting or typing regression.
    """
    from pathlib import Path

    import tomllib

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    pin = next(d.split("==")[1] for d in dev if d.startswith("black=="))
    installed = __import__("black").__version__
    assert installed == pin, (
        f"black {installed} is installed but the guards are calibrated "
        f"against {pin}. Run: pip install -e .[dev]"
    )
