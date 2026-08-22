"""`--help` must answer, not run the demonstration.

Typing `--help` is the first thing anyone does with an unfamiliar program.
Seventeen of this repository's thirty-eight examples used to respond by running
the whole demonstration and writing their figures -- measured, not assumed:

    python -m ch3_estimators.example_least_squares --help
    -> the full Chapter 3 least-squares output, then a saved figure

They had no `ArgumentParser` at all, so the flag was simply ignored, and
`ch6_dead_reckoning.example_allan_variance` was worse than that: it read
`'--debug' in sys.argv` by hand, which accepted its own flag while still
ignoring `--help`.

**The check is behavioural rather than structural, because the structural
version does not say the right thing.** "The module builds an ArgumentParser"
is satisfied by a parser constructed at the *end* of `main()`, after the work
and the figures. What a reader cares about is that the process prints usage and
stops, so that is what this runs.

Each example gets its own subprocess, but a `--help` run exits during argument
parsing, so the cost is import time -- around a second each.

`description=__doc__` with `RawDescriptionHelpFormatter` is the house spelling.
It costs three lines and makes `--help` genuinely useful: the module docstrings
here list what the example demonstrates and which book equations it implements,
which is exactly what someone typing `--help` wants to know.

Author: Li-Ta Hsu
"""

import os
import subprocess
import sys

import pytest

from tests.example_runner import WORKSPACE_ROOT

#: A `--help` run parses arguments and exits, so this is a deadlock guard
#: rather than a budget. An example that *runs* on --help blows through it,
#: which is the failure this exists to catch.
HELP_TIMEOUT_S = 90


def _examples():
    return sorted(
        p for p in WORKSPACE_ROOT.glob("ch*_*/example_*.py") if p.is_file()
    )


def _module(path):
    return f"{path.parent.name}.{path.stem}"


@pytest.mark.parametrize("example", _examples(), ids=_module)
def test_help_prints_usage_and_stops(example, tmp_path):
    """`python -m <example> --help` must print usage without doing the work."""
    module = _module(example)
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(WORKSPACE_ROOT),
        "PYTHONIOENCODING": "utf-8",
        # If the example ignores --help and runs anyway, its figures must not
        # land on the committed ones.
        "IPIN_FIGS_DIR": str(tmp_path),
    })

    try:
        run = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=HELP_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{module} --help did not finish in {HELP_TIMEOUT_S}s, so it is "
            "running the demonstration instead of answering."
        )

    assert run.returncode == 0, (
        f"{module} --help exited {run.returncode}.\n{run.stderr[-1500:]}"
    )
    assert "usage:" in run.stdout[:400], (
        f"{module} --help printed no usage line. Its first output was:\n\n"
        f"{run.stdout[:400]!r}\n\n"
        "Parse arguments before doing any work:\n\n"
        "    argparse.ArgumentParser(\n"
        "        description=__doc__,\n"
        "        formatter_class=argparse.RawDescriptionHelpFormatter,\n"
        "    ).parse_args()\n"
    )
    assert not list(tmp_path.rglob("*")), (
        f"{module} --help wrote files into {tmp_path}, so it did the work "
        "before answering. Parse arguments first."
    )
