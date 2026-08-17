"""The tools in tools/ that validate the repo must run, and must pass.

Both of these existed and neither ran anywhere. `check_equation_index.py`
validates the book-equation mapping that is this repository's stated premise --
180 indexed equations, every file path and object reference resolved, every
implemented equation backed by a conformance test. It was passing, and nothing
would have noticed the day it stopped.

`validate_dataset_docs.py` was worse: it reported 72 errors across 12 of the 20
datasets, so its answer was "red" and had presumably been "red" for long enough
that nobody looked. Most of those 72 were the tool's own fault -- it demanded
.npz files of datasets that ship .txt, demanded one specific set of table column
headers, and could not see the three ch5 datasets at all. Those are fixed, and
what remained is recorded in its KNOWN_INCOMPLETE register.

Run as subprocesses rather than imported and called. Both are user-facing tools
that print a report a person is meant to read, and on failure that report is
exactly what a contributor needs; capturing it whole and attaching it to the
assertion beats re-deriving a summary here. It also keeps the tools the single
source of truth for their own rules.

The workflow file makes the same argument for running these at all: "Their
entire value is that they run, and until this file there was nothing making them
run."

Author: Li-Ta Hsu
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (script, args). check_equation_index needs --strict, and that is the whole
#: reason this file was worth writing carefully.
#:
#: Without it the tool detects a broken index, prints "[ERROR] Object reference
#: errors" and "Object ref errors: 1", and then returns 0. Wiring the bare
#: command into CI would have produced a green tick that meant nothing --
#: confirmed by pointing one indexed object at a name that does not exist:
#: the default run exits 0, --strict exits 1.
VALIDATORS = [
    ("tools/check_equation_index.py", ["--strict"]),
    ("tools/validate_dataset_docs.py", []),
]


@pytest.mark.parametrize(
    "script,args", VALIDATORS, ids=[Path(s).stem for s, _ in VALIDATORS]
)
def test_validator_passes(script, args):
    """The validator exits 0 against the repository as committed."""
    proc = subprocess.run(
        [sys.executable, script, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )

    assert proc.returncode == 0, (
        f"{' '.join([script, *args])} exited {proc.returncode}.\n\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
