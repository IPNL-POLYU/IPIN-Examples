"""A command a document tells a reader to run has to work on a fresh clone.

Two spellings of "run this example" exist, and only one of them does:

    python ch3_estimators/example_least_squares.py    # script form
    python -m ch3_estimators.example_least_squares    # module form

The script form puts the *script's* directory on ``sys.path[0]``. The module
form puts the working directory there. Since every example does ``from core...
import``, and ``core`` lives at the repository root, the script form only works
once ``pip install -e .`` has been run, while the module form works straight
from a clone. Measured, not assumed -- with the editable install's finder
removed, the script form raises

    ModuleNotFoundError: No module named 'core'

and the module form runs to completion.

The repository had drifted into using both. The seven chapter READMEs used the
module form 71 times against 11 script-form uses, but the two documents a
newcomer actually lands on -- the top-level README's Quick Start and
``notebooks/README.md`` -- used the script form throughout, and the top-level
Quick Start sat *above* the Setup section that installs the package. So the
three most prominent commands in the repository were the only ones that could
not work in the state the reader was in when they read them.

That is the class of bug nothing else here catches: `test_documented_paths_exist`
confirms the file is there, and `test_documented_flags_exist` confirms the flag
is declared, but both would pass a command that cannot import its own library.

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: `python ch4_rf_point_positioning/example_toa_positioning.py`, with or without
#: trailing flags. Anchored to a chapter directory so that prose about a path
#: (`see ch4_rf_point_positioning/example_toa_positioning.py`) is untouched --
#: this is a claim about how to *invoke* something, not about where it lives.
SCRIPT_FORM = re.compile(
    r"python\s+(ch\d+_[a-z0-9_]+)/([A-Za-z0-9_]+)\.py"
)

#: Directories whose documents record a past state rather than instruct a reader.
#: `.claude` holds other sessions' worktrees, which are separate checkouts.
EXEMPT_DIRS = (".git", ".claude", ".dev", "node_modules")

#: Agent working notes, not reader instructions. CLAUDE.md documents the
#: editable-install `sys.path` trap and quotes the script form *as the failing
#: case*, which is the one place the spelling is deliberate.
EXEMPT_FILES = {"CLAUDE.md"}


def _documents():
    """Every reader-facing markdown and notebook file, in a stable order."""
    found = []
    for pattern in ("*.md", "*.ipynb"):
        for path in sorted(REPO_ROOT.rglob(pattern)):
            relative = path.relative_to(REPO_ROOT)
            if any(part in EXEMPT_DIRS for part in relative.parts):
                continue
            if str(relative).replace("\\", "/") in EXEMPT_FILES:
                continue
            found.append(path)
    return found


def _name(path):
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_documented_commands_use_module_form(document):
    """Every documented example invocation must be `python -m`."""
    text = document.read_text(encoding="utf-8")
    offenders = {}
    for match in SCRIPT_FORM.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        offenders.setdefault(match.group(0), []).append(line)

    assert not offenders, (
        f"{_name(document)} tells the reader to run {len(offenders)} example(s) "
        "in the script form, which raises ModuleNotFoundError on a clone that "
        "has not been pip-installed:\n  "
        + "\n  ".join(
            f"{cmd}  (line {', '.join(map(str, lines))})"
            for cmd, lines in offenders.items()
        )
        + "\n\nRewrite as the module form, which needs no install:\n  "
        + "\n  ".join(
            "python -m {}.{}".format(*SCRIPT_FORM.match(cmd).groups())
            for cmd in offenders
        )
    )
