"""A flag a document tells a reader to pass has to exist.

`test_documented_paths_exist.py` next door checks that the *program* is there.
This checks the rest of the command line, which is the half that drifts without
moving a file: a renamed option, a deprecated one, or -- the case that turned up
most here -- an option the program never had.

It reads argparse out of the target's AST rather than running it with `--help`.
That is not a performance choice. A `--help` scan of this repository was written
first and reported nine broken files; it shelled out without
`PYTHONIOENCODING=utf-8`, the child printed a degree sign, cp950 could not
encode it, the child died, and empty help text read as "this program has no
flags". Parsing the source cannot be fooled that way, and it costs milliseconds
instead of minutes. **A guard that runs the thing has to be able to read the
thing.**

What it found on its first run, in five reader-facing documents:

- `ch6_dead_reckoning.example_imu_strapdown` and `example_zupt` were being
  passed `--data <dataset>` in about twenty places. Neither declares any
  arguments, and neither reads `data/sim` at all -- both build their own
  trajectory. The flag was not a stale spelling, it was **a capability that
  never existed**, and every "run it on these three variants" experiment built
  on it was unrunnable as written.
- `ch5_fingerprinting.example_deterministic` likewise: it loads the shipped grid
  database and declares no flags.
- `ch8_sensor_fusion.compare_lc_tc --alpha 0.01` -- the real flag is
  `--confidence`, and `alpha` was deprecated in favour of it, so 0.01 became
  0.99.

Stripping the flag was not enough for any of them. The blocks exist to *compare*
variants, and three identical commands compare nothing, so each was rewritten to
point at the mechanism the reader actually has.

Author: Li-Ta Hsu
"""

import ast
import re

import pytest

from tests.docs.test_documented_paths_exist import _documents, _name, REPO_ROOT

#: `python -m a.b --x --y=1` and `python dir/script.py --x 2`.
#:
#: The flag run is captured lazily so that prose after the command is not
#: swallowed; anything that is not a leading `-` ends it.
INVOCATION = re.compile(
    r"python\s+(?:-m\s+(?P<module>[A-Za-z0-9_][A-Za-z0-9_.]*)"
    r"|(?P<script>[A-Za-z0-9_][A-Za-z0-9_/]*\.py))"
    r"(?P<args>(?:\s+-{1,2}[A-Za-z0-9][A-Za-z0-9-]*(?:[= ][^\s\\]+)?)*)"
)

FLAG = re.compile(r"(?<!\S)(--[A-Za-z0-9][A-Za-z0-9-]*)")

#: Flags every argparse program accepts without declaring them.
IMPLICIT = {"--help"}


def _target_path(match):
    """The file a documented invocation runs, or None if it is not in the repo."""
    module = match.group("module")
    if module:
        return REPO_ROOT / (module.replace(".", "/") + ".py")
    return REPO_ROOT / match.group("script")


def _declared_flags(path):
    """(has_argparse, flags) read from the source.

    `has_argparse` matters on its own: a program with no parser at all cannot
    take any flag, which is a stronger and more useful statement than "this
    particular flag is missing".
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags, parser_seen = set(), False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "ArgumentParser":
            parser_seen = True
        if node.func.attr != "add_argument":
            continue
        parser_seen = True
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return parser_seen, flags


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_documented_flags_are_declared(document):
    """Every flag passed to a repo program must be one that program declares."""
    text = document.read_text(encoding="utf-8")
    # Join shell line continuations so a wrapped command reads as one.
    joined = text.replace("\\\n", " ")

    problems = []
    for match in INVOCATION.finditer(joined):
        used = {f for f in FLAG.findall(match.group("args") or "")} - IMPLICIT
        if not used:
            continue
        path = _target_path(match)
        if not path.is_file():
            continue  # the path guard next door owns missing programs
        try:
            parser_seen, declared = _declared_flags(path)
        except SyntaxError:  # pragma: no cover - a broken program is not our news
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if not parser_seen:
            problems.append(
                f"{relative} declares no arguments at all, but is passed "
                f"{', '.join(sorted(used))}"
            )
            continue
        unknown = sorted(used - declared)
        if unknown:
            problems.append(
                f"{relative} does not declare {', '.join(unknown)} "
                f"(it has {', '.join(sorted(declared)) or 'none'})"
            )

    assert not problems, (
        f"{_name(document)} passes flags that do not exist:\n  "
        + "\n  ".join(problems)
        + "\n\nIf the program declares no arguments, the flag is not a stale "
        "spelling but a capability it never had -- check whether the surrounding "
        "example is runnable at all before editing the flag away, because a "
        "comparison of three variants that becomes three identical commands has "
        "lost the thing it was for."
    )
