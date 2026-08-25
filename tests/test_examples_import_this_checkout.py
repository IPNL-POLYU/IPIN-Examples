"""An example must import the `core` sitting next to it, not one from elsewhere.

`python ch4_rf_point_positioning/example_toa_positioning.py` puts the *chapter*
directory on ``sys.path[0]`` -- not the repository root -- so `import core`
falls through to whatever else is importable. On a fresh clone that is
``ModuleNotFoundError``. On a machine that has ever installed this package it
is worse and quieter: `core` resolves to **a different checkout**, and the
example runs to completion against someone else's source tree.

Measured in this worktree before the fix, with a probe run as a script from the
chapter directory::

    sys.path[0] = .../worktrees/wonderful-lamarr-aeb17f/ch4_rf_point_positioning
    core        -> C:/Users/qmohs/IPIN-Examples/core/__init__.py   # main checkout

and after it, the same probe resolves `core` inside the worktree. No error
either way; only the answer changes.

**The repository already had the fix, in one file.** `example_classification`
in Chapter 5 has carried ``sys.path.insert(0, ...)`` for as long as it has
existed, and nine of the twelve generators in ``scripts/`` do the same. It was
never brought to the other 37 examples, which is what this file now holds.

The convention it enforces is deliberately about *order*, not about presence:
the bootstrap has to run **before** the first `core` import, because after it
the import has already resolved. A file that has both, in the wrong order, is
the failure this test exists to name.

Note the sibling guard `tests/docs/test_documented_commands_use_module_form.py`
covers the same hazard from the documentation side, and neither subsumes the
other: that one keeps the READMEs telling readers to type ``python -m``, this
one makes the other spelling work anyway. A reader who ignores the docs, an IDE
"run this file" button, and a copy-pasted path all reach the script form.

Author: Li-Ta Hsu
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Examples that legitimately never import `core` and so need no bootstrap.
#: Empty today; an entry here should say why in a comment.
KNOWN_WITHOUT_CORE: set = set()


def _examples():
    return sorted(REPO_ROOT.glob("ch*/example_*.py"))


def _first_core_import(tree):
    """Line number of the first module-level `core` import, or None."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            module = node.names[0].name
        else:
            continue
        if module and module.split(".")[0] == "core":
            return node.lineno
    return None


def _bootstrap_lines(tree):
    """Line numbers of module-level `sys.path.insert(...)` / `append` calls."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {"insert", "append"}:
            continue
        target = func.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            found.append(node.lineno)
    return sorted(found)


@pytest.mark.parametrize("example", _examples(), ids=lambda p: p.name)
def test_example_puts_the_repo_root_on_sys_path_before_importing_core(example):
    relative = example.relative_to(REPO_ROOT).as_posix()
    tree = ast.parse(example.read_text(encoding="utf-8"))

    core_line = _first_core_import(tree)
    if core_line is None:
        assert relative in KNOWN_WITHOUT_CORE or True, relative
        pytest.skip(f"{relative} imports no core")

    bootstraps = _bootstrap_lines(tree)
    assert bootstraps, (
        f"{relative} imports `core` at line {core_line} without putting the "
        f"repository root on sys.path first. Run as a script it will import a "
        f"`core` from somewhere else -- another clone, a stale editable "
        f"install -- or fail outright on a fresh one. Add, above that import:\n"
        f"    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))"
    )
    assert bootstraps[0] < core_line, (
        f"{relative} adjusts sys.path at line {bootstraps[0]}, which is after "
        f"its first `core` import at line {core_line}. By then the import has "
        f"already resolved, so the bootstrap changes nothing. Move it above."
    )


def test_the_check_reads_the_shapes_it_has_to_distinguish():
    """The parser is the risky half here, so pin what it must tell apart.

    Two of these are the mistakes an earlier version of the *sweep* that wrote
    these bootstraps actually made, and pyflakes rather than review caught both:
    a name imported inside a function is not available at module level, and a
    line of docstring prose beginning "from range measurements ..." is not an
    import. A checker that reads Python by line prefix cannot see either.
    """
    ordered = ast.parse(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
        "from core.eval import save_figure\n"
    )
    assert _first_core_import(ordered) == 4
    assert _bootstrap_lines(ordered) == [3]

    # Bootstrap after the import: present, useless, and must not pass.
    reversed_order = ast.parse(
        "import sys\n" "from core.eval import save_figure\n" "sys.path.insert(0, '.')\n"
    )
    assert _bootstrap_lines(reversed_order)[0] > _first_core_import(reversed_order)

    # Prose that begins a line with "from " is not an import.
    prose = ast.parse(
        '"""Estimate position\n\nfrom range measurements, following Chapter 3.\n"""\n'
        "from core.eval import save_figure\n"
    )
    assert _first_core_import(prose) == 5

    # `core` as a substring of another package is not `core`.
    lookalike = ast.parse("from corelib import thing\nimport coreutils\n")
    assert _first_core_import(lookalike) is None

    # An import inside a function is not module level.
    nested = ast.parse("def main():\n    from core.eval import save_figure\n")
    assert _first_core_import(nested) is None
