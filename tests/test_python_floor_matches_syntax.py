"""The declared Python floor has to be one the source can actually run on.

`pyproject.toml` claimed ``requires-python = ">=3.8"`` and shipped a 3.8
classifier, and both were false. Five modules --

    core/fusion/adaptive.py     core/fusion/gating.py
    core/sensors/strapdown.py   core/sensors/types.py
    core/slam/factors.py

-- annotate with PEP 585 built-in generics (``tuple[bool, Optional[str]]``,
``list[np.ndarray]``) and none of them carries ``from __future__ import
annotations``. Function and dataclass annotations are *evaluated* at definition
time without that import, and ``tuple[...]`` is a 3.9 feature, so on 3.8 those
modules raise ``TypeError: 'type' object is not subscriptable`` the moment they
are imported. That is `core.fusion`, `core.slam` and `core.sensors` -- Chapters
6, 7 and 8 in their entirety.

Nothing caught it because nothing was looking. CI runs a single job on 3.11.
mypy's own config said ``python_version = "3.8"`` and would have flagged every
one of these, but CI never invokes mypy -- a checker that is configured and not
run is indistinguishable from one that passes.

**The floor is 3.10, not 3.9, and a hand survey said 3.9.** Grepping for
built-in generics found the five modules above and stopped there. This check,
reading the AST, immediately added ``core/coords/transforms.py``, which
annotates three parameters ``NDArray[np.float64] | None = None`` -- PEP 604,
where ``types.GenericAlias.__or__`` arrives in 3.10. That is `core.coords`, the
module Chapter 2 opens with, so the package could never have run on 3.9 either.
A grep sees the spellings you thought to look for; the parser sees the ones you
did not.

So this guard reads the floor the package *declares* and the syntax the source
*uses*, and requires the first to support the second. It fails in both
directions, which is the point: lowering `requires-python` back to 3.8 turns it
red, and so does adding a ``match`` statement to a package whose floor has
since been lowered.

Author: Li-Ta Hsu
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Names whose subscription is PEP 585, new in 3.9.
PEP585_NAMES = frozenset({"list", "dict", "tuple", "set", "frozenset", "type"})

#: Directories that are not this package's source.
EXEMPT_DIRS = (".git", ".claude", ".dev", "node_modules", "build", "dist")

VERSION_FEATURES = {
    (3, 9): "PEP 585 built-in generics in an evaluated annotation (list[int])",
    (3, 10): "PEP 604 unions (X | Y) or a match statement",
}


def _declared_floor():
    """The minimum Python version `pyproject.toml` promises to support."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[><=~^]*\s*(\d+)\.(\d+)"', text)
    assert match, "pyproject.toml has no parseable requires-python"
    return int(match.group(1)), int(match.group(2))


def _source_files():
    """Every .py file that ships as part of this repository."""
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if any(part in EXEMPT_DIRS for part in relative.parts):
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        yield path


def _annotation_nodes(tree):
    """Every AST node that sits inside an annotation."""
    inside = set()
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
            args = node.args
            for arg in (
                list(args.args)
                + list(args.posonlyargs)
                + list(args.kwonlyargs)
                + [args.vararg, args.kwarg]
            ):
                if arg is not None:
                    annotations.append(arg.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            if annotation is not None:
                inside.update(id(child) for child in ast.walk(annotation))
    return inside


def _required_version(path):
    """The lowest Python this file can run on, and why."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lazy = "from __future__ import annotations" in source
    in_annotation = _annotation_nodes(tree)

    needed = (3, 0)
    reasons = []
    for node in ast.walk(tree):
        annotated = id(node) in in_annotation
        # An annotation is a string under `from __future__ import annotations`,
        # so nothing in it is evaluated and the syntax costs nothing at runtime.
        if lazy and annotated:
            continue
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in PEP585_NAMES
        ):
            if (3, 9) > needed:
                needed = (3, 9)
            reasons.append(f"line {node.lineno}: {node.value.id}[...]")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and annotated:
            needed = max(needed, (3, 10))
            reasons.append(f"line {node.lineno}: X | Y annotation")
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            needed = max(needed, (3, 10))
            reasons.append(f"line {node.lineno}: match statement")
    return needed, reasons


def test_declared_python_floor_supports_the_syntax_in_use():
    """No source file may need a newer Python than pyproject.toml promises."""
    floor = _declared_floor()
    too_new = {}
    for path in _source_files():
        result = _required_version(path)
        if result is None:
            continue
        needed, reasons = result
        if needed > floor:
            name = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            too_new[name] = (needed, reasons[:3])

    assert not too_new, (
        "pyproject.toml declares requires-python >={}.{}, but {} file(s) use "
        "syntax that needs a newer interpreter and would raise at import "
        "time:\n  ".format(floor[0], floor[1], len(too_new))
        + "\n  ".join(
            "{} needs {}.{} -- {} ({})".format(
                name, need[0], need[1], VERSION_FEATURES.get(need, "?"), "; ".join(why)
            )
            for name, (need, why) in sorted(too_new.items())
        )
        + "\n\nEither raise requires-python (and the classifiers, black/ruff "
        "target-version and mypy python_version alongside it), or add "
        "`from __future__ import annotations` so the annotations are not "
        "evaluated."
    )


def test_classifiers_do_not_advertise_versions_below_the_floor():
    """A `Programming Language :: Python :: 3.x` classifier is a promise too."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    floor = _declared_floor()
    advertised = [
        (int(a), int(b))
        for a, b in re.findall(
            r'"Programming Language :: Python :: (\d+)\.(\d+)"', text
        )
    ]
    below = [v for v in advertised if v < floor]
    assert not below, (
        "pyproject.toml classifiers advertise {} but requires-python is "
        ">={}.{}. pip trusts requires-python and PyPI shows the classifiers, so "
        "the two disagreeing sends readers to an interpreter that cannot import "
        "the package.".format(
            ", ".join(f"{a}.{b}" for a, b in sorted(below)), floor[0], floor[1]
        )
    )
