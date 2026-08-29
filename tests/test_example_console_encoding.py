"""
Guard test: every example's console output must survive a legacy Windows console.

Readers run the `ch*/example_*.py` scripts exactly as printed in the book, often
in a default Windows console whose stdout encoding is a legacy code page rather
than UTF-8. A single unencodable character in a `print()` aborts the run with
UnicodeEncodeError -- and if it fires before the figure is saved, the reader
never sees the figure at all.

This test parses each example and checks the string literals reaching `print()`
or argparse's console-facing help against every code page a reader plausibly
has. Checking one code page is not enough, because they disagree about which
symbols they carry:

    - U+00B0 degree sign  -- present in all of them, safe to print
    - U+00B2 superscript two -- missing from cp932/cp950/cp936 (CJK Windows)
    - U+00D7 multiplication sign -- missing from cp437 (US OEM console)

So the bar is "encodable in all of CONSOLE_ENCODINGS", not "pure ASCII": the
degree sign printed across ch2/ch3/ch4/ch6 stays legal, while Greek letters,
arrows, combining accents, and the two symbols above do not.

Math symbols in *matplotlib* labels are deliberately not checked -- figure text
is rendered by matplotlib, never written to the console.

Author: Li-Ta Hsu
Run with: python -m pytest tests/test_example_console_encoding.py -v
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

# Default stdout encodings a reader can realistically be dropped into: Western
# European Windows (cp1252), a US OEM console (cp437), Western OEM (cp850), and
# CJK Windows (cp932 Japanese, cp950 Traditional Chinese, cp936 Simplified,
# cp949 Korean).
CONSOLE_ENCODINGS = (
    "cp1252",
    "cp437",
    "cp850",
    "cp932",
    "cp950",
    "cp936",
    "cp949",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_example_scripts() -> list[Path]:
    """Collect every chapter example script in the repo.

    Returns:
        Sorted list of paths to `ch*/example_*.py`.
    """
    return sorted(REPO_ROOT.glob("ch*/example_*.py"))


def iter_print_string_literals(source: str) -> Iterator[tuple[int, str]]:
    """Yield string literals that are arguments to a `print()` call.

    Walks into f-strings, so the literal parts of an f-string are covered while
    the interpolated values (resolved at runtime) are not.

    Args:
        source: Python source text.

    Yields:
        Tuples of (line number, string literal).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        is_print = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        )
        if not is_print:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                yield child.lineno, child.value


def iter_argparse_string_literals(source: str) -> Iterator[tuple[int, str]]:
    """Yield literals that argparse can print to the console.

    Module docstrings deserve special handling because the repository's normal
    parser construction passes ``description=__doc__``. The old guard inspected
    only explicit ``print()`` calls, so an unencodable arrow or combining accent
    in that docstring made ``--help`` crash even though the guard stayed green.

    Args:
        source: Python source text.

    Yields:
        Tuples of (line number, console-facing string literal).
    """
    tree = ast.parse(source)
    module_docstring = ast.get_docstring(tree, clean=False)
    module_docstring_line = tree.body[0].lineno if module_docstring and tree.body else 1

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        if function_name == "ArgumentParser":
            relevant_keywords = {"description", "epilog"}
        elif function_name == "add_argument":
            relevant_keywords = {"help"}
        else:
            continue

        for keyword in node.keywords:
            if keyword.arg not in relevant_keywords:
                continue
            if isinstance(keyword.value, ast.Name) and keyword.value.id == "__doc__":
                if module_docstring is not None:
                    yield module_docstring_line, module_docstring
                continue
            for child in ast.walk(keyword.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    yield child.lineno, child.value


def unencodable_chars(text: str) -> list[tuple[str, list[str]]]:
    """Find characters of `text` that some console encoding cannot represent.

    Args:
        text: String destined for stdout.

    Returns:
        Sorted list of (character, names of encodings that reject it) pairs,
        one entry per distinct offending character. Empty if `text` is safe
        everywhere.
    """
    offenders = {}
    for char in set(text):
        rejected_by = []
        for encoding in CONSOLE_ENCODINGS:
            try:
                char.encode(encoding)
            except UnicodeEncodeError:
                rejected_by.append(encoding)
        if rejected_by:
            offenders[char] = rejected_by
    return sorted(offenders.items())


EXAMPLE_SCRIPTS = find_example_scripts()


def test_example_scripts_were_found():
    """Fail loudly if the glob stops matching -- an empty sweep proves nothing."""
    assert EXAMPLE_SCRIPTS, f"no ch*/example_*.py found under {REPO_ROOT}"


@pytest.mark.parametrize(
    "script", EXAMPLE_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_print_output_encodable_on_legacy_console(script: Path):
    """No `print()` literal may hold a character some console cannot encode."""
    source = script.read_text(encoding="utf-8")

    failures = []
    for lineno, literal in iter_print_string_literals(source):
        for char, rejected_by in unencodable_chars(literal):
            failures.append(
                f"  {script.name}:{lineno}: {char!r} (U+{ord(char):04X}) "
                f"breaks {', '.join(rejected_by)}"
            )

    assert not failures, (
        "print() output would raise UnicodeEncodeError on a default "
        "(non-UTF-8) console:\n"
        + "\n".join(failures)
        + "\n\nUse an ASCII equivalent (lambda, x_hat, R^2, ->, x). "
        "Math symbols are fine in matplotlib labels, just not in print()."
    )


@pytest.mark.parametrize(
    "script", EXAMPLE_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_argparse_help_encodable_on_legacy_console(script: Path):
    """Module descriptions and argument help must survive legacy consoles."""
    source = script.read_text(encoding="utf-8")

    failures = []
    for lineno, literal in iter_argparse_string_literals(source):
        for char, rejected_by in unencodable_chars(literal):
            failures.append(
                f"  {script.name}:{lineno}: {char!r} (U+{ord(char):04X}) "
                f"breaks {', '.join(rejected_by)}"
            )

    assert not failures, (
        "argparse help would raise UnicodeEncodeError on a default "
        "(non-UTF-8) console:\n"
        + "\n".join(failures)
        + "\n\nUse an ASCII equivalent in module descriptions, epilog text, and "
        "argument help. Math symbols remain fine in matplotlib labels."
    )
