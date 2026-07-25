"""
Guard test: every example's console output must survive a legacy Windows console.

Readers run the `ch*/example_*.py` scripts exactly as printed in the book, often
in a default Windows console whose stdout encoding is cp1252 rather than UTF-8.
A single unencodable character in a `print()` aborts the run with
UnicodeEncodeError -- and if it fires before the figure is saved, the reader
never sees the figure at all.

This test parses each example and checks the string literals reaching `print()`.
The bar is "encodable as cp1252", not "pure ASCII": characters such as the
degree sign (U+00B0, used across ch2/ch3/ch4/ch6) live in cp1252 and print
fine, whereas Greek letters, arrows, and combining accents do not.

Math symbols in *matplotlib* labels are deliberately not checked -- figure text
is rendered by matplotlib, never written to the console.

Author: Li-Ta Hsu
Run with: python -m pytest tests/test_example_console_encoding.py -v
"""

import ast
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

# Encoding of a default (non-UTF-8) Windows console, and the worst case a
# reader is realistically dropped into.
LEGACY_CONSOLE_ENCODING = "cp1252"

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_example_scripts() -> List[Path]:
    """Collect every chapter example script in the repo.

    Returns:
        Sorted list of paths to `ch*/example_*.py`.
    """
    return sorted(REPO_ROOT.glob("ch*/example_*.py"))


def iter_print_string_literals(source: str) -> Iterator[Tuple[int, str]]:
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


def unencodable_chars(text: str) -> List[str]:
    """Return the characters of `text` that a legacy console cannot encode.

    Args:
        text: String destined for stdout.

    Returns:
        Sorted, de-duplicated list of offending characters.
    """
    offenders = set()
    for char in text:
        try:
            char.encode(LEGACY_CONSOLE_ENCODING)
        except UnicodeEncodeError:
            offenders.add(char)
    return sorted(offenders)


EXAMPLE_SCRIPTS = find_example_scripts()


def test_example_scripts_were_found():
    """Fail loudly if the glob stops matching -- an empty sweep proves nothing."""
    assert EXAMPLE_SCRIPTS, f"no ch*/example_*.py found under {REPO_ROOT}"


@pytest.mark.parametrize(
    "script", EXAMPLE_SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_print_output_encodable_on_legacy_console(script: Path):
    """No `print()` literal may contain a cp1252-unencodable character."""
    source = script.read_text(encoding="utf-8")

    failures = []
    for lineno, literal in iter_print_string_literals(source):
        offenders = unencodable_chars(literal)
        if offenders:
            detail = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in offenders)
            failures.append(f"  {script.name}:{lineno}: {detail}")

    assert not failures, (
        f"print() output would raise UnicodeEncodeError on a "
        f"{LEGACY_CONSOLE_ENCODING} console:\n"
        + "\n".join(failures)
        + "\n\nUse an ASCII equivalent (lambda, x_hat, R^2, ->, x). "
        "Math symbols are fine in matplotlib labels, just not in print()."
    )
