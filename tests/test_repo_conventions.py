"""Repo conventions enforced as tests rather than as prose.

Written after a survey found the split: the convention that *was* in
`.cursor/rules` ("chapter figures belong in chX_*/figs/") is followed
everywhere, while the convention that lived only in reviewers' heads ("write
them through core.eval.save_figure") is violated in ten files. A rule nobody
can run is a rule that decays, so the runnable ones live here.

Each check ratchets: pre-existing violations are listed and skipped, new ones
fail. That keeps the debt visible and countable instead of letting a red suite
force an unrelated cleanup into whatever change happens to notice it.

Author: Li-Ta Hsu
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Chapter example scripts: the files that produce the book's figures.
CHAPTER_GLOB = "ch*_*/*.py"

# Files known to write figures with a raw savefig, predating this test.
#
# THIS LIST MUST ONLY SHRINK. Adding an entry means shipping a figure that is
# PNG-only (the book needs SVG and PDF) and that misses save_figure's
# reproducible writer, which pins the SVG hash salt and drops creation
# timestamps so a committed figure diff means the picture actually changed.
#
# The size of this list is the point. A hand survey of five likely files had
# found three violations; running the check over every chapter found ten, all
# of Chapters 7 and 8. That gap is the argument for a runnable rule.
#
# Deliberately not asserting that listed files still violate: a stale-entry
# check would turn one branch's fix into another's red build.
#
# Empty, and it took two passes to get here: ten at first count, seven after
# the SLAM front-end fix converted three in passing, then zero. Every chapter
# example now writes through save_figure. Keep it empty -- if a new entry is
# genuinely warranted, record why next to it.
KNOWN_RAW_SAVEFIG: set = set()

# Files known to draw from the unseeded global RNG, predating this test.
# Same ratchet: only shrink it. Each entry was an example whose committed
# figures could not be regenerated, so a figure diff there meant nothing.
#
# Empty. All five now own a np.random.default_rng(DEFAULT_SEED); the two that
# draw in more than one place thread a single generator through the run rather
# than seeding twice, so the streams stay independent instead of being two
# copies of the same sequence.
KNOWN_UNSEEDED_RNG: set = set()


def _chapter_scripts():
    """Every chapter-level Python file, as repo-relative paths."""
    return sorted(
        path for path in REPO_ROOT.glob(CHAPTER_GLOB) if not path.name.startswith("_")
    )


def _relative(path: Path) -> str:
    """Repo-relative POSIX path, for stable comparison against the allowlist."""
    return path.relative_to(REPO_ROOT).as_posix()


def _savefig_lines(source: str):
    """Line numbers of ``.savefig(`` calls, ignoring comments and strings.

    Uses the AST rather than a text scan so that a mention of savefig inside a
    docstring -- for instance one explaining why not to use it -- is not
    reported as a violation.
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "savefig"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_chapter_figures_go_through_save_figure(script):
    """No new raw savefig in chapter code.

    core.eval.save_figure is the single output path: it writes svg/pdf/png
    together and makes the bytes reproducible. Bypassing it silently costs the
    book two of its three formats.
    """
    relative = _relative(script)
    hits = _savefig_lines(script.read_text(encoding="utf-8"))

    if relative in KNOWN_RAW_SAVEFIG:
        pytest.skip(f"known pre-existing raw savefig ({relative})")

    assert not hits, (
        f"{relative} calls savefig directly at line(s) "
        f"{', '.join(str(n) for n in hits)}. Use core.eval.save_figure so the "
        f"figure is written in every format and reproducibly. If this is "
        f"genuinely not a book figure, say so in a comment and add the file "
        f"to KNOWN_RAW_SAVEFIG with the reason."
    )


def test_no_figures_written_to_the_repo_root():
    """Figures belong in chX_*/figs/, per .cursor/rules/020.

    The prose rule has held so far; this pins it so it keeps holding.
    """
    stray = [
        path.name
        for path in REPO_ROOT.glob("*")
        if path.suffix.lower() in {".png", ".svg", ".pdf", ".gif"}
    ]

    assert not stray, (
        f"figure files at the repo root: {stray}. Chapter figures belong in "
        f"the owning chapter's figs/ directory."
    )


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_examples_do_not_draw_from_the_unseeded_global_rng(script):
    """Figures must be regenerable, which means the randomness is seeded.

    Chapter 6's comparison drew biases from the unseeded global RNG, so every
    run sent the unaided IMU off in a different direction and the committed
    figures could not be reproduced -- which also made every figure diff
    meaningless noise. Either seed the global RNG explicitly or, better, own a
    np.random.default_rng(seed).
    """
    relative = _relative(script)
    source = script.read_text(encoding="utf-8")

    legacy_calls = re.findall(r"np\.random\.(\w+)", source)
    draws = [name for name in legacy_calls if name not in {"seed", "default_rng"}]

    if not draws:
        return

    if relative in KNOWN_UNSEEDED_RNG:
        pytest.skip(f"known pre-existing unseeded RNG ({relative})")

    seeded = "np.random.seed(" in source or "default_rng(" in source
    assert seeded, (
        f"{relative} draws from the global RNG "
        f"({', '.join(sorted(set(draws)))}) without ever seeding it, so its "
        f"figures cannot be regenerated. Prefer threading an explicit "
        f"rng = np.random.default_rng(seed)."
    )


# Distribution name in pyproject.toml -> module name to import, where the two
# differ. Only the exceptions need listing.
IMPORT_NAMES = {"scikit-learn": "sklearn"}


def _declared_dependencies():
    """Runtime dependencies declared in pyproject.toml, as distribution names."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no [project] dependencies block"

    # Strip the version specifier and any extras: "scikit-learn>=1.0.0" -> name.
    return [
        re.split(r"[<>=!~\[]", line.strip().strip('",'))[0].strip()
        for line in block.group(1).splitlines()
        if line.strip().startswith('"')
    ]


@pytest.mark.parametrize("dist", _declared_dependencies())
def test_declared_dependency_is_importable(dist):
    """Every declared dependency must actually be installed.

    This exists because a missing dependency does not fail loudly here -- it
    fails *silently*. ``core/fingerprinting/classification.py`` guards its
    scikit-learn import and sets SKLEARN_AVAILABLE, and the matching test
    module skips itself when that is False. scikit-learn was installed in the
    development environment and declared nowhere, so:

      - locally, 21 tests passed and nobody could tell
      - on a clean install, the same 21 skipped and the suite still went green
      - and a reader running ch5's example_classification got an ImportError,
        because that guard re-raises rather than degrading

    The first CI run made it visible: 1692 passed / 30 skipped on the runner
    against 1713 / 9 locally. A test that quietly stops running is worse than
    one that fails, so this asserts the environment the suite claims to need
    is the environment it got.
    """
    module = IMPORT_NAMES.get(dist, dist)

    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(
            f"pyproject.toml declares {dist!r} but `import {module}` fails "
            f"({exc}). Any test guarded by an optional-import flag will skip "
            f"rather than fail, so the suite would stay green while running "
            f"less than it appears to."
        )
