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
import io
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

# Files known to build their own matplotlib animation, predating this test.
#
# Same ratchet: only shrink it. The savefig check above guards static figures
# but says nothing about animations, and two Chapter 7 examples used that gap
# for as long as it existed -- they hand-rolled FuncAnimation + PillowWriter
# and passed `anim.save()` a literal `ch7_slam/figs/*.gif`. A path that never
# reaches core.eval cannot be redirected, so IPIN_FIGS_DIR did not apply to
# them: running either with --animate wrote into the tracked figs/ directory,
# which is precisely the failure the variable was introduced to stop. The
# clean-tree check in CI did not catch it because animations sit behind
# --animate and no test passes it.
#
# Empty. Both now call core.eval.save_animation, which resolves the directory,
# warns on oversized GIFs, and builds the identical FuncAnimation.
KNOWN_RAW_ANIMATION: set = set()

# Files known to draw from the unseeded global RNG, predating this test.
# Same ratchet: only shrink it. Each entry was an example whose committed
# figures could not be regenerated, so a figure diff there meant nothing.
#
# Empty. All five now own a np.random.default_rng(DEFAULT_SEED); the two that
# draw in more than one place thread a single generator through the run rather
# than seeding twice, so the streams stay independent instead of being two
# copies of the same sequence.
KNOWN_UNSEEDED_RNG: set = set()

# Files outside tests/ that define test_-prefixed functions, predating this
# test. Same ratchet: only shrink it.
#
# Empty. core/estimators/ held 18 of them -- three in kalman_filter, three in
# extended_kalman_filter, two each in unscented, iterated and particle_filter,
# six in factor_graph -- printing "UNIT TESTS" banners under
# if __name__ == "__main__". Because testpaths = ["tests"], pytest never
# collected any of them, so 18 functions named like tests contributed nothing to
# the suite and nothing would have noticed them breaking. They are now check_*,
# which is what they are: self-checks a reader runs by hand. The real coverage
# was never missing -- tests/core/estimators/ already holds an
# equation-anchored test file for every one of those six modules.
KNOWN_UNCOLLECTED_TESTS: set = set()

# Generators whose preset branch assigns output_dir a bare literal, predating
# this check. Same ratchet: only shrink it.
#
# Empty. Seven of them did: ch2, ch3, ch4, ch6 env/pdr/wheel_odom and ch7 each
# wrote `output_dir = "data/sim/..."` inside their preset chain, unconditionally,
# so `--preset X --output somewhere` silently regenerated the *shipped* dataset
# and left `somewhere` empty. ch5's copy was found the hard way in an earlier
# session, by overwriting three shipped datasets while surveying them; this
# check is what stops the other six being found the same way.
KNOWN_PRESET_OVERRIDES_OUTPUT: set = set()

# Library functions in core/ that draw from numpy's global RNG directly,
# predating this check. Same ratchet: only shrink it.
#
# Empty. Six did -- simulate_rss_measurement and simulate_rtt_measurement in
# core/rf, generate_scan_with_occlusion and generate_dense_wall_scan in
# core/slam, and ParticleFilter's __init__ and _resample -- for eleven draws in
# total. Nothing was broken: every example reaching them calls np.random.seed,
# which is why the committed figures reproduce. The hazard is that the fix this
# file *recommends* one check below ("prefer threading an explicit
# rng = np.random.default_rng(seed)") would have broken them, because a local
# Generator does not cover a library's global draws. Following good advice is a
# poor way to lose reproducibility.
#
# The remaining np.random calls under core/ are all inside check_* self-check
# demos, which seed themselves and are not library surface.
KNOWN_GLOBAL_RNG_IN_CORE: set = set()


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


def _raw_animation_lines(source: str):
    """Line numbers where chapter code builds its own matplotlib animation.

    Two signals, either of which means the write bypasses save_animation:
    importing anything from ``matplotlib.animation``, and calling ``.save()``
    with a ``writer=`` argument (which catches the case where the module is
    imported wholesale rather than by name).

    AST rather than a text scan, for the same reason the savefig check uses
    one: the two Chapter 7 files carry a comment naming ``FuncAnimation`` and
    ``PillowWriter`` to explain why they no longer use them, and a regex would
    report that comment as the violation it describes.
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "matplotlib.animation":
            hits.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(a.name.startswith("matplotlib.animation") for a in node.names):
                hits.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
            and any(kw.arg == "writer" for kw in node.keywords)
        ):
            hits.append(node.lineno)
    return sorted(set(hits))


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_chapter_animations_go_through_save_animation(script):
    """No new hand-rolled animation in chapter code.

    core.eval.save_animation is the single output path for GIFs, and the only
    one that resolves IPIN_FIGS_DIR. A hand-rolled FuncAnimation writing to a
    literal chX_*/figs path cannot be redirected, so a test run that reaches it
    rewrites a committed binary.
    """
    relative = _relative(script)
    hits = _raw_animation_lines(script.read_text(encoding="utf-8"))

    if relative in KNOWN_RAW_ANIMATION:
        pytest.skip(f"known pre-existing raw animation ({relative})")

    assert not hits, (
        f"{relative} builds its own matplotlib animation at line(s) "
        f"{', '.join(str(n) for n in hits)}. Use core.eval.save_animation so "
        f"the GIF honours IPIN_FIGS_DIR and is size-checked. If this is "
        f"genuinely not a book figure, say so in a comment and add the file "
        f"to KNOWN_RAW_ANIMATION with the reason."
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


# Chapter examples known to build a generator without a seed, predating this
# check. Same ratchet as the others: entries come out, never in.
#
# Empty. example_allan_variance was the last, and it is the reason this check
# exists at all.
KNOWN_UNSEEDED_GENERATOR: set = set()


def _bare_default_rng_lines(source: str):
    """Line numbers of ``default_rng()`` calls that pass no seed.

    AST rather than a regex, for the same reason the savefig check uses one: a
    comment or docstring explaining why not to write ``np.random.default_rng()``
    should not itself be reported as a violation. This file's own docstrings
    contain that string.
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if name == "default_rng" and not node.args and not node.keywords:
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_examples_seed_the_generators_they_create(script):
    """``np.random.default_rng()`` without a seed is not seeded randomness.

    The sibling check above accepts the mere *presence* of ``default_rng(`` as
    evidence that a file seeds its randomness. That is a hole, and Chapter 6's
    Allan variance example sat in it: it built a bare
    ``np.random.default_rng()``, so every run drew a different record. Bias
    instability came out 11.15 deg/hr on one run and 7.85 on the next -- a 42%
    swing in a number the figure reports, with the committed figure
    unreproducible and any diff against it meaningless.

    Reaching for the modern generator is the right instinct, which is exactly
    why it needs its own check: it looks more careful than np.random.seed()
    while giving the same guarantee as no seed at all.
    """
    relative = _relative(script)
    hits = _bare_default_rng_lines(script.read_text(encoding="utf-8"))

    if relative in KNOWN_UNSEEDED_GENERATOR:
        pytest.skip(f"known pre-existing unseeded generator ({relative})")

    assert not hits, (
        f"{relative} calls np.random.default_rng() with no seed at line(s) "
        f"{', '.join(str(n) for n in hits)}, so its figures cannot be "
        f"regenerated. Pass a seed -- a module-level DEFAULT_SEED threaded "
        f"through the generating function is the pattern used elsewhere in "
        f"Chapter 6."
    )


# Files whose pyflakes warnings are tolerated, predating this check.
#
# Same ratchet as the others: entries come out, never in. It starts empty and
# should stay that way, because it was emptied the hard way -- the sweep that
# made this check possible cleared 214 warnings across the 30 chapter examples
# (185 redundant f prefixes, 27 unused imports, 22 dead assignments), after
# earlier passes had done core/, scripts/ and tests/.
#
# An entry here is a claim that a warning is wrong. Pyflakes reports only
# things that are true statements about the code -- a name that is not defined,
# an import that is not used, a variable assigned and discarded -- so that
# claim is a strong one. Prefer restructuring the code.
KNOWN_PYFLAKES: set = set()

# Where Python lives in this repo. Chapter directories are globbed rather than
# listed so a new chapter is covered the day it is added.
PYFLAKES_AREAS = {
    "core": ["core/**/*.py"],
    "scripts": ["scripts/**/*.py"],
    "tests": ["tests/**/*.py"],
    "chapters": [CHAPTER_GLOB],
    # tools/ was outside this check until its two validators were wired into the
    # suite. It held six findings at that point -- an unused import in a script
    # named like a test that pytest never collects, an assigned-and-discarded
    # local, and four f-strings with no placeholders. All cleared; the area is
    # here so they cannot come back now that this code runs in CI.
    "tools": ["tools/**/*.py"],
}


def _area_files(area: str):
    """Every Python file in one area, as repo-relative paths."""
    found = set()
    for pattern in PYFLAKES_AREAS[area]:
        found.update(REPO_ROOT.glob(pattern))
    return sorted(path for path in found if path.is_file())


def _pyflakes_warnings(paths):
    """Run pyflakes over `paths`, returning its report lines.

    Uses the API rather than a subprocess per file: this covers just under 300
    files, and at ~0.85 s per interpreter spawn the subprocess form would take
    about four minutes against the API's six seconds.
    Syntax errors are reported on the error stream and count too -- a file that
    cannot be parsed is a worse failure than an unused import, not an exempt
    one.
    """
    from pyflakes.api import check
    from pyflakes.reporter import Reporter

    warnings_out, errors_out = io.StringIO(), io.StringIO()
    reporter = Reporter(warnings_out, errors_out)

    for path in paths:
        if _relative(path) in KNOWN_PYFLAKES:
            continue
        check(path.read_text(encoding="utf-8"), _relative(path), reporter)

    lines = warnings_out.getvalue().splitlines() + errors_out.getvalue().splitlines()
    return [line for line in lines if line.strip()]


@pytest.mark.parametrize("area", sorted(PYFLAKES_AREAS))
def test_no_pyflakes_warnings(area):
    """Pyflakes must stay silent, everywhere.

    This is the cheapest of the ratchets and the one with the widest reach: it
    is the check that would have caught the mistake made while writing the
    sweep it enforces. A two-line replace() with count=1 deleted a `d_ref`
    definition in a function where the name was still live, because the same
    two lines appeared in an earlier function too. `python -m compileall`
    accepted the file happily -- an undefined name is a runtime error, not a
    syntax one, and that branch of the example is not covered by a test.
    Pyflakes named it immediately.

    That is the general shape: pyflakes finds the class of defect that survives
    both the compiler and a test suite, because it is about names rather than
    behaviour. Unused imports and dead assignments are the tidy half; undefined
    names are the half that bites.
    """
    pytest.importorskip(
        "pyflakes",
        reason="pyflakes is declared in the dev extra; install with pip install -e '.[dev]'",
    )

    reported = _pyflakes_warnings(_area_files(area))

    assert not reported, (
        f"pyflakes reports {len(reported)} warning(s) under {area}/:\n  "
        + "\n  ".join(reported[:40])
        + ("\n  ..." if len(reported) > 40 else "")
        + "\n\nThe repo is pyflakes-clean; keep it that way. If a warning is "
        "genuinely wrong, add the file to KNOWN_PYFLAKES with the reason."
    )


def _source_files_outside_tests():
    """Every Python file in the library, chapters, scripts and tools."""
    found = []
    # Recursive everywhere on purpose. scripts/ and tools/ are flat today, so
    # a single-star glob would be complete -- and would silently stop being
    # complete the day someone adds a subpackage under either.
    patterns = ("core/**/*.py", "ch*_*/**/*.py", "scripts/**/*.py", "tools/**/*.py")
    for pattern in patterns:
        found.extend(REPO_ROOT.glob(pattern))
    return sorted(set(found))


@pytest.mark.parametrize("source", _source_files_outside_tests(), ids=_relative)
def test_no_test_functions_outside_the_tests_tree(source):
    """A test_-prefixed function outside tests/ is never collected.

    pyproject sets testpaths = ["tests"], so pytest does not look anywhere else.
    A function named test_something in core/ therefore looks like coverage from
    every angle except the one that matters: it does not run, and nothing
    reports it when it stops working.

    core/estimators/ carried 18 such functions. All 18 passed when checked by
    hand, which is exactly why the shape survives -- it fails silently by never
    speaking at all. Two of them, in particle_filter, asserted nothing whatever
    and printed "[PASS] Test passed" unconditionally beneath a comment reading
    "Check that filter ran successfully".

    Name a by-hand self-check check_* or demo_*. If it is a real test, move it
    under tests/ where it will be collected.
    """
    relative = _relative(source)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    offenders = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]

    if not offenders:
        return

    if relative in KNOWN_UNCOLLECTED_TESTS:
        pytest.skip(f"known pre-existing uncollected tests ({relative})")

    assert not offenders, (
        f"{relative} defines {len(offenders)} test_-prefixed function(s) that "
        f'pytest never collects, because testpaths = ["tests"]: '
        f"{', '.join(sorted(offenders))}. Rename to check_* or demo_* if it is a "
        f"by-hand self-check, or move it under tests/ if it is a real test."
    )


def _generators():
    """Every dataset generation script."""
    return sorted(REPO_ROOT.glob("scripts/generate_*.py"))


@pytest.mark.parametrize("script", _generators(), ids=_relative)
def test_preset_does_not_overwrite_an_explicit_output(script):
    """A preset may supply the default directory, never override the caller.

    `output_dir = "data/sim/whatever"` inside a preset branch discards whatever
    the caller passed. The failure is silent and destructive in the same breath:
    the command appears to write where you asked, and actually rewrites the
    shipped dataset. Write `output_dir = output_dir or "data/sim/whatever"` so
    the preset only fills a gap.

    The paired hazard is the inverse, and it is why this check is not enough on
    its own: if --output also declares an argparse default, output_dir is never
    empty and the preset's directory becomes unreachable. --output must default
    to None, with the module's own directory supplied as a fallback after the
    preset chain.
    """
    relative = _relative(script)
    tree = ast.parse(script.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "output_dir"
            for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.startswith("data/sim/"):
                offenders.append((node.lineno, value.value))

    if not offenders:
        return

    if relative in KNOWN_PRESET_OVERRIDES_OUTPUT:
        pytest.skip(f"known pre-existing preset override ({relative})")

    listed = ", ".join(f"line {n}: {v}" for n, v in offenders)
    assert not offenders, (
        f"{relative} assigns output_dir a bare literal ({listed}), which "
        f"discards an explicit --output and rewrites the shipped dataset "
        f'instead. Use `output_dir = output_dir or "..."` so the preset only '
        f"supplies a default, and make sure --output itself defaults to None."
    )


def _core_modules():
    """Every module in the shared library."""
    return sorted(REPO_ROOT.glob("core/**/*.py"))


@pytest.mark.parametrize("module", _core_modules(), ids=_relative)
def test_core_library_takes_its_randomness_from_the_caller(module):
    """A library function must let the caller own the stream it draws from.

    `np.random.normal(...)` inside core/ reads numpy's global stream, which the
    caller can only control with np.random.seed. That works, and every example
    here does it -- but it means an example that modernises to
    `rng = np.random.default_rng(seed)` silently stops covering the library, and
    its figures stop reproducing. The advice and the hazard point the same way,
    which is what makes this worth a check rather than a note.

    Take an `rng` parameter and default it to np.random. That keeps today's
    behaviour exactly -- verified by regenerating the Chapter 3 and Chapter 4
    figures byte-identically -- while making the correct thing reachable.

    Assigning np.random (`rng = np.random if rng is None else rng`) is the
    intended default and is not a call, so it does not trip this. Only *drawing*
    does.
    """
    relative = _relative(module)
    tree = ast.parse(module.read_text(encoding="utf-8"))

    # check_*/demo_* are by-hand self-checks, not library surface; they seed
    # themselves and are covered by the uncollected-tests ratchet above.
    demo_lines = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith(("check_", "demo_")):
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    demo_lines.add(child.lineno)

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if not (
            isinstance(owner, ast.Attribute)
            and owner.attr == "random"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "np"
        ):
            continue
        if node.func.attr in {"seed", "default_rng", "Generator", "RandomState"}:
            continue
        if node.lineno in demo_lines:
            continue
        offenders.append(f"line {node.lineno}: np.random.{node.func.attr}")

    if not offenders:
        return

    if relative in KNOWN_GLOBAL_RNG_IN_CORE:
        pytest.skip(f"known pre-existing global RNG draw ({relative})")

    assert not offenders, (
        f"{relative} draws from the global RNG in library code:\n  "
        + "\n  ".join(offenders)
        + "\n\nAccept an rng parameter and default it to np.random, so a caller "
        "threading its own Generator actually covers this draw. Note "
        "Generator has no randn -- use standard_normal, which draws the same "
        "values from the same stream."
    )


# Files sitting in a chapter directory that are not example_*.py, predating
# this check. Same ratchet: only shrink it.
#
# Empty. Nine were listed, all Chapter 8: seven runnable demos named after what
# they do rather than what they are, and two library modules that have since
# moved to core/fusion/. Chapter 8 now holds nothing but example_*.py and
# __init__.py, the same as the six chapters that already did.
#
# `ls ch8_sensor_fusion/example_*.py` reported one file when this was written.
# It reports eight now, which is how many runnable demos the chapter has.
KNOWN_NON_EXAMPLE_CHAPTER_FILES: set = set()

# Chapter modules that import a sibling from their own chapter, predating this
# check. Same ratchet: only shrink it.
#
# Seven files, twelve imports, all Chapter 8. Every other chapter has zero, so
# this is the structural fact underneath the naming one above: ch8's examples
# are not leaves. example_tc_fusion is both a demo and the module four others
# import load_fusion_dataset and run_tc_fusion from.
#
# Which is why renaming alone would make things worse rather than better: it
# would produce examples that import examples, a shape no other chapter has.
# The library half belongs in core/fusion/, next to the gating, adaptive and
# tuning modules it already sits beside.
# Empty. Seven files carried twelve of these, all Chapter 8, and they are gone:
# tc_models and lc_models now live in core/fusion/ alongside the gating,
# adaptive and tuning modules they always sat beside, and load_fusion_dataset,
# run_tc_fusion and run_lc_fusion moved to core/fusion/dataset.py,
# tightly_coupled.py and loosely_coupled.py. The two EKF demos went from
# 658 and 474 lines to 255 and 210, which is what an example is supposed to
# weigh: evaluate, plot, main.
#
# Verified by the chapter's own gate rather than by reading: all eight README
# transcripts and all 27 committed figure files are unchanged, byte for byte.
KNOWN_INTRA_CHAPTER_IMPORTS: set = set()


def _chapter_dirs():
    """Every chapter directory."""
    return sorted(p for p in REPO_ROOT.glob("ch*_*") if p.is_dir())


@pytest.mark.parametrize("chapter", _chapter_dirs(), ids=lambda p: p.name)
def test_chapter_directories_hold_only_examples(chapter):
    """A chapter directory is a set of runnable examples and nothing else.

    The convention is not written down anywhere, which is exactly the problem:
    six chapters follow it exactly, so readers learn it by induction and then
    apply it to the seventh. `ls chX/example_*.py` is how you find out what a
    chapter offers, and it under-reports Chapter 8 by a factor of eight.

    Shared code goes in core/. If a chapter file has no __main__ it is a
    library, and a library in a chapter directory is one no other chapter can
    reach.
    """
    offenders = sorted(
        _relative(path)
        for path in chapter.glob("*.py")
        if path.name != "__init__.py"
        and not path.name.startswith("example_")
        and _relative(path) not in KNOWN_NON_EXAMPLE_CHAPTER_FILES
    )

    assert not offenders, (
        f"{chapter.name} holds {len(offenders)} file(s) that are neither "
        f"example_*.py nor __init__.py:\n  "
        + "\n  ".join(offenders)
        + "\n\nName a runnable demo example_<what it shows>.py so `ls "
        "chX/example_*.py` reports it. Move anything importable into core/."
    )


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_chapter_module_does_not_import_its_sibling(script):
    """An example is a leaf: it imports core, never the example next door.

    Chapters 2 through 7 have zero intra-chapter imports between them. The
    moment one example imports another, the second one stops being an example
    and becomes a library that only its own chapter can use -- invisible to
    core/, untested as library surface, and impossible to rename without
    touching every caller.
    """
    relative = _relative(script)
    package = script.parent.name
    tree = ast.parse(script.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == package or module.startswith(f"{package}."):
                offenders.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(f"{package}."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")

    if not offenders:
        return

    if relative in KNOWN_INTRA_CHAPTER_IMPORTS:
        pytest.skip(f"known pre-existing intra-chapter import ({relative})")

    assert not offenders, (
        f"{relative} imports {len(offenders)} name(s) from its own chapter:\n  "
        + "\n  ".join(offenders)
        + "\n\nShared code belongs in core/, where every chapter can reach it "
        "and where it is tested as library surface."
    )


# Examples calling plt.show() directly, predating this check. Same ratchet:
# only shrink it.
#
# Empty. Nineteen of the thirty-eight examples ended in a bare `plt.show()` and
# nineteen did not, undocumented either way, so a reader could not predict
# whether running one would open a window. The ones that did **block** under a
# GUI backend until the window is closed -- which makes running several in
# sequence an exercise in clicking, and cost this repository's own first
# usability walkthrough several minutes of believing an example had hung -- and
# warn under Agg.
#
# `core.eval.show_figures_if_requested` is the one place that decides, reading
# IPIN_SHOW_FIGURES, so the answer is the same for all of them and is written
# down once. The figures are saved either way.
KNOWN_DIRECT_PLT_SHOW: set = set()


@pytest.mark.parametrize("script", _chapter_scripts(), ids=_relative)
def test_examples_do_not_call_plt_show_directly(script):
    """Whether a demo opens a window is one decision, not thirty-eight.

    A bare ``plt.show()`` blocks under a GUI backend and warns under Agg, and
    doing it in some examples and not others leaves a reader unable to predict
    either. Call ``core.eval.show_figures_if_requested()`` instead; it honours
    ``IPIN_SHOW_FIGURES`` and does nothing by default.
    """
    relative = _relative(script)
    tree = ast.parse(script.read_text(encoding="utf-8"))

    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "show"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "plt"
    ]

    if not offenders:
        return

    if relative in KNOWN_DIRECT_PLT_SHOW:
        pytest.skip(f"known pre-existing plt.show ({relative})")

    assert not offenders, (
        f"{relative} calls plt.show() directly at {', '.join(offenders)}.\n\n"
        "Use core.eval.show_figures_if_requested(), which opens the windows "
        "only when IPIN_SHOW_FIGURES is set. A bare plt.show() blocks under a "
        "GUI backend, and having it in some examples and not others means a "
        "reader cannot predict what running one does."
    )
