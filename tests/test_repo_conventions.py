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
# Deliberately not asserting that listed files still violate: one of them is
# being fixed in a parallel session, and a stale-entry check would turn their
# fix into someone else's red build.
KNOWN_RAW_SAVEFIG = {
    "ch7_slam/example_bundle_adjustment.py",
    "ch7_slam/example_pose_graph_slam.py",
    "ch7_slam/example_slam_frontend.py",
    "ch8_sensor_fusion/calibration_demo.py",
    "ch8_sensor_fusion/compare_lc_tc.py",
    "ch8_sensor_fusion/lc_uwb_imu_ekf.py",
    "ch8_sensor_fusion/observability_demo.py",
    "ch8_sensor_fusion/tc_uwb_imu_ekf.py",
    "ch8_sensor_fusion/temporal_calibration_demo.py",
    "ch8_sensor_fusion/tuning_robust_demo.py",
}

# Files known to draw from the unseeded global RNG, predating this test.
# Same ratchet: only shrink it. Each entry is an example whose committed
# figures cannot be regenerated, so a figure diff there means nothing.
KNOWN_UNSEEDED_RNG = {
    "ch5_fingerprinting/example_classification.py",
    "ch6_dead_reckoning/example_environment.py",
    "ch6_dead_reckoning/example_imu_strapdown.py",
    "ch6_dead_reckoning/example_pdr.py",
    "ch6_dead_reckoning/example_wheel_odometry.py",
}


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
