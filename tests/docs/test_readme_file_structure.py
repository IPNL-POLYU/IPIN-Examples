"""A chapter's "File Structure" block has to describe the chapter that exists.

Two claims, both reader-facing and neither previously checked:

- Every `example_*.py` in a chapter directory appears in that chapter README's
  `## File Structure` block, and nothing the block names there is absent. A
  student picks what to run from this list; a script missing from it is a script
  they never find.
- Every file the block names resolves on disk, at both tree levels.

`tests/docs/test_documented_paths_exist.py` cannot see any of this, by design:
it matches *anchored* repo-relative paths like `core/utils/angles.py`, while a
tree block writes a directory header once and then bare filenames under it
(`├── angles.py`). CLAUDE.md already records that shape -- six of one file's
seven stale names carried no directory prefix, "so no path regex would ever have
matched them".

Unchecked, it had drifted in both forms. Five chapters omitted an example each
-- the same five the deleted `docs/architecture/` diagrams omitted, which is the
tell that these lists were copied once and never revisited. Two named a `core/`
module under a name it had lost (`core/utils/angle_diff.py` for `angles.py`,
`core/sim/trajectories.py` for `imu_from_trajectory.py`). And sixteen entries
described dataset contents that do not exist in any form: ch6's two IMU datasets
were documented as `time.txt` / `accel.txt` / `gyro.txt` when they ship
`imu.npz` and `truth.npz`, and all three ch5 fingerprint datasets as
`fingerprints.csv` when they ship three `.npy` arrays and a `metadata.json`.

**The parser is the risky part of this guard, not the assertions.** Writing it
produced three false findings in a row, each of which looked exactly like drift:

- `(example_[a-z0-9_]+\\.py)` matches inside `test_example_pose_graph_runs.py`,
  so a test file in the block was reported as a stale example. Hence the
  lookbehind.
- A character class without `-` truncates `ch7_prompts_1-6_COMPLETE.md` to
  `ch7_prompts_1` and reports a file that exists as missing.
- The same class without `*` truncates the glob `ch7_prompt*_*.md` to
  `ch7_prompt`. Globs are legitimate in these blocks and must be skipped whole,
  which means matching the `*` first in order to see it.

That is the "a tool that cannot read the thing reports the thing as broken"
pattern from CLAUDE.md, three more times. The `_entries` helper below is
exercised by `test_the_parser_reads_the_shapes_that_fooled_it` so those cases
cannot silently regress.

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: `example_*.py` not preceded by an identifier character, so `test_example_x.py`
#: does not read as a claim about an example named `example_x.py`.
EXAMPLE = re.compile(r"(?<![A-Za-z0-9_])(example_[a-z0-9_]+\.py)")

#: A tree entry name. `-` and `*` are both in the class deliberately -- see the
#: module docstring; leaving either out silently truncates a name and
#: manufactures a finding.
NAME = r"[A-Za-z0-9_.*\-]+/?"
TOP = re.compile(r"^[├└]── (" + NAME + ")")
NESTED = re.compile(r"^(?:│   |    )[├└]── (" + NAME + ")")

#: Entries that name a pattern rather than a file.
PLACEHOLDERS = ("...", "…")


def _block(chapter: Path) -> str:
    text = (chapter / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^## File Structure\n(.*?)(?=^## )", text, re.S | re.M)
    assert match, f"{chapter.name}/README.md has no '## File Structure' section"
    return match.group(1)


def _entries(block: str) -> list[str]:
    """Repo-relative paths the tree block claims exist.

    Globs and `...` are skipped: they are legitimate shorthand in these blocks
    and not a claim about a particular file.
    """
    found: list[str] = []
    top_dir: str | None = None
    sub_dir: str | None = None
    for raw in block.split("\n"):
        line = raw.rstrip()
        header = line.split("#")[0].strip()
        if header.endswith("/") and not line.startswith(("├", "└", "│", " ")):
            top_dir, sub_dir = header, None
            continue
        if top_dir is None:
            continue
        match = TOP.match(line)
        if match:
            name = match.group(1)
            sub_dir = name if name.endswith("/") else None
            if not sub_dir and "*" not in name and name not in PLACEHOLDERS:
                found.append(top_dir + name)
            continue
        match = NESTED.match(line)
        if match and sub_dir:
            name = match.group(1)
            if "*" not in name and name not in PLACEHOLDERS:
                found.append(top_dir + sub_dir + name)
    return found


def _chapters() -> list[Path]:
    return sorted(p for p in REPO_ROOT.glob("ch[2-8]_*") if p.is_dir())


@pytest.mark.parametrize("chapter", _chapters(), ids=lambda p: p.name)
def test_file_structure_lists_every_example(chapter):
    """The block is exhaustive over the chapter's own runnable demos, both ways."""
    block = _block(chapter)
    on_disk = {p.name for p in chapter.glob("example_*.py")}
    listed = set(EXAMPLE.findall(block))

    missing = sorted(on_disk - listed)
    stale = sorted(listed - on_disk)
    assert not missing, (
        f"{chapter.name}/README.md's File Structure omits {missing}. A reader "
        f"picks what to run from that list."
    )
    assert not stale, (
        f"{chapter.name}/README.md's File Structure names {stale}, which no "
        f"longer exists."
    )


@pytest.mark.parametrize("chapter", _chapters(), ids=lambda p: p.name)
def test_file_structure_entries_exist(chapter):
    """Every file the tree names resolves, at both levels of nesting."""
    absent = [
        name for name in _entries(_block(chapter)) if not (REPO_ROOT / name).exists()
    ]
    assert (
        not absent
    ), f"{chapter.name}/README.md's File Structure names absent files: {absent}"


def test_the_parser_reads_the_shapes_that_fooled_it():
    """Pin the three parsing traps, each of which produced a false finding.

    Without these the guard stays green while quietly mis-reading the blocks it
    is supposed to police, which is the failure mode that made the old diagrams
    worth deleting in the first place.
    """
    block = (
        "ch7_slam/\n"
        "├── example_pose_graph_slam.py       # real\n"
        "└── figs/\n"
        "    └── slam_with_maps.png\n"
        "\n"
        "tests/ch7_slam/\n"
        "└── test_example_pose_graph_runs.py  # a test, not an example\n"
        "\n"
        ".dev/\n"
        "├── ch7_prompts_1-6_COMPLETE.md      # hyphen must survive\n"
        "└── ch7_prompt*_*.md                 # a glob, skipped whole\n"
    )
    entries = _entries(block)

    assert "ch7_slam/example_pose_graph_slam.py" in entries
    assert "ch7_slam/figs/slam_with_maps.png" in entries, "nested entry lost"
    assert ".dev/ch7_prompts_1-6_COMPLETE.md" in entries, "hyphenated name truncated"
    assert not any(
        "ch7_prompt*" in e or e == ".dev/ch7_prompt" for e in entries
    ), "a glob was truncated into a claim about a concrete file"
    assert EXAMPLE.findall(block) == [
        "example_pose_graph_slam.py"
    ], "test_example_*.py must not read as an example"
