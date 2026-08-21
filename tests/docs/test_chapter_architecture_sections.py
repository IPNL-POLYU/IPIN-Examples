"""The architecture a README draws has to be the architecture the code has.

Every chapter README, and the top-level README, carries a generated Architecture
section: a small Mermaid pipeline and a table of example -> core modules ->
dataset, both derived from the imports by `tools/chapter_dependencies.py`. This
regenerates them and fails if a README has drifted.

It replaces seventeen PlantUML/SVG pairs, and the reason they had to go is worth
keeping, because "the diagram is ugly" was the least of it:

- **Nothing could regenerate them.** The `.puml` files were not the source of the
  shipped `.svg` files -- the SVGs were emitted by graphviz 2.43.0, and the two
  disagreed on the text they drew (ch6's `.puml` labelled an edge "save plots";
  its SVG said "write"). `dot` is not installed and CI has never run it. A
  picture nobody can rebuild is a claim nothing checks.
- **They had drifted into being wrong.** Six of seven chapter diagrams named
  files that no longer existed or omitted examples that did; every one of
  Chapter 8's nine nodes carried a pre-rename filename. Worse than a stale name,
  ch6's drew five `--data` arrows into examples that cannot load a dataset at
  all -- only `example_pdr` declares the flag. Same shape as the `--data` drift
  in `test_documented_flags_exist.py`: not a stale spelling, a capability that
  never existed.
- **They were unreadable where readers meet them.** Rendered into a GitHub
  README column of about 830 px, the eight execution-flow diagrams put their
  body text at 2.4 to 4.9 px -- Chapter 8's canvas was 3436 px wide. The nine
  component diagrams were legible but tangled: ch6 crossed its own edges 64
  times across 31 edges, and 45 to 67 percent of those edges were boilerplate
  (README to every example, every example to `figs/`) carrying no information.

Generating the section from the AST means the claim and the code cannot
disagree. The parts a generator cannot own -- the surrounding prose -- stay
prose, per the "an API coupling is a gate, a prose claim is a rule" distinction
in CLAUDE.md.

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest

from tools.chapter_dependencies import (
    BEGIN_MARKER,
    END_MARKER,
    REPO_BEGIN_MARKER,
    REPO_END_MARKER,
    chapter_dependencies,
    render_repo_section,
    render_section,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Mermaid diagram types this repository uses. A block opening with anything else
#: is a typo, and GitHub renders it as a red error box rather than a diagram.
KNOWN_DIAGRAM_TYPES = ("flowchart ", "graph ", "sequenceDiagram", "stateDiagram")

#: Things inside a Mermaid label that name something in the repository. Checked
#: because a hand-written diagram is exactly as prone to naming a renamed file as
#: the PlantUML ones were, and nothing else in the suite reads a Mermaid block.
REPO_PATH = re.compile(r"\b(?:core|data/sim|tools|scripts)/[A-Za-z0-9_/]*")
CHAPTER_DIR = re.compile(r"\bch[2-8]_[a-z_]+/")


def _sections(text: str, begin: str, end: str) -> list[str]:
    return re.findall(re.escape(begin) + r"(.*?)" + re.escape(end), text, re.S)


def _mermaid_blocks(text: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)\n```", text, re.S)


def _markdown_files() -> list[Path]:
    found = [REPO_ROOT / "README.md"]
    found += sorted(REPO_ROOT.glob("ch[2-8]_*/README.md"))
    return [p for p in found if p.is_file()]


@pytest.mark.parametrize("chapter", sorted(chapter_dependencies(REPO_ROOT)))
def test_chapter_section_matches_the_code(chapter):
    """Regenerate the chapter's Architecture section and require a verbatim match."""
    deps = chapter_dependencies(REPO_ROOT)
    readme = REPO_ROOT / chapter / "README.md"
    text = readme.read_text(encoding="utf-8")

    blocks = _sections(text, BEGIN_MARKER, END_MARKER)
    assert len(blocks) == 1, (
        f"{chapter}/README.md should carry exactly one generated Architecture "
        f"section, found {len(blocks)}"
    )

    expected = render_section(chapter, deps[chapter])
    actual = BEGIN_MARKER + blocks[0] + END_MARKER
    assert actual == expected, (
        f"{chapter}/README.md's Architecture section no longer matches what the "
        f"chapter imports. Regenerate it rather than editing by hand -- see "
        f"tools/chapter_dependencies.py."
    )


def test_repo_section_matches_the_code():
    """The top-level README's chapter-to-core map, same contract."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = _sections(text, REPO_BEGIN_MARKER, REPO_END_MARKER)
    assert len(blocks) == 1, f"README.md should carry one generated section, found {len(blocks)}"

    expected = render_repo_section(chapter_dependencies(REPO_ROOT))
    assert REPO_BEGIN_MARKER + blocks[0] + REPO_END_MARKER == expected, (
        "README.md's Architecture section no longer matches what the chapters "
        "import. Regenerate it -- see tools/chapter_dependencies.py."
    )


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.parent.name or "root")
def test_mermaid_blocks_declare_a_diagram_type(path):
    """A Mermaid block that opens with a typo renders as an error box, not a diagram."""
    for index, block in enumerate(_mermaid_blocks(path.read_text(encoding="utf-8"))):
        first = next((line for line in block.split("\n") if line.strip()), "")
        assert first.startswith(KNOWN_DIAGRAM_TYPES), (
            f"{path.relative_to(REPO_ROOT)} mermaid block {index} opens with "
            f"{first!r}, which is not one of {KNOWN_DIAGRAM_TYPES}"
        )


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.parent.name or "root")
def test_mermaid_blocks_name_real_paths(path):
    """Every repository path a diagram label names has to exist.

    The failure this exists for is not hypothetical: the diagrams it replaced
    drew `lc_uwb_imu_ekf.py` and eight siblings for a year after those files were
    renamed, and no test in the suite could see inside a picture.
    """
    missing = []
    for block in _mermaid_blocks(path.read_text(encoding="utf-8")):
        cited = set(REPO_PATH.findall(block)) | set(CHAPTER_DIR.findall(block))
        for name in cited:
            # `core/eval/` and `example_*.py` are directory or glob forms; resolve
            # the concrete part and require that much to be present.
            target = REPO_ROOT / name.rstrip("/")
            if not target.exists():
                missing.append(name)
    assert not missing, (
        f"{path.relative_to(REPO_ROOT)} draws paths that do not exist: {sorted(set(missing))}"
    )


def test_the_deleted_diagram_directory_stays_deleted():
    """`docs/architecture/` held a second, unrebuildable source of truth.

    Asserted rather than assumed because the whole point of removing it was that
    two descriptions of one architecture will diverge, and re-adding a `.puml`
    beside a generated README section recreates exactly that.
    """
    architecture = REPO_ROOT / "docs" / "architecture"
    assert not architecture.exists(), (
        "docs/architecture/ is back. The chapter READMEs now generate their own "
        "diagrams from the code; a hand-maintained copy alongside them is the "
        "drift this replaced."
    )
    stragglers = sorted(REPO_ROOT.rglob("*.puml"))
    stragglers = [p for p in stragglers if ".claude" not in p.parts]
    assert not stragglers, f"PlantUML sources nothing renders: {stragglers}"
