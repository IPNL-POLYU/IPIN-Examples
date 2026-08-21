"""Derive each chapter's real dependencies from the source, not from prose.

The chapter READMEs carry an "Architecture" section: a small Mermaid pipeline
and a table of example -> core modules -> dataset. Both are generated from what
the code actually imports, and `tests/docs/test_chapter_architecture_sections.py`
regenerates them and fails if a README has drifted.

This exists because the diagrams it replaced had drifted badly and nothing
noticed. They were PlantUML sources rendered to SVG by some other toolchain, so
the two disagreed (ch6's `.puml` labelled an edge "save plots"; the shipped SVG
said "write"), `dot` is not installed anywhere in this repository, and the SVGs
named files that had since been renamed -- every one of Chapter 8's nine nodes.
Worse than stale names, they were *wrong about capability*: ch6's diagram drew
five `--data` arrows where only `example_pdr` can load a dataset at all.

A picture nobody can regenerate is a claim nothing checks. Deriving the section
from the AST means the claim and the code cannot disagree.

Author: Li-Ta Hsu
"""

from __future__ import annotations

import ast
from pathlib import Path

CHAPTER_GLOB = "ch[2-8]_*"
EXAMPLE_GLOB = "example_*.py"

# Modules worth naming at their submodule level in the table. Everything else is
# reported at package level (`core.slam`, not `core.slam.se2`), because the point
# of the table is "where does this algorithm live", not a full import list.
SUBMODULE_PACKAGES = ("core.sensors", "core.fusion")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalise(module: str) -> str:
    """Collapse an import to the level the table reports."""
    parts = module.split(".")
    for pkg in SUBMODULE_PACKAGES:
        if module.startswith(pkg + ".") and len(parts) >= 3:
            return ".".join(parts[:3])
    return ".".join(parts[:2])


def core_imports(source: str) -> list[str]:
    """Every `core.*` module a source file imports, at reporting level."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("core"):
                found.add(_normalise(node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("core"):
                    found.add(_normalise(alias.name))
    return sorted(found)


def datasets_read(source: str, known: set[str]) -> list[str]:
    """Shipped datasets this source names.

    A name only counts when `data/sim/` actually holds a directory by that name.
    Matching on a bare `chN_*` pattern instead sweeps up figure basenames --
    `ch6_zupt_drift` is a GIF, not a dataset -- which is how an earlier pass at
    this produced five dataset arrows for Chapter 6 where the truth is one.
    """
    words = set()
    token = []
    for char in source:
        if char.isalnum() or char == "_":
            token.append(char)
        else:
            if token:
                words.add("".join(token))
            token = []
    if token:
        words.add("".join(token))
    return sorted(words & known)


def known_datasets(root: Path | None = None) -> set[str]:
    root = root or _repo_root()
    sim = root / "data" / "sim"
    if not sim.is_dir():
        return set()
    return {p.name for p in sim.iterdir() if p.is_dir()}


def chapter_dependencies(root: Path | None = None) -> dict[str, dict[str, dict]]:
    """{chapter: {example_stem: {"core": [...], "data": [...]}}}."""
    root = root or _repo_root()
    known = known_datasets(root)
    result: dict[str, dict[str, dict]] = {}
    for chapter in sorted(root.glob(CHAPTER_GLOB)):
        if not chapter.is_dir():
            continue
        per_example: dict[str, dict] = {}
        for path in sorted(chapter.glob(EXAMPLE_GLOB)):
            source = path.read_text(encoding="utf-8")
            per_example[path.stem] = {
                "core": core_imports(source),
                "data": datasets_read(source, known),
            }
        if per_example:
            result[chapter.name] = per_example
    return result


BEGIN_MARKER = "<!-- BEGIN GENERATED: architecture (tools/chapter_dependencies.py) -->"
END_MARKER = "<!-- END GENERATED: architecture -->"


def _packages(per_example: dict[str, dict]) -> list[str]:
    """Distinct `core.<package>` names the chapter depends on."""
    return sorted({".".join(m.split(".")[:2]) for v in per_example.values() for m in v["core"]})


def render_section(chapter: str, per_example: dict[str, dict]) -> str:
    """The generated body of a chapter README's Architecture section.

    Deliberately small. The diagram this replaced tried to draw one edge per
    (example, module) pair and per (example, figure) pair; for Chapter 6 that was
    31 edges crossing each other 64 times, and roughly half of them said nothing
    -- every example imports the chapter's own package, and every example writes
    a figure. Those two facts are one arrow each here, and the per-example detail
    is a table, which is the right medium for a lookup and stays readable at any
    column width.
    """
    datasets = sorted({d for v in per_example.values() for d in v["data"]})
    readers = [name for name, v in per_example.items() if v["data"]]
    packages = _packages(per_example)

    lines = [BEGIN_MARKER, "", "```mermaid", "flowchart TB"]
    if datasets:
        if len(readers) == len(per_example):
            who = "every example reads it"
        elif len(readers) == 1:
            who = f"only {readers[0]} reads it"
        else:
            who = f"{len(readers)} of {len(per_example)} examples read one"
        # Full `data/sim/<name>` rather than a bare dataset name: the reader can
        # copy it, and `tests/docs/test_chapter_architecture_sections.py` can
        # check it exists. A bare name is invisible to a path check, which is how
        # the first version of that guard passed over a deliberately broken label.
        shown = "<br/>".join(f"data/sim/{d}" for d in datasets)
        lines.append(f'    D["<b>optional input</b><br/>{shown}<br/><i>{who}</i>"]')
    lines += [
        f'    E["<b>{chapter}/example_*.py</b><br/>'
        f'{len(per_example)} runnable demos"]',
        '    C["<b>the reusable library</b><br/>'
        + " · ".join(p.replace("core.", "core/") + "/" for p in packages)
        + '"]',
        f'    F["<b>{chapter}/figs/</b><br/>svg + pdf + png"]',
    ]
    if datasets:
        lines.append('    D -. "--data" .-> E')
    lines += ["    E ==> C", "    C ==> F", "```", ""]

    lines += ["| Example | Core modules | Optional dataset |", "| --- | --- | --- |"]
    for name, v in per_example.items():
        mods = ", ".join(f"`{m}`" for m in v["core"]) or "—"
        data = ", ".join(f"`{d}`" for d in v["data"]) or "—"
        lines.append(f"| `{name}` | {mods} | {data} |")
    lines += ["", END_MARKER]
    return "\n".join(lines)


REPO_BEGIN_MARKER = "<!-- BEGIN GENERATED: repo-architecture (tools/chapter_dependencies.py) -->"
REPO_END_MARKER = "<!-- END GENERATED: repo-architecture -->"

# A package this many chapters or more import is shown as prose, not as edges.
CROSS_CUTTING_CHAPTERS = 4

# Short label for each chapter node, so the top-level diagram reads as topics
# rather than as directory names.
CHAPTER_TOPICS = {
    "ch2_coords": "coordinate frames",
    "ch3_estimators": "estimation",
    "ch4_rf_point_positioning": "RF positioning",
    "ch5_fingerprinting": "fingerprinting",
    "ch6_dead_reckoning": "dead reckoning",
    "ch7_slam": "SLAM",
    "ch8_sensor_fusion": "sensor fusion",
}


def render_repo_section(deps: dict[str, dict[str, dict]]) -> str:
    """The generated body of the top-level README's Architecture section.

    Only the packages that distinguish a chapter get an edge. `core/eval` and
    `core/utils` are imported by nearly every chapter, and as edges they were
    most of what made the old repository diagram a braid of long horizontals --
    35 edges crossing 23 times. They are one box of prose here instead.

    Attribution is by use, not by exclusivity. An earlier version drew only
    packages a single chapter owned, which silently dropped `ch3_estimators`
    from the picture entirely: `core/estimators` is shared with Chapters 7 and
    8, so ch3 had nothing exclusive and vanished. A rule that can delete a
    chapter from the architecture diagram is the wrong rule.
    """
    users: dict[str, list[str]] = {}
    for chapter, per in deps.items():
        for package in _packages(per):
            users.setdefault(package, []).append(chapter)
    cross_cutting = {p for p, u in users.items() if len(u) >= CROSS_CUTTING_CHAPTERS}

    lines = [REPO_BEGIN_MARKER, "", "```mermaid", "flowchart LR"]
    for index, (chapter, per) in enumerate(sorted(deps.items())):
        distinctive = [p for p in _packages(per) if p not in cross_cutting]
        topic = CHAPTER_TOPICS.get(chapter, chapter)
        target = "<br/>".join(p.replace("core.", "core/") + "/" for p in distinctive)
        lines.append(f'    A{index}["<b>{chapter}/</b><br/>{topic}"] --> B{index}["{target}"]')
    lines += [
        '    S["<b>imported by nearly every chapter</b><br/>'
        + " · ".join(p.replace("core.", "core/") + "/" for p in sorted(cross_cutting))
        + '"]',
        "```",
        "",
        "| Chapter | Core packages it imports |",
        "| --- | --- |",
    ]
    for chapter, per in sorted(deps.items()):
        packages = ", ".join(f"`{p}`" for p in _packages(per))
        lines.append(f"| [`{chapter}/`]({chapter}/) | {packages} |")
    lines += ["", REPO_END_MARKER]
    return "\n".join(lines)


def _replace_between(text: str, begin: str, end: str, body: str) -> str:
    start = text.index(begin)
    stop = text.index(end) + len(end)
    return text[:start] + body + text[stop:]


def rewrite_readmes(root: Path | None = None) -> list[str]:
    """Regenerate every marked Architecture section in place.

    This is what the guard's failure message tells you to run, so it exists
    rather than being implied:

        python -m tools.chapter_dependencies
    """
    root = root or _repo_root()
    deps = chapter_dependencies(root)
    touched = []
    for chapter, per_example in deps.items():
        readme = root / chapter / "README.md"
        text = readme.read_text(encoding="utf-8")
        if BEGIN_MARKER not in text:
            continue
        new = _replace_between(text, BEGIN_MARKER, END_MARKER, render_section(chapter, per_example))
        if new != text:
            readme.write_text(new, encoding="utf-8")
            touched.append(str(readme.relative_to(root)))
    top = root / "README.md"
    text = top.read_text(encoding="utf-8")
    if REPO_BEGIN_MARKER in text:
        new = _replace_between(text, REPO_BEGIN_MARKER, REPO_END_MARKER, render_repo_section(deps))
        if new != text:
            top.write_text(new, encoding="utf-8")
            touched.append("README.md")
    return touched


if __name__ == "__main__":
    changed = rewrite_readmes()
    print("\n".join(changed) if changed else "already up to date")
