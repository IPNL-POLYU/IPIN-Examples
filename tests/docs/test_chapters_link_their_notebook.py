"""A chapter README must offer the interactive version of itself.

Seven chapters ship a Jupyter notebook, every one of them carrying a Colab
bootstrap cell that clones and installs the repository, so a reader with a
browser and no Python can run the chapter. **Not one chapter README mentioned
it** -- measured, 0 of 7 for the word "notebook" and 0 of 7 for a Colab link.
The badges lived only in the top-level README's table.

That matters more than a missing link usually does, because of where readers
arrive. Someone following a reference from the book, or clicking into
`ch3_estimators/` from the repository listing, lands on the chapter README and
never sees the front page. For them the runnable version simply did not exist.

It also breaks the step in the middle. A reader who has just been shown a figure
and wants to change a parameter and see what happens has, from the chapter
README, nowhere to do that -- and changing a parameter to see what happens is
most of what these examples are for.

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

#: Chapter directory -> the notebook that covers it. Spelled out rather than
#: derived, because the names do not match: ch4_rf_point_positioning is covered
#: by ch4_rf_positioning.ipynb.
NOTEBOOKS = {
    "ch2_coords": "ch2_coordinate_systems",
    "ch3_estimators": "ch3_state_estimation",
    "ch4_rf_point_positioning": "ch4_rf_positioning",
    "ch5_fingerprinting": "ch5_fingerprinting",
    "ch6_dead_reckoning": "ch6_dead_reckoning",
    "ch7_slam": "ch7_slam",
    "ch8_sensor_fusion": "ch8_sensor_fusion",
}

COLAB = re.compile(
    r"https://colab\.research\.google\.com/github/[\w-]+/[\w-]+/blob/\w+/"
    r"notebooks/([\w]+)\.ipynb"
)

#: How far into the file the badge has to appear. A link to the interactive
#: version is worth nothing three screens down, past the equation tables.
FIRST_LINES = 12


def _chapters():
    return sorted(NOTEBOOKS)


@pytest.mark.parametrize("chapter", _chapters())
def test_every_chapter_readme_links_its_notebook(chapter):
    """The badge must be near the top and point at this chapter's notebook."""
    expected = NOTEBOOKS[chapter]
    assert (
        WORKSPACE_ROOT / "notebooks" / f"{expected}.ipynb"
    ).is_file(), f"NOTEBOOKS maps {chapter} to {expected}.ipynb, which does not exist."

    readme = WORKSPACE_ROOT / chapter / "README.md"
    head = "\n".join(readme.read_text(encoding="utf-8").splitlines()[:FIRST_LINES])
    linked = COLAB.findall(head)

    assert linked, (
        f"{chapter}/README.md has no Colab badge in its first {FIRST_LINES} "
        "lines. A reader who arrives here from the book, or from the repository "
        "listing, never sees the top-level README's table and so never learns "
        "the chapter has a runnable version. Add:\n\n"
        "    [![Open In Colab](https://colab.research.google.com/assets/"
        f"colab-badge.svg)](https://colab.research.google.com/github/"
        f"IPNL-POLYU/IPIN-Examples/blob/main/notebooks/{expected}.ipynb)\n"
    )
    assert expected in linked, (
        f"{chapter}/README.md links {linked}, not its own notebook "
        f"{expected}.ipynb."
    )
