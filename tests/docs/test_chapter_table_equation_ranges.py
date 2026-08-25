"""The README's chapter table must agree with the equation index.

The top-level README's Chapter Overview carries an **Equations** column --
"Eqs. 4.1-4.69" and so on. It is the first thing a reader consults to find out
whether the equation they are looking at in the book has code here, and it had
drifted in both directions:

- **Ch4 said 4.1-4.69** while `docs/equation_index.yml` maps equations up to
  **4.108** -- the AOA OVE and PLE closed-form solvers, and the whole GDOP /
  HDOP / VDOP family at 4.104-4.108. A reader looking for GDOP would have
  concluded from the table that it was out of scope, with the implementation
  sitting in `core/rf/dop.py` the whole time.
- **Ch8 said 8.3-8.9**, omitting the observability definition at 8.1 and the
  least-squares cost at 8.2. **Ch2 said 2.9-2.23** against an indexed 2.1, and
  **Ch5 said 5.1-5.5** against 5.6.

Underselling is the failure mode to expect here, and it is the one nobody
reports: a reader who is told something is missing does not go looking for it,
so the table can be wrong for as long as it likes without generating a single
complaint. Only Ch3, Ch6 and Ch7 were right.

The invariant is the *span of what the index maps for that chapter*, including
entries with an empty `files:` list -- those are the book's vector and quantity
definitions, which the index deliberately records without a standalone
function. The span is not a promise that every number inside it is implemented;
the README says so directly beneath the table, because the index has real gaps
(Chapter 3 has no 3.5, 3.6, 3.7, 3.10 or 3.13) and pretending otherwise would
be the opposite error.

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX = REPO_ROOT / "docs" / "equation_index.yml"
README = REPO_ROOT / "README.md"

#: `| **Ch4** | RF Positioning | ... | Eqs. 4.1-4.108 |`
TABLE_ROW = re.compile(
    r"^\|\s*\*\*Ch(\d+)\*\*\s*\|.*\|\s*Eqs\.\s*"
    r"(\d+)\.(\d+)\s*-\s*(\d+)\.(\d+)\s*\|\s*$",
    re.M,
)

#: `Eq. (4.108)` -- the identifier form the index uses.
EQUATION = re.compile(r"\((\d+)\.(\d+)\)")


def _indexed_spans():
    """Chapter -> (lowest, highest) equation number the index maps."""
    entries = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    spans = {}
    for entry in entries:
        match = EQUATION.search(entry["eq"])
        if not match:
            continue
        chapter = entry["chapter"]
        number = int(match.group(2))
        low, high = spans.get(chapter, (number, number))
        spans[chapter] = (min(low, number), max(high, number))
    return spans


def _table_rows():
    """Chapter -> the span the README advertises, as written."""
    text = README.read_text(encoding="utf-8")
    rows = {}
    for match in TABLE_ROW.finditer(text):
        chapter = int(match.group(1))
        rows[chapter] = (
            (int(match.group(2)), int(match.group(3))),
            (int(match.group(4)), int(match.group(5))),
        )
    return rows


def test_the_table_has_a_row_for_every_indexed_chapter():
    """A chapter the index covers must appear in the table, and vice versa."""
    assert set(_table_rows()) == set(_indexed_spans()), (
        "The README chapter table and docs/equation_index.yml disagree about "
        "which chapters exist. Either the table gained a row the index does not "
        "cover, or a chapter was indexed without being listed."
    )


@pytest.mark.parametrize("chapter", sorted(_indexed_spans()))
def test_chapter_equation_span_matches_the_index(chapter):
    """The advertised span must be the span the index actually maps."""
    (low_ch, low_n), (high_ch, high_n) = _table_rows()[chapter]
    indexed_low, indexed_high = _indexed_spans()[chapter]

    assert (low_ch, high_ch) == (
        chapter,
        chapter,
    ), f"Ch{chapter}'s row advertises equations from chapter {low_ch}-{high_ch}."
    assert (low_n, high_n) == (indexed_low, indexed_high), (
        f"README says Ch{chapter} covers Eqs. {chapter}.{low_n}-{chapter}.{high_n}, "
        f"but docs/equation_index.yml maps Eqs. {chapter}.{indexed_low}-"
        f"{chapter}.{indexed_high}.\n\n"
        "If the index gained an entry, widen the README row to match. If the "
        "README is right and the index is missing an equation, add it there -- "
        "the index is the thing readers grep, and a narrower table quietly "
        "tells them an implemented equation does not exist."
    )
