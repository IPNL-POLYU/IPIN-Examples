"""The notebooks must execute, in a real kernel, top to bottom.

Seven notebooks, 59 code cells, and nothing had ever run them. They were the
last body of executable documentation here without a guard: the dataset READMEs
have had one for a while, the chapter READMEs got one recently, and the
transcripts are checked against live output. The notebooks had only the check
that their ``from core ...`` imports resolve, which is the shallowest thing that
can be verified about a notebook.

**They all pass, and that is worth saying plainly**, because the expectation
going in was the opposite. Every previous time a guard was added over unexecuted
documentation here it found real defects -- eight API drifts in the dataset
READMEs, five in the chapter READMEs. On that base rate the notebooks looked
certain to be broken. Measured, all 59 cells run clean in about 56 s.

A first attempt with a bare ``exec`` reported 37 of 59 failing, which was
entirely the harness: every notebook opens with ``%matplotlib inline``, stored
in source form as ``get_ipython().run_line_magic(...)``. Outside IPython that
name does not exist, and because a notebook shares one namespace top to bottom,
the NameError in cell 1 cascaded into every later cell that needed its imports.
**A tool that cannot run the thing reports the thing as broken**, and the shape
is worth recognising: a first-cell failure in a shared namespace makes the
whole file look dead. So this runs a real kernel, which is also what a reader
has.

Being green on the day it lands is the point rather than a disappointment. The
equation index was green and unwatched for months before it was wired into CI;
that is the state this prevents.

Author: Li-Ta Hsu
"""

import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

# Imported plainly rather than through pytest.importorskip. Both arrive with the
# declared `jupyter` dependency and both are named in the dev extra, so a
# missing one is a broken environment and should say so. An importorskip here
# would turn that into a silent skip -- the exact failure the pyflakes entry in
# pyproject.toml was written to avoid, where a suite goes green having quietly
# run fewer tests.

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS = sorted(REPO_ROOT.glob("notebooks/*.ipynb"))

#: Per-notebook ceiling. A guard against a hung kernel, not a speed budget --
#: the slowest is ch5 at ~18 s, so this is an order of magnitude clear of it.
CELL_TIMEOUT_S = 300


def _name(path):
    return path.name


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_name)
def test_notebook_executes(notebook):
    """Every code cell runs without raising, in order, in one kernel."""
    document = nbformat.read(notebook, as_version=4)

    # allow_errors so the report can name every failing cell rather than only
    # the first. Later ones are often consequences of the first -- a notebook
    # shares its namespace -- so the message says which came first.
    client = NotebookClient(
        document,
        timeout=CELL_TIMEOUT_S,
        kernel_name="python3",
        allow_errors=True,
        # The notebooks chdir to the repo root themselves when started here,
        # and resolve data/sim relative to it.
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()

    failures = []
    for number, cell in enumerate(document.cells, start=1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                opening = (cell.source.strip().splitlines() or [""])[0][:70]
                failures.append(
                    f"cell {number} ({opening!r}): "
                    f"{output.get('ename')}: {(output.get('evalue') or '')[:200]}"
                )

    assert not failures, (
        f"{notebook.name}: {len(failures)} cell(s) raised.\n  "
        + "\n  ".join(failures)
        + "\n\nThe first one listed is the one to fix -- a notebook shares one "
        "namespace, so a failure early on turns every later cell that needed "
        "its names into a NameError."
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_name)
def test_notebook_carries_no_stored_output(notebook):
    """Committed notebooks stay output-free.

    All seven are, and keeping them that way is what makes a notebook diff
    readable: stored outputs put base64 images and full stdout into every
    commit that touches a cell. Run them to read them; do not commit the run.
    """
    document = json.loads(notebook.read_text(encoding="utf-8"))
    with_output = [
        number
        for number, cell in enumerate(document.get("cells", []), start=1)
        if cell.get("cell_type") == "code" and cell.get("outputs")
    ]

    assert not with_output, (
        f"{notebook.name}: cells {with_output} carry stored output. Clear them "
        f"before committing -- `jupyter nbconvert --clear-output --inplace "
        f"notebooks/{notebook.name}` -- so the diff stays readable."
    )
