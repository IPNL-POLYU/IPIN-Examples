"""Every source file a document tells a reader to run has to exist.

The guards next door all work by *executing* something -- the ```python blocks in
the dataset and chapter READMEs, the notebooks, an example's stdout against its
transcript. That leaves one class of claim untouched: a path written in prose or
inside a ```bash fence. Nothing parses `python scripts/generate_x.py`, so a file
rename is invisible to CI no matter how many readers it strands.

It had drifted accordingly. Of 212 distinct repo-relative `.py` paths cited
across the documentation, 17 did not resolve in reader-facing files, and the
worst of them were regeneration commands in shipped dataset READMEs -- three
Chapter 8 datasets and one Chapter 5 dataset each told the reader to run a
generator under a name it lost when the scripts were prefixed by chapter.

Two conventions this check depends on:

- **A historical document is exempt.** `references/design_doc.md` carries
  HISTORICAL_MARKER and records the *intended* API, six names of which were
  never built; `tests/docs/test_design_doc_is_historical.py` owns that file and
  maps its old names to new ones. Flagging it here would duplicate that policy,
  and duplicated policy only has to be forgotten once.
- **`.dev/` is exempt** for the same reason without needing a marker: those are
  dated per-chapter session summaries, records of what was true when written.

Anything else that names a file it does not expect to exist has to say so and be
listed in ASPIRATIONAL below. There is exactly one such case today, and the
phrasing is the point -- ch7's "**Future work**: Could add as
`core/slam/loam.py`" is *correct* prose about a file that is deliberately
absent, and a guard that cannot tell it from a stale rename would train people
to delete honest sentences.

Author: Li-Ta Hsu
"""

import functools
import re
from pathlib import Path

import pytest

from tests.docs.test_design_doc_is_historical import HISTORICAL_MARKER

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Repo-relative Python paths: the top-level packages plus any chapter directory.
#: Deliberately anchored, so prose like "see metrics.py" is not a claim about a
#: path and is not treated as one.
PATH = re.compile(
    r"\b(?:core|scripts|tools|tests|ch[0-9]_[a-z0-9_]+)/[A-Za-z0-9_/]+\.py\b"
)

#: Paths a document names *as not existing*. Each needs the prose to say so.
#:
#: Not an escape hatch for a broken link: if the document reads as though the
#: file is there, the file has to be there. Check the sentence, not the path.
ASPIRATIONAL = {
    # "**Future work**: Could add as `core/slam/loam.py` with ..." -- LOAM is
    # discussed in the chapter but deliberately not implemented.
    "core/slam/loam.py",
    # The new-dataset walkthrough, where the reader writes this file themselves.
    "scripts/generate_my_dataset.py",
}

#: Directories whose documents record a past state rather than instruct a reader.
EXEMPT_DIRS = (".git", ".claude", ".dev", "node_modules")


def _documents():
    """Every reader-facing markdown file, in a stable order."""
    found = []
    for path in sorted(REPO_ROOT.rglob("*.md")):
        relative = path.relative_to(REPO_ROOT)
        if any(part in EXEMPT_DIRS for part in relative.parts):
            continue
        if HISTORICAL_MARKER in path.read_text(encoding="utf-8"):
            continue
        found.append(path)
    return found


def _name(path):
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_documented_python_paths_resolve(document):
    """A path a document points at must be a file that is there."""
    text = document.read_text(encoding="utf-8")
    missing = {}
    for match in PATH.finditer(text):
        cited = match.group(0)
        if cited in ASPIRATIONAL or (REPO_ROOT / cited).is_file():
            continue
        line = text.count("\n", 0, match.start()) + 1
        missing.setdefault(cited, []).append(line)

    assert not missing, (
        f"{_name(document)} cites {len(missing)} path(s) that do not exist:\n  "
        + "\n  ".join(f"{p}  (line {', '.join(map(str, ls))})" for p, ls in missing.items())
        + "\n\nA renamed file is the usual cause -- point the document at the new "
        "name rather than deleting the line. If the document names the file as "
        "future work the reader is expected to write, say so in the prose and "
        "add it to ASPIRATIONAL. For a metasyntactic placeholder, write it as "
        "scripts/<name>.py -- every check here skips angle brackets."
    )


#: A backticked bare filename is a claim that the file exists. Anchored to the
#: backticks on purpose: unquoted prose mentioning a name is not a path claim,
#: and treating it as one would flag every sentence about a module.
BARE = re.compile(r"`([A-Za-z0-9_]+\.py)`")


@functools.lru_cache(maxsize=1)
def _repo_filenames():
    """Every .py basename in the repository, ignoring the exempt directories."""
    names = set()
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in EXEMPT_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        names.add(path.name)
    return frozenset(names)


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_documented_bare_filenames_resolve(document):
    """`some_file.py` in backticks must name a file that exists somewhere.

    Deliberately weaker than the path check above -- it asks only that the name
    exist, not that it sit where the reader would infer. That is enough to catch
    a rename, which is what actually happens, and it is what the path check
    cannot see: `scripts/README.md` heads each section with a bare
    **`generate_x.py`**, so six of its seven stale names carried no directory
    prefix and no anchored path regex would ever have matched them.
    """
    text = document.read_text(encoding="utf-8")
    have = _repo_filenames()
    missing = {}
    for match in BARE.finditer(text):
        name = match.group(1)
        if name in have or any(a.endswith("/" + name) for a in ASPIRATIONAL):
            continue
        line = text.count("\n", 0, match.start()) + 1
        missing.setdefault(name, []).append(line)

    assert not missing, (
        f"{_name(document)} names {len(missing)} file(s) that do not exist:\n  "
        + "\n  ".join(f"{n}  (line {', '.join(map(str, ls))})" for n, ls in missing.items())
        + "\n\nUsually a rename the document did not follow. Point it at the "
        "current name; if the file is future work, say so and list it in "
        "ASPIRATIONAL."
    )


#: `python -m package.module` -- the third way a document names runnable code,
#: and the one neither check above can see: it carries no slash and no `.py`.
MODULE = re.compile(r"python\s+-m\s+([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)")


@pytest.mark.parametrize("document", _documents(), ids=_name)
def test_documented_modules_resolve(document):
    """`python -m a.b` must name a module that is there.

    ch8's time-offset dataset told the reader to run
    `python -m ch8_sensor_fusion.tc_uwb_imu_ekf_augmented` twice, in two
    separate worked experiments. No such module was ever written -- online
    time-offset estimation is discussed in the chapter but not implemented --
    and neither the path check nor the bare-name check could see it, because the
    invocation contains no slash and no `.py`.
    """
    text = document.read_text(encoding="utf-8")
    missing = {}
    for match in MODULE.finditer(text):
        module = match.group(1)
        # `python -m ch7_slam.example_*` is a glob standing for "any of these".
        if text[match.end() : match.end() + 1] in ("*", "<"):
            continue
        if (REPO_ROOT / (module.replace(".", "/") + ".py")).is_file():
            continue
        line = text.count("\n", 0, match.start()) + 1
        missing.setdefault(module, []).append(line)

    assert not missing, (
        f"{_name(document)} runs {len(missing)} module(s) that do not exist:\n  "
        + "\n  ".join(f"{m}  (line {', '.join(map(str, ls))})" for m, ls in missing.items())
        + "\n\nIf the feature was never built, remove the worked example rather "
        "than leaving a command the reader cannot run."
    )
