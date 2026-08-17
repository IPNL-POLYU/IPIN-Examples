"""The transcripts in the chapter READMEs must be what the examples print.

Every chapter README shows sample output in a fenced block. Those blocks were
typed in by hand once and then left behind by the code, and an audit found the
drift had reached the point of teaching the opposite of the truth:

- ``ch6`` reported the IMU strapdown final error as 252.0 m against an actual
  645.6 m, and described a PDR failure mode -- paths stretched outward, ~40%
  step over-count, 239 steps detected against 171 -- that had been fixed. The
  example now finds 166 steps against 168 taken.
- ``ch4``'s comparison block was an entirely different format: a summary table
  with a "Success Rate" column that no longer exists. It claimed AOA succeeded
  100% of the time, where the example now reports AOA diverging even at zero
  angular noise, which is the whole lesson of that table.
- ``ch5``'s comparison table was roughly half the real error, and missing a
  method the example has since grown.
- ``ch7`` showed 5 loop closures and +35.10%; the run produces 1 and +26.93%.
- ``ch8`` carried *three* mutually inconsistent sets of LC/TC numbers in one
  file, none of which matched the code.

Fixing the numbers once would buy nothing -- they were right once too. So the
blocks are checked here instead, and the contract is deliberately strict:

    **A marked block must be output the example actually produced.**

Not a paraphrase of it. The old blocks were lightly edited transcripts --
"Results (IMU-only, no corrections):" where the program prints "RESULTS
(IMU-only, no corrections)" -- and any check loose enough to accept that
rewriting is also loose enough to accept ch4's block, where nothing matched
any more and a fuzzy check would simply have found nothing to compare.

So each non-blank line of a marked block must appear, in order, in the live
run's stdout. Two escapes make an excerpt possible without weakening that:

- a line containing only ``...`` elides however much output you like;
- a bare ``~`` stands in for one token that varies between runs.

``~`` exists for the timing columns in the comparison tables. A wall-clock
figure measured on whoever last regenerated the block is not a fact about the
code, and pinning it would make this suite fail on a slow CI runner for no
reason -- but blanking the column silently would hide that a *metric* had
started varying, so the placeholder is visible in the rendered README.

Marking a block opts it in::

    <!-- example-output: ch6_dead_reckoning.example_imu_strapdown -->
    ```
    ...
    ```

Arguments go after the module::

    <!-- example-output: ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square -->

``UNCHECKED_TRANSCRIPTS`` is the ratchet, in the same spirit as
``FRAGMENT_BLOCKS`` next door: it pins how many transcript-looking blocks each
README still carries unmarked, so leaving one behind is a deliberate line in a
diff rather than an omission nobody sees. **It must only shrink.**

Author: Li-Ta Hsu
"""

import re
from pathlib import Path

import pytest

from tests.example_runner import run_example

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Unmarked transcript-looking blocks still in each chapter README.
#:
#: THIS REGISTER MUST ONLY SHRINK. An entry is a block whose numbers nothing
#: checks, which is exactly the state the audit found the whole set in.
#:
#: One entry, and it is here because marking the block would enshrine a bug.
#:
#: ch2's transcript prints `Target: 100m East -> ENU: [6405.80, 2.49, -3.21] m`.
#: The example builds that target as `np.deg2rad(-122.4194) + 100 / 78800`:
#: 78800 is roughly metres per *degree* of longitude, but it is added to a value
#: already in radians, so the offset is 57.3x too large. 100 / 78800 rad is
#: 0.0727 deg, which at this latitude is the 6405 m the run reports. "100m
#: North" is wrong the same way (5729 m); "50m Up" is right, because a height is
#: metres either way.
#:
#: Pinning that transcript would make this suite defend the defect, so the block
#: stays unmarked until the example is fixed. Mark it in the same change.
#:
#: Every other transcript in every chapter README is now checked. Four blocks
#: were left off an earlier draft of this register with reasons -- unseeded
#: RNGs, a single un-averaged draw -- that turned out on measurement to be
#: false: all four are bit-reproducible across processes, and the ch4 example
#: already reports a 2000-draw RMS beside the single draw. Verify before
#: registering; a plausible reason is not a reason.
UNCHECKED_TRANSCRIPTS = {
    "ch2_coords": 1,
}

#: Spans that legitimately differ between runs and machines, masked before
#: comparison. Deliberately short: every entry is a licence for the block to
#: be wrong about something, so a metric must never end up in here.
VOLATILE_SPANS = [
    # "Computation time: 0.251 s (39891x real-time)"
    (re.compile(r"\d+(?:\.\d+)?\s*s\s*\(\d+x real-time\)"), "<time>"),
    (re.compile(r"(?<=time:)\s*\d+(?:\.\d+)?\s*s"), " <time>"),
    (re.compile(r"\d+(?:\.\d+)?\s*(?:ms|MB)\b"), "<varies>"),
    # Absolute figure paths, which contain the scratch directory under test.
    (re.compile(r"\S*[/\\]figs[/\\]?\S*"), "<path>"),
]

MARKER = re.compile(
    r"^<!--\s*example-output:\s*([\w.]+)([^>]*?)-->\s*$", re.M
)
ELISION = "..."
WILDCARD = "~"

# A transcript looks like console output rather than prose or a directory tree.
TRANSCRIPT_SIGNALS = ("====", "RMSE", "Results", "results", "Error", "error")


def _normalise(line: str) -> str:
    """Collapse whitespace and mask run-to-run variation."""
    for pattern, replacement in VOLATILE_SPANS:
        line = pattern.sub(replacement, line)
    return re.sub(r"\s+", " ", line).strip()


def _matches(expected: str, actual: str) -> bool:
    """Whitespace-insensitive equality, with ``~`` matching one token."""
    if WILDCARD not in expected:
        return expected == actual
    want = expected.split(" ")
    got = actual.split(" ")
    if len(want) != len(got):
        return False
    return all(w == WILDCARD or w == g for w, g in zip(want, got))


def _find_from(live, expected, start):
    """Index of the first line at or after ``start`` matching ``expected``."""
    for i in range(start, len(live)):
        if _matches(expected, live[i]):
            return i
    return -1


def _fenced_blocks(text: str):
    """Yield (language, body_lines, marker_or_None) for every fenced block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        fence = re.match(r"^```(\w*)\s*$", lines[i])
        if not fence:
            i += 1
            continue
        marker = None
        for back in (i - 1, i - 2):
            if back >= 0:
                found = MARKER.match(lines[back].strip())
                if found:
                    marker = (found.group(1), found.group(2).split())
                    break
        body = []
        j = i + 1
        while j < len(lines) and not re.match(r"^```\s*$", lines[j]):
            body.append(lines[j])
            j += 1
        yield fence.group(1), body, marker
        i = j + 1


def _looks_like_transcript(language: str, body: list) -> bool:
    """A console transcript, as opposed to prose, a tree or a code sample."""
    if language not in ("", "text", "console"):
        return False
    joined = "\n".join(body)
    if not re.search(r"\d+\.\d+", joined):
        return False
    # Directory trees are drawn with box characters and are not output. They
    # are excluded explicitly rather than by signal count: ch7's tree annotates
    # its dataset directories with the SLAM improvement figures, so the moment
    # those were corrected to one decimal place it started matching on
    # "contains a decimal number" alone.
    if sum(1 for line in body if "├──" in line or "└──" in line) >= 3:
        return False
    return sum(joined.count(s) for s in TRANSCRIPT_SIGNALS) >= 3


def _marked_blocks():
    """Every opted-in transcript, as (readme, module, args, body)."""
    found = []
    for readme in sorted(REPO_ROOT.glob("ch*_*/README.md")):
        text = readme.read_text(encoding="utf-8")
        for _language, body, marker in _fenced_blocks(text):
            if marker:
                module, args = marker
                found.append((readme, module, tuple(args), tuple(body)))
    return found


def _block_id(case) -> str:
    readme, module, args, _body = case
    return f"{readme.parent.name}:{module.split('.')[-1]}{'+' + '+'.join(args) if args else ''}"


@pytest.mark.slow
@pytest.mark.parametrize("case", _marked_blocks(), ids=_block_id)
def test_readme_transcript_is_what_the_example_prints(case):
    """Every line of a marked block appears, in order, in the live output."""
    readme, module, args, body = case

    run = run_example(module, *args)
    assert run.process.returncode == 0, (
        f"{module} exited {run.process.returncode}; its README transcript "
        f"cannot be checked.\n{run.process.stderr[-2000:]}"
    )

    live = [_normalise(line) for line in run.process.stdout.splitlines()]
    live = [line for line in live if line]

    cursor = 0
    for raw in body:
        expected = _normalise(raw)
        if not expected or expected == ELISION:
            continue
        hit = _find_from(live, expected, cursor)
        if hit >= 0:
            cursor = hit + 1
        else:
            nearby = "\n    ".join(live[max(0, cursor - 2):cursor + 8])
            pytest.fail(
                f"{readme.relative_to(REPO_ROOT).as_posix()} shows a line that "
                f"`python -m {module} {' '.join(args)}`.strip() no longer "
                f"produces, at or after line {cursor} of its output:\n\n"
                f"  README: {expected}\n\n"
                f"  output around there:\n    {nearby}\n\n"
                f"Rerun the example and paste the current output into the "
                f"block. Do not hand-edit it into agreement -- the point of "
                f"the marker is that the block is a real transcript."
            )


@pytest.mark.parametrize("readme", sorted(REPO_ROOT.glob("ch*_*/README.md")),
                         ids=lambda p: p.parent.name)
def test_unmarked_transcripts_match_the_register(readme):
    """No new unchecked transcript blocks."""
    chapter = readme.parent.name
    text = readme.read_text(encoding="utf-8")

    unmarked = sum(
        1
        for language, body, marker in _fenced_blocks(text)
        if marker is None and _looks_like_transcript(language, body)
    )
    allowed = UNCHECKED_TRANSCRIPTS.get(chapter, 0)

    assert unmarked <= allowed, (
        f"{chapter} has {unmarked} unmarked transcript blocks, more than the "
        f"{allowed} on the register. A block showing example output needs an "
        f"<!-- example-output: <module> --> marker above its fence so this "
        f"suite checks it, or a deliberate entry in UNCHECKED_TRANSCRIPTS "
        f"saying why it cannot be checked yet."
    )
    assert unmarked == allowed, (
        f"{chapter} now has {unmarked} unmarked transcript blocks where the "
        f"register expects {allowed}. Lower the UNCHECKED_TRANSCRIPTS entry -- "
        f"it is a debt register and only shrinks."
    )
