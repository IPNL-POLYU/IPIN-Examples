# CLAUDE.md

Working notes for agents on this repo. Facts that are non-obvious, cost someone
an investigation, or keep getting re-typed into task briefs.

Conventions live in `.cursor/rules/` and are authoritative — read
`030-figures-and-claims.mdc` before touching anything that produces a figure.
This file is the environment, not the rules.

## What this is

Code examples for an Indoor Positioning and Indoor Navigation textbook.
`core/` holds the shared library (`coords`, `sensors`, `sim`, `eval`, `fusion`,
`fingerprinting`); each `chX_*/` holds runnable examples that produce the
book's figures into `chX_*/figs/`. The figures are the deliverable.

Author line on new files: `Li-Ta Hsu`. Tests go under `tests/`, mirroring the
source tree.

## Running an example from a worktree — read this first

An editable install (`__editable__.ipin_examples-0.1.0`) points `core` at the
main checkout `C:\Users\qmohs\IPIN-Examples`. Running a script directly puts
the *script's* directory on `sys.path[0]`, not the worktree root, so `core`
resolves to the main checkout — including whatever uncommitted changes happen
to sit there. You can edit `core/` in your worktree, run an example, and test
a completely different copy without any error.

Pin the worktree root, which lands ahead of the editable finder:

```bash
PYTHONPATH=$(pwd) python ch6_dead_reckoning/example_comparison.py
```

The tell-tale symptom is a `TypeError` about an argument you just added, while
`python -c "import core..."` shows the right signature — the `-c` form puts cwd
on the path and picks up the worktree.

pytest is unaffected; it inserts rootdir itself.

## Tests

```bash
python -m pytest tests/core tests/ch6_dead_reckoning -q
```

The full suite is slow (10-16 minutes) because several tests shell out to
chapter examples.

`tests/ch7_slam` used to fail intermittently under load, and the history is
worth knowing because the shape recurs. Five tests each `subprocess.run` the
same `ch7_slam.example_pose_graph_slam` at `timeout=180`, three of them with
identical arguments. The script takes ~135 s standalone but 170-207 s inside
the suite, so the limit was being grazed and the result depended on what else
was running: identical content gave 2 failed / 1351 passed in 954 s under
load, then 1353 passed / 0 failed in 647 s when quieter.

Fixed by `tests/ch7_slam/slam_example_runner.py`, which memoises each distinct
invocation so it runs once per session (5 pose-graph runs down to 3) and sets a
timeout generous enough to be a deadlock guard rather than a performance
budget. **If you add a test that shells out to an example, go through that
runner.** It now owns every Chapter 7 example subprocess — five distinct
invocations: the pose graph inline, on the square dataset and on the high-drift
dataset, plus the front-end and the scan-matching visualisation. The
scan-matching test was the last holdout and kept its own correct copy of the
env setup for a while, which is the failure mode worth avoiding: duplicated
policy only has to be forgotten once.

Those tests still run the example with `cwd` at the repo root, because the
example resolves `data/sim` relative to the working directory. The figures are
no longer written there, though: the runner points `IPIN_FIGS_DIR` at a
scratch directory and exposes the result as `ExampleRun.figs_dir`. **Assert
against that, never against the in-repo path** — an existence check on
`ch7_slam/figs/slam_with_maps.png` passes on the committed file whether or not
your run wrote anything.

If `slam_with_maps.png` shows up in `git status` after a test run now, that is
a real change and not the churn this note used to describe.

## `docs/` is deliberately not executed, and this is why

Four bodies of documentation here carry runnable Python. Three are executed by
CI -- `data/sim/*/README.md`, `ch*/README.md`, and `notebooks/*.ipynb`. **`docs/`
is not, and that was a decision rather than an oversight.** Measured before
deciding, so the numbers are here to save the next person repeating it:

- 119 ` ```python ` blocks across 15 files. **21 run. 90 are placeholder
  fragments** (`position`, `anchors`, `x`, `landmarks` supplied by the reader).
  Two are not Python at all: a prose block with degree signs, and a `pytest ...`
  shell line.
- **Zero defects found.** Every `from core ... import` resolves; all 10
  `(symbol, file)` table claims are correct; no block touches matplotlib, so
  none carries the version exposure that broke six examples and two notebooks.

Guarding it would mean re-fencing ~98 blocks to protect 21 clean ones. The other
three bodies each earned their guard by finding real bugs on the first run --
eight API drifts in the dataset READMEs, five in the chapter READMEs, two
matplotlib crashes in the notebooks. `docs/` found none, and its ` ```python `
fence has never meant "runnable" here, only "highlight as Python".

**If you re-measure this, dedent the blocks first.** Fenced blocks nested in
numbered lists are indented, and a naive `` ```python
(.*?)
``` `` regex
captures that indentation and reports `IndentationError` on valid code. That
mistake inflated the first count of this survey from 8 non-Python blocks to 20,
and the block count from 119 to 102. The same shape cost an afternoon on the
notebooks: a bare `exec` harness reported 37 of 59 cells broken, because
`%matplotlib inline` is stored as `get_ipython().run_line_magic(...)` and the
resulting `NameError` in cell 1 cascades through a shared namespace.
**A tool that cannot read the thing reports the thing as broken** -- check the
harness before believing a survey that says most of a directory is rotten.

## A path in prose is the one claim nothing executes

Every other guard here works by running something -- the ` ```python ` blocks in
the dataset and chapter READMEs, the notebooks, an example's stdout against its
transcript. None of them look at `python scripts/generate_<name>.py` inside a
` ```bash ` fence, so a file rename is invisible to CI however many readers it
strands. `tests/docs/test_documented_paths_exist.py` closes that, and it found
**17 stale references across 10 reader-facing documents** on its first run.

The three citation forms are separate checks, because each is blind to the
others and the drift was distributed across all three:

- `scripts/<name>.py` -- an anchored repo-relative path.
- `` `<name>.py` `` -- a backticked bare filename. Six of `scripts/README.md`'s
  seven stale names carried no directory prefix, so no path regex would ever
  have matched them.
- `python -m <pkg>.<module>` -- no slash, no `.py`. This is how ch8's
  time-offset dataset
  told the reader twice to run `tc_uwb_imu_ekf_augmented`, which was never
  written.

Exempt: documents carrying `HISTORICAL_MARKER` (owned by
`test_design_doc_is_historical.py`, which maps their old names to new ones) and
`.dev/`, which is dated session summaries. Anything else naming a file it does
not expect to exist has to say so in the prose and be listed in `ASPIRATIONAL`.
There is one such case, and its phrasing is the point: ch7's "**Future work**:
Could add as `core/slam/loam.py`" is correct prose about a deliberately absent
file, and a guard that cannot tell it from a stale rename teaches people to
delete honest sentences.

**The worst of what this found was not a broken link.** Chasing
`tc_uwb_imu_ekf_augmented` showed the whole experiment around it was
unrunnable: all three commands in that block passed flags
(`--no-time-correction`, `--time-offset`, `--output`) that `tc_uwb_imu_ekf` does
not have, and the results table reported RMSE, NIS and convergence for runs
nobody could perform. Its "offline correction" row claimed 0.08-0.12 m against
an uncorrected 0.18-0.25 m -- correction more than halving the error -- where
the real demo measures 0.211 m to 0.185 m, 12.5%. **The kinematic bound was
already written down in the code**: `temporal_calibration_demo.py`'s docstring
records that a 50 ms offset at 1 m/s displaces the platform 5 cm and so cannot
cost more, having had this exact inflation removed once before. The doc kept the
old story. When a document and a docstring disagree about the size of an effect,
the one with the derivation is right.

So expect a stale path to be a thread, not a typo: pull it and check whether the
surrounding worked example ever ran.

**Flag drift is the next layer and is not guarded yet.** A scan of `--help`
against every documented invocation found five real cases in three files
(`ch6_foot_zupt_walk`, `docs/ch8_lc_tc_comparison_guide.md`, `scripts/README.md`),
plus parameter tables that were largely fictional -- all five flags ch4's
inventory entry documented were invented, and four of ch5's seven. Those tables
are now rebuilt from `--help`, with the "Range" column dropped because nothing
sourced it, and `--help` named as the authority so the tables cannot quietly
become it.

**That scan reported 9 files before it reported 3.** The first run shelled out
to `--help` without `PYTHONIOENCODING=utf-8`; the child printed a degree sign,
cp950 could not encode it, the child died, and empty help text read as "this
program has no flags". Third time in this repo that a tool which could not read
the thing reported the thing as broken -- see the notebook `exec` harness and
the docs fence regex above. `tests/example_runner.py` already sets that variable
for exactly this reason. **Set it in any subprocess you scan with, and treat an
empty result as unreadable rather than as evidence.**

## Editing a dataset README

`tests/docs/test_readme_code_blocks.py` executes every ` ```python ` block in
`data/sim/*/README.md` into one shared namespace, as a reader reads them top to
bottom, with the working directory at the repo root. All of them run today, so
a block you add must too.

Fences carry meaning here:

- ` ```python ` — a reader is expected to run this, and the guard executes it.
- ` ```py ` — an illustrative fragment (`quat = quat / np.linalg.norm(quat)`,
  or a block opening `# In tc_uwb_imu_ekf.py, add parameter:`). Not executed.
  Both fences still highlight as Python in GitHub, VS Code and mkdocs.

`FRAGMENT_BLOCKS` in that file pins the ` ```py ` count per README, because the
fence is otherwise an unreviewed escape hatch: demoting a genuinely broken
example to ` ```py ` would silence the guard in a one-word diff.

When a block does fail, **the exception type tells you which kind you have.** A
`NameError` on an undefined placeholder means it was never meant to run alone.
A `TypeError`, `ValueError`, `ImportError` or `AttributeError` means every name
resolved and the *call* is wrong — that is API drift, and it is always real.
That one rule separated eight genuine drifts from twenty-one fragments in a
register that had counted all twenty-nine as "broken".

## Editing a chapter README's "Expected Output"

`tests/docs/test_readme_example_output.py` runs the example and requires every
non-blank line of the marked block to appear, in order, in its stdout. So
**paste a real transcript, never a tidied one** — `...` on its own line elides,
`~` stands in for one varying token, and that is the whole vocabulary.
Numbers get 5% for platform noise; text must match exactly.

`UNCHECKED_TRANSCRIPTS` is the ratchet and **it is empty today**. Its last
entry was ch2's, deliberately held open because marking the block would have
pinned a bug: the example built "100m East" as
`np.deg2rad(-122.4194) + 100 / 78800`, adding a per-*degree* constant to a
value already in radians, and printed the resulting 6405.80 m as 100 m.

Two things about that shape are worth carrying forward:

- **A unit error hides behind whichever output is unit-agnostic.** Of the three
  targets, "50m Up" was right — a height is metres either way — and one correct
  line out of three made the block look like it worked. The same reflex as the
  frame audit in ch6: if one component of a vector cannot express the error,
  it will be the one you look at.
- **A constant borrowed from a passing test is not thereby correct.** 78800 came
  from `tests/core/coords/test_transforms.py`, where it is metres per degree of
  longitude at 45° and used correctly, with `deg2rad`. At this example's 37.77°
  the true figure is 88100, so it was independently wrong by 12% — an error the
  57.3x one would have hidden completely had anyone gone looking.

Derive such a conversion rather than pasting it. `core.coords.enu_to_llh_offset`
takes metres and returns radians, from the WGS84 radii of curvature at the given
latitude — no per-degree constant to mislay and no latitude to be wrong about.
Sub-millimetre residuals remain and are real: the second-order curvature term,
growing as the square of the offset, which the example prints.

## Dataset files have to agree with each other

Several datasets ship the same information in more than one form, and those
forms are checked against each other now — see
`tests/ch2_coords/test_rotation_files_agree.py` and
`test_coordinate_files_agree.py`. Both tests exist because ch2 shipped two
defects of this shape:

- `quaternions.txt` was `euler_to_quat(pitch, roll, yaw)` and
  `rotation_matrices.txt` was the same swap transposed, so the quaternion and
  the matrix for a point described rotations up to 169° apart.
- `reference_llh.txt` carried `heights[0]` — the first sampled point, which
  drew floor 5 — while the ENU was built about `height_ground = 0.0`, putting
  Up out by exactly 15 m everywhere.

Neither was caught because the READMEs' round-trip experiments **recompute both
sides from the same source file** and compare those, which passes whatever the
shipped bytes contain. If you add such a check, read the shipped file.

Every dataset has now been audited this way. Eighteen came back consistent —
though see the next section: ch2's coordinates were among the eighteen and were
wrong by 57.3x, which consistency could not see. The three defects *this* shape
found were ch2's two above and one in ch6 strapdown, where
`imu.npz/accel_xy` was map-frame acceleration while the README's Eq. (6.19)
integrates it as `v += C(theta) f dt` and so rotates it a second time. On the
circular trajectory the accelerometer carried no centripetal term at all:
integrating it drew a 16.9 m radius against a true 10.0 m. See
`tests/ch6_dead_reckoning/test_imu_is_body_frame.py`.

**A wrong frame hides as clean noise.** The residual against map-frame
acceleration was 0.1002 m/s² against a declared 0.1 — a perfect match, and
wrong. Checking a vector quantity against only one candidate frame will
confirm whichever you picked, so compare against both and let the systematic
mean pick the winner.

## Agreeing with each other is not enough — check `config.json`

The audit above passed ch2's coordinates, and they were wrong by 57.3x. The
generator added a value it named `lat_offset_deg` straight to a latitude in
radians, so a building `config.json` declares as 50 m was sampled across
2666 m × 2612 m. `test_coordinate_files_agree.py` was green throughout, because
the ENU was derived from the same inflated LLH.

**A unit or frame error is common-mode: it moves the data and every check
recomputed from that data together.** Consistency is structurally blind to it,
so it cannot be the only guard. `config.json` is the one file in a dataset
stating *intent* rather than a derived quantity — compare against it. See
`tests/ch2_coords/test_dataset_matches_its_config.py`.

Two things about that one are worth carrying:

- **The documentation had already met the bug and explained it away.** The
  dataset README carried "Issue 2: ENU Range Seems Wrong — Symptoms: ENU
  coordinates in km instead of m — Cause: Wrong reference point". Someone saw
  the symptom, guessed a cause that could not produce it, and wrote it up as
  reader error. A troubleshooting entry describing your own output is a bug
  report, not documentation; check it against the code before believing it.
- **Its neighbour was the same shape.** `config.json` also advertised
  `"rotation_roundtrip_deg": 360.0`, which the README explained as gimbal lock
  and prescribed quaternions for. It was a branch cut: yaw sampled on [0, 2π)
  and recovered on (−π, π], differenced without wrapping. Wrapped, the error is
  0.0. **A rotation error of 360° is the identity** — like a classifier at its
  base rate, the number is the arithmetic signature of a measurement bug, and
  the prescribed fix would have changed nothing.

Convert metres to radians with `core.coords.enu_to_llh_offset`, which takes
metres and returns radians so there is no per-degree quantity left to mislay.
Both ch2 instances were a hand-rolled constant; the third would have been too.

## Wrapping an angle difference

All eleven generators have now been swept for the two ch2 shapes. The unit half
is clean and structurally cannot recur — ch2 was the only geodetic generator,
and the four remaining degree/radian boundaries (`ch3:165`, `ch4:220`) convert
correctly. The wrap half turned up one more, in ch6 env sensors.

**`min(|d|, 2*pi - |d|)` is not a wrap.** It is the shorter arc only while
`|d| <= 2*pi`, and it returns a *negative* number above that. The ch6 building
walk runs true yaw to 7.84 rad while `mag_heading` returns `(-pi, pi]`, so 221
of 1800 samples got a negative "error" and the mean written into `config.json`
was 2.66° against a true 3.51°.

Two properties of this one are worth expecting:

- **It was invisible on clean data.** Noiseless, the difference is exactly 0 or
  exactly 2*pi, and `2*pi - 2*pi` is 0 — so the broken form is *right*. Only
  the noisy path that actually ships produces `2*pi + eps`, hence `-eps`.
  Checking a reduction against ideal input will confirm whichever you wrote.
- **The ratio survived.** The README's tilt-compensation experiment divides two
  of these means and still prints "1.9x worse", because both sides were
  understated alike. A ratio can stay convincing while the magnitudes under it
  are wrong, so 030's "look at the ratio" rule needs its converse: check the
  absolute numbers too.

Use `core.sensors.wrap_angle_diff` for a difference and `wrap_heading` for a
single angle. Both already existed; the generator and the dataset README each
hand-rolled a copy and both got it wrong. That is now four wrap helpers in the
repo (those two, plus `core.utils.angle_diff` and
`example_comparison._wrap_to_pi`) — prefer the shared ones, which
`tests/core/test_angle_differences_are_wrapped.py` holds to one another. See
also `tests/ch6_dead_reckoning/test_heading_error_is_wrapped.py`, whose first
two assertions had to be rewritten because `abs(wrap_angle_diff(...))` is
non-negative and under 180° *by construction*: they were testing numpy, not the
dataset. Pin what distinguishes the two reductions, not what the helper
guarantees.

The chapter examples have now been swept too, and **none had a live defect** —
every angular error they actually report already went through a correct helper.
The sweep's value was elsewhere:

- **One real bug, in `core/` again.** `factor_graph.py` built a bearing
  residual as `predicted - z` with no wrap, the case `angle_diff`'s own
  docstring names. An anchor due west and a 1 cm perturbation flips the raw
  residual to −6.2807 rad where the truth is +0.0025 — wrong magnitude *and
  wrong sign*, so the optimiser is pushed the wrong way. Third time a sweep of
  the examples found the defect underneath them.
- **Four latent subtractions**, in ch2's round-trip `PASS/FAIL` gate, ch7's two
  loop-closure checks, ch7's front-end no-op gate and ch8's calibration. All
  printed identical output before and after the fix. **That is the argument for
  wrapping anyway:** a branch-cut bug is invisible until someone changes an
  angle, and a textbook invites exactly that. ch2's gate would have printed
  `FAIL` for a perfect round-trip at a yaw of 200°.

One thing that sweep turned up is not about angles, and it is now fixed and
ratcheted: `core/estimators/*.py` held **18 `test_`-named functions that pytest
never collected**, because `testpaths = ["tests"]`. They printed "UNIT TESTS"
banners under `if __name__ == "__main__"`. All 18 passed — which is exactly why
the shape survives. **It fails silently by never speaking at all**, so there is
no red to notice; the functions look like coverage from every angle except the
one that decides whether they run.

They are now `check_*`, which is what they are: self-checks a reader runs by
hand. **The real coverage was never missing** — `tests/core/estimators/` already
holds an equation-anchored test file for every one of those six modules, so the
inline ones were duplicates wearing the name of the thing that already existed.
Before assuming an uncollected test is a coverage gap, look for the collected
file next door.

`KNOWN_UNCOLLECTED_TESTS` in `tests/test_repo_conventions.py` keeps it that way,
over `core/`, the chapters, `scripts/` and `tools/`. It is empty, and it found
five functions in two more files that the hand survey had missed: three
`test_`-named evaluation stages in
`ch5_fingerprinting/example_classification.py`, all taking required arguments so
pytest could only ever have errored on them, and two helpers in a *tool* whose
own filename began with `test_` — now `check_all_datasets.py`, since a CLI
matching `python_files` is the same confusion one level up.

Two of the 18, in `particle_filter.py`, **asserted nothing at all** and printed
`"[PASS] Test passed"` unconditionally, beneath a comment reading "Check that
filter ran successfully". They also seeded *after* constructing the filter, so
the particle cloud was drawn unseeded and three runs gave 0.2245, 0.0516 and
0.1486 m. Seeded first they are a fixed 0.2149 m, which is what made a real
assertion possible — **an unseeded check cannot be given a bound, so it tends to
be given a print instead.**

Fixing them surfaced a third thing worth knowing: that demo's `likelihood_func`
returned a shape-`(1,)` array where the signature says
`Callable[..., float]`, so `weights[i] *= likelihood` assigned an array into a
scalar slot — a numpy DeprecationWarning that says it *will* become an error.
`ch3_estimators/example_particle_bimodal.py` had it right with `np.sum` all
along. Same lesson as the matplotlib `labels=` removal that broke CI twice this
year: **a warning naming a future version is a scheduled breakage**, and the
correct form is usually already in a sibling file.

So expect a red here to be real, but confirm the tolerance first:

**Data files are written at `%.3f` or `%.6f`, so quantisation is the floor.**
Coordinates quantise at 1 mm, DOP at 5e-7, and a finite difference divides that
by `dt` — heading stored at 1e-6 rad becomes 1e-4 rad/s of apparent gyro error
at 100 Hz. Four apparent findings in that audit were tolerances set below the
storage precision, against one real defect. An exact-looking residual is the
tell that it is yours: 0.7854 rad/s turned out to be π/4 from an off-by-one in
the check, while the genuine bug was a dull, perfectly uniform 15 m.

## Attributing a dataset diff

If regenerating a dataset produces a diff, find out whether your change caused
it before believing either answer. Stash the edits, regenerate from a clean
tree, and compare:

```bash
git checkout -- data/ && git stash push -- core/ scripts/
python scripts/generate_chX_....py --preset <the one in config.json>
git status --short data/          # same files? then the drift is pre-existing
git checkout -- data/ && git stash pop
```

That is how the ch2 rotation drift was separated from a lint sweep that merely
surfaced it. Pass the preset `config.json` records — running a generator with
no arguments can use different defaults than the shipped data was built with,
which manufactures a diff that looks like drift.

**`.npz` files are not byte-reproducible.** They are zip archives carrying
member timestamps, so every regeneration rewrites the bytes whether or not a
single number changed. `git status` is therefore no evidence about `.npz`
content — load both versions and compare the arrays. Regenerating the two ch6
IMU datasets touched four files and changed exactly one array. (Figures are
the opposite: `core.eval.save_figure` is byte-reproducible, so there a diff
does mean the picture changed.)

## Auditing a chapter's numbers

A sweep of Chapters 4 and 8 found a reported number wrong in nine of eleven
examples. One question found almost all of them: **what should this number be?**
Compute it from the noise, the geometry, or the kinematics, then compare. A
50 ms clock offset at 1 m/s cannot cost 17.78 m. Range errors of 0.035 m do
not produce 0.739 m of fusion error. An alignment residual between two sensors
each carrying 0.05 m of noise should be 0.10 m, not 0.05 m.

Two things made these survive, and both are worth expecting:

- **The number looked plausible in isolation.** Nobody checks 0.67 m. It was
  wrong by sqrt(2), and the tell was that it contradicted the per-axis figures
  printed on the next two lines.
- **The correct version was usually already in the same file.** The AOA
  example's second table averaged properly while its first reported single
  draws; Chapter 4 contains `example_dop_geometry`, which validates its own
  prediction against Monte-Carlo and reports the 3.5% disagreement. When
  something looks wrong, look for the sibling that does it right before
  inventing an approach.

A later sweep across every dataset README turned up the same kind of thing in
five of the thirteen datasets it checked, plus two further shapes worth
recognising on sight, both now written up in
`.cursor/rules/030-figures-and-claims.mdc`: an example arguing that some
correction matters while its own output prints a ratio of 1.0, and a solver
whose reported accuracy is really the accuracy of the solves that happened to
converge. Read that file before trusting a printed comparison.

Every chapter's console output has now been swept this way. Chapters 2 and 6
came back clean; 3, 5 and 7 did not, and the two expectations that sweep
overturned are the useful part.

**The deepest defects were in `core/`, not in the examples.** Earlier rounds
had trained the habit of reading the example and trusting the library beneath
it. Two of the three were library bugs, and each had been quietly wrong for
every caller:

- `hierarchical_localize(coarse_method="floor")` inferred the floor by running
  `nn_localize`, which returns an `(x, y)` location with the floor already
  discarded, then taking the floor of the RP nearest that point *in x-y*. A
  multi-floor survey stacks its RPs at the same coordinates — the shipped ch5
  grid has 363 RPs on 121 distinct `(x, y)` — so the argmin was a three-way tie
  that always resolved to the lowest index. It returned floor 0 for every query
  and scored that floor's base rate, 32.7% against a 33.3% chance level, which
  reads as a hard problem rather than a broken one.
- `icp_point_to_point` returned a sum of squared errors where every caller
  gated it in metres, so the SLAM front-end rejected all 145 of its own good
  alignments and returned odometry unchanged. Written up with the
  bundle-adjustment instance of the same shape under "A sum of squares is not a
  distance" in `.cursor/rules/030-figures-and-claims.mdc`.

**A number at chance, or exactly equal to its baseline, is a bug signature.**
Both library defects printed a plausible-looking result — 29.0% floor accuracy,
"Frontend improvement: -0.00%" — and both numbers were the arithmetic signature
of a stage doing nothing. When a classifier lands on its base rate or a stage
matches its input exactly, check that its output *varies at all* before tuning
anything. That check is one line and it is cheaper than any of the
investigations it replaces:

```python
predicted = {classify(q) for q in queries}
assert len(predicted) > 1        # a constant predictor scores the base rate
```

Note that an accuracy threshold set anywhere near chance would have passed the
ch5 bug, which is why the test asserts variety rather than quality. The ch7
equivalent already existed: a previous session had left
`tests/ch7_slam/test_frontend_actually_corrects.py` asserting the front-end was
*still* a no-op, written to fail the moment it was repaired. It did, and that
is the pattern to copy when you find something broken you are not fixing now.

The most reusable habit from these sweeps is duller than any of the rules:
**check that the check can fail.** A new assertion, a new tolerance, a new
consistency test — run it against the broken input before believing the green.
It matters in both directions and both were hit: a Chapter 6 strapdown test
shipped ending in `pass  # Will fix in next version`, asserting nothing while
still appearing in the count, and a replacement assertion written during the
sweep to remove exactly that antipattern turned out to hold whether or not the
code under test did its job.

If a ch7 test does fail, re-run it in isolation before believing it.

## Exact float equality does not survive a change of CI runner

`tests/ch5_fingerprinting/test_dataset_reproduces_from_its_seed.py` compared
regenerated arrays with `np.array_equal`. It passed locally, it passed on CI,
and then it failed on CI with **max|difference| = 2.8e-14** on values of order
100 -- one to two ulp -- with **identical numpy 2.4.6 and scipy 1.17.1** in both
runs, from a branch that could not reach the generator at all (it imports only
`core.fingerprinting`, and `core/__init__.py` is empty).

Measured, so the next person does not repeat it: two local runs of the generator
and the shipped files agree **exactly**, max|difference| 0.0. The generator is
bit-reproducible on one machine and not across machines. The likely mechanism is
numpy dispatching a different SIMD kernel for `np.log10` on a different CPU,
which the heterogeneous Actions runner pool makes a coin flip; that is not
proven, but the last bit plainly does not survive the move.

**So do not assert bit equality on computed floats, only on stored ones.** The
committed figures are byte-reproducible because `save_figure` writes bytes; a
`.npy` of RSS values is arithmetic, and arithmetic is portable only to a
tolerance.

The rest of the suite was swept for the same shape and is clean, so the test to
apply is narrower than "never compare floats exactly". Exact equality is right
when both sides come from **one process** (`test_allan_variance...` calls the
generator twice and requires identical output -- correct, and the point of the
test), from **stored bytes** (save then load), from an **assignment** rather than
arithmetic, or from **integers**. It is wrong only when a fresh computation is
compared against a file some other machine produced, and this was the repository's
only instance. `tests/ch8_sensor_fusion/test_batch_update.py` already had the
distinction right on adjacent lines: `assert_array_equal` for the integer anchor
indices, `assert_array_almost_equal` for the float ranges beside them.

The tolerance is now 1e-9, and both sides of it were measured, which is the part
worth copying. Regenerating with `seed + 1` gives **23.06 dB** of difference, so
the bound sits ten orders below the defect it must still catch and four orders
above the noise it has to tolerate. **Justify a tolerance against both** --
against the noise or it is flaky, against the defect or it is decorative. Note
that the first version of this test would have caught the seed regression too;
it was not too weak, it was too strong, and the failure mode of too-strong is a
red that teaches people to distrust the suite.


## Figures

`core.eval.save_figure` is the only output path; it writes svg/pdf/png together
and is byte-reproducible, so a committed figure diff means the picture changed.

One exception, worth knowing before you chase a phantom: `bbox_inches="tight"`
derives the canvas size from measured text extents, and for a figure whose
tight box lands on a rounding boundary that size can wobble by one ulp between
processes -- `width="999.498437pt"` against `999.498438pt`. Every coordinate in
the SVG is relative to it, so the whole file differs while the picture is
identical. `ch3_estimator_comparison` does this; the Chapter 6 and 8 figures do
not. If a figure diff looks total but the printed numbers are unchanged, check
the width attribute before looking for a real cause.

Regenerate the figures for any code you change, and **open the PNGs** — this
repo has repeatedly shipped figure defects that no test caught.

Every chapter's figures have now been regenerated and looked at. All of them
reproduce byte-identically from current code, so a diff here is a real signal.
Two were wrong, and both were the same shape — a direction convention applied
backwards, in an example that builds its own data inline and so was untouched
by the dataset fixes:

- Chapter 6's `environment_mag_heading` built its field as
  `R_body_to_map.T @ [1,0,0]`, for which `atan2(m_y, m_x)` is minus the
  heading. The error sawtoothed 0-180° across the whole run while the figure
  claimed bounded heading, and the shaded disturbance windows it exists to
  highlight were indistinguishable from the baseline.
- Chapter 4's `ch4_aoa_geometry` drew each bearing ray from the anchor along
  `(sin psi, cos psi)`, but `aoa_azimuth` measures psi *from the agent toward
  the anchor*. All four rays pointed away, and none passed through the fix
  they are drawn to intersect at.

Both survived because every test on those examples checks that files were
written. **A dataset audit will not find these** — the examples do not read
`data/sim`. Only opening the picture does.

`tests/test_repo_conventions.py` enforces the mechanical parts (no raw
`savefig`, no unseeded global RNG, nothing at the repo root) as a ratchet:
pre-existing violations are listed and skipped, new ones fail. Those lists are
the current debt register and should only shrink. Every one of them is empty
today, as is `KNOWN_PYFLAKES` — see below.

## Lint

pyflakes is clean across the whole repo and `test_no_pyflakes_warnings` in
`tests/test_repo_conventions.py` keeps it that way, over `core/`, `scripts/`,
`tests/` and the chapter directories — 259 files, about 5 seconds, via the
pyflakes API rather than a subprocess each. Syntax errors count as failures
too. `pyflakes` is declared in the `dev` extra rather than inherited from
flake8, because a tool present by accident makes a test skip in silence.

**The half of that check that earns it is undefined names, not tidiness.** An
undefined name is a runtime error, so `python -m compileall` accepts the file,
and if the branch has no test nothing else notices either. That is not
hypothetical: the sweep that made the repo clean introduced one. A two-line
`replace(..., count=1)` deleted a `d_ref` definition that was still live in an
*earlier* function, because the same two lines appeared twice in the file.
compileall was happy; pyflakes named it immediately.

So **a lint sweep is not safe by construction**, and the bigger it is the less
that argument is worth. When a mechanical edit touches many files, verify by
running the examples before and after and diffing their output — the 30-file
sweep was checked that way, and only incidental differences survived (figure
paths, wall-clock durations, and warning line numbers shifted by deleted
lines). Two traps found doing it:

- `f"{{}}"` has no placeholders but renders `{}`, so dropping the prefix
  changes the output. Nothing in this repo hit it, which is only knowable
  because it was checked for.
- **Bytecode equality is the wrong equivalence test.** `f"\n" + "=" * 50`
  cannot constant-fold across the JoinedStr while `"\n" + "=" * 50` folds to a
  single constant, so one file compiled differently while printing exactly the
  same thing. Compare output, not code objects.

## Parallel sessions

Several agents often work this repo at once, on separate worktrees off `main`.
Consequences worth expecting:

- `main` moves during your task, and its working tree may be dirty with someone
  else's in-progress work. Never stash or revert there.
- Before regenerating a shared figure or editing `core/eval/plots.py`, check
  whether `main` already moved under you.
- Prefer merging `main` into your branch over rebasing your own if another
  session has been told to build on your commits.
