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

**Run it as a module and the problem does not arise.** `python -m` puts the
working directory on `sys.path[0]`, which lands ahead of the editable finder,
so from the worktree root you get the worktree's `core`:

```bash
python -m ch6_dead_reckoning.example_comparison
```

Measured against a scratch worktree carrying a marked `core/`: the script form
resolved to the main checkout, the module form to the worktree copy. Setting
`PYTHONPATH=$(pwd)` also works and is what this note used to say; it is now
redundant, because every documented command in the repository is the module
form and `tests/docs/test_documented_commands_use_module_form.py` keeps it that
way. Two problems, one fix — see the section below for the other one.

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
- **A tree block in `## File Structure`** -- a directory header written once,
  then bare filenames under it (`├── angles.py`). Structurally invisible to the
  path regex above, for the same reason as the backticked bare filename.
  `tests/docs/test_readme_file_structure.py` covers it and found **23** claims:
  five chapters omitting an example each, two `core/` modules under names they
  had lost, and sixteen entries describing dataset contents that do not exist --
  ch6's two IMU datasets documented as `time.txt`/`accel.txt`/`gyro.txt` when
  they ship `imu.npz` and `truth.npz`, and all three ch5 fingerprint datasets as
  `fingerprints.csv` when they ship three `.npy` arrays. That guard is
  exhaustive both ways over `example_*.py` (a demo missing from the list is a
  demo nobody runs) and existence-only for everything else.

  **Its parser was the risky half, not its assertions.** Three regexes in a row
  produced false findings that each looked exactly like drift: `example_[a-z_]+`
  matches inside `test_example_pose_graph_runs.py`; a character class without
  `-` truncates `ch7_prompts_1-6_COMPLETE.md` and reports a file that exists as
  missing; the same class without `*` truncates the glob `ch7_prompt*_*.md`.
  `test_the_parser_reads_the_shapes_that_fooled_it` pins all three, and mutating
  the pattern back turns it red. When a guard has to *parse* prose rather than
  execute it, test the parser separately -- otherwise a green means only that
  your reader agreed with itself.

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
(`--no-time-correction`, `--time-offset`, `--output`) that `example_tc_fusion` does
not have, and the results table reported RMSE, NIS and convergence for runs
nobody could perform. Its "offline correction" row claimed 0.08-0.12 m against
an uncorrected 0.18-0.25 m -- correction more than halving the error -- where
the real demo measures 0.211 m to 0.185 m, 12.5%. **The kinematic bound was
already written down in the code**: `example_temporal_calibration.py`'s docstring
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
  or a block opening `# In example_tc_fusion.py, add parameter:`). Not executed.
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

## `--preset` and `--output`, and the shape a one-instance fix leaves behind

A generator's preset may supply the output directory. It may not *override* one
the caller gave, and seven of them did: ch2, ch3, ch4, ch6 env/pdr/wheel_odom
and ch7 each ran `output_dir = "data/sim/..."` unconditionally inside the preset
chain. `--preset X --output somewhere` therefore regenerated the **shipped**
dataset and left `somewhere` empty -- silent and destructive in one step, since
the command appears to write where you asked.

**ch5's copy of this was found and fixed an earlier session, by overwriting
three shipped datasets while surveying them. The other six were never looked
for.** That is the lesson worth carrying: a defect found in one file is evidence
about a *shape*, and the sweep costs one grep. This one was found again the same
way -- by stashing the fix, regenerating, and watching the run write into
`data/sim` -- which is a slow way to learn something a search would have said.

The inverse hazard is real and I walked straight into it. `output_dir or "..."`
only helps when `output_dir` can be empty, and `--output` declared an argparse
default pointing at the *baseline* directory, so it never was: every preset would
have written to the baseline dataset. So the contract has two halves and needs
both:

- `--output` defaults to `None`.
- The preset chain supplies a default with `output_dir = output_dir or "..."`,
  and the module's own directory is a fallback *after* the chain.

`KNOWN_PRESET_OVERRIDES_OUTPUT` in `tests/test_repo_conventions.py` holds the
first half as an AST check and is empty. The second half it cannot see, so the
docstring says so.

## A trajectory written as independent closed forms will not join up

`ch6_env_sensors_heading_altitude` sat in `KNOWN_DISCONTINUOUS` described as "not
a pedestrian" at 51 m/s. Measured, the walk was ordinary -- mean 1.20 m/s, median
1.05 -- with **two** bad samples out of 1799, both at phase boundaries. Each
phase was an absolute closed form with nothing tying it to the one before, so
the stairwells sat at points the walk never reached: 10.1 m of teleport at t=60,
7.8 m at t=120.

**Read the distribution before believing the peak.** The register's summary was
true and still gave the wrong picture of what to fix; two teleports in an
otherwise sane walk is a different job from a trajectory that is wrong
throughout.

Two further defects were hiding under the first, and both came from asserting
the heading per phase instead of deriving it:

- The corridor was a line walked back and forth with `y` fixed, so the heading
  reversed 180 degrees in one 0.1 s sample -- **1800 deg/s, in the chapter about
  heading**. Now a flattened loop, which turns through the same 180 degrees
  continuously.
- The figure-8 set `yaw = atan2(8 cos p * 2, 4 cos 2p * 2)`. For
  `x = 10 + 8 sin p, y = 10 + 4 sin 2p` the velocity is `(8 cos p, 8 cos 2p)`,
  so the arguments were swapped and the heading was wrong for a sixth of the
  run. Deriving yaw from the velocity removes this **by construction** -- the
  trajectory and the heading can no longer disagree, because there is only one
  of them.

Result: 51.05 -> 2.37 m/s, 26 g -> 1.2 g, 1800 -> 59.8 deg/s. The barometer,
floor labels and altitude error are byte-identical, which is the check that the
change stayed inside the trajectory.

Two things that happened while fixing it are worth expecting:

- **The numbers named my own bug faster than reading did.** Splitting the loop
  in two left `altitude = z` reading the *position* loop's final value, pinning
  every sample at 7.0 m. The tells were exact: `barometer_clean` differed by
  84.09 Pa, which is the whole 0-7 m range, and the altitude error was 4.06 m,
  which is the mean distance from 7 m. An error equal to a range or a mean is
  arithmetic, not noise.
- **A guard-the-guard test caught a regression I would not have.** `atan2`
  returns `(-pi, pi]`, and storing that would have kept the true yaw wrapped --
  at which point `min(|d|, 2pi - |d|)` never goes negative here and
  `tests/ch6_dead_reckoning/test_heading_error_is_wrapped.py` would have passed
  for a broken implementation. Someone had written
  `test_the_trajectory_still_exercises_the_wrap` for exactly this, and it failed
  the moment I regenerated. `np.unwrap` restores the accumulating heading, and
  the demonstration is now stronger than before: 702 of 1800 samples go negative
  under the naive form, against 221, understating 3.41 deg as 0.70.
  **When you rewrite the data a test depends on, run that test before believing
  the rewrite is an improvement.**

## Seventeen seeds nobody was running

All twenty datasets record `seed: 42`. Until now three of them -- Chapter 5's --
had anything that checked it. `tests/test_datasets_reproduce_from_their_recipe.py`
now runs every generator into a temporary directory and compares against the
shipped bytes. **The whole sweep is about 30 seconds**, which is the first thing
worth knowing: the expensive part of this repository is the examples, not the
data, and the reason this went unguarded for so long was an assumption about
cost that nobody measured.

Nineteen of twenty reproduced immediately. The findings were in the other one and
in the machinery around it:

- **ch6 PDR's `--preset baseline` did not produce the shipped baseline.** The
  preset carried cleaner sensors -- 0.15 / 0.005 / 0.05 / 0.002 against defaults
  of 0.2 / 0.01 / 0.1 / 0.005 -- and wrote to the same directory, so the
  regeneration command *its own README offers* replaced the reference dataset
  with different data. Its two sibling ch6 generators are fine: their datasets
  record `preset=baseline` and regenerate from it exactly. The preset now
  matches the defaults, so the command in the README is true and no shipped byte
  moved.
- **ch8's presets had no directory of their own.** `--preset nlos_severe` wrote
  wherever `--output` defaulted, which was the *baseline* dataset. The sweep in
  the previous session skipped ch8 because it had no literal `output_dir =
  "data/sim/..."` to find -- the AST check looks for the shape, and ch8 had the
  same hazard in a different shape.
- **`nlos_severe` is not the shipped NLOS dataset.** It is the 1.5 m bias case;
  the shipped one is 0.8 m, as its README and `--all-variants` both say. So
  pointing the preset at that directory, which is what I did first, would have
  replaced a shipped dataset with a different scenario under the same name. It
  writes somewhere else now.

**The comparison has to be NaN-aware, and this nearly cost the whole result.**
UWB dropouts are stored as NaN -- 129 per Chapter 8 dataset. `np.abs(a - b).max()`
returns `nan` across one, and `nan > worst` is False, so a running-maximum scan
silently skips every array carrying a NaN. The first version of the survey did
exactly that and reported all three Chapter 8 datasets as byte-perfect,
**including the one that differs by 0.7 m**. A comparison that cannot see a
difference reports agreement, which is indistinguishable from success.

## Read argparse, do not run it

`tests/docs/test_documented_flags_exist.py` checks that a flag a document passes
is one the program declares. It reads the target's AST rather than running
`--help`.

That is not a performance choice, though it is also 200x faster. A `--help`
scan was written first and reported nine broken files; it shelled out without
`PYTHONIOENCODING=utf-8`, the child printed a degree sign, cp950 could not
encode it, and empty help text read as "this program declares no flags". The
AST version cannot be fooled that way -- and it found **more**, because a
program with no `ArgumentParser` at all returns "no flags" from `--help` too,
which the subprocess version could not distinguish from a crash.

That distinction turned out to be the whole finding. `example_imu_strapdown`,
`example_zupt` and `example_deterministic` were being passed `--data <dataset>`
in about twenty places across five documents. None of the three declares any
argument, and the first two do not read `data/sim` at all -- they build their own
trajectory. **`--data` was not a stale spelling; it was a capability that never
existed**, and every "run this on the three variants you just generated"
experiment built on it was unrunnable as written.

**Deleting the flag is not the fix.** Those blocks exist to compare variants, and
a mechanical strip turns three commands into three identical ones, which compares
nothing -- that is what the first attempt produced, and it is worth looking at
your own diff for exactly that shape. Each block now points at the mechanism the
reader has: the dataset README's own loading block, which takes `dataset_path`
and can be pointed anywhere.

Also found and fixed: `--alpha` on the ch8 fusion runs (the flag is
`--confidence`, and `alpha` was deprecated in favour of it, so 0.05 becomes
0.95), `--output results.json` on `example_tc_fusion` (it has `--save`, which takes
a *figure* path), `--no-correction` on `example_temporal_calibration` (it runs both
paths in one pass -- there is nothing to switch off), `--imu-grade` and
`--accel-bias` on the ch6 strapdown generator (`--preset`, and per-axis
`--accel-bias-x/-y`), and `--r-scale` on `example_tc_fusion` -- where the paragraph
directly above the block already said "modify the fusion script", so the doc
contradicted itself in adjacent lines.

**My debugging harness disagreed with the guard, and the harness was wrong.**
Twice I wrote a quick script to enumerate the findings and it reported one file
where pytest reported five. Both times the answer was to stop trusting the
scratch tool and use the guard as the oracle. That is the third harness in this
audit to misreport the thing it was pointed at.

## The first ten minutes, and the documents nobody re-reads

Walking the repository as a first-time reader — fresh clone, README top to
bottom, run the first thing it tells you to run — found four blockers before
any of the physics. All 38 runnable demos passed throughout; none of this was
broken code.

**The entry documents had the invocation form backwards.** The seven chapter
READMEs used `python -m` 71 times against 11 script uses, but the top-level
README's Quick Start and `notebooks/README.md` — the two documents a newcomer
actually lands on — used the script form throughout, and Quick Start sat
*above* the Setup section that installs the package. The script form puts the
*script's* directory on `sys.path`, so on a clone that has not been installed
it is `ModuleNotFoundError: No module named 'core'`. The three most prominent
commands in the repository were the only ones that could not work in the state
the reader was in when they read them.

Expect that shape: **the most-read documents are the least re-read.** A chapter
README gets revisited every time its chapter changes; the front page does not.
All 46 occurrences across nine documents are module form now, held by
`tests/docs/test_documented_commands_use_module_form.py`.

**Exit status is not evidence that a stage ran.** `cd`-ing into a chapter
folder used to break the twelve dataset-reading examples, because
`Path("data/sim")` is cwd-relative; `core.utils.resolve_data_path` now tries
the working directory and then the repository root. The guard over it asserts
on *output*, and has to: ch4's `--compare-geometry` prints
`Skipping <name> (not found)` and **exits 0**, so an exit-code check stays
green while the comparison compares nothing. ch2, ch3 and ch6's `--data`
handling is the same shape — a message and `return`. Same family as the
"number at chance" signature below: a stage that does nothing rarely says so.

**A grep sees the spellings you thought to look for.** `requires-python`
claimed `>=3.8` and was false — five modules annotate with PEP 585 generics and
carry no `from __future__ import annotations`, so 3.8 raises at *import* time.
Grepping for built-in generics found those five and said the floor was 3.9. The
AST check written to hold the new floor immediately added a sixth,
`core/coords/transforms.py`, annotating `NDArray[np.float64] | None` — PEP 604,
where `types.GenericAlias.__or__` arrives in **3.10**. That is the module
Chapter 2 opens with, so the package could never have run on 3.9 either. Parse,
do not grep, when the question is "what syntax does this use" — the same reason
`test_documented_flags_exist.py` reads argparse instead of running `--help`.

**A configured checker that is not run is indistinguishable from one that
passes.** mypy's own config said `python_version = "3.8"` and would have
flagged every one of those files. CI runs a single pytest job on 3.11 and never
invokes mypy, ruff, black or pylint, all four of which the README tells readers
to run. Before concluding a class of defect is covered, check the workflow file
rather than the config.

**Underselling generates no bug reports.** The chapter table advertised
Ch4 as "Eqs. 4.1-4.69" while `docs/equation_index.yml` maps to 4.108 — hiding
the AOA closed-form solvers and the whole GDOP/HDOP/VDOP family. Ch2, Ch5 and
Ch8 were narrow too. A reader told something is absent does not go looking for
it, so this can stay wrong indefinitely without a single complaint, which is
the opposite of the failure modes the rest of this file records.
`tests/docs/test_chapter_table_equation_ranges.py` recomputes each span.

Two things left deliberately undone, because they are judgement calls rather
than defects: **Chapter 8 has nine runnable demos and only one is named
`example_*`**, so a reader who learned the pattern from ch2-ch7 finds almost
nothing there; and **`plt.show()` is in 12 of 31 examples and absent from 19**,
undocumented, blocking under a GUI backend and warning under Agg. Also, 17 of
31 examples have no `argparse`, so `--help` runs the whole demo.

## Four guards that measurement killed, and why that was the right answer

PR #66 fixed equation-number drift in four notebooks by hand: ch2's intro
claimed Eq. (2.1)-(2.10) mapped sequentially to LLH/ECEF/ENU/rotations, when
`docs/equation_index.yml` and the chapter README had carried the corrected
numbers since an earlier audit -- LLH/ECEF is (2.9). The obvious follow-up is a
guard so it cannot drift again. **Four designs, four rejected by measurement.**
They are written down here because each one is the first thing the next person
will think of, and each takes an hour to disprove:

- **Every `Eq. (N.M)` cited anywhere must exist in the index.** The invariant
  itself is wrong: the index maps equations *to code*, not to the book, and ch3
  has no 3.5, 3.6, 3.7, 3.10 or 3.13. Thirteen files legitimately cite
  unimplemented equations, and this guard calls every one of them drift.
- **Within a cell, a cited equation must match the indexed object called
  there.** Can never fire: equations live in markdown cells and calls in code
  cells, so **0** such pairs exist.
- **Within a notebook, calling an indexed object requires citing its
  equation.** 68% noise -- 17 of 25 called objects have no equation cited, and
  that is normal prose.
- **A symbol named in notebook prose must exist.** Vacuous: notebook markdown
  contains **0** backticked identifiers.

The first is the instructive one. I proposed it before measuring, and it would
have turned thirteen honest documents red. **Check what the reference data
actually covers before asserting things against it** -- `equation_index.yml` is
a code map with deliberate gaps, not a table of contents, and nothing about its
name says so.

**The coupling people worry about is already guarded.** "If we change the Python,
do the notebooks follow?" -- yes, `tests/docs/test_notebooks_run.py` executes all
seven in a real kernel and asserts no cell produced an `error` output. Verified
by mutation rather than by reading: adding a required argument to
`core.coords.rotations.euler_to_quat` turns the ch2 notebook red in 8.7 seconds.
What is *not* guarded is notebook prose -- equation numbers, described behaviour
-- and that is the same class as the ` ```bash ` paths in the section above,
except that here no precise invariant exists to check it against.

So the useful distinction, and it is worth keeping: **an API coupling is a gate,
a prose claim about numbering is a rule.** A guard built anyway would land in one
of two states -- never firing, or noisy enough that someone turns it off -- and
both are worse than the rule, because both look like coverage.

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

`tests/test_datasets_reproduce_from_their_recipe.py` compared
regenerated arrays with `np.array_equal` (as the Chapter 5 test it grew out of
did). It passed locally, it passed on CI,
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


## The library used to own the randomness, and the recommended fix broke it

Six functions in `core/` drew straight from numpy's global RNG: the two
`simulate_*_measurement` in `core/rf`, the two scan generators in `core/slam`,
and `ParticleFilter.__init__` and `_resample`. Eleven draws.

Nothing was broken. Every example that reaches them calls `np.random.seed(...)`,
which is exactly why the committed figures reproduce. **The hazard is that the
repair this repo's own convention check recommends would have broken them:**
`test_examples_do_not_draw_from_the_unseeded_global_rng` says "prefer threading
an explicit `rng = np.random.default_rng(seed)`", and a local Generator does not
cover a library's global draws. Modernise an example that way and its figures
quietly stop reproducing, having followed the advice.

That is the shape worth recognising: **a latent defect whose trigger is someone
doing the right thing.** It cannot be found by looking for wrong code, because
none of it is wrong yet.

Each function now takes `rng`, defaulting to `np.random` itself:

```python
def _rng(rng):
    return np.random if rng is None else rng
```

**The default has to be `np.random` and not `default_rng()`.** A fresh Generator
ignores `np.random.seed`, so switching the default would have moved every figure
in the repository. Verified rather than assumed: the Chapter 3 and Chapter 4
figures regenerate byte-identically after the change, which is the whole
argument that this was refactoring and not editing.

One mechanical detail with a trap in it: `Generator` has no `randn`, so those
sites became `standard_normal`. The two draw **the same values from the same
seeded stream** -- checked directly, not inferred from the docs -- so the swap
is free. Had they differed, this change would have silently rewritten seven RSS
draws.

Two guards, because one is not enough and it is worth knowing why:

- `KNOWN_GLOBAL_RNG_IN_CORE` in `tests/test_repo_conventions.py` is an AST check
  that no library function calls `np.random.<draw>` at all. It sees *shape*.
- `tests/core/test_rng_injection_is_honoured.py` sees *behaviour*, because a
  function can accept `rng`, ignore it, and satisfy the AST check perfectly.

**Which assertion catches an ignored parameter was established by mutation, and
it was not the one I expected.** Making `_rng` return `np.random`
unconditionally is caught by "the same Generator gives the same draws whatever
`np.random.seed` says" -- not by "the injected result differs from the default",
which was written first and deleted. Two consecutive draws from the global
stream differ whether or not anyone is listening to the argument, so that
assertion discriminated nothing while reading like the main event. Run the
mutation before deciding which of your assertions is load-bearing; the
convincing-sounding one is not reliably the one doing the work.

## An RMSE over the solves that converged hides the failure twice

Chapter 4's `--compare-geometry` aggregated each method as an RMSE over the
solves reporting `converged`, and that single choice produced three wrong
answers at once. Nonsense that claimed success was averaged **in** -- AOA on the
collinear beacons reported 2.2e10 m, from three fixes at 1e11 m among 95
"converged" ones. Honest refusals dropped **out** of the denominator, so TOA
and TDOA on that geometry printed no row and drew no bar at all. And the third
was quieter than either: `errors[:n]` was paired against `gdop[:n]` with the
errors already compacted to the successes, so point i's error sat beside some
other point's GDOP the moment anything failed.

Net effect: **no method appeared on more than two of the three geometries**, in
the mode whose entire purpose is comparing methods across geometries.

- **The correct version was next door.** `030-figures-and-claims.mdc` already
  said "report the median and the failure count separately" and already listed
  the fourth condition people forget (never left the initial guess), and
  `generate_ch4_rf_2d_positioning_dataset.py` already implemented all of it --
  it is where every `failed_count` in every ch4 `config.json` comes from. The
  example simply never adopted it. It is now `core.rf.solve_batch`, used by
  both; the extraction was verified by regenerating all four ch4 datasets and
  diffing **every byte, `config.json` included**, which is the check that turns
  "should be equivalent" into "is".
- **A rule that is not reachable as a function gets applied where someone
  happened to look.** The chapter's *inline table* had this exact defect fixed
  once before -- CLAUDE.md records "the comparison table reported AOA at 5.3e9 m
  with zero noise". The dataset path kept it, for the reason the figure ratchet
  exists: nobody had ever committed or opened `ch4_geometry_comparison.png`.
- **On a linear axis one absurd bar is the whole figure.** 2.2e10 m flattened
  every other bar to zero height. The honest numbers span 0.08 m to 14 m, so
  the axis is logarithmic, the height is a median, and the failure rate is its
  own panel -- because an accuracy plot cannot express "this did not work",
  which is the thing it most needs to say.

**The thread under it was worth more than the figure.** Asking CLAUDE.md's
standard question of the surviving numbers -- *what should this be?* -- TDOA on
the square array has GDOP 0.87 and 0.1 m of range noise, so `sigma_position =
GDOP x sigma_range` predicts 0.087 m. It reported **13.75 m**. The generator
builds each measurement as `tdoa_range_difference(beacons[0], beacons[j], pos)`
= `d_ref - d_k`, while `TDOAPositioner` predicts `d_k - d_ref`: every shipped
TDOA measurement is negated. Negate them back and the square array gives
0.074 m with 0 failures instead of 13.753 m with 11. A factor of **158**,
across every ch4 dataset, in the deliverable of a chapter about RF positioning.

- **Three documents had already met it and explained it away.** The square
  dataset README calls TDOA "the fragile one" and attributes the gap to "the
  hyperbolic geometry"; its troubleshooting section carries an entry whose
  stated symptom -- ">10m errors while TOA gives <0.5m" -- *is* the defect.
  Third instance of this shape after ch2's "Issue 2: ENU Range Seems Wrong" and
  its 360-degree rotation error. **A troubleshooting entry describing your own
  output is a bug report.**
- **The collinear dataset could not see it.** Both signs stall at the beacon
  centroid, 100/100, so the one geometry a reader would suspect is the one that
  looks identical either way. Pick the well-conditioned case to test a sign.
- **Fixed in a follow-up change**, which is what the pinning test was for: it
  went red on all six assertions the moment the argument order was swapped, and
  was then deleted. Regenerating from each dataset's recorded preset touched
  `tdoa_diffs.txt` in all four and the TDOA block of three `config.json`s and
  **nothing else** -- TOA, AOA and every GDOP file byte-identical, which is the
  check that the sign flip did not disturb the RNG stream. Square and NLOS now
  give 0.075 m with 0 failures, optimal 0.085 m; the collinear variant is
  unchanged at 6.770 m / 100 failures, exactly as predicted.
- **Nothing caught it because nothing read the shipped bytes.** Chapter 4 had
  two good accuracy tests on either side of this -- `test_toa_attains_the_dop_bound`
  and `test_tdoa_error_scales_linearly` -- and both *synthesise their own
  measurements* from the correct convention before solving. They characterise
  the solvers, which were never wrong. Same blind spot as the figure defects:
  a test that builds its own input cannot see a defect in the stored input, just
  as a test that checks a figure was written cannot see what it depicts.
  `tests/ch4_rf_point_positioning/test_shipped_measurements_match_the_solver_convention.py`
  is the durable replacement, and it compares each stored measurement file
  against the forward model rather than against a solved position -- so a
  solver, seed or tolerance change cannot move it, because both sides are
  geometry. It covers TOA and AOA too, which were correct and cost nothing to
  pin, and carries a test that re-measures its own tolerance against both the
  noise and the defect on every run, rather than trusting a number written once.
  Every arm's margin is measured against six named corruptions.
- **The first version of that guard was itself broken, and the mutations I chose
  to validate it were the ones it passes.** This is the antipattern CLAUDE.md
  already warns about -- "a replacement assertion written during the sweep to
  remove exactly that antipattern turned out to hold whether or not the code
  under test did its job" -- arriving one level up, in the guard rather than in
  the code. Reviewing the diff caught it. Two statistics were wrong:
  - **A signed mean cancels.** Negating *every* azimuth in the square dataset
    moves the signed mean residual from 0.17 deg to 0.17 deg, so a fully
    sign-inverted AOA file passed. Per column the same defect is 90.15 deg.
    Swapping `atan2`'s two arguments was missed the same way. And a signed mean
    fails on TDOA even *per column* -- 0.12 sigma either way -- because
    `d_j - d_ref` averages to nothing over a symmetric grid.
  - **A mean over all columns dilutes.** One beacon is one of four, so an
    undeclared +1.1 m bias on a single beacon -- larger than the 0.8 m the NLOS
    dataset legitimately ships -- read 0.280 m against a 0.300 m gate.
  The right statistic is the **worst column's mean |residual|**: absolute so it
  cannot cancel, per column so it cannot dilute. Honest values land at
  0.81-0.85 sigma for all three measurement types, which is `sqrt(2/pi)` = 0.798,
  the mean absolute deviation of the noise itself -- so the residual is all noise
  and no systematic part, a stronger statement than "under the threshold".
- **Pick the mutation that is hardest for your statistic, not the one that comes
  to mind.** Of three AOA convention errors, the reverse bearing (+pi) is the
  only one a signed mean detects, and it is the one I reached for first. The
  general form: a defect that is *antisymmetric* about the array defeats a
  signed reduction, and a defect confined to *one* sensor defeats a reduction
  over all of them. Enumerate corruptions along both axes before believing a
  green. The tolerance test now carries all six as data rather than as prose.
- **Correcting the data falsified a claim written about the broken data, and it
  did not read like a number.** `data/sim/ch4_rf_2d_linear/README.md` said
  "TDOA fails from every starting point tried", and explained it with flat
  hyperbolae and a GDOP of 10.36 -- physics, not a measurement, so it survived
  every numeric sweep. On the corrected file TDOA solves **83 of 100** from an
  off-line seed. Half the sentence was right and the half that was wrong was
  the half that sounded like a conclusion. Its loop evaluated only TOA and AOA
  and described TDOA in prose; it evaluates all three now, which is the general
  fix -- **when a defect is fixed, re-run the claims that were measured against
  it, and prefer to put the third method in the loop over describing it.**
- **`--compare-geometry` now prints an emptier failure panel, and that is
  correct.** With TDOA no longer failing 11 and 13 times, two of the three
  geometry groups show zero failures. A panel with nothing in it looks like
  missing data and is in fact the result.

**The `iwls` question next door was not a defect, and was still worth
changing.** `run_positioning` used to solve TOA with
`TOAPositioner(beacons, method="iwls")`, a deprecated alias resolving to
`range_weighted` (W_ii = 1/d_i^2) rather than to iterative WLS. That reads like
drift and was not: at the commit that created these datasets `iwls` *was* the
1/d^2 branch, and the alias was added later specifically to preserve it, so the
behaviour had never changed.

**The argument for changing it is the data, not the name.** `range_weighted`
assumes sigma_i proportional to d_i. These generators add
`rng.normal(0, toa_noise)` with one fixed std, so that assumption is false here
and uniform weights are the maximum-likelihood choice -- which is also
Eq. (4.20), the book default. Measured both ways before switching:
`range_weighted` vs `iterative_ls` gives 0.0951 vs 0.0881 m on the square,
0.0995 vs 0.0792 on optimal, identical 6.7696 on the collinear, and 0.6030 vs
0.6145 on NLOS with failures dropping 4 to 1. Uniform weights win on both
well-conditioned arrays, as BLUE says they must for constant noise.

**The tell that it needed resolving was two files disagreeing about the same
measurements.** `example_comparison.solve_every_method` already used
`iterative_ls`, so it reported the square's TOA median as 0.088 m while the
`config.json` beside it recorded 0.095 m -- and the example carried a docstring
paragraph *explaining the discrepancy* rather than removing it. When a
codebase documents why two of its own numbers disagree, that note is a
deferred decision, not an explanation. They agree now and the paragraph is
gone.

Blast radius, for calibration: only the TOA block of three `config.json`s. No
measurement file moved, because the estimator affects what is *reported*, not
what is *measured* -- and `ch4_rf_2d_linear` did not change at all, since
100/100 fixes stall at the centroid under either weighting.

**"I-WLS" is a label three Chapter 4 demos apply to two estimators that are not
it.** Chasing the alias turned up the same confusion in the printed output.
There are three weightings and the difference is one argument:

- `TOAPositioner(method="iterative_ls")` and `TDOAPositioner.solve()` **without**
  a `covariance` are W = I: iterative *LS*, Eq. (4.20) / (4.34)-(4.41).
- `.solve(..., covariance=cov)` is genuinely I-WLS, W = Sigma^-1, Eq. (4.23).
- `method="range_weighted"` is W_ii = 1/d_i^2, a heuristic the library's own
  docstring flags as "NOT from the book".

`example_tdoa_positioning` printed all three under "I-WLS" -- including inside a
single demo, where Chan's perfect-measurement run is unweighted and its Monte
Carlo beside it does pass a covariance. The labels now name what runs, and
`method='iwls'` is spelled `method='range_weighted'`, which is **bit-identical**
because that is literally what the alias maps to. Verified the honest way: every
number in the demo's 242 lines of output is unchanged, only labels moved.

**The other two files carried the same defect and are fixed too.**
`example_aoa_positioning` constructed `AOAPositioner(anchors)` at all seven call
sites with no `sigma_*` and then said "I-WLS" fourteen times;
`example_toa_positioning` solves with `method="iterative_ls"` throughout while
its module docstring claimed "using Iterative Weighted Least Squares".

**The AOA case needed measuring, not reading, because "uniform weights" does not
obviously mean "unweighted".** It does here: a uniform W is a multiple of the
identity, so it cancels out of `(H' W H)^-1 H' W`. Confirmed by solving the same
bearings three ways -- no sigma, a scalar sigma, and a per-anchor sigma. The
first two are **bit-identical**; only the third moves the answer. So the seven
call sites are the Eq. (4.63)-(4.78) solver run unweighted, and `--compare-geometry`
is the one place in the chapter that supplies a per-anchor sigma and earns the W.

Same verification as before: every number in both examples' output is unchanged,
and both committed figures are byte-identical, because neither carried the label
in a tick or an axis. Note what that leaves -- `example_toa_positioning`'s stdout
is the transcript pinned in the chapter README, and it did **not** move, because
the wrong label lived only in the module docstring. A pinned transcript is not a
guard against a mislabelled docstring; nothing here is.

## A success rate can measure your harness instead of your method

`example_comparison` injected a shared 1.5 m receiver clock bias into the TOA
pseudoranges -- correctly, to show that TDOA differences it away -- and then
solved them with a **position-only** `(x, y)` state. A common bias is
unobservable to that state: no position makes four uniformly inflated ranges
consistent, so the residual never reaches `tol` and the solve is discarded. The
figure's "Convergence Success Rate" panel therefore read **2-5 of 100** for TOA
at every noise level, next to a median-error table where TOA was among the best.
Both cannot be true of a method, and the panel was the one lying: it was
reporting the survival rate of a model mismatch.

- **The tell was the noiseless row.** TOA printed 0.153 m of median error at
  zero measurement noise, where TDOA and AOA both printed 0.000. A method
  solving perfect data is exact, so a nonzero number there is never noise --
  it is a model that cannot represent the data it was handed. Check the
  zero-noise row first; it is the one row with a known right answer.
- **"The survivors are accurate" was wrong, and worth disbelieving on sight.**
  The few solves that converged were the geometries where the bias could be
  partly absorbed *into the position* -- so they were the least-inaccurate, not
  the accurate. When a filter passes 4% of cases, ask what those 4% have in
  common before treating them as a clean sample.
- **The fix was already exported.** `toa_solve_with_clock_bias` is
  Eqs. (4.24)-(4.26), lives in `core/rf/positioning.py`, and is in `core.rf`'s
  `__all__`. The example injected a clock bias and then declined to use the
  solver written for it, one import away. Same shape as the ch5 and ch7 library
  bugs: look for the sibling that does it right before designing anything.
  With it: 100/100 converge at every level, the bias returns as +1.500 m on
  noiseless data, and TOA tracks TDOA at the small cost of the extra unknown.

**Fixing it made TOA invisible, which is the same defect mirrored.** Three of
the four methods then sat on 100%, and four solid lines at one value show only
the last one drawn -- so the panel that used to say "TOA barely converges" would
have said "TOA is missing". Distinct dash patterns per method, no nudging of
values. Whenever a fix moves several series onto the same constant, re-open the
figure: the overlap is new and it is not visible in any number.

**My own probe reproduced the bug while measuring it.** Checking the fix, I
called `toa_solve_with_clock_bias` expecting two return values where it returns
three -- `(position, bias, info)` -- inside a `try/except Exception: pass`, and
got a confident `0/100 converged`. That is the issue's own defect one level up:
a blanket except turning a caller error into a result about the callee. Fourth
harness in this file to report the thing it could not read as broken.
## Every example now bootstraps its own sys.path, and the sweep took three tries

The trap at the top of this file -- `python chX/example.py` resolving `core` to
the main checkout -- is closed at the source now: all 38 examples insert the
repository root before their first `core` import, matching what
`ch5_fingerprinting/example_classification` and nine of twelve `scripts/`
generators already did. `tests/test_examples_import_this_checkout.py` holds it,
and checks *order* rather than presence, because a bootstrap below the import
changes nothing.

Two things worth knowing before repeating this kind of sweep.

**Three of the three bugs in the sweep script were found by pyflakes, none by
reading.** Each was a plausible-looking way to locate an insertion point:

1. `ast.walk` to decide whether `sys` was already imported -- it counts an
   `import sys` inside a *function*, so four files got a module-level
   `sys.path.insert` with no `sys` bound.
2. Finding the stdlib group with `line.startswith(("import ", "from "))` --
   which matched a line of *module docstring prose* beginning "from range
   measurements ...", and wrote `import sys` inside the docstring.
3. Treating any module-level import of a name as sufficient -- ch7's pose-graph
   example imports `pathlib.Path` three lines *below* its first `core` import,
   so the name existed but not yet at the point the bootstrap runs.

All three produce files that `compileall` accepts. The rule from the earlier
lint sweep holds and is worth restating in the stronger form: **a tool that
locates Python by line prefix cannot tell an import from prose that starts like
one, and `ast.walk` cannot tell module scope from function scope.** Use
`tree.body`, and compare line numbers.

**The import-order ratchet fired, correctly, and fixing it improved the
baseline.** Inserting `import sys` at the top of a stdlib group is unsorted, so
I001 went 113 -> 147. `ruff --select I001 --fix` over the chapter directories
cleared 63, including 29 that predated this change, leaving **84** -- so the
baseline moved down, not up. E402 does *not* fire: ruff exempts imports that
follow a `sys.path` manipulation, which is what makes this idiom viable at all.

Verification was the cheap strong one: `--help` for all 38 examples, captured
before and diffed after, **byte-identical both times** -- once after the
insertions and again after ruff reordered the imports. `--help` exits during
argument parsing but only after every module-level import has run, so it tests
exactly what this change touches, in about a second per example.

## A scratch probe in the scratchpad imports the *other* checkout

The editable-install trap at the top of this file has a second face that the
`python -m` rule does not cover. A throwaway script written to
`.../Temp/claude/probe.py` cannot be run as a module from the worktree, so
`sys.path[0]` is the temp directory, `core` resolves through the editable
finder to `C:\Users\qmohs\IPIN-Examples`, and the probe measures the **main
checkout** while you read its output as evidence about your branch.

A helper you just *added* fails loudly with `ImportError`, which is how this
was noticed. A function you just *changed* does not: it runs the old one and
returns a plausible number. Two probes in this session ran that way before the
third failed and gave it away.

`PYTHONPATH=$(pwd)` in front of the command fixes it, and is the one place that
spelling is still needed -- everything documented in the repository is the
module form, which is why the note above says the variable is redundant. It is
redundant for the repository's own commands, not for yours.

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

## A figure is byte-reproducible on one machine, not across two

Do not build a check that regenerates a figure and compares it against the
committed bytes. It passes locally and it cannot pass on CI. Measured on the
Ubuntu runner against Chapter 8's committed set: **all 27 files differ.** svg by
about 0.1%, pdf by 2-4%, png by **5-27%** — 252546 bytes against 321177 for one
of them. That is font metrics and rasterisation, not a changed picture.

This is the figure form of the float-equality rule above: exact comparison is
right for **stored** bytes (save then load), wrong for a fresh computation
compared against what some other machine produced. Rendering is a computation.

**It hung CI three times rather than failing, and that part is worth carrying
beyond figures.** pytest builds `bytes == bytes` failure diffs element by
element, so a mismatch on a 300 KB PNG is not a failure report but an
astronomical one: three jobs were cancelled at the 45-minute limit having
printed `.s` and nothing further. **If you assert equality on large binary
blobs, compare digests** — a failure then reads as two short hex strings.

What survives is portable and is the half that earned itself: every committed
figure must still be produced by some demo, and every produced figure must be
committed. That found four PNGs in `ch8_sensor_fusion/figs/` that no code
writes, then caught a mechanical rename moving three demos' default figure paths
— via `is_file()`, not via bytes — and then, widened to every chapter in
`tests/test_every_figure_has_a_demo_behind_it.py`, three more orphans:
`ch6_dead_reckoning/figs/strapdown_trajectory.svg` and
`zupt_trajectory_stance.svg`, both still *displayed* in the chapter README so
readers saw a picture the code had stopped producing, and
`ch7_slam/figs/pose_graph_slam_results.png`.

**An incomplete svg/pdf/png set is the tell, and it costs nothing to look for.**
`save_figure` writes all three together, so a lone `.svg` or `.png` was made by
something else. All seven orphans had that shape, and an old mtime beside their
chapter's live figures.

Where a real figure needs a flag to appear, run the flag rather than excusing
the file — ch6's Allan component plots (`--debug`) and PDR dataset panel
(`--data`) are fifteen files that a plain run does not write and that are
perfectly real. That leaves the exemption list holding animations only.

To check that a change left the pictures alone, do what this file already says:
regenerate on your own machine, read `git status`, open the PNGs.

## When CI hangs and your machine does not, measure on CI

Narrowing that one took six rounds because each hypothesis was cheap to state
and expensive to test. Three died: the demos are fine there (all eight exit 0 in
3-8 s), `run_example` is fine (the same eight, 36 s), and buffering was not
hiding the location — with `PYTHONUNBUFFERED` the run stopped in the same place,
so the last line printed really was where it was.

Two things make this cheap next time, and both are in the workflow now:
`PYTHONUNBUFFERED`, so a cancelled run's last line means what it looks like, and
`--durations=25`. Beyond that, a **temporary diagnostic step** that measures the
suspect directly — before the suite, with `continue-on-error` and its own
`timeout-minutes` — answers in minutes where a cancelled pytest run answers
nothing. Skipping the suite step with `if: false` for one such run is worth it.
Note that the job's shell runs with `-e`, so `cmd; rc=$?` does **not** survive a
non-zero `cmd`; the step exits first.

What finally named it was two characters of pytest progress, `.s`: the first
test passed, so every subprocess had completed, and the stop was in the byte
comparison after them.

## A convention is what a repo-wide sweep selects on

`tests/test_example_console_encoding.py` globs `ch*/example_*.py`. Chapter 8
named seven of its eight runnable demos after what they do rather than what they
are, so that sweep had been seeing **one** of them. Renaming the other seven
into scope turned it red at once, on two real defects — `'²'` and `'•'` inside
`print()`, which raise `UnicodeEncodeError` on a default Windows or Japanese
console.

So the argument for the naming convention is not tidiness. **A file outside the
convention is a file nothing sweeps**, and the sweeps here are most of the
safety net. `KNOWN_NON_EXAMPLE_CHAPTER_FILES` and `KNOWN_INTRA_CHAPTER_IMPORTS`
in `tests/test_repo_conventions.py` hold both halves of it — chapter
directories contain only `example_*.py`, and an example imports `core`, never
the example next door. Both are empty.

**`functools.lru_cache` does not cache exceptions.** A memoised helper that
`assert`s inside itself re-runs its whole body for every test that calls it once
anything goes wrong — 28 parametrised tests × eight subprocesses, in the
instance that taught this. Collect failures and return them, then assert in a
separate test, so a failure is reported once and the rest skip.

## What an example does before it does the work

Two things a reader hits on every example were decided per-file for years, so
they disagreed roughly half and half:

- **`--help` ran the whole demonstration** in 17 of 38, because those had no
  `ArgumentParser` at all and the flag was simply ignored. It is the first thing
  anyone types at an unfamiliar program. Each of the 17 now parses before doing
  any work, spelled `description=__doc__` with
  `RawDescriptionHelpFormatter` — three lines, and `--help` becomes genuinely
  useful, because the module docstrings here already list what the example shows
  and which equations it implements.
- **`plt.show()` was in 19 of 38**, undocumented either way. Under a GUI backend
  it *blocks* until the window is closed, which is what made the first usability
  walkthrough of this repo spend minutes believing an example had hung. It is
  one decision now, in `core.eval.show_figures_if_requested`, reading
  `IPIN_SHOW_FIGURES` and off by default. The figures are saved regardless.

`tests/test_examples_answer_help.py` holds the first and
`KNOWN_DIRECT_PLT_SHOW` in `tests/test_repo_conventions.py` the second; both are
empty.

**The `--help` check is behavioural on purpose.** The structural version — "the
module builds an `ArgumentParser`" — is satisfied by a parser constructed at the
*end* of `main()`, after the work and the figures. So it runs the process and
requires a usage line and an empty `IPIN_FIGS_DIR`. About a second per example,
since a `--help` run exits during argument parsing.

Two things that sweep turned up, both familiar shapes:

- **A grep sees the spellings you thought to look for**, again. Searching for
  the literal `plt.show()` missed two Chapter 7 examples calling
  `plt.show(block=False)` behind hand-rolled `DISPLAY`/`MPLBACKEND` sniffing and
  a bare `except: pass`. The AST check found them immediately. Same lesson as
  the PEP 585 floor and `test_documented_flags_exist`.
- **`example_allan_variance` parsed `'--debug' in sys.argv` by hand**, which
  accepted its own flag while ignoring `--help` — and would have accepted
  `--debug` anywhere, including as the value of another option. Adding a strict
  parser to a file like that breaks the undocumented flag unless you look, so
  check for `sys.argv` before adding one.

## A diagram nobody can regenerate is a claim nothing checks

`docs/architecture/` held 34 files -- a PlantUML source and a rendered SVG for a
component view and an execution flow, per chapter, plus a repository overview.
Every chapter README embedded two of them. They are gone; the chapter READMEs
now generate their own Architecture section from the AST, via
`tools/chapter_dependencies.py` and `python -m tools.chapter_dependencies`, held
by `tests/docs/test_chapter_architecture_sections.py`.

The complaint that started it was that the lines were unreadable, and measuring
it split one complaint into **two unrelated defects** that would have had
different fixes:

- **The eight flow diagrams were not tangled, they were microscopic.** Canvases
  up to 3436 x 598 px scaled into a GitHub README column of about 830 px put
  their body text at **2.4 to 4.9 px**; six of eight were under 5 px. ch8's had
  zero edge crossings and was still unreadable.
- **The nine component diagrams were legible and genuinely tangled.** ch6 crossed
  its own edges **64 times across 31 edges**, ch5 36 across 28. And 45-67% of
  those edges carried no information: README to every example, every example to
  `figs/`, and in ch5 an arrow from each PNG to the `figs/` folder that contains
  it -- containment drawn as an edge.

**The presentation was the least of it.** Chasing why the pictures disagreed
with each other found that the `.puml` files were never the source of the
`.svg` files: the SVGs say `Generated by graphviz version 2.43.0`, `dot` is
installed nowhere here and CI has never run it, and the two disagreed on the
text they drew -- ch6's `.puml` labelled an edge `save plots` where its SVG said
`write`. So the READMEs pointed readers at a "source" that could not produce the
picture above it, and nothing in the suite could see inside either one.

Left unchecked, they had drifted into being wrong rather than merely stale. Six
of seven chapter diagrams named files that no longer existed or omitted examples
that did, and **every one of Chapter 8's nine nodes carried a pre-rename
filename**. The worst was not a name: ch6's drew **five `--data` arrows into
examples that cannot load a dataset at all** -- only `example_pdr` declares the
flag. Same shape as the `--data` drift under "Read argparse, do not run it": not
a stale spelling, a capability that never existed.

Two things worth carrying beyond diagrams:

- **A table beats a graph for a lookup.** The per-example detail those 31 edges
  encoded is seven rows of `example -> core modules -> dataset`. It reflows at
  any column width, diffs line by line, and a generator can produce it. What
  stayed a picture is the four-box pipeline every chapter shares, which is the
  part that is actually shaped like a graph.
- **A guard over a picture can only see the text you put in the picture.** The
  first version of the path check passed a deliberately broken label, because
  the diagram wrote package names bare (`estimators/`) while the regex anchored
  on `core/`. Rendering them fully qualified made them checkable *and* clearer
  to the reader. Five mutations were run before believing the green -- a spurious
  import, a broken `core/` path, a broken `data/sim/` path, a typo'd diagram
  type, and a resurrected `docs/architecture/` -- and only two of the five fired
  against the first draft.

## Lint

pyflakes is clean across the whole repo and `test_no_pyflakes_warnings` in
`tests/test_repo_conventions.py` keeps it that way, over five areas — `core/`,
`scripts/`, `tests/`, `tools/` and the chapter directories, just under 300 files
today. It runs them through the pyflakes API rather than spawning a subprocess
per file, and the reason is worth the sentence: a spawn costs about 0.85 s on
this machine, so the subprocess form would take **roughly four minutes** where
the API takes **about six seconds**. Syntax errors count as failures too.
`pyflakes` is declared in the `dev` extra rather than inherited from
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

### The other four linters: black passes now, mypy is the gap

pyflakes is the exception, not the rule. The README used to tell readers to run
`black .`, `ruff check .`, `mypy .` and `pylint`, as though the repository
passed. Measured over `core/`, the chapters, `scripts/`, `tools/` and `tests/`:
**ruff 5836, black 237 of 288 files, mypy 404 errors in `core/` alone.** A
reader following that section got thousands of complaints and could only
conclude they had broken something. The README says what is true now, which is
the cheap half of the fix and the one that had to happen first.

**83% of ruff's number was whitespace.** 4737 W293 (a blank line containing
spaces) plus 131 W291. `ruff --fix` cleared 3961 of them and every changed line
was checked to differ only by trailing whitespace before the change was
believed — 3961 removed lines against 3961 added, zero differing by anything
else. That took the count to 1879 at no semantic risk.

**Ruff refused the other 907 and was right to** — they sit inside string
literals, where whitespace is content rather than layout. Reaching for
`--unsafe-fixes` there would have been wrong.

**Black cleared 889 of those 907, and that is the interesting part.** Black
knows which triple-quoted strings are *docstrings* and normalises those, where
ruff could only see a string and had to stop. So the answer to a tool declining
an unsafe fix was a tool that could tell the difference, not overriding the
first one. The twelve that survive both are in argparse `epilog=` strings, which
are not docstrings and whose blank lines get printed.

Running black took ruff from 1879 to **951** and made `black --check` pass on
all 299 files. What remains is mostly not lint: 727 are
`List[int]`-for-`list[int]` modernisations that only became legal when the floor
moved to 3.10. The ~140 after that are the ones with content, and **B905 is the
group to read first** — 41 `zip()` calls with no `strict=`, which truncate to
the shorter argument without saying so.

`tests/test_lint_debt_only_shrinks.py` records the count **per rule**, not as a
total: a total lets ten fixed W293 pay for ten new B905, which is the opposite
of what a ratchet is for. It fails in both directions — a rule that grows, and a
baseline left above the real count, because a stale number hides the debt it
exists to expose.

**mypy is now the honest remaining gap**, at 406 errors in `core/` alone, and
it is the one of the four that would be red on arrival and stay red. Nothing
about the black run touched it; formatting does not change types.

The black run itself was 241 files, and it changed `.py` only — no figure or
data byte moved. **The full suite reads 3404 passed / 21 skipped on both sides
of it**, measured on trees verified identical beforehand. What made that safe to
*believe* was not the argument that black is a formatter and therefore harmless;
it was that the README transcripts and the figure gate would have said
otherwise if it were not.

## Parallel sessions

Several agents often work this repo at once, on separate worktrees off `main`.
Consequences worth expecting:

- `main` moves during your task, and its working tree may be dirty with someone
  else's in-progress work. Never stash or revert there.
- Before regenerating a shared figure or editing `core/eval/plots.py`, check
  whether `main` already moved under you.
- Prefer merging `main` into your branch over rebasing your own if another
  session has been told to build on your commits.
