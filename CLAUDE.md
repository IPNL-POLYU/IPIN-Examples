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

Every dataset has now been audited this way. Eighteen came back consistent;
the three defects were ch2's two above and one in ch6 strapdown, where
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

The most reusable habit from both sweeps is duller than any of the rules:
**check that the check can fail.** A new assertion, a new tolerance, a new
consistency test — run it against the broken input before believing the green.
It matters in both directions and both were hit: a Chapter 6 strapdown test
shipped ending in `pass  # Will fix in next version`, asserting nothing while
still appearing in the count, and a replacement assertion written during the
sweep to remove exactly that antipattern turned out to hold whether or not the
code under test did its job.

If a ch7 test does fail, re-run it in isolation before believing it.

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
the current debt register and should only shrink.

## Parallel sessions

Several agents often work this repo at once, on separate worktrees off `main`.
Consequences worth expecting:

- `main` moves during your task, and its working tree may be dirty with someone
  else's in-progress work. Never stash or revert there.
- Before regenerating a shared figure or editing `core/eval/plots.py`, check
  whether `main` already moved under you.
- Prefer merging `main` into your branch over rebasing your own if another
  session has been told to build on your commits.
