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
invocation so it runs once per session (5 runs down to 3) and sets a timeout
generous enough to be a deadlock guard rather than a performance budget. **If
you add a test that shells out to an example, go through that runner.**

Those tests still run the example with `cwd` at the repo root, because the
example resolves `data/sim` relative to the working directory. The figures are
no longer written there, though: the runner points `IPIN_FIGS_DIR` at a
scratch directory and exposes the result as `ExampleRun.figs_dir`. **Assert
against that, never against the in-repo path** — an existence check on
`ch7_slam/figs/slam_with_maps.png` passes on the committed file whether or not
your run wrote anything.

If `slam_with_maps.png` shows up in `git status` after a test run now, that is
a real change and not the churn this note used to describe.

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
