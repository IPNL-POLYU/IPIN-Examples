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

**`tests/ch7_slam` fails intermittently under load. Do not bisect it.** Five
tests each `subprocess.run` the same `ch7_slam.example_pose_graph_slam` with
`timeout=180`; the script takes ~135 s standalone, a margin of 1.33x, so any
concurrent work pushes it over. Verified as flakiness rather than inferred:
identical content gave 2 failed / 1351 passed in 954 s under load, then
1353 passed / 0 failed in 647 s when quieter. **Re-run a ch7 failure in
isolation before believing it.**

Those tests also run the example with `cwd` at the repo root, so running them
rewrites tracked files under `ch7_slam/figs/`. That is why
`slam_with_maps.png` keeps reappearing in `git status` — it is test churn, not
someone's edit.

## Figures

`core.eval.save_figure` is the only output path; it writes svg/pdf/png together
and is byte-reproducible, so a committed figure diff means the picture changed.
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
