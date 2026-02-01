---
name: bug-triage-evidence
description: Triage bugs and package reproducible evidence cleanly (.dev artifacts, chapter figures, and README updates). Use when fixing a bug or mismatch vs expected results.
---

# Bug triage & evidence packaging

Use this skill when:
- A simulation result looks wrong, plots mismatch expected figures, or a test fails.
- You need to hand off a clear bug report to another engineer.

## Workflow

1) **Reproduce**
- Write the minimal reproduction steps (command, script entrypoint, seed if relevant).
- Capture expected vs actual behavior in 1–3 bullets.

2) **Create a bug bundle in `.dev/`**
Create a folder:
- `.dev/bug-<short-slug>/`
Include:
- `README.md` containing:
  - symptoms
  - reproduction steps
  - expected vs actual
  - suspected root cause (if any)
  - fix summary + verification steps
- Any logs / short diffs / notes needed to validate.

3) **Figures**
- If the bug is chapter-specific: save plots in `chX_*/figs/`
- If it's general debugging evidence: also copy or link into `.dev/bug-<slug>/`

4) **Update chapter README when relevant**
If the bug affects a `chX_*/README.md`:
- Add a short student-centric note:
  - what was wrong
  - what was fixed
  - how to run the corrected example
  - reference the figure files you saved in `chX_*/figs/`

5) **Verification**
- Add/adjust tests under `tests/` to prevent regression.
- Re-run the minimal reproduction + full test suite if reasonable.

## Output standard
A reviewer should be able to:
- reproduce in <5 minutes
- see the mismatch in one figure
- see the fix rationale in one page
