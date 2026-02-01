---
name: python-quality-gate
description: Run the Python quality gate (format, lint, types, tests) and iterate until clean. Use after implementing or refactoring Python code.
---

# Python quality gate workflow

Use this skill when:
- You changed Python code and want it to pass formatting/lint/type/tests.
- You're preparing a PR/release and need a consistent "done" checklist.

## Steps

1) **Locate project tooling config**
- Check for `pyproject.toml`, `ruff.toml`, `setup.cfg`, `mypy.ini`, `pytest.ini`.
- Follow repo config over default tool flags.

2) **Format**
- Run: `python -m black .`
- If Black isn't installed, propose adding it to the dev environment and explain how to install.

3) **Lint**
- Run: `python -m ruff check . --fix`
- If Ruff isn't installed, propose adding it and re-run.

4) **Types (only if configured/expected)**
- If `mypy.ini` or mypy config exists, run: `python -m mypy .`
- Fix type errors by improving annotations (avoid weakening to `Any` unless unavoidable).

5) **Tests**
- Run: `python -m pytest -q`
- If failures: fix implementation first; only adjust tests if the test is wrong or spec changed.

6) **Exit criteria**
- No formatting diffs from Black.
- Ruff clean (or only allowed ignores).
- Mypy clean (if used).
- Pytest passing.

## Common fixes
- Replace `== None` with `is None`.
- Prefer `Pathlib` over manual path string ops when appropriate.
- Tighten return types and avoid ambiguous `Optional` behavior.
