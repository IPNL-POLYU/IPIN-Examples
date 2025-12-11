# Development Setup Guide

## Initial Setup

1. **Install Python 3.8+** (if not already installed)

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Pre-commit Setup (Optional but Recommended)

Install pre-commit hooks to automatically format and lint code:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## IDE Configuration

### VS Code / Cursor

Recommended extensions:
- Python
- Pylance
- Black Formatter
- Ruff

Settings (`.vscode/settings.json`):
```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.rulers": [88]
  }
}
```

## Running Code Quality Checks

Before committing, run:

```bash
# Format code
black .

# Check style
ruff check .

# Type check
mypy .

# Run tests
pytest
```

Or use the convenience script (create `scripts/check.sh` or `scripts/check.bat`):

```bash
#!/bin/bash
black . && ruff check . && mypy . && pytest
```

