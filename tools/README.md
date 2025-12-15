# Tools

Maintenance and CI/CD scripts for the IPIN Book Examples repository.

## Available Tools

### `check_equation_index.py`

Verifies consistency between equation references in code and the `docs/equation_index.yml` documentation.

**Usage:**
```bash
# Basic check
python tools/check_equation_index.py

# Verbose output (shows file locations)
python tools/check_equation_index.py --verbose

# Strict mode (fails CI if any issues)
python tools/check_equation_index.py --strict
```

**What it checks:**
1. All equations referenced in code docstrings (e.g., `Eq. (2.1)`) are documented in `equation_index.yml`
2. All file paths in the equation index point to existing files
3. Reports equations in index that aren't referenced in code (informational)

**Example output:**
```
Project root: /path/to/IPIN_Book_Examples

Equations in index: 91
Equations referenced in code: 103

[OK] All equations in code are documented in index
[OK] All file paths in index are valid

============================================================
SUMMARY
============================================================
  Indexed equations:     91
  Code references:       103
  Missing from index:    0
  File path errors:      0

[PASSED]
```

## CI Integration

Add to your CI pipeline (e.g., GitHub Actions):

```yaml
- name: Check equation index
  run: python tools/check_equation_index.py --strict
```

## Adding New Tools

When adding new tools:

1. Create the Python script in `tools/`
2. Add a docstring explaining usage
3. Update this README with documentation
4. Ensure the tool works without external dependencies when possible

