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

---

### `plot_fusion_dataset.py`

Visualizes 2D IMU + UWB fusion datasets with comprehensive plots showing trajectory, measurements, noise characteristics, and data quality metrics.

**Usage:**
```bash
# Generate all plots for a dataset
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb

# Save to custom output directory
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb --output plots/baseline

# Generate PNG instead of SVG
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb --format png

# Display plots interactively
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb --show
```

**Generated plots:**
1. `trajectory.svg` - 2D trajectory with UWB anchors (NLOS anchors highlighted)
2. `velocity_heading.svg` - Velocity and heading over time
3. `imu_measurements.svg` - Accelerometer X/Y and gyroscope Z
4. `uwb_ranges.svg` - Per-anchor range measurements over time
5. `range_errors.svg` - Range error histograms with mean bias indicators

---

### `compare_fusion_variants.py`

Creates side-by-side comparison plots of different dataset configurations (e.g., baseline vs. NLOS vs. time offset).

**Usage:**
```bash
# Compare all 3 standard variants
python tools/compare_fusion_variants.py \
    data/sim/ch8_fusion_2d_imu_uwb \
    data/sim/ch8_fusion_2d_imu_uwb_nlos \
    data/sim/ch8_fusion_2d_imu_uwb_timeoffset

# Compare baseline vs. NLOS only
python tools/compare_fusion_variants.py \
    data/sim/ch8_fusion_2d_imu_uwb \
    data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --output comparison_baseline_vs_nlos

# Save as PNG instead of SVG
python tools/compare_fusion_variants.py \
    data/sim/ch8_fusion_2d_imu_uwb \
    data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --format png

# Display interactively
python tools/compare_fusion_variants.py \
    data/sim/ch8_fusion_2d_imu_uwb \
    data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --show
```

**Generated outputs:**
- `comparison_trajectories.svg` - Overlaid trajectories from all datasets
- `comparison_range_errors.svg` - Per-anchor error histograms (NLOS highlighted in red)
- `comparison_imu_measurements.svg` - IMU signals side-by-side
- Console summary table comparing key parameters across datasets

---

### `test_all_datasets.py`

Tests all code examples in dataset READMEs across all chapters. Validates that documentation is accurate and examples are runnable.

**Usage:**
```bash
# Test all datasets
python tools/test_all_datasets.py

# Test specific chapter only
python tools/test_all_datasets.py --chapter 8

# Verbose output (detailed test results)
python tools/test_all_datasets.py --verbose
```

**What it tests:**
1. README.md exists for each dataset
2. config.json is valid JSON
3. Python code blocks in READMEs are runnable
4. Skips non-runnable snippets (ellipsis, output examples, configs)

**Example output:**
```
======================================================================
DATASET TESTING SUITE
======================================================================
Testing 12 datasets...

CH8_FUSION:
  Testing ch8_fusion_2d_imu_uwb...
  [PASS] ch8_fusion_2d_imu_uwb
      Code examples: 3/3 passed

======================================================================
OVERALL: 12/12 datasets passed
Total issues found: 0
======================================================================
```

---

### `validate_dataset_docs.py`

Validates dataset documentation completeness following the standards in `references/design_doc.md` Section 5.3.

**Usage:**
```bash
# Check all datasets
python tools/validate_dataset_docs.py

# Check specific dataset
python tools/validate_dataset_docs.py ch8_fusion_2d_imu_uwb

# Quiet mode (only summary)
python tools/validate_dataset_docs.py --quiet

# Strict mode (warnings treated as errors)
python tools/validate_dataset_docs.py --strict
```

**What it validates:**
1. Required files present (config.json, data files)
2. Required README sections (per Section 5.3.2):
   - Overview
   - Scenario Description
   - Files and Data Structure
   - Loading Example
   - Configuration Parameters
   - Parameter Effects and Learning Experiments
   - Visualization Example
   - Connection to Book Equations
   - Recommended Experiments
3. Parameter effects table present
4. Code examples with Python loading snippets

**Example output:**
```
Dataset Documentation Validator
Checking 12 dataset(s)...

Checking: ch8_fusion_2d_imu_uwb
Path: data/sim/ch8_fusion_2d_imu_uwb
  [OK] All required files present
  [OK] README.md exists
  [OK] All required sections present
  [OK] 8 code blocks found
  [OK] Parameter effects table present
  Status: VALID [OK]

======================================================================
VALIDATION SUMMARY
======================================================================
Total datasets checked: 12
Valid datasets: 12
Invalid datasets: 0

All datasets have complete documentation! [OK]
```

---

## CI Integration

Add to your CI pipeline (e.g., GitHub Actions):

```yaml
- name: Check equation index
  run: python tools/check_equation_index.py --strict

- name: Validate dataset documentation
  run: python tools/validate_dataset_docs.py --strict

- name: Test dataset examples
  run: python tools/test_all_datasets.py
```

## Adding New Tools

When adding new tools:

1. Create the Python script in `tools/`
2. Add a docstring explaining usage
3. Update this README with documentation
4. Ensure the tool works without external dependencies when possible

