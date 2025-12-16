# Developer Quick Start Guide

> **Note**: This guide is for developers/contributors extending the IPIN Book Examples repository. For student documentation, see the main `README.md` and chapter-specific READMEs.

## What's in `.templates/`

This folder contains templates for creating new content:

- `dataset_README_template.md` - Copy this for new dataset READMEs
- `generation_script_CLI_template.py` - Copy this for new generation scripts
- `parameter_effects_table_template.md` - Examples and guidelines for parameter tables

## Quick Test

Let's verify everything works:

```bash
# 1. Check the validator runs
python tools/validate_dataset_docs.py --help

# 2. Try validating existing datasets (will show what's missing)
python tools/validate_dataset_docs.py

# 3. Read the central documentation
cat data/sim/README.md
cat scripts/README.md
```

---

## Adding a New Dataset

### 1. Generate the data

```bash
python scripts/generate_my_dataset.py --output data/sim/my_dataset
```

### 2. Create README

```bash
cp .templates/dataset_README_template.md data/sim/my_dataset/README.md
# Fill in all sections
```

### 3. Update catalog

```bash
# Edit data/sim/README.md - add row to appropriate chapter table
```

### 4. Document generation

```bash
# Edit scripts/README.md - add to inventory and create 2 scenarios
```

### 5. Validate

```bash
python tools/validate_dataset_docs.py my_dataset
```

---

## Documentation Checklist for New Datasets

### Files Required
- [ ] `README.md` (use template)
- [ ] `config.json` (with all generation parameters)
- [ ] Data files (`.npz`, `.npy`)

### README Sections Required (9)
- [ ] ## Overview
- [ ] ## Scenario Description
- [ ] ## Files and Data Structure
- [ ] ## Loading Example
- [ ] ## Configuration Parameters
- [ ] ## Parameter Effects and Learning Experiments
- [ ] ## Visualization Example
- [ ] ## Connection to Book Equations
- [ ] ## Recommended Experiments

### README Sections Recommended (3)
- [ ] ## Dataset Variants
- [ ] ## Troubleshooting / Common Student Questions
- [ ] ## Generation

### Validation
- [ ] `python tools/validate_dataset_docs.py [dataset_name]` passes
- [ ] All code examples tested and working
- [ ] Generation script has CLI with `--help`

---

## Tips for Success

1. **Start with templates**: Don't write from scratch
2. **Use examples**: All templates include Ch4-Ch8 examples
3. **Validate early**: Run validator after each major section
4. **Test code**: All code examples must run without errors
5. **Connect theory**: Every parameter should map to book equations
6. **Student-first**: Write for students discovering concepts, not experts


