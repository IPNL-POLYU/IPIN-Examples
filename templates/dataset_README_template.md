# [Dataset Name]

## Overview

**Purpose**: [Brief description of what this dataset demonstrates]

**Learning Objectives**:
- [Objective 1: What students should learn]
- [Objective 2: How this connects to book concepts]
- [Objective 3: What experiments this enables]

**Related Chapter**: Chapter [X] - [Chapter Title]

**Related Book Equations**: [List key equations, e.g., Eqs. (X.Y)-(X.Z)]

---

## Scenario Description

**[Trajectory/Environment]**: [Describe the motion/environment, e.g., "2D rectangular walking path"]

**Duration**: [X] seconds

**Motion Characteristics**: [Speed, path type, etc.]

**Sensors**:
- **[Sensor 1]**: [Rate] Hz, [Description]
- **[Sensor 2]**: [Rate] Hz, [Description]

**[Additional elements]**: [e.g., "Anchors placed at: (x1,y1), (x2,y2), ..."]

---

## Files and Data Structure

| File | Shape | Description | Units |
|------|-------|-------------|-------|
| `truth.npz` | | Ground truth states | |
| ├─ `t` | (N,) | Timestamps | seconds |
| ├─ `[field1]` | (N, D) | [Description] | [units] |
| └─ `[field2]` | (N,) | [Description] | [units] |
| `[sensor1].npz` | | [Sensor] measurements | |
| ├─ `t` | (M,) | Timestamps | seconds |
| └─ `[measurement]` | (M, D) | [Description] | [units] |
| `config.json` | | Configuration params | see below |

---

## Loading Example

```python
import numpy as np
import json
from pathlib import Path

# Set dataset path
dataset_path = Path('data/sim/[dataset_name]')

# Load ground truth
truth = np.load(dataset_path / 'truth.npz')
t = truth['t']          # (N,) timestamps
# Add other fields...

# Load sensor data
sensor = np.load(dataset_path / '[sensor].npz')
t_sensor = sensor['t']
measurements = sensor['[measurement]']

# Load configuration
with open(dataset_path / 'config.json') as f:
    config = json.load(f)
    
print(f"Duration: {config['dataset_info']['duration_sec']} seconds")
print(f"[Sensor] rate: {config['[sensor]']['rate_hz']} Hz")
# Print other relevant config info...
```

---

## Configuration Parameters

From `config.json`:

### [Sensor 1] Parameters

- `rate_hz`: [X.X] (sampling rate)
- `[parameter1]`: [value] ([description])
- `[parameter2]`: [value] ([description])

### [Sensor 2] Parameters

- `rate_hz`: [X.X] (measurement rate)
- `[parameter1]`: [value] ([description])
- `[parameter2]`: [value] ([description])

### [Other Section]

- `[parameter]`: [value] ([description])

---

## Parameter Effects and Learning Experiments

| Parameter | Default | Experiment Range | Effect on [Algorithm] | Learning Objective |
|-----------|---------|------------------|----------------------|-------------------|
| `[param1]` | [X] | [X1]-[X2] | [Effect description] | [What students learn] |
| `[param2]` | [Y] | [Y1]-[Y2] | [Effect description] | [What students learn] |
| `[param3]` | [Z] | [Z1]-[Z2] | [Effect description] | [What students learn] |

---

## Dataset Variants

[If applicable, describe related datasets with different configurations]

**1. [dataset_name_variant1]/** - [Purpose]
   - [Key differences]
   - Use for: [Specific learning scenario]
   - Expected observation: [What should happen]

**2. [dataset_name_variant2]/** - [Purpose]
   - [Key differences]
   - Use for: [Specific learning scenario]
   - Expected observation: [What should happen]

---

## Visualization Example

```python
import matplotlib.pyplot as plt

# Plot [main visualization]
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([...], [...], 'k-', label='Ground Truth', linewidth=2)
# Add other plot elements...
ax.set_xlabel('[Label] [units]')
ax.set_ylabel('[Label] [units]')
ax.set_title('[Title]')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('[output_name].svg')
```

**Quick visualization**:
```bash
python tools/plot_dataset_overview.py data/sim/[dataset_name]
```

---

## Connection to Book Equations

This dataset is designed to demonstrate:

**[Equation Group 1]** (Ch[X], Eqs. [Y]-[Z])
- Equation description and how it relates to dataset parameters
- [Specific parameter] → [Specific equation term]

**[Equation Group 2]** (Ch[X], Eq. [Y])
- Equation description and observable behavior in data
- Expected behavior: [Description]

---

## Recommended Experiments

### Experiment 1: [Name]

**Objective**: [What students learn from this experiment]

**Setup**:
```bash
# Generate dataset variants
[command 1]
[command 2]

# Run algorithm on each
[command 3]
[command 4]
```

**Expected Observations**:
- [Observation 1]
- [Observation 2]
- [Key insight for students]

**Analysis**:
- Compare [metric 1] across variants
- Plot [visualization]
- Verify relationship: [theoretical prediction]

### Experiment 2: [Name]

**Objective**: [What students learn]

**Setup**:
```bash
[commands]
```

**Expected Observations**:
- [Observations]

---

## Troubleshooting / Common Student Questions

**Q: [Common question 1]?**
A: [Clear answer with explanation]

**Q: [Common question 2]?**
A: [Clear answer with explanation]

**Q: [Common question 3]?**
A: [Clear answer with guidance]

---

## Generation

This dataset was generated using:
```bash
python scripts/generate_[dataset]_dataset.py [options]
```

See `scripts/README.md` for parameter customization options.

**Regenerate with custom parameters**:
```bash
python scripts/generate_[dataset]_dataset.py \
    --[param1] [value1] \
    --[param2] [value2] \
    --output data/sim/my_custom_dataset
```

---

## References

- **Book Chapter**: Chapter [X], Section [Y]
- **Key Equations**: Eqs. ([X.Y])-([X.Z])
- **Related Examples**: `ch[X]_[topic]/example_[name].py`
- **Generation Script**: `scripts/generate_[dataset]_dataset.py`


