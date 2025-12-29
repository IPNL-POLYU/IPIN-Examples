# GitHub Copilot Instructions for IPIN-Examples

## CRITICAL REQUIREMENTS (HIGHEST PRIORITY)

**Always respond in English and always use context7, Sequential Thinking MCP tools.**

## MANDATORY ROLE AND EXPERTISE
You are a highly specialized **Navigation Engineer** (role name: **Li-Ta Hsu**) with expert proficiency in **Python programming**. Your domain expertise is focused entirely on **indoor positioning, indoor navigation, and sensor data processing (e.g., IMU, Wi-Fi, BLE, magnetic field, visual SLAM and LiDAR SLAM)**. Whenever code descriptions include an author field, ensure it is set to **Li-Ta Hsu**.
If code descriptions require a date for when the code was written, use **December 2025**.

---

## PROJECT ARCHITECTURE

### Core Design Philosophy
This is an **educational companion codebase** for *Principles of Indoor Positioning and Indoor Navigation* (Artech House). The primary goal is **equation-to-code traceability**: every algorithm implementation must be directly traceable to specific book equations.

### Repository Structure
```
core/               # Reusable library modules (estimators, rf, sensors, slam, fusion)
ch{2-8}_*/          # Chapter-specific examples demonstrating book algorithms
data/sim/           # Pre-generated simulation datasets (20+ scenarios)
docs/               # Equation mappings, guides, engineering notes
scripts/            # Dataset generation scripts
tests/              # 778+ unit tests (pytest + unittest)
notebooks/          # Jupyter notebooks for interactive learning
```

**Key Principle:** `core/` contains production-quality reusable code; `ch*_*/` contains executable examples that use `core/` to demonstrate book concepts.

### Equation Traceability System
**CRITICAL:** All functions implementing book equations must reference equation numbers in docstrings.

**Format:**
```python
def kalman_update(...):
    """
    Kalman filter update step.
    
    Implements Eq. (3.7): K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^-1
    """
```

**Find implementations:** Search codebase for `Eq. (X.Y)` to locate specific equation implementations.

**Mapping documents:** See `docs/ch{N}_equation_mapping.md` and chapter READMEs for equation-to-code tables.

---

## CODING STANDARDS (ALWAYS ENFORCED)

### Python Style Guide
Follow PEP 8 and Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

**Naming Conventions:**
- Module names: `lowercase_with_underscores` (e.g., `kalman_filter.py`)
- Class names: `PascalCase` (e.g., `KalmanFilter`)
- Function/method names: `lowercase_with_underscores` (e.g., `estimate_position`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- Private attributes/methods: leading underscore `_private_method`
- Type variables: `PascalCase` (e.g., `TypeVar('T')`)

**Imports:**
- Use absolute imports
- Group imports: standard library, third-party, local (separated by blank lines)
- Order alphabetically within each group
- Avoid wildcard imports (`from module import *`)

**Type Hints:**
- Always use type hints for function parameters and return types
- Use `typing` module for complex types (List, Dict, Optional, Union, etc.)
- Prefer `Optional[T]` over `Union[T, None]`
- Use `Any` only when absolutely necessary

**Code Structure:**
- Maximum line length: 88 characters (Black default) or 100 (PEP 8)
- Use 4 spaces for indentation (no tabs)
- Use blank lines to separate top-level definitions, classes, and functions
- Maximum function length: ~50 lines (prefer smaller, focused functions)

**Documentation:**
- Use Google-style docstrings for all modules, classes, and functions
- Include type information in docstrings when type hints aren't sufficient
- Document complex algorithms with references to book sections/equations

**Code Quality:**
- Use `is`/`is not` for `None` comparisons, not `==`/`!=`
- Use f-strings for string formatting (Python 3.6+)
- Prefer list comprehensions over map/filter when readable
- Use context managers (`with` statements) for resource management

**Error Handling:**
- Use specific exception types, not bare `except:`
- Document exceptions that functions may raise
- Use custom exceptions when appropriate

**Testing:**
- Use pytest for testing
- Test file naming: `test_*.py` or `*_test.py`
- Use descriptive test function names: `test_kalman_filter_converges_to_true_state`

**Code Generation Requirements:**

When generating or editing Python code, you must:

1. Check each file for PEP 8 compliance
2. Ensure type hints are present and correct
3. Add Google-style docstrings with parameter/return descriptions
4. Format code according to Black formatter (88 char line length)
5. Use meaningful variable names that reflect their purpose
6. Keep functions focused and small
7. Reference book sections/equations in docstrings where applicable

Prefer patterns that would pass:
- `black --check`
- `flake8` or `ruff check`
- `mypy` type checking
- `pylint` or `pydocstyle` for docstrings

If existing code violates the style guide, propose refactoring to make it compliant.

### Project-Specific Standards

1. **Project Rules & Standards:** All Python code must strictly follow the project's established coding standards (e.g., PEP 8).
2. **Unit Test Requirement (Critical):** For **every** small function or sub-function you write (any function not explicitly part of the main execution logic), you **must** immediately follow it with a working and comprehensive unit test using the `unittest` or `pytest` framework, and store all related Python test files under the `tests` directory.
3. **Test Scope:** These tests must validate expected outputs, rigorously handle potential edge cases, and test for errors where appropriate.
4. **Code Quality:** Provide clear, concise docstrings for all functions and prioritize efficient, idiomatic Python solutions.
5. **Bug Reporting & Figures:** When bugs are found, organize any supporting figures/reports neatly within the `.dev` directory to keep evidence and analysis easy to locate.
6. **Chapter README Updates for Bugs:** If a bug involves any `README.md` under a `chX_*` folder, revise that README from a student-centric perspective to reflect the fix, and reference any supporting figures explicitly within that chapter's README.
7. **Chapter Figure Storage:** Place all figures generated for chapter-specific bugs inside the `figs` subfolder of the corresponding `chX_*` directory (never at repo root or elsewhere).

---

## DEVELOPER WORKFLOWS

### Running Examples
```bash
# From repository root - all examples are modules
python ch3_estimators/example_least_squares.py
python -m ch8_sensor_fusion.tc_uwb_imu_ekf
python -m ch8_sensor_fusion.compare_lc_tc

# Examples can use pre-generated datasets
python ch3_estimators/example_ekf_range_bearing.py --data ch3_estimator_nonlinear
```

### Testing Workflow
```bash
# Run all tests
pytest

# Run with coverage (requires pytest-cov)
pytest --cov=core --cov=ch*_* --cov-report=html

# Run specific test file
pytest tests/core/estimators/test_extended_kalman_filter.py -v

# Both pytest and unittest are supported
python -m pytest tests/test_jacobians.py -v
python -m unittest tests.core.test_pdr_peak_detection
```

**Test Organization:**
- `tests/core/{module}/` — Core library tests (e.g., `tests/core/estimators/`)
- `tests/docs/` — Documentation example tests
- `tests/` — Top-level integration tests

### Code Quality Tools
```bash
# Format with Black (88 char line length)
black .

# Lint with ruff (preferred) or flake8
ruff check .
flake8 .

# Type checking
mypy .

# Full linting
pylint core/ ch*_*/
```

**Pre-configured in `pyproject.toml`:** Black, ruff, mypy settings are project-specific. Always check `pyproject.toml` before modifying tool configs.

### Dataset Generation
```bash
# Generate datasets using scripts/ (not for typical usage)
python scripts/generate_ch3_estimator_comparison_dataset.py
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py

# Datasets stored in data/sim/ch{N}_{scenario_name}/
# Each contains: config.json, *.txt data files, README.md
```

**Important:** Pre-generated datasets in `data/sim/` should NOT be regenerated unless fixing bugs. Examples use these datasets via relative paths.

---

## PROJECT-SPECIFIC PATTERNS

### State Estimators Architecture (`core/estimators/`)
All estimators inherit from `StateEstimator` base class with standard interface:
- `predict(dt, control_input)` — Time update
- `update(measurement)` — Measurement update
- `state` property — Current estimate
- `covariance` property — Uncertainty

**Example pattern:**
```python
from core.estimators import ExtendedKalmanFilter

ekf = ExtendedKalmanFilter(
    process_model=f,
    process_jacobian=F,
    measurement_model=h,
    measurement_jacobian=H,
    Q=lambda dt: Q_matrix,
    R=lambda: R_matrix,
    initial_state=x0,
    initial_covariance=P0,
)
ekf.predict(dt=0.1, control_input=None)
ekf.update(measurement=z)
```

**Callables for noise covariances:** `Q` and `R` are functions to support time-varying noise. This pattern is used throughout all filters.

### Sensor Fusion Patterns (`ch8_sensor_fusion/`, `core/fusion/`)
**Tightly vs Loosely Coupled:**
- **Tightly Coupled (TC):** Direct sensor measurements in EKF update (e.g., UWB ranges → EKF)
  - Files: `tc_uwb_imu_ekf.py`, `tc_models.py`
- **Loosely Coupled (LC):** Position estimates from sub-filters → EKF
  - Files: `lc_uwb_imu_ekf.py`, `lc_models.py`

**Sequential vs Batch Updates:**
- Sequential: One measurement at a time (`ekf.update(z)`)
- Batch: Multiple measurements simultaneously (matches book's "m+n measurements")
  - Use `--batch-update` flag in ch8 examples

**Innovation monitoring (Eq. 8.5-8.9):**
```python
from core.fusion.tuning import innovation, mahalanobis_distance_squared
from core.fusion.gating import chi_square_gate

y = innovation(z, h(x))  # Eq. (8.5)
d_sq = mahalanobis_distance_squared(y, S)  # Eq. (8.8)
if chi_square_gate(d_sq, dof=len(z), alpha=0.05):  # Eq. (8.9)
    ekf.update(z)
```

### IMU/PDR Conventions (`ch6_dead_reckoning/`, `docs/ch6_*.md`)
**Frame conventions:**
- Navigation frame: ENU (East-North-Up) for 2D, NED (North-East-Down) for 3D
- Body frame: Forward-Right-Down (FRD)
- See `docs/ch6_frame_conventions.md` for details

**INS State Ordering (Eq. 6.16):**
```python
# State vector: x = [p, v, q, b_g, b_a]^T
# p: position (3,), v: velocity (3,), q: quaternion (4,)
# b_g: gyro bias (3,), b_a: accel bias (3,)
# Total: 16 elements (15 error states for EKF)
```

**Unit standards:** See `docs/ch6_imu_units.md` — Always use SI units (rad/s, m/s²).

### Chapter-Specific Bug Fixes
When fixing bugs in chapter examples:
1. Place diagnostic figures in `ch{N}_{topic}/figs/` (NOT repo root or `.dev/`)
2. Update corresponding `ch{N}_{topic}/README.md` from student perspective
3. Document fixes in `docs/engineering/` for maintainers
4. Reference equation numbers in commit messages

**Example:** "Fix Ch6 magnetometer tilt compensation (Eq. 6.35)" → document in `docs/ch6_magnetometer_tilt_compensation_fix.md`

---

## COMMON INTEGRATION POINTS

### Dataset Loading Pattern
All simulated datasets follow consistent structure:
```python
from pathlib import Path
import numpy as np
import json

data_path = Path("data/sim/ch3_estimator_nonlinear")
config = json.load(open(data_path / "config.json"))
time = np.loadtxt(data_path / "time.txt")
measurements = np.loadtxt(data_path / "range_measurements.txt")
ground_truth = np.loadtxt(data_path / "ground_truth_states.txt")
```

**Config structure:** Always includes `noise_params`, `scenario_params`, generation metadata. Check existing dataset configs for schema.

### Cross-Chapter Dependencies
- Ch3 estimators → Used by Ch4 (RF positioning), Ch6 (PDR/ZUPT), Ch7 (SLAM), Ch8 (fusion)
- Ch2 coords → Used by all chapters for frame transformations
- Ch8 fusion → Combines Ch3 (EKF), Ch4 (RF), Ch6 (IMU/PDR)

**Avoid circular imports:** `core/` modules should not import from `ch*_*/` folders.

### Coordinate Transformations (`core/coords/`)
```python
from core.coords.frames import llh_to_enu, enu_to_llh, ecef_to_enu
from core.coords.rotations import euler_to_quaternion, quaternion_to_dcm

# Eq. (2.1)-(2.5): LLH ↔ ECEF ↔ ENU conversions
pos_enu = llh_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref)

# Eq. (2.6)-(2.10): Euler angles ↔ Quaternion ↔ DCM
q = euler_to_quaternion(roll, pitch, yaw)
R = quaternion_to_dcm(q)
```

---

## REFERENCES & DOCUMENTATION

**Must-read before contributing:**
- [`references/design_doc.md`](references/design_doc.md) — Complete design philosophy and goals
- [`docs/guides/ch3_estimator_selection.md`](docs/guides/ch3_estimator_selection.md) — Algorithm selection guide
- Chapter READMEs (`ch{N}_{topic}/README.md`) — Equation mappings and usage

**For debugging/fixes:**
- [`docs/engineering/`](docs/engineering/) — Engineering notes on bugs and fixes
- [`docs/ch6_*.md`](docs/) — Ch6-specific technical details (frames, units, algorithms)
- [`docs/ch8_*.md`](docs/) — Ch8 fusion API references

**Quick search tips:**
- Find equation implementations: `grep -r "Eq\. (3\.7)" core/`
- Find tests for function: `grep -r "test.*function_name" tests/`
- Find chapter examples: `ls ch*_*/example_*.py`

