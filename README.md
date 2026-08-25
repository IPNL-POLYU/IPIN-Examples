# Principle of IPIN Open-Sourced Code and Data

## Book

This repository is the companion code + datasets for the book:

**Principles of Indoor Positioning and Indoor Navigation** — Li-Ta Hsu, Guohao Zhang, Weisong Wen.

Publisher page (Artech House):
https://us.artechhouse.com/Principles-of-Indoor-Positioning-and-Indoor-Navigation-P2459.aspx

## Project Structure

```
IPIN-Examples/
├── core/                        # Reusable math & models
│   ├── coords/                  # Coordinate systems (ENU/NED/LLH, rotations)
│   ├── estimators/              # LS, robust LS, KF/EKF/UKF, PF
│   ├── rf/                      # RF models (RSS, TOA/TDOA/AOA, DOP)
│   ├── sensors/                 # IMU, wheel odom, PDR, mag, barometer
│   ├── fingerprinting/          # Wi-Fi/magnetic fingerprinting algorithms
│   ├── slam/                    # SLAM geometry, scan matching, factors
│   ├── fusion/                  # Multi-sensor fusion utilities
│   ├── models/                  # Common motion & measurement models
│   ├── sim/                     # Synthetic sensor data from a trajectory
│   ├── utils/                   # Angles, geometry, observability
│   └── eval/                    # Metrics, error stats, plots
├── ch2_coords/                  # Chapter 2: Coordinate Systems
├── ch3_estimators/              # Chapter 3: State Estimation
├── ch4_rf_point_positioning/    # Chapter 4: RF Point Positioning
├── ch5_fingerprinting/          # Chapter 5: Fingerprinting
├── ch6_dead_reckoning/          # Chapter 6: Dead Reckoning & PDR
├── ch7_slam/                    # Chapter 7: SLAM Technologies
├── ch8_sensor_fusion/           # Chapter 8: Sensor Fusion
├── data/sim/                    # Simulated datasets
├── docs/                        # Documentation & equation mappings
├── notebooks/                   # Jupyter notebooks for interactive learning
├── scripts/                     # Dataset generation scripts
├── tools/                       # CI/maintenance scripts
├── references/                  # Design specifications
└── tests/                       # Unit tests, plus the guards over docs and figures
```

## Architecture

Each chapter is a folder of runnable examples over one package of the shared
`core/` library. Both the diagram and the table are generated from the imports
themselves by `tools/chapter_dependencies.py`, so neither can drift from the code.

<!-- BEGIN GENERATED: repo-architecture (tools/chapter_dependencies.py) -->

```mermaid
flowchart LR
    A0["<b>ch2_coords/</b><br/>coordinate frames"] --> B0["core/coords/"]
    A1["<b>ch3_estimators/</b><br/>estimation"] --> B1["core/estimators/"]
    A2["<b>ch4_rf_point_positioning/</b><br/>RF positioning"] --> B2["core/rf/"]
    A3["<b>ch5_fingerprinting/</b><br/>fingerprinting"] --> B3["core/fingerprinting/"]
    A4["<b>ch6_dead_reckoning/</b><br/>dead reckoning"] --> B4["core/sensors/<br/>core/sim/"]
    A5["<b>ch7_slam/</b><br/>SLAM"] --> B5["core/estimators/<br/>core/slam/"]
    A6["<b>ch8_sensor_fusion/</b><br/>sensor fusion"] --> B6["core/estimators/<br/>core/fusion/"]
    S["<b>imported by nearly every chapter</b><br/>core/eval/ · core/utils/"]
```

| Chapter | Core packages it imports |
| --- | --- |
| [`ch2_coords/`](ch2_coords/) | `core.coords`, `core.eval`, `core.utils` |
| [`ch3_estimators/`](ch3_estimators/) | `core.estimators`, `core.eval`, `core.utils` |
| [`ch4_rf_point_positioning/`](ch4_rf_point_positioning/) | `core.eval`, `core.rf`, `core.utils` |
| [`ch5_fingerprinting/`](ch5_fingerprinting/) | `core.eval`, `core.fingerprinting` |
| [`ch6_dead_reckoning/`](ch6_dead_reckoning/) | `core.eval`, `core.sensors`, `core.sim`, `core.utils` |
| [`ch7_slam/`](ch7_slam/) | `core.estimators`, `core.eval`, `core.slam`, `core.utils` |
| [`ch8_sensor_fusion/`](ch8_sensor_fusion/) | `core.estimators`, `core.eval`, `core.fusion` |

<!-- END GENERATED: repo-architecture -->

## Chapter Overview

Each chapter folder contains example scripts and a README with equation-to-code mappings:

| Chapter | Topic | Key Algorithms | Equations |
|---------|-------|----------------|-----------|
| **Ch2** | Coordinate Systems | LLH↔ECEF↔ENU, Euler/Quaternion/Matrix rotations | Eqs. 2.1-2.23 |
| **Ch3** | State Estimation | LS, WLS, KF, EKF, IEKF, UKF, PF, FGO | Eqs. 3.1-3.56 |
| **Ch4** | RF Positioning | TOA, TDOA, AOA, RSS, DOP | Eqs. 4.1-4.108 |
| **Ch5** | Fingerprinting | NN, k-NN, MAP, Posterior Mean, Classification | Eqs. 5.1-5.6 |
| **Ch6** | Dead Reckoning | IMU Strapdown, PDR, ZUPT, Wheel Odometry, Allan Variance | Eqs. 6.2-6.61 |
| **Ch7** | SLAM | ICP, NDT, Pose Graph, Bundle Adjustment | Eqs. 7.10-7.70 |
| **Ch8** | Sensor Fusion | LC/TC EKF, Observability, Gating, Calibration | Eqs. 8.1-8.9 |

The **Equations** column is the span of book equations that
[`docs/equation_index.yml`](docs/equation_index.yml) maps to code for that
chapter. It is a span, not a complete list — the index has deliberate gaps
where the book states an equation this repository does not implement.

See [Quick Start](#quick-start) below to run the examples.

## Notebooks

Each chapter also has an interactive Jupyter notebook in [`notebooks/`](notebooks/) — open it directly in Google Colab, no local install required:

| Notebook | Chapter | Open in Colab |
|----------|---------|----------------|
| `ch2_coordinate_systems.ipynb` | 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch2_coordinate_systems.ipynb) |
| `ch3_state_estimation.ipynb` | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch3_state_estimation.ipynb) |
| `ch4_rf_positioning.ipynb` | 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch4_rf_positioning.ipynb) |
| `ch5_fingerprinting.ipynb` | 5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch5_fingerprinting.ipynb) |
| `ch6_dead_reckoning.ipynb` | 6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch6_dead_reckoning.ipynb) |
| `ch7_slam.ipynb` | 7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch7_slam.ipynb) |
| `ch8_sensor_fusion.ipynb` | 8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch8_sensor_fusion.ipynb) |

See [`notebooks/README.md`](notebooks/README.md) for details.

## Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IPIN-Examples
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install the package and development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

Run any chapter's example as a module:

```bash
python -m ch3_estimators.example_least_squares
python -m ch5_fingerprinting.example_comparison
python -m ch6_dead_reckoning.example_comparison
```

`python -m` puts the repository root on `sys.path`, so these run straight from
a fresh clone even before step 4 above, and it is the form every command in
this repository is written in.

The script form — `python <chapter>/<example>.py` — puts the *script's*
directory there instead, so `core` would not be importable from a fresh clone.
Each example now adds the repository root itself before importing `core`, so
that form works too. It is worth knowing why the line is there: without it, on
a machine that has ever installed this package, `import core` does not fail —
it quietly resolves to **whichever checkout the install points at**, and the
example runs to completion against a different copy of the library. The error,
when there is one, names a directory you have never heard of.

Examples find their datasets from any working directory, so `cd`-ing into a
chapter folder first is fine. Every example takes `--help`, which prints what
it demonstrates and which book equations it implements without running
anything:

```bash
python -m ch3_estimators.example_least_squares --help
```

### Figures

Each example prints its results and writes its figures to `ch*_*/figs/`. Two
environment variables change that, and neither is per-example — they mean the
same thing everywhere:

| Variable | Effect |
|---|---|
| `IPIN_FIGS_DIR` | Write the figures somewhere else instead of `ch*_*/figs/`. |
| `IPIN_SHOW_FIGURES` | Also open them in a window. Off by default, because with a GUI backend that **blocks** until you close it. |

```bash
IPIN_SHOW_FIGURES=1 python -m ch6_dead_reckoning.example_zupt
```

## Code Style

New code follows **PEP 8** and the **Google Python Style Guide**: type hints on
every function, Google-style docstrings, PascalCase classes, snake_case
functions, 88-character lines.

### What is actually enforced

The test suite is the gate, and it enforces more than style. Everything below
runs in CI on every pull request:

| Check | What it holds |
|---|---|
| `tests/test_repo_conventions.py` | No raw `savefig`, no unseeded RNG, no `plt.show()` in an example, chapter folders hold only `example_*.py`, and **pyflakes is clean across ~300 files** |
| `tests/docs/` | Every documented command, path and flag exists; every README transcript matches what the example prints; the notebooks run |
| `tests/test_every_figure_has_a_demo_behind_it.py` | Every committed figure is still produced, and everything produced is committed |
| `tests/test_examples_answer_help.py` | `--help` answers instead of running the demonstration |

```bash
pytest
```

### Where the linters stand

Stated plainly, because this section used to imply the repository passed all of
them and a reader who ran them got thousands of complaints. Measured over
`core/`, the chapters, `scripts/`, `tools/` and `tests/`:

| Tool | Today | Was |
|---|---|---|
| `black --check` | **passes** — 299 files unchanged | 237 of 288 reformatted |
| `ruff check` | **951 findings.** 727 are annotation modernisations (`List[int]` → `list[int]`) that only became available when the floor moved to 3.10; the ~140 after that are the ones with content, `zip()` without `strict=` first | 5836 |
| `mypy` | 406 errors in `core/` alone — **the remaining gap** | 404 |

`tests/test_lint_debt_only_shrinks.py` records the ruff count per rule and fails
both when one grows and when a baseline sits above the real count, so the number
can only go down. It is not pass/fail on the linters themselves; mypy would be
red on arrival and stay red.

```bash
ruff check core ch*_* tests
black --check core ch*_* tests
mypy core
```

## Testing

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=core --cov=ch*_* --cov-report=html
```

A test run leaves the working tree clean. Some tests run a chapter example end
to end to check its figures are still produced, so figure output is diverted to
a temporary directory for the duration of the run rather than overwriting the
committed figures in `ch*_*/figs/`. Set `IPIN_FIGS_DIR` yourself to send an
example's figures somewhere other than its chapter:

```bash
IPIN_FIGS_DIR=/tmp/figs python -m ch5_fingerprinting.example_probabilistic
```

## Development Workflow

For each chapter/topic, follow this 5-step process:

1. **Spec extraction**: Define function signatures and APIs
2. **Core module skeleton**: Implement with type hints and docstrings
3. **Unit tests**: Write tests for core functionality
4. **Example/notebook**: Create demonstration notebooks
5. **Documentation**: Update docs with usage examples

## How to cite

If you use this repository in academic work, please cite the book:

APA (7th) - Hsu, L.-T., Zhang, G., & Wen, W. (2025). Principles of indoor positioning and indoor navigation. Artech House.

IEEE - L.-T. Hsu, G. Zhang, and W. Wen, Principles of Indoor Positioning and Indoor Navigation. Norwood, MA, USA: Artech House, 2025. ISBN: 978-1-63081-977-4.

### BibTeX
```bibtex
@book{Hsu2025IPIN,
  title     = {Principles of Indoor Positioning and Indoor Navigation},
  author    = {Hsu, Li-Ta and Zhang, Guohao and Wen, Weisong},
  publisher = {Artech House},
  address   = {Norwood, MA},
  year      = {2025},
  isbn      = {978-1-63081-977-4}
}
```

## Acknowledgements

This repository is supported by **The Hong Kong Polytechnic University (PolyU)** under the **Financial Support for Book Writing** scheme. This support enables the development, testing, documentation, and release of the companion code and datasets for the book *Principles of Indoor Positioning and Indoor Navigation*.

## License

This repository is intended to be **academic-friendly** (research/teaching) while requiring **prior permission for commercial use**.

- **Code** (e.g., `core/`, `ch*_*/`, `scripts/`, `tools/`, `tests/`) is licensed under the **PolyForm Noncommercial License 1.0.0** — full text in [`LICENSE`](LICENSE).
- **Data** (e.g., `data/`) is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** unless otherwise noted in the corresponding folder — full text in [`LICENSE-DATA`](LICENSE-DATA).

### Commercial use

Commercial use is **not permitted** under the licenses above. If you want to use this repository for product development, commercial services, internal commercial evaluation, or other for-profit purposes, please contact the maintainers to discuss a separate commercial license.

### Book content notice

This GitHub repository does **not** distribute the book PDF or other publisher-copyrighted book content. It provides original companion implementations and datasets intended to support learning and reproducible experiments.

