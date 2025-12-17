# Principle of IPIN Open-Sourced Code and Data

## Book

This repository is the companion code + datasets for the book:

**Principles of Indoor Positioning and Indoor Navigation** — Li-Ta Hsu, Guohao Zhang, Weisong Wen.

Publisher page (Artech House):
https://us.artechhouse.com/Principles-of-Indoor-Positioning-and-Indoor-Navigation-P2459.aspx

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
## Project Structure

```
IPIN_Book_Examples/
├── core/                        # Reusable math & models
│   ├── coords/                  # Coordinate systems (ENU/NED/LLH, rotations)
│   ├── estimators/              # LS, robust LS, KF/EKF/UKF, PF
│   ├── rf/                      # RF models (RSS, TOA/TDOA/AOA, DOP)
│   ├── sensors/                 # IMU, wheel odom, PDR, mag, barometer
│   ├── fingerprinting/          # Wi-Fi/magnetic fingerprinting algorithms
│   ├── slam/                    # SLAM geometry, scan matching, factors
│   ├── fusion/                  # Multi-sensor fusion utilities
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
└── tests/                       # Unit tests (778 test cases)
```

## Chapter Overview

Each chapter folder contains example scripts and a README with equation-to-code mappings:

| Chapter | Topic | Key Algorithms | Equations |
|---------|-------|----------------|-----------|
| **Ch2** | Coordinate Systems | LLH↔ECEF↔ENU, Euler/Quaternion/Matrix rotations | Eqs. 2.1-2.10 |
| **Ch3** | State Estimation | LS, WLS, Robust LS, KF, EKF | Eqs. 3.1-3.9 |
| **Ch4** | RF Positioning | TOA, TDOA, AOA, RSS, DOP | Eqs. 4.1-4.69 |
| **Ch5** | Fingerprinting | k-NN, MAP, Posterior Mean, Linear Regression | Eqs. 5.1-5.5 |
| **Ch6** | Dead Reckoning | IMU Strapdown, PDR, ZUPT, Wheel Odometry | Eqs. 6.2-6.61 |
| **Ch7** | SLAM | ICP, NDT, Pose Graph, Bundle Adjustment | Eqs. 7.10-7.70 |
| **Ch8** | Sensor Fusion | Loosely/Tightly Coupled EKF, Observability | Practical methods |

**Quick Start:** Run any chapter's example script:
```bash
python ch3_estimators/example_least_squares.py
python ch5_fingerprinting/example_comparison.py
python ch6_dead_reckoning/example_comparison.py
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IPIN_Book_Examples
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

## Code Style

This project follows **PEP 8** and the **Google Python Style Guide**. All code should:

- Use type hints for all functions
- Include Google-style docstrings
- Follow naming conventions (PascalCase for classes, snake_case for functions)
- Be formatted with Black (88 character line length)
- Pass linting checks (flake8/ruff, mypy, pylint)

### Formatting and Linting

Format code:
```bash
black .
```

Check code style:
```bash
ruff check .
flake8 .
```

Type checking:
```bash
mypy .
```

Linting:
```bash
pylint core/ ch*_*/
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

## Development Workflow

For each chapter/topic, follow this 5-step process:

1. **Spec extraction**: Define function signatures and APIs
2. **Core module skeleton**: Implement with type hints and docstrings
3. **Unit tests**: Write tests for core functionality
4. **Example/notebook**: Create demonstration notebooks
5. **Documentation**: Update docs with usage examples

## Acknowledgements

This repository is supported by **The Hong Kong Polytechnic University (PolyU)** under the **Financial Support for Book Writing** scheme. This support enables the development, testing, documentation, and release of the companion code and datasets for the book *Principles of Indoor Positioning and Indoor Navigation*.

## License

This repository is intended to be **academic-friendly** (research/teaching) while requiring **prior permission for commercial use**.

- **Code** (e.g., `core/`, `ch*_*/`, `scripts/`, `tools/`, `tests/`) is licensed under the **PolyForm Noncommercial License 1.0.0**.
- **Data** (e.g., `data/`) is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** unless otherwise noted in the corresponding folder.

### Commercial use

Commercial use is **not permitted** under the licenses above. If you want to use this repository for product development, commercial services, internal commercial evaluation, or other for-profit purposes, please contact the maintainers to discuss a separate commercial license.

### Book content notice

This GitHub repository does **not** distribute the book PDF or other publisher-copyrighted book content. It provides original companion implementations and datasets intended to support learning and reproducible experiments.

