# IPIN Book Examples

Code examples for the IPIN (Indoor Positioning and Indoor Navigation) book, organized by chapter with shared core modules.

## Project Structure

```
ipin-examples/
├── core/                    # Reusable math & models
│   ├── coords/              # Coordinate systems (ENU/NED/LLH, rotations)
│   ├── estimators/         # LS, robust LS, KF/EKF/UKF, PF, FGO
│   ├── rf/                  # RF models (RSS, TOA/TDOA/AOA)
│   ├── sensors/             # IMU, wheel odom, mag, barometer
│   ├── sim/                 # Generic simulators
│   └── eval/                # Metrics, error stats, DOP
├── ch2_coords/              # Chapter 2 examples
├── ch3_estimators/          # Chapter 3 examples
├── ch4_rf_point_positioning/# Chapter 4 examples
├── ch5_fingerprinting/      # Chapter 5 examples
├── ch6_dead_reckoning/      # Chapter 6 examples
├── ch7_slam/                # Chapter 7 examples
├── ch8_sensor_fusion/       # Chapter 8 examples
├── ch9_advanced/            # Chapter 9 examples
├── data/                    # Data files
│   ├── sim/                 # Simulated data
│   └── real/                # Real data (optional)
├── notebooks/               # Jupyter notebooks per chapter
└── tests/                   # Unit tests

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

## License

MIT License
