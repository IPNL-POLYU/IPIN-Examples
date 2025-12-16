# Jupyter Notebooks

This directory will contain interactive Jupyter notebooks for each chapter of *Principles of Indoor Positioning and Indoor Navigation*.

## Planned Notebooks

| Notebook | Chapter | Description | Status |
|----------|---------|-------------|--------|
| `ch2_coordinates.ipynb` | 2 | Coordinate transforms and rotations | Planned |
| `ch3_estimators.ipynb` | 3 | LS, KF, EKF, UKF, PF comparison | Planned |
| `ch4_rf_positioning.ipynb` | 4 | TOA, TDOA, AOA positioning demos | Planned |
| `ch5_fingerprinting.ipynb` | 5 | Wi-Fi fingerprinting methods | Planned |
| `ch6_dead_reckoning.ipynb` | 6 | IMU, PDR, wheel odometry | Planned |
| `ch7_slam.ipynb` | 7 | ICP, NDT, pose graph SLAM | Planned |
| `ch8_sensor_fusion.ipynb` | 8 | LC vs TC fusion comparison | Planned |

## Current Status

Notebooks are planned for a future release. In the meantime, please use:

1. **Example Scripts** - Each `ch*_*/` directory contains runnable Python examples
2. **Chapter READMEs** - Comprehensive documentation in each chapter folder
3. **Equation Index** - `docs/equation_index.yml` maps equations to code

## Running Example Scripts

```bash
# Chapter 3: State Estimation
python ch3_estimators/example_least_squares.py
python ch3_estimators/example_comparison.py

# Chapter 5: Fingerprinting
python ch5_fingerprinting/example_comparison.py

# Chapter 6: Dead Reckoning
python ch6_dead_reckoning/example_comparison.py

# Chapter 8: Sensor Fusion
python -m ch8_sensor_fusion.tc_uwb_imu_ekf
python -m ch8_sensor_fusion.compare_lc_tc
```

## Contributing

To contribute a notebook:

1. Use the naming convention `ch{N}_{topic}.ipynb`
2. Include markdown cells explaining the algorithms and equations
3. Reference book equations using format: `Eq. (X.Y)`
4. Generate visualizations that can be saved as static images
5. Ensure the notebook runs without errors using `jupyter nbconvert --execute`

## Dependencies

Notebooks will require:
- `jupyter` or `jupyterlab`
- All dependencies in `pyproject.toml`
- Optionally: `ipywidgets` for interactive demos

```bash
pip install jupyter
jupyter notebook notebooks/
```


