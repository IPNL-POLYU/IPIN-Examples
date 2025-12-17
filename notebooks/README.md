# Jupyter Notebooks

This directory will contain interactive Jupyter notebooks for each chapter of *Principles of Indoor Positioning and Indoor Navigation*.

## Notebooks

| Notebook | Chapter | Description | Status |
|----------|---------|-------------|--------|
| `ch2_coordinate_systems.ipynb` | 2 | LLH/ECEF/ENU transforms, rotations | ✅ Available |
| `ch3_state_estimation.ipynb` | 3 | LS, WLS, Robust LS, Kalman Filter | ✅ Available |
| `ch4_rf_positioning.ipynb` | 4 | TOA, TDOA, AOA, RSS positioning | ✅ Available |
| `ch4_rf_positioning.ipynb` | 4 | TOA, TDOA, AOA positioning demos | Planned |
| `ch5_fingerprinting.ipynb` | 5 | Wi-Fi fingerprinting methods | Planned |
| `ch6_dead_reckoning.ipynb` | 6 | IMU, PDR, wheel odometry | Planned |
| `ch7_slam.ipynb` | 7 | ICP, NDT, pose graph SLAM | Planned |
| `ch8_sensor_fusion.ipynb` | 8 | LC vs TC fusion comparison | Planned |

## 🚀 Quick Start with Google Colab

The easiest way to run notebooks is with Google Colab (no installation required!):

1. Open Google Colab: https://colab.research.google.com
2. File → Open notebook → GitHub
3. Enter the repository URL and select a notebook
4. Run the first setup cell to clone and install dependencies

**Or run locally:**
```bash
cd IPIN_Book_Examples
jupyter notebook notebooks/
```

## Current Status

One notebook is now available! More are planned for future releases. In the meantime, please use the extensive resources already available:

1. **Example Scripts** - Each `ch*_*/` directory contains runnable Python examples (24+ scripts total)
2. **Chapter READMEs** - Comprehensive documentation in each chapter folder with equation-to-code mappings
3. **Documentation** - Additional guides in `docs/` folder
4. **Simulated Datasets** - Pre-generated datasets in `data/sim/` for testing and experimentation

## Running Example Scripts

### Chapter 2: Coordinate Systems

```bash
python ch2_coords/example_coordinate_transforms.py
```

### Chapter 3: State Estimation

```bash
python ch3_estimators/example_least_squares.py
python ch3_estimators/example_kalman_1d.py
python ch3_estimators/example_ekf_range_bearing.py
python ch3_estimators/example_comparison.py
```

### Chapter 4: RF Point Positioning

```bash
python ch4_rf_point_positioning/example_toa_positioning.py
python ch4_rf_point_positioning/example_tdoa_positioning.py
python ch4_rf_point_positioning/example_aoa_positioning.py
python ch4_rf_point_positioning/example_comparison.py
```

### Chapter 5: Fingerprinting

```bash
python ch5_fingerprinting/example_deterministic.py
python ch5_fingerprinting/example_probabilistic.py
python ch5_fingerprinting/example_pattern_recognition.py
python ch5_fingerprinting/example_comparison.py
```

### Chapter 6: Dead Reckoning

```bash
python ch6_dead_reckoning/example_imu_strapdown.py
python ch6_dead_reckoning/example_pdr.py
python ch6_dead_reckoning/example_wheel_odometry.py
python ch6_dead_reckoning/example_zupt.py
python ch6_dead_reckoning/example_allan_variance.py
python ch6_dead_reckoning/example_environment.py
python ch6_dead_reckoning/example_comparison.py
```

### Chapter 7: SLAM

```bash
python ch7_slam/example_pose_graph_slam.py
python ch7_slam/example_bundle_adjustment.py
```

### Chapter 8: Sensor Fusion

```bash
python -m ch8_sensor_fusion.lc_uwb_imu_ekf
python -m ch8_sensor_fusion.tc_uwb_imu_ekf
python -m ch8_sensor_fusion.compare_lc_tc
python -m ch8_sensor_fusion.observability_demo
python -m ch8_sensor_fusion.temporal_calibration_demo
python -m ch8_sensor_fusion.tuning_robust_demo
```

## Available Documentation

The `docs/` folder contains additional guides and references:

| Document | Description |
|----------|-------------|
| `equation_index.yml` | Maps book equations to code implementations |
| `ch2_equation_mapping.md` | Chapter 2 equation-to-code mappings |
| `CH2_QUICK_REFERENCE.md` | Quick reference for coordinate transforms |
| `ch7_slam.md` | SLAM algorithms documentation |
| `ch8_fusion_api_reference.md` | Sensor fusion API reference |
| `ch8_lc_tc_comparison_guide.md` | Loosely vs Tightly Coupled fusion comparison |
| `data_simulation_guide.md` | Guide for generating simulated datasets |

## Simulated Datasets

The `data/sim/` folder contains pre-generated datasets for each chapter:

| Dataset Folder | Description |
|----------------|-------------|
| `ch2_coordinate_transforms/` | Coordinate system test data |
| `ch3_estimator_comparison/` | State estimation scenarios |
| `ch4_rf_2d_positioning/` | RF positioning with anchors |
| `ch5_wifi_fingerprint_*/` | Wi-Fi fingerprinting datasets (dense/sparse) |
| `ch6_pdr_*/` | PDR datasets (indoor/outdoor, consumer/tactical) |
| `ch6_strapdown_*/` | IMU strapdown navigation data |
| `ch6_wheel_odom_*/` | Wheel odometry datasets |
| `ch6_zupt_*/` | Zero-velocity update scenarios |
| `ch6_env_*/` | Environmental sensor data (barometer, magnetometer) |
| `ch7_slam_2d/` | 2D SLAM test environments |
| `ch8_fusion_2d_imu_uwb_*/` | IMU-UWB fusion datasets |

Run the dataset generation scripts in `scripts/` to create custom datasets.

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
