# Jupyter Notebooks

This directory will contain interactive Jupyter notebooks for each chapter of *Principles of Indoor Positioning and Indoor Navigation*.

## Notebooks

| Notebook | Chapter | Description | Open in Colab | Status |
|----------|---------|-------------|----------------|--------|
| `ch2_coordinate_systems.ipynb` | 2 | LLH/ECEF/ENU transforms, rotations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch2_coordinate_systems.ipynb) | ✅ Available |
| `ch3_state_estimation.ipynb` | 3 | LS, WLS, Kalman Filter | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch3_state_estimation.ipynb) | ✅ Available |
| `ch4_rf_positioning.ipynb` | 4 | TOA, TDOA, AOA, RSS positioning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch4_rf_positioning.ipynb) | ✅ Available |
| `ch5_fingerprinting.ipynb` | 5 | NN, k-NN, Bayesian fingerprinting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch5_fingerprinting.ipynb) | ✅ Available |
| `ch6_dead_reckoning.ipynb` | 6 | IMU strapdown, PDR, environmental sensors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch6_dead_reckoning.ipynb) | ✅ Available |
| `ch7_slam.ipynb` | 7 | Pose graph SLAM, ICP scan matching | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch7_slam.ipynb) | ✅ Available |
| `ch8_sensor_fusion.ipynb` | 8 | TC/LC fusion, chi-square gating | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPNL-POLYU/IPIN-Examples/blob/main/notebooks/ch8_sensor_fusion.ipynb) | ✅ Available |

## 🚀 Quick Start with Google Colab

1. Click an **Open in Colab** badge above (or open any notebook in this folder directly in Colab).
2. Run the setup cell — `GITHUB_REPO` is already set to this repository, so it clones and installs dependencies automatically.
   - Running from your own fork? Change `GITHUB_REPO` in the setup cell to your fork's URL first:
   ```python
   GITHUB_REPO = "https://github.com/YOUR_USERNAME/IPIN-Examples.git"
   ```

**Or run locally:**
```bash
cd IPIN-Examples
jupyter notebook notebooks/
```

## Important Resources

In addition to notebooks, please explore the extensive resources already available:

1. **Example Scripts** - Each `ch*_*/` directory contains runnable Python examples (24+ scripts total)
2. **Chapter READMEs** - Comprehensive documentation in each chapter folder with equation-to-code mappings
3. **Documentation** - Additional guides in `docs/` folder
4. **Simulated Datasets** - Pre-generated datasets in `data/sim/` for testing and experimentation

## Running Example Scripts

### Chapter 2: Coordinate Systems

```bash
python -m ch2_coords.example_coordinate_transforms
```

### Chapter 3: State Estimation

```bash
python -m ch3_estimators.example_least_squares
python -m ch3_estimators.example_kalman_1d
python -m ch3_estimators.example_ekf_range_bearing
python -m ch3_estimators.example_comparison
```

### Chapter 4: RF Point Positioning

```bash
python -m ch4_rf_point_positioning.example_toa_positioning
python -m ch4_rf_point_positioning.example_tdoa_positioning
python -m ch4_rf_point_positioning.example_aoa_positioning
python -m ch4_rf_point_positioning.example_comparison
```

### Chapter 5: Fingerprinting

```bash
python -m ch5_fingerprinting.example_deterministic
python -m ch5_fingerprinting.example_probabilistic
python -m ch5_fingerprinting.example_pattern_recognition
python -m ch5_fingerprinting.example_comparison
```

### Chapter 6: Dead Reckoning

```bash
python -m ch6_dead_reckoning.example_imu_strapdown
python -m ch6_dead_reckoning.example_pdr
python -m ch6_dead_reckoning.example_wheel_odometry
python -m ch6_dead_reckoning.example_zupt
python -m ch6_dead_reckoning.example_allan_variance
python -m ch6_dead_reckoning.example_environment
python -m ch6_dead_reckoning.example_comparison
```

### Chapter 7: SLAM

```bash
python -m ch7_slam.example_pose_graph_slam
python -m ch7_slam.example_bundle_adjustment
```

### Chapter 8: Sensor Fusion

```bash
python -m ch8_sensor_fusion.example_lc_fusion
python -m ch8_sensor_fusion.example_tc_fusion
python -m ch8_sensor_fusion.example_comparison
python -m ch8_sensor_fusion.example_observability
python -m ch8_sensor_fusion.example_temporal_calibration
python -m ch8_sensor_fusion.example_robust_tuning
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
