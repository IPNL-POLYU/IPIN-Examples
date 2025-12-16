# Simulation Datasets

This folder contains pre-generated simulation datasets for learning indoor positioning and navigation algorithms. Each dataset is designed to demonstrate specific concepts from the IPIN book and enable hands-on experimentation with algorithm parameters.

## Purpose

Simulation datasets in this repository serve three key learning objectives:

1. **Reproducibility**: Fixed-seed datasets allow students to reproduce book examples exactly
2. **Experimentation**: Students can modify parameters to observe cause-and-effect relationships
3. **Algorithm Comparison**: Same datasets enable fair comparison across different algorithms (e.g., LS vs EKF, LC vs TC)

## Quick Start

```python
import numpy as np
import json

# Load any dataset
truth = np.load('data/sim/ch8_fusion_2d_imu_uwb/truth.npz')
config = json.load(open('data/sim/ch8_fusion_2d_imu_uwb/config.json'))

# Visualize dataset
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb
```

---

## 🔗 Example Scripts ↔ Dataset Connections

Each chapter example folder has scripts that can load from these datasets. Use `--data <dataset_name>` to run with pre-generated data:

### Chapter 2: Coordinate Systems (`ch2_coords/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_coordinate_transforms.py` | `ch2_coords_san_francisco/` | `python -m ch2_coords.example_coordinate_transforms --data ch2_coords_san_francisco` |

### Chapter 3: State Estimation (`ch3_estimators/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_ekf_range_bearing.py` | `ch3_estimator_nonlinear/` | `python -m ch3_estimators.example_ekf_range_bearing --data ch3_estimator_nonlinear` |
| `example_ekf_range_bearing.py` | `ch3_estimator_high_nonlinear/` | `python -m ch3_estimators.example_ekf_range_bearing --data ch3_estimator_high_nonlinear` |

### Chapter 4: RF Point Positioning (`ch4_rf_point_positioning/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_comparison.py` | `ch4_rf_2d_square/` | `python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_square` |
| `example_comparison.py` | `ch4_rf_2d_optimal/` | `python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_optimal` |
| `example_comparison.py` | `ch4_rf_2d_linear/` | `python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_linear` |
| `example_comparison.py` | `ch4_rf_2d_nlos/` | `python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_nlos` |
| `example_comparison.py` | *(compare all)* | `python -m ch4_rf_point_positioning.example_comparison --compare-geometry` |

### Chapter 5: Fingerprinting (`ch5_fingerprinting/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_deterministic.py` | `ch5_wifi_fingerprint_grid/` | `python -m ch5_fingerprinting.example_deterministic` |
| `example_probabilistic.py` | `ch5_wifi_fingerprint_grid/` | `python -m ch5_fingerprinting.example_probabilistic` |
| `example_pattern_recognition.py` | `ch5_wifi_fingerprint_grid/` | `python -m ch5_fingerprinting.example_pattern_recognition` |
| `example_comparison.py` | `ch5_wifi_fingerprint_grid/` | `python -m ch5_fingerprinting.example_comparison` |

> **Note**: CH5 examples use `ch5_wifi_fingerprint_grid/` by default. Edit the script to use `ch5_wifi_fingerprint_dense/` or `ch5_wifi_fingerprint_sparse/` for different density experiments.

### Chapter 6: Dead Reckoning (`ch6_dead_reckoning/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_pdr.py` | `ch6_pdr_corridor_walk/` | `python -m ch6_dead_reckoning.example_pdr --data ch6_pdr_corridor_walk` |

> **Other CH6 datasets**: `ch6_strapdown_basic/`, `ch6_wheel_odom_square/`, `ch6_foot_zupt_walk/`, `ch6_env_sensors_heading_altitude/` are available for additional dead reckoning experiments.

### Chapter 7: SLAM (`ch7_slam/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `example_pose_graph_slam.py` | `ch7_slam_2d_square/` | `python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square` |
| `example_pose_graph_slam.py` | `ch7_slam_2d_high_drift/` | `python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_high_drift` |

### Chapter 8: Sensor Fusion (`ch8_sensor_fusion/`)

| Example Script | Dataset | Command |
|----------------|---------|---------|
| `lc_uwb_imu_ekf.py` | `ch8_fusion_2d_imu_uwb/` | `python -m ch8_sensor_fusion.lc_uwb_imu_ekf --data data/sim/ch8_fusion_2d_imu_uwb` |
| `tc_uwb_imu_ekf.py` | `ch8_fusion_2d_imu_uwb/` | `python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/ch8_fusion_2d_imu_uwb` |
| `compare_lc_tc.py` | `ch8_fusion_2d_imu_uwb/` | `python -m ch8_sensor_fusion.compare_lc_tc --data data/sim/ch8_fusion_2d_imu_uwb` |
| `tuning_robust_demo.py` | `ch8_fusion_2d_imu_uwb_nlos/` | `python -m ch8_sensor_fusion.tuning_robust_demo --data data/sim/ch8_fusion_2d_imu_uwb_nlos` |
| `temporal_calibration_demo.py` | `ch8_fusion_2d_imu_uwb_timeoffset/` | `python -m ch8_sensor_fusion.temporal_calibration_demo --data data/sim/ch8_fusion_2d_imu_uwb_timeoffset` |

### Quick Reference: Dataset → Example Mapping

| Dataset | Primary Example | Chapter |
|---------|-----------------|---------|
| `ch2_coords_san_francisco/` | `ch2_coords/example_coordinate_transforms.py` | CH2 |
| `ch3_estimator_nonlinear/` | `ch3_estimators/example_ekf_range_bearing.py` | CH3 |
| `ch3_estimator_high_nonlinear/` | `ch3_estimators/example_ekf_range_bearing.py` | CH3 |
| `ch4_rf_2d_square/` | `ch4_rf_point_positioning/example_comparison.py` | CH4 |
| `ch4_rf_2d_optimal/` | `ch4_rf_point_positioning/example_comparison.py` | CH4 |
| `ch4_rf_2d_linear/` | `ch4_rf_point_positioning/example_comparison.py` | CH4 |
| `ch4_rf_2d_nlos/` | `ch4_rf_point_positioning/example_comparison.py` | CH4 |
| `ch5_wifi_fingerprint_grid/` | `ch5_fingerprinting/example_*.py` | CH5 |
| `ch5_wifi_fingerprint_dense/` | `ch5_fingerprinting/example_*.py` | CH5 |
| `ch5_wifi_fingerprint_sparse/` | `ch5_fingerprinting/example_*.py` | CH5 |
| `ch6_pdr_corridor_walk/` | `ch6_dead_reckoning/example_pdr.py` | CH6 |
| `ch6_strapdown_basic/` | *(manual loading)* | CH6 |
| `ch6_wheel_odom_square/` | *(manual loading)* | CH6 |
| `ch6_foot_zupt_walk/` | *(manual loading)* | CH6 |
| `ch6_env_sensors_heading_altitude/` | *(manual loading)* | CH6 |
| `ch7_slam_2d_square/` | `ch7_slam/example_pose_graph_slam.py` | CH7 |
| `ch7_slam_2d_high_drift/` | `ch7_slam/example_pose_graph_slam.py` | CH7 |
| `ch8_fusion_2d_imu_uwb/` | `ch8_sensor_fusion/lc_uwb_imu_ekf.py`, `tc_uwb_imu_ekf.py` | CH8 |
| `ch8_fusion_2d_imu_uwb_nlos/` | `ch8_sensor_fusion/tuning_robust_demo.py` | CH8 |
| `ch8_fusion_2d_imu_uwb_timeoffset/` | `ch8_sensor_fusion/temporal_calibration_demo.py` | CH8 |

---

## 🎓 Suggested Learning Paths

### Path A: RF Positioning Fundamentals (Chapter 4)
1. **Start**: `ch4_rf_2d_square/` - Understand TOA/TDOA/AOA with good geometry
2. **Compare**: `ch4_rf_2d_optimal/` vs `ch4_rf_2d_linear/` - Learn GDOP impact (10× difference!)
3. **Challenge**: `ch4_rf_2d_nlos/` - Handle NLOS bias in real environments

### Path B: Dead Reckoning (Chapter 6)
1. **Start**: `ch6_strapdown_basic/` - See unbounded IMU drift
2. **Explore**: `ch6_pdr_corridor_walk/` - Smartphone-based navigation
3. **Advanced**: `ch6_foot_zupt_walk/` - Constraint-based corrections

### Path C: Sensor Fusion (Chapters 6 → 8)
1. **Background**: `ch6_strapdown_basic/` - Understand why fusion is needed
2. **Baseline**: `ch8_fusion_2d_imu_uwb/` - See how fusion bounds IMU drift
3. **Robustness**: `ch8_fusion_2d_imu_uwb_nlos/` - Handle measurement outliers
4. **Calibration**: `ch8_fusion_2d_imu_uwb_timeoffset/` - Time synchronization challenges

### Path D: SLAM (Chapter 7)
1. **Start**: `ch7_slam_2d_square/` - Pose graph basics with loop closure
2. **Challenge**: `ch7_slam_2d_high_drift/` - When SLAM really matters (20× improvement!)

---

## Available Datasets

### Chapter 2: Coordinate Systems

| Dataset | Purpose | Key Learning | Documentation |
|---------|---------|--------------|---------------|
| `ch2_coords_san_francisco/` | LLH↔ECEF↔ENU transforms, rotations | Coordinate frame fundamentals, sub-mm precision | [README](ch2_coords_san_francisco/README.md) |

**Key Equations**: Eqs. (2.1)-(2.10) - Coordinate transforms, rotation representations

**Generation Script**: `scripts/generate_ch2_coordinate_transforms_dataset.py`

---

### Chapter 3: State Estimation

| Dataset | Purpose | Nonlinearity | Key Learning | Documentation |
|---------|---------|--------------|--------------|---------------|
| `ch3_estimator_nonlinear/` | KF vs EKF vs UKF vs PF comparison | Moderate (circular) | When to use which estimator | [README](ch3_estimator_nonlinear/README.md) |
| `ch3_estimator_high_nonlinear/` | High nonlinearity stress test | High (figure-8) | EKF breakdown, UKF advantage | [README](ch3_estimator_high_nonlinear/README.md) |

**Key Equations**: 
- Eqs. (3.11)-(3.19): Linear Kalman Filter
- Eq. (3.21): Extended Kalman Filter
- Eqs. (3.24)-(3.30): Unscented Kalman Filter
- Eqs. (3.32)-(3.34): Particle Filter

**Generation Script**: `scripts/generate_ch3_estimator_comparison_dataset.py`

---

### Chapter 4: RF Point Positioning

| Dataset | Purpose | Geometry | Mean GDOP | Key Learning | Documentation |
|---------|---------|----------|-----------|--------------|---------------|
| `ch4_rf_2d_square/` | Baseline TOA/TDOA/AOA | Square (4 corners) | ~1.0 | Good geometry fundamentals | [README](ch4_rf_2d_square/README.md) |
| `ch4_rf_2d_optimal/` | Best-case geometry | Circular (evenly spaced) | ~0.8 | Optimal beacon placement | [README](ch4_rf_2d_optimal/README.md) |
| `ch4_rf_2d_linear/` | Worst-case geometry | Linear array | >10 | **GDOP degradation (10×!)** | [README](ch4_rf_2d_linear/README.md) |
| `ch4_rf_2d_nlos/` | NLOS robustness | Square + NLOS bias | ~1.0 | Systematic measurement bias | [README](ch4_rf_2d_nlos/README.md) |

**Key Equations**:
- Eqs. (4.1)-(4.3): TOA range measurements
- Eqs. (4.27)-(4.33): TDOA range differences
- Eqs. (4.63)-(4.66): AOA angle measurements
- Section 4.5: DOP calculations

**Generation Script**: `scripts/generate_ch4_rf_2d_positioning_dataset.py`

---

### Chapter 5: Fingerprinting

| Dataset | Purpose | Grid Spacing | RPs/Floor | Key Learning | Documentation |
|---------|---------|--------------|-----------|--------------|---------------|
| `ch5_wifi_fingerprint_grid/` | Baseline Wi-Fi fingerprinting | 5m | 121 (11×11) | NN, k-NN, MAP, Posterior Mean | [README](ch5_wifi_fingerprint_grid/README.md) |
| `ch5_wifi_fingerprint_dense/` | High-accuracy variant | 2m | 676 (26×26) | Dense grids = better accuracy | [README](ch5_wifi_fingerprint_dense/README.md) |
| `ch5_wifi_fingerprint_sparse/` | Quick deployment variant | 10m | 25 (5×5) | Accuracy vs. effort trade-off | [README](ch5_wifi_fingerprint_sparse/README.md) |

**Key Equations**:
- Eq. (5.1): Nearest-Neighbor (NN)
- Eq. (5.2): k-Nearest-Neighbor (k-NN)
- Eq. (5.3): Log-likelihood (Gaussian Naive Bayes)
- Eq. (5.4): MAP estimation
- Eq. (5.5): Posterior Mean estimation

**Generation Script**: `scripts/generate_ch5_wifi_fingerprint_dataset.py`

---

### Chapter 6: Dead Reckoning

| Dataset | Purpose | Sensors | Key Learning | Documentation |
|---------|---------|---------|--------------|---------------|
| `ch6_strapdown_basic/` | IMU integration drift | IMU (100Hz) | **Unbounded drift without corrections** | [README](ch6_strapdown_basic/README.md) |
| `ch6_wheel_odom_square/` | Vehicle odometry | Wheel encoders + IMU | Lever arm, slip effects | [README](ch6_wheel_odom_square/README.md) |
| `ch6_foot_zupt_walk/` | ZUPT corrections | Foot-mounted IMU | Zero-velocity constraint-based correction | [README](ch6_foot_zupt_walk/README.md) |
| `ch6_pdr_corridor_walk/` | Smartphone PDR | Smartphone IMU | **Heading errors dominate (1°→1.7% error)** | [README](ch6_pdr_corridor_walk/README.md) |
| `ch6_env_sensors_heading_altitude/` | Environmental sensors | Magnetometer + Barometer | Heading/altitude estimation | [README](ch6_env_sensors_heading_altitude/README.md) |

**Key Equations**:
- Eqs. (6.2)-(6.5): IMU strapdown integration
- Eq. (6.46): Step detection (accelerometer magnitude)
- Eq. (6.49): Step length (Weinberg model)
- Eq. (6.50): 2D position update from step
- Eqs. (6.51)-(6.53): Magnetometer heading

**Generation Scripts**: 
- `scripts/generate_ch6_strapdown_dataset.py`
- `scripts/generate_ch6_wheel_odom_dataset.py`
- `scripts/generate_ch6_zupt_dataset.py`
- `scripts/generate_ch6_pdr_dataset.py`
- `scripts/generate_ch6_env_sensors_dataset.py`

---

### Chapter 7: SLAM

| Dataset | Purpose | Odometry Drift | Loop Closures | Key Learning | Documentation |
|---------|---------|----------------|---------------|--------------|---------------|
| `ch7_slam_2d_square/` | Pose graph SLAM baseline | ~0.5m | 1 | **10× improvement with loop closure** | [README](ch7_slam_2d_square/README.md) |
| `ch7_slam_2d_high_drift/` | High drift stress test | ~2.0m | 1 | **20× improvement (SLAM essential!)** | [README](ch7_slam_2d_high_drift/README.md) |

**Key Equations**:
- Eqs. (7.10)-(7.11): ICP scan matching
- Section 7.3: Pose graph optimization

**Generation Script**: `scripts/generate_ch7_slam_2d_dataset.py`

---

### Chapter 8: Sensor Fusion

| Dataset | Purpose | Sensors | Key Parameters | Key Learning | Documentation |
|---------|---------|---------|----------------|--------------|---------------|
| `ch8_fusion_2d_imu_uwb/` | Baseline LC/TC fusion | IMU (100Hz) + UWB (10Hz) | No bias, no offset | Multi-rate fusion fundamentals | [README](ch8_fusion_2d_imu_uwb/README.md) |
| `ch8_fusion_2d_imu_uwb_nlos/` | NLOS robustness | IMU + UWB | Anchors 1,2 biased +0.8m | **Chi-square gating, robust loss** | [README](ch8_fusion_2d_imu_uwb_nlos/README.md) |
| `ch8_fusion_2d_imu_uwb_timeoffset/` | Temporal calibration | IMU + UWB | 50ms offset, 100ppm drift | **Time synchronization challenges** | [README](ch8_fusion_2d_imu_uwb_timeoffset/README.md) |

**Key Equations**:
- Eqs. (8.1)-(8.7): Tightly-coupled EKF fusion
- Eqs. (8.8)-(8.9): Chi-square gating, NIS monitoring
- Eqs. (8.13)-(8.18): Loosely-coupled fusion

**Generation Script**: `scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py`

---

## File Format Reference

### Common File Types

**`.npz` (NumPy Compressed Archive)**
- Primary format for numerical data (trajectories, measurements)
- Efficient, supports multiple arrays in one file
- Loading: `data = np.load('file.npz'); array = data['key']`

**`.npy` (NumPy Array)**
- Single array data (e.g., anchor positions)
- Loading: `array = np.load('file.npy')`

**`.txt` (Text Files)**
- Human-readable format for smaller datasets
- Loading: `array = np.loadtxt('file.txt')`

**`.json` (Configuration Files)**
- Human-readable parameter storage
- Contains generation parameters, metadata, frame definitions
- Loading: `config = json.load(open('file.json'))`

### Standard Fields

Most datasets follow this structure:

**Ground Truth** (`truth.npz` or `ground_truth_*.txt`):
- `t`: Timestamps (N,) in seconds
- `p_xy` or `p_xyz`: Positions in meters
- `v_xy` or `v_xyz`: Velocities in m/s (if applicable)
- `yaw` or `q`: Orientation in radians or quaternions

**Sensor Data** (`[sensor].npz` or `[sensor].txt`):
- `t`: Sensor timestamps (M,) in seconds
- Sensor-specific fields (see individual dataset READMEs)

**Configuration** (`config.json`):
- `dataset_info`: Description, seed, duration, sample counts
- `[sensor]`: Sensor-specific parameters (noise, biases, rates)
- `coordinate_frame`: Frame definition and units

### Loading Examples

```python
import numpy as np
import json
from pathlib import Path

# General loading function
def load_dataset(dataset_path):
    """Load a simulation dataset.
    
    Args:
        dataset_path: Path to dataset folder (e.g., 'data/sim/ch8_fusion_2d_imu_uwb')
    
    Returns:
        Dictionary with 'truth', 'sensors', and 'config' keys
    """
    path = Path(dataset_path)
    
    # Load ground truth (try .npz first, then .txt)
    if (path / 'truth.npz').exists():
        truth = dict(np.load(path / 'truth.npz'))
    else:
        # Try text files
        truth = {}
        for f in path.glob('ground_truth_*.txt'):
            key = f.stem.replace('ground_truth_', '')
            truth[key] = np.loadtxt(f)
    
    # Load configuration
    with open(path / 'config.json') as f:
        config = json.load(f)
    
    return {'truth': truth, 'config': config}

# Example usage
dataset = load_dataset('data/sim/ch8_fusion_2d_imu_uwb')
print(f"Duration: {dataset['truth']['t'][-1]:.1f} seconds")
```

---

## Coordinate Frames

All datasets use consistent coordinate frame conventions:

**ENU (East-North-Up)** - Primary frame for most datasets
- X-axis: East
- Y-axis: North  
- Z-axis: Up
- Right-handed coordinate system

**Body Frame** - Sensor-specific frame
- Varies by sensor type (see `core/coords/` for transformations)
- IMU: typically forward-right-down or forward-left-up
- Always documented in dataset README

**Map Frame** - Global reference
- Usually aligned with ENU
- Origin and orientation specified in `config.json`

---

## Parameter Effects Guide

Understanding how parameters affect algorithm behavior is key to learning. Each dataset README includes a parameter effects table. Here are common parameters across datasets:

### IMU Noise Parameters

| Parameter | Typical Range | Effect on Algorithm |
|-----------|---------------|---------------------|
| `accel_noise_std` | 0.01-1.0 m/s² | Higher → faster velocity drift in integration |
| `gyro_noise_std` | 0.001-0.1 rad/s | Higher → faster heading drift |
| `accel_bias` | 0-0.5 m/s² | Causes systematic drift (unbounded without corrections) |
| `gyro_bias` | 0-0.1 rad/s | Causes heading drift proportional to bias × time |

### RF Measurement Parameters

| Parameter | Typical Range | Effect on Algorithm |
|-----------|---------------|---------------------|
| `range_noise_std` | 0.01-0.5 m | Higher → noisier position fixes, more uncertainty |
| `nlos_bias` | 0-2.0 m | Positive bias on affected beacons/anchors |
| `dropout_rate` | 0-0.5 | Probability of missing measurements |

### Temporal Parameters

| Parameter | Typical Range | Effect on Algorithm |
|-----------|---------------|---------------------|
| `time_offset` | -0.5 to 0.5 s | Sensor timestamp misalignment → systematic residuals |
| `clock_drift` | 0-0.001 | Relative clock drift rate (e.g., 100 ppm = 0.0001) |

---

## Generating Custom Datasets

All datasets can be regenerated with custom parameters. See `scripts/README.md` for detailed instructions.

**Quick Examples**:

```bash
# Generate fusion dataset with high IMU noise
python scripts/generate_ch8_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/my_high_noise_experiment

# Generate fingerprint database with dense AP coverage
python scripts/generate_ch5_wifi_fingerprint_dataset.py \
    --n-aps 16 --grid-spacing 3.0 \
    --output data/sim/my_dense_fingerprint

# Generate SLAM data with high drift
python scripts/generate_ch7_slam_2d_dataset.py \
    --preset high_drift \
    --output data/sim/my_challenging_slam
```

---

## Visualization Tools

Quick visualization of any dataset:

```bash
# Plot dataset overview (trajectory, sensors, noise characteristics)
python tools/plot_fusion_dataset.py data/sim/ch8_fusion_2d_imu_uwb

# Compare multiple dataset variants side-by-side
python tools/compare_fusion_variants.py \
    data/sim/ch8_fusion_2d_imu_uwb \
    data/sim/ch8_fusion_2d_imu_uwb_nlos \
    --output comparison.svg

# Validate dataset format and report statistics
python tools/validate_dataset_docs.py
```

---

## Learning Workflow

Recommended workflow for students:

1. **Explore**: Start with pre-generated datasets
   - Read dataset README
   - Load and visualize data
   - Run relevant chapter examples

2. **Understand**: Connect data to theory
   - Identify parameters in `config.json`
   - Map parameters to book equations
   - Predict how changes affect algorithms

3. **Experiment**: Generate custom variants
   - Modify one parameter at a time
   - Compare results systematically
   - Document observations

4. **Analyze**: Interpret results
   - Plot metrics (RMSE, NIS, convergence)
   - Verify predictions match theory
   - Understand failure modes

---

## Dataset Summary Table

| Chapter | Datasets | Key Algorithms | Primary Learning |
|---------|----------|----------------|------------------|
| Ch2 | 1 | LLH↔ECEF↔ENU | Coordinate fundamentals |
| Ch3 | 2 | KF, EKF, UKF, PF | Estimator selection |
| Ch4 | 4 | TOA, TDOA, AOA, DOP | Geometry matters! |
| Ch5 | 3 | NN, k-NN, MAP | Pattern matching |
| Ch6 | 5 | Strapdown, PDR, ZUPT | Drift accumulation |
| Ch7 | 2 | ICP, Pose Graph | Loop closure = consistency |
| Ch8 | 3 | LC-EKF, TC-EKF | Multi-sensor fusion |
| **Total** | **20 datasets** | | |

---

## Troubleshooting

**Q: How do I know which dataset to use for my experiment?**
A: Check the chapter-specific tables above, or see the relevant chapter folder (e.g., `ch8_sensor_fusion/`) for examples that reference specific datasets.

**Q: Can I modify existing datasets?**
A: No, datasets are read-only for reproducibility. Generate new variants instead using the generation scripts.

**Q: What if a dataset is missing?**
A: Some datasets may need to be generated first. Check `scripts/README.md` for generation instructions.

**Q: How much disk space do these datasets require?**
A: Most datasets are < 10 MB. All datasets together: ~100-200 MB.

**Q: Are these datasets sufficient for a thesis/paper?**
A: These are educational datasets for algorithm validation. For research, you'll need real-world data or larger-scale simulations.

**Q: Why do some datasets use `.txt` and others use `.npz`?**
A: Both formats are supported. `.npz` is more efficient for large arrays; `.txt` is human-readable for smaller datasets. See individual dataset READMEs for loading instructions.

---

## Adding New Datasets

If you're contributing a new dataset, follow these requirements:

1. ✅ Create dataset folder in `data/sim/[ch{N}_dataset_name]/` (use chapter prefix!)
2. ✅ Include `README.md` following existing dataset README patterns
3. ✅ Save data in standard formats (`.npz`, `.json`, or `.txt`)
4. ✅ Include `config.json` with all generation parameters
5. ✅ Create generation script in `scripts/` with CLI
6. ✅ Add entry to this catalog (tables above)
7. ✅ Document in `scripts/README.md` with example commands
8. ✅ Provide at least 2 experimentation scenarios
9. ✅ Connect to specific book equations
10. ✅ Test all code examples in README

---

## References

- **Main README**: `README.md` (project root)
- **Generation Scripts**: `scripts/` folder and `scripts/README.md`
- **Learning Guide**: `docs/data_simulation_guide.md`
- **Tools**: `tools/` folder (visualization, validation, comparison)
- **Book**: *Principles of Indoor Positioning and Indoor Navigation*, Chapters 2-8
