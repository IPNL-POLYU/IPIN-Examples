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
truth = np.load('data/sim/fusion_2d_imu_uwb/truth.npz')
config = json.load(open('data/sim/fusion_2d_imu_uwb/config.json'))

# Visualize dataset
python tools/plot_dataset_overview.py data/sim/fusion_2d_imu_uwb
```

---

## Available Datasets

### Chapter 8: Sensor Fusion

| Dataset | Purpose | Sensors | Key Parameters | Used In | Documentation |
|---------|---------|---------|----------------|---------|---------------|
| `fusion_2d_imu_uwb/` | Baseline fusion | IMU (100Hz) + UWB (10Hz) | No bias, no offset | Ch8 LC/TC demos | [README](fusion_2d_imu_uwb/README.md) |
| `fusion_2d_imu_uwb_nlos/` | NLOS robustness | IMU + UWB | Anchors 1,2 biased +0.8m | Robust loss demo | [README](fusion_2d_imu_uwb_nlos/README.md) |
| `fusion_2d_imu_uwb_timeoffset/` | Temporal calibration | IMU + UWB | 50ms offset, 100ppm drift | Time sync demo | [README](fusion_2d_imu_uwb_timeoffset/README.md) |

**Generation Script**: `scripts/generate_fusion_2d_imu_uwb_dataset.py`

### Chapter 6: Dead Reckoning

| Dataset | Purpose | Sensors | Key Parameters | Used In | Documentation |
|---------|---------|---------|----------------|---------|---------------|
| `ch6_strapdown_basic/` | IMU drift demo | IMU (100Hz) | Various noise levels | Strapdown integration | [README](ch6_strapdown_basic/README.md) |
| `ch6_wheel_odom_square/` | Vehicle DR | Wheel encoders + IMU | Lever arm, slip | Wheel odometry | [README](ch6_wheel_odom_square/README.md) |
| `ch6_foot_zupt_walk/` | Constraint-based correction | Foot-mounted IMU | ZUPT thresholds | ZUPT demo | [README](ch6_foot_zupt_walk/README.md) |
| `ch6_pdr_corridor_walk/` | Pedestrian navigation | Smartphone IMU | Step length params | PDR demo | [README](ch6_pdr_corridor_walk/README.md) |
| `ch6_env_sensors_heading_altitude/` | Environmental sensors | Magnetometer + Barometer | Disturbances | Heading/altitude | [README](ch6_env_sensors_heading_altitude/README.md) |

**Generation Scripts**: `scripts/generate_ch6_*.py`

### Chapter 5: Fingerprinting

| Dataset | Purpose | Sensors | Key Parameters | Used In | Documentation |
|---------|---------|---------|----------------|---------|---------------|
| `wifi_fingerprint_grid/` | RSS fingerprinting | Wi-Fi (8 APs) | Grid 5m, 3 floors | k-NN, MAP, Bayesian | [README](wifi_fingerprint_grid/README.md) |

**Generation Script**: `scripts/generate_wifi_fingerprint_dataset.py`

### Chapter 4: RF Point Positioning

| Dataset | Purpose | Sensors | Key Parameters | Used In | Documentation |
|---------|---------|---------|----------------|---------|---------------|
| `rf_2d_floor/` | RF positioning | TOA/TDOA/AOA/RSS | Beacon geometry, NLOS | TOA/TDOA/AOA demos | [README](rf_2d_floor/README.md) |

**Generation Script**: `scripts/generate_rf_2d_floor_dataset.py`

### Chapter 7: SLAM

| Dataset | Purpose | Sensors | Key Parameters | Used In | Documentation |
|---------|---------|---------|----------------|---------|---------------|
| `slam_lidar2d/` | LiDAR SLAM | 2D LiDAR scans | Outliers, loop closures | ICP, NDT, pose graph | [README](slam_lidar2d/README.md) |
| `slam_visual_bearing2d/` | Visual SLAM | Camera observations | Landmarks, outliers | Bundle adjustment | [README](slam_visual_bearing2d/README.md) |

**Generation Scripts**: `scripts/generate_slam_*.py`

### Chapter 3: Estimators (Toy Datasets)

| Dataset | Purpose | Used In | Documentation |
|---------|---------|---------|---------------|
| `toy_ls_linear/` | Linear least squares testing | LS/WLS unit tests | [README](toy_ls_linear/README.md) |
| `toy_kf_1d_cv/` | 1D Kalman filter | KF unit tests | [README](toy_kf_1d_cv/README.md) |
| `toy_nonlinear_1d/` | Nonlinear filtering comparison | EKF/UKF/PF comparison | [README](toy_nonlinear_1d/README.md) |

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

**`.json` (Configuration Files)**
- Human-readable parameter storage
- Contains generation parameters, metadata, frame definitions
- Loading: `config = json.load(open('file.json'))`

### Standard Fields

Most datasets follow this structure:

**Ground Truth** (`truth.npz`):
- `t`: Timestamps (N,) in seconds
- `p_xy` or `p_xyz`: Positions in meters
- `v_xy` or `v_xyz`: Velocities in m/s (if applicable)
- `yaw` or `q`: Orientation in radians or quaternions

**Sensor Data** (`[sensor].npz`):
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
        dataset_path: Path to dataset folder (e.g., 'data/sim/fusion_2d_imu_uwb')
    
    Returns:
        Dictionary with 'truth', 'sensors', and 'config' keys
    """
    path = Path(dataset_path)
    
    # Load ground truth
    truth = dict(np.load(path / 'truth.npz'))
    
    # Load configuration
    with open(path / 'config.json') as f:
        config = json.load(f)
    
    # Load sensors (dataset-specific)
    sensors = {}
    # Add sensor loading based on config
    
    return {'truth': truth, 'sensors': sensors, 'config': config}

# Example usage
dataset = load_dataset('data/sim/fusion_2d_imu_uwb')
print(f"Duration: {dataset['truth']['t'][-1]:.1f} seconds")
print(f"Trajectory length: {len(dataset['truth']['t'])} samples")
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
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/my_high_noise_experiment

# Generate fingerprint database with dense AP coverage
python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 16 --grid-spacing 3.0 \
    --output data/sim/my_dense_fingerprint

# Generate SLAM data with more outliers
python scripts/generate_slam_lidar2d_dataset.py \
    --outlier-rate 0.3 \
    --output data/sim/my_challenging_slam
```

---

## Visualization Tools

Quick visualization of any dataset:

```bash
# Plot dataset overview (trajectory, sensors, noise characteristics)
python tools/plot_dataset_overview.py data/sim/fusion_2d_imu_uwb

# Compare multiple dataset variants side-by-side
python tools/compare_dataset_variants.py \
    data/sim/fusion_2d_imu_uwb \
    data/sim/fusion_2d_imu_uwb_nlos \
    --output comparison.svg

# Validate dataset format and report statistics
python tools/validate_dataset.py data/sim/fusion_2d_imu_uwb
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

## Troubleshooting

**Q: How do I know which dataset to use for my experiment?**
A: Check the "Used In" column in the tables above, or see the relevant chapter folder (e.g., `ch8_sensor_fusion/`) for examples.

**Q: Can I modify existing datasets?**
A: No, datasets are read-only for reproducibility. Generate new variants instead using the generation scripts.

**Q: What if a dataset is missing?**
A: Some datasets may need to be generated first. Check `scripts/README.md` for generation instructions.

**Q: How much disk space do these datasets require?**
A: Most datasets are < 10 MB. All datasets together: ~100-200 MB.

**Q: Are these datasets sufficient for a thesis/paper?**
A: These are educational datasets for algorithm validation. For research, you'll need real-world data or larger-scale simulations.

---

## Adding New Datasets

If you're contributing a new dataset, follow these requirements:

1. ✓ Create dataset folder in `data/sim/[dataset_name]/`
2. ✓ Include `README.md` using `templates/dataset_README_template.md`
3. ✓ Save data in standard formats (`.npz`, `.json`)
4. ✓ Include `config.json` with all generation parameters
5. ✓ Create generation script in `scripts/` with CLI
6. ✓ Add entry to this catalog (tables above)
7. ✓ Document in `scripts/README.md` with example commands
8. ✓ Provide at least 2 experimentation scenarios
9. ✓ Connect to specific book equations
10. ✓ Test all code examples in README

See `templates/` folder for templates and Section 5.3 of `references/design_doc.md` for detailed requirements.

---

## References

- **Design Document**: `references/design_doc.md` (Section 5: Simulation Datasets)
- **Generation Scripts**: `scripts/` folder and `scripts/README.md`
- **Learning Guide**: `docs/data_simulation_guide.md`
- **Tools**: `tools/` folder (visualization, validation, comparison)
- **Book**: *Principles of Indoor Positioning and Indoor Navigation*, Chapters 2-8

