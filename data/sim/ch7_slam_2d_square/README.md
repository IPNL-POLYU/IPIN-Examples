# Ch7 SLAM 2D Dataset: Square Loop with Odometry Drift

## Overview

This dataset demonstrates **2D LiDAR-based pose graph SLAM** with odometry drift and loop closure detection. It showcases the **critical importance of loop closure** for correcting accumulated drift.

**Key Learning Objective**: Understand that loop closure detection + pose graph optimization can reduce final positioning error by 10× compared to dead-reckoning alone!

## Dataset Purpose

### Learning Goals
1. **Odometry Drift**: See how small per-step errors accumulate over time
2. **Loop Closure Detection**: Learn how revisiting places enables drift correction
3. **Pose Graph Optimization**: Understand factor graphs for trajectory correction
4. **ICP Scan Matching**: Use point cloud alignment for loop verification
5. **SLAM vs. Dead-Reckoning**: Quantify 10× error reduction with SLAM!

### Implemented Equations
- **Eqs. (7.10-7.11)**: ICP (Iterative Closest Point) scan matching
  ```
  E(R, t) = Σᵢ ||pᵢ' - (R·pᵢ + t)||²
  Find optimal rigid transformation between point clouds
  ```

- **Section 7.3**: Pose graph optimization
  ```
  Minimize: Σ(odometry_residuals) + Σ(loop_closure_residuals)
  Variables: Robot poses x₀, x₁, ..., xₙ
  Constraints: Odometry + loop closures
  ```

## Dataset Variants

| Variant | Drift Noise | Final Drift | Loop Closures | Key Demonstration |
|---------|-------------|-------------|---------------|-------------------|
| **Baseline** | 0.1m, 0.02rad | ~0.5m | 1 | Standard SLAM scenario |
| **Low Drift** | 0.02m, 0.005rad | ~0.1m | 1 | High-quality odometry |
| **High Drift** | 0.3m, 0.05rad | ~2.0m | 1 | **Poor odometry (SLAM essential!)** |
| **Figure-8** | 0.1m, 0.02rad | ~0.8m | Multiple | **Complex trajectory, multiple loops** |

**Generate variants**:
```bash
python scripts/generate_ch7_slam_2d_dataset.py --preset baseline
python scripts/generate_ch7_slam_2d_dataset.py --preset low_drift
python scripts/generate_ch7_slam_2d_dataset.py --preset high_drift
python scripts/generate_ch7_slam_2d_dataset.py --preset figure8
```

## Files

### Trajectory Data
- `ground_truth_poses.txt`: True robot poses [N×3] (x, y, yaw in m, rad)
- `odometry_poses.txt`: Odometry estimates [N×3] (with cumulative drift)

### Environment Data
- `landmarks.txt`: Static landmark positions [M×2] (x, y in m)
- `scans.npz`: LiDAR scans in robot frame [compressed, N scans]

### SLAM Constraints
- `loop_closures.txt`: Loop closure index pairs [K×2] (pose_i, pose_j)

### Configuration
- `config.json`: All dataset parameters and statistics

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path

# Load dataset
data_dir = Path("data/sim/ch7_slam_2d_square")

ground_truth = np.loadtxt(data_dir / "ground_truth_poses.txt")
odometry = np.loadtxt(data_dir / "odometry_poses.txt")
landmarks = np.loadtxt(data_dir / "landmarks.txt")
loop_closures = np.loadtxt(data_dir / "loop_closures.txt", dtype=int)

# Load scans
scans_data = np.load(data_dir / "scans.npz")
scans = [scans_data[f"scan_{i}"] for i in range(len(ground_truth))]

print(f"Loaded {len(ground_truth)} poses")
print(f"Landmarks: {len(landmarks)}")
print(f"Loop closures: {len(loop_closures)}")

# Compute drift
final_drift = np.linalg.norm(odometry[-1, :2] - ground_truth[-1, :2])
print(f"Final odometry drift: {final_drift:.2f}m")
```

## Configuration Parameters

### Trajectory Configuration
```json
{
  "trajectory": {
    "type": "square",
    "size_m": 20.0,
    "n_poses_per_side": 10,
    "total_poses": 41
  }
}
```

**Key Parameters**:
- **type**: Trajectory shape (square, figure8, random_walk)
- **size**: Characteristic dimension (20m × 20m square)
- **total_poses**: 41 poses around square loop

### Sensor Configuration
```json
{
  "sensor": {
    "max_range_m": 15.0,
    "scan_noise_std_m": 0.05
  }
}
```

**Key Parameters**:
- **max_range**: LiDAR maximum range (15m)
- **scan_noise**: Range measurement noise (0.05m std dev)

### Odometry Configuration
```json
{
  "odometry": {
    "translation_noise_std_m": 0.1,
    "rotation_noise_std_rad": 0.02,
    "final_drift_m": 0.55
  }
}
```

**Key Parameters**:
- **translation_noise**: Per-step translation error (0.1m)
- **rotation_noise**: Per-step rotation error (0.02 rad ≈ 1.1°)
- **final_drift**: Accumulated error at loop closure (0.55m)

## Quick Start Examples

### Example 1: Visualize Odometry Drift
```python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_dir = Path("data/sim/ch7_slam_2d_square")
ground_truth = np.loadtxt(data_dir / "ground_truth_poses.txt")
odometry = np.loadtxt(data_dir / "odometry_poses.txt")
landmarks = np.loadtxt(data_dir / "landmarks.txt")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Landmarks
ax.scatter(landmarks[:, 0], landmarks[:, 1], c='gray', s=20, alpha=0.3, label='Landmarks')

# Ground truth
ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', linewidth=2, label='Ground Truth')
ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c='green', s=200, marker='o', label='Start')

# Odometry (with drift)
ax.plot(odometry[:, 0], odometry[:, 1], 'r--', linewidth=2, label='Odometry')
ax.scatter(odometry[-1, 0], odometry[-1, 1], c='red', s=200, marker='x', label='End (drifted)')

# Drift vector
ax.arrow(odometry[-1, 0], odometry[-1, 1],
         ground_truth[-1, 0] - odometry[-1, 0],
         ground_truth[-1, 1] - odometry[-1, 1],
         head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2, label='Drift')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Odometry Drift: {np.linalg.norm(odometry[-1, :2] - ground_truth[-1, :2]):.2f}m')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.show()
```

**Expected**: See odometry trajectory drift away from true trajectory. Final error ~0.5m.

**Learning Point**: Small per-step errors (0.1m) accumulate to large drift over 41 poses!

### Example 2: Run ICP Scan Matching
```python
from core.slam import icp_point_to_point

# Load scans
scans_data = np.load("data/sim/ch7_slam_2d_square/scans.npz")
scan_0 = scans_data["scan_0"]  # First scan
scan_40 = scans_data["scan_40"]  # Last scan (loop closure!)

# Run ICP to find relative pose
pose_rel, iterations, residual, converged = icp_point_to_point(
    source_scan=scan_40,
    target_scan=scan_0,
    initial_pose=np.array([0.0, 0.0, 0.0]),  # Start from no offset
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=0.5
)

print(f"ICP Result:")
print(f"  Relative pose: x={pose_rel[0]:.3f}m, y={pose_rel[1]:.3f}m, yaw={pose_rel[2]:.3f}rad")
print(f"  Iterations: {iterations}")
print(f"  Residual: {residual:.6f}")
print(f"  Converged: {converged}")
```

**Expected**: ICP finds near-zero relative pose (scans 0 and 40 are at same location!)

**Learning Point**: Scan matching enables loop closure detection!

### Example 3: Build and Optimize Pose Graph
```python
from core.slam import create_pose_graph
from core.estimators import optimize_factor_graph

# Load poses and loop closures
odometry = np.loadtxt("data/sim/ch7_slam_2d_square/odometry_poses.txt")
ground_truth = np.loadtxt("data/sim/ch7_slam_2d_square/ground_truth_poses.txt")
loop_closures_idx = np.loadtxt("data/sim/ch7_slam_2d_square/loop_closures.txt", dtype=int)

# Build pose graph
graph = create_pose_graph(
    initial_poses=[odometry[i] for i in range(len(odometry))],
    odometry_measurements=[(i, i+1) for i in range(len(odometry)-1)],
    loop_closures=[(int(lc[0]), int(lc[1])) for lc in loop_closures_idx.reshape(-1, 2)]
)

# Optimize
optimized_poses, info = optimize_factor_graph(graph, max_iterations=20)

# Compute errors
odom_error = np.linalg.norm(odometry[-1, :2] - ground_truth[-1, :2])
slam_error = np.linalg.norm(optimized_poses[-1][:2] - ground_truth[-1, :2])

print(f"Errors:")
print(f"  Odometry: {odom_error:.3f}m")
print(f"  SLAM: {slam_error:.3f}m")
print(f"  Improvement: {odom_error/slam_error:.1f}×")
```

**Expected**: SLAM reduces error from ~0.5m to ~0.05m (10× improvement!)

**Learning Point**: Loop closure + optimization = global consistency!

## Visualization

### Plot Scan Matching
```python
import matplotlib.pyplot as plt
import numpy as np

# Load scans for loop closure
scans_data = np.load("data/sim/ch7_slam_2d_square/scans.npz")
scan_0 = scans_data["scan_0"]
scan_40 = scans_data["scan_40"]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Before ICP
ax1.scatter(scan_0[:, 0], scan_0[:, 1], c='blue', s=10, label='Scan 0 (target)')
ax1.scatter(scan_40[:, 0], scan_40[:, 1], c='red', s=10, alpha=0.5, label='Scan 40 (source)')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Before ICP (Loop Closure)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# After ICP (aligned)
from core.slam import se2_apply, icp_point_to_point

pose_rel, _, _, _ = icp_point_to_point(scan_40, scan_0)
scan_40_aligned = np.array([se2_apply(pose_rel, pt) for pt in scan_40])

ax2.scatter(scan_0[:, 0], scan_0[:, 1], c='blue', s=10, label='Scan 0 (target)')
ax2.scatter(scan_40_aligned[:, 0], scan_40_aligned[:, 1], c='red', s=10, alpha=0.5, label='Scan 40 (aligned)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('After ICP (Aligned)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()
```

**Learning Point**: ICP aligns overlapping scans to detect loop closures!

## Parameter Effects

### Effect of Odometry Noise

| Drift Noise | Final Drift | SLAM Improvement | Notes |
|-------------|-------------|------------------|-------|
| 0.02m, 0.005rad (low) | ~0.1m | ~5× | High-quality odometry |
| 0.1m, 0.02rad (baseline) | ~0.5m | ~10× | Standard odometry |
| 0.3m, 0.05rad (high) | ~2.0m | ~20× | **Poor odometry (SLAM critical!)** |

**Generate comparison**:
```bash
python scripts/generate_ch7_slam_2d_dataset.py --preset low_drift
python scripts/generate_ch7_slam_2d_dataset.py --preset baseline
python scripts/generate_ch7_slam_2d_dataset.py --preset high_drift
```

**Learning Point**: Worse odometry → bigger drift → larger SLAM benefit!

### Effect of Trajectory Type

| Trajectory | Loop Closures | Drift Complexity | Notes |
|------------|---------------|------------------|-------|
| **Square** | 1 (end-start) | Simple | Single closed loop |
| **Figure-8** | Multiple | Complex | **Two loops, more constraints** |
| **Random Walk** | 0-few | Variable | **No guaranteed closure** |

**Generate comparison**:
```bash
python scripts/generate_ch7_slam_2d_dataset.py --preset baseline
python scripts/generate_ch7_slam_2d_dataset.py --preset figure8
```

**Learning Point**: More loop closures → better global consistency!

## Experiments

### Experiment 1: Quantify Drift Accumulation

**Objective**: Measure how odometry error grows with trajectory length.

**Procedure**:
1. Load ground truth and odometry poses
2. Compute error at each pose along trajectory
3. Plot error vs. distance traveled

**Expected Results**:
- Error grows approximately linearly with distance
- Final drift ~0.5m after 80m path (0.6% drift rate)

**Code**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
ground_truth = np.loadtxt("data/sim/ch7_slam_2d_square/ground_truth_poses.txt")
odometry = np.loadtxt("data/sim/ch7_slam_2d_square/odometry_poses.txt")

# Compute error at each pose
errors = np.linalg.norm(odometry[:, :2] - ground_truth[:, :2], axis=1)

# Compute cumulative distance
distances = np.zeros(len(ground_truth))
for i in range(1, len(ground_truth)):
    distances[i] = distances[i-1] + np.linalg.norm(ground_truth[i, :2] - ground_truth[i-1, :2])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(distances, errors, 'r-', linewidth=2)
plt.xlabel('Distance Traveled (m)')
plt.ylabel('Position Error (m)')
plt.title('Odometry Drift Accumulation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Drift rate: {errors[-1]/distances[-1]*100:.2f}% of distance")
```

**Learning Point**: Drift is linear in distance without loop closures!

### Experiment 2: Loop Closure Impact

**Objective**: Quantify how loop closure reduces final error.

**Procedure**:
1. Run odometry-only (no optimization)
2. Run SLAM with loop closure
3. Compare final errors

**Expected Results**:
- Odometry: ~0.5m final error
- SLAM: ~0.05m final error
- **10× improvement!**

**Code**: See Example 3 above

**Learning Point**: Loop closure is THE key to SLAM accuracy!

### Experiment 3: ICP Convergence

**Objective**: Study ICP alignment quality for loop closure verification.

**Procedure**:
1. Load loop closure scans (pose 0 and pose 40)
2. Run ICP with different initial guesses
3. Check convergence and residuals

**Expected Results**:
- Converges in ~10-20 iterations
- Final residual < 0.01
- Robust to moderate initial pose error

**Code**: See Example 2 above

**Learning Point**: ICP enables reliable loop closure detection!

## Performance Metrics (Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| **Trajectory** | Square loop | 20m × 20m |
| **Total Poses** | 41 | ~2m spacing |
| **Path Length** | ~80m | Perimeter of square |
| **Landmarks** | 50 | Static features |
| **Loop Closures** | 1 | End-to-start |
| **Odometry Drift** | 0.5m | Dead-reckoning final error |
| **SLAM Error** | ~0.05m | After optimization |
| **Improvement** | **10×** | SLAM vs. odometry |
| **ICP Iterations** | ~15 | For loop closure |
| **Scan Points** | ~24 per scan | Within 15m range |

**Comparison**:
- Low drift: 0.1m → 0.02m (5× improvement)
- Baseline: 0.5m → 0.05m (10× improvement)
- High drift: 2.0m → 0.1m (20× improvement)

**Key Insight**: Loop closure benefit increases with odometry quality!

## Book Connection

### Chapter 7: SLAM

This dataset directly demonstrates Chapter 7 SLAM concepts:

1. **Scan Matching (Section 7.2, Eqs. 7.10-7.11)**
   - ICP aligns point clouds
   - Finds optimal rigid transformation
   - Enables loop closure detection
   - **Key Insight**: Shared landmarks → pose constraints!

2. **Pose Graph Optimization (Section 7.3)**
   - Nodes: Robot poses
   - Edges: Odometry + loop closures
   - Optimization: Minimize constraint violations
   - **Key Insight**: Global consistency from local constraints!

3. **Loop Closure Detection**
   - Distance-based candidate selection
   - ICP verification
   - Adds constraints to pose graph
   - **Key Insight**: Revisiting places corrects drift!

4. **SLAM Pipeline**:
   ```
   Odometry → Scans → Scan Matching → Loop Detection → 
   Pose Graph → Optimization → Corrected Trajectory
   ```

**Formula**: Final SLAM error << Odometry drift (often 10-20× better!)

## Common Issues & Solutions

### Issue 1: ICP Doesn't Converge

**Symptoms**: Large residual, no convergence after 50 iterations

**Likely Cause**: Poor initial guess or insufficient overlap

**Solution**: Provide better initial guess or increase max_correspondence_distance:
```python
pose_rel, _, residual, converged = icp_point_to_point(
    scan_40, scan_0,
    initial_pose=np.array([0.5, 0.0, 0.0]),  # Better guess
    max_correspondence_distance=1.0  # Increase from 0.5
)
```

### Issue 2: No Loop Closures Detected

**Symptoms**: Loop closure file is empty or has no entries

**Likely Cause**: Trajectory doesn't actually close loop, or detection threshold too strict

**Solution**: Check trajectory type and relax distance threshold:
```python
# In generation script
loop_closures = detect_loop_closures(
    poses,
    min_index_diff=10,  # Reduce from 15
    max_distance=3.0  # Increase from 2.0
)
```

### Issue 3: SLAM Doesn't Improve Over Odometry

**Symptoms**: SLAM error ≈ Odometry error

**Likely Cause**: Loop closure constraints not strong enough, or optimization didn't converge

**Solution**: Check loop closure uncertainty and increase iterations:
```python
optimized_poses, info = optimize_factor_graph(
    graph,
    max_iterations=50  # Increase from 20
)
print(f"Optimization converged: {info['converged']}")
```

## Troubleshooting

### Error: Scan file not found

**Cause**: Scans are stored in compressed .npz format

**Fix**: Use np.load() with correct key format:
```python
scans_data = np.load("data/sim/ch7_slam_2d_square/scans.npz")
scan_i = scans_data[f"scan_{i}"]  # Note f-string format
```

### Warning: Large ICP residual

**Cause**: Scans don't overlap well (false loop closure)

**Fix**: This is expected for false positives. Filter by residual:
```python
if residual < 0.05:  # Threshold
    # Accept loop closure
else:
    # Reject as false positive
```

## Next Steps

After understanding pose graph SLAM:

1. **3D SLAM**: Extend to 3D with SE(3) poses
2. **Feature-Based SLAM**: Use extracted features instead of raw scans
3. **Visual SLAM**: Camera instead of LiDAR (Ch7 bundle adjustment example)
4. **Online SLAM**: Incremental updates as new data arrives
5. **Robust Estimation**: Handle outliers with M-estimators

## Citation

If you use this dataset in your research, please cite:

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={7},
  note={SLAM (Simultaneous Localization and Mapping)}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

