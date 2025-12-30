# Chapter 7: SLAM (Simultaneous Localization and Mapping)

## Overview

This module implements SLAM algorithms described in **Chapter 7** of *Principles of Indoor Positioning and Indoor Navigation*.

SLAM addresses the chicken-and-egg problem:
- **Localization** requires a map
- **Mapping** requires knowing the robot's location
- **SLAM** solves both simultaneously

The chapter implements:
- **Scan matching** (ICP, NDT) for relative pose estimation
- **Factor graph optimization** for trajectory correction
- **Loop closure detection** for drift reduction
- **Visual SLAM** with camera models and bundle adjustment

## Quick Start

```bash
# Run LiDAR SLAM example (inline data)
python -m ch7_slam.example_pose_graph_slam

# Run with pre-generated dataset
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square

# Run high-drift scenario (demonstrates SLAM value)
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_high_drift

# Run Visual SLAM example
python -m ch7_slam.example_bundle_adjustment
```

## 📂 Dataset Connection

| Example Script | Dataset | Description |
|----------------|---------|-------------|
| `example_pose_graph_slam.py` | `data/sim/ch7_slam_2d_square/` | Square trajectory with loop closure |
| `example_pose_graph_slam.py` | `data/sim/ch7_slam_2d_high_drift/` | High drift scenario (20x improvement with SLAM!) |

**Load dataset manually:**
```python
import numpy as np
import json
from pathlib import Path

path = Path("data/sim/ch7_slam_2d_square")
true_poses = np.loadtxt(path / "ground_truth_poses.txt")
odom_poses = np.loadtxt(path / "odometry_poses.txt")
landmarks = np.loadtxt(path / "landmarks.txt")
loop_closures = np.loadtxt(path / "loop_closures.txt")
scans = np.load(path / "scans.npz")
config = json.load(open(path / "config.json"))
```

## Equation Reference

### 7.3.1 Point-cloud based LiDAR SLAM - ICP

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `icp_point_to_point()` | `core/slam/scan_matching.py` | Eq. (7.10)-(7.11) | ICP alignment with SVD |

### 7.3.2 Feature-based LiDAR SLAM - NDT

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `build_ndt_map()` | `core/slam/ndt.py` | Eq. (7.12), (7.13) | Voxel mean and covariance (note: uses n_k-1) |
| `ndt_score()` | `core/slam/ndt.py` | Eq. (7.14)-(7.16) | Negative log-likelihood objective |
| `ndt_align()` | `core/slam/ndt.py` | Eq. (7.12)-(7.16) | Full NDT alignment (2D implementation) |

**Note**: The book presents NDT for 3D LiDAR (Eq. 7.9), but this implementation uses 2D for pedagogical clarity.

### Pose Graph Optimization (GraphSLAM)

| Function | Location | Reference | Description |
|----------|----------|----------|-------------|
| `create_odometry_factor()` | `core/slam/factors.py` | Section 7.1.2, Table 7.2 | Connect consecutive poses |
| `create_loop_closure_factor()` | `core/slam/factors.py` | Section 7.3.5, Eq. (7.22) | Loop closure constraints |
| `create_prior_factor()` | `core/slam/factors.py` | Section 7.1.2 | Anchor first pose |

### 7.4 Visual SLAM

| Function | Location | Equation | Description |
|----------|----------|----------|-------------|
| `project_point()` | `core/slam/camera.py` | Eq. (7.40), (7.41), (7.42)-(7.43) | Full camera projection + distortion |
| `distort_normalized()` | `core/slam/camera.py` | Eq. (7.41) | Distortion model (k1,k2,k3,p1,p2) |
| `create_reprojection_factor()` | `core/slam/factors.py` | Eq. (7.70) | Bundle adjustment reprojection error |

**Note on Bundle Adjustment (Section 7.4.2)**: The book's Eq. (7.70) uses full SE(3) poses with rotation matrix R_i and translation vector t_i. This implementation uses SE(2) planar poses [x, y, yaw] for pedagogical consistency with other 2D SLAM examples. The core principle (minimizing reprojection error) remains the same.

## Expected Output

### LiDAR Pose Graph SLAM

Running `python -m ch7_slam.example_pose_graph_slam` produces:

```
====================================================================================
CHAPTER 7: 2D POSE GRAPH SLAM EXAMPLE
====================================================================================

1. Generating square trajectory...
   Generated 22 poses in square loop

2. Generating environment landmarks...
   Generated 60 landmarks

3. Generating LiDAR scans...
   Generated 22 scans (avg 25.4 points/scan)

4. Simulating odometry with drift...
   Final drift (without SLAM): 1.234 m

5. Detecting loop closures...
   Detected 1 loop closures

6. Building pose graph...
   Factors: 1 prior + 21 odometry + 1 loop closures

7. Optimizing pose graph...
   Error reduction: 99.73%

8. Evaluating results...
   Odometry RMSE: 0.8234 m
   Optimized RMSE: 0.0567 m
   Improvement: 93.11%
```

**Visual Output:**

![Pose Graph SLAM Results](figs/pose_graph_slam_results.png)

*This figure shows two plots:*
- **Left:** Trajectories comparing ground truth (green), odometry with drift (red dashed), and optimized SLAM (blue)
- **Right:** Position error over time showing how loop closure corrects accumulated drift

### Visual Bundle Adjustment

Running `python -m ch7_slam.example_bundle_adjustment` produces:

```
================================================================================
CHAPTER 7: VISUAL BUNDLE ADJUSTMENT EXAMPLE
================================================================================

1. Setting up camera parameters...
   Camera: fx=500.0, fy=500.0

2. Generating ground truth...
   Generated 8 camera poses (circular trajectory)
   Generated 15 3D landmarks

3. Simulating camera observations...
   Generated 117 observations

4. Creating noisy initial estimates...
   Initial pose RMSE: 0.0696 m
   Initial landmark RMSE: 0.1708 m

5. Running bundle adjustment optimization...
   Error reduction: 98.61%
```

**Visual Output:**

![Bundle Adjustment Results](figs/bundle_adjustment_results.png)

*This figure shows three plots:*
- **Left:** Camera trajectory and 3D landmarks (top view)
- **Middle:** Position errors before and after optimization
- **Right:** Optimization convergence curve

## Performance Summary

| Method | Input | RMSE | Improvement |
|--------|-------|------|-------------|
| **Odometry only** | Wheel encoders | ~1-2 m | Baseline |
| **Pose Graph SLAM** | + Loop closures | ~0.05 m | 93-95% |
| **Bundle Adjustment** | Camera images | ~0.01 m | 98%+ |

## Key Concepts

### 7.3.1 ICP (Iterative Closest Point)

- Minimizes point-to-point distances between scans (Eq. 7.10)
- Uses SVD to solve rotation and translation
- **Used for**: Scan-to-scan matching and loop closure detection

### 7.3.2 NDT (Normal Distributions Transform)

- Represents voxels as 3D normal distributions (Eq. 7.12-7.13)
- Maximum likelihood estimation via nonlinear optimization (Eq. 7.14-7.16)
- More robust to noise than raw point matching

### Pose Graph Structure (GraphSLAM - Section 7.1.2)

```
pose_0 --odom--> pose_1 --odom--> ... --odom--> pose_N
  ^                                                  |
  +--------------- loop closure (Eq. 7.22) ----------+
```

Based on GraphSLAM (Section 7.1.2): poses and landmarks form graph nodes, measurements create edges (constraints). The SLAM problem is solved by finding the configuration that best satisfies all constraints through sparse graph optimization.

**Loop Closure Constraints (Section 7.3.5, Eq. 7.22):**

When the robot returns to a previously visited location (e.g., completing a loop), a loop closure is detected by scan matching. The close-loop constraint enforces consistency between:
- The **scan-matched transform** ΔT_ij' (from ICP/NDT)
- The **pose chain transform** T_i^{-1} T_j (from odometry)

The residual from Eq. (7.22):
```
residual = ln((ΔT_ij')^{-1} T_i^{-1} T_j)^∨
```
where T_i is an earlier pose, T_j is the current pose, and ΔT_ij' is the observed relative transform from scan matching. This constraint "bends" the trajectory to close loops and eliminate accumulated drift.

### Bundle Adjustment (Section 7.4.2)

- **Variables**: Camera poses {Ri, ti} + 3D landmarks {pk}
- **Objective**: Minimize sum of squared reprojection errors (Eq. 7.70)
- **Challenge**: Scale uncertainty in monocular vision (Section 7.4.2)
- **Result**: Globally consistent reconstruction across multiple views

## File Structure

```
ch7_slam/
├── README.md                       # This file (student documentation)
├── example_pose_graph_slam.py      # 2D LiDAR SLAM pipeline demo
├── example_bundle_adjustment.py    # Visual bundle adjustment demo
└── figs/                           # Generated figures
    ├── pose_graph_slam_results.png
    └── bundle_adjustment_results.png

core/slam/
├── scan_matching.py                # ICP algorithm (Section 7.3.1)
├── ndt.py                          # NDT algorithm (Section 7.3.2)
├── factors.py                      # Pose graph factors and bundle adjustment
├── camera.py                       # Camera projection and distortion (Section 7.4)
├── se2.py                          # SE(2) transformations for 2D SLAM
└── types.py                        # Type definitions
```

**Note**: The current implementation is **2D SLAM** (SE(2) poses) for educational clarity, while the book's Chapter 7 discusses general 3D LiDAR SLAM. The mathematical principles (ICP, NDT, pose graph optimization) apply to both 2D and 3D cases.

## Not Implemented (Future Work)

The following topics from Chapter 7 are **not currently implemented** in this repository:

### 7.3.3 LOAM (LiDAR Odometry and Mapping)
- **Book coverage**: Section 7.3.3, Eqs. (7.17)-(7.19)
- **What it is**: State-of-the-art feature-based LiDAR SLAM using edge and planar features
- **Key innovations**:
  - Scan-to-map matching (vs scan-to-scan) to reduce drift
  - Two-step approach: scan-to-scan odometry + scan-to-map refinement
  - Point-to-line and point-to-plane distance metrics
- **Why not implemented**: Significantly more complex than ICP/NDT; requires feature extraction, curvature analysis, and two-stage optimization
- **Future work**: Could add as `core/slam/loam.py` with `extract_edge_features()`, `extract_planar_features()`, and `loam_align()`

### 7.3.4 Advanced LiDAR SLAM Topics
- **Motion distortion compensation** (Eqs. 7.20-7.21): Compensating for ego-motion during LiDAR scan sweep using IMU
- **Dynamic object handling**: Detecting and filtering moving objects in SLAM
- **LiDAR-IMU integration**: Tightly-coupled LiDAR-inertial odometry (e.g., LIO-SAM)
- **Why not implemented**: These are advanced topics requiring IMU integration and real-time processing considerations

### 7.4.3 RGB-D SLAM
- **Book coverage**: Section 7.4.3
- **What it is**: SLAM using RGB-D cameras (e.g., Microsoft Kinect) with depth information
- **Why not implemented**: Requires depth sensor data and integration of visual and depth information
- **Future work**: Could extend visual SLAM examples to use depth measurements

### 7.4.4 Advanced Visual SLAM Topics
- **Stereo SLAM**: Using stereo camera pairs for depth estimation
- **Deep learning features**: Neural network-based feature detection and tracking
- **Visual-inertial fusion**: Camera-IMU integration (e.g., VINS-Mono)
- **Why not implemented**: Advanced topics beyond the scope of introductory examples

### 7.4.5 LiDAR-Camera Integration
- **Book coverage**: Section 7.4.5
- **What it is**: Mapping camera pixels to LiDAR point clouds for colored 3D maps
- **Why not implemented**: Requires sensor calibration and multi-modal data fusion

### What IS Implemented

This repository focuses on **foundational SLAM concepts** for educational purposes:

✅ **ICP** (Section 7.3.1): Point-to-point scan matching  
✅ **NDT** (Section 7.3.2): Probabilistic scan matching with normal distributions  
✅ **Pose Graph Optimization** (Section 7.3.5): Factor graph-based trajectory optimization  
✅ **Loop Closure** (Section 7.3.5): Drift correction via loop detection  
✅ **Camera Models** (Section 7.4.1): Pinhole projection with distortion  
✅ **Bundle Adjustment** (Section 7.4.2): Visual SLAM with reprojection error minimization  

These implementations provide a solid foundation for understanding SLAM principles. For production systems, consider established frameworks like:
- **LiDAR SLAM**: LIO-SAM, LeGO-LOAM, LOAM
- **Visual SLAM**: ORB-SLAM3, VINS-Mono, OpenVSLAM
- **Multi-sensor**: Cartographer, RTAB-Map

## References

- **Chapter 7**: Indoor Simultaneous Localization and Mapping (SLAM)
  - Section 7.1.2: SLAM Frameworks and Evolution (GraphSLAM)
  - Section 7.3: LiDAR SLAM
    - Section 7.3.1: Point-cloud based LiDAR SLAM - ICP ✅
    - Section 7.3.2: Feature-based LiDAR SLAM - NDT ✅
    - Section 7.3.3: Feature-based LiDAR SLAM - LOAM ❌ (not implemented)
    - Section 7.3.4: Challenges of LiDAR SLAM ❌ (not implemented)
    - Section 7.3.5: Close-loop Constraints ✅
  - Section 7.4: Visual SLAM
    - Section 7.4.1: Monocular Camera (pinhole model, distortion) ✅
    - Section 7.4.2: Monocular SLAM (bundle adjustment) ✅
    - Section 7.4.3: RGB-D SLAM ❌ (not implemented)
    - Section 7.4.4: Challenges of Visual SLAM ❌ (not implemented)
    - Section 7.4.5: LiDAR-Camera Integration ❌ (not implemented)

