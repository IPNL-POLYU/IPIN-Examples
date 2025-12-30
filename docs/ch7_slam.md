# Chapter 7: SLAM Technologies

## Overview

This document provides a comprehensive guide to the SLAM (Simultaneous Localization and Mapping) implementations from **Chapter 7** of *Principles of Indoor Positioning and Indoor Navigation*.

SLAM solves the fundamental chicken-and-egg problem:
- **Localization** requires a known map
- **Mapping** requires knowing the robot's position
- **SLAM** solves both problems **simultaneously**

## Implemented Algorithms

### 1. Scan Matching (Section 7.3 - LiDAR SLAM)

#### 1.1 ICP - Iterative Closest Point (Section 7.3.1)

**Purpose:** Align two point clouds by finding optimal rigid body transformation.

**Key Equations:**
- **Eq. (7.10)**: Point-to-point matching objective
  ```
  E(R, t) = Σᵢ ||pᵢ' - (R·pᵢ + t)||²
  ```
- **Eq. (7.11)**: Correspondence association with distance threshold ε

**Implementation:**
```python
from core.slam import icp_point_to_point

pose, iterations, residual, converged = icp_point_to_point(
    source_scan=scan_current,
    target_scan=scan_previous,
    initial_pose=np.array([0.0, 0.0, 0.0]),  # [x, y, yaw]
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=0.5,  # Eq. (7.11) threshold ε
)
```

**Algorithm Steps:**
1. Transform source points by current pose estimate
2. Find nearest neighbors (correspondence search)
3. Filter by distance threshold (Eq. 7.11)
4. Compute optimal transformation using SVD
5. Update pose estimate
6. Repeat until convergence

**Key Functions:**
- `find_correspondences()` - Implements Eq. (7.11)
- `compute_icp_residual()` - Implements Eq. (7.10)
- `align_svd()` - SVD-based rigid alignment
- `icp_point_to_point()` - Complete ICP pipeline

**Tests:** `tests/core/slam/test_scan_matching.py` (37 tests)

#### 1.2 NDT - Normal Distributions Transform (Section 7.3.2)

**Purpose:** Probabilistic scan matching using voxel-based Gaussian distributions.

**Key Equations:**
- **Eqs. (7.12)-(7.13)**: Per-voxel mean μₖ and covariance Σₖ
- **Eqs. (7.14)-(7.15)**: Likelihood of points under voxel Gaussians
- **Eq. (7.16)**: MLE optimization (minimize negative log-likelihood)

**Implementation:**
```python
from core.slam import ndt_align, build_ndt_map

# Build NDT map from reference scan
ndt_map = build_ndt_map(
    points=reference_scan,
    voxel_size=1.0,
    min_points_per_voxel=3
)

# Align current scan to NDT map
pose, iterations, score, converged = ndt_align(
    source_scan=current_scan,
    target_scan=reference_scan,
    initial_pose=np.array([0.0, 0.0, 0.0]),
    voxel_size=1.0,
    max_iterations=50,
)
```

**Algorithm Steps:**
1. Discretize reference scan into voxel grid
2. Compute Gaussian (μₖ, Σₖ) for each voxel (Eqs. 7.12-7.13)
3. Transform source points by current pose
4. Compute negative log-likelihood score (Eqs. 7.14-7.15)
5. Optimize pose via gradient descent (Eq. 7.16)
6. Repeat until convergence

**Key Functions:**
- `build_ndt_map()` - Implements Eqs. (7.12)-(7.13)
- `ndt_score()` - Implements Eqs. (7.14)-(7.15)
- `ndt_gradient()` - Numerical gradient for optimization
- `ndt_align()` - Implements Eq. (7.16)

**Tests:** `tests/core/slam/test_ndt.py` (31 tests)

**Advantages over ICP:**
- More robust to noise and outliers
- Smoother objective function (better convergence)
- Handles partial overlaps better

---

### 2. Pose Graph Optimization (Section 7.1.2 - GraphSLAM)

**Purpose:** Globally optimize robot trajectory using odometry and loop closure constraints.

**Note:** Pose graph optimization is the back-end of GraphSLAM (Section 7.1.2, Table 7.2). Loop closure constraints are detailed in Section 7.3.5 (Eq. 7.22).

**Framework:** Factor graph with poses as variables and constraints as factors.

**Implementation:**
```python
from core.slam import create_pose_graph

# Define measurements
poses = [pose0, pose1, pose2, ..., poseN]  # Initial estimates
odometry = [(0, 1, rel_pose01), (1, 2, rel_pose12), ...]  # Consecutive
loop_closures = [(10, 0, rel_pose10_0)]  # Non-consecutive (from ICP/NDT)

# Build and optimize
graph = create_pose_graph(
    poses=poses,
    odometry_measurements=odometry,
    loop_closures=loop_closures,
    prior_pose=np.array([0, 0, 0]),  # Anchor first pose
)

optimized_poses, error_history = graph.optimize(
    method="gauss_newton",
    max_iterations=50,
    tol=1e-6,
)
```

**Factor Types:**

#### 2.1 Odometry Factor
**Purpose:** Connect consecutive poses from dead-reckoning.

**Residual:**
```
r = (xᵢ⁻¹ ⊕ xⱼ) - zᵢⱼ
```
where `zᵢⱼ` is measured relative pose, `⊕` is SE(2) composition.

**Usage:**
```python
from core.slam import create_odometry_factor

factor = create_odometry_factor(
    pose_id_from=i,
    pose_id_to=i+1,
    relative_pose=np.array([dx, dy, dyaw]),  # Measured
    information=np.diag([1.0, 1.0, 0.1]),  # Inverse covariance
)
```

#### 2.2 Loop Closure Factor (Section 7.3.5)
**Purpose:** Connect non-consecutive poses detected via scan matching.

**Close-loop constraint (Eq. 7.22):**
```
residual = ln((ΔT_ij')^{-1} T_i^{-1} T_j)^∨
```
where ΔT_ij' is the scan-matched transform and T_i, T_j are poses.

**Usage:**
```python
from core.slam import create_loop_closure_factor

# Detect loop closure using ICP
rel_pose, _, _, converged = icp_point_to_point(scan_i, scan_j)
if converged:
    factor = create_loop_closure_factor(
        pose_id_from=i,
        pose_id_to=j,
        relative_pose=rel_pose,
        information=np.eye(3) * 10.0,  # Higher confidence from ICP
    )
```

#### 2.3 Prior Factor
**Purpose:** Anchor pose to known value (fix first pose, GPS measurement).

**Residual:**
```
r = x - x_prior
```

**Usage:**
```python
from core.slam import create_prior_factor

# Fix first pose to prevent gauge freedom
factor = create_prior_factor(
    pose_id=0,
    prior_pose=np.array([0, 0, 0]),
    information=np.eye(3) * 1e6,  # Strong prior
)
```

#### 2.4 Landmark Factor
**Purpose:** Connect pose to landmark via range-bearing measurement.

**Residual:**
```
r = h(xᵢ, lₖ) - zᵢₖ
```
where `h` computes expected [range, bearing] from pose to landmark.

**Usage:**
```python
from core.slam import create_landmark_factor

factor = create_landmark_factor(
    pose_id=i,
    landmark_id=k,
    measurement=np.array([range, bearing]),
    information=np.diag([10.0, 1.0]),  # Range/bearing uncertainty
)
```

**Tests:** `tests/core/slam/test_factors.py` (18 tests)

---

### 3. Visual SLAM (Section 7.4)

#### 3.1 Camera Model

**Purpose:** Project 3D points to 2D pixels with lens distortion.

**Key Equations:**
- **Eq. (7.40)**: Pinhole projection
  ```
  u = fx · x_d + cx
  v = fy · y_d + cy
  ```
- **Eqs. (7.43)-(7.46)**: Brown-Conrady distortion model
  - Radial distortion: `1 + k1·r² + k2·r⁴`
  - Tangential distortion: `[2p1xy + p2(r²+2x²), p1(r²+2y²) + 2p2xy]`

**Implementation:**
```python
from core.slam import CameraIntrinsics, project_point

# Define camera
intrinsics = CameraIntrinsics(
    fx=500.0, fy=500.0,  # Focal lengths
    cx=320.0, cy=240.0,  # Principal point
    k1=-0.1, k2=0.01,    # Radial distortion
    p1=0.001, p2=0.001,  # Tangential distortion
)

# Project 3D point to pixel
point_3d = np.array([1.0, 0.5, 5.0])  # [X, Y, Z] in camera frame
pixel = project_point(intrinsics, point_3d)  # [u, v]
```

**Key Functions:**
- `distort_normalized()` - Implements Eqs. (7.43)-(7.46)
- `undistort_normalized()` - Inverse distortion (Newton-Raphson)
- `project_point()` - Full projection pipeline (Eq. 7.40 + distortion)
- `unproject_pixel()` - Inverse projection (pixel → 3D ray)
- `compute_reprojection_error()` - Core residual for BA

**Tests:** `tests/core/slam/test_camera.py` (28 tests)

#### 3.2 Bundle Adjustment (Section 7.4.2)

**Purpose:** Jointly optimize camera poses and 3D landmark positions.

**Key Equations:**
- **Eq. (7.70)**: Bundle adjustment objective
  ```
  E = Σᵢⱼ ||uᵢⱼ - π(Tᵢ, Lⱼ)||²
  ```
  where `uᵢⱼ` is observed pixel, `π(Tᵢ, Lⱼ)` is projected landmark
- **Eq. (7.68)**: Robust kernel formulation
- **Eq. (7.69)**: Reprojection error definition

**Implementation:**
```python
from core.slam import create_reprojection_factor
from core.estimators.factor_graph import FactorGraph

graph = FactorGraph()

# Add camera pose variables
for i, pose in enumerate(initial_poses):
    graph.add_variable(i, pose)  # [x, y, yaw] for 2D

# Add landmark variables
for j, landmark in enumerate(initial_landmarks):
    graph.add_variable(n_poses + j, landmark)  # [x, y, z]

# Add reprojection factors for all observations
for (pose_id, landmark_id, observed_pixel) in observations:
    factor = create_reprojection_factor(
        camera_pose_id=pose_id,
        landmark_id=n_poses + landmark_id,
        observed_pixel=observed_pixel,  # [u, v]
        camera_intrinsics=intrinsics,
        information=np.eye(2),  # Pixel uncertainty
    )
    graph.add_factor(factor)

# Optimize
optimized_vars, error_history = graph.optimize(
    method="gauss_newton",
    max_iterations=20,
)
```

**Key Concepts:**
- **Reprojection error**: 2D distance between observed and projected pixel
- **Joint optimization**: Refines both camera trajectory and 3D map
- **Sparsity**: Large but sparse Jacobian matrix (efficient solvers needed)

**Example:** `ch7_slam/example_bundle_adjustment.py`

**Tests:** `tests/core/slam/test_camera.py::TestComputeReprojectionError`

---

## SE(2) Operations

**Purpose:** 2D rigid body transformations for pose manipulation.

**Representation:** Pose = `[x, y, yaw]` (position + heading)

**Operations:**
```python
from core.slam import (
    se2_compose,    # p1 ⊕ p2 (composition)
    se2_inverse,    # p⁻¹ (inverse)
    se2_apply,      # Transform points
    se2_relative,   # p_from⁻¹ ⊕ p_to
    wrap_angle,     # Normalize angle to [-π, π)
)

# Compose two poses
pose_composed = se2_compose(pose1, pose2)

# Inverse transformation
pose_inv = se2_inverse(pose)

# Transform points
points_transformed = se2_apply(pose, points)  # (N, 2) array

# Relative pose
rel_pose = se2_relative(pose_from, pose_to)
```

**Tests:** `tests/core/slam/test_se2.py` (49 tests)

---

## Complete SLAM Pipeline

### Example 1: Pose Graph SLAM

**File:** `ch7_slam/example_pose_graph_slam.py`

**Pipeline:**
1. **Generate trajectory** (ground truth)
2. **Simulate LiDAR scans** at each pose
3. **Add odometry noise** (simulate drift)
4. **Detect loop closures** using ICP
5. **Build pose graph** (odometry + loop closure factors)
6. **Optimize** using Gauss-Newton
7. **Visualize** results

**Run:**
```bash
python -m ch7_slam.example_pose_graph_slam
```

**Expected Results:**
- Odometry RMSE: ~0.3-0.5 m (with drift)
- SLAM RMSE: ~0.05-0.1 m (after optimization)
- Error reduction: 80-95%

### Example 2: Bundle Adjustment

**File:** `ch7_slam/example_bundle_adjustment.py`

**Pipeline:**
1. **Generate camera trajectory** (circular)
2. **Generate 3D landmarks** (scene features)
3. **Simulate observations** (pixel coordinates)
4. **Add noise** to initial estimates
5. **Build factor graph** (reprojection factors)
6. **Optimize** poses + landmarks
7. **Visualize** convergence

**Run:**
```bash
python -m ch7_slam.example_bundle_adjustment
```

**Note:** Bundle adjustment optimization is numerically challenging and may require careful tuning.

---

## Equation-to-Code Mapping

Complete mappings are in `docs/equation_index.yml`. Quick reference:

| Equation | Description | Implementation |
|----------|-------------|----------------|
| Eq. (7.10) | ICP objective | `compute_icp_residual()` |
| Eq. (7.11) | ICP correspondence | `find_correspondences()` |
| Eqs. (7.12)-(7.13) | NDT voxel stats | `build_ndt_map()` |
| Eqs. (7.14)-(7.15) | NDT score | `ndt_score()` |
| Eq. (7.16) | NDT optimization | `ndt_align()` |
| Eq. (7.40) | Camera projection | `project_point()` |
| Eqs. (7.43)-(7.46) | Lens distortion | `distort_normalized()` |
| Eqs. (7.68)-(7.70) | Bundle adjustment | `create_reprojection_factor()` |

---

## Testing

**Total SLAM Tests:** 163 (all passing ✅)
- SE(2) operations: 49 tests
- ICP scan matching: 37 tests
- NDT alignment: 31 tests
- Pose graph factors: 18 tests
- Camera model: 28 tests

**Run all tests:**
```bash
pytest tests/core/slam/ -v
```

**Run specific module:**
```bash
pytest tests/core/slam/test_scan_matching.py -v  # ICP tests
pytest tests/core/slam/test_ndt.py -v             # NDT tests
pytest tests/core/slam/test_factors.py -v         # Factor tests
pytest tests/core/slam/test_camera.py -v          # Camera tests
```

---

## API Reference

### Core Modules

- **`core/slam/se2.py`** - SE(2) transformations
- **`core/slam/scan_matching.py`** - ICP algorithm
- **`core/slam/ndt.py`** - NDT alignment
- **`core/slam/factors.py`** - Pose graph factors
- **`core/slam/camera.py`** - Camera projection & distortion
- **`core/slam/types.py`** - Data structures (Pose2, CameraIntrinsics)

### Examples

- **`ch7_slam/example_pose_graph_slam.py`** - Complete 2D SLAM pipeline
- **`ch7_slam/example_bundle_adjustment.py`** - Visual BA demonstration

### Tests

- **`tests/core/slam/test_se2.py`** - SE(2) operations
- **`tests/core/slam/test_scan_matching.py`** - ICP algorithm
- **`tests/core/slam/test_ndt.py`** - NDT alignment
- **`tests/core/slam/test_factors.py`** - Pose graph factors
- **`tests/core/slam/test_camera.py`** - Camera model

---

## Design Notes

### Scope

This is a **reference implementation** for educational purposes:
- ✅ Demonstrates key SLAM algorithms from Chapter 7
- ✅ Equation-level traceability (book → code)
- ✅ Comprehensive unit tests
- ✅ Working examples on synthetic data
- ❌ NOT a production SLAM system
- ❌ NOT optimized for real-time performance
- ❌ NOT a full robotics framework

### Simplifications

1. **2D-first approach**: Most examples use SE(2) (3 DOF) rather than SE(3) (6 DOF)
2. **Synthetic data**: Examples use generated data with known ground truth
3. **Simplified models**: No full image processing, no complex sensor models
4. **Numerical Jacobians**: Bundle adjustment uses finite differences (slower but clearer)

### Future Extensions

Potential additions for completeness (not required):
- LOAM-style feature extraction (Eqs. 7.17-7.19)
- IMU integration for scan matching initialization
- Analytical Jacobians for faster bundle adjustment
- Real dataset examples (e.g., TUM RGB-D, KITTI)

---

## References

- **Book**: *Principles of Indoor Positioning and Indoor Navigation*, Chapter 7
- **Design Document**: `references/design_doc.md`, Section 7.6
- **Equation Index**: `docs/equation_index.yml`
- **README**: `ch7_slam/README.md`

---

## Support

For issues or questions:
1. Check equation mappings in `docs/equation_index.yml`
2. Review test files for usage examples
3. Run examples with `python -m ch7_slam.example_*`
4. Read implementation comments in `core/slam/*.py`

**Author:** Navigation Engineer  
**Date:** 2024  
**Version:** 0.1.0


