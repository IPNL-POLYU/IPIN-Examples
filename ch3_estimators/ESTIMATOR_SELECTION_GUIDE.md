# Estimator Selection Guide

## Overview

This guide helps you choose the right state estimator for your indoor positioning application. Each estimator has trade-offs in accuracy, computational cost, and applicability.

---

## Quick Selection Table

| Scenario | Recommended Estimator | Alternative | Why |
|----------|----------------------|-------------|-----|
| **Linear system, Gaussian noise** | Kalman Filter (KF) | - | Optimal for this case |
| **Nonlinear, moderate** | Extended Kalman Filter (EKF) | UKF | Fast, analytically tractable |
| **Nonlinear, severe** | Unscented Kalman Filter (UKF) | Particle Filter | Better linearization than EKF |
| **Multi-modal posterior** | Particle Filter (PF) | - | Can represent multiple hypotheses |
| **Non-Gaussian noise** | Particle Filter (PF) | Robust LS | No Gaussian assumption |
| **Batch optimization** | Factor Graph Optimization (FGO) | Batch LS | Smoother, globally optimal |
| **Static positioning** | Least Squares (LS) | Iterative LS | Single-shot solution |
| **Positioning with outliers** | Robust Least Squares | PF with resampling | Downweights bad measurements |

---

## Estimator Details

### 1. Least Squares (LS) Methods

**When to use:**
- Static positioning (no dynamics)
- Single time instant
- Overdetermined system (more measurements than unknowns)
- Batch processing acceptable

**Variants:**
- **Linear LS**: When problem is linear or after linearization
- **Weighted LS**: Different measurement accuracies
- **Iterative LS**: Nonlinear problems (Gauss-Newton)
- **Robust LS**: Presence of outliers (IRLS with Huber/Cauchy/Tukey)

**Pros:**
- ✅ Simple to implement
- ✅ No initialization required
- ✅ Analytically solvable (linear case)
- ✅ Well-understood theory

**Cons:**
- ❌ No time filtering (doesn't use dynamics)
- ❌ No uncertainty propagation
- ❌ Requires all measurements simultaneously
- ❌ Sensitive to outliers (unless robust variant)

**Complexity:** O(n³) for n unknowns

**Example use case:**
```
TOA positioning from UWB ranges
- 4+ anchors
- Static target
- Need quick position fix
→ Use Iterative Least Squares
```

---

### 2. Kalman Filter (KF)

**When to use:**
- Linear dynamics and measurements
- Gaussian noise
- Real-time filtering
- Need uncertainty estimates

**Requirements:**
- System must be linear: x_{k+1} = F x_k + w, z_k = H x_k + v
- w ~ N(0, Q), v ~ N(0, R)

**Pros:**
- ✅ **Optimal** for linear Gaussian systems
- ✅ Recursive (online processing)
- ✅ Provides uncertainty (covariance)
- ✅ Very fast: O(n³) per update
- ✅ Theoretically well-understood

**Cons:**
- ❌ **Only linear systems**
- ❌ Requires accurate noise models
- ❌ Poor with outliers
- ❌ Diverges if model is wrong

**Complexity:** O(n³) per time step for n-dimensional state

**Example use case:**
```
1D constant velocity tracking
- Position measurements (linear)
- Gaussian noise
- Want velocity estimate too
→ Use Kalman Filter
```

**Code:**
```python
from core.estimators import KalmanFilter
from core.models import ConstantVelocity1D

# Setup
F = ConstantVelocity1D.F(dt=0.1)
Q = ConstantVelocity1D.Q(dt=0.1, q=0.5)
H = np.array([[1, 0]])  # Observe position only
R = np.array([[0.5**2]])  # Measurement noise

kf = KalmanFilter(F, Q, H, R, x0, P0)

# Run
for z in measurements:
    kf.predict(dt=0.1)
    kf.update(z)
```

---

### 3. Extended Kalman Filter (EKF)

**When to use:**
- Nonlinear dynamics or measurements
- Moderate nonlinearity
- Need real-time performance
- Gaussian noise assumption acceptable

**How it works:**
- Linearizes nonlinear functions using Jacobians
- Applies KF to linearized system

**Pros:**
- ✅ Handles nonlinear systems
- ✅ Fast: O(n³) per update (same as KF)
- ✅ Widely used, well-tested
- ✅ Recursive (online)
- ✅ Provides uncertainty

**Cons:**
- ❌ **Linearization errors** can cause divergence
- ❌ Requires Jacobian computation (analytical or numerical)
- ❌ Poor for highly nonlinear systems
- ❌ Only first-order linearization
- ❌ Sensitive to initial conditions

**Complexity:** O(n³) per time step

**When NOT to use:**
- Highly nonlinear measurements (use UKF)
- Large initial uncertainty (use PF)
- Multi-modal distributions (use PF)

**Example use case:**
```
2D positioning from range-bearing measurements
- Nonlinear: range = sqrt(dx² + dy²), bearing = atan2(dy, dx)
- Moderate nonlinearity (smooth trajectory)
- Need real-time performance
→ Use Extended Kalman Filter
```

**Code:**
```python
from core.estimators import ExtendedKalmanFilter
from core.models import ConstantVelocity2D, RangeBearingMeasurement2D

# Setup
model_motion = ConstantVelocity2D()
model_meas = RangeBearingMeasurement2D(landmarks)

ekf = ExtendedKalmanFilter(
    process_model=model_motion.f,
    process_jacobian=model_motion.F,
    measurement_model=model_meas.h,
    measurement_jacobian=model_meas.H,
    Q=lambda dt: model_motion.Q(dt, q=0.5),
    R=lambda: np.diag([0.5**2, 0.05**2] * len(landmarks)),
    x0=x0, P0=P0
)
```

---

### 4. Unscented Kalman Filter (UKF)

**When to use:**
- Nonlinear systems with severe nonlinearity
- EKF performs poorly
- Can't compute Jacobians easily
- Computational cost acceptable (≈2× EKF)

**How it works:**
- Unscented Transform: propagates "sigma points" through nonlinearity
- Captures mean and covariance to 2nd order (better than EKF's 1st order)

**Pros:**
- ✅ **No Jacobians needed!**
- ✅ Better than EKF for strong nonlinearity
- ✅ 2nd-order accurate (EKF is 1st-order)
- ✅ Same asymptotic complexity as EKF
- ✅ Handles discontinuities better

**Cons:**
- ❌ 2-3× slower than EKF (more function evaluations)
- ❌ Still assumes Gaussian noise
- ❌ Can't handle multi-modal posteriors
- ❌ Tuning parameters (α, β, κ)

**Complexity:** O(n³) per time step, but ~2-3× EKF due to sigma points

**When to prefer over EKF:**
- Jacobians difficult to derive
- Severe nonlinearity
- Measurement function has discontinuities

**Example use case:**
```
Highly curved trajectory with range measurements
- Large angular rates (high nonlinearity)
- EKF diverges
- Still need real-time performance
→ Use Unscented Kalman Filter
```

**Performance comparison (from ch3_estimators/example_comparison.py):**
```
Scenario: 2D tracking, range measurements, curved path
EKF RMSE:  0.32 m   (0.016 s)
UKF RMSE:  0.31 m   (0.017 s)  ← Slightly better
```

---

### 5. Particle Filter (PF)

**When to use:**
- Highly nonlinear systems
- Non-Gaussian noise
- Multi-modal posterior distributions
- Ambiguity (multiple hypotheses)
- EKF/UKF diverge

**How it works:**
- Represents posterior with particles (samples)
- Each particle is a hypothesis
- Importance sampling + resampling

**Pros:**
- ✅ **No Gaussian assumption!**
- ✅ **Multi-modal posteriors**
- ✅ Handles severe nonlinearity
- ✅ Flexible noise models
- ✅ Can represent any distribution

**Cons:**
- ❌ **Very slow:** O(N×m) for N particles, m measurements
- ❌ Particle degeneracy problem
- ❌ Requires many particles (100-1000+)
- ❌ Not suitable for high-dimensional states (>10D)
- ❌ Random (non-deterministic)

**Complexity:** O(N×m) per time step for N particles

**Number of particles needed:**
- 2D positioning: 100-500 particles
- 3D positioning: 500-2000 particles
- 6D pose (pos+ori): 1000-5000 particles

**When to prefer over EKF/UKF:**
- Multi-modal ambiguity (e.g., symmetry in environment)
- Heavy-tailed noise (outliers common)
- Highly nonlinear measurements
- Computational resources available

**Example use case:**
```
Robot localization with symmetric environment
- Multiple valid pose hypotheses
- Need to track all until disambiguated
- Have computational power
→ Use Particle Filter
```

**Performance comparison:**
```
Scenario: 2D tracking (from ch3_estimators/example_comparison.py)
EKF RMSE:  0.32 m   (0.016 s)
PF RMSE:   0.45 m   (1.178 s)  ← Worse here (Gaussian case)
```
Note: PF underperforms EKF in Gaussian scenarios but excels with ambiguity/outliers.

---

### 6. Factor Graph Optimization (FGO)

**When to use:**
- Batch or smoothing problem
- Have all data (not streaming)
- Need globally optimal solution
- SLAM, mapping, trajectory estimation

**How it works:**
- Graph representation: variables (states) + factors (constraints)
- Solve: minimize Σ ||residual||² over all factors
- Gauss-Newton or Levenberg-Marquardt

**Pros:**
- ✅ **Globally optimal** (smoother, not filter)
- ✅ Better accuracy than filtering
- ✅ Flexible: easy to add constraints
- ✅ Sparse structure → efficient
- ✅ Widely used in SLAM

**Cons:**
- ❌ **Not real-time** (batch processing)
- ❌ Requires all data upfront
- ❌ Slower than filtering for online use
- ❌ More complex implementation

**Complexity:** O(T×n³) for T time steps, but sparse structure helps

**When to prefer over EKF:**
- Offline processing acceptable
- Need best possible accuracy
- Have loop closures or global constraints
- SLAM applications

**Example use case:**
```
Post-process trajectory after data collection
- Collected IMU + GPS data
- Want smoother, more accurate trajectory
- Processing offline is fine
→ Use Factor Graph Optimization
```

**Performance comparison:**
```
Scenario: 2D tracking (from ch3_estimators/example_comparison.py)
EKF RMSE:  0.32 m   (0.016 s)  - Online filtering
FGO RMSE:  0.28 m   (0.231 s)  - Batch smoothing ← Best accuracy
```

---

## Decision Tree

```
START: What's your problem?

├─ Static positioning (single time instant)?
│  ├─ Outliers present?
│  │  ├─ YES → Robust Least Squares (Huber/Cauchy)
│  │  └─ NO → Iterative Least Squares
│  └─ → Least Squares

├─ Online filtering (streaming data)?
│  ├─ System linear?
│  │  └─ YES → Kalman Filter (optimal!)
│  │
│  ├─ Moderate nonlinearity?
│  │  ├─ Can compute Jacobians?
│  │  │  ├─ YES → Extended Kalman Filter
│  │  │  └─ NO → Unscented Kalman Filter
│  │  └─ EKF diverges? → Try UKF
│  │
│  ├─ Severe nonlinearity or multi-modal?
│  │  └─ Particle Filter (if computational resources available)
│  │
│  └─ Not sure? → Start with EKF, iterate if needed

└─ Batch processing (offline)?
   └─ Factor Graph Optimization (best accuracy)
```

---

## Accuracy vs Speed Trade-off

```
Fast ←→ Accurate

KF (linear only)
  ↓
EKF (moderate nonlinearity)
  ↓
UKF (strong nonlinearity)
  ↓
Particle Filter (severe nonlinearity, multi-modal)
  ↓
Factor Graph (batch, globally optimal)

---

Real-time ←→ Batch

KF, EKF, UKF (online, recursive)
  ↓
PF (online, but slow)
  ↓
FGO (batch, offline)
```

---

## Common Pitfalls

### EKF Divergence
**Symptom:** Covariance shrinks, estimates diverge from truth

**Causes:**
1. Poor linearization (highly nonlinear)
2. Process noise Q too small
3. Bad initial estimate
4. Jacobian errors

**Solutions:**
- Increase Q (allow more uncertainty)
- Better initial guess
- Verify Jacobians numerically
- Switch to UKF or PF

### Particle Degeneracy
**Symptom:** All particles have similar weights, diversity lost

**Solutions:**
- More particles
- Better proposal distribution
- Systematic resampling
- Add process noise

### Robust LS Not Working
**Symptom:** Outliers not rejected

**Causes:**
- Insufficient redundancy (need 6-8+ anchors for 2D)
- Poor geometry
- Threshold too high

**Solutions:**
- Add more measurements
- Check anchor geometry
- Tune threshold parameter

---

## References

- **Chapter 3:** State estimation fundamentals
- **Chapter 8:** Observability and robustness
- Bar-Shalom et al., "Estimation with Applications to Tracking and Navigation"
- Thrun et al., "Probabilistic Robotics" (for PF)
- Dellaert & Kaess, "Factor Graphs for Robot Perception" (for FGO)

---

## Summary

| Estimator | Best For | Speed | Accuracy | Complexity |
|-----------|----------|-------|----------|------------|
| **LS** | Static, batch | ⚡⚡⚡ | ⭐⭐ | Simple |
| **KF** | Linear, Gaussian | ⚡⚡⚡ | ⭐⭐⭐⭐ | Simple |
| **EKF** | Moderate nonlinear | ⚡⚡⚡ | ⭐⭐⭐ | Medium |
| **UKF** | Strong nonlinear | ⚡⚡ | ⭐⭐⭐⭐ | Medium |
| **PF** | Multimodal, non-Gaussian | ⚡ | ⭐⭐⭐⭐ | Complex |
| **FGO** | Batch, globally optimal | ⚡ | ⭐⭐⭐⭐⭐ | Complex |

**Default recommendation:** Start with EKF. It handles most real-world cases well and is fast enough for real-time use.

