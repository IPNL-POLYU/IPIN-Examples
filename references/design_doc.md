# 1. Project Overview

## 1.1 Purpose

This repository provides open-source, simulation-based examples of the algorithms in *Principles of Indoor Positioning and Indoor Navigation* (Chapters 2–8).

The repo is intended as a companion to the book: readers can read the equations and derivations in the text and then run concrete, minimal code examples that directly implement those equations on simulated datasets.

## 1.2 Target Users

- Master’s students in navigation / robotics / geodesy / EE / CS.
- PhD students and early-career researchers entering indoor positioning.
- Engineers who want a reference implementation of classical algorithms before building production systems.

## 1.3 Goals

The repo should:

- Demonstrate key algorithms from Ch.2–8 on small, reproducible simulations:
  - Ch.2: coordinate systems & attitude.
  - Ch.3: LS/WLS, robust LS, KF/EKF/UKF/PF, FGO.
  - Ch.4–7: RF positioning, fingerprinting, PDR/sensors, SLAM.
  - Ch.8: practical sensor fusion.
- Provide reproducible experiments based on open simulation datasets.
- Offer simple, inspectable reference implementations rather than production systems.
- Support per-chapter examples that map back to the book.

### New Goal: Equation-Level Traceability

For every important equation in the book that is implemented in code, there must be a clear mapping:

**Book → Code**

For a given equation number (e.g. Eq. (3.12)), users can quickly locate:

- The implementing function/class.
- The tests and example notebooks that exercise it.

**Code → Book**

From a function or class, users can see:

- Which equation(s) it implements.
- The chapter/section context.

Design constraints:

Mapping must be:

- Searchable via plain text (e.g. searching Eq. (3.12) in the repo).
- Maintainable when new code is added.
- Visible in docstrings, comments, docs, and notebooks.

## 1.4 Out of Scope

- Advanced techniques from Chapter 9 (crowdsourcing, collaborative, deep AI PDR, RIS) beyond small demos.
- Real-time deployment, mobile apps, or large-scale real-data pipelines.
- Full SLAM frameworks or high-performance mapping stacks.

---

# 2. Scope & Use Cases

## 2.1 Scope

The repo implements simulation-first reference versions of:

- Coordinate transforms (body/map/ENU/NED/LLH) and attitude representations.
- LS/WLS and robust LS; KF/EKF/UKF/PF; basic FGO wrappers.
- RF point positioning (TOA / two-way TOA / TDOA / AOA / RSS).
- Fingerprinting (Wi-Fi / magnetic / hybrid; deterministic & probabilistic).
- Proprioceptive sensor models (IMU, wheel odom, PDR, barometer, magnetometer).
- Minimal 2D SLAM & sensor fusion demos (Loosely vs Tightly coupled; observability).

## 2.2 Typical Use Cases

- “Run a notebook to reproduce a figure similar to chapter X.”
- “Swap KF ↔ FGO on the same simulated trajectory.”
- “Compare RF trilateration vs TDOA vs AOA in a toy floor.”
- “Compare k-NN fingerprinting vs a simple ML model on a synthetic RF map.”

This section sets what problems the repo must solve, not just what code exists.

---

# 3. High-Level Architecture

Repo layout:

```text
ipin-examples/
  core/
    coords/
    estimators/
    rf/
    sensors/
    slam/
    sim/
    eval/
    fingerprinting/
    fusion/
  ch2_coords/
  ch3_estimators/
  ch4_rf_point_positioning/
  ch5_fingerprinting/
  ch6_dead_reckoning/
  ch7_slam/
  ch8_sensor_fusion/
  data/
    sim/
    real/    # optional, small demo logs only
  notebooks/
  docs/
  tools/
    check_equation_index.py

Directory roles
core/

Reusable math & models used across chapters (never import from chX_.../):

coords/ – frames, LLH↔ECEF↔ENU, ENU↔NED, attitude conversions.

estimators/ – LS/WLS, robust LS, KF/EKF/UKF/PF, FGO wrappers.

rf/ – TOA/TDOA/AOA/RSS models, DOP utilities.

sensors/ – IMU, wheel odom, PDR, mag, barometer models.

slam/ – SLAM geometry + models: SE(2) helpers, scan matching (ICP/NDT), LOAM-style feature residuals, camera model + reprojection factors (Chapter 7).

sim/ – trajectory generators, scenario definitions, noise injection.

eval/ – error metrics, CDFs, NEES/NIS, DOP.

fingerprinting/ – deterministic (NN / k-NN) and probabilistic (Bayesian) fingerprinting algorithms reused across Chapter 5 and fusion examples.

fusion/ – practical multi-sensor fusion utilities (multi-rate measurement queues, gating, robust/adaptive noise, time alignment, and calibration glue) used in ch8_sensor_fusion/ and as optional helpers elsewhere.

chX_.../

Thin chapter-specific examples:

Wiring code & scripts.

Plots and figure reproduction.

Minimal glue code between core/ and data/.

data/sim/

Standardized simulation datasets (Section 5).

data/real/

Optional, very small demonstration logs (not the main focus).

notebooks/

One notebook (or small set) per chapter, referencing equations and pointing to core/ modules.

docs/

Markdown docs per chapter + equation_index.yml and usage docs.

tools/

CI/maintenance scripts, especially equation mapping checker.

# 4. Core Functional Requirements (Basic Functions)

For each core submodule, the design doc specifies:

Purpose

Key data structures

Required functions / classes (with rough signatures)

Equation mapping expectations (docstring conventions, index entries)

Owner: navigation engineer vs software engineer.

4.1 core/coords
Purpose

Implement coordinate and attitude foundations of Ch.2.

Basic functions

Geodetic / ECEF / local frames:

llh_to_ecef(llh) -> np.ndarray

ecef_to_llh(ecef) -> np.ndarray

ecef_to_enu(ecef, ref_llh) -> np.ndarray

enu_to_ecef(enu, ref_llh) -> np.ndarray

enu_to_ned(enu) -> np.ndarray

ned_to_enu(ned) -> np.ndarray

Attitude conversions:

rpy_to_rotmat(roll, pitch, yaw) -> np.ndarray

rotmat_to_rpy(R) -> Tuple[float, float, float]

quat_to_rotmat(q) -> np.ndarray

rotmat_to_quat(R) -> np.ndarray

Equation mapping

All core transforms have docstrings like:

"Implements Eq. (2.x) in Chapter 2: ..."

Each implemented equation appears in docs/equation_index.yml.

Unique tasks

Support multiple map frames (multi-floor, building frames).

Enforce consistent frame naming/metadata used by RF, PDR, SLAM.

4.2 core/estimators
4.2.1 Purpose

This module implements the core estimation and filtering algorithms from Chapter 3 and exposes them through reusable APIs that other modules (core/rf, core/sensors, chX_...) can call.

Goals:

Provide reference implementations of:

Least squares (LS), weighted LS (WLS), and robust LS.

Linear KF, EKF, UKF, PF.

Factor graph optimization (FGO) with basic numerical solvers.

Maintain equation-level traceability to Chapter 3:

Each algorithm implementation references the exact equation numbers it follows.

Reuse the same estimator APIs across RF positioning, PDR, SLAM, and fusion examples.

4.2.2 Key abstractions & data structures

State vector

State = np.ndarray of shape (n,).

Covariance matrix

Covariance = np.ndarray of shape (n, n).

Measurement vector

Measurement = np.ndarray of shape (m,).

ProcessModel (for KF/EKF/UKF/FGO)

Encapsulates system dynamics:

f(x, u, dt) -> x_next: nonlinear state propagation (Eq. (3.21) for EKF, Eq. (3.25) for UKF).

F(x, u, dt) -> Fk: Jacobian ∂f/∂x for EKF (used in Eq. (3.22)).

Q(x, u, dt) -> Qk: process noise covariance.

MeasurementModel

Encapsulates sensor model:

h(x) -> z_pred: linear or nonlinear measurement (e.g. Eq. (3.8) for KF, Eq. (3.21) measurement part).

H(x) -> Hk: Jacobian ∂h/∂x for EKF.

R(x) -> Rk: measurement noise covariance (Σ_{w,z,k}).

Factor / FactorGraph (for FGO)

Factor:

Holds residual function r(x_subset) and its Jacobian (or automatic differentiation stub).

FactorGraph:

Collection of variables and factors representing the MAP optimization problem of Eq. (3.35)–(3.38).

All estimators share these abstractions so that changing the estimator does not require changing the sensor or motion models.

4.2.3 Least Squares (LS / WLS / Robust LS)
Relevant equations (Chapter 3)

Cost function:
J(x) = Σ (yᵢ − hᵢ(x))² → Eq. (3.1).

Linear model normal equations:
HᵀH x̂ = Hᵀ y → Eq. (3.2).
x̂ = (HᵀH)⁻¹ Hᵀ y → Eq. (3.3).

Nonlinear LS optimality condition:
Σ (yᵢ − hᵢ(x)) ∇hᵢ(x) = 0 → Eq. (3.4).

Weighted LS cost (no explicit equation number, defined in the text in 3.1.1):
J(x) = Σ wᵢ (yᵢ − hᵢ(x))².

Planned API (in core/estimators/least_squares.py)
def linear_least_squares(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Implements Eq. (3.2)–(3.3).
    # Docstring must include:
    # “Implements Eqs. (3.2)–(3.3) (normal equations and closed form LS solution).”
    ...

def weighted_least_squares(H: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    # W is diagonal or full weight (inverse covariance).
    # Generalizes Eq. (3.2)–(3.3) to WLS.
    # Docstring: “Implements weighted least squares as defined in Section 3.1.1.”
    ...

def gauss_newton_solve(h_fun, jac_fun, y, x0, max_iters, tol) -> np.ndarray:
    # Implements iterative solution of Eq. (3.4): residual gradient = 0 via Gauss–Newton.
    # Used for nonlinear LS in RF / SLAM.
    ...

def robust_least_squares(...) -> np.ndarray:
    # Wraps LS with robust loss functions (Huber, Cauchy, etc.) as described in Section 3.1.1.
    # Docstring references Table 3.1 + robust LS text, even if no explicit equation number.
    ...

Equation mapping requirements

linear_least_squares docstring: references Eqs. (3.2)–(3.3).

gauss_newton_solve: references Eq. (3.4).

equation_index.yml must map:

"Eq. (3.2)" and "Eq. (3.3)" → core/estimators/least_squares.py::linear_least_squares.

"Eq. (3.4)" → core/estimators/least_squares.py::gauss_newton_solve.

4.2.4 Kalman Filter Family (KF / EKF / UKF)
(a) Linear Kalman Filter
Relevant equations (Chapter 3)

Linear measurement model:
z_k = H_k x_k + w_{z,k} → Eq. (3.8).

Likelihood mean/covariance:
E[z_k | x_k] = H_k x_k, Cov[z_k | x_k] = Σ_{w,z,k} → Eq. (3.9).

Propagation model and prior mean/cov:
xₖ,ₖ₋₁ = F_k x_{k−1} + u_k + w_{x,k−1} → Eq. (3.11).
P_{k,k−1} = F_k Σ_{x,k−1} F_kᵀ + Σ_{w,u,k} → Eq. (3.12).

MAP derivation and posterior:
Posterior density (3.13), MAP via derivative (3.15), closed form (3.16).

KF update form:
x̂_{k,MAP} = x_{k,k−1} + K_k (z_k − H_k x_{k,k−1}) → Eq. (3.17).
K_k = P_{k,k−1} H_kᵀ (H_k P_{k,k−1} H_kᵀ + Σ_{w,z,k})⁻¹ → Eq. (3.18).
Covariance: Σ_{x,k} = P_{k,k−1} − F_k K_k H_k P_{k,k−1} → Eq. (3.19).
Summary of “five equations” → grouped as Eq. (3.20).

Planned API (in core/estimators/kalman.py)
class KalmanFilter:
    # __init__(self, F, Q, H, R) – constant matrices or callables.

    def predict(self, u: np.ndarray = None):
        # Implements propagation from Eqs. (3.11), (3.12) and the first two lines of (3.20).
        ...

    def update(self, z: np.ndarray):
        # Implements Eqs. (3.17)–(3.19) and full set in (3.20).
        ...

Equation mapping requirements

predict docstring: “Implements the prediction step of the linear Kalman filter (Eqs. (3.11), (3.12), (3.20)).”

update docstring: “Implements the update step (Eqs. (3.17)–(3.20)).”

equation_index.yml:

"Eq. (3.11)", "Eq. (3.12)", "Eq. (3.17)", "Eq. (3.18)", "Eq. (3.19)", "Eq. (3.20)" → KalmanFilter.predict / KalmanFilter.update.

(b) Extended Kalman Filter (EKF)
Relevant equations (Chapter 3)

Nonlinear state & measurement model:
x_k = f(x_{k−1}, u_k) + w_k, z_k = h(x_k) + v_k → Eq. (3.21).

EKF prediction:
x̂_k⁻ = f(x̂_{k−1}, u_k),
P_k⁻ = F_{k−1} P_{k−1} F_{k−1}ᵀ + Q → Eq. (3.22).

EKF update (on the next page, same section; equation number continues, but we primarily tie to Eq. (3.21)–(3.23)).

Planned API
class ExtendedKalmanFilter(KalmanFilter):
    # Accepts ProcessModel and MeasurementModel objects instead of fixed matrices.

    def predict(self, u, dt):
        # Uses f, F, and Q → Eq. (3.21) and (3.22).
        ...

    def update(self, z):
        # Uses h, H, R and standard KF structure, referencing EKF update equations
        # in the EKF subsection.
        ...

Equation mapping

predict docstring: “Implements EKF prediction (Eqs. (3.21)–(3.22)).”

update docstring: “Implements EKF update (EKF section following Eq. (3.21)).”

equation_index.yml: map "Eq. (3.21)", "Eq. (3.22)" to ExtendedKalmanFilter.predict.

(c) Unscented Kalman Filter (UKF)
Relevant equations (Chapter 3)

Sigma point generation:
χ₀ = x̂_{k−1}, χᵢ = x̂_{k−1} + δᵢ, χ_{i+n} = x̂_{k−1} − δᵢ, i=1..n → Eq. (3.24).

Sigma point propagation through f (process) and h (measurement):
χᵢ⁻ = f(χᵢ, u_k) → Eq. (3.25).

UKF update and gain:
K_k, x̂_k, P_k defined via cross covariances and measurement covariances → Eq. (3.30).

Planned API
class UnscentedKalmanFilter:
    def __init__(self, process_model, measurement_model, alpha, beta, kappa):
        ...

    def predict(self, u, dt):
        # Generates sigma points (Eq. (3.24)), propagates via f (Eq. (3.25)),
        # computes predicted mean/cov.
        ...

    def update(self, z):
        # Computes predicted measurement, cross covariance, Kalman gain,
        # and posterior state following Eq. (3.30).
        ...

Equation mapping

Docstrings reference Eqs. (3.24), (3.25), (3.30) explicitly.

equation_index.yml maps these equations to the UKF methods.

4.2.5 Particle Filter (PF)
Relevant equations (Chapter 3)

Recursive Bayes update:
p(x_k | z₁:k) ∝ p(z_k | x_k) p(x_k | z₁:k−1) → Eq. (3.32).

Sampling step:
x_k⁽ⁱ⁾ ∼ p(x_k | x_{k−1}⁽ⁱ⁾) → Eq. (3.33).

Weight update:
ẇ_k⁽ⁱ⁾ = w_{k−1}⁽ⁱ⁾ p(z_k | x_k⁽ⁱ⁾) → Eq. (3.34).

Planned API (in core/estimators/particle.py)
class ParticleFilter:
    # Attributes: particles (N×n), weights (N,), process_model, measurement_model.

    def predict(self, u, dt):
        # Sample new particles using the process model (Eq. (3.33)).
        ...

    def update(self, z):
        # Update weights via likelihood (Eq. (3.34)), normalize, resample.
        ...

    def estimate(self) -> np.ndarray:
        # Return weighted mean or best weight particle.
        ...

Equation mapping

predict docstring references Eq. (3.33).

update docstring references Eqs. (3.32)–(3.34).

equation_index.yml: "Eq. (3.32)", "Eq. (3.33)", "Eq. (3.34)" → ParticleFilter.

4.2.6 Factor Graph Optimization (FGO) & Numerical Methods
Relevant equations (Chapter 3)

MAP for full trajectory:
X̂_{MAP} = argmax_X p(X | Z) = argmax_X p(Z | X) p(X) / p(Z) → Eq. (3.35).

Simplified MAP form:
X̂_{MAP} = argmax_X ℓ(X; Z) p(X) with ℓ(X; Z) ∝ p(Z | X) → Eqs. (3.36)–(3.37).

Conversion from product of Gaussians to sum of squared residuals (negative log posterior) → Eq. (3.38) and following.

Gradient descent update:
x_{k+1} = x_k + α d → Eq. (3.42).

Descent condition:
f(x_{k+1}) = f(x_k + α d) < f(x_k) → Eq. (3.43).

Planned API (in core/estimators/fgo.py and core/estimators/optim.py)
class Factor:
    def residual(self, x: np.ndarray) -> np.ndarray:
        ...
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        ...

class FactorGraph:
    # Stores variable ordering, list of factors.
    def evaluate(self, x: np.ndarray):
        # returns stacked residuals and Jacobian.
        ...

def gradient_descent_step(x, grad, alpha) -> np.ndarray:
    # Encodes Eq. (3.42) and Eq. (3.43) logic.
    ...

def gauss_newton_step(x, J, r) -> np.ndarray:
    # Equivalent to LS step on residuals derived from Eq. (3.38).
    ...

def levenberg_marquardt_step(...) -> np.ndarray:
    # LM step bridging gradient descent and Gauss–Newton, consistent with 3.4.1 discussion.
    ...

def solve_fgo(graph: FactorGraph, x0, method='gn', max_iters=..., tol=...) -> np.ndarray:
    # High-level solver that iteratively applies GN/LM using the factor graph.
    ...

Equation mapping

solve_fgo docstring references Eqs. (3.35)–(3.38) (MAP via factor graph).

gradient_descent_step docstring references Eqs. (3.42)–(3.43).

equation_index.yml ties these equations to the corresponding functions.

4.2.7 Simulation data requirements (for estimators)

Although most estimators are exercised through chapter specific scenarios, core/estimators also needs simple synthetic datasets for unit tests and minimal examples:

toy_ls_linear/

Small linear regression / linear system:

H (m×n), y (m), known true x_true, noise variance.

Used to test:

linear_least_squares, weighted_least_squares, robust_least_squares.

Must be documented in Section 5.2 as one of the dataset families.

toy_kf_1d_cv/

1D constant velocity (CV) model:

State: [position, velocity].

Measurements: noisy position only.

Used to test:

KalmanFilter convergence vs. analytical solution.

toy_nonlinear_1d/

1D or 2D nonlinear tracking:

Nonlinear measurement, e.g. range², to exercise EKF, UKF, PF, and FGO.

Used to:

Compare KF vs. EKF vs. UKF vs. PF on the same synthetic data.

Test FGO smoothing vs. recursive filters.

These datasets should be stored under data/sim/ (see Section 5.2) and used both by:

Unit tests in tests/test_estimators_*.py.

A small ch3_estimators notebook that reproduces selected plots / behaviors from Chapter 3.

4.3 core/rf
Purpose

RF signal measurement models from Ch.4–5.

Basic functions

Ranging / angles:

toa_range(tx_pos, rx_pos, c, clock_bias)

two_way_toa_range(...)

tdoa_range_diff(anchor_i, anchor_j, state, c)

aoa_bearing(anchor_pos, state)

RSS:

rss_pathloss(tx_power, distance, n, sigma_shadow)

DOP:

compute_dop(geometry, weights=None) -> dict with HDOP/VDOP/PDOP.

Equation mapping

Each measurement model documents which equations in Ch.4 it implements (e.g. path-loss, hyperbolic TDOA, AOA geometry).

Unique tasks

Support multiple technologies (Wi-Fi, BLE, UWB, 5G) through parameterization rather than separate code.

4.4 core/sensors
Purpose

Proprioceptive & environmental sensor models from Ch.6.

Basic functions

IMU:

Continuous/discrete error models (bias, noise, RW).

Simple strapdown integration for 2D/3D.

PDR:

Step detection, step length models, heading from IMU/mag.

Environmental:

Barometer altitude, floor detection.

Magnetometer heading and magnetic fingerprint stubs.

Wheel odometry:

Differential drive, steering model, noise models.

Equation mapping

IMU propagation, error models, ZUPT updates etc. point to Ch.6 equations.

Unique tasks

Define a unified sensor packet structure used across simulation and logs.

4.4.1 Chapter 6 scope and design stance

This module implements the proprioceptive + environmental sensor models and dead-reckoning algorithms from Chapter 6:

IMU strapdown propagation (attitude/velocity/position).

Wheel odometry dead reckoning (vehicle).

Integrated IMU + wheel speed fusion (EKF).

Drift correction constraints: ZUPT / ZARU / NHC.

Pedestrian Dead Reckoning (PDR): step detection + step length + heading + step update.

Environmental sensors: magnetometer heading, barometer altitude / floor change.

Calibration utilities: Allan variance; IMU scale/misalignment model.

Design stance

Dead-reckoning drifts. Examples must make drift clearly visible, then show how constraints/fusion reduce it.

Keep two tracks for teaching clarity:

Pure propagation (strapdown / PDR) to show drift.

Constraint/fusion (EKF updates with ZUPT/ZARU/NHC, mag/baro) to show drift reduction.

Avoid production-grade INS complexity unless needed. Implement the chapter equations faithfully, with minimal state choices.

4.4.2 Key data structures and packet formats

All structures live in core/sensors/types.py.

Time base

Use float seconds t (monotonic), or timestamp arrays np.ndarray (preferred).

Sensor series packets

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ImuSeries:
    t: np.ndarray        # (N,)
    accel: np.ndarray    # (N,3) m/s^2 in body frame B
    gyro: np.ndarray     # (N,3) rad/s in body frame B
    meta: dict

@dataclass(frozen=True)
class WheelSpeedSeries:
    t: np.ndarray        # (N,)
    v_s: np.ndarray      # (N,3) velocity in speed frame S (per Chapter 6)
    meta: dict

@dataclass(frozen=True)
class MagnetometerSeries:
    t: np.ndarray
    mag: np.ndarray      # (N,3) in device/body frame
    meta: dict

@dataclass(frozen=True)
class BarometerSeries:
    t: np.ndarray
    pressure: np.ndarray # (N,) Pa (or hPa, but must be explicit in meta)
    meta: dict


Navigation state (minimal Chapter-6 state)

Base state is consistent with Chapter-6 integrated formulation (quaternion + velocity + position), e.g. Eq. (6.16).

@dataclass
class NavStateQPVP:
    q: np.ndarray  # (4,) quaternion scalar-first (q0,q1,q2,q3)
    v: np.ndarray  # (3,) velocity in map frame M
    p: np.ndarray  # (3,) position in map frame M


Bias-augmented state (recommended for realism)

Add gyro and accel biases when implementing Eq. (6.5)/(6.9) style models.

@dataclass
class NavStateQPVPBias:
    q: np.ndarray      # (4,)
    v: np.ndarray      # (3,)
    p: np.ndarray      # (3,)
    b_g: np.ndarray    # (3,) gyro bias
    b_a: np.ndarray    # (3,) accel bias


Rule

Chapter-6 examples may start with NavStateQPVP to keep concepts simple, then optionally switch to NavStateQPVPBias in “advanced” notebooks/tests.

4.4.3 Module layout

Recommended layout:

core/sensors/
  types.py
  imu_models.py          # measurement correction + calibration helpers
  strapdown.py           # quaternion/vel/pos propagation
  wheel_odometry.py      # wheel speed DR + lever-arm compensation
  ins_ekf_models.py      # ProcessModel/MeasurementModel for IMU+wheel EKF
  constraints.py         # ZUPT / ZARU / NHC detectors + pseudo-measurements
  pdr.py                 # step detection, step length, PDR propagation
  environment.py         # mag heading + baro altitude (+ small smoothing helper)
  calibration.py         # Allan variance + IMU scale/misalignment model

4.4.4 IMU strapdown propagation APIs (Chapter 6 quaternion/velocity/position)

Quaternion propagation (Eqs. (6.2)–(6.4), Ω matrix Eq. (6.3))

def omega_matrix(omega_b: np.ndarray) -> np.ndarray:
    """
    Build Ω(ω) used in quaternion kinematics.
    Implements Eq. (6.3).
    """
    ...

def quat_integrate(q_prev: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    """
    Discrete quaternion update.
    Implements Eqs. (6.2)–(6.4).
    Note: normalize output quaternion each step.
    """
    ...


IMU correction (gyro/accel bias/noise; Eqs. (6.5)–(6.6), Eq. (6.9))

def correct_gyro(
    gyro_meas: np.ndarray,
    b_g: np.ndarray,
    n_g: np.ndarray | None = None
) -> np.ndarray:
    """
    ω = ω~ - b_g - n_g.
    Implements Eq. (6.6) (consistent with Eq. (6.5)).
    """
    ...

def correct_accel(
    accel_meas: np.ndarray,
    b_a: np.ndarray,
    n_a: np.ndarray | None = None
) -> np.ndarray:
    """
    Specific force correction consistent with Eq. (6.9).
    """
    ...


Velocity, gravity, position updates (Eqs. (6.7)–(6.10))

def gravity_vector() -> np.ndarray:
    """
    Gravity approximation.
    Implements Eq. (6.8).
    """
    ...

def vel_update(v_prev: np.ndarray, q_prev: np.ndarray, f_b: np.ndarray, dt: float) -> np.ndarray:
    """
    v_k = v_{k-1} + (C_B^M(q) f_b + g) dt.
    Implements Eq. (6.7).
    """
    ...

def pos_update(p_prev: np.ndarray, v_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    p_k = p_{k-1} + v_k dt.
    Implements Eq. (6.10).
    """
    ...


Note on coordinate transforms

If core/coords/quat_to_rotmat is used for Chapter-6 propagation, add Chapter-6 equation references (e.g. Eq. (6.13) if the book defines the quaternion rotation matrix there) to its docstring in addition to Chapter-2 references.

4.4.5 Wheel odometry dead reckoning APIs (Eqs. (6.11)–(6.15))

Skew matrix / lever arm / velocity transform / position update:

def skew(v: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix [v×].
    Implements Eq. (6.12).
    """
    ...

def wheel_speed_to_attitude_velocity(v_s: np.ndarray, omega_b: np.ndarray, lever_arm_b: np.ndarray) -> np.ndarray:
    """
    Convert wheel speed measurement to attitude-frame velocity with lever arm compensation.
    Implements Eq. (6.11).
    """
    ...

def attitude_to_map_velocity(v_a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    v^M = C_A^M(q) v^A.
    Implements Eq. (6.14).
    """
    ...

def odom_pos_update(p_prev: np.ndarray, v_map: np.ndarray, dt: float) -> np.ndarray:
    """
    p_k^M = p_{k-1}^M + v_k^M Δt.
    Implements Eq. (6.15).
    """
    ...

4.4.6 Integrated IMU + wheel EKF models (Eqs. (6.16)–(6.43))

Implement as ProcessModel and MeasurementModel so it plugs into core/estimators/ExtendedKalmanFilter.

Process model (Eqs. (6.17)–(6.32); state Eq. (6.16))

class InsWheelProcessModel(ProcessModel):
    """
    EKF process model for state x=[q, v, p] (Eq. (6.16)),
    with propagation and covariance per Eqs. (6.17)–(6.32).
    """

    def f(self, x: np.ndarray, u: dict, dt: float) -> np.ndarray:
        """
        Implements Eq. (6.17) and the concrete sub-updates in the Chapter 6 text.
        """
        ...

    def F(self, x: np.ndarray, u: dict, dt: float) -> np.ndarray:
        """
        State Jacobian.
        Implements Eq. (6.28) (and uses Eq. (6.29) where needed).
        """
        ...

    def Q(self, x: np.ndarray, u: dict, dt: float) -> np.ndarray:
        """
        Process noise covariance.
        Implements Eqs. (6.30)–(6.31).
        """
        ...


Wheel measurement model (Eqs. (6.33)–(6.38))

class WheelSpeedMeasurementModel(MeasurementModel):
    """
    Wheel-speed measurement model.
    Implements Eqs. (6.33)–(6.38).
    """

    def h(self, x: np.ndarray) -> np.ndarray:
        """Implements Eqs. (6.33)–(6.35)."""
        ...

    def H(self, x: np.ndarray) -> np.ndarray:
        """Implements Eqs. (6.37)–(6.38)."""
        ...

    def R(self, x: np.ndarray) -> np.ndarray:
        """Measurement covariance (configurable)."""
        ...


EKF update equations (Eqs. (6.39)–(6.43))

Do not duplicate EKF update logic in core/sensors.

core/estimators/ExtendedKalmanFilter.update() is the single source of truth.

Add docstring references in ExtendedKalmanFilter.update() to include both:

Chapter 3 EKF update block, and

Chapter 6 EKF update block (Eqs. (6.39)–(6.43)).

4.4.7 Drift correction constraints (ZUPT / ZARU / NHC)

ZUPT detector (Eq. (6.44)) and pseudo-measurement (Eq. (6.45))

def detect_zupt(gyro_b: np.ndarray, accel_b: np.ndarray, delta_omega: float, delta_f: float) -> bool:
    """
    Stationary detector.
    Implements Eq. (6.44).
    """
    ...

class ZuptMeasurementModel(MeasurementModel):
    """
    Zero-velocity pseudo-measurement.
    Implements Eq. (6.45).
    """
    ...


ZARU pseudo-measurement (Eq. (6.60))

class ZaruMeasurementModel(MeasurementModel):
    """
    Zero angular rate pseudo-measurement.
    Implements Eq. (6.60).
    """
    ...


NHC pseudo-measurement (Eq. (6.61))

class NhcMeasurementModel(MeasurementModel):
    """
    Nonholonomic constraint pseudo-measurement.
    Implements Eq. (6.61).
    """
    ...


Practical warning (must be shown in examples)

These constraints are only valid under their motion assumptions.

Include at least one “assumption violated” segment (e.g., wheel slip, foot sliding) and show the failure mode.

4.4.8 PDR APIs (Eqs. (6.46)–(6.50))
def total_accel_magnitude(accel_b: np.ndarray) -> float:
    """Implements Eq. (6.46)."""
    ...

def remove_gravity_from_magnitude(a_mag: float, g: float = 9.81) -> float:
    """Implements Eq. (6.47)."""
    ...

def step_frequency(delta_t: float) -> float:
    """Implements Eq. (6.48)."""
    ...

def step_length(h: float, f_step: float, a: float = 0.371, b: float = 0.227, c: float = 1.0) -> float:
    """Implements Eq. (6.49)."""
    ...

def pdr_step_update(p_prev_xy: np.ndarray, step_len: float, heading_rad: float) -> np.ndarray:
    """
    2D PDR step update.
    Implements Eq. (6.50) specialized to 2D.
    """
    ...


Heading sources for PDR examples

Gyro-integrated yaw (drifts).

Magnetometer heading (suffers indoor disturbance).

Optional smoothing filter using Eq. (6.55) style model.

4.4.9 Environmental sensor models (magnetometer + barometer) (Eqs. (6.51)–(6.55))

Magnetometer heading (Eqs. (6.51)–(6.53))

def mag_tilt_compensate(mag_b: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Implements Eq. (6.52)."""
    ...

def mag_heading(mag_b: np.ndarray, roll: float, pitch: float) -> float:
    """Implements Eqs. (6.51)–(6.53)."""
    ...


Barometer altitude (Eq. (6.54))

def pressure_to_altitude(p: float, p0: float, T: float) -> float:
    """Implements Eq. (6.54)."""
    ...


Optional smoothing helper (Eq. (6.55))

Provide a small reusable “altitude smoothing KF” or “heading smoothing KF” helper that references Eq. (6.55) as the generic state/measurement model, but still uses core/estimators KF/EKF APIs.

4.4.10 Calibration utilities (Eqs. (6.56)–(6.59))

Allan variance (Eqs. (6.56)–(6.58))

def allan_variance(x: np.ndarray, fs: float, taus: np.ndarray) -> np.ndarray:
    """Implements Eqs. (6.56)–(6.58)."""
    ...


IMU scale/misalignment model (Eq. (6.59))

def apply_imu_scale_misalignment(u: np.ndarray, M: np.ndarray, S: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Implements Eq. (6.59)."""
    ...

4.4.11 Simulation data requirements (for sensors / Chapter 6)

Although most Chapter-6 algorithms are exercised through ch6_dead_reckoning, core/sensors must be testable with standardized simulation datasets stored in data/sim/ (see Section 5.2):

ch6_strapdown_basic/

ch6_wheel_odom_square/

ch6_foot_zupt_walk/

ch6_pdr_corridor_walk/

ch6_env_sensors_heading_altitude/

These datasets must be deterministic (fixed seeds, config files), small, and documented.

2) INSERT into Section 5.2 Dataset Families (no deletion required)

In Section 5.2 Dataset Families, after your existing pdr_corridor_walk/ entry (or right after wifi_fingerprint_grid/ if you prefer), insert the following additional dataset families:

ch6_strapdown_basic/

Baseline strapdown IMU propagation dataset (attitude/velocity/position).

Files (NPZ preferred):

truth.npz: t, q, v, p

imu.npz: t, accel, gyro (optional b_g_true, b_a_true)

config.json: dt, noise, bias, initial conditions, frame definitions

ch6_wheel_odom_square/

Vehicle-style dataset for wheel DR and IMU+wheel EKF.

Files:

truth.npz: t, q, v, p

wheel.npz: t, v_s (speed frame S)

imu.npz: t, accel, gyro

config.json: lever arm, wheel noise, optional slip segments

ch6_foot_zupt_walk/

Foot-mounted INS dataset with stance phases for ZUPT/ZARU.

Files:

truth.npz: t, q, v, p

imu.npz

events.npz: stance_mask (optional for evaluation)

config.json: detector thresholds (δ_ω, δ_f), sample rate

ch6_pdr_corridor_walk/

Step-and-heading PDR dataset (lightweight PDR baseline).

Files:

truth_xy.npz: t, p_xy, heading_true

imu.npz

config.json: user height h, (a,b,c) parameters for step length model

ch6_env_sensors_heading_altitude/

Environmental sensor dataset (mag heading + barometer altitude).

Files:

truth.npz: heading_true, altitude_true

mag.npz, baro.npz

config.json: disturbance segments, p0 and T for barometer model

4.5 core/sim
Purpose

Shared simulation tools for all examples.

Basic functions

Trajectories:

generate_2d_trajectory(shape, params, dt, duration)

generate_3d_trajectory(...)

Scenarios:

Rooms/floor plans, beacons/anchors, obstacles.

Sensor noise injection:

add_imu_noise(traj, model_params)

simulate_rf_measurements(traj, anchors, rf_config)

Unique tasks

Scenario configurations in JSON/YAML so all chapters share the same definitions.

4.6 core/eval
Purpose

Evaluation and visualization utilities shared across chapters.

Provide a small, consistent API for:

computing errors and consistency metrics, and

generating publication-quality, vector graphics for trajectories, errors, and algorithm comparisons.

Basic functions (numerical evaluation)

Position errors:

compute_position_errors(truth, est) -> dict
Per-epoch error vectors (ENU / NED).

compute_rmse(errors) -> float | dict
Scalar RMSE or per-axis RMSE.

compute_error_stats(errors) -> dict
mean / median / percentiles.

Consistency metrics:

compute_nees(truth, est, cov) -> np.ndarray

compute_nis(innov, S) -> np.ndarray

helpers to check NEES/NIS against χ² bounds.

RF / geometry utilities:

compute_dop(geometry, weights=None) -> dict (HDOP/VDOP/PDOP), reusing core/coords for frame handling.

Basic functions (visualization)

All plotting helpers live in core/eval/plots.py. They must:

accept plain NumPy arrays / dicts (no hidden global state), and

return a matplotlib.figure.Figure (the caller decides whether to show or save).

Planned APIs:

Trajectory / map views

plot_trajectory_2d(truth_xy, est_xy_dict, anchors_xy=None) -> Figure
truth vs. one or more estimates on a 2D floor plan.
optional RF anchors / landmarks overlay.

plot_trajectory_3d(truth_xyz, est_xyz_dict, anchors_xyz=None) -> Figure
simple 3D view for SLAM / multi-floor examples.

Error over time

plot_position_error_time(errors_dict, dt, axes="enu") -> Figure
per-axis error vs. time, multiple algorithms on one plot.

Error distributions

plot_error_hist(errors_dict, bins=…) -> Figure

plot_error_cdf(errors_dict) -> Figure
used heavily in Ch.4–5 to compare RF / fingerprinting methods.

Consistency plots

plot_nees(nees, chi2_bounds) -> Figure

plot_nis(nis, chi2_bounds) -> Figure

RF geometry / DOP

plot_rf_geometry(anchors_xy, traj_xy=None) -> Figure

plot_dop_map(dop_grid, floorplan=None) -> Figure
visualize how beacon layouts (or AP layouts) affect HDOP / VDOP.

SLAM / map views (minimal)

plot_occupancy_grid(grid, poses=None, landmarks=None) -> Figure

plot_factor_graph_skeleton(nodes, edges) -> Figure (optional, diagnostic only).

Standard vector export

To make results easy to reuse in papers and slides, every plotting helper must work with a single, shared export utility:

save_figure(fig, out_dir, name, formats=("svg", "pdf")) -> list[path]

ensures:

vector formats by default (.svg, .pdf),

deterministic naming, and

creation of the output directory if needed.

File and naming conventions

All chapter examples should save figures into:

chX_.../figs/ for scripts, or

notebooks/figs/chX_.../ for notebooks.

File names must follow:

ch3_estimators_kf_vs_pf_rmse.svg

ch4_rf_toa_vs_tdoa_cdf.pdf

ch6_pdr_drift_correction_trajectory.svg

Notebooks embed the figure inline, but also call save_figure so users get reusable vector files with a single run.

Visual style guidelines

To keep plots visually consistent across the repo:

Always label axes with units, e.g. Position error [m], Time [s], Heading [deg].

Use the same color / linestyle mapping for estimator families across chapters:

LS/WLS: solid blue

KF/EKF/UKF: solid green

PF: dashed orange

FGO / smoothing: solid red

Keep backgrounds white and gridlines light (MATLAB/Matplotlib default is fine).

Avoid heavy custom styling in core; chapter examples can adjust if needed.

Unique tasks

Define a small set of “standard plots” per chapter (see Section 7) and implement them using core/eval/plots.py.

Make sure every example script / notebook:

prints a small numerical summary (RMSE, final bias, etc.), and

generates at least one vector figure via save_figure(...).

Ensure that evaluation APIs are reused across RF, PDR, SLAM, and fusion examples so that algorithm comparisons are meaningful and visually comparable.

4.7 core/fingerprinting
4.7.1 Purpose

This module collects the generic fingerprinting algorithms from Chapter 5 into reusable code:

Deterministic fingerprinting (NN, k-NN).

Probabilistic fingerprinting (Bayesian / Naive Bayes style).

Optional model-based utilities to support ray-tracing / position-hypothesized methods.

Goals:

Provide clean, reusable APIs for fingerprint-based localization.

Maintain equation-level traceability to Chapter 5:

Each algorithm implementation references the relevant Eq. (5.x).

Allow chapter examples (ch5_fingerprinting/, ch8_sensor_fusion/) to reuse the same algorithms.

4.7.2 Key data structures

All defined in core/fingerprinting/dataset.py and core/fingerprinting/types.py.

Location vector

Location = np.ndarray of shape (d,) (typically d = 2 or 3).

Fingerprint / feature vector

Fingerprint = np.ndarray of shape (N,), where N is number of features (e.g. RSSI from N APs).

FingerprintDatabase

@dataclass
class FingerprintDatabase:
    locations: np.ndarray  # shape (M, d), x_i
    features: np.ndarray   # shape (M, N), f_i
    meta: dict             # AP IDs, floor labels, etc.


Query fingerprint

z: np.ndarray of shape (N,), same dimension and ordering as database f_i.

Equation mapping rule:

Any algorithm that directly implements an equation from Chapter 5 must:

Include that equation ID in its docstring (e.g. “Implements Eq. (5.1)”).

Have a corresponding entry in docs/equation_index.yml.

4.7.3 Deterministic fingerprinting (NN / k-NN)

Relevant equations (Chapter 5):

Nearest neighbor (1-NN): decision rule based on distance
i* = argmin_i D(z, f_i) (Eq. (5.1)).

Weighted k-NN interpolation:
x̂ = ∑_{i∈K(z)} w_i x_i / ∑_{i∈K(z)} w_i (Eq. (5.2)),
with typical choice w_i = 1/(D(z,f_i)+ε).

Planned API (in core/fingerprinting/deterministic.py):

Distance utilities:

def distance(z: np.ndarray, f: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute D(z, f) between query fingerprint z and reference fingerprint f.

    Implements the distance metric D(·, ·) used in Eq. (5.1) and Eq. (5.2)
    in Chapter 5 (e.g., Euclidean and Manhattan distances).
    """
    ...

def pairwise_distances(z: np.ndarray, F: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute distances D(z, f_i) for all fingerprints f_i in F.

    Implements the distance evaluation required in Eq. (5.1) across
    all i = 1, ..., M.
    """
    ...


Nearest-neighbor localizer (Eq. (5.1)):

def nn_localize(
    z: np.ndarray,
    db: FingerprintDatabase,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Nearest-neighbor deterministic fingerprinting.

    Implements Eq. (5.1) in Chapter 5:
        i* = argmin_{1<=i<=M} D(z, f_i),
    and returns x_hat = x_{i*}.
    """
    ...


k-NN localizer (Eq. (5.2)):

def knn_localize(
    z: np.ndarray,
    db: FingerprintDatabase,
    k: int = 3,
    metric: str = "euclidean",
    weighting: str = "inverse_distance",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    k-nearest-neighbor fingerprinting with interpolation.

    Implements Eq. (5.2) in Chapter 5:
        x_hat = sum_{i in K(z)} w_i x_i / sum_{i in K(z)} w_i,

    with weights w_i defined from D(z, f_i), typically
        w_i = 1 / (D(z, f_i) + eps).
    """
    ...


Equation mapping:

nn_localize docstring references Eq. (5.1).

knn_localize docstring references Eq. (5.2).

equation_index.yml entries:

"Eq. (5.1)" → core/fingerprinting/deterministic.py::nn_localize.

"Eq. (5.2)" → core/fingerprinting/deterministic.py::knn_localize.

4.7.4 Probabilistic fingerprinting (Bayesian)

Relevant equations (Chapter 5):

Posterior over locations:
P(x_i | Z_i = z) = P(Z_i = z | x_i) P(x_i) / P(Z_i = z) (Eq. (5.3)).

ML / MAP estimate:
i* = argmax_i P(Z_i = z | x_i) (Eq. (5.4)).

Posterior-mean estimate:
x̂ = ∑_{i=1}^M P(x_i | Z_i = z) x_i (Eq. (5.5)).

Planned API (in core/fingerprinting/probabilistic.py):

Model:

@dataclass
class NaiveBayesFingerprintModel:
    """
    Probabilistic fingerprinting model P(z | x_i) under naive Bayes assumptions.

    Implements the Bayesian formulation in Eqs. (5.3)–(5.5) of Chapter 5:
        - Eq. (5.3): posterior P(x_i | Z_i = z)
        - Eq. (5.4): ML/MAP selection of i*
        - Eq. (5.5): posterior-mean location estimate x_hat
    """
    db: FingerprintDatabase
    priors: np.ndarray  # shape (M,), P(x_i)
    means: np.ndarray   # shape (M, N)
    vars: np.ndarray    # shape (M, N)


Fitting:

def fit_gaussian_naive_bayes(
    db: FingerprintDatabase,
    samples_per_rp: np.ndarray,
    var_floor: float = 1e-4,
    prior: str = "uniform",
) -> NaiveBayesFingerprintModel:
    """
    Fit a Gaussian naive Bayes P(z | x_i) model from offline survey data.

    Each reference point i has empirical mean and variance of each feature,
    which parameterize P(Z_i = z | x_i) in Eq. (5.3).
    """
    ...


Inference helpers:

def log_likelihoods(model: NaiveBayesFingerprintModel, z: np.ndarray) -> np.ndarray:
    """
    Compute log P(Z_i = z | x_i) for all i.

    This corresponds to the likelihood term in Eq. (5.3) & Eq. (5.4).
    """
    ...

def posterior(model: NaiveBayesFingerprintModel, z: np.ndarray) -> np.ndarray:
    """
    Compute posterior P(x_i | Z_i = z) over all i.

    Implements Eq. (5.3) with normalization over i.
    """
    ...

def map_localize(model: NaiveBayesFingerprintModel, z: np.ndarray) -> np.ndarray:
    """
    ML/MAP estimate based on Eq. (5.4):

        i* = argmax_i P(Z_i = z | x_i) (with uniform priors),

    and returns x_hat = x_{i*}.
    """
    ...

def posterior_mean_localize(model: NaiveBayesFingerprintModel, z: np.ndarray) -> np.ndarray:
    """
    Posterior-mean estimate of location:

        x_hat = sum_i P(x_i | Z_i = z) * x_i    (Eq. (5.5))
    """
    ...


Equation mapping:

posterior ↔ Eq. (5.3).

map_localize ↔ Eq. (5.4).

posterior_mean_localize ↔ Eq. (5.5).

Corresponding entries in equation_index.yml.

4.7.5 Advanced / model-based fingerprinting (optional)

For the ray-tracing / position-hypothesized method (Eqs. (5.7)–(5.11)), implement only lightweight helpers in core/fingerprinting/model_based.py:

gaussian_likelihood(z, z_pred, sigma) – measurement likelihood reused by demos.

raytrace_likelihood(z, z_pred, sigma) – convenience wrapper for “ray-traced” features.

The full position-hypothesized particle method (sampling states from a Gaussian prior, computing likelihoods, then weighted averaging as in Eq. (5.11)) is implemented as a chapter demo in ch5_fingerprinting (Section 7.4.5), not as a generic core API.

Docstrings for these helpers reference the likelihood concept used in Eqs. (5.8)–(5.10).

4.7.6 Simulation data requirements (for fingerprinting)

core/fingerprinting relies on at least one standardized dataset family in data/sim/:

wifi_fingerprint_grid/:

locations.*: (M, d) reference point coordinates.

fingerprints.*: (M, N) mean feature vectors f_i.

Optional fingerprints_samples/: raw RSS samples per RP for probabilistic model fitting.

query_trajectories/: sequences (x_true(t), z(t)) for evaluation.

Metadata: AP IDs, floor labels, building ID, etc.

These are used by:

Unit tests:

tests/test_fingerprinting_deterministic.py

tests/test_fingerprinting_probabilistic.py

Chapter 5 examples in ch5_fingerprinting (Section 7.4).

4.8 core/slam
4.8.1 Purpose

Provide minimal, reusable building blocks for Chapter 7 (SLAM) examples:

- LiDAR scan matching (scan-to-scan and scan-to-map) using ICP-style objectives (Eqs. (7.10)–(7.11)).
- NDT (normal distribution transform) scan matching (Eqs. (7.12)–(7.16)).
- LOAM-style feature residuals (edge/plane) for didactic odometry + mapping (Eqs. (7.17)–(7.19)).
- Camera projection/distortion helpers and reprojection residuals for (synthetic) visual SLAM / bundle adjustment (Eqs. (7.43)–(7.46), (7.68)–(7.70)).

Design rule (important):

- This is not a full SLAM framework.
- core/slam provides geometry + residual models.
- The nonlinear solvers live in core/estimators (Gauss–Newton / LM / FGO).
- ch7_slam/ wires datasets + factors + solver + plots.

4.8.2 Key abstractions & data structures

Pose representations

- Pose2 (SE(2)) state: x = [x, y, yaw] in meters/radians.
- Pose3 (SE(3)) is optional; the repo can stay 2D-first.

Point clouds / scans

- PointCloud2D: np.ndarray of shape (N,2) in meters.
- PointCloud3D (optional): np.ndarray of shape (N,3).

VoxelGrid (for NDT)

- Voxel key: integer tuple (ix, iy, iz) for 3D or (ix, iy) for 2D.
- Store per-voxel mean p_k and covariance Σ_k (Eqs. (7.12)–(7.13)).

Feature sets (for LOAM-style residuals)

- EdgeFeatures: np.ndarray of shape (Ne,3) in 3D (or (Ne,2) in 2D).
- PlaneFeatures: np.ndarray of shape (Np,3) (3D only).
- Each feature set may carry indices back to the original scan.

Camera model (for synthetic visual SLAM)

- CameraIntrinsics: fx, fy, cx, cy, distortion parameters (k1, k2, p1, p2).
- Observations: (frame_id, landmark_id, u, v) pixel coordinates.

4.8.3 Recommended module layout

core/slam/
  types.py                # dataclasses: Pose2, CameraIntrinsics, etc.
  se2.py                  # SE(2) compose/inverse/apply + angle wrapping
  scan_matching.py        # ICP-style matching + correspondence gating (Eqs. (7.10)–(7.11))
  ndt.py                  # voxel stats + NDT objective/optimizer hooks (Eqs. (7.12)–(7.16))
  loam.py                 # curvature + feature selection + residuals (Eqs. (7.17)–(7.19))
  camera.py               # distortion / projection helpers (Eqs. (7.43)–(7.46), plus projection Eq. (7.40) if needed)
  factors.py              # FactorGraph factor classes for pose graph + reprojection

4.8.4 Required functions / APIs (with equation hooks)

(a) SE(2) helpers (used by all Chapter 7 examples)

core/slam/se2.py

- se2_compose(p: np.ndarray, dp: np.ndarray) -> np.ndarray
  Compose poses in [x,y,yaw] form.

- se2_inverse(p: np.ndarray) -> np.ndarray

- se2_apply(p: np.ndarray, pts: np.ndarray) -> np.ndarray
  Apply pose to points. Used by scan matching residuals.

- wrap_angle(theta: float) -> float

(b) ICP-style scan matching (Eqs. (7.10)–(7.11))

core/slam/scan_matching.py

- build_correspondences(
    pts_ref: np.ndarray,   # scan at t-1
    pts_cur: np.ndarray,   # scan at t
    T_init: np.ndarray,    # Pose2 initial guess
    epsilon: float
  ) -> tuple[np.ndarray, np.ndarray]

  Implements the gating/association variable in Eq. (7.11) using nearest-neighbor + distance threshold epsilon.

- icp_point_to_point(
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    T_init: np.ndarray,
    max_iters: int = 20,
    epsilon: float = 0.5,
    robust_kernel: str | None = None,
  ) -> np.ndarray

  Minimizes the LS objective in Eq. (7.10) (with correspondences from Eq. (7.11)).
  Should expose per-iteration diagnostics: number of correspondences, residual RMS, Δpose.

Implementation notes

- Use scipy.spatial.cKDTree for fast nearest-neighbor queries.
- For the rigid transform update step, use an SVD/Procrustes solve (point-to-point ICP); keep the implementation readable.

(c) NDT scan matching (Eqs. (7.12)–(7.16))

core/slam/ndt.py

- voxel_stats(points: np.ndarray, voxel_size: float) -> dict
  Compute per-voxel mean p_k and covariance Σ_k (Eqs. (7.12)–(7.13)).

- ndt_negative_log_likelihood(
    pts_cur: np.ndarray,
    voxel_map: dict,
    T: np.ndarray
  ) -> float

  Implements the negative log of the likelihood product in Eqs. (7.14)–(7.15).

- ndt_align(
    pts_cur: np.ndarray,
    voxel_map: dict,
    T_init: np.ndarray,
    max_iters: int = 20,
  ) -> np.ndarray

  Optimization wrapper corresponding to the MLE in Eq. (7.16).
  Should call core/estimators/optim (GN/LM) rather than implementing its own solver.

(d) LOAM-style feature residuals (Eqs. (7.17)–(7.19))

core/slam/loam.py

- compute_curvature(points: np.ndarray, k: int = 10) -> np.ndarray

- select_loam_features(curvature: np.ndarray, edge_q: float, plane_q: float) -> tuple[np.ndarray, np.ndarray]
  Select “edge” and “plane” points based on curvature thresholding (as described in the LOAM subsection).

- point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float
  Implements Eq. (7.18).

- point_to_plane_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float
  Implements Eq. (7.19) (3D only; can be stubbed if repo stays 2D-first).

- loam_odometry_step(...)
  Wrapper for the odometry objective in Eq. (7.17). In the repo this is a didactic demo:
  a small number of features, small maps, and clear plots.

Robust weights

- LOAM uses robust/bisquare-like weighting for residuals; implement as a robust kernel option in core/estimators or core/slam/loam.py.

(e) Visual SLAM helpers + reprojection residuals (Eqs. (7.43)–(7.46), (7.68)–(7.70))

core/slam/camera.py

- distort_normalized(xy: np.ndarray, k1,k2,p1,p2) -> np.ndarray
  Encodes the radial + tangential distortion model (Eqs. (7.43)–(7.46)).

- project_point(K: CameraIntrinsics, p_cam: np.ndarray) -> np.ndarray
  Projection function referenced by Eq. (7.40) and used in the reprojection cost.

core/slam/factors.py

- ReprojectionFactor
  Residual implements the bundle adjustment cost structure in Eqs. (7.68)–(7.70).

- BetweenPoseFactorSE2
  Used for pose graph SLAM constraints produced by scan matching.

4.8.5 Simulation data requirements (for SLAM)

core/slam examples depend on dataset families in data/sim/:

- slam_lidar2d/ (scan matching + pose graph).
- slam_visual_bearing2d/ (synthetic camera observations + bundle adjustment).

Section 5.2 defines the exact file formats.

4.8.6 Ownership split

Navigation engineer

- Decide which Chapter-7 equations are “must-implement” vs “optional demo”.
- Choose simplifications (2D vs 3D; which residuals to include).
- Set nominal noise/outlier parameters for the synthetic datasets.

Software engineer

- Implement file layout, dataclasses, and stable APIs.
- Provide unit tests + CI.
- Keep performance reasonable (KD-tree for ICP; vectorized operations).


4.9 core/fusion
Purpose

Chapter 8 is about *practical* aspects of sensor fusion (coupling choices, observability, tuning, calibration, time synchronization, and efficiency). This repo needs a small cross-cutting module that makes those topics concrete without turning into a production fusion stack.

Design stance

- Do NOT re-implement KF/EKF/UKF/PF/FGO here. Those live in core/estimators.
- core/fusion provides *utilities and glue* that let chapter examples:
  - process asynchronous / multi-rate sensors in timestamp order,
  - log innovations + consistency metrics (NIS/NEES),
  - reject or downweight outliers (gating + robust loss),
  - compensate time offsets (temporal calibration) and interpolate signals,
  - demonstrate calibration impact (extrinsic / lever arm toys).

4.9.1 Key data structures

All structures live in core/fusion/types.py.

StampedMeasurement

from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class StampedMeasurement:
    """Generic time-stamped measurement packet used by fusion demos."""
    t: float                 # seconds (float)
    sensor: str              # e.g. 'imu', 'uwb_range', 'lidar_odom'
    z: np.ndarray            # measurement vector
    R: np.ndarray            # measurement covariance
    meta: dict = field(default_factory=dict)

Time synchronization model

@dataclass(frozen=True)
class TimeSyncModel:
    """Map sensor-local time to a common fusion time."""
    offset: float = 0.0      # seconds
    drift: float = 0.0       # seconds/second (0 means no drift)

    def to_fusion_time(self, t_sensor: float) -> float:
        # t_fusion = (1 + drift) * t_sensor + offset
        ...

Rule

All fusion examples must pass timestamps through TimeSyncModel (even if offset=0) so that temporal calibration is a first-class concept.

4.9.2 Tuning & gating helpers (Chapter 8, Section 8.3)

Innovation (Eq. (8.5))

def innovation(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    """Implements Eq. (8.5): y_k = z_k - h(x_k|k-1)."""
    ...

Innovation covariance (Eq. (8.6))

def innovation_covariance(H: np.ndarray, P_pred: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Implements Eq. (8.6): S_k = H_k P_pred H_k^T + R_k."""
    ...

Robust scaling of measurement covariance (Eq. (8.7))

def scale_measurement_covariance(R: np.ndarray, w: float) -> np.ndarray:
    """Implements Eq. (8.7): R_k <- w(y_k) * R_k."""
    ...

Chi-square gating (Eqs. (8.8)–(8.9))

def chi_square_gate(innov: np.ndarray, S: np.ndarray, alpha: float) -> bool:
    """Implements Eqs. (8.8)–(8.9): accept if d^2 < chi2(m, alpha)."""
    ...

Implementation note

- core/eval.compute_nis already computes d^2 = y^T S^{-1} y. core/fusion should call into core/eval rather than duplicate math.

4.9.3 Multi-rate EKF runner (didactic)

Provide a small runner in core/fusion/run.py that processes a list of StampedMeasurement objects.

def run_multisensor_ekf(
    ekf,
    measurements: list[StampedMeasurement],
    *,
    time_sync: dict[str, TimeSyncModel] | None = None,
    gate=None,
) -> dict:
    """
    Time-order measurements (after applying time_sync), propagate, update, and log:
    - x_hat(t), P(t)
    - innovations y_k, innovation covariances S_k
    - NIS (and optionally NEES if truth is available)
    - accepted/rejected flags per measurement
    """
    ...

This runner is used by the Chapter-8 LC/TC demos to keep each script short and consistent.

4.9.4 Temporal alignment utilities (Chapter 8, Section 8.5)

Provide small, reusable helpers in core/fusion/time_alignment.py:

- interpolate_series(t_src, y_src, t_query, method='linear')
- resample_series(t_src, y_src, dt, method='linear')

Goal: make it trivial to evaluate a measurement at an arbitrary time (e.g., interpolate a state history to a LiDAR timestamp).

4.9.5 Calibration utilities (Chapter 8, Section 8.4)

Provide toy extrinsic calibration helpers in core/fusion/calibration.py:

- estimate_rigid_transform(A_pts, B_pts) -> (R, t)  # SVD / Procrustes
- apply_rigid_transform(R, t, pts) -> pts_transformed

This supports a minimal demo showing how wrong extrinsics degrade fusion.

4.9.6 Equation mapping requirements

docs/equation_index.yml must include:

- Eq. (8.5) -> core/fusion/tuning.py::innovation
- Eq. (8.6) -> core/fusion/tuning.py::innovation_covariance
- Eq. (8.7) -> core/fusion/tuning.py::scale_measurement_covariance
- Eq. (8.8) -> core/eval/metrics.py::compute_nis
- Eq. (8.9) -> core/fusion/gating.py::chi_square_gate

Eq. (8.1)–(8.2) are observability definitions; the repo implements them as a behavioral demo (see Section 7.7.4) rather than a single reusable math routine.

4.9.7 Ownership split

Navigation engineer

- Choose the state for each demo (what is estimated vs assumed known).
- Choose nominal Q/R values, gating confidence alpha, and robust-loss parameters.

Software engineer

- Implement stable APIs and logging so examples can inspect innovations/NIS and accepted/rejected measurements.
- Implement time alignment utilities with unit tests.
- Keep the runner deterministic and well-tested (fixed seeds and exact outputs on toy datasets).



5. Simulation Datasets & Open Data
5.1 Principles

All primary examples must run only on data in data/sim/.

Datasets must be:

Small enough to clone and run quickly.

Reproducible (fixed seeds, recorded configs).

Clearly licensed (e.g. CC-BY-4.0) and documented in docs/data.md.

**Student-accessible and well-documented** (Section 5.3):

Every dataset family must have a README.md explaining purpose, parameters, and usage.

Every generation script must have CLI interface and parameter documentation.

Parameter effects on algorithm performance must be documented for learning purposes.

5.2 Dataset Families

Planned families under data/sim/:

rf_2d_floor/

2D or 2.5D floor map with known beacon/anchor positions (in ENU coordinates).

True agent trajectories:

Static points (for pure point positioning).

Moving paths (for time-series demos and filter comparisons).

Generated measurement files (or generation configs) for:

TOA / two-way TOA ranges d^a_i (Eqs. (4.1)–(4.3), (4.6)–(4.7)).

RSS values p^R_{a,i} with path-loss parameters (Eqs. (4.11)–(4.13)).

TDOA range differences d^a_{i,j} (Eqs. (4.27)–(4.33)).

AOA azimuth/elevation (Eqs. (4.63)–(4.67)).

Noise configuration file (YAML/JSON) specifying:

Timing noise std, oscillator jitter, RSS variance, angle noise.

NLOS bias templates for selected beacons (used in the Chapter 4 “RF challenges” example).

Coordinate frames:

All positions expressed in ENU with a documented origin and height reference.

wifi_fingerprint_grid/

Reference points on a grid, RSS vectors, test trajectories.

pdr_corridor_walk/

IMU time series, reference path, step labels, floor labels.

fusion_2d_imu_uwb/

Multi-sensor fusion dataset designed for Chapter 8 (LC vs TC, tuning, and temporal calibration).

Scenario

- 2D walking or robot trajectory on a simple floor.
- High-rate IMU (or PDR increments) + low-rate UWB range measurements to fixed anchors.
- Optional injected time offset and drift between IMU and UWB timestamps.

Required files

- truth.npz: t, p_xy, v_xy, yaw (ground truth)
- imu.npz: t, accel_xy (or accel_xyz), gyro_z (or gyro_xyz)
- uwb_anchors.npy: (A,2) anchor positions in map frame
- uwb_ranges.npz: t, ranges  # shape (T,A), with NaNs allowed for dropouts
- config.json: dt_imu, dt_uwb, noise params, optional time_offset_sec, optional clock_drift

Used by

- ch8_sensor_fusion LC UWB-position-fix + IMU EKF example.
- ch8_sensor_fusion TC raw-range + IMU EKF example.
- tuning + gating demos (NIS plots + chi-square gating).
- temporal calibration demo (recover performance after compensating time_offset_sec).


slam_lidar2d/

Minimal 2D LiDAR SLAM dataset used in Chapter 7 demos (scan-to-scan, scan-to-map, and loop closure).

Required files

- map_occupancy.npy: (H,W) occupancy grid in map frame M.
- map_meta.json: meters_per_cell, origin (ENU), and frame conventions.
- poses_true.npy: (T,3) ground-truth poses [x,y,yaw] in map frame M.
- scans/: per-timestep point clouds (already converted from range-bearing)
  - scan_0000.npy: (N,2) xy points in the LiDAR/body frame B_t (meters).
  - scan_0001.npy, ...
- timestamps.npy: (T,) timestamps for aligning scans and optional IMU.

Optional (recommended for teaching)

- raw_ranges/: range + bearing arrays if you want to show “sensor → points” explicitly.
- loop_closures.csv: synthetic loop closure constraints (i,j, dx,dy,dyaw, weight/info).
- noise_config.yml: correspondence threshold epsilon (Eq. (7.11)), outlier rate, and scan noise.

Used by

- ch7_slam ICP + NDT + LOAM-style odometry examples.
- tests/ch7 regression tests (pose RMSE thresholds on fixed seeds).

slam_visual_bearing2d/

Toy visual SLAM / bundle adjustment dataset (no real images; synthetic feature tracks).

Required files

- camera_intrinsics.json: fx, fy, cx, cy + distortion (k1,k2,p1,p2).
- poses_true.npy: (T,3) ground-truth camera poses [x,y,yaw] (2D demo).
- landmarks_true.npy: (L,2) landmark positions in map frame.
- observations.csv: (t, landmark_id, u, v) pixel observations + optional outlier flags.

Used by

- ch7_slam bundle adjustment example (Eqs. (7.68)–(7.70)).

toy_ls_linear/

Simple linear system for least squares testing:

Files: H.npy, y.npy, x_true.npy, noise_config.json.

Used by: core/estimators/least_squares.py unit tests and ch3_estimators examples.

toy_kf_1d_cv/

1D constant velocity system:

Files: x_true.npy, z.npy, F.npy, H.npy, Q.npy, R.npy, dt.

Used to validate KalmanFilter implementation and tuning.

toy_nonlinear_1d/

Nonlinear state/measurement example:

State trajectory file, nonlinear measurement config, noise statistics.

Used by: EKF, UKF, PF, and FGO examples to illustrate behavior under non-Gaussian/nonlinear conditions.

For each dataset:

File formats: CSV / NPZ / HDF5.

Coordinate frames: clearly defined, link to core/coords.

Ground truth semantics.

5.3 Dataset Documentation Standards

To enable student learning and experimentation, all simulation datasets and generation tools must follow these documentation standards:

5.3.1 data/sim/README.md (Dataset Catalog)

This file must be the entry point for students exploring simulation data. Required sections:

**Overview**

Purpose of simulation datasets in the IPIN learning context.

How datasets connect to book chapters and algorithms.

**Available Datasets Table**

| Dataset | Purpose | Sensors | Key Parameters | Used In | Generation Script |
|---------|---------|---------|----------------|---------|-------------------|
| `fusion_2d_imu_uwb/` | Baseline fusion | IMU + UWB | No bias, no offset | Ch8 TC/LC | `scripts/generate_fusion_2d_imu_uwb_dataset.py` |
| `wifi_fingerprint_grid/` | Fingerprinting | Wi-Fi RSS | 8 APs, 3 floors | Ch5 kNN/MAP | `scripts/generate_wifi_fingerprint_dataset.py` |

**File Format Reference**

Python code examples showing how to load each dataset format (.npz, .json, .csv).

Explanation of coordinate frames and units.

**Quick Start**

Step-by-step instructions for loading and visualizing a dataset.

Example code snippets for common operations.

**Parameter Effects Guide**

Table explaining how key parameters affect algorithm performance:

| Parameter | Range | Effect on Filter/Algorithm |
|-----------|-------|----------------------------|
| `accel_noise_std` | 0.01-1.0 | Higher = more velocity drift in IMU integration |
| `range_noise_std` | 0.01-0.5 | Higher = noisier UWB fixes, more gating rejections |
| `nlos_bias` | 0-2.0 | Positive bias on affected anchors |

**Generating Custom Datasets**

Pointer to `scripts/README.md` for generation instructions.

5.3.2 Per-Dataset README (e.g., data/sim/fusion_2d_imu_uwb/README.md)

Each dataset folder should contain a README.md with:

**Dataset Description**

Scenario overview (e.g., "2D rectangular walking trajectory with IMU + UWB ranging").

Learning objectives: what students should learn from this dataset.

**Files Included**

List and describe each file:
- `truth.npz`: Ground truth (t, p_xy, v_xy, yaw)
- `imu.npz`: IMU measurements (t, accel_xy, gyro_z)
- `uwb_anchors.npy`: UWB anchor positions
- `uwb_ranges.npz`: UWB range measurements (with NaN for dropouts)
- `config.json`: Dataset configuration parameters

**Loading Example**

```python
import numpy as np
import json

# Load ground truth
truth = np.load('data/sim/fusion_2d_imu_uwb/truth.npz')
t = truth['t']          # (6000,) timestamps
p_xy = truth['p_xy']    # (6000, 2) positions

# Load configuration
with open('data/sim/fusion_2d_imu_uwb/config.json') as f:
    config = json.load(f)
```

**Configuration Parameters**

Detailed explanation of each parameter in `config.json`:
- Purpose and typical range
- Effect on data characteristics
- Relationship to book equations

**Visualization Example**

Code snippet to visualize the trajectory and sensor data.

**Connection to Book**

Which chapter(s) and equations this dataset is designed to demonstrate.

Related example scripts in `chX_*/` folders.

**Variants**

If multiple variants exist (e.g., baseline, NLOS, time offset), explain:
- Purpose of each variant
- Key parameter differences
- When to use each variant for learning

5.3.3 scripts/README.md (Data Generation Guide)

This file must enable students to generate custom datasets for experimentation. Required sections:

**Overview**

Purpose: Allow students to explore parameter sensitivity.

Prerequisites: Python environment setup.

**Quick Start**

```bash
# Generate default datasets
python scripts/generate_fusion_2d_imu_uwb_dataset.py

# Generate with custom parameters
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 \
    --output data/sim/my_experiment
```

**Generation Scripts Inventory**

Table of all generation scripts:

| Script | Generates | Key Parameters | Typical Use |
|--------|-----------|----------------|-------------|
| `generate_fusion_2d_imu_uwb_dataset.py` | IMU + UWB fusion data | noise levels, NLOS, time offset | Ch8 fusion experiments |
| `generate_wifi_fingerprint_dataset.py` | Wi-Fi RSS fingerprint DB | grid spacing, AP count, floors | Ch5 fingerprinting experiments |

**Parameter Reference**

For each major generation script, provide a parameter table:

**generate_fusion_2d_imu_uwb_dataset.py Parameters**

| Parameter | Default | Range | Description | Impact on Learning |
|-----------|---------|-------|-------------|-------------------|
| `--duration` | 60.0 | 10-300 | Trajectory duration (sec) | Longer = more drift accumulation |
| `--accel-noise` | 0.1 | 0.01-1.0 | Accelerometer noise σ (m/s²) | Higher = faster velocity drift |
| `--gyro-noise` | 0.01 | 0.001-0.1 | Gyroscope noise σ (rad/s) | Higher = faster heading drift |
| `--range-noise` | 0.05 | 0.01-0.5 | UWB range noise σ (m) | Higher = noisier position fixes |
| `--nlos-anchors` | [] | [0-3] | NLOS anchor indices | Tests robustness to biased measurements |
| `--nlos-bias` | 0.5 | 0-2.0 | NLOS positive bias (m) | Larger = more severe outliers |
| `--dropout-rate` | 0.05 | 0-0.5 | Measurement dropout probability | Tests missing data handling |
| `--time-offset` | 0.0 | -0.5 to 0.5 | Sensor time offset (sec) | Tests temporal calibration |

**Experimentation Scenarios**

Provide ready-to-use command examples for common learning experiments:

**Scenario 1: Effect of IMU Noise on Drift**

```bash
# Low noise (tactical-grade IMU)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --output data/sim/experiment_low_noise \
    --accel-noise 0.01 --gyro-noise 0.001

# Medium noise (consumer-grade IMU)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --output data/sim/experiment_med_noise \
    --accel-noise 0.1 --gyro-noise 0.01

# High noise (degraded IMU)
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --output data/sim/experiment_high_noise \
    --accel-noise 0.5 --gyro-noise 0.05

# Compare results:
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/experiment_low_noise
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/experiment_med_noise
python -m ch8_sensor_fusion.tc_uwb_imu_ekf --data data/sim/experiment_high_noise
```

Expected learning outcome: Students observe how IMU noise propagates to position drift over time.

**Scenario 2: NLOS Severity Study**

```bash
for bias in 0.2 0.5 1.0 2.0; do
    python scripts/generate_fusion_2d_imu_uwb_dataset.py \
        --output data/sim/nlos_bias_${bias} \
        --nlos-anchors 1 2 --nlos-bias $bias
done

# Compare robustness with and without gating
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/nlos_bias_0.5 --no-gating
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/nlos_bias_0.5  # default uses gating
```

Expected learning outcome: Students see how chi-square gating rejects NLOS outliers.

**Scenario 3: Temporal Calibration Impact**

```bash
# Generate dataset with time offset
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --output data/sim/time_offset_50ms \
    --time-offset -0.05

# Run without correction (degraded performance)
python -m ch8_sensor_fusion.temporal_calibration_demo \
    --data data/sim/time_offset_50ms --no-correction

# Run with correction (recovered performance)
python -m ch8_sensor_fusion.temporal_calibration_demo \
    --data data/sim/time_offset_50ms
```

Expected learning outcome: Students understand importance of temporal synchronization.

**Adding CLI to Generation Scripts**

All generation scripts must support command-line arguments using `argparse`:

```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D IMU + UWB fusion dataset for Chapter 8"
    )
    
    # Trajectory parameters
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Trajectory duration (seconds)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Walking speed (m/s)')
    
    # IMU parameters
    parser.add_argument('--accel-noise', type=float, default=0.1,
                       help='Accelerometer noise std (m/s²)')
    parser.add_argument('--gyro-noise', type=float, default=0.01,
                       help='Gyroscope noise std (rad/s)')
    
    # UWB parameters
    parser.add_argument('--range-noise', type=float, default=0.05,
                       help='UWB range noise std (meters)')
    parser.add_argument('--nlos-anchors', type=int, nargs='*', default=[],
                       help='List of NLOS anchor indices (e.g., 1 2)')
    parser.add_argument('--nlos-bias', type=float, default=0.5,
                       help='NLOS positive bias (meters)')
    parser.add_argument('--dropout-rate', type=float, default=0.05,
                       help='Measurement dropout probability [0-1]')
    
    # Temporal calibration
    parser.add_argument('--time-offset', type=float, default=0.0,
                       help='Sensor time offset (seconds)')
    
    # Output
    parser.add_argument('--output', type=str, 
                       default='data/sim/fusion_2d_imu_uwb',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    generate_fusion_2d_imu_uwb_dataset(
        output_dir=args.output,
        seed=args.seed,
        duration=args.duration,
        speed=args.speed,
        accel_noise_std=args.accel_noise,
        gyro_noise_std=args.gyro_noise,
        range_noise_std=args.range_noise,
        nlos_anchors=args.nlos_anchors,
        nlos_bias=args.nlos_bias,
        dropout_rate=args.dropout_rate,
        time_offset_sec=args.time_offset
    )

if __name__ == "__main__":
    main()
```

**Troubleshooting**

Common issues students may encounter:
- Import errors → check package installation
- Memory errors → reduce duration or sampling rate
- Slow generation → expected for large datasets

5.3.4 docs/data_simulation_guide.md (Comprehensive Learning Guide)

Create a standalone learning guide that bridges theory and practice:

**Purpose**

Connect book equations to simulation parameters.

Guide students through systematic parameter sensitivity experiments.

Provide interpretation frameworks for experimental results.

**Theory-to-Simulation Mapping**

For each major concept, show the connection:

**IMU Error Models (Ch6, Eqs. 6.5-6.9)**

Book equations:
- Gyro error model: ω̃ = ω + b_g + n_g (Eq. 6.5)
- Accel error model: f̃ = f + b_a + n_a (similar to Eq. 6.9)

Simulation parameters:
- `gyro_noise_std`: σ_g for white noise n_g
- `accel_noise_std`: σ_a for white noise n_a
- `gyro_bias`: constant bias b_g
- `accel_bias`: constant bias b_a

Expected behavior:
- Gyro noise → heading drift accumulation
- Accel noise → velocity and position drift
- Bias → systematic drift (unbounded without corrections)

**NLOS Bias (Ch4, RF challenges)**

Book concept: Non-line-of-sight propagation adds positive bias to range measurements.

Simulation parameters:
- `nlos_anchors`: which anchors are affected
- `nlos_bias`: magnitude of positive bias (meters)

Expected behavior:
- Unmitigated: systematic position bias
- With chi-square gating (Ch8): outlier rejection
- With robust loss (Ch8): down-weighting

**Step-by-Step Experiment Guides**

**Experiment 1: IMU Drift Characterization**

Objective: Understand how IMU noise propagates to position error over time.

Setup:
1. Generate three datasets with different noise levels
2. Run pure IMU strapdown integration (no corrections)
3. Compare position error vs. time

Commands: [provided]

Analysis:
- Plot position RMSE vs. time for all three cases
- Compute drift rate (m/s) from linear fit
- Verify relationship: higher noise → faster drift

Key insight: IMU-only positioning is unbounded without corrections (Ch6 finding).

**Experiment 2: Filter Tuning Sensitivity**

Objective: Understand impact of measurement covariance on fusion performance.

Setup:
1. Use fixed dataset (baseline fusion_2d_imu_uwb)
2. Run TC fusion with different R scaling factors
3. Monitor NIS and position accuracy

Expected observations:
- Under-estimated R (scaling < 1): overconfident, possible divergence
- Correctly estimated R (scaling = 1): consistent, optimal
- Over-estimated R (scaling > 1): conservative, suboptimal

Key insight: Proper covariance tuning is critical (Ch8, Section 8.3).

**Parameter Sensitivity Reference Tables**

Provide quick-reference tables for all major parameters showing:
- Typical ranges for different sensor grades
- Expected impact on algorithm metrics (RMSE, drift rate, etc.)
- Recommended values for different learning scenarios

**Common Student Questions**

Q: How do I choose realistic noise parameters?
A: [guidance on sensor datasheets, Allan variance, typical ranges]

Q: Why does my filter diverge with low measurement noise?
A: [explanation of over-confidence and innovation monitoring]

Q: How much data do I need for reliable conclusions?
A: [guidance on trajectory length, statistical significance]

5.3.5 Visualization and Analysis Tools

To support student experimentation, provide helper scripts in `tools/`:

**tools/plot_dataset_overview.py**

Generate summary visualizations for any dataset:
- Trajectory plot with sensor positions
- Sensor measurement timelines
- Noise characteristics (histograms, Allan variance)

Usage:
```bash
python tools/plot_dataset_overview.py data/sim/fusion_2d_imu_uwb
```

**tools/compare_dataset_variants.py**

Compare parameter effects across multiple datasets:
- Side-by-side trajectory plots
- Parameter impact tables
- Performance metric summaries

Usage:
```bash
python tools/compare_dataset_variants.py \
    data/sim/fusion_2d_imu_uwb \
    data/sim/fusion_2d_imu_uwb_nlos \
    data/sim/fusion_2d_imu_uwb_timeoffset
```

5.3.6 Implementation Ownership (Dataset Documentation)

Navigation engineer responsibilities:

- Define parameter ranges and typical values for each sensor type.
- Write theory-to-simulation mapping sections.
- Design experimentation scenarios with clear learning outcomes.
- Validate that parameter effects match theoretical predictions.

Software engineer responsibilities:

- Implement CLI interfaces for all generation scripts.
- Create dataset loading utility functions and examples.
- Build visualization and comparison tools.
- Ensure all documentation examples run correctly.
- Maintain consistency across all dataset READMEs.

5.4 Optional Real Data

data/real/ may contain small demo logs (not full research datasets) to show real-world quirks.

Not a core dependency for running per-chapter examples.

6. Equation–to–Code Mapping Design

This section defines how equation mapping is implemented, enforced, and integrated with core/ and chapter modules.

6.1 Canonical Equation ID Format

Equations are referenced using a canonical string:

Eq. (C.NN) for Chapter C, Equation NN.

If the book uses section-based numbering, adapt to Eq. (C.SS).

Examples:

Chapter 2, equation 5 → Eq. (2.5)

Chapter 3, equation 12 → Eq. (3.12)

Rules:

This exact string must appear in:

Docstrings.

Comments (for line-level mapping).

Notebooks (markdown cells).

docs/equation_index.yml.

That way, a simple repo search for "Eq. (3.12)" finds all relevant code/tests.

6.2 Code Conventions (Docstrings & Comments)

Requirement:

Every function/class that directly implements a book equation must mention it in its docstring.

Example: LS

def linear_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the linear least squares problem x* = argmin ||Ax - b||_2.

    Implements Eqs. (3.2)–(3.3) in Chapter 3 of the book:
    - Normal equations: (A^T A) x* = A^T b   (Eq. (3.2))
    - Closed-form solution: x* = (A^T A)^{-1} A^T b   (Eq. (3.3))

    """
    ...


Example: EKF predict

def predict(self, u: np.ndarray) -> None:
    """
    Extended Kalman filter prediction step.

    Implements Eqs. (3.21)–(3.22) in Chapter 3:
    - Nonlinear process model definition: Eq. (3.21)
    - EKF prediction (state + covariance propagation using the Jacobian F): Eq. (3.22)
    """
    ...


Line-level comments

If a specific line encodes an equation:

# Eq. (3.22): P_k^- = F_{k-1} P_{k-1} F_{k-1}^T + Q
P_pred = F @ P @ F.T + Q


Design rule:

Whole function = one equation (or block) → docstring reference.

Single expression or key step = equation → inline comment with the exact Eq. (C.NN) reference.

6.3 Central Equation Index File

Create docs/equation_index.yml (or .json).

Purpose:

From the book side, you can look up "Eq. (3.12)" and see:

Which modules/objects implement it.

Which tests/notebooks exercise it.

YAML structure (example):

- eq: "Eq. (2.3)"
  chapter: 2
  description: "ECEF to ENU coordinate transformation"
  files:
    - path: "core/coords/frames.py"
      object: "ecef_to_enu"
    - path: "ch2_coords/examples/coord_demo.py"
      object: "demo_ecef_to_enu"
  tests:
    - "tests/test_coords.py::test_ecef_enu_roundtrip"

- eq: "Eq. (3.3)"
  chapter: 3
  description: "Linear least squares closed-form solution"
  files:
    - path: "core/estimators/least_squares.py"
      object: "linear_least_squares"
  notebooks:
    - "notebooks/ch3_estimators/ls_vs_ekf.ipynb"


Design choices:

One entry per implemented equation.

Keys:

eq – canonical ID ("Eq. (3.12)").

chapter

description

files – list of {path, object}.

Optional: tests, notebooks, notes.

6.4 Notebooks & Docs Conventions

Notebooks:

At the top of each chapter notebook, add an “Equations used” list, e.g.:

Eq. (3.12): Linear least squares solution.

Eqs. (3.25)–(3.27): EKF prediction equations.

And reference equations in text:

We now implement the linear least squares estimator from Eq. (3.12).

Docs:

In docs/ch3_estimators.md:

Equation map

Eq. (3.12): core/estimators/least_squares.py::linear_least_squares

Eqs. (3.25)–(3.27): core/estimators/kalman.py::ExtendedKalmanFilter.predict

This gives users three navigation routes:

Search "Eq. (3.12)" in the repo.

Check docs/equation_index.yml.

Read chapter docs listing mappings.

6.5 Tooling: Equation Index Checker Script

Add tools/check_equation_index.py:

Responsibilities:

Parse docs/equation_index.yml.

For every {path, object} entry:

Verify the module and object exist.

Verify the docstring or comments contain the mapped "Eq. (C.NN)".

Sketch (conceptually):

Use importlib to load modules.

Inspect obj.__doc__.

Fail CI if any mismatch.

Integrate in CI:

Run pytest.

Run python tools/check_equation_index.py.

6.6 Integration into Epics / Phases

For each epic, “definition of done” includes:

Code & tests written.

Example notebook added.

Equation mapping completed:

Docstring references added.

equation_index.yml updated.

Notebook/docs list relevant equations.

Concretely:

Epic 1 – Coordinates (Ch.2)
All transforms tagged with Eq. (2.x) where applicable.

Epic 2 – Estimators (Ch.3)
LS, KF, EKF, UKF, PF, FGO all mapped to Ch.3 equations.

Epic 3 – RF positioning (Ch.4)
TOA/TDOA/AOA/RSS and DOP models mapped.

Subsequent epics (fingerprinting, PDR, SLAM, fusion) follow the same pattern.

7. Per-Chapter Example Modules (Unique Tasks)

For each chX_... folder, list:

Example scripts / notebooks.

Which core/ functions they exercise.

Which equations they highlight.

Standard visualization outputs (all chapters)

For each chX_... module, at least one example must:

call core/eval metrics to compute errors / statistics, and

call core/eval/plots.py to generate:

a trajectory or map view, and

at least one error/time or error/CDF plot.

All examples should save figures via save_figure(...) into:

chX_.../figs/ (scripts), or

notebooks/figs/chX_.../ (notebooks),

so that:

readers can quickly compare different algorithms or datasets by opening the SVG/PDF files, and

instructors can directly reuse the vector figures in slides and course materials without manual re-plotting.

7.1 ch2_coords/

Examples

demo_frames.py
Build body/map/ENU/NED frames and print transforms.

Simple plotting script to reproduce or echo Ch.2 figures.

Dependencies

core.coords, core.sim, core.eval.

7.2 ch3_estimators/

Examples

ls_vs_robust_ls.py
Outlier demo: LS vs Huber vs Cauchy.

kf_vs_ekf_vs_ukf_vs_pf_vs_fgo.ipynb
1D/2D toy system comparing estimator families.

Unique tasks

Compare convergence, RMSE, computation time.

Show qualitatively how different estimators behave under same conditions.

7.3 ch4_rf_point_positioning/ – Point Positioning by Radio Signals (Chapter 4)
7.3.1 Purpose

This module provides end-to-end simulation examples for Chapter 4: Point Positioning by Radio Signals.

Main goals:

Demonstrate TOA, two-way TOA/RTT, RSS-based ranging, TDOA, and AOA positioning on a common simulated floorplan (rf_2d_floor dataset).

Compare:

Iterative WLS / I-WLS vs closed-form algorithms (Fang, Chan).

Different measurement types (TOA vs TDOA vs AOA vs RSS) under the same geometry and noise levels.

Make the link between Chapter 4 equations and actual numerical behavior explicit via equation-level traceability.

This is a Chapter-specific wiring layer: it uses core/rf, core/estimators, core/coords, core/sim, and core/eval, plus data/sim/rf_2d_floor/.

7.3.2 Dependencies

Core modules

core/rf – TOA/TDOA/AOA/RSS measurement models, DOP utilities.

core/estimators – LS / WLS / robust LS, Gauss–Newton, LM, FGO.

core/coords – ENU frame and beacon/agent coordinates.

core/sim – floorplan, anchors, trajectory generation, noise injection.

core/eval – error metrics, CDFs, DOP and geometry plots.

Data

data/sim/rf_2d_floor/ (Section 5.2).

Notebooks

notebooks/ch4_rf_point_positioning/*.ipynb.

7.3.3 Example scenarios & scripts
Scenario A – Direct ranging: TOA / two-way TOA / RSS (Section 4.2)

Key equations

TOA models:

Basic TOA range and variants with clock offsets (Eqs. (4.1)–(4.3)).

Two-way TOA (RTT):

RTT distance with and without processing delay (Eqs. (4.6)–(4.7)).

RSS path-loss:

Path-loss inversion and fading/noise models (Eqs. (4.11)–(4.13)).

Nonlinear TOA LS/I-WLS:

Measurement vector, state, nonlinear range, linearization and iterative WLS update (Eqs. (4.14)–(4.23)).

Joint position + clock bias:

Extended state and modified TOA model (Eqs. (4.24)–(4.26)).

Planned examples

ch4_rf_point_positioning/direct_ranging_tutorial.ipynb

Generate TOA / two-way TOA / RSS measurements from rf_2d_floor using core/rf.

Implement:

Plain LS on ranges (linearized Eq. (4.16)).

I-WLS using Eqs. (4.14)–(4.23).

Compare TOA vs RTT vs RSS positioning under different noise levels.

Show how initial guess and weighting matrix affect convergence and accuracy.

ch4_rf_point_positioning/direct_ranging_clock_offset.py

Include clock bias in the state:

State vector and distance model from Eqs. (4.24)–(4.26).

Demonstrate joint estimation of position and clock bias.

Equation mapping

Docstrings explicitly reference:

“TOA model in Eqs. (4.1)–(4.3), (4.6)–(4.7)”

“Nonlinear TOA I-WLS algorithm from Eqs. (4.14)–(4.23)”

“Joint position + clock bias formulation from Eqs. (4.24)–(4.26)”

equation_index.yml maps these equations to:

TOA/RSS functions in core/rf.

Example helpers in ch4_rf_point_positioning.

Scenario B – TDOA positioning: LS / WLS vs Fang & Chan (Section 4.3)

Key equations

TDOA definition and range differences:

Uplink/downlink TDOA and range differences (Eqs. (4.27)–(4.33)).

TDOA LS/WLS:

Linearized TDOA system, LS and WLS solutions, weighting matrix (Eqs. (4.34)–(4.42)).

Fang & Chan closed-form hyperbolic algorithms:

Hyperbolic relationships and algebraic LS forms (Eqs. (4.43)–(4.48), plus refinement in Eq. (4.58)).

Planned examples

ch4_rf_point_positioning/tdoa_ls_vs_iwls.ipynb

Use TDOA measurements from rf_2d_floor.

Implement:

Basic TDOA LS.

WLS with correct covariance weighting.

Compare with TOA-based positioning (Scenario A).

ch4_rf_point_positioning/tdoa_fang_chan.py

Implement Fang’s closed-form TDOA algorithm.

Implement Chan’s two-step algorithm:

Step 1: LS.

Step 2: WLS refinement with covariance.

Compare Fang, Chan, and LS/I-WLS under:

Varying noise levels.

Good vs. poor anchor geometry.

Equation mapping

Docstrings reference:

“TDOA LS/WLS from Eqs. (4.34)–(4.42).”

“Fang’s algorithm from Eqs. (4.43)–(4.48).”

“Chan’s two-step refinement including Eq. (4.58).”

equation_index.yml maps these equations to:

TDOA helpers in core/rf.

Scenario scripts in ch4_rf_point_positioning.

Scenario C – AOA positioning: LS/I-WLS, OVE, 3D PLE (Section 4.4)

Key equations

AOA geometry:

Vertical and azimuth angle relationships (Eqs. (4.63)–(4.64)).

AOA measurement vectors and stacking:

Single-beacon and multi-beacon measurement vectors (Eqs. (4.65)–(4.66)).

State vector consistency:

State definition reused from range-based positioning (Eq. (4.35)).

Linearization and LS/I-WLS:

Linearized form and Jacobians of the AOA model (Eqs. (4.63)–(4.67)).

Planned examples

ch4_rf_point_positioning/aoa_ls_iwls.ipynb

Generate azimuth/elevation AOA measurements for anchors in rf_2d_floor.

Implement:

LS AOA solver based on linearized Eqs. (4.63)–(4.67).

I-WLS refinement using multiple iterations.

ch4_rf_point_positioning/aoa_ove_3dple.py

Implement OVE and 3D PLE methods as described in Section 4.4.

Compare OVE and 3D PLE vs LS/I-WLS under:

Different noise levels.

Different beacon height distributions and geometries.

Equation mapping

Docstrings indicate:

“AOA LS/I-WLS based on Eqs. (4.63)–(4.67).”

“OVE / 3D PLE built on the same AOA geometry (Eqs. (4.63)–(4.67)).”

equation_index.yml links these equations to:

AOA models in core/rf.

AOA demo scripts in ch4_rf_point_positioning.

Scenario D – RF challenges & limitations (Section 4.5)

Focus

Illustrate limitations and realistic imperfections discussed in Section 4.5:

Thermal noise and oscillator instability.

Multipath and NLOS biases.

Poor beacon geometry (high DOP).

Sensitivity to initialization in nonlinear WLS.

Planned example

ch4_rf_point_positioning/rf_challenges.ipynb

Baseline scenario:

Well-conditioned TOA/TDOA/AOA geometry and low noise.

Progressive stress tests:

Add oscillator noise:

Show how timing noise propagates into range and position errors.

Tie back to assumptions in TOA/TDOA models (Eqs. (4.1)–(4.3), (4.27)–(4.33)).

Add NLOS:

Inject biases on selected anchors using rf_2d_floor noise config.

Compare LS vs I-WLS robustness.

Change geometry:

Cluster anchors or limit them to one side; visualize DOP using core/eval.

Relate GDOP/HDOP/PDOP to position error statistics.

Bad initialization:

Show TOA I-WLS divergence or convergence to wrong local minima for poor initial guesses.

Equation mapping

Markdown cells explicitly mention which ideal assumptions from Chapter 4’s equations are being violated in each experiment.

These are behavior-focused demos rather than new implementations, but they still reference:

TOA/TDOA/AOA model equations.

DOP formulas (from core/rf and Chapter 4).

7.3.4 Tests

Scenario-level regression tests to complement core/ unit tests:

tests/ch4/test_direct_ranging_accuracy.py
For a fixed rf_2d_floor config and random seed:
Check TOA I-WLS RMSE is below a threshold (e.g. < ε m).

tests/ch4/test_tdoa_chan_fang_consistency.py
For a simple configuration:
Confirm Fang and Chan solutions are close to LS/I-WLS when noise is small.

tests/ch4/test_aoa_methods.py
Verify LS/I-WLS/OVE/3D PLE solutions converge within tolerance in a nominal AOA scenario.

These tests ensure the wiring between datasets, core/ models, and Chapter-4 examples remains intact.

7.3.5 Dataset Documentation Requirements (Chapter 4)

Following Section 5.3 standards, Chapter 4 must include:

**data/sim/rf_2d_floor/README.md**

Must document:
- Beacon/anchor placement rationale and DOP implications
- RF measurement types available (TOA, TDOA, AOA, RSS)
- Noise model parameters and typical ranges
- NLOS bias injection for RF challenges demo
- Loading examples for each measurement type

Parameter effects table:
| Parameter | Range | Effect on Positioning |
|-----------|-------|----------------------|
| `timing_noise_std` | 1-50 ns | Directly propagates to range error (× c) |
| `rss_sigma` | 2-10 dBm | Larger = more variable RSS-based ranging |
| `nlos_bias` | 0-5 m | Positive bias on affected beacons |
| `beacon_geometry` | varies | Poor geometry → high DOP → amplified errors |

**scripts/generate_rf_2d_floor_dataset.py** (if not yet implemented)

Must support CLI for:
- Number and placement of beacons (geometry experiments)
- Trajectory type (static grid, moving paths)
- Noise levels for each RF measurement type
- NLOS beacon selection

Example experimentation scenario:
```bash
# Study DOP impact: clustered vs distributed beacons
python scripts/generate_rf_2d_floor_dataset.py \
    --beacon-layout clustered --output data/sim/rf_high_dop

python scripts/generate_rf_2d_floor_dataset.py \
    --beacon-layout distributed --output data/sim/rf_low_dop
```

Connection to book:
- Beacon geometry → DOP (Ch4, Section 4.1)
- Timing noise → TOA accuracy (Ch4, Eqs. 4.1-4.3)
- NLOS modeling → RF challenges (Ch4, Section 4.5)

7.4 ch5_fingerprinting/
7.4.1 Goals

Demonstrate deterministic vs probabilistic fingerprinting as introduced in Chapter 5.

Show pattern-recognition formulations (classification/regression) using the same synthetic fingerprints.

Optionally demonstrate a model-based + particle/hypothesis example using ray-tracing-like simulation and the position-hypothesized method.

Every example must:

Use core/fingerprinting for algorithms.

Use core/eval for error metrics and CDF/trajectory plots.

Call save_figure(...) to output vector figures into ch5_fingerprinting/figs/.

7.4.2 Example 1 – Deterministic NN / k-NN (Eqs. (5.1)–(5.2))

Files:

ch5_fingerprinting/demo_deterministic_nn_knn.py

notebooks/ch5_fingerprinting/demo_deterministic_nn_knn.ipynb

Core functions used:

core.fingerprinting.deterministic.nn_localize (Eq. (5.1)).

core.fingerprinting.deterministic.knn_localize (Eq. (5.2)).

core.eval.compute_position_errors, core.eval.plot_error_cdf, core.eval.plot_trajectory_2d.

Equations to highlight:

Eq. (5.1): nearest-neighbor decision rule.

Eq. (5.2): weighted k-NN interpolation.

Notebook structure:

Markdown header “Equations used in this notebook”:

Eq. (5.1): Nearest-neighbor fingerprinting rule.

Eq. (5.2): Weighted k-NN interpolated position estimate.

Code cells:

Load wifi_fingerprint_grid/.

Simulate query fingerprints along a test trajectory (add noise).

Run NN and k-NN for different k (e.g. 1, 3, 5, 10).

Compute RMSE and error CDFs using core.eval.

Plots:

ch5_fingerprinting/figs/ch5_nn_vs_knn_cdf.svg.

ch5_fingerprinting/figs/ch5_nn_vs_knn_traj.svg.

Equation mapping:

Notebook header explicitly lists Eqs. (5.1)–(5.2).

docs/ch5_fingerprinting.md “Equation map” section will include:

Eq. (5.1): core/fingerprinting/deterministic.py::nn_localize.

Eq. (5.2): core/fingerprinting/deterministic.py::knn_localize.

7.4.3 Example 2 – Probabilistic (Bayesian) fingerprinting (Eqs. (5.3)–(5.5))

Files:

ch5_fingerprinting/demo_probabilistic_bayes.py

notebooks/ch5_fingerprinting/demo_probabilistic_bayes.ipynb

Core functions used:

fit_gaussian_naive_bayes, posterior, map_localize, posterior_mean_localize from core.fingerprinting.probabilistic.

core.eval.compute_position_errors, core.eval.plot_error_cdf.

Equations to highlight:

Eq. (5.3): posterior
P(x_i | Z_i = z).

Eq. (5.4): ML/MAP estimate.

Eq. (5.5): posterior-mean estimate.

Notebook structure:

Markdown header “Equations used in this notebook”:

Eq. (5.3): Bayesian posterior over locations.

Eq. (5.4): ML/MAP fingerprint-based location estimate.

Eq. (5.5): Posterior-mean location.

Code:

Load wifi_fingerprint_grid/ with multiple samples per RP.

Fit Naive Bayes model (means/variances per RP).

Compare:

Deterministic NN (from Example 1).

MAP estimator (Eq. 5.4).

Posterior mean estimator (Eq. 5.5).

Plots:

ch5_fingerprinting/figs/ch5_det_vs_prob_cdf.svg.

Optional: posterior heatmap for a single query point.

Equation mapping:

Notebook references Eqs. (5.3)–(5.5) in text.

equation_index.yml entries point to posterior, map_localize, posterior_mean_localize.

7.4.4 Example 3 – Pattern recognition (classification & regression) – Section 5.2

File:

notebooks/ch5_fingerprinting/demo_pattern_recognition.ipynb

Core functions used:

New simple wrappers in core/fingerprinting/pattern_recognition.py, e.g.:

LinearRegressionLocalizer (x̂ = W f + b).

Optional: simple MLP classifier/regressor (thin wrapper around scikit-learn or PyTorch if allowed).

Concepts to highlight:

Classification view: treating each reference point / area as a class.

Regression view: mapping features directly to continuous coordinates.

Notebook structure:

Markdown references to Chapter 5, Section 5.2.1 (classification) and 5.2.2 (regression).

Code:

Build a classifier across regions/cells of the grid.

Build a regression model for continuous x,y from features.

Compare performance to NN/k-NN and Bayesian methods.

Equation mapping:

If the book defines explicit equations (e.g. linear regression form), docstrings in LinearRegressionLocalizer reference the Eq. (5.x); otherwise, they reference the section numbers (5.2.1 / 5.2.2).

7.4.5 Example 4 – Model-based + PF demo (Eqs. (5.7)–(5.11), optional/advanced)

File:

notebooks/ch5_fingerprinting/demo_raytracing_pf.ipynb

Core modules used:

core.sim – to generate a simple 2D map and simulate “ray-tracing-like” features.

core.fingerprinting.model_based – likelihood helpers.

Optionally core.estimators.ParticleFilter as a base implementation.

Equations to highlight:

Eq. (5.7): Gaussian prior over hypothesized states around an initial guess.

Eqs. (5.8)–(5.10): Gaussian likelihoods of simulated measurements.

Eq. (5.11): weighted average over hypotheses to produce an estimate.

Notebook structure:

Header “Equations used in this notebook” listing Eqs. (5.7)–(5.11).

Code:

Define a small 2D environment and “ray-tracing-like” simulator (simplified).

Sample particles around a prior mean (Eq. 5.7).

Compute likelihoods with gaussian_likelihood / raytrace_likelihood (Eqs. 5.8–5.10).

Compute weighted average of particles as position estimate (Eq. 5.11).

Compare performance vs a simpler deterministic method.

Plots:

Particle cloud colored by likelihood.

Position error statistics compared to deterministic fingerprinting.

Equation mapping:

Eqs. (5.7)–(5.11) are referenced in notebook markdown and in equation_index.yml as notebook-based implementations (not core functions).

7.4.6 Dependencies

core.fingerprinting (deterministic, probabilistic, pattern-recognition wrappers).

core.rf and core.sim (to generate synthetic fingerprints for wifi_fingerprint_grid/).

core.eval for metrics and visualization.

data/sim/wifi_fingerprint_grid/.

7.4.7 Dataset Documentation Requirements (Chapter 5)

Following Section 5.3 standards, Chapter 5 must include:

**data/sim/wifi_fingerprint_grid/README.md**

Must document:
- Fingerprint database structure (reference points, features, metadata)
- Path-loss model parameters and multi-floor attenuation
- AP placement strategy and coverage characteristics
- Loading examples for deterministic and probabilistic methods
- Query trajectory generation for evaluation

Parameter effects table:
| Parameter | Range | Effect on Fingerprinting |
|-----------|-------|--------------------------|
| `grid_spacing` | 1-10 m | Finer grid → better resolution but more data |
| `n_aps` | 4-20 | More APs → better uniqueness, less ambiguity |
| `shadow_fading_std` | 2-8 dBm | Higher → more variable RSS, harder matching |
| `floor_attenuation` | 10-20 dB | Affects cross-floor discrimination |

**scripts/generate_wifi_fingerprint_dataset.py** enhancement

Must support CLI for:
- Grid spacing and area size
- Number and placement of APs
- Path-loss parameters (exponent, shadow fading)
- Number of floors and floor height
- Number of RSS samples per reference point (for probabilistic methods)

Example experimentation scenario:
```bash
# Study AP density impact
python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 4 --output data/sim/wifi_sparse_aps

python scripts/generate_wifi_fingerprint_dataset.py \
    --n-aps 12 --output data/sim/wifi_dense_aps

# Compare k-NN performance
python -m ch5_fingerprinting.example_deterministic \
    --data data/sim/wifi_sparse_aps
python -m ch5_fingerprinting.example_deterministic \
    --data data/sim/wifi_dense_aps
```

Connection to book:
- Grid resolution → NN/k-NN accuracy (Ch5, Eqs. 5.1-5.2)
- RSS variance → probabilistic uncertainty (Ch5, Eqs. 5.3-5.5)
- AP coverage → feature uniqueness

7.5 ch6_dead_reckoning/

(Keep this short summary at the top for readability)

Examples (summary)

Foot-mounted PDR with ZUPT:

Walk along corridor, accumulate drift, correct with constraints.

Vehicle odom + IMU:

Simple car-like motion on a map.

7.5.1 Purpose

This module provides end-to-end simulation examples for Chapter 6:

Strapdown IMU propagation (shows drift).

Wheel odometry dead reckoning (shows drift + sensitivity).

Integrated IMU + wheel EKF (reduces drift).

Foot-mounted ZUPT INS (constraint-based drift correction).

Step-and-heading PDR baseline (simple but fragile).

Environmental sensors (magnetometer heading and barometer altitude).

All examples must:

Use core/sensors for models and algorithms.

Use core/estimators for KF/EKF update machinery (single source of truth).

Use core/eval for metrics and vector plot export (save_figure).

7.5.2 Dependencies

Core modules

core/sensors – Chapter 6 models.

core/estimators – KF/EKF used for constraint and fusion updates.

core/coords – quaternion/rotmat conversions.

core/sim – trajectory generation + sensor noise injection.

core/eval – metrics and standardized plots.

Data

data/sim/ch6_* (see Section 5.2).

Notebooks

notebooks/ch6_dead_reckoning/*.ipynb.

7.5.3 Example scenarios & notebooks/scripts
Scenario 6A – Strapdown IMU propagation (drift demo)

Key equations (Chapter 6)

Quaternion kinematics and discrete update: Eqs. (6.2)–(6.4), Ω matrix Eq. (6.3).

Gyro model/correction: Eqs. (6.5)–(6.6).

Velocity and gravity: Eqs. (6.7)–(6.8).

Accelerometer model and position update: Eqs. (6.9)–(6.10).

Notebook

notebooks/ch6_dead_reckoning/strapdown_imu_drift.ipynb

Must show

Perfect IMU vs biased IMU drift.

Trajectory plot (truth vs estimate) + error vs time.

Save vector outputs into notebooks/figs/ch6_dead_reckoning/.

Scenario 6B – Wheel odometry dead reckoning (vehicle)

Key equations

Lever arm compensation and skew matrix: Eqs. (6.11)–(6.12).

Velocity transform and position update: Eqs. (6.14)–(6.15).

Notebook

notebooks/ch6_dead_reckoning/wheel_odometry_dr.ipynb

Must show

DR drift under noise.

Failure mode under injected slip.

Scenario 6C – Integrated IMU + wheel EKF (vehicle)

Key equations

State definition: Eq. (6.16).

Process propagation and covariance: Eqs. (6.17)–(6.32).

Wheel measurement model and Jacobian: Eqs. (6.33)–(6.38).

EKF update block: Eqs. (6.39)–(6.43).

Notebook

notebooks/ch6_dead_reckoning/imu_wheel_ekf.ipynb

Must show

Compare:

IMU-only strapdown

wheel-only DR

integrated EKF

Report RMSE and drift summary.

Implementation rule

Do not duplicate EKF update math here; call core/estimators EKF update.

Scenario 6D – Foot-mounted ZUPT INS (drift correction)

Key equations

ZUPT detector: Eq. (6.44)

ZUPT pseudo-measurement: Eq. (6.45)

Optional: ZARU pseudo-measurement: Eq. (6.60)

Notebook

notebooks/ch6_dead_reckoning/foot_zupt_ins.ipynb

Must show

IMU drift vs ZUPT-corrected drift.

Plot detector firing timeline.

Scenario 6E – Step-and-heading PDR baseline

Key equations

Acceleration magnitude and gravity removal: Eqs. (6.46)–(6.47).

Step frequency, step length, step update: Eqs. (6.48)–(6.50).

Notebook

notebooks/ch6_dead_reckoning/pdr_step_heading.ipynb

Must show

Sensitivity to user height h and personal factor c.

Compare heading sources:

gyro yaw integration (drifts)

magnetometer heading (can jump under disturbance)

Scenario 6F – Magnetometer + barometer demos

Key equations

Magnetometer heading: Eqs. (6.51)–(6.53).

Barometer altitude: Eq. (6.54).

Generic state/measurement form: Eq. (6.55) (for smoothing helper).

Notebook

notebooks/ch6_dead_reckoning/env_heading_altitude.ipynb

Must show

Mag disturbance intervals causing heading jumps.

Barometer altitude trend and offset handling.

Optional smoothing using core/estimators.

7.5.4 Tests (Chapter 6)

Add tests under tests/ch6/:

tests/ch6/test_quaternion_propagation.py
Validate quat_integrate normalization and sanity for simple rotation cases.
(Eqs. (6.2)–(6.4))

tests/ch6/test_strapdown_stationary.py
Stationary IMU should maintain near-zero velocity and stable position (within tolerance).
(Eqs. (6.7)–(6.10))

tests/ch6/test_wheel_odometry_update.py
Known straight-line motion: wheel DR matches truth within tolerance.
(Eqs. (6.11)–(6.15))

tests/ch6/test_imu_wheel_ekf_smoke.py
Integrated EKF RMSE < min(IMU-only, wheel-only) for a fixed seed and config.
(Eqs. (6.16)–(6.43))

tests/ch6/test_zupt_detector.py
Synthetic stance intervals trigger Eq. (6.44) detector behavior.

tests/ch6/test_pdr_step_length.py
Validate step length computation and step update correctness.
(Eqs. (6.48)–(6.50))

tests/ch6/test_mag_baro_models.py
Validate magnetometer heading and barometer altitude transforms.
(Eqs. (6.51)–(6.54))

7.5.5 Standard visualization outputs (Chapter 6)

Each notebook/script must generate at least:

Trajectory plot (truth vs estimate) → SVG/PDF

Error vs time plot (position error, heading error, altitude error) → SVG/PDF

(Optional) CDF plot for position error if comparing multiple methods

Naming examples

ch6_strapdown_drift_traj.svg

ch6_imu_wheel_ekf_error_time.pdf

ch6_zupt_detector_timeline.svg

ch6_pdr_heading_comparison.svg

7.5.6 Equation mapping entries to add (Chapter 6)

Add entries in docs/equation_index.yml mapping Chapter-6 equations to code objects, for example:

Eqs. (6.2)–(6.4) → core/sensors/strapdown.py::quat_integrate

Eq. (6.3) → core/sensors/strapdown.py::omega_matrix

Eqs. (6.5)–(6.6) → core/sensors/imu_models.py::correct_gyro

Eq. (6.7) → core/sensors/strapdown.py::vel_update

Eq. (6.8) → core/sensors/strapdown.py::gravity_vector

Eq. (6.9) → core/sensors/imu_models.py::correct_accel

Eq. (6.10) → core/sensors/strapdown.py::pos_update

Eqs. (6.11)–(6.15) → core/sensors/wheel_odometry.py::{skew, wheel_speed_to_attitude_velocity, attitude_to_map_velocity, odom_pos_update}

Eqs. (6.16)–(6.32) → core/sensors/ins_ekf_models.py::InsWheelProcessModel

Eqs. (6.33)–(6.38) → core/sensors/ins_ekf_models.py::WheelSpeedMeasurementModel

Eqs. (6.39)–(6.43) → core/estimators/kalman.py::ExtendedKalmanFilter.update
(extend docstring to mention Chapter 6 EKF update block)

Eq. (6.44) → core/sensors/constraints.py::detect_zupt

Eq. (6.45) → core/sensors/constraints.py::ZuptMeasurementModel

Eqs. (6.46)–(6.47) → core/sensors/pdr.py::{total_accel_magnitude, remove_gravity_from_magnitude}

Eqs. (6.48)–(6.50) → core/sensors/pdr.py::{step_frequency, step_length, pdr_step_update}

Eqs. (6.51)–(6.53) → core/sensors/environment.py::mag_heading

Eq. (6.54) → core/sensors/environment.py::pressure_to_altitude

Eq. (6.55) → core/sensors/environment.py::(smoothing helper, if implemented)

Eqs. (6.56)–(6.58) → core/sensors/calibration.py::allan_variance

Eq. (6.59) → core/sensors/calibration.py::apply_imu_scale_misalignment

Eq. (6.60) → core/sensors/constraints.py::ZaruMeasurementModel

Eq. (6.61) → core/sensors/constraints.py::NhcMeasurementModel

7.5.7 Dataset Documentation Requirements (Chapter 6)

Following Section 5.3 standards, Chapter 6 must include READMEs for all dataset families:

**data/sim/ch6_strapdown_basic/README.md**

Parameter effects table:
| Parameter | Range | Effect on Strapdown |
|-----------|-------|---------------------|
| `gyro_bias` | 0-0.1 rad/s | Causes heading drift proportional to bias × time |
| `accel_bias` | 0-0.5 m/s² | Causes velocity drift → quadratic position drift |
| `gyro_noise_std` | 0.001-0.1 rad/s | Random walk in heading |
| `accel_noise_std` | 0.01-1.0 m/s² | Random walk in velocity and position |

Expected learning: IMU drift is unbounded without corrections (Ch6 key insight).

**data/sim/ch6_wheel_odom_square/README.md**

Parameter effects:
| Parameter | Range | Effect |
|-----------|-------|--------|
| `wheel_noise_std` | 0.01-0.2 m/s | Odometry drift rate |
| `lever_arm_error` | 0-0.5 m | Systematic error during turns |
| `slip_segments` | varies | Failure mode demonstration |

**data/sim/ch6_foot_zupt_walk/README.md**

Must document ZUPT detector thresholds (δ_ω, δ_f from Eq. 6.44):
| Parameter | Range | Effect |
|-----------|-------|--------|
| `delta_omega` | 0.1-0.5 rad/s | Lower → more ZUPT detections (may include false positives) |
| `delta_f` | 0.5-2.0 m/s² | Lower → more ZUPT detections |

Expected learning: ZUPT dramatically reduces drift during stance phases.

**data/sim/ch6_pdr_corridor_walk/README.md**

Parameter effects:
| Parameter | Range | Effect on PDR |
|-----------|-------|---------------|
| `user_height` | 1.5-2.0 m | Affects step length model (Eq. 6.49) |
| `step_length_params` (a,b,c) | varies | Personal calibration factors |
| `heading_source` | gyro/mag | Gyro drifts; mag has disturbances |

**data/sim/ch6_env_sensors_heading_altitude/README.md**

Documents magnetometer disturbance segments and barometer drift characteristics.

**Generation script requirements**

All ch6 dataset generation scripts must support CLI for IMU grade selection:

```bash
python scripts/generate_ch6_strapdown_dataset.py \
    --imu-grade tactical \  # presets: tactical, consumer, mems
    --output data/sim/my_strapdown_test

# Or custom parameters:
python scripts/generate_ch6_strapdown_dataset.py \
    --gyro-noise 0.05 --accel-noise 0.3 \
    --output data/sim/my_custom_imu
```

Connection to book:
- Noise parameters → Allan variance characterization (Ch6, Eqs. 6.56-6.58)
- Drift behavior → Error propagation analysis (Ch6, Sections 6.1-6.2)
- ZUPT effectiveness → Constraint-based correction (Ch6, Eqs. 6.44-6.45)


7.6 ch7_slam/
7.6.1 Goals

Chapter 7 is about SLAM technologies (LiDAR SLAM and visual SLAM) and why they matter for indoor positioning.

This repo’s goal is to make Chapter-7 math executable without turning the repository into a full robotics SLAM stack.

What Chapter-7 examples must achieve

- Demonstrate scan matching objectives (ICP / NDT) and how they produce relative pose constraints.
- Show how feature selection (LOAM-style) reduces compute while keeping accuracy.
- Show loop closure and why global optimization (pose graph / factor graph) reduces drift.
- Show the basic bundle adjustment objective on synthetic camera observations.
- Keep everything small and reproducible (data/sim/, fixed seeds, fast runtime).

What Chapter-7 examples must NOT become

- A production LOAM/LIO-SAM/VINS implementation.
- A ROS2 pipeline.
- A massive dataset + tuning project.

7.6.2 Core Chapter-7 equations to implement (minimum set)

LiDAR scan matching (ICP-style)

- Eq. (7.10): scan-to-scan matching LS objective.
- Eq. (7.11): correspondence gating / association variable.

NDT scan matching

- Eqs. (7.12)–(7.13): per-voxel mean and covariance.
- Eqs. (7.14)–(7.15): likelihood of scan points under voxel Gaussians.
- Eq. (7.16): MLE as nonlinear LS.

LOAM-style residuals

- Eq. (7.17): odometry objective using edge + plane features.
- Eqs. (7.18)–(7.19): point-to-line and point-to-plane distances.

Visual model + bundle adjustment

- Eqs. (7.43)–(7.46): camera distortion model.
- Eqs. (7.68)–(7.70): bundle adjustment (BA) reprojection cost over poses + landmarks.

7.6.3 Example 1 – LiDAR scan-to-scan matching (ICP-style) (Eqs. (7.10)–(7.11))

Files

- ch7_slam/lidar_icp_scan_matching.py
- notebooks/ch7_slam/lidar_icp_scan_matching.ipynb

Core functions used

- core.slam.scan_matching.build_correspondences (Eq. (7.11))
- core.slam.scan_matching.icp_point_to_point (Eq. (7.10))
- core.slam.se2 utilities for applying/compounding poses
- core.eval plots + metrics

Dataset

- data/sim/slam_lidar2d/

Notebook structure

1) Load two consecutive scans and ground truth relative pose.

2) Run ICP with:

- a good initial guess (ground truth + noise),
- a poor initial guess, and
- no initial guess (identity).

3) For each run, plot:

- correspondence count per iteration,
- residual RMS per iteration,
- final alignment overlay (ref points vs transformed current points).

4) Convert per-pair relative poses into an odometry trajectory and show drift.

Plots

- ch7_slam/figs/ch7_icp_alignment_overlay.svg
- ch7_slam/figs/ch7_icp_residual_vs_iter.svg
- ch7_slam/figs/ch7_icp_odometry_traj.svg
- ch7_slam/figs/ch7_icp_odometry_error_cdf.svg

Equation mapping

- icp_point_to_point docstring references Eq. (7.10).
- build_correspondences docstring references Eq. (7.11).
- equation_index.yml maps:
  - Eq. (7.10) → core/slam/scan_matching.py::icp_point_to_point
  - Eq. (7.11) → core/slam/scan_matching.py::build_correspondences

7.6.4 Example 2 – NDT scan matching (Eqs. (7.12)–(7.16))

Files

- ch7_slam/lidar_ndt_alignment.py
- notebooks/ch7_slam/lidar_ndt_alignment.ipynb

Core functions used

- core.slam.ndt.voxel_stats (Eqs. (7.12)–(7.13))
- core.slam.ndt.ndt_negative_log_likelihood (Eqs. (7.14)–(7.15))
- core.slam.ndt.ndt_align (Eq. (7.16), via core/estimators optimizer)

Experiment

- Build an NDT voxel map from a “reference scan” or a short local map.
- Align the current scan against the voxel map.
- Compare ICP vs NDT on:
  - random outliers,
  - reduced point density,
  - larger initial pose error.

Plots

- ch7_slam/figs/ch7_icp_vs_ndt_rmse_bar.svg
- ch7_slam/figs/ch7_ndt_convergence.svg

Equation mapping

- voxel_stats docstring references Eqs. (7.12)–(7.13).
- ndt_align docstring references Eq. (7.16).

7.6.5 Example 3 – LOAM-style feature odometry (Eqs. (7.17)–(7.19))

Files

- ch7_slam/lidar_loam_feature_odometry.py
- notebooks/ch7_slam/lidar_loam_feature_odometry.ipynb

Core functions used

- core.slam.loam.compute_curvature
- core.slam.loam.select_loam_features
- core.slam.loam.point_to_line_distance (Eq. (7.18))
- core.slam.loam.point_to_plane_distance (Eq. (7.19), optional)

What to demonstrate

- Select a small subset of “edge” and “plane” features.
- Solve a tiny nonlinear LS step for the relative pose using only these features (Eq. (7.17)).
- Compare runtime and accuracy vs point-to-point ICP.

Important simplification

- It is acceptable to implement a 2D-only LOAM demo:
  - “edge features” in 2D are points on high-curvature corners.
  - Use only point-to-line residuals (a 2D analogue of Eq. (7.18)).
  - Keep plane features as optional.

Plots

- ch7_slam/figs/ch7_loam_feature_selection.svg
- ch7_slam/figs/ch7_loam_vs_icp_runtime.svg

Equation mapping

- point_to_line_distance docstring references Eq. (7.18).
- point_to_plane_distance docstring references Eq. (7.19) (if implemented).
- loam_odometry_step docstring references Eq. (7.17).

7.6.6 Example 4 – Pose graph SLAM with loop closure (uses Ch3 MAP + Ch7 constraints)

Goal

Turn scan matching outputs into a global trajectory estimate:

- consecutive scan matches → between-pose factors
- loop closure constraints (synthetic) → long-range between-pose factors
- solve via FGO (Chapter 3 MAP) to reduce drift

Files

- ch7_slam/pose_graph_loop_closure.py
- notebooks/ch7_slam/pose_graph_loop_closure.ipynb

Core functions used

- core.slam.factors.BetweenPoseFactorSE2
- core.estimators.fgo.solve_fgo
- core.eval metrics + plotting

Must show

- Odometry-only trajectory vs optimized (loop-closed) trajectory.
- Error CDF and RMSE reduction.

7.6.7 Example 5 – Visual SLAM: camera model + bundle adjustment (Eqs. (7.43)–(7.46), (7.68)–(7.70))

This repo avoids full image processing. Instead, it uses synthetic “feature tracks” so the estimation math is clear.

Files

- ch7_slam/visual_bundle_adjustment.py
- notebooks/ch7_slam/visual_bundle_adjustment.ipynb

Core functions used

- core.slam.camera.distort_normalized (Eqs. (7.43)–(7.46))
- core.slam.factors.ReprojectionFactor (BA cost; Eqs. (7.68)–(7.70))
- core.estimators.fgo.solve_fgo

Dataset

- data/sim/slam_visual_bearing2d/

Notebook structure

1) Load true poses, landmarks, and noisy pixel observations.

2) Initialize:

- poses with drifted odometry,
- landmarks with noisy guesses.

3) Run BA and plot:

- reprojection RMS vs iteration
- before/after trajectory
- before/after landmark map

Equation mapping

- distort_normalized docstring references Eqs. (7.43)–(7.46).
- ReprojectionFactor references Eqs. (7.68)–(7.70).

7.6.8 IMU roles in LiDAR SLAM (Section 7.5) – minimal demo requirement

Chapter 7 emphasizes IMU as a high-rate motion source that can provide a strong initial guess for scan matching and help bridge gaps between scans.

Repo requirement (minimal)

- Add one small demo showing ICP/NDT convergence with:
  - constant-velocity initial guess, vs
  - IMU-integrated initial guess (using Chapter-6 strapdown helpers).

This is a “bridging” demo only; full IMU preintegration factors belong in Chapter 8 (tightly coupled fusion).

7.6.9 Tests (Chapter 7)

Add tests under tests/ch7/:

tests/ch7/test_icp_converges.py
- On a fixed synthetic pair of scans, ICP recovers the known transform within tolerance.
- References Eqs. (7.10)–(7.11).

tests/ch7/test_ndt_voxel_stats.py
- Validate voxel mean/covariance against a hand-checked mini point set.
- References Eqs. (7.12)–(7.13).

tests/ch7/test_point_to_line_plane_distance.py
- Unit tests for Eq. (7.18) and (optionally) Eq. (7.19).

tests/ch7/test_pose_graph_loop_closure_smoke.py
- Odometry + loop closure optimization reduces RMSE vs odometry-only.

tests/ch7/test_bundle_adjustment_smoke.py
- BA decreases reprojection RMS and reduces pose error on a small synthetic problem.

7.6.10 Dataset Documentation Requirements (Chapter 7)

Following Section 5.3 standards, Chapter 7 must include:

**data/sim/slam_lidar2d/README.md**

Must document:
- Map structure and occupancy grid format
- Scan generation process and coordinate frames
- Loop closure constraint format
- Noise characteristics and outlier injection

Parameter effects table:
| Parameter | Range | Effect on SLAM |
|-----------|-------|----------------|
| `correspondence_epsilon` | 0.1-1.0 m | ICP correspondence threshold (Eq. 7.11) |
| `outlier_rate` | 0-0.3 | Tests robustness of scan matching |
| `scan_noise` | 0.01-0.1 m | Point measurement noise |
| `loop_closure_rate` | varies | Global drift reduction |

**data/sim/slam_visual_bearing2d/README.md**

Must document:
- Camera intrinsics and distortion parameters (Eqs. 7.43-7.46)
- Landmark distribution and visibility
- Observation noise characteristics
- Bundle adjustment problem structure

Parameter effects:
| Parameter | Range | Effect on BA |
|-----------|-------|--------------|
| `pixel_noise_std` | 0.5-5.0 px | Measurement uncertainty |
| `n_landmarks` | 10-100 | Observability and conditioning |
| `outlier_rate` | 0-0.2 | Tests robust kernels |

**Generation script requirements**

```bash
# Generate SLAM datasets with varying difficulty
python scripts/generate_slam_lidar2d_dataset.py \
    --outlier-rate 0.1 \
    --scan-noise 0.05 \
    --output data/sim/slam_lidar_moderate

python scripts/generate_slam_lidar2d_dataset.py \
    --outlier-rate 0.3 \
    --scan-noise 0.1 \
    --output data/sim/slam_lidar_hard
```

Experimentation scenario:
```bash
# Study loop closure impact
python -m ch7_slam.pose_graph_loop_closure \
    --data data/sim/slam_lidar2d \
    --no-loop-closure  # odometry-only

python -m ch7_slam.pose_graph_loop_closure \
    --data data/sim/slam_lidar2d  # with loop closure
```

Expected learning: Loop closure dramatically reduces accumulated drift.

Connection to book:
- Correspondence gating → Eq. 7.11
- NDT voxel parameters → Eqs. 7.12-7.16
- Reprojection error → Eqs. 7.68-7.70



7.7 ch8_sensor_fusion/

Chapter 8 is cross-cutting: it focuses on the *practical aspects* that determine whether a fusion system works outside of idealized assumptions:

- coupling architecture (loosely vs tightly coupled),
- observability and failure modes,
- tuning (covariance selection, thresholds, robust losses),
- calibration (intrinsic/extrinsic),
- temporal calibration (synchronization and interpolation),
- computational efficiency.

Repo stance

- Implement small, inspectable building blocks + didactic demos.
- Prefer 2D where possible.
- Reuse Chapter 3 estimators and Chapter 4–7 measurement models.

7.7.1 Minimum deliverables

A) LC vs TC pair on the *same dataset*

- Loosely coupled: first compute per-sensor outputs (e.g., UWB position fixes), then fuse those outputs.
- Tightly coupled: fuse raw measurements directly (e.g., UWB ranges), so the estimator sees the real measurement model.

B) Observability demo

A minimal example that shows an unobservable mode (e.g., global translation with odometry-only) and how an absolute measurement makes it observable.

C) Tuning + gating demo

A minimal example showing:

- how overly optimistic R (measurement covariance) can destabilize a filter,
- innovation/NIS monitoring,
- chi-square gating,
- robust loss / down-weighting instead of hard rejection.

D) Calibration demo (toy)

Show that incorrect extrinsic parameters cause systematic residuals and worse accuracy, and demonstrate a toy extrinsic calibration (rigid transform or lever arm).

E) Temporal calibration demo (toy)

Show that a fixed time offset between two sensor streams degrades performance, and that correcting the offset (plus interpolation) recovers accuracy.

7.7.2 Core Chapter-8 equations to implement (minimum set)

These are the “must implement” equations from Chapter 8 because they directly translate into reusable code and tests.

Observability definitions

- Eq. (8.1)–(8.2): definition of unobservability via indistinguishable measurements.

Tuning and gating

- Eq. (8.5): innovation y_k.
- Eq. (8.6): innovation covariance S_k.
- Eq. (8.7): robust scaling of measurement covariance R_k <- w(y_k) R_k.
- Eq. (8.8): squared Mahalanobis distance d_k^2 = y_k^T S_k^{-1} y_k (same quantity as NIS).
- Eq. (8.9): chi-square gating threshold d_k^2 < chi2(m, alpha).

Equation mapping requirements

- Eq. (8.5) -> core/fusion/tuning.py::innovation
- Eq. (8.6) -> core/fusion/tuning.py::innovation_covariance
- Eq. (8.7) -> core/fusion/tuning.py::scale_measurement_covariance
- Eq. (8.8) -> core/eval/metrics.py::compute_nis
- Eq. (8.9) -> core/fusion/gating.py::chi_square_gate

7.7.3 Required plotting outputs (for every Chapter-8 demo)

All Chapter-8 notebooks/scripts must generate at least:

- trajectory plot: truth vs estimate(s)
- position error vs time
- error CDF

And for tuning-focused demos:

- NIS over time with chi-square bounds (core/eval.plot_nis)
- accepted vs rejected measurements (a simple boolean timeline or marker plot)

7.7.4 Example set (scripts + notebooks)

Folder structure

ch8_sensor_fusion/

- lc_uwb_imu_ekf.py
- tc_uwb_imu_ekf.py
- observability_translation_demo.py
- tuning_gating_robust_demo.py
- calib_extrinsic_demo.py
- temporal_sync_demo.py
- figs/

notebooks/ch8_sensor_fusion/

- lc_vs_tc_uwb_imu.ipynb
- observability.ipynb
- tuning_and_gating.ipynb
- calibration_and_sync.ipynb

Each script must be runnable as:

python -m ch8_sensor_fusion.lc_uwb_imu_ekf --data data/sim/fusion_2d_imu_uwb

7.7.5 Example A: Loosely coupled fusion (UWB position fixes + IMU)

Goal

Demonstrate a standard loosely coupled architecture: compute UWB position fixes first (from raw ranges), then fuse those fixes as measurements in an EKF.

Inputs

- IMU (high rate) for propagation (or PDR increments as a simplified control).
- UWB range measurements to anchors (low rate).

LC pipeline

1) UWB position fixes (per epoch)

- For each UWB timestamp, compute p_uwb_xy via LS/WLS (Chapter 4 positioning solver).
- Treat p_uwb_xy and its covariance R_uwb_fix as a measurement.

2) EKF fusion

- State (minimal): x = [p_x, p_y, v_x, v_y, yaw] or a simpler [p_x, p_y] if using PDR increments as control.
- Propagate using IMU/strapdown (Chapter 6) or using a simplified constant-velocity model.
- Update with the UWB position fix.

Outlier handling

- Apply chi-square gating on the innovation (Eqs. (8.8)–(8.9)).
- If rejected, skip the update.

Required plots

- LC EKF vs UWB-only fixes vs dead-reckoning only.
- NIS with chi-square bounds.

7.7.6 Example B: Tightly coupled fusion (raw UWB ranges + IMU)

Goal

Demonstrate a tightly coupled architecture: fuse the *raw* UWB ranges directly in the EKF (or optionally FGO), instead of first computing a position fix.

TC pipeline

- State and propagation are the same as Example A.
- Measurement model uses the range function h(x) = ||p - a_i|| (anchors known).
- EKF update is done per anchor measurement (or per vector of ranges).

Why this example matters

- It makes the coupling difference obvious: LC uses a derived position fix; TC uses the real sensor measurement model.
- It reuses Chapter 4 measurement models and Chapter 3 estimators, and adds Chapter 8 tuning/gating logic.

Required plots

- TC EKF vs LC EKF on the same dataset.
- Show behavior when one anchor has injected NLOS bias: gating/robust loss should reduce impact.

7.7.7 Example C: Observability demo (odometry-only vs odometry + absolute fixes)

Goal

Demonstrate the Chapter-8 observability definition by showing that an odometry-only system cannot determine its global translation (two different initial positions produce the same odometry measurements).

Demo definition

- Define an odometry measurement function that observes only *increments* (delta position), not absolute position.
- Create two trajectories that are identical up to a constant translation.
- Show that the generated measurements are identical (Eq. (8.1)–(8.2) condition).

Then add an absolute measurement source (e.g., occasional UWB position fixes or a synthetic 'Wi-Fi fix') and show:

- the translation becomes observable,
- the estimate converges to the true global position.

Required outputs

- plot two translated trajectories with identical odometry increments
- plot estimation results with and without absolute fixes

7.7.8 Example D: Tuning, thresholds, and robust losses

Goal

Make tuning concepts concrete by showing:

- a filter with under-estimated R becomes overconfident and may diverge,
- inflating R increases robustness,
- innovation-based monitoring and robust loss scaling can stabilize performance.

Required code features

- log innovation y_k (Eq. (8.5)) and innovation covariance S_k (Eq. (8.6)) for every update.
- compute NIS (= d^2) and compare to chi-square bounds.
- implement at least:
  - hard chi-square gating (Eq. (8.9)), and
  - soft robust down-weighting via R_k <- w(y_k) R_k (Eq. (8.7)) with Huber or Cauchy weight.

Required plots

- NIS timeline with gates marked
- trajectory and error comparisons across:
  - baseline (no gating)
  - chi-square gating
  - robust down-weighting

7.7.9 Example E: Calibration (toy extrinsic)

Goal

Show that wrong extrinsic parameters degrade fusion quality.

Minimum toy

- Rigid transform demo:
  - generate points in sensor A frame
  - transform with known (R,t) to sensor B frame
  - add noise
  - recover (R,t) with estimate_rigid_transform

Fusion-oriented toy (optional but recommended)

- lever arm demo:
  - UWB tag is offset from IMU origin by a fixed translation
  - simulate ranges from the tag
  - run fusion with wrong lever arm and observe systematic residuals

7.7.10 Example F: Temporal calibration and interpolation

Goal

Show that even a small time offset between two sensor streams can cause significant fusion error, and that:

- applying a corrected offset, and
- interpolating signals to the correct measurement time

recovers accuracy.

Minimum toy

- inject a fixed time offset between IMU and UWB range timestamps (config.json)
- run the TC fusion:
  - without correction
  - with correction via TimeSyncModel
- show trajectory + error difference

Required utilities

- core/fusion/time_alignment.py interpolation helpers
- TimeSyncModel applied consistently

7.7.11 Tests (Chapter 8)

Add tests under tests/ch8/:

tests/ch8/test_innovation_and_S.py
- Validate Eq. (8.5) and Eq. (8.6) on a small linear KF example.

tests/ch8/test_chi_square_gate.py
- Validate Eqs. (8.8)–(8.9): accept/reject decisions match known chi-square quantiles.

tests/ch8/test_robust_R_scaling.py
- Validate Eq. (8.7) scaling behavior.

tests/ch8/test_time_sync_model.py
- Validate TimeSyncModel offset/drift mapping.

tests/ch8/test_interpolation_helpers.py
- Validate interpolation on a known function.

tests/ch8/test_lc_tc_smoke.py
- Runs LC and TC scripts on a tiny dataset and asserts TC or LC improves over DR-only (smoke-level assertion, not a strict benchmark).

7.7.12 Ownership split (Chapter 8)

Navigation engineer

- Decide which fusion pair to emphasize (e.g., UWB+IMU baseline, plus optional LiDAR/IMU extension).
- Define the state vector for each demo and justify observability.
- Select noise levels, gating alpha, and robust-loss parameters.

Software engineer

- Implement the core/fusion utilities and keep interfaces stable.
- Ensure every demo logs innovations/NIS and generates the required plots.
- Add unit tests and keep runtime short.

7.7.13 Dataset Documentation Requirements (Chapter 8)

Following Section 5.3 standards, Chapter 8 fusion datasets must be exceptionally well-documented since they are the primary vehicle for teaching practical fusion concepts.

**data/sim/fusion_2d_imu_uwb/README.md** (Comprehensive Example)

This README must be the gold standard for dataset documentation. Required sections:

**Overview**

Purpose: Demonstrate loosely vs tightly coupled fusion on a realistic 2D scenario.

Learning objectives:
- Understand multi-rate sensor fusion (100 Hz IMU, 10 Hz UWB)
- Observe innovation monitoring and chi-square gating (Ch8, Eqs. 8.5-8.9)
- Compare LC and TC architectures
- Study parameter sensitivity (noise, dropout, NLOS)

**Scenario Description**

Trajectory: 2D rectangular walking path (20m × 15m)
Duration: 60 seconds
Motion: Constant speed (1.0 m/s), 70m perimeter, ~0.86 laps

Sensors:
- IMU: 100 Hz, 2D accelerometer + 1D gyroscope
- UWB: 10 Hz, ranges to 4 corner anchors
- Anchors placed at: (0,0), (20,0), (20,15), (0,15)

**Files and Data Structure**

| File | Shape | Description | Units |
|------|-------|-------------|-------|
| `truth.npz` | | Ground truth states | |
| ├─ `t` | (6000,) | Timestamps | seconds |
| ├─ `p_xy` | (6000, 2) | 2D positions | meters |
| ├─ `v_xy` | (6000, 2) | 2D velocities | m/s |
| └─ `yaw` | (6000,) | Heading angles | radians |
| `imu.npz` | | IMU measurements | |
| ├─ `t` | (6000,) | Timestamps | seconds |
| ├─ `accel_xy` | (6000, 2) | 2D accelerations | m/s² |
| └─ `gyro_z` | (6000,) | Yaw rates | rad/s |
| `uwb_anchors.npy` | (4, 2) | Anchor positions | meters |
| `uwb_ranges.npz` | | UWB measurements | |
| ├─ `t` | (600,) | Timestamps | seconds |
| └─ `ranges` | (600, 4) | Ranges (NaN = dropout) | meters |
| `config.json` | | Configuration params | see below |

**Loading Example**

```python
import numpy as np
import json

# Load ground truth
truth = np.load('data/sim/fusion_2d_imu_uwb/truth.npz')
t = truth['t']          # (6000,) timestamps
p_xy = truth['p_xy']    # (6000, 2) positions [x, y]
v_xy = truth['v_xy']    # (6000, 2) velocities [vx, vy]
yaw = truth['yaw']      # (6000,) heading angles

# Load IMU
imu = np.load('data/sim/fusion_2d_imu_uwb/imu.npz')
t_imu = imu['t']              # (6000,) at 100 Hz
accel_xy = imu['accel_xy']    # (6000, 2) in body frame
gyro_z = imu['gyro_z']        # (6000,) in body frame

# Load UWB
anchors = np.load('data/sim/fusion_2d_imu_uwb/uwb_anchors.npy')  # (4, 2)
uwb = np.load('data/sim/fusion_2d_imu_uwb/uwb_ranges.npz')
t_uwb = uwb['t']        # (600,) at 10 Hz
ranges = uwb['ranges']  # (600, 4), NaN indicates dropout

# Load configuration
with open('data/sim/fusion_2d_imu_uwb/config.json') as f:
    config = json.load(f)
    
print(f"IMU rate: {config['imu']['rate_hz']} Hz")
print(f"UWB rate: {config['uwb']['rate_hz']} Hz")
print(f"IMU accel noise: {config['imu']['accel_noise_std_m_s2']} m/s²")
print(f"UWB range noise: {config['uwb']['range_noise_std_m']} m")
```

**Configuration Parameters** (from `config.json`)

IMU parameters:
- `rate_hz`: 100.0 (sampling rate)
- `accel_noise_std_m_s2`: 0.1 (white noise σ)
- `gyro_noise_std_rad_s`: 0.01 (white noise σ)
- `accel_bias_m_s2`: [0.0, 0.0] (constant bias)
- `gyro_bias_rad_s`: 0.0 (constant bias)

UWB parameters:
- `rate_hz`: 10.0 (measurement rate)
- `n_anchors`: 4
- `range_noise_std_m`: 0.05 (white noise σ)
- `nlos_anchors`: [] (list of biased anchor indices)
- `nlos_bias_m`: 0.5 (positive bias magnitude)
- `dropout_rate`: 0.05 (probability of missing measurement)

Temporal calibration:
- `time_offset_sec`: 0.0 (UWB time offset relative to IMU)
- `clock_drift`: 0.0 (relative clock drift rate)

**Parameter Effects and Learning Experiments**

| Parameter | Default | Experiment Range | Effect on Fusion | Learning Objective |
|-----------|---------|------------------|------------------|-------------------|
| `accel_noise_std` | 0.1 | 0.01-0.5 | Higher → IMU velocity drift faster → fusion relies more on UWB | Understand process model uncertainty |
| `gyro_noise_std` | 0.01 | 0.001-0.05 | Higher → heading drift faster | Observe heading error propagation |
| `range_noise_std` | 0.05 | 0.01-0.5 | Higher → UWB fixes noisier → EKF trusts IMU more | Understand measurement uncertainty balance |
| `nlos_anchors` | [] | [1], [1,2] | Introduces systematic bias → tests gating/robust loss | Learn outlier rejection (Ch8, Eqs. 8.8-8.9) |
| `nlos_bias` | 0.5 | 0.2-2.0 | Larger bias → more rejections by chi-square gate | Quantify gating effectiveness |
| `dropout_rate` | 0.05 | 0.1-0.3 | More dropouts → longer IMU-only intervals | Understand missing data handling |
| `time_offset_sec` | 0.0 | -0.1 to 0.1 | Non-zero → systematic residuals → degraded fusion | Learn temporal calibration importance |

**Dataset Variants**

This baseline dataset is accompanied by two variants:

1. **fusion_2d_imu_uwb_nlos/** - NLOS corruption
   - Anchors 1 and 2 have positive bias (+0.8 m)
   - Use for robust loss and gating demonstrations
   - Expected: chi-square gating rejects ~30-50% of NLOS measurements

2. **fusion_2d_imu_uwb_timeoffset/** - Temporal misalignment
   - UWB 50ms behind IMU (`time_offset_sec = -0.05`)
   - 100 ppm clock drift (`clock_drift = 0.0001`)
   - Use for temporal calibration demonstrations
   - Expected: without correction, position RMSE increases by 50-100%

**Visualization Example**

```python
import matplotlib.pyplot as plt

# Plot trajectory
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p_xy[:, 0], p_xy[:, 1], 'k-', label='Ground Truth', linewidth=2)
ax.scatter(anchors[:, 0], anchors[:, 1], 
          marker='^', s=200, c='red', 
          edgecolors='black', linewidths=2,
          label='UWB Anchors', zorder=10)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('2D Walking Trajectory with UWB Anchors')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig('trajectory_overview.svg')
```

**Connection to Book Equations**

This dataset is designed to demonstrate:
- Ch6, Eqs. 6.2-6.10: IMU strapdown integration (propagation)
- Ch4, Eqs. 4.14-4.23: UWB range-based positioning
- Ch3, Eqs. 3.21-3.22: EKF prediction and update
- Ch8, Eq. 8.5: Innovation y = z - h(x)
- Ch8, Eq. 8.6: Innovation covariance S
- Ch8, Eqs. 8.8-8.9: Chi-square gating threshold

**Recommended Experiments**

**Experiment 1: LC vs TC Comparison**

```bash
# Run both architectures on baseline dataset
python -m ch8_sensor_fusion.lc_uwb_imu_ekf
python -m ch8_sensor_fusion.tc_uwb_imu_ekf
python -m ch8_sensor_fusion.compare_lc_tc
```

Expected observations:
- TC typically achieves 10-20% better RMSE
- TC handles dropouts more gracefully (continues with partial updates)
- TC has higher computational cost (4× more updates per epoch)

**Experiment 2: Gating Effectiveness**

```bash
# Test on NLOS-corrupted data with/without gating
python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_2d_imu_uwb_nlos --no-gating

python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_2d_imu_uwb_nlos  # gating enabled

# Visualize NIS and rejected measurements
python -m ch8_sensor_fusion.tuning_robust_demo \
    --data data/sim/fusion_2d_imu_uwb_nlos
```

Expected observations:
- Without gating: NLOS measurements corrupt position estimate
- With gating: ~30-50% of NLOS measurements rejected
- NIS plot shows rejected measurements exceeding χ²(α=0.05) threshold

**Experiment 3: Parameter Sensitivity**

Generate custom datasets to study specific effects:

```bash
# High IMU noise scenario
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --accel-noise 0.5 --gyro-noise 0.05 \
    --output data/sim/fusion_high_imu_noise

python -m ch8_sensor_fusion.tc_uwb_imu_ekf \
    --data data/sim/fusion_high_imu_noise

# Result: Fusion relies more heavily on UWB updates,
# longer IMU-only intervals cause larger prediction uncertainty
```

**Troubleshooting / Common Student Questions**

Q: Why are there NaN values in the UWB ranges?
A: These represent measurement dropouts (packet loss, occlusion). The fusion system must handle missing data gracefully.

Q: Why is the IMU rate 100x faster than UWB?
A: This reflects realistic sensor characteristics. IMU provides high-rate motion prior; UWB provides sparse absolute corrections.

Q: How do I know if my filter is working correctly?
A: Check:
1. Position RMSE decreases compared to DR-only
2. NIS values are statistically consistent (within χ² bounds 95% of time)
3. Covariance trace decreases after updates, increases during prediction

Q: What if I want 3D instead of 2D?
A: Modify generation script to include z-axis. Core fusion logic extends naturally to 3D state.

**scripts/generate_fusion_2d_imu_uwb_dataset.py Enhancement**

The generation script must have a comprehensive CLI as documented in Section 5.3.3.

Add preset configurations for common scenarios:

```python
PRESETS = {
    'baseline': {
        'accel_noise': 0.1, 'gyro_noise': 0.01,
        'range_noise': 0.05, 'nlos_anchors': [], 'dropout_rate': 0.05
    },
    'nlos_severe': {
        'accel_noise': 0.1, 'gyro_noise': 0.01,
        'range_noise': 0.05, 'nlos_anchors': [1, 2], 'nlos_bias': 1.5, 'dropout_rate': 0.05
    },
    'high_dropout': {
        'accel_noise': 0.1, 'gyro_noise': 0.01,
        'range_noise': 0.05, 'nlos_anchors': [], 'dropout_rate': 0.3
    },
    'degraded_imu': {
        'accel_noise': 0.5, 'gyro_noise': 0.05,
        'range_noise': 0.05, 'nlos_anchors': [], 'dropout_rate': 0.05
    }
}

# Usage:
parser.add_argument('--preset', choices=PRESETS.keys(),
                   help='Use preset configuration')
```

Example usage:
```bash
python scripts/generate_fusion_2d_imu_uwb_dataset.py \
    --preset nlos_severe \
    --output data/sim/fusion_nlos_severe
```

For each chapter section, the design doc should separate:

- Core functions reused from core/.
- Chapter-specific "unique tasks" and plots.
- **Dataset documentation requirements** (following Section 5.3 standards).


8. Non-Functional Requirements
8.1 Language & Tooling

Python 3.x.

NumPy, SciPy, matplotlib, optional JAX/PyTorch for ML examples (kept light).

Packaging via pyproject.toml / poetry or similar.

8.2 Quality

Unit tests for all core functions (tests/).

pytest-based CI.

Equation index checker included in CI.

8.3 Performance

All examples should run on a typical laptop in minutes, not hours.

Datasets and notebooks: no GPU dependency required.

8.4 Documentation

docs/:

Per-chapter docs (overview, how to run examples).

docs/equation_index.yml and usage instructions.

README.md:

Explain structure, target audience, and how to map book ↔ repo.

9. Implementation Roadmap

Use epics as a roadmap for you + contributing engineers.

9.1 Epic 0 – Repo & Infrastructure

Initialize repo structure.

Set up:

core/, chX_..., data/, notebooks/, docs/, tools/.

CI with pytest + equation checker.

9.2 Epic 1 – Coordinates (Ch.2)

Implement core/coords.

Add tests for:

LLH↔ECEF↔ENU round-trip.

ENU↔NED conversions.

Implement ch2_coords examples.

Add Ch.2 equations to equation_index.yml.

9.3 Epic 2 – Estimators (Ch.3)

Implement LS/WLS + robust LS in core/estimators.

Implement KF/EKF/UKF/PF skeletons.

Add simple FGO wrapper.

Unit tests (simple linear systems).

ch3_estimators example notebook.

Update equation_index.yml for Ch.3.

9.4 Epic 3 – RF Positioning (Ch.4)

Implement measurement models and DOP in core/rf.

Build RF simulation tools in core/sim.

ch4_rf_point_positioning examples:

TOA, TDOA (Chan/Fang), AOA demos, and RF challenges notebook.

Add Ch.4 equations to equation_index.yml.

**Dataset documentation deliverables (Section 5.3):**

- Create data/sim/rf_2d_floor/README.md with parameter effects table
- Add CLI to scripts/generate_rf_2d_floor_dataset.py (if not present)
- Document beacon geometry impact on DOP
- Provide experimentation scenarios for students

9.5 Epic 4 – Sensors & Dead Reckoning (Ch.6)

Implement core/sensors detailed in Section 4.4.

Add Chapter 6 dataset families under data/sim/ (Section 5.2).

Implement ch6_dead_reckoning notebooks (Section 7.5).

Add tests under tests/ch6/.

Update equation_index.yml for Chapter 6 and ensure equation checker passes.

**Dataset documentation deliverables (Section 5.3, 7.5.7):**

- Create READMEs for all ch6 datasets:
  - data/sim/ch6_strapdown_basic/README.md
  - data/sim/ch6_wheel_odom_square/README.md
  - data/sim/ch6_foot_zupt_walk/README.md
  - data/sim/ch6_pdr_corridor_walk/README.md
  - data/sim/ch6_env_sensors_heading_altitude/README.md
- Add CLI with IMU grade presets (tactical, consumer, MEMS)
- Document parameter-to-drift relationships
- Provide experimentation scenarios showing constraint effectiveness

9.6 Epic 5 – Fingerprinting (Ch.5)

Implement core/fingerprinting (Section 4.7) + ch5_fingerprinting notebooks (Section 7.4).

- Unit tests for deterministic and probabilistic methods.
- Add Ch.5 equations to equation_index.yml.

**Dataset documentation deliverables (Section 5.3, 7.4.7):**

- Create data/sim/wifi_fingerprint_grid/README.md
- Enhance scripts/generate_wifi_fingerprint_dataset.py with CLI
- Document path-loss model parameters and AP placement strategy
- Provide experimentation scenarios (grid spacing, AP density impact on k-NN)

9.7 Epic 6 – SLAM (Ch.7)

Implement core/slam (Section 4.8) + ch7_slam notebooks (Section 7.6).

Minimum deliverables

- ICP scan matching with correspondence gating (Eqs. (7.10)–(7.11)).
- NDT scan matching (Eqs. (7.12)–(7.16)).
- LOAM-style residual helpers (Eqs. (7.17)–(7.19)) as a didactic demo.
- Pose graph loop closure example using FGO.
- Synthetic visual BA example (Eqs. (7.43)–(7.46), (7.68)–(7.70)).
- Tests under tests/ch7/.
- Update equation_index.yml for all implemented Chapter-7 equations.

**Dataset documentation deliverables (Section 5.3, 7.6.10):**

- Create data/sim/slam_lidar2d/README.md with:
  - Scan format and coordinate frames
  - Loop closure constraint structure
  - Parameter effects on scan matching (correspondence threshold, noise, outliers)
- Create data/sim/slam_visual_bearing2d/README.md with:
  - Camera model documentation (intrinsics, distortion)
  - Landmark visibility and observation format
- Add CLI to SLAM dataset generation scripts
- Provide experimentation scenarios (loop closure impact, outlier robustness)

9.8 Epic 7 – Sensor Fusion (Ch.8)

Deliverables (minimum)

Core

- Add core/fusion/ (Section 4.9) and wire it into ch8 demos.
- Implement and unit-test:
  - innovation + innovation covariance helpers (Eqs. (8.5)–(8.6))
  - robust covariance scaling (Eq. (8.7))
  - NIS + chi-square gating (Eqs. (8.8)–(8.9))
  - time sync + interpolation utilities

Data

- Add data/sim/fusion_2d_imu_uwb/ (Section 5.2).

Examples

- LC vs TC pair on the same dataset:
  - LC: UWB position fixes + IMU EKF
  - TC: raw UWB ranges + IMU EKF
- Observability demo (odometry-only vs odometry + absolute fixes).
- Tuning/gating/robust demo (NIS plots + outlier injection).
- Calibration + temporal sync toy demos.

Quality gates

- Tests under tests/ch8/.
- Update docs/equation_index.yml for all implemented Chapter-8 equations and ensure equation checker passes.
- Ensure all scripts run in minutes on a laptop.

**Dataset documentation deliverables (Section 5.3, 7.7.13) - HIGHEST PRIORITY:**

Chapter 8 datasets must be the gold standard for documentation since they demonstrate the most practical concepts for students.

- Create **comprehensive** data/sim/fusion_2d_imu_uwb/README.md:
  - Complete with all sections from Section 7.7.13 (overview, files, loading examples, parameter effects, variants, experiments)
  - This README serves as a template for all other dataset documentation
- Create data/sim/fusion_2d_imu_uwb_nlos/README.md (focus on NLOS handling)
- Create data/sim/fusion_2d_imu_uwb_timeoffset/README.md (focus on temporal calibration)
- Enhance scripts/generate_fusion_2d_imu_uwb_dataset.py with:
  - Full CLI interface with all parameters
  - Preset configurations (baseline, nlos_severe, high_dropout, degraded_imu)
  - Parameter validation and user-friendly error messages
- Update scripts/README.md with comprehensive Ch8 examples:
  - All three experimentation scenarios from Section 5.3.3
  - Expected learning outcomes for each scenario
  - Troubleshooting section for common student issues
- Create docs/data_simulation_guide.md:
  - Theory-to-simulation mapping for IMU error models
  - Theory-to-simulation mapping for NLOS effects
  - Step-by-step experiment guides (IMU drift, filter tuning, gating effectiveness)
  - Parameter sensitivity reference tables
- Create visualization tools:
  - tools/plot_dataset_overview.py (works with fusion datasets)
  - tools/compare_dataset_variants.py (compares baseline vs NLOS vs timeoffset)


10. Working Effectively: SW vs Navigation Engineers

Throughout the design doc, separate:

Algorithm / modeling decisions (navigation engineer):

Which equation(s) in the book to implement.

What approximations and parameters to use.

Implementation details (software engineer):

File layout, function signatures, performance, tests, CI.

The equation mapping system bridges the two:

Navigation engineers can check correctness by tracing Eq. → code.

Software engineers can preserve traceability by following docstring + index conventions and the equation checker.

**Dataset documentation bridges learning and practice:**

Navigation engineers:
- Define parameter ranges that demonstrate key theoretical concepts
- Write theory-to-simulation mappings connecting book equations to data parameters
- Design experimentation scenarios with clear learning outcomes
- Validate that parameter effects match theoretical predictions

Software engineers:
- Implement CLI interfaces enabling easy parameter exploration
- Create loading utilities and visualization tools
- Maintain documentation consistency across all datasets
- Ensure all documentation examples are tested and working

Students benefit from:
- READMEs that explain "what" and "why" for each dataset
- Parameter tables showing cause-and-effect relationships
- Ready-to-run experimentation scenarios
- Troubleshooting guides for common issues

**Quality gate for dataset documentation:**

Every dataset must have:
1. README.md with all required sections (Section 5.3)
2. Generation script with CLI interface
3. At least 2 experimentation scenarios in scripts/README.md
4. Parameter effects table with learning objectives
5. Working code examples for loading and visualization

This documentation is **not optional** – it is how students learn to manipulate simulation parameters and observe algorithm behavior, which is a core learning objective of this repository.