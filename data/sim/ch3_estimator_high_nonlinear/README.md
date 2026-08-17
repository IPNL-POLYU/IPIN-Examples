# ch3_estimator_high_nonlinear

## Overview

A figure-8 trajectory observed by four range-and-bearing beacons, generated to
stress the linearisation that the EKF depends on. It is the harder sibling of
[`ch3_estimator_nonlinear`](../ch3_estimator_nonlinear/README.md), which uses the
same beacons, the same noise and the same duration on a circular path.

Use the pair together. The circular dataset shows the estimators agreeing; this
one shows them disagreeing, and the reason they disagree is the whole content of
Section 3.2.

| | `ch3_estimator_nonlinear` | `ch3_estimator_high_nonlinear` |
|---|---|---|
| Trajectory | circle | figure-8 |
| Speed | constant 3.000 m/s | 2.646 – 5.657 m/s |
| Path length | 89.70 m | 114.51 m |
| Extent | x ±10, y ±10 | x ±10, y ±5 |
| Self-intersecting | no | yes, at the origin |

## Scenario Description

A target follows a figure-8 for 30 s at 10 Hz. Four beacons sit at the corners
of a 30 m square, and each returns a range and a bearing to the target at every
epoch.

Two things make this harder than the circle, and neither is the noise — the
noise is identical:

**Speed is not constant.** A figure-8 traversed in equal time steps runs fast
through the straight diagonals and slows at the two turns, from 2.646 to
5.657 m/s. A constant-velocity process model is therefore wrong twice per lap,
in opposite directions. On the circular dataset the same model is wrong only
about the *heading*, and its speed error is zero.

**It crosses itself.** At the origin the target passes through the same point
travelling in a different direction. Range measurements alone cannot tell those
two passes apart, so the estimator has to hold the distinction in its velocity
state. That is a genuinely different demand from tracking a circle, where
position determines phase.

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `time.txt` | (300,) | Timestamps, 0.0 – 29.9 s at dt = 0.1 s |
| `ground_truth_states.txt` | (300, 4) | `[x, y, vx, vy]` in metres and m/s |
| `beacons.txt` | (4, 2) | Beacon positions `[x, y]` in metres |
| `range_measurements.txt` | (300, 4) | Noisy range to each beacon, metres |
| `bearing_measurements.txt` | (300, 4) | Noisy bearing to each beacon, radians |
| `config.json` | — | Generation parameters and seed |

The state layout is `[x, y, vx, vy]`, matching the constant-velocity model in
`core.models.motion_models.ConstantVelocity2D`.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

hn_dir = Path("data/sim/ch3_estimator_high_nonlinear")

hn_t = np.loadtxt(hn_dir / "time.txt")
hn_truth = np.loadtxt(hn_dir / "ground_truth_states.txt")
hn_beacons = np.loadtxt(hn_dir / "beacons.txt")
hn_ranges = np.loadtxt(hn_dir / "range_measurements.txt")
hn_bearings = np.loadtxt(hn_dir / "bearing_measurements.txt")
hn_config = json.load(open(hn_dir / "config.json"))

print(f"epochs:    {len(hn_t)} at dt = {hn_t[1] - hn_t[0]:.1f} s")
print(f"state:     {hn_truth.shape[1]} elements [x, y, vx, vy]")
print(f"beacons:   {len(hn_beacons)}")
print(f"ranges:    {hn_ranges.shape}, {hn_ranges.min():.2f} - {hn_ranges.max():.2f} m")
print(f"bearings:  {hn_bearings.shape}, "
      f"{hn_bearings.min():+.2f} - {hn_bearings.max():+.2f} rad")
```

Expected output:

```
epochs:    300 at dt = 0.1 s
state:     4 elements [x, y, vx, vy]
beacons:   4
ranges:    (300, 4), 11.27 - 31.53 m
bearings:  (300, 4), -2.87 - +2.97 rad
```

## Configuration Parameters

```python
print(f"preset:          {hn_config['preset']}")
print(f"trajectory:      {hn_config['trajectory']['type']}")
print(f"duration:        {hn_config['trajectory']['duration_s']} s")
print(f"range noise:     {hn_config['measurements']['range_noise_std_m']} m")
print(f"bearing noise:   {hn_config['measurements']['bearing_noise_std_deg']} deg")
print(f"outlier rate:    {hn_config['measurements']['outlier_rate']}")
print(f"seed:            {hn_config['seed']}")
```

| Parameter | Value | Effect |
|---|---|---|
| `trajectory.type` | `figure8` | The whole difference from the baseline dataset |
| `measurements.range_noise_std_m` | 0.5 | Identical to the baseline, so any accuracy difference is nonlinearity, not noise |
| `measurements.bearing_noise_std_deg` | 5.0 | Identical to the baseline |
| `measurements.outlier_rate` | 0.0 | No outliers: this dataset isolates linearisation error from robustness |
| `seed` | 42 | Fixed, so the comparison is reproducible |

The noise being identical to the baseline is the design of the pair. If the
estimators separate here and not there, the cause is in the model, not the data.

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| Process noise `Q` | ×0.1, ×1, ×10 | The EKF needs a larger `Q` here than on the circle to stay consistent, because its model error is larger. Watch NEES, not just RMSE |
| Estimator | KF, EKF, UKF, PF | The gap between EKF and UKF is the point of this dataset |
| `range_noise_std_m` | 0.1, 0.5, 2.0 | At low noise the linearisation error dominates; at high noise it is buried |
| Beacon geometry | move beacons inward | Shorter ranges make the range Jacobian vary faster along the path |

**Measure the speed profile first** — it is the mechanism behind everything
else:

```python
hn_speed = np.linalg.norm(hn_truth[:, 2:4], axis=1)
print(f"speed: {hn_speed.min():.3f} - {hn_speed.max():.3f} m/s")

hn_step = np.linalg.norm(np.diff(hn_truth[:, :2], axis=0), axis=1)
print(f"path length: {hn_step.sum():.2f} m")

# Acceleration is what a constant-velocity model assumes away.
hn_accel = np.linalg.norm(np.diff(hn_truth[:, 2:4], axis=0), axis=1) / 0.1
print(f"implied acceleration: up to {hn_accel.max():.2f} m/s^2")
```

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(hn_truth[:, 0], hn_truth[:, 1], "b-", linewidth=1.5, label="truth")
ax1.plot(hn_beacons[:, 0], hn_beacons[:, 1], "r^", markersize=11, label="beacons")
ax1.plot(hn_truth[0, 0], hn_truth[0, 1], "go", markersize=8, label="start")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("Figure-8 trajectory and beacon geometry")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axis("equal")

ax2.plot(hn_t, np.linalg.norm(hn_truth[:, 2:4], axis=1), "b-")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Speed [m/s]")
ax2.set_title("Speed is not constant -- a CV model is wrong twice per lap")
ax2.grid(alpha=0.3)

fig.tight_layout()
print("figure built")
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eqs. (3.11)–(3.19) | Linear KF. Fed the *range* measurements directly it is mis-specified, which is the baseline to beat |
| Eq. (3.21) | EKF. The first-order expansion of the range/bearing model is where this dataset bites |
| Eqs. (3.24)–(3.30) | UKF. Sigma points do not linearise, so the figure-8's curvature costs it less |
| Eqs. (3.32)–(3.34) | Particle filter. No linearisation at all, at a cost in particles |

The range model is `h(x) = ||[x, y] - b_i||` and the bearing model is
`atan2(y - b_iy, x - b_ix)`; both are implemented in
`core.models.measurement_models.RangeBearingMeasurement2D`.

## Recommended Experiments

1. **Run the chapter example on both datasets and compare.**

   ```bash
   python ch3_estimators/example_ekf_range_bearing.py --data ch3_estimator_nonlinear
   python ch3_estimators/example_ekf_range_bearing.py --data ch3_estimator_high_nonlinear
   ```

2. **Check consistency, not just error.** A filter can have a respectable RMSE
   and still be lying about its covariance. `core.eval.compute_nees` and the
   chi-square bounds in `core.fusion.gating` are the tools; a NEES well above
   the state dimension means the EKF is overconfident, which is the classic
   symptom of unmodelled nonlinearity.

3. **Predict before measuring.** Compute the acceleration the trajectory
   implies, multiply by `dt`, and compare that with the velocity uncertainty the
   filter carries. If the model error is larger than the noise the filter
   expects, no amount of tuning `Q` will make it consistent — the model is
   wrong, and that is the lesson.

4. **Find the crossing.** Locate the epochs where the path passes through the
   origin, and look at each estimator's velocity error there specifically. Range
   measurements are identical at those two epochs; only the velocity state
   distinguishes them.

## Generation

```bash
python scripts/generate_ch3_estimator_comparison_dataset.py --preset high_nonlinearity
```

The `nonlinear` preset produces the circular sibling. Both use seed 42.

## References

- Chapter 3, *Principles of Indoor Positioning and Indoor Navigation*
- [`ch3_estimator_nonlinear`](../ch3_estimator_nonlinear/README.md) — the circular baseline
- `docs/guides/ch3_estimator_selection.md` — which estimator to reach for
