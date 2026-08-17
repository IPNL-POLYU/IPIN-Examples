# ch7_slam_2d_high_drift

## Overview

The same 20 m square trajectory, landmarks and scans as
[`ch7_slam_2d_square`](../ch7_slam_2d_square/README.md), with the odometry noise
tripled. It is the dataset for asking what a pose graph is actually worth when
the odometry it is correcting is bad.

Measured, on the shipped data:

| | `ch7_slam_2d_square` | **`ch7_slam_2d_high_drift`** |
|---|---|---|
| Odometry translation noise | 0.1 m | **0.3 m** |
| Odometry rotation noise | 0.02 rad | **0.05 rad** |
| Final odometry drift | 0.546 m | **1.124 m** |
| Odometry position RMSE | 0.3281 m | **0.7968 m** |
| Max odometry yaw error | 4.00° | **10.01°** |
| Loop closures in the data | 1 | 1 |
| Pipeline improvement | +33.4% | **+13.1%** |

**Read the last row before reaching for this dataset.** SLAM helps *less* here,
not more. That is the opposite of what the name suggests and of what the
catalogue used to claim, and the reason is worth understanding: a single loop
closure fixes one constraint. It can pin the end of the trajectory back to the
start, but it cannot recover the shape of what happened in between, and there is
three times as much shape error to recover.

More drift needs *more constraints*, not a bigger correction from the same one.

## Scenario Description

A robot drives a 20 m square, 10 poses per side, 41 poses in total, returning to
its start. Fifty landmarks are scattered around it, and at each pose a 2-D LiDAR
returns between 19 and 31 points, depending on what is in range.

Odometry is integrated from noisy relative motion, so its error accumulates:
1.124 m of position error and 0.72° of yaw error by the final pose, peaking at
1.549 m and 10.01° mid-run. The single loop closure links pose 0 to pose 40.

Everything except the odometry noise is identical to the baseline dataset —
same trajectory, same landmarks, same scan noise, same seed. So a comparison
between the two isolates the effect of odometry quality on a pose graph.

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `ground_truth_poses.txt` | (41, 3) | True poses `[x, y, yaw]`, metres and radians |
| `odometry_poses.txt` | (41, 3) | Dead-reckoned poses, same layout |
| `landmarks.txt` | (50, 2) | Landmark positions `[x, y]` in metres |
| `loop_closures.txt` | (1, 2) | Index pairs — here `[[0, 40]]` |
| `scans.npz` | 41 arrays | LiDAR scans in the robot frame, keys `scan_0` … `scan_40` |
| `config.json` | — | Generation parameters and seed |

**`scans.npz` holds one array per pose, not one array called `scans`.** Index it
as `scans_data[f"scan_{i}"]`; `scans_data["scans"]` raises `KeyError`.

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

hd_dir = Path("data/sim/ch7_slam_2d_high_drift")

hd_truth = np.loadtxt(hd_dir / "ground_truth_poses.txt")
hd_odom = np.loadtxt(hd_dir / "odometry_poses.txt")
hd_landmarks = np.loadtxt(hd_dir / "landmarks.txt")
hd_closures = np.loadtxt(hd_dir / "loop_closures.txt", dtype=int, ndmin=2)
hd_scan_data = np.load(hd_dir / "scans.npz")
hd_scans = [hd_scan_data[f"scan_{i}"] for i in range(len(hd_truth))]
hd_config = json.load(open(hd_dir / "config.json"))

print(f"poses:          {len(hd_truth)}")
print(f"landmarks:      {len(hd_landmarks)}")
print(f"loop closures:  {hd_closures.tolist()}")
print(f"scans:          {len(hd_scans)}, "
      f"{min(len(s) for s in hd_scans)}-{max(len(s) for s in hd_scans)} points each")
print(f"trajectory:     {hd_config['trajectory']['type']}, "
      f"{hd_config['trajectory']['size_m']} m")
```

Expected output:

```
poses:          41
landmarks:      50
loop closures:  [[0, 40]]
scans:          41, 19-31 points each
trajectory:     square, 20.0 m
```

## Configuration Parameters

```python
print(f"preset:            {hd_config['preset']}")
print(f"translation noise: "
      f"{hd_config['odometry']['translation_noise_std_m']} m")
print(f"rotation noise:    "
      f"{hd_config['odometry']['rotation_noise_std_rad']} rad")
print(f"recorded drift:    {hd_config['odometry']['final_drift_m']:.4f} m")
print(f"scan noise:        {hd_config['sensor']['scan_noise_std_m']} m")
print(f"max range:         {hd_config['sensor']['max_range_m']} m")
print(f"seed:              {hd_config['seed']}")
```

| Parameter | Value | Effect |
|---|---|---|
| `odometry.translation_noise_std_m` | 0.3 | 3× the baseline. Drives the 1.12 m final drift |
| `odometry.rotation_noise_std_rad` | 0.05 | 2.5× the baseline. Yaw error is what bends the square |
| `sensor.scan_noise_std_m` | 0.05 | **Unchanged.** Scan matching is no harder here; only the prior is worse |
| `sensor.max_range_m` | 15.0 | Unchanged, so the same landmarks are visible |
| `loop_closures.count` | 1 | Unchanged — and this is why the improvement is smaller |
| `seed` | 42 | Same as the baseline, so the comparison is controlled |

That the scan noise is unchanged is the design of the pair. The front end has
exactly the same measurements to work with; what differs is the quality of the
prediction it starts from.

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| `translation_noise_std_m` | 0.1, 0.3, 0.6 | Odometry RMSE should scale roughly with the square root of the number of steps × noise |
| `loop_closures.count` | 1, 3, 5 | **The one that matters.** More constraints, not less drift, is what makes the backend pay |
| `rotation_noise_std_rad` | 0.02, 0.05, 0.1 | Yaw error compounds into position error; it is usually the dominant term |
| `min_index_diff` | 5, 15, 25 | How far apart two poses must be to count as a closure. Short-baseline closures constrain almost nothing |

**Verify the drift is what the config claims** before drawing conclusions from
it:

```python
hd_err = np.linalg.norm(hd_odom[:, :2] - hd_truth[:, :2], axis=1)
hd_yaw_err = np.abs(np.arctan2(
    np.sin(hd_odom[:, 2] - hd_truth[:, 2]),
    np.cos(hd_odom[:, 2] - hd_truth[:, 2]),
))

print(f"final position error: {hd_err[-1]:.3f} m "
      f"(config says {hd_config['odometry']['final_drift_m']:.3f})")
print(f"max position error:   {hd_err.max():.3f} m")
print(f"odometry RMSE:        {np.sqrt((hd_err ** 2).mean()):.4f} m")
print(f"max yaw error:        {np.degrees(hd_yaw_err.max()):.2f} deg")
```

Note that the *maximum* error (1.549 m) is larger than the *final* error
(1.124 m). The trajectory returns to its start, so the final error understates
the drift — a closed loop flatters dead reckoning. Read the RMSE and the maximum,
not the endpoint. This is the same trap the Chapter 6 comparison example warns
about with its `Path` column.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(hd_truth[:, 0], hd_truth[:, 1], "g-", linewidth=2, label="truth")
ax1.plot(hd_odom[:, 0], hd_odom[:, 1], "r--", linewidth=1.5, label="odometry")
ax1.plot(hd_landmarks[:, 0], hd_landmarks[:, 1], "k.", markersize=4,
         label="landmarks")
for i, j in hd_closures:
    ax1.plot([hd_odom[i, 0], hd_odom[j, 0]], [hd_odom[i, 1], hd_odom[j, 1]],
             "m-", linewidth=2, label="loop closure")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.set_title("One closure against 1.12 m of drift")
ax1.legend(fontsize=8)
ax1.axis("equal")

ax2.plot(hd_err, "r-", label="position error")
ax2.axhline(hd_err.max(), color="gray", linestyle=":",
            label=f"max {hd_err.max():.2f} m")
ax2.axhline(hd_err[-1], color="k", linestyle="--",
            label=f"final {hd_err[-1]:.2f} m")
ax2.set_xlabel("Pose index")
ax2.set_ylabel("Error [m]")
ax2.set_title("The endpoint flatters it -- the loop closes")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
print("figure built")
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eqs. (7.10)–(7.11) | ICP scan matching, from a worse initial guess than the baseline gives |
| Section 7.3 | Pose graph optimisation, and its dependence on constraint count |
| Eqs. (3.35)–(3.38) | The MAP formulation the pose graph solves |
| Eqs. (3.42)–(3.43) | Gauss-Newton / LM, via `core.estimators.factor_graph` |

`FactorGraph.optimize` is the solver; `core.slam.factors.create_pose_graph`
builds the graph from `(from, to, relative_pose)` triples.

## Recommended Experiments

1. **Run both datasets and compare improvements, not errors.**

   ```bash
   python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square
   python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_high_drift
   ```

   Baseline: 0.3281 → 0.2184 m, +33.4%. High drift: 0.7968 → 0.6922 m, +13.1%.
   The absolute correction is similar in metres; as a *fraction* it is much
   smaller, because the error it cannot reach has grown.

2. **Add constraints and watch the improvement recover.** Regenerate with
   `--loop-closures 5` at this noise level. If the hypothesis above is right, the
   improvement should climb back toward the baseline's — and if it does not, the
   limit is elsewhere and worth finding.

3. **Check the front end separately.** Scan noise is unchanged, so scan-to-map
   ICP has the same measurements and a worse prior. Report the front-end RMSE on
   its own before the backend runs: a worse prior can push ICP outside its basin
   of convergence, which is a different failure from "the backend needs more
   constraints".

4. **Do not read the endpoint.** Verify for yourself that final error understates
   drift here, then check which number the chapter example reports and which one
   you were about to quote.

## Dataset Variants

- [`ch7_slam_2d_square`](../ch7_slam_2d_square/README.md) — the same scenario at 0.1 m / 0.02 rad odometry noise

## Generation

```bash
python scripts/generate_ch7_slam_2d_dataset.py --preset high_drift
```

`--preset baseline` produces the sibling. Both use seed 42, so the trajectory,
landmarks and scans are identical and only the odometry differs.

## References

- Chapter 7, *Principles of Indoor Positioning and Indoor Navigation*
- [`ch7_slam_2d_square`](../ch7_slam_2d_square/README.md) — the baseline
- `ch7_slam/README.md` — the pipeline and its measured stage-by-stage results
