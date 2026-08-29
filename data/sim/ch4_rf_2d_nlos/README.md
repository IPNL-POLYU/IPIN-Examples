# ch4_rf_2d_nlos

## Overview

[`ch4_rf_2d_square`](../ch4_rf_2d_square/README.md) with a non-line-of-sight bias
added to two of the four beacons. The geometry, the query points, the noise and
the seed are all identical — **the only difference is +0.8 m of bias on beacons
1 and 2.**

That makes it the dataset for one specific lesson: geometry is not the problem
here, and DOP cannot see the problem at all. GDOP is byte-identical to the
baseline:

| | `ch4_rf_2d_square` | `ch4_rf_2d_nlos` |
|---|---|---|
| Mean GDOP (TOA) | 1.022 | 1.022 |
| Mean GDOP (TDOA) | 1.067 | 1.067 |
| Beacon 0 range residual | +0.0026 ± 0.0851 m | +0.0026 ± 0.0851 m |
| Beacon 1 range residual | +0.0089 ± 0.1060 m | **+0.8089 ± 0.1060 m** |
| Beacon 2 range residual | +0.0062 ± 0.0965 m | **+0.8062 ± 0.0965 m** |
| Beacon 3 range residual | +0.0019 ± 0.1066 m | +0.0019 ± 0.1066 m |

Look at the biased rows against their clean counterparts: the mean is up by
exactly 0.8000 and the standard deviation is **identical to four decimal
places**. Same seed, same noise draw, one constant added. That is what makes this
pair usable as a controlled experiment.

DOP is a function of geometry alone. It answers "how much does measurement noise
get amplified", and a bias is not noise. A solver that trusts DOP will report a
confident, wrong position — which is why Section 8.3's gating exists.

## Scenario Description

Four beacons at the corners of a 20 × 20 m floor:

```
(0,20) ---------- (20,20)
   |      biased      |     beacons 1 and 2: +0.8 m
   |                  |
   |     clean        |     beacons 0 and 3: unbiased
(0,0) ----------- (20,0)
```

100 query positions on a grid from (2, 2) to (18, 18). Each returns a TOA range,
a TDOA range difference against beacon 0, and an AOA azimuth.

The bias is **positive and constant**: an NLOS path is longer than the direct
one, never shorter, so a reflected or diffracted signal always reports too far.
That one-sidedness is what makes NLOS harder than noise. Zero-mean noise averages
out over redundant measurements; a positive bias does not, and least squares will
happily move the estimate to reduce the residual it creates.

Note also that beacon 0 is the TDOA reference and is *unbiased*, so the TDOA
differences inherit the bias of beacons 1 and 2 directly rather than having it
partly cancel. Choosing a biased beacon as the reference would spread the error
across every difference instead — worth trying.

**That paragraph described the intended dataset for longer than it described
the shipped one.** The generator applied the NLOS bias inside its TOA loop
only, and built the range differences from *true* ranges, so until recently
`tdoa_diffs.txt` here was bias-free: its residual column means were
`[-0.012, +0.001, +0.002]` where the sentence above predicts `[+0.8, +0.8, 0]`.
The differences are now formed from the same biased, noisy ranges the TOA file
carries — `tdoa_diffs.txt` is exactly `toa_ranges.txt` differenced against its
first column — and they measure `[+0.806, +0.804, -0.001]`. TDOA's median error
moved from 0.075 m to 0.604 m as a result, which is the right order: it is now
degraded by NLOS about as much as TOA is (0.614 m), instead of being immune to
a bias the dataset exists to demonstrate. A document that predicts your data is
worth checking against it.

## Files and Data Structure

| File | Shape | Contents |
|---|---|---|
| `beacons.txt` | (4, 2) | Beacon positions `[x, y]` in metres |
| `ground_truth_positions.txt` | (100, 2) | Query positions `[x, y]` in metres |
| `toa_ranges.txt` | (100, 4) | Noisy **and biased** range to each beacon |
| `tdoa_diffs.txt` | (100, 3) | Range differences against beacon 0 |
| `aoa_angles.txt` | (100, 4) | Noisy azimuth to each beacon, radians |
| `gdop_toa.txt` | (100,) | GDOP for TOA — identical to the baseline |
| `gdop_tdoa.txt` | (100,) | GDOP for TDOA |
| `gdop_aoa.txt` | (100,) | GDOP for AOA |
| `config.json` | — | Generation parameters, including which beacons are biased |

## Loading Example

```python
import json
from pathlib import Path

import numpy as np

nlos_dir = Path("data/sim/ch4_rf_2d_nlos")

nlos_beacons = np.loadtxt(nlos_dir / "beacons.txt")
nlos_truth = np.loadtxt(nlos_dir / "ground_truth_positions.txt")
nlos_toa = np.loadtxt(nlos_dir / "toa_ranges.txt")
nlos_tdoa = np.loadtxt(nlos_dir / "tdoa_diffs.txt")
nlos_gdop = np.loadtxt(nlos_dir / "gdop_toa.txt")
nlos_config = json.load(open(nlos_dir / "config.json"))

print(f"beacons:        {len(nlos_beacons)}")
print(f"query points:   {len(nlos_truth)}")
print(f"biased beacons: {nlos_config['nlos']['beacon_indices']}")
print(f"bias:           {nlos_config['nlos']['bias_m']} m")
print(f"mean GDOP:      {nlos_gdop.mean():.3f}")
```

Expected output:

```
beacons:        4
query points:   100
biased beacons: [1, 2]
bias:           0.8 m
mean GDOP:      1.022
```

## Configuration Parameters

```python
print(f"preset:      {nlos_config['preset']}")
print(f"TOA noise:   {nlos_config['measurements']['toa_noise_std_m']} m")
print(f"NLOS bias:   {nlos_config['nlos']['bias_m']} m "
      f"on beacons {nlos_config['nlos']['beacon_indices']}")
```

| Parameter | Value | Effect |
|---|---|---|
| `nlos.bias_m` | 0.8 | 8x the measurement noise, so the bias dominates. Positive only |
| `nlos.beacon_indices` | `[1, 2]` | Two of four biased. With three or more, robust methods lose their majority and fail |
| `measurements.toa_noise_std_m` | 0.1 | Unchanged from the baseline, which is what isolates the bias |
| `preset` | `nlos` | Same geometry and seed as `baseline`; only the bias differs |

**Two of four is the interesting choice.** A robust estimator works by deciding
which measurements to distrust, and it can only do that if the trustworthy ones
outnumber the rest. With two clean and two biased there is no majority, and
whether a robust loss recovers the right answer depends on the geometry — which
is a far more instructive situation than one clear outlier.

## Parameter Effects and Learning Experiments

| Parameter | Try | What to watch |
|---|---|---|
| `nlos.bias_m` | 0.2, 0.8, 3.0 | Position error grows with bias, but *not* by GDOP x bias — the relationship is geometry-dependent |
| `nlos.beacon_indices` | `[1]`, `[1,2]`, `[1,2,3]` | One outlier is detectable, two is marginal, three defeats a majority vote |
| Estimator | LS, Huber, Cauchy, Geman-McClure | The Table 3.1 losses, on a real bias rather than a synthetic outlier |
| TDOA reference | beacon 0 vs beacon 1 | An unbiased reference passes the bias through; a biased one spreads it |

**Measure the bias rather than reading it from the config** — this is the
check that shows why residuals are the diagnostic:

```python
# True ranges from the known geometry, so the residual is bias + noise.
nlos_true_r = np.linalg.norm(
    nlos_truth[:, None, :] - nlos_beacons[None, :, :], axis=2
)
nlos_resid = nlos_toa - nlos_true_r

for j in range(nlos_resid.shape[1]):
    tag = "BIASED" if j in nlos_config["nlos"]["beacon_indices"] else "clean "
    print(f"beacon {j} [{tag}]: mean {nlos_resid[:, j].mean():+.4f} m, "
          f"std {nlos_resid[:, j].std():.4f} m")
```

The clean beacons show a mean near zero and a std near the 0.1 m noise; the
biased ones show a mean near +0.8 with the *same* std. **A bias moves the mean
and leaves the spread alone**, which is exactly what a chi-square gate on
innovations is looking for.

## Visualization Example

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

nlos_biased = nlos_config["nlos"]["beacon_indices"]
for j in range(nlos_resid.shape[1]):
    ax1.hist(nlos_resid[:, j], bins=18, alpha=0.6,
             label=f"beacon {j}" + (" (NLOS)" if j in nlos_biased else ""))
ax1.axvline(0.0, color="k", linewidth=1)
ax1.set_xlabel("Range residual [m]")
ax1.set_ylabel("Query points")
ax1.set_title("Two distributions, shifted -- not widened")
ax1.legend(fontsize=8)

ax2.plot(nlos_truth[:, 0], nlos_truth[:, 1], "o", color="lightgray",
         markersize=5, label="query points")
for j, b in enumerate(nlos_beacons):
    colour = "crimson" if j in nlos_biased else "seagreen"
    ax2.plot(b[0], b[1], "^", color=colour, markersize=14)
    ax2.annotate(f"b{j}", (b[0], b[1]), textcoords="offset points",
                 xytext=(6, 6))
ax2.set_xlabel("East [m]")
ax2.set_ylabel("North [m]")
ax2.set_title("Red beacons carry +0.8 m of NLOS bias")
ax2.legend(fontsize=8)
ax2.axis("equal")

fig.tight_layout()
print("figure built")
```

## Connection to Book Equations

| Equations | What this dataset exercises |
|---|---|
| Eqs. (4.1)–(4.3) | TOA range model — and what happens when it is wrong by a constant |
| Eqs. (4.27)–(4.33) | TDOA, where the reference beacon's own bias matters |
| Eq. (4.107) | `sigma_position = DOP x sigma_measurement`, and its silence about bias |
| Table 3.1 | Robust losses: Huber, Cauchy, Geman-McClure |
| Eqs. (8.5)–(8.9) | Innovation monitoring and chi-square gating, the detection route |

Robust solvers live in `core.estimators.least_squares.robust_least_squares` and
`core.estimators.nonlinear_least_squares.robust_gauss_newton`; the gate is
`core.fusion.gating.chi_square_gate`.

## Recommended Experiments

1. **Solve it with plain least squares first, and look at the error map.**

   ```bash
   python -m ch4_rf_point_positioning.example_comparison --data ch4_rf_2d_nlos
   ```

   The error is not uniform over the floor. Positions near the two biased
   beacons are pulled differently from positions near the clean pair, because the
   bias enters through the geometry.

2. **Predict before measuring.** DOP is 1.022 and the noise is 0.1 m, so
   noise alone predicts ~0.10 m. Measure the actual RMSE. The gap is the bias,
   and it will be much larger than 0.8 x 1.022 — a systematic error does not
   propagate the way a random one does. Work out why from Eq. (4.5).

3. **Then make it robust, and check what it costs.** Re-solve with a Huber and a
   Cauchy loss. Compare against the *baseline* dataset too: a robust loss should
   cost a little accuracy when there is nothing to be robust against. If it costs
   nothing, check that the weights are actually varying.

4. **Try three biased beacons.** Regenerate with
   `--nlos-beacons 1 2 3` and watch every robust method fail together. The
   failure is not a tuning problem; it is that "which measurements are wrong" has
   stopped being answerable from the data.

## Generation

```bash
python scripts/generate_ch4_rf_2d_positioning_dataset.py --preset nlos
```

Same geometry and seed as `--preset baseline`, so a diff between the two
datasets' `toa_ranges.txt` is the bias and nothing else.

## References

- Chapter 4, and Section 8.3 for gating, *Principles of Indoor Positioning and Indoor Navigation*
- [`ch4_rf_2d_square`](../ch4_rf_2d_square/README.md) — the unbiased baseline
- [`ch8_fusion_2d_imu_uwb_nlos`](../ch8_fusion_2d_imu_uwb_nlos/README.md) — the same problem inside a filter
