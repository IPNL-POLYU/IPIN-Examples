# Ch5 Wi-Fi Fingerprinting Dataset: Grid-Based Radio Map

## Overview

This dataset demonstrates **Wi-Fi fingerprinting for indoor positioning** using RSS (Received Signal Strength) measurements. It provides a comprehensive **radio map** with reference points (RPs) covering a multi-floor building, enabling exploration of deterministic and probabilistic fingerprinting methods.

**Key Learning Objective**: Understand fingerprinting as a pattern-matching approach where positioning = finding the best match between a query measurement and a pre-built database.

## Dataset Purpose

### Learning Goals
1. **Fingerprinting Paradigm**: Learn pattern-matching vs. geometric positioning
2. **Method Comparison**: Compare NN, k-NN, and probabilistic approaches (Eqs. 5.1-5.5)
3. **Grid Spacing Impact**: Dense vs. sparse grids (2m vs. 10m → 5× accuracy difference!)
4. **Multi-Floor Positioning**: Floor identification from RSS attenuation
5. **Database Size Trade-offs**: Accuracy vs. storage/computation costs

### Implemented Equations
- **Eq. (5.1)**: Nearest-Neighbor (NN)
  ```
  i* = argmin_i D(z, f_i)
  x_hat = x_{i*}
  where D is distance metric (Euclidean or Manhattan)
  ```

- **Eq. (5.2)**: k-Nearest-Neighbor (k-NN)
  ```
  x_hat = Σ_{i ∈ K(z)} w_i x_i / Σ_{i ∈ K(z)} w_i
  where K(z) are k nearest neighbors, w_i = 1/(D(z, f_i) + ε)
  ```

- **Eq. (5.3)**: Log-likelihood (Gaussian Naive Bayes)
  ```
  log p(z|x_i) = Σ_j log N(z_j; μ_{ij}, σ_{ij}^2)
  ```

- **Eq. (5.4)**: MAP (Maximum A Posteriori)
  ```
  i* = argmax_i p(x_i|z)
  x_hat = x_{i*}
  ```

- **Eq. (5.5)**: Posterior Mean
  ```
  x_hat = Σ_i p(x_i|z) x_i
  ```

## Dataset Variants

| Variant | Grid Spacing | RPs/Floor | Total RPs | Positioning Accuracy | Use Case |
|---------|--------------|-----------|-----------|---------------------|----------|
| **Baseline** | 5m | 121 (11×11) | 363 | ~2-3m | Standard radio map |
| **Dense** | 2m | 676 (26×26) | 2,028 | ~1-1.5m | High accuracy, high cost |
| **Sparse** | 10m | 25 (5×5) | 75 | ~5-8m | Quick deployment, lower accuracy |
| **Few APs** | 5m (4 APs) | 121 | 363 | ~3-5m | Limited infrastructure |

**Generate variants**:
```bash
python scripts/generate_wifi_fingerprint_dataset.py --preset baseline
python scripts/generate_wifi_fingerprint_dataset.py --preset dense
python scripts/generate_wifi_fingerprint_dataset.py --preset sparse
python scripts/generate_wifi_fingerprint_dataset.py --preset few_aps
```

## Files

### Fingerprint Database
- `locations.npy`: Reference point 2D positions [N×2] (x, y in meters)
- `features.npy`: RSS measurements [N×M] (M = number of APs, in dBm)
- `floor_ids.npy`: Floor labels [N×1] (0=ground, 1=first, 2=second)
- `metadata.json`: Database metadata (AP positions, area size, model parameters)

**Database Structure**:
```
N = number of reference points (363 for baseline)
M = number of access points (8 for baseline)

locations[i] = [x_i, y_i]           2D position of RP i
features[i]  = [RSS_1, ..., RSS_M]  RSS vector from M APs at RP i
floor_ids[i] = floor number         Floor label for RP i
```

## Loading Data

### Python
```python
import numpy as np
from pathlib import Path
from core.fingerprinting import load_fingerprint_database

# Load database
db = load_fingerprint_database("data/sim/ch5_wifi_fingerprint_grid")

print(f"Database: {db}")
print(f"Reference points: {db.n_samples}")
print(f"APs: {db.n_features}")
print(f"Floors: {np.unique(db.floor_ids)}")
print(f"Area: {db.meta['area_size']}")
print(f"Grid spacing: {db.meta['grid_spacing']}m")

# Access data
print(f"Locations shape: {db.locations.shape}")  # (363, 2)
print(f"Features shape: {db.features.shape}")    # (363, 8)
print(f"RSS range: [{db.features.min():.1f}, {db.features.max():.1f}] dBm")
```

### Manual Loading (without library)
```python
import numpy as np
from pathlib import Path

data_dir = Path("data/sim/ch5_wifi_fingerprint_grid")

locations = np.load(data_dir / "locations.npy")
features = np.load(data_dir / "features.npy")
floor_ids = np.load(data_dir / "floor_ids.npy")

import json
with open(data_dir / "metadata.json") as f:
    metadata = json.load(f)

print(f"Loaded {len(locations)} reference points")
print(f"AP positions: {metadata['ap_positions']}")
```

## Configuration Parameters

### Area Configuration
```json
{
  "area_size": [50.0, 50.0],
  "grid_spacing": 5.0,
  "n_floors": 3,
  "floor_height": 3.0
}
```

**Key Parameters**:
- **area_size**: Building footprint (50m × 50m)
- **grid_spacing**: Distance between RPs (5m → 11×11 grid)
- **n_floors**: Number of floors (3)
- **floor_height**: Vertical spacing (3m)

### Access Point Configuration
```json
{
  "ap_ids": ["AP1", "AP2", ..., "AP8"],
  "ap_positions": [
    [0, 0, 2.5],      // Corner 1
    [50, 0, 2.5],     // Corner 2
    [50, 50, 2.5],    // Corner 3
    [0, 50, 2.5],     // Corner 4
    [25, 0, 2.5],     // Mid-wall 1
    [25, 50, 2.5],    // Mid-wall 2
    [0, 25, 2.5],     // Mid-wall 3
    [50, 25, 2.5]     // Mid-wall 4
  ]
}
```

**AP Placement Strategy**: Corners + mid-walls for good coverage

### Path-Loss Model Configuration
```json
{
  "path_loss_model": {
    "type": "log_distance",
    "P0_dBm": -30.0,
    "path_loss_exponent": 2.5,
    "shadow_fading_std_dBm": 4.0,
    "floor_attenuation_dB": 15.0
  }
}
```

**Model**: RSS(d) = P₀ - 10×n×log₁₀(d/d₀) + X_σ - floor_attenuation

## Quick Start Examples

### Example 1: Nearest-Neighbor Positioning
```python
from core.fingerprinting import load_fingerprint_database, nn_localize
import numpy as np

# Load database
db = load_fingerprint_database("data/sim/ch5_wifi_fingerprint_grid")

# Create query measurement (simulate at position [25, 25] on floor 0)
query_rss = np.array([-45, -50, -60, -65, -42, -58, -48, -52])  # dBm

# NN positioning (Eq. 5.1)
estimated_pos = nn_localize(query_rss, db, metric="euclidean", floor_id=0)

print(f"Estimated position: {estimated_pos}")
# Expected: close to [25, 25] (center of area)
```

**Expected Accuracy**: ~2-3m for baseline grid

### Example 2: k-NN Positioning
```python
from core.fingerprinting import knn_localize

# k-NN with k=5 (Eq. 5.2)
estimated_pos_knn = knn_localize(
    query_rss, db,
    k=5,
    metric="euclidean",
    weighting="inverse_distance",
    floor_id=0
)

print(f"k-NN estimated position: {estimated_pos_knn}")
```

**Expected**: Smoother estimate than NN, often more accurate

### Example 3: Probabilistic (MAP) Positioning
```python
from core.fingerprinting import fit_gaussian_naive_bayes, map_localize

# Train Gaussian Naive Bayes model
model = fit_gaussian_naive_bayes(db)

# MAP positioning (Eq. 5.4)
estimated_pos_map = map_localize(query_rss, model, floor_id=0)

print(f"MAP estimated position: {estimated_pos_map}")
```

**Expected**: Similar to NN but probabilistic

### Example 4: Posterior Mean Positioning
```python
from core.fingerprinting import posterior_mean_localize

# Posterior mean (Eq. 5.5)
estimated_pos_pm = posterior_mean_localize(query_rss, model, floor_id=0)

print(f"Posterior mean position: {estimated_pos_pm}")
```

**Expected**: Smoother than MAP, similar to k-NN

## Visualization

### Plot Radio Map (Single AP)
```python
import matplotlib.pyplot as plt
from core.fingerprinting import load_fingerprint_database

db = load_fingerprint_database("data/sim/ch5_wifi_fingerprint_grid")

# Filter floor 0
floor_0_mask = db.floor_ids == 0
locs_floor_0 = db.locations[floor_0_mask]
rss_ap1_floor_0 = db.features[floor_0_mask, 0]  # AP1

# Plot heatmap
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    locs_floor_0[:, 0],
    locs_floor_0[:, 1],
    c=rss_ap1_floor_0,
    cmap='RdYlGn',
    s=100,
    edgecolors='black'
)
plt.colorbar(scatter, label='RSS from AP1 (dBm)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Radio Map: AP1 on Floor 0')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

**Learning Point**: Visualize how RSS varies spatially!

### Plot All AP Coverage
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i in range(8):  # 8 APs
    ax = axes[i]
    scatter = ax.scatter(
        locs_floor_0[:, 0],
        locs_floor_0[:, 1],
        c=db.features[floor_0_mask, i],
        cmap='RdYlGn',
        s=50
    )
    ax.set_title(f'AP{i+1}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()
```

## Parameter Effects

### Effect of Grid Spacing

| Grid Spacing | RPs/Floor | Total RPs | Positioning Error | Storage | Collection Time |
|--------------|-----------|-----------|-------------------|---------|-----------------|
| 2m (dense) | 676 | 2,028 | ~1-1.5m | 5.6× | 5.6× |
| 5m (baseline) | 121 | 363 | ~2-3m | 1× | 1× |
| 10m (sparse) | 25 | 75 | ~5-8m | 0.2× | 0.2× |

**Formula**: Positioning error ≈ grid_spacing / 2 (rule of thumb)

**Generate comparison**:
```bash
python scripts/generate_wifi_fingerprint_dataset.py --preset dense
python scripts/generate_wifi_fingerprint_dataset.py --preset baseline
python scripts/generate_wifi_fingerprint_dataset.py --preset sparse
```

**Learning Point**: Dense grids → better accuracy but 5× more effort!

### Effect of k in k-NN

| k Value | Positioning Error | Characteristics | Best Use Case |
|---------|-------------------|-----------------|---------------|
| 1 (NN) | ~2.5m | Discrete, fast | Real-time |
| 3 | ~2.2m | Smoother | General purpose |
| 5 | ~2.0m | More robust | Moderate noise |
| 7 | ~2.1m | Over-smoothing | High noise |
| 10+ | ~2.5m | Too smooth | Not recommended |

**Optimal k**: Typically 3-5 for 5m grid spacing

### Effect of Number of APs

| Number of APs | RSS Dimension | Positioning Error | Notes |
|---------------|---------------|-------------------|-------|
| 4 (few_aps) | 4D | ~3-5m | Minimum viable |
| 8 (baseline) | 8D | ~2-3m | Good coverage |
| 12+ | 12+D | ~1.5-2.5m | Diminishing returns |

**Generate comparison**:
```bash
python scripts/generate_wifi_fingerprint_dataset.py --preset baseline      # 8 APs
python scripts/generate_wifi_fingerprint_dataset.py --preset few_aps      # 4 APs
python scripts/generate_wifi_fingerprint_dataset.py --n-aps 12 --output data/sim/wifi_fp_many_aps
```

**Learning Point**: 8 APs is sweet spot (good accuracy, manageable infrastructure)

## Experiments

### Experiment 1: Method Comparison

**Objective**: Compare NN, k-NN, MAP, and Posterior Mean positioning.

**Procedure**:
1. Load baseline database
2. Generate 200 test queries with 2 dBm noise
3. Run all four methods
4. Compare errors

**Expected Results**:
- NN: ~2.5m error (discrete estimates)
- k-NN (k=5): ~2.0m error (smooth, best deterministic)
- MAP: ~2.3m error (discrete, probabilistic)
- Posterior Mean: ~2.1m error (smooth, probabilistic)

**Code**: See `ch5_fingerprinting/example_comparison.py`

**Learning Point**: k-NN and Posterior Mean are most accurate!

### Experiment 2: Grid Spacing Impact

**Objective**: Quantify how grid spacing affects positioning accuracy.

**Procedure**:
1. Generate databases with 2m, 5m, and 10m spacing
2. Test k-NN positioning on each
3. Measure errors

**Expected Results**:
- 2m grid: ~1.5m error (excellent)
- 5m grid: ~2.5m error (good)
- 10m grid: ~6.0m error (poor)

**Rule of Thumb**: Positioning error ≈ grid_spacing / 2

### Experiment 3: Multi-Floor Positioning

**Objective**: Test floor identification from RSS patterns.

**Procedure**:
1. Generate queries on different floors
2. Run positioning without floor constraint
3. Check floor identification accuracy

**Expected Results**:
- Floor identification: ~95% accuracy (15 dB attenuation/floor helps!)
- Same-floor positioning: ~2-3m error
- Cross-floor errors: Large (>20m) but rare

## Performance Metrics (Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| **Database Size** | 363 RPs | 121 per floor × 3 floors |
| **Grid Spacing** | 5m | 11×11 grid |
| **Area Coverage** | 50m × 50m | 2,500 m² |
| **Features** | 8 APs | RSS dimensionality |
| **RSS Range** | -115 to -28 dBm | Path-loss + noise |
| **NN Error** | ~2.5m | Euclidean distance |
| **k-NN Error (k=5)** | ~2.0m | Inverse distance weighting |
| **MAP Error** | ~2.3m | Gaussian Naive Bayes |
| **Posterior Mean Error** | ~2.1m | Probabilistic average |
| **Floor ID Accuracy** | ~95% | 15 dB attenuation/floor |

**Comparison**:
- Dense (2m): ~1.5m error (2.5× better, 5.6× larger database)
- Sparse (10m): ~6.0m error (3× worse, 5× smaller database)

## Book Connection

### Chapter 5: Fingerprinting-based Positioning

This dataset directly supports Chapter 5 fingerprinting methods:

1. **Deterministic Methods (Section 5.1)**
   - Nearest-Neighbor (Eq. 5.1): Finds closest match
   - k-Nearest-Neighbor (Eq. 5.2): Weighted average of k matches
   - **Key Insight**: Simple pattern matching works well!

2. **Probabilistic Methods (Section 5.2)**
   - Gaussian Naive Bayes model (Eq. 5.3)
   - MAP estimation (Eq. 5.4): Most likely position
   - Posterior Mean (Eq. 5.5): Expected position
   - **Key Insight**: Probabilistic ≈ Deterministic for this problem!

3. **Database Construction (Section 5.3)**
   - Grid-based sampling strategy
   - Path-loss model for RSS generation
   - Multi-floor considerations
   - **Key Insight**: Database quality = positioning quality!

4. **Key Trade-offs**:
   - Accuracy vs. database size (grid spacing)
   - Computation vs. storage
   - Coverage vs. cost (number of APs)

## Common Issues & Solutions

### Issue 1: Poor Positioning Accuracy (>5m error)

**Symptoms**: All methods give large errors

**Likely Causes**:
- Database and query use different floors
- Too few APs (< 4)
- Grid too sparse (> 10m)

**Solution**: Check floor constraint and grid spacing:
```python
# Always specify floor if known
estimated_pos = knn_localize(query, db, k=5, floor_id=0)  # Correct

# Without floor constraint, may match wrong floor
estimated_pos = knn_localize(query, db, k=5)  # May be wrong!
```

### Issue 2: NN Gives Discrete Jumps

**Symptoms**: Position estimate jumps between grid points

**Cause**: NN always returns a reference point (discrete)

**Solution**: Use k-NN or Posterior Mean for smooth estimates:
```python
# NN: discrete
pos_nn = nn_localize(query, db)  # One of the RPs

# k-NN: smooth
pos_knn = knn_localize(query, db, k=5)  # Weighted average
```

### Issue 3: Slow Positioning (>100ms)

**Symptoms**: Real-time requirements not met

**Cause**: Large database (dense grid)

**Solutions**:
- Use NN instead of k-NN (3× faster)
- Spatial indexing (k-d tree)
- GPU acceleration for batch queries

## Troubleshooting

### Error: Database dimension mismatch

**Cause**: Query has different number of APs than database

**Fix**: Ensure query matches database:
```python
print(f"Database features: {db.n_features}")  # Should be 8
print(f"Query features: {len(query)}")  # Must match!
```

### Warning: High RSS variability

**Cause**: Shadow fading or measurement noise

**Fix**: This is realistic! Real Wi-Fi has σ = 4-8 dBm variability

## Next Steps

After understanding Wi-Fi fingerprinting:

1. **Chapter 8**: Sensor fusion (fingerprinting + IMU/PDR)
2. **Advanced Methods**: Gaussian Process regression, neural networks
3. **Real Data**: Collect actual fingerprints (trickier than simulation!)
4. **Hybrid**: Combine fingerprinting with RF ranging

## Citation

If you use this dataset in your research, please cite:

```bibtex
@book{IPIN2024,
  title={Principles of Indoor Positioning and Indoor Navigation},
  author={[Authors]},
  year={2024},
  chapter={5},
  note={Fingerprinting-based Indoor Positioning}
}
```

## License

This dataset is part of the IPIN Book Examples repository. See repository LICENSE for details.

---

**Dataset Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: See repository README for contact information

