# Prompt 8: Truth-Free Odometry - Quick Reference

## What Was Changed

### Change 1: Dataset Mode (Line 126-130)

**BEFORE (Ground Truth Oracle):**
```python
# Prepare odometry measurements
odometry_measurements = []
for i in range(n_poses - 1):
    rel_pose = se2_relative(np.array(true_poses[i]), np.array(true_poses[i + 1]))
    rel_pose[0] += np.random.normal(0, 0.05)
    rel_pose[1] += np.random.normal(0, 0.05)
    rel_pose[2] += np.random.normal(0, 0.01)
    odometry_measurements.append((i, i + 1, rel_pose))
```

**AFTER (Sensor Data):**
```python
# Prepare odometry measurements (from noisy odometry, NOT ground truth)
odometry_measurements = []
for i in range(n_poses - 1):
    # CRITICAL: Use odom_poses (sensor data), never true_poses
    rel_pose = se2_relative(np.array(odom_poses[i]), np.array(odom_poses[i + 1]))
    odometry_measurements.append((i, i + 1, rel_pose))
```

**Why:** `odom_poses` comes from the dataset file `odometry_poses.txt` which has realistic drift. No need to add extra noise.

---

### Change 2: Inline Mode (Line 670-676)

**BEFORE (Ground Truth Oracle):**
```python
# Prepare odometry measurements (consecutive poses)
odometry_measurements = []
for i in range(n_poses - 1):
    rel_pose = se2_relative(true_poses[i], true_poses[i + 1])
    # Add odometry noise (simulate sensor noise)
    rel_pose[0] += np.random.normal(0, 0.05)
    rel_pose[1] += np.random.normal(0, 0.05)
    rel_pose[2] += np.random.normal(0, 0.01)
    odometry_measurements.append((i, i + 1, rel_pose))
```

**AFTER (Sensor Data):**
```python
# Prepare odometry measurements (from noisy odometry, NOT ground truth)
odometry_measurements = []
for i in range(n_poses - 1):
    # CRITICAL: Use odom_poses (sensor data with drift), never true_poses
    # The drift is already baked in from add_odometry_noise()
    rel_pose = se2_relative(odom_poses[i], odom_poses[i + 1])
    odometry_measurements.append((i, i + 1, rel_pose))
```

**Why:** `odom_poses` was already generated with drift from `add_odometry_noise(true_poses, ...)`. No need to add more noise.

---

## Data Flow Diagram

### BEFORE (Oracle-Based)
```
┌─────────────┐
│ true_poses  │ (Ground Truth)
└──────┬──────┘
       │
       ├──────────────────────────┐
       │                          │
       ↓                          ↓
  se2_relative()           add_odometry_noise()
       │                          │
       ↓                          ↓
  odometry_measurements       odom_poses
  (ORACLE CONSTRAINTS)        (Used for initial guess only)
       │
       ↓
  create_pose_graph()
       │
       ↓
   OPTIMIZATION
  (99% correct already!)
```

### AFTER (Sensor-Based)
```
┌─────────────┐
│ true_poses  │ (Ground Truth - for simulation only)
└──────┬──────┘
       │
       ↓
  add_odometry_noise()
       │
       ↓
  ┌──────────┐
  │odom_poses│ (Simulated Sensor Data)
  └────┬─────┘
       │
       ├──────────────────────────┐
       │                          │
       ↓                          ↓
  se2_relative()           Initial guess
       │
       ↓
  odometry_measurements
  (REALISTIC CONSTRAINTS with drift)
       │
       ↓
  create_pose_graph()
       │
       ↓
   OPTIMIZATION
  (Must correct real drift!)
```

---

## Testing Commands

Run these three tests to verify the fix:

```bash
# Test 1: Inline mode (generates synthetic data)
python -m ch7_slam.example_pose_graph_slam

# Test 2: Square trajectory dataset
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_square

# Test 3: High drift scenario
python -m ch7_slam.example_pose_graph_slam --data ch7_slam_2d_high_drift
```

**Expected output:**
- All three should run without errors
- Figures saved to `ch7_slam/figs/pose_graph_slam_results.png`
- Console output shows ~90-95% error reduction
- RMSE values should be similar to before (but now with honest constraints!)

---

## Verification Checklist

Run this verification manually:

```bash
# 1. Check for ground truth contamination in odometry
grep -n "se2_relative.*true_poses\|true_poses.*se2_relative" ch7_slam/example_pose_graph_slam.py

# Expected: Only ONE match at line 373 (inside add_odometry_noise function)
# If you see matches at lines ~129 or ~675, the fix failed!

# 2. Check odometry measurements use odom_poses
grep -B5 "odometry_measurements.append" ch7_slam/example_pose_graph_slam.py | grep se2_relative

# Expected: Both occurrences should show "odom_poses", NOT "true_poses"

# 3. Check for linter errors
# (Should be none)
```

---

## Impact

### What Changed
- ✅ Odometry factors now come from **realistic sensor data** (`odom_poses`)
- ✅ Removed oracle: no longer using `true_poses` for measurements
- ✅ Removed redundant noise addition (drift already present)

### What Stayed the Same
- ✅ Graph structure unchanged
- ✅ Optimization algorithm unchanged
- ✅ Evaluation metrics unchanged (still use ground truth for RMSE)
- ✅ Output format unchanged

### What's Still Wrong (Future Prompts)
- ❌ Loop closure detection uses position-based oracle (line 431)
- ❌ No scan-to-scan odometry (uses pre-generated wheel odometry)
- ❌ No map building/updating

---

## Next Steps

**If tests pass:**
1. ✅ Mark Prompt 8 as complete
2. → Proceed to **Prompt 9**: Fix loop closure detection (use observation similarity, not position oracle)

**If tests fail:**
1. Check Python environment setup
2. Verify the changes were applied correctly (grep verification above)
3. Check for new errors in console output
4. Report issues with full error message

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `ch7_slam/example_pose_graph_slam.py` | 126-130 | Dataset mode: use `odom_poses` |
| `ch7_slam/example_pose_graph_slam.py` | 670-676 | Inline mode: use `odom_poses` |

**Total:** 2 code blocks, ~10 lines changed

**Breaking changes:** None (API unchanged)

**New dependencies:** None

---

**Status:** ✅ Code changes complete, ready for testing
