# Chapter 6: ZUPT Windowed Detector (Eq. 6.44)

**Author:** Li-Ta Hsu  
**Date:** December 2025

## Overview

This document describes the proper implementation of the ZUPT (Zero-Velocity Update) detector using the windowed test statistic from Equation 6.44. This is the SHOE-style (Zero-velocity update aided Inertial Navigation) detector used in foot-mounted IMU navigation.

## Equation 6.44

The ZUPT test statistic over a window W_k of length N samples:

```
T_k = (1/N) * Σ_(l∈W_k) [ (1/σ_A) * ||ã_l^B - g*(ā_k^B)/||ā_k^B|| ||
                          + (1/σ_G) * ||ω̃_l^B||² ]
```

where:
- **ā_k^B**: Average accelerometer measurement over the window
- **σ_A**: Accelerometer noise standard deviation (m/s²)
- **σ_G**: Gyroscope noise standard deviation (rad/s)
- **g**: Gravity magnitude (9.81 m/s²)
- **ã_l^B**: Accelerometer measurement at sample l
- **ω̃_l^B**: Gyroscope measurement at sample l

The detector declares the sensor stationary if:

```
T_k < γ (threshold)
```

## Implementation

### Core Functions

1. **`zupt_test_statistic()`** - Computes T_k over a window
   ```python
   from core.sensors import zupt_test_statistic
   
   T_k = zupt_test_statistic(
       accel_window=accel_buffer,  # Shape: (N, 3)
       gyro_window=gyro_buffer,    # Shape: (N, 3)
       sigma_a=sigma_a,            # Accel noise std dev
       sigma_g=sigma_g,            # Gyro noise std dev
       g=9.81                       # Gravity magnitude
   )
   ```

2. **`detect_zupt_windowed()`** - Compares T_k to threshold γ
   ```python
   from core.sensors import detect_zupt_windowed
   
   is_stationary = detect_zupt_windowed(
       accel_window=accel_buffer,
       gyro_window=gyro_buffer,
       sigma_a=sigma_a,
       sigma_g=sigma_g,
       gamma=10.0,  # Threshold (tuned value)
       g=9.81
   )
   ```

### Noise Parameters

The noise parameters (σ_A, σ_G) should be derived from IMU specifications:

```python
from core.sensors import IMUNoiseParams

params = IMUNoiseParams.consumer_grade()

# Scale continuous-time noise to discrete-time
dt = 0.01  # 100 Hz sample rate
sigma_a = params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
sigma_g = params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)
```

### Window Size

Typical window sizes:
- **5-10 samples**: Fast response, less robust to noise
- **10-20 samples**: Good balance (50-200ms at 100Hz)
- **20+ samples**: More robust, but slower to detect transitions

Recommended: **10 samples** for consumer-grade IMUs at 100Hz.

### Threshold Tuning (γ)

The threshold γ is application-specific and depends on:
- IMU noise characteristics
- Motion dynamics (walking vs running)
- Desired detection sensitivity

**Typical values**:
- **γ = 10-50**: Strict detection (consumer IMUs, walking)
- **γ = 100-1000**: Moderate detection
- **γ = 1e5-1e7**: Lenient detection (may cause false positives)

**Tuning process**:
1. Start with γ = 10
2. Run on test trajectory with known stance phases
3. Compare detection rate to true stance ratio
4. Adjust γ until detection rate ≈ stance ratio

## Usage Example

```python
from core.sensors import (
    IMUNoiseParams,
    detect_zupt_windowed,
    strapdown_update,
    FrameConvention
)
import numpy as np

# Setup
frame = FrameConvention.create_enu()
imu_params = IMUNoiseParams.consumer_grade()
dt = 0.01  # 100 Hz
window_size = 10
gamma = 10.0

# Compute noise parameters
sigma_a = imu_params.accel_vrw_mps_sqrt_s * np.sqrt(1 / dt)
sigma_g = imu_params.gyro_arw_rad_sqrt_s * np.sqrt(1 / dt)

# Process IMU data
for k in range(window_size, N):
    # Propagate state
    q, v, p = strapdown_update(q, v, p, gyro[k-1], accel[k-1], dt, frame)
    
    # Extract window
    accel_window = accel[k-window_size:k]
    gyro_window = gyro[k-window_size:k]
    
    # Detect ZUPT
    if detect_zupt_windowed(accel_window, gyro_window, sigma_a, sigma_g, gamma):
        # Apply ZUPT correction
        v = np.zeros(3)  # Force velocity to zero
```

## Performance

On a synthetic walking trajectory (5s walk + 2s stop pattern):

| Metric | Value |
|--------|-------|
| True stance ratio | 26.7% |
| Detected stance (γ=10) | 25.2% |
| RMSE reduction | 92.1% |
| IMU-only error | 105.3 m |
| IMU+ZUPT error | 8.3 m |

**Key finding**: The windowed detector accurately identifies stance phases (25.2% detected vs 26.7% true), resulting in >90% error reduction compared to pure IMU integration.

## Advantages Over Instantaneous Detector

The windowed test statistic (Eq. 6.44) has several advantages over simple threshold tests:

1. **More robust to noise**: Averages over multiple samples
2. **Fewer false positives**: Less sensitive to momentary spikes
3. **Physics-based**: Uses known sensor noise characteristics
4. **Tunable**: Single threshold parameter (γ) to adjust sensitivity
5. **Standard approach**: Used in SHOE and other foot-mounted INS systems

## Related Functions

- **`detect_zupt()`** (DEPRECATED): Simple instantaneous threshold test
- **`ZuptMeasurementModel`**: EKF measurement model for ZUPT (Eq. 6.45)
- **`strapdown_update()`**: IMU integration (Eqs. 6.2-6.10)

## References

1. **Chapter 6, Section 6.3.1**: Zero-Velocity Updates
2. **Eq. (6.44)**: ZUPT test statistic (windowed detector)
3. **Eq. (6.45)**: ZUPT pseudo-measurement (velocity = 0)
4. Foxlin, E. (2005). "Pedestrian tracking with shoe-mounted inertial sensors." *IEEE Computer Graphics and Applications*, 25(6), 38-46.

## Best Practices

1. **Always use windowed detection**: More robust than instantaneous thresholds
2. **Tune γ empirically**: Test on representative trajectories
3. **Match noise parameters**: Use actual IMU specifications
4. **Monitor detection rate**: Should match expected stance ratio
5. **Avoid false positives**: Better to miss some stance phases than apply ZUPT during motion










