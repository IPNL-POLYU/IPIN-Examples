"""Verification script for temporal calibration dataset.

Tests that the timeoffset dataset has real time offsets and drift applied
to UWB sensor timestamps, not just cosmetic config values.

Author: Li-Ta Hsu
Date: December 2025
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fusion import TimeSyncModel


def verify_temporal_calibration(dataset_path: str = "data/sim/ch8_fusion_2d_imu_uwb_timeoffset"):
    """Verify that temporal calibration is real, not cosmetic.
    
    Args:
        dataset_path: Path to the timeoffset dataset
    """
    print(f"\n{'='*70}")
    print("Temporal Calibration Verification")
    print(f"{'='*70}\n")
    
    dataset_path = Path(dataset_path)
    
    # Load data
    print(f"Loading dataset from: {dataset_path}")
    
    # Load config
    import json
    with open(dataset_path / "config.json") as f:
        config = json.load(f)
    
    time_offset = config['temporal_calibration']['time_offset_sec']
    clock_drift = config['temporal_calibration']['clock_drift']
    
    print(f"\nConfig parameters:")
    print(f"  Time offset: {time_offset*1000:.1f} ms")
    print(f"  Clock drift: {clock_drift*1e6:.1f} ppm")
    
    # Load timestamps
    truth = np.load(dataset_path / "truth.npz")
    imu = np.load(dataset_path / "imu.npz")
    uwb = np.load(dataset_path / "uwb_ranges.npz")
    
    t_truth = truth['t']
    t_imu = imu['t']
    t_uwb_sensor = uwb['t']
    
    print(f"\nTimestamp ranges:")
    print(f"  Truth: [{t_truth[0]:.3f}, {t_truth[-1]:.3f}] s")
    print(f"  IMU:   [{t_imu[0]:.3f}, {t_imu[-1]:.3f}] s")
    print(f"  UWB:   [{t_uwb_sensor[0]:.3f}, {t_uwb_sensor[-1]:.3f}] s")
    
    # ========================================================================
    # TEST 1: Verify that UWB timestamps differ from fusion time
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: UWB timestamps are in sensor time (not fusion time)")
    print(f"{'='*70}")
    
    # Expected sensor time at t_fusion = 0
    t_fusion_0 = t_truth[0]  # Should be 0.0
    t_sensor_expected_0 = (t_fusion_0 - time_offset) / (1.0 + clock_drift)
    
    # Actual sensor time at first UWB measurement
    t_sensor_actual_0 = t_uwb_sensor[0]
    
    # Expected sensor time at t_fusion = 10.0
    idx_fusion_10 = np.argmin(np.abs(t_truth - 10.0))
    t_fusion_10 = t_truth[idx_fusion_10]
    t_sensor_expected_10 = (t_fusion_10 - time_offset) / (1.0 + clock_drift)
    
    # Find corresponding UWB measurement near fusion time 10.0
    idx_uwb_10 = np.argmin(np.abs(t_uwb_sensor - t_sensor_expected_10))
    t_sensor_actual_10 = t_uwb_sensor[idx_uwb_10]
    
    print(f"\nAt fusion time t=0.0s:")
    print(f"  Expected sensor time: {t_sensor_expected_0:.6f} s")
    print(f"  Actual sensor time:   {t_sensor_actual_0:.6f} s")
    print(f"  Difference:           {abs(t_sensor_actual_0 - t_sensor_expected_0)*1000:.3f} ms")
    
    print(f"\nAt fusion time t~10.0s:")
    print(f"  Expected sensor time: {t_sensor_expected_10:.6f} s")
    print(f"  Actual sensor time:   {t_sensor_actual_10:.6f} s")
    print(f"  Difference:           {abs(t_sensor_actual_10 - t_sensor_expected_10)*1000:.3f} ms")
    
    # Check that UWB timestamps ARE different from fusion time
    offset_at_start = t_uwb_sensor[0] - t_truth[0]
    print(f"\nOffset at start (sensor - fusion): {offset_at_start*1000:.3f} ms")
    
    # With offset=-0.05 and drift=0.0001:
    # t_sensor = (t_fusion - (-0.05)) / (1 + 0.0001) = (t_fusion + 0.05) / 1.0001
    # At t_fusion=0: t_sensor ≈ 0.05 / 1.0001 ≈ 0.04999
    # So sensor is ahead by ~50ms (because offset is negative = sensor behind fusion)
    
    test1_pass = abs(offset_at_start*1000 - (-time_offset*1000)) < 1.0  # Within 1ms
    
    if test1_pass:
        print(f"[PASS] TEST 1 PASSED: UWB timestamps are in sensor time")
    else:
        print(f"[FAIL] TEST 1 FAILED: UWB timestamps appear to be in fusion time (cosmetic offset)")
    
    # ========================================================================
    # TEST 2: Verify that drift accumulates over time
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Clock drift accumulates over time")
    print(f"{'='*70}")
    
    # Check drift accumulation at different time points
    test_fusion_times = [0.0, 20.0, 40.0, 60.0]
    drift_errors = []
    
    print(f"\nDrift accumulation:")
    for t_fus in test_fusion_times:
        if t_fus > t_truth[-1]:
            continue
        
        # Expected sensor time
        t_sens_expected = (t_fus - time_offset) / (1.0 + clock_drift)
        
        # Find nearest UWB measurement
        idx_uwb = np.argmin(np.abs(t_uwb_sensor - t_sens_expected))
        t_sens_actual = t_uwb_sensor[idx_uwb]
        
        # Drift contribution (should grow linearly with time)
        drift_contribution = t_fus * clock_drift / (1.0 + clock_drift)
        
        error = abs(t_sens_actual - t_sens_expected) * 1000
        drift_errors.append(error)
        
        print(f"  t_fusion={t_fus:5.1f}s: sensor={t_sens_actual:7.4f}s, "
              f"expected={t_sens_expected:7.4f}s, error={error:.3f}ms, "
              f"drift_contrib={drift_contribution*1000:.2f}ms")
    
    # Drift errors should be small and not grow significantly
    # (errors from nearest-neighbor sampling should dominate, not systematic drift error)
    max_drift_error = max(drift_errors)
    test2_pass = max_drift_error < 2.0  # Within 2ms (generous for UWB rate)
    
    if test2_pass:
        print(f"[PASS] TEST 2 PASSED: Drift correctly applied (max error {max_drift_error:.3f}ms)")
    else:
        print(f"[FAIL] TEST 2 FAILED: Drift not correctly applied (max error {max_drift_error:.3f}ms)")
    
    # ========================================================================
    # TEST 3: TimeSyncModel can recover fusion time
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST 3: TimeSyncModel recovers fusion time")
    print(f"{'='*70}")
    
    # Create TimeSyncModel with same offset/drift (to convert sensor→fusion)
    # The TimeSyncModel formula: t_fusion = (1 + drift) * t_sensor + offset
    # which inverts the generation formula: t_sensor = (t_fusion - offset) / (1 + drift)
    time_sync = TimeSyncModel(offset=time_offset, drift=clock_drift)
    
    # Convert first few UWB sensor timestamps back to fusion time
    t_uwb_recovered = np.array([time_sync.to_fusion_time(t) for t in t_uwb_sensor[:5]])
    
    # Expected: The recovered fusion times should be close to the original fusion time grid
    # UWB was sampled at 10 Hz starting at fusion t=0, so fusion grid is [0, 0.1, 0.2, 0.3, 0.4, ...]
    uwb_rate = config['uwb']['rate_hz']
    dt_uwb = 1.0 / uwb_rate
    
    # The generation code samples at np.arange(t[0], t[-1], dt_uwb)
    # So first UWB measurement is at fusion time t=0.0
    t_fusion_grid = np.arange(0.0, 0.5, dt_uwb)
    
    print(f"\nFirst 5 UWB measurements:")
    print(f"{'Sensor Time':>12} {'Recovered Fusion':>18} {'Expected Fusion':>18} {'Error (ms)':>12}")
    print(f"{'-'*70}")
    
    recovery_errors = []
    for i in range(min(5, len(t_uwb_recovered))):
        error = (t_uwb_recovered[i] - t_fusion_grid[i]) * 1000
        recovery_errors.append(abs(error))
        print(f"{t_uwb_sensor[i]:12.6f} {t_uwb_recovered[i]:18.6f} "
              f"{t_fusion_grid[i]:18.6f} {error:12.3f}")
    
    max_recovery_error = max(recovery_errors)
    test3_pass = max_recovery_error < 0.5  # Within 0.5ms
    
    if test3_pass:
        print(f"[PASS] TEST 3 PASSED: TimeSyncModel correctly recovers fusion time "
              f"(max error {max_recovery_error:.3f}ms)")
    else:
        print(f"[FAIL] TEST 3 FAILED: TimeSyncModel does not recover fusion time "
              f"(max error {max_recovery_error:.3f}ms)")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print(f"[PASS] ALL TESTS PASSED")
        print(f"\nTemporal calibration is REAL (not cosmetic):")
        print(f"  - UWB timestamps are in sensor time")
        print(f"  - Clock drift accumulates correctly")
        print(f"  - TimeSyncModel recovers fusion time")
        print(f"\nThe dataset correctly implements time offset and drift!")
        return 0
    else:
        print(f"[FAIL] SOME TESTS FAILED")
        print(f"\nTemporal calibration may be cosmetic (config-only).")
        print(f"Please check the dataset generation script.")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify temporal calibration in timeoffset dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sim/ch8_fusion_2d_imu_uwb_timeoffset",
        help="Path to timeoffset dataset (default: data/sim/ch8_fusion_2d_imu_uwb_timeoffset)"
    )
    
    args = parser.parse_args()
    
    exit_code = verify_temporal_calibration(args.dataset)
    sys.exit(exit_code)

