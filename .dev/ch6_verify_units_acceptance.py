"""
Verification script for Prompt 3 acceptance criterion.

Acceptance criterion:
If you set "10 deg/hr bias," the printed bias should be ~0.0028 deg/s, not ~10 deg/s.

Author: Li-Ta Hsu
Date: December 2025
"""

import numpy as np
from core.sensors import IMUNoiseParams, units

def main():
    print("\n" + "="*70)
    print("VERIFYING ACCEPTANCE CRITERION FOR PROMPT 3")
    print("="*70)
    print("\nCriterion: 10 deg/hr bias should print as ~0.0028 deg/s\n")
    
    # Create consumer-grade IMU with 10 deg/hr gyro bias
    params = IMUNoiseParams.consumer_grade()
    
    # Check the gyro bias value
    bias_rad_s = params.gyro_bias_rad_s
    bias_deg_hr = units.rad_per_sec_to_deg_per_hour(bias_rad_s)
    bias_deg_s = units.rad_per_sec_to_deg_per_sec(bias_rad_s)
    
    print("Consumer-grade IMU gyro bias:")
    print(f"  Specification:     10.00 deg/hr")
    print(f"  Stored value:      {bias_rad_s:.6e} rad/s")
    print(f"  Back to deg/hr:    {bias_deg_hr:.2f} deg/hr")
    print(f"  In deg/s:          {bias_deg_s:.6f} deg/s")
    print(f"\nFormatted: {units.format_gyro_bias(bias_rad_s)}")
    
    # Check acceptance criterion
    expected_deg_s = 10.0 / 3600.0  # 0.002778 deg/s
    error = abs(bias_deg_s - expected_deg_s)
    
    print(f"\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"Expected value:      {expected_deg_s:.6f} deg/s")
    print(f"Actual value:        {bias_deg_s:.6f} deg/s")
    print(f"Error:               {error:.9f} deg/s")
    print(f"Tolerance:           {0.0001:.6f} deg/s")
    
    if error < 0.0001:
        print("\n[PASS] Acceptance criterion met!")
        print("  10 deg/hr bias correctly converts to ~0.0028 deg/s")
        return True
    else:
        print("\n[FAIL] Acceptance criterion not met!")
        print(f"  Error ({error:.6f}) exceeds tolerance")
        return False


if __name__ == "__main__":
    success = main()
    print("="*70 + "\n")
    exit(0 if success else 1)

