"""
Hand-check geometry verification for AOA measurement models.

This script verifies the AOA implementation against book equations 4.63-4.65.
"""

import numpy as np

from core.rf import (
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    aoa_sin_elevation,
    aoa_tan_azimuth,
)


def main():
    """Run hand-check geometry verification."""
    print("=" * 60)
    print("Hand-check Geometry Verification (Book Eqs. 4.63-4.65)")
    print("=" * 60)

    # Hand-check geometry from book:
    # Beacon at (0, 10, 5) in ENU
    # Agent at (5, 5, 0) in ENU
    anchor = np.array([0.0, 10.0, 5.0])  # (E=0, N=10, U=5)
    agent = np.array([5.0, 5.0, 0.0])  # (E=5, N=5, U=0)

    print(f"\nAnchor position (E, N, U): {anchor}")
    print(f"Agent position (E, N, U): {agent}")

    # Compute differences
    delta_e = anchor[0] - agent[0]  # 0 - 5 = -5
    delta_n = anchor[1] - agent[1]  # 10 - 5 = 5
    delta_u = anchor[2] - agent[2]  # 5 - 0 = 5
    distance = np.linalg.norm(agent - anchor)

    print("\nDifferences (anchor - agent):")
    print(f"  Delta_E = {delta_e}")
    print(f"  Delta_N = {delta_n}")
    print(f"  Delta_U = {delta_u}")
    print(f"  distance = sqrt({delta_e**2} + {delta_n**2} + {delta_u**2}) = {distance:.4f}")

    # Book Eq. 4.63: sin(theta) = Delta_U / d
    expected_sin_theta = delta_u / distance
    sin_theta = aoa_sin_elevation(anchor, agent)
    match1 = np.isclose(sin_theta, expected_sin_theta)
    print(f"\nEq. 4.63: sin(theta) = Delta_U / d = {delta_u} / {distance:.4f} = {expected_sin_theta:.6f}")
    print(f"  Computed: sin(theta) = {sin_theta:.6f}")
    print(f"  Match: {match1}")

    # Book Eq. 4.64: tan(psi) = Delta_E / Delta_N
    expected_tan_psi = delta_e / delta_n
    tan_psi = aoa_tan_azimuth(anchor, agent)
    match2 = np.isclose(tan_psi, expected_tan_psi)
    print(f"\nEq. 4.64: tan(psi) = Delta_E / Delta_N = {delta_e} / {delta_n} = {expected_tan_psi:.6f}")
    print(f"  Computed: tan(psi) = {tan_psi:.6f}")
    print(f"  Match: {match2}")

    # Azimuth angle
    azimuth = aoa_azimuth(anchor, agent)
    expected_azimuth = np.arctan2(delta_e, delta_n)
    match3 = np.isclose(azimuth, expected_azimuth)
    print(f"\nAzimuth: psi = atan2(Delta_E, Delta_N) = atan2({delta_e}, {delta_n}) = {np.rad2deg(expected_azimuth):.2f} deg")
    print(f"  Computed: psi = {np.rad2deg(azimuth):.2f} deg")
    print(f"  Match: {match3}")

    # Elevation angle
    elevation = aoa_elevation(anchor, agent)
    horiz_dist = np.sqrt(delta_e**2 + delta_n**2)
    expected_elevation = np.arctan2(delta_u, horiz_dist)
    match4 = np.isclose(elevation, expected_elevation)
    print(f"\nElevation: theta = arctan2(Delta_U, horiz_dist) = {np.rad2deg(expected_elevation):.2f} deg")
    print(f"  Computed: theta = {np.rad2deg(elevation):.2f} deg")
    print(f"  Match: {match4}")

    # Measurement vector (Eq. 4.65)
    z = aoa_measurement_vector(anchor.reshape(1, -1), agent, include_elevation=True)
    match5 = np.allclose(z, [expected_sin_theta, expected_tan_psi])
    print(f"\nEq. 4.65 Measurement vector: z = [sin(theta), tan(psi)]")
    print(f"  Expected: [{expected_sin_theta:.6f}, {expected_tan_psi:.6f}]")
    print(f"  Computed: {z}")
    print(f"  Match: {match5}")

    print("\n" + "=" * 60)
    all_pass = match1 and match2 and match3 and match4 and match5
    if all_pass:
        print("All hand-check verifications PASSED!")
    else:
        print("Some verifications FAILED!")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

