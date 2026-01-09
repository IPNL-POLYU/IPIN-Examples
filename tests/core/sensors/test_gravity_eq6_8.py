"""
Unit tests for gravity magnitude computation (Book Eq. 6.8).

Tests verify:
    1. Correct implementation of WGS-84 gravity formula
    2. Expected values at reference latitudes (0°, 45°, 90°)
    3. Monotonic increase from equator to pole
    4. Symmetric behavior for North/South hemispheres
    5. Backward compatibility with default fallback
    6. Degree/radian conversion helpers

Author: Li-Ta Hsu
Date: December 2025
"""

import unittest
import numpy as np
from core.sensors.gravity import (
    gravity_magnitude_eq6_8,
    gravity_magnitude,
    gravity_magnitude_from_lat_deg,
)


class TestGravityMagnitudeEq6_8(unittest.TestCase):
    """
    Test suite for Book Eq. (6.8) gravity magnitude computation.
    
    Reference values computed from WGS-84 formula:
        g(φ) = 9.7803 * (1 + 0.0053024·sin²(φ) - 0.000005·sin²(2φ))
    """
    
    def test_gravity_at_equator(self):
        """
        Test gravity magnitude at equator (φ = 0°).
        
        At equator:
            sin(0) = 0
            sin(2·0) = 0
            g(0) = 9.7803 * (1 + 0 - 0) = 9.7803 m/s²
        
        This is the minimum gravity on Earth (strongest centrifugal effect).
        """
        lat_rad = 0.0  # Equator
        g = gravity_magnitude_eq6_8(lat_rad)
        
        # Expected: exactly 9.7803 m/s²
        self.assertAlmostEqual(g, 9.7803, places=6,
                               msg="Gravity at equator should be 9.7803 m/s²")
    
    def test_gravity_at_45_degrees(self):
        """
        Test gravity magnitude at 45° latitude.
        
        At 45° (π/4 rad):
            sin(π/4) = √2/2 ≈ 0.7071
            sin²(π/4) = 0.5
            sin(π/2) = 1
            sin²(π/2) = 1
            g(π/4) = 9.7803 * (1 + 0.0053024·0.5 - 0.000005·1)
                   = 9.7803 * (1 + 0.0026512 - 0.000005)
                   = 9.7803 * 1.0026462
                   ≈ 9.8062 m/s²
        
        This is close to the commonly used approximation g = 9.81 m/s².
        """
        lat_rad = np.deg2rad(45.0)  # 45° North
        g = gravity_magnitude_eq6_8(lat_rad)
        
        # Expected: approximately 9.8062 m/s²
        expected_g = 9.7803 * (1 + 0.0053024 * 0.5 - 0.000005 * 1.0)
        self.assertAlmostEqual(g, expected_g, places=6,
                               msg="Gravity at 45° should match hand-calculated value")
        self.assertAlmostEqual(g, 9.8062, places=3,
                               msg="Gravity at 45° should be approximately 9.806 m/s²")
    
    def test_gravity_at_north_pole(self):
        """
        Test gravity magnitude at North Pole (φ = 90°).
        
        At North Pole (π/2 rad):
            sin(π/2) = 1
            sin²(π/2) = 1
            sin(π) = 0
            sin²(π) = 0
            g(π/2) = 9.7803 * (1 + 0.0053024·1 - 0.000005·0)
                   = 9.7803 * (1 + 0.0053024)
                   = 9.7803 * 1.0053024
                   ≈ 9.8322 m/s²
        
        This is the maximum gravity on Earth (no centrifugal effect).
        """
        lat_rad = np.pi / 2  # North Pole (90°)
        g = gravity_magnitude_eq6_8(lat_rad)
        
        # Expected: approximately 9.8322 m/s²
        expected_g = 9.7803 * (1 + 0.0053024)
        self.assertAlmostEqual(g, expected_g, places=6,
                               msg="Gravity at North Pole should match hand-calculated value")
        self.assertAlmostEqual(g, 9.8322, places=3,
                               msg="Gravity at pole should be approximately 9.832 m/s²")
    
    def test_gravity_at_south_pole(self):
        """
        Test gravity magnitude at South Pole (φ = -90°).
        
        Gravity should be symmetric: g(-90°) = g(+90°).
        """
        lat_rad_south = -np.pi / 2  # South Pole (-90°)
        lat_rad_north = +np.pi / 2  # North Pole (+90°)
        
        g_south = gravity_magnitude_eq6_8(lat_rad_south)
        g_north = gravity_magnitude_eq6_8(lat_rad_north)
        
        self.assertAlmostEqual(g_south, g_north, places=10,
                               msg="Gravity should be symmetric at North/South poles")
    
    def test_gravity_increases_from_equator_to_pole(self):
        """
        Test that gravity increases monotonically from equator to pole.
        
        Physical expectation:
            - Minimum at equator (strongest centrifugal force)
            - Maximum at poles (no centrifugal force)
        """
        latitudes_deg = np.array([0, 15, 30, 45, 60, 75, 90])
        latitudes_rad = np.deg2rad(latitudes_deg)
        
        gravities = [gravity_magnitude_eq6_8(lat) for lat in latitudes_rad]
        
        # Check monotonic increase
        for i in range(len(gravities) - 1):
            self.assertGreater(gravities[i + 1], gravities[i],
                               msg=f"Gravity should increase from {latitudes_deg[i]}° "
                                   f"to {latitudes_deg[i + 1]}°")
    
    def test_gravity_symmetric_north_south(self):
        """
        Test that gravity is symmetric between Northern/Southern hemispheres.
        
        g(+φ) should equal g(-φ) for any latitude φ.
        """
        test_latitudes_deg = [10, 25, 40, 55, 70, 85]
        
        for lat_deg in test_latitudes_deg:
            lat_rad_north = np.deg2rad(lat_deg)
            lat_rad_south = np.deg2rad(-lat_deg)
            
            g_north = gravity_magnitude_eq6_8(lat_rad_north)
            g_south = gravity_magnitude_eq6_8(lat_rad_south)
            
            self.assertAlmostEqual(g_north, g_south, places=10,
                                   msg=f"Gravity should be symmetric at ±{lat_deg}°")
    
    def test_gravity_variation_range(self):
        """
        Test that gravity variation is within expected WGS-84 range.
        
        Expected range: [9.78, 9.84] m/s² (approximately).
        Total variation: ~0.052 m/s² (~0.5% of g).
        """
        # Sample latitudes across full range
        latitudes_rad = np.linspace(-np.pi / 2, np.pi / 2, 100)
        gravities = [gravity_magnitude_eq6_8(lat) for lat in latitudes_rad]
        
        g_min = min(gravities)
        g_max = max(gravities)
        
        # Check bounds
        self.assertGreater(g_min, 9.77,
                           msg="Minimum gravity should be above 9.77 m/s²")
        self.assertLess(g_max, 9.85,
                        msg="Maximum gravity should be below 9.85 m/s²")
        
        # Check variation
        variation = g_max - g_min
        self.assertAlmostEqual(variation, 0.0519, places=2,
                               msg="Gravity variation should be approximately 0.052 m/s²")
    
    def test_gravity_at_typical_city_latitudes(self):
        """
        Test gravity at representative city latitudes.
        
        Validates against known approximate values for common locations.
        """
        # Tokyo, Japan: 35.6762° N
        g_tokyo = gravity_magnitude_eq6_8(np.deg2rad(35.6762))
        self.assertAlmostEqual(g_tokyo, 9.7976, places=3,
                               msg="Tokyo gravity should be ~9.798 m/s²")
        
        # New York City, USA: 40.7128° N
        g_nyc = gravity_magnitude_eq6_8(np.deg2rad(40.7128))
        self.assertAlmostEqual(g_nyc, 9.8023, places=3,
                               msg="NYC gravity should be ~9.802 m/s²")
        
        # Singapore: 1.3521° N (near equator)
        g_singapore = gravity_magnitude_eq6_8(np.deg2rad(1.3521))
        self.assertAlmostEqual(g_singapore, 9.7804, places=3,
                               msg="Singapore gravity should be ~9.780 m/s²")
        
        # London, UK: 51.5074° N
        g_london = gravity_magnitude_eq6_8(np.deg2rad(51.5074))
        self.assertAlmostEqual(g_london, 9.8117, places=3,
                               msg="London gravity should be ~9.812 m/s²")


class TestGravityMagnitudeWithFallback(unittest.TestCase):
    """
    Test suite for gravity_magnitude() with automatic fallback.
    
    Verifies backward compatibility and flexible API.
    """
    
    def test_default_fallback_when_no_latitude(self):
        """
        Test that default gravity is returned when lat_rad=None.
        
        This ensures backward compatibility with existing code.
        """
        g = gravity_magnitude(lat_rad=None, default_g=9.81)
        self.assertEqual(g, 9.81,
                         msg="Should return default_g when lat_rad is None")
    
    def test_default_fallback_with_custom_default(self):
        """
        Test custom default gravity value.
        """
        custom_g = 9.798
        g = gravity_magnitude(lat_rad=None, default_g=custom_g)
        self.assertEqual(g, custom_g,
                         msg="Should return custom default_g when lat_rad is None")
    
    def test_eq6_8_when_latitude_provided(self):
        """
        Test that Eq. (6.8) is used when latitude is provided.
        """
        lat_rad = np.deg2rad(45.0)
        g = gravity_magnitude(lat_rad=lat_rad, default_g=9.81)
        
        # Should match Eq. (6.8) result, NOT default
        g_expected = gravity_magnitude_eq6_8(lat_rad)
        self.assertAlmostEqual(g, g_expected, places=10,
                               msg="Should use Eq. (6.8) when lat_rad provided")
        self.assertNotEqual(g, 9.81,
                            msg="Should NOT return default when latitude provided")
    
    def test_default_parameter_values(self):
        """
        Test that default parameters work as expected.
        """
        g = gravity_magnitude()  # No arguments
        self.assertEqual(g, 9.81,
                         msg="Should default to 9.81 m/s² with no arguments")


class TestGravityMagnitudeFromDegrees(unittest.TestCase):
    """
    Test suite for convenience function with degree input.
    """
    
    def test_degree_conversion_at_45(self):
        """
        Test that degree input matches radian input after conversion.
        """
        lat_deg = 45.0
        lat_rad = np.deg2rad(lat_deg)
        
        g_from_deg = gravity_magnitude_from_lat_deg(lat_deg)
        g_from_rad = gravity_magnitude_eq6_8(lat_rad)
        
        self.assertAlmostEqual(g_from_deg, g_from_rad, places=10,
                               msg="Degree and radian versions should match")
    
    def test_degree_conversion_at_multiple_latitudes(self):
        """
        Test degree conversion at various latitudes.
        """
        test_latitudes_deg = [0, 15, 30, 45, 60, 75, 90]
        
        for lat_deg in test_latitudes_deg:
            lat_rad = np.deg2rad(lat_deg)
            
            g_from_deg = gravity_magnitude_from_lat_deg(lat_deg)
            g_from_rad = gravity_magnitude_eq6_8(lat_rad)
            
            self.assertAlmostEqual(g_from_deg, g_from_rad, places=10,
                                   msg=f"Degree/radian mismatch at {lat_deg}°")


class TestGravityEdgeCases(unittest.TestCase):
    """
    Test suite for edge cases and numerical stability.
    """
    
    def test_very_small_latitude(self):
        """
        Test that very small latitudes (near zero) work correctly.
        """
        lat_rad = 1e-10  # Very close to equator
        g = gravity_magnitude_eq6_8(lat_rad)
        
        # Should be very close to equator value (9.7803)
        self.assertAlmostEqual(g, 9.7803, places=4,
                               msg="Near-zero latitude should give equator gravity")
    
    def test_negative_latitudes(self):
        """
        Test that negative latitudes (Southern hemisphere) work correctly.
        """
        lat_deg = -35.5  # Example: Southern hemisphere
        g = gravity_magnitude_from_lat_deg(lat_deg)
        
        # Should be in valid range
        self.assertGreater(g, 9.77)
        self.assertLess(g, 9.85)
    
    def test_large_latitude_array(self):
        """
        Test that function works with array-like inputs (vectorized).
        """
        latitudes_deg = np.linspace(-90, 90, 1000)
        
        # Should work without errors
        gravities = [gravity_magnitude_from_lat_deg(lat) for lat in latitudes_deg]
        
        self.assertEqual(len(gravities), 1000,
                         msg="Should handle array-like inputs")
        self.assertTrue(all(9.77 < g < 9.85 for g in gravities),
                        msg="All gravity values should be in valid range")


class TestGravityIntegrationWithCh6Algorithms(unittest.TestCase):
    """
    Integration tests simulating Chapter 6 algorithm usage.
    """
    
    def test_strapdown_propagation_use_case(self):
        """
        Simulate strapdown propagation use case.
        
        Test that gravity magnitude can be computed and used in
        typical strapdown integration workflow.
        """
        # Scenario: IMU at 40° North latitude
        lat_deg = 40.0
        lat_rad = np.deg2rad(lat_deg)
        
        # Compute gravity magnitude using Eq. (6.8)
        g_mag = gravity_magnitude(lat_rad=lat_rad, default_g=9.81)
        
        # Verify it's book-accurate (not default)
        self.assertNotEqual(g_mag, 9.81)
        self.assertAlmostEqual(g_mag, 9.8017, places=3)
    
    def test_pdr_gravity_removal_use_case(self):
        """
        Simulate PDR gravity removal use case.
        
        Test Eq. (6.47) usage: a_dynamic = a_mag - g
        """
        lat_rad = np.deg2rad(35.0)
        g_mag = gravity_magnitude(lat_rad=lat_rad)
        
        # Simulate accelerometer magnitude during walking
        a_mag = 11.5  # m/s² (includes motion + gravity)
        
        # Remove gravity (Eq. 6.47)
        a_dynamic = a_mag - g_mag
        
        # Dynamic component should be reasonable for walking (~1-2 m/s²)
        self.assertGreater(a_dynamic, 1.0)
        self.assertLess(a_dynamic, 3.0)
    
    def test_backward_compatibility_no_latitude(self):
        """
        Test that old code without latitude still works.
        
        Ensures no breaking changes to existing Ch6 examples.
        """
        # Old code path: no latitude provided
        g_old = gravity_magnitude(lat_rad=None, default_g=9.81)
        
        # Should return exactly 9.81 (backward compatible)
        self.assertEqual(g_old, 9.81)
    
    def test_new_code_with_latitude(self):
        """
        Test that new code with latitude uses Eq. (6.8).
        
        Validates book-accurate path for updated examples.
        """
        # New code path: latitude provided
        lat_deg = 45.0
        lat_rad = np.deg2rad(lat_deg)
        g_new = gravity_magnitude(lat_rad=lat_rad, default_g=9.81)
        
        # Should use Eq. (6.8), not default
        g_expected = gravity_magnitude_eq6_8(lat_rad)
        self.assertAlmostEqual(g_new, g_expected, places=10)
        self.assertNotEqual(g_new, 9.81)


if __name__ == "__main__":
    unittest.main()

