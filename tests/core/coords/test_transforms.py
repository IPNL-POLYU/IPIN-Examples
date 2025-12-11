"""Unit tests for coordinate transformations (LLH, ECEF, ENU).

This module tests the transformation functions between geodetic (LLH),
Earth-Centered Earth-Fixed (ECEF), and local East-North-Up (ENU)
coordinate systems.

Test cases include:
- Round-trip transformations (LLH -> ECEF -> LLH)
- Round-trip transformations (ECEF -> ENU -> ECEF)
- Known reference points (e.g., equator, poles, Greenwich)
- Numerical accuracy and edge cases

Reference: Chapter 2, Section 2.3 - Coordinate Transformations
"""

import unittest

import numpy as np

from core.coords.transforms import (
    ecef_to_enu,
    ecef_to_llh,
    enu_to_ecef,
    llh_to_ecef,
)


class TestLLHtoECEF(unittest.TestCase):
    """Test cases for LLH to ECEF transformation."""

    def test_equator_prime_meridian(self) -> None:
        """Test conversion at equator and prime meridian (0°N, 0°E)."""
        lat = 0.0
        lon = 0.0
        height = 0.0

        xyz = llh_to_ecef(lat, lon, height)

        # At equator and prime meridian: x ≈ 6378137m, y ≈ 0, z ≈ 0
        expected = np.array([6378137.0, 0.0, 0.0])
        np.testing.assert_allclose(xyz, expected, rtol=1e-9)

    def test_north_pole(self) -> None:
        """Test conversion at North Pole (90°N)."""
        lat = np.pi / 2.0  # 90°N
        lon = 0.0
        height = 0.0

        xyz = llh_to_ecef(lat, lon, height)

        # At North Pole: x ≈ 0, y ≈ 0, z ≈ 6356752m (semi-minor axis)
        expected = np.array([0.0, 0.0, 6356752.314245])
        np.testing.assert_allclose(xyz, expected, atol=1e-6)

    def test_south_pole(self) -> None:
        """Test conversion at South Pole (90°S)."""
        lat = -np.pi / 2.0  # 90°S
        lon = 0.0
        height = 0.0

        xyz = llh_to_ecef(lat, lon, height)

        # At South Pole: x ≈ 0, y ≈ 0, z ≈ -6356752m
        expected = np.array([0.0, 0.0, -6356752.314245])
        np.testing.assert_allclose(xyz, expected, atol=1e-6)

    def test_greenwich_observatory(self) -> None:
        """Test conversion at Greenwich Observatory (51.4769°N, 0°E)."""
        lat = np.deg2rad(51.4769)
        lon = np.deg2rad(0.0)
        height = 0.0

        xyz = llh_to_ecef(lat, lon, height)

        # Expected ECEF coordinates (approximate, values may vary by reference)
        # Main check: ensure coordinates are in the right ballpark
        self.assertGreater(xyz[0], 3980000.0)
        self.assertLess(xyz[0], 3981000.0)
        self.assertAlmostEqual(xyz[1], 0.0, delta=1.0)
        self.assertGreater(xyz[2], 4966000.0)
        self.assertLess(xyz[2], 4967000.0)

    def test_with_height(self) -> None:
        """Test conversion with non-zero height."""
        lat = np.deg2rad(45.0)
        lon = np.deg2rad(90.0)
        height = 1000.0  # 1 km above ellipsoid

        xyz = llh_to_ecef(lat, lon, height)

        # Check that height increases the magnitude
        xyz_zero_height = llh_to_ecef(lat, lon, 0.0)
        distance = np.linalg.norm(xyz)
        distance_zero = np.linalg.norm(xyz_zero_height)

        self.assertAlmostEqual(distance - distance_zero, height, delta=0.1)


class TestECEFtoLLH(unittest.TestCase):
    """Test cases for ECEF to LLH transformation."""

    def test_equator_prime_meridian(self) -> None:
        """Test conversion at equator and prime meridian."""
        x = 6378137.0
        y = 0.0
        z = 0.0

        llh = ecef_to_llh(x, y, z)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(llh, expected, atol=1e-6)

    def test_north_pole(self) -> None:
        """Test conversion at North Pole."""
        x = 0.0
        y = 0.0
        z = 6356752.314245

        llh = ecef_to_llh(x, y, z)

        expected = np.array([np.pi / 2.0, 0.0, 0.0])
        np.testing.assert_allclose(llh[:2], expected[:2], atol=1e-6)
        self.assertAlmostEqual(llh[2], 0.0, delta=0.01)

    def test_arbitrary_point(self) -> None:
        """Test conversion at arbitrary point (45°N, 90°E, 100m)."""
        lat = np.deg2rad(45.0)
        lon = np.deg2rad(90.0)
        height = 100.0

        # Forward transform
        xyz = llh_to_ecef(lat, lon, height)

        # Backward transform
        llh_result = ecef_to_llh(*xyz)

        np.testing.assert_allclose(llh_result[0], lat, rtol=1e-9)
        np.testing.assert_allclose(llh_result[1], lon, rtol=1e-9)
        np.testing.assert_allclose(llh_result[2], height, atol=1e-3)


class TestRoundTripLLHECEF(unittest.TestCase):
    """Test round-trip transformations between LLH and ECEF."""

    def test_round_trip_multiple_points(self) -> None:
        """Test LLH -> ECEF -> LLH for multiple points."""
        test_points = [
            (0.0, 0.0, 0.0),  # Equator, prime meridian
            (np.deg2rad(45.0), np.deg2rad(90.0), 100.0),  # Mid-latitude
            (np.deg2rad(-30.0), np.deg2rad(-120.0), 500.0),  # Southern hemisphere
            (np.deg2rad(51.4769), np.deg2rad(0.0), 0.0),  # Greenwich
            (np.deg2rad(89.0), np.deg2rad(180.0), 1000.0),  # Near North Pole
        ]

        for lat, lon, height in test_points:
            with self.subTest(lat=lat, lon=lon, height=height):
                # Forward: LLH -> ECEF
                xyz = llh_to_ecef(lat, lon, height)

                # Backward: ECEF -> LLH
                llh_result = ecef_to_llh(*xyz)

                # Check round-trip accuracy
                np.testing.assert_allclose(llh_result[0], lat, rtol=1e-9)
                np.testing.assert_allclose(llh_result[1], lon, rtol=1e-9)
                np.testing.assert_allclose(llh_result[2], height, atol=1e-3)


class TestECEFtoENU(unittest.TestCase):
    """Test cases for ECEF to ENU transformation."""

    def test_reference_point_is_origin(self) -> None:
        """Test that reference point maps to ENU origin."""
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(90.0)
        height_ref = 0.0

        # Reference point in ECEF
        xyz_ref = llh_to_ecef(lat_ref, lon_ref, height_ref)

        # Transform reference point to ENU
        enu = ecef_to_enu(*xyz_ref, lat_ref, lon_ref, height_ref)

        # Should be at origin
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(enu, expected, atol=1e-6)

    def test_point_east_of_reference(self) -> None:
        """Test point 100m east of reference."""
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(0.0)
        height_ref = 0.0

        # Point slightly east (small longitude increase)
        # At 45° latitude, 1° longitude ≈ 78.8 km
        delta_lon = 100.0 / 78800.0  # Approximately 100m east
        lon_target = lon_ref + np.deg2rad(delta_lon)

        xyz_target = llh_to_ecef(lat_ref, lon_target, height_ref)
        enu = ecef_to_enu(*xyz_target, lat_ref, lon_ref, height_ref)

        # Should have positive east component, near-zero north and up
        self.assertGreater(enu[0], 90.0)  # Approximately 100m
        self.assertLess(enu[0], 110.0)
        self.assertAlmostEqual(enu[1], 0.0, delta=1.0)
        self.assertAlmostEqual(enu[2], 0.0, delta=1.0)

    def test_point_north_of_reference(self) -> None:
        """Test point 100m north of reference."""
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(0.0)
        height_ref = 0.0

        # Point slightly north (small latitude increase)
        # 1° latitude ≈ 111 km
        delta_lat = 100.0 / 111000.0  # Approximately 100m north
        lat_target = lat_ref + np.deg2rad(delta_lat)

        xyz_target = llh_to_ecef(lat_target, lon_ref, height_ref)
        enu = ecef_to_enu(*xyz_target, lat_ref, lon_ref, height_ref)

        # Should have positive north component, near-zero east and up
        self.assertAlmostEqual(enu[0], 0.0, delta=1.0)
        self.assertGreater(enu[1], 90.0)  # Approximately 100m
        self.assertLess(enu[1], 110.0)
        self.assertAlmostEqual(enu[2], 0.0, delta=1.0)

    def test_point_above_reference(self) -> None:
        """Test point 100m above reference."""
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(90.0)
        height_ref = 0.0

        # Point 100m above reference
        xyz_target = llh_to_ecef(lat_ref, lon_ref, 100.0)
        enu = ecef_to_enu(*xyz_target, lat_ref, lon_ref, height_ref)

        # Should have positive up component, near-zero east and north
        self.assertAlmostEqual(enu[0], 0.0, delta=1.0)
        self.assertAlmostEqual(enu[1], 0.0, delta=1.0)
        self.assertAlmostEqual(enu[2], 100.0, delta=0.1)


class TestENUtoECEF(unittest.TestCase):
    """Test cases for ENU to ECEF transformation."""

    def test_enu_origin_to_ecef(self) -> None:
        """Test that ENU origin maps to reference ECEF point."""
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(90.0)
        height_ref = 100.0

        # ENU origin
        enu = np.array([0.0, 0.0, 0.0])

        # Transform to ECEF
        xyz = enu_to_ecef(*enu, lat_ref, lon_ref, height_ref)

        # Should equal reference point
        xyz_ref = llh_to_ecef(lat_ref, lon_ref, height_ref)
        np.testing.assert_allclose(xyz, xyz_ref, rtol=1e-9)

    def test_enu_displacement(self) -> None:
        """Test ENU displacement to ECEF."""
        lat_ref = np.deg2rad(0.0)  # Equator for simplicity
        lon_ref = np.deg2rad(0.0)
        height_ref = 0.0

        # 100m east, 200m north, 50m up
        enu = np.array([100.0, 200.0, 50.0])

        # Transform to ECEF
        xyz = enu_to_ecef(*enu, lat_ref, lon_ref, height_ref)

        # Reference point in ECEF
        xyz_ref = llh_to_ecef(lat_ref, lon_ref, height_ref)

        # Check that displacement makes sense
        displacement = xyz - xyz_ref
        self.assertGreater(np.linalg.norm(displacement), 0.0)


class TestRoundTripECEFENU(unittest.TestCase):
    """Test round-trip transformations between ECEF and ENU."""

    def test_round_trip_multiple_points(self) -> None:
        """Test ECEF -> ENU -> ECEF for multiple points."""
        # Reference point
        lat_ref = np.deg2rad(45.0)
        lon_ref = np.deg2rad(90.0)
        height_ref = 0.0

        # Test points in ECEF
        test_points_llh = [
            (lat_ref, lon_ref, height_ref),  # Reference point
            (np.deg2rad(45.01), lon_ref, height_ref),  # North
            (lat_ref, np.deg2rad(90.01), height_ref),  # East
            (lat_ref, lon_ref, 100.0),  # Up
            (np.deg2rad(45.01), np.deg2rad(90.01), 50.0),  # Combined
        ]

        for lat, lon, height in test_points_llh:
            with self.subTest(lat=lat, lon=lon, height=height):
                # Convert to ECEF
                xyz = llh_to_ecef(lat, lon, height)

                # Forward: ECEF -> ENU
                enu = ecef_to_enu(*xyz, lat_ref, lon_ref, height_ref)

                # Backward: ENU -> ECEF
                xyz_result = enu_to_ecef(*enu, lat_ref, lon_ref, height_ref)

                # Check round-trip accuracy
                np.testing.assert_allclose(xyz_result, xyz, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
