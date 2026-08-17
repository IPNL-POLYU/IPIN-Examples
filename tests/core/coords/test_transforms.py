"""Unit tests for coordinate transformations (LLH, ECEF, ENU).

This module tests the transformation functions between geodetic (LLH),
Earth-Centered Earth-Fixed (ECEF), and local East-North-Up (ENU)
coordinate systems.

Test cases include:
- Round-trip transformations (LLH -> ECEF -> LLH)
- Round-trip transformations (ECEF -> ENU -> ECEF)
- Known reference points (e.g., equator, poles, Greenwich)
- Numerical accuracy and edge cases

Reference: Chapter 2, Section 2.1 - Coordinate Systems and Transformations
"""

import unittest

import numpy as np

from core.coords.transforms import (
    body_to_enu,
    body_to_map,
    ecef_to_enu,
    ecef_to_llh,
    enu_to_body,
    enu_to_ecef,
    enu_to_llh_offset,
    enu_to_ned,
    llh_to_ecef,
    map_to_body,
    ned_to_enu,
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


class TestMapBody(unittest.TestCase):
    """Test cases for local map <-> local body transforms (Eq. (2.3))."""

    def test_pure_yaw_known_value(self) -> None:
        """A 90 deg yaw maps map-frame +X to body-frame -Y (passive)."""
        x_body = map_to_body(np.array([1.0, 0.0, 0.0]), np.pi / 2.0)
        np.testing.assert_allclose(x_body, [0.0, -1.0, 0.0], atol=1e-9)

    def test_translation_offset(self) -> None:
        """Body origin offset in the map frame is subtracted before rotating."""
        x_map = np.array([3.0, 1.0, 2.0])
        origin = np.array([1.0, 1.0, 0.0])
        x_body = map_to_body(x_map, 0.0, origin)
        np.testing.assert_allclose(x_body, [2.0, 0.0, 2.0], atol=1e-9)

    def test_round_trip(self) -> None:
        """map -> body -> map recovers the original point."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            x_map = rng.uniform(-50.0, 50.0, 3)
            yaw = rng.uniform(-np.pi, np.pi)
            origin = rng.uniform(-10.0, 10.0, 3)
            x_body = map_to_body(x_map, yaw, origin)
            recovered = body_to_map(x_body, yaw, origin)
            np.testing.assert_allclose(recovered, x_map, atol=1e-9)


class TestEnuNed(unittest.TestCase):
    """Test cases for ENU <-> NED transforms (Eq. (2.5))."""

    def test_known_value(self) -> None:
        """[E, N, U] -> [N, E, -U]."""
        np.testing.assert_allclose(
            enu_to_ned(np.array([1.0, 2.0, 3.0])), [2.0, 1.0, -3.0], atol=1e-12
        )

    def test_self_inverse(self) -> None:
        """The ENU<->NED matrix is its own inverse."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            v = rng.uniform(-100.0, 100.0, 3)
            np.testing.assert_allclose(ned_to_enu(enu_to_ned(v)), v, atol=1e-12)


class TestEnuBody(unittest.TestCase):
    """Test cases for ENU <-> local body transforms (Eqs. (2.6), (2.7))."""

    def test_identity_attitude(self) -> None:
        """Zero attitude with no offset is the identity map."""
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(enu_to_body(v, 0.0, 0.0, 0.0), v, atol=1e-9)

    def test_round_trip(self) -> None:
        """enu -> body -> enu recovers the original point (coincident origins)."""
        rng = np.random.default_rng(2)
        for _ in range(20):
            x_enu = rng.uniform(-50.0, 50.0, 3)
            roll, pitch, yaw = rng.uniform(-np.pi, np.pi, 3)
            x_body = enu_to_body(x_enu, roll, pitch, yaw)
            recovered = body_to_enu(x_body, roll, pitch, yaw)
            np.testing.assert_allclose(recovered, x_enu, atol=1e-9)


class TestENUToLLHOffset(unittest.TestCase):
    """A displacement in metres becomes the right offset in radians.

    This function exists because the conversion was twice done by hand with a
    "metres per degree" constant and twice got the units wrong: an example
    printed 6405.80 m as "100m East", and the Ch2 dataset generator sampled a
    declared 50 m footprint across 2.7 km. Both were a per-degree value added
    to a radian coordinate. Taking metres in and returning radians removes the
    quantity that was being mislaid, so these tests check the contract in the
    units the caller uses: metres in, metres back out through the real
    transform.
    """

    LAT = np.deg2rad(37.7749)
    LON = np.deg2rad(-122.4194)

    def _round_trip(self, east: float, north: float) -> np.ndarray:
        """Offset by (east, north) metres and read the ENU back."""
        dlat, dlon = enu_to_llh_offset(east, north, self.LAT)
        xyz = llh_to_ecef(self.LAT + dlat, self.LON + dlon, 0.0)
        return ecef_to_enu(*xyz, self.LAT, self.LON, 0.0)

    def test_one_hundred_metres_east_comes_back_as_one_hundred(self) -> None:
        """The headline claim, in the units of the argument."""
        enu = self._round_trip(100.0, 0.0)
        self.assertAlmostEqual(enu[0], 100.0, delta=0.01)
        self.assertAlmostEqual(enu[1], 0.0, delta=0.01)

    def test_one_hundred_metres_north_comes_back_as_one_hundred(self) -> None:
        enu = self._round_trip(0.0, 100.0)
        self.assertAlmostEqual(enu[0], 0.0, delta=0.01)
        self.assertAlmostEqual(enu[1], 100.0, delta=0.01)

    def test_the_neglected_term_is_second_order(self) -> None:
        """The residual scales as the square of the offset, and only so.

        This is what distinguishes an honest first-order approximation from a
        scale error. A wrong constant would give a residual growing linearly;
        the curvature term grows as d^2, so the error per metre must grow by
        10x for each 10x of distance.
        """
        errors = {}
        for distance in (10.0, 100.0, 1000.0):
            enu = self._round_trip(distance, 0.0)
            errors[distance] = abs(np.linalg.norm(enu - [distance, 0.0, 0.0]))

        # 100x the distance, 10000x the error -- with room for the fact that
        # the east residual mixes the drop with the parallel-vs-normal-section
        # difference.
        ratio = errors[1000.0] / errors[10.0]
        self.assertGreater(ratio, 5e3)
        self.assertLess(ratio, 2e4)

    def test_a_degree_sized_offset_is_not_produced(self) -> None:
        """Guard the actual defect: the result must be radian-scaled.

        100 m of latitude is 1.57e-5 rad. The bug produced 9.0e-4 -- the same
        number read as degrees. Anything above 1e-4 rad for a 100 m offset is
        that mistake, whatever produced it.
        """
        dlat, dlon = enu_to_llh_offset(100.0, 100.0, self.LAT)
        self.assertLess(abs(dlat), 1e-4)
        self.assertLess(abs(dlon), 1e-4)

    def test_a_pole_is_refused_rather_than_answered(self) -> None:
        """Near the pole the linearisation stops meaning anything, and says so.

        `cos(pi/2)` is 6.1e-17 rather than 0, so this divides by something
        tiny-but-finite and raises no warning: before the guard,
        `enu_to_llh_offset(100.0, 0.0, pi/2)` returned 2.55e+11 rad. A silent
        absurdity is worse than an exception, so it is now an exception.
        """
        with self.assertRaises(ValueError):
            enu_to_llh_offset(100.0, 0.0, np.pi / 2)
        with self.assertRaises(ValueError):
            enu_to_llh_offset(100.0, 0.0, -np.pi / 2)

    def test_high_latitudes_still_work(self) -> None:
        """The guard must not fire on anywhere anyone actually positions.

        Svalbard is about as far north as an indoor-positioning deployment
        gets. A check that also refused legitimate input would be worse than
        the silent number it replaced.

        The tolerance is derived rather than picked, because a flat one is
        wrong here: a parallel curves toward the pole, so an eastward offset
        picks up a northward residual of about ``d^2 tan(lat) / 2R``. That is
        1.4 mm at 60 deg but 44.8 mm at 89 deg, and the measured miss tracks it
        to within a few percent. A fixed 50 mm delta would have passed 89 deg
        for the wrong reason and failed at 89.5.
        """
        for lat_deg in (60.0, 78.2, 89.0):
            with self.subTest(lat=lat_deg):
                lat = np.deg2rad(lat_deg)
                dlat, dlon = enu_to_llh_offset(100.0, 100.0, lat)
                xyz = llh_to_ecef(lat + dlat, dlon, 0.0)
                enu = ecef_to_enu(*xyz, lat, 0.0, 0.0)

                predicted = 100.0**2 * np.tan(lat) / (2 * 6.4e6)
                tolerance = 2.0 * predicted + 0.005
                self.assertAlmostEqual(enu[0], 100.0, delta=tolerance)
                self.assertAlmostEqual(enu[1], 100.0, delta=tolerance)

    def test_longitude_scaling_follows_latitude(self) -> None:
        """cos(lat) is applied to east, and is not silently constant.

        The borrowed 78800 was correct at 45 deg and 12% wrong at 37.77, which
        is the failure a fixed constant guarantees. A degree of longitude must
        shrink as latitude rises.
        """
        equator = enu_to_llh_offset(100.0, 0.0, 0.0)[1]
        mid = enu_to_llh_offset(100.0, 0.0, np.deg2rad(45.0))[1]
        high = enu_to_llh_offset(100.0, 0.0, np.deg2rad(60.0))[1]

        self.assertLess(equator, mid)
        self.assertLess(mid, high)
        # At 60 deg, cos(lat) = 0.5, so the same metres need ~2x the longitude.
        self.assertAlmostEqual(high / equator, 2.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
