"""
Unit tests for core/sensors/environment.py (magnetometer and barometer).

Tests cover:
    - Magnetometer tilt compensation (Eq. 6.52)
    - Magnetometer heading (Eqs. 6.51-6.53)
    - Barometric altitude (Eq. 6.54)
    - Floor change detection
    - Measurement smoothing
    - Hard-iron calibration
    - Edge cases and validation

Run with: pytest tests/test_sensors_environment.py -v
"""

import unittest
import numpy as np
import pytest

from core.sensors.environment import (
    earth_field_body,
    earth_field_map,
    mag_tilt_compensate,
    mag_heading,
    pressure_to_altitude,
    detect_floor_change,
    smooth_measurement_simple,
    compensate_hard_iron,
)
from core.sensors.types import FrameConvention


class TestMagTiltCompensate(unittest.TestCase):
    """Test suite for magnetometer tilt compensation (Eq. 6.52)."""

    def test_mag_tilt_compensate_level(self) -> None:
        """Test tilt compensation for level device (no tilt)."""
        mag = np.array([20.0, 0.0, -40.0])  # pointing north and down
        roll = 0.0
        pitch = 0.0

        mag_comp = mag_tilt_compensate(mag, roll, pitch)

        # Should be unchanged when level
        np.testing.assert_array_almost_equal(mag_comp, mag)

    def test_mag_tilt_compensate_roll_only(self) -> None:
        """Test tilt compensation with roll only."""
        mag = np.array([20.0, 0.0, -40.0])
        roll = np.pi / 4  # 45° roll
        pitch = 0.0

        mag_comp = mag_tilt_compensate(mag, roll, pitch)

        # After roll compensation, y and z should be affected
        assert not np.allclose(mag_comp, mag)
        # x-component should be unchanged (roll is about x-axis)
        assert np.isclose(mag_comp[0], mag[0])

    def test_mag_tilt_compensate_pitch_only(self) -> None:
        """Test tilt compensation with pitch only."""
        mag = np.array([20.0, 0.0, -40.0])
        roll = 0.0
        pitch = np.pi / 6  # 30° pitch

        mag_comp = mag_tilt_compensate(mag, roll, pitch)

        # After pitch compensation, x and z should be affected
        assert not np.allclose(mag_comp, mag)
        # y-component should be unchanged (pitch is about y-axis)
        assert np.isclose(mag_comp[1], mag[1])

    def test_mag_tilt_compensate_both_angles(self) -> None:
        """Test tilt compensation with both roll and pitch."""
        mag = np.array([20.0, 10.0, -40.0])
        roll = np.deg2rad(15)
        pitch = np.deg2rad(20)

        mag_comp = mag_tilt_compensate(mag, roll, pitch)

        # Should be different from input
        assert not np.allclose(mag_comp, mag)
        # Magnitude should be preserved (rotation is orthogonal)
        mag_original = np.linalg.norm(mag)
        mag_compensated = np.linalg.norm(mag_comp)
        assert np.isclose(mag_original, mag_compensated, atol=0.01)

    def test_mag_tilt_compensate_matches_book_eq_6_52(self) -> None:
        """Tilt compensation must equal the closed form of book Eq. (6.52).

        Written out componentwise here so the code and the book can be read
        side by side. The composition is R_y(pitch) R_x(roll), which inverts
        the roll/pitch part of C_B^M = R_z R_y R_x; the other order agrees
        with it whenever roll or pitch is zero, so the two-angle case below is
        the one that discriminates.
        """
        mag = np.array([20.0, 10.0, -40.0])
        roll, pitch = 0.2, 0.3
        mx, my, mz = mag
        ct, st = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        Mx = mx * ct + my * sr * st + mz * cr * st
        My = my * cr - mz * sr
        out = mag_tilt_compensate(mag, roll, pitch)
        np.testing.assert_allclose(out[:2], [Mx, My], atol=1e-12)

    def test_the_two_tilt_orders_are_not_the_same_function(self) -> None:
        """Guard the guard: the closed form above must be discriminating.

        `Rx(roll) Ry(pitch)` -- the composition this module used to carry --
        agrees with the implemented one to machine precision whenever roll or
        pitch is zero. Every level test therefore passes under both, which is
        how the wrong order survived. Measured here at 0.2/0.3 rad so the test
        above cannot silently become a tautology.
        """
        mag = np.array([20.0, 10.0, -40.0])
        roll, pitch = 0.2, 0.3
        mx, my, mz = mag
        ct, st = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        other_order = np.array(
            [mx * ct + mz * st, my * cr + mx * st * sr - mz * ct * sr]
        )
        implemented = mag_tilt_compensate(mag, roll, pitch)[:2]
        # Measured gap at this attitude: 0.82 uT on Mx, 0.82 on My, against a
        # field of 20/10/-40. The bound is half that -- low enough not to be
        # brittle, high enough that it cannot be met by rounding.
        assert np.abs(implemented - other_order).max() > 0.4, (
            "the two rotation orders agree here, so the Eq. (6.52) test above "
            "no longer distinguishes them"
        )

    def test_mag_tilt_compensate_invalid_shape(self) -> None:
        """Test that invalid mag shape raises error."""
        mag_bad = np.array([20.0, 10.0])  # Wrong: should be (3,)

        with pytest.raises(ValueError, match="must have shape"):
            mag_tilt_compensate(mag_bad, 0.0, 0.0)


class TestMagHeading(unittest.TestCase):
    """Test suite for magnetometer heading (Eqs. 6.51-6.53)."""

    # The default frame is ENU, so mag_b is ordered [East, North, Up] and the
    # returned heading is measured from East toward North.
    #
    # These four cases name what the PLATFORM is doing, not where the field
    # points, and the distinction is the whole of the compass-vs-ENU defect.
    # mag_b is a field reading in the body frame; the field itself is fixed
    # and points at magnetic north. So a reading that lands on the body +x
    # (forward) axis means the platform is facing north -- heading pi/2 in ENU
    # -- and a reading on the body +y (left) axis means it faces east. They
    # used to be labelled the other way round, which is exactly the reflection.

    def test_mag_heading_field_on_the_forward_axis_means_facing_north(self) -> None:
        """Reading along body +x: the platform is looking at magnetic north."""
        mag = np.array([20.0, 0.0, -40.0])

        heading = mag_heading(mag, 0.0, 0.0)

        assert np.isclose(heading, np.pi / 2, atol=0.01)

    def test_mag_heading_field_on_the_left_axis_means_facing_east(self) -> None:
        """Reading along body +y (left): north is to the left, so face east."""
        mag = np.array([0.0, 20.0, -40.0])

        heading = mag_heading(mag, 0.0, 0.0)

        assert np.isclose(heading, 0.0, atol=0.01)

    def test_mag_heading_field_on_the_right_axis_means_facing_west(self) -> None:
        """Reading along body -y (right): north is to the right, face west."""
        mag = np.array([0.0, -20.0, -40.0])

        heading = mag_heading(mag, 0.0, 0.0)

        assert np.isclose(abs(heading), np.pi, atol=0.01)

    def test_mag_heading_field_on_the_back_axis_means_facing_south(self) -> None:
        """Reading along body -x: north is behind, so the platform faces south."""
        mag = np.array([-20.0, 0.0, -40.0])

        heading = mag_heading(mag, 0.0, 0.0)

        assert np.isclose(heading, -np.pi / 2, atol=0.01)

    def test_the_frame_argument_changes_the_answer(self) -> None:
        """`frame` is load-bearing, and this is where it shows.

        It used to accept `create_ned()` and return the identical number,
        because the only thing it selected was an atan2 that reads the same in
        both conventions. It now also selects where magnetic north *sits* --
        heading pi/2 in ENU, heading 0 in NED -- so the same three numbers,
        read under the two frames, describe platforms a quarter turn apart.
        Neither answer is wrong: the numbers mean different things.
        """
        mag = np.array([3.0, 4.0, -40.0])

        enu = mag_heading(mag, 0.0, 0.0, frame=FrameConvention.create_enu())
        ned = mag_heading(mag, 0.0, 0.0, frame=FrameConvention.create_ned())

        gap = np.arctan2(np.sin(enu - ned), np.cos(enu - ned))
        assert np.isclose(abs(gap), np.pi / 2)

    def test_mag_heading_inverts_the_field_the_platform_is_sitting_in(self) -> None:
        """The round trip that matters: attitude -> reading -> attitude.

        The previous version round-tripped a *direction* through
        `heading_to_unit_vector` and `mag_heading`, which passed under the
        compass readout and the ENU one alike -- it pinned that the two halves
        were mutual inverses without ever asking whether the pair was the right
        convention. Starting from a fixed Earth field and a platform attitude
        cannot be satisfied that way.
        """
        for frame in (FrameConvention.create_enu(), FrameConvention.create_ned()):
            field_map = earth_field_map(frame=frame)
            for psi in np.linspace(-np.pi, np.pi, 17)[:-1]:  # -pi and pi alias
                mag = earth_field_body(0.0, 0.0, psi, field_map=field_map)

                recovered = mag_heading(mag, 0.0, 0.0, frame=frame)

                # +pi and -pi are the same heading; compare the wrapped gap.
                gap = np.arctan2(np.sin(recovered - psi), np.cos(recovered - psi))
                assert np.isclose(
                    gap, 0.0, atol=1e-9
                ), f"{frame.map_frame}: {psi} -> {mag} -> {recovered}"

    def test_mag_heading_with_declination(self) -> None:
        """East declination SUBTRACTS in ENU, because ENU runs the other way.

        "True = magnetic + east declination" is the compass rule, for bearings
        measured clockwise from north. ENU heading is counter-clockwise from
        east, so the same physical correction reverses sign. The frame owns it
        (`FrameConvention.magnetic_north_heading`) and this pins the result.
        """
        mag = np.array([20.0, 0.0, -40.0])  # north is dead ahead
        declination = np.deg2rad(10)  # 10 deg east declination

        heading = mag_heading(mag, 0.0, 0.0, declination)

        assert np.isclose(heading, np.pi / 2 - np.deg2rad(10), atol=0.01)

    def test_mag_heading_tilted_device(self) -> None:
        """Test heading computation with tilted device."""
        mag = np.array([20.0, 0.0, -40.0])
        roll = np.deg2rad(30)
        pitch = np.deg2rad(20)

        heading = mag_heading(mag, roll, pitch)

        # Should still compute heading (after tilt compensation)
        assert -np.pi <= heading <= np.pi

    def test_mag_heading_tilt_invariant(self) -> None:
        """Heading must be invariant to roll/pitch (the point of Eq. 6.52).

        The platform holds one heading and is tilted six ways; the recovered
        heading must not move. The forward model is `earth_field_body`, the
        transpose of C_B^M = Rz Ry Rx, so this is a statement about the
        chapter's attitude convention rather than about a rotation invented
        here to match the implementation.
        """
        target = np.deg2rad(37.0)
        for roll, pitch in [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.0, 0.4),
            (0.3, -0.4),
            (-0.5, 0.6),
            (0.7, 0.7),
        ]:
            mag_b = earth_field_body(roll, pitch, target)
            heading = mag_heading(mag_b, roll, pitch)
            assert np.isclose(
                heading, target, atol=1e-9
            ), f"heading not tilt-invariant at roll={roll}, pitch={pitch}"

    def test_mag_heading_wrapping(self) -> None:
        """Test that heading is wrapped to [-π, π]."""
        mag = np.array([20.0, 0.0, -40.0])
        roll = 0.0
        pitch = 0.0
        declination = np.deg2rad(180)  # large declination

        heading = mag_heading(mag, roll, pitch, declination)

        # Should be wrapped
        assert -np.pi <= heading <= np.pi


class TestPressureToAltitude(unittest.TestCase):
    """Test suite for barometric altitude (Eq. 6.54)."""

    def test_pressure_to_altitude_at_reference(self) -> None:
        """Test altitude is zero when pressure equals reference."""
        p = 101325.0  # Pa (sea level)
        p0 = 101325.0
        T = 288.15  # K (15°C)

        h = pressure_to_altitude(p, p0, T)

        assert np.isclose(h, 0.0, atol=0.01)

    def test_pressure_to_altitude_above_reference(self) -> None:
        """Test positive altitude for lower pressure."""
        p0 = 101325.0
        p = 101325.0 - 120.0  # ~10m higher (pressure drops ~12 Pa/m)
        T = 288.15

        h = pressure_to_altitude(p, p0, T)

        # Should be positive (above reference)
        assert h > 0
        # Should be roughly 10 m
        assert 8 < h < 12

    def test_pressure_to_altitude_below_reference(self) -> None:
        """Test negative altitude for higher pressure."""
        p0 = 101325.0
        p = 101325.0 + 120.0  # ~10m lower
        T = 288.15

        h = pressure_to_altitude(p, p0, T)

        # Should be negative (below reference)
        assert h < 0
        # Should be roughly -10 m
        assert -12 < h < -8

    def test_pressure_to_altitude_one_floor(self) -> None:
        """Test altitude change for one floor (~3m)."""
        p0 = 101325.0
        p_floor1 = p0 - 36  # 3m higher (12 Pa/m × 3m)
        T = 288.15

        h = pressure_to_altitude(p_floor1, p0, T)

        # Should be around 3 m
        assert 2.5 < h < 3.5

    def test_pressure_to_altitude_custom_temperature(self) -> None:
        """Test altitude with different temperature."""
        p0 = 101325.0
        p = 100000.0  # lower pressure
        T_cold = 273.15  # 0°C
        T_warm = 293.15  # 20°C

        h_cold = pressure_to_altitude(p, p0, T_cold)
        h_warm = pressure_to_altitude(p, p0, T_warm)

        # Altitude depends on temperature
        assert h_cold != h_warm

    def test_pressure_to_altitude_invalid_inputs(self) -> None:
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError, match="pressure.*must be positive"):
            pressure_to_altitude(p=0.0, p0=101325.0)

        with pytest.raises(ValueError, match="reference pressure"):
            pressure_to_altitude(p=100000.0, p0=-101325.0)

        with pytest.raises(ValueError, match="temperature"):
            pressure_to_altitude(p=100000.0, p0=101325.0, T=0.0)


class TestDetectFloorChange(unittest.TestCase):
    """Test suite for floor change detection."""

    def test_floor_change_no_change(self) -> None:
        """Test no floor change for small altitude difference."""
        alt_prev = 10.0
        alt_curr = 10.5  # small change (< threshold)

        change = detect_floor_change(alt_prev, alt_curr, floor_height=3.0)

        assert change == 0

    def test_floor_change_up_one_floor(self) -> None:
        """Test detection of moving up one floor."""
        alt_prev = 10.0
        alt_curr = 13.5  # moved up ~3.5m (one floor)

        change = detect_floor_change(alt_prev, alt_curr, floor_height=3.0)

        assert change == +1

    def test_floor_change_down_one_floor(self) -> None:
        """Test detection of moving down one floor."""
        alt_prev = 13.0
        alt_curr = 10.0  # moved down 3m

        change = detect_floor_change(alt_prev, alt_curr, floor_height=3.0)

        assert change == -1

    def test_floor_change_exactly_at_threshold(self) -> None:
        """Test behavior at detection threshold."""
        alt_prev = 10.0
        alt_curr = 11.5  # exactly at threshold
        threshold = 1.5

        # At threshold boundary: implementation uses < (strict), so this triggers
        change = detect_floor_change(alt_prev, alt_curr, threshold=threshold)

        # With delta_h = 1.5 and threshold = 1.5: abs(1.5) < 1.5 is False, so detects change
        assert change == +1


class TestSmoothMeasurementSimple(unittest.TestCase):
    """Test suite for simple exponential smoothing."""

    def test_smooth_no_change(self) -> None:
        """Test smoothing when measurement equals previous estimate."""
        x_prev = 10.0
        z = 10.0
        alpha = 0.1

        x_next = smooth_measurement_simple(x_prev, z, alpha)

        # Should stay the same
        assert x_next == 10.0

    def test_smooth_small_alpha(self) -> None:
        """Test heavy smoothing (small alpha)."""
        x_prev = 10.0
        z = 15.0  # large jump
        alpha = 0.1  # heavy smoothing

        x_next = smooth_measurement_simple(x_prev, z, alpha)

        # Should move only 10% toward measurement
        expected = 0.9 * 10.0 + 0.1 * 15.0
        assert np.isclose(x_next, expected)
        assert x_next < 11.0  # mostly kept previous value

    def test_smooth_large_alpha(self) -> None:
        """Test light smoothing (large alpha)."""
        x_prev = 10.0
        z = 15.0
        alpha = 0.9  # light smoothing (track measurement closely)

        x_next = smooth_measurement_simple(x_prev, z, alpha)

        # Should move 90% toward measurement
        expected = 0.1 * 10.0 + 0.9 * 15.0
        assert np.isclose(x_next, expected)
        assert x_next > 14.0  # mostly followed measurement

    def test_smooth_multiple_steps(self) -> None:
        """Test exponential smoothing over multiple steps."""
        x = 0.0
        z_target = 10.0
        alpha = 0.2

        # Smooth toward target over 20 steps
        for _ in range(20):
            x = smooth_measurement_simple(x, z_target, alpha)

        # Should be close to target (converges exponentially)
        assert x > 9.0  # most of the way there
        assert x < 10.0  # but not quite at target

    def test_smooth_invalid_alpha(self) -> None:
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            smooth_measurement_simple(10.0, 12.0, alpha=0.0)

        with pytest.raises(ValueError):
            smooth_measurement_simple(10.0, 12.0, alpha=1.0)

        with pytest.raises(ValueError):
            smooth_measurement_simple(10.0, 12.0, alpha=1.5)


class TestCompensateHardIron(unittest.TestCase):
    """Test suite for hard-iron calibration."""

    def test_compensate_hard_iron_zero_offset(self) -> None:
        """Test hard-iron correction with zero offset."""
        mag_raw = np.array([25.0, 5.0, -35.0])
        offset = np.zeros(3)

        mag_corrected = compensate_hard_iron(mag_raw, offset)

        np.testing.assert_array_equal(mag_corrected, mag_raw)

    def test_compensate_hard_iron_with_offset(self) -> None:
        """Test hard-iron correction removes offset."""
        mag_raw = np.array([25.0, 5.0, -35.0])
        offset = np.array([5.0, 2.0, 5.0])

        mag_corrected = compensate_hard_iron(mag_raw, offset)

        expected = mag_raw - offset
        np.testing.assert_array_almost_equal(mag_corrected, expected)

    def test_compensate_hard_iron_negative_offset(self) -> None:
        """Test hard-iron correction with negative offset."""
        mag_raw = np.array([15.0, -5.0, -45.0])
        offset = np.array([-5.0, 5.0, -10.0])

        mag_corrected = compensate_hard_iron(mag_raw, offset)

        expected = np.array([20.0, -10.0, -35.0])
        np.testing.assert_array_almost_equal(mag_corrected, expected)

    def test_compensate_hard_iron_invalid_shapes(self) -> None:
        """Test that invalid shapes raise errors."""
        mag = np.array([25.0, 5.0, -35.0])

        with pytest.raises(ValueError, match="mag_raw must have shape"):
            compensate_hard_iron(np.array([25.0, 5.0]), np.zeros(3))

        with pytest.raises(ValueError, match="offset must have shape"):
            compensate_hard_iron(mag, np.zeros(2))


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests for environmental sensors."""

    def test_pressure_to_altitude_large_difference(self) -> None:
        """Test altitude for large pressure difference (tall building)."""
        p0 = 101325.0
        p_top = 98000.0  # much lower pressure (~300m higher)

        h = pressure_to_altitude(p_top, p0)

        # Should be several hundred meters
        assert h > 200
        assert h < 400

    def test_mag_heading_all_four_quadrants(self) -> None:
        """Every quadrant of heading is reachable, and lands where it should.

        Driven from the platform side: a walker facing into each quadrant must
        get a heading back in that quadrant. Driving it from the *reading*
        side, as this used to, describes the field rather than the platform
        and reads as though the two were the same thing.
        """
        for psi_deg, low, high in [
            (45.0, 0.0, np.pi / 2),
            (135.0, np.pi / 2, np.pi),
            (-45.0, -np.pi / 2, 0.0),
            (-135.0, -np.pi, -np.pi / 2),
        ]:
            mag = earth_field_body(0.0, 0.0, np.deg2rad(psi_deg))
            heading = mag_heading(mag, 0.0, 0.0)
            assert low < heading < high, f"facing {psi_deg} deg gave {heading} rad"

    def test_altitude_conversion_roundtrip(self) -> None:
        """Test pressure→altitude→pressure roundtrip (approximately)."""
        p0 = 101325.0
        T = 288.15
        h_target = 50.0  # m

        # Compute pressure at h_target using inverse formula (approximate)
        # p = p0 * (1 - L*h/T)^(g*M/(R*L))
        L = 0.0065
        R = 8.31432
        g = 9.80665
        M = 0.0289644
        alpha = (R * L) / (g * M)

        p_at_h = p0 * (1.0 - L * h_target / T) ** (1 / alpha)

        # Convert back to altitude
        h_computed = pressure_to_altitude(p_at_h, p0, T)

        # Should match target (within numerical tolerance)
        assert np.isclose(h_computed, h_target, atol=0.1)

    def test_smooth_negative_measurements(self) -> None:
        """Test smoothing works with negative values."""
        x_prev = -5.0
        z = -8.0
        alpha = 0.2

        x_next = smooth_measurement_simple(x_prev, z, alpha)

        expected = 0.8 * (-5.0) + 0.2 * (-8.0)
        assert np.isclose(x_next, expected)

    def test_floor_change_large_jump(self) -> None:
        """Test floor change with large altitude jump (elevator)."""
        alt_prev = 0.0
        alt_curr = 20.0  # jumped 20m (e.g., elevator to 7th floor)

        change = detect_floor_change(alt_prev, alt_curr, floor_height=3.0)

        # Simplified: only returns ±1
        assert change == +1


if __name__ == "__main__":
    unittest.main()
