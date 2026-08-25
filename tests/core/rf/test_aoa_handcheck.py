"""AOA measurement model against the book's hand-checked geometry.

Was a script named like a test: `tests/core/rf/test_aoa_handcheck.py` with a
`main()` and no `def test_`, so pytest collected nothing from it and it had
never run. Unlike ch7's equivalent, its content was *right* -- every value it
prints matches an independent calculation. It simply asserted nothing that
anything would notice.

The geometry is the book's, at Eqs. (4.63)-(4.65): anchor at (0, 10, 5) ENU,
agent at (5, 5, 0), so the deltas are (-5, +5, +5) and every quantity below is
exact rather than approximate -- which is what makes it a hand-check.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.63)-(4.65)
"""

import numpy as np
import pytest

from core.rf import (
    aoa_azimuth,
    aoa_elevation,
    aoa_measurement_vector,
    aoa_sin_elevation,
    aoa_tan_azimuth,
)

#: Argument order is (anchor, agent) throughout core.rf -- checked, not
#: assumed: the first attempt at this file had it the other way round and the
#: signs came out inverted.
ANCHOR = np.array([0.0, 10.0, 5.0])
AGENT = np.array([5.0, 5.0, 0.0])

#: (-5, 5, 5): equal magnitudes, so the answers are exact.
EXPECTED_RANGE = np.sqrt(75.0)


def test_the_geometry_is_the_one_that_makes_this_a_hand_check():
    """Deltas of equal magnitude, so range is sqrt(75) exactly."""
    delta = ANCHOR - AGENT

    assert delta == pytest.approx([-5.0, 5.0, 5.0])
    assert np.linalg.norm(delta) == pytest.approx(EXPECTED_RANGE)


def test_azimuth_is_minus_45_degrees():
    """Eq. (4.64): tan(psi) = dE / dN = -5/5, so psi = -45 exactly."""
    assert aoa_tan_azimuth(ANCHOR, AGENT) == pytest.approx(-1.0)
    assert np.degrees(aoa_azimuth(ANCHOR, AGENT)) == pytest.approx(-45.0)


def test_elevation_is_the_arcsine_of_one_over_root_three():
    """dU / range = 5 / sqrt(75) = 1/sqrt(3), so theta = 35.2644 degrees."""
    assert aoa_sin_elevation(ANCHOR, AGENT) == pytest.approx(1 / np.sqrt(3))
    assert np.degrees(aoa_elevation(ANCHOR, AGENT)) == pytest.approx(
        35.264389, abs=1e-5
    )


def test_the_measurement_vector_is_sin_elevation_then_tan_azimuth():
    """Eq. (4.65): z = [sin(theta), tan(psi)], and the order matters."""
    z = aoa_measurement_vector(ANCHOR[None, :], AGENT)

    assert np.asarray(z).ravel() == pytest.approx([1 / np.sqrt(3), -1.0])
