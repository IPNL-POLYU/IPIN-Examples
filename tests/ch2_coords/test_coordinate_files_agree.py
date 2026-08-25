"""The Ch2 coordinate files must describe one set of points, in three frames.

`llh_coordinates.txt`, `ecef_coordinates.txt` and `enu_coordinates.txt` are the
same 20 points in three frames, and `reference_llh.txt` is the origin the ENU
axes are built about. Converting between them with `core.coords` has to
reproduce the shipped bytes, or the dataset teaches a transform it does not
itself satisfy.

The Up axis did not. `reference_llh.txt` was written as
`[lat_center, lon_center, heights[0]]` -- the height of the *first sampled
point*, which drew floor 5 and so read 15 m -- while the ENU was computed about
`height_ground = 0.0`. East and North agreed to storage precision and Up was
out by exactly 15 m on every point, which is the signature of an origin
mismatch rather than a numerical one. `config.json` recorded the reference
height correctly as 0.0 the whole time; only this file disagreed.

That went unnoticed for the same reason as the rotation files (see
`test_rotation_files_agree.py`): the README's round-trip experiment recomputes
ECEF and ENU from `llh_coordinates.txt` and compares those, which passes
whatever the shipped reference says.

Tolerances come from the files. Coordinates are written at `fmt="%.3f"`, so
millimetre quantisation is the floor and anything tighter fails on storage
rather than on content.

Author: Li-Ta Hsu
References: Chapter 2, Eqs. (2.1)-(2.3)
"""

import unittest
from pathlib import Path

import numpy as np

from core.coords import ecef_to_enu, llh_to_ecef

DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "sim" / "ch2_coords_san_francisco"
)

#: Coordinates are stored at 3 decimals, so 1 mm is the quantisation floor.
STORAGE_TOL_M = 2e-3


def _load():
    llh = np.loadtxt(DATA_DIR / "llh_coordinates.txt")
    ecef = np.loadtxt(DATA_DIR / "ecef_coordinates.txt")
    enu = np.loadtxt(DATA_DIR / "enu_coordinates.txt")
    ref = np.loadtxt(DATA_DIR / "reference_llh.txt")
    return llh, ecef, enu, ref


class TestCh2CoordinateFilesAgree(unittest.TestCase):
    """Cross-file checks against the shipped bytes."""

    def test_ecef_matches_the_llh_it_came_from(self):
        llh, ecef, _, _ = _load()
        predicted = np.array([llh_to_ecef(*row) for row in llh])

        np.testing.assert_allclose(predicted, ecef, atol=STORAGE_TOL_M)

    def test_enu_matches_the_shipped_reference(self):
        """The check the old reference height failed.

        Uses `reference_llh.txt` rather than a recomputed origin, because that
        file is what a reader loads and the bug lived in it. Passing the right
        origin from elsewhere would have hidden it.
        """
        _, ecef, enu, ref = _load()
        predicted = np.array([ecef_to_enu(*point, *ref) for point in ecef])

        np.testing.assert_allclose(predicted, enu, atol=STORAGE_TOL_M)

    def test_the_reference_height_is_the_enu_origin(self):
        """Stated separately so a failure names the cause, not just the symptom.

        An origin offset shows up only in Up, so this pins that the reference
        height is the one the Up axis is measured from.
        """
        _, ecef, enu, ref = _load()
        up_predicted = np.array([ecef_to_enu(*point, *ref)[2] for point in ecef])

        offset = float(np.mean(up_predicted - enu[:, 2]))
        self.assertLess(
            abs(offset),
            STORAGE_TOL_M,
            f"Up is offset by {offset:.3f} m, so reference_llh.txt's height "
            f"({ref[2]:.3f} m) is not the origin the ENU was built about",
        )


if __name__ == "__main__":
    unittest.main()
