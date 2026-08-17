"""The shipped Ch2 dataset spans 2.7 km where its config declares 50 m.

**This file asserts a defect, and is written to fail when the defect is fixed.**
It is the pattern `tests/ch7_slam/test_frontend_actually_corrects.py` used
before the SLAM front-end was repaired: something is broken, it is not being
fixed in this change, and a comment would be lost.

`scripts/generate_ch2_coordinate_transforms_dataset.py` samples the building
footprint like this::

    meters_per_degree = 111000.0 * np.cos(lat_center)   # lat_center: RADIANS
    lat_offset_deg = building_size_m / 111000.0         # ...degrees
    lon_offset_deg = building_size_m / meters_per_degree
    lats = lat_center + rng.uniform(-lat_offset_deg/2, lat_offset_deg/2, n)

The last line adds a value the code itself names ``_deg`` to a latitude in
radians. No ``deg2rad`` anywhere, so every offset is 180/pi = 57.3x too large.
It is the same defect that `ch2_coords/example_coordinate_transforms.py` had in
its inline ENU targets -- fixed there, in the change that added this file --
sitting one directory away in the generator, and already baked into the
committed `.txt` files.

`config.json` declares ``building.size_m = 50.0``. The shipped
`enu_coordinates.txt` spans 2666 m east by 2612 m north. (Not the full 57.3x of
the 50 m width: 20 uniform draws do not reach both extremes, so the observed
span runs ~53x.)

Why the existing suite is green on this:

- `test_coordinate_files_agree.py` checks LLH, ECEF and ENU against *each
  other*. They agree perfectly -- the ENU was derived from the same inflated
  LLH. Self-consistency cannot see a common-mode error, which is the trap
  CLAUDE.md records for the ch2 rotation files and the ch6 strapdown frame.
- Nothing compares the data against `config.json`, which is the one file
  carrying the intended answer.

Fixing it means regenerating the dataset, which rewrites three committed `.txt`
files and moves every number the ch2 dataset README quotes. That is its own
change, with the diff-attribution procedure in CLAUDE.md to follow, and it is
deliberately not bundled here.

**When you fix it:** this file will fail. Delete it and replace the assertions
with their inverse -- the footprint fits within `building.size_m`, the way
`test_enu_offsets_are_metres.py` next door states its claim in metres.

Author: Li-Ta Hsu
References: Chapter 2, Section 2.1
"""

import json
import unittest
from pathlib import Path

import numpy as np

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "sim" / "ch2_coords_san_francisco"
)

#: Degrees per radian -- the factor the missing conversion costs.
DEG_PER_RAD = 180.0 / np.pi


class TestDatasetFootprintIsStillInflated(unittest.TestCase):
    """Pin the inflated footprint so repairing it is visible, not silent."""

    @classmethod
    def setUpClass(cls):
        with open(DATA_DIR / "config.json") as handle:
            cls.config = json.load(handle)
        cls.enu = np.loadtxt(DATA_DIR / "enu_coordinates.txt")
        cls.declared_m = float(cls.config["building"]["size_m"])

    def test_the_footprint_does_not_match_the_declared_building_size(self) -> None:
        """The shipped span is tens of times the size config.json declares.

        This is the assertion that inverts on repair. A correct dataset puts
        every sampled point inside the declared footprint, so the ratio drops
        to <= 1 and this fails -- which is the intent.
        """
        east_span = float(np.ptp(self.enu[:, 0]))
        north_span = float(np.ptp(self.enu[:, 1]))
        ratio = max(east_span, north_span) / self.declared_m

        self.assertGreater(
            ratio,
            10.0,
            "The ch2 dataset footprint now fits its declared size -- the "
            "generator bug this file pins appears to be fixed. Delete this "
            "file and assert the correct property instead: every ENU point "
            "inside building.size_m.",
        )

    def test_the_inflation_is_the_missing_degrees_to_radians_conversion(self) -> None:
        """The factor is 180/pi, not some arbitrary scale.

        Naming the cause is what makes this a diagnosis rather than a
        tolerance. 20 uniform draws do not reach both ends of the range, so the
        observed span is a fraction of the full 57.3x width; the check is that
        it is that order, and could not be anything else.
        """
        east_span = float(np.ptp(self.enu[:, 0]))
        north_span = float(np.ptp(self.enu[:, 1]))

        for axis, span in (("east", east_span), ("north", north_span)):
            with self.subTest(axis=axis):
                implied = span / self.declared_m
                self.assertGreater(
                    implied, 0.5 * DEG_PER_RAD,
                    f"{axis} span {span:.1f} m is {implied:.1f}x the declared "
                    f"{self.declared_m} m; expected the ~{DEG_PER_RAD:.1f}x of "
                    f"a missing deg2rad, so the cause may have changed.",
                )
                self.assertLess(
                    implied, DEG_PER_RAD,
                    f"{axis} span {span:.1f} m exceeds even the full "
                    f"{DEG_PER_RAD:.1f}x width of the declared footprint, "
                    f"which a uniform sample cannot reach.",
                )

    def test_the_files_still_agree_with_each_other(self) -> None:
        """The defect is common-mode, and that is why nothing caught it.

        If this ever fails, the dataset has a *second*, independent problem:
        the inflation is in the LLH the other files derive from, so ENU and
        LLH stay perfectly consistent while both describe the wrong building.
        """
        llh = np.loadtxt(DATA_DIR / "llh_coordinates.txt")
        self.assertEqual(
            len(llh), len(self.enu),
            "llh_coordinates.txt and enu_coordinates.txt describe different "
            "numbers of points.",
        )


if __name__ == "__main__":
    unittest.main()
