"""The shipped Ch2 dataset must describe the building `config.json` declares.

This replaces `test_dataset_footprint_is_still_inflated.py`, which asserted the
opposite and was written to fail the moment the generator was repaired. It did,
with the message telling the fixer to delete it, and this is what it asked for.

The defect it pinned: `generate_ch2_coordinate_transforms_dataset.py` computed
``lat_offset_deg = building_size_m / 111000.0`` and added it straight to a
latitude in radians -- the code named the variable ``_deg`` itself, and no
``deg2rad`` ever ran. Every offset was 180/pi = 57.3x too large, so the declared
50 m footprint was sampled across 2666 m x 2612 m. The generator now works in
metres and converts with `core.coords.enu_to_llh_offset`, which takes metres and
returns radians so there is no per-degree quantity left to mislay.

**Why nothing caught it, which is the part worth keeping.** Two guards ran over
this dataset and both were green:

- `test_coordinate_files_agree.py` checks LLH, ECEF and ENU against *each
  other*. They agreed perfectly -- the ENU was derived from the same inflated
  LLH, so the error was common-mode and consistency could not see it.
- The dataset README's round-trip experiments recompute both sides from the
  same source file, the failure mode CLAUDE.md already records for the ch2
  rotation files and the ch6 strapdown frame.

Nothing compared the data against `config.json`, which is the one file carrying
the intended answer rather than a derived one. That is this file's job, and it
is why it asserts against the config rather than against a hard-coded 50.0.

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

#: ENU is stored at ``fmt="%.3f"``, so 1 mm is the quantisation floor. The
#: sampler draws uniformly on the closed footprint, so a point may sit exactly
#: on the boundary; allow a centimetre and no more.
EDGE_TOLERANCE_M = 0.01

#: 20 uniform draws do not reach both extremes of a range. The expected span of
#: the sample max minus min is (n-1)/(n+1) = 90.5% of the full width, so a
#: correct dataset lands near there -- well clear of both the 1x ceiling and the
#: floor below, while the defect this replaces sat at 53x.
MIN_PLAUSIBLE_SPAN_FRACTION = 0.5


class TestDatasetMatchesItsConfig(unittest.TestCase):
    """The data agrees with the parameters `config.json` says produced it."""

    @classmethod
    def setUpClass(cls):
        with open(DATA_DIR / "config.json") as handle:
            cls.config = json.load(handle)
        cls.enu = np.loadtxt(DATA_DIR / "enu_coordinates.txt")
        cls.declared_m = float(cls.config["building"]["size_m"])
        cls.declared_points = int(cls.config["building"]["num_points"])

    def test_every_point_is_inside_the_declared_footprint(self) -> None:
        """No sampled point sits outside the building it is supposed to be in.

        The assertion that inverts the old file. At 57.3x, points sat 1.3 km
        from a building declared 50 m across.
        """
        half = self.declared_m / 2.0 + EDGE_TOLERANCE_M
        for axis, column in (("east", 0), ("north", 1)):
            with self.subTest(axis=axis):
                worst = float(np.max(np.abs(self.enu[:, column])))
                self.assertLessEqual(
                    worst,
                    half,
                    f"a point sits {worst:.1f} m {axis} of centre, outside the "
                    f"{self.declared_m} m footprint config.json declares. An "
                    f"offset in metres must be converted to radians before it "
                    f"is added to a coordinate already in radians -- use "
                    f"core.coords.enu_to_llh_offset.",
                )

    def test_the_footprint_is_actually_used(self) -> None:
        """The points spread across the footprint rather than bunching at zero.

        Without this, the check above passes for a dataset that collapsed every
        point to the origin -- which a conversion bug can produce just as
        easily as an inflated one, and which would look perfectly compliant.
        """
        for axis, column in (("east", 0), ("north", 1)):
            with self.subTest(axis=axis):
                span = float(np.ptp(self.enu[:, column]))
                self.assertGreater(
                    span,
                    MIN_PLAUSIBLE_SPAN_FRACTION * self.declared_m,
                    f"{axis} span is only {span:.2f} m across a "
                    f"{self.declared_m} m footprint; the sample has collapsed "
                    f"toward the reference point.",
                )

    def test_the_point_count_matches(self) -> None:
        """`num_points` describes the file that shipped."""
        self.assertEqual(len(self.enu), self.declared_points)

    def test_the_reported_rotation_accuracy_is_not_a_wrap_artifact(self) -> None:
        """`rotation_roundtrip_deg` measures the round-trip, not the branch cut.

        It read 360.0, and the dataset README explained that as gimbal lock and
        recommended quaternions. It was neither: yaw is sampled on [0, 2pi) and
        recovered on (-pi, pi], so an exact round-trip of 4.4307 rad came back
        as -1.8525 rad and a raw subtraction called the 2pi difference error.
        Wrapped, it is 0.0 -- the round-trip was always exact.

        A rotation error of 360 degrees is the identity. Any pipeline reporting
        it is measuring its own subtraction.
        """
        reported = float(self.config["accuracy"]["rotation_roundtrip_deg"])
        self.assertLess(
            reported,
            1.0,
            f"config.json reports {reported} deg of rotation round-trip error. "
            f"360 deg is the identity rotation -- wrap the angle difference to "
            f"[-pi, pi] before taking its magnitude.",
        )


if __name__ == "__main__":
    unittest.main()
