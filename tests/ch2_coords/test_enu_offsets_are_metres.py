"""A target named "100m East" must come back from ENU as 100 metres east.

Section 3 of ``example_coordinate_transforms`` offsets a reference point in
LLH, converts through ECEF to ENU, and prints the result under a label that
states the answer. For a long time the answer was wrong by a factor of 57.3::

    Target: 100m East   ->  ENU: [6405.80, 2.49, -3.21] m
    Target: 100m North  ->  ENU: [-0.00, 5729.20, -2.58] m
    Target: 50m Up      ->  ENU: [0.00, -0.00, 50.00] m

The offsets were built as ``np.deg2rad(-122.4194) + 100 / 78800`` -- 78800 is
roughly metres per *degree* of longitude, added to a value already in radians,
so ``100 / 78800`` rad is 0.0727 deg and lands 6.4 km out. (78800 is also the
figure for 45 deg latitude, borrowed from ``tests/core/coords/test_transforms``
where it is used correctly; at 37.77 deg it is a further 12% wrong. Two unit
errors, one of which would have hidden the other.)

Two things let it survive:

- **One of the three lines was right.** A height offset is metres either way,
  so "50m Up" printed 50.00 and the block looked like it was working.
- **Nothing read the numbers.** The README transcript showed them, but that
  block was the last unmarked entry in ``UNCHECKED_TRANSCRIPTS`` next door,
  precisely because marking it would have pinned the defect.

The example now derives the offset from the local radii of curvature, so this
test states the claim in the units the label is written in. The transcript
check in ``tests/docs`` pins the printed text; this pins the physics, and would
survive a reformatting of the output.

Author: Li-Ta Hsu
References: Chapter 2, Eqs. (2.9)-(2.10)
"""

import re
import unittest

import numpy as np

from tests.example_runner import run_example

MODULE = "ch2_coords.example_coordinate_transforms"

#: The ENU each target's label promises, in metres.
NAMED_OFFSETS = {
    "100m East": np.array([100.0, 0.0, 0.0]),
    "100m North": np.array([0.0, 100.0, 0.0]),
    "50m Up": np.array([0.0, 0.0, 50.0]),
}

#: Printed at ``%.2f``, so 5 mm is the quantisation floor; the second-order
#: curvature term the linear metres-to-radians conversion drops adds ~1 mm at
#: 100 m. 10 mm clears both and still fails on anything worth reporting -- the
#: defect this file exists for was out by 6.4 km.
TOLERANCE_M = 0.01

# "Target: 100m East" then, on the next line, "  ENU: [100.00, 0.00, -0.00] m".
TARGET_BLOCK = re.compile(
    r"^Target:\s*(?P<name>.+?)\s*$\s*^\s*ENU:\s*\["
    r"(?P<e>[-\d.]+),\s*(?P<n>[-\d.]+),\s*(?P<u>[-\d.]+)\]\s*m\s*$",
    re.M,
)


class TestNamedENUOffsets(unittest.TestCase):
    """The ENU printed for each labelled target agrees with its label."""

    @classmethod
    def setUpClass(cls):
        cls.example = run_example(MODULE)
        cls.reported = {
            match.group("name"): np.array(
                [float(match.group("e")),
                 float(match.group("n")),
                 float(match.group("u"))]
            )
            for match in TARGET_BLOCK.finditer(cls.example.process.stdout)
        }

    def test_example_runs(self) -> None:
        """The example exits cleanly, so its output means something."""
        self.assertEqual(
            self.example.process.returncode,
            0,
            f"{MODULE} exited {self.example.process.returncode}:\n"
            f"{self.example.process.stderr[-2000:]}",
        )

    def test_every_named_target_was_found(self) -> None:
        """Guard the parse itself.

        Without this, a change to the print format leaves ``reported`` empty
        and the assertions below iterate over nothing -- passing green while
        checking the example not at all. Read as: this is the assertion that
        makes the next one able to fail.
        """
        self.assertEqual(
            set(self.reported),
            set(NAMED_OFFSETS),
            "Section 3's output no longer parses. Found "
            f"{sorted(self.reported)}, expected {sorted(NAMED_OFFSETS)}.",
        )

    def test_each_offset_returns_the_metres_it_is_named_for(self) -> None:
        """"100m East" is 100 m east, not 100 units of whatever the code did."""
        for name, expected in NAMED_OFFSETS.items():
            with self.subTest(target=name):
                actual = self.reported.get(name)
                self.assertIsNotNone(actual, f"no ENU printed for {name!r}")
                error = float(np.linalg.norm(actual - expected))
                self.assertLess(
                    error,
                    TOLERANCE_M,
                    f"{name} came back as ENU {actual.tolist()} m, "
                    f"{error:.3f} m from the {expected.tolist()} m its label "
                    f"promises. An offset in metres has to be divided by a "
                    f"radius (or converted from degrees) before it is added "
                    f"to a latitude or longitude already in radians.",
                )

    def test_the_horizontal_targets_are_not_merely_small(self) -> None:
        """East and North must be distinguishable, and each on its own axis.

        The bug put 6.4 km on the east axis, which the tolerance above catches.
        The opposite failure -- an offset that collapses to zero, so every
        target reports the origin and every error is small -- would not be, and
        a conversion bug can land either way.
        """
        east = self.reported["100m East"]
        north = self.reported["100m North"]
        self.assertGreater(east[0], 50.0, "'100m East' has no east component")
        self.assertGreater(north[1], 50.0, "'100m North' has no north component")
        self.assertLess(abs(east[1]), 1.0, "'100m East' drifted north")
        self.assertLess(abs(north[0]), 1.0, "'100m North' drifted east")


if __name__ == "__main__":
    unittest.main()
