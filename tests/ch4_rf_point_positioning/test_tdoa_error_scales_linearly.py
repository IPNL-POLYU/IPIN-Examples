"""TDOA position error is linear in the range-difference noise.

The example's noise table solved once per level and reported 0.046, 0.136 and
0.857 m for 0.1, 0.5 and 1.0 m of noise -- error ratios of 3.0x and 6.3x where
the noise ratios are 5x and 2x. Single draws tabulated as a trend, the same
defect as the TOA and AOA examples. Demos 3 and 4 in that same file always
averaged and were fine.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.27)-(4.40)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.rf import TDOAPositioner

ANCHORS = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
TRUE_POS = np.array([7.0, 12.0])
SEED = 42


def _median_error(noise_std, trials=200):
    """Median position error over repeated range-difference noise draws."""
    dist_ref = np.linalg.norm(TRUE_POS - ANCHORS[0])
    tdoa_true = np.array(
        [
            np.linalg.norm(TRUE_POS - ANCHORS[i]) - dist_ref
            for i in range(1, len(ANCHORS))
        ]
    )
    rng = np.random.default_rng(SEED)
    errors = []
    for _ in range(trials):
        noisy = tdoa_true + rng.standard_normal(len(tdoa_true)) * noise_std
        est, info = TDOAPositioner(ANCHORS, reference_idx=0).solve(
            noisy, initial_guess=np.array([10.0, 10.0])
        )
        if info["converged"]:
            errors.append(float(np.linalg.norm(est - TRUE_POS)))
    return float(np.median(errors))


class TestTdoaErrorScalesLinearly(unittest.TestCase):
    """The law the noise table is supposed to demonstrate."""

    def test_error_is_proportional_to_noise(self):
        """Error divided by noise is constant down the column.

        This is the self-check the example now prints. A table whose ratios
        wander is either non-linear -- which would be the interesting finding --
        or is not measuring what it claims.
        """
        slopes = [_median_error(s) / s for s in (0.1, 0.5, 1.0)]

        for slope in slopes:
            self.assertAlmostEqual(slope / slopes[0], 1.0, delta=0.2)

    def test_geometry_beats_the_raw_measurement_noise(self):
        """Four anchors with good geometry give sub-unity error per metre.

        Guards the setup as much as the result: a slope near or above 1 would
        mean the solve is not combining the measurements.
        """
        self.assertLess(_median_error(0.5) / 0.5, 1.0)


if __name__ == "__main__":
    unittest.main()
