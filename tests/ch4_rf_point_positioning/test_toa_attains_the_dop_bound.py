"""TOA positioning accuracy must match Eq. (4.107), not one lucky draw.

The example solved once with noise, printed "Position error: 0.055 m" and
"Error/Noise ratio: 0.55", and left the reader to conclude that positioning is
twice as good as the ranging behind it. It is not: 0.055 m is a single sample
from a distribution whose 10th-90th percentiles span 0.03 to 0.15 m.

The quantity that does characterise the geometry is the chapter's own
sigma_position = HDOP * sigma_range. This solver attains it -- 0.1012 m
measured against 0.1010 m predicted -- which is a far stronger statement than
any single draw, and it is the relationship Section 4.8 exists to teach.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.20), (4.103)-(4.107)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.rf import (
    TOAPositioner,
    compute_dop,
    position_error_from_dop,
    toa_range,
)

ANCHORS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
TRUE_POS = np.array([3.0, 7.0])
NOISE_STD = 0.1
TRIALS = 2000
SEED = 42


def _solve_many(anchors, true_pos, noise_std, trials, seed):
    """Position error over repeated noise draws."""
    true_ranges = np.array([toa_range(a, true_pos) for a in anchors])
    rng = np.random.default_rng(seed)
    errors = np.empty(trials)
    for k in range(trials):
        noisy = true_ranges + rng.standard_normal(len(anchors)) * noise_std
        estimate, _ = TOAPositioner(anchors, method="iterative_ls").solve(
            noisy, initial_guess=np.array([5.0, 5.0])
        )
        errors[k] = np.linalg.norm(estimate - true_pos)
    return errors


def _hdop(anchors, position):
    """HDOP of the unit-line-of-sight geometry at ``position``."""
    geometry = (position - anchors) / np.linalg.norm(
        position - anchors, axis=1, keepdims=True
    )
    return compute_dop(geometry)["HDOP"]


class TestToaAttainsTheDopBound(unittest.TestCase):
    """The estimator should be efficient, and be shown to be."""

    @classmethod
    def setUpClass(cls):
        cls.errors = _solve_many(ANCHORS, TRUE_POS, NOISE_STD, TRIALS, SEED)
        cls.hdop = _hdop(ANCHORS, TRUE_POS)

    def test_rms_error_matches_hdop_times_sigma(self):
        """Eq. (4.107), as a measurement rather than an assertion.

        Brackets from both sides. Materially above the bound means the solver
        is leaving information on the table; materially below means the test
        setup is not measuring what it claims, since no unbiased estimator
        beats it.
        """
        predicted = position_error_from_dop(self.hdop, NOISE_STD)
        measured = float(np.sqrt(np.mean(self.errors**2)))

        self.assertAlmostEqual(measured / predicted, 1.0, delta=0.1)

    def test_a_single_draw_is_not_an_accuracy_figure(self):
        """Why the example now reports a distribution.

        The spread is wide enough that any one solve can look twice as good,
        or half as good, as the geometry allows. That is what made the old
        "error/noise ratio: 0.55" misleading rather than merely imprecise.
        """
        low, high = np.percentile(self.errors, [10, 90])

        self.assertLess(low, 0.5 * np.sqrt(np.mean(self.errors**2)))
        self.assertGreater(high / low, 2.0)

    def test_worse_geometry_gives_proportionally_worse_position(self):
        """DOP has to track geometry, or it is not measuring geometry.

        Collapsing the anchors towards a line inflates HDOP, and the position
        error must follow it by the same factor -- that proportionality is the
        content of Eq. (4.107).
        """
        # Anchors clustered so they subtend a narrow angle from the target;
        # a line of anchors is not enough, since a target off that line still
        # sees them spread. Chosen by computing HDOP, not by eye: this gives
        # 6.98 against the square layout's 1.01.
        poor_anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
        poor_hdop = _hdop(poor_anchors, TRUE_POS)
        self.assertGreater(poor_hdop, 3.0 * self.hdop)

        errors = _solve_many(poor_anchors, TRUE_POS, NOISE_STD, 500, SEED)
        predicted = position_error_from_dop(poor_hdop, NOISE_STD)
        measured = float(np.sqrt(np.mean(errors**2)))

        self.assertAlmostEqual(measured / predicted, 1.0, delta=0.25)


if __name__ == "__main__":
    unittest.main()
