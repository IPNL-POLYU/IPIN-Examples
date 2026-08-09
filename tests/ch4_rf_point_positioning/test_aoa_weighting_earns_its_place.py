"""AOA measurement weighting has to beat not weighting, or it is decoration.

The Chapter 4 comparison passes a per-anchor sigma so the solver can
down-weight a degraded anchor (W_a in Eq. 4.77). That is a claim, and the
example prints an unweighted control beside it precisely so the claim can be
checked rather than assumed.

The subtlety worth pinning is which sigma does anything. In angle space the
weight matrix is diag(1/sigma^2), so a *uniform* sigma makes W a scalar
multiple of the identity and it cancels exactly out of (H'WH)^-1 H'W. Only the
spread between anchors carries information.

That is easy to get wrong, and was: a change proposed in PR #3 passed a scalar
sigma_psi to this solver believing it stabilised the output. It did change
results at the time, but only because the old tan parameterisation propagated
variance as var(tan psi) = sec^4(psi) var(psi), making the weights
angle-dependent. That amplification is what let near-singular anchors dominate
the normal equations -- it was the bug, not the feature. With residuals formed
in angle space (see test_aoa_initialisation_basin.py) a scalar sigma is inert,
so passing one would be a parameter that silently does nothing.

Author: Li-Ta Hsu
References: Chapter 4, Eqs. (4.63)-(4.66), (4.77)
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from core.rf import AOAPositioner, aoa_azimuth

ANCHORS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
DEGRADED = 3
SEED = 1


def _sigma(base_rad, scale):
    sigma = np.full(len(ANCHORS), base_rad, dtype=float)
    sigma[DEGRADED] *= scale
    return sigma


def _median_error(sigma_truth, sigma_told, trials=400):
    """Median error when the solver is told `sigma_told` about the noise.

    `sigma_told=None` means solve unweighted. The measurements always carry
    `sigma_truth`, so the only thing varying is what the solver believes.
    """
    rng = np.random.default_rng(SEED)
    points = rng.uniform(1, 9, size=(trials, 2))
    centroid = ANCHORS.mean(axis=0)
    errors = []
    for p in points:
        angles = np.array([aoa_azimuth(a, p) for a in ANCHORS])
        angles = angles + rng.normal(0, sigma_truth)
        kwargs = {"initial_guess": centroid}
        if sigma_told is not None:
            kwargs["sigma_psi"] = sigma_told
        est, info = AOAPositioner(ANCHORS).solve(angles, **kwargs)
        if info["converged"]:
            errors.append(float(np.linalg.norm(est - p)))
    return float(np.median(errors)), len(errors)


class TestAoaWeightingEarnsItsPlace(unittest.TestCase):
    """One anchor ten times noisier than the rest."""

    def test_per_anchor_sigma_beats_no_weighting(self):
        """The claim the comparison table makes, asserted."""
        sigma = _sigma(np.deg2rad(1.0), 10.0)

        weighted, n_w = _median_error(sigma, sigma)
        unweighted, n_u = _median_error(sigma, None)

        self.assertEqual(n_w, n_u, "both arms must solve the same count")
        self.assertLess(weighted, unweighted / 2.0)

    def test_a_uniform_sigma_changes_nothing(self):
        """A scalar sigma is inert here, and silently so.

        Pinned because passing one looks like applying a weighting. If this
        ever fails, angle-space weighting has acquired a dependence on the
        estimate and the claim in `aoa_positioning_test` needs revisiting.
        """
        sigma = _sigma(np.deg2rad(1.0), 10.0)
        rng = np.random.default_rng(SEED)
        centroid = ANCHORS.mean(axis=0)

        worst = 0.0
        for p in rng.uniform(1, 9, size=(100, 2)):
            angles = np.array([aoa_azimuth(a, p) for a in ANCHORS])
            angles = angles + rng.normal(0, sigma)
            plain, _ = AOAPositioner(ANCHORS).solve(angles, initial_guess=centroid)
            scaled, _ = AOAPositioner(ANCHORS).solve(
                angles, initial_guess=centroid, sigma_psi=np.deg2rad(7.5)
            )
            worst = max(worst, float(np.linalg.norm(plain - scaled)))

        self.assertLess(worst, 1e-9)

    def test_weighting_helps_less_as_the_good_anchors_degrade(self):
        """The gain is not a constant, and the example says so.

        Weighting buys back what one bad sensor costs only while the others
        are still worth listening to. Asserted so the table's tapering column
        is a property rather than an accident of one seed.
        """
        gain = {}
        for base_deg in (1.0, 10.0):
            sigma = _sigma(np.deg2rad(base_deg), 10.0)
            weighted, _ = _median_error(sigma, sigma)
            unweighted, _ = _median_error(sigma, None)
            gain[base_deg] = unweighted / weighted

        self.assertGreater(gain[1.0], 2.0)
        self.assertLess(gain[10.0], gain[1.0])


if __name__ == "__main__":
    unittest.main()
