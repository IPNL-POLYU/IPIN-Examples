"""Two Chapter 3 examples printed an improvement measured from a single draw.

Both are the antipattern in `.cursor/rules/030-figures-and-claims.mdc`: an error
computed from one noise realisation, printed as a percentage, invites the reader
to treat it as a property of the method. Neither survived being averaged.

  - `example_least_squares` printed "Improvement: 36.7%" for WLS over unweighted
    LS. Over 5000 draws the RMS improvement is ~14%, the per-draw median ~9%,
    and WLS is *worse* on nearly 30% of draws. With four ranges, two unknowns
    and 36x more weight on one anchor, WLS inherits that anchor's luck.
  - `example_iekf_range_bearing` printed "IEKF improvement: 0.6%" from a mean
    over steps [5:]. That window is the one place the effect cannot appear: by
    step 5 both filters have converged and no linearisation error is left to
    remove. The first update is where it can appear -- but in this scenario the
    median gain there is only +3.7% over 200 seeds, with IEKF worse on 42%.

That last point is the substantive finding. The scenario is named "high
nonlinearity", yet its measurement noise (0.30 m range, 0.08 rad bearing) is
large enough to swamp the linearisation error, so the example does not
demonstrate its own thesis. Tightening the noise to 0.05 m / 0.01 rad, holding
the geometry fixed, lifts the first-update median to +39.5%; starting 8.49 m
out instead gives +48.1%; both together give +90.4% with IEKF never losing.
The last test pins that mechanism, so the claim the example now prints is
falsifiable rather than decorative.

A note on how these tests are built, because getting it wrong cost a full round
of wrong numbers. The IEKF helper reproduces the example's scenario exactly --
same dt, step count, noise, P0, initial offset, and the same legacy
`np.random.seed` stream -- and is checked against it by `test_helper_reproduces
_the_example`. An earlier version re-implemented the scenario with different
parameters and a `default_rng` stream; every test passed while measuring a
different experiment, and reported +20.3% where the truth is +3.7%. The WLS
helper does use `default_rng`, which is sound there because only the
*distribution* enters a Monte-Carlo average and the sigmas match; that was
confirmed against the example's own stream (14.3/8.5/29% against 14.4/9.0/27%).

Author: Li-Ta Hsu
References: Chapter 3, Sections 3.1.1 and 3.2.3
"""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: no display during tests

import numpy as np

from ch3_estimators.example_iekf_range_bearing import (
    create_models,
    create_range_bearing_innovation_func,
    setup_high_nonlinearity_scenario,
)
from ch3_estimators.example_least_squares import (
    compute_ranges,
    create_range_model,
    setup_positioning_scenario,
)
from core.estimators import (
    ExtendedKalmanFilter,
    IteratedExtendedKalmanFilter,
    linear_least_squares,
    weighted_least_squares,
)

# The example's own constants (ch3_estimators/example_iekf_range_bearing.py).
DT = 0.5
N_STEPS = int(25.0 / DT)
Q_SCALE = 0.3
RANGE_STD = 0.3
BEARING_STD = 0.08
X0_EST = np.array([4.0, 4.0, 0.0, 0.0])
P0 = np.diag([3.0, 3.0, 2.0, 2.0])

_CACHE = {}


def _q_func(dt):
    return Q_SCALE * np.array([
        [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
        [0, dt ** 3 / 3, 0, dt ** 2 / 2],
        [dt ** 2 / 2, 0, dt, 0],
        [0, dt ** 2 / 2, 0, dt],
    ])


def _iekf_errors(seed, range_std=RANGE_STD, bearing_std=BEARING_STD):
    """Position-error series for both filters, replicating the example.

    Returns arrays indexed as the example indexes them: element 0 is x0_est
    before any update (identical for both filters by construction), so the
    first update is element 1.
    """
    key = (seed, range_std, bearing_std)
    if key in _CACHE:
        return _CACHE[key]

    landmarks, true_x0 = setup_high_nonlinearity_scenario()
    process_model, process_jac, meas_model, meas_jac = create_models(landmarks)

    def r_func():
        diag = []
        for _ in landmarks:
            diag.extend([range_std ** 2, bearing_std ** 2])
        return np.diag(diag)

    innovation = create_range_bearing_innovation_func(len(landmarks))
    common = (process_model, process_jac, meas_model, meas_jac, _q_func, r_func)
    ekf = ExtendedKalmanFilter(
        *common, X0_EST.copy(), P0.copy(), innovation_func=innovation)
    iekf = IteratedExtendedKalmanFilter(
        *common, X0_EST.copy(), P0.copy(), max_iterations=5,
        convergence_tol=1e-6, innovation_func=innovation)

    np.random.seed(seed)  # legacy stream, as the example uses
    true_states, state = [true_x0.copy()], true_x0.copy()
    for _ in range(N_STEPS):
        state = process_model(state, None, DT) + np.random.multivariate_normal(
            np.zeros(4), _q_func(DT))
        true_states.append(state.copy())

    measurements = []
    for state in true_states[1:]:
        truth = meas_model(state)
        measurements.append(truth + np.random.multivariate_normal(
            np.zeros(len(truth)), r_func()))

    ekf_est, iekf_est = [X0_EST.copy()], [X0_EST.copy()]
    for z in measurements:
        ekf.predict(dt=DT)
        ekf.update(z)
        ekf_est.append(ekf.get_state()[0].copy())
        iekf.predict(dt=DT)
        iekf.update(z)
        iekf_est.append(iekf.get_state()[0].copy())

    truth_arr = np.array(true_states)[:, :2]
    out = (
        np.linalg.norm(np.array(ekf_est)[:, :2] - truth_arr, axis=1),
        np.linalg.norm(np.array(iekf_est)[:, :2] - truth_arr, axis=1),
    )
    _CACHE[key] = out
    return out


def _gains(window, seeds=40, **kw):
    """Percentage improvement of IEKF over EKF in `window`, one per seed."""
    out = []
    for seed in range(seeds):
        ekf, iekf = _iekf_errors(seed, **kw)
        e, i = ekf[window].mean(), iekf[window].mean()
        out.append((e - i) / e * 100.0)
    return np.asarray(out)


class TestWlsImprovementIsModest(unittest.TestCase):
    """WLS beats LS on average here, but not by 36.7% and not every time."""

    @staticmethod
    def _errors(trials=2000, seed=0):
        anchors, truth = setup_positioning_scenario()
        h, jacobian = create_range_model(anchors)
        stds = np.array([0.05, 0.3, 0.3, 0.3])
        W = np.diag(1.0 / stds ** 2)
        x0 = np.array([5.0, 5.0])
        A = jacobian(x0)

        rng = np.random.default_rng(seed)
        e_wls, e_ls = [], []
        for _ in range(trials):
            y = np.array([
                np.linalg.norm(truth - anchors[i]) + rng.normal(0, stds[i])
                for i in range(len(anchors))
            ])
            r = y - h(x0)
            e_wls.append(np.linalg.norm(x0 + weighted_least_squares(A, r, W)[0] - truth))
            e_ls.append(np.linalg.norm(x0 + linear_least_squares(A, r)[0] - truth))
        return np.asarray(e_wls), np.asarray(e_ls)

    def test_weighting_helps_on_average(self):
        e_wls, e_ls = self._errors()
        rms_gain = 1.0 - np.sqrt((e_wls ** 2).mean()) / np.sqrt((e_ls ** 2).mean())

        self.assertGreater(rms_gain, 0.05)

    def test_the_gain_is_far_below_the_single_draw_that_was_reported(self):
        """36.7% was a lucky realisation, not the method's accuracy."""
        e_wls, e_ls = self._errors()
        rms_gain = 1.0 - np.sqrt((e_wls ** 2).mean()) / np.sqrt((e_ls ** 2).mean())

        self.assertLess(rms_gain, 0.25)

    def test_weighting_loses_on_a_substantial_minority_of_draws(self):
        """The part a percentage hides: WLS inherits its best sensor's luck."""
        e_wls, e_ls = self._errors()

        self.assertGreater(float(np.mean(e_wls > e_ls)), 0.10)

    def test_the_example_still_generates_ranges_the_way_this_test_assumes(self):
        """Guards the one thing the `default_rng` shortcut above depends on.

        Only the noise *distribution* enters the averages, so a different
        stream is fine -- but only while the example really does add
        N(0, sigma_i) to the true range of anchor i.
        """
        anchors, truth = setup_positioning_scenario()
        np.random.seed(7)
        drawn = np.array([
            compute_ranges(truth, anchors[i:i + 1], noise_std=0.0)[0]
            for i in range(len(anchors))
        ])
        exact = np.array([np.linalg.norm(truth - a) for a in anchors])

        np.testing.assert_allclose(drawn, exact, atol=1e-12)


class TestIekfHelpsOnlyBeforeConvergence(unittest.TestCase):
    """The effect lives in the window the example discarded -- and is small."""

    def test_helper_reproduces_the_example(self):
        """Without this, the rest of the class can measure another experiment.

        Seed 42 is the example's own. It prints -19.6% on the first update and
        +0.6% over [5:]; those are the numbers to match.
        """
        ekf, iekf = _iekf_errors(42)
        first = (ekf[1] - iekf[1]) / ekf[1] * 100.0
        late = (ekf[5:].mean() - iekf[5:].mean()) / ekf[5:].mean() * 100.0

        self.assertAlmostEqual(first, -19.6, delta=0.1)
        self.assertAlmostEqual(late, 0.6, delta=0.1)

    def test_index_zero_is_identical_for_both_filters(self):
        """Why the first update is index 1: slicing [:1] prints a sure 0.0%."""
        ekf, iekf = _iekf_errors(42)

        self.assertAlmostEqual(ekf[0], iekf[0], places=12)

    def test_iekf_helps_on_the_first_update(self):
        """Where the linearisation point is 2.83 m from the truth."""
        self.assertGreater(float(np.median(_gains(slice(1, 2)))), 0.5)

    def test_but_only_slightly_at_this_operating_point(self):
        """The example's thesis is not carried by a ~3% median.

        Pinned as an upper bound so that if someone strengthens the scenario,
        this fails and the example's prose has to be revisited with it.
        """
        self.assertLess(float(np.median(_gains(slice(1, 2)))), 10.0)

    def test_iekf_is_worse_on_a_large_minority_of_seeds(self):
        self.assertGreater(float(np.mean(_gains(slice(1, 2)) < 0)), 0.25)

    def test_iekf_stops_helping_once_the_filter_has_converged(self):
        """The window the example used to average over, and why it read ~0."""
        self.assertLess(abs(float(np.median(_gains(slice(5, None))))), 1.5)

    def test_the_advantage_appears_when_nonlinearity_dominates_noise(self):
        """The mechanism behind the claim the example now prints.

        Same geometry and initial error; only the measurement noise shrinks.
        If this ever fails, the printed explanation is wrong, not just stale.
        """
        tight = _gains(slice(1, 2), seeds=30, range_std=0.05, bearing_std=0.01)
        loose = _gains(slice(1, 2), seeds=30)

        self.assertGreater(float(np.median(tight)), 20.0)
        self.assertGreater(float(np.median(tight)), 5.0 * float(np.median(loose)))


if __name__ == "__main__":
    unittest.main()
