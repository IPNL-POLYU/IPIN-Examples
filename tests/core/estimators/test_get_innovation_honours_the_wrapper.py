"""`get_innovation` must wrap the same way `update` does.

`ExtendedKalmanFilter` and `IteratedExtendedKalmanFilter` both accept an
`innovation_func`, and both exist to be given one: a range-bearing measurement
needs `angle_diff` on its bearing components, because a raw subtraction across
the pi/-pi branch cut returns a residual that is ~2*pi wrong **and carries the
wrong sign**. `update()` honoured the function. `get_innovation` did not, and
subtracted the angles raw.

That is the same defect CLAUDE.md records for `core/slam/factor_graph.py`,
whose bearing residual was `predicted - z` with no wrap: an anchor due west and
a 1 cm perturbation flip the raw residual to -6.2807 rad where the truth is
+0.0025.

**Nothing was calling the broken helper, which is why it survived.** The tell
was a comment in a caller: `ch3_estimators/example_ekf_range_bearing.py` needed
a normalised innovation squared for its consistency line and computed it by
hand, with a note saying `get_innovation` "subtracts the angles raw and so
cannot be used on bearings". A workaround comment in a caller is a bug report
against the library, and this file is what it turned into.

The linear `KalmanFilter` is deliberately not covered: it has no
`innovation_func` (zero occurrences in the file) and a linear measurement's raw
subtraction is correct.

**These assertions were run against the unfixed code first.** Both filters
reported an innovation of +6.2578 rad where the wrapped residual is -0.0254 --
two orders of magnitude too large and pointing the other way, which is what a
branch-cut residual looks like. The consequence showed in the state: `update`
moved the IEKF by [0.00024, 0.02309] while the helper's innovation predicted
[0.0722, -5.6884].

**What this file does not demand.** `get_innovation` is a *pre-update
diagnostic*, linearised at the prediction, and the IEKF moves its linearisation
point as it iterates -- so "the helper equals the innovation update consumed"
is the right identity for the EKF and the wrong one for the IEKF. It is pinned
where it holds (exactly for the EKF; exactly for the IEKF's first iteration,
via `max_iterations=1`) and replaced by a measured comparison where it does
not.

Author: Li-Ta Hsu
"""

import numpy as np
import pytest

from core.estimators import ExtendedKalmanFilter
from core.estimators.iterated_extended_kalman_filter import (
    IteratedExtendedKalmanFilter,
)
from core.utils import angle_diff

#: State is [x, y]; the single measurement is the bearing from the origin.
#: The predicted state sits just below the +x axis so its bearing is just under
#: -pi... in fact just over -pi, and the measurement sits just under +pi. The
#: two are 0.0254 rad apart the short way round and 6.2578 rad apart the long
#: way, which is the whole point of the fixture.
_PREDICTED_BEARING = -np.pi + 0.0127
_MEASURED_BEARING = np.pi - 0.0127


def _bearing_models():
    """A one-dimensional bearing measurement of a 2-D position."""

    def process_model(x, u, dt):
        return np.asarray(x, dtype=float)

    def process_jacobian(x, u, dt):
        return np.eye(2)

    def measurement_model(x):
        return np.array([np.arctan2(x[1], x[0])])

    def measurement_jacobian(x):
        r_sq = max(x[0] ** 2 + x[1] ** 2, 1e-12)
        return np.array([[-x[1] / r_sq, x[0] / r_sq, 0, 0]])[:, :2]

    return process_model, process_jacobian, measurement_model, measurement_jacobian


def _wrapping_innovation(z, z_pred):
    return np.array([angle_diff(z[0], z_pred[0])])


def _straddling_state():
    """A state whose predicted bearing sits just the far side of -pi."""
    return np.array([np.cos(_PREDICTED_BEARING), np.sin(_PREDICTED_BEARING)])


def _build(cls, **kwargs):
    """A predicted filter of the given class, holding the straddling state."""
    process, process_jac, measure, measure_jac = _bearing_models()
    kf = cls(
        process,
        process_jac,
        measure,
        measure_jac,
        lambda dt: 1e-6 * np.eye(2),
        lambda: np.array([[0.01]]),
        _straddling_state(),
        0.1 * np.eye(2),
        innovation_func=_wrapping_innovation,
        **kwargs,
    )
    kf.predict(dt=1.0)
    return kf


FILTERS = [ExtendedKalmanFilter, IteratedExtendedKalmanFilter]
FILTER_IDS = [cls.__name__ for cls in FILTERS]


@pytest.mark.parametrize("cls", FILTERS, ids=FILTER_IDS)
def test_get_innovation_wraps_across_the_branch_cut(cls):
    """The residual is the short way round, and points the right way."""
    kf = _build(cls)
    z = np.array([_MEASURED_BEARING])

    innovation, _ = kf.get_innovation(z)
    expected = angle_diff(_MEASURED_BEARING, _PREDICTED_BEARING)

    assert innovation[0] == pytest.approx(expected, abs=1e-12), (
        f"{cls.__name__}.get_innovation returned {innovation[0]:.4f} rad where "
        f"the wrapped difference is {expected:.4f}. A raw subtraction gives "
        f"{_MEASURED_BEARING - _PREDICTED_BEARING:.4f} -- the long way round, "
        "and with the opposite sign."
    )


@pytest.mark.parametrize("cls", FILTERS, ids=FILTER_IDS)
def test_the_raw_subtraction_would_have_been_wrong_here(cls):
    """Guard the guard: this fixture actually exercises the branch cut.

    Without this, the test above passes on any measurement whose two angles
    happen not to straddle pi -- which is most of them -- and would be
    testing nothing. It is the same reason
    `tests/ch6_dead_reckoning/test_heading_error_is_wrapped.py` carries
    `test_the_trajectory_still_exercises_the_wrap`.
    """
    raw = _MEASURED_BEARING - _PREDICTED_BEARING
    wrapped = angle_diff(_MEASURED_BEARING, _PREDICTED_BEARING)

    assert abs(raw - wrapped) == pytest.approx(2 * np.pi, abs=1e-9)
    assert np.sign(raw) != np.sign(wrapped)
    assert abs(raw) > 6.0 and abs(wrapped) < 0.05


def _predicted_steps(kf, z):
    """(realised step, step the helper predicts, step a raw residual predicts)."""
    innovation, S = kf.get_innovation(z)
    x_pred, P_pred = kf.get_state()
    H = _bearing_models()[3](x_pred)
    gain = P_pred @ H.T @ np.linalg.inv(S)
    raw = np.array([z[0] - _bearing_models()[2](x_pred)[0]])

    kf.update(z)
    x_post, _ = kf.get_state()
    return x_post - x_pred, gain @ innovation, gain @ raw


def test_get_innovation_agrees_with_what_ekf_update_actually_uses():
    """The helper and the update must not disagree about the residual.

    Checked through the state rather than by reading both: `update` moves the
    state by `K @ innovation`, so recovering the innovation from the step it
    took is independent of how either routine spells the subtraction.

    Exact only for the EKF, which linearises once. The IEKF gets the same
    check in a form its relinearisation allows, below.
    """
    kf = _build(ExtendedKalmanFilter)
    realised, from_helper, _ = _predicted_steps(kf, np.array([_MEASURED_BEARING]))

    assert np.allclose(realised, from_helper, atol=1e-9), (
        f"update moved the state by {realised}, but the innovation "
        f"get_innovation reports predicts a step of {from_helper}. The two "
        "disagree about the residual."
    )


def test_the_iekf_helper_is_the_first_iterations_innovation():
    """For the IEKF the helper is the *first* linearisation, exactly.

    `get_innovation` is a pre-update diagnostic evaluated at the prediction,
    while the IEKF's `update()` moves its linearisation point as it iterates.
    So "the helper equals what update consumed" is the wrong identity to demand
    of this class -- but it is exactly right for the first iteration, and
    `max_iterations` lets that be pinned through the class's own update path
    rather than by recomputing the innovation here, which would be circular.

    Held to machine precision: 8.9e-17 at one iteration, against 5.3e-04 once
    four iterations of relinearisation have moved the point.
    """
    kf = _build(IteratedExtendedKalmanFilter, max_iterations=1)
    realised, from_helper, _ = _predicted_steps(kf, np.array([_MEASURED_BEARING]))

    assert kf.get_last_iterations() == 1
    assert np.allclose(realised, from_helper, atol=1e-12), (
        f"at one iteration the IEKF must move by exactly K @ nu; it moved by "
        f"{realised} against a predicted {from_helper}."
    )


def test_the_iekf_update_lands_where_the_wrapped_helper_says_and_not_the_raw_one():
    """Same claim for the IEKF, which relinearises so cannot match exactly.

    `update` runs four iterations here, so `K(x_pred) @ nu(x_pred)` is only the
    first of them and the realised step differs from it by the relinearisation
    residue. That residue is 5.3e-04, against 5.7 for the step a raw
    subtraction would predict -- a factor of **10713**, which is what makes
    this a comparison rather than a tolerance. Asserting the wrapped side alone
    would pass at any loose enough bound; the raw side is what the assertion
    has to beat.
    """
    kf = _build(IteratedExtendedKalmanFilter)
    realised, from_helper, from_raw = _predicted_steps(
        kf, np.array([_MEASURED_BEARING])
    )

    err_helper = np.linalg.norm(realised - from_helper)
    err_raw = np.linalg.norm(realised - from_raw)

    assert err_helper < 1e-2, (
        f"the wrapped helper predicts a step {err_helper:.3e} from the one "
        f"update took, which is far more than the 5.3e-04 of relinearisation "
        "this scenario carries."
    )
    assert err_raw > 1.0
    assert err_raw / err_helper > 100


@pytest.mark.parametrize("cls", FILTERS, ids=FILTER_IDS)
def test_without_an_innovation_func_the_subtraction_stays_raw(cls):
    """No wrapper supplied, no wrapping -- the default must not change."""
    process, process_jac, measure, measure_jac = _bearing_models()
    kf = cls(
        process,
        process_jac,
        measure,
        measure_jac,
        lambda dt: 1e-6 * np.eye(2),
        lambda: np.array([[0.01]]),
        _straddling_state(),
        0.1 * np.eye(2),
    )
    kf.predict(dt=1.0)

    innovation, _ = kf.get_innovation(np.array([_MEASURED_BEARING]))
    raw = _MEASURED_BEARING - _PREDICTED_BEARING

    assert innovation[0] == pytest.approx(raw, abs=1e-9)
