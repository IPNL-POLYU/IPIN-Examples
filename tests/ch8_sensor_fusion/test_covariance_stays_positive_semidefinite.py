"""A covariance cannot be negative, and Chapter 8's filters made it so.

`AdaptiveGatingManager` inflates P by lambda after the innovation covariance S
has already been formed for the chi-square gate. Both fusion runners then built
the gain from that stale S and applied the *short form* update, K = P H' S^-1
followed by P <- (I - K H) P.

The short form equals the Joseph form only at the optimal gain. Off it, the
variance along H is scaled by

    1 - lambda * HPH' / (HPH' + R)

which is negative as soon as lambda * HPH' > HPH' + R -- exactly the situation
inflation creates. The filter then reports a negative variance, the covariance
panel of `ch8_sensor_fusion/figs/lc_uwb_imu_results.png` drew it, and nothing
in the suite objected.

Measured on the corrected datasets with the short form still in place:

    dataset                      filter   min trace(P)      negative samples
    ch8_fusion_2d_imu_uwb        LC         0.000383            0 / 6587
    ch8_fusion_2d_imu_uwb        TC         0.000384            0 / 8271
    ch8_fusion_2d_imu_uwb_nlos   LC        -0.012310           11 / 6587
    ch8_fusion_2d_imu_uwb_nlos   TC   -21032532.843825        600 / 8271

**The baseline dataset is not where this lives, and that is the trap.** Once
the accelerometer frame was corrected the filter became consistent, the gate
stopped rejecting runs of measurements, and the inflation branch fires zero
times on the baseline -- so a guard that only ran the baseline would be green
while exercising none of the code it exists to protect. The NLOS dataset is
the one with outliers, and `test_the_inflation_branch_actually_fires` pins
that it stays that way.

`core.fusion.tuning.kalman_update` is the fix: it derives S from the covariance
it is handed, so the gain can never be matched to a covariance the filter no
longer has, and it uses the Joseph form, which is a sum of two congruence
transforms and therefore positive semidefinite at *any* gain.

Author: Li-Ta Hsu
References: Chapter 8, Section 8.2; Joseph form correcting Eq. (3.19) (see docs/book_errata.md E-01)
"""

import numpy as np
import pytest

import core.fusion.adaptive as adaptive
from core.fusion import load_fusion_dataset
from core.fusion.loosely_coupled import run_lc_fusion
from core.fusion.tightly_coupled import run_tc_fusion
from core.fusion.tuning import kalman_update

#: Baseline exercises the ordinary path; NLOS is the one that trips inflation.
DATASETS = ("ch8_fusion_2d_imu_uwb", "ch8_fusion_2d_imu_uwb_nlos")

FILTERS = (("LC", run_lc_fusion), ("TC", run_tc_fusion))


def _count_inflations(fn, dataset):
    """Run a filter, returning (history, number of P inflations applied)."""
    original = adaptive.AdaptiveGatingManager.inflate_covariance
    calls = []

    def counting(self, P):
        calls.append(1)
        return original(self, P)

    adaptive.AdaptiveGatingManager.inflate_covariance = counting
    try:
        history = fn(load_fusion_dataset(f"data/sim/{dataset}"), verbose=False)
    finally:
        adaptive.AdaptiveGatingManager.inflate_covariance = original
    return history, len(calls)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("name,fn", FILTERS, ids=[n for n, _ in FILTERS])
def test_the_trace_never_goes_negative(name, fn, dataset):
    """trace(P) is a sum of variances; it cannot be below zero."""
    history, _ = _count_inflations(fn, dataset)
    trace = np.asarray(history["P_trace"], dtype=float)

    negative = int((trace < 0).sum())
    assert negative == 0, (
        f"{name} on {dataset}: {negative} of {trace.size} stored covariances "
        f"have negative trace, worst {trace.min():.6g}. A gain built from a "
        f"stale S and applied in the short form does this."
    )


@pytest.mark.parametrize("name,fn", FILTERS, ids=[n for n, _ in FILTERS])
def test_the_inflation_branch_actually_fires(name, fn):
    """Guard the guard: the NLOS run must still reach the inflation path.

    On the baseline dataset it fires zero times, so if this ever drops to
    zero the test above is green without exercising the branch that broke.
    """
    _, inflations = _count_inflations(fn, "ch8_fusion_2d_imu_uwb_nlos")
    assert inflations > 0, (
        f"{name} never inflated P on the NLOS dataset, so "
        f"test_the_trace_never_goes_negative no longer covers the path it "
        f"was written for. Find a dataset that trips the gate, or force it."
    )


def test_joseph_stays_psd_where_the_short_form_does_not():
    """The mechanism, without a dataset: an inflated P and a stale gain.

    S is formed at P0, then P is inflated by lambda, then the gain built from
    the stale S is applied. The short form drives the variance along H
    negative; the Joseph form cannot, whatever the gain.
    """
    P0 = np.diag([1.0, 1.0])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.01]])
    y = np.array([0.5])
    x = np.zeros(2)

    S_stale = H @ P0 @ H.T + R  # formed for the gate, before inflation
    P_inflated = 2.0 * P0  # AdaptiveGatingManager's lambda

    # The short form, as both runners used to spell it.
    K_stale = P_inflated @ H.T @ np.linalg.inv(S_stale)
    P_short = (np.eye(2) - K_stale @ H) @ P_inflated
    assert float(np.linalg.eigvalsh(P_short).min()) < 0.0, (
        "this test no longer reproduces the defect it guards against; the "
        "short form is supposed to go indefinite here"
    )

    # kalman_update recomputes S from the covariance it is given.
    _, P_joseph = kalman_update(x, P_inflated, y, H, R)
    assert float(np.linalg.eigvalsh(P_joseph).min()) >= 0.0, (
        f"Joseph form produced an indefinite covariance: "
        f"eigenvalues {np.linalg.eigvalsh(P_joseph)}"
    )
    np.testing.assert_allclose(P_joseph, P_joseph.T, atol=1e-12)
