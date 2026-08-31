"""
Comparison of Fingerprinting Methods

This script compares all fingerprinting methods from Chapter 5:
    - Deterministic: NN, k-NN (Eqs. 5.1-5.2)
    - Probabilistic: MAP, Posterior Mean (Eqs. 5.3-5.5)
    - Pattern Recognition: Linear Regression

Evaluates under various conditions:
    - Different noise levels
    - Multi-floor scenarios
    - Sparse vs dense reference points

Author: Li-Ta Hsu
Date: December 2024
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# `core` must come from this checkout. Running this file as a script puts
# its *chapter* directory on sys.path[0], not the repository root, so
# without this line `import core` silently resolves to whatever else is
# installed -- another clone, a stale editable install -- or fails outright
# on a fresh one. See issue #86.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval import plot_error_cdf, save_figure, show_figures_if_requested
from core.fingerprinting import (
    LinearRegressionLocalizer,
    ShadowingField,
    fit_gaussian_naive_bayes,
    k_nearest_neighbor_localize,
    load_fingerprint_database,
    maximum_a_posteriori_localize,
    nearest_neighbor_localize,
    posterior_mean_localize,
)
from core.fingerprinting.probabilistic import log_posterior

DEFAULT_DATA = "data/sim/ch5_wifi_fingerprint_grid"
DEFAULT_MULTI_DATA = "data/sim/ch5_wifi_fingerprint_multisamples"
PANEL_LABELS = "ABCDEFGHI"

#: One stroke per method, used in every line panel of the comparison figure.
#:
#: Two pairs of these six series are *arithmetically identical* on the shipped
#: single-sample database, and both coincidences are findings of this figure
#: rather than accidents of it:
#:
#: - ``MAP`` is ``NN (Euclidean)`` exactly, because one sample per reference
#:   point gives one constant sigma and a constant sigma makes Eq. (5.4) the
#:   argmin of Eq. (5.1).
#: - ``Post.Mean (k=10)`` is ``Posterior Mean`` to five significant figures,
#:   because the posterior concentrates on a handful of RPs and truncating the
#:   rest changes nothing.
#:
#: Drawn solid, each pair shows only the line drawn last: the legend names six
#: series and four are visible, which reads as a plotting bug. ``None`` means
#: solid; every other entry is a dash pattern in points. No value is nudged --
#: only the stroke changes, so a reader can still see the two curves land on
#: each other.
METHOD_DASHES = {
    "NN (Euclidean)": None,
    "k-NN (k=3)": (6, 2),
    "MAP": (2, 2),  # lies on NN (Euclidean)
    "Posterior Mean": None,
    "Post.Mean (k=10)": (1, 2),  # lies on Posterior Mean
    "Linear Regression": (6, 2, 1, 2),
}

#: One marker per method, for the panels that plot points rather than lines.
#: Drawn hollow and in descending size so a marker landing exactly on another
#: still shows both outlines.
METHOD_MARKERS = {
    "NN (Euclidean)": "o",
    "k-NN (k=3)": "s",
    "MAP": "D",
    "Posterior Mean": "^",
    "Post.Mean (k=10)": "v",
    "Linear Regression": "P",
}


def apply_method_dashes(ax, methods):
    """Give the lines already drawn on ``ax`` their per-method stroke.

    ``methods`` must be in draw order. Used after :func:`plot_error_cdf`, which
    picks its own styles from a shared cycle and cannot know that two of these
    series coincide.
    """
    for line, method in zip(ax.get_lines(), methods, strict=True):
        dash = METHOD_DASHES[method]
        if dash is None:
            line.set_linestyle("solid")
        else:
            line.set_dashes(dash)


def label_panel(ax, label):
    """Add a compact panel label so dense README-scale grids remain navigable."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )


def generate_test_queries(
    db,
    n_queries: int = 100,
    floor_id: int = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple:
    """Generate unbiased test queries via log-distance path-loss model.

    Queries are synthesised from AP positions stored in ``db.meta`` using the
    same physical propagation model that created the radio-map, so no
    localization method is structurally favoured.  When AP metadata is
    unavailable the function falls back to a train/test hold-out split of
    existing reference-point fingerprints (with additive noise).

    Args:
        db: Fingerprint database (must carry ``ap_positions`` and
            ``path_loss_model`` in ``meta`` for physics-based generation).
        n_queries: Number of query points to generate.
        floor_id: Restrict queries to a single floor.  ``None`` = random
            floors.
        noise_std: Additional Gaussian noise std (dBm), on top of the fast
            fading the propagation model already gives every sample. It is
            genuinely additional now, and small next to nothing else: a query is
            built as ``pathloss(p) - floor_attenuation + S_ap(p) + fast``, where
            ``S_ap`` is the *same spatially correlated shadowing field the radio
            map was built from*, evaluated at the query's own position, and
            ``fast`` is the per-sample term at the database's
            ``fast_fading_std_dBm`` (1.5 dB for the shipped databases).

            **This used to redraw the whole 4 dB of shadowing per query**, which
            made the query inconsistent with the map at its own position -- the
            correlation fingerprinting exists to exploit. It also made the map
            itself a table of random numbers rather than a smooth function of
            position, because the generator redrew shadowing per (RP, AP) too.
            Both are fixed; the field is drawn once per (floor, AP) and everyone
            reads it from ``db.meta['shadow_field']``.

            The decomposition, measured on the grid database with
            nearest-neighbour matching over 200 queries on floor 0:

                clean map, noiseless query                           2.18 m
                shipped map, query = pathloss + S(p)                 3.01 m
                shipped map, full query (+ fast fading)              3.82 m
                shipped map, query carrying no shadowing at all      9.34 m

            **The floor to read those against is 2.04 m**, which is
            ``sqrt(2 s^2 / 12)`` for a grid of side ``s = 5`` -- the *rms*
            distance from a uniformly placed query to the nearest node, and so
            the right statistic to compare an RMSE to. Sampled directly it is
            2.043 m over 200k uniform positions; the *mean* of the same
            distance is 1.91 m, which is a different number for a different
            question and is the one the dense dataset's survey-effort table
            uses. This block used to quote 2.27 m, which is neither.

            The 2.18 m first row is therefore not the floor but what nearest
            neighbour *achieves* on a noiseless map: 7% above the geometric
            bound, because the argmin over eight RSS channels is not quite the
            argmin over distance.

            The last row is what the old model effectively measured, and it is
            correctly bad: a query that ignores the building's shadowing is not
            a measurement taken in that building. The 3.01 m row is the one that
            matters -- it was 6.93 m before this change, against a floor of
            2.04 m, and the residual gap is the map's own 1.5 dB of fast fading
            plus the field changing between reference points 5 m apart. Over 20
            query seeds that row runs 2.95 to 3.44 m with a mean of 3.24, so
            read it as "about 3 m" rather than as three significant figures.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(query_fingerprints, true_locations, floor_ids)``.
    """
    np.random.seed(seed)

    ap_positions = db.meta.get("ap_positions")
    pl_cfg = db.meta.get("path_loss_model", {})

    if ap_positions is not None and pl_cfg:
        return _generate_queries_pathloss(
            db,
            np.asarray(ap_positions),
            pl_cfg,
            n_queries=n_queries,
            floor_id=floor_id,
            noise_std=noise_std,
        )

    return _generate_queries_holdout(
        db,
        n_queries=n_queries,
        floor_id=floor_id,
        noise_std=noise_std,
    )


def _generate_queries_pathloss(
    db,
    ap_positions,
    pl_cfg,
    *,
    n_queries,
    floor_id,
    noise_std,
    include_shadowing=True,
):
    """Physics-based query generation using the log-distance path-loss model.

    The shadowing term is read from the database rather than redrawn, so a query
    at position p carries the same ``S_ap(p)`` the radio map was built with.

    Args:
        db: Fingerprint database, carrying ``shadow_field`` in ``meta``.
        ap_positions: AP coordinates, shape (N, 3).
        pl_cfg: The database's ``path_loss_model`` block. Passing a modified
            copy is how an experiment varies one term: zeroing
            ``fast_fading_std_dBm`` gives a noiseless query.
        n_queries: Number of queries.
        floor_id: Restrict to one floor, or None for random floors.
        noise_std: Extra Gaussian noise std (dBm), beyond fast fading.
        include_shadowing: Evaluate the map's shadowing field at the query
            position. ``False`` reproduces what the old per-query redraw
            effectively measured -- a query inconsistent with the map -- and is
            here so that comparison stays runnable rather than only described.
    """
    p0 = pl_cfg.get("P0_dBm", -30.0)
    n_exp = pl_cfg.get("path_loss_exponent", 2.5)
    sigma_fast = pl_cfg.get("fast_fading_std_dBm", 1.5)
    floor_att_db = pl_cfg.get("floor_attenuation_dB", 15.0)
    floor_height = db.meta.get("floor_height", 3.0)

    # The map's own field, not a fresh draw. This one line is the whole point of
    # the change: without it the query and the map disagree about where the
    # walls are, by the full shadowing std, at every position.
    shadow_field = ShadowingField.from_meta(db.meta) if include_shadowing else None

    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        rp_locs = db.locations[mask]
        floor_ids_out = np.full(n_queries, floor_id)
    else:
        rp_locs = db.locations
        floor_ids_out = np.random.choice(db.floor_list, n_queries)

    min_x, max_x = rp_locs[:, 0].min(), rp_locs[:, 0].max()
    min_y, max_y = rp_locs[:, 1].min(), rp_locs[:, 1].max()

    true_locs = np.column_stack(
        [
            np.random.uniform(min_x, max_x, n_queries),
            np.random.uniform(min_y, max_y, n_queries),
        ]
    )

    n_aps = len(ap_positions)
    query_fingerprints = np.empty((n_queries, n_aps))

    for qi in range(n_queries):
        fid = int(floor_ids_out[qi])
        device_z = fid * floor_height + 1.5
        pos_3d = np.array([true_locs[qi, 0], true_locs[qi, 1], device_z])

        # Shadowing at this position, for every AP at once.
        if shadow_field is not None:
            shadow = shadow_field(true_locs[qi], fid)
        else:
            shadow = np.zeros(n_aps)

        for ai in range(n_aps):
            ap = np.asarray(ap_positions[ai])
            d = np.linalg.norm(pos_3d - ap)
            d = max(d, 0.1)

            rss = p0 - 10.0 * n_exp * np.log10(d)
            rss += shadow[ai]

            ap_floor = int(ap[2] / floor_height) if len(ap) > 2 else 0
            rss -= abs(fid - ap_floor) * floor_att_db

            # Fast fading: the per-sample term, and the only one a repeat visit
            # to this exact spot would redraw.
            if sigma_fast > 0:
                rss += np.random.randn() * sigma_fast

            if noise_std > 0:
                rss += np.random.randn() * noise_std

            query_fingerprints[qi, ai] = rss

    return query_fingerprints, true_locs, floor_ids_out


def _generate_queries_holdout(db, *, n_queries, floor_id, noise_std):
    """Fallback: hold-out a random subset of RPs as test queries."""
    if floor_id is not None:
        mask = db.get_floor_mask(floor_id)
        rp_locs = db.locations[mask]
        rp_features = db.features[mask]
    else:
        rp_locs = db.locations
        rp_features = db.features

    n_rps = len(rp_locs)
    indices = np.random.choice(n_rps, size=min(n_queries, n_rps), replace=True)

    true_locs = rp_locs[indices]
    query_fingerprints = rp_features[indices].copy().astype(float)

    if noise_std > 0:
        query_fingerprints += np.random.randn(*query_fingerprints.shape) * noise_std

    if floor_id is not None:
        floor_ids_out = np.full(len(true_locs), floor_id)
    else:
        floor_ids_out = np.random.choice(db.floor_list, len(true_locs))

    return query_fingerprints, true_locs, floor_ids_out


def report_what_a_repeat_survey_buys(db_single, db_multi, queries, true_locs):
    """Compare a one-visit survey against a ten-visit one, on the same queries.

    This section exists because the summary table above shows MAP and
    NN (Euclidean) scoring identically, and a reader is entitled to know whether
    that is a coincidence, a bug, or a theorem. It is a theorem, and it is a
    property of the *survey*, not of the method:

    - A single-sample database has no variance to estimate, so
      ``fit_gaussian_naive_bayes`` sets one global sigma and the Gaussian
      log-likelihood becomes a monotone function of Euclidean distance. MAP is
      then 1-NN exactly -- not approximately.
    - A multi-sample database estimates sigma per (RP, AP), so the match is
      weighted and MAP can disagree.

    What the sweep over ``min_std`` shows is the part that is not in the book:
    the sigma a repeat survey measures is the spread of repeat visits *at a
    reference point*, while the spread the likelihood needs is the disagreement
    between a query and the nearest reference point -- which also contains the
    radio map changing over the gap between them. Those are different numbers,
    1.5 dB against 2.09 dB on this grid, and ``min_std`` is the only knob that
    reaches the difference. Raising it widens the posterior honestly and, at the
    same time, floors away the per-RP sigma that made MAP interesting.

    **MAP does not beat 1-NN on the shipped survey, and on the shipped survey
    the reason is entirely estimation noise.** The sweep below prints the
    experiment that settles it: an *oracle* row that keeps the model and the
    queries fixed and replaces the estimated sigma with the true constant the
    samples were drawn from, 1.5 dB. MAP then differs from 1-NN on **0 of 200**
    queries and scores the identical RMSE -- because a constant sigma of any
    value collapses Eq. (5.4) onto Eq. (5.1), which is the theorem this whole
    section is about. So every metre MAP loses here is the ten-visit sigma
    estimate wobbling around a flat truth, and nothing else. The sweep's own
    ``min_std`` column says the same thing from the other side: raise the floor
    until the estimate is erased and the penalty goes to zero with it.

    This paragraph used to argue the opposite -- that estimation noise was "the
    obvious diagnosis" and "wrong". It is right for the data this example runs
    on, and what follows is why the mistake was easy to make.

    **A prototype that is not shipped.** Giving the fast fading a physically
    standard attenuation dependence -- RSS variance growing as SNR falls, mean
    sigma held at 1.5 dB so the total noise is unchanged -- makes MAP
    monotonically worse as the dependence steepens, and *there* the mechanism is
    not estimation noise. In that model the spatial mismatch (a query stands
    between reference points, so it disagrees with the nearest one by the radio
    map's change over the gap) is **anti-correlated** with the per-visit sigma,
    ``corr = -0.34``, because the path-loss gradient
    ``d(pathloss)/dd = -10 n / (d ln 10)`` is steepest close to the AP -- exactly
    where the signal is strongest and, under that model, the fading quietest. So
    Naive Bayes up-weights the APs whose unmodelled spatial error is worst.

    **None of that describes the shipped datasets**, whose fast fading is one
    constant for the whole building. Measured on the shipped multisample survey
    the same correlation is ``+0.0156`` -- zero, as it must be: a constant has
    nothing to anti-correlate with. Read the prototype as a conditional -- *if
    sigma_fast varied with attenuation, then Eq. (5.6) would be actively
    harmful* -- and not as an account of the numbers printed below.

    The spatial mismatch itself is shipped-verifiable and is worth carrying,
    because it sets the ceiling on what any per-visit sigma could buy. It is the
    rms change of the noiseless map between a query and its nearest reference
    point, and it grows with the survey grid::

        survey        spacing   spatial mismatch    (measured on shipped data)
        dense            2 m         0.54 dB
        baseline         5 m         1.43 dB
        sparse          10 m         2.98 dB

    At 5 m that term is already comparable to the entire 1.5 dB fast-fading
    budget, which is the honest statement for the chapter: **Eq. (5.6) pays only
    when the variability it models dominates the variability it does not.** That
    happens on a grid dense relative to the radio map's correlation length, or
    with a likelihood whose sigma includes the interpolation term rather than
    only the per-visit one. On a 5 m grid with 8 APs, it does not, and the
    chapter is better for saying so than for tuning until a table agrees.

    The ``MAP - NN`` column that used to sit beside those three spacings is
    gone. Only one multisample survey ships -- the 5 m one -- so a dense and a
    sparse repeat survey had to be generated to produce it, under the
    distance-dependent-sigma prototype above. Three numbers measured elsewhere
    in a table whose other column is measured here is the mixture this
    repository has been bitten by before.

    Args:
        db_single: Single-sample database (one visit per RP).
        db_multi: Multi-sample database of the same building.
        queries: Query fingerprints, shape (n, N).
        true_locs: True query positions, shape (n, 2).
    """
    print("\n" + "=" * 70)
    print("WHAT A REPEAT SURVEY BUYS (why MAP == NN above)")
    print("=" * 70)

    def characterise(db, min_std, model=None):
        if model is None:
            model = fit_gaussian_naive_bayes(db, min_std=min_std)
        differ = 0
        max_weight, effective, err_map, err_nn = [], [], [], []
        for query, true_loc in zip(queries, true_locs, strict=True):
            x_map = maximum_a_posteriori_localize(query, model, floor_id=0)
            x_nn = nearest_neighbor_localize(query, db, metric="euclidean", floor_id=0)
            if not np.allclose(x_map, x_nn):
                differ += 1
            weights = np.exp(log_posterior(query, model, floor_id=0))
            max_weight.append(weights.max())
            # Participation ratio: how many RPs the posterior really spreads
            # over. 1.0 is a delta; the count of RPs is uniform.
            effective.append(1.0 / np.sum(weights**2))
            err_map.append(np.linalg.norm(x_map - true_loc))
            err_nn.append(np.linalg.norm(x_nn - true_loc))
        return {
            "model": model,
            "differ": differ,
            "max_weight": float(np.median(max_weight)),
            "effective": float(np.median(effective)),
            "rmse_map": float(np.sqrt(np.mean(np.array(err_map) ** 2))),
            "rmse_nn": float(np.sqrt(np.mean(np.array(err_nn) ** 2))),
        }

    n = len(queries)
    single = characterise(db_single, min_std=2.0)
    print(f"\nSingle-sample survey ({db_single.n_reference_points} RPs, 1 visit each)")
    print(f"  {single['model']}")
    print(f"  {single['model'].sigma_summary()}")
    print(f"  MAP differs from 1-NN on {single['differ']}/{n} queries")
    print(f"  median max posterior weight  {single['max_weight']:.4f}")
    print(f"  median effective RP count    {single['effective']:.2f}")
    print(f"  RMSE  MAP {single['rmse_map']:.2f} m   NN {single['rmse_nn']:.2f} m")

    print(
        f"\nMulti-sample survey ({db_multi.n_reference_points} RPs, "
        f"{db_multi.n_samples_per_rp} visits each)"
    )
    print(
        f"  {'min_std':>7}  {'sigma range':>14}  {'MAP!=NN':>9}  "
        f"{'max wt':>7}  {'eff RPs':>7}  {'MAP':>7}  {'NN':>7}"
    )
    for min_std in (0.5, 1.0, 1.5, 2.0, 2.5):
        row = characterise(db_multi, min_std=min_std)
        stds = row["model"].stds
        print(
            f"  {min_std:7.1f}  {stds.min():5.2f} - {stds.max():5.2f}  "
            f"{row['differ']:5d}/{n:<3d}  {row['max_weight']:7.4f}  "
            f"{row['effective']:7.2f}  {row['rmse_map']:5.2f} m  "
            f"{row['rmse_nn']:5.2f} m"
        )

    # The experiment that says what MAP's penalty above is made of. Same
    # database, same queries, same estimator -- only sigma is replaced, by the
    # true constant the samples were drawn from. If any of the gap were the
    # weighting exploiting real structure, the oracle would keep some of it.
    oracle_sigma = float(db_multi.meta["path_loss_model"]["fast_fading_std_dBm"])
    fitted = fit_gaussian_naive_bayes(db_multi, min_std=0.5)
    oracle_model = dataclasses.replace(
        fitted, stds=np.full_like(fitted.stds, oracle_sigma)
    )
    oracle = characterise(db_multi, min_std=None, model=oracle_model)
    print(
        f"  {'oracle':>7}  {oracle_sigma:5.2f} - {oracle_sigma:5.2f}  "
        f"{oracle['differ']:5d}/{n:<3d}  {oracle['max_weight']:7.4f}  "
        f"{oracle['effective']:7.2f}  {oracle['rmse_map']:5.2f} m  "
        f"{oracle['rmse_nn']:5.2f} m"
    )
    print(
        f"\n  The oracle row substitutes the true sigma these samples were drawn "
        f"with\n"
        f"  ({oracle_sigma} dB, one constant for the whole building) for the "
        f"estimated one.\n"
        f"  MAP then differs from 1-NN on {oracle['differ']}/{n} queries and "
        f"scores {oracle['rmse_map']:.2f} m against NN's "
        f"{oracle['rmse_nn']:.2f} m:\n"
        f"  every metre MAP loses in the rows above is the ten-visit sigma "
        f"estimate\n"
        f"  wobbling around a flat truth. A constant sigma of ANY value is 1-NN "
        f"-- that\n"
        f"  is the theorem, and the oracle row is it, run rather than asserted."
    )

    print(
        "\n  Read the two ends against each other. A floor below the measured "
        "1.5 dB\n"
        "  lets the estimated sigma vary, so MAP genuinely stops being 1-NN -- "
        "and\n"
        "  is slightly worse, because with one fast-fading std for the whole\n"
        "  building that variation is estimation noise rather than signal. A "
        "floor\n"
        "  near the 2.09 dB a query really disagrees by gives the honest "
        "posterior\n"
        "  width and the better fix, at the cost of flooring the per-RP sigma "
        "away.\n"
        "  Neither column is the answer on its own; the gap between them is."
    )


def per_query_operation_counts(db, floor_id=None):
    """Count the work one query costs each method, exactly as implemented.

    This is what the cost figures plot, in place of the wall-clock timings they
    used to. A measured millisecond cannot be committed to a figure: it differs
    on every run and on every machine, so the figure churned on every
    regeneration while telling a reader nothing about their own hardware. How
    much work each method does per query is exact, reproducible, and the thing
    the chapter is actually comparing.

    One operation is counted per elementary term the implementation evaluates:

        - one per (RP, AP) pair whose distance or Gaussian log-density is formed
        - one per RP examined by the estimator step: argmin/argmax, exp, or one
          term of the posterior-weighted sum
        - one per multiply-add in the linear model's matrix-vector product

    Two implementation details dominate the result and are worth reading off the
    figure:

        - The floor constraint is applied at different points.
          `nearest_neighbor_localize` and `k_nearest_neighbor_localize` slice
          the database by floor *before* computing distances, so they only touch
          that floor's RPs. `log_likelihood` evaluates every RP in the database
          and masks the other floors to -inf afterwards. Here that is one
          floor's RPs against all three.
        - `top_k` does not avoid the dominant term. `posterior_mean_localize`
          computes the full posterior over every RP first and only then
          truncates the weighted sum, so top-k trims an O(M*d) step while the
          O(M*N) likelihood stands.

    Args:
        db: Fingerprint database being queried.
        floor_id: Floor constraint passed to the localisers, or None for all.

    Returns:
        Dict mapping method name to its per-query operation count (int).
    """
    n_all = db.n_reference_points
    n_features = db.features.shape[1]
    n_dims = db.locations.shape[1]

    # Distances are computed over the floor-filtered set; likelihoods are not.
    if floor_id is not None:
        n_searched = int(np.sum(db.get_floor_mask(floor_id)))
    else:
        n_searched = n_all

    distance_terms = n_searched * n_features
    likelihood_terms = n_all * n_features

    return {
        # distances + argmin
        "NN (Euclidean)": distance_terms + n_searched,
        # distances + partial selection + weighted average of k locations
        "k-NN (k=3)": distance_terms + n_searched + 3 * n_dims,
        # log-densities + argmax
        "MAP": likelihood_terms + n_all,
        # log-densities + exp + full posterior-weighted sum
        "Posterior Mean": likelihood_terms + n_all + n_all * n_dims,
        # log-densities + exp + partial selection + weighted sum of k locations
        "Post.Mean (k=10)": likelihood_terms + 2 * n_all + 10 * n_dims,
        # W z + b: one multiply-add per weight, plus the bias
        "Linear Regression": n_dims * n_features + n_dims,
    }


def evaluate_scenario(scenario_name, db, queries, true_locs, floor_id=None):
    """
    Evaluate all methods on a specific scenario.

    Returns:
        List of result dictionaries.
    """
    ops_per_query = per_query_operation_counts(db, floor_id=floor_id)

    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'=' * 70}")

    results = []

    # Deterministic methods
    print("\nDeterministic Methods (Eqs. 5.1-5.2):")

    # NN
    method_name = "NN (Euclidean)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = nearest_neighbor_localize(
            query, db, metric="euclidean", floor_id=floor_id
        )
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Deterministic",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    # k-NN
    method_name = "k-NN (k=3)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = k_nearest_neighbor_localize(
            query,
            db,
            k=3,
            metric="euclidean",
            weighting="inverse_distance",
            floor_id=floor_id,
        )
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Deterministic",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    # Probabilistic methods
    print("\nProbabilistic Methods (Eqs. 5.3-5.5):")

    # Train Bayesian model.
    #
    # min_std = 2.0 dBm is not an arbitrary floor. On a single-sample database
    # it is the *entire* likelihood model, so it should be the spread a query
    # actually has about the reference point it is matched to -- measured at
    # 2.09 dB on this grid. That is larger than the 1.5 dB of fast fading a
    # repeat survey would measure, because a query stands between reference
    # points and the radio map changes over the gap. See
    # fit_gaussian_naive_bayes' docstring.
    print("  Training Bayesian model...", end=" ", flush=True)
    model_bayes = fit_gaussian_naive_bayes(db, min_std=2.0)
    print("Done")

    # MAP
    method_name = "MAP"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = maximum_a_posteriori_localize(query, model_bayes, floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Probabilistic",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    # Posterior Mean (Full)
    method_name = "Posterior Mean"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = posterior_mean_localize(query, model_bayes, floor_id=floor_id)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Probabilistic",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    # Posterior Mean (Top-k) - Book guidance: typically sufficient
    method_name = "Post.Mean (k=10)"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = posterior_mean_localize(
            query, model_bayes, floor_id=floor_id, top_k=10
        )
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Probabilistic",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    # Pattern Recognition
    print("\nPattern Recognition:")

    # Train Linear Regression
    print("  Training Linear Regression...", end=" ", flush=True)
    model_lr = LinearRegressionLocalizer.fit(db, floor_id=floor_id, regularization=1.0)
    print("Done")

    method_name = "Linear Regression"
    print(f"  {method_name}...", end=" ", flush=True)
    errors, times = [], []
    for query, true_loc in zip(queries, true_locs, strict=True):
        t_start = time.perf_counter()
        est_loc = model_lr.predict(query)
        t_end = time.perf_counter()
        errors.append(np.linalg.norm(est_loc - true_loc))
        times.append((t_end - t_start) * 1000)

    results.append(
        {
            "method": method_name,
            "category": "Pattern Recognition",
            "errors": np.array(errors),
            "times": np.array(times),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "mean_time_ms": np.mean(times),
            "ops_per_query": ops_per_query[method_name],
        }
    )
    print(f"RMSE={results[-1]['rmse']:.2f}m")

    return results


def main():
    """Run comprehensive comparison of fingerprinting methods."""
    # Parse arguments before doing any work, so --help answers instead of
    # running the whole demonstration.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Fingerprint database directory (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--multi-data",
        default=DEFAULT_MULTI_DATA,
        help=(
            "Multi-sample fingerprint database for the repeat-survey comparison "
            f"(default: {DEFAULT_MULTI_DATA})"
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Chapter 5: Fingerprinting Methods Comparison")
    print("=" * 70)

    # Load database
    print("\nLoading fingerprint database...")
    db_path = Path(args.data)
    db = load_fingerprint_database(db_path)
    print(f"Database: {db}")

    all_results = {}

    # Scenario 1: Baseline (low noise, single floor)
    #
    # The three scenario labels name what the sweep varies and the term it
    # sits on top of. That second term is **fast fading at 1.5 dB**, the
    # per-sample part of the noise model -- not the 4 dB of shadowing these
    # labels used to name. Shadowing is a property of the location: the query
    # carries the same S_ap(p) the radio map was built with, so it is *shared*
    # with the map rather than added on top of it, and naming it here
    # overstated the unlabelled half by nearly a factor of three.
    print("\n" + "=" * 70)
    print("SCENARIO 1: Baseline (low noise, single floor)")
    print("=" * 70)

    queries1, true_locs1, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=1.0, seed=42
    )
    all_results["Baseline"] = evaluate_scenario(
        "Baseline (extra sigma=1dBm on top of 1.5dBm fast fading, Floor 0)",
        db,
        queries1,
        true_locs1,
        floor_id=0,
    )

    # Scenario 2: Moderate noise
    print("\n" + "=" * 70)
    print("SCENARIO 2: Moderate Noise")
    print("=" * 70)

    queries2, true_locs2, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=2.0, seed=43
    )
    all_results["Moderate Noise"] = evaluate_scenario(
        "Moderate (extra sigma=2dBm on top of 1.5dBm fast fading, Floor 0)",
        db,
        queries2,
        true_locs2,
        floor_id=0,
    )

    # Scenario 3: High noise
    print("\n" + "=" * 70)
    print("SCENARIO 3: High Noise")
    print("=" * 70)

    queries3, true_locs3, _ = generate_test_queries(
        db, n_queries=200, floor_id=0, noise_std=5.0, seed=44
    )
    all_results["High Noise"] = evaluate_scenario(
        "High (extra sigma=5dBm on top of 1.5dBm fast fading, Floor 0)",
        db,
        queries3,
        true_locs3,
        floor_id=0,
    )

    # Why MAP and NN agree above: it is the survey, not the method.
    db_multi = load_fingerprint_database(Path(args.multi_data))
    report_what_a_repeat_survey_buys(db, db_multi, queries1, true_locs1)

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)

    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        print(
            f"{'Method':<20} {'Category':<20} {'RMSE (m)':<12} {'Median (m)':<12} {'P90 (m)':<12} {'Time (ms)':<12}"
        )
        print("-" * 90)
        for r in results:
            print(
                f"{r['method']:<20} {r['category']:<20} {r['rmse']:<12.2f} "
                f"{r['median']:<12.2f} {r['p90']:<12.2f} {r['mean_time_ms']:<12.3f}"
            )

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating comparison visualizations...")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 13))

    # Plot 1: RMSE comparison across scenarios
    ax1 = plt.subplot(3, 3, 1)
    methods = [r["method"] for r in all_results["Baseline"]]
    x = np.arange(len(methods))
    width = 0.25

    for i, (scenario_name, results) in enumerate(all_results.items()):
        rmses = [r["rmse"] for r in results]
        ax1.bar(x + i * width, rmses, width, label=scenario_name, alpha=0.8)

    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("RMSE Comparison Across Scenarios")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    label_panel(ax1, PANEL_LABELS[0])

    # Plot 2: Error CDF (Baseline scenario)
    ax2 = plt.subplot(3, 3, 2)
    plot_error_cdf(
        {r["method"]: r["errors"] for r in all_results["Baseline"]},
        title="Error CDF (Baseline)",
        ax=ax2,
        title_fontweight="normal",
    )
    # Same strokes as panel E, for the same reason: NN and MAP draw one curve
    # between them, and so do the two posterior means. Solid, this panel shows
    # four lines for six legend entries.
    apply_method_dashes(ax2, [r["method"] for r in all_results["Baseline"]])
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 15)
    label_panel(ax2, PANEL_LABELS[1])

    # Plot 3: Computation time comparison
    ax3 = plt.subplot(3, 3, 3)
    methods = [r["method"] for r in all_results["Baseline"]]
    ops = [r["ops_per_query"] for r in all_results["Baseline"]]
    colors = ["blue", "cyan", "red", "orange", "green", "purple"]
    ax3.barh(methods, ops, color=colors[: len(methods)], alpha=0.7)
    ax3.set_xscale("log")
    ax3.set_xlabel("Operations per query")
    ax3.set_title("Per-Query Cost (Baseline)")
    ax3.grid(True, alpha=0.3, axis="x")
    label_panel(ax3, PANEL_LABELS[2])

    # Plot 4: Box plot comparison (Baseline)
    ax4 = plt.subplot(3, 3, 4)
    error_data = [r["errors"] for r in all_results["Baseline"]]
    bp = ax4.boxplot(error_data, tick_labels=methods, patch_artist=True)
    # One light shade per entry of `colors` above, in the same order. It was a
    # shade short, so the sixth method's box kept matplotlib's default facecolor
    # and read as deliberately highlighted among five pastel ones. strict=True
    # below is what stops the two lists drifting apart again.
    colors_box = [
        "lightblue",
        "lightcyan",
        "lightcoral",
        "lightsalmon",
        "lightgreen",
        "plum",
    ]
    for patch, color in zip(bp["boxes"], colors_box, strict=True):
        patch.set_facecolor(color)
    ax4.set_ylabel("Positioning Error (m)")
    ax4.set_title("Error Distribution (Baseline)")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")
    label_panel(ax4, PANEL_LABELS[3])

    # Plot 5: Robustness to noise (RMSE vs noise std)
    ax5 = plt.subplot(3, 3, 5)
    noise_levels = [1.0, 2.0, 5.0]
    scenario_names = ["Baseline", "Moderate Noise", "High Noise"]

    # Distinct dash patterns from METHOD_DASHES, because two pairs of these
    # series lie exactly on top of each other and solid lines would show only
    # the last one drawn. The table is shared with panels B and I so a method
    # keeps one stroke across the whole figure.
    for method in methods:
        rmses = []
        for scenario_name in scenario_names:
            method_result = [
                r for r in all_results[scenario_name] if r["method"] == method
            ][0]
            rmses.append(method_result["rmse"])
        (line,) = ax5.plot(
            noise_levels, rmses, "o-", label=method, linewidth=2, markersize=6
        )
        dash = METHOD_DASHES[method]
        if dash is not None:
            line.set_dashes(dash)

    ax5.set_xlabel("RSS Noise Std (dBm)")
    ax5.set_ylabel("RMSE (m)")
    ax5.set_title("Robustness to Measurement Noise")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    label_panel(ax5, PANEL_LABELS[4])

    # Plot 6: Cost vs Accuracy (Baseline)
    #
    # Labelled by legend, not by annotate(). Every point used the same (5, 5)
    # offset, and MAP, Posterior Mean and Post.Mean (k=10) sit at almost the
    # same cost and RMSE, so three labels printed on top of each other and
    # none of them could be read. That the points coincide is the panel's
    # actual finding, so it should not be what breaks the labelling. Colours
    # match the per-query cost panel above, so a method reads the same in both.
    ax6 = plt.subplot(3, 3, 6)
    # Hollow, distinctly shaped and in descending size: the two posterior means
    # sit at almost the same cost and the same RMSE, so filled circles of one
    # size drew one point where there are two. An outline nested inside another
    # outline says "these coincide" where a single disc says "one method".
    sizes = np.linspace(300, 130, len(all_results["Baseline"]))
    for r, color, size in zip(all_results["Baseline"], colors, sizes, strict=True):
        ax6.scatter(
            r["ops_per_query"],
            r["rmse"],
            s=size,
            facecolors="none",
            edgecolors=color,
            linewidths=2.2,
            marker=METHOD_MARKERS[r["method"]],
            label=r["method"],
        )
    ax6.legend(fontsize=7)
    ax6.set_xscale("log")
    ax6.set_xlabel("Operations per query")
    ax6.set_ylabel("RMSE (m)")
    ax6.set_title("Cost vs Accuracy Trade-off")
    ax6.grid(True, alpha=0.3)
    label_panel(ax6, PANEL_LABELS[5])

    # Plot 7: Category comparison
    ax7 = plt.subplot(3, 3, 7)
    categories = ["Deterministic", "Probabilistic", "Pattern Recognition"]
    cat_rmses = {}
    for cat in categories:
        cat_methods = [r for r in all_results["Baseline"] if r["category"] == cat]
        cat_rmses[cat] = [r["rmse"] for r in cat_methods]

    positions = [1, 2, 3]
    # The category count goes in the tick label, and every method is drawn as a
    # point on top of its box. "Pattern Recognition" holds one method, and a
    # box plot of one value is a bare horizontal line that reads as missing
    # data rather than as a category with a single member.
    bp = ax7.boxplot(
        cat_rmses.values(),
        positions=positions,
        tick_labels=[f"{cat}\n(n={len(vals)})" for cat, vals in cat_rmses.items()],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightyellow")
    for position, values in zip(positions, cat_rmses.values(), strict=True):
        ax7.scatter(
            np.full(len(values), position),
            values,
            s=40,
            color="black",
            zorder=3,
            label="one method" if position == 1 else None,
        )
    # Not "upper left": that is where label_panel puts the panel letter, and the
    # legend covered it.
    ax7.legend(fontsize=7, loc="center left")
    ax7.set_ylabel("RMSE (m)")
    ax7.set_title("Performance by Category")
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha="right")
    ax7.grid(True, alpha=0.3, axis="y")
    label_panel(ax7, PANEL_LABELS[6])

    # Plot 8: Percentile comparison
    ax8 = plt.subplot(3, 3, 8)
    x = np.arange(len(methods))
    p50 = [r["median"] for r in all_results["Baseline"]]
    p90 = [r["p90"] for r in all_results["Baseline"]]
    width = 0.35
    ax8.bar(x - width / 2, p50, width, label="Median (P50)", alpha=0.8)
    ax8.bar(x + width / 2, p90, width, label="P90", alpha=0.8)
    ax8.set_ylabel("Error (m)")
    ax8.set_title("Median vs P90 Errors")
    ax8.set_xticks(x)
    ax8.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis="y")
    label_panel(ax8, PANEL_LABELS[7])

    # Plot 9: Summary radar chart
    ax9 = plt.subplot(3, 3, 9, projection="polar")

    # Normalize metrics for radar chart
    baseline_results = all_results["Baseline"]
    metrics = ["RMSE", "Median", "P90"]

    # Select 3 representative methods
    selected_methods = ["NN (Euclidean)", "MAP", "Linear Regression"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for method_name in selected_methods:
        method_result = [r for r in baseline_results if r["method"] == method_name][0]
        values = [
            method_result["rmse"] / 10,  # Normalize
            method_result["median"] / 10,
            method_result["p90"] / 15,
        ]
        values += values[:1]
        (line,) = ax9.plot(angles, values, "o-", linewidth=2, label=method_name)
        # NN and MAP trace the same triangle here -- all three metrics are the
        # same number -- so without the shared stroke table this panel shows two
        # polygons for three legend entries. The fill is left off the pair for
        # the same reason: two identical translucent triangles read as one
        # darker one.
        dash = METHOD_DASHES[method_name]
        if dash is not None:
            line.set_dashes(dash)
        else:
            ax9.fill(angles, values, alpha=0.15)

    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics)
    ax9.set_ylim(0, 1)
    ax9.set_title("Performance Profile\n(Normalized)", pad=20)
    ax9.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax9.grid(True)
    label_panel(ax9, PANEL_LABELS[8])

    plt.tight_layout()

    # Save (svg + pdf + png via the shared layer)
    paths = save_figure(fig, Path(__file__).parent / "figs", "comparison_all_methods")
    print(f"Saved: {paths[0]}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print("  1. Speed: Linear Regression >> NN > k-NN ~= MAP ~= Posterior Mean")
    print("  2. Accuracy (low noise): Probabilistic ~= k-NN > NN > Linear Reg")
    print("  3. Robustness: k-NN and Posterior Mean most stable with noise")
    print("  4. Smoothness: Posterior Mean > k-NN > Linear Reg > MAP ~= NN")
    print("  5. Training: Linear Reg requires training, others just use database")

    # Two pairs of these six series coincide, so the CDF panel draws four
    # curves for six legend entries and a reader has no way to tell a
    # coincidence from a plotting bug. Both are real and both are findings, so
    # they are named here and given distinct strokes in the figure rather than
    # left to overlap. Computed, not asserted, so neither can go stale if the
    # database or the noise changes.
    by_name = {r["method"]: r for r in all_results["Baseline"]}
    full, topk = by_name["Posterior Mean"], by_name["Post.Mean (k=10)"]
    nn, map_ = by_name["NN (Euclidean)"], by_name["MAP"]
    gap = abs(full["rmse"] - topk["rmse"])
    map_gap = abs(nn["rmse"] - map_["rmse"])
    print(
        f"  6. Two of the six curves are drawn twice, for two different "
        f"reasons, so panels B, E, F and I dash them apart rather than let one "
        f"hide the other:\n"
        f"     (a) MAP scores {map_['rmse']:.4f} m and NN {nn['rmse']:.4f} m, a "
        f"difference of {map_gap * 1000:.2f} mm. That is the theorem, not a "
        f"tie: one visit per RP gives one constant sigma, and a constant sigma "
        f"makes Eq. (5.4) the argmin of Eq. (5.1). Survey each point twice and "
        f"they part company -- see 'What a repeat survey buys' above.\n"
        f"     (b) The full posterior mean scores {full['rmse']:.4f} m and the "
        f"top-10 {topk['rmse']:.4f} m, a difference of {gap * 1000:.2f} mm. The "
        f"posterior concentrates on a handful of RPs, so dropping the rest "
        f"costs nothing in accuracy -- but it saves little either, only "
        f"{full['ops_per_query'] / topk['ops_per_query']:.2f}x, because "
        f"posterior_mean_localize evaluates the likelihood at every RP before "
        f"it truncates. Truncating the weighted sum trims O(M*d) while the "
        f"O(M*N) likelihood stands."
    )
    print("\nRecommendations:")
    print("  - Real-time apps: Use NN or Linear Regression for speed")
    print("  - High accuracy: Use k-NN or Bayesian methods")
    print("  - Noisy environments: k-NN with k=3-5 or Posterior Mean")
    print("  - Dense RPs: NN sufficient")
    print("  - Sparse RPs: k-NN or Linear Regression for interpolation")
    show_figures_if_requested()


if __name__ == "__main__":
    main()
