# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pure functions for the doubly-robust (AIPW) outcome-model estimate ``μ̂_DR``.

Balance's estimand is a single **target-population mean** ``μ = E_T[Y]`` (not a
treatment-effect contrast). This module combines the outcome model ``ĝ`` from
:mod:`balance.outcome_models.outcome_model` with balance weights ``w`` to form
the augmented / one-sample AIPW (doubly-robust) estimator:

    μ̂_DR = wmean_T(ĝ(X_T), w_T)  +  wmean_S( Y − ĝ(X_S), w )

where the first term is the g-computation estimate ``μ̂_OM`` transported to the
target and the second is the IPW-weighted mean of the responders' residuals
(over responders with an observed ``Y``). It is **doubly robust**: consistent if
*either* ``ĝ`` is correct (then ``E[Y−ĝ|X]=0`` so the augmentation vanishes and
``μ̂_DR → μ̂_OM``) *or* the weights correctly reweight the responders to the
target (then the ``ĝ`` terms cancel and ``μ̂_DR → μ̂_IPW``). Equivalently, it is a
GREG (model-assisted) estimator with balance's weights as the design weights.

Note the collapse for a linear ``ĝ`` with intercept: if ``ĝ`` is fit with the
same weights ``w`` used here, ``wmean_S(Y, w) = wmean_S(ĝ(X_S), w)`` exactly, so
the augmentation is zero and ``μ̂_DR = μ̂_OM``. A non-trivial correction therefore
requires either a non-linear ``ĝ`` or a fit-weighting different from ``w`` (e.g.
an unweighted ``ĝ`` combined with non-uniform balance weights).

This module provides the **point estimate only** (see the TODOs below for
cross-fitting and honest variance/CI).

The public :meth:`balance.balance_frame.BalanceFrame.aipw` entry point enforces
the estimator's normalization contract: responder weights must come from
``adjust()`` and their total must match the target-weight total. Direct callers
of :func:`aipw_point_estimate` are responsible for supplying weights on that
same target-population scale.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from balance.outcome_models.outcome_model import predict_outcome
from balance.stats_and_plots.weighted_stats import weighted_mean

logger: logging.Logger = logging.getLogger(__package__)

_AIPW_WEIGHT_SUM_RTOL: float = 1e-6

# TODO (cross-fitting): the augmentation uses in-sample ĝ(X_S) — the model was
# fit on these same responders — which is optimistic for flexible learners. Add
# K-fold cross-fitted (out-of-fold) predictions for the residual term, and
# average ĝ^(-k)(X_T) over folds for the target term (n_folds=5 default),
# reusing learner_from_model() for the per-fold refits.
#
# TODO (variance / CI — preferred: analytic influence function): this returns
# the point estimate only. The efficient next step is an influence-function /
# sandwich SE. It MUST account for estimating BOTH nuisances (ĝ AND the balance
# weighting model) to be honest: a conditional plug-in that treats the weights
# as fixed (e.g. summing var_of_weighted_mean of the target term and the
# residual term) UNDER-COVERS, because it ignores weight- and ĝ-estimation
# uncertainty (cross-fitting makes the ĝ term asymptotically negligible; the
# weighting influence still needs the propensity model's contribution).
#   TODO (variance — bootstrap alternative): resample responders, refit ĝ AND
#   recompute the balance weights per replicate, recompute μ̂_DR, percentile CI.
#   TODO (ideal end-to-end CI — larger redesign, scope separately): bootstrap
#   the WHOLE pipeline jointly — refit the IPW/CBPS/rake weighting model and the
#   outcome model together on each resample and re-transport to the target — so
#   selection-model + outcome-model + finite-target uncertainty all propagate.
#   This spans weighting_methods + outcome_models and needs a shared resampling
#   harness; it is a prerequisite for a fully honest .summary() interval.


def _validate_aipw_weight_scale(
    sample_weight: pd.Series | np.ndarray,
    target_weight: pd.Series | np.ndarray,
) -> None:
    """Validate the same-population-scale contract for public AIPW estimates.

    Zero-valued row weights are valid (for example, uncovered cells can receive
    zero weight), but both vectors must be non-empty, one-dimensional, finite,
    non-negative, and have positive totals. Their totals must differ by less
    than the internal ``1e-6`` tolerance, relative to the target total.

    Args:
        sample_weight: Adjusted responder weights.
        target_weight: Target design weights.

    Raises:
        ValueError: If either vector or the relationship between their totals
            violates the AIPW normalization contract.
    """

    arrays: dict[str, np.ndarray] = {}
    for name, weight in (
        ("responder", sample_weight),
        ("target", target_weight),
    ):
        try:
            array = np.asarray(weight, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aipw() requires numeric {name} weights.") from exc
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"aipw() requires a non-empty, one-dimensional {name} weight vector."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"aipw() requires finite {name} weights.")
        if (array < 0).any():
            raise ValueError(f"aipw() requires non-negative {name} weights.")
        arrays[name] = array

    try:
        sample_weight_total = math.fsum(arrays["responder"])
        target_weight_total = math.fsum(arrays["target"])
    except OverflowError as exc:
        raise ValueError(
            "aipw() requires finite responder and target weight totals."
        ) from exc
    if not math.isfinite(sample_weight_total) or not math.isfinite(target_weight_total):
        raise ValueError("aipw() requires finite responder and target weight totals.")
    if sample_weight_total <= 0 or target_weight_total <= 0:
        raise ValueError("aipw() requires positive responder and target weight totals.")

    relative_difference = (
        abs(sample_weight_total - target_weight_total) / target_weight_total
    )
    if relative_difference >= _AIPW_WEIGHT_SUM_RTOL:
        raise ValueError(
            "aipw() requires adjust()-calibrated responder and target weights "
            "on the same population scale: the relative weight-total "
            f"difference is {relative_difference:.6g}, which must be less than "
            f"{_AIPW_WEIGHT_SUM_RTOL:g}. Re-run adjust(...) without changing "
            "the resulting weights."
        )


def aipw_point_estimate(
    sample_covars: pd.DataFrame,
    outcomes: pd.DataFrame,
    sample_weight: pd.Series | np.ndarray | None,
    target_covars: pd.DataFrame,
    target_weight: pd.Series | np.ndarray | None,
    model: Dict[str, Any],
) -> Dict[str, float]:
    """Doubly-robust (AIPW) point estimate ``μ̂_DR`` per outcome column.

    Replays the stored outcome model on the responder and target covariates,
    then combines the target g-computation term with the IPW-weighted responder
    residuals (over responders with an observed outcome):

        μ̂_DR[c] = wmean(ĝ_c(X_T), w_T) + wmean(Y_c − ĝ_c(X_S), w)   (observed Y)

    Args:
        sample_covars: Responder covariates ``X_S`` (row-aligned to ``outcomes``
            and ``sample_weight``).
        outcomes: Observed responder outcome(s); must contain every column in
            ``model["outcome_columns"]``. ``NaN`` rows are dropped from the
            residual term (weights realigned), matching ``fit_outcome_model``.
        sample_weight: Responder (balance) weights ``w``, or ``None`` for an
            unweighted augmentation. Direct callers must ensure that these are
            on the same population scale as ``target_weight``.
        target_covars: Target covariates ``X_T``.
        target_weight: Target weights ``w_T``, or ``None`` for a simple mean.
        model: A fitted model dict from :func:`fit_outcome_model`.

    Returns:
        Dict[str, float]: ``{outcome_column: μ̂_DR}``.
    """
    # predict_outcome scores rows positionally and the residual term below
    # indexes into np.asarray(...) of the outcomes/weights positionally, so a
    # caller whose (already row-aligned) frames carry mismatched or shuffled
    # index LABELS must not be able to silently misalign Y against ĝ(X_S).
    # Normalising the index here is a no-op for the common already-aligned path
    # (row order is untouched); it only guards against index-based drift.
    outcomes = outcomes.reset_index(drop=True)
    if isinstance(sample_weight, pd.Series):
        sample_weight = sample_weight.reset_index(drop=True)

    outcome_columns: List[str] = [str(c) for c in model["outcome_columns"]]
    preds_sample = predict_outcome(model, sample_covars)
    preds_target = predict_outcome(model, target_covars)

    sample_weight_arr = (
        None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    )

    result: Dict[str, float] = {}
    for col in outcome_columns:
        y = np.asarray(outcomes[col], dtype=float)
        yhat_sample = np.asarray(preds_sample[col], dtype=float)

        observed = ~np.isnan(y)
        if not observed.any():
            # No responder has an observed outcome, so the residual mean is
            # undefined (weighted_mean of an empty array). Emit NaN rather than
            # computing an undefined augmentation.
            logger.warning(
                "aipw_point_estimate: outcome column %r has no observed "
                "(non-NaN) responder outcomes; the residual augmentation is "
                "undefined, so μ̂_DR is set to NaN for this column.",
                col,
            )
            result[col] = float("nan")
            continue

        mu_om_target = float(
            weighted_mean(pd.Series(preds_target[col]), target_weight).iloc[0]
        )

        residuals = y[observed] - yhat_sample[observed]
        residual_weight = (
            None if sample_weight_arr is None else sample_weight_arr[observed]
        )
        augmentation = float(
            weighted_mean(
                pd.Series(residuals),
                None if residual_weight is None else pd.Series(residual_weight),
            ).iloc[0]
        )
        result[col] = mu_om_target + augmentation
    return result
