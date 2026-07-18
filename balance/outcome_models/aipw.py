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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from balance.outcome_models.outcome_model import predict_outcome
from balance.stats_and_plots.weighted_stats import weighted_mean

logger: logging.Logger = logging.getLogger(__package__)

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
            unweighted augmentation.
        target_covars: Target covariates ``X_T``.
        target_weight: Target weights ``w_T``, or ``None`` for a simple mean.
        model: A fitted model dict from :func:`fit_outcome_model`.

    Returns:
        Dict[str, float]: ``{outcome_column: μ̂_DR}``.
    """
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

        mu_om_target = float(
            weighted_mean(pd.Series(preds_target[col]), target_weight).iloc[0]
        )

        observed = ~np.isnan(y)
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
