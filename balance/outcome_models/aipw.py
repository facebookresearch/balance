# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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

This module provides full-sample and K-fold cross-fitted **point estimates**.
Honest variance/CI remains deferred (see the TODO below).

The public :meth:`balance.balance_frame.BalanceFrame.aipw` entry point enforces
the estimator's normalization contract: responder weights must come from
``adjust()`` and their total must match the target-weight total. Direct callers
of :func:`aipw_point_estimate` are responsible for supplying weights on that
same target-population scale.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from balance.outcome_models.outcome_model import fit_outcome_model, predict_outcome
from balance.stats_and_plots.weighted_stats import weighted_mean
from balance.utils.input_validation import _is_discrete_series

logger: logging.Logger = logging.getLogger(__package__)

_AIPW_WEIGHT_SUM_RTOL: float = 1e-6

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


def cross_fitted_aipw_point_estimate(
    sample_covars: pd.DataFrame,
    outcomes: pd.DataFrame,
    sample_weight: pd.Series | np.ndarray,
    target_covars: pd.DataFrame,
    target_weight: pd.Series | np.ndarray,
    *,
    fit_kwargs: Dict[str, Any],
    fit_sample_weight: pd.Series | np.ndarray | None = None,
    n_folds: int = 5,
    random_seed: int = 2020,
) -> Dict[str, float]:
    """Return a K-fold cross-fitted AIPW target-mean estimate.

    Each responder prediction is produced by a model that was not trained on
    that responder.  The target prediction is the arithmetic average of the
    predictions from the ``n_folds`` nuisance fits. Fold assignment is shuffled
    and deterministic given ``random_seed``. Nuisance learners remain subject
    to their own ``random_state`` configuration.

    ``sample_weight`` always supplies the AIPW residual weights, whereas
    ``fit_sample_weight`` independently reproduces a weighted outcome-model fit.
    This distinction is important when the outcome model was fit unweighted.

    Raises:
        ValueError: If fold or seed configuration is invalid; inputs are empty,
            non-numeric, non-finite, negative, or not row-aligned; weight totals
            violate the AIPW scale contract; an outcome lacks enough observed
            rows; or a classification outcome lacks two sufficiently populated
            classes.

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.outcome_models import cross_fitted_aipw_point_estimate
        from sklearn.linear_model import LinearRegression

        sample_covars = pd.DataFrame({"x": np.arange(6, dtype=float)})
        outcomes = pd.DataFrame({"y": 1.0 + 2.0 * sample_covars["x"]})
        target_covars = pd.DataFrame({"x": [0.5, 2.5, 4.5]})
        estimates = cross_fitted_aipw_point_estimate(
            sample_covars,
            outcomes,
            sample_weight=np.ones(6),
            target_covars=target_covars,
            target_weight=np.full(3, 2.0),
            fit_kwargs={"model": LinearRegression()},
            n_folds=3,
            random_seed=7,
        )
        round(estimates["y"], 6)
        # 6.0
    """
    from sklearn.model_selection import KFold

    n_rows = len(sample_covars)
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds must be an integer greater than or equal to 2.")
    n_folds = int(n_folds)
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
        or random_seed > np.iinfo(np.uint32).max
    ):
        raise ValueError(
            "random_seed must be a non-negative integer no greater than 2**32 - 1."
        )
    random_seed = int(random_seed)
    if n_rows == 0:
        raise ValueError("cross-fitted aipw() requires at least one responder.")
    if len(outcomes) != n_rows:
        raise ValueError(
            "sample_covars and outcomes must have the same number of rows: "
            f"got {n_rows} and {len(outcomes)}."
        )
    if outcomes.shape[1] == 0:
        raise ValueError("outcomes must contain at least one outcome column.")
    if "sample_weight" in fit_kwargs:
        raise ValueError(
            "fit_kwargs must not contain sample_weight; pass fit_sample_weight "
            "to keep nuisance-fit weights separate from AIPW residual weights."
        )
    if len(target_covars) == 0:
        raise ValueError("cross-fitted aipw() requires at least one target row.")

    try:
        residual_weight = np.asarray(sample_weight, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must contain numeric values.") from exc
    try:
        target_weight_array = np.asarray(target_weight, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_weight must contain numeric values.") from exc
    if residual_weight.ndim != 1 or len(residual_weight) != n_rows:
        raise ValueError(
            "sample_weight must be one-dimensional and have one value per "
            f"responder ({n_rows}); got shape {residual_weight.shape}."
        )
    if target_weight_array.ndim != 1 or len(target_weight_array) != len(target_covars):
        raise ValueError(
            "target_weight must be one-dimensional and have one value per "
            f"target row ({len(target_covars)}); got shape {target_weight_array.shape}."
        )
    _validate_aipw_weight_scale(residual_weight, target_weight_array)

    sample_covars = sample_covars.reset_index(drop=True)
    outcomes = outcomes.reset_index(drop=True)
    try:
        fit_weight = (
            None
            if fit_sample_weight is None
            else np.asarray(fit_sample_weight, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("fit_sample_weight must contain numeric values.") from exc
    if fit_weight is not None and (fit_weight.ndim != 1 or len(fit_weight) != n_rows):
        raise ValueError(
            "fit_sample_weight must be one-dimensional and have one value per "
            f"responder ({n_rows}); got shape {fit_weight.shape}."
        )
    if fit_weight is not None and (
        not np.isfinite(fit_weight).all() or (fit_weight <= 0).any()
    ):
        raise ValueError(
            "fit_sample_weight must contain only finite, strictly positive values."
        )
    outcome_columns = [str(column) for column in outcomes.columns]
    oof_predictions = pd.DataFrame(
        index=range(n_rows), columns=outcome_columns, dtype=float
    )
    target_prediction_sum = pd.DataFrame(
        0.0, index=range(len(target_covars)), columns=outcome_columns
    )

    result: Dict[str, float] = {}
    for column in outcome_columns:
        observed_index = np.flatnonzero(outcomes[column].notna().to_numpy())
        if n_folds > len(observed_index):
            raise ValueError(
                f"n_folds ({n_folds}) cannot exceed the number of observed "
                f"responders for outcome {column!r} ({len(observed_index)})."
            )
        observed_outcome = outcomes.loc[observed_index, column]
        try:
            observed_numeric = observed_outcome.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"outcome {column!r} must contain numeric or binary values."
            ) from exc
        if not np.isfinite(observed_numeric).all():
            raise ValueError(
                f"outcome {column!r} must contain only finite observed values."
            )
        if math.fsum(residual_weight[observed_index]) <= 0:
            raise ValueError(
                f"observed responders for outcome {column!r} must have positive "
                "total AIPW residual weight."
            )
        if _is_discrete_series(observed_outcome):
            class_values = observed_outcome.to_numpy()
            classes, class_counts = np.unique(class_values, return_counts=True)
            if len(classes) != 2 or class_counts.min() < 2:
                raise ValueError(
                    f"cross-fitted classification for outcome {column!r} requires "
                    "exactly two classes with at least two observed responders "
                    f"per class; got counts {dict(zip(classes, class_counts))}."
                )
            # Distribute each shuffled class round-robin. Unlike
            # StratifiedKFold, this remains warning-free when a class has fewer
            # observations than folds, while ensuring every training fold keeps
            # at least one observation from each class.
            rng = np.random.default_rng(random_seed)
            validation_folds: list[list[int]] = [[] for _ in range(n_folds)]
            fold_offset = 0
            for class_value in classes:
                class_positions = np.flatnonzero(class_values == class_value)
                rng.shuffle(class_positions)
                for offset, position in enumerate(class_positions):
                    validation_folds[(fold_offset + offset) % n_folds].append(
                        int(position)
                    )
                fold_offset = (fold_offset + len(class_positions)) % n_folds
            fold_positions = []
            all_positions = np.arange(len(observed_index))
            for validation in validation_folds:
                validation_position = np.asarray(validation, dtype=int)
                train_position = np.setdiff1d(
                    all_positions, validation_position, assume_unique=True
                )
                fold_positions.append((train_position, validation_position))
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            fold_positions = list(splitter.split(observed_index))

        for train_position, validation_position in fold_positions:
            train_index = observed_index[train_position]
            validation_index = observed_index[validation_position]
            fold_model = fit_outcome_model(
                sample_covars.iloc[train_index].reset_index(drop=True),
                outcomes.loc[train_index, [column]].reset_index(drop=True),
                sample_weight=None if fit_weight is None else fit_weight[train_index],
                **fit_kwargs,
            )
            validation_predictions = predict_outcome(
                fold_model,
                sample_covars.iloc[validation_index].reset_index(drop=True),
            )
            oof_predictions.loc[validation_index, column] = np.asarray(
                validation_predictions[column], dtype=float
            )
            target_prediction_sum[column] += np.asarray(
                predict_outcome(fold_model, target_covars)[column], dtype=float
            )

        target_predictions = target_prediction_sum[column] / float(n_folds)
        target_term = float(weighted_mean(target_predictions, target_weight).iloc[0])
        observed = outcomes[column].notna()
        residual_term = float(
            weighted_mean(
                outcomes.loc[observed, column] - oof_predictions.loc[observed, column],
                pd.Series(residual_weight[observed.to_numpy()]),
            ).iloc[0]
        )
        result[column] = target_term + residual_term
    return result
