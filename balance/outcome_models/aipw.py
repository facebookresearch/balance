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

This module provides full-sample and outcome-cross-fitted **point estimates**.
Honest variance and confidence intervals remain separate work.

The public :meth:`balance.balance_frame.BalanceFrame.aipw` entry point enforces
the estimator's normalization contract: responder weights must come from
``adjust()`` and their total must match the target-weight total. Direct callers
of :func:`aipw_point_estimate` are responsible for supplying weights on that
same target-population scale.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from numbers import Integral
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from balance.outcome_models.outcome_model import (
    _build_fit_matrix,
    _resolve_learner,
    _resolve_use_model_matrix,
    _validate_fit_inputs,
    fit_outcome_model,
    learner_from_model,
    predict_outcome,
)
from balance.stats_and_plots.weighted_stats import weighted_mean
from balance.utils.input_validation import _is_discrete_series
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

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


def _cross_fit_config(n_folds: int, random_seed: int) -> tuple[int, int]:
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds must be an integer greater than or equal to 2.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or not 0 <= random_seed <= np.iinfo(np.uint32).max
    ):
        raise ValueError("random_seed must be an integer between 0 and 2**32 - 1.")
    return int(n_folds), int(random_seed)


def _cross_fit_weight_array(
    weight: pd.Series | np.ndarray, length: int, name: str
) -> np.ndarray:
    try:
        array = np.asarray(weight, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric weights.") from exc
    if array.ndim != 1 or len(array) != length:
        raise ValueError(
            f"{name} must be one-dimensional with {length} row-aligned values."
        )
    return array


def _cross_fit_fold_positions(
    outcome: pd.Series, residual_weight: np.ndarray, n_folds: int, random_seed: int
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    observed = np.flatnonzero(outcome.notna().to_numpy())
    if len(observed) < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds observed responders for outcome "
            f"{outcome.name!r} ({len(observed)})."
        )
    try:
        values = outcome.iloc[observed].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Outcome {outcome.name!r} must contain numeric values."
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            f"Outcome {outcome.name!r} must contain finite observed values."
        )
    if math.fsum(residual_weight[observed]) <= 0:
        raise ValueError(
            f"Observed responders for {outcome.name!r} need positive total weight."
        )
    if not _is_discrete_series(outcome):
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        return observed, list(splitter.split(observed))
    classes, counts = np.unique(values, return_counts=True)
    if (
        not np.array_equal(classes, [0, 1])
        or counts.min() < 2
        or not set(outcome.iloc[observed].unique()).issubset({0, 1})
    ):
        raise ValueError(
            f"Cross-fitting outcome {outcome.name!r} requires binary 0/1 classes "
            "with at least two observed responders per class."
        )
    rng = np.random.default_rng(random_seed)
    validation: list[list[int]] = [[] for _ in range(n_folds)]
    offset = 0
    for value in classes:
        positions = np.flatnonzero(values == value)
        rng.shuffle(positions)
        for i, position in enumerate(positions):
            validation[(offset + i) % n_folds].append(int(position))
        offset = (offset + len(positions)) % n_folds
    all_positions = np.arange(len(observed))
    folds = [
        (np.setdiff1d(all_positions, held_out), np.asarray(held_out, dtype=int))
        for held_out in validation
    ]
    return observed, folds


def _cross_fit_settings(
    covars: pd.DataFrame, outcomes: pd.DataFrame, fit_kwargs: dict[str, Any]
) -> dict[str, Any]:
    settings = dict(fit_kwargs)
    if "sample_weight" in settings:
        raise ValueError("Pass fit_sample_weight separately from fit_kwargs.")
    _validate_fit_inputs(
        str(settings.get("na_action", "add_indicator")),
        settings.get("transformations"),
        outcomes,
        covars,
    )
    learners = [
        _resolve_learner(settings.get("model", "auto"), outcomes[col])[0]
        for col in outcomes.columns
    ]
    use_matrix = any(
        _resolve_use_model_matrix(settings.get("use_model_matrix", "auto"), learner)
        for learner in learners
    )
    settings["use_model_matrix"] = use_matrix
    state = settings.get("preprocessing_state")
    if state is None:
        state = _build_fit_matrix(
            covars,
            use_model_matrix=use_matrix,
            formula=settings.get("formula"),
            na_action=str(settings.get("na_action", "add_indicator")),
            sample_weight_arr=None,
            densify=False,
        )["preprocessing_state"]
    settings["preprocessing_state"] = deepcopy(state)
    settings["_store_training_inputs"] = False
    settings["_outcome_is_discrete"] = {
        str(col): _is_discrete_series(outcomes[col]) for col in outcomes.columns
    }
    return settings


def _cross_fit_outcome(
    covars: pd.DataFrame,
    outcomes: pd.DataFrame,
    residual_weight: np.ndarray,
    target_covars: pd.DataFrame,
    target_weight: np.ndarray,
    fit_weight: np.ndarray | None,
    settings: dict[str, Any],
    plan: tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]],
) -> float:
    observed, folds = plan
    column = str(outcomes.columns[0])
    oof = np.full(len(covars), np.nan)
    target_sum = np.zeros(len(target_covars))
    for fold, (train, held_out) in enumerate(folds):
        train_rows, score_rows = observed[train], observed[held_out]
        fold_settings: dict[str, Any] = {
            **settings,
            "_outcome_is_discrete": {column: settings["_outcome_is_discrete"][column]},
        }
        fold_outcomes = outcomes.iloc[train_rows].reset_index(drop=True)
        if settings["_outcome_is_discrete"][column]:
            fold_outcomes[column] = fold_outcomes[column].to_numpy(dtype=np.int64)
        try:
            fitted = fit_outcome_model(
                covars.iloc[train_rows].reset_index(drop=True),
                fold_outcomes,
                sample_weight=None if fit_weight is None else fit_weight[train_rows],
                **fold_settings,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cross-fitting outcome {column!r}, training fold {fold + 1}: {exc} "
                "Choose fold sizes and a learner/calibration compatible with the training data."
            ) from exc
        oof[score_rows] = predict_outcome(fitted, covars.iloc[score_rows])[column]
        target_sum += predict_outcome(fitted, target_covars)[column]
    target_term = float(
        weighted_mean(pd.Series(target_sum / len(folds)), target_weight).iloc[0]
    )
    residuals = outcomes.iloc[observed, 0].to_numpy(dtype=float) - oof[observed]
    correction = float(
        weighted_mean(pd.Series(residuals), residual_weight[observed]).iloc[0]
    )
    return target_term + correction


def cross_fitted_aipw_point_estimate(
    sample_covars: pd.DataFrame,
    outcomes: pd.DataFrame,
    sample_weight: pd.Series | np.ndarray,
    target_covars: pd.DataFrame,
    target_weight: pd.Series | np.ndarray,
    *,
    fit_kwargs: dict[str, Any],
    fit_sample_weight: pd.Series | np.ndarray | None = None,
    n_folds: int = 5,
    random_seed: int = 2020,
) -> dict[str, float]:
    """Return outcome-cross-fitted AIPW target-population means, not intervals.

    Each observed responder is predicted without its own outcome in training.
    The target term averages the fold models' predictions; residuals use the
    supplied balance weights. ``fit_sample_weight`` independently controls
    outcome fitting. Missing outcomes and formula-omitted prediction rows retain
    the existing per-outcome omission behavior.

    The shared categorical domains and feature schema use covariates only;
    each fold refits any scaler and its learners using training rows. The
    weighting model is not refit or cross-fitted. Temporary models do not retain
    training snapshots. Fold assignment is deterministic given ``random_seed``;
    learners still need their own random-state configuration.

    Args:
        sample_covars: Responder covariates, row-aligned with outcomes/weights.
        outcomes: Observed outcomes; missingness is handled independently per column.
        sample_weight: Finite non-negative residual weights on the target's scale.
        target_covars: Covariates of the target population.
        target_weight: Finite non-negative target weights with matching total.
        fit_kwargs: Outcome-fit settings, including the learner, formula and
            optional frozen preprocessing schema; not ``sample_weight``.
        fit_sample_weight: Optional original outcome-fit weights, strictly positive.
        n_folds: At least two, no more than observed rows per outcome. Defaults to five.
        random_seed: Non-negative uint32 fold seed. Defaults to 2020.

    Returns:
        One point estimate per outcome, without standard errors or confidence intervals.

    Raises:
        ValueError: For invalid folds, weights, outcomes or incompatible fold fits.

    Example:
        With six rows ``x=0..5``, ``y=1+2*x``, responder weights all one,
        target ``x=[0.5, 2.5, 4.5]`` and target weights all two, calling
        ``cross_fitted_aipw_point_estimate(X, Y, w, XT, wT,
        fit_kwargs={'model': LinearRegression()}, n_folds=3)['y']`` gives 6.0.
    """
    n_folds, random_seed = _cross_fit_config(n_folds, random_seed)
    if len(sample_covars) != len(outcomes):
        raise ValueError(
            "sample_covars and outcomes must have the same number of rows."
        )
    if not len(sample_covars) or not len(target_covars) or not outcomes.shape[1]:
        raise ValueError(
            "Cross-fitting needs non-empty responders, target and outcomes."
        )
    if not outcomes.columns.is_unique:
        raise ValueError("Cross-fitting requires unique outcome column names.")
    residual_weight = _cross_fit_weight_array(
        sample_weight, len(sample_covars), "sample_weight"
    )
    target_weight_array = _cross_fit_weight_array(
        target_weight, len(target_covars), "target_weight"
    )
    _validate_aipw_weight_scale(residual_weight, target_weight_array)
    fit_weight = None
    if fit_sample_weight is not None:
        fit_weight = _cross_fit_weight_array(
            fit_sample_weight, len(sample_covars), "fit_sample_weight"
        )
        if not np.isfinite(fit_weight).all() or (fit_weight <= 0).any():
            raise ValueError("fit_sample_weight must be finite and strictly positive.")
    covars, outcomes = sample_covars.reset_index(drop=True), outcomes.reset_index(
        drop=True
    )
    plans = {
        col: _cross_fit_fold_positions(
            outcomes[col], residual_weight, n_folds, random_seed
        )
        for col in outcomes.columns
    }
    settings = _cross_fit_settings(covars, outcomes, fit_kwargs)
    return {
        str(col): _cross_fit_outcome(
            covars,
            outcomes[[col]],
            residual_weight,
            target_covars,
            target_weight_array,
            fit_weight,
            settings,
            plans[col],
        )
        for col in outcomes.columns
    }


def _cross_fit_model_inputs(
    model: dict[str, Any], covars: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, dict[str, Any]]:
    """Recover the fitted variable selection and original fitting weights safely."""
    training = model.get("training_inputs")
    message = (
        "Cross-fitting requires the stored original outcome-fit inputs on the current "
        "responders. Refit with fit_outcome_model(...) on this frame first."
    )
    if not isinstance(training, dict):
        raise ValueError(message)
    fitted_covars, fitted_outcomes = training.get("covars"), training.get("outcomes")
    if not isinstance(fitted_covars, pd.DataFrame) or not isinstance(
        fitted_outcomes, pd.DataFrame
    ):
        raise ValueError(message)
    columns = model["outcome_columns"]
    try:
        selected_covars = covars.loc[:, fitted_covars.columns]
        selected_outcomes = outcomes.loc[:, columns]
        original_outcomes = fitted_outcomes.loc[:, columns]
    except KeyError as exc:
        raise ValueError(message) from exc
    complete = selected_outcomes.notna().all(axis=1)
    if not selected_covars.loc[complete].equals(
        fitted_covars
    ) or not selected_outcomes.loc[complete].equals(original_outcomes):
        raise ValueError(message)
    fit_weight = training.get("sample_weight") if model.get("weighted") else None
    if model.get("weighted") and not complete.all():
        raise ValueError(
            "Cross-fitting this weighted model needs original fitting weights for every "
            "responder; its stored fit omitted incomplete outcomes. Refit unweighted, "
            "or use cross_fitted_aipw_point_estimate with explicit fit_sample_weight."
        )
    if model.get("weighted") and (
        not isinstance(fit_weight, pd.Series)
        or not fit_weight.index.equals(fitted_covars.index)
    ):
        raise ValueError(message)
    return (
        selected_covars,
        selected_outcomes,
        fit_weight,
        _cross_fit_model_settings(model),
    )


def _cross_fit_model_settings(model: dict[str, Any]) -> dict[str, Any]:
    """Clone the actual learner family, unwrapping only automatic calibration."""
    learners = {
        col: (
            learner_from_model({"fit": {col: estimator}})[col]
            if model.get("calibrated") and isinstance(estimator, CalibratedClassifierCV)
            else clone(estimator)
        )
        for col, estimator in model["fit"].items()
    }
    return {
        "model": learners,
        "formula": model.get("formula"),
        "transformations": model.get("transformations"),
        "na_action": str(model.get("na_action", "add_indicator")),
        "use_model_matrix": bool(model.get("use_model_matrix", True)),
        "calibrate": bool(model.get("calibrated", False)),
        "preprocessing_state": model.get("preprocessing_state"),
    }
