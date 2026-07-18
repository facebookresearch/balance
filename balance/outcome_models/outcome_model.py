# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pure DataFrame functions for fitting and applying outcome models.

This module is the ``outcome_models/`` counterpart to ``weighting_methods/``:
it fits a regressor/classifier of an observed *outcome* on covariates and
produces a stored model dict plus predictions, mirroring the IPW
fit-store-replay conventions (:mod:`balance.weighting_methods.ipw`).

It contains **pure functions on ``pandas`` objects only** — there is no
``SampleFrame``/``BalanceFrame`` wiring here (that lives in a later change).
The estimator ``ĝ(X) ≈ E[Y|X]`` is fit on the responders' covariates and the
observed outcome; applying it to new (target) covariates yields the
``outcomes_hat`` predictions whose weighted mean is the g-computation
(outcome-model) population estimate ``μ̂_OM``.

Design notes:

* Preprocessing is **learner-dependent** (``use_model_matrix="auto"``): tree /
  boosting learners use the native-categorical path (no scaler) when
  scikit-learn >= 1.4 is available (``categorical_features="from_dtype"``), and
  fall back to a one-hot + :class:`~sklearn.preprocessing.StandardScaler` path
  on scikit-learn < 1.4; linear learners always use the one-hot + scaler path.
* :class:`~sklearn.ensemble.HistGradientBoostingRegressor` /
  :class:`~sklearn.ensemble.HistGradientBoostingClassifier` reject sparse input,
  so the design matrix is densified for them (reusing
  :func:`balance.weighting_methods.ipw._convert_to_dense_array`).
* The native-categorical path stores the fit-time category levels
  (``categorical_levels``) and re-applies them on replay so novel/missing target
  categories do not shift the integer codes the learner sees.

Example:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.outcome_models import fit_outcome_model, predict_outcome

        rng = np.random.default_rng(0)
        covars_R = pd.DataFrame(
            {"age": rng.normal(50, 10, 200), "grp": rng.choice(["a", "b"], 200)}
        )
        outcomes_R = pd.DataFrame({"happiness": covars_R["age"] + rng.normal(0, 1, 200)})
        w_R = pd.Series(np.ones(200))

        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        model["method"]  # -> "outcome_model"

        covars_T = pd.DataFrame(
            {"age": rng.normal(55, 10, 5), "grp": rng.choice(["a", "b"], 5)}
        )
        predict_outcome(model, covars_T)["happiness"][:3]  # -> ŷ on new data
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_stats import weighted_mean, weighted_r2
from balance.typing import OutcomeLearner
from balance.utils.input_validation import (
    _assert_type,
    _check_weighting_methods_input,
    _is_discrete_series,
)
from balance.utils.model_matrix import build_design_matrix
from balance.weighting_methods.ipw import _compute_deviance, _convert_to_dense_array
from scipy.sparse import issparse
from sklearn.base import clone, is_classifier
from sklearn.utils.validation import has_fit_parameter

logger: logging.Logger = logging.getLogger(__package__)

# The categorical NA sentinel that add_na_indicator appends on the raw path (its
# replace_val_obj default in balance.utils.data_transformation). Fit-time category
# levels capture it; the replay's add_na_indicator re-adds it, so it must be
# stripped from restored levels to avoid a double-add of the category.
_NA_CATEGORY_SENTINEL: str = "_NA"

# Reserved keys for a type-dispatch ``model`` map, e.g.
# ``{"_discrete": clf, "_continuous": reg}``. The leading underscore keeps them
# from colliding with real outcome-column names (any other keys are treated as a
# ``{outcome_column: estimator}`` column-name map).
_MODEL_KEY_DISCRETE: str = "_discrete"
_MODEL_KEY_CONTINUOUS: str = "_continuous"


def _has_sklearn_1_4() -> bool:
    """Return True when scikit-learn >= 1.4 (native-categorical support)."""
    # Parse the version without the (undeclared) ``packaging`` dependency; this
    # mirrors ``balance.testutil._has_sklearn_1_4`` so the two agree.
    try:
        import sklearn

        return tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4)
    except Exception:
        return False


def _auto_estimator(discrete: bool) -> Any:
    """Return a fresh ``HistGradientBoosting`` estimator for the outcome type.

    Classifier for a discrete (binary 0/1) outcome, regressor for a continuous
    one; configured with ``categorical_features="from_dtype"`` on scikit-learn
    >= 1.4 so it can consume the native-categorical design matrix.
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    kwargs: Dict[str, Any] = {"random_state": 2020}
    if _has_sklearn_1_4():
        kwargs["categorical_features"] = "from_dtype"
    return (
        HistGradientBoostingClassifier(**kwargs)
        if discrete
        else HistGradientBoostingRegressor(**kwargs)
    )


def _resolve_learner(model: OutcomeLearner, y_series: pd.Series) -> tuple[Any, str]:
    """Resolve the per-outcome estimator + a short note on how it was chosen.

    Dispatch (using ``_is_discrete_series`` for the regressor-vs-classifier split):

    * ``"auto"`` — a fresh ``HistGradientBoosting`` classifier (discrete outcome)
      or regressor (continuous), per :func:`_auto_estimator`.
    * ``{"_discrete": clf, "_continuous": reg}`` — a **type** map (its keys are the
      reserved ``_MODEL_KEY_*`` names); an outcome whose type is absent from the
      map falls back to ``"auto"``.
    * ``{outcome_column: estimator}`` — a **column-name** map (any other keys),
      looked up by ``y_series.name`` and cloned.
    * a single sklearn estimator — cloned for this outcome column.

    Args:
        model: ``"auto"``, an sklearn estimator, a ``{"_discrete"/"_continuous":
            estimator}`` type map, or a ``{outcome_column: estimator}`` map.
        y_series: The observed outcome column (its ``name`` is used for column
            dispatch, its values for the regressor-vs-classifier dispatch).

    Returns:
        ``(unfitted_estimator, note)`` — ``note`` is a short human string for the
        INFO fit log (e.g. ``"auto"``, ``"model['_discrete']"``).

    Raises:
        ValueError: For an unknown string, or a column-name map missing this
            outcome column.

    Examples:
    .. code-block:: python

        import pandas as pd
        from balance.outcome_models.outcome_model import _resolve_learner
        from sklearn.base import is_classifier

        est, note = _resolve_learner("auto", pd.Series([0, 1, 1, 0], name="y"))
        is_classifier(est), note
        # (True, "auto")
    """
    discrete = _is_discrete_series(y_series)
    if isinstance(model, str):
        if model != "auto":
            raise ValueError(
                f"Unknown model string {model!r}. The only supported string mode "
                "is 'auto'; otherwise pass an sklearn estimator, a "
                "{'_discrete': clf, '_continuous': reg} type map, or a "
                "{outcome_column: estimator} map."
            )
        return _auto_estimator(discrete), "auto"

    if isinstance(model, dict):
        if set(model).issubset({_MODEL_KEY_DISCRETE, _MODEL_KEY_CONTINUOUS}):
            key = _MODEL_KEY_DISCRETE if discrete else _MODEL_KEY_CONTINUOUS
            if key in model:
                return clone(model[key]), f"model[{key!r}]"
            return _auto_estimator(discrete), f"auto (no model[{key!r}])"
        outcome_col = str(y_series.name)
        if outcome_col not in model:
            raise ValueError(
                f"model dict does not contain an estimator for outcome column "
                f"{outcome_col!r}. model keys: {sorted(map(str, model.keys()))}."
            )
        return clone(model[outcome_col]), f"model[{outcome_col!r}]"

    # A single sklearn estimator: clone so each outcome column is fit independently.
    return clone(model), f"single {type(model).__name__}"


def _resolve_use_model_matrix(
    use_model_matrix: bool | str,
    estimator: Any,
) -> bool:
    """Resolve ``use_model_matrix="auto"`` for a given estimator.

    HistGradientBoosting estimators prefer the native-categorical path
    (``use_model_matrix=False``); linear learners prefer patsy one-hot +
    :class:`~sklearn.preprocessing.StandardScaler` (``use_model_matrix=True``).
    On scikit-learn < 1.4 the native path is unavailable, so ``"auto"`` falls
    back to one-hot for every learner so the default works everywhere.

    Args:
        use_model_matrix: ``True``/``False`` (explicit) or ``"auto"``.
        estimator: The resolved (unfitted) sklearn estimator for this outcome.

    Returns:
        ``True`` to use the one-hot/scaler path, ``False`` for the raw
        native-categorical path.
    """
    if isinstance(use_model_matrix, bool):
        return use_model_matrix
    if use_model_matrix != "auto":
        raise ValueError(
            f"use_model_matrix must be True, False, or 'auto'; got "
            f"{use_model_matrix!r}."
        )
    # "auto": only HistGradientBoosting supports scikit-learn's native-categorical
    # path (categorical_features="from_dtype"), and only on sklearn >= 1.4; every
    # other estimator (incl. RandomForest / plain GradientBoosting) uses one-hot.
    estimator_name = type(estimator).__name__.lower()
    prefers_native = "histgradientboosting" in estimator_name
    if prefers_native and _has_sklearn_1_4():
        return False
    return True


def _needs_dense(estimator: Any) -> bool:
    """Return True when the estimator rejects sparse input (HistGradientBoosting)."""
    return "histgradientboosting" in type(estimator).__name__.lower()


def _capture_categorical_levels(matrix: Any) -> Dict[str, List[Any]] | None:
    """Capture fit-time category levels from a raw-path design matrix.

    Only meaningful for the native-categorical path, where ``matrix`` is a
    :class:`~pandas.DataFrame` with categorical columns. Returns ``None`` for the
    one-hot/scaler path (sparse/dense matrices have no categorical columns).
    """
    if not isinstance(matrix, pd.DataFrame):
        return None
    levels: Dict[str, List[Any]] = {}
    for col in matrix.columns:
        series = matrix[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels[col] = list(series.cat.categories)
    return levels or None


def _restore_categorical_levels(
    covars: pd.DataFrame,
    categorical_levels: Dict[str, List[Any]] | None,
) -> pd.DataFrame:
    """Re-apply stored fit-time category levels to new covariates before replay.

    Casting each stored categorical column to ``pd.Categorical`` with the
    fit-time ``categories`` keeps the integer codes aligned with fit time: known
    values keep their codes, missing fit-time categories stay in the category set
    (so codes do not shift), and novel target categories become ``NaN`` (handled
    by the NA-indicator path / native missing support) instead of stealing an
    existing code.

    Only raw covariate columns present in ``categorical_levels`` are touched. The
    fit-time levels can include the ``"_NA"`` sentinel that ``add_na_indicator``
    appends for a categorical with missing values; it is **stripped** here because
    the replay's ``add_na_indicator`` re-adds it, and keeping it would double-add
    the category (a hard ``ValueError: new categories must not include old
    categories``).
    """
    if not categorical_levels:
        return covars
    out = covars.copy()
    for col, levels in categorical_levels.items():
        if col in out.columns:
            restore_levels = [
                level for level in levels if level != _NA_CATEGORY_SENTINEL
            ]
            out[col] = pd.Categorical(out[col], categories=restore_levels)
    return out


def _classifier_perf(
    y: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray,
) -> Dict[str, Any]:
    """Weighted classifier performance: deviance-explained + log-loss.

    Reuses the IPW proportion-of-deviance idea (:func:`_compute_deviance`,
    ``2 * log_loss``): ``deviance_explained = 1 - dev / null_dev`` where the null
    model predicts the weighted prevalence for everyone. Kept intentionally
    simple; ``deviance_explained`` is the classifier analog of ``r2`` and is
    bounded above by 1 (it can be negative for a badly-miscalibrated model).
    """
    n = int(len(y))
    prevalence = float(np.sum(sample_weight * y) / np.sum(sample_weight))
    null_dev = _compute_deviance(
        y, np.full(n, prevalence), sample_weight, labels=[0, 1]
    )
    dev = _compute_deviance(y, proba, sample_weight, labels=[0, 1])
    deviance_explained = float("nan") if null_dev == 0 else 1.0 - dev / null_dev
    return {
        "deviance_explained": deviance_explained,
        "log_loss": float(dev / 2.0),
        "n": n,
    }


def _build_fit_matrix(
    covars_df: pd.DataFrame,
    *,
    use_model_matrix: bool,
    formula: str | list[str] | None,
    na_action: str,
    sample_weight_arr: np.ndarray | None,
    densify: bool,
) -> Dict[str, Any]:
    """Build the TRAIN-mode design matrix on the responders' covariates alone.

    Uses ``build_design_matrix`` with the covariates in the sample slot and an
    empty target slot (the fit needs only the responders). Fits a
    :class:`~sklearn.preprocessing.StandardScaler` (weighted by
    ``sample_weight_arr``) on the one-hot path; densifies for boosting learners.
    """
    empty_target = covars_df.iloc[0:0]
    dm = build_design_matrix(
        covars_df,
        empty_target,
        use_model_matrix=use_model_matrix,
        formula=formula,
        na_action=na_action,
        scaler_weights=sample_weight_arr if use_model_matrix else None,
    )
    sample_n = dm["sample_n"]
    matrix = dm["combined_matrix"]
    # build_design_matrix concatenates sample+target; slice back to sample rows.
    fit_matrix = matrix[:sample_n]
    categorical_levels = _capture_categorical_levels(fit_matrix)
    if densify and issparse(fit_matrix):
        # Only sparse output is densified here; a raw-path DataFrame is kept
        # as-is for the native-categorical learner.
        fit_matrix = _convert_to_dense_array(fit_matrix)
    return {
        "fit_matrix": fit_matrix,
        "columns": dm["columns"],
        "fit_scaler": dm["fit_scaler"],
        "resolved_formula": dm["resolved_formula"],
        "categorical_levels": categorical_levels,
    }


def _fit_matrix_type(matrix: Any) -> str:
    """Classify a matrix as ``"sparse"`` / ``"dense"`` / ``"dataframe"`` (cf IPW)."""
    if issparse(matrix):
        return "sparse"
    if isinstance(matrix, pd.DataFrame):
        return "dataframe"
    return "dense"


def _validate_fit_inputs(
    na_action: str,
    transformations: str | dict[str, Any] | None,
    outcomes_df: pd.DataFrame,
    covars_df: pd.DataFrame,
) -> None:
    """Validate inputs for :func:`fit_outcome_model` before any computation.

    Raises early with actionable messages for unsupported/empty inputs.
    """
    if na_action == "drop":
        raise ValueError(
            "na_action='drop' is incompatible with outcome-model transfer: "
            "replay of the stored preprocessing needs NA-indicator columns to "
            "stay aligned across fit and score data. Use "
            "na_action='add_indicator' (the default)."
        )
    if transformations is not None:
        # TODO (transformations): support balance's data-dependent transformations
        # (quantize / fct_lump; adjust's transformations="default"), rejected today
        # because they can't be replayed deterministically in the fit -> predict-
        # later flow. Target design:
        #   * Fit on a SampleFrame: FREEZE the transform params at fit (bin edges,
        #     kept levels) next to the scaler / categorical_levels, and RE-APPLY the
        #     identical bins/levels to the target when used within a BalanceFrame.
        #   * Constructed from a BalanceFrame (sample + target both available at
        #     fit): OPTION to compute the transforms on the JOINT sample + target
        #     (as adjust does), so no freezing is needed.
        raise ValueError(
            "transformations are not yet supported for outcome models: balance's "
            "data-dependent transformations (e.g. quantize/fct_lump) can't yet be "
            "frozen for deterministic replay on the target/new data. Pass "
            "transformations=None (see the TODO in outcome_model.py)."
        )
    if outcomes_df.shape[1] == 0:
        raise ValueError("outcomes_df must contain at least one outcome column.")
    if covars_df.shape[0] == 0:
        raise ValueError(
            "covars_df has no rows; cannot fit an outcome model on an empty "
            "sample. Provide at least one row with observed covariates and outcome."
        )


def _prepare_sample_weight(
    sample_weight: pd.Series | np.ndarray | None,
    covars_df: pd.DataFrame,
) -> tuple[bool, np.ndarray | None, str | None]:
    """Prepare sample weights for fitting, returning (weighted, arr, column_name).

    Validates the weights against the covariate index and normalises them to a
    numpy array.  Returns ``(False, None, None)`` when no weights are provided.
    """
    if sample_weight is None:
        return False, None, None
    sample_weight_series = (
        sample_weight
        if isinstance(sample_weight, pd.Series)
        else pd.Series(sample_weight, index=covars_df.index)
    )
    # TODO (weights): validate the weights before fitting — reject zero /
    # negative / NaN / inf sample weights with an actionable error (mirrors
    # the copilot-instructions weighting-input checklist) rather than passing
    # them straight to the estimator.
    _check_weighting_methods_input(covars_df, sample_weight_series, "sample")
    sample_weight_arr = sample_weight_series.to_numpy(dtype=float)
    fit_weight_column: str | None = (
        str(sample_weight_series.name)
        if sample_weight_series.name is not None
        else None
    )
    return True, sample_weight_arr, fit_weight_column


def _validate_classifier_outcome(
    y_series: pd.Series,
    outcome_col: str,
) -> None:
    """Validate that a discrete outcome is binary 0/1 with at least two classes.

    Raises:
        ValueError: If the outcome is not binary 0/1 (or boolean), or has only a
            single observed class.
    """
    # TODO (multiclass): only binary 0/1 outcomes are supported today. To
    # add multiclass, either (a) one-hot the outcome and fit a binary model
    # per level, or (b) fit a native multiclass classifier and average its
    # predict_proba matrix. Until then, non-0/1 discrete outcomes raise.
    observed = pd.unique(y_series.dropna())
    is_binary_01 = pd.api.types.is_bool_dtype(y_series) or set(observed).issubset(
        {0, 1}
    )
    if not is_binary_01:
        raise ValueError(
            f"Outcome column {outcome_col!r} was detected as categorical "
            f"but is not a binary 0/1 (or boolean) outcome (observed "
            f"values: {sorted(str(v) for v in observed)}). Outcome-model "
            "classification supports only binary 0/1 outcomes; encode the "
            "outcome as 0/1, or pass a regression learner explicitly."
        )
    if len(observed) < 2:
        raise ValueError(
            f"Outcome column {outcome_col!r} has a single observed class "
            f"({sorted(str(v) for v in observed)}); a classifier needs at "
            "least two classes present. Provide an outcome with both "
            "classes, or use a continuous outcome."
        )


def fit_outcome_model(
    covars_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
    model: OutcomeLearner = "auto",
    formula: str | list[str] | None = None,
    transformations: str | dict[str, Any] | None = None,
    na_action: str = "add_indicator",
    use_model_matrix: bool | str = "auto",
    calibrate: bool = False,
) -> Dict[str, Any]:
    """Fit an outcome model ``ĝ(X) ≈ E[Y|X]`` and return a stored model dict.

    A regressor (continuous outcome) or classifier (binary outcome) is fit for
    each column of ``outcomes_df`` on a design matrix built from ``covars_df``.
    The returned dict stores the fitted estimators together with the fitted
    preprocessing (design-matrix columns, scaler, category levels, matrix type),
    so :func:`predict_outcome` can replay the exact transformation on new data.
    This mirrors the IPW model dict (:mod:`balance.weighting_methods.ipw`) so the
    same :func:`~balance.utils.model_matrix.build_design_matrix` replay works.

    The fit is **weighted** whenever ``sample_weight`` is provided (passed as
    ``sample_weight=`` to the learner's ``fit`` and to the ``StandardScaler``);
    HistGradientBoosting and LinearRegression both accept sample weights.

    Args:
        covars_df: Responder covariates ``X_R`` (one row per responder).
        outcomes_df: Observed outcome(s) ``Y_R``; one fitted estimator per column.
        sample_weight: Optional per-row weights ``w_R``. When ``None`` the fit is
            unweighted. Its index must match ``covars_df`` / ``outcomes_df``.
        model: ``"auto"`` (HistGradientBoosting regressor/classifier by outcome
            type), a single sklearn estimator (cloned per outcome column — must be
            a single type when the outcomes are mixed), a ``{"_discrete": clf,
            "_continuous": reg}`` type map (a type absent from the map falls back
            to ``"auto"``), or a ``{outcome_column: estimator}`` column map.
        formula: Optional patsy formula(s) forwarded to the one-hot path; ignored
            on the native-categorical path.
        transformations: Reserved for parity with IPW; must be ``None`` for now.
            Balance's data-dependent transforms (quantize/fct_lump) can't yet be
            frozen for deterministic replay on the target, so a non-``None`` value
            raises (see the module TODO).
        na_action: How to handle missing values. Only ``"add_indicator"`` (the
            default) is supported — ``"drop"`` is rejected because replay needs
            the NA-indicator columns to stay aligned across fit and score data.
        use_model_matrix: ``"auto"`` (default) picks the native-categorical path
            for tree/boosting learners on scikit-learn >= 1.4 and the one-hot +
            scaler path otherwise; pass ``True``/``False`` to force a path.
        calibrate: When ``True``, wrap each classifier in
            :class:`~sklearn.calibration.CalibratedClassifierCV` (binary outcomes
            only). Ignored for regressors. Defaults to ``False``.

    Returns:
        Dict[str, Any]: the ``_outcome_model`` dict. Keys:

            ``"method"`` (``"outcome_model"``), ``"fit"`` (``{col: estimator}``),
            ``"X_matrix_columns"``, ``"fit_scaler"``, ``"formula"``,
            ``"na_action"``, ``"one_hot_encoding"``, ``"use_model_matrix"``,
            ``"transformations"``, ``"fit_matrix_type"``, ``"weighted"``,
            ``"fit_weight"`` (``{"column": str | None, "uniform": bool}``),
            ``"prediction_kind"``
            (``{col: "regression"|"proba"}``), ``"calibrated"``,
            ``"categorical_levels"`` (or ``None``), ``"outcome_columns"``,
            ``"learner"`` (repr), ``"perf"`` (``{col: {...}}``), and
            ``"training_sample_index"``.

    Raises:
        ValueError: If ``na_action="drop"``, if ``transformations`` is not
            ``None``, if ``outcomes_df`` has no columns, if ``covars_df`` has no
            rows, if a discrete outcome is not binary 0/1 or has a single observed
            class, if a single ``model`` estimator is given for mixed-type
            outcomes, or if a ``model`` column map lacks an outcome column.
        TypeError: If ``weighted`` (``sample_weight`` given) but the resolved
            estimator's ``fit`` does not accept ``sample_weight``.

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.outcome_models import fit_outcome_model, predict_outcome

        rng = np.random.default_rng(0)
        covars_R = pd.DataFrame(
            {"age": rng.normal(50, 10, 200), "grp": rng.choice(["a", "b"], 200)}
        )
        outcomes_R = pd.DataFrame(
            {"happiness": covars_R["age"] + rng.normal(0, 1, 200)}
        )
        w_R = pd.Series(np.ones(200))

        model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)
        model["method"]
        # 'outcome_model'
        sorted(k for k in ("fit", "X_matrix_columns", "perf") if k in model)
        # ['X_matrix_columns', 'fit', 'perf']
    """
    _validate_fit_inputs(na_action, transformations, outcomes_df, covars_df)

    weighted, sample_weight_arr, fit_weight_column = _prepare_sample_weight(
        sample_weight, covars_df
    )

    outcome_columns: List[str] = [str(c) for c in outcomes_df.columns]
    outcome_is_discrete: Dict[str, bool] = {
        col: _is_discrete_series(outcomes_df[col]) for col in outcome_columns
    }
    # A single estimator can't serve a mixed-type outcome set; require "auto" or a
    # type/column map instead (a type map picks per outcome; "auto" dispatches).
    if (
        not isinstance(model, str)
        and not isinstance(model, dict)
        and len(set(outcome_is_discrete.values())) > 1
    ):
        raise ValueError(
            "a single estimator was passed as `model`, but the outcomes are of "
            f"mixed type (discrete + continuous): {outcome_is_discrete}. Pass "
            "model='auto', a {'_discrete': clf, '_continuous': reg} type map, or a "
            "{outcome_column: estimator} map instead."
        )
    fit: Dict[str, Any] = {}
    perf: Dict[str, Any] = {}
    prediction_kind: Dict[str, str] = {}
    stored_columns: List[str] | None = None
    stored_scaler: Any = None
    stored_formula: str | list[str] | None = None
    stored_categorical_levels: Dict[str, List[Any]] | None = None
    stored_matrix_type: str | None = None
    resolved_use_model_matrix: bool | None = None
    fit_matrix: Any = None

    for outcome_col in outcome_columns:
        y_series = outcomes_df[outcome_col]
        estimator, model_note = _resolve_learner(model, y_series)
        logger.info(
            "outcome_model: outcome %r (%s) -> %s [%s]",
            outcome_col,
            "discrete" if outcome_is_discrete[outcome_col] else "continuous",
            type(estimator).__name__,
            model_note,
        )
        col_use_mm = _resolve_use_model_matrix(use_model_matrix, estimator)
        # All columns share one design matrix; resolve on the first column and
        # reuse (a mixed regressor/classifier set still shares covariate encoding).
        if resolved_use_model_matrix is None:
            resolved_use_model_matrix = col_use_mm
            built = _build_fit_matrix(
                covars_df,
                use_model_matrix=resolved_use_model_matrix,
                formula=formula,
                na_action=na_action,
                sample_weight_arr=sample_weight_arr,
                densify=_needs_dense(estimator),
            )
            stored_columns = built["columns"]
            stored_scaler = built["fit_scaler"]
            stored_formula = built["resolved_formula"]
            stored_categorical_levels = built["categorical_levels"]
            fit_matrix = built["fit_matrix"]
            stored_matrix_type = _fit_matrix_type(fit_matrix)
            # The native-categorical path (use_model_matrix=False) feeds raw
            # categorical dtypes to the estimator, which needs scikit-learn >= 1.4
            # (categorical_features="from_dtype"). On older scikit-learn the
            # boosting fit fails with a cryptic "could not convert string to
            # float", so raise an actionable error instead (mirrors IPW's
            # use_model_matrix guard); use_model_matrix="auto" already falls back
            # to one-hot on < 1.4.
            if (
                not resolved_use_model_matrix
                and stored_categorical_levels
                and not _has_sklearn_1_4()
            ):
                raise ValueError(
                    "use_model_matrix=False (the native-categorical path) needs "
                    "scikit-learn >= 1.4 to fit categorical covariates; the "
                    "installed scikit-learn is older. Use use_model_matrix='auto' "
                    "(one-hot fallback) or use_model_matrix=True, or upgrade "
                    "scikit-learn."
                )
        # Densify per-estimator if this estimator needs it but the shared matrix
        # is still sparse (e.g. a boosting learner mixed with a linear one).
        estimator_matrix = fit_matrix
        if _needs_dense(estimator) and issparse(estimator_matrix):
            estimator_matrix = _convert_to_dense_array(estimator_matrix)

        y_values = y_series.to_numpy()
        is_clf = _is_discrete_series(y_series)
        if is_clf:
            _validate_classifier_outcome(y_series, outcome_col)
            from sklearn.base import is_classifier

            if not is_classifier(estimator):
                raise ValueError(
                    f"outcome column {outcome_col!r} is discrete (binary), so it "
                    f"needs a classifier, but the resolved estimator "
                    f"{type(estimator).__name__} is a regressor. Pass a classifier "
                    f"(e.g. model={{'_discrete': <classifier>, '_continuous': "
                    f"<regressor>}} for mixed outcomes), or use model='auto'."
                )
        if weighted and not has_fit_parameter(estimator, "sample_weight"):
            raise TypeError(
                f"weighted=True, but the estimator for outcome {outcome_col!r} "
                f"({type(estimator).__name__}) does not accept `sample_weight` in "
                "its fit(). Pass weighted=False (unweighted g-computation) or use "
                "an estimator that supports sample weights."
            )
        if is_clf and calibrate:
            from sklearn.calibration import CalibratedClassifierCV

            estimator = CalibratedClassifierCV(estimator)

        if weighted:
            estimator.fit(estimator_matrix, y_values, sample_weight=sample_weight_arr)
        else:
            estimator.fit(estimator_matrix, y_values)
        fit[outcome_col] = estimator

        perf_weight = (
            sample_weight_arr
            if sample_weight_arr is not None
            else np.ones(len(y_values))
        )
        if is_clf:
            proba = _predict_proba_class1(estimator, estimator_matrix)
            prediction_kind[outcome_col] = "proba"
            perf[outcome_col] = _classifier_perf(
                y_values.astype(float), proba, perf_weight
            )
        else:
            y_hat = np.asarray(estimator.predict(estimator_matrix), dtype=float)
            prediction_kind[outcome_col] = "regression"
            perf[outcome_col] = {
                "r2": weighted_r2(y_values.astype(float), y_hat, perf_weight),
                "n": int(len(y_values)),
            }

    # TODO (AIPW/DR seam): a doubly-robust (AIPW) estimator will combine these
    # predictions with the propensity weights; it will need the responder
    # residuals Y - ĝ(X) and, ideally, CROSS-FITTED (out-of-fold) ĝ to avoid
    # own-observation optimism. Don't lock this stored model into in-sample-only
    # predictions when adding DR.
    return {
        "method": "outcome_model",
        "fit": fit,
        "X_matrix_columns": _assert_type(stored_columns, list),
        "fit_scaler": stored_scaler,
        "formula": stored_formula,
        "na_action": na_action,
        "one_hot_encoding": False,
        "use_model_matrix": bool(resolved_use_model_matrix),
        "transformations": transformations,
        "fit_matrix_type": stored_matrix_type,
        "weighted": weighted,
        "fit_weight": {"column": fit_weight_column, "uniform": not weighted},
        "prediction_kind": prediction_kind,
        "calibrated": bool(calibrate)
        and any(type(e).__name__ == "CalibratedClassifierCV" for e in fit.values()),
        "categorical_levels": stored_categorical_levels,
        "outcome_columns": outcome_columns,
        "learner": repr(model),
        "perf": perf,
        "training_sample_index": covars_df.index.copy(),
    }


def _predict_proba_class1(estimator: Any, matrix: Any) -> np.ndarray:
    """Return ``P̂(Y=1)`` from a fitted classifier's ``predict_proba``.

    Selects the column of ``predict_proba`` corresponding to the positive class
    label ``1`` (falling back to the last column when the classes are not the
    canonical ``{0, 1}``), mirroring the class-index lookup IPW performs.
    """
    proba = np.asarray(estimator.predict_proba(matrix))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        class_index = classes.index(1)
    else:
        class_index = proba.shape[1] - 1
    return proba[:, class_index].astype(float)


def _build_replay_matrix(
    model: Dict[str, Any],
    new_covars_df: pd.DataFrame,
) -> Any:
    """Build the REPLAY-mode design matrix on new covariates from a model dict.

    Restores stored ``categorical_levels`` on the raw path (so category codes
    match fit time), then calls ``build_design_matrix`` with the stored
    ``X_matrix_columns`` / ``fit_scaler`` / ``na_action`` / ``fit_matrix_type``.
    """
    use_model_matrix = bool(model.get("use_model_matrix", True))
    covars = new_covars_df
    if not use_model_matrix:
        covars = _restore_categorical_levels(
            new_covars_df, model.get("categorical_levels")
        )
    empty_target = covars.iloc[0:0]
    dm = build_design_matrix(
        covars,
        empty_target,
        use_model_matrix=use_model_matrix,
        formula=model.get("formula"),
        na_action=str(model.get("na_action", "add_indicator")),
        project_to_columns=_assert_type(model.get("X_matrix_columns"), list),
        fit_scaler=model.get("fit_scaler"),
        matrix_type=model.get("fit_matrix_type"),
    )
    sample_n = dm["sample_n"]
    return dm["combined_matrix"][:sample_n]


def predict_outcome(
    model: Dict[str, Any],
    new_covars_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Apply a stored outcome model to new covariates, returning ``outcomes_hat``.

    Rebuilds the design matrix on ``new_covars_df`` in **replay mode** (using the
    stored fit-time columns, scaler, category levels, and matrix type), then per
    outcome column returns the learner's prediction: ``.predict`` for a
    regressor, or ``P̂(Y=1)`` (``predict_proba[:, class-1]``) for a classifier.

    Args:
        model: A dict produced by :func:`fit_outcome_model`.
        new_covars_df: Covariates to score (e.g. the target ``X_T``). Must expose
            the fit-time covariate columns.

    Returns:
        Dict[str, np.ndarray]: ``{outcome_column: ŷ}`` with one prediction array
        per fitted outcome, aligned row-wise with ``new_covars_df``.

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.outcome_models import fit_outcome_model, predict_outcome

        rng = np.random.default_rng(0)
        covars_R = pd.DataFrame({"age": rng.normal(50, 10, 200)})
        outcomes_R = pd.DataFrame(
            {"happiness": covars_R["age"] + rng.normal(0, 1, 200)}
        )
        model = fit_outcome_model(
            covars_R, outcomes_R, sample_weight=pd.Series(np.ones(200))
        )
        covars_T = pd.DataFrame({"age": rng.normal(55, 10, 5)})
        preds = predict_outcome(model, covars_T)
        preds["happiness"].shape
        # (5,)
    """
    matrix = _build_replay_matrix(model, new_covars_df)
    fit: Dict[str, Any] = model["fit"]
    prediction_kind: Dict[str, str] = model.get("prediction_kind", {})
    out: Dict[str, np.ndarray] = {}
    for outcome_col, estimator in fit.items():
        estimator_matrix = matrix
        if _needs_dense(estimator) and issparse(estimator_matrix):
            estimator_matrix = _convert_to_dense_array(estimator_matrix)
        kind = prediction_kind.get(
            outcome_col, "proba" if is_classifier(estimator) else "regression"
        )
        if kind == "proba":
            out[outcome_col] = _predict_proba_class1(estimator, estimator_matrix)
        else:
            out[outcome_col] = np.asarray(
                estimator.predict(estimator_matrix), dtype=float
            )
    return out


def _unwrap_calibrated(estimator: Any) -> Any:
    """Return the inner estimator of a ``CalibratedClassifierCV`` wrapper.

    The bootstrap refit re-applies the stored ``calibrate`` flag itself (which
    re-wraps in :class:`~sklearn.calibration.CalibratedClassifierCV`), so the
    learner it refits with must be the *unwrapped* estimator; otherwise the
    classifier would be double-wrapped.  ``.estimator`` is the sklearn >= 1.2
    attribute; ``.base_estimator`` covers older versions.
    """
    inner = getattr(estimator, "estimator", None)
    if inner is None:
        inner = getattr(estimator, "base_estimator", None)
    return inner if inner is not None else estimator


def learner_from_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a per-outcome ``{col: unfitted estimator}`` learner from a model.

    Clones each fitted estimator stored under ``model["fit"]`` (unwrapping any
    :class:`~sklearn.calibration.CalibratedClassifierCV`) to recover an unfitted
    estimator with the **same hyper-parameters**.  Passing this back as the
    ``model`` to :func:`fit_outcome_model` reproduces the exact estimators used at
    fit time — for both ``model="auto"`` and pluggable estimators — without
    depending on the (lossy) stored ``"learner"`` repr in the model dict.

    Args:
        model: A dict produced by :func:`fit_outcome_model`.

    Returns:
        Dict[str, Any]: ``{outcome_column: unfitted sklearn estimator}`` suitable
        as the ``model`` argument to :func:`fit_outcome_model`.
    """
    return {
        outcome_col: clone(_unwrap_calibrated(estimator))
        for outcome_col, estimator in model["fit"].items()
    }


def _percentile_ci(values: np.ndarray, conf_level: float) -> Tuple[float, float]:
    """Percentile confidence interval from a 1-D array of bootstrap estimates."""
    alpha = 1.0 - conf_level
    low = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_outcome_estimate(
    sample_covars: pd.DataFrame,
    outcomes: pd.DataFrame,
    sample_weight: pd.Series | np.ndarray | None,
    target_covars: pd.DataFrame,
    target_weight: pd.Series | np.ndarray | None,
    *,
    fit_kwargs: Dict[str, Any],
    n_bootstrap: int = 200,
    random_seed: int = 2020,
    conf_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Nonparametric bootstrap CI for the outcome-model estimate ``μ̂_OM``.

    Captures the dominant uncertainty in ``μ̂_OM`` — the responder sample plus the
    outcome-model fit — while treating the **target as a fixed, known
    population** (only the responders resample).  For each of ``n_bootstrap``
    replicates it resamples the responder rows **with replacement** (keeping each
    resampled row's weight), refits ``ĝ*`` via :func:`fit_outcome_model` with the
    **same fit configuration** as the full-sample model (``fit_kwargs`` — so the
    resampled weights honour the stored ``weighted``/fit-weighting), predicts on
    the fixed target covariates, and averages the predictions with the target
    weights (:func:`~balance.stats_and_plots.weighted_stats.weighted_mean`) to get
    one scalar ``μ̂_OM*(b)`` per outcome.  Only the ``B`` scalar outputs per
    outcome are kept — never ``B`` fitted models — so memory stays flat in ``B``.

    The reported **point estimate** is the full-sample ``μ̂_OM`` — ``ĝ`` refit once
    on all responders (via ``fit_kwargs``) and averaged over the fixed target; for a
    deterministic estimator this equals ``outcomes_hat().mean()``.  The bootstrap only
    supplies the interval.  The interval is a **percentile** CI over ``{μ̂_OM*(b)}``.  The
    routine is **deterministic** given ``random_seed`` (it uses
    :func:`numpy.random.default_rng`), so the same seed yields identical CIs.

    Args:
        sample_covars: Responder covariates ``X_R`` (rows are resampled).
        outcomes: Observed responder outcome(s) ``Y_R``; must be row-aligned to
            ``sample_covars``.  One estimate is returned per outcome column.
        sample_weight: Responder weights ``w_R`` (resampled alongside the rows),
            or ``None`` for an unweighted fit.  Row-aligned to ``sample_covars``.
        target_covars: The **fixed** target covariates ``X_T`` scored every
            replicate.
        target_weight: Target weights ``w_T`` used to average ``ĝ*(X_T)``, or
            ``None`` for a simple mean.  Row-aligned to ``target_covars``.
        fit_kwargs: Keyword arguments forwarded verbatim to
            :func:`fit_outcome_model` on every replicate (e.g. ``model``,
            ``formula``, ``na_action``, ``use_model_matrix``, ``calibrate``) so
            the refit matches the stored model's configuration.  Whether the
            refit is weighted is controlled by passing (or omitting)
            ``sample_weight``, not by ``fit_kwargs``.
        n_bootstrap: Number of bootstrap replicates ``B``.  Defaults to 200.
        random_seed: Seed for the resampling RNG.  Defaults to 2020.
        conf_level: Confidence level for the percentile interval.  Defaults to
            0.95.

    Returns:
        Dict[str, Dict[str, float]]: ``{outcome_column: {"estimate": μ̂_OM,
        "ci_low": ..., "ci_high": ...}}`` — the full-sample point estimate plus
        the percentile CI bounds, per outcome.

    Examples:
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from balance.outcome_models.outcome_model import (
            bootstrap_outcome_estimate,
        )
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        covars_R = pd.DataFrame({"age": rng.normal(50, 10, 200)})
        outcomes_R = pd.DataFrame({"happiness": covars_R["age"] + rng.normal(0, 1, 200)})
        w_R = pd.Series(np.ones(200))
        covars_T = pd.DataFrame({"age": rng.normal(55, 10, 50)})
        w_T = pd.Series(np.ones(50))

        res = bootstrap_outcome_estimate(
            covars_R, outcomes_R, w_R, covars_T, w_T,
            fit_kwargs={"model": LinearRegression()},
            n_bootstrap=25, random_seed=2020,
        )
        sorted(res["happiness"])  # -> ['ci_high', 'ci_low', 'estimate']
    """
    outcome_columns: List[str] = [str(c) for c in outcomes.columns]

    # Full-sample point estimate: fit once on all responders, average over the
    # fixed target with the target weights.
    full_model = fit_outcome_model(
        sample_covars, outcomes, sample_weight=sample_weight, **fit_kwargs
    )
    full_predictions = predict_outcome(full_model, target_covars)
    point_estimate: Dict[str, float] = {
        col: float(
            weighted_mean(pd.Series(full_predictions[col]), target_weight).iloc[0]
        )
        for col in outcome_columns
    }

    n_responders = sample_covars.shape[0]
    rng = np.random.default_rng(random_seed)
    boot_estimates: Dict[str, List[float]] = {col: [] for col in outcome_columns}
    skipped = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_responders, size=n_responders)
        # Resample rows (and weights) with replacement; fresh positional index
        # so the design-matrix / weighting-input validators stay aligned.
        boot_covars = sample_covars.iloc[idx].reset_index(drop=True)
        boot_outcomes = outcomes.iloc[idx].reset_index(drop=True)
        boot_weight: pd.Series | None = None
        if sample_weight is not None:
            weight_values = np.asarray(sample_weight, dtype=float)[idx]
            boot_weight = pd.Series(weight_values, index=boot_covars.index)

        try:
            boot_model = fit_outcome_model(
                boot_covars, boot_outcomes, sample_weight=boot_weight, **fit_kwargs
            )
        except ValueError:
            # A resample can be degenerate (e.g. a single observed class for a
            # rare binary outcome); skip it rather than aborting the whole CI.
            skipped += 1
            continue
        boot_predictions = predict_outcome(boot_model, target_covars)
        for col in outcome_columns:
            mu_star = float(
                weighted_mean(pd.Series(boot_predictions[col]), target_weight).iloc[0]
            )
            boot_estimates[col].append(mu_star)

    if skipped:
        logger.warning(
            "bootstrap_outcome_estimate: skipped %d of %d resample(s) that could "
            "not be fit (e.g. a degenerate single-class resample of a rare binary "
            "outcome); the CI is formed from the %d successful replicate(s).",
            skipped,
            n_bootstrap,
            n_bootstrap - skipped,
        )

    result: Dict[str, Dict[str, float]] = {}
    for col in outcome_columns:
        estimates = boot_estimates[col]
        if len(estimates) < 2:
            raise ValueError(
                "bootstrap_outcome_estimate could not fit enough valid resamples "
                f"({len(estimates)} of {n_bootstrap} succeeded) to form a "
                "confidence interval — the outcome may be too rare or degenerate "
                "for a nonparametric bootstrap. Use mean_with_ci(ci_method="
                "'analytic') or provide more data."
            )
        ci_low, ci_high = _percentile_ci(np.asarray(estimates, dtype=float), conf_level)
        result[col] = {
            "estimate": point_estimate[col],
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return result
