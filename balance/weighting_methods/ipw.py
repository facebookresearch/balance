# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
from typing import Any, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots.weighted_comparisons_stats import asmd
from balance.stats_and_plots.weights_stats import design_effect
from balance.testutil import _verify_value_type

from scipy.sparse import csc_matrix, csr_matrix, issparse
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


logger: logging.Logger = logging.getLogger(__package__)


def model_coefs(
    model: ClassifierMixin,
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Extract coefficient-like information from sklearn classifiers.

    For linear models such as :class:`~sklearn.linear_model.LogisticRegression`,
    this returns the fitted coefficients (and intercept when available).  For
    classifiers that do not expose a ``coef_`` attribute (e.g. tree ensembles),
    an empty :class:`pandas.Series` is returned so downstream diagnostics can
    handle the absence of coefficients gracefully.

    Args:
        model (ClassifierMixin): Fitted sklearn classifier.
        feature_names (Optional[list], optional): Feature names associated with
            the model matrix columns. When provided and the model exposes a
            one-dimensional ``coef_`` array, the returned Series is indexed by
            ``["intercept"] + feature_names``.

    Returns:
        Dict[str, Any]: Dictionary containing a ``coefs`` key with a
        :class:`pandas.Series` of coefficients (which may be empty when the
        model does not expose linear coefficients).
    """

    if not hasattr(model, "coef_"):
        return {"coefs": pd.Series(dtype=float)}
    coefs = np.asarray(cast(Any, model).coef_)
    intercept = getattr(model, "intercept_", None)

    if coefs.ndim > 1 and coefs.shape[0] == 1:
        coefs = coefs.reshape(-1)

    # If the model was trained on scaled data, map the coefficients back to
    # the original feature space for interpretability. When the attribute is
    # absent, we assume the coefficients are already on the original scale.
    scale_factor = getattr(model, "_balance_feature_scale_factor", None)
    if scale_factor is not None and coefs.ndim == 1:
        coefs = coefs * np.asarray(scale_factor)[: len(coefs)]

    if feature_names is not None and coefs.ndim == 1:
        index: List[str] = list(feature_names[: len(coefs)])
        values = coefs
        if intercept is not None:
            intercept_array = np.asarray(intercept).ravel()
            if intercept_array.size == 1:
                index = ["intercept"] + index
                values = np.hstack((intercept_array, coefs))
        coefs_series = pd.Series(data=values, index=index)
    else:
        coefs_series = pd.Series(data=np.ravel(coefs))

    return {
        "coefs": coefs_series,
    }


def link_transform(pred: np.ndarray) -> np.ndarray:
    """Transforms probabilities into log odds (link function).

    Args:
        pred (np.ndarray): LogisticRegression probability predictions from sklearn.

    Returns:
        np.ndarray: Array of log odds.

    """
    pred = np.asarray(pred, dtype=float)
    # Clip probabilities to avoid dividing by zero or taking log of zero
    eps = np.finfo(float).eps
    pred = np.clip(pred, eps, 1 - eps)
    return np.log(pred / (1 - pred))


def _compute_deviance(
    y: np.ndarray,
    pred: np.ndarray,
    model_weights: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """Compute deviance (2 * log loss).

    Used multiple times throughout ipw() and calc_dev() functions.

    Args:
        y (np.ndarray): True labels.
        pred (np.ndarray): Predicted probabilities.
        model_weights (np.ndarray): Sample weights.
        labels (Optional[List[int]], optional): Label specification. Defaults to None.

    Returns:
        float: Deviance value.
    """
    if labels is not None:
        return 2 * log_loss(y, pred, sample_weight=model_weights, labels=labels)
    return 2 * log_loss(y, pred, sample_weight=model_weights)


def _compute_proportion_deviance(dev: float, null_dev: float) -> float:
    """Compute proportion of null deviance explained.

    Used multiple times in ipw() for model evaluation.

    Args:
        dev (float): Model deviance.
        null_dev (float): Null model deviance.

    Returns:
        float: Proportion of deviance explained (1 - dev/null_dev).
    """
    return 1 - dev / null_dev


def _convert_to_dense_array(
    X_matrix: Union[csc_matrix, csr_matrix, np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """Convert sparse matrix or DataFrame to dense numpy array.

    If the input is a CSC matrix, first convert to CSR for efficiency,
    then convert to dense array. If already a dense numpy array or DataFrame,
    return as-is (note: DataFrames will be returned unchanged and may need
    explicit conversion to numpy array elsewhere if needed).

    Args:
        X_matrix: Input matrix - can be a sparse matrix (csc_matrix, csr_matrix),
            dense numpy array, or pandas DataFrame.

    Returns:
        np.ndarray: Dense numpy array (or DataFrame if input was DataFrame).
    """
    if isinstance(X_matrix, csc_matrix):
        X_matrix = X_matrix.tocsr()

    if issparse(X_matrix):
        X_matrix = X_matrix.toarray()

    return X_matrix


def calc_dev(
    X_matrix: csr_matrix,
    y: np.ndarray,
    model: ClassifierMixin,
    model_weights: np.ndarray,
    foldids: np.ndarray,
) -> Tuple[float, float]:
    """10 fold cross validation to calculate holdout deviance.

    Args:
        X_matrix (csr_matrix): Model matrix,
        y (np.ndarray): Vector of sample inclusion (1=sample, 0=target),
        model (_type_): LogisticRegression object from sklearn,
        model_weights (np.ndarray): Vector of sample and target weights,
        foldids (np.ndarray): Vector of cross-validation fold indices.

    Returns:
        float, float: mean and standard deviance of holdout deviance.

    """
    unique_folds = np.unique(foldids)
    cv_dev = [0.0 for _ in unique_folds]

    for fold_index, fold_value in enumerate(unique_folds):
        X_train = X_matrix[foldids != fold_value, :]
        X_test = X_matrix[foldids == fold_value, :]
        y_train = y[foldids != fold_value]
        y_test = y[foldids == fold_value]
        model_weights_train = model_weights[foldids != fold_value]
        model_weights_test = model_weights[foldids == fold_value]
        if X_test.shape[0] == 0:
            raise ValueError(
                f"Cross-validation fold {fold_value} contains no samples; verify foldids covers all observations."
            )
        # pyre-ignore[16]: ClassifierMixin has fit method at runtime
        model_fit = model.fit(X_train, y_train, sample_weight=model_weights_train)
        pred_test = model_fit.predict_proba(X_test)[:, 1]
        cv_dev[fold_index] = _compute_deviance(
            y_test, pred_test, model_weights_test, labels=[0, 1]
        )

    logger.debug(
        f"dev_mean: {np.mean(cv_dev)}, dev_sd: {np.std(cv_dev, ddof=1) / np.sqrt(len(unique_folds))}"
    )
    return np.mean(cv_dev), np.std(cv_dev, ddof=1) / np.sqrt(len(unique_folds))


def weights_from_link(
    link: Any,
    balance_classes: bool,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    weight_trimming_mean_ratio: None | float | int = None,
    weight_trimming_percentile: float | None = None,
    keep_sum_of_weights: bool = True,
    normalize_to: str = "target",
) -> pd.Series:
    """Transform link predictions into weights, by exponentiating them, and optionally balancing the classes and trimming
    the weights, then normalize the weights to have sum equal to the sum of the target weights.

    Args:
        link (Any): link predictions
        balance_classes (bool): whether balance_classes used
        sample_weights (pd.Series): vector of sample weights
        target_weights (pd.Series): vector of sample weights
        weight_trimming_mean_ratio (Union[None, float, int], optional): to be used in :func:`trim_weights`. Defaults to None.
        weight_trimming_percentile (Optional[float], optional): to be used in :func:`trim_weights`. Defaults to None.
        keep_sum_of_weights (bool, optional): to be used in :func:`trim_weights`. Defaults to True.
        normalize_to (str, optional): whether to normalize the final weights to the sum of
            the target weights ("target") or the sample weights ("sample"). Defaults to
            "target".

    Returns:
        pd.Series: A vecotr of normalized weights (for sum of target weights)
    """
    link = link.reshape((link.shape[0],))
    if balance_classes:
        odds = np.sum(sample_weights) / np.sum(target_weights)
        link = link + np.log(odds)
    weights = sample_weights / np.exp(link)
    weights = balance_adjustment.trim_weights(
        weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
        keep_sum_of_weights=keep_sum_of_weights,
    )
    # Normalize weights such that the sum will be aligned with the chosen population
    if normalize_to == "target":
        weights = weights * np.sum(target_weights) / np.sum(weights)
    elif normalize_to == "sample":
        weights = weights * np.sum(sample_weights) / np.sum(weights)
    else:
        raise ValueError("normalize_to must be either 'target' or 'sample'.")
    return weights


def choose_regularization(
    links: List[Any],
    lambdas: np.ndarray,
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    balance_classes: bool,
    max_de: float = 1.5,
    normalize_to: str = "target",
    trim_options: Tuple[
        int, int, int, float, float, float, float, float, float, float
    ] = (20, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 0.05, 0.01),
    n_asmd_candidates: int = 10,
) -> Dict[str, Any]:
    """Score regularisation parameters and trimming levels by mean squared ASMD.

    Each combination is evaluated on the covariate ASMD between ``sample_df``
    and ``target_df`` using the provided weights. Candidates exceeding the
    design effect threshold (``max_de``) are discarded. The remaining
    configurations are ranked by mean squared ASMD and the best combination is
    returned.

    Args:
        links (Links[Any]): list of link predictions from sklearn
        lambdas (np.ndarray): the lambda values for regularization
        sample_df (pd.DataFrame): a dataframe representing the sample
        target_df (pd.DataFrame): a dataframe representing the target
        sample_weights (pd.Series): design weights for sample
        target_weights (pd.Series): design weights for target
        balance_classes (bool): whether balance_classes used
        max_de (float, optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
        normalize_to (str, optional): population to which the weights are normalized when
            evaluating candidates. Defaults to "target".
        trim_options (Tuple[ int, int, int, float, float, float, float, float, float, float ], optional):
            options for weight_trimming_mean_ratio. Defaults to (20, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 0.05, 0.01).
        n_asmd_candidates (int, optional): number of candidates for grid search.. Defaults to 10.

    Returns:
        Dict[str, Any]: Dict of the value of the chosen lambda, the value of trimming, model description.
            Shape is
                {
                    "best": {"s": best.s.values, "trim": best.trim.values[0]},
                    "perf": all_perf,
                }
    """

    logger.info("Starting choosing regularisation parameters")
    # get all non-null links
    links = [link for link in links if link is not None]

    # Evaluate combinations with mean squared ASMD as the score while
    # enforcing the design effect constraint. The candidate with the
    # lowest mean squared ASMD is selected, using the same design-effect
    # based pre-filtering as the previous grid search implementation to
    # maintain consistency with historical behaviour.
    design_effects: list[dict[str, Any]] = []
    for trim_ratio in trim_options:
        for lambda_index, link in enumerate(links):
            weights = weights_from_link(
                link,
                balance_classes,
                sample_weights,
                target_weights,
                weight_trimming_mean_ratio=trim_ratio,
                normalize_to=normalize_to,
            )
            design_effects.append(
                {
                    "s_index": lambda_index,
                    "trim": trim_ratio,
                    "design_effect": design_effect(weights),
                }
            )

    perf = pd.DataFrame(design_effects)
    candidates = perf[perf.design_effect < max_de]
    if candidates.empty:
        raise ValueError(
            "No regularization parameters satisfy the design effect constraint."
        )

    candidates = candidates.sort_values("design_effect").tail(n_asmd_candidates)
    scored_configs: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        trim_ratio = row.trim
        lambda_index = int(row.s_index)
        link = links[lambda_index]
        weights = weights_from_link(
            link,
            balance_classes,
            sample_weights,
            target_weights,
            weight_trimming_mean_ratio=trim_ratio,
            normalize_to=normalize_to,
        )

        adjusted_df = sample_df[sample_df.index.isin(weights.index)]
        asmd_after = asmd(
            sample_df=adjusted_df,
            target_df=target_df,
            sample_weights=weights,
            target_weights=target_weights,
        )
        asmd_components = asmd_after.drop(labels=["mean(asmd)"], errors="ignore")
        if asmd_components.empty:
            mean_squared_asmd = np.inf
        else:
            asmd_array = np.asarray(asmd_components.astype(float))
            finite_values = asmd_array[np.isfinite(asmd_array)]
            if finite_values.size:
                mean_squared_asmd = float(np.mean(np.square(finite_values)))
            else:
                mean_asmd = asmd_after.get("mean(asmd)")
                if pd.notna(mean_asmd):
                    mean_squared_asmd = float(mean_asmd**2)
                else:
                    mean_squared_asmd = np.inf
        scored_configs.append(
            {
                "s": lambdas[lambda_index],
                "s_index": lambda_index,
                "trim": trim_ratio,
                "design_effect": row.design_effect,
                "mean_squared_asmd": mean_squared_asmd,
            }
        )

    if not scored_configs:
        raise ValueError(
            "No regularization parameters satisfy the design effect constraint."
        )

    all_perf = pd.DataFrame(scored_configs).sort_values("mean_squared_asmd")
    best = all_perf.head(1)
    logger.info(f"Best regularisation: \n {best}")
    return {
        "best": {"s_index": int(best.s_index.values[0]), "trim": best.trim.values[0]},
        "perf": all_perf.head(n_asmd_candidates),
    }


# Lambda regularization parameters can be used to speedup the IPW algorithm,
# counteracting the slow computational speed of sklearn.
def ipw(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: list[str] | None = None,
    model: str | ClassifierMixin | None = "sklearn",
    weight_trimming_mean_ratio: int | float | None = 20,
    weight_trimming_percentile: float | None = None,
    balance_classes: bool = True,
    normalize_weights_to: str = "target",
    transformations: str | None = "default",
    na_action: str = "add_indicator",
    max_de: float | None = None,
    lambda_min: float = 1e-05,
    lambda_max: float = 10,
    num_lambdas: int = 250,
    formula: str | list[str] | None = None,
    penalty_factor: list[float] | None = None,
    one_hot_encoding: bool = False,
    logistic_regression_kwargs: Dict[str, Any] | None = None,
    random_seed: int = 2020,
    sklearn_model: ClassifierMixin | None = None,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit an ipw (inverse propensity score weighting) for the sample using the target.

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target
        variables (Optional[List[str]], optional): list of variables to include in the model.
            If None all joint variables of sample_df and target_df are used. Defaults to None.
        model (Union[str, ClassifierMixin, None], optional): Model used for modeling the
            propensity scores. Provide "sklearn" (default) to use logistic regression,
            or pass an sklearn classifier implementing ``fit`` and ``predict_proba``
            (for example :class:`sklearn.ensemble.RandomForestClassifier` or
            :class:`sklearn.linear_model.LogisticRegression`).
        weight_trimming_mean_ratio (Optional[Union[int, float]], optional): indicating the ratio from above according to which
            the weights are trimmed by mean(weights) * ratio.
            Defaults to 20.
        weight_trimming_percentile (Optional[float], optional): if weight_trimming_percentile is not none, winsorization is applied.
            if None then trimming is applied. Defaults to None.
        balance_classes (bool, optional): whether to balance the sample and target size for running the model.
            True is preferable for imbalanced cases.
            It shouldn't have an effect on the final weights as this is factored
            into the computation of the weights through the odds adjustment.
            Defaults to True.
        transformations (str, optional): what transformations to apply to data before fitting the model.
            See apply_transformations function. Defaults to "default".
        normalize_weights_to (str, optional): normalize final weights to either the "target" (default)
            or "sample" population totals.
        na_action (str, optional): what to do with NAs.
            See add_na_indicator function. Defaults to "add_indicator".
        max_de (Optional[float], optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
        formula (Union[str, List[str], None], optional): The formula according to which build the model.
            In case of list of formula, the model matrix will be built in steps and
            concatenated together. Defaults to None.
        penalty_factor (Optional[List[float]], optional): the penalty factors used in ipw. The penalty
            should have the same length as the formula list (and applies to each element of formula).
            Smaller penalty on some formula will lead to elements in that formula to get more adjusted, i.e. to have a higher chance to get into the model (and not zero out). A penalty of 0 will make sure the element is included in the model.
            If not provided, assume the same penalty (1) for all variables. Defaults to None.
        one_hot_encoding (bool, optional): whether to encode all factor variables in the model matrix with
            almost_one_hot_encoding. This is recomended in case of using
            LASSO on the data (Default: False).
            one_hot_encoding_greater_3 creates one-hot-encoding for all
            categorical variables with more than 2 categories (i.e. the
            number of columns will be equal to the number of categories),
            and only 1 column for variables with 2 levels (treatment contrast). Defaults to False.
        logistic_regression_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments
            passed to :class:`sklearn.linear_model.LogisticRegression`. When None, the
            model defaults to ``penalty="l2"``, ``solver="lbfgs"``, ``tol=1e-4``,
            ``max_iter=5000``, and ``warm_start=True``. Defaults to None.
        random_seed (int, optional): Random seed to use. Defaults to 2020.
        sklearn_model (Optional[ClassifierMixin], optional): Deprecated alias for
            providing a custom sklearn classifier. Use ``model`` instead. The
            estimator must implement ``fit`` and ``predict_proba``. When provided,
            ``logistic_regression_kwargs`` and ``penalty_factor`` are ignored.
            Defaults to None.

    Examples:
        Use the default logistic regression implementation::

            result = ipw(sample_df, sample_w, target_df, target_w)
            weights = result["weight"]

        Fit a custom sklearn classifier (here, ``RandomForestClassifier``)::

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=200, random_state=123)
            result = ipw(
                sample_df,
                sample_w,
                target_df,
                target_w,
                model=rf,
            )
            rf_weights = result["weight"]

    Raises:
        Exception: f"Sample indicator only has value {_n_unique}. This can happen when your sample or target are empty from unknown reason"
        NotImplementedError: if model is not "sklearn"

    Returns:
        Dict[str, Any]: A dictionary includes:
            "weight" --- The weights for the sample.
            "model" --- parameters of the model:fit, performance, X_matrix_columns, lambda,
                        weight_trimming_mean_ratio
            Shape of the Dict:
            {
                "weight": weights,
                "model": {
                    "method": "ipw",
                    "X_matrix_columns": X_matrix_columns_names,
                    "fit": fit,
                    "perf": performance,
                    "lambda": best_s,
                    "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
                },
            }
    """
    custom_model: ClassifierMixin | None = None
    model_name: str | None

    if isinstance(model, ClassifierMixin):
        custom_model = model
        model_name = "sklearn"
    elif model is None:
        model_name = "sklearn"
    elif isinstance(model, str):
        model_name = model
    else:
        raise TypeError(
            "model must be 'sklearn', an sklearn classifier implementing predict_proba, or None"
        )

    if sklearn_model is not None:
        if custom_model is not None:
            raise ValueError("Provide either 'model' or 'sklearn_model', not both.")
        custom_model = sklearn_model

    if model_name == "glmnet":
        raise NotImplementedError("glmnet is no longer supported")
    elif model_name != "sklearn":
        raise NotImplementedError(
            f"Model '{model_name}' is not supported. Only 'sklearn' is currently implemented."
        )

    logger.info("Starting ipw function")
    np.random.seed(
        random_seed
    )  # setting random seed for cases of variations in sklearn

    balance_util._check_weighting_methods_input(sample_df, sample_weights, "sample")
    balance_util._check_weighting_methods_input(target_df, target_weights, "target")

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)
    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    sample_n = sample_df.shape[0]
    target_n = target_df.shape[0]

    # Applying transformations
    # Important! Variables that don't need transformations
    # should be transformed with the *identity function*,
    # otherwise will be dropped from the model
    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations=transformations
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model: {variables}")

    logger.info("Building model matrix")
    # Convert formula to List[str] if it's a single string
    formula_list: list[str] | None = [formula] if isinstance(formula, str) else formula
    model_matrix_output = balance_util.model_matrix(
        sample_df,
        target_df,
        variables,
        add_na=(na_action == "add_indicator"),
        return_type="one",
        return_var_type="sparse",
        formula=formula_list,
        penalty_factor=penalty_factor,
        one_hot_encoding=one_hot_encoding,
    )
    X_matrix = cast(
        Union[pd.DataFrame, np.ndarray, csc_matrix],
        model_matrix_output["model_matrix"],
    )
    X_matrix_columns_names = cast(
        List[str], model_matrix_output["model_matrix_columns_names"]
    )
    penalty_factor_expanded = cast(List[float], model_matrix_output["penalty_factor"])
    logger.info(
        f"The formula used to build the model matrix: {model_matrix_output['formula']}"
    )
    logger.info(f"The number of columns in the model matrix: {X_matrix.shape[1]}")
    logger.info(f"The number of rows in the model matrix: {X_matrix.shape[0]}")

    y = np.concatenate((np.ones(sample_n), np.zeros(target_n)))
    _n_unique = np.unique(y.reshape(y.shape[0]))
    if len(_n_unique) == 1:
        raise Exception(
            f"Sample indicator only has value {_n_unique}. This can happen when your "
            "sample or target are empty from unknown reason"
        )

    if balance_classes:
        odds = np.sum(sample_weights) / np.sum(target_weights)
    else:
        odds = 1
    logger.debug(f"odds for balancing classes: {odds}")

    model_weights = np.concatenate((sample_weights, target_weights * odds))

    logger.debug(f"X_matrix shape: {X_matrix.shape}")
    logger.debug(f"y input shape: {y.shape}")
    logger.debug(
        f"penalty_factor frequency table {pd.crosstab(index=penalty_factor_expanded, columns='count')}"
    )

    foldids = np.resize(range(10), y.shape[0])
    np.random.shuffle(
        foldids
    )  # shuffels the values of foldid - note that we set the seed in the beginning of the function, so this order is fixed
    logger.debug(
        f"foldid frequency table {pd.crosstab(index=foldids, columns='count')}"
    )
    logger.debug(f"first 10 elements of foldids: {foldids[0:9]}")

    logger.debug("Fitting propensity model")

    null_dev = _compute_deviance(
        y,
        np.full(len(y), np.sum(model_weights * y) / np.sum(model_weights)),
        model_weights,
    )

    using_default_logistic = custom_model is None

    if using_default_logistic:
        # Standardize columns of the X matrix and penalize the columns of the X matrix according to the penalty_factor.
        # Workaround for sklearn, which doesn't allow for covariate specific penalty terms.
        # Note that penalty = 0 is not truly supported, and large differences in penalty factors
        # may affect convergence speed.

        scaler = StandardScaler(with_mean=False, copy=False)

        X_matrix = scaler.fit_transform(X_matrix, sample_weight=model_weights)

        penalty_floor = 0.1
        small_penalty_multiplier = 10.0
        penalties_skl = [
            # Clamp very small penalty factors to a fixed multiplier to avoid
            # exploding coefficients while still keeping relative shrinkage
            # across features. Otherwise translate glmnet-style penalties
            # into sklearn's single "C" hyper-parameter space.
            1 / pf if pf > penalty_floor else small_penalty_multiplier
            for pf in penalty_factor_expanded
        ]

        for i in range(len(penalties_skl)):
            X_matrix[:, i] *= penalties_skl[i]

        scale = np.asarray(scaler.scale_)
        if np.any(scale == 0):
            raise ValueError(
                "Encountered zero scale during standardization; remove constant columns and retry."
            )
        feature_scale_factor = np.asarray(penalties_skl) / scale

        X_matrix = csr_matrix(X_matrix)

        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num_lambdas)

        # Using L2 regression since L1 is too slow. Observed "lbfgs" was the most computationally efficient solver.
        lr_kwargs: Dict[str, Any] = {
            "penalty": "l2",
            "solver": "lbfgs",
            "tol": 1e-4,
            "max_iter": 5000,
            "warm_start": True,
        }
        if logistic_regression_kwargs is not None:
            lr_kwargs.update(logistic_regression_kwargs)

        lr = LogisticRegression(**lr_kwargs)
        lr._balance_feature_scale_factor = feature_scale_factor
        fits: list[ClassifierMixin | None] = [None for _ in range(len(lambdas))]
        links: list[np.ndarray | None] = [None for _ in range(len(lambdas))]
        prop_dev = [np.nan for _ in range(len(lambdas))]
        dev = [np.nan for _ in range(len(lambdas))]
        cv_dev_mean = [np.nan for _ in range(len(lambdas))]
        cv_dev_sd = [np.nan for _ in range(len(lambdas))]

        prev_prop_dev = None

        for i in range(len(lambdas)):
            # Conversion between glmnet lambda penalty parameter and sklearn 'C' parameter referenced
            # from https://stats.stackexchange.com/questions/203816/logistic-regression-scikit-learn-vs-glmnet
            lr.C = 1 / (sum(model_weights) * lambdas[i])

            model = lr.fit(X_matrix, y, sample_weight=model_weights)
            pred = model.predict_proba(X_matrix)[:, 1]
            dev[i] = _compute_deviance(y, pred, model_weights)
            prop_dev[i] = _compute_proportion_deviance(dev[i], null_dev)

            # Early stopping criteria: improvement in prop_dev is less than 1e-5 (mirrors glmnet)
            if (
                np.sum(np.abs(model.coef_)) > 0
                and prev_prop_dev is not None
                and prop_dev[i] - prev_prop_dev < 1e-5
            ):
                break

            # Cross-validation procedure is only used for choosing best lambda if max_de is None
            # Previously, cross validation was run even when max_de is not None,
            # but the results weren't used for model selection.
            if max_de is not None:
                logger.debug(
                    f"iter {i}: lambda: {lambdas[i]}, dev: {dev[i]}, prop_dev: {prop_dev[i]}"
                )
            elif num_lambdas > 1:
                dev_mean, dev_sd = calc_dev(X_matrix, y, lr, model_weights, foldids)
                logger.debug(
                    f"iter {i}: lambda: {lambdas[i]}, cv_dev: {dev_mean}, dev_diff: {dev_mean - dev[i]}, prop_dev: {prop_dev[i]}"
                )
                cv_dev_mean[i] = dev_mean
                cv_dev_sd[i] = dev_sd

            prev_prop_dev = prop_dev[i]
            links[i] = link_transform(pred)[:sample_n,]
            fits[i] = copy.deepcopy(model)

    else:
        if logistic_regression_kwargs is not None:
            raise ValueError(
                "logistic_regression_kwargs cannot be used when providing a custom model"
            )
        if penalty_factor is not None:
            logger.warning("penalty_factor is ignored when using a custom model.")

        custom_model = clone(cast(ClassifierMixin, custom_model))
        if not hasattr(custom_model, "predict_proba"):
            raise ValueError(
                "The provided custom model must implement predict_proba for propensity estimation."
            )

        X_matrix = _convert_to_dense_array(X_matrix)

        lambdas = np.array([np.nan])
        fits: list[ClassifierMixin | None] = [None]
        links: list[np.ndarray | None] = [None]
        prop_dev = [np.nan]
        dev = [np.nan]
        cv_dev_mean = [np.nan]
        cv_dev_sd = [np.nan]

        model = custom_model.fit(X_matrix, y, sample_weight=model_weights)
        probas = model.predict_proba(X_matrix)
        if probas.ndim != 2 or probas.shape[1] < 2:
            raise ValueError(
                "The provided custom model predict_proba must return probability estimates for both classes."
            )
        try:
            class_index = list(model.classes_).index(1)
        except ValueError as error:
            raise ValueError(
                "The provided custom model must be trained on the binary labels {0, 1}."
            ) from error
        pred = probas[:, class_index]
        dev[0] = _compute_deviance(y, pred, model_weights)
        prop_dev[0] = _compute_proportion_deviance(dev[0], null_dev)
        links[0] = link_transform(pred)[:sample_n,]
        fits[0] = copy.deepcopy(model)

    logger.info("Done with sklearn")

    logger.info(f"max_de: {max_de}")

    best_s_index = 0
    regularisation_perf = None
    min_s_index = 0

    if max_de is not None:
        regularisation_perf = choose_regularization(
            links,
            lambdas,
            sample_df,
            target_df,
            sample_weights,
            target_weights,
            balance_classes,
            max_de,
            normalize_to=normalize_weights_to,
        )
        best_s_index = regularisation_perf["best"]["s_index"]
        weight_trimming_mean_ratio = regularisation_perf["best"]["trim"]
        weight_trimming_percentile = None
    elif num_lambdas > 1 and using_default_logistic:
        # Cross-validation procedure
        logger.info("Starting model selection")

        min_s_index = np.nanargmin(cv_dev_mean)
        min_dev_mean = cv_dev_mean[min_s_index]
        min_dev_sd = cv_dev_sd[min_s_index]

        # Mirrors 'lambda.1se' from glmnet:
        # 'the most regularized model such that the cross-validated error is within one standard error of the minimum.'
        best_s_index = np.argmax(
            [
                (
                    l
                    if (loss is not np.nan) and (loss < min_dev_mean + min_dev_sd)
                    else 0
                )
                for loss, l in zip(cv_dev_mean, lambdas)
            ]
        )

    best_model = fits[best_s_index]
    link = links[best_s_index]
    best_s = lambdas[best_s_index]

    logger.debug("Predicting")
    weights = weights_from_link(
        link,
        balance_classes,
        sample_weights,
        target_weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
        normalize_to=normalize_weights_to,
    )

    logger.info(f"Chosen lambda: {best_s}")
    best_model = _verify_value_type(best_model)
    performance = model_coefs(
        best_model,
        feature_names=list(X_matrix_columns_names),
    )
    performance["null_deviance"] = null_dev
    performance["deviance"] = dev[best_s_index]
    performance["prop_dev_explained"] = prop_dev[best_s_index]
    if max_de is None and num_lambdas > 1 and using_default_logistic:
        performance["cv_dev_mean"] = cv_dev_mean[best_s_index]
        performance["lambda_min"] = lambdas[min_s_index]
        performance["min_cv_dev_mean"] = cv_dev_mean[min_s_index]
        performance["min_cv_dev_sd"] = cv_dev_sd[min_s_index]

    dev = performance["prop_dev_explained"]
    logger.info(f"Proportion null deviance explained {dev}")

    if (np.max(weights) - np.min(weights)) / np.mean(
        weights
    ) < 1e-04:  # All weights are (essentially) the same
        logger.warning("All weights are identical. The estimates will not be adjusted")

    if dev < 0.10:
        logger.warning(
            "The propensity model has low fraction null deviance explained "
            f"({dev}). Results may not be accurate"
        )

    out = {
        "weight": weights,
        "model": {
            "method": "ipw",
            "X_matrix_columns": X_matrix_columns_names,
            "fit": fits[best_s_index],
            "perf": performance,
            "lambda": best_s,
            "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
            "regularisation_perf": regularisation_perf,
        },
    }

    logger.debug("Done ipw function")

    return out
