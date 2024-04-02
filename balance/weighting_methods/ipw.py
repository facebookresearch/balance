# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from contextlib import contextmanager
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import scipy

from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots.weighted_comparisons_stats import asmd
from balance.stats_and_plots.weights_stats import design_effect

from scipy.sparse.csc import csc_matrix

logger: logging.Logger = logging.getLogger(__package__)


# Allows us to control exactly where monkey patching is applied (e.g.: for better code readability and exceptions tracking).
@contextmanager
def _patch_nan_in_amin_amax(*args, **kwds) -> Generator:
    """This allows us to use nanmin and nanmax instead of amin and amax (thus removing nan from their computation)

    This is needed in cases that the cvglmnet yields some nan values for the cross validation folds.

    Returns:
        Generator: replaces amin and amax, and once done, turns them back to their original version.
    """

    # swap amin and amax with nanmin and nanmax
    # so that they won't return nan when their input has some nan values
    tmp_amin, tmp_amax = scipy.amin, scipy.amax  # pyre-ignore[16]

    # Wrapping amin and amax with logger calls so to alert the user in case nan were present.
    # This comes with the strong assumption that this will occur within the cross validation step!
    def new_amin(a, *args, **kwds) -> Callable:
        nan_count = scipy.count_nonzero(scipy.isnan(a))  # pyre-ignore[16]
        if nan_count > 0:
            logger.warning(
                "The scipy.amin function was replaced with scipy.nanmin."
                f"While running, it removed {nan_count} `nan` values from an array of size {a.size}"
                f"(~{round(100 * nan_count / a.size, 1)}% of the values)."
            )
        return scipy.nanmin(a, *args, **kwds)  # pyre-ignore[16]

    def new_amax(a, *args, **kwds) -> Callable:
        nan_count = scipy.count_nonzero(scipy.isnan(a))  # pyre-ignore[16]
        if nan_count > 0:
            logger.warning(
                "The scipy.amax function was replaced with scipy.nanmax. "
                f"While running, it removed {nan_count} `nan` values from an array of size {a.size}"
                f"(~{round(100 * nan_count / a.size, 1)}% of the values)."
            )
        return scipy.nanmax(a, *args, **kwds)  # pyre-ignore[16]

    scipy.amin = new_amin  # scipy.nanmin
    scipy.amax = new_amax  # scipy.nanmax
    try:
        yield
    finally:
        # undo the function swap
        scipy.amin, scipy.amax = tmp_amin, tmp_amax


@contextmanager
def _patch_scipy_random(*args, **kwds) -> Generator:
    """Monkey-patch scipy.random(), used by glmnet_python
    but removed in scipy 1.9.0"""

    tmp_scipy_random_func = (
        # pyre-ignore[16]
        scipy.random
        if hasattr(scipy, "random")
        else None
    )
    scipy.random = np.random
    try:
        yield
    finally:
        # undo the function swap
        scipy.random = tmp_scipy_random_func


# TODO: consider add option to normalize weights to sample size
def weights_from_link(
    link: Any,
    balance_classes: bool,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    weight_trimming_mean_ratio: Union[None, float, int] = None,
    weight_trimming_percentile: Optional[float] = None,
    keep_sum_of_weights: bool = True,
) -> pd.Series:
    """Transform output of cvglmnetPredict(..., type='link') into weights, by
    exponentiating them, and optionally balancing the classes and trimming
    the weights, then normalize the weights to have sum equal to the sum of the target weights.

    Args:
        link (Any): output of cvglmnetPredict(..., type='link')
        balance_classes (bool): whether balance_classes used in glmnet
        sample_weights (pd.Series): vector of sample weights
        target_weights (pd.Series): vector of sample weights
        weight_trimming_mean_ratio (Union[None, float, int], optional): to be used in :func:`trim_weights`. Defaults to None.
        weight_trimming_percentile (Optional[float], optional): to be used in :func:`trim_weights`. Defaults to None.
        keep_sum_of_weights (bool, optional): to be used in :func:`trim_weights`. Defaults to True.

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
    # Normalize weights such that the sum will be the sum of the weights of target
    weights = weights * np.sum(target_weights) / np.sum(weights)
    return weights


def cv_logistic_regression_performance(
    model, X_matrix, y, cv, feature_names: Optional[list] = None
) -> Dict[str, Any]:
    """Extract elements from GridSearchCV to describe the fitness quality."""
    best_model = model.best_estimator_
    coefs = best_model.coef_[0]
    if feature_names is not None:
        coefs = pd.Series(data=coefs, index=feature_names)
    else:
        coefs = pd.Series(data=coefs)

    return {
        "prop_dev_explained": best_model.score(X_matrix, y),
        "mean_cv_error": -model.best_score_,
        "coefs": coefs,
    }


def ipw(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Optional[List[str]] = None,
    model: str = "logistic",
    weight_trimming_mean_ratio: Optional[Union[int, float]] = 20,
    weight_trimming_percentile: Optional[float] = None,
    balance_classes: bool = True,
    transformations: str = "default",
    na_action: str = "add_indicator",
    max_de: Optional[float] = None,
    formula: Union[str, List[str], None] = None,
    one_hot_encoding: bool = False,
    random_seed: int = 2020,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Fit an ipw (inverse propensity score weighting) for the sample using the target."""
    logger.info("Starting ipw function")
    np.random.seed(random_seed)

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

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations=transformations
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model: {variables}")

    logger.info("Building model matrix")
    model_matrix_output = balance_util.model_matrix(
        sample_df,
        target_df,
        variables,
        add_na=(na_action == "add_indicator"),
        return_type="one",
        return_var_type="sparse",
        formula=formula,
        one_hot_encoding=one_hot_encoding,
    )
    X_matrix = cast(
        Union[pd.DataFrame, np.ndarray, csc_matrix],
        model_matrix_output["model_matrix"],
    )
    X_matrix_columns_names = cast(
        List[str], model_matrix_output["model_matrix_columns_names"]
    )
    logger.info(
        f"The formula used to build the model matrix: {model_matrix_output['formula']}"
    )
    logger.info(f"The number of columns in the model matrix: {X_matrix.shape[1]}")
    logger.info(f"The number of rows in the model matrix: {X_matrix.shape[0]}")

    in_sample = np.concatenate((np.ones(sample_n), np.zeros(target_n)))
    _n_unique = np.unique(in_sample.reshape(in_sample.shape[0]))
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

    y = np.concatenate((np.ones(sample_n), np.zeros(target_n)))
    sample_weight = np.concatenate((sample_weights, target_weights * odds))

    logger.debug(f"X_matrix shape: {X_matrix.shape}")
    logger.debug(f"y input shape: {y.shape}")

    if model == "logistic":
        logger.info("Fitting logistic model")
        param_grid = {
            "C": np.logspace(-4, 4, 20),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }
        lr = LogisticRegression(random_state=random_seed, max_iter=5000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=10, scoring="neg_log_loss", n_jobs=-1
        )
        grid_search.fit(X_matrix, y, sample_weight=sample_weight)
        fit = grid_search.best_estimator_
        logger.debug("Done with GridSearchCV")
    else:
        raise NotImplementedError()

    X_matrix_sample = X_matrix[:sample_n]

    link = fit.predict_proba(X_matrix_sample)[:, 1]
    logger.debug("Predicting")
    weights = weights_from_link(
        link,
        balance_classes,
        sample_weights,
        target_weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
    )

    logger.info(f"Best hyperparameters: {grid_search.best_params_}")

    performance = cv_logistic_regression_performance(
        grid_search, X_matrix, y, cv=10, feature_names=list(X_matrix_columns_names)
    )
    dev = performance["prop_dev_explained"]
    logger.info(f"Proportion null deviance explained {dev}")

    if np.unique(weights).shape[0] == 1:  # All weights are the same
        logger.warning("All weights are identical. The estimates will not be adjusted")

    coefs = performance["coefs"][1:]  # exclude intercept
    if all(abs(coefs) < 1e-14):
        logger.warning(
            (
                "All propensity model coefficients are zero, your covariates do "
                "not predict inclusion in the sample. The estimates will not be "
                "adjusted"
            )
        )

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
            "fit": fit,
            "perf": performance,
            "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
        },
    }

    logger.debug("Done ipw function")

    return out
