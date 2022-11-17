# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from contextlib import contextmanager
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union

import glmnet_python  # noqa  # Required so that cvglmnet import works

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots.weighted_comparisons_stats import asmd
from balance.stats_and_plots.weights_stats import design_effect

from cvglmnet import cvglmnet  # pyre-ignore[21]: this module exists
from cvglmnetPredict import cvglmnetPredict  # pyre-ignore[21]: this module exists
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
    import scipy

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


def cv_glmnet_performance(
    fit, feature_names: Optional[list] = None, s: Union[str, float, None] = "lambda_1se"
) -> Dict[str, Any]:
    """Extract elements from cvglmnet to describe the fitness quality.

    Args:
        fit (_type_): output of cvglmnet
        feature_names (Optional[list], optional): The coeficieents of which features should be included.
            None = all features are included. Defaults to None.
        s (Union[str, float, None], optional): lambda avlue for cvglmnet. Defaults to "lambda_1se".

    Raises:
        Exception: _description_

    Returns:
        Dict[str, Any]: Dict of the shape:
            {
                "prop_dev_explained": fit["glmnet_fit"]["dev"][optimal_lambda_index],
                "mean_cv_error": fit["cvm"][optimal_lambda_index],
                "coefs": coefs,
            }
    """
    if isinstance(s, str):
        optimal_lambda = fit[s]
    else:
        optimal_lambda = s
    optimal_lambda_index = np.where(fit["lambdau"] == optimal_lambda)[0]
    if len(optimal_lambda_index) != 1:
        raise Exception(
            f"No lambda found for s={s}. You must specify a "
            'numeric value which exists in the model or "lambda_min", or '
            '"lambda_1se"'
        )
    coefs = cvglmnetPredict(
        fit,
        newx=np.empty([0]),
        ptype="coefficients",
        s=optimal_lambda,
    )
    coefs = coefs.reshape(coefs.shape[0])
    if feature_names is not None:
        coefs = pd.Series(data=coefs, index=["intercept"] + feature_names)
    else:
        coefs = pd.Series(data=coefs)

    return {
        "prop_dev_explained": fit["glmnet_fit"]["dev"][optimal_lambda_index],
        "mean_cv_error": fit["cvm"][optimal_lambda_index],
        "coefs": coefs,
    }


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


def choose_regularization(
    fit,
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_weights: pd.Series,
    X_matrix_sample,
    balance_classes: bool,
    max_de: float = 1.5,
    trim_options: Tuple[
        int, int, int, float, float, float, float, float, float, float
    ] = (20, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 0.05, 0.01),
    n_asmd_candidates: int = 10,
) -> Dict[str, Any]:
    """Searches through the regularisation parameters of the model and weight
    trimming levels to find the combination with the highest covariate
    ASMD reduction (in sample_df and target_df, NOT in the model matrix used for modeling
    the response) subject to the design effect being lower than max_de (deafults to 1.5).
    The function preforms a grid search over the n_asmd_candidates (deafults to 10) models
    with highest DE lower than max_de (assuming higher DE means more bias reduction).

    Args:
        fit (_type_): output of cvglmnet
        sample_df (pd.DataFrame): a dataframe representing the sample
        target_df (pd.DataFrame): a dataframe representing the target
        sample_weights (pd.Series): design weights for sample
        target_weights (pd.Series): design weights for target
        X_matrix_sample (_type_): the matrix that was used to consturct the model
        balance_classes (bool): whether balance_classes used in glmnet
        max_de (float, optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
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
    # get all links
    links = cvglmnetPredict(fit, X_matrix_sample, ptype="link", s=fit["lambdau"])

    asmd_before = asmd(
        sample_df=sample_df,
        target_df=target_df,
        sample_weights=sample_weights,
        target_weights=target_weights,
    )
    # Grid search over regularisation parameter and weight trimming
    # First calculates design effects for all combinations, because this is cheap
    all_perf = []
    for wr in trim_options:
        for i in range(links.shape[1]):

            s = fit["lambdau"][i]
            link = links[:, i]
            weights = weights_from_link(
                link,
                balance_classes,
                sample_weights,
                target_weights,
                weight_trimming_mean_ratio=wr,
            )

            deff = design_effect(weights)
            all_perf.append(
                {
                    "s": s,
                    "s_index": i,
                    "trim": wr,
                    "design_effect": deff,
                }
            )
    all_perf = pd.DataFrame(all_perf)
    best = (
        # pyre-fixme[16]: `Optional` has no attribute `tail`.
        all_perf[all_perf.design_effect < max_de]
        # pyre-fixme[6]: For 1st param expected `Union[typing_extensions.Literal[0],
        #  typing_extensions.Literal['index']]` but got
        #  `typing_extensions.Literal['design_effect']`.
        .sort_values("design_effect").tail(n_asmd_candidates)
    )
    logger.debug(f"Regularisation with design effect below {max_de}: \n {best}")

    # Calculate ASMDS for best 10 candidates (assuming that higher DE means
    #  more bias reduction)
    all_perf = []
    for _, r in best.iterrows():
        wr = r.trim
        s_index = int(r.s_index)
        s = fit["lambdau"][s_index]
        link = links[:, s_index]

        weights = weights_from_link(
            link,
            balance_classes,
            sample_weights,
            target_weights,
            weight_trimming_mean_ratio=wr,
        )
        adjusted_df = sample_df[sample_df.index.isin(weights.index)]

        asmd_after = asmd(
            # pyre-fixme[6]: For 1st param expected `DataFrame` but got `Series`.
            sample_df=adjusted_df,
            target_df=target_df,
            sample_weights=weights,
            target_weights=target_weights,
        )
        # TODO: use asmd_improvement function for that
        asmd_improvement = (
            asmd_before.loc["mean(asmd)"] - asmd_after.loc["mean(asmd)"]
        ) / asmd_before.loc["mean(asmd)"]
        deff = design_effect(weights)
        all_perf.append(
            {
                "s": s,
                # pyre-fixme[61]: `i` is undefined, or not always defined.
                "s_index": i,
                "trim": wr,
                "design_effect": deff,
                "asmd_improvement": asmd_improvement,
                "asmd": asmd_after.loc["mean(asmd)"],
            }
        )

    all_perf = pd.DataFrame(all_perf)
    best = (
        all_perf[all_perf.design_effect < max_de]
        # pyre-fixme[6]: For 1st param expected `Union[typing_extensions.Literal[0],
        #  typing_extensions.Literal['index']]` but got
        #  `typing_extensions.Literal['asmd_improvement']`.
        .sort_values("asmd_improvement").tail(1)
    )
    logger.info(f"Best regularisation: \n {best}")
    solution = {
        "best": {"s": best.s.values, "trim": best.trim.values[0]},
        "perf": all_perf,
    }
    return solution


# TODO: add memoaization (maybe in the adjust stage?!)
def ipw(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Optional[List[str]] = None,
    model: str = "glmnet",
    weight_trimming_mean_ratio: Optional[Union[int, float]] = 20,
    weight_trimming_percentile: Optional[float] = None,
    balance_classes: bool = True,
    transformations: str = "default",
    na_action: str = "add_indicator",
    # TODO: set max_de to None as default
    max_de: Optional[float] = 1.5,
    formula: Union[str, List[str], None] = None,
    penalty_factor: Optional[List[float]] = None,
    one_hot_encoding: bool = False,
    # TODO: This is set to be false in order to keep reproducibility of works that uses balance.
    # The best practice is for this to be true.
    random_seed: int = 2020,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Fit an ipw (inverse propensity score weighting) for the sample using the target.

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target
        variables (Optional[List[str]], optional): list of variables to include in the model.
            If None all joint variables of sample_df and target_df are used. Defaults to None.
        model (str, optional): the model used for modeling the propensity scores.
            "glmnet" is logistic model. Defaults to "glmnet".
        weight_trimming_mean_ratio (Optional[Union[int, float]], optional): indicating the ratio from above according to which
            the weights are trimmed by mean(weights) * ratio.
            Defaults to 20.
        weight_trimming_percentile (Optional[float], optional): if weight_trimming_percentile is not none, winsorization is applied.
            if None then trimming is applied. Defaults to None.
        balance_classes (bool, optional): whether to balance the sample and target size for running the model.
            True is preferable for imbalanced cases.
            It is done to make the computation of the glmnet more efficient.
            It shouldn't have an effect on the final weights as this is factored
            into the computation of the weights. TODO: add ref. Defaults to True.
        transformations (str, optional): what transformations to apply to data before fitting the model.
            See apply_transformations function. Defaults to "default".
        na_action (str, optional): what to do with NAs.
            See add_na_indicator function. Defaults to "add_indicator".
        max_de (Optional[float], optional): upper bound for the design effect of the computed weights.
            Used for choosing the model regularization and trimming.
            If set to None, then it uses 'lambda_1se'. Defaults to 1.5.
        formula (Union[str, List[str], None], optional): The formula according to which build the model.
            In case of list of formula, the model matrix will be built in steps and
            concatenated together. Defaults to None.
        penalty_factor (Optional[List[float]], optional): the penalty used in the glment function in ipw. The penalty
            should have the same length as the formula list. If not provided,
            assume the same panelty for all variables. Defaults to None.
        one_hot_encoding (bool, optional): whether to encode all factor variables in the model matrix with
            almost_one_hot_encoding. This is recomended in case of using
            LASSO on the data (Default: False).
            one_hot_encoding_greater_3 creates one-hot-encoding for all
            categorical variables with more than 2 categories (i.e. the
            number of columns will be equal to the number of categories),
            and only 1 column for variables with 2 levels (treatment contrast). Defaults to False.
        random_seed (int, optional): Random seed to use. Defaults to 2020.

    Raises:
        Exception: _description_
        NotImplementedError: _description_

    Returns:
        Dict[str, Any]: A dictionary includes:
            "weights" --- The weights for the sample.
            "model" --- parameters of the model:fit, performance, X_matrix_columns, lambda,
                        weight_trimming_mean_ratio
            Shape of the Dict:
            {
                "weights": weights,
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
    logger.info("Starting ipw function")
    np.random.seed(
        random_seed
    )  # setting random seed for cases of variations in cvglmnet

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
    model_matrix_output = balance_util.model_matrix(
        sample_df,
        target_df,
        variables,
        add_na=(na_action == "add_indicator"),
        return_type="one",
        return_var_type="sparse",
        # pyre-fixme[6]: for 7th parameter `formula` expected `Optional[List[str]]` but got `Union[None, List[str], str]`.
        # TODO: fix pyre issue
        formula=formula,
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
    penalty_factor = cast(Optional[List[float]], model_matrix_output["penalty_factor"])
    # in cvglmnet: "penalty factors are internally rescaled to sum to nvars."
    # (https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html)
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

    y0 = np.concatenate((np.zeros(sample_n), target_weights * odds))
    y1 = np.concatenate((sample_weights, np.zeros(target_n)))
    y = np.column_stack((y0, y1))
    # From glmnet documentation: https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html
    # "For binomial logistic regression, the response variable y should be either
    # a factor with two levels, or a two-column matrix of counts or proportions."

    logger.debug(f"X_matrix shape: {X_matrix.shape}")
    logger.debug(f"y input shape: {y.shape}")
    logger.debug(
        f"penalty_factor frequency table {pd.crosstab(index=penalty_factor, columns='count')}"
        f"Note that these are normalized by cvglmnet"
    )

    if model == "glmnet":
        logger.info("Fitting logistic model")
        foldid = np.resize(range(10), y.shape[0])
        np.random.shuffle(
            foldid
        )  # shuffels the values of foldid - note that we set the seed in the beginning of the function, so this order is fixed
        logger.debug(
            f"foldid frequency table {pd.crosstab(index=foldid, columns='count')}"
        )
        logger.debug(f"first 10 elements of foldid: {foldid[0:9]}")
        with _patch_nan_in_amin_amax():
            # we use _patch_nan_in_amin_amax here since sometimes
            # cvglmnet could have one of the cross validated samples that
            # produce nan. In which case, the lambda search returns nan
            # instead of a value from the cross-validated options that successfully computed a lambda
            # The current monkey-patch solves this problem and makes the function fail less.

            with np.errstate(
                divide="ignore"
            ):  # ignoring np warning "divide by zero encountered in log"
                fit = cvglmnet(
                    x=X_matrix,
                    y=y,
                    family="binomial",
                    ptype="deviance",
                    alpha=1,
                    penalty_factor=penalty_factor,
                    nlambda=250,
                    lambda_min=np.array([1e-6]),
                    nfolds=10,
                    foldid=foldid,
                    maxit=5000,
                    *args,
                    **kwargs,
                )
        logger.debug("Done with cvglmnet")
    else:
        raise NotImplementedError()
    logger.debug(f"fit['lambda_1se']: {fit['lambda_1se']}")

    X_matrix_sample = X_matrix[
        :sample_n,
    ].toarray()

    if max_de is not None:
        regularisation_perf = choose_regularization(
            fit,
            sample_df,
            target_df,
            sample_weights,
            target_weights,
            X_matrix_sample,
            balance_classes,
            max_de,
        )
        best_s = regularisation_perf["best"]["s"]
        weight_trimming_mean_ratio = regularisation_perf["best"]["trim"]
        weight_trimming_percentile = None
    else:
        best_s = fit["lambda_1se"]
        regularisation_perf = None

    link = cvglmnetPredict(fit, X_matrix_sample, ptype="link", s=best_s)
    logger.debug("Predicting")
    weights = weights_from_link(
        link,
        balance_classes,
        sample_weights,
        target_weights,
        weight_trimming_mean_ratio,
        weight_trimming_percentile,
    )

    logger.info(f"Chosen lambda for cv: {best_s}")

    performance = cv_glmnet_performance(
        fit,
        feature_names=list(X_matrix_columns_names),
        s=best_s,
    )
    dev = performance["prop_dev_explained"]
    logger.info(f"Proportion null deviance explained {dev}")

    if np.unique(weights).shape[0] == 1:  # All weights are the same
        logger.warning("All weights are identical. The estimates will not be adjusted")

    coefs = performance["coefs"][1:]  # exclude intercept
    if all(abs(coefs) < 1e-14):
        # The value was determined by the unit-test test_adjustment/test_ipw_bad_adjustment_warnings
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
        "weights": weights,
        "model": {
            "method": "ipw",
            "X_matrix_columns": X_matrix_columns_names,
            "fit": fit,
            "perf": performance,
            "lambda": best_s,
            "weight_trimming_mean_ratio": weight_trimming_mean_ratio,
        },
    }
    if max_de is not None:
        out["model"]["regularisation_perf"] = regularisation_perf

    logger.debug("Done ipw function")

    return out
