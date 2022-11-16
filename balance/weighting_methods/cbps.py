# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from typing import Any, cast, Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy
import sklearn.utils.extmath
import statsmodels.api as sm

from balance import adjustment as balance_adjustment, util as balance_util
from balance.stats_and_plots.weights_stats import design_effect
from numpy import ndarray
from scipy.sparse import csc_matrix

logger: logging.Logger = logging.getLogger(__package__)


def logit_truncated(
    X: Union[np.ndarray, pd.DataFrame], beta: np.ndarray, truncation_value: float = 1e-5
) -> np.ndarray:
    """
    This is a helper function for cbps.
    Given an X matrx and avector of coeeficients beta, it computes the truncated
    version of the logit function.

    Arguments:
        X --- (np.ndarray) Covariate matrix
        beta --- (np.ndarray) vector of coefficients
        truncation_value --- (float) upper and lower bound for the computed probabilities

    Returns:
        np.ndarray of computed probablities
    """
    probs = 1.0 / (1 + np.exp(-1 * (np.matmul(X, beta))))
    return np.minimum(np.maximum(probs, truncation_value), 1 - truncation_value)


def compute_pseudo_weights_from_logit_probs(
    probs: np.ndarray,
    design_weights: Union[np.ndarray, pd.DataFrame],
    in_pop: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """
    This is a helper function for cbps.
    Given computed probs, it computes the weights: N/N_t * (in_pop - p_i)/(1 - p_i).
    (Note that these weights on sample are negative for convenience of notations)

    Arguments:
        probs --- (np.ndarray) vector of probabilities
        design_weights --- (np.ndarray) vector of design weights of sample and target
        in_pop --- (np.ndarray) indicator vector for target

    Returns:
        np.ndarray of computed weights
    """
    N = np.sum(design_weights)
    N_target = np.sum(design_weights[in_pop == 1.0])
    return N / N_target * (in_pop - probs) / (1 - probs)


def bal_loss(
    beta: np.ndarray,
    X: np.ndarray,
    design_weights: np.ndarray,
    in_pop: np.ndarray,
    XtXinv: np.ndarray,
) -> np.float64:
    """
    This is a helper function for cbps.
    It computes the balance loss.

    Arguments:
        beta --- (np.ndarray) vector of coefficients
        X --- (np.ndarray) Covariate matrix
        design_weights --- (np.ndarray) vector of design weights of sample and target
        in_pop --- (np.ndarray) indicator vector for target
        XtXinv --- (np.ndarray) (X.T %*% X)^(-1)

    Returns:
        (float) computed balance loss
    """
    probs = logit_truncated(X, beta)
    N = np.sum(design_weights)
    weights = (
        1.0 / N * compute_pseudo_weights_from_logit_probs(probs, design_weights, in_pop)
    )

    Xprimew = np.matmul((X * design_weights[:, None]).T, weights)
    loss = np.absolute(np.matmul(np.matmul(Xprimew.T, XtXinv), Xprimew))

    return loss


def gmm_function(
    beta: np.ndarray,
    X: Union[np.ndarray, pd.DataFrame],
    design_weights: Union[np.ndarray, pd.DataFrame],
    in_pop: Union[np.ndarray, pd.DataFrame],
    invV: Union[np.ndarray, None] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    This is a helper function for cbps.
    It computes the gmm loss.

    Arguments:
        beta --- (np.ndarray) vector of coefficients
        X --- (np.ndarray) covariates matrix
        design_weights --- (np.ndarray) vector of design weights of sample and target
        in_pop --- (np.ndarray) indicator vector for target
        invV --- (np.ndarray) the inverse weighting matrix for GMM. Default is None.

    Returns:
        loss --- (float) computed gmm loss
        invV --- (np.ndarray) the weighting matrix for GMM
    """
    probs = logit_truncated(X, beta)
    N = np.sum(design_weights)
    N_target = np.sum(design_weights[in_pop == 1.0])

    weights = compute_pseudo_weights_from_logit_probs(probs, design_weights, in_pop)

    # Generate gbar
    gbar = np.concatenate(
        (
            1.0 / N * (np.matmul((X * design_weights[:, None]).T, (in_pop - probs))),
            1.0 / N * (np.matmul((X * design_weights[:, None]).T, weights)),
        )
    )

    if invV is None:
        # Compute inverse sigma matrix to use in GMM estimate
        design_weights_sq = np.sqrt(design_weights)[:, None]
        X1 = design_weights_sq * X * np.sqrt((1 - probs) * probs)[:, None]
        X2 = design_weights_sq * X * np.sqrt(probs / (1 - probs))[:, None]
        X11 = design_weights_sq * X * np.sqrt(probs)[:, None]
        X11TX11 = np.matmul(X11.T, X11 * N / N_target)

        V = (
            1
            / N
            * np.vstack(
                (
                    np.hstack((np.matmul(X1.T, X1), X11TX11)),
                    np.hstack(
                        (X11TX11, np.matmul(X2.T, X2 * np.power(N / N_target, 2)))
                    ),
                )
            )
        )
        # Note - The R CBPS code is using sum(treat) for N_target instead of
        # the sum of the design weights on the treated
        invV = np.linalg.pinv(V)

    # Compute loss
    loss = np.matmul(np.matmul(gbar.T, invV), gbar)
    return {"loss": loss, "invV": invV}


def gmm_loss(
    beta: ndarray,
    X: Union[ndarray, pd.DataFrame],
    design_weights: Union[ndarray, pd.DataFrame],
    in_pop: Union[ndarray, pd.DataFrame],
    invV: Optional[ndarray] = None,
) -> Union[float, ndarray]:
    """
    This is a helper function for cbps.
    It computes the gmm loss.
    See gmm_function for detials.
    """
    return gmm_function(beta, X, design_weights, in_pop, invV)["loss"]


def alpha_function(
    alpha,
    beta,
    X: Union[ndarray, pd.DataFrame],
    design_weights: Union[ndarray, pd.DataFrame],
    in_pop: Union[ndarray, pd.DataFrame],
) -> Union[float, ndarray]:
    """
    This is a helper function for cbps.
    It computes the gmm loss of alpha*beta.
    """
    return gmm_loss(alpha * beta, X, design_weights, in_pop)


def compute_deff_from_beta(
    X: np.ndarray, beta: np.ndarray, design_weights: np.ndarray, in_pop: np.ndarray
) -> np.float64:
    """
    This is a helper function for cbps. It computes the design effect of
    the estimated weights on the sample given a value of beta.
    It is used for setting a constraints on max_de.

    Arguments:
        beta --- (np.ndarray) vector of coefficients
        X --- (np.ndarray) covariates matrix
        design_weights --- (np.ndarray) vector of design weights of sample and target
        in_pop --- (np.ndarray) indicator vector for target

    Returns:
        (float) design effect
    """
    probs = logit_truncated(X, beta)
    weights = np.absolute(
        compute_pseudo_weights_from_logit_probs(probs, design_weights, in_pop)
    )
    weights = design_weights[in_pop == 0.0] * weights[in_pop == 0.0]
    return design_effect(weights)


def _standardize_model_matrix(
    model_matrix, model_matrix_columns_names
) -> Dict[str, Any]:
    """
    This is a helper function for cbps. It standardizes the columns of the model matrix.
    """
    # TODO: Verify if scikit-learn have something similar
    model_matrix_std = np.asarray(np.std(model_matrix, axis=0)).reshape(
        -1
    )  # This is needed if the input is 2 dim numpy array
    if np.sum(model_matrix_std == 0) > 0:
        variables_to_omit = model_matrix_std == 0
        names_variables_to_omit = np.array(model_matrix_columns_names)[
            variables_to_omit
        ]
        logger.warning(
            f"The following variables have only one level, and are omitted: {names_variables_to_omit}"
        )
        model_matrix_columns_names = list(
            np.array(model_matrix_columns_names)[np.invert(variables_to_omit)]
        )
        model_matrix = model_matrix[:, np.invert(variables_to_omit)]
        model_matrix_std = model_matrix_std[np.invert(variables_to_omit)]
    model_matrix_mean = np.mean(model_matrix, axis=0)
    model_matrix = (model_matrix - model_matrix_mean) / model_matrix_std
    return {
        "model_matrix": model_matrix,
        "model_matrix_columns_names": model_matrix_columns_names,
        "model_matrix_mean": model_matrix_mean,
        "model_matrix_std": model_matrix_std,
    }


def _reverse_svd_and_centralization(beta, U, s, Vh, X_matrix_mean, X_matrix_std):
    """
    This is a helper function for cbps. It revrse the svd and the centralization to get an estimate of beta
    Source: https://github.com/kosukeimai/CBPS/blob/master/R/CBPSMain.R#L353
    """
    # TODO: update SVD and reverse SVD to the functions in scikit-learn
    # Invert s
    s_inv = s
    s_inv[s_inv > 1e-5] = 1 / s_inv[s_inv > 1e-5]
    s_inv[s_inv <= 1e-5] = 0
    # Compute beta
    beta = np.matmul(Vh.T * s_inv, beta)
    beta_new = np.delete(beta, 0) / X_matrix_std
    beta_new = np.insert(
        beta_new,
        0,
        beta[
            0,
        ]
        - np.matmul(X_matrix_mean, beta_new),
    )
    return beta_new


def cbps(  # noqa
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables=None,
    transformations: str = "default",
    na_action: str = "add_indicator",
    formula=None,
    balance_classes: bool = True,
    cbps_method: str = "over",  # other option: "exact"
    max_de=None,
    opt_method: str = "COBYLA",
    opt_opts=None,
    weight_trimming_mean_ratio: Union[None, float, int] = 20,
    weight_trimming_percentile: Optional[float] = None,
    random_seed: int = 2020,
    *args,
    **kwargs,
):
    """
    Fit cbps (covariate balancing propensity score model) for the sample using the target.
    Final weights are normalized to target size.
    We use a two-step GMM estimator (as in the default R package), unlike the suggeted continuous-updating
    estimator in the paper. The reason is that it runs much faster than the continuous one.

    Paper: Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society: Series B: Statistical Methodology, 243-263.
    https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12027
    R code source: https://github.com/kosukeimai/CBPS
    two-step GMM: https://en.wikipedia.org/wiki/Generalized_method_of_moments

    Arguments:
    sample_df --- (pandas dataframe) a dataframe representing the sample
    sample_weights --- (pandas series) design weights for sample
    target_df ---  (pandas dataframe) a dataframe representing the target
    target_weights --- (pandas series) design weights for target
    variables ---  (list of strings) list of variables to include in the model.
                   If None all joint variables of sample_df and target_df are used
    transformations --- (dict) what transformations to apply to data before fitting the model. §
                        Default is "default" (see apply_transformations function)
    na_action --- (string) what to do with NAs. Default is "add_indicator"
                  (see add_na_indicator function)
    formula --- (string or list of strings) The formula according to which build the model.
                In case of list of formula, the model matrix will be built in steps and
                concatenated together.
    balance_classes --- (boolean) whether to balance the sample and target size for running the model.
                              Default is True (preferable for imbalanced cases).
    cbps_method --- (string) method used for cbps. "over" fits an over-identified model that combines
                    the propensity score and covariate balancing conditions; "exact" fits a model that only c
                    ontains the covariate balancing conditions.
    max_de --- (float) upper bound for the design effect of the computed weights.
                Default is None.
    opt_method --- (string) type of optimization solver. Default is "COBYLA". See scipy.optimize.minimize
                   for other options.
    opt_opts --- (dict) A dictionary of solver options. Default is None. See scipy.optimize.minimize
                   for other options.
    weight_trimming_mean_ratio ---  (float or int) indicating the ratio from above according to which
                                    the weights are trimmed by mean(weights) * ratio.
                                    Default is 20.
    weight_trimming_percentile --- (float) if weight_trimming_percentile is not none, winsorization is applied.
                                    Default is None, i.e. trimming is applied.
    random_seed --- (int) a random seed

    Returns:
    A dictionary includes:
        "weights" --- The weights for the sample.
        "model" -- dictionary with details about the fitted model:
            X_matrix_columns, deviance, beta_optimal, balance_optimize_result,
            gmm_optimize_result_glm_init, gmm_optimize_result_bal_init
    """
    logger.info("Starting cbps function")
    np.random.seed(random_seed)  # setting random seed for cases of variations in glmnet

    balance_util._check_weighting_methods_input(sample_df, sample_weights, "sample")
    balance_util._check_weighting_methods_input(target_df, target_weights, "target")

    # Choose joint variables from sample and target
    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)
    logger.debug(f"Joint variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    # keeping index of sample df to use for final weights
    sample_index = sample_df.index

    # Applying transformations
    # Important! Variables that don't need transformations
    # should be transformed with the *identity function*,
    # otherwise will be dropped from the model
    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations=transformations
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model: {variables}")

    # Build X matrix
    model_matrix_output = balance_util.model_matrix(
        sample_df,
        target_df,
        variables,
        add_na=(na_action == "add_indicator"),
        return_type="one",
        return_var_type="sparse",
        formula=formula,
        one_hot_encoding=False,
    )
    # TODO: Currently using a dense version of the X matrix. We might change to using the sparse version if need.
    X_matrix = cast(
        Union[pd.DataFrame, np.ndarray, csc_matrix],
        (model_matrix_output["model_matrix"]),
    ).toarray()
    X_matrix_columns_names = model_matrix_output["model_matrix_columns_names"]
    logger.info(
        f"The formula used to build the model matrix: {model_matrix_output['formula']}"
    )

    # Standardize the X_matrix for SVD
    model_matrix_standardized = _standardize_model_matrix(
        X_matrix, X_matrix_columns_names
    )
    X_matrix = model_matrix_standardized["model_matrix"]
    X_matrix_columns_names = model_matrix_standardized["model_matrix_columns_names"]
    logger.info(f"The number of columns in the model matrix: {X_matrix.shape[1]}")
    logger.info(f"The number of rows in the model matrix: {X_matrix.shape[0]}")

    # Adding intercept since model_matrix removes it
    X_matrix = np.c_[np.ones(X_matrix.shape[0]), X_matrix]
    X_matrix_columns_names.insert(0, "Intercept")

    # SVD for X_matrix
    U, s, Vh = scipy.linalg.svd(X_matrix, full_matrices=False)
    # Make the sign of the SVD deterministic
    U, Vh = sklearn.utils.extmath.svd_flip(U, Vh, u_based_decision=False)
    # TODO: add stop: if (k < ncol(X)) stop("X is not full rank")

    sample_n = sample_df.shape[0]
    target_n = target_df.shape[0]
    total_n = sample_n + target_n

    # Create "treatment" (in_sample) variable
    in_sample = np.concatenate((np.ones(sample_n), np.zeros(target_n)))
    in_pop = np.concatenate((np.zeros(sample_n), np.ones(target_n)))
    if len(np.unique(in_sample.reshape(in_sample.shape[0]))) == 1:
        _number_unique = np.unique(in_sample.reshape(in_sample.shape[0]))
        raise Exception(
            f"Sample indicator only has value {_number_unique}. This can happen when your sample or target are empty from unknown reason"
        )

    # balance classes
    if balance_classes:
        design_weights = (
            total_n
            / 2
            * np.concatenate(
                (
                    sample_weights / np.sum(sample_weights),
                    target_weights / np.sum(target_weights),
                )
            )
        )

        XtXinv = np.linalg.pinv(
            # pyre-fixme[16] Undefined attribute [16]: `float` has no attribute `__getitem__`.
            np.matmul((U * design_weights[:, None]).T, U * design_weights[:, None])
        )
    else:
        design_weights = np.concatenate((sample_weights, target_weights))
        design_weights = design_weights / np.mean(design_weights)
        XtXinv = np.linalg.pinv(np.matmul(U.T, U))

    # Define constraints for optimization to limit max_de
    constraints = []
    if max_de is not None:
        constraints += [
            # max de effect
            {
                "type": "ineq",
                "fun": lambda x: (
                    max_de - compute_deff_from_beta(U, x, design_weights, in_pop)
                ),
            }
        ]

    # Optimization using Generalized Methods of Moments:
    # Step 0 - initial estimation for beta
    logger.info("Finding initial estimator for GMM optimization")
    glm_mod = sm.GLM(
        in_pop, U, family=sm.families.Binomial(), freq_weights=design_weights
    )
    beta_0 = glm_mod.fit().params
    # TODO: add some safty for when this fails?

    # Step 1 - rescale initial estimation for beta by minimizing the gmm loss
    rescale_initial_result = scipy.optimize.minimize(
        alpha_function,
        x0=[1],
        args=(
            beta_0,
            U,
            design_weights,
            in_pop,
        ),
        bounds=[(0.8, 1.1)],  # These are the bounds used in the R CBPS package
    )
    if rescale_initial_result["success"] is np.bool_(False):
        logger.warning(
            f"Convergence of alpha_function has failed due to '{rescale_initial_result['message']}'"
        )
    gmm_init = beta_0 * rescale_initial_result["x"][0]
    invV = gmm_function(gmm_init, U, design_weights, in_pop, invV=None)["invV"]

    # Step 2 - find initial estimation for beta by minimizing balance loss
    logger.info(
        "Finding initial estimator for GMM optimization that minimizes the balance loss"
    )
    balance_optimize_result = scipy.optimize.minimize(
        fun=bal_loss,
        x0=gmm_init,
        args=(
            U,
            design_weights,
            in_pop,
            XtXinv,
        ),
        method=opt_method,
        options=opt_opts,
        constraints=constraints,
    )
    if balance_optimize_result["success"] is np.bool_(False):
        logger.warning(
            f"Convergence of bal_loss function has failed due to '{balance_optimize_result['message']}'"
        )
    beta_balance = balance_optimize_result["x"]

    gmm_optimize_result_glm_init = None
    if cbps_method == "exact":
        if (
            balance_optimize_result["success"] is np.bool_(False)
            and "Did not converge to a solution satisfying the constraints"
            in balance_optimize_result["message"]
        ):
            raise Exception("There is no solution satisfying the constraints.")
        beta_opt = beta_balance

    # Step 3 - minimize gmm_loss from two starting points: beta_balance and gmm_init,
    # and choose the solution that minimize the gmm_loss.
    elif cbps_method == "over":
        logger.info("Running GMM optimization")
        gmm_optimize_result_glm_init = scipy.optimize.minimize(
            fun=gmm_loss,
            x0=gmm_init,
            args=(
                U,
                design_weights,
                in_pop,
                invV,
            ),
            method=opt_method,
            options=opt_opts,
            constraints=constraints,
        )
        if gmm_optimize_result_glm_init["success"] is np.bool_(False):
            logger.warning(
                f"Convergence of gmm_loss function with gmm_init start point has failed due to '{gmm_optimize_result_glm_init['message']}'"
            )

        gmm_optimize_result_bal_init = scipy.optimize.minimize(
            fun=gmm_loss,
            x0=beta_balance,
            args=(
                U,
                design_weights,
                in_pop,
                invV,
            ),
            method=opt_method,
            options=opt_opts,
            constraints=constraints,
        )
        if gmm_optimize_result_bal_init["success"] is np.bool_(False):
            logger.warning(
                f"Convergence of gmm_loss function with beta_balance start point has failed due to '{gmm_optimize_result_bal_init['message']}'"
            )

        # If the constraints cannot be satisfied, exit the function
        if (
            gmm_optimize_result_glm_init["success"] is np.bool_(False)
            and "Did not converge to a solution satisfying the constraints"
            in gmm_optimize_result_glm_init["message"]
            and gmm_optimize_result_bal_init["success"] is np.bool_(False)
            and "Did not converge to a solution satisfying the constraints"
            in gmm_optimize_result_bal_init["message"]
        ):
            raise Exception("There is no solution satisfying the constraints.")

        # Choose beta that gives a smaller loss in GMM and that satisfy the constraints
        if (
            gmm_optimize_result_bal_init["fun"] < gmm_optimize_result_glm_init["fun"]
            and "Did not converge to a solution satisfying the constraints"
            not in gmm_optimize_result_bal_init["message"]
        ):
            beta_opt = gmm_optimize_result_bal_init["x"]
        else:
            beta_opt = gmm_optimize_result_glm_init["x"]

    else:
        raise Exception(f"cbps_method '{cbps_method}' is not a valid option")

    # Compute final probs and weights with beta_opt
    probs = logit_truncated(U, beta_opt)
    logger.debug(f"Minimum probs for sample: {np.min(probs[in_sample == 1.0])}")
    logger.debug(f"Maximum probs for sample: {np.max(probs[in_sample == 1.0])}")
    weights = np.absolute(
        compute_pseudo_weights_from_logit_probs(probs, design_weights, in_pop)
    )
    logger.debug(f"Minimum weight for sample: {np.min(weights[in_sample == 1.0])}")
    logger.debug(f"Maximum weight for sample: {np.max(weights[in_sample == 1.0])}")
    weights = design_weights[in_sample == 1.0] * weights[in_sample == 1.0]

    # Compute deviance (Minus twice the log-likelihood of the CBPS fit)
    deviance = -2 * np.sum(
        in_pop * design_weights * np.log(probs)
        + (1 - in_pop) * design_weights * np.log(1 - probs)
    )

    # trim weights
    weights = balance_adjustment.trim_weights(
        weights, weight_trimming_mean_ratio, weight_trimming_percentile
    )
    # normalize to target size
    original_sum_weights = np.sum(weights)
    logger.debug(f"original sum of weights for sample: {original_sum_weights}")
    weights = weights / original_sum_weights * np.sum(target_weights)

    # set index to sample_df
    weights = pd.DataFrame({"weights": weights}).set_index(sample_index)["weights"]

    if np.unique(weights).shape[0] == 1 or weights.describe()["std"] < 1e-4:
        # All weights are the same
        logger.warning(
            "All weights are identical (or almost identical). The estimates will not be adjusted"
        )

    # Revrse SVD and centralization
    beta_opt = _reverse_svd_and_centralization(
        beta_opt,
        U,
        s,
        Vh,
        model_matrix_standardized["model_matrix_mean"],
        model_matrix_standardized["model_matrix_std"],
    )

    out = {
        "weights": weights,
        "model": {
            "method": "cbps",
            "X_matrix_columns": X_matrix_columns_names,
            "deviance": deviance,
            "original_sum_weights": original_sum_weights,  # This can be used to reconstruct the propensity probablities
            "beta_optimal": beta_opt,
            "beta_init_glm": beta_0,  # The initial estimator by glm
            "gmm_init": gmm_init,  # The rescaled initial estimator
            # The following are the results of the optimizations
            "rescale_initial_result": rescale_initial_result,
            "balance_optimize_result": balance_optimize_result,
            "gmm_optimize_result_glm_init": gmm_optimize_result_glm_init
            if cbps_method == "over"
            else None,
            # pyre-fixme[61]: `gmm_optimize_result_bal_init` is undefined, or not
            #  always defined.
            "gmm_optimize_result_bal_init": gmm_optimize_result_bal_init
            if cbps_method == "over"
            else None,
        },
    }
    logger.info("Done cbps function")

    return out
