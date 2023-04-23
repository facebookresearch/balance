# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util
from ipfn import ipfn

logger = logging.getLogger(__package__)


# TODO: Add options for only marginal distributions input
def rake(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Union[List[str], None] = None,
    transformations: Union[Dict[str, Callable], str] = "default",
    na_action: str = "add_indicator",
    max_iteration: int = 1000,
    convergence_rate: float = 0.0005,
    rate_tolerance: float = 1e-8,
    *args,
    **kwargs,
) -> Dict:
    """
    Perform raking (using the iterative proportional fitting algorithm).
    See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    Returns weights normalised to sum of target weights

    Arguments:
    sample_df --- (pandas dataframe) a dataframe representing the sample.
    sample_weights --- (pandas series) design weights for sample.
    target_df ---  (pandas dataframe) a dataframe representing the target.
    target_weights --- (pandas series) design weights for target.
    variables ---  (list of strings) list of variables to include in the model.
                   If None all joint variables of sample_df and target_df are used.
    transformations --- (dict) what transformations to apply to data before fitting the model.
                               Default is "default" (see apply_transformations function).
    na_action --- (string) what to do with NAs. Default is "add_indicator", which adds NaN as a
                  group (called "__NaN__") for each weighting variable (post-transformation);
                  "drop" removes rows with any missing values on any variable from both sample
                  and target.
    max_iteration --- (int) maximum number of iterations for iterative proportional fitting algorithm
    convergence_rate --- (float) convergence criteria; the maximum difference in proportions between
                          sample and target marginal distribution on any covariate in order
                          for algorithm to converge.
    rate_tolerance --- (float) convergence criteria; if convergence rate does not move more
                               than this amount than the algorithm is also considered to
                               have converged.

    Returns:
    A dictionary including:
    "weight" --- The weights for the sample.
    "model" --- parameters of the model: iterations (dataframe with iteration numbers and
                convergence rate information at all steps), converged (Flag with the output
                status: 0 for failure and 1 for success).
    """
    assert (
        "weight" not in sample_df.columns.values
    ), "weight shouldn't be a name for covariate in the sample data"
    assert (
        "weight" not in target_df.columns.values
    ), "weight shouldn't be a name for covariate in the target data"

    # TODO: move the input checks into separate funnction for rake, ipw, poststratify
    assert isinstance(sample_df, pd.DataFrame), "sample_df must be a pandas DataFrame"
    assert isinstance(target_df, pd.DataFrame), "target_df must be a pandas DataFrame"
    assert isinstance(
        sample_weights, pd.Series
    ), "sample_weights must be a pandas Series"
    assert isinstance(
        target_weights, pd.Series
    ), "target_weights must be a pandas Series"
    assert sample_df.shape[0] == sample_weights.shape[0], (
        "sample_weights must be the same length as sample_df"
        f"{sample_df.shape[0]}, {sample_weights.shape[0]}"
    )
    assert target_df.shape[0] == target_weights.shape[0], (
        "target_weights must be the same length as target_df"
        f"{target_df.shape[0]}, {target_weights.shape[0]}"
    )

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)

    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    assert len(variables) > 1, (
        "Must weight on at least two variables for raking. "
        f"Currently have variables={variables} only"
    )

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations
    )

    # TODO: separate into a function that handles NA (for rake, ipw, poststratify)
    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        target_df = target_df.fillna("__NaN__")
        sample_df = sample_df.fillna("__NaN__")
    else:
        raise ValueError("`na_action` must be 'add_indicator' or 'drop'")

    # Cast all data types as string to be explicit about each unique value
    # being its own group and to handle that `fillna()` above creates
    # series of type Object, which won't work for the ipfn script
    levels_dict = {}
    for variable in variables:
        target_df[variable] = target_df[variable].astype(str)
        sample_df[variable] = sample_df[variable].astype(str)

        sample_var_set = set(sample_df[variable].unique())
        target_var_set = set(target_df[variable].unique())
        sample_over_set = sample_var_set - target_var_set
        target_over_set = target_var_set - sample_var_set
        if len(sample_over_set):
            raise ValueError(
                "All variable levels in sample must be present in target. "
                f"'{variable}' in target is missing these levels: {sample_over_set}."
            )
        if len(target_over_set):
            logger.warning(
                f"'{variable}' has more levels in target than in sample. "
                f"'{variable}' in sample is missing these levels: {target_over_set}. "
                "These levels are treated as if they do not exist for that variable."
            )
        levels_dict[variable] = list(sample_var_set.intersection(target_var_set))

    logger.info(
        f"Final covariates and levels that will be used in raking: {levels_dict}."
    )

    target_df = target_df.assign(weight=target_weights)
    sample_df = sample_df.assign(weight=sample_weights)

    sample_sum_weights = sample_df["weight"].sum()
    target_sum_weights = target_df["weight"].sum()

    # Alphabetize variables to ensure consistency across covariate order
    # (ipfn algorithm is iterative and variable order can matter on the margins)
    alphabetized_variables = list(variables)
    alphabetized_variables.sort()

    logger.debug(
        f"Alphabetized variable order is as follows: {alphabetized_variables}."
    )

    # margins from population
    target_margins = [
        (
            (
                target_df.groupby(variable)["weight"].sum()
                / (sum(target_df.groupby(variable)["weight"].sum()))
                * sample_sum_weights
            )
        )
        for variable in alphabetized_variables
    ]

    # sample cells (joint distribution of covariates)
    sample_cells = (
        sample_df.groupby(alphabetized_variables)["weight"].sum().reset_index()
    )

    logger.debug(
        "Raking algorithm running following settings: "
        f" convergence_rate: {convergence_rate}; max_iteration: {max_iteration}; rate_tolerance: {rate_tolerance}"
    )
    # returns dataframe with joint distribution of covariates and total weight
    # for that specific set of covariates
    ipfn_obj = ipfn.ipfn(
        original=sample_cells,
        aggregates=target_margins,
        dimensions=[[var] for var in alphabetized_variables],
        weight_col="weight",
        convergence_rate=convergence_rate,
        max_iteration=max_iteration,
        verbose=2,
        rate_tolerance=rate_tolerance,
    )
    fit, converged, iterations = ipfn_obj.iteration()

    logger.debug(
        f"Raking algorithm terminated with following convergence: {converged}; "
        f"and iteration meta data: {iterations}."
    )

    if not converged:
        logger.warning("Maximum iterations reached, convergence was not achieved")

    raked = pd.merge(
        sample_df.reset_index(),
        fit.rename(columns={"weight": "rake_weight"}),
        how="left",
        on=list(variables),
    )

    # Merge back to original sample weights
    raked_rescaled = pd.merge(
        raked,
        (
            sample_df.groupby(list(variables))["weight"]
            .sum()
            .reset_index()
            .rename(columns={"weight": "total_survey_weight"})
        ),
        how="left",
        on=list(variables),
    ).set_index(
        "index"
    )  # important if dropping rows due to NA

    # use above merge to give each individual sample its proportion of the
    # cell's total weight
    raked_rescaled["rake_weight"] = (
        raked_rescaled["rake_weight"] / raked_rescaled["total_survey_weight"]
    )
    # rescale weights to sum to target_sum_weights (sum of initial target weights)
    w = (
        raked_rescaled["rake_weight"] / raked_rescaled["rake_weight"].sum()
    ) * target_sum_weights
    return {
        "weight": w,
        "model": {
            "method": "rake",
            "iterations": iterations,
            "converged": converged,
            "perf": {"prop_dev_explained": np.array([np.nan])},
            # TODO: fix functions that use the perf and remove it from here
        },
    }
