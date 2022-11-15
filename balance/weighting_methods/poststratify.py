# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict

import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util

logger: logging.Logger = logging.getLogger(__package__)


# TODO: Add tests for all arguments of function
# TODO: Add argument for na_action
def poststratify(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables=None,
    transformations: str = "default",
    transformations_drop: bool = True,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform cell-based post-stratification. The output weights take into account
    the design weights and the post-stratification weights.
    Reference: https://docs.wfp.org/api/documents/WFP-0000121326/download/

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target
        variables (list of strings or None): list of variables to include in the model.
            If None all joint variables of sample_df and target_df are used
        transformations (dict): what transformations to apply to data before fitting the model.
            Default is "default" (see apply_transformations function)
        transformations_drop (bool): whether the function should drop non-transformed variables.
            Default is True.

    Returns:
        Dict:
            weights (pd.Series): final weights (sum up to target's sum of weights)
            model (dict): method of adjustment
    """
    balance_util._check_weighting_methods_input(sample_df, sample_weights, "sample")
    balance_util._check_weighting_methods_input(target_df, target_weights, "target")

    if ("weight" in sample_df.columns.values) or ("weight" in target_df.columns.values):
        raise ValueError(
            "weight can't be a name of a column in sample or target when applying poststratify"
        )

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)
    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df),
        transformations=transformations,
        drop=transformations_drop,
    )
    variables = list(sample_df.columns)
    logger.debug(f"Final variables in the model after transformations: {variables}")

    target_df = target_df.assign(weight=target_weights)
    target_cell_props = target_df.groupby(list(variables))["weight"].sum()

    sample_df = sample_df.assign(design_weight=sample_weights)
    sample_cell_props = sample_df.groupby(list(variables))["design_weight"].sum()

    combined = pd.merge(
        target_cell_props,
        sample_cell_props,
        right_index=True,
        left_index=True,
        how="outer",
    )

    # check that all combinations of cells in sample_df are also in target_df
    if any(combined["weight"].isna()):
        raise ValueError("all combinations of cells in sample_df must be in target_df")

    combined["weight"] = combined["weight"] / combined["design_weight"]
    sample_df = sample_df.join(combined["weight"], on=variables)
    w = sample_df.weight * sample_df.design_weight

    return {
        "weights": w,
        "model": {"method": "poststratify"},
    }
