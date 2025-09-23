# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Optional, Union

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
    variables: Optional[List[str]] = None,
    transformations: str = "default",
    transformations_drop: bool = True,
    strict_matching: bool = True,
    *args,
    **kwargs,
) -> Dict[str, Union[pd.Series, Dict[str, str]]]:
    """Perform cell-based post-stratification. The output weights take into account
    the design weights and the post-stratification weights.
    Reference: https://docs.wfp.org/api/documents/WFP-0000121326/download/

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target
        variables (Optional[List[str]], optional): list of variables to include in the model.
            If None all joint variables of sample_df and target_df are used
        transformations (str, optional): what transformations to apply to data before fitting the model.
            Default is "default" (see apply_transformations function)
        transformations_drop (bool, optional): whether the function should drop non-transformed variables.
            Default is True.
        strict_matching (bool, optional): whether to require all cells in the sample be in the target.
            Default is True. When False, a warning is provided and
            samples in cells not covered by the target are given weight 0.
    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Dict[str, Union[pd.Series, Dict[str, str]]]:
            weight (pd.Series): final weights (sum up to target's sum of weights)
            model (dict): method of adjustment

            Dict shape:
            {
                "weight": w,
                "model": {"method": "poststratify"},
            }
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
        if strict_matching:
            raise ValueError(
                "all combinations of cells in sample_df must be in target_df. Set strict_matching=False to continue."
            )
        else:
            logger.warning(
                "Detected some cells in sample_df that are not in target_df. "
                "Samples in cells not covered by the target are given weight 0."
            )
            from balance.util import _safe_fillna_and_infer

            combined["weight"] = _safe_fillna_and_infer(combined["weight"], 0)

    combined["weight"] = combined["weight"] / combined["design_weight"]
    sample_df = sample_df.join(combined["weight"], on=variables)
    w = sample_df.weight * sample_df.design_weight

    return {
        "weight": w,
        "model": {"method": "poststratify"},
    }
