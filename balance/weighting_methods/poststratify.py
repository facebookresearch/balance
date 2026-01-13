# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from balance import adjustment as balance_adjustment, util as balance_util

logger: logging.Logger = logging.getLogger(__package__)


# TODO: Add tests for all arguments of function
def poststratify(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Optional[List[str]] = None,
    transformations: str = "default",
    transformations_drop: bool = True,
    strict_matching: bool = True,
    na_action: Union[str, bool] = "add_indicator",
    weight_trimming_mean_ratio: Union[float, int, None] = None,
    weight_trimming_percentile: Union[float, None] = None,
    keep_sum_of_weights: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Union[pd.Series, Dict[str, str]]]:
    """
    Perform cell-based post-stratification to adjust sample weights so that the sample matches the joint distribution of, one or more, specified variables in the target population.

    This method computes one weight per *cell* - a unique combination of the supplied variables - so that the weighted sample reproduces the cell distribution observed in the target population.
    When more than one variable is supplied, the function operates on cells from the joint distribution (as opposed to raking, which operates on the marginals distribution).

    Reference: https://docs.wfp.org/api/documents/WFP-0000121326/download/

    Args:
        sample_df (pd.DataFrame): DataFrame representing the sample.
        sample_weights (pd.Series): Design weights for the sample.
        target_df (pd.DataFrame): DataFrame representing the target population.
        target_weights (pd.Series): Design weights for the target.
        variables (Optional[List[str]], optional): List of variables to define post-stratification cells. If None, uses the intersection of columns in sample_df and target_df.
        transformations (str, optional): Transformations to apply to data before fitting the model. Default is "default". See `balance.adjustment.apply_transformations`.
        transformations_drop (bool, optional): If True, drops variables not affected by transformations. Default is True.
        strict_matching (bool, optional): If True, requires all sample cells to be present in the target. If False, cells missing in the target are assigned weight 0 (and a warning is raised). Default is True.
        na_action (Union[str, bool], optional): How to handle missing values. Use
            ``True``/``"add_indicator"`` to treat missing values as their own category, or
            ``False``/``"drop"`` to remove rows with missing values from both sample and
            target. Defaults to ``"add_indicator"``.
        weight_trimming_mean_ratio (Union[float, int, None], optional): Forwarded to
            :func:`balance.adjustment.trim_weights` to clip weights at a multiple of the mean.
        weight_trimming_percentile (Union[float, None], optional): Percentile limit(s) for
            winsorisation, passed to :func:`balance.adjustment.trim_weights`.
        keep_sum_of_weights (bool, optional): Preserve the sum of weights during trimming before
            the final normalisation to the target total. Defaults to True.
        *args: Additional positional arguments (currently unused).
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        dict:
            weight (pd.Series): Final weights for the sample, summing to the target's total weight.
            model (dict): Description of the adjustment method used.

    Raises:
        ValueError: If strict_matching is True and some sample cells are missing in the target.

    Notes:
        * The function expects that every combination of ``variables`` present
          in ``sample_df`` is also present in ``target_df``. Set
          ``strict_matching=False`` to keep rows whose cell is missing in the
          target and assign them weight 0.
        * When no ``variables`` are provided, the intersection of columns in
          ``sample_df`` and ``target_df`` is used. In practice you will
          usually provide a small number of categorical variables (often one
          or two) describing the post-stratification cells.

    Examples:
        Post-stratifying on a single categorical variable:

            >>> import pandas as pd
            >>> sample_df = pd.DataFrame({"gender": ["Female", "Male", "Female"]})
            >>> target_df = pd.DataFrame({"gender": ["Female", "Female", "Male", "Male"]})
            >>> design = pd.Series(1, index=sample_df.index)
            >>> target_design = pd.Series(1, index=target_df.index)
            >>> weights = poststratify(
            ...     sample_df=sample_df,
            ...     sample_weights=design,
            ...     target_df=target_df,
            ...     target_weights=target_design,
            ...     variables=["gender"],
            ... )["weight"]
            >>> weights.tolist()
            [1.0, 2.0, 1.0]

        Post-stratifying on the joint distribution of two variables (the
        resulting weights depend on the combination of both columns rather
        than their marginals):

            >>> sample_df = pd.DataFrame(
            ...     {
            ...         "gender": ["Female", "Female", "Male", "Male"],
            ...         "age_group": ["18-34", "35+", "18-34", "35+"],
            ...     }
            ... )
            >>> target_df = pd.DataFrame(
            ...     {
            ...         "gender": ["Female", "Female", "Female", "Male", "Male", "Male"],
            ...         "age_group": ["18-34", "18-34", "35+", "18-34", "35+", "35+"],
            ...     }
            ... )
            >>> design = pd.Series(1, index=sample_df.index)
            >>> target_design = pd.Series(1, index=target_df.index)
            >>> weights = poststratify(
            ...     sample_df=sample_df,
            ...     sample_weights=design,
            ...     target_df=target_df,
            ...     target_weights=target_design,
            ...     variables=["gender", "age_group"],
            ... )["weight"]
            >>> weights.tolist()
            [2.0, 1.0, 1.0, 2.0]
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

    if na_action is True:
        na_action = "add_indicator"
    elif na_action is False:
        na_action = "drop"

    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        from balance.util import _safe_fillna_and_infer

        sample_df = _safe_fillna_and_infer(sample_df, "__NaN__")
        target_df = _safe_fillna_and_infer(target_df, "__NaN__")
    else:
        raise ValueError("`na_action` must be 'add_indicator' or 'drop'")

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
    raw_weights = sample_df.weight * sample_df.design_weight
    target_total = raw_weights.sum()
    w = balance_adjustment.trim_weights(
        raw_weights,
        target_sum_weights=target_total,
        weight_trimming_mean_ratio=weight_trimming_mean_ratio,
        weight_trimming_percentile=weight_trimming_percentile,
        keep_sum_of_weights=keep_sum_of_weights,
    ).rename(raw_weights.name)

    return {
        "weight": w,
        "model": {"method": "poststratify"},
    }
