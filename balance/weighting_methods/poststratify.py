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
    """Perform cell-based post-stratification.

    The method computes one weight per *cell* — a unique combination of the
    supplied variables — so that the weighted sample reproduces the cell
    distribution observed in the target population. When more than one
    variable is supplied, the function still operates on those joint cells; it
    does **not** fall back to raking (which would instead match the marginals
    of each variable one at a time).

    Reference: https://docs.wfp.org/api/documents/WFP-0000121326/download/

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample.
        sample_weights (pd.Series): design weights for the sample.
        target_df (pd.DataFrame): a dataframe representing the target.
        target_weights (pd.Series): design weights for the target.
        variables (Optional[List[str]], optional): list of variables to include
            in the model. If ``None`` all joint variables shared by
            ``sample_df`` and ``target_df`` are used.
        transformations (str, optional): what transformations to apply to data
            before fitting the model. Default is ``"default"`` (see
            :func:`balance.adjustment.apply_transformations`).
        transformations_drop (bool, optional): whether the function should
            drop non-transformed variables. Default is ``True``.
        strict_matching (bool, optional): whether to require all cells in the
            sample to be present in the target. Default is ``True``. When set
            to ``False`` a warning is provided and samples in cells that are
            missing from the target receive weight ``0``.

    Returns:
        Dict[str, Union[pd.Series, Dict[str, str]]]:
            ``weight`` contains the final weights (they sum up to the target's
            total weight) and ``model`` describes the adjustment method.

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
        >>> from balance.weighting_methods.poststratify import poststratify
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
