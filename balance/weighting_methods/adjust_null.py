# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import logging

import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)


def adjust_null(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    *args: object,
    **kwargs: object,
) -> dict[str, dict[str, str] | pd.Series]:
    """Doesn't apply any adjustment to the data. Returns the design weights as they are.
    This may be useful when one needs the output of Sample.adjust() (i.e.: an adjusted object),
    but wishes to not run any model for it.

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample
        sample_weights (pd.Series): design weights for sample
        target_df (pd.DataFrame): a dataframe representing the target
        target_weights (pd.Series): design weights for target

    Returns:
        Dict[str, Union[Dict[str, str], pd.Series]]: Dict of weights (original sample weights) and model (with method = null_adjustment)

    Examples:
    .. code-block:: python

        import pandas as pd
        from balance.weighting_methods.adjust_null import adjust_null
        sample_df = pd.DataFrame({"x": [0, 1]})
        target_df = pd.DataFrame({"x": [0, 1]})
        weights = pd.Series([1.0, 2.0])
        result = adjust_null(sample_df, weights, target_df, weights)
        result["model"]["method"]
        # 'null_adjustment'
    """

    return {
        "weight": sample_weights,
        "model": {
            "method": "null_adjustment",
        },
    }
