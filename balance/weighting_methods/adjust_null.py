# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, Union

import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)


def adjust_null(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    *args,
    **kwargs,
) -> Dict[str, Union[Dict[str, str], pd.Series]]:
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
    """

    return {
        "weights": sample_weights,
        "model": {
            "method": "null_adjustment",
        },
    }
