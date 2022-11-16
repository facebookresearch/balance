# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict

logger: logging.Logger = logging.getLogger(__package__)


def adjust_null(
    sample_df, sample_weights, target_df, target_weights, *args, **kwargs
) -> Dict[str, Dict[str, str]]:
    """
    Doesn't apply any adjustment to the data. Returns the design weights as they are.
    This may be useful when one needs the output of Sample.adjust() (i.e.: an adjusted object),
    but wishes to not run any model for it.

    Arguments:
        sample_df --- (pandas dataframe) a dataframe representing the sample
        sample_weights --- (pandas series) design weights for sample
        target_df ---  (pandas dataframe) a dataframe representing the target
        target_weights --- (pandas series) design weights for target

    Returns:
        Dict of weights (original sample weights) and model (with method = null_adjustment)
    """
    return {
        "weights": sample_weights,
        "model": {
            "method": "null_adjustment",
        },
    }
