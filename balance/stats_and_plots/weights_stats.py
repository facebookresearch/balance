# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__package__)


##########################################
# Weights diagnostics - functions for analyzing weights
##########################################


def _check_weights_are_valid(
    w: Union[
        List,
        pd.Series,
        np.ndarray,
        pd.DataFrame,
        None,
    ]
) -> None:
    """Check weights.

    Args:
        w (Union[ List, pd.Series, np.ndarray, pd.DataFrame, None, ]): input weights.
            If w is pd.DataFrame then only the first column will be checked (assuming it is a column of weights).
            If input is None, then the function returns None with no errors (since None is a valid weights input for various functions).

    Raises:
        ValueError: if weights are not numeric.
        ValueError: if weights include a negative value.

    Returns:
        _type_: None
    """
    if w is None:
        return None
    if isinstance(w, pd.DataFrame):
        w = w.iloc[:, 0]  # if DataFrame, we check only the first column.
    if not isinstance(w, pd.Series):
        w = pd.Series(w)
        # TODO: (p2) consider having a check for each type of w, instead of
        #            turning w into pd.Series (since this solution might not be very efficient)
    if not np.issubdtype(w, np.number):
        raise TypeError(
            f"weights (w) must be a number but instead they are of type: {w.dtype}."
        )
    if any(w < 0):
        raise ValueError("weights (w) must all be non-negative values.")
    # TODO: do we also want to verify that at least one weight is larger than 0?!

    return None


# TODO: if the input is pd.DataFrame than the output will be pd.Series.
#       we could make the support of this more official in the future.
def design_effect(w: pd.Series) -> np.float64:
    """
    Kish's design effect measure.

    For details, see:
    - https://en.wikipedia.org/wiki/Design_effect
    - https://en.wikipedia.org/wiki/Effective_sample_size

    Args:
        w (pd.Series): A pandas series of weights (non negative, float/int) values.

    Returns:
        np.float64: An estimator saying by how much the variance of the mean is expected to increase, compared to a random sample mean,
        due to application of the weights.

    Examples:
        ::

            from balance.stats_and_plots.weights_stats import design_effect
            import pandas as pd

            design_effect(pd.Series((0, 1, 2, 3)))
                # output:
                # 1.5555555555555556
            design_effect(pd.Series((1, 1, 1000)))
                # 2.9880418803112336
                # As expected. With a single dominating weight - the Deff is almost equal to the sample size.
    """
    _check_weights_are_valid(w)
    return (w**2).mean() / (w.mean() ** 2)


def nonparametric_skew(w: pd.Series) -> np.float64:
    # TODO (p2): consider adding other skew measures (https://en.wikipedia.org/wiki/Skewness)
    #            look more in the literature (are there references for using this vs another, or none at all?)
    #            update the doc with insights, once done:
    #            what is more accepted in the literature in the field and what are the advantages of each.
    #            Any reference to literature where this is used to analyze weights of survey?
    #            Add reference to some interpretation of these values?
    """
    The nonparametric skew is the difference between the mean and the median, divided by the standard deviation.
    See:
    - https://en.wikipedia.org/wiki/Nonparametric_skew

    Args:
        w (pd.Series): A pandas series of weights (non negative, float/int) values.

    Returns:
        np.float64: A value of skew, between -1 to 1, but for weights it's often positive (i.e.: right tailed distribution).
        The value returned will be 0 if the standard deviation is 0 (i.e.: all values are identical), or if the input is of length 1.

    Examples:
        ::

            from balance.stats_and_plots.weights_stats import nonparametric_skew

            nonparametric_skew(pd.Series((1, 1, 1, 1)))  # 0
            nonparametric_skew(pd.Series((1)))           # 0
            nonparametric_skew(pd.Series((1, 2, 3, 4)))  # 0
            nonparametric_skew(pd.Series((1, 1, 1, 2)))  # 0.5
            nonparametric_skew(pd.Series((-1,1,1, 1)))   #-0.5

    """
    _check_weights_are_valid(w)
    if (len(w) == 1) or (w.std() == 0):
        return np.float64(0)
    return (w.mean() - w.median()) / w.std()


def prop_above_and_below(
    w: pd.Series,
    below: Union[Tuple[float, ...], List[float], None] = (
        1 / 10,
        1 / 5,
        1 / 3,
        1 / 2,
        1,
    ),
    above: Union[Tuple[float, ...], List[float], None] = (1, 2, 3, 5, 10),
    return_as_series: bool = True,
) -> Union[pd.Series, Dict[Any, Any], None]:
    # TODO (p2): look more in the literature (are there references for using this vs another, or none at all?)
    #            update the doc with insights, once done.
    """
    The proportion of weights, normalized to sample size, that are above and below some numbers (E.g. 1,2,3,5,10 and their inverse: 1, 1/2, 1/3, etc.).
    This is similar to returning percentiles of the (normalized) weighted distribution. But instead of focusing on the 25th percentile, the median, etc,
    We focus instead on more easily interpretable weights values.

    For example, saying that some proportion of users had a weight of above 1 gives us an indication of how many users
    we got that we don't "loose" their value after using the weights. Saying which proportion of users had a weight below 1/10 tells us how many users
    had basically almost no contribution to the final analysis (after applying the weights).

    Note that below and above can overlap, be unordered, etc. The user is responsible for the order.

    Args:
        w (pd.Series): A pandas series of weights (float, non negative) values.
        below (Union[Tuple[float, ...], List[float], None], optional):
            values to check which proportion of normalized weights are *below* them.
            Using None returns None.
            Defaults to (1/10, 1/5, 1/3, 1/2, 1).
        above (Union[Tuple[float, ...], List[float], None], optional):
            values to check which proportion of normalized weights are *above* (or equal) to them.
            Using None returns None.
            Defaults to (1, 2, 3, 5, 10).
        return_as_series (bool, optional): If true returns one pd.Series of values.
            If False will return a dict with two pd.Series (one for below and one for above).
            Defaults to True.

    Returns:
        Union[pd.Series, Dict]:
        If return_as_series is True we get pd.Series with proportions of (normalized weights)
        that are below/above some numbers, the index indicates which threshold was checked
        (the values in the index are rounded up to 3 points for printing purposes).
        If return_as_series is False we get a dict with 'below' and 'above' with the relevant pd.Series (or None).

    Examples:
        ::

            from balance.stats_and_plots.weights_stats import prop_above_and_below
            import pandas as pd

            # normalized weights:
            print(pd.Series((1, 2, 3, 4)) / pd.Series((1, 2, 3, 4)).mean())
                # 0    0.4
                # 1    0.8
                # 2    1.2
                # 3    1.6

            # checking the function:
            prop_above_and_below(pd.Series((1, 2, 3, 4)))
                # dtype: float64
                # prop(w < 0.1)      0.00
                # prop(w < 0.2)      0.00
                # prop(w < 0.333)    0.00
                # prop(w < 0.5)      0.25
                # prop(w < 1.0)      0.50
                # prop(w >= 1)       0.50
                # prop(w >= 2)       0.00
                # prop(w >= 3)       0.00
                # prop(w >= 5)       0.00
                # prop(w >= 10)      0.00
                # dtype: float64

            prop_above_and_below(pd.Series((1, 2, 3, 4)), below = (0.1, 0.5), above = (2,3))
                # prop(w < 0.1)    0.00
                # prop(w < 0.5)    0.25
                # prop(w >= 2)     0.00
                # prop(w >= 3)     0.00
                # dtype: float64

            prop_above_and_below(pd.Series((1, 2, 3, 4)), return_as_series = False)
                # {'below': prop(w < 0.1)      0.00
                # prop(w < 0.2)      0.00
                # prop(w < 0.333)    0.00
                # prop(w < 0.5)      0.25
                # prop(w < 1)        0.50
                # dtype: float64, 'above': prop(w >= 1)     0.5
                # prop(w >= 2)     0.0
                # prop(w >= 3)     0.0
                # prop(w >= 5)     0.0
                # prop(w >= 10)    0.0
                # dtype: float64}

    """
    _check_weights_are_valid(w)

    # normalize weight to sample size:
    w = w / w.mean()

    if below is None and above is None:
        return None

    # calculate props from below:
    if below is not None:
        prop_below = [(w < i).mean() for i in below]
        prop_below_index = ["prop(w < " + str(round(i, 3)) + ")" for i in below]
        prop_below_series = pd.Series(prop_below, index=prop_below_index)
    else:
        prop_below_series = None

    # calculate props from above:
    if above is not None:
        prop_above = [(w >= i).mean() for i in above]
        prop_above_index = ["prop(w >= " + str(round(i, 3)) + ")" for i in above]
        prop_above_series = pd.Series(prop_above, index=prop_above_index)
    else:
        prop_above_series = None

    # decide if to return one series or a dict
    if return_as_series:
        out = pd.concat(
            [  # pyre-ignore[6]: pd.concat supports Series.
                prop_below_series,
                prop_above_series,
            ]
        )
    else:
        out = {"below": prop_below_series, "above": prop_above_series}

    return out  # pyre-ignore[7]:  TODO: see if we can fix this pyre


def weighted_median_breakdown_point(w: pd.Series) -> np.float64:
    # TODO (p2): do we want to have weighted_quantile_breakdown_point
    # so to check for quantiles other than 50%?
    """
    Calculates the minimal percent of users that have at least 50% of the weights.
    This gives us the breakdown point of calculating the weighted median.
    This can be thought of as reflecting a similar metric to the design effect.
    See also:
    - https://en.wikipedia.org/wiki/Weighted_median
    - https://en.wikipedia.org/wiki/Robust_statistics#Breakdown_point

    Args:
        w (pd.Series): A pandas series of weights (float, non negative values).

    Returns:
        np.float64: A minimal percent of users that contain at least 50% of the weights.

    Examples:
        ::

            w = pd.Series([1,1,1,1])
            print(weighted_median_breakdown_point(w)) # 0.5

            w = pd.Series([2,2,2,2])
            print(weighted_median_breakdown_point(w)) # 0.5

            w = pd.Series([1,1,1, 10])
            print(weighted_median_breakdown_point(w)) # 0.25

            w = pd.Series([1,1,1,1, 10])
            print(weighted_median_breakdown_point(w)) # 0.2
    """
    _check_weights_are_valid(w)

    # normalize weight to sample size:

    n = len(w)  # n users
    w = w / w.sum()  # normalize to 1
    # get a cumsum of sorted weights to find the median:
    w_freq_cumsum = w.sort_values(  # pyre-ignore[16]: it does have a cumsum method.
        ascending=False
    ).cumsum()
    numerator = (w_freq_cumsum <= 0.5).sum()
    if numerator == 0:
        numerator = (
            1  # this happens if one observation has more than 50% of the weights
        )
    # find minimal proportion of samples needed to reach 50%
    # the +1 trick is to deal with cases that 1 user has a weight that is larget then 50%.
    return numerator / n  # breakdown_point
