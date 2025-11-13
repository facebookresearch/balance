# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from balance import util as balance_util
from balance.weighting_methods import (
    adjust_null as balance_adjust_null,
    cbps as balance_cbps,
    ipw as balance_ipw,
    poststratify as balance_poststratify,
    rake as balance_rake,
)
from pandas.api.types import is_bool_dtype, is_numeric_dtype

logger: logging.Logger = logging.getLogger(__package__)


BALANCE_WEIGHTING_METHODS = {
    "ipw": balance_ipw.ipw,
    "cbps": balance_cbps.cbps,
    "null": balance_adjust_null.adjust_null,
    "poststratify": balance_poststratify.poststratify,
    "rake": balance_rake.rake,
}


def trim_weights(
    weights: Union[pd.Series, npt.NDArray],
    # TODO: add support to more types of input weights? (e.g. list? other?)
    weight_trimming_mean_ratio: Union[float, int, None] = None,
    weight_trimming_percentile: Union[float, None] = None,
    verbose: bool = False,
    keep_sum_of_weights: bool = True,
) -> pd.Series:
    """Trim extreme weights.

    The user cannot supply both weight_trimming_mean_ratio and weight_trimming_percentile.
    If none are supplied, the original weights are returned.

    If `weight_trimming_mean_ratio` is not None, the weights are trimmed from above by
    mean(weights) * ratio. The weights are then normalized to have the original mean.
    Note that trimmed weights aren't actually bounded by trimming.ratio because the
    reduced weight is redistributed to arrive at the original mean.

    If `weight_trimming_percentile` is not None, the weights are trimmed according to the percentiles of the distribution of the weights.
    Note that weight_trimming_percentile by default clips both sides of the distribution, unlike
    trimming that only trims the weights from above.
    For example, `weight_trimming_percentile=0.1` trims below the 10th percentile AND above the 90th.
    If you only want to trim the upper side, specify `weight_trimming_percentile = (0, 0.1)`. If you only want to trim the lower side, specify
    `weight_trimming_percentile = (0.1, 0)`.

    Args:
        weights (Union[pd.Series, np.ndarray]): pd.Series of weights to trim. np.ndarray will be turned into pd.Series) of weights.
        weight_trimming_mean_ratio (Union[float, int], optional): indicating the ratio from above according to which
            the weights are trimmed by mean(weights) * ratio. Defaults to None.
        weight_trimming_percentile (Union[float], optional): if `weight_trimming_percentile` is not None,
            then we apply winsorization using :func:`scipy.stats.mstats.winsorize`. Ranges between 0 and 1.
            If a single value is passed, indicates the percentiles on both sides of the weight distribution beyond which the weights will be winsorized.
            If two values are passed, the first value is the lower percentiles below which winsorizing will be applied, and the second is the 1. - upper percentile above which winsorizing will be applied.
            For example, `weight_trimming_percentile=(0.01, 0.05)` will trim the weights with values below the 1st percentile and above the 95th percentile of the weight distribution.
            See also: [https://en.wikipedia.org/wiki/Winsorizing].
            Defaults to None.
        verbose (bool, optional): whether to add to logger printout of trimming process.
            Defaults to False.
        keep_sum_of_weights (bool, optional): Set if the sum of weights after trimming
            should be the same as the sum of weights before trimming.
            Defaults to True.

    Raises:
        TypeError: If weights is not np.array or pd.Series.
        ValueError: If both weight_trimming_mean_ratio and weight_trimming_percentile are set.

    Returns:
        pd.Series (of type float64): Trimmed weights

    Examples:
        ::

            import pandas as pd
            from balance.adjustment import trim_weights
            print(trim_weights(pd.Series(range(1, 101)), weight_trimming_mean_ratio = None))
                # 0       1.0
                # 1       2.0
                # 2       3.0
                # 3       4.0
                # 4       5.0
                #     ...
                # 95     96.0
                # 96     97.0
                # 97     98.0
                # 98     99.0
                # 99    100.0
                # Length: 100, dtype: float64

            print(trim_weights(pd.Series(range(1, 101)), weight_trimming_mean_ratio = 1.5))
                # 0      1.064559
                # 1      2.129117
                # 2      3.193676
                # 3      4.258235
                # 4      5.322793
                #         ...
                # 95    80.640316
                # 96    80.640316
                # 97    80.640316
                # 98    80.640316
                # 99    80.640316
                # Length: 100, dtype: float64

            print(pd.DataFrame(trim_weights(pd.Series(range(1, 101)), weight_trimming_percentile=.01)))
                # 0    2.0
                # 1    2.0
                # 2    3.0
                # 3    4.0
                # 4    5.0
                # ..   ...
                # 95  96.0
                # 96  97.0
                # 97  98.0
                # 98  99.0
                # 99  99.0
                # [100 rows x 1 columns]

            print(pd.DataFrame(trim_weights(pd.Series(range(1, 101)), weight_trimming_percentile=(0., .05))))
                # 0    1.002979
                # 1    2.005958
                # 2    3.008937
                # 3    4.011917
                # 4    5.014896
                # ..        ...
                # 95  95.283019
                # 96  95.283019
                # 97  95.283019
                # 98  95.283019
                # 99  95.283019
    """

    if isinstance(weights, pd.Series):
        weights = weights.astype(np.float64, copy=False)
        weights_index = weights.index
    elif isinstance(weights, np.ndarray):
        weights = pd.Series(weights, dtype=np.float64)
        weights_index = weights.index
    else:
        raise TypeError(
            f"weights must be np.array or pd.Series, are of type: {type(weights)}"
        )

    n_weights = len(weights)
    if n_weights == 0:
        return pd.Series(dtype=np.float64)

    if (weight_trimming_mean_ratio is not None) and (
        weight_trimming_percentile is not None
    ):
        raise ValueError(
            "Only one of weight_trimming_mean_ratio and "
            "weight_trimming_percentile can be set"
        )

    original_mean = float(np.mean(weights))

    if weight_trimming_mean_ratio is not None:
        max_val = weight_trimming_mean_ratio * original_mean
        percent_trimmed = weights[weights > max_val].count() / weights.count()
        weights = weights.clip(upper=max_val)
        if verbose:
            if percent_trimmed > 0:
                logger.debug("Clipping weights to %s (before renormalizing)" % max_val)
                logger.debug("Clipped %s of the weights" % percent_trimmed)
            else:
                logger.debug("No extreme weights were trimmed")
    elif weight_trimming_percentile is not None:
        # Winsorize
        percentile = weight_trimming_percentile
        if isinstance(percentile, (list, tuple, np.ndarray)):
            if len(percentile) != 2:
                raise ValueError(
                    "weight_trimming_percentile must be a single value or a length-2 iterable"
                )
            lower_limit, upper_limit = percentile
        else:
            lower_limit = upper_limit = percentile

        def _validate_limit(limit: Union[float, int, None]) -> Union[float, None]:
            if limit is None:
                return None
            limit = float(limit)
            if limit < 0 or limit > 1:
                raise ValueError("Percentile limits must be between 0 and 1")
            if limit == 0:
                return 0.0
            if np.isfinite(limit):
                extra = min(2.0 / max(n_weights, 1), limit / 10.0)
                adjusted = min(limit + extra, 1.0)
                return adjusted
            return limit

        adjusted_limits = (
            _validate_limit(lower_limit),
            _validate_limit(upper_limit),
        )

        weights = scipy.stats.mstats.winsorize(
            weights, limits=adjusted_limits, inplace=False
        )
        if verbose:
            logger.debug(
                "Winsorizing weights to %s percentile" % str(weight_trimming_percentile)
            )

        weights = pd.Series(np.asarray(weights, dtype=np.float64), index=weights_index)

    if keep_sum_of_weights:
        weights = weights / np.mean(weights) * original_mean

    return weights


def default_transformations(
    dfs: Union[Tuple[pd.DataFrame, ...], List[pd.DataFrame]],
) -> Dict[str, Callable]:
    """
    Apply default transformations to dfs, i.e.
    quantize to numeric columns and fct_lump to non-numeric and boolean

    Args:
        dfs (Union[Tuple[pd.DataFrame, ...], List[pd.DataFrame]]): A list or tuple of dataframes

    Returns:
        Dict[str, Callable]: Dict of transformations
    """
    dtypes = {}
    for d in dfs:
        dtypes.update(d.dtypes.to_dict())

    transformations = {}
    for k, v in dtypes.items():
        # Notice that in pandas: pd.api.types.is_numeric_dtype(pd.Series([True, False])) == True
        # Hence, we need to explicitly check that not is_bool_dtype(v)
        # see: https://github.com/pandas-dev/pandas/issues/38378
        if (is_numeric_dtype(v)) and (not is_bool_dtype(v)):
            transformations[k] = balance_util.quantize
        else:
            transformations[k] = balance_util.fct_lump
    return transformations


def apply_transformations(
    dfs: Tuple[pd.DataFrame, ...],
    transformations: Union[Dict[str, Callable], str, None],
    drop: bool = True,
) -> Tuple[pd.DataFrame, ...]:
    """Apply the transformations specified in transformations to all of the dfs
        - if a column specified in `transformations` does not exist in the dataframes,
        it is added
        - if a column is not specified in `transformations`, it is dropped,
        unless drop==False
        - the dfs are concatenated together before transformations are applied,
        so functions like `max` are relative to the column in all dfs
        - Cannot transform the same variable twice, or add a variable and then transform it
        (i.e. the definition of the added variable should include the transformation)
        - if you get a cryptic error about mismatched data types, make sure your
        transformations are not being treated as additions because of missing
        columns (use `_set_warnings("DEBUG")` to check)

    Args:
        dfs (Tuple[pd.DataFrame, ...]): The DataFrames on which to operate
        transformations (Union[Dict[str, Callable], str, None]): Mapping from column name to function to apply.
            Transformations of existing columns should be specified as functions
            of those columns (e.g. `lambda x: x*2`), whereas additions of new
            columns should be specified as functions of the DataFrame
            (e.g. `lambda x: x.column_a + x.column_b`).
        drop (bool, optional): Whether to drop columns which are
              not specified in `transformations`. Defaults to True.

    Raises:
        NotImplementedError: When passing an unknown "transformations" argument.

    Returns:
        Tuple[pd.DataFrame, ...]: tuple of pd.DataFrames

    Examples:
        ::

            from balance.adjustment import apply_transformations
            import pandas as pd
            import numpy as np

            apply_transformations(
                (pd.DataFrame({'d': [1, 2, 3], 'e': [4, 5, 6]}),),
                {'d': lambda x: x*2, 'f': lambda x: x.d+x.e}
            )

                # (   f  d
                #  0  5  2
                #  1  7  4
                #  2  9  6,)

    """
    # TODO: change assert to raise
    assert isinstance(dfs, tuple), "'dfs' argument must be a tuple of DataFrames"
    assert all(
        isinstance(x, pd.DataFrame) for x in dfs
    ), "'dfs' must contain DataFrames"

    if transformations is None:
        return dfs
    elif isinstance(transformations, str):
        if transformations == "default":
            transformations = default_transformations(dfs)
        else:
            raise NotImplementedError(f"Unknown transformations {transformations}")

    ns = [0] + list(np.cumsum([x.shape[0] for x in dfs]))
    boundaries = [(ns[i], ns[i + 1]) for i in range(0, len(ns) - 1)]
    indices = [x.index for x in dfs]

    all_data = pd.concat(dfs).reset_index(drop=True)
    # This is to avoid issues with trnasformations that cannot
    # be done on object with duplicate indecies

    # additions is new columns to add to data. i.e.: column names that appear in transformations
    #   but are not present in all_data.
    additions = {k: v for k, v in transformations.items() if k not in all_data.columns}
    transformations = {
        k: v for k, v in transformations.items() if k in all_data.columns
    }
    logger.info(f"Adding the variables: {list(additions.keys())}")
    logger.info(f"Transforming the variables: {list(transformations.keys())}")
    logger.debug(
        f"Total number of added or transformed variables: {len(additions) + len(transformations)}"
    )

    # TODO: change assert to raise
    assert (
        len(additions) + len(transformations)
    ) > 0, "No transformations or additions passed"

    if len(additions) > 0:
        added = all_data.assign(**additions).loc[:, list(additions.keys())]
    else:
        added = None

    if len(transformations) > 0:
        # NOTE: .copy(deep=False) is used to avoid a false alarm that sometimes happen.
        # When we take a slice of the DataFrame (all_data[k]), it is passed to a function
        # inside v from transformations (e.g.: fct_lump), it would then sometimes raise:
        # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
        # Adding .copy(deep=False) solves this.
        # See: https://stackoverflow.com/a/54914752
        transformed = pd.DataFrame(
            {k: v(all_data.copy(deep=False)[k]) for k, v in transformations.items()}
        )
    else:
        transformed = None

    out = pd.concat((added, transformed), axis=1)

    dropped_columns = list(set(all_data.columns.values) - set(out.columns.values))

    if len(dropped_columns) > 0:
        if drop:
            logger.warning(f"Dropping the variables: {dropped_columns}")
        else:
            out = pd.concat((out, all_data.loc[:, dropped_columns]), axis=1)
    logger.info(f"Final variables in output: {list(out.columns)}")

    for column in out:
        # Create frequency table without using value_counts to avoid FutureWarning
        freq_table = out[column].groupby(out[column], dropna=False).size()
        logger.debug(f"Frequency table of column {column}:\n{freq_table}")
        logger.debug(
            f"Number of levels of column {column}:\n{out[column].nunique(dropna=False)}"
        )

    res = tuple(out[i:j] for (i, j) in boundaries)
    res = tuple(x.set_index(i) for x, i in zip(res, indices))

    return res


def _find_adjustment_method(
    method: Literal["cbps", "ipw", "null", "poststratify", "rake"],
    WEIGHTING_METHODS: Dict[str, Callable] = BALANCE_WEIGHTING_METHODS,
) -> Callable:
    """This function translates a string method argument to the function itself.

    Args:
        method (Literal["cbps", "ipw", "null", "poststratify", "rake"]): method for adjustment: cbps, ipw, null, poststratify
        WEIGHTING_METHODS (Dict[str, Callable]): A dict where keys are strings of function names, and the values are
            the functions themselves.

    Returns:
        Callable: The function for adjustment
    """
    if method in WEIGHTING_METHODS.keys():
        adjustment_function = WEIGHTING_METHODS[method]
    else:
        raise ValueError(f"Unknown adjustment method: '{method}'")

    return adjustment_function
