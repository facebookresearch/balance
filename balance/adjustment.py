# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from balance import util as balance_util
from balance.util import _verify_value_type
from balance.weighting_methods import (
    adjust_null as balance_adjust_null,
    cbps as balance_cbps,
    ipw as balance_ipw,
    poststratify as balance_poststratify,
    rake as balance_rake,
)
from pandas.api.types import is_bool_dtype, is_numeric_dtype

logger: logging.Logger = logging.getLogger(__package__)


BALANCE_WEIGHTING_METHODS: Dict[str, Callable[..., Any]] = {
    "ipw": balance_ipw.ipw,
    "cbps": balance_cbps.cbps,
    "null": balance_adjust_null.adjust_null,
    "poststratify": balance_poststratify.poststratify,
    "rake": balance_rake.rake,
}


def _validate_limit(limit: float | int | None, n_weights: int) -> float | None:
    """Validate and adjust a percentile limit for use with scipy.stats.mstats.winsorize.

    This function prepares percentile limits for winsorization by:
    1. Validating that finite limits are within valid bounds (0-1)
    2. Adding a small adjustment to finite, non-zero limits to ensure at least
       one value gets winsorized at the boundary percentile

    The adjustment prevents edge cases where the exact percentile value might not
    trigger winsorization due to floating-point precision or discrete data distributions.
    The adjustment is the minimum of (2/n_weights, limit/10), capped at 1.0.

    Special cases:
    - None: Returns None unchanged (no winsorization on this side)
    - 0: Returns 0.0 (no winsorization on this side)
    - Non-finite (inf): Returns as-is without validation or adjustment

    Args:
        limit (Union[float, int, None]): The percentile limit to validate.
            For finite values, should be between 0 and 1.
        n_weights (int): The number of weights in the dataset. Used to calculate
            an adjustment factor that scales inversely with sample size.

    Returns:
        Union[float, None]: The validated and adjusted limit, or None if the
            input limit was None.

    Raises:
        ValueError: If the limit is finite and not between 0 and 1.
    """
    if limit is None:
        return None
    limit = float(limit)
    if limit == 0:
        return 0.0
    # Check for non-finite values before validating range
    if not np.isfinite(limit):
        return limit
    # Validate range only for finite values
    if limit < 0 or limit > 1:
        raise ValueError("Percentile limits must be between 0 and 1")
    # Apply adjustment for finite values within valid range
    extra = min(2.0 / max(n_weights, 1), limit / 10.0)
    adjusted = min(limit + extra, 1.0)
    return adjusted


def trim_weights(
    weights: pd.Series | npt.NDArray,
    # TODO: add support to more types of input weights? (e.g. list? other?)
    weight_trimming_mean_ratio: float | int | None = None,
    weight_trimming_percentile: float | Tuple[float, float] | None = None,
    verbose: bool = False,
    keep_sum_of_weights: bool = True,
    target_sum_weights: float | int | np.floating | None = None,
) -> pd.Series:
    """Trim extreme weights using mean ratio clipping or percentile-based winsorization.

    The user cannot supply both weight_trimming_mean_ratio and weight_trimming_percentile.
    If neither is supplied, the original weights are returned unchanged.

    **Mean Ratio Trimming (weight_trimming_mean_ratio)**:
    When specified, weights are clipped from above at mean(weights) * ratio, then
    renormalized to preserve the original mean. This is a hard upper bound.
    Note: Final weights may slightly exceed the trimming ratio due to renormalization
    redistributing the clipped weight mass across all observations.

    **Percentile-Based Winsorization (weight_trimming_percentile)**:
    When specified, extreme weights are replaced with less extreme values using
    scipy.stats.mstats.winsorize. By default, winsorization affects both tails
    of the distribution symmetrically, unlike mean ratio trimming which only
    clips from above.

    Behavior:
    - Single value (e.g., 0.1): Winsorizes below 10th AND above 90th percentile
    - Tuple (lower, upper): Winsorizes independently on each side
      - (0.1, 0): Only winsorizes below 10th percentile
      - (0, 0.1): Only winsorizes above 90th percentile
      - (0.01, 0.05): Winsorizes below 1st AND above 95th percentile

    Important implementation detail: Percentile limits are automatically adjusted
    upward slightly (via _validate_limit) to ensure at least one value gets
    winsorized at boundary percentiles. This prevents edge cases where discrete
    distributions or floating-point precision might prevent winsorization at the
    exact percentile value. The adjustment is min(2/n_weights, limit/10), capped at 1.0.

    After trimming/winsorization, if keep_sum_of_weights=True (default), weights
    are rescaled to preserve the original sum of weights.  Alternatively, pass a
    ``target_sum_weights`` to rescale the trimmed weights so their sum matches a
    desired total.

    Args:
        weights (pd.Series | np.ndarray): Weights to trim. np.ndarray will be
            converted to pd.Series internally.
        weight_trimming_mean_ratio (float | int | None, optional): Ratio for upper bound
            clipping as mean(weights) * ratio. Mutually exclusive with
            weight_trimming_percentile. Defaults to None.
        weight_trimming_percentile (float | tuple[float, float] | None, optional):
            Percentile limits for winsorization. Value(s) must be between 0 and 1.
            - Single float: Symmetric winsorization on both tails
            - tuple[float, float]: (lower_percentile, upper_percentile) for
              independent control of each tail
            Mutually exclusive with weight_trimming_mean_ratio. Defaults to None.
        verbose (bool, optional): Whether to log details about the trimming process.
            Defaults to False.
        keep_sum_of_weights (bool, optional): Whether to rescale weights after trimming
            to preserve the original sum of weights. Defaults to True.
        target_sum_weights (float | int | np.floating | None, optional): If
            provided, rescale the trimmed weights so their sum equals this
            target. ``None`` (default) leaves the post-trimming sum unchanged.

    Raises:
        TypeError: If weights is not np.array or pd.Series.
        ValueError: If both weight_trimming_mean_ratio and weight_trimming_percentile
            are specified, or if weight_trimming_percentile tuple has length != 2.

    Returns:
        pd.Series (of type float64): Trimmed weights with the same index as input

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

    original_name = getattr(weights, "name", None)

    if isinstance(weights, pd.Series):
        weights = weights.astype(np.float64, copy=False)
    elif isinstance(weights, np.ndarray):
        weights = pd.Series(weights, dtype=np.float64, name=original_name)
    else:
        raise TypeError(
            f"weights must be np.array or pd.Series, are of type: {type(weights)}"
        )
    weights_index = weights.index

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

        weights = pd.Series(
            np.asarray(weights, dtype=np.float64),
            index=weights_index,
            name=original_name,
        )
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

        # Keep the original requested percentiles for exact clipping bounds,
        # but validate/adjust separately for the winsorization call so at least
        # one value is affected at the requested edge.
        clip_limits = (
            None if (lower_limit is None or lower_limit == 0) else lower_limit,
            None if (upper_limit is None or upper_limit == 0) else upper_limit,
        )
        adjusted_limits = (
            _validate_limit(lower_limit, n_weights),
            _validate_limit(upper_limit, n_weights),
        )

        # Preserve the pre-trim weights to calculate strict clipping bounds.
        original_weights_for_bounds = weights.copy()

        weights = scipy.stats.mstats.winsorize(
            weights, limits=adjusted_limits, inplace=False
        )
        if verbose:
            logger.debug(
                "Winsorizing weights to %s percentile" % str(weight_trimming_percentile)
            )

        weights = pd.Series(
            np.asarray(weights, dtype=np.float64),
            index=weights_index,
            name=original_name,
        )

        # Clip to the exact percentile bounds to avoid small numerical overshoots
        # from scipy.stats.mstats.winsorize on certain inputs.
        lower_bound = (
            None
            if clip_limits[0] is None
            else np.quantile(
                original_weights_for_bounds, clip_limits[0], method="lower"
            )
        )
        upper_bound = (
            None
            if clip_limits[1] is None
            else np.quantile(
                original_weights_for_bounds,
                1 - _verify_value_type(clip_limits[1]),
                method="lower",
            )
        )
        weights = weights.clip(lower=lower_bound, upper=upper_bound)

    if keep_sum_of_weights:
        weights = weights / np.mean(weights) * original_mean

    if target_sum_weights is not None:
        target_total = float(target_sum_weights)
        current_total = float(weights.sum())
        if np.isclose(current_total, 0.0):
            raise ValueError("Cannot normalise weights because their sum is zero.")
        weights = weights * (target_total / current_total)

    weights = weights.rename(original_name)

    return weights


def default_transformations(
    dfs: tuple[pd.DataFrame, ...] | list[pd.DataFrame],
) -> Dict[str, Callable[..., Any]]:
    """
    Apply default transformations to dfs, i.e.
    quantize to numeric columns and fct_lump to non-numeric and boolean

    Args:
        dfs (tuple[pd.DataFrame, ...] | list[pd.DataFrame]): A list or tuple of dataframes

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
    transformations: Dict[str, Callable[..., Any]] | str | None,
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
        transformations (Dict[str, Callable[..., Any]] | str | None): Mapping from column name to function to apply.
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
    WEIGHTING_METHODS: Dict[str, Callable[..., Any]] = BALANCE_WEIGHTING_METHODS,
) -> Callable[..., Any]:
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
