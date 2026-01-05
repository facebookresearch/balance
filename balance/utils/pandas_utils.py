# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Dict, NamedTuple

import numpy as np
import pandas as pd
import pandas.api.types as pd_types

logger: logging.Logger = logging.getLogger(__package__)

# TODO: Allow configuring this threshold globally if we need different sensitivity
HIGH_CARDINALITY_RATIO_THRESHOLD: float = 0.8


class HighCardinalityFeature(NamedTuple):
    column: str
    unique_count: int
    unique_ratio: float
    has_missing: bool


def _compute_cardinality_metrics(series: pd.Series) -> HighCardinalityFeature:
    """Compute cardinality metrics for a feature series.

    The function counts unique non-missing values and their proportion relative
    to non-missing rows, while also tracking whether any missing values are
    present.

    Args:
        series: Feature column to evaluate.

    Returns:
        HighCardinalityFeature: Metrics describing uniqueness and missingness.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series(["a", "b", "c", None, "c"])
        >>> _compute_cardinality_metrics(s)
        HighCardinalityFeature(column='', unique_count=3, unique_ratio=0.75, has_missing=True)
    """
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique()) if not non_missing.empty else 0
    unique_ratio = (
        float(unique_count) / float(len(non_missing)) if len(non_missing) > 0 else 0.0
    )
    return HighCardinalityFeature(
        column="",
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        has_missing=series.isna().any(),
    )


def _detect_high_cardinality_features(
    df: pd.DataFrame,
    threshold: float = HIGH_CARDINALITY_RATIO_THRESHOLD,
) -> list[HighCardinalityFeature]:
    """Identify categorical columns whose non-missing values are mostly unique.

    A feature is flagged when the ratio of unique non-missing values to total
    non-missing rows meets or exceeds ``threshold``. Only categorical columns
    (object, category, string dtypes) are checked, as high cardinality in
    numeric columns is expected and not problematic. Results are sorted by
    descending unique counts for clearer reporting.

    Args:
        df: Dataframe containing candidate features.
        threshold: Minimum unique-to-count ratio to flag a column.

    Returns:
        list[HighCardinalityFeature]: High-cardinality categorical columns
            sorted by descending uniqueness.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"id": ["a", "b", "c"], "group": ["a", "a", "b"]})
        >>> _detect_high_cardinality_features(df, threshold=0.8)
        [HighCardinalityFeature(column='id', unique_count=3, unique_ratio=1.0, has_missing=False)]
    """
    high_cardinality_features: list[HighCardinalityFeature] = []

    for column in df.columns:
        # Only check categorical columns (object, category, string dtypes)
        if not _is_categorical_dtype(df[column]):
            continue

        metrics = _compute_cardinality_metrics(df[column])

        if metrics.unique_count == 0:
            continue

        if metrics.unique_ratio < threshold:
            continue

        high_cardinality_features.append(
            HighCardinalityFeature(
                column=column,
                unique_count=metrics.unique_count,
                unique_ratio=metrics.unique_ratio,
                has_missing=metrics.has_missing,
            )
        )

    high_cardinality_features.sort(
        key=lambda feature: feature.unique_count, reverse=True
    )
    return high_cardinality_features


def _coerce_scalar(value: Any) -> float:
    """Safely convert a scalar value to ``float`` for diagnostics.

    ``None`` and non-scalar inputs are converted to ``NaN``. Scalar inputs are
    coerced to ``float`` when possible; otherwise, ``NaN`` is returned instead
    of raising a ``TypeError`` or ``ValueError``. Arrays and sequences return
    ``NaN`` so callers do not need to special-case these inputs.

    Args:
        value: Candidate value to coerce.

    Returns:
        float: ``float`` representation of ``value`` when possible, otherwise
        ``NaN``.

    Example:
        >>> _coerce_scalar(3)
        3.0
        >>> _coerce_scalar("7.125")
        7.125
        >>> _coerce_scalar(True)
        1.0
        >>> _coerce_scalar(complex(1, 2))
        nan
        >>> _coerce_scalar(())
        nan
        >>> _coerce_scalar([1, 2, 3])
        nan
    """

    if value is None:
        return float("nan")

    if np.isscalar(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return float("nan")


def _is_categorical_dtype(series: pd.Series) -> bool:
    """Check if a pandas Series has a categorical dtype.

    A dtype is considered categorical if it is object, category, or string type.

    Args:
        series: A pandas Series to check the dtype of.

    Returns:
        bool: True if the Series dtype is categorical (object, category, or string),
            False otherwise.

    Example:
        >>> import pandas as pd
        >>> _is_categorical_dtype(pd.Series(["a", "b"]))
        True
        >>> _is_categorical_dtype(pd.Series([1, 2]))
        False
    """
    dtype = series.dtype
    return (
        pd_types.is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd_types.is_string_dtype(dtype)
    )


def _process_series_for_missing_mask(series: pd.Series) -> pd.Series:
    """
    Helper function to process a pandas Series for missing value detection
    while avoiding deprecation warnings from replace and infer_objects.

    Args:
        series (pd.Series): Input series to process

    Returns:
        pd.Series: Boolean series indicating missing values
    """
    # Use _safe_replace_and_infer to avoid downcasting warnings
    replaced_series = _safe_replace_and_infer(series, [np.inf, -np.inf], np.nan)
    return replaced_series.isna()


def _safe_replace_and_infer(
    data: pd.Series | pd.DataFrame,
    to_replace: Any | None = None,
    value: Any | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Helper function to safely replace values and infer object dtypes
    while avoiding pandas deprecation warnings.
    Args:
        data: pandas Series or DataFrame to process
        to_replace: Value(s) to replace (default: [np.inf, -np.inf])
        value: Value to replace with (default: np.nan)
    Returns:
        Processed Series or DataFrame with proper dtype inference
    """
    if to_replace is None:
        to_replace = [np.inf, -np.inf]
    if value is None:
        value = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
        )
        return data.replace(to_replace, value).infer_objects(copy=False)


def _safe_fillna_and_infer(
    data: pd.Series | pd.DataFrame, value: Any | None = None
) -> pd.Series | pd.DataFrame:
    """
    Helper function to safely fill NaN values and infer object dtypes
    while avoiding pandas deprecation warnings.

    Args:
        data: pandas Series or DataFrame to process
        value: Value to fill NaN with (default: np.nan)

    Returns:
        Processed Series or DataFrame with proper dtype inference
    """
    if value is None:
        value = np.nan

    # Suppress pandas FutureWarnings about downcasting during fillna operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        filled_data = data.fillna(value)

    return filled_data.infer_objects(copy=False)


def _safe_groupby_apply(
    data: pd.DataFrame,
    groupby_cols: str | list[str],
    apply_func: Any,
) -> pd.Series:
    """
    Helper function to safely apply groupby operations while handling
    the include_groups parameter for pandas compatibility.

    Args:
        data: DataFrame to group
        groupby_cols: Column(s) to group by
        apply_func: Function to apply to each group

    Returns:
        Result of groupby apply operation
    """
    # Use include_groups=False to avoid FutureWarning about operating on grouping columns
    # Fall back to old behavior if include_groups parameter is not supported
    try:
        return data.groupby(groupby_cols, include_groups=False).apply(apply_func)
    except TypeError:
        # Suppress pandas FutureWarnings about downcasting during fillna operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Fallback for older pandas versions that don't support include_groups parameter
            return data.groupby(groupby_cols).apply(apply_func)


def _safe_show_legend(axis: Any) -> None:
    """
    Helper function to safely show legend only if there are labeled artists,
    avoiding matplotlib UserWarning about no artists with labels.

    Args:
        axis: matplotlib axis object
    """
    _, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend()


def _safe_divide_with_zero_handling(numerator: Any, denominator: Any) -> Any:
    """
    Helper function to safely perform division while handling divide by zero
    warnings with proper numpy error state management.

    Args:
        numerator: Numerator for division
        denominator: Denominator for division

    Returns:
        Result of division with proper handling of divide by zero cases
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Use numpy.divide to handle division by zero properly
        result = np.divide(numerator, denominator)
        return result


def _dict_intersect(d: Dict[Any, Any], d_for_keys: Dict[Any, Any]) -> Dict[Any, Any]:
    """Returns dict 1, but only with the keys that are also in d2

    Args:
        d1 (Dict): First dictionary.
        d2 (Dict): Second dictionary.

    Returns:
        Dict: Intersection of d1 and d2 (with values from d1)

    Examples:
        ::
            d1 = {"a": 1, "b": 2}
            d2 = {"c": 3, "b": 2}
            _dict_intersect(d1, d2)
            # {'b': 2}
    """
    intersect_keys = d.keys() & d_for_keys.keys()
    return {k: d[k] for k in intersect_keys}


# TODO: using _astype_in_df_from_dtypes to turn sample.df to original df dtypes may not be a good long term solution.
#       A better solution might require a redesign of some core features.
def _astype_in_df_from_dtypes(
    df: pd.DataFrame, target_dtypes: pd.Series
) -> pd.DataFrame:
    """Returns df with dtypes cast as specified in df_orig.
       Columns that were not in the original dataframe are kept the same.

    Args:
        df (pd.DataFrame): df to convert
        target_dtypes (pd.Series): DataFrame.dtypes to use as target dtypes for conversion

    Returns:
        pd.DataFrame: df with dtypes cast as specified in target_dtypes

    Examples:
        ::
            df = pd.DataFrame({"id": ("1", "2"), "a": (1.0, 2.0), "weight": (1.0,2.0)})
            df_orig = pd.DataFrame({"id": (1, 2), "a": (1, 2), "forest": ("tree", "banana")})

            df.dtypes.to_dict()
                # {'id': dtype('O'), 'a': dtype('float64'), 'weight': dtype('float64')}
            df_orig.dtypes.to_dict()
                # {'id': dtype('int64'), 'a': dtype('int64'), 'forest': dtype('O')}

            target_dtypes = df_orig.dtypes
            _astype_in_df_from_dtypes(df, target_dtypes).dtypes.to_dict()
                # {'id': dtype('int64'), 'a': dtype('int64'), 'weight': dtype('float64')}
    """
    dict_of_target_dtypes = _dict_intersect(
        target_dtypes.to_dict(),
        df.dtypes.to_dict(),
    )
    return df.astype(dict_of_target_dtypes)


def _are_dtypes_equal(
    dt1: pd.Series, dt2: pd.Series
) -> Dict[str, bool | pd.Series | set[Any]]:
    """Returns True if both dtypes are the same and False otherwise.

    If dtypes have an unequal set of items, the comparison will only be about the same set of keys.
    If there are no shared keys, then return False.

    Args:
        dt1 (pd.Series): first dtype (output from DataFrame.dtypes)
        dt2 (pd.Series): second dtype (output from DataFrame.dtypes)

    Returns:
        Dict[str, Union[bool, pd.Series, set]]: a dict of the following structure
            {
                'is_equal': False,
                'comparison_of_dtypes':
                                    flt    True
                                    int    False
                                    dtype: bool,
                'shared_keys': {'flt', 'int'}
            }

    Examples:
        ::
            df1 = pd.DataFrame({'int':np.arange(5), 'flt':np.random.randn(5)})
            df2 = pd.DataFrame({'flt':np.random.randn(5), 'int':np.random.randn(5)})
            df11 = pd.DataFrame({'int':np.arange(5), 'flt':np.random.randn(5), 'miao':np.random.randn(5)})

            _are_dtypes_equal(df1.dtypes, df1.dtypes)['is_equal']  # True
            _are_dtypes_equal(df1.dtypes, df2.dtypes)['is_equal']  # False
            _are_dtypes_equal(df11.dtypes, df2.dtypes)['is_equal'] # False
    """
    shared_keys = set.intersection(set(dt1.keys()), set(dt2.keys()))
    shared_keys_list = list(shared_keys)
    comparison_of_dtypes = dt1[shared_keys_list] == dt2[shared_keys_list]
    is_equal = np.all(comparison_of_dtypes)
    return {
        "is_equal": is_equal,
        "comparison_of_dtypes": comparison_of_dtypes,
        "shared_keys": shared_keys,
    }


def _warn_of_df_dtypes_change(
    original_df_dtypes: pd.Series,
    new_df_dtypes: pd.Series,
    original_str: str = "df",
    new_str: str = "new_df",
) -> None:
    """Prints a warning if the dtypes of some original df and some modified df differs.

    Args:
        original_df_dtypes (pd.Series): dtypes of original dataframe
        new_df_dtypes (pd.Series): dtypes of modified dataframe
        original_str (str, optional): string to use for warnings when referring to the original. Defaults to "df".
        new_str (str, optional): string to use for warnings when referring to the modified df. Defaults to "new_df".

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from copy import deepcopy
            import balance

            df = pd.DataFrame({"int": np.arange(5), "flt": np.random.randn(5)})
            new_df = deepcopy(df)
            new_df.int = new_df.int.astype(float)
            new_df.flt = new_df.flt.astype(int)

            balance.util._warn_of_df_dtypes_change(df.dtypes, new_df.dtypes)

                # WARNING (2023-02-07 08:01:19,961) [util/_warn_of_df_dtypes_change (line 1696)]: The dtypes of new_df were changed from the original dtypes of the input df, here are the differences -
                # WARNING (2023-02-07 08:01:19,963) [util/_warn_of_df_dtypes_change (line 1707)]: The (old) dtypes that changed for df (before the change):
                # WARNING (2023-02-07 08:01:19,966) [util/_warn_of_df_dtypes_change (line 1710)]:
                # flt    float64
                # int      int64
                # dtype: object
                # WARNING (2023-02-07 08:01:19,971) [util/_warn_of_df_dtypes_change (line 1711)]: The (new) dtypes saved in df (after the change):
                # WARNING (2023-02-07 08:01:19,975) [util/_warn_of_df_dtypes_change (line 1712)]:
                # flt      int64
                # int    float64
                # dtype: object
    """
    compare_df_dtypes_before_and_after = _are_dtypes_equal(
        original_df_dtypes, new_df_dtypes
    )
    if not compare_df_dtypes_before_and_after["is_equal"]:
        logger.warning(
            f"The dtypes of {new_str} were changed from the original dtypes of the input {original_str}, here are the differences - "
        )
        compared_dtypes = compare_df_dtypes_before_and_after["comparison_of_dtypes"]
        dtypes_that_changed = (
            # pyre-ignore[16]: we're only using the pd.Series, so no worries
            compared_dtypes[np.bitwise_not(compared_dtypes.values)].keys().to_list()
        )
        logger.debug(compare_df_dtypes_before_and_after)
        logger.warning(
            f"The (old) dtypes that changed for {original_str} (before the change):"
        )
        logger.warning("\n" + str(original_df_dtypes[dtypes_that_changed]))
        logger.warning(f"The (new) dtypes saved in {original_str} (after the change):")
        logger.warning("\n" + str(new_df_dtypes[dtypes_that_changed]))


def _make_df_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame column names unique by adding suffixes to duplicates.

    This function iterates through the column names of the input DataFrame
    and appends a suffix to duplicate column names to make them distinct.
    The suffix is an underscore followed by an integer value representing
    the number of occurrences of the column name.

    Args:
        df (pd.DataFrame): The input DataFrame with potentially duplicate
            column names.

    Returns:
        pd.DataFrame: A DataFrame with unique column names where any
            duplicate column names have been renamed with a suffix.

    Examples:
        ::
            import pandas as pd

            # Sample DataFrame with duplicate column names
            data = {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "A2": [7, 8, 9],
                "C": [10, 11, 12],
            }

            df1 = pd.DataFrame(data)
            df1.columns = ["A", "B", "A", "C"]

            _make_df_column_names_unique(df1).to_dict()
            # {'A': {0: 1, 1: 2, 2: 3},
            #  'B': {0: 4, 1: 5, 2: 6},
            #  'A_1': {0: 7, 1: 8, 2: 9},
            #  'C': {0: 10, 1: 11, 2: 12}}
    """
    # Check if all column names are unique
    unique_columns = set(df.columns)
    if len(unique_columns) == len(df.columns):
        return df

    # Else: fix duplicate column names
    logger.warning(
        """Duplicate column names exists in the DataFrame.
                    A suffix will be added to them but their order might change from one iteration to another.
                    To avoid issues, make sure to change your original column names to be unique (and without special characters)."""
    )
    col_counts = {}
    new_columns = []

    for col in df.columns:
        if col in col_counts:
            col_counts[col] += 1
            new_col_name = f"{col}_{col_counts[col]}"
            logger.warning(
                f"Column {col} already exists in the DataFrame, renaming it to be {new_col_name}"
            )
        else:
            col_counts[col] = 0
            new_col_name = col
        new_columns.append(new_col_name)

    df.columns = new_columns

    return df


def _pd_convert_all_types(
    df: pd.DataFrame, input_type: str, output_type: str
) -> pd.DataFrame:
    """Converts columns in the input dataframe to a specified type.

    Args:
        df (pd.DataFrame): Input df
        input_type (str): A string of the input type to change.
        output_type (str): A string of the desired output type for the columns of type input_type.

    Returns:
        pd.DataFrame: Output df with columns converted from input_type to output_type.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            df = pd.DataFrame({"a": pd.array([1,2], dtype = pd.Int64Dtype()), "a2": pd.array([1,2], dtype = np.int64)})

            df.dtypes
                # a     Int64
                # a2    int64
                # dtype: object
            df.dtypes.to_numpy()
                # array([Int64Dtype(), dtype('int64')], dtype=object)

            df2 =_pd_convert_all_types(df, "Int64", "int64")

            df2.dtypes.to_numpy()
                # array([dtype('int64'), dtype('int64')], dtype=object)

            # Might be requires some casting to float64 so that it will handle missing values
            # For details, see: https://stackoverflow.com/a/53853351
            df3 =_pd_convert_all_types(df, "Int64", "float64")
            df3.dtypes.to_numpy()
                # array([dtype('float64'), dtype('float64')], dtype=object)
    """
    df = copy.deepcopy(df)
    # source: https://stackoverflow.com/questions/39904889/
    df = pd.concat(
        [
            df.select_dtypes([], [input_type]),
            df.select_dtypes([input_type]).apply(pd.Series.astype, dtype=output_type),
        ],
        axis=1,
    ).reindex(df.columns, axis=1)
    return df
