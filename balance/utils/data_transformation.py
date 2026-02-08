# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import logging
import warnings
from itertools import combinations
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from balance.utils.input_validation import choose_variables
from balance.utils.pandas_utils import _safe_fillna_and_infer

logger: logging.Logger = logging.getLogger(__package__)


def add_na_indicator(
    df: pd.DataFrame, replace_val_obj: str = "_NA", replace_val_num: int = 0
) -> pd.DataFrame:
    """If a column in the DataFrame contains NAs, replace these with 0 for
    numerical columns or "_NA" for non-numerical columns,
    and add another column of an indicator variable for which rows were NA.

    Args:
        df (pd.DataFrame): The input DataFrame
        replace_val_obj (str, optional): The value to put instead of nulls for object columns. Defaults to "_NA".
        replace_val_num (int, optional): The value to put instead of nulls for numeric columns. Defaults to 0.

    Raises:
        Exception: Can't add NA indicator to DataFrame which contains columns which start with '_is_na_'
        Exception: Can't add NA indicator to columns containing NAs and the value '{replace_val_obj}',

    Returns:
        pd.DataFrame: New dataframe with additional columns
    """
    already_na_cols = [c for c in df.columns if c.startswith("_is_na_")]
    if len(already_na_cols) > 0:
        raise ValueError(
            "Can't add NA indicator to DataFrame which contains"
            f"columns which start with '_is_na_': {already_na_cols}"
        )

    na = df.isnull()
    na_cols = list(df.columns[na.any(axis="index")])
    na_indicators = na.loc[:, na_cols]
    na_indicators.columns = ("_is_na_" + c for c in na_indicators.columns)

    categorical_cols = list(df.columns[df.dtypes == "category"])
    non_numeric_cols = list(
        df.columns[(df.dtypes == "object") | (df.dtypes == "string")]
    )

    for c in list(na_cols):
        if replace_val_obj in set(df[c]):
            raise ValueError(
                f"Can't add NA indicator to columns containing NAs and the value '{replace_val_obj}', "
                f"i.e. column: {c}"
            )
        if c in categorical_cols:
            filled_col = (
                df[c].cat.add_categories(replace_val_obj).fillna(replace_val_obj)
            )
            df[c] = filled_col.infer_objects()
        elif c in non_numeric_cols:
            df[c] = _safe_fillna_and_infer(df[c], replace_val_obj)
        else:
            df[c] = _safe_fillna_and_infer(df[c], replace_val_num)

    return pd.concat((df, na_indicators), axis=1)


def add_na_indicator_to_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Add NA indicator columns to a DataFrame, handling pre-existing ``_is_na_*`` columns.

    :func:`add_na_indicator` raises when the input already contains columns whose
    names start with ``_is_na_``.  This wrapper splits those columns out first,
    applies :func:`add_na_indicator` to the remaining base columns, and then
    re-attaches the original indicator columns so that nothing is duplicated.

    Args:
        df (pd.DataFrame): The input DataFrame, which may or may not already
            contain ``_is_na_*`` columns.

    Returns:
        pd.DataFrame: The DataFrame with NA indicator columns added for every
            base column that contains missing values.
    """
    existing_indicator_cols = [
        col for col in df.columns if isinstance(col, str) and col.startswith("_is_na_")
    ]
    if not existing_indicator_cols:
        return add_na_indicator(df)

    base_cols = [col for col in df.columns if col not in existing_indicator_cols]
    combined_base = add_na_indicator(df[base_cols])
    # add_na_indicator will create "_is_na_<col>" for every base column that has
    # NAs.  If the input already carried a matching indicator (e.g. "_is_na_foo"
    # exists and "foo" still has NAs), the newly created column would clash with
    # the pre-existing one.  Drop the duplicates so the original indicators are
    # preserved unchanged when we re-attach them below.
    overlapping = [c for c in existing_indicator_cols if c in combined_base.columns]
    if overlapping:
        logger.debug(
            "add_na_indicator_to_combined: dropping %d newly created indicator "
            "column(s) that overlap with pre-existing ones: %s",
            len(overlapping),
            overlapping,
        )
    combined_base = combined_base.drop(columns=overlapping, errors="ignore")
    return pd.concat([combined_base, df[existing_indicator_cols]], axis=1)


def drop_na_rows(
    sample_df: pd.DataFrame, sample_weights: pd.Series, name: str = "sample object"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows with missing values in sample_df and their corresponding weights, and the same in target_df.

    Args:
        sample_df (pd.DataFrame): a dataframe representing the sample or target
        sample_weights (pd.Series): design weights for sample or target
        name (str, optional): name of object checked (used for warnings prints). Defaults to "sample object".

    Raises:
        ValueError: Dropping rows led to empty {name}. Maybe try na_action='add_indicator'?

    Returns:
        Tuple[pd.DataFrame, pd.Series]: sample_df, sample_weights without NAs rows
    """
    sample_n = sample_df.shape[0]
    sample_df = sample_df.dropna()
    sample_weights = sample_weights[sample_df.index]
    sample_n_after = sample_df.shape[0]

    _sample_rate = f"{sample_n - sample_n_after}/{sample_n}"
    logger.warning(f"Dropped {_sample_rate} rows of {name}")
    if sample_n_after == 0:
        raise ValueError(
            f"Dropping rows led to empty {name}. Maybe try na_action='add_indicator'?"
        )
    return (sample_df, sample_weights)


def qcut(
    s: np.ndarray | pd.Series,
    q: int | float,
    duplicates: str = "drop",
    **kwargs: Any,
) -> np.ndarray | pd.Series:
    """Discretize variable into equal-sized buckets based quantiles.
    This is a wrapper to pandas qcut function.

    Args:
        s (_type_): 1d ndarray or Series.
        q (_type_): Number of quantiles (int or float).
        duplicates (str, optional): whether to drop non unique bin edges or raise error ("raise" or "drop").
            Defaults to "drop".

    Returns:
        Series of type object with intervals.
    """
    if s.shape[0] < q:  # pyre-ignore[58]: Comparison is valid in practice
        logger.warning("Not quantizing, too few values")
        return s
    else:
        return pd.qcut(s, q, duplicates=duplicates, **kwargs).astype("O")


def quantize(
    df: pd.DataFrame | pd.Series,
    q: int = 10,
    variables: List[str] | None = None,
) -> pd.DataFrame | np.ndarray | pd.Series:
    """Cut numeric variables of a DataFrame into quantiles buckets

    Args:
        df (Union[pd.DataFrame, pd.Series]): a DataFrame to transform
        q (int, optional): Number of buckets to create for each variable. Defaults to 10.
        variables (optional): variables to transform.
                    If None, all numeric variables are transformed. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame after quantization. numpy.nan values are kept as is.
    """
    if not (isinstance(df, pd.Series) or isinstance(df, pd.DataFrame)):
        # Necessary because pandas calls the function on the first item on its own
        #  https://stackoverflow.com/questions/21635915/
        df = pd.Series(df)

    if isinstance(df, pd.Series):
        if not pd.api.types.is_numeric_dtype(df.dtype):
            raise TypeError("series must be numeric")
        return qcut(df, q, duplicates="drop")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    variables = choose_variables(df, variables=variables)
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)

    variables = [v for v in variables if v in numeric_columns]

    original_columns = list(df.columns)
    transformed_data = df.loc[:, variables].transform(
        lambda c: qcut(c, q, duplicates="drop")
    )
    untransformed_columns = df.columns.difference(variables)
    transformed_data = pd.concat(
        (df.loc[:, untransformed_columns], transformed_data), axis=1
    )
    return transformed_data.loc[:, original_columns]


def row_pairwise_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the differences between every pair of rows of df

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with differences between all combinations of rows
    """
    c = combinations(sorted(df.index), 2)
    diffs = []
    for j, i in c:
        d = df.loc[i] - df.loc[j]
        d = d.to_frame().transpose().assign(source=f"{i} - {j}").set_index("source")
        diffs.append(d)
    return pd.concat([df] + diffs)


def auto_spread(
    data: pd.DataFrame, features: List[str] | None = None, id_: str = "id"
) -> pd.DataFrame:
    """Automatically transform a 'long' DataFrame into a 'wide' DataFrame
    by guessing which column should be used as a key, treating all
    other columns as values. At the moment, this will only find a single key column

    Args:
        data (pd.DataFrame):
        features (Optional[list], optional): Defaults to None.
        id_ (str, optional): Defaults to "id".

    Returns:
        pd.DataFrame
    """
    if features is None:
        features = [c for c in data.columns.values if c != id_]

    is_unique = {}
    for c in features:
        # Use include_groups=False to avoid FutureWarning about operating on grouping columns
        # Fall back to old behavior if include_groups parameter is not supported
        try:
            unique_userids = data.groupby(c, include_groups=False)[id_].apply(
                lambda x: len(set(x)) == len(x)
            )
        except TypeError:
            # Fallback for older pandas versions that don't support include_groups parameter
            unique_userids = data.groupby(c)[id_].apply(lambda x: len(set(x)) == len(x))
        is_unique[c] = all(unique_userids.values)

    unique_groupings = [k for k, v in is_unique.items() if v]
    if len(unique_groupings) < 1:
        logger.warning(f"no unique groupings {is_unique}")
        return data
    elif len(unique_groupings) > 1:
        logger.warning(
            f"{len(unique_groupings)} possible groupings: {unique_groupings}"
        )

    # Always chooses the first unique grouping
    unique_grouping = unique_groupings[0]
    logger.warning(f"Grouping by {unique_grouping}")

    data = data.loc[:, features + [id_]].pivot(index=id_, columns=unique_grouping)
    data.columns = [
        "_".join(map(str, ((unique_grouping,) + c[-1:] + c[:-1])))
        for c in data.columns.values
    ]
    data = data.reset_index()
    return data


def auto_aggregate(
    data: pd.DataFrame,
    features: None = None,
    _id: str = "id",
    # NOTE: we use str as default since using a lambda function directly would make this argument mutable -
    # so if one function call would change it, another function call would get the revised aggfunc argument.
    # Thus, using str is important so to keep our function idempotent.
    aggfunc: str | Any = "sum",
) -> pd.DataFrame:
    # The default aggregation function is a lambda around sum(x), because as of
    # Pandas 0.22.0, Series.sum of an all-na Series is 0, not nan

    if features is not None:
        warnings.warn(
            "features argument is unused, it will be removed in the future",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(aggfunc, str):
        if aggfunc == "sum":

            def _f(x: Any) -> int:
                return sum(x)

            aggfunc = _f
        else:
            raise ValueError(
                f"unknown aggregate function name {aggfunc}, accepted values are ('sum',)."
            )

    try:
        data_without_id = data.drop(columns=[_id])
    except KeyError:
        raise ValueError(f"data must have a column named {_id}")

    all_columns = data_without_id.columns.to_list()

    numeric_columns = data_without_id.select_dtypes(
        include=[np.number]
    ).columns.to_list()

    if set(all_columns) != set(numeric_columns):
        raise ValueError(
            "Not all covariates are numeric. The function will not aggregate automatically."
        )

    return pd.pivot_table(data, index=_id, aggfunc=aggfunc).reset_index()


def fct_lump(s: pd.Series, prop: float = 0.05) -> pd.Series:
    """Lumps infrequent levels into '_lumped_other'.
    Note that all values with proportion less than prop output the same value '_lumped_other'.

    Args:
        s (pd.Series): pd.series to lump, with dtype of integer, numeric, object, or category (category will be converted to object)
        prop (float, optional): the proportion of infrequent levels to lump. Defaults to 0.05.

    Returns:
        pd.Series: pd.series (with category dtype converted to object, if applicable)
    """
    # Handle value_counts with object-dtype to maintain consistent behavior
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of value_counts with object-dtype is deprecated.*",
            category=FutureWarning,
        )
        props = s.value_counts() / s.shape[0]

    # Ensure proper dtype inference on the index
    props.index = props.index.infer_objects()

    small_categories = props[props < prop].index.tolist()

    remainder_category_name = "_lumped_other"
    while remainder_category_name in props.index:
        remainder_category_name = remainder_category_name * 2

    # Convert to object dtype unless already string dtype
    if not pd.api.types.is_string_dtype(s.dtype):
        s = s.astype("object")

    # Replace small categories with the remainder category name
    mask = s.isin(small_categories).fillna(False)
    s.loc[mask] = remainder_category_name
    return s


def fct_lump_by(s: pd.Series, by: pd.Series, prop: float = 0.05) -> pd.Series:
    """Lumps infrequent levels into '_lumped_other, only does so per
    value of the grouping variable `by`. Useful, for example, for keeping the
    most important interactions in a model.

    Args:
        s (pd.Series): pd.series to lump
        by (pd.Series): pd.series according to which group the data
        prop (float, optional): the proportion of infrequent levels to lump. Defaults to 0.05.

    Returns:
        pd.Series: pd.series, we keep the index of s as the index of the result.
    """
    res = copy.deepcopy(s)
    # pandas groupby doesnt preserve order
    for subgroup in pd.unique(by):
        mask = by == subgroup
        grouped_res = fct_lump(res.loc[mask], prop=prop)
        # Ensure dtype compatibility before assignment
        if not pd.api.types.is_string_dtype(res.dtype):
            res = res.astype("object")
        res.loc[mask] = grouped_res
    return res
