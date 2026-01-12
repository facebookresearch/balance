# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import collections
import logging
from functools import reduce
from typing import Any, Callable, List, Optional, overload, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from balance.utils.pandas_utils import _process_series_for_missing_mask

logger: logging.Logger = logging.getLogger(__package__)

T = TypeVar("T")


@overload
def _verify_value_type(  # noqa: E704
    optional: Optional[Any],
    expected_type: Type[T],
) -> T: ...


@overload
def _verify_value_type(  # noqa: E704
    optional: Optional[T],
    expected_type: None = None,
) -> T: ...


def _verify_value_type(
    optional: Optional[T],
    expected_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
) -> T:
    """Assert that optional value is not None and return it.

    Args:
        optional: The optional value to check
        expected_type: Optional type or tuple of types to check with isinstance()

    Returns:
        The non-None value

    Raises:
        ValueError: If optional is None
        TypeError: If expected_type is provided and isinstance check fails
    """
    if optional is None:
        raise ValueError("Unexpected None value")
    if expected_type is not None and not isinstance(optional, expected_type):
        raise TypeError(f"Expected type {expected_type}, got {type(optional).__name__}")
    return optional


def _float_or_none(value: float | int | str | None) -> float | None:
    """Return a float (if float or int) or None if it's None or "None".

    This helper keeps argument parsing explicit about optional float inputs.
    """

    if value is None or value == "None":
        return None
    return float(value)


def _check_weighting_methods_input(
    df: pd.DataFrame,
    weights: pd.Series,
    object_name: str,
) -> None:
    """
    This is a helper function fo weighting methods functions.
    It checks the inputs are of the correct type and shapes.

    Args:
        df (pd.DataFrame):
        weights (pd.Series):
        object_name (str):

    Raises:
        TypeError: if df is not a DataFrame
        TypeError: if weights is not a pd.Series
        ValueError: {object_name}_weights must be the same length as {object_name}_df
        ValueError: {object_name}_df index must be the same as {object_name}_weights index
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{object_name}_df must be a pandas DataFrame, is {type(df)}")
    if not isinstance(weights, pd.Series):
        raise TypeError(
            f"{object_name}_weights must be a pandas Series, is {type(weights)}"
        )
    if df.shape[0] != weights.shape[0]:
        raise ValueError(
            f"{object_name}_weights must be the same length as {object_name}_df: "
            f"{df.shape[0]}, {weights.shape[0]}"
        )
    if not df.index.equals(weights.index):
        raise ValueError(
            f"{object_name}_df index must be the same as {object_name}_weights index"
        )


# This is so to avoid various cyclic imports (since various files call sample_class, and then sample_class also calls these files)
# TODO: (p2) move away from this method once we restructure Sample and BalanceDF objects...
def _isinstance_sample(obj: Any) -> bool:
    try:
        from balance import sample_class
    except ImportError:
        return False

    return isinstance(obj, sample_class.Sample)


def guess_id_column(dataset: pd.DataFrame, column_name: str | None = None) -> str:
    """
    Guess the id column of a given dataset.
    Possible values for guess: 'id'.

    Args:
        dataset (pd.DataFrame): dataset to guess id column
        column_name (str, optional): Given id column name. Defaults to None,
            which will guess the id column or raise exception.

    Returns:
        str: name of guessed id column
    """
    # TODO: add a general argument for the user so they could set
    # a list of possible userid column names instead of only "id".
    # This should go as an input into Sample.from_frame as well.
    columns = list(dataset.columns)
    if column_name is not None:
        if column_name in columns:
            return column_name
        else:
            raise ValueError(f"Dataframe does not have column '{column_name}'")
    else:
        possible_columns = [i for i in ["id"] if i in columns]
        if len(possible_columns) != 1:
            raise ValueError(
                "Cannot guess id column name for this DataFrame. "
                "Please provide a value in id_column"
            )
        else:
            column_name = possible_columns[0]
            logger.warning(f"Guessed id column name {column_name} for the data")
            return column_name


def _is_arraylike(o: Any) -> bool:
    """Test (returns True) if an object is an array-ish type (a numpy array, or
    a sequence, but not a string). Not the same as numpy's arraylike,
    which also applies to scalars which can be turned into arrays.

    Args:
        o: Object to test.

    Returns:
        bool: returns True if an object is an array-ish type.
    """
    return (
        isinstance(o, np.ndarray)
        or isinstance(o, pd.Series)
        or (
            hasattr(pd.arrays, "NumpyExtensionArray")
            and isinstance(o, pd.arrays.NumpyExtensionArray)
        )
        or isinstance(o, pd.arrays.StringArray)
        or isinstance(o, pd.arrays.IntegerArray)
        or isinstance(o, pd.arrays.BooleanArray)
        or "pandas.core.arrays" in str(type(o))  # support any pandas array type.
        or (isinstance(o, collections.abc.Sequence) and not isinstance(o, str))
    )


def rm_mutual_nas(*args: Any) -> List[Any]:
    """
    Remove entries in a position which is na or infinite in any of the arguments.

    Ignores args which are None.

    Can accept multiple array-like arguments or a single array-like argument. Handles pandas and numpy arrays.

    Raises:
        ValueError: If any argument is not array-like. (see: :func:`_is_arraylike`)
        ValueError: If arguments include arrays of different lengths.

    Returns:
        List: A list containing the original input arrays, after removing elements that have a missing or infinite value in the same position as any of the other arrays.
    """
    if any(not (a is None or _is_arraylike(a)) for a in args):
        raise ValueError("All arguments must be arraylike")
    # create a set of lengths of all arrays, and see if there are is more than
    # one array length: (we shouldn't, since we expect all arrays to have the same length)
    if len({len(a) for a in args if a is not None}) > 1:
        raise ValueError("All arrays must be of same length")

    missing_mask = reduce(
        lambda x, y: x | y,
        [
            _process_series_for_missing_mask(pd.Series(x, dtype="object"))
            for x in args
            if x is not None
        ],
    )
    nonmissing_mask = ~missing_mask

    def _return_type_creation_function(x: Any) -> Callable | Any:
        # The numpy.ndarray constructor doesn't take the same arguments as np.array
        if isinstance(x, np.ndarray):
            return lambda obj: np.array(obj, dtype=x.dtype)
        # same with pd.arrays.PandasArray, pd.arrays.StringArray, etc.
        elif "pandas.core.arrays" in str(type(x)):
            return lambda obj: pd.array(obj, dtype=x.dtype)
        else:
            return type(x)

    #  Need to convert each argument to a type that can be indexed and then
    #  convert back
    original_types = [_return_type_creation_function(x) for x in args]
    r = [pd.Series(x)[nonmissing_mask].tolist() if x is not None else x for x in args]

    # Reapply the index for pd.Series
    r = [
        (
            pd.Series(data, index=pd.Series(orig_data)[nonmissing_mask].index)
            if isinstance(orig_data, pd.Series)
            else data
        )
        for data, orig_data in zip(r, args)
    ]

    # reproduce the type of each array in the result
    r = [(t(x) if x is not None else x) for t, x in zip(original_types, r)]
    if len(args) == 1:
        r = r[0]
    return r


# TODO: (p2) create choose_variables_df that only works with pd.DataFrames as input, and wrap it with something that deals with Sample.
#       This would help clarify the logic of each function.
def choose_variables(
    *dfs: pd.DataFrame | Any,
    variables: List[str] | set[str] | None = None,
    df_for_var_order: int = 0,
) -> List[str]:
    """
     Returns a list of joint (intersection of) variables present in all the input dataframes and also in the `variables` set or list
     if provided. The order of the returned variables is conditional on the input:
         - If a `variables` argument is supplied as a list - the order will be based on the order in the variables list.
         - If a `variables` is not a list (e.g.: set or None), the order is determined by the order of the columns in the dataframes
             supplied. The dataframe chosen for the order is determined by the `df_for_var_order` argument. 0 means the order from the first df,
             1 means the order from the second df, etc.

    Args:
         *dfs (pd.DataFrame | Any): One or more pandas.DataFrames or balance.Samples.
         variables (List[str] | set[str] | None): The variables to choose from. If None, returns all joint variables found
             in the input dataframes. Defaults to None.
         df_for_var_order (int): Index of the dataframe used to determine the order of the variables in the output list.
             Defaults to 0. This is used only if the `variables` argument is not a list (e.g.: a set or None).

     Raises:
         ValueError: If one or more requested variables are not present in all dataframes.

     Returns:
         List[str]: A list of the joint variables present in all dataframes and in the `variables` set or list, ordered
             based on the input conditions specified.
    """

    if (variables is not None) and (len(variables) == 0):
        variables = None

    # This is a list of lists with the variable names of the input dataframes
    dfs_variables = [
        d.covars().names() if _isinstance_sample(d) else d.columns.values.tolist()
        for d in dfs
        if d is not None
    ]

    var_list_for_order = (
        variables if (isinstance(variables, list)) else dfs_variables[df_for_var_order]
    )

    intersection_variables = set(
        reduce(lambda x, y: set(x).intersection(set(y)), dfs_variables)
    )

    union_variables = reduce(lambda x, y: set(x).union(set(y)), dfs_variables)

    if len(set(union_variables).symmetric_difference(intersection_variables)) > 0:
        logger.warning(
            f"Ignoring variables not present in all Samples: {union_variables.difference(intersection_variables)}"
        )

    if variables is None:
        variables = intersection_variables
    else:
        variables = set(variables)
        variables_not_in_df = variables.difference(intersection_variables)

        if len(variables_not_in_df) > 0:
            logger.warning(
                f"These variables are not included in the dataframes: {variables_not_in_df}"
            )
            raise ValueError(
                f"{len(variables_not_in_df)} requested variables are not in all Samples: "
                f"{variables_not_in_df}"
            )
        variables = intersection_variables.intersection(variables)
    logger.debug(f"Joint variables in all dataframes: {list(variables)}")

    if (variables is None) or (len(variables) == 0):
        logger.warning("Sample and target have no variables in common")
        return []

    ordered_variables = []
    for val in var_list_for_order:
        if val in variables and val not in ordered_variables:
            ordered_variables.append(val)
    # NOTE: the above is just like:
    # seen = set()
    # ordered_variables = [val for val in dfs_variables[df_for_var_order] if val in variables and val not in seen and not seen.add(val)]

    # TODO: consider changing the return form list to a tuple. But doing so would require to deal with various edge cases around the codebase.
    return ordered_variables


def find_items_index_in_list(a_list: List[Any], items: List[Any]) -> List[int]:
    """Finds the index location of a given item in an array.

    Helpful references:
        - https://stackoverflow.com/a/48898363
        - https://stackoverflow.com/a/176921

    Args:
        x (List[Any]): a list of items to find their index
        items (List[Any]): a list of items to search for

    Returns:
        List[int]: a list of indices of the items in x that appear in the items list.
    """
    # TODO: (p2) Optimization note: checking that i is in set each time is expensive -
    #       there are probably faster ways to do it. Consider using a dict-based approach for large lists.
    return [a_list.index(i) for i in items if i in set(a_list)]


def get_items_from_list_via_indices(a_list: List[Any], indices: List[int]) -> List[Any]:
    """Gets a subset of items from a list via indices

    Source code (there doesn't seem to be a better solution): https://stackoverflow.com/a/6632209

    Args:
        a_list (List[Any]): a list of items to extract a list from
        indices (List[int]): a list of indexes of items to get

    Returns:
        List[Any]: a list of extracted items
    """
    return [a_list[i] for i in indices]


def _true_false_str_to_bool(x: str) -> bool:
    """Changes strings such as 'false' to False and 'true' to True.

    Args:
        x (str): String to be converted (ideally 'true' or 'false' - case is ignored).

    Raises:
        ValueError: If x is not 'true' or 'false'.

    Returns:
        bool: True if x is 'true', False if x is 'false'.
    """
    if x.lower() == "false":
        return False
    elif x.lower() == "true":
        return True
    else:
        raise ValueError(
            f"{x} is not an accepted value, please pass either 'True' or 'False' (lower/upper case is ignored)"
        )
