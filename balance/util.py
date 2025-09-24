# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import collections.abc

import copy
import logging
import tempfile
import uuid
import warnings
from functools import reduce
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from IPython.lib.display import FileLink
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from patsy.contrasts import ContrastMatrix
from patsy.highlevel import dmatrix, ModelDesc
from scipy.sparse import csc_matrix, hstack

logger: logging.Logger = logging.getLogger(__package__)

# TODO: split util and adjustment files into separate files: transformations, model_matrix, others..


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
def _isinstance_sample(obj) -> bool:
    try:
        from balance import sample_class
    except ImportError:
        return False

    return isinstance(obj, sample_class.Sample)


def guess_id_column(dataset: pd.DataFrame, column_name: Optional[str] = None):
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
        # TODO: change to ValueError?!
        raise Exception(
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
            # TODO: change to ValueError?!
            raise Exception(
                f"Can't add NA indicator to columns containing NAs and the value '{replace_val_obj}', "
                f"i.e. column: {c}"
            )
        if c in categorical_cols:
            filled_col = (
                df[c].cat.add_categories(replace_val_obj).fillna(replace_val_obj)
            )
            df[c] = filled_col.infer_objects(copy=False)
        elif c in non_numeric_cols:
            df[c] = _safe_fillna_and_infer(df[c], replace_val_obj)
        else:
            df[c] = _safe_fillna_and_infer(df[c], replace_val_num)

    return pd.concat((df, na_indicators), axis=1)


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


def formula_generator(variables, formula_type: str = "additive") -> str:
    """Create formula to build the model matrix
        Default is additive formula.
    Args:
        variables: list with names of variables (as strings) to combine into a formula
        formula_type (str, optional): how to construct the formula. Currently only "additive" is supported. Defaults to "additive".

    Raises:
        Exception: "This formula type is not supported.'" "Please provide a string formula"

    Returns:
        str: A string representing the formula

    Examples:
        ::

            formula_generator(['a','b','c'])
            # returns 'c + b + a'
    """
    if formula_type == "additive":
        rhs_formula = " + ".join(sorted(variables, reverse=True))
    else:
        # TODO ValueError?!
        raise Exception(
            "This formula type is not supported.'" "Please provide a string formula"
        )

    logger.debug(f"Model default formula: {rhs_formula}")
    return rhs_formula


def dot_expansion(formula, variables: List):
    """Build a formula string by replacing "." with "summing" all the variables,
    If no dot appears, returns the formula as is.

    This function is named for the 'dot' operators in R, where a formula given
    as ' ~ .' means "use all variables in dataframe.

    Args:
        formula: The formula to expand.
        variables (List): List of all variables in the dataframe we build the formula for.

    Raises:
        Exception: "Variables should not be empty. Please provide a list of strings."
        Exception:  "Variables should be a list of strings and have to be included."

    Returns:
        A string formula replacing the '.'' with all variables in variables.
        If no '.' is present, then the original formula is returned as is.

    Examples:
        ::

            dot_expansion('.', ['a','b','c','d']) # (a+b+c+d)
            dot_expansion('b:(. - a)', ['a','b','c','d']) # b:((a+b+c+d) - a)
            dot_expansion('a*b', ['a','b','c','d']) # a*b
            dot_expansion('.', None) # Raise error

            import pandas as pd
            d = {'a': ['a1','a2','a1','a1'], 'b': ['b1','b2','b3','b3'],
                        'c': ['c1','c1','c2','c1'], 'd':['d1','d1','d2','d3']}
            df = pd.DataFrame(data=d)
            dot_expansion('.', df) # Raise error
            dot_expansion('.', list(df.columns)) # (a+b+c+d)
    """
    if variables is None:
        # TODO: TypeError?
        raise Exception(
            "Variables should not be empty. Please provide a list of strings."
        )

    if not isinstance(variables, list):
        # TODO: TypeError?
        raise Exception(
            "Variables should be a list of strings and have to be included."
            "Please provide a list of your variables. If you would like to use all variables in"
            "a dataframe, insert variables = list(df.columns)"
        )
    if formula.find(".") == -1:
        rhs = formula
    else:
        dot = "(" + "+".join(x for x in variables) + ")"
        rhs = str(formula).replace(".", dot)
    return rhs


class one_hot_encoding_greater_2:
    """
    This class creates a special encoding for factor variable to be used in a LASSO model.
    For variables with exactly two levels using this in dmatrix will only keep one level, i.e.
    will create one column with a 0 or 1 indicator for one of the levels. The level kept will
    be the second one, based on loxicographical order of the levels.
    For variables with more than 2 levels, using this in dmatrix will keep all levels
    as columns of the matrix.

    References:
    1. More about this encoding:
    # https://stats.stackexchange.com/questions/69804/group-categorical-variables-in-glmnet/107958#107958
    3. Source code: adaptation of
    # https://patsy.readthedocs.io/en/latest/categorical-coding.html

    Examples:
        ::

        import pandas as pd
        d = {'a': ['a1','a2','a1','a1'], 'b': ['b1','b2','b3','b3'],
        'c': ['c1','c1','c2','c1'], 'd':['d1','d1','d2','d3']}
        df = pd.DataFrame(data=d)

        print(dmatrix('C(a, one_hot_encoding_greater_2)', df, return_type = 'dataframe'))
            #   Intercept  C(a, one_hot_encoding_greater_2)[a2]
            #0        1.0                                   0.0
            #1        1.0                                   1.0
            #2        1.0                                   0.0
            #3        1.0                                   0.0
        print(dmatrix('C(a, one_hot_encoding_greater_2)-1', df, return_type = 'dataframe'))
            #   C(a, one_hot_encoding_greater_2)[a2]
            #0                                   0.0
            #1                                   1.0
            #2                                   0.0
            #3                                   0.0
        print(dmatrix('C(b, one_hot_encoding_greater_2)', df, return_type = 'dataframe'))
            #   Intercept  C(b, one_hot_encoding_greater_2)[b1]  \
            #0        1.0                                   1.0
            #1        1.0                                   0.0
            #2        1.0                                   0.0
            #3        1.0                                   0.0
            #
            #   C(b, one_hot_encoding_greater_2)[b2]  C(b, one_hot_encoding_greater_2)[b3]
            #0                                   0.0                                   0.0
            #1                                   1.0                                   0.0
            #2                                   0.0                                   1.0
            #3                                   0.0                                   1.0
        print(dmatrix('C(b, one_hot_encoding_greater_2)-1', df, return_type = 'dataframe'))
            #   C(b, one_hot_encoding_greater_2)[b1]  C(b, one_hot_encoding_greater_2)[b2]  \
            #0                                   1.0                                   0.0
            #1                                   0.0                                   1.0
            #2                                   0.0                                   0.0
            #3                                   0.0                                   0.0
            #
            #   C(b, one_hot_encoding_greater_2)[b3]
            #0                                   0.0
            #1                                   0.0
            #2                                   1.0

        d = {'a': ['a1','a1','a1','a1'], 'b': ['b1','b2','b3','b3']}
        df = pd.DataFrame(data=d)

        print(dmatrix('C(a, one_hot_encoding_greater_2)-1', df, return_type = 'dataframe'))
        print(dmatrix('C(a, one_hot_encoding_greater_2):C(b, one_hot_encoding_greater_2)-1', df, return_type = 'dataframe'))   C(a, one_hot_encoding_greater_2)[a1]
            #0                                   1.0
            #1                                   1.0
            #2                                   1.0
            #3                                   1.0
            #   C(a, one_hot_encoding_greater_2)[a1]:C(b, one_hot_encoding_greater_2)[b1]  \
            #0                                                1.0
            #1                                                0.0
            #2                                                0.0
            #3                                                0.0
            #
            #   C(a, one_hot_encoding_greater_2)[a1]:C(b, one_hot_encoding_greater_2)[b2]  \
            #0                                                0.0
            #1                                                1.0
            #2                                                0.0
            #3                                                0.0

    """

    def __init__(self, reference: int = 0) -> None:
        self.reference = reference

    def code_with_intercept(self, levels):
        if len(levels) == 2:
            eye = np.eye(len(levels) - 1)
            contrasts = np.vstack(
                (
                    eye[: self.reference, :],
                    np.zeros((1, len(levels) - 1)),
                    eye[self.reference :, :],
                )
            )
            suffixes = [
                f"[{level}]"
                for level in levels[: self.reference] + levels[self.reference + 1 :]
            ]
            contrasts_mat = ContrastMatrix(contrasts, suffixes)
        else:
            contrasts_mat = ContrastMatrix(
                np.eye(len(levels)), [f"[{level}]" for level in levels]
            )
        return contrasts_mat

    def code_without_intercept(self, levels):
        return self.code_with_intercept(levels)


def process_formula(formula, variables: List, factor_variables=None):
    """Process a formula string:
        1. Expand .  notation using dot_expansion function
        2. Remove intercept (if using ipw, it will be added automatically by sklearn)
        3. If factor_variables is not None, one_hot_encoding_greater_2 is applied
        to factor_variables


    Args:
        formula: A string representing the formula
        variables (List): list of all variables to include (usually all variables in data)
        factor_variables: list of names of factor variables that we use
            one_hot_encoding_greater_2 for. Note that these should be also
            part of variables.
            Default is None, in which case no special contrasts are
            applied (using patsy defaults). one_hot_encoding_greater_2
            creates one-hot-encoding for all categorical variables with
            more than 2 categories (i.e. the number of columns will
            be equal to the number of categories), and only 1
            column for variables with 2 levels (treatment contrast).

    Raises:
        Exception: "Not all factor variables are contained in variables"

    Returns:
        a ModelDesc object to build a model matrix using patsy.dmatrix.

    Examples:
        ::

            f1 = process_formula('a:(b+aab)', ['a','b','aab'])
            print(f1)
                # ModelDesc(lhs_termlist=[],
                #       rhs_termlist=[Term([EvalFactor('a'), EvalFactor('b')]),
                #                     Term([EvalFactor('a'), EvalFactor('aab')])])
            f2 = process_formula('a:(b+aab)', ['a','b','aab'], ['a','b'])
            print(f2)
                # ModelDesc(lhs_termlist=[],
                #       rhs_termlist=[Term([EvalFactor('C(a, one_hot_encoding_greater_2)'),
                #                           EvalFactor('C(b, one_hot_encoding_greater_2)')]),
                #                     Term([EvalFactor('C(a, one_hot_encoding_greater_2)'),
                #                           EvalFactor('aab')])])
    """
    # Check all factor variables are in variables:
    if (factor_variables is not None) and (not set(factor_variables) <= set(variables)):
        # TODO: ValueError?!
        raise Exception("Not all factor variables are contained in variables")

    formula = dot_expansion(formula, variables)
    # Remove the intercept since it is added by sklearn/cbps
    formula = formula + " -1"
    desc = ModelDesc.from_formula(formula)

    if factor_variables is not None:
        # We use one_hot_encoding_greater_2 for building the model matrix for factor_variables
        # Reference: https://patsy.readthedocs.io/en/latest/categorical-coding.html
        for i, term_i in enumerate(desc.rhs_termlist):
            for j, factor_j in enumerate(term_i.factors):
                if factor_j.code in factor_variables:
                    var = desc.rhs_termlist[i].factors[j].code
                    desc.rhs_termlist[i].factors[
                        j
                    ].code = f"C({var}, one_hot_encoding_greater_2)"

    return desc


def build_model_matrix(
    df: pd.DataFrame,
    formula: str = ".",
    factor_variables: Optional[List] = None,
    return_sparse: bool = False,
) -> Dict[str, Any]:
    """Build a model matrix from a formula (using patsy.dmatrix)

    Args:
        df (pd.DataFrame): The data from which to create the model matrix (pandas dataframe)
        formula (str, optional): a string representing the formula to use for building the model matrix.
                Default is additive formula with all variables in df. Defaults to ".".
        factor_variables (LisOptional[List]t, optional): list of names of factor variables that we use
                         one_hot_encoding_greater_2 for.
                         Default is None, in which case no special contrasts are applied
                         (uses patsy defaults).
                         one_hot_encoding_greater_2 creates one-hot-encoding for all
                         categorical variables with more than 2 categories (i.e. the
                         number of columns will be equal to the number of categories), and only 1
                         column for variables with 2 levels (treatment contrast).
        return_sparse (bool, optional): whether to return a sparse matrix using scipy.sparse.csc_matrix. Defaults to False.

    Raises:
        Exception: "Variable names cannot contain characters '[' or ']'"
        Exception: "Not all factor variables are contained in df"

    Returns:
        Dict[str, Any]:     A dictionary of 2 elements:
            1. model_matrix - this is a pd dataframe or a csc_matrix (depends on return_sparse), ordered by columns names
            2. model_matrix_columns - A list of the columns names of model_matrix
            (We include model_matrix_columns as a separate argument since if we return a sparse X_matrix,
            it doesn't have a columns names argument and these need to be kept separately,
            see here:
            https://stackoverflow.com/questions/35086940/how-can-i-give-row-and-column-names-to-scipys-csr-matrix.)

    Examples:
        ::

            import pandas as pd
            d = {'a': ['a1','a2','a1','a1'], 'b': ['b1','b2','b3','b3']}
            df = pd.DataFrame(data=d)

            print(build_model_matrix(df, 'a'))
                # {'model_matrix':    a[a1]  a[a2]
                # 0    1.0    0.0
                # 1    0.0    1.0
                # 2    1.0    0.0
                # 3    1.0    0.0,
                # 'model_matrix_columns': ['a[a1]', 'a[a2]']}


            print(build_model_matrix(df, '.'))
                # {'model_matrix':    a[a1]  a[a2]  b[T.b2]  b[T.b3]
                # 0    1.0    0.0      0.0      0.0
                # 1    0.0    1.0      1.0      0.0
                # 2    1.0    0.0      0.0      1.0
                # 3    1.0    0.0      0.0      1.0,
                # 'model_matrix_columns': ['a[a1]', 'a[a2]', 'b[T.b2]', 'b[T.b3]']}


            print(build_model_matrix(df, '.', factor_variables=['a']))
                # {'model_matrix':    C(a, one_hot_encoding_greater_2)[a2]  b[T.b2]  b[T.b3]
                # 0                                0.0      0.0      0.0
                # 1                                1.0      1.0      0.0
                # 2                                0.0      0.0      1.0
                # 3                                0.0      0.0      1.0,
                # 'model_matrix_columns': ['C(a, one_hot_encoding_greater_2)[a2]', 'b[T.b2]', 'b[T.b3]']}


            print(build_model_matrix(df, 'a', return_sparse=True))
                # {'model_matrix': <4x2 sparse matrix of type '<class 'numpy.float64'>'
                # with 4 stored elements in Compressed Sparse Column format>, 'model_matrix_columns': ['a[a1]', 'a[a2]']}
            print(build_model_matrix(df, 'a', return_sparse=True)["model_matrix"].toarray())
                # [[1. 0.]
                # [0. 1.]
                # [1. 0.]
                # [1. 0.]]
    """
    variables = list(df.columns)

    bracket_variables = [v for v in variables if ("[" in v) or ("]" in v)]
    if len(bracket_variables) > 0:
        # TODO: ValueError?
        raise Exception(
            "Variable names cannot contain characters '[' or ']'"
            f"because patsy uses them to denote one-hot encoded categoricals: ({bracket_variables})"
        )

    # Check all factor variables are in variables:
    if factor_variables is not None:
        if not (set(factor_variables) <= set(variables)):
            # TODO: ValueError?
            raise Exception("Not all factor variables are contained in df")

    model_desc = process_formula(formula, variables, factor_variables)
    # dmatrix cannot get Int64Dtype as data type. Hence converting all numeric columns to float64.
    for x in df.columns:
        if (is_numeric_dtype(df[x])) and (not is_bool_dtype(df[x])):
            df[x] = df[x].astype("float64")

    X_matrix = dmatrix(model_desc, data=df, return_type="dataframe")
    # Sorting the output in order to eliminate edge cases that cause column order to be stochastic
    X_matrix = X_matrix.sort_index(axis=1)
    logger.debug(f"X_matrix shape: {X_matrix.shape}")
    X_matrix_columns = list(X_matrix.columns)
    if return_sparse:
        X_matrix = csc_matrix(X_matrix)

    return {"model_matrix": X_matrix, "model_matrix_columns": X_matrix_columns}


def _prepare_input_model_matrix(
    sample: Union[pd.DataFrame, Any],
    target: Union[pd.DataFrame, Any, None] = None,
    variables: Optional[List] = None,
    add_na: bool = True,
    fix_columns_names: bool = True,
) -> Dict[str, Any]:
    """Helper function to model_matrix. Prepare and check input of sample and target:
        - Choose joint variables to sample and target (or by given variables)
        - Extract sample and target dataframes
        - Concat dataframes together
        - Add na indicator if required.

    Args:
        sample (Union[pd.DataFrame, Any]): This can either be a DataFrame or a Sample object. TODO: add text.
        target (Union[pd.DataFrame, Any, None], optional): This can either be a DataFrame or a Sample object.. Defaults to None.
        variables (Optional[List], optional): Defaults to None. TODO: add text.
        add_na (bool, optional): Defaults to True. TODO: add text.
        fix_columns_names (bool, optional): Defaults to True. If to fix the column names of the DataFrame by changing special characters to '_'.

    Raises:
        Exception: "Variable names cannot contain characters '[' or ']'"

    Returns:
        Dict[str, Any]: returns a dictionary containing two keys: 'all_data' and 'sample_n'.
            The 'all_data' is a pd.DataFrame with all the rows of 'sample' (including 'target', if supplied)
            The'sample_n' is the number of rows in the first input DataFrame ('sample').

    Examples:
        ::

            import pandas as pd
            import balance.util

            df = pd.DataFrame(
                {"a": ["a1", "a2", "a1", "a1"], "b": ["b1", "b2", "b3", "b3"]}
            )

            print(balance.util._prepare_input_model_matrix(df, df))
                # {'all_data':     a   b
                # 0  a1  b1
                # 1  a2  b2
                # 2  a1  b3
                # 3  a1  b3
                # 0  a1  b1
                # 1  a2  b2
                # 2  a1  b3
                # 3  a1  b3, 'sample_n': 4}


            # It automatically fixes the column names for you from special characters
            df = pd.DataFrame(
                {"a": ["a1", "a2", "a1", "a1"], "b./ * b": ["b1", "b2", "b3", "b3"]}
            )

            print(balance.util._prepare_input_model_matrix(df, df))
                # {'all_data':     a b_____b
                # 0  a1      b1
                # 1  a2      b2
                # 2  a1      b3
                # 3  a1      b3
                # 0  a1      b1
                # 1  a2      b2
                # 2  a1      b3
                # 3  a1      b3, 'sample_n': 4}
    """
    variables = choose_variables(sample, target, variables=variables)

    bracket_variables = [v for v in variables if ("[" in v) or ("]" in v)]
    if len(bracket_variables) > 0:
        raise Exception(
            "Variable names cannot contain characters '[' or ']'"
            f"because patsy uses them to denote one-hot encoded categoricals: ({bracket_variables})"
        )

    if _isinstance_sample(sample):
        sample_df = sample._df
    else:
        sample_df = sample
    assert sample_df.shape[0] > 0, "sample must have more than zero rows"
    sample_n = sample_df.shape[0]
    sample_df = sample_df.loc[:, variables]

    if target is None:
        target_df = None
    elif _isinstance_sample(target):
        target_df = target._df.loc[:, variables]
    else:
        target_df = target.loc[:, variables]

    all_data = pd.concat((sample_df, target_df))

    if add_na:
        all_data = add_na_indicator(all_data)
    else:
        logger.warning("Dropping all rows with NAs")

    if fix_columns_names:
        all_data.columns = all_data.columns.str.replace(
            r"[^\w]", "_", regex=True
        ).infer_objects(copy=False)
        all_data = _make_df_column_names_unique(all_data)

    return {"all_data": all_data, "sample_n": sample_n}


def model_matrix(
    sample: Union[pd.DataFrame, Any],
    target: Union[pd.DataFrame, Any, None] = None,
    variables: Optional[List] = None,
    add_na: bool = True,
    return_type: str = "two",
    return_var_type: str = "dataframe",
    formula: Optional[List[str]] = None,
    penalty_factor: Optional[List[float]] = None,
    one_hot_encoding: bool = False,
) -> Dict[str, Union[List[Any], np.ndarray, pd.DataFrame, csc_matrix, None]]:
    """Create a model matrix from a sample (and target).
    The default is to use an additive formula for all variables (or the ones specified).
    Can also create a custom model matrix if a formula is provided.

    Args:
        sample (Union[pd.DataFrame, Any]): The Samples from which to create the model matrix. This can either be a DataFrame or a Sample object.
        target (Union[pd.DataFrame, Any, None], optional): See sample. Defaults to None. This can either be a DataFrame or a Sample object.
        variables (Optional[List]): the names of the variables to include (when 'None' then
            all joint variables to target and sample are used). Defaults to None.
        add_na (bool, optional): whether to call add_na_indicator on the data before constructing
            the matrix.If add_na = True, then the function add_na_indicator is applied,
            i.e. if a column in the DataFrame contains NAs, replace these with 0 or "_NA", and
            add another column of an indicator variable for which rows were NA.
            If add_na is False, observations with any missing data will be
            omitted from the model. Defaults to True.
        return_type (str, optional): whether to return a single matrix ('one'), or a dict of
            sample and target matrices. Defaults to "two".
        return_var_type (str, optional): whether to return a "dataframe" (pd.dataframe) a "matrix" (np.ndarray)
            (i.e. only values of the output dataframe), or a "sparse" matrix. Defaults to "dataframe".
        formula (Optional[List[str]], optional): according to what formula to construct the matrix. If no formula is provided an
            additive formula is applied. This may be a string or a list of strings
            representing different parts of the formula that will be concated together.
            Default is None, which will create an additive formula from the available variables. Defaults to None.
        penalty_factor (Optional[List[float]], optional): the penalty used in the sklearn function in ipw. The penalty
            should have the same length as the formula list. If not provided,
            assume the same penalty for all variables. Defaults to None.
        one_hot_encoding (bool, optional): whether to encode all factor variables in the model matrix with
            one_hot_encoding_greater_2. This is recommended in case of using
            LASSO on the data (Default: False).
            one_hot_encoding_greater_2 creates one-hot-encoding for all
            categorical variables with more than 2 categories (i.e. the
            number of columns will be equal to the number of categories),
            and only 1 column for variables with 2 levels (treatment contrast). Defaults to False.

    Returns:
        Dict[str, Union[List[Any], np.ndarray, pd.DataFrame, csc_matrix, None]]
            a dict of:
                1. "model_matrix_columns_names": columns names of the model matrix
                2. "penalty_factor ": a penalty_factor for each column in the model matrix
                3. "model_matrix" (or: "sample" and "target"): the DataFrames for the sample and target
                (one or two, according to return_type)
                    If return_sparse="True" returns a sparse matrix (csc_matrix)

    Examples:
        ::

            import pandas as pd
            d = {'a': ['a1','a2','a1','a1'], 'b': ['b1','b2','b3','b3']}
            df = pd.DataFrame(data=d)

            model_matrix(df)
                # {'model_matrix_columns_names': ['b[b1]', 'b[b2]', 'b[b3]', 'a[T.a2]'],
                #  'penalty_factor': array([1, 1, 1, 1]),
                #  'sample':    b[b1]  b[b2]  b[b3]  a[T.a2]
                #  0    1.0    0.0    0.0      0.0
                #  1    0.0    1.0    0.0      1.0
                #  2    0.0    0.0    1.0      0.0
                #  3    0.0    0.0    1.0      0.0,
                #  'target': None}

            model_matrix(df, formula = 'a*b')
                # {'model_matrix_columns_names': ['a[a1]',
                #   'a[a2]',
                #   'b[T.b2]',
                #   'b[T.b3]',
                #   'a[T.a2]:b[T.b2]',
                #   'a[T.a2]:b[T.b3]'],
                #  'penalty_factor': array([1, 1, 1, 1, 1, 1]),
                #  'sample':    a[a1]  a[a2]  b[T.b2]  b[T.b3]  a[T.a2]:b[T.b2]  a[T.a2]:b[T.b3]
                #  0    1.0    0.0      0.0      0.0              0.0              0.0
                #  1    0.0    1.0      1.0      0.0              1.0              0.0
                #  2    1.0    0.0      0.0      1.0              0.0              0.0
                #  3    1.0    0.0      0.0      1.0              0.0              0.0,
                #  'target': None}

            model_matrix(df, formula = ['a','b'], penalty_factor=[1,2])
                # {'model_matrix_columns_names': ['a[a1]', 'a[a2]', 'b[b1]', 'b[b2]', 'b[b3]'],
                #  'penalty_factor': array([1, 1, 2, 2, 2]),
                #  'sample':    a[a1]  a[a2]  b[b1]  b[b2]  b[b3]
                #  0    1.0    0.0    1.0    0.0    0.0
                #  1    0.0    1.0    0.0    1.0    0.0
                #  2    1.0    0.0    0.0    0.0    1.0
                #  3    1.0    0.0    0.0    0.0    1.0,
                #  'target': None}

            model_matrix(df, formula = ['a','b'], penalty_factor=[1,2], one_hot_encoding=True)
                # {'model_matrix_columns_names': ['C(a, one_hot_encoding_greater_2)[a2]',
                #   'C(b, one_hot_encoding_greater_2)[b1]',
                #   'C(b, one_hot_encoding_greater_2)[b2]',
                #   'C(b, one_hot_encoding_greater_2)[b3]'],
                #  'penalty_factor': array([1, 2, 2, 2]),
                #  'sample':    C(a, one_hot_encoding_greater_2)[a2]  ...  C(b, one_hot_encoding_greater_2)[b3]
                #  0                                0.0  ...                                0.0
                #  1                                1.0  ...                                0.0
                #  2                                0.0  ...                                1.0
                #  3                                0.0  ...                                1.0
                # [4 rows x 4 columns],
                # 'target': None}

            model_matrix(df, formula = ['a','b'], penalty_factor=[1,2], return_sparse = True)
                # {'model_matrix_columns_names': ['a[a1]', 'a[a2]', 'b[b1]', 'b[b2]', 'b[b3]'],
                #  'penalty_factor': array([1, 1, 2, 2, 2]),
                #  'sample': <4x5 sparse matrix of type '<class 'numpy.float64'>'
                #  	with 8 stored elements in Compressed Sparse Column format>,
                #  'target': None}

            model_matrix(df, target = df)
                # {'model_matrix_columns_names': ['b[b1]', 'b[b2]', 'b[b3]', 'a[T.a2]'],
                #  'penalty_factor': array([1, 1, 1, 1]),
                #  'sample':    b[b1]  b[b2]  b[b3]  a[T.a2]
                #  0    1.0    0.0    0.0      0.0
                #  1    0.0    1.0    0.0      1.0
                #  2    0.0    0.0    1.0      0.0
                #  3    0.0    0.0    1.0      0.0,
                #  'target':    b[b1]  b[b2]  b[b3]  a[T.a2]
                #  0    1.0    0.0    0.0      0.0
                #  1    0.0    1.0    0.0      1.0
                #  2    0.0    0.0    1.0      0.0
                #  3    0.0    0.0    1.0      0.0}

            model_matrix(df, target = df, return_type = "one")
                # {'model_matrix_columns_names': ['b[b1]', 'b[b2]', 'b[b3]', 'a[T.a2]'],
                #  'penalty_factor': array([1, 1, 1, 1]),
                #  'model_matrix':    b[b1]  b[b2]  b[b3]  a[T.a2]
                #  0    1.0    0.0    0.0      0.0
                #  1    0.0    1.0    0.0      1.0
                #  2    0.0    0.0    1.0      0.0
                #  3    0.0    0.0    1.0      0.0
                #  0    1.0    0.0    0.0      0.0
                #  1    0.0    1.0    0.0      1.0
                #  2    0.0    0.0    1.0      0.0
                #  3    0.0    0.0    1.0      0.0}

            model_matrix(df, target = df, formula=['a','b'],return_type = "one")
                # {'model_matrix_columns_names': ['a[a1]', 'a[a2]', 'b[b1]', 'b[b2]', 'b[b3]'],
                #  'penalty_factor': array([1, 1, 1, 1, 1]),
                #  'model_matrix':    a[a1]  a[a2]  b[b1]  b[b2]  b[b3]
                #  0    1.0    0.0    1.0    0.0    0.0
                #  1    0.0    1.0    0.0    1.0    0.0
                #  2    1.0    0.0    0.0    0.0    1.0
                #  3    1.0    0.0    0.0    0.0    1.0
                #  0    1.0    0.0    1.0    0.0    0.0
                #  1    0.0    1.0    0.0    1.0    0.0
                #  2    1.0    0.0    0.0    0.0    1.0
                #  3    1.0    0.0    0.0    0.0    1.0}
    """
    logger.debug("Starting building the model matrix")
    input_data = _prepare_input_model_matrix(sample, target, variables, add_na)
    all_data = input_data["all_data"]
    sample_n = input_data["sample_n"]

    # Arrange formula
    if formula is None:
        # if no formula is provided, we create an additive formula from available columns
        formula = formula_generator(list(all_data.columns), formula_type="additive")
    if not isinstance(formula, list):
        formula = [formula]
    logger.debug(f"The formula used to build the model matrix: {formula}")
    # If formula is given we rely on patsy formula checker to check it.

    # Arrange penalty factor
    if penalty_factor is None:
        penalty_factor = [1] * len(formula)
    assert len(formula) == len(
        penalty_factor
    ), "penalty factor and formula must have the same length"

    # Arrange factor variables
    if one_hot_encoding:
        factor_variables = list(
            all_data.select_dtypes(["category", "string", "boolean", "object"]).columns
        )
        logger.debug(
            f"These variables will be encoded using one-hot encoding: {factor_variables}"
        )
    else:
        factor_variables = None

    X_matrix = []
    X_matrix_columns = []
    pf = []
    for idx, formula_item in enumerate(formula):
        logger.debug(f"Building model matrix for formula item {formula_item}")

        model_matrix_result = build_model_matrix(
            all_data,
            formula_item,
            factor_variables=factor_variables,
            return_sparse=(return_var_type == "sparse"),
        )
        X_matrix_columns = (
            X_matrix_columns + model_matrix_result["model_matrix_columns"]
        )
        X_matrix.append(model_matrix_result["model_matrix"])
        pf.append(
            np.repeat(
                penalty_factor[idx],
                model_matrix_result["model_matrix"].shape[1],
                axis=0,
            )
        )

    penalty_factor_updated = np.concatenate(pf, axis=0)
    if return_var_type == "sparse":
        X_matrix = hstack(X_matrix, format="csc")
    elif return_var_type == "matrix":
        X_matrix = pd.concat(X_matrix, axis=1).values
    else:
        X_matrix = pd.concat(X_matrix, axis=1)
    logger.debug("The number of columns in the model matrix: {X_matrix.shape[1]}")
    logger.debug("The number of rows in the model matrix: {X_matrix.shape[0]}")

    result = {
        "model_matrix_columns_names": X_matrix_columns,
        "penalty_factor": penalty_factor_updated,
        "formula": formula,
    }
    if return_type == "one":
        result["model_matrix"] = X_matrix
    elif return_type == "two":
        sample_matrix = X_matrix[0:sample_n]
        if target is None:
            target_matrix = None
        else:
            target_matrix = X_matrix[sample_n:]
        result["sample"] = sample_matrix
        result["target"] = target_matrix

    logger.debug("Finished building the model matrix")
    return result


# TODO: add type hinting
def qcut(s, q, duplicates: str = "drop", **kwargs):
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
    if s.shape[0] < q:
        logger.warning("Not quantizing, too few values")
        return s
    else:
        return pd.qcut(s, q, duplicates=duplicates, **kwargs).astype("O")


# TODO: fix it so that the order of the returned columns is the same as the original order in the DataFrame
def quantize(
    df: Union[pd.DataFrame, pd.Series], q: int = 10, variables=None
) -> pd.DataFrame:
    """Cut numeric variables of a DataFrame into quantiles buckets

    Args:
        df (Union[pd.DataFrame, pd.Series]): a DataFrame to transform
        q (int, optional): Number of buckets to create for each variable. Defaults to 10.
        variables (optional): variables to transform.
                    If None, all numeric variables are transformed. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame after quantization. numpy.nan values are kept as is.

    Examples:
        ::

            from balance.util import quantize
            import numpy as np

            df = pd.DataFrame({"a": [1,1,2,20,22,23,np.nan], "b": range(7), "c": range(7), "d": [1,1,np.nan,20,5,23,np.nan]})
            print(quantize(df, q = 3))

                #             b               d              c                a
                # 0  (-0.001, 2.0]  (0.999, 2.333]  (-0.001, 2.0]   (0.999, 1.667]
                # 1  (-0.001, 2.0]  (0.999, 2.333]  (-0.001, 2.0]   (0.999, 1.667]
                # 2  (-0.001, 2.0]             NaN  (-0.001, 2.0]  (1.667, 20.667]
                # 3     (2.0, 4.0]    (15.0, 23.0]     (2.0, 4.0]  (1.667, 20.667]
                # 4     (2.0, 4.0]   (2.333, 15.0]     (2.0, 4.0]   (20.667, 23.0]
                # 5     (4.0, 6.0]    (15.0, 23.0]     (4.0, 6.0]   (20.667, 23.0]
                # 6     (4.0, 6.0]             NaN     (4.0, 6.0]              NaN
    """
    if not (isinstance(df, pd.Series) or isinstance(df, pd.DataFrame)):
        # Necessary because pandas calls the function on the first item on its own
        #  https://stackoverflow.com/questions/21635915/
        df = pd.Series(df)

    # TODO: change assert to raise
    if isinstance(df, pd.Series):
        assert is_numeric_dtype(df.dtype), "series must be numeric"
        return qcut(df, q, duplicates="drop")

    assert isinstance(df, pd.DataFrame)

    variables = choose_variables(df, variables=variables)
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)

    variables = [v for v in variables if v in numeric_columns]

    transformed_data = df.loc[:, variables].transform(
        lambda c: qcut(c, q, duplicates="drop")
    )
    untransformed_columns = df.columns.difference(variables)
    transformed_data = pd.concat(
        (df.loc[:, untransformed_columns], transformed_data), axis=1
    )
    return transformed_data


def row_pairwise_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the differences between every pair of rows of df

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with differences between all combinations of rows

    Examples:
        ::

            d = pd.DataFrame({"a": (1, 2, 3), "b": (-42, 8, 2)})
            row_pairwise_diffs(d)
                #        a   b
                # 0      1 -42
                # 1      2   8
                # 2      3   2
                # 1 - 0  1  50
                # 2 - 0  2  44
                # 2 - 1  1  -6
    """
    c = combinations(sorted(df.index), 2)
    diffs = []
    for j, i in c:
        d = df.loc[i] - df.loc[j]
        d = d.to_frame().transpose().assign(source=f"{i} - {j}").set_index("source")
        diffs.append(d)
    return pd.concat([df] + diffs)


def _is_arraylike(o) -> bool:
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
    data: Union[pd.Series, pd.DataFrame], to_replace=None, value=None
) -> Union[pd.Series, pd.DataFrame]:
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
    data: Union[pd.Series, pd.DataFrame], value=None
) -> Union[pd.Series, pd.DataFrame]:
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
    data: pd.DataFrame, groupby_cols: Union[str, List[str]], apply_func
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


def _safe_show_legend(axis):
    """
    Helper function to safely show legend only if there are labeled artists,
    avoiding matplotlib UserWarning about no artists with labels.

    Args:
        axis: matplotlib axis object
    """
    _, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend()


def _safe_divide_with_zero_handling(numerator, denominator):
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


def rm_mutual_nas(*args) -> List:
    """
    Remove entries in a position which is na or infinite in any of the arguments.

    Ignores args which are None.

    Can accept multiple array-like arguments or a single array-like argument. Handles pandas and numpy arrays.

    Raises:
        ValueError: If any argument is not array-like. (see: :func:`_is_arraylike`)
        ValueError: If arguments include arrays of different lengths.

    Returns:
        List: A list containing the original input arrays, after removing elements that have a missing or infinite value in the same position as any of the other arrays.

    Examples:
        ::

            import pandas as pd
            import numpy as np

            x1 = pd.array([1,2, None, np.nan, pd.NA, 3])
            x2 = pd.array([1.1,2,3, None, np.nan, pd.NA])
            x3 = pd.array([1.1,2,3, 4,5,6])
            x4 = pd.array(["1.1",2,3, None, np.nan, pd.NA])
            x5 = pd.array(["1.1","2","3", None, np.nan, pd.NA], dtype = "string")
            x6 = np.array([1,2,3.3,4,5,6])
            x7 = np.array([1,2,3.3,4,"5","6"])
            x8 = [1,2,3.3,4,"5","6"]
            (x1,x2, x3, x4, x5, x6, x7, x8)
                # (<IntegerArray>
                # [1, 2, <NA>, <NA>, <NA>, 3]
                # Length: 6, dtype: Int64,
                # <PandasArray>
                # [1.1, 2, 3, None, nan, <NA>]
                # Length: 6, dtype: object,
                # <PandasArray>
                # [1.1, 2.0, 3.0, 4.0, 5.0, 6.0]
                # Length: 6, dtype: float64,
                # <PandasArray>
                # ['1.1', 2, 3, None, nan, <NA>]
                # Length: 6, dtype: object,
                # <StringArray>
                # ['1.1', '2', '3', <NA>, <NA>, <NA>]
                # Length: 6, dtype: string,
                # array([1. , 2. , 3.3, 4. , 5. , 6. ]),
                # array(['1', '2', '3.3', '4', '5', '6'], dtype='<U32'),
                # [1, 2, 3.3, 4, '5', '6'])

            from balance.util import rm_mutual_nas
            rm_mutual_nas(x1,x2, x3, x4, x5,x6,x7,x8)
                # [<IntegerArray>
                #  [1, 2]
                #  Length: 2, dtype: Int64,
                #  <PandasArray>
                #  [1.1, 2]
                #  Length: 2, dtype: object,
                #  <PandasArray>
                #  [1.1, 2.0]
                #  Length: 2, dtype: float64,
                #  <PandasArray>
                #  ['1.1', 2]
                #  Length: 2, dtype: object,
                #  <StringArray>
                #  ['1.1', '2']
                #  Length: 2, dtype: string,
                #  array([1., 2.]),
                #  array(['1', '2'], dtype='<U32'),
                #  [1, 2]]

                # Preserve the index values in the resulting pd.Series:
                x1 = pd.Series([1, 2, 3, 4])
                x2 = pd.Series([np.nan, 2, 3, 4])
                x3 = np.array([1, 2, 3, 4])
                print(rm_mutual_nas(x1, x2)[0])
                print(rm_mutual_nas(x1.sort_values(ascending=False), x2)[0])
                print(rm_mutual_nas(x1, x3)[0])
                    # 1    2
                    # 2    3
                    # 3    4
                    # dtype: int64
                    # 3    4
                    # 2    3
                    # 1    2
                    # dtype: int64
                    # 0    1
                    # 1    2
                    # 2    3
                    # 3    4
                    # dtype: int64
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

    def _return_type_creation_function(x: Any) -> Union[Callable, Any]:
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
    *dfs: Union[pd.DataFrame, Any],
    variables: Optional[Union[List, set]] = None,
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
         *dfs (Union[pd.DataFrame, Any]): One or more pandas.DataFrames or balance.Samples.
         variables (Optional[Union[List, set]]): The variables to choose from. If None, returns all joint variables found
             in the input dataframes. Defaults to None.
         df_for_var_order (int): Index of the dataframe used to determine the order of the variables in the output list.
             Defaults to 0. This is used only if the `variables` argument is not a list (e.g.: a set or None).

     Raises:
         ValueError: If one or more requested variables are not present in all dataframes.

     Returns:
         List[str]: A list of the joint variables present in all dataframes and in the `variables` set or list, ordered
             based on the input conditions specified.

     Examples:
        ::
            import pandas as pd
            from balance.util import choose_variables

            df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'E': [1,1], 'F': [1,1]})
            df2 = pd.DataFrame({'C': [7, 8], 'J': [9, 10], 'B': [11, 12], 'K': [1,1], 'A': [1,1]})

            print(choose_variables(df1, df2))
            print(choose_variables(df1, df2,df_for_var_order = 1))
            print(choose_variables(df1, df2,variables=["B", "A"]))

                # WARNING (2023-04-02 10:12:01,337) [util/choose_variables (line 1206)]: Ignoring variables not present in all Samples: {'K', 'F', 'E', 'J'}
                # WARNING (2023-04-02 10:12:01,338) [util/choose_variables (line 1206)]: Ignoring variables not present in all Samples: {'K', 'F', 'E', 'J'}
                # WARNING (2023-04-02 10:12:01,340) [util/choose_variables (line 1206)]: Ignoring variables not present in all Samples: {'K', 'F', 'E', 'J'}
                # ['A', 'B', 'C']
                # ['C', 'B', 'A']
                # ['B', 'A']
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
                "These variables are not included in the dataframes: {variables_not_in_df}"
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


def auto_spread(
    data: pd.DataFrame, features: Optional[list] = None, id_: str = "id"
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
    aggfunc: Union[str, Callable] = "sum",
) -> pd.DataFrame:
    # The default aggregation function is a lambda around sum(x), because as of
    # Pandas 0.22.0, Series.sum of an all-na Series is 0, not nan

    if features is not None:
        warnings.warn(
            "features argument is unused, it will be removed in the future",
            warnings.DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(aggfunc, str):
        if aggfunc == "sum":

            def _f(x):
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

    Examples:
        ::

            from balance.util import fct_lump
            import pandas as pd

            s = pd.Series(['a','a','b','b','c','a','b'], dtype = 'category')
            fct_lump(s, 0.25)
                # 0                a
                # 1                a
                # 2                b
                # 3                b
                # 4    _lumped_other
                # 5                a
                # 6                b
                # dtype: object
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
    props.index = props.index.infer_objects(copy=False)

    small_categories = props[props < prop].index.tolist()

    remainder_category_name = "_lumped_other"
    while remainder_category_name in props.index:
        remainder_category_name = remainder_category_name * 2

    # Convert to object dtype
    s = s.astype("object")

    # Replace small categories with the remainder category name
    s.loc[s.apply(lambda x: x in small_categories)] = remainder_category_name
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

    Examples:
        ::

            s = pd.Series([1,1,1,2,3,1,2])
            by = pd.Series(['a','a','a','a','a','b','b'])
            fct_lump_by(s, by, 0.5)
                # 0                1
                # 1                1
                # 2                1
                # 3    _lumped_other
                # 4    _lumped_other
                # 5                1
                # 6                2
                # dtype: object
    """
    res = copy.deepcopy(s)
    pd.options.mode.copy_on_write = True
    # pandas groupby doesnt preserve order
    for subgroup in pd.unique(by):
        mask = by == subgroup
        grouped_res = fct_lump(res.loc[mask], prop=prop)
        # Ensure dtype compatibility before assignment
        res = res.astype("object")
        res.loc[mask] = grouped_res
    return res


# TODO: add tests
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


# TODO: add tests
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

    Examples:
        ::

            l = [1,2,3,4,5,6,7]
            items = [2,7]
            find_items_index_in_list(l, items)
                # [1, 6]

            items = [1000]
            find_items_index_in_list(l, items)
                # []

            l = ["a", "b", "c"]
            items = ["c", "c", "a"]
            find_items_index_in_list(l, items)
                # [2, 2, 0]

            type(find_items_index_in_list(l, items)[0])
                # int
    """
    # TODO: checking that i is in set each time is expensive -
    #       there are probably faster ways to do it.
    return [a_list.index(i) for i in items if i in set(a_list)]


# TODO: add tests
def get_items_from_list_via_indices(a_list: List[Any], indices: List[int]) -> List[Any]:
    """Gets a subset of items from a list via indices

    Source code (there doesn't seem to be a better solution): https://stackoverflow.com/a/6632209

    Args:
        a_list (List[Any]): a list of items to extract a list from
        indices (List[int]): a list of indexes of items to get

    Returns:
        List[Any]: a list of extracted items

    Examples:
        ::

            l = ["a", "b", "c", "d"]
            get_items_from_list_via_indices(l, [2, 0])
                # ['c', 'a']

            get_items_from_list_via_indices(l, [100])
                # IndexError
    """
    return [a_list[i] for i in indices]


################################################################################
#  logging
################################################################################


def _truncate_text(s: str, length: int) -> str:
    """Truncate string s to be of length 'length'. If the length of s is larger than 'length', then the
    function will add '...' at the end of the truncated text.

    Args:
        s (str):
        length (int):

    Returns:
        str:
    """

    return s[:length] + "..." * (len(s) > length)


class TruncationFormatter(logging.Formatter):
    """
    Logging formatter which truncates the logged message to 500 characters.

    This is useful in the cases where the logging message includes objects
    --- like DataFrames --- whose string representation is very long.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TruncationFormatter, self).__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord):
        result = super(TruncationFormatter, self).format(record)
        return _truncate_text(result, 2000)


################################################################################
#  File handling
################################################################################
def _to_download(
    df: pd.DataFrame,
    tempdir: Optional[str] = None,
) -> FileLink:
    """Creates a downloadable link of the DataFrame (df).

    File name starts with tmp_balance_out_, and some random file name (using :func:`uuid.uuid4`).

    Args:
        self (BalanceDF): Object.
        tempdir (Optional[str], optional): Defaults to None (which then uses a temporary folder using :func:`tempfile.gettempdir`).

    Returns:
        FileLink: Embedding a local file link in an IPython session, based on path. Using :func:FileLink.
    """
    if tempdir is None:
        tempdir = tempfile.gettempdir()
    path = f"{tempdir}/tmp_balance_out_{uuid.uuid4()}.csv"

    df.to_csv(path_or_buf=path, index=False)
    return FileLink(path, result_html_prefix="Click here to download: ")


################################################################################
#  pandas utils
################################################################################


def _dict_intersect(d: Dict, d_for_keys: Dict) -> Dict:
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


def _true_false_str_to_bool(x: str) -> bool:
    """Changes strings such as 'false' to False and 'true' to True.

    Args:
        x (str): String to be converted (ideally 'true' or 'false' - case is ignored).

    Raises:
        ValueError: If x is not 'true' or 'false'.

    Returns:
        bool: True if x is 'true', False if x is 'false'.

    Examples:
        ::
            _true_false_str_to_bool('falsE')  # False
            _true_false_str_to_bool('TrUe')  # True

            _true_false_str_to_bool('Banana')
            # ValueError: Banana is not an accepted value, please pass either 'True' or 'False' (lower/upper case is ignored)
    """
    if x.lower() == "false":
        return False
    elif x.lower() == "true":
        return True
    else:
        raise ValueError(
            f"{x} is not an accepted value, please pass either 'True' or 'False' (lower/upper case is ignored)"
        )


def _are_dtypes_equal(
    dt1: pd.Series, dt2: pd.Series
) -> Dict[str, Union[bool, pd.Series, set]]:
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
