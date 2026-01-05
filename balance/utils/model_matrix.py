# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from patsy.contrasts import ContrastMatrix
from patsy.highlevel import ModelDesc, dmatrix
from scipy.sparse import csc_matrix, hstack

from balance.utils.data_transformation import add_na_indicator
from balance.utils.input_validation import choose_variables, _isinstance_sample
from balance.utils.pandas_utils import _make_df_column_names_unique

logger: logging.Logger = logging.getLogger(__package__)


def formula_generator(variables: List[str], formula_type: str = "additive") -> str:
    """Create formula to build the model matrix
        Default is additive formula.
    Args:
        variables: list with names of variables (as strings) to combine into a formula
        formula_type (str, optional): how to construct the formula. Currently only "additive" is supported. Defaults to "additive".

    Raises:
        Exception: "This formula type is not supported.'" "Please provide a string formula"

    Returns:
        str: A string representing the formula
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


def dot_expansion(formula: str, variables: List[str]) -> str:
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
    """

    def __init__(self, reference: int = 0) -> None:
        self.reference = reference

    def code_with_intercept(self, levels: List[Any]) -> ContrastMatrix:
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

    def code_without_intercept(self, levels: List[Any]) -> ContrastMatrix:
        return self.code_with_intercept(levels)


def process_formula(
    formula: str, variables: list[str], factor_variables: list[str] | None = None
) -> ModelDesc:
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
    factor_variables: List[str] | None = None,
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
    sample: pd.DataFrame | Any,
    target: pd.DataFrame | Any | None = None,
    variables: List[str] | None = None,
    add_na: bool = True,
    fix_columns_names: bool = True,
) -> Dict[str, Any]:
    """Helper function to model_matrix. Prepare and check input of sample and target:
        - Choose joint variables to sample and target (or by given variables)
        - Extract sample and target dataframes
        - Concat dataframes together
        - Add na indicator if required.

    Args:
        sample (pd.DataFrame | Any): This can either be a DataFrame or a Sample object. TODO: add text.
        target (pd.DataFrame | Any | None, optional): This can either be a DataFrame or a Sample object.. Defaults to None.
        variables (List[str] | None, optional): Defaults to None. TODO: add text.
        add_na (bool, optional): Defaults to True. TODO: add text.
        fix_columns_names (bool, optional): Defaults to True. If to fix the column names of the DataFrame by changing special characters to '_'.

    Raises:
        Exception: "Variable names cannot contain characters '[' or ']'"

    Returns:
        Dict[str, Any]: returns a dictionary containing two keys: 'all_data' and 'sample_n'.
            The 'all_data' is a pd.DataFrame with all the rows of 'sample' (including 'target', if supplied)
            The'sample_n' is the number of rows in the first input DataFrame ('sample').
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
        # TODO: add code to drop all rows with NAs (columns are left as is)

    if fix_columns_names:
        all_data.columns = all_data.columns.str.replace(
            r"[^\w]", "_", regex=True
        ).infer_objects(copy=False)
        all_data = _make_df_column_names_unique(all_data)

    return {"all_data": all_data, "sample_n": sample_n}


def model_matrix(
    sample: pd.DataFrame | Any,
    target: pd.DataFrame | Any | None = None,
    variables: List[str] | None = None,
    add_na: bool = True,
    return_type: str = "two",
    return_var_type: str = "dataframe",
    formula: str | List[str] | None = None,
    penalty_factor: List[float] | None = None,
    one_hot_encoding: bool = False,
) -> Dict[str, List[Any] | np.ndarray | pd.DataFrame | csc_matrix | None]:
    """Create a model matrix from a sample (and target).
    The default is to use an additive formula for all variables (or the ones specified).
    Can also create a custom model matrix if a formula is provided.
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
