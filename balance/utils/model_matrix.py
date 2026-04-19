# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import re
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from balance.utils.data_transformation import (
    add_na_indicator,
    add_na_indicator_to_combined,
    NA_INDICATOR_TOKEN_PATTERN,
)
from balance.utils.input_validation import (
    _assert_type,
    _isinstance_sample,
    choose_variables,
)
from balance.utils.pandas_utils import _make_df_column_names_unique
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from patsy.contrasts import ContrastMatrix
from patsy.highlevel import dmatrix, ModelDesc
from scipy.sparse import csc_matrix, diags, hstack, spmatrix
from sklearn.preprocessing import StandardScaler

logger: logging.Logger = logging.getLogger(__package__)


def _ordered_unique_missing_columns(
    requested_columns: list[str], existing_columns: pd.Index
) -> list[str]:
    """Return missing requested columns (deduplicated, original order preserved)."""
    existing_set = set(existing_columns)
    return [col for col in dict.fromkeys(requested_columns) if col not in existing_set]


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
        raise ValueError(
            "This formula type is not supported.'Please provide a string formula"
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
        raise TypeError(
            "Variables should not be empty. Please provide a list of strings."
        )

    if not isinstance(variables, list):
        raise TypeError(
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
        raise ValueError("Not all factor variables are contained in variables")

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
        raise ValueError(
            "Variable names cannot contain characters '[' or ']'"
            f"because patsy uses them to denote one-hot encoded categoricals: ({bracket_variables})"
        )

    # Check all factor variables are in variables:
    if factor_variables is not None:
        if not (set(factor_variables) <= set(variables)):
            raise ValueError("Not all factor variables are contained in df")

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


def _concat_frames(
    sample_df: pd.DataFrame, target_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Return a combined DataFrame from sample/target, skipping empty inputs.

    Args:
        sample_df: The sample DataFrame (must be non-empty).
        target_df: The optional target DataFrame.

    Returns:
        A DataFrame containing the concatenated rows or a copy of the single
        non-empty frame.
    """
    frames = [df for df in (sample_df, target_df) if df is not None and not df.empty]
    if len(frames) == 1:
        return frames[0].copy()
    return pd.concat(frames)


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
        sample (pd.DataFrame | Any): Input sample data as either a
            ``pandas.DataFrame`` or a ``Sample`` object from
            ``balance.sample_class`` (recognized via ``_isinstance_sample``).
        target (pd.DataFrame | Any | None, optional): Optional target data as
            either a ``DataFrame`` or a ``Sample`` object. If provided, the
            model-matrix inputs are prepared from a sample/target union of
            variables and rows. Defaults to None.
        variables (List[str] | None, optional): Variables to use from both
            inputs. If provided, `choose_variables` validates that each
            requested variable exists in both sample and target (when target is
            supplied), otherwise it raises ``ValueError``. For ``Sample``
            inputs, this validation/inference is based on covariate names
            (``sample.covars().names()``), not all raw ``._df`` columns. If
            None, variables are inferred by `choose_variables`.
        add_na (bool, optional): If True, add NA indicator columns before
            model-matrix creation. If False, drop rows containing missing
            values; this can raise ``ValueError`` if dropping rows empties the
            sample or target. Defaults to True.
        fix_columns_names (bool, optional): Whether to sanitize column names by
            replacing non-word characters with ``_`` and making duplicate names
            unique. Defaults to True.

    Raises:
        ValueError: If requested ``variables`` are not present in the
            provided input frame(s) (and in both sample and target when target
            is supplied), if variables contain ``[`` or ``]``, or if
            ``add_na=False`` drops all rows from sample/target, or if
            sample has zero rows.

    Returns:
        Dict[str, Any]: returns a dictionary containing two keys: 'all_data' and 'sample_n'.
            The 'all_data' is a pd.DataFrame with all the rows of 'sample' (including 'target', if supplied)
            The'sample_n' is the number of rows in the first input DataFrame ('sample').
    """
    variables = choose_variables(sample, target, variables=variables)

    bracket_variables = [v for v in variables if ("[" in v) or ("]" in v)]
    if len(bracket_variables) > 0:
        raise ValueError(
            "Variable names cannot contain characters '[' or ']'"
            f"because patsy uses them to denote one-hot encoded categoricals: ({bracket_variables})"
        )

    if _isinstance_sample(sample):
        sample_df = sample._df
    else:
        sample_df = sample
    if sample_df.shape[0] == 0:
        raise ValueError("sample must have more than zero rows")
    # NOTE: .copy() not needed as it is copied anyway in _concat_frames
    sample_n = sample_df.shape[0]
    sample_df = sample_df.loc[:, variables]

    if target is None:
        target_df = None
    elif _isinstance_sample(target):
        target_df = target._df.loc[:, variables]
    else:
        target_df = target.loc[:, variables]

    if add_na:
        # Build a combined frame so NA indicators reflect sample/target union
        # (target-only missingness should still add NA indicator columns).
        all_data = _concat_frames(sample_df, target_df)
        all_data = add_na_indicator(all_data)
    else:
        logger.warning("Dropping all rows with NAs")
        target_was_all_na = False
        if target_df is not None and target_df.dropna(how="all").empty:
            target_was_all_na = True
            target_df = None
        all_data = _concat_frames(sample_df, target_df)
        if target_was_all_na:
            raise ValueError(
                "Dropping rows led to empty target. Consider using add_na=True to add "
                "NA indicator columns instead of dropping rows."
            )
        category_levels: Dict[str, List[Any]] = {}
        for column in all_data.columns:
            column_series = all_data[column]
            if isinstance(column_series.dtype, pd.CategoricalDtype):
                category_levels[column] = list(column_series.cat.categories)
            elif is_object_dtype(column_series) or is_string_dtype(column_series):
                category_levels[column] = list(column_series.dropna().unique())

        sample_df = sample_df.dropna()
        if sample_df.empty:
            raise ValueError(
                "Dropping rows led to empty sample. Consider using add_na=True to add "
                "NA indicator columns instead of dropping rows."
            )
        sample_n = sample_df.shape[0]
        if target_df is not None:
            target_df = target_df.dropna()
            if target_df.empty:
                raise ValueError(
                    "Dropping rows led to empty target. Consider using add_na=True to add "
                    "NA indicator columns instead of dropping rows."
                )
        if category_levels:
            for column, levels in category_levels.items():
                if column in sample_df.columns:
                    sample_df = sample_df.assign(
                        **{column: pd.Categorical(sample_df[column], categories=levels)}
                    )
                if target_df is not None and column in target_df.columns:
                    target_df = target_df.assign(
                        **{column: pd.Categorical(target_df[column], categories=levels)}
                    )
        all_data = _concat_frames(sample_df, target_df)

    if fix_columns_names:
        all_data.columns = all_data.columns.str.replace(
            r"[^\w]", "_", regex=True
        ).infer_objects()
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


# ---------------------------------------------------------------------------
# Shared design-matrix builder for IPW (training & holdout paths)
# ---------------------------------------------------------------------------


# NOTE: Use typing.Tuple/Optional here (not builtin tuple/|) because this is
# a module-level type alias evaluated at runtime. `from __future__ import
# annotations` only defers evaluation inside function signatures, not here.
# Python 3.9 raises TypeError for `list[float] | None` at module level.
_MatrixBuildResult = Tuple[
    Union[pd.DataFrame, np.ndarray, csc_matrix],  # combined_matrix
    List[str],  # columns
    Optional[List[float]],  # penalty_factor_expanded
    Optional[Union[str, List[str]]],  # resolved_formula
]


def _build_projected_model_matrix(
    sample_covars: pd.DataFrame,
    target_covars: pd.DataFrame,
    *,
    formula: str | list[str] | None,
    one_hot_encoding: bool,
    na_action: str,
    project_to_columns: list[str],
) -> _MatrixBuildResult:
    """Holdout path: build model matrix and project to stored fit-time columns."""
    if na_action == "drop":
        raise ValueError(
            "Recomputing stored IPW artifacts for design_matrix/predict_proba is "
            "unsupported when na_action='drop' because row dropping can "
            "change sample/target boundaries. Re-fit with "
            "na_action='add_indicator' or score using matching fit-time rows."
        )

    # Pre-add NA indicators so the model matrix sees the same columns
    # that were present at fit time.
    combined = pd.concat((sample_covars, target_covars), axis=0)
    combined = add_na_indicator_to_combined(combined)
    formula_items: List[str] = (
        [formula]
        if isinstance(formula, str)
        else formula if isinstance(formula, list) else []
    )
    expected_na_indicators: set[str] = {
        token
        for formula_item in formula_items
        for token in re.findall(NA_INDICATOR_TOKEN_PATTERN, formula_item)
    }
    for indicator in expected_na_indicators:
        if indicator not in combined.columns:
            combined[indicator] = False

    model_matrix_out = model_matrix(
        combined,
        return_type="one",
        return_var_type="sparse",
        formula=formula,
        one_hot_encoding=one_hot_encoding,
        add_na=False,
    )
    sparse_matrix: csc_matrix = _assert_type(
        model_matrix_out["model_matrix"], csc_matrix
    )
    sparse_columns: list[str] = _assert_type(
        model_matrix_out["model_matrix_columns_names"], list
    )

    # Project to the stored column set.
    sparse_col_to_idx = {col: i for i, col in enumerate(sparse_columns)}
    target_idx = np.array(
        [sparse_col_to_idx.get(col, -1) for col in project_to_columns]
    )
    present_mask = target_idx >= 0
    if present_mask.any():
        projection = csc_matrix(
            (
                np.ones(int(present_mask.sum()), dtype=float),
                (target_idx[present_mask], np.flatnonzero(present_mask)),
            ),
            shape=(sparse_matrix.shape[1], len(project_to_columns)),
        )
        combined_matrix: Union[pd.DataFrame, np.ndarray, csc_matrix] = (
            sparse_matrix @ projection
        )
    else:
        combined_matrix = csc_matrix((sparse_matrix.shape[0], len(project_to_columns)))

    # In the projection path, penalty_factor_expanded is not used downstream
    # (fit_penalties_skl is applied separately), but we return a correctly-
    # sized vector for consistency.
    penalty_factor_expanded: Optional[List[float]] = [1.0] * len(project_to_columns)
    resolved_formula: str | list[str] | None = cast(
        Optional[Union[str, List[str]]], model_matrix_out.get("formula")
    )
    return (
        combined_matrix,
        project_to_columns,
        penalty_factor_expanded,
        resolved_formula,
    )


def _build_training_model_matrix(
    sample_covars: pd.DataFrame,
    target_covars: pd.DataFrame,
    *,
    formula: str | list[str] | None,
    one_hot_encoding: bool,
    na_action: str,
    penalty_factor: list[float] | None,
) -> _MatrixBuildResult:
    """Training path: build model matrix via formula expansion."""
    formula_list: list[str] | None = [formula] if isinstance(formula, str) else formula
    model_matrix_output = model_matrix(
        sample_covars,
        target_covars,
        add_na=(na_action == "add_indicator"),
        return_type="one",
        return_var_type="sparse",
        formula=formula_list,
        penalty_factor=penalty_factor,
        one_hot_encoding=one_hot_encoding,
    )
    combined_matrix: Union[pd.DataFrame, np.ndarray, csc_matrix] = cast(
        Union[pd.DataFrame, np.ndarray, csc_matrix],
        model_matrix_output["model_matrix"],
    )
    # `_assert_type(..., list)` narrows to `list` without element typing, so
    # cast to `List[str]` to preserve the expected element type for downstream
    # callers and for strict Pyre `len()`/iteration checks.
    columns: list[str] = cast(
        List[str],
        _assert_type(model_matrix_output["model_matrix_columns_names"], list),
    )
    # model_matrix() stores `penalty_factor` as a 1D np.ndarray of floats; fall
    # back to a uniform vector when the key is absent. Narrowing via cast is
    # required because `model_matrix_output` has a broad union value type.
    penalty_factor_raw = model_matrix_output.get("penalty_factor")
    penalty_factor_expanded: Optional[List[float]] = (
        [1.0] * len(columns)
        if penalty_factor_raw is None
        else list(cast(np.ndarray, penalty_factor_raw))
    )
    resolved_formula: str | list[str] | None = cast(
        Optional[Union[str, List[str]]], model_matrix_output.get("formula")
    )
    return combined_matrix, columns, penalty_factor_expanded, resolved_formula


def _build_raw_covariates(
    sample_covars: pd.DataFrame,
    target_covars: pd.DataFrame,
    *,
    na_action: str,
) -> _MatrixBuildResult:
    """Raw covariates path: concat DataFrames without model matrix encoding."""
    combined_matrix = pd.concat((sample_covars, target_covars), axis=0)
    if na_action == "add_indicator":
        combined_matrix = add_na_indicator_to_combined(combined_matrix)

    categorical_cols = combined_matrix.select_dtypes(
        ["string", "boolean", "object"]
    ).columns
    for col in categorical_cols:
        combined_matrix[col] = pd.Categorical(combined_matrix[col])

    columns = combined_matrix.columns.tolist()
    return combined_matrix, columns, [1.0] * len(columns), None


def build_design_matrix(
    sample_covars: pd.DataFrame,
    target_covars: pd.DataFrame,
    *,
    use_model_matrix: bool = True,
    formula: str | list[str] | None = None,
    one_hot_encoding: bool = False,
    na_action: str = "add_indicator",
    penalty_factor: list[float] | None = None,
    project_to_columns: list[str] | None = None,
    fit_scaler: StandardScaler | None = None,
    scaler_weights: np.ndarray | None = None,
    fit_penalties_skl: list[float] | None = None,
    matrix_type: str | None = None,
) -> dict[str, Any]:
    """Build the combined design matrix used by IPW weighting.

    This function consolidates the preprocessing pipeline shared by the
    ``ipw()`` training path and the ``_compute_ipw_matrices()``
    holdout path.  It performs (in order):

    1. Model-matrix construction (formula expansion, one-hot encoding, NA
       indicators) **or** raw-covariate passthrough when
       ``use_model_matrix=False``.
    2. Optional column projection to align holdout matrices with columns
       stored at fit time (``project_to_columns``).
    3. Optional standardisation via :class:`~sklearn.preprocessing.StandardScaler`.
    4. Optional per-column penalty rescaling.
    5. Conversion to the requested matrix type (sparse / dense / dataframe).

    Parameters
    ----------
    sample_covars:
        Sample covariate DataFrame (rows = sample units).
    target_covars:
        Target covariate DataFrame (rows = target units).
    use_model_matrix:
        If ``True`` (default), construct a model matrix via
        :func:`model_matrix`.  If ``False``, concatenate the raw DataFrames
        and treat categorical columns as ``pd.Categorical``.
    formula:
        Formula string(s) forwarded to :func:`model_matrix`.  Ignored when
        ``use_model_matrix=False``.
    one_hot_encoding:
        Whether to apply one-hot encoding for categorical variables.
        Ignored when ``use_model_matrix=False``.
    na_action:
        How to handle missing values.  ``"add_indicator"`` (default) adds
        boolean NA-indicator columns; ``"drop"`` is **not** supported in the
        holdout path and will raise ``ValueError`` when
        ``project_to_columns`` is set.
    penalty_factor:
        Per-formula penalty factors forwarded to :func:`model_matrix`.
        Ignored when ``use_model_matrix=False``.
    project_to_columns:
        When provided (holdout path), the model-matrix columns are projected
        / reindexed to match this list.  Missing columns are filled with
        zeros.
    fit_scaler:
        An already-fitted ``StandardScaler``.  When provided, the function
        calls ``.transform()`` (holdout path).  When ``None`` **and**
        ``scaler_weights`` is provided, a *new* scaler is created and
        ``fit_transform()`` is called (training path).
    scaler_weights:
        Sample weights passed to ``StandardScaler.fit_transform()`` during
        training.  Ignored when ``fit_scaler`` is provided.
    fit_penalties_skl:
        Pre-computed per-column penalty multipliers.  Each column *i* of the
        matrix is multiplied by ``fit_penalties_skl[i]``.
    matrix_type:
        Desired output type: ``"sparse"`` (:class:`csc_matrix`) or
        ``"dense"`` (:class:`numpy.ndarray`).  ``None`` leaves the
        matrix as-is (may be sparse, dense, or DataFrame depending on
        the path).

    Returns
    -------
    dict[str, Any]
        ``"combined_matrix"`` -- the preprocessed combined (sample + target)
        matrix.

        ``"sample_n"`` -- number of sample rows (for splitting).

        ``"columns"`` -- column names of the model matrix.

        ``"penalty_factor_expanded"`` -- expanded penalty factor (one entry
        per column), or ``None``.

        ``"resolved_formula"`` -- the formula(s) actually used, or ``None``.

        ``"fit_scaler"`` -- the ``StandardScaler`` that was used (either the
        one passed in or the newly created one), or ``None``.

        ``"fit_penalties_skl"`` -- the per-column penalty multipliers that
        were applied, or ``None``.
    """
    sample_n = sample_covars.shape[0]

    # -- Step 1: Build the combined matrix via one of three paths ----------
    if use_model_matrix and project_to_columns is not None:
        combined_matrix, columns, penalty_factor_expanded, resolved_formula = (
            _build_projected_model_matrix(
                sample_covars,
                target_covars,
                formula=formula,
                one_hot_encoding=one_hot_encoding,
                na_action=na_action,
                project_to_columns=project_to_columns,
            )
        )
    elif use_model_matrix:
        combined_matrix, columns, penalty_factor_expanded, resolved_formula = (
            _build_training_model_matrix(
                sample_covars,
                target_covars,
                formula=formula,
                one_hot_encoding=one_hot_encoding,
                na_action=na_action,
                penalty_factor=penalty_factor,
            )
        )
    else:
        combined_matrix, columns, penalty_factor_expanded, resolved_formula = (
            _build_raw_covariates(
                sample_covars,
                target_covars,
                na_action=na_action,
            )
        )

    # -- Step 2: Reindex DataFrame to stored columns (raw holdout path) ----
    if isinstance(combined_matrix, pd.DataFrame) and project_to_columns is not None:
        missing_projection_columns = _ordered_unique_missing_columns(
            project_to_columns, combined_matrix.columns
        )
        if missing_projection_columns:
            logger.warning(
                "build_design_matrix: holdout data is missing %d fit-time "
                "column(s); zero-filling unseen columns: %s",
                len(missing_projection_columns),
                missing_projection_columns,
            )
        combined_matrix = combined_matrix.reindex(
            columns=project_to_columns, fill_value=0
        )
        columns = project_to_columns

    # -- Step 3: Scaler ----------------------------------------------------
    out_scaler: StandardScaler | None = fit_scaler
    if fit_scaler is not None:
        combined_matrix = fit_scaler.transform(combined_matrix)
    elif scaler_weights is not None:
        scaler = StandardScaler(with_mean=False)
        combined_matrix = scaler.fit_transform(
            combined_matrix, sample_weight=scaler_weights
        )
        out_scaler = scaler

    # -- Step 4: Penalty rescaling -----------------------------------------
    out_penalties: list[float] | None = fit_penalties_skl
    if fit_penalties_skl is not None:
        penalties_arr = np.asarray(fit_penalties_skl, dtype=float)
        n_cols = combined_matrix.shape[1] if hasattr(combined_matrix, "shape") else len(combined_matrix.columns)  # type: ignore[union-attr]
        if len(penalties_arr) != n_cols:
            raise ValueError(
                f"fit_penalties_skl length ({len(penalties_arr)}) does not match "
                f"the number of matrix columns ({n_cols})."
            )
        if isinstance(combined_matrix, spmatrix):
            penalty_diag = diags(penalties_arr, format="csc")
            combined_matrix = combined_matrix @ penalty_diag  # pyre-ignore[58]
        else:
            combined_matrix = np.asarray(combined_matrix) * penalties_arr

    # -- Step 5: Matrix type conversion ------------------------------------
    if matrix_type == "dense":
        if isinstance(combined_matrix, spmatrix):
            combined_matrix = combined_matrix.toarray()  # pyre-ignore[16]
        elif isinstance(combined_matrix, pd.DataFrame):
            combined_matrix = np.asarray(combined_matrix)
    elif matrix_type == "sparse":
        if isinstance(combined_matrix, np.ndarray):
            combined_matrix = csc_matrix(combined_matrix)
        elif isinstance(combined_matrix, pd.DataFrame):
            combined_matrix = csc_matrix(np.asarray(combined_matrix))

    return {
        "combined_matrix": combined_matrix,
        "sample_n": sample_n,
        "columns": columns,
        "penalty_factor_expanded": penalty_factor_expanded,
        "resolved_formula": resolved_formula,
        "fit_scaler": out_scaler,
        "fit_penalties_skl": out_penalties,
    }
