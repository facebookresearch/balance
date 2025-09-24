# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

import logging

import math
from fractions import Fraction

from functools import reduce
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util
from ipfn import ipfn

logger = logging.getLogger(__package__)


# TODO: Add options for only marginal distributions input
def rake(
    sample_df: pd.DataFrame,
    sample_weights: pd.Series,
    target_df: pd.DataFrame,
    target_weights: pd.Series,
    variables: Union[List[str], None] = None,
    transformations: Union[Dict[str, Callable], str] = "default",
    na_action: str = "add_indicator",
    max_iteration: int = 1000,
    convergence_rate: float = 0.0005,
    rate_tolerance: float = 1e-8,
    *args,
    **kwargs,
) -> Dict:
    """
    Perform raking (using the iterative proportional fitting algorithm).
    See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    Returns weights normalised to sum of target weights

    Arguments:
    sample_df --- (pandas dataframe) a dataframe representing the sample.
    sample_weights --- (pandas series) design weights for sample.
    target_df ---  (pandas dataframe) a dataframe representing the target.
    target_weights --- (pandas series) design weights for target.
    variables ---  (list of strings) list of variables to include in the model.
                   If None all joint variables of sample_df and target_df are used.
    transformations --- (dict) what transformations to apply to data before fitting the model.
                               Default is "default" (see apply_transformations function).
    na_action --- (string) what to do with NAs. Default is "add_indicator", which adds NaN as a
                  group (called "__NaN__") for each weighting variable (post-transformation);
                  "drop" removes rows with any missing values on any variable from both sample
                  and target.
    max_iteration --- (int) maximum number of iterations for iterative proportional fitting algorithm
    convergence_rate --- (float) convergence criteria; the maximum difference in proportions between
                          sample and target marginal distribution on any covariate in order
                          for algorithm to converge.
    rate_tolerance --- (float) convergence criteria; if convergence rate does not move more
                               than this amount than the algorithm is also considered to
                               have converged.

    Returns:
    A dictionary including:
    "weight" --- The weights for the sample.
    "model" --- parameters of the model: iterations (dataframe with iteration numbers and
                convergence rate information at all steps), converged (Flag with the output
                status: 0 for failure and 1 for success).
    """
    assert (
        "weight" not in sample_df.columns.values
    ), "weight shouldn't be a name for covariate in the sample data"
    assert (
        "weight" not in target_df.columns.values
    ), "weight shouldn't be a name for covariate in the target data"

    # TODO: move the input checks into separate funnction for rake, ipw, poststratify
    assert isinstance(sample_df, pd.DataFrame), "sample_df must be a pandas DataFrame"
    assert isinstance(target_df, pd.DataFrame), "target_df must be a pandas DataFrame"
    assert isinstance(
        sample_weights, pd.Series
    ), "sample_weights must be a pandas Series"
    assert isinstance(
        target_weights, pd.Series
    ), "target_weights must be a pandas Series"
    assert sample_df.shape[0] == sample_weights.shape[0], (
        "sample_weights must be the same length as sample_df"
        f"{sample_df.shape[0]}, {sample_weights.shape[0]}"
    )
    assert target_df.shape[0] == target_weights.shape[0], (
        "target_weights must be the same length as target_df"
        f"{target_df.shape[0]}, {target_weights.shape[0]}"
    )

    variables = balance_util.choose_variables(sample_df, target_df, variables=variables)

    logger.debug(f"Join variables for sample and target: {variables}")

    sample_df = sample_df.loc[:, variables]
    target_df = target_df.loc[:, variables]

    assert len(variables) > 1, (
        "Must weight on at least two variables for raking. "
        f"Currently have variables={variables} only"
    )

    sample_df, target_df = balance_adjustment.apply_transformations(
        (sample_df, target_df), transformations
    )

    # TODO: separate into a function that handles NA (for rake, ipw, poststratify)
    if na_action == "drop":
        (sample_df, sample_weights) = balance_util.drop_na_rows(
            sample_df, sample_weights, "sample"
        )
        (target_df, target_weights) = balance_util.drop_na_rows(
            target_df, target_weights, "target"
        )
    elif na_action == "add_indicator":
        from balance.util import _safe_fillna_and_infer

        target_df = _safe_fillna_and_infer(target_df, "__NaN__")
        sample_df = _safe_fillna_and_infer(sample_df, "__NaN__")
    else:
        raise ValueError("`na_action` must be 'add_indicator' or 'drop'")

    # Alphabetize variables to ensure consistency across covariate order
    # (ipfn algorithm is iterative and variable order can matter on the margins)
    alphabetized_variables = list(variables)
    alphabetized_variables.sort()

    logger.debug(
        f"Alphabetized variable order is as follows: {alphabetized_variables}."
    )

    # Cast all data types as string to be explicit about each unique value
    # being its own group and to handle that `fillna()` above creates
    # series of type Object, which won't work for the ipfn script
    categories = []
    for variable in alphabetized_variables:
        target_df[variable] = target_df[variable].astype(str)
        sample_df[variable] = sample_df[variable].astype(str)

        sample_var_set = set(sample_df[variable].unique())
        target_var_set = set(target_df[variable].unique())
        sample_over_set = sample_var_set - target_var_set
        target_over_set = target_var_set - sample_var_set
        if len(sample_over_set):
            raise ValueError(
                "All variable levels in sample must be present in target. "
                f"'{variable}' in target is missing these levels: {sample_over_set}."
            )
        if len(target_over_set):
            logger.warning(
                f"'{variable}' has more levels in target than in sample. "
                f"'{variable}' in sample is missing these levels: {target_over_set}. "
                "These levels are treated as if they do not exist for that variable."
            )
        categories.append(sorted(sample_var_set.intersection(target_var_set)))

    logger.info(
        f"Final covariates and levels that will be used in raking: {dict(zip(alphabetized_variables, categories))}."
    )

    target_df = target_df.assign(weight=target_weights)
    sample_df = sample_df.assign(weight=sample_weights)

    sample_sum_weights = sample_df["weight"].sum()
    target_sum_weights = target_df["weight"].sum()

    # Calculate {# covariates}-dimensional array representation of the sample
    # for the ipfn algorithm

    # Create a multi-index DataFrame with all possible combinations
    index = pd.MultiIndex.from_product(categories, names=alphabetized_variables)
    full_df = pd.DataFrame(index=index).reset_index()

    # Group by covariates and sum weights
    grouped_sample = (
        sample_df.groupby(alphabetized_variables)["weight"].sum().reset_index()
    )

    # Merge to ensure all combinations are present (fill missing with 0)
    merged = (
        pd.merge(full_df, grouped_sample, on=alphabetized_variables, how="left")
        .fillna(0)
        .infer_objects(copy=False)
    )

    # Reshape to n-dimensional array
    m_sample = merged["weight"].values.reshape([len(c) for c in categories])

    # Calculate target margins for ipfn
    target_margins = []
    for col, cats in zip(alphabetized_variables, categories):
        sums = (
            target_df.groupby(col)["weight"].sum()
            / target_sum_weights
            * sample_sum_weights
        )
        sums = sums.reindex(cats, fill_value=0)
        target_margins.append(sums.values)
    dimensions = [[i] for i in range(len(alphabetized_variables))]

    logger.debug(
        "Raking algorithm running following settings: "
        f" convergence_rate: {convergence_rate}; max_iteration: {max_iteration}; rate_tolerance: {rate_tolerance}"
    )

    # returns array with joint distribution of covariates and total weight
    # for that specific set of covariates
    # no longer uses the dataframe version of the ipfn algorithm
    # due to incompatability with latest Python versions
    ipfn_obj = ipfn.ipfn(
        original=m_sample,
        aggregates=target_margins,
        dimensions=dimensions,
        convergence_rate=convergence_rate,
        max_iteration=max_iteration,
        verbose=2,
        rate_tolerance=rate_tolerance,
    )
    m_fit, converged, iterations = ipfn_obj.iteration()

    logger.debug(
        f"Raking algorithm terminated with following convergence: {converged}; "
        f"and iteration meta data: {iterations}."
    )

    if not converged:
        logger.warning("Maximum iterations reached, convergence was not achieved")

    # Convert array representation of the weighted sample into a dataframe
    # Generate all possible combinations of categories (cartesian product)
    combos = list(itertools.product(*categories))
    # Flatten the array to match the order of combos
    weights = m_fit.flatten()
    # Build the DataFrame
    fit = pd.DataFrame(combos, columns=alphabetized_variables)
    fit["rake_weight"] = weights

    raked = pd.merge(
        sample_df.reset_index(),
        fit,
        how="left",
        on=alphabetized_variables,
    )

    # Merge back to original sample weights
    raked_rescaled = pd.merge(
        raked,
        (grouped_sample.rename(columns={"weight": "total_survey_weight"})),
        how="left",
        on=alphabetized_variables,
    ).set_index("index")  # important if dropping rows due to NA

    # use above merge to give each individual sample its proportion of the
    # cell's total weight
    raked_rescaled["rake_weight"] = (
        raked_rescaled["rake_weight"] / raked_rescaled["total_survey_weight"]
    )
    # rescale weights to sum to target_sum_weights (sum of initial target weights)
    w = (
        raked_rescaled["rake_weight"] / raked_rescaled["rake_weight"].sum()
    ) * target_sum_weights
    return {
        "weight": w,
        "model": {
            "method": "rake",
            "iterations": iterations,
            "converged": converged,
            "perf": {"prop_dev_explained": np.array([np.nan])},
            # TODO: fix functions that use the perf and remove it from here
        },
    }


def _lcm(a: int, b: int) -> int:
    """
    Calculates the least common multiple (LCM) of two integers.

    The least common multiple (LCM) of two or more numbers is the smallest positive integer that is divisible by each of the given numbers.
    In other words, it is the smallest multiple that the numbers have in common.
    The LCM is useful when you need to find a common denominator for fractions or synchronize repeating events with different intervals.

    For example, let's find the LCM of 4 and 6:

    The multiples of 4 are: 4, 8, 12, 16, 20, ...
    The multiples of 6 are: 6, 12, 18, 24, 30, ...
    The smallest multiple that both numbers have in common is 12, so the LCM of 4 and 6 is 12.

    The calculation is based on the property that the product of two numbers is equal to the product of their LCM and GCD: a * b = LCM(a, b) * GCD(a, b).
    (proof: https://math.stackexchange.com/a/589299/1406)

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The least common multiple of the two integers.
    """
    # NOTE: this function uses math.gcd which calculates the greatest common divisor (GCD) of two integers.
    # The greatest common divisor (GCD) of two or more integers is the largest positive integer that divides each of the given integers without leaving a remainder.
    # In other words, it is the largest common factor of the given integers.
    # For example, the GCD of 12 and 18 is 6, since 6 is the largest integer that divides both 12 and 18 without leaving a remainder.
    # Similarly, the GCD of 24, 36, and 48 is 12, since 12 is the largest integer that divides all three of these numbers without leaving a remainder.
    return abs(a * b) // math.gcd(a, b)


def _proportional_array_from_dict(
    input_dict: Dict[str, float], max_length: int = 10000
) -> List[str]:
    """
    Generates a proportional array based on the input dictionary.

    Args:
        input_dict (Dict[str, float]): A dictionary where keys are strings and values are their proportions (float).
        max_length (int): check if the length of the output exceeds the max_length. If it does, it will be scaled down to that length. Default is 10k.

    Returns:
        A list of strings where each key is repeated according to its proportion.

    Examples:
        ::
            _proportional_array_from_dict({"a":0.2, "b":0.8})
                # ['a', 'b', 'b', 'b', 'b']
            _proportional_array_from_dict({"a":0.5, "b":0.5})
                # ['a', 'b']
            _proportional_array_from_dict({"a":1/3, "b":1/3, "c": 1/3})
                # ['a', 'b', 'c']
            _proportional_array_from_dict({"a": 3/8, "b": 5/8})
                # ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
            _proportional_array_from_dict({"a":3/5, "b":1/5, "c": 2/10})
                # ['a', 'a', 'a', 'b', 'c']
    """
    # Filter out items with zero value
    filtered_dict = {k: v for k, v in input_dict.items() if v > 0}

    # Normalize the values if they don't sum to 1
    sum_values = sum(filtered_dict.values())
    if sum_values != 1:
        filtered_dict = {k: v / sum_values for k, v in filtered_dict.items()}

    # Convert the proportions to fractions
    fraction_dict = {
        k: Fraction(v).limit_denominator() for k, v in filtered_dict.items()
    }

    # Find the least common denominator (LCD) of the fractions
    lcd = 1
    for fraction in fraction_dict.values():
        lcd *= fraction.denominator // math.gcd(lcd, fraction.denominator)

    # Calculate the scaling factor based on max_length
    scaling_factor = min(1, max_length / lcd)

    result = []
    for key, fraction in fraction_dict.items():
        # Calculate the count for each key based on its proportion and scaling_factor
        k_count = round((fraction * lcd).numerator * scaling_factor)
        # Extend the result array with the key repeated according to its count
        result.extend([key] * k_count)

    return result


def _find_lcm_of_array_lengths(arrays: Dict[str, List[str]]) -> int:
    """
    Finds the least common multiple (LCM) of the lengths of arrays in the input dictionary.

    Args:
        arrays: A dictionary where keys are strings and values are lists of strings.

    Returns:
        The LCM of the lengths of the arrays in the input dictionary.

    Example:
        ::
            arrays = {
                        "v1": ["a", "b", "b", "c"],
                        "v2": ["aa", "bb"]
                    }
            _find_lcm_of_array_lengths(arrays)
                # 4

            arrays = {
                        "v1": ["a", "b", "b", "c"],
                        "v2": ["aa", "bb"],
                        "v3": ["a1", "a2", "a3"]
                    }
            _find_lcm_of_array_lengths(arrays)
                # 12
    """
    array_lengths = [len(arr) for arr in arrays.values()]
    lcm_length = reduce(lambda x, y: _lcm(x, y), array_lengths)
    return lcm_length


def _realize_dicts_of_proportions(
    dict_of_dicts: Dict[str, Dict[str, float]],
) -> Dict[str, List[str]]:
    """
    Generates proportional arrays of equal length for each input dictionary.

    This can be used to get an input dict of proportions of values, and produce a dict with arrays that realizes these proportions.
    It can be used as input to the Sample object so it could be used for running raking.

    Args:
        dict_of_dicts: A dictionary of dictionaries, where each key is a string and
                   each value is a dictionary with keys as strings and values as their proportions (float).

    Returns:
        A dictionary with the same keys as the input and equal length arrays as values.

    Examples:
        ::
            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5}
            }

            realize_dicts_of_proportions(dict_of_dicts)
            # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'], 'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb']}


            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5},
                "v3": {"A": 0.2, "B": 0.8},
            }
            realize_dicts_of_proportions(dict_of_dicts)
                # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'],
                #  'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb'],
                #  'v3': ['A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B']}
            # The above example could have been made shorter. But that's a limitation of the function.

            dict_of_dicts = {
                "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
                "v2": {"aa": 0.5, "bb": 0.5},
                "v3": {"A": 0.2, "B": 0.8},
                "v4": {"A": 0.1, "B": 0.9},
            }
            realize_dicts_of_proportions(dict_of_dicts)
                # {'v1': ['a', 'b', 'b', 'b', 'c', 'a', 'b', 'b', 'b', 'c'],
                #  'v2': ['aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb', 'aa', 'bb'],
                #  'v3': ['A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B'],
                #  'v4': ['A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']}
    """
    # Generate proportional arrays for each dictionary
    arrays = {k: _proportional_array_from_dict(v) for k, v in dict_of_dicts.items()}

    # Find the LCM over the lengths of all the arrays
    lcm_length = _find_lcm_of_array_lengths(arrays)

    # Extend each array to have the same LCM length while maintaining proportions
    result = {}
    for k, arr in arrays.items():
        factor = lcm_length // len(arr)
        extended_arr = arr * factor
        result[k] = extended_arr

    return result


def prepare_marginal_dist_for_raking(
    dict_of_dicts: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Realizes a nested dictionary of proportions into a DataFrame.

    Args:
        dict_of_dicts: A nested dictionary where the outer keys are column names
                   and the inner dictionaries have keys as category labels
                   and values as their proportions (float).

    Returns:
        A DataFrame with columns specified by the outer keys of the input dictionary
        and rows containing the category labels according to their proportions.
        An additional "id" column is added with integer values as row identifiers.

    Example:
        ::
            print(prepare_marginal_dist_for_raking({
                            "A": {"a": 0.5, "b": 0.5},
                            "B": {"x": 0.2, "y": 0.8}
                        }))
            # Returns a DataFrame with columns A, B, and id
                #    A  B  id
                # 0  a  x   0
                # 1  b  y   1
                # 2  a  y   2
                # 3  b  y   3
                # 4  a  y   4
                # 5  b  x   5
                # 6  a  y   6
                # 7  b  y   7
                # 8  a  y   8
                # 9  b  y   9
    """
    target_dict_from_marginals = _realize_dicts_of_proportions(dict_of_dicts)
    target_df_from_marginals = pd.DataFrame.from_dict(target_dict_from_marginals)
    # Add an id column:
    target_df_from_marginals["id"] = range(target_df_from_marginals.shape[0])

    return target_df_from_marginals
