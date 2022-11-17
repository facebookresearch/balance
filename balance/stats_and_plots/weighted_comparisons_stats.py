# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import re
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_stats import (
    descriptive_stats,
    weighted_mean,
    weighted_var,
)
from balance.stats_and_plots.weights_stats import _check_weights_are_valid

logger: logging.Logger = logging.getLogger(__package__)

# TODO: add?
# from scipy.stats import wasserstein_distance

##########################################
# Weighted comparisons - functions to compare one or two data sources with one or two sources of weights
##########################################


# TODO: fix the r_indicator function. The current implementation is broken since it
#       seems to wrongly estimate N.
#       This seems to attempt to reproduce equation 2.2.2, in page 5 in
#       "Indicators for the representativeness of survey response"
#       by Jelke Bethlehem, Fannie Cobben, and Barry Schouten
#       See pdf: https://www150.statcan.gc.ca/n1/en/pub/11-522-x/2008000/article/10976-eng.pdf?st=Zi4d4zld
#       From: https://www150.statcan.gc.ca/n1/pub/11-522-x/2008000/article/10976-eng.pdf
# def r_indicator(sample_p: np.float64, target_p: np.float64) -> np.float64:
#     p = np.concatenate((sample_p, target_p))
#     N = len(sample_p) + len(target_p)
#     return 1 - 2 * np.sqrt(1 / (N - 1) * np.sum((p - np.mean(p)) ** 2))


def _weights_per_covars_names(covar_names: List) -> pd.DataFrame:
    # TODO (p2): consider if to give weights that are proportional to the proportion of this covar in the population
    #           E.g.: if merging varios platforms, maybe if something like windows has very few users, it's impact on the ASMD
    #           should be smaller (similar to how kld works).
    #           The current function structure won't support it, and this would require some extra input (i.e.: baseline target pop proportion)
    """
    Figure out how much weight to give to each column name for ASMD averaging.
    This is meant for post-processing df produced from model_matrix that include
    one-hot categorical variables. This function helps to resolve the needed weight to add
    to columns after they are broken down by the one-hot encoding used in model_matrix.
    Each of these will count for 1/num_levels.
    It's OK to assume that variables with '[]' in the name are one-hot
    categoricals because model_matrix enforces it.

    Args:
        covar_names (List): A list with names of covariate.

    Returns:
        pd.DataFrame: with two columns, one for weights and another for main_covar_names,
        with rows for each of the columns from 'covar_names'

    Examples:
        ::

            asmd_df = pd.DataFrame(
            {
                'age': 0.5,
                'education[T.high_school]': 1,
                'education[T.bachelor]': 1,
                'education[T.masters]': 1,
                'education[T.phd]': 1,
            }, index = ('self', ))

            input = asmd_df.columns.values.tolist()
                # input
                # ['age',
                #  'education[T.high_school]',
                #  'education[T. bachelor]',
                #  'education[T. masters]',
                #  'education[T. phd]']

            _weights_per_covars_names(input).to_dict()
                # Output:
                # {'weight': {'age': 1.0,
                #   'education[T.high_school]': 0.25,
                #   'education[T.bachelor]': 0.25,
                #   'education[T.masters]': 0.25,
                #   'education[T.phd]': 0.25},
                #  'main_covar_names': {'age': 'age',
                #   'education[T.high_school]': 'education',
                #   'education[T.bachelor]': 'education',
                #   'education[T.masters]': 'education',
                #   'education[T.phd]': 'education'}}
    """
    columns_to_original_variable = {v: re.sub(r"\[.*\]$", "", v) for v in covar_names}
    counts = collections.Counter(columns_to_original_variable.values())
    weights = pd.DataFrame(
        {k: 1 / counts[v] for k, v in columns_to_original_variable.items()},
        index=("weight",),
    )
    _check_weights_are_valid(weights)  # verify nothing odd has occured.
    main_covar_names = pd.DataFrame.from_dict(
        columns_to_original_variable,
        orient="index",
        columns=[
            "main_covar_names",
        ],
    )
    return pd.concat([weights.transpose(), main_covar_names], axis=1)


# TODO: add memoization
def asmd(
    sample_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_weights: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
    target_weights: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
    std_type: Literal["target", "sample", "pooled"] = "target",
    aggregate_by_main_covar: bool = False,
) -> pd.Series:
    """
    Calculate the Absolute Standardized Mean Deviation (ASMD, a.k.a
    absolute standardized mean difference) between the columns of two DataFrames
    (or BalanceDFs).
    It uses weighted average and std for the calculations.
    This is the same as taking the absolute value of Cohen's d statistic,
    with a specific choice of the standard deviation (std).

    https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

    As opposed to Cohen's d, the current asmd implementation has several options of std calculation,
    see the arguments section for details.

    Unlike in R package {cobalt}, in the current implementation of asmd:
    - the absolute value is taken
    - un-represented levels of categorical variables are treated as missing,
      not as 0.
    - differences for categorical variables are also weighted by default

    The function omits columns that starts with the name "_is_na_"

    If column names of sample_df and target_df are different, it will only calculate asmd for
    the overlapping columns. The rest will be np.nan.
    The mean(asmd) will be calculated while treating the nan values as 0s.

    Args:
        sample_df (pd.DataFrame): source group of the asmd comparison
        target_df (pd.DataFrame): target group of the asmd comparison.
            The column names should be the same as the ones from sample_df.
        sample_weights (Union[ List, pd.Series, np.ndarray, ], optional): weights to use.
            The default is None.
            If no weights are passed (None), it will use an array of 1s.
        target_weights (Union[ List, pd.Series, np.ndarray, ], optional): weights to use.
            The default is None.
            If no weights are passed (None), it will use an array of 1s.
        std_type (Literal["target", "sample", "pooled"], optional):
            How the standard deviation should be calculated.
            The options are: "target", "sample" and "pooled". Defaults to "target".
            "target" means we use the std from the target population.
            "sample" means we use the std from the sample population.
            "pooled" means we use the simple arithmetic average of
                the variance from the sample and target population.
                We then take the square root of that value to get the pooled std.
                Notice that this is done while giving the same weight to both sources.
                I.e.: there is NO over/under weighting sample or target based on their
                respective sample sizes (in contrast to how pooled sd is calculated in
                Cohen's d statistic).
        aggregate_by_main_covar (bool):
            If to use :func:`_aggregate_asmd_by_main_covar` to aggregate (average) the asmd based on variable name.
            Default is False.
    Returns:
        pd.Series: a Series indexed on the names of the columns in the input DataFrames.
        The values (of type np.float64) are of the ASMD calculation.
        The last element is 'mean(asmd)', which is the average of the calculated ASMD values.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots import weighted_comparisons_stats

            a1 = pd.Series((1, 2))
            b1 = pd.Series((-1, 1))
            a2 = pd.Series((3, 4))
            b2 = pd.Series((-2, 2))
            w1 = pd.Series((1, 1))
            w2 = w1

            r = weighted_comparisons_stats.asmd(
                        pd.DataFrame({"a": a1, "b": b1}),
                        pd.DataFrame({"a": a2, "b": b2}),
                        w1,
                        w2,
                    )

            exp_a = np.abs(a1.mean() - a2.mean()) / a2.std()
            exp_b = np.abs(b1.mean() - b2.mean()) / b2.std()
            print(r)
            print(exp_a)
            print(exp_b)

            # output:
            # {'a': 2.82842712474619, 'b': 0.0, 'mean(asmd)': 1.414213562373095}
            # 2.82842712474619
            # 0.0



            a1 = pd.Series((1, 2))
            b1_A = pd.Series((1, 3))
            b1_B = pd.Series((-1, -3))
            a2 = pd.Series((3, 4))
            b2_A = pd.Series((2, 3))
            b2_B = pd.Series((-2, -3))
            w1 = pd.Series((1, 1))
            w2 = w1

            r = weighted_comparisons_stats.asmd(
                pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
                pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
                w1,
                w2,
            ).to_dict()

            print(r)
            # {'a': 2.82842712474619, 'b[A]': 0.7071067811865475, 'b[B]': 0.7071067811865475, 'mean(asmd)': 1.7677669529663689}

            # Check that using aggregate_by_main_covar works
            r = weighted_comparisons_stats.asmd(
                pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
                pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
                w1,
                w2,
                "target",
                True,
            ).to_dict()

            print(r)
            # {'a': 2.82842712474619, 'b': 0.7071067811865475, 'mean(asmd)': 1.7677669529663689}

    """
    if not isinstance(sample_df, pd.DataFrame):
        raise ValueError(f"sample_df must be pd.DataFrame, is {type(sample_df)}")
    if not isinstance(target_df, pd.DataFrame):
        raise ValueError(f"target_df must be pd.DataFrame, is {type(target_df)}")
    possible_std_type = ("sample", "target", "pooled")
    if not (std_type in possible_std_type):
        raise ValueError(f"std_type must be in {possible_std_type}, is {std_type}")
    if sample_df.columns.values.tolist() != target_df.columns.values.tolist():
        logger.warning(
            f"""
            sample_df and target_df must have the same column names.
            sample_df column names: {sample_df.columns.values.tolist()}
            target_df column names: {target_df.columns.values.tolist()}"""
        )

    sample_mean = descriptive_stats(sample_df, sample_weights, "mean")
    target_mean = descriptive_stats(target_df, target_weights, "mean")

    if std_type == "sample":
        std = descriptive_stats(sample_df, sample_weights, "std")
    elif std_type == "target":
        std = descriptive_stats(target_df, target_weights, "std")
    elif std_type == "pooled":
        target_std = descriptive_stats(target_df, target_weights, "std")
        sample_std = descriptive_stats(sample_df, sample_weights, "std")
        std = np.sqrt(((sample_std**2) + (target_std**2)) / 2)

    out = abs(sample_mean - target_mean) / std  # pyre-ignore[61]: std is always defined

    #  Remove na indicator columns; it's OK to assume that these columns are
    #  indicators because add_na_indicator enforces it
    out = out.loc[:, (c for c in out.columns.values if not c.startswith("_is_na_"))]
    out = out.replace([np.inf, -np.inf], np.nan)

    # TODO (p2): verify that df column names are unique (otherwise throw an exception).
    #            it should probably be upstream during in the Sample creation process.
    weights = _weights_per_covars_names(out.columns.values.tolist())[
        ["weight"]
    ].transpose()
    out = pd.concat((out, weights))
    mean = weighted_mean(out.iloc[0], out.loc["weight"])
    out["mean(asmd)"] = mean

    out = out.iloc[0]
    out.name = None

    if aggregate_by_main_covar:
        out = _aggregate_asmd_by_main_covar(out)

    return out


def _aggregate_asmd_by_main_covar(asmd_series: pd.Series) -> pd.Series:
    """
    This function helps to aggregate the ASMD, which is broken down per level for a category variable, into one value per covariate.
    This is useful since it allows us to get high level view of features such as country, locale, etc.
    It also allows us to filter out variables (such as is_today) before the overall averaging of the ASMD.

    Args:
        asmd_series (pd.Series): a value for each covariate. Covariate name are in the index, the values are the asmd values.

    Returns:
        pd.Series: If asmd_series had several items broken by one-hot encoding,
            then they would be averaged (with equal weight to each).

    Examples:
        ::

            from balance.stats_and_plots.weighted_comparisons_stats import _aggregate_asmd_by_main_covar

            asmd_series = pd.Series(
            {
                'age': 0.5,
                'education[T.high_school]': 1,
                'education[T.bachelor]': 2,
                'education[T.masters]': 3,
                'education[T.phd]': 4,
            })

            _aggregate_asmd_by_main_covar(asmd_series).to_dict()

            # output:
            # {'age': 0.5, 'education': 2.5}
    """
    weights = _weights_per_covars_names(asmd_series.index.values.tolist())

    # turn things into DataFrame to make it easy to aggregate.
    out = pd.concat((asmd_series, weights), axis=1)

    def _weighted_mean_for_our_df(x: pd.DataFrame) -> pd.Series:
        values = x.iloc[:, 0]
        weights = x["weight"]
        weighted_mean = pd.Series(
            ((values * weights) / weights.sum()).sum(), index=["mean"]
        )
        return weighted_mean

    out = out.groupby("main_covar_names").apply(_weighted_mean_for_our_df).iloc[:, 0]
    out.name = None
    out.index.name = None

    return out


# TODO: (p2) sample_before and sample_after are redundant, the moment weights of
#           before and after are supplied directly.
#           In the future, we can either omit sample_after, or change the names to
#           reflect a support a comparison of two panels to some target populations.
def asmd_improvement(
    sample_before: pd.DataFrame,
    sample_after: pd.DataFrame,
    target: pd.DataFrame,
    sample_before_weights: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
    sample_after_weights: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
    target_weights: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
) -> np.float64:
    """Calculates the improvement in mean(asmd) from before to after applying some weight adjustment.

    Args:
        sample_before (pd.DataFrame): DataFrame of the sample before adjustments.
        sample_after (pd.DataFrame): DataFrame of the sample after adjustments
            (should be identical to sample_before. But could be used to compare two populations).
        target (pd.DataFrame): DataFrame of the target population.
        sample_before_weights (Union[ List, pd.Series, np.ndarray, ], optional): Weights before adjustments (i.e.: design weights). Defaults to None.
        sample_after_weights (Union[ List, pd.Series, np.ndarray, ], optional): Weights after some adjustment. Defaults to None.
        target_weights (Union[ List, pd.Series, np.ndarray, ], optional): Design weights of the target population. Defaults to None.

    Returns:
        np.float64: The improvement is taking the (before_mean_asmd-after_mean_asmd)/before_mean_asmd.
        The asmd is calculated using :func:`asmd`.
    """
    asmd_mean_before = asmd(
        sample_before, target, sample_before_weights, target_weights
    ).loc["mean(asmd)"]
    asmd_mean_after = asmd(
        sample_after, target, sample_after_weights, target_weights
    ).loc["mean(asmd)"]
    return (asmd_mean_before - asmd_mean_after) / asmd_mean_before


def outcome_variance_ratio(
    df_numerator: pd.DataFrame,
    df_denominator: pd.DataFrame,
    w_numerator: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
    w_denominator: Union[
        List,
        pd.Series,
        np.ndarray,
        None,
    ] = None,
) -> pd.Series:
    """Calculate ratio of weighted variances of two DataFrames

    Directly calculating the empirical ratio of variance of the outcomes before and after weighting.
    Notice that this is different than design effect.
    The Deff estimates the ratio of variances of the weighted means, while this function calculates the ratio
    of empirical weighted variance of the data.

    Args:
        df_numerator (pd.DataFrame): df_numerator
        df_denominator (pd.DataFrame): df_denominator
        w_numerator (Union[ List, pd.Series, np.ndarray, None, ], optional): w_numerator. Defaults to None.
        w_denominator (Union[ List, pd.Series, np.ndarray, None, ], optional): w_denominator. Defaults to None.

    Returns:
        pd.Series: (np.float64) A series of calculated ratio of variances for each outcome.
    """
    numerator_w_var = weighted_var(df_numerator, w_numerator)
    denominator_w_var = weighted_var(df_denominator, w_denominator)
    return numerator_w_var / denominator_w_var
