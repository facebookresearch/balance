# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from balance.stats_and_plots.weights_stats import _check_weights_are_valid
from balance.util import model_matrix, rm_mutual_nas
from scipy.stats import norm

from statsmodels.stats.weightstats import DescrStatsW

logger: logging.Logger = logging.getLogger(__package__)


##########################################
# Weighted statistics
##########################################


def _prepare_weighted_stat_args(
    v: Union[
        List,
        pd.Series,
        pd.DataFrame,
        npt.NDArray,
        np.matrix,
    ],
    w: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    inf_rm: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Checks that the values (v) and weights (w) are of the supported types and of the same length. Can also replace np.Inf to np.nan.

    Args:
        v (Union[List, pd.Series, pd.DataFrame, np.ndarray, np.matrix,]): a series of values. Notice that pd.DataFrame and np.matrix should store the Series/np.ndarry as columns (not rows).
        w (Union[List, pd.Series, np.ndarray, None]): a series of weights to be used with v (could also be none). Defaults to None.
        inf_rm (bool, optional): should np.inf values be replaced by np.nan? Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: the original v and w after they have been validated, and turned to pd.DataFrame and pd.Series (if needed).
        The values might be int64 or float64, depending on the input.
    """

    supported_types = (
        list,
        pd.Series,
        np.ndarray,
    )

    tmp_supported_types = supported_types + (
        pd.DataFrame,
        np.matrix,
    )
    if type(v) not in tmp_supported_types:
        raise TypeError(f"argument must be {tmp_supported_types}, is {type(v)}")

    tmp_supported_types = supported_types + (type(None),)
    if type(w) not in supported_types + (type(None),):
        raise TypeError(f"argument must be {tmp_supported_types}, is {type(w)}")

    # NOTE: np.matrix is an instance of np.ndarray, so we must turn it to pd.Dataframe before moving forward.
    if isinstance(v, np.matrix):
        v = pd.DataFrame(v)
    if isinstance(v, np.ndarray):
        # Check if it's a 2D array (matrix-like)
        if v.ndim == 2:
            v = pd.DataFrame(v)
        else:
            v = pd.Series(v)
    elif isinstance(v, list):
        v = pd.Series(v)
    if isinstance(v, pd.Series):
        v = v.to_frame()
    if isinstance(w, np.ndarray) or isinstance(w, list):
        w = pd.Series(w)
    if w is None:
        w = pd.Series(np.ones(len(v)), index=v.index)

    if v.shape[0] != w.shape[0]:
        raise ValueError(
            f"weights (w) and data (v) must have same number of rows, ({v.shape[0]}, {w.shape[0]})"
        )

    dtypes = v.dtypes if hasattr(v.dtypes, "__iter__") else [v.dtypes]

    if not all(np.issubdtype(x, np.number) for x in dtypes):
        raise TypeError("all columns must be numeric")

    if inf_rm:
        from balance.util import _safe_replace_and_infer

        # Since v is guaranteed to be DataFrame and w is guaranteed to be Series at this point,
        # we can safely cast the results
        v = _safe_replace_and_infer(v)
        w = _safe_replace_and_infer(w)

    v = v.reset_index(drop=True)
    w = w.reset_index(drop=True)

    _check_weights_are_valid(w)

    # Type cast to satisfy pyre - we know these are correct types based on the function logic
    return pd.DataFrame(v), pd.Series(w)


def weighted_mean(
    v: Union[
        List,
        pd.Series,
        pd.DataFrame,
        np.matrix,
    ],
    w: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    inf_rm: bool = False,
) -> pd.Series:
    """
    Computes the weighted average of a pandas Series or DataFrame.

    If no weights are supplied, it just computes the simple arithmetic mean.

    See:
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    Uses :func:`_prepare_weighted_stat_args`.

    Args:
        v (Union[ List, pd.Series, pd.DataFrame, np.matrix, ]): values to average. None (or np.nan) values are treated like 0.
            If v is a DataFrame than the average of the values from each column will be returned, all using the same set of weights from w.
        w (Union[ List, pd.Series], optional): weights. Defaults to None.
            If there is None value in the weights, that value will be ignored from the calculation.
        inf_rm (bool, optional): whether to remove infinite (from weights or values) from the computation. Defaults to False.

    Returns:
        pd.Series(dtype=np.float64): The weighted mean.
        If v is a DataFrame with several columns than the pd.Series will have a value for the weighted mean of each of the columns.
        If inf_rm=False then:
            If v has Inf then the output will be Inf.
            If w has Inf then the output will be np.nan.

    Examples:
        ::
            from balance.stats_and_plots.weighted_stats import weighted_mean

            weighted_mean(pd.Series((1, 2, 3, 4)))
                # 0    2.5
                # dtype: float64

            weighted_mean(pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4)))
                # 0    3.0
                # dtype: float64

            df = pd.DataFrame(
                {"a": [1,2,3,4], "b": [1,1,1,1]}
            )
            w = pd.Series((1, 2, 3, 4))
            weighted_mean(df, w)
                # a    3.0
                # b    1.0
                # dtype: float64
    """
    v, w = _prepare_weighted_stat_args(v, w, inf_rm)
    return v.multiply(w, axis="index").sum() / w.sum()


def var_of_weighted_mean(
    v: Union[List, pd.Series, pd.DataFrame, np.matrix],
    w: Optional[Union[List, pd.Series, npt.NDArray]] = None,
    inf_rm: bool = False,
) -> pd.Series:
    """
    Computes the variance of the weighted average (pi estimator for ratio-mean) of a list of values and their corresponding weights.

    If no weights are supplied, it assumes that all values have equal weights of 1.0.

    See: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_of_the_weighted_mean_(%CF%80-estimator_for_ratio-mean)

    Uses :func:`_prepare_weighted_stat_args`.

    v (Union[List, pd.Series, pd.DataFrame, np.matrix]): A series of values. If v is a DataFrame, the weighted variance will be calculated for each column using the same set of weights from `w`.
    w (Optional[Union[List, pd.Series, np.ndarray]]): A series of weights to be used with `v`. If None, all values will be weighted equally.
    inf_rm (bool, optional): Whether to remove infinite (from weights or values) from the computation. Defaults to False.

    Returns:
        pd.Series: The variance of the weighted mean. If `v` is a DataFrame with several columns, the pd.Series will have a value for the weighted variance of each column. The values are of data type `np.float64`.
        If `inf_rm` is False:
            If `v` has infinite values, the output will be `Inf`.
            If `w` has infinite values, the output will be `np.nan`.


    Examples:
        ::
            from balance.stats_and_plots.weighted_stats import var_of_weighted_mean

            #  In R: sum((1:4 - mean(1:4))^2 / 4) / (4)
            #  [1] 0.3125
            var_of_weighted_mean(pd.Series((1, 2, 3, 4)))
                # pd.Series(0.3125)

            #  For a reproducible R example, see: https://gist.github.com/talgalili/b92cd8cdcbfc287e331a8f27db265c00
            var_of_weighted_mean(pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4)))
                # pd.Series(0.24)

            df = pd.DataFrame(
                {"a": [1,2,3,4], "b": [1,1,1,1]}
            )
            w = pd.Series((1, 2, 3, 4))
            var_of_weighted_mean(df, w)

                # a    0.24
                # b    0.00
                # dtype: float64
    """
    v, w = _prepare_weighted_stat_args(v, w, inf_rm)
    weighed_mean_of_v = weighted_mean(v, w, inf_rm)  # This is a pd.Series
    # NOTE: the multiply needs to be called from the pd.Series. Unfortunately this makes the code less readable.
    sum_of_squared_weighted_diffs = (
        ((v - weighed_mean_of_v).multiply(w, axis="index")).pow(2).sum()
    )
    squared_sum_of_w = w.sum() ** 2  # This is a pd.series
    return sum_of_squared_weighted_diffs / squared_sum_of_w


def ci_of_weighted_mean(
    v: Union[List[float], pd.Series, pd.DataFrame, npt.NDArray],
    w: Optional[Union[List[float], pd.Series, npt.NDArray]] = None,
    conf_level: float = 0.95,
    round_ndigits: Optional[int] = None,
    inf_rm: bool = False,
) -> pd.Series:
    """
    Computes the confidence interval of the weighted mean of a list of values and their corresponding weights.

    If no weights are supplied, it assumes that all values have equal weights of 1.0.

    v (Union[List[float], pd.Series, pd.DataFrame, np.ndarray]): A series of values. If v is a DataFrame, the weighted mean and its confidence interval will be calculated for each column using the same set of weights from `w`.
    w (Optional[Union[List[float], pd.Series, np.ndarray]]): A series of weights to be used with `v`. If None, all values will be weighted equally.
    conf_level (float, optional): Confidence level for the interval, between 0 and 1. Defaults to 0.95.
    round_ndigits (Optional[int], optional): Number of decimal places to round the confidence interval. If None, the values will not be rounded. Defaults to None.
    inf_rm (bool, optional): Whether to remove infinite (from weights or values) from the computation. Defaults to False.

    Returns:
        pd.Series: The confidence interval of the weighted mean. If `v` is a DataFrame with several columns, the pd.Series will have a value for the confidence interval of each column. The values are of data type Tuple[np.float64, np.float64].
        If `inf_rm` is False:
            If `v` has infinite values, the output will be `Inf`.
            If `w` has infinite values, the output will be `np.nan`.

    Examples:
        ::
            from balance.stats_and_plots.weighted_stats import ci_of_weighted_mean

            ci_of_weighted_mean(pd.Series((1, 2, 3, 4)))
                # 0    (1.404346824279273, 3.5956531757207273)
                # dtype: object

            ci_of_weighted_mean(pd.Series((1, 2, 3, 4)), round_ndigits = 3).to_list()
                # [(1.404, 3.596)]

            ci_of_weighted_mean(pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4)))
                # 0    (2.039817664728938, 3.960182335271062)
                # dtype: object

            ci_of_weighted_mean(pd.Series((1, 2, 3, 4)), pd.Series((1, 2, 3, 4)), round_ndigits = 3).to_list()
                # [(2.04, 3.96)]

            df = pd.DataFrame(
                {"a": [1,2,3,4], "b": [1,1,1,1]}
            )
            w = pd.Series((1, 2, 3, 4))
            ci_of_weighted_mean(df, w, conf_level = 0.99, round_ndigits = 3)
                # a    (1.738, 4.262)
                # b        (1.0, 1.0)
                # dtype: object
            ci_of_weighted_mean(df, w, conf_level = 0.99, round_ndigits = 3).to_dict()
                # {'a': (1.738, 4.262), 'b': (1.0, 1.0)}
    """
    v, w = _prepare_weighted_stat_args(v, w, inf_rm)
    weighed_mean_of_v = weighted_mean(v, w, inf_rm)
    var_weighed_mean_of_v = var_of_weighted_mean(v, w, inf_rm)
    z_value = norm.ppf((1 + conf_level) / 2)

    if isinstance(v, pd.Series):
        ci_index = v.index
    elif isinstance(v, pd.DataFrame):
        ci_index = v.columns
    else:
        ci_index = None

    ci = pd.Series(
        [
            (
                weighed_mean_of_v.iloc[i]
                - z_value * np.sqrt(var_weighed_mean_of_v.iloc[i]),
                weighed_mean_of_v.iloc[i]
                + z_value * np.sqrt(var_weighed_mean_of_v.iloc[i]),
            )
            for i in range(len(weighed_mean_of_v))
        ],
        index=ci_index,
    )

    if round_ndigits is not None:
        # Apply a lambda function to round a pd.Series of tuples to x decimal places
        ci = ci.apply(lambda t: tuple(round(x, round_ndigits) for x in t))

    # pyre-ignore
    return ci


def weighted_var(
    v: Union[
        List,
        pd.Series,
        pd.DataFrame,
        npt.NDArray,
        np.matrix,
    ],
    w: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    inf_rm: bool = False,
) -> pd.Series:
    """
    Calculate the sample weighted variance (a.k.a 'reliability weights').
    This is described here:
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights_2
    And also used in SDMTools, see:
    https://www.gnu.org/software/gsl/doc/html/statistics.html#weighted-samples

    Uses :func:`weighted_mean` and :func:`_prepare_weighted_stat_args`.

    Args:
        v (Union[ List, pd.Series, pd.DataFrame, np.matrix, ]): values to get the weighted variance for. None values are treated like 0.
            If v is a DataFrame than the average of the values from each column will be returned, all using the same set of weights from w.
        w (Union[ List, pd.Series], optional): weights. Defaults to None.
            If there is None value in the weights, that value will be ignored from the calculation.
        inf_rm (bool, optional): whether to remove infinite (from weights or values) from the computation. Defaults to False.

    Returns:
        pd.Series[np.float64]: The weighted variance.
        If v is a DataFrame with several columns than the pd.Series will have a value for the weighted mean of each of the columns.
        If inf_rm=False then:
            If v has Inf then the output will be Inf.
            If w has Inf then the output will be np.nan.
    """
    v, w = _prepare_weighted_stat_args(v, w, inf_rm)
    means = weighted_mean(v, w)
    v1 = w.sum()
    v2 = (w**2).sum()
    # Avoid divide by zero warning
    denominator = v1**2 - v2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (v1 / denominator) * ((v - means) ** 2).multiply(w, axis="index").sum()
    return result


def weighted_sd(
    v: Union[
        List,
        pd.Series,
        pd.DataFrame,
        np.matrix,
    ],
    w: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    inf_rm: bool = False,
) -> pd.Series:
    """Calculate the sample weighted standard deviation

    See :func:`weighted_var` for details.

    Args:
        v (Union[ List, pd.Series, pd.DataFrame, np.matrix, ]): Values.
        w (Union[ List, pd.Series, np.ndarray, None, ], optional): Weights. Defaults to None.
        inf_rm (bool, optional): Remove inf. Defaults to False.

    Returns:
        pd.Series: np.sqrt of :func:`weighted_var` (np.float64)
    """
    return pd.Series(np.sqrt(weighted_var(v, w, inf_rm)))


def weighted_quantile(
    v: Union[
        List,
        pd.Series,
        pd.DataFrame,
        npt.NDArray,
        np.matrix,
    ],
    quantiles: Union[
        List,
        pd.Series,
        npt.NDArray,
    ],
    w: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    inf_rm: bool = False,
) -> pd.DataFrame:
    """
    Calculates the weighted quantiles (q) of values (v) based on weights (w).

    See :func:`_prepare_weighted_stat_args` for the pre-processing done to v and w.

    Based on :func:`statsmodels.stats.weightstats.DescrStatsW`.

    Args:
        v (Union[ List, pd.Series, pd.DataFrame, np.array, np.matrix, ]): values to get the weighted quantiles for.
        quantiles (Union[ List, pd.Series, ]): the quantiles to calculate.
        w (Union[ List, pd.Series, np.array, ] optional): weights. Defaults to None.
        inf_rm (bool, optional): whether to remove infinite (from weights or values) from the computation. Defaults to False.

    Returns:
        pd.DataFrame: The index (names p) has the values from quantiles. The columns are based on v:
            If it's a pd.Series it's one column, if it's a pd.DataFrame with several columns, than each column
            in the output corrosponds to the column in v.
    """

    v, w = _prepare_weighted_stat_args(v, w, inf_rm)

    v = np.array(v)
    w = np.array(w)
    quantiles = np.array(quantiles)

    d = DescrStatsW(v.astype(float), weights=w)
    return d.quantile(quantiles)


def descriptive_stats(
    df: pd.DataFrame,
    weights: Union[
        List,
        pd.Series,
        npt.NDArray,
        None,
    ] = None,
    stat: Literal["mean", "std", "var_of_mean", "ci_of_mean", "..."] = "mean",
    # relevant only if stat is None
    weighted: bool = True,
    # relevant only if we have non-numeric columns and we want to use model_matrix on them
    numeric_only: bool = False,
    add_na: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Computes weighted statistics (e.g.: mean, std) on a DataFrame

    This function gets a DataFrame + weights and apply some weighted aggregation function (mean, std, or DescrStatsW).
    The main benefit of the function is that if the DataFrame includes non-numeric columns, then descriptive_stats will first
    run :func:`model_matrix` to create some numeric dummary variable that will then be processed.

    Args:
        df (pd.DataFrame): Some DataFrame to get stats (mean, std, etc.) for.
        weights (Union[ List, pd.Series, np.ndarray, ], optional): Weights to apply for the computation. Defaults to None.
        stat (Literal["mean", "std", "var_of_mean", ...], optional): Which statistic to calculate on the data.
            If mean - uses :func:`weighted_mean` (with inf_rm=True)
            If std - uses :func:`weighted_sd` (with inf_rm=True)
            If var_of_mean - uses :func:`var_of_weighted_mean` (with inf_rm=True)
            If ci_of_mean - uses :func:`ci_of_weighted_mean` (with inf_rm=True)
            If something else - tries to use :func:`statsmodels.stats.weightstats.DescrStatsW`.
                This supports stat such as: std_mean, sum_weights, nobs, etc. See function documentation to see more.
                (while removing mutual nan using :func:`rm_mutual_nas`)
            Defaults to "mean".
        weighted (bool, optional): If stat is not "mean" or "std", if to use the weights with the DescrStatsW function.
            Defaults to True.
        numeric_only (bool, optional): Should the statistics be computed only on numeric columns?
            If True - then non-numeric columns will be omitted.
            If False - then :func:`model_matrix` (with no formula argument) will be used to transfer non-numeric columns to dummy variables.
            Defaults to False.
        add_na (bool, optional): Passed to :func:`model_matrix`.
            Relevant only if numeric_only == False and df has non-numeric columns.
            Defaults to True.
        **kwargs: extra args to be passed to functions (e.g.: ci_of_weighted_mean)

    Returns:
        pd.DataFrame: Returns pd.DataFrame of the output (based on stat argument), for each of the columns in df.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots.weighted_stats import descriptive_stats, weighted_mean, weighted_sd

            # Without weights
            x = [1, 2, 3, 4]
            print(descriptive_stats(pd.DataFrame(x), stat="mean"))
            print(np.mean(x))
            print(weighted_mean(x))
                #     0
                # 0  2.5
                # 2.5
                # 0    2.5
                # dtype: float64

            print(descriptive_stats(pd.DataFrame(x), stat="var_of_mean"))
                #         0
                # 0  0.3125

            print(descriptive_stats(pd.DataFrame(x), stat="std"))
            print(weighted_sd(x))
            x2 = pd.Series(x)
            print(np.sqrt( np.sum((x2 - x2.mean())**2) / (len(x)-1) ))
                #         0
                # 0  1.290994
                # 0    1.290994
                # dtype: float64
                # 1.2909944487358056
                # Notice that it is different from
                # print(np.std(x))
                # which gives: 1.118033988749895
                # Which is the MLE (i.e.: biased, dividing by n and not n-1) estimator for std:
                # (np.sqrt( np.sum((x2 - x2.mean())**2) / (len(x)) ))

            x2 = pd.Series(x)
            tmp_sd = np.sqrt(np.sum((x2 - x2.mean()) ** 2) / (len(x) - 1))
            tmp_se = tmp_sd / np.sqrt(len(x))
            print(descriptive_stats(pd.DataFrame(x), stat="std_mean").iloc[0, 0])
            print(tmp_se)
                # 0.6454972243679029
                # 0.6454972243679028

            # Weighted results
            x, w = [1, 2, 3, 4], [1, 2, 3, 4]
            print(descriptive_stats(pd.DataFrame(x), w, stat="mean"))
                #      0
                # 0  3.0
            print(descriptive_stats(pd.DataFrame(x), w, stat="std"))
                #           0
                # 0  1.195229
            print(descriptive_stats(pd.DataFrame(x), w, stat="std_mean"))
                #           0
                # 0  0.333333
            print(descriptive_stats(pd.DataFrame(x), w, stat="var_of_mean"))
                #       0
                # 0  0.24
            print(descriptive_stats(pd.DataFrame(x), w, stat="ci_of_mean", conf_level = 0.99, round_ndigits=3))
                #                 0
                # 0  (1.738, 4.262)

    """
    if len(df.select_dtypes(np.number).columns) != len(df.columns):
        # If we have non-numeric columns, and want faster results,
        # then we can set numeric_only == True.
        # This will skip the model_matrix computation for non-numeric variables.
        if numeric_only:
            # TODO: (p2) does this check takes a long time?
            #       if it does - then maybe add an option of numeric_only = None
            #       to just use df as is.
            df = df.select_dtypes(include=[np.number])
        else:
            # TODO: add the ability to pass formula argument to model_matrix
            df = model_matrix(  # pyre-ignore[9]: this uses the DataFrame onlyÍ
                df, add_na=add_na, return_type="one"
            )["model_matrix"]

    if stat == "mean":
        return weighted_mean(df, weights, inf_rm=True).to_frame().transpose()
    elif stat == "std":
        return weighted_sd(df, weights, inf_rm=True).to_frame().transpose()
    elif stat == "var_of_mean":
        return var_of_weighted_mean(df, weights, inf_rm=True).to_frame().transpose()
    elif stat == "ci_of_mean":
        conf_level = kwargs.get("conf_level", 0.95)
        round_ndigits = kwargs.get("round_ndigits", None)
        return (
            ci_of_weighted_mean(
                df,
                weights,
                inf_rm=True,
                conf_level=conf_level,
                round_ndigits=round_ndigits,
            )
            .to_frame()
            .transpose()
        )

    # TODO: (p2) check which input DescrStatsW takes, not sure we need to run the next two lines.
    if weights is not None:
        weights = pd.Series(weights)
    # TODO: (p2) consider to remove the "weighted" argument, and just use
    #       whatever is passed to "weights" (if None, than make sure it's replaced with 1s)
    #  Fallback to statsmodels function
    _check_weights_are_valid(weights)
    wdf = {}
    for c in df.columns.values:
        df_c, w = rm_mutual_nas(df.loc[:, c], weights)
        # pyre-fixme[16]: `DescrStatsW` has no attribute `...`.
        wdf[c] = [getattr(DescrStatsW(df_c, w if weighted else None), stat)]
    return pd.DataFrame(wdf)


def relative_frequency_table(
    df: Union[pd.DataFrame, pd.Series],
    column: Optional[str] = None,
    w: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Creates a relative frequency table by aggregating over a categorical column (`column`) - optionally weighted by `w`.
    I.e.: produce the proportion (or weighted proportion) of rows that appear in each category, relative to the total number of rows (or sum of weights).
    See: https://en.wikipedia.org/wiki/Frequency_(statistics)#Types.

    Used in plotting functions.

    Args:
        df (pd.DataFrame): A DataFrame with categorical columns, or a pd.Series of the grouping column.
        column (Optional[str]): The name of the column to be aggregated.
            If None (default), then it takes the first column of df (if pd.DataFrame), or just uses as is (if pd.Series)
        w (Optional[pd.Series], optional): Optional weights to use when aggregating the relative proportions.
            If None than assumes weights is 1 for all rows. The defaults is None.

    Returns:
        pd.DataFrame: a pd.DataFrame with columns:
            - `column`, the aggregation variable, and,
            -  'prop', the aggregated (weighted) proportion of rows from each group in 'column'.

    Examples:
        ::

            from balance.stats_and_plots.weighted_stats import relative_frequency_table
            import pandas as pd

            df = pd.DataFrame({
                'group': ('a', 'b', 'c', 'c'),
                'v1': (1, 2, 3, 4),
            })
            print(relative_frequency_table(df, 'group'))
                #     group  prop
                #   0     a  0.25
                #   1     b  0.25
                #   2     c  0.50
            print(relative_frequency_table(df, 'group', pd.Series((2, 1, 1, 1),)))
                #     group  prop
                #   0     a   0.4
                #   1     b   0.2
                #   2     c   0.4

            # Using a pd.Series:
            a_series = df['group']
            print(relative_frequency_table(a_series))
                #   group  prop
                # 0     a  0.25
                # 1     b  0.25
                # 2     c  0.50
    """
    _check_weights_are_valid(w)

    if not (w is None or isinstance(w, pd.Series)):
        raise TypeError("argument `w` must be a pandas Series or None")
    if w is None:
        w = pd.Series(np.ones(df.shape[0]))

    w = w.rename("Freq")

    if column is None:
        if isinstance(df, pd.DataFrame):
            column = df.columns.values[0]
        elif isinstance(df, pd.Series):
            if df.name is None:
                df.name = "group"
            column = df.name
        else:
            raise TypeError("argument `df` must be a pandas DataFrame or Series")

    relative_frequency_table_data = pd.concat((df, w), axis=1)

    relative_frequency_table_data = relative_frequency_table_data.groupby(
        column, as_index=False
    ).sum()
    relative_frequency_table_data["prop"] = relative_frequency_table_data["Freq"] / sum(
        relative_frequency_table_data["Freq"]
    )

    return relative_frequency_table_data.loc[:, (column, "prop")]
