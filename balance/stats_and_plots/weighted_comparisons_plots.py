# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import random

from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline
import seaborn as sns

from balance.stats_and_plots.weighted_stats import (
    relative_frequency_table,
    weighted_quantile,
)
from balance.util import _safe_show_legend, choose_variables, rm_mutual_nas
from matplotlib.colors import rgb2hex

logger: logging.Logger = logging.getLogger(__package__)


################################################################################
#  seaborn plots below
################################################################################


def _return_sample_palette(
    names: List[str],
) -> Union[Dict[str, str], str]:
    """Returns sample palette for seaborn plots.

    Named colors for matplotlib: https://stackoverflow.com/a/37232760

    Args:
        names (List[str]): e.g. ['self', 'target']

    Returns:
        Union[Dict[str, str], str]: e.g. {'self': 'tomato', 'target': 'dodgerblue'}
    """
    # colors to match the plotly colors
    col_unadjusted = rgb2hex((222 / 255, 45 / 255, 38 / 255, 0.8), keep_alpha=True)
    col_adjusted = rgb2hex((52 / 255, 165 / 255, 48 / 255, 0.5), keep_alpha=True)
    col_target = rgb2hex((158 / 255, 202 / 255, 225 / 255, 0.8), keep_alpha=True)

    if set(names) == {"self", "target"}:
        sample_palette = {
            "self": col_unadjusted,
            "target": col_target,
        }
    elif set(names) == {"self", "unadjusted"}:
        sample_palette = {
            "self": col_adjusted,
            "unadjusted": col_unadjusted,
        }
    elif set(names) == {"self", "unadjusted", "target"}:
        sample_palette = {
            "self": col_adjusted,
            "unadjusted": col_unadjusted,
            "target": col_target,
        }
    elif set(names) == {"adjusted", "unadjusted", "target"}:
        sample_palette = {
            "adjusted": col_adjusted,
            "unadjusted": col_unadjusted,
            "target": col_target,
        }
    else:
        sample_palette = "muted"
    return sample_palette


def _plotly_marker_color(
    name: Literal["sample", "unadjusted", "self", "adjusted"],
    only_self_and_target: bool,
    color_type: Literal["color", "line"],
) -> str:
    """
    Returns a color string for a marker in a plotly plot based on the given parameters.

    Args:
        name (Literal["sample", "unadjusted", "self", "adjusted"]): Name of the marker.
        only_self_and_target (bool): Determines if only self and target groups are available, or if it's self, unadjusted and target.
        color_type (Literal["color", "line"]): The type of color, either "color" or "line".

    Returns:
        str: A string representing the color in RGBA format.

    Raises:
        ValueError: If the color_type is not one of the accepted options.
    """
    if color_type == "color":
        col1 = 0.8
        col2 = 0.5
    elif color_type == "line":
        col1 = 1
        col2 = 1
    else:
        raise ValueError(
            "Invalid value for 'tycolor_typepe'. Must be either 'color' or 'line'."
        )

    if name.lower() in ["sample", "unadjusted"] or (
        name.lower() == "self" and only_self_and_target
    ):
        return f"rgba(222,45,38,{col1})"
    elif name.lower() in ["self", "adjusted"]:
        return f"rgba(52,165,48,{col2})"
    else:
        return f"rgb(158,202,225,{col1})"


def plot_bar(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: List[str],
    column: str,
    axis: Optional[plt.Axes] = None,
    weighted: bool = True,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Shows a (weighted) sns.barplot using a relative frequency table of several DataFrames (with optional control over the y-axis limits).

    If weighted is True, then mutual NA values are removed using :func:`rm_mutual_nas`.

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating the variable using :func:`relative_frequency_table`.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
        column (str): The column to be used to aggregate using :func:`relative_frequency_table`.
        axis (Optional[plt.Axes], optional): matplotlib Axes object to draw the plot onto, otherwise uses the current Axes. Defaults to None.
        weighted (bool, optional): If to pass the weights from the dicts inside dfs. Defaults to True.
        title (str, optional): Title of the plot. Defaults to "barplot of covar '{column}'".
        ylim (Optional[Tuple[float, float]], optional): A tuple with two float values representing the lower and upper limits of the y-axis.
            If not provided, the y-axis range is determined automatically. Defaults to None.

    Examples:
        ::

            from balance.stats_and_plots.weighted_comparisons_plots import plot_bar
            import pandas as pd
            import numpy as np

            df = pd.DataFrame({
                'group': ('a', 'b', 'c', 'c'),
                'v1': (1, 2, 3, 4),
            })

            plot_bar(
                [{"df": df, "weight": pd.Series((1, 1, 1, 1))}, {"df": df, "weight": pd.Series((2, 1, 1, 1))}],
                names = ["self", "target"],
                column = "group",
                axis = None,
                weighted = True)

            # The same as above just with ylim set to (0, 1).
            plot_bar(
                [{"df": df, "weight": pd.Series((1, 1, 1, 1))}, {"df": df, "weight": pd.Series((2, 1, 1, 1))}],
                names = ["self", "target"],
                column = "group",
                axis = None,
                weighted = True,
                ylim = (0, 1))

            # Also deals with np.nan weights
            a = plot_bar(
                [{"df": df, "weight": pd.Series((1, 1, 1, np.nan))}, {"df": df, "weight": pd.Series((2, 1, 1, np.nan))}],
                names = ["self", "target"],
                column = "group",
                axis = None,
                weighted = True)
    """
    plot_data = []
    for ii, i in enumerate(dfs):
        a_series = i["df"][column]
        _w = i["weight"]
        if weighted:
            a_series, _w = rm_mutual_nas(a_series, _w)
            a_series.name = column  # rm_mutual_nas removes name, so we set it back

        df_plot_data = relative_frequency_table(a_series, w=_w if weighted else None)
        df_plot_data["dataset"] = names[ii]  # a recycled column for barplot's hue.

        plot_data.append(df_plot_data)

    sample_palette = _return_sample_palette(names)
    if title is None:
        title = f"barplot of covar '{column}'"

    ax = sns.barplot(
        x=column,
        y="prop",
        hue="dataset",
        data=pd.concat(plot_data),
        ax=axis,
        palette=sample_palette,
        saturation=1,
        alpha=0.6,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)


def plot_hist_kde(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: List[str],
    column: str,
    axis: Optional[plt.Axes] = None,
    weighted: bool = True,
    dist_type: Literal["hist", "kde", "ecdf"] = "hist",
    title: Optional[str] = None,
) -> None:
    """Shows a (weighted) distribution plot ():func:`sns.displot`) of data from several DataFrame objects.

    Options include histogram (hist), kernel density estimate (kde), and empirical cumulative density function (ecdf).

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating the variable using :func:`relative_frequency_table`.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
        column (str): The column to be used to aggregate using :func:`relative_frequency_table`.
        axis (Optional[plt.Axes], optional): matplotlib Axes object to draw the plot onto, otherwise uses the current Axes. Defaults to None.
        weighted (bool, optional): If to pass the weights from the dicts inside dfs. Defaults to True.
        dist_type (Literal["hist", "kde", "ecdf"], optional): The type of plot to draw. Defaults to "hist".
        title (str, optional): Title of the plot. Defaults to "distribution plot of covar '{column}'".


    Examples:
        ::

            from balance.stats_and_plots.weighted_comparisons_plots import plot_hist_kde
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt

            df = pd.DataFrame({
                'group': ('a', 'b', 'c', 'c'),
                'v1': (1, 2, 3, 4),
            })

            dfs1 = [{"df": pd.DataFrame(pd.Series([1,2,2,2,3,4,5,5,7,8,9,9,9,9,5,2,5,4,4,4], name = "v1")), "weight": None}, {"df": df, "weight": pd.Series((200, 1, 0, 20))}]

            plt.figure(1)

            # kde: no weights
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = None,
                weighted = False, dist_type = "kde")

            plt.figure(2)

            # kde: with weights
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = None,
                weighted = True, dist_type = "kde")

            plt.figure(3)

            # hist
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = None,
                weighted = True, dist_type = "hist")

            plt.figure(4)

            # ecdf
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = None,
                weighted = True, dist_type = "ecdf")


            # can work nicely with plt.subplots:
            f, axes = plt.subplots(1, 2, figsize=(7, 7 * 1))
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = axes[0],
                weighted = False, dist_type = "kde")
            plot_hist_kde(
                dfs1,
                names = ["self", "target"],
                column = "v1",
                axis = axes[1],
                weighted = False, dist_type = "kde")
    """
    possible_dist_function = {
        "hist": sns.histplot,
        "kde": sns.kdeplot,
        "ecdf": sns.ecdfplot,
    }
    dist_function = possible_dist_function[dist_type]
    # NOTE: the reason we don't use sns.displot directly is that it doesn't accept an ax= kwarg.
    # see also here: https://stackoverflow.com/a/63895570/256662

    plot_data = []
    for ii, i in enumerate(dfs):
        a_series = i["df"][column]
        _w = i["weight"]
        if _w is None:
            _w = pd.Series(np.ones(len(a_series)), index=a_series.index)
        if weighted:
            a_series, _w = rm_mutual_nas(a_series, _w)
            a_series.name = column  # rm_mutual_nas removes name, so we set it back
            # TODO: verify if this normalization to sum to 1 is needed (if so, how come we don't do it when _w is None)?
            _w = _w / np.sum(_w)

        df_plot_data = pd.DataFrame({column: a_series, "_w": _w})
        df_plot_data["dataset"] = names[ii]  # a recycled column for barplot's hue.

        plot_data.append(df_plot_data)

    plot_data = pd.concat(plot_data)

    sample_palette = _return_sample_palette(names)
    if title is None:
        title = f"distribution plot of covar '{column}'"

    kwargs_for_dist_function = {
        "data": plot_data,
        "x": column,
        "hue": "dataset",
        "ax": axis,
        "weights": plot_data["_w"] if weighted else None,
        # common_norm:False,
        "palette": sample_palette,
        "linewidth": 2,
    }
    if dist_type != "ecdf":
        kwargs_for_dist_function["common_norm"] = False
    ax = dist_function(**kwargs_for_dist_function)
    ax.set_title(title)


def plot_qq(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: List[str],
    column: str,
    axis: Optional[plt.Axes] = None,
    weighted: bool = True,
) -> None:
    """Plots a qq plot of the weighted data from a DataFrame object against some target.

    See: https://en.wikipedia.org/wiki/Q-Q_plot

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating the variable using :func:`weighted_quantile`.
            Uses the last df item in the list as the target.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
        column (str): The column to be used to aggregate using :func:`weighted_quantile`.
        axis (Optional[plt.Axes], optional): matplotlib Axes object to draw the plot onto, otherwise uses the current Axes. Defaults to None.
        weighted (bool, optional): If to pass the weights from the dicts inside dfs. Defaults to True.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots.weighted_comparisons_plots import plot_qq
            from numpy import random

            df = pd.DataFrame({
                'v1': random.uniform(size=100),
            }).sort_values(by=['v1'])

            dfs1 = [
                {"df": df, "weight": pd.Series(np.ones(100))},
                {"df": df, "weight": pd.Series(range(100))},
                {"df": df, "weight": pd.Series(np.ones(100))},
            ]

            # plot_qq(dfs1, names=["self", "unadjusted", "target"], column="v1", axis=None, weighted=False)
            plot_qq(dfs1, names=["self", "unadjusted", "target"], column="v1", axis=None, weighted=True)
    """
    target = dfs[-1]  # assumes the last item is target
    dfs = dfs[:-1]  # assumes the non-last item are dfs to be compared to target

    for ii, i in enumerate(dfs):
        d = i["df"]
        _w = i["weight"]

        target_q = weighted_quantile(
            target["df"][column],
            np.arange(0, 1, 0.001),
            # pyre-fixme[6]: TODO:
            # This is because of using:
            # dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
            # When in fact we want to be clear that the first element is called
            # "df" and the second "weight", and that the first is a pd.DataFrame and
            # the second pd.Series. Until this is not clear - the following line will raise an error.
            target["weight"] if weighted else None,
        )
        sample_q = weighted_quantile(
            d.loc[:, column],
            np.arange(0, 1, 0.001),
            # pyre-fixme[6]
            _w if weighted else None,
        )

        axis = sns.scatterplot(
            x=target_q.iloc[:, 0].values,
            y=sample_q.iloc[:, 0].values,
            label=names[ii],
            ax=axis,
        )
    set_xy_axes_to_use_the_same_lim(axis)
    axis.plot(axis.get_xlim(), axis.get_ylim(), "--")
    axis.set_title(f"quantile-quantile plot of covar '{column}' in target vs sample")
    _safe_show_legend(axis)


def plot_qq_categorical(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: List[str],
    column: str,
    axis: Optional[plt.Axes] = None,
    weighted: bool = True,
    label_threshold: int = 30,
) -> None:
    """A scatter plot of weighted relative frequencies of categories from each df.

    Notice that this is not a "real" qq-plot, but rather a scatter plot of (estimated, weighted) probabilities for each category.

    X-axis is the sample (adjusted, unadjusted) and Y-axis is the target.

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating the variable using :func:`weighted_quantile`.
            Uses the last df item in the list as the target.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
        column (str): The column to be used to aggregate using :func:`relative_frequency_table`.
        axis (Optional[plt.Axes], optional): matplotlib Axes object to draw the plot onto, otherwise uses the current Axes. Defaults to None.
        weighted (bool, optional): If to pass the weights from the dicts inside dfs. Defaults to True.
        label_threshold (int, optional): All labels that are larger from the threshold will be omitted from the scatter plot (so to reduce clutter). Defaults to 30.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots.weighted_comparisons_plots import plot_qq_categorical
            from numpy import random

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100),
            }).sort_values(by=['v1'])

            dfs1 = [
                {"df": df, "weight": pd.Series(np.ones(100))},
                {"df": df, "weight": pd.Series(np.ones(99).tolist() + [1000])},
                {"df": df, "weight": pd.Series(np.ones(100))},
            ]

            import matplotlib.pyplot as plt

            plt.rcParams["figure.figsize"] = (20, 6) # (w, h)

            fig, axs = plt.subplots(1,3)

            # Without using weights
            plot_qq_categorical(dfs1, names=["self", "unadjusted", "target"], column="v1", axis=axs[0], weighted=False)
            # With weights
            plot_qq_categorical(dfs1, names=["self", "unadjusted", "target"], column="v1", axis=axs[1], weighted=True)
            # With label trimming if the text is longer than 3.
            plot_qq_categorical(dfs1, names=["self", "unadjusted", "target"], column="v1", axis=axs[2], weighted=True, label_threshold=3)
    """
    target = dfs[-1]["df"]
    target_weights = dfs[-1]["weight"]
    dfs = dfs[:-1]

    # pyre-fixme[6]
    target_plot_data = relative_frequency_table(target, column, target_weights)

    for ii, i in enumerate(dfs):
        d = i["df"]
        _w = i["weight"] if weighted else None

        # pyre-fixme[6]
        sample_plot_data = relative_frequency_table(d, column, _w)
        plot_data = pd.merge(
            sample_plot_data,
            target_plot_data,
            on=column,
            how="outer",
            suffixes=("_sample", "_target"),
        )

        axis = sns.scatterplot(
            x=plot_data.prop_sample, y=plot_data.prop_target, label=names[ii], ax=axis
        )

        if plot_data.shape[0] < label_threshold:
            for r in plot_data.itertuples():
                # pyre-ignore
                axis.text(x=r.prop_sample, y=r.prop_target, s=r[1])

    axis.set_ylim(-0.1, 1.1)
    axis.set_xlim(-0.1, 1.1)
    axis.plot(axis.get_xlim(), axis.get_ylim(), "--")
    axis.set_title(
        f"proportion-proportion plot of covar '{column}' in target vs sample"
    )
    _safe_show_legend(axis)


# TODO: add control (or just change) the default theme
# TODO: add a separate dist_type control parameter for categorical and numeric variables.
def seaborn_plot_dist(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: Optional[List[str]] = None,
    variables: Optional[List] = None,
    numeric_n_values_threshold: int = 15,
    weighted: bool = True,
    dist_type: Optional[Literal["qq", "hist", "kde", "ecdf"]] = None,
    return_axes: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> Union[List[plt.Axes], npt.NDArray, None]:
    """Plots to compare the weighted distributions of an arbitrary number of variables from
    an arbitrary number of DataFrames.

    Uses: :func:`plot_qq_categorical`, :func:`plot_qq`, :func:`plot_hist_kde`, :func:`plot_bar`.

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating by the column variable.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
        variables (Optional[List], optional): The list of variables to use, by default (None) will plot all of them.
        numeric_n_values_threshold (int, optional): How many unique values (or less) should be in a column so that it is considered to be a "category"? Defaults to 15.
            This is compared against the maximum number of distinct values (for each of the variables) across all DataFrames.
            Setting this value to 0 will disable this check.
        weighted (bool, optional): If to pass the weights from the dicts inside dfs. Defaults to True.
        dist_type (Optional[str], optional): can be "hist", "kde", or "qq". Defaults to None.
        return_axes (bool, optional): if to returns axes or None. Defaults to False,
        ylim (Optional[Tuple[float, float]], optional): A tuple with two float values representing the lower and upper limits of the y-axis.
            If not provided, the y-axis range is determined automatically. Defaults to None.
            Passed only for categorical variables and when dist_type is not 'qq' (i.e.: for bar plots).


    Returns:
        Union[List[plt.Axes], np.ndarray, None]: Returns None.
        However, if return_axes is True then either it returns a list or an np.array of matplotlib AxesSubplot (plt.Subplot).
        NOTE: There is no AxesSubplot class until one is invoked and created on the fly.
            See details here: https://stackoverflow.com/a/11690800/256662

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from balance.stats_and_plots.weighted_comparisons_plots import seaborn_plot_dist
            from numpy import random

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dfs1 = [
                {"df": df, "weight": pd.Series(np.ones(100))},
                {"df": df, "weight": pd.Series(np.ones(99).tolist() + [1000])},
                {"df": df, "weight": pd.Series(np.random.uniform(size=100))},
            ]

            seaborn_plot_dist(dfs1, names=["self", "unadjusted", "target"], dist_type = "qq")  # default
            seaborn_plot_dist(dfs1, names=["self", "unadjusted", "target"], dist_type = "hist")
            seaborn_plot_dist(dfs1, names=["self", "unadjusted", "target"], dist_type = "kde")
            seaborn_plot_dist(dfs1, names=["self", "unadjusted", "target"], dist_type = "ecdf")

            # With limiting the y axis range to (0,1)
            seaborn_plot_dist(dfs1, names=["self", "unadjusted", "target"], dist_type = "kde", ylim = (0,1))
    """
    if dist_type is None:
        if len(dfs) == 1:
            dist_type = "hist"
        else:
            dist_type = "qq"

    #  Choose set of variables to plot
    variables = choose_variables(*(d["df"] for d in dfs), variables=variables)
    logger.debug(f"plotting variables {variables}")

    #  Set up subplots
    f, axes = plt.subplots(len(variables), 1, figsize=(7, 7 * len(variables)))
    if not isinstance(axes, np.ndarray):  # If only one subplot
        axes = [axes]

    # TODO: patch choose_variables to return outcome_types from multiple_objects
    numeric_variables = dfs[0]["df"].select_dtypes(exclude=["object"]).columns.values

    for io, o in enumerate(variables):
        #  Find the maximum number of non-missing values of this variable accross
        #  all the dataframes
        n_values = max(len(set(rm_mutual_nas(d["df"].loc[:, o].values))) for d in dfs)
        if n_values == 0:
            logger.warning(f"No nonmissing values for variable '{o}', skipping")
            continue

        #  Plot categorical variables as histogram
        categorical = (o not in numeric_variables) or (
            n_values < numeric_n_values_threshold
        )

        if categorical:
            if dist_type == "qq":
                # pyre-fixme[6]
                plot_qq_categorical(dfs, names, o, axes[io], weighted)
            else:
                # pyre-fixme[6]
                plot_bar(dfs, names, o, axes[io], weighted, ylim=ylim)
        else:
            if dist_type == "qq":
                # pyre-fixme[6]
                plot_qq(dfs, names, o, axes[io], weighted)
            else:
                # pyre-fixme[6]
                plot_hist_kde(dfs, names, o, axes[io], weighted, dist_type)

    if return_axes:
        return axes
    # else (default) will return None


def set_xy_axes_to_use_the_same_lim(ax: plt.Axes) -> None:
    """Set the x and y axes limits to be the same.

    Done by taking the min and max from xlim and ylim and using these global min/max on both x and y axes.

    Args:
        ax (plt.Axes): matplotlib Axes object to draw the plot onto.

    Examples:
        ::

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.scatter(x= [1,2,3], y = [3,4,5])

            plt.figure(2)
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
            plt.scatter(x= [1,2,3], y = [3,4,5])
            set_xy_axes_to_use_the_same_lim(ax)
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))


################################################################################
#  plotly plots below
################################################################################


def plotly_plot_qq(
    dict_of_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    plot_it: bool = True,
    return_dict_of_figures: bool = False,
    **kwargs,
) -> Optional[Dict[str, go.Figure]]:
    """
    Plots interactive QQ plot of the given variables.

    Creates a plotly qq plot of the given variables from multiple DataFrames.
    This ASSUMES there is a df with key 'target'.


    Args:
        dict_of_dfs (Dict[str, pd.DataFrame]): The key is the name of the DataFrame (E.g.: self, unadjusted, target),
            and the value is the DataFrame that contains the variables that we want to plot.
        variables (List[str]): a list of variables to use for plotting.
        plot_it (bool, optional): If to plot the plots interactively instead of returning a dictionary. Defaults to True.
        return_dict_of_figures (bool, optional): If to return the dictionary containing the plots rather than just returning None. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the update_layout method of the plotly figure object. (e.g.: width and height are 700 and 450, and could be set using the kwargs).

    Returns:
        Optional[Dict[str, go.Figure]]: Dictionary containing plots if return_dict_of_figures is True. None otherwise.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from numpy import random
            from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_qq

            random.seed(96483)

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dict_of_dfs = {
                "self": pd.concat([df, pd.Series(random.random(size = 100) + 0.5, name = "weight")], axis = 1),
                "unadjusted": pd.concat([df, pd.Series(np.ones(99).tolist() + [1000], name = "weight")], axis = 1),
                "target": pd.concat([df, pd.Series(np.ones(100), name = "weight")], axis = 1),
            }

            # It won't work with "v1" since it is not numeric.
            plotly_plot_qq(dict_of_dfs, variables= ["v2", "v3"])
    """
    dict_of_qqs = {}
    for variable in variables:
        variable_specific_dict_of_plots = {}

        assert "target" in dict_of_dfs.keys(), "Must pass target"

        # Extract 'col1' because weighted_quantile will return a DataFrame
        # https://www.statsmodels.org/dev/_modules/statsmodels/stats/weightstats.html#DescrStatsW.quantile

        line_data = list(
            weighted_quantile(
                dict_of_dfs["target"][variable],
                np.arange(0, 1, 0.001),
                (
                    dict_of_dfs["target"]["weight"]
                    if "weight" in dict_of_dfs["target"].columns
                    else None
                ),
            )["col1"]
        )

        # Indicate if we have only self and target (without unadjusted)
        # since in this case the color of self should be red, since it's likely unadjusted.
        only_self_and_target = set(dict_of_dfs.keys()) == {"self", "target"}

        for name in dict_of_dfs:
            if name.lower() == "target":
                variable_specific_dict_of_plots[name] = go.Scatter(
                    x=line_data,
                    y=line_data,
                    showlegend=False,
                    line={"color": ("blue"), "width": 2, "dash": "dash"},
                )
            else:
                variable_specific_dict_of_plots[name] = go.Scatter(
                    x=line_data,
                    y=list(
                        weighted_quantile(
                            dict_of_dfs[name][variable],
                            np.arange(0, 1, 0.001),
                            (
                                dict_of_dfs[name]["weight"]
                                if "weight" in dict_of_dfs[name].columns
                                else None
                            ),
                        )["col1"]
                    ),
                    marker={
                        "color": _plotly_marker_color(
                            # pyre-ignore[6]: it cannot get to this point if name=="target".
                            name,
                            only_self_and_target,
                            "color",
                        )
                    },
                    mode="markers",
                    name=naming_legend(name, list(dict_of_dfs.keys())),
                    opacity=0.6,
                )

        data = [variable_specific_dict_of_plots[name] for name in dict_of_dfs]
        layout = {
            "title": f"QQ Plots of {variable}",
            "paper_bgcolor": "rgb(255, 255, 255)",
            "plot_bgcolor": "rgb(255, 255, 255)",
            "xaxis": {
                "zeroline": False,
                "linewidth": 1,
                "mirror": True,
                "title": variable,
            },
            "yaxis": {"zeroline": False, "linewidth": 1, "mirror": True},
        }
        fig = go.Figure(data=data, layout=layout)
        # Set the default PNG image size to 1400 x 1000 for when downloading the image
        fig.update_layout(**kwargs)
        dict_of_qqs[variable] = fig
        if plot_it:
            offline.iplot(fig)
    if return_dict_of_figures:
        return dict_of_qqs


def plotly_plot_density(
    dict_of_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    plot_it: bool = True,
    return_dict_of_figures: bool = False,
    plot_width: int = 800,
    **kwargs,
) -> Optional[Dict[str, go.Figure]]:
    """
    Plots interactive density plots of the given variables using kernel density estimation.

    Creates a plotly plot of the kernel density estimate for each variable in the given list
    across multiple DataFrames. The function assumes there is a DataFrame with the key 'target'.
    The density plot shows the distribution of the variable for each DataFrame in the dictionary.
    It looks for a `weights` column and uses it to normalize the data. If no weight column is found, it assumes all weights are equal to 1.
    It relies on the seaborn library to create the KDE (`sns.kdeplot`).

    Args:
        dict_of_dfs (Dict[str, pd.DataFrame]): A dictionary where each key is a name for the DataFrame
            and the value is the DataFrame that contains the variables to plot.
        variables (List[str]): A list of variables to plot.
        plot_it (bool, optional): Whether to plot the figures interactively using plotly. Defaults to True.
        return_dict_of_figures (bool, optional): Whether to return a dictionary of plotly figures.
            Defaults to False.
        plot_width (int, optional): The width of the plot in pixels. Defaults to 800.
        **kwargs: Additional keyword arguments to pass to the update_layout method of the plotly figure object. (e.g.: width and height are 700 and 450, and could be set using the kwargs).

    Returns:
        Optional[Dict[str, go.Figure]]: A dictionary containing plotly figures for each variable
        in the given list if `return_dict_of_figures` is True. Otherwise, returns None.

    Examples:
        ::
            import numpy as np
            import pandas as pd
            from numpy import random
            from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_density, plot_dist

            random.seed(96483)

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dict_of_dfs = {
                "self": pd.concat([df, pd.Series(random.random(size = 100) + 0.5, name = "weight")], axis = 1),
                "unadjusted": pd.concat([df, pd.Series(np.ones(99).tolist() + [1000], name = "weight")], axis = 1),
                "target": pd.concat([df, pd.Series(np.ones(100), name = "weight")], axis = 1),
            }

            # It won't work with "v1" since it is not numeric.
            plotly_plot_density(dict_of_dfs, variables= ["v2", "v3"], plot_width = 550)


            # The above gives the same results as:
            dfs1 = [
                {"df": df, "weight": dict_of_dfs['self']["weight"]},
                {"df": df, "weight": dict_of_dfs['unadjusted']["weight"]},
                {"df": df, "weight": dict_of_dfs['target']["weight"]},
            ]
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn", dist_type = "kde", variables= ["v2", "v3"])


            # This gives the same shape of plots (notice how we must have the column "weight" for the plots to work)
            df = pd.DataFrame({
                'group': ('a', 'b', 'c', 'c'),
                'v1': (1, 2, 3, 4),
            })

            dfs1 = [{"df": pd.DataFrame(pd.Series([1,2,2,2,3,4,5,5,7,8,9,9,9,9,5,2,5,4,4,4], name = "v1")), "weight": None}, {"df": df, "weight": pd.Series((200, 1, 0, 200000))}]
            # dfs1[1]{'df'}

            dict_of_dfs = {
                "self": dfs1[0]['df'], # pd.concat([df, pd.Series(random.random(size = 100) + 0.5, name = "weight")], axis = 1),
                "target": pd.concat([dfs1[1]['df'], pd.Series(dfs1[1]["weight"], name = "weight")], axis = 1),
            }

            plotly_plot_density(dict_of_dfs, variables= ["v1"], plot_width = 550)

            plot_dist(dfs1, names=["self", "target"], library="seaborn", dist_type = "kde", variables= ["v1"],numeric_n_values_threshold = 1)

    """
    dict_of_density_plots = {}

    for variable in variables:
        data = []

        # Indicate if we have only self and target (without unadjusted)
        # since in this case the color of self should be red, since it's likely unadjusted.
        only_self_and_target = set(dict_of_dfs.keys()) == {"self", "target"}

        for name, df in dict_of_dfs.items():
            if "weight" in df.columns:
                weights = df["weight"]
                # TODO: verify if this normalization to sum to 1 is needed (if so, how come we don't do it when _w is None)?
                weights = weights / weights.sum()  # normalize weights by sum of weights
            else:
                weights = np.ones(len(df))

            # Convert the data to long format
            long_df = pd.DataFrame({"value": df[variable], "weight": weights})

            # Replace KDE calculation with sns.kdeplot
            with plt.xkcd():
                plt.figure()
                ax = sns.kdeplot(
                    data=long_df, x="value", weights="weight", common_norm=False
                )
                x = ax.get_lines()[-1].get_xdata()
                y = ax.get_lines()[-1].get_ydata()
                # print(name)
                # print(x)
                # print(y)
                plt.close()

            trace = go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=naming_legend(name, list(dict_of_dfs.keys())),
                line={
                    # pyre-ignore[6]: it cannot get to this point if name=="target".
                    "color": _plotly_marker_color(name, only_self_and_target, "line"),
                    "width": 1.5,
                },
            )
            data.append(trace)

        layout = {
            "title": f"Density Plots of '{variable}'",
            "paper_bgcolor": "rgb(255, 255, 255)",
            "plot_bgcolor": "rgb(255, 255, 255)",
            "width": plot_width,
            "xaxis": {
                "title": variable,
                "gridcolor": "rgba(128, 128, 128, 0.5)",
                "gridwidth": 1,
                "showgrid": True,
            },
            "yaxis": {
                "title": "Density",
                "gridcolor": "rgba(128, 128, 128, 0.5)",
                "gridwidth": 1,
                "showgrid": True,
            },
        }
        fig = go.Figure(data=data, layout=layout)
        dict_of_density_plots[variable] = fig

        fig.update_layout(**kwargs)

        if plot_it:
            offline.iplot(fig)

    if return_dict_of_figures:
        return dict_of_density_plots


def plotly_plot_bar(
    dict_of_dfs: Dict[str, pd.DataFrame],
    variables: List[str],
    plot_it: bool = True,
    return_dict_of_figures: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Optional[Dict[str, go.Figure]]:
    """
    Plots interactive bar plots of the given variables (with optional control over the y-axis limits).

    Args:
        dict_of_dfs (Dict[str, pd.DataFrame]): A dictionary with keys as names of the DataFrame (e.g., 'self', 'unadjusted', 'target'),
            and values as the DataFrames containing the variables to plot.
        variables (List[str]): A list of variables to use for plotting.
        plot_it (bool, optional): If True, plots the graphs interactively instead of returning a dictionary. Defaults to True.
        return_dict_of_figures (bool, optional): If True, returns the dictionary containing the plots rather than just returning None. Defaults to False.
        ylim (Optional[Tuple[float, float]], optional): A tuple with two float values representing the lower and upper limits of the y-axis.
            If not provided, the y-axis range is determined automatically. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the update_layout method of the plotly figure object. (e.g.: width and height are 700 and 450, and could be set using the kwargs).

    Returns:
        Optional[Dict[str, go.Figure]]: Dictionary containing plots if return_dict_of_figures is True. None otherwise.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from numpy import random
            from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_bar

            random.seed(96483)

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dict_of_dfs = {
                "self": pd.concat([df, pd.Series(random.random(size = 100) + 0.5, name = "weight")], axis = 1),
                "unadjusted": pd.concat([df, pd.Series(np.ones(99).tolist() + [1000], name = "weight")], axis = 1),
                "target": pd.concat([df, pd.Series(np.ones(100), name = "weight")], axis = 1),
            }

            # It can work with "v2" and "v3", but it would be very sparse
            plotly_plot_bar(dict_of_dfs, variables= ["v1"])

            # Plots the same as above, but this time the range of the yaxis is from 0 to 1.
            plotly_plot_bar(dict_of_dfs, variables= ["v1"], ylim = (0,1))
    """
    dict_of_bars = {}
    for variable in variables:
        # for each variable
        variable_specific_dict_of_plots = {}
        # create plot for each df using that variable

        # Indicate if we have only self and target (without unadjusted)
        # since in this case the color of self should be red, since it's likely unadjusted.
        only_self_and_target = set(dict_of_dfs.keys()) == {"self", "target"}

        # filter dict_of_dfs
        for name in dict_of_dfs:
            df_plot_data = relative_frequency_table(
                dict_of_dfs[name],
                variable,
                (
                    dict_of_dfs[name]["weight"]
                    if "weight" in dict_of_dfs[name].columns
                    else None
                ),
            )

            variable_specific_dict_of_plots[name] = go.Bar(
                x=list(df_plot_data[variable]),
                y=list(df_plot_data["prop"]),
                marker={
                    # pyre-ignore[6]: it cannot get to this point if name=="target".
                    "color": _plotly_marker_color(name, only_self_and_target, "color"),
                    "line": {
                        "color": _plotly_marker_color(
                            # pyre-ignore[6]: it cannot get to this point if name=="target".
                            name,
                            only_self_and_target,
                            "line",
                        ),
                        "width": 1.5,
                    },
                },
                opacity=0.6,
                name=naming_legend(name, list(dict_of_dfs.keys())),
                visible=True,
            )
        data = [variable_specific_dict_of_plots[name] for name in dict_of_dfs]

        layout = go.Layout(
            title=f"Sample Vs Target {variable}",
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor="rgb(255, 255, 255)",
            xaxis={"title": variable},
            yaxis={
                "title": "Proportion of Total",
                "range": ylim if ylim is not None else None,
            },
        )

        fig = go.Figure(data=data, layout=layout)

        fig.update_layout(**kwargs)

        dict_of_bars[variable] = fig
        if plot_it:
            offline.iplot(fig)
    if return_dict_of_figures:
        return dict_of_bars


# TODO: add more plots other than qq for numeric (e.g.: hist, ecdf,)
# see https://plotly.com/python/distplot/
# Notice that these plots do not support the 'weight' column, so it requires a different approach.
# See the plotly_plot_density solution that uses seaborn's output
def plotly_plot_dist(
    dict_of_dfs: Dict[str, pd.DataFrame],
    variables: Optional[List[str]] = None,
    numeric_n_values_threshold: int = 15,
    weighted: bool = True,
    dist_type: Optional[Literal["kde", "qq"]] = None,
    plot_it: bool = True,
    return_dict_of_figures: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Optional[Dict[str, go.Figure]]:
    """
    Plots interactive distribution plots (qq and bar plots) of the given variables.

    The plots compare the weighted distributions of an arbitrary number
    of variables from an arbitrary number of DataFrames.
    Numeric variables are plotted as either qq's using :func:`plotly_plot_qq`, or as kde desnity plots using :func:`plotly_plot_density`.
    categorical variables as barplots using :func:`plotly_plot_bar`.

    Args:
        dict_of_dfs (Dict[str, pd.DataFrame]): The key is the name of the DataFrame (E.g.: self, unadjusted, target),
            and the value is the DataFrame that contains the variables that we want to plot.
        variables (Optional[List[str]], optional): a list of variables to use for plotting. Defaults (i.e.: if None) is to use the list of all variables.
        numeric_n_values_threshold (int, optional): How many numbers should be in a column so that it is considered to be a "category"? Defaults to 15.
        weighted (bool, optional): If to use the weights with the plots. Defaults to True.
        dist_type (Optional[Literal["kde", "qq"]], optional): The type of plot to draw (relevant only for numerical variables). Defaults to None (which fallbacks to "kde").
        plot_it (bool, optional): If to plot the plots interactively instead of returning a dictionary. Defaults to True.
        return_dict_of_figures (bool, optional): If to return the dictionary containing the plots rather than just returning None. Defaults to False.
            If returned - the dictionary is of plots.
            Keys in this dictionary are the variable names for each plot.
            Values are plotly plot objects plotted like:
                offline.iplot(dict_of_all_plots['age'])
            Or simply:
                dict_of_all_plots['age']
        ylim (Optional[Tuple[float, float]], optional): A tuple with two float values representing the lower and upper limits of the y-axis.
            If not provided, the y-axis range is determined automatically. Defaults to None.
            passed to bar plots only.
        **kwargs: Additional keyword arguments to pass to the update_layout method of the plotly figure object. (e.g.: width and height are 700 and 450, and could be set using the kwargs).

    Returns:
        Optional[Dict[str, go.Figure]]: Dictionary containing plots if return_dict_of_figures is True. None otherwise.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from numpy import random
            from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_dist

            random.seed(96483)

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dict_of_dfs = {
                "self": pd.concat([df, pd.Series(random.random(size = 100) + 0.5, name = "weight")], axis = 1),
                "unadjusted": pd.concat([df, pd.Series(np.ones(99).tolist() + [1000], name = "weight")], axis = 1),
                "target": pd.concat([df, pd.Series(np.ones(100), name = "weight")], axis = 1),
            }

            plotly_plot_dist(dict_of_dfs)

            # Make sure the bar plot is plotted with y in the range of 0 to 1.
            plotly_plot_dist(dict_of_dfs, ylim = (0,1))

            # See the qqplots version
            plotly_plot_dist(dict_of_dfs, dist_type="qq")
    """
    dict_of_all_plots = {}
    #  Choose set of variables to plot
    variables = choose_variables(
        *(dict_of_dfs[name] for name in dict_of_dfs), variables=variables
    )

    variables = [v for v in variables if v != "weight"]

    logger.debug(f"plotting variables {variables}")

    # TODO: patch choose_variables to return outcome_types from multiple_objects
    # find numeric values, using 'sample' df. If for some reason sample is
    # not an option, use a random df.
    if "sample" in dict_of_dfs:
        numeric_variables = (
            dict_of_dfs["sample"].select_dtypes(exclude=["object"]).columns.values
        )
    else:
        numeric_variables = (
            dict_of_dfs[random.choice(list(dict_of_dfs.keys()))]
            .select_dtypes(exclude=["object"])
            .columns.values
        )

    for _, o in enumerate(variables):
        #  Find the maximum number of non-missing values of this variable accross
        #  all the dataframes

        # Look at the first element in the dict: (name, type and values)
        logger.debug(list(dict_of_dfs.keys())[0])
        logger.debug(type(dict_of_dfs[list(dict_of_dfs.keys())[0]].loc[:, o].values))
        logger.debug(dict_of_dfs[list(dict_of_dfs.keys())[0]].loc[:, o].values)

        n_values = max(
            len(set(rm_mutual_nas(dict_of_dfs[name].loc[:, o].values)))
            for name in dict_of_dfs
        )

        if n_values == 0:
            logger.warning(f"No nonmissing values for variable '{o}', skipping")

            continue

        #  Plot categorical variables as histogram
        categorical = (o not in numeric_variables) or (
            n_values < numeric_n_values_threshold
        )

        if (dist_type is None) or dist_type == "kde":
            plotly_numeric_plot = plotly_plot_density
        elif dist_type == "qq":
            plotly_numeric_plot = plotly_plot_qq
        else:
            raise NotImplementedError(
                f"dist_type of type {dist_type} is not implemented."
            )

        # the below functions will create plotly plots
        if categorical:
            dict_of_plot = plotly_plot_bar(
                dict_of_dfs, [o], plot_it, return_dict_of_figures, ylim=ylim, **kwargs
            )
        else:
            # plotly_plot_density
            dict_of_plot = plotly_numeric_plot(
                dict_of_dfs, [o], plot_it, return_dict_of_figures, **kwargs
            )
        # the below functions will add the plotly dict outputs
        # to the dictionary 'dict_of_all_plots' (if return_dict_of_figures is True).
        if dict_of_plot is not None and return_dict_of_figures:
            dict_of_all_plots.update(dict_of_plot)

    if return_dict_of_figures:
        return dict_of_all_plots


def naming_legend(object_name: str, names_of_dfs: List[str]) -> str:
    """Returns a name for a legend of a plot given the other dfs.
    If one of the dfs we would like to plot is "unadjusted", it means
    that the Sample object contains the adjusted object as self.
    If not, then the self object is sample.

    Args:
        object_name (str): the name of the object to plot.
        names_of_dfs (List[str]): the names of the other dfs to plot.

    Returns:
        str: a string with the desired name

    Examples:
        ::

            naming_legend('self', ['self', 'target', 'unadjusted']) #'adjusted'
            naming_legend('unadjusted', ['self', 'target', 'unadjusted']) #'sample'
            naming_legend('self', ['self', 'target']) #'sample'
            naming_legend('other_name', ['self', 'target']) #'other_name'
    """
    if object_name in names_of_dfs:
        return {
            "unadjusted": "sample",
            "self": "adjusted" if "unadjusted" in names_of_dfs else "sample",
            "target": "population",
        }[object_name]
    else:
        return object_name


# TODO: set colors of the lines and dots in plots to be fixed by the object name
# (sample, target and adjusted sample).
# See the examples in Balance wiki Diagnostic_Plots page for the current plots
# (the original sample is green when we don't plot the adjusted and red when we are)


################################################################################
#  A master plotting function to navigate between seaborn and plotly plots
################################################################################


def plot_dist(
    dfs: List[Dict[str, Union[pd.DataFrame, pd.Series]]],
    names: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    numeric_n_values_threshold: int = 15,
    weighted: bool = True,
    dist_type: Optional[Literal["kde", "hist", "qq", "ecdf"]] = None,
    library: Literal["plotly", "seaborn"] = "plotly",
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Union[Union[List, npt.NDArray], Dict[str, go.Figure], None]:
    """Plots the variables of a DataFrame by using either seaborn or plotly.

    If using plotly then using kde (or qq) plots for numeric variables and bar plots for categorical variables. Uses :func:`plotly_plot_dist`.
    If using seaborn then various types of plots are possible for the variables (see dist_type for details). Uses :func:`seaborn_plot_dist`

    Args:
        dfs (List[Dict[str, Union[pd.DataFrame, pd.Series]]]): a list (of length 1 or more) of dictionaries which describe the DataFrames and weights
            The structure is as follows:
            [
                {'df': pd.DataFrame(...), "weight": pd.Series(...)},
                ...
            ]
            The 'df' is a DataFrame which includes the column name that was supplied through 'column'.
            The "weight" is a pd.Series of weights that are used when aggregating the variable using :func:`relative_frequency_table`.
        names (List[str]): a list of the names of the DataFrames that are plotted. E.g.: ['adjusted', 'unadjusted', 'target']
            If None, then all DataFrames will be plotted, but only if library == "seaborn". (TODO: to remove this restriction)
        variables (Optional[List[str]], optional): a list of variables to use for plotting. Default (i.e.: if None) is to use the list of all variables.
        numeric_n_values_threshold (int, optional): How many numbers should be in a column so that it is considered to be a "category"? Defaults to 15.
        weighted (bool, optional): If to use the weights with the plots. Defaults to True.
        dist_type (Literal["kde", "hist", "qq", "ecdf"], optional): The type of plot to draw. The 'qq' and 'kde' options are available for library="plotly",
            While all options are available if using library="seaborn". Defaults to "kde".
        library (Literal["plotly", "seaborn"], optional): Whichever library to use for the plot. Defaults to "plotly".
        ylim (Optional[Tuple[float, float]], optional): A tuple with two float values representing the lower and upper limits of the y-axis.
            If not provided, the y-axis range is determined automatically. Defaults to None.
            passed to bar plots only.
        **kwargs: Additional keyword arguments to pass to plotly_plot_dist or seaborn_plot_dist.

    Raises:
        ValueError: if library is not in ("plotly", "seaborn").

    Returns:
        Union[Union[List, np.ndarray], Dict[str, go.Figure], None]:
            If library="plotly" then returns a dictionary containing plots if return_dict_of_figures is True. None otherwise.
            If library="seaborn" then returns None, unless return_axes is True. Then either a list or an np.array of matplotlib axis.

    Examples:
        ::

            import numpy as np
            import pandas as pd
            from numpy import random
            from balance.stats_and_plots.weighted_comparisons_plots import plotly_plot_bar

            random.seed(96483)

            df = pd.DataFrame({
                'v1': random.random_integers(11111, 11114, size=100).astype(str),
                'v2': random.normal(size = 100),
                'v3': random.uniform(size = 100),
            }).sort_values(by=['v2'])

            dfs1 = [
                {"df": df, "weight": pd.Series(random.random(size = 100) + 0.5)},
                {"df": df, "weight": pd.Series(np.ones(99).tolist() + [1000])},
                {"df": df, "weight": pd.Series(np.ones(100))},
            ]


            from balance.stats_and_plots.weighted_comparisons_plots import plot_dist

            # defaults to plotly with bar and qq plots. Returns None.
            plot_dist(dfs1, names=["self", "unadjusted", "target"])

            # Using seaborn, deafults to kde plots
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn") # like using dist_type = "kde"
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn", dist_type = "hist")
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn", dist_type = "qq")
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn", dist_type = "ecdf")

            plot_dist(dfs1, names=["self", "unadjusted", "target"], ylim = (0,1))
            plot_dist(dfs1, names=["self", "unadjusted", "target"], library="seaborn", dist_type = "qq", ylim = (0,1))
    """
    if library not in ("plotly", "seaborn"):
        raise ValueError(f"library must be either 'plotly' or 'seaborn', is {library}")

    #  Set default names for samples
    # TODO: this will work only with seaborn. Will need to change to something that also works for plotly.
    if names is None:
        names = [f"sample {i}" for i in range(1, len(dfs) + 1)]

    if library == "seaborn":
        return seaborn_plot_dist(
            dfs=dfs,
            names=names,
            variables=variables,
            numeric_n_values_threshold=numeric_n_values_threshold,
            weighted=weighted,
            dist_type=dist_type,
            ylim=ylim,
            **kwargs,
        )
    elif library == "plotly":
        dict_of_dfs = dict(
            zip(
                names,
                (
                    pd.concat((d["df"], pd.Series(d["weight"], name="weight")), axis=1)
                    for d in dfs
                ),
            )
        )

        if dist_type is not None:
            logger.warning("plotly plots ignore dist_type. Consider library='seaborn'")

        return plotly_plot_dist(
            dict_of_dfs,
            variables,
            numeric_n_values_threshold,
            weighted,
            # pyre-ignore[6]: plotly_plot_dist will raise a NotImplemented error if dist_type is not None, 'kde', or 'qq'
            dist_type=dist_type,
            ylim=ylim,
            **kwargs,
        )


# TODO: add plots to compare ASMD
