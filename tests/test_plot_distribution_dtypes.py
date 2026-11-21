# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from balance.stats_and_plots.weighted_comparisons_plots import (
    DataFrameWithWeight,
    plot_dist,
)
from balance.stats_and_plots.weighted_stats import relative_frequency_table
from matplotlib.axes import Axes


@pytest.fixture
def plot_data() -> list[DataFrameWithWeight]:
    frame = pd.DataFrame(
        {
            "text": pd.Series(["a", "b", "a", "c"], dtype=object),
            "string": pd.array(["a", "b", "a", "c"], dtype="string"),
            "category": pd.Categorical([1, 2, 1, 3], categories=[1, 2, 3, 4]),
            "flag": pd.array([True, False, True, False], dtype="boolean"),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return [{"df": frame, "weight": None}]


def test_ascii_distribution_uses_categorical_dtypes(
    plot_data: list[DataFrameWithWeight],
) -> None:
    result = plot_dist(plot_data, library="balance", numeric_n_values_threshold=0)

    assert isinstance(result, str)
    for column in ("text", "string", "category", "flag"):
        assert f"=== {column} (categorical) ===" in result
    assert "=== value (numeric, comparative) ===" in result


def test_seaborn_distribution_uses_categorical_dtypes(
    plot_data: list[DataFrameWithWeight],
) -> None:
    try:
        result = plot_dist(
            plot_data,
            library="seaborn",
            dist_type="hist",
            numeric_n_values_threshold=0,
            return_axes=True,
        )

        assert isinstance(result, list)
        assert all(isinstance(axis, Axes) for axis in result)
        assert {axis.get_title() for axis in result} == {
            "barplot of covar 'text'",
            "barplot of covar 'string'",
            "barplot of covar 'category'",
            "barplot of covar 'flag'",
            "distribution plot of covar 'value'",
        }
    finally:
        plt.close("all")


@pytest.mark.parametrize("name", ["sample", "self"])
def test_plotly_distribution_uses_categorical_dtypes(
    plot_data: list[DataFrameWithWeight], name: str
) -> None:
    result = plot_dist(
        plot_data,
        names=[name],
        library="plotly",
        numeric_n_values_threshold=0,
        plot_it=False,
        return_dict_of_figures=True,
    )

    assert isinstance(result, dict)
    assert {column: figure.data[0].type for column, figure in result.items()} == {
        "text": "bar",
        "string": "bar",
        "category": "bar",
        "flag": "bar",
        "value": "scatter",
    }
    category_trace = result["category"].data[0]
    assert list(category_trace.x) == [1, 2, 3]
    assert list(category_trace.y) == [0.5, 0.25, 0.25]
    assert result["category"].layout.xaxis.type == "category"


@pytest.mark.parametrize("library", ["balance", "plotly", "seaborn"])
def test_weighted_distributions_with_different_dtypes_across_frames(
    library: Literal["balance", "plotly", "seaborn"],
) -> None:
    dfs: list[DataFrameWithWeight] = [
        {
            "df": pd.DataFrame({"group": [1, 2]}),
            "weight": pd.Series([1.0, 3.0]),
        },
        {
            "df": pd.DataFrame({"group": pd.Categorical([2, 3], categories=[1, 2, 3])}),
            "weight": pd.Series([1.0, 1.0]),
        },
    ]
    options_by_library = {
        "balance": {},
        "plotly": {"plot_it": False, "return_dict_of_figures": True},
        "seaborn": {"return_axes": True},
    }

    try:
        result = plot_dist(
            dfs,
            names=["sample", "reference"],
            library=library,
            numeric_n_values_threshold=0,
            **options_by_library[library],
        )

        if library == "balance":
            assert isinstance(result, str)
            assert "=== group (categorical) ===" in result
            assert "75.0%" in result
        elif library == "plotly":
            assert isinstance(result, dict)
            sample_trace, target_trace = result["group"].data
            assert sample_trace.type == target_trace.type == "bar"
            assert (list(sample_trace.x), list(sample_trace.y)) == (
                [1, 2],
                [0.25, 0.75],
            )
            assert (list(target_trace.x), list(target_trace.y)) == ([2, 3], [0.5, 0.5])
        else:
            assert isinstance(result, list)
            axis = result[0]
            assert isinstance(axis, Axes)
            assert axis.get_title().startswith("proportion-proportion plot")
            points = axis.collections[0].get_offsets()
            assert {tuple(point) for point in points} == {
                (0.25, 0.0),
                (0.75, 0.5),
                (0.0, 0.5),
            }
    finally:
        plt.close("all")


@pytest.mark.parametrize("column", ["group", "Freq"])
def test_plot_frequency_weights_do_not_collide_with_input_columns(column: str) -> None:
    frame = pd.DataFrame(
        {"group": ["a", "b", "a"], "Freq": pd.Categorical(["a", "b", "a"])},
        index=pd.Index([10, 20, 30], name="category"),
    )
    weights = pd.Series([3.0, 2.0, 5.0], index=[30, 10, 20])

    result = relative_frequency_table(frame, column, weights)

    assert result[column].tolist() == ["a", "b"]
    assert result["prop"].tolist() == [0.5, 0.5]
