# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_plots import (
    DataFrameWithWeight,
    plot_dist,
)
from balance.stats_and_plots.weighted_stats import relative_frequency_table
from balance.util import _assert_type
from matplotlib.axes import Axes


def _plot_data() -> list[DataFrameWithWeight]:
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


class PlotDistributionDtypesTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_ascii_distribution_uses_categorical_dtypes(self) -> None:
        result = plot_dist(
            _plot_data(), library="balance", numeric_n_values_threshold=0
        )

        output = _assert_type(result, str)
        for column in ("text", "string", "category", "flag"):
            self.assertIn(f"=== {column} (categorical) ===", output)
        self.assertIn("=== value (numeric, comparative) ===", output)

    def test_ascii_distribution_routes_complex_values_to_categories(self) -> None:
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype.__name__):
                frame = pd.DataFrame({"value": np.array([1 + 2j, 2 + 1j], dtype=dtype)})

                result = plot_dist(
                    [{"df": frame, "weight": None}],
                    library="balance",
                    numeric_n_values_threshold=0,
                )

                output = _assert_type(result, str)
                self.assertIn("=== value (categorical) ===", output)
                self.assertIn("(1+2j)", output)
                self.assertIn("(2+1j)", output)

    def test_seaborn_distribution_uses_categorical_dtypes(self) -> None:
        result = plot_dist(
            _plot_data(),
            library="seaborn",
            dist_type="hist",
            numeric_n_values_threshold=0,
            return_axes=True,
        )

        axes = _assert_type(result, list)
        self.assertTrue(all(isinstance(axis, Axes) for axis in axes))
        self.assertEqual(
            {axis.get_title() for axis in axes},
            {
                "barplot of covar 'text'",
                "barplot of covar 'string'",
                "barplot of covar 'category'",
                "barplot of covar 'flag'",
                "distribution plot of covar 'value'",
            },
        )

    def test_plotly_distribution_uses_categorical_dtypes(self) -> None:
        for name in ("sample", "self"):
            with self.subTest(name=name):
                result = plot_dist(
                    _plot_data(),
                    names=[name],
                    library="plotly",
                    numeric_n_values_threshold=0,
                    plot_it=False,
                    return_dict_of_figures=True,
                )

                figures = _assert_type(result, dict)
                self.assertEqual(
                    {column: figure.data[0].type for column, figure in figures.items()},
                    {
                        "text": "bar",
                        "string": "bar",
                        "category": "bar",
                        "flag": "bar",
                        "value": "scatter",
                    },
                )
                category_trace = figures["category"].data[0]
                self.assertEqual(list(category_trace.x), [1, 2, 3])
                self.assertEqual(list(category_trace.y), [0.5, 0.25, 0.25])
                self.assertEqual(figures["category"].layout.xaxis.type, "category")

    def test_plotly_bars_keep_level_order_when_first_frame_lacks_a_level(
        self,
    ) -> None:
        levels = ["low", "medium", "high"]
        cases: dict[str, tuple[pd.Series, pd.Series, list[Any]]] = {
            "numeric codes": (pd.Series([1, 3]), pd.Series([1, 2, 3]), [1, 2, 3]),
            "ordered categorical": (
                pd.Series(
                    pd.Categorical(["low", "high"], categories=levels, ordered=True)
                ),
                pd.Series(pd.Categorical(levels, categories=levels, ordered=True)),
                levels,
            ),
        }
        for case, (sample_values, target_values, expected_order) in cases.items():
            with self.subTest(case=case):
                dfs: list[DataFrameWithWeight] = [
                    {"df": pd.DataFrame({"group": sample_values}), "weight": None},
                    {"df": pd.DataFrame({"group": target_values}), "weight": None},
                ]

                result = plot_dist(
                    dfs,
                    names=["self", "target"],
                    library="plotly",
                    plot_it=False,
                    return_dict_of_figures=True,
                )

                xaxis = _assert_type(result, dict)["group"].layout.xaxis
                self.assertEqual(xaxis.categoryorder, "array")
                self.assertEqual(list(xaxis.categoryarray), expected_order)

    def test_weighted_distributions_with_different_dtypes_across_frames(
        self,
    ) -> None:
        dfs: list[DataFrameWithWeight] = [
            {
                "df": pd.DataFrame({"group": [1, 2]}),
                "weight": pd.Series([1.0, 3.0]),
            },
            {
                "df": pd.DataFrame(
                    {"group": pd.Categorical([2, 3], categories=[1, 2, 3])}
                ),
                "weight": pd.Series([1.0, 1.0]),
            },
        ]
        names = ["sample", "reference"]

        with self.subTest(library="balance"):
            result = plot_dist(
                dfs, names=names, library="balance", numeric_n_values_threshold=0
            )

            output = _assert_type(result, str)
            self.assertIn("=== group (categorical) ===", output)
            self.assertIn("75.0%", output)

        with self.subTest(library="plotly"):
            result = plot_dist(
                dfs,
                names=names,
                library="plotly",
                numeric_n_values_threshold=0,
                plot_it=False,
                return_dict_of_figures=True,
            )

            sample_trace, target_trace = _assert_type(result, dict)["group"].data
            self.assertEqual(sample_trace.type, "bar")
            self.assertEqual(target_trace.type, "bar")
            self.assertEqual(list(sample_trace.x), [1, 2])
            self.assertEqual(list(sample_trace.y), [0.25, 0.75])
            self.assertEqual(list(target_trace.x), [2, 3])
            self.assertEqual(list(target_trace.y), [0.5, 0.5])

        with self.subTest(library="seaborn"):
            result = plot_dist(
                dfs,
                names=names,
                library="seaborn",
                numeric_n_values_threshold=0,
                return_axes=True,
            )

            axis = _assert_type(_assert_type(result, list)[0], Axes)
            self.assertTrue(axis.get_title().startswith("proportion-proportion plot"))
            points = np.asarray(axis.collections[0].get_offsets()).tolist()
            self.assertEqual(
                {tuple(point) for point in points},
                {(0.25, 0.0), (0.75, 0.5), (0.0, 0.5)},
            )

    def test_plot_frequency_weights_do_not_collide_with_input_columns(self) -> None:
        frame = pd.DataFrame(
            {"group": ["a", "b", "a"], "Freq": pd.Categorical(["a", "b", "a"])},
            index=pd.Index([10, 20, 30], name="category"),
        )
        weights = pd.Series([3.0, 2.0, 5.0], index=[30, 10, 20])

        for column in ("group", "Freq"):
            with self.subTest(column=column):
                result = relative_frequency_table(frame, column, weights)

                self.assertEqual(result[column].tolist(), ["a", "b"])
                self.assertEqual(result["prop"].tolist(), [0.5, 0.5])
