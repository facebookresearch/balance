# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_plots import plot_qq, plotly_plot_qq
from balance.stats_and_plots.weighted_stats import weighted_quantile
from balance.util import _assert_type


class QuantileZeroWeightRegressionTest(unittest.TestCase):
    def test_nondefault_value_index_and_array_weights_are_positional(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
        self.assertEqual(
            weighted_quantile(values, [0.5], np.array([1.0, 0.0, 1.0])).iloc[0, 0], 2.0
        )

    def test_scalar_quantiles_with_and_without_positive_mass(self) -> None:
        for weights, expected in (
            ([1.0, 1.0, 1.0], 2.0),
            ([0.0] * 3, np.nan),
            ([np.nan] * 3, np.nan),
        ):
            with self.subTest(weights=weights), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = weighted_quantile([1.0, 2.0, 3.0], 0.5, weights)
                self.assertEqual(result.shape, (1, 1))
                self.assertEqual(result.index.tolist(), [0.5])
                if np.isnan(expected):
                    self.assertTrue(np.isnan(result.iloc[0, 0]))
                else:
                    self.assertEqual(result.iloc[0, 0], expected)

    def test_undefined_quantiles_preserve_frame_shape(self) -> None:
        for weights in ([0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]):
            with self.subTest(weights=weights):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", UserWarning)
                    result = weighted_quantile(
                        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
                        [0.25, 0.5],
                        weights,
                    )
                self.assertEqual(result.shape, (2, 2))
                self.assertEqual(result.index.tolist(), [0.25, 0.5])
                self.assertEqual(result.columns.tolist(), ["col1", "col2"])
                self.assertTrue(np.asarray(result.isna(), dtype=bool).all())
                self.assertTrue(
                    any(issubclass(w.category, UserWarning) for w in caught)
                )

    def test_qq_plots_accept_undefined_sample_quantiles(self) -> None:
        target = pd.DataFrame({"x": [1.0, 2.0, 3.0], "weight": [1.0, 1.0, 1.0]})
        for weights in ([0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]):
            with self.subTest(weights=weights), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sample = pd.DataFrame({"x": [1.0, 2.0, 3.0], "weight": weights})
                figures = _assert_type(
                    plotly_plot_qq(
                        {"self": sample, "target": target},
                        ["x"],
                        plot_it=False,
                        return_dict_of_figures=True,
                    ),
                    dict,
                )
                self.assertTrue(
                    np.isnan(np.asarray(figures["x"].data[0].y, dtype=float)).all()
                )
                figure, axis = plt.subplots()
                try:
                    plot_qq(
                        [
                            {"df": sample, "weight": sample["weight"]},
                            {"df": target, "weight": target["weight"]},
                        ],
                        ["self", "target"],
                        "x",
                        axis=axis,
                    )
                    self.assertEqual(
                        sum(
                            len(collection.get_offsets())
                            for collection in axis.collections
                        ),
                        0,
                    )
                finally:
                    plt.close(figure)

    def test_ignores_noncontributing_intermediate_values(self) -> None:
        self.assertEqual(
            weighted_quantile([0.0, 100.0, 200.0], [0.5], [1.0, 0.0, 1.0]).iloc[0, 0],
            100.0,
        )
        self.assertEqual(
            weighted_quantile(
                [1.0, 2.0, 3.0], [0.5], [1.0, np.inf, 1.0], inf_rm=True
            ).iloc[0, 0],
            2.0,
        )
