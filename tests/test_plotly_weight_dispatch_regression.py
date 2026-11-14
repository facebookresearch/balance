# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.stats_and_plots.weighted_comparisons_plots import (
    plot_dist,
    plotly_plot_dist,
)
from balance.util import _assert_type


class PlotlyWeightDispatchRegressionTest(unittest.TestCase):
    def test_unweighted_dispatch_ignores_supplied_weights(self) -> None:
        frame = pd.DataFrame({"group": ["a", "b"]})
        result = plot_dist(
            [{"df": frame, "weight": pd.Series([9.0, 1.0])}],
            names=["self"],
            weighted=False,
            plot_it=False,
            return_dict_of_figures=True,
        )
        self.assertEqual(
            list(_assert_type(result, dict)["group"].data[0].y), [0.5, 0.5]
        )
        result = plotly_plot_dist(
            {"self": frame.assign(weight=[9.0, 1.0])},
            weighted=False,
            plot_it=False,
            return_dict_of_figures=True,
        )
        self.assertEqual(
            list(_assert_type(result, dict)["group"].data[0].y), [0.5, 0.5]
        )

    def test_none_weights_mean_uniform_weights(self) -> None:
        result = plot_dist(
            [{"df": pd.DataFrame({"group": ["a", "b"]}), "weight": None}],
            names=["self"],
            plot_it=False,
            return_dict_of_figures=True,
        )
        self.assertEqual(
            list(_assert_type(result, dict)["group"].data[0].y), [0.5, 0.5]
        )
