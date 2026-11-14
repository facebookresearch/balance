# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_stats import (
    ci_of_weighted_mean,
    var_of_weighted_mean,
    weighted_mean,
    weighted_sd,
    weighted_var,
)


class MissingVarianceRegressionTest(unittest.TestCase):
    def test_missing_observation_and_its_weight_are_omitted(self) -> None:
        values = [1.0, np.nan, 3.0]
        weights = [1.0, 100.0, 1.0]
        self.assertEqual(weighted_mean(values, weights).iloc[0], 2.0)
        self.assertEqual(weighted_var(values, weights).iloc[0], 2.0)
        self.assertEqual(var_of_weighted_mean(values, weights).iloc[0], 0.5)
        self.assertEqual(
            ci_of_weighted_mean(values, weights, round_ndigits=3).iloc[0],
            (0.614, 3.386),
        )
        for statistic in (
            weighted_mean,
            weighted_var,
            var_of_weighted_mean,
            weighted_sd,
        ):
            with self.subTest(statistic=statistic.__name__):
                self.assertEqual(
                    statistic(values).iloc[0], statistic([1.0, 3.0]).iloc[0]
                )

    def test_each_column_omits_only_its_missing_rows(self) -> None:
        values = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [np.nan, 10.0, 14.0]},
            index=pd.Index([7, 7, 8]),
        )
        weights = pd.Series([1.0, 2.0, 3.0], index=[90, 91, 92])
        original = values.copy(deep=True)
        original_weights = weights.copy(deep=True)
        for statistic in (
            weighted_mean,
            weighted_var,
            var_of_weighted_mean,
            weighted_sd,
            ci_of_weighted_mean,
        ):
            for column, rows in (("a", [0, 2]), ("b", [1, 2])):
                with self.subTest(statistic=statistic.__name__, column=column):
                    actual = statistic(values, weights).loc[column]
                    expected = statistic(
                        values[column].iloc[rows], weights.iloc[rows]
                    ).iloc[0]
                    self.assertEqual(actual, expected)
        pd.testing.assert_frame_equal(values, original)
        pd.testing.assert_series_equal(weights, original_weights)

    def test_missing_weights_and_removed_infinities_are_omitted(self) -> None:
        for statistic in (weighted_mean, weighted_var, var_of_weighted_mean):
            with self.subTest(statistic=statistic.__name__):
                expected = statistic([1.0, 3.0], [1.0, 2.0]).iloc[0]
                self.assertEqual(
                    statistic([1.0, 999.0, 3.0], [1.0, np.nan, 2.0]).iloc[0],
                    expected,
                )
                self.assertEqual(
                    statistic([1.0, np.inf, 3.0], [1.0, 100.0, 2.0], inf_rm=True).iloc[
                        0
                    ],
                    expected,
                )
                self.assertEqual(
                    statistic([1.0, 999.0, 3.0], [1.0, np.inf, 2.0], inf_rm=True).iloc[
                        0
                    ],
                    expected,
                )

    def test_real_zeros_are_not_missing_observations(self) -> None:
        self.assertEqual(weighted_mean([0.0, np.nan, 2.0]).iloc[0], 1.0)
        self.assertEqual(weighted_var([0.0, np.nan, 2.0]).iloc[0], 2.0)
        self.assertEqual(var_of_weighted_mean([0.0, np.nan, 2.0]).iloc[0], 0.5)

    def test_nullable_numeric_missing_values_use_observed_rows(self) -> None:
        for dtype in ("Float64", "Int64"):
            values = pd.Series([1, pd.NA, 3], dtype=dtype)
            for statistic in (weighted_mean, weighted_var, var_of_weighted_mean):
                with self.subTest(dtype=dtype, statistic=statistic.__name__):
                    self.assertEqual(
                        statistic(values, [1.0, 100.0, 1.0]).iloc[0],
                        statistic([1, 3]).iloc[0],
                    )

    def test_no_observed_weight_mass_has_no_estimate(self) -> None:
        for values, weights in (
            ([np.nan, np.nan], [1.0, 2.0]),
            ([np.nan, 3.0], [2.0, 0.0]),
        ):
            for statistic in (
                weighted_mean,
                weighted_var,
                var_of_weighted_mean,
                weighted_sd,
            ):
                with self.subTest(values=values, statistic=statistic.__name__):
                    self.assertTrue(np.isnan(statistic(values, weights).iloc[0]))
        self.assertTrue(np.isnan(weighted_var([1.0, np.nan]).iloc[0]))
        self.assertEqual(weighted_mean([1.0, np.nan]).iloc[0], 1.0)
        self.assertEqual(var_of_weighted_mean([1.0, np.nan]).iloc[0], 0.0)
