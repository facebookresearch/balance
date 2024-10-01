# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import balance.testutil

import numpy as np
import pandas as pd

from balance.sample_class import Sample
from balance.weighting_methods.rake import (
    _proportional_array_from_dict,
    _realize_dicts_of_proportions,
    prepare_marginal_dist_for_raking,
    rake,
)


class Testrake(
    balance.testutil.BalanceTestCase,
):
    def test_rake_input_assertions(self):
        N = 20
        sample = pd.DataFrame(
            {
                "a": np.random.normal(size=N),
                "b": np.random.normal(size=N),
                "weight": [1.0] * N,
            }
        )
        target = pd.DataFrame(
            {
                "a": np.random.normal(size=N),
                "b": np.random.normal(size=N),
            }
        )

        # Cannot have weight in df that is not the weight column
        self.assertRaisesRegex(
            AssertionError,
            "weight shouldn't be a name for covariate in the sample data",
            rake,
            sample,
            pd.Series((1,) * N),
            target,
            pd.Series((1,) * N),
        )

        target["weight"] = [2.0] * N
        self.assertRaisesRegex(
            AssertionError,
            "weight shouldn't be a name for covariate in the target data",
            rake,
            sample[["a", "b"]],
            pd.Series((1,) * N),
            target,
            pd.Series((1,) * N),
        )

        # Must pass more than one varaible
        self.assertRaisesRegex(
            AssertionError,
            "Must weight on at least two variables",
            rake,
            sample[["a"]],
            pd.Series((1,) * N),
            target[["a"]],
            pd.Series((1,) * N),
        )

        # Must pass weights for sample
        self.assertRaisesRegex(
            AssertionError,
            "sample_weights must be a pandas Series",
            rake,
            sample[["a", "b"]],
            None,
            target[["a", "b"]],
            pd.Series((1,) * N),
        )

        # Must pass weights for sample
        self.assertRaisesRegex(
            AssertionError,
            "target_weights must be a pandas Series",
            rake,
            sample[["a", "b"]],
            pd.Series((1,) * N),
            target[["a", "b"]],
            None,
        )

        # Must pass weights of same length as sample
        self.assertRaisesRegex(
            AssertionError,
            "sample_weights must be the same length as sample_df",
            rake,
            sample[["a", "b"]],
            pd.Series((1,) * (N - 1)),
            target[["a", "b"]],
            pd.Series((1,) * N),
        )

        # Must pass weights for sample
        self.assertRaisesRegex(
            AssertionError,
            "target_weights must be the same length as target_df",
            rake,
            sample[["a", "b"]],
            pd.Series((1,) * N),
            target[["a", "b"]],
            pd.Series((1,) * (N - 1)),
        )

    def test_rake_fails_when_all_na(self):
        df_sample_nas = pd.DataFrame(
            {
                "a": np.array([np.nan] * 12),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_sample = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target_nas = pd.DataFrame(
            {
                "a": pd.Series([np.nan] * 6 + ["2"] * 6, dtype=object),
                "b": pd.Series(["a"] * 6 + [np.nan] * 6, dtype=object),
                "id": range(0, 12),
            }
        )

        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty",
            rake,
            df_sample_nas,
            pd.Series((1,) * 12),
            df_target,
            pd.Series((1,) * 12),
            na_action="drop",
            transformations=None,
        )

        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty",
            rake,
            df_sample,
            pd.Series((1,) * 12),
            df_target_nas,
            pd.Series((1,) * 12),
            na_action="drop",
            transformations=None,
        )

    def test_rake_weights(self):
        df_sample = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )

        sample = Sample.from_frame(df_sample)
        target = Sample.from_frame(df_target)
        sample = sample.set_target(target)

        adjusted = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
        )

        self.assertEqual(
            adjusted["weight"].round(2),
            pd.Series([1.67, 0.33] * 6, name="rake_weight").rename_axis("index"),
        )

    def test_rake_weights_with_weighted_input(self):
        df_sample = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["a"] * 6 + ["b"] * 6,
                "weight": [0.5, 1.0] * 6,
                "id": range(0, 12),
            }
        )

        sample = Sample.from_frame(df_sample)
        target = Sample.from_frame(df_target)
        sample = sample.set_target(target)

        adjusted = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
        )

        self.assertEqual(
            adjusted["weight"].round(2),
            pd.Series([1.25, 0.25] * 6, name="rake_weight").rename_axis("index"),
        )

    def test_rake_weights_scale_to_pop(self):
        df_sample = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 5),
                "b": ["a"] * 6 + ["b"] * 9,
                "id": range(0, 15),
            }
        )

        sample = Sample.from_frame(df_sample)
        target = Sample.from_frame(df_target)
        sample = sample.set_target(target)

        adjusted = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
        )

        self.assertEqual(round(sum(adjusted["weight"]), 2), 15.0)

    def test_rake_expected_weights_with_na(self):
        dfsamp = pd.DataFrame(
            {
                "a": np.array([1.0, 2.0, np.nan] * 6),
                "b": ["a", "b"] * 9,
                "id": range(0, 18),
            }
        )
        dfpop = pd.DataFrame(
            {
                "a": np.array([1.0] * 10 + [2.0] * 6 + [np.nan] * 2),
                "b": ["a", "b"] * 9,
                "id": range(18, 36),
            }
        )

        sample = Sample.from_frame(dfsamp)
        target = Sample.from_frame(dfpop)
        sample = sample.set_target(target)

        # Dropping NAs (example calculation for test values):
        # Note, 'b' does not matter here, always balanced
        # In sample, a=1.0 is 6/12=0.5
        # In target, a=1.0 is 10/16=0.625
        # So a=1.0 in sample needs 1.25 weight (when weights sum to sample size)
        # Now that weights sum to target size, we need to scale 1.25 by relative population sizes
        # 1.25 * (16/12) = 1.6666667, final weight

        adjusted = sample.adjust(method="rake", transformations=None, na_action="drop")
        self.assertEqual(
            adjusted.weight_column.round(2),
            pd.Series([1.67, 1.0, np.nan] * 6, name="weight"),
        )

        # Dropping NAs (example calculation for test values):
        # Note, 'b' does not matter here, always balanced
        # In sample, a=1.0 is 6/18=0.333333
        # In target, a=1.0 is 10/18=0.5555556
        # So a=1.0 in sample needs 1.6667 weight (when weights sum to sample size)
        # sample size = target size, so no need to rescale
        adjusted = sample.adjust(
            method="rake", transformations=None, na_action="add_indicator"
        )

        self.assertEqual(
            adjusted.weight_column.round(2),
            pd.Series([1.67, 1.0, 0.33] * 6, name="weight"),
        )

    # Test consistency result of rake
    def test_rake_consistency_with_default_arguments(self):
        # This test is meant to check the consistency of the rake function with the default arguments
        np.random.seed(2021)
        n_sample = 1000
        n_target = 2000

        sample_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 10, size=n_sample), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_sample, 4)), columns=range(1, 5)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_sample, 5)
                    ),
                    columns=range(5, 10),
                ),
            ],
            axis=1,
        )
        sample_df = sample_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        target_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 18, size=n_target), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_target, 4)), columns=range(1, 5)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_target, 5)
                    ),
                    columns=range(5, 10),
                ),
            ],
            axis=1,
        )
        target_df = target_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        # Add some NAN values
        sample_df.loc[[0, 1], "a"] = np.nan
        target_df.loc[[100, 101], "a"] = np.nan

        sample_weights = pd.Series(np.random.uniform(0, 1, size=n_sample))
        target_weights = pd.Series(np.random.uniform(0, 1, size=n_target))

        res = rake(sample_df, sample_weights, target_df, target_weights)

        # Compare output weights (examples and distribution)
        self.assertEqual(round(res["weight"][4], 4), 1.3221)
        self.assertEqual(round(res["weight"][997], 4), 0.8985)
        self.assertEqual(
            np.around(res["weight"].describe().values, 4),
            np.array(
                [
                    1.0000e03,
                    1.0167e00,
                    3.5000e-01,
                    3.4260e-01,
                    7.4790e-01,
                    9.7610e-01,
                    1.2026e00,
                    2.8854e00,
                ]
            ),
        )

    def test_variable_order_alphabetized(self):
        # Note: 'a' is always preferred, and due to perfect collinearity
        # with 'b', 'b' never gets weighted to, even if we reverse the
        # order. This is not a perfect test, but it broke pre-alphabetization!
        df_sample = pd.DataFrame(
            {
                "a": ["1"] * 6 + ["2"] * 6,
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["a"] * 3 + ["b"] * 9,
                "id": range(0, 12),
            }
        )

        sample = Sample.from_frame(df_sample)
        target = Sample.from_frame(df_target)

        adjusted = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
            variables=["a", "b"],
        )

        adjusted_two = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
            variables=["b", "a"],
        )

        self.assertEqual(
            adjusted["weight"],
            adjusted_two["weight"],
        )

    def test_rake_levels_warnings(self):
        df_sample = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["a"] * 6 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_sample_excess_levels = pd.DataFrame(
            {
                "a": np.array(["1", "2"] * 6),
                "b": ["alpha"] * 2 + ["a"] * 4 + ["b"] * 6,
                "id": range(0, 12),
            }
        )
        df_target_excess_levels = pd.DataFrame(
            {
                "a": np.array(["1"] * 10 + ["2"] * 2),
                "b": ["omega"] * 2 + ["a"] * 4 + ["b"] * 6,
                "id": range(0, 12),
            }
        )

        sample = Sample.from_frame(df_sample)
        sample_excess_levels = Sample.from_frame(df_sample_excess_levels)
        target = Sample.from_frame(df_target)
        target_excess_levels = Sample.from_frame(df_target_excess_levels)

        self.assertRaisesRegex(
            ValueError,
            "'b' in target is missing.*alpha",
            rake,
            sample_excess_levels.covars().df,
            sample_excess_levels.weight_column,
            target.covars().df,
            target.weight_column,
        )
        self.assertWarnsRegexp(
            "'b' in sample is missing.*omega",
            rake,
            sample.covars().df,
            sample.weight_column,
            target_excess_levels.covars().df,
            target_excess_levels.weight_column,
        )

    def test__proportional_array_from_dict(self):
        self.assertEqual(
            _proportional_array_from_dict({"a": 0.2, "b": 0.8}),
            ["a", "b", "b", "b", "b"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 0.5, "b": 0.5}), ["a", "b"]
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
            ["a", "b", "c"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 3 / 8, "b": 5 / 8}),
            ["a", "a", "a", "b", "b", "b", "b", "b"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 3 / 5, "b": 1 / 5, "c": 2 / 10}),
            ["a", "a", "a", "b", "c"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 3 / 8, "b": 5 / 8}, max_length=5),
            ["a", "a", "b", "b", "b"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 3 / 8, "b": 5 / 8}, max_length=50),
            ["a", "a", "a", "b", "b", "b", "b", "b"],
        )

    def test__realize_dicts_of_proportions(self):
        dict_of_dicts = {
            "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
            "v2": {"aa": 0.5, "bb": 0.5},
        }

        self.assertEqual(
            _realize_dicts_of_proportions(dict_of_dicts),
            {
                "v1": ["a", "b", "b", "b", "c", "a", "b", "b", "b", "c"],
                "v2": ["aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb"],
            },
        )

        dict_of_dicts = {
            "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
            "v2": {"aa": 0.5, "bb": 0.5},
            "v3": {"A": 0.2, "B": 0.8},
        }
        self.assertEqual(
            _realize_dicts_of_proportions(dict_of_dicts),
            {
                "v1": ["a", "b", "b", "b", "c", "a", "b", "b", "b", "c"],
                "v2": ["aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb"],
                "v3": ["A", "B", "B", "B", "B", "A", "B", "B", "B", "B"],
            },
        )

        dict_of_dicts = {
            "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
            "v2": {"aa": 0.5, "bb": 0.5},
            "v3": {"A": 0.2, "B": 0.8},
            "v4": {"A": 0.1, "B": 0.9},
        }
        self.assertEqual(
            _realize_dicts_of_proportions(dict_of_dicts),
            {
                "v1": ["a", "b", "b", "b", "c", "a", "b", "b", "b", "c"],
                "v2": ["aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb"],
                "v3": ["A", "B", "B", "B", "B", "A", "B", "B", "B", "B"],
                "v4": ["A", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            },
        )

    def test_prepare_marginal_dist_for_raking(self):
        self.assertEqual(
            prepare_marginal_dist_for_raking(
                {"A": {"a": 0.5, "b": 0.5}, "B": {"x": 0.2, "y": 0.8}}
            ).to_dict(),
            {
                "A": {
                    0: "a",
                    1: "b",
                    2: "a",
                    3: "b",
                    4: "a",
                    5: "b",
                    6: "a",
                    7: "b",
                    8: "a",
                    9: "b",
                },
                "B": {
                    0: "x",
                    1: "y",
                    2: "y",
                    3: "y",
                    4: "y",
                    5: "x",
                    6: "y",
                    7: "y",
                    8: "y",
                    9: "y",
                },
                "id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
            },
        )

    # TODO: test convergence rate
    # TODO: test max iteration
    # TODO: test logging
