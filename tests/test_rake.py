# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    """
    Test suite for the rake weighting method.

    This test class validates the functionality of the rake() function and related
    utilities used for iterative proportional fitting (raking) in survey weighting.
    Rake weighting adjusts sample weights to match known population marginal distributions.
    """

    def _assert_rake_raises_with_message(
        self,
        expected_message,
        sample_df,
        sample_weights,
        target_df,
        target_weights,
        **kwargs,
    ):
        """
        Helper method to assert that rake raises an error with a specific message.

        Args:
            expected_message: Expected error message pattern
            sample_df: Sample DataFrame
            sample_weights: Sample weights Series
            target_df: Target DataFrame
            target_weights: Target weights Series
            **kwargs: Additional arguments to pass to rake()
        """
        self.assertRaisesRegex(
            AssertionError,
            expected_message,
            rake,
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            **kwargs,
        )

    def test_rake_input_assertions(self):
        """
        Test that rake() properly validates input parameters.

        This test ensures that the rake function correctly identifies and raises
        appropriate errors for various invalid input scenarios:
        - Presence of 'weight' column in input data
        - Insufficient number of variables (must be at least 2)
        - Missing or invalid weight Series
        - Mismatched lengths between DataFrames and weight Series
        """
        n_rows = 20
        np.random.seed(42)
        sample = pd.DataFrame(
            {
                "a": np.random.normal(size=n_rows),
                "b": np.random.normal(size=n_rows),
                "weight": [1.0] * n_rows,
            }
        )
        np.random.seed(43)
        target = pd.DataFrame(
            {
                "a": np.random.normal(size=n_rows),
                "b": np.random.normal(size=n_rows),
            }
        )

        # Cannot have weight in df that is not the weight column
        self._assert_rake_raises_with_message(
            "weight shouldn't be a name for covariate in the sample data",
            sample,
            pd.Series((1,) * n_rows),
            target,
            pd.Series((1,) * n_rows),
        )

        target["weight"] = [2.0] * n_rows
        self._assert_rake_raises_with_message(
            "weight shouldn't be a name for covariate in the target data",
            sample[["a", "b"]],
            pd.Series((1,) * n_rows),
            target,
            pd.Series((1,) * n_rows),
        )

        # Must pass more than one variable
        self._assert_rake_raises_with_message(
            "Must weight on at least two variables",
            sample[["a"]],
            pd.Series((1,) * n_rows),
            target[["a"]],
            pd.Series((1,) * n_rows),
        )

        # Must pass weights for sample
        self._assert_rake_raises_with_message(
            "sample_weights must be a pandas Series",
            sample[["a", "b"]],
            None,
            target[["a", "b"]],
            pd.Series((1,) * n_rows),
        )

        # Must pass weights for target
        self._assert_rake_raises_with_message(
            "target_weights must be a pandas Series",
            sample[["a", "b"]],
            pd.Series((1,) * n_rows),
            target[["a", "b"]],
            None,
        )

        # Must pass weights of same length as sample
        self._assert_rake_raises_with_message(
            "sample_weights must be the same length as sample_df",
            sample[["a", "b"]],
            pd.Series((1,) * (n_rows - 1)),
            target[["a", "b"]],
            pd.Series((1,) * n_rows),
        )

        # Must pass weights of same length as target
        self._assert_rake_raises_with_message(
            "target_weights must be the same length as target_df",
            sample[["a", "b"]],
            pd.Series((1,) * n_rows),
            target[["a", "b"]],
            pd.Series((1,) * (n_rows - 1)),
        )

    def test_rake_fails_when_all_na(self):
        """
        Test that rake() properly handles cases where all values are NaN.

        This test verifies that rake() raises appropriate errors when:
        - Sample data contains all NaN values in a column
        - Target data contains all NaN values in a column
        This should result in empty DataFrames after dropping NAs, which is invalid.
        """
        # Create test data with NaN values
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

        # Test that sample with all NaN values fails
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

        # Test that target with NaN values that result in empty data fails
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
        """
        Test basic rake weighting functionality with categorical data.

        This test verifies that rake() correctly calculates weights to match
        target marginal distributions. It uses a simple case with categorical
        variables and checks that the resulting weights achieve the desired
        population balance.
        """
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
        """
        Test rake weighting with pre-weighted target data.

        This test verifies that rake() correctly handles cases where the target
        data already has non-uniform weights. The function should properly account
        for these existing weights when calculating the raking adjustments.
        """
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
        """
        Test that rake weights properly scale to match target population size.

        This test verifies that when the target population is larger than the sample,
        the rake weights sum to the target population size, effectively scaling up
        the sample to represent the larger population.
        """
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
        """
        Test rake weighting behavior with NaN values using different na_action strategies.

        This test verifies that rake() correctly handles missing values with two approaches:
        1. 'drop': Remove rows with NaN values before raking
        2. 'add_indicator': Create indicator variables for NaN values

        The test includes detailed calculations showing expected weight values
        for both strategies.
        """
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

    def test_rake_consistency_with_default_arguments(self):
        """
        Test consistency of rake function results with default parameters.

        This test verifies that the rake function produces consistent and expected
        results when applied to large datasets with mixed data types (continuous
        and categorical variables). It includes NaN values to test real-world
        scenarios and validates specific weight values and distributions.
        """
        # Create test data inline
        n_sample = 1000
        n_target = 2000
        np.random.seed(2021)

        # Create sample DataFrame with mixed data types
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

        # Create target DataFrame with mixed data types
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

        # Add some NaN values for realistic testing
        sample_df.loc[[0, 1], "a"] = np.nan
        target_df.loc[[100, 101], "a"] = np.nan

        # Create random weights
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
        """
        Test that variable ordering is consistent and alphabetized.

        This test ensures that the rake function produces identical results
        regardless of the order in which variables are specified. The function
        should internally alphabetize variables to ensure consistent behavior,
        preventing issues where different variable orders could lead to
        different weighting results.
        """
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
        """
        Test warning and error handling for mismatched categorical levels.

        This test verifies that rake() properly handles cases where:
        1. Sample data contains levels not present in target data (should raise ValueError)
        2. Target data contains levels not present in sample data (should issue warning)

        This ensures data quality and prevents silent failures in weighting.
        """
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
        """
        Test the _proportional_array_from_dict utility function.

        This test verifies that the helper function correctly converts a dictionary
        of proportions into an array representation where each key appears
        proportionally to its value. This is used internally by rake() to
        create proportional distributions for raking calculations.
        """
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
        """
        Test the _realize_dicts_of_proportions utility function.

        This test verifies that the helper function correctly processes a dictionary
        of dictionaries containing proportions, converting them into arrays where
        each variable has proportional representation. This function ensures
        consistent lengths across all variables for raking operations.
        """
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
        """
        Test the prepare_marginal_dist_for_raking utility function.

        This test verifies that the function correctly prepares marginal distributions
        for raking by converting proportion dictionaries into a DataFrame format
        suitable for the raking algorithm. The function ensures proper alignment
        and indexing of marginal distributions across multiple variables.
        """
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
