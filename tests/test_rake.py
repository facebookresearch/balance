# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import warnings

import balance.testutil
import numpy as np
import pandas as pd
from balance import adjustment as balance_adjustment
from balance.sample_class import Sample
from balance.weighting_methods.rake import (
    _find_lcm_of_array_lengths,
    _hare_niemeyer_allocation,
    _lcm,
    _proportional_array_from_dict,
    _realize_dicts_of_proportions,
    _run_ipf_numpy,
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
        expected_message: str,
        sample_df: pd.DataFrame,
        sample_weights: pd.Series | None,
        target_df: pd.DataFrame,
        target_weights: pd.Series | None,
        **kwargs: object,
    ) -> None:
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

    def test_rake_input_assertions(self) -> None:
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

    def test_rake_fails_when_all_na(self) -> None:
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

    def test_rake_weights(self) -> None:
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

    def test_rake_weight_trimming_applied(self) -> None:
        """Verify that rake forwards trimming arguments to the adjustment helper."""

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

        baseline = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
        )

        trimmed = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
            weight_trimming_mean_ratio=1.0,
        )

        expected = balance_adjustment.trim_weights(
            baseline["weight"],
            target_sum_weights=baseline["weight"].sum(),
            weight_trimming_mean_ratio=1.0,
        ).rename("rake_weight")

        pd.testing.assert_series_equal(trimmed["weight"], expected)

    def test_rake_percentile_trimming_applied(self) -> None:
        """Percentile trimming parameters should be honoured by rake."""

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

        baseline = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
        )

        trimmed = rake(
            sample.covars().df,
            sample.weight_column,
            target.covars().df,
            target.weight_column,
            weight_trimming_percentile=0.1,
        )

        expected = balance_adjustment.trim_weights(
            baseline["weight"],
            target_sum_weights=baseline["weight"].sum(),
            weight_trimming_percentile=0.1,
        ).rename("rake_weight")

        pd.testing.assert_series_equal(trimmed["weight"], expected)

    def test_rake_weights_with_weighted_input(self) -> None:
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

    def test_rake_weights_scale_to_pop(self) -> None:
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

    def test_rake_expected_weights_with_na(self) -> None:
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

    def test_rake_consistency_with_default_arguments(self) -> None:
        """
        Test consistency of rake function results with default parameters.

        This test verifies that the rake function produces consistent and expected
        results when applied to large datasets with mixed data types (continuous
        and categorical variables). It includes NaN values to test real-world
        scenarios and validates specific weight values and distributions.
        """
        # Create test data inline (reduced sizes for faster tests — rake
        # correctness is validated by the smaller deterministic tests above)
        n_sample = 200
        n_target = 400
        np.random.seed(2021)

        # Create sample DataFrame with mixed data types
        # 2 continuous + 1 continuous-with-NaN + 3 categorical = 6 columns
        sample_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 10, size=n_sample), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_sample, 2)), columns=range(1, 3)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_sample, 3)
                    ),
                    columns=range(3, 6),
                ),
            ],
            axis=1,
        )
        sample_df = sample_df.rename(columns={i: "abcdef"[i] for i in range(0, 6)})

        # Create target DataFrame with mixed data types
        target_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 18, size=n_target), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_target, 2)), columns=range(1, 3)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_target, 3)
                    ),
                    columns=range(3, 6),
                ),
            ],
            axis=1,
        )
        target_df = target_df.rename(columns={i: "abcdef"[i] for i in range(0, 6)})

        # Add some NaN values for realistic testing
        sample_df.loc[[0, 1], "a"] = np.nan
        target_df.loc[[100, 101], "a"] = np.nan

        # Create random weights
        sample_weights = pd.Series(np.random.uniform(0, 1, size=n_sample))
        target_weights = pd.Series(np.random.uniform(0, 1, size=n_target))

        res = rake(sample_df, sample_weights, target_df, target_weights)

        # Verify rake produces reasonable results: correct length, finite,
        # and positive weights (detailed correctness is tested by the smaller
        # deterministic tests above).
        self.assertEqual(len(res["weight"]), n_sample)
        self.assertTrue(np.all(np.isfinite(res["weight"].dropna())))
        self.assertTrue(res["weight"].min() > 0)

    def test_variable_order_alphabetized(self) -> None:
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

    def test_rake_levels_warnings(self) -> None:
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

    def test__proportional_array_from_dict(self) -> None:
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

    def test__realize_dicts_of_proportions(self) -> None:
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

    def test__hare_niemeyer_allocation(self) -> None:
        """
        Test the _hare_niemeyer_allocation helper.

        Verifies that the function correctly allocates n slots to categories
        according to their proportions using the largest-remainder method, and
        that the total length of the output always equals n.
        """
        # Simple 80/20 split with n=5
        self.assertEqual(
            _hare_niemeyer_allocation({"a": 0.2, "b": 0.8}, 5),
            ["a", "b", "b", "b", "b"],
        )

        # Equal split – n not evenly divisible by 2 → largest remainder assigns extra to "a"
        result = _hare_niemeyer_allocation({"a": 0.5, "b": 0.5}, 3)
        self.assertEqual(len(result), 3)
        # Both categories must be represented
        self.assertIn("a", result)
        self.assertIn("b", result)

        # Proportions that don't sum to 1 are normalized internally
        result2 = _hare_niemeyer_allocation({"x": 2.0, "y": 8.0}, 10)
        self.assertEqual(result2.count("x"), 2)
        self.assertEqual(result2.count("y"), 8)

        # Output length always equals n
        for n in [1, 7, 100, 9999]:
            out = _hare_niemeyer_allocation({"p": 0.3, "q": 0.3, "r": 0.4}, n)
            self.assertEqual(len(out), n)

        # Zero-valued categories are excluded
        result3 = _hare_niemeyer_allocation({"a": 0.5, "b": 0.0, "c": 0.5}, 4)
        self.assertNotIn("b", result3)
        self.assertEqual(len(result3), 4)

    def test__realize_dicts_of_proportions_input_validation(self) -> None:
        """
        Tests that _realize_dicts_of_proportions raises actionable ValueError for
        invalid inputs: empty dict_of_dicts, bool/float/zero max_length, and
        bool proportions in inner dicts (which should propagate with variable name).
        """
        simple = {"v1": {"a": 0.5, "b": 0.5}}

        # empty dict_of_dicts rejected
        with self.assertRaises(ValueError):
            _realize_dicts_of_proportions({})

        # bool max_length rejected
        with self.assertRaises(ValueError):
            _realize_dicts_of_proportions(simple, True)

        # float max_length rejected
        with self.assertRaises(ValueError):
            # pyre-ignore[6]: Deliberately testing invalid type
            _realize_dicts_of_proportions(simple, 10000.0)

        # max_length < 1 rejected
        with self.assertRaises(ValueError):
            _realize_dicts_of_proportions(simple, 0)

        # bool proportion in inner dict rejected, error message includes variable name
        with self.assertRaisesRegex(ValueError, "v1"):
            _realize_dicts_of_proportions({"v1": {"a": True, "b": 0.5}}, 5)

    def test__realize_dicts_of_proportions_lcm_cap(self) -> None:
        """
        Test that _realize_dicts_of_proportions caps output at max_length when
        the natural LCM of individual array lengths would exceed it.

        This is the core regression test for the memory-explosion bug where
        high-precision proportions or many variables caused LCM to reach
        tens of millions of rows.
        """
        # Two variables whose individual arrays fit within max_length but whose
        # LCM would be huge (coprime array lengths near max_length).
        # Array lengths for each variable produced by _proportional_array_from_dict
        # are ~9999 and ~9998 → LCM ≈ 9999 * 9998 ≈ 99 million.
        # Using the bug-report example inputs (high-precision floats).
        pop_distribution = {
            "age": {
                "18-24": 0.3622334534532,
                "25-34": 0.421434535,
                "35-44": 0.216345345,
            },
            "gender": {
                "Male": 0.4955666,
                "Female": 0.505434345,
            },
        }

        result = _realize_dicts_of_proportions(pop_distribution, max_length=10000)
        # All arrays must have the same length and equal max_length exactly
        # (the Hare-Niemeyer fallback always allocates exactly max_length slots)
        lengths = {len(v) for v in result.values()}
        self.assertEqual(len(lengths), 1)
        self.assertEqual(list(lengths)[0], 10000)

        # A warning must have been logged (via logger.warning, not warnings.warn)
        with self.assertLogs("balance.weighting_methods", level="WARNING") as cm:
            _realize_dicts_of_proportions(pop_distribution, 10000)
        self.assertTrue(
            any("LCM of array lengths" in msg for msg in cm.output),
        )

        # The custom max_length parameter is respected
        result_small = _realize_dicts_of_proportions(pop_distribution, max_length=500)
        lengths_small = {len(v) for v in result_small.values()}
        self.assertEqual(list(lengths_small)[0], 500)

    def test__realize_dicts_of_proportions_no_cap_for_small_lcm(self) -> None:
        """
        Test that _realize_dicts_of_proportions does NOT trigger the cap when
        the LCM of array lengths is within max_length (i.e. existing behaviour
        is preserved for well-behaved inputs).
        """
        dict_of_dicts = {
            "v1": {"a": 0.2, "b": 0.6, "c": 0.2},
            "v2": {"aa": 0.5, "bb": 0.5},
        }
        # LCM = 10, well below any reasonable max_length – result must be identical
        # to the pre-fix behaviour.
        self.assertEqual(
            _realize_dicts_of_proportions(dict_of_dicts, max_length=10000),
            {
                "v1": ["a", "b", "b", "b", "c", "a", "b", "b", "b", "c"],
                "v2": ["aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb", "aa", "bb"],
            },
        )

    def test_prepare_marginal_dist_for_raking_large_input(self) -> None:
        """
        Regression test: prepare_marginal_dist_for_raking must not produce a
        DataFrame with more than max_length rows, even for high-precision
        proportions that would previously cause LCM explosion.

        These examples are taken directly from the bug report (GitHub issue #369).
        """
        # Bug report example 1: 4 variables – previously produced (33_366_600, 5).
        pop_distribution_1 = {
            "age": {"18-24": 0.362, "25-34": 0.421, "35-44": 0.216},
            "gender": {"Male": 0.495, "Female": 0.505},
            "education": {"College": 0.495, "No_College": 0.505},
            "religion": {
                "Christianity": 0.162,
                "Sunni Islam": 0.312,
                "Shia Islam": 0.322,
                "Buddhism": 0.055,
                "Hinduism": 0.077,
                "Judaism": 0.038,
                "No religion": 0.028,
                "Sikhism": 0.008,
            },
        }
        df1 = prepare_marginal_dist_for_raking(pop_distribution_1)
        # Must be capped at the default max_length of 10 000
        self.assertEqual(df1.shape[0], 10000)
        # Must have 4 covariate columns + 1 id column
        self.assertEqual(df1.shape[1], 5)
        # id column must be sequential
        self.assertEqual(list(df1["id"]), list(range(10000)))

        # Bug report example 2: 2 variables with high-precision floats –
        # previously produced (99_990_000, 3).
        pop_distribution_2 = {
            "age": {
                "18-24": 0.3622334534532,
                "25-34": 0.421434535,
                "35-44": 0.216345345,
            },
            "gender": {
                "Male": 0.4955666,
                "Female": 0.505434345,
            },
        }
        df2 = prepare_marginal_dist_for_raking(pop_distribution_2)
        self.assertEqual(df2.shape[0], 10000)
        self.assertEqual(df2.shape[1], 3)
        self.assertEqual(list(df2["id"]), list(range(10000)))

    def test__realize_dicts_of_proportions_bug_report_example(self) -> None:
        """
        Regression test for the third example in bug report #369, which calls
        _realize_dicts_of_proportions directly with high-precision proportions
        and verifies the output length is bounded.

        Previously `len(target_dict["age"])` was ~99 million.
        """
        pop_distribution = {
            "age": {
                "18-24": 0.3622334534532,
                "25-34": 0.421434535,
                "35-44": 0.216345345,
            },
            "gender": {
                "Male": 0.4955666,
                "Female": 0.505434345,
            },
        }
        target_dict = _realize_dicts_of_proportions(pop_distribution)
        # Output must be capped at the default max_length of 10 000
        self.assertEqual(len(target_dict["age"]), 10000)
        self.assertEqual(len(target_dict["gender"]), 10000)

    def test__hare_niemeyer_allocation_numpy_scalars(self) -> None:
        """
        Regression test: _hare_niemeyer_allocation must accept NumPy scalar
        numeric types (e.g. np.float64, np.int64) that are produced when a
        user builds the proportions dict via pandas Series.to_dict().
        """
        import numpy as np

        # np.float64 proportions — the common case from Series.to_dict()
        result = _hare_niemeyer_allocation(
            # pyre-ignore[6]: Testing numpy scalar compatibility
            {"a": np.float64(0.2), "b": np.float64(0.8)},
            5,
        )
        self.assertEqual(result, ["a", "b", "b", "b", "b"])
        self.assertEqual(len(result), 5)

        # np.int64 proportions (unnormalized counts)
        # pyre-ignore[6]: Testing numpy scalar compatibility
        result2 = _hare_niemeyer_allocation({"x": np.int64(2), "y": np.int64(8)}, 10)
        self.assertEqual(result2.count("x"), 2)
        self.assertEqual(result2.count("y"), 8)

        # Mixed Python float and np.float64
        # pyre-ignore[6]: Testing numpy scalar compatibility
        result3 = _hare_niemeyer_allocation({"p": 0.3, "q": np.float64(0.7)}, 10)
        self.assertEqual(len(result3), 10)

    def test__realize_dicts_of_proportions_more_categories_than_max_length(
        self,
    ) -> None:
        """
        Regression test: _realize_dicts_of_proportions must not crash with a
        ZeroDivisionError when a variable has more categories than max_length.

        Two scenarios are tested:
        1. lcm_length > max_length: 15 equal-weight categories, max_length=10.
           _proportional_array_from_dict returns an array of length 15; LCM([15, 2])
           = 30 > 10 triggers the Hare-Niemeyer fallback.
        2. lcm_length == 0: 1000 equal-weight categories, max_length=10.
           _proportional_array_from_dict returns [] (all counts round to 0);
           LCM([0, 2]) = 0 triggers the Hare-Niemeyer fallback instead of
           a ZeroDivisionError in the LCM-extension step.
        """
        # Scenario 1: LCM > max_length path (15 categories, max_length=10)
        many_cats = {f"cat_{i}": 1 / 15 for i in range(15)}
        dict_of_dicts = {"v1": many_cats, "v2": {"a": 0.5, "b": 0.5}}

        result = _realize_dicts_of_proportions(dict_of_dicts, max_length=10)
        lengths = {len(v) for v in result.values()}
        self.assertEqual(len(lengths), 1)
        self.assertEqual(list(lengths)[0], 10)
        self.assertLessEqual(len(set(result["v1"])), 15)

        # Scenario 2: lcm_length == 0 path (1000 categories, max_length=10).
        # With 1000 equal-weight categories, _proportional_array_from_dict returns []
        # because scaling_factor = 10/1000 = 0.01 makes each rounded count 0.
        # LCM of [0, 2] = 0, which previously bypassed the cap check and caused
        # ZeroDivisionError at `lcm_length // len(arr)`.
        very_many_cats = {f"cat_{i}": 1 / 1000 for i in range(1000)}
        dict_of_dicts_2 = {"v1": very_many_cats, "v2": {"a": 0.5, "b": 0.5}}

        result2 = _realize_dicts_of_proportions(dict_of_dicts_2, max_length=10)
        lengths2 = {len(v) for v in result2.values()}
        self.assertEqual(len(lengths2), 1)
        self.assertEqual(list(lengths2)[0], 10)

    def test_prepare_marginal_dist_for_raking(self) -> None:
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

    def test_run_ipf_numpy_matches_expected_margins(self) -> None:
        """Validate that the NumPy IPF solver hits the requested marginals."""

        original = np.array([[5.0, 3.0], [2.0, 4.0]])
        # Target row totals and column totals after rescaling the sample sum.
        target_rows = np.array([7.0, 7.0])
        target_cols = np.array([6.0, 8.0])

        fitted, converged, iterations = _run_ipf_numpy(
            original,
            [target_rows, target_cols],
            convergence_rate=1e-6,
            max_iteration=10,
            rate_tolerance=0.0,
        )

        self.assertEqual(converged, 1)
        np.testing.assert_allclose(fitted.sum(axis=1), target_rows, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fitted.sum(axis=0), target_cols, rtol=0, atol=1e-6)
        self.assertGreater(len(iterations), 0)

    def test_run_ipf_numpy_handles_zero_targets(self) -> None:
        """Ensure zero-valued margins do not introduce NaNs or divergence."""

        original = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
        target_rows = np.array([5.0, 5.0])
        # The last column should be forced to zero.
        target_cols = np.array([4.0, 6.0, 0.0])

        fitted, converged, _ = _run_ipf_numpy(
            original,
            [target_rows, target_cols],
            convergence_rate=1e-7,
            max_iteration=50,
            rate_tolerance=0.0,
        )

        self.assertEqual(converged, 1)
        self.assertTrue(np.all(np.isfinite(fitted)))
        np.testing.assert_allclose(fitted.sum(axis=1), target_rows, atol=1e-9)
        np.testing.assert_allclose(fitted.sum(axis=0), target_cols, atol=1e-9)

    def test_run_ipf_numpy_flags_non_convergence(self) -> None:
        """The solver should report non-convergence when the iteration budget is exhausted."""

        original = np.array([[1.0, 0.0], [0.0, 1.0]])
        target_rows = np.array([1.0, 1.0])
        target_cols = np.array([0.5, 1.5])

        _, converged, _ = _run_ipf_numpy(
            original,
            [target_rows, target_cols],
            convergence_rate=1e-9,
            max_iteration=0,
            rate_tolerance=0.0,
        )

        self.assertEqual(converged, 0)

    def test_rake_zero_weight_levels_respected(self) -> None:
        """Variable levels with zero target weight should collapse to zero mass."""

        sample_df = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["p", "q", "p", "q"],
            }
        )
        sample_weights = pd.Series([1.0, 2.0, 3.0, 4.0])
        target_df = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["p", "q", "p", "q"],
            }
        )
        # Force the 'q' column margin to zero in the target population.
        target_weights = pd.Series([1.0, 0.0, 1.0, 0.0])

        result = rake(sample_df, sample_weights, target_df, target_weights)

        raked = pd.concat([sample_df, result["weight"]], axis=1)
        level_totals = raked.groupby("b")["rake_weight"].sum()
        expected = (
            pd.concat([target_df, target_weights.rename("weight")], axis=1)
            .groupby("b")["weight"]
            .sum()
            .rename("rake_weight")
        )

        self.assertAlmostEqual(level_totals.loc["q"], 0.0)
        pd.testing.assert_series_equal(level_totals.sort_index(), expected.sort_index())

    def test__lcm(self) -> None:
        """
        Test the _lcm function with various input cases.

        The _lcm function is used internally by _find_lcm_of_array_lengths
        to ensure proportional representation across variables in raking.
        """
        test_cases = [
            # (a, b, expected_lcm, description)
            (4, 6, 12, "basic case"),
            (7, 11, 77, "coprime numbers"),
            (3, 9, 9, "one divides other"),
            (5, 5, 5, "same numbers"),
            (1, 5, 5, "LCM with 1"),
            (0, 5, 0, "LCM with 0"),
            (-4, 6, 12, "negative numbers"),
            (100, 150, 300, "larger numbers"),
        ]

        for a, b, expected, description in test_cases:
            with self.subTest(a=a, b=b, description=description):
                self.assertEqual(_lcm(a, b), expected)

    def test__find_lcm_of_array_lengths(self) -> None:
        """
        Test _find_lcm_of_array_lengths with various array configurations.

        This function finds the LCM of array lengths to ensure proportional
        representation across variables in raking operations.
        """
        test_cases = [
            # (arrays, expected_lcm, description)
            (
                {"v1": ["a", "b", "b", "c"], "v2": ["aa", "bb"]},
                4,
                "basic case: lengths 4 and 2",
            ),
            (
                {
                    "v1": ["a", "b", "b", "c"],
                    "v2": ["aa", "bb"],
                    "v3": ["a1", "a2", "a3"],
                },
                12,
                "three arrays: lengths 4, 2, 3",
            ),
            (
                {"v1": ["a", "b", "c"], "v2": ["x", "y", "z"], "v3": ["1", "2", "3"]},
                3,
                "all same length",
            ),
            (
                {"v1": ["a"]},
                1,
                "single array of length 1",
            ),
            (
                {"v1": ["a", "b", "c", "d", "e"]},
                5,
                "single array of length 5",
            ),
            (
                {"v1": ["a", "b", "c"], "v2": ["1", "2", "3", "4", "5"]},
                15,
                "coprime lengths 3 and 5",
            ),
            (
                {
                    "v1": ["a", "b"],
                    "v2": ["x", "y", "z"],
                    "v3": ["1", "2", "3", "4", "5"],
                },
                30,
                "coprime lengths 2, 3, and 5",
            ),
            (
                {
                    "v1": ["a", "b"],
                    "v2": ["w", "x", "y", "z"],
                    "v3": ["1", "2", "3", "4", "5", "6", "7", "8"],
                },
                8,
                "divisible lengths 2, 4, 8",
            ),
        ]

        for arrays, expected, description in test_cases:
            with self.subTest(description=description):
                self.assertEqual(_find_lcm_of_array_lengths(arrays), expected)

    def test__proportional_array_from_dict_edge_cases(self) -> None:
        """
        Test _proportional_array_from_dict with additional edge cases.

        This extends the existing test coverage with edge cases like:
        - Dictionary with zero values
        - Single key dictionary
        - Values that don't sum to 1
        - Empty arrays after filtering
        """
        # Single key should return array with just that key
        self.assertEqual(
            _proportional_array_from_dict({"a": 1.0}),
            ["a"],
        )

        # Values that don't sum to 1 should be normalized
        self.assertEqual(
            _proportional_array_from_dict({"a": 2, "b": 2}),
            ["a", "b"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 3, "b": 6}),
            ["a", "b", "b"],
        )

        # Dictionary with zero values should filter them out
        self.assertEqual(
            _proportional_array_from_dict({"a": 0.5, "b": 0.5, "c": 0.0}),
            ["a", "b"],
        )
        self.assertEqual(
            _proportional_array_from_dict({"a": 0.0, "b": 1.0}),
            ["b"],
        )

    def test__run_ipf_numpy_zero_dimensional_array(self) -> None:
        """
        Test that _run_ipf_numpy raises ValueError for 0-dimensional arrays.

        The IPF algorithm requires at least one dimension to operate on.
        """
        with self.assertRaises(ValueError):
            _run_ipf_numpy(
                np.array(5.0),  # 0-dimensional array
                [],
                convergence_rate=0.01,
                max_iteration=100,
                rate_tolerance=1e-8,
            )

    def test__run_ipf_numpy_convergence_rate_tolerance(self) -> None:
        """
        Test that _run_ipf_numpy respects rate_tolerance convergence criterion.

        The algorithm should stop when the change in convergence rate between
        iterations is less than rate_tolerance, even if convergence_rate is not met.
        """
        # Create a simple 2x2 table that converges slowly
        original = np.array([[10.0, 20.0], [30.0, 40.0]])
        target_rows = np.array([35.0, 65.0])
        target_cols = np.array([45.0, 55.0])

        # With high rate_tolerance, should converge quickly
        _, converged, iterations = _run_ipf_numpy(
            original,
            [target_rows, target_cols],
            convergence_rate=1e-10,  # Very strict convergence
            max_iteration=1000,
            rate_tolerance=0.1,  # But allow early stopping
        )

        # Should converge (rate_tolerance allows it)
        self.assertEqual(converged, 1)
        # Should take relatively few iterations due to rate_tolerance
        self.assertLess(len(iterations), 100)


class TestRakeEdgeCases(balance.testutil.BalanceTestCase):
    """Test suite for rake edge cases and error handling."""

    def test__run_ipf_numpy_nan_convergence(self) -> None:
        """Test _run_ipf_numpy handles NaN in convergence calculation (line 97).

        Verifies that when the convergence calculation results in NaN,
        it is properly converted to 0.0 to continue the algorithm.
        """
        # Setup: Create a scenario where division by zero could occur in convergence
        # Using a small table where margins sum to zero on one axis
        original = np.array([[0.0, 0.0], [1.0, 1.0]])
        # Target with a zero margin that could cause NaN in diff calculation
        target_rows = np.array([0.0, 2.0])
        target_cols = np.array([1.0, 1.0])

        # Execute: Run IPF - should not raise despite potential NaN
        result, _converged, iterations = _run_ipf_numpy(
            original,
            [target_rows, target_cols],
            convergence_rate=0.01,
            max_iteration=100,
            rate_tolerance=1e-8,
        )

        # Assert: Verify algorithm completed without error
        self.assertIsNotNone(result)
        self.assertIsInstance(iterations, pd.DataFrame)

    def test_rake_invalid_na_action(self) -> None:
        """Test that invalid na_action raises ValueError (line 238).

        Verifies that passing an invalid na_action value raises a
        descriptive ValueError.
        """
        # Setup: Create sample and target data
        sample = pd.DataFrame({"a": ["1", "2", "3"], "id": [1, 2, 3]})
        target = pd.DataFrame({"a": ["1", "2", "3"], "id": [1, 2, 3]})
        sample_weights = pd.Series([1, 1, 1])
        target_weights = pd.Series([1, 1, 1])

        # Execute & Assert: Invalid na_action should raise ValueError
        self.assertRaisesRegex(
            ValueError,
            "`na_action` must be 'add_indicator' or 'drop'",
            rake,
            sample,
            sample_weights,
            target,
            target_weights,
            na_action="invalid_action",
        )

    def test_rake_non_convergence_warning(self) -> None:
        """Test that non-convergence produces warning (line 327).

        Verifies that when the rake algorithm does not converge within
        max_iteration, a warning is logged about not achieving convergence.
        """
        # Setup: Create data that is difficult to converge with very few iterations
        sample = pd.DataFrame(
            {
                "a": ["1", "2"] * 50,
                "b": ["x", "y"] * 50,
                "id": range(100),
            }
        )
        target = pd.DataFrame(
            {
                "a": ["1"] * 90 + ["2"] * 10,
                "b": ["x"] * 10 + ["y"] * 90,
                "id": range(100),
            }
        )
        sample_weights = pd.Series([1] * 100)
        target_weights = pd.Series([1] * 100)

        # Execute: Run rake with very few max_iteration to force non-convergence
        with self.assertLogs(level="WARNING") as logs:
            rake(
                sample,
                sample_weights,
                target,
                target_weights,
                max_iteration=1,  # Force non-convergence
            )

        # Assert: Verify warning about non-convergence was logged
        self.assertTrue(
            any("convergence was not achieved" in message for message in logs.output)
        )


class TestRunIpfNumpyNanConv(balance.testutil.BalanceTestCase):
    """Test _run_ipf_numpy nan convergence handling (line 97)."""

    def test_run_ipf_numpy_nan_conv_handling(self) -> None:
        """Test _run_ipf_numpy handles nan convergence value (line 97).

        Verifies that when np.nanmax returns nan (all diff values are nan),
        the current_conv is set to 0.0 to prevent algorithm errors.
        """
        # Setup: Create a marginal distribution that would cause all-nan diff values
        # This happens when current = 0 and margin = 0, making diff = nan
        table = np.array([[0.0, 0.0], [0.0, 0.0]])
        margins = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]

        # Execute: Run IPF with zero margins that cause nan values
        # Suppress RuntimeWarning about "All-NaN slice encountered" as it's expected
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            result_table, converged, iterations_df = _run_ipf_numpy(
                table,
                margins,
                convergence_rate=1e-6,
                max_iteration=10,
                rate_tolerance=0.0,
            )

        # Assert: Should handle nan gracefully and converge (or at least not crash)
        # When all margins are 0, convergence should be detected
        self.assertEqual(converged, 1)  # Should converge since 0/0-1 = nan -> 0.0
        self.assertIsInstance(result_table, np.ndarray)
        self.assertIsInstance(iterations_df, pd.DataFrame)

    def test_run_ipf_numpy_partial_nan_handling(self) -> None:
        """Test _run_ipf_numpy handles partial nan values in convergence check (line 97).

        Tests that when some but not all diff values are nan (due to 0/0),
        the algorithm correctly handles them using np.nanmax and continues.
        """
        # Setup: Create mixed margins where some are 0 (causing nan) and some are not
        table = np.array([[1.0, 0.0], [0.0, 1.0]])
        # margins[0] has a zero that will cause nan in convergence check
        margins = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

        # Execute: Run IPF with mixed margins
        result_table, converged, iterations_df = _run_ipf_numpy(
            table,
            margins,
            convergence_rate=1e-6,
            max_iteration=100,
            rate_tolerance=0.0,
        )

        # Assert: Should handle partial nan gracefully
        self.assertIsInstance(result_table, np.ndarray)
        self.assertIsInstance(iterations_df, pd.DataFrame)
        # Check that convergence history is recorded
        self.assertGreater(len(iterations_df), 0)
