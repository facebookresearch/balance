# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import random

import balance

import balance.testutil

import numpy as np
import pandas as pd
from balance.adjustment import (
    apply_transformations,
    default_transformations,
    trim_weights,
)

from balance.sample_class import Sample
from balance.util import fct_lump, quantize
from balance.weighting_methods import (
    cbps as balance_cbps,
    ipw as balance_ipw,
    poststratify as balance_poststratify,
)

# Tolerance for floating point comparisons
EPSILON = 0.00001


class TestAdjustment(balance.testutil.BalanceTestCase):
    """
    Test suite for the balance adjustment module functionality.

    This test class validates the core adjustment functions including:
    - Weight trimming operations
    - Data transformations (default and custom)
    - Transformation application across multiple dataframes
    - Adjustment method discovery

    The tests ensure proper handling of edge cases, error conditions,
    and maintain backward compatibility for the balance library's
    adjustment capabilities.
    """

    def test_trim_weights(self):
        """
        Test weight trimming functionality including no trimming, percentile trimming,
        and mean ratio trimming scenarios.

        Validates that:
        - Weights are properly converted to float64 dtype
        - No trimming preserves original values
        - Percentile and mean ratio trimming work correctly
        - Error conditions are properly handled
        """
        # Test no trimming - verify dtype conversion to float64
        input_weights = pd.Series([0, 1, 2])
        expected_weights = pd.Series([0.0, 1.0, 2.0])

        result_weights = trim_weights(input_weights)
        pd.testing.assert_series_equal(result_weights, expected_weights)
        self.assertEqual(type(result_weights), pd.Series)
        self.assertEqual(result_weights.dtype, np.float64)

        # Test that no trimming parameters preserves original weights
        random.seed(42)
        random_weights = np.random.uniform(0, 1, 10000)
        untrimmed_result = trim_weights(
            random_weights,
            weight_trimming_percentile=None,
            weight_trimming_mean_ratio=None,
            keep_sum_of_weights=False,
        )
        self.assertEqual(untrimmed_result, random_weights)

        # Test error handling for invalid input types
        with self.assertRaisesRegex(
            TypeError, "weights must be np.array or pd.Series, are of type*"
        ):
            trim_weights("Strings don't get trimmed", weight_trimming_mean_ratio=1)

        # Test error when both trimming parameters are provided
        with self.assertRaisesRegex(ValueError, "Only one"):
            trim_weights(np.array([0, 1, 2]), 1, 1)

        # Test weight_trimming_mean_ratio functionality
        random.seed(42)
        original_weights = np.random.uniform(0, 1, 10000)
        mean_ratio_result = trim_weights(original_weights, weight_trimming_mean_ratio=1)

        # Mean should be preserved and ratio constraints should be applied
        self.assertAlmostEqual(
            np.mean(original_weights), np.mean(mean_ratio_result), delta=EPSILON
        )
        self.assertAlmostEqual(
            np.mean(original_weights) / np.min(original_weights),
            np.max(mean_ratio_result) / np.min(mean_ratio_result),
            delta=EPSILON,
        )

        # Test weight_trimming_percentile functionality
        random.seed(42)
        test_weights = np.random.uniform(0, 1, 10000)

        # Test upper percentile trimming
        upper_trimmed = trim_weights(
            test_weights,
            weight_trimming_percentile=(0, 0.11),
            keep_sum_of_weights=False,
        )
        self.assertTrue(max(upper_trimmed) < 0.9)

        # Test lower percentile trimming
        lower_trimmed = trim_weights(
            test_weights,
            weight_trimming_percentile=(0.11, 0),
            keep_sum_of_weights=False,
        )
        self.assertTrue(min(lower_trimmed) > 0.1)

        # Test both-sided percentile trimming
        both_trimmed = trim_weights(
            test_weights, weight_trimming_percentile=(0.11, 0.11)
        )
        self.assertTrue(min(both_trimmed) > 0.1)
        self.assertTrue(max(both_trimmed) < 0.9)

    def test_default_transformations(self):
        """
        Test automatic detection of appropriate transformations for different data types.

        Validates that:
        - Numeric columns get quantize transformation
        - Categorical/string columns get fct_lump transformation
        - Works with multiple dataframes
        - Handles boolean and nullable Int64 dtypes correctly
        """
        # Test with multiple dataframes
        multiple_dfs_input = (
            pd.DataFrame({"a": (1, 2), "b": ("a", "b")}),
            pd.DataFrame({"c": (1, 2), "d": ("a", "b")}),
        )
        multiple_result = default_transformations(multiple_dfs_input)
        expected_multiple = {
            "a": quantize,
            "b": fct_lump,
            "c": quantize,
            "d": fct_lump,
        }
        self.assertEqual(multiple_result, expected_multiple)

        # Test with single dataframe
        single_df_input = pd.DataFrame({"a": (1, 2), "b": ("a", "b")})
        single_result = default_transformations([single_df_input])
        expected_single = {
            "a": quantize,
            "b": fct_lump,
        }
        self.assertEqual(single_result, expected_single)

        # Test with boolean and nullable Int64 dtypes
        typed_df_input = pd.DataFrame({"a": (1, 2), "b": (True, False)})
        typed_df_input = typed_df_input.astype(
            dtype={
                "a": "Int64",
                "b": "boolean",
            }
        )
        typed_result = default_transformations([typed_df_input])
        expected_typed = {
            "a": quantize,
            "b": fct_lump,
        }
        self.assertEqual(typed_result, expected_typed)

    def test_default_transformations_pd_int64(self):
        """
        Test that nullable Int64 dtype is handled the same as regular int64.

        Ensures compatibility between pandas nullable integers and numpy integers
        for transformation detection.
        """
        nullable_int_df = pd.DataFrame({"a": pd.array((1, 2), dtype="Int64")})
        numpy_int_df = nullable_int_df.astype(np.int64)

        nullable_transformations = default_transformations([nullable_int_df])
        numpy_transformations = default_transformations([numpy_int_df])

        self.assertEqual(nullable_transformations, numpy_transformations)

    def test_apply_transformations(self):
        """Test basic transformations with modifications and additions."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        transformations = {"d": lambda x: x * 2, "f": lambda x: x.d + 1}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (
            pd.DataFrame({"d": [2, 4, 6], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "f": [5, 6, 7, 8]}),
        )
        self.assertEqual(result[0], expected[0], lazy=True)
        self.assertEqual(result[1], expected[1], lazy=True)

    def test_apply_transformations_none_transformations(self):
        """Test that None transformations return dataframes unchanged."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        result = apply_transformations((source_df, target_df), None)
        self.assertEqual(result, (source_df, target_df))

    def test_apply_transformations_only_modifications(self):
        """Test transformations with only column modifications."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        result = apply_transformations((source_df, target_df), {"d": lambda x: x * 2})
        expected = (
            pd.DataFrame({"d": [2, 4, 6]}),
            pd.DataFrame({"d": [8, 10, 12, 14]}),
        )
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_only_additions(self):
        """Test transformations with only column additions."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        result = apply_transformations((source_df, target_df), {"f": lambda x: x.d + 1})
        expected = (pd.DataFrame({"f": [2, 3, 4]}), pd.DataFrame({"f": [5, 6, 7, 8]}))
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_drop_warning(self):
        """Test that drop warning is properly issued."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        transformations = {"d": lambda x: x * 2, "f": lambda x: x.d + 1}
        self.assertWarnsRegexp(
            r"Dropping the variables: \['e'\]",
            apply_transformations,
            (source_df, target_df),
            transformations,
        )

    def test_apply_transformations_drop_false(self):
        """Test that drop=False preserves all original columns."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        transformations = {"d": lambda x: x * 2, "f": lambda x: x.d + 1}
        result = apply_transformations(
            (source_df, target_df), transformations, drop=False
        )
        expected = (
            pd.DataFrame({"d": [2, 4, 6], "e": [1, 2, 3], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "e": [1, 2, 3, 4], "f": [5, 6, 7, 8]}),
        )
        self.assertEqual(result[0], expected[0], lazy=True)
        self.assertEqual(result[1], expected[1], lazy=True)

    def test_apply_transformations_three_dataframes(self):
        """Test that transformations work correctly with three dataframes."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        third_df = pd.DataFrame({"d": [8, 9], "g": [1, 2]})
        transformations = {"d": lambda x: x * 2, "f": lambda x: x.d + 1}

        result = apply_transformations(
            (source_df, target_df, third_df), transformations
        )
        expected = (
            pd.DataFrame({"d": [2, 4, 6], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "f": [5, 6, 7, 8]}),
            pd.DataFrame({"d": [16, 18], "f": [9, 10]}),
        )
        self.assertEqual(result[0], expected[0], lazy=True)
        self.assertEqual(result[1], expected[1], lazy=True)
        self.assertEqual(result[2], expected[2], lazy=True)

    def test_apply_transformations_global_computation(self):
        """Test that transformations are computed over all dataframes, not individually."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        transformations = {"d": lambda x: x / max(x)}

        result = apply_transformations((source_df, target_df), transformations)
        # Max across both dataframes is 7, so all values are divided by 7
        expected = (
            pd.DataFrame({"d": [1 / 7, 2 / 7, 3 / 7]}),
            pd.DataFrame({"d": [4 / 7, 5 / 7, 6 / 7, 7 / 7]}),
        )
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_missing_column_transform(self):
        """Test transforming a column that exists in only one dataframe."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7]})  # missing column 'e'
        transformations = {"e": lambda x: x * 2}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (
            pd.DataFrame({"e": [2.0, 4.0, 6.0]}),
            pd.DataFrame({"e": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_missing_column_add(self):
        """Test adding a column based on a column that exists in only one dataframe."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7]})  # missing column 'e'
        transformations = {"f": lambda x: x.e * 2}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (
            pd.DataFrame({"f": [2.0, 4.0, 6.0]}),
            pd.DataFrame({"f": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_missing_column_specified(self):
        """Test transformation of a specified column that's missing in one dataframe."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [0, 0, 0]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7]})  # missing column 'e'
        transformations = {"e": lambda x: x + 1}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (
            pd.DataFrame({"e": [1.0, 1.0, 1.0]}),
            pd.DataFrame({"e": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_index_handling(self):
        """Test that dataframe indices are properly handled during transformations."""
        source_df = pd.DataFrame({"d": [1, 2, 3]}, index=(5, 6, 7))
        target_df = pd.DataFrame({"d": [4, 5, 6, 7]}, index=(0, 1, 2, 3))
        transformations = {"d": lambda x: x}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (source_df, target_df)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_index_reset_case(self):
        """Test specific index handling case that requires reset_index internally."""
        source_df = pd.DataFrame({"a": (0, 0, 0, 0, 0, 0, 0, 0)})
        target_df = pd.DataFrame({"a": (1, 1, 1, 1)})
        result = apply_transformations((source_df, target_df), "default")
        expected = (
            pd.DataFrame({"a": ["(-0.001, 0.7]"] * 8}),
            pd.DataFrame({"a": ["(0.7, 1.0]"] * 4}),
        )
        self.assertEqual(result[0].astype(str), expected[0])
        self.assertEqual(result[1].astype(str), expected[1])

    def test_apply_transformations_default_comprehensive(self):
        """Test default transformations with comprehensive data."""
        source_df = pd.DataFrame({"d": range(0, 100), "e": ["a"] * 96 + ["b"] * 4})
        target_df = pd.DataFrame({"d": range(0, 100), "e": ["a"] * 96 + ["b"] * 4})
        result_source, result_target = apply_transformations(
            (source_df, target_df), "default"
        )

        # Numeric column should be quantized into 10 bins
        self.assertEqual(result_source["d"].drop_duplicates().values.shape[0], 10)
        self.assertEqual(result_target["d"].drop_duplicates().values.shape[0], 10)

        # Categorical column should be lumped (rare categories combined)
        expected_categories = ("a", "_lumped_other")
        self.assertEqual(
            result_source["e"].drop_duplicates().values, expected_categories
        )
        self.assertEqual(
            result_target["e"].drop_duplicates().values, expected_categories
        )

    def test_invalid_input_to_apply_transformations(self):
        """Test error handling for invalid inputs to apply_transformations function."""
        # Sample data for testing with mixed data types and weights
        self.sample_data = Sample.from_frame(
            df=pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "b": (-42, 8, 2, -42),
                    "o": (7, 8, 9, 10),
                    "c": ("x", "y", "z", "x"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
            outcome_columns="o",
        )

        # Test non-existent transformation
        self.assertRaisesRegex(
            NotImplementedError,
            "Unknown transformations",
            apply_transformations,
            (self.sample_data.df,),
            "foobar",
        )

        # Test non-dataframe input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' must contain DataFrames",
            apply_transformations,
            (self.sample_data,),
            "foobar",
        )

        # Test non-tuple input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' argument must be a tuple of DataFrames",
            apply_transformations,
            self.sample_data.df,
            "foobar",
        )

    def test__find_adjustment_method(self):
        """
        Test the internal adjustment method discovery function.

        Validates that:
        - Known adjustment methods (ipw, cbps, poststratify) are correctly resolved
        - Unknown methods raise appropriate ValueError
        """
        # Test known adjustment methods
        self.assertTrue(
            balance.adjustment._find_adjustment_method("ipw") is balance_ipw.ipw
        )
        self.assertTrue(
            balance.adjustment._find_adjustment_method("cbps") is balance_cbps.cbps
        )
        self.assertTrue(
            balance.adjustment._find_adjustment_method("poststratify")
            is balance_poststratify.poststratify
        )

        # Test unknown adjustment method
        with self.assertRaisesRegex(ValueError, "Unknown adjustment method*"):
            balance.adjustment._find_adjustment_method("some_other_value")
