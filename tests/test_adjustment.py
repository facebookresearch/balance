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

import random

import balance

import balance.testutil

import numpy as np
import pandas as pd
from balance.adjustment import (
    _validate_limit,
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

    def test_trim_weights(self) -> None:
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
            # pyre-ignore[6]: Testing error handling with intentionally wrong type
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
            test_weights,
            weight_trimming_percentile=(0.11, 0.11),
        )
        self.assertTrue(min(both_trimmed) > 0.1)
        self.assertTrue(max(both_trimmed) < 0.9)

    def test_trim_weights_doc_examples(self) -> None:
        """Docstring examples stay synchronized with implementation results."""

        weights = pd.Series(range(1, 101))

        symmetric_trim = trim_weights(
            weights,
            weight_trimming_percentile=0.01,
            keep_sum_of_weights=False,
        )
        expected_symmetric = pd.Series(range(1, 101)).clip(3, 98).astype(float)
        pd.testing.assert_series_equal(symmetric_trim, expected_symmetric)

        upper_trim = trim_weights(
            weights,
            weight_trimming_percentile=(0.0, 0.05),
            keep_sum_of_weights=False,
        )
        expected_upper = pd.Series(range(1, 101)).clip(upper=94).astype(float)
        pd.testing.assert_series_equal(upper_trim, expected_upper)

    def test_trim_weights_return_type_consistency(self) -> None:
        """
        Test that both weight_trimming_mean_ratio and weight_trimming_percentile
        return pd.Series with dtype=float64 and preserve the index.

        This validates the explicit conversions added to ensure consistent
        return types across all trimming methods.
        """
        # Create test data with custom index
        custom_index = pd.Index([10, 20, 30, 40, 50])
        test_weights = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5], index=custom_index)

        # Test weight_trimming_mean_ratio returns pd.Series with correct dtype and index
        mean_ratio_result = trim_weights(
            test_weights, weight_trimming_mean_ratio=1.5, keep_sum_of_weights=False
        )
        self.assertEqual(type(mean_ratio_result), pd.Series)
        self.assertEqual(mean_ratio_result.dtype, np.float64)
        pd.testing.assert_index_equal(mean_ratio_result.index, custom_index)

        # Test weight_trimming_percentile returns pd.Series with correct dtype and index
        percentile_result = trim_weights(
            test_weights,
            weight_trimming_percentile=(0.1, 0.1),
            keep_sum_of_weights=False,
        )
        self.assertEqual(type(percentile_result), pd.Series)
        self.assertEqual(percentile_result.dtype, np.float64)
        pd.testing.assert_index_equal(percentile_result.index, custom_index)

    def test_trim_weights_target_sum_scaling(self) -> None:
        """Trimming can directly scale to a target sum of weights."""

        weights = pd.Series([1.0, 2.0, 3.0], name="w")

        scaled = trim_weights(weights, target_sum_weights=12.0)

        self.assertAlmostEqual(scaled.sum(), 12.0, delta=EPSILON)
        self.assertEqual(scaled.name, "w")

    def test_trim_weights_target_sum_zero_raises(self) -> None:
        """Scaling to a target sum fails when trimmed weights sum to zero."""

        zeros = pd.Series([0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "sum is zero"):
            trim_weights(zeros, keep_sum_of_weights=False, target_sum_weights=1.0)

    def test_default_transformations(self) -> None:
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

    def test_default_transformations_pd_int64(self) -> None:
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

    def test_apply_transformations(self) -> None:
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

    def test_apply_transformations_none_transformations(self) -> None:
        """Test that None transformations return dataframes unchanged."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        result = apply_transformations((source_df, target_df), None)
        self.assertEqual(result, (source_df, target_df))

    def test_apply_transformations_only_modifications(self) -> None:
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

    def test_apply_transformations_only_additions(self) -> None:
        """Test transformations with only column additions."""
        source_df = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        target_df = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})
        result = apply_transformations((source_df, target_df), {"f": lambda x: x.d + 1})
        expected = (pd.DataFrame({"f": [2, 3, 4]}), pd.DataFrame({"f": [5, 6, 7, 8]}))
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_drop_warning(self) -> None:
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

    def test_apply_transformations_drop_false(self) -> None:
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

    def test_apply_transformations_three_dataframes(self) -> None:
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

    def test_apply_transformations_global_computation(self) -> None:
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

    def test_apply_transformations_missing_column_transform(self) -> None:
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

    def test_apply_transformations_missing_column_add(self) -> None:
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

    def test_apply_transformations_missing_column_specified(self) -> None:
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

    def test_apply_transformations_index_handling(self) -> None:
        """Test that dataframe indices are properly handled during transformations."""
        source_df = pd.DataFrame({"d": [1, 2, 3]}, index=(5, 6, 7))
        target_df = pd.DataFrame({"d": [4, 5, 6, 7]}, index=(0, 1, 2, 3))
        transformations = {"d": lambda x: x}
        result = apply_transformations((source_df, target_df), transformations)
        expected = (source_df, target_df)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_apply_transformations_index_reset_case(self) -> None:
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

    def test_apply_transformations_default_comprehensive(self) -> None:
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

    def test_invalid_input_to_apply_transformations(self) -> None:
        """Test error handling for invalid inputs to apply_transformations function."""
        # Sample data for testing with mixed data types and weights
        sample_data = Sample.from_frame(
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
            (sample_data.df,),
            "foobar",
        )

        # Test non-dataframe input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' must contain DataFrames",
            apply_transformations,
            (sample_data,),
            "foobar",
        )

        # Test non-tuple input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' argument must be a tuple of DataFrames",
            apply_transformations,
            sample_data.df,
            "foobar",
        )

    def test__find_adjustment_method(self) -> None:
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
            # pyre-ignore[6]: Testing error handling with intentionally wrong type
            balance.adjustment._find_adjustment_method("some_other_value")

    def test_validate_limit_none_input(self) -> None:
        """
        Test that _validate_limit returns None when given None input.

        This validates the function's handling of the None sentinel value
        which indicates no limit should be applied.
        """
        # Setup: None limit
        limit = None
        n_weights = 100

        # Execute: validate the limit
        result = _validate_limit(limit, n_weights)

        # Assert: None should be returned unchanged
        self.assertIsNone(result)

    def test_validate_limit_zero_input(self) -> None:
        """
        Test that _validate_limit returns 0.0 when given 0 input.

        This validates the special case handling for zero limits.
        """
        # Setup: zero limit
        limit = 0
        n_weights = 100

        # Execute: validate the limit
        result = _validate_limit(limit, n_weights)

        # Assert: should return 0.0
        self.assertEqual(result, 0.0)
        self.assertEqual(type(result), float)

    def test_validate_limit_finite_value_adjustment(self) -> None:
        """
        Test that _validate_limit properly adjusts finite limits.

        For finite limits between 0 and 1, the function should add a small
        adjustment based on n_weights to ensure proper winsorization behavior.
        """
        # Setup: finite limit between 0 and 1
        limit = 0.1
        n_weights = 100

        # Execute: validate and adjust the limit
        result = _validate_limit(limit, n_weights)

        # Assert: result should be slightly larger than input
        self.assertIsNotNone(result)
        self.assertGreater(result, limit)
        # Extra is min(2.0/100, 0.1/10) = min(0.02, 0.01) = 0.01
        expected = limit + 0.01
        self.assertAlmostEqual(result, expected, delta=EPSILON)

    def test_validate_limit_finite_value_with_small_n_weights(self) -> None:
        """
        Test _validate_limit with small n_weights values.

        When n_weights is small, the adjustment factor should be larger.
        """
        # Setup: finite limit with small n_weights
        limit = 0.1
        n_weights = 10

        # Execute: validate and adjust the limit
        result = _validate_limit(limit, n_weights)

        # Assert: result should have larger adjustment due to small n_weights
        self.assertIsNotNone(result)
        self.assertGreater(result, limit)
        # Extra is min(2.0/10, 0.1/10) = min(0.2, 0.01) = 0.01
        expected = limit + 0.01
        self.assertAlmostEqual(result, expected, delta=EPSILON)

    def test_validate_limit_finite_value_capped_at_one(self) -> None:
        """
        Test that _validate_limit caps adjusted values at 1.0.

        Even if the adjustment would push the limit above 1.0,
        the result should be capped at 1.0.
        """
        # Setup: limit close to 1.0
        limit = 0.99
        n_weights = 100

        # Execute: validate and adjust the limit
        result = _validate_limit(limit, n_weights)

        # Assert: result should be capped at 1.0
        self.assertIsNotNone(result)
        self.assertLessEqual(result, 1.0)
        # Extra is min(2.0/100, 0.99/10) = min(0.02, 0.099) = 0.02
        # 0.99 + 0.02 = 1.01, but should be capped at 1.0
        self.assertEqual(result, 1.0)

    def test_validate_limit_invalid_negative_value(self) -> None:
        """
        Test that _validate_limit raises ValueError for negative limits.

        Percentile limits must be in the range [0, 1], so negative values
        should raise a ValueError.
        """
        # Setup: invalid negative limit
        limit = -0.1
        n_weights = 100

        # Execute and Assert: should raise ValueError
        with self.assertRaisesRegex(
            ValueError, "Percentile limits must be between 0 and 1"
        ):
            _validate_limit(limit, n_weights)

    def test_validate_limit_invalid_greater_than_one(self) -> None:
        """
        Test that _validate_limit raises ValueError for limits > 1.

        Percentile limits must be in the range [0, 1], so values greater than 1
        should raise a ValueError.
        """
        # Setup: invalid limit greater than 1
        limit = 1.5
        n_weights = 100

        # Execute and Assert: should raise ValueError
        with self.assertRaisesRegex(
            ValueError, "Percentile limits must be between 0 and 1"
        ):
            _validate_limit(limit, n_weights)

    def test_validate_limit_non_finite_value(self) -> None:
        """
        Test that _validate_limit returns non-finite values unchanged.

        If a non-finite value (like infinity) is passed, it should be returned
        without adjustment, after basic validation.
        """
        # Setup: non-finite limit (infinity)
        limit = float("inf")
        n_weights = 100

        # Execute: validate the limit
        result = _validate_limit(limit, n_weights)

        # Assert: should return infinity unchanged
        self.assertEqual(result, float("inf"))

    def test_validate_limit_integer_input(self) -> None:
        """
        Test that _validate_limit handles integer inputs correctly.

        The function should accept integer inputs and convert them to float.
        """
        # Setup: integer limit
        limit = 1
        n_weights = 100

        # Execute: validate the limit
        result = _validate_limit(limit, n_weights)

        # Assert: should return float type
        self.assertEqual(type(result), float)
        self.assertEqual(result, 1.0)

    def test_validate_limit_edge_case_n_weights_one(self) -> None:
        """
        Test _validate_limit with n_weights=1.

        With only one weight, the adjustment calculation should handle
        the edge case properly using max(n_weights, 1).
        """
        # Setup: single weight scenario
        limit = 0.5
        n_weights = 1

        # Execute: validate and adjust the limit
        result = _validate_limit(limit, n_weights)

        # Assert: should calculate adjustment correctly
        self.assertIsNotNone(result)
        self.assertGreater(result, limit)
        # Extra is min(2.0/max(1, 1), 0.5/10) = min(2.0, 0.05) = 0.05
        expected = limit + 0.05
        self.assertAlmostEqual(result, expected, delta=EPSILON)
