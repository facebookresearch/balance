# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import sys

import balance.testutil

import numpy as np
import pandas as pd


class TestTestUtil(
    balance.testutil.BalanceTestCase,
):
    """Test cases for the testutil module functions."""

    def _get_test_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Helper method to create test DataFrames with original and reordered columns.

        Returns:
            Tuple of (original_df, reordered_df) where both have same data but different column order.
        """
        df_original = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)})
        df_reordered = pd.DataFrame(
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")  # pyre-ignore[6]
        )
        return df_original, df_reordered

    def _get_test_indices(self) -> tuple[pd.Index, pd.Index]:
        """Helper method to create test indices with original and reordered values.

        Returns:
            Tuple of (original_index, reordered_index) where both have same values but different order.
        """
        index_original = pd.Index([1, 2, 3])
        index_reordered = pd.Index([1, 3, 2])
        return index_original, index_reordered

    def test_assert_frame_equal_lazy_with_different_values(self) -> None:
        """Test that _assert_frame_equal_lazy raises AssertionError when DataFrames have different values."""
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_frame_equal_lazy,
            pd.DataFrame({"a": (1, 2, 3)}),
            pd.DataFrame({"a": (1, 2, 4)}),
        )

    def test_assert_frame_equal_lazy_with_different_columns(self) -> None:
        """Test that _assert_frame_equal_lazy raises AssertionError when DataFrames have different columns."""
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_frame_equal_lazy,
            pd.DataFrame({"a": (1, 2, 3), "b": (1, 2, 3)}),
            pd.DataFrame({"a": (1, 2, 4), "c": (1, 2, 3)}),
        )

    def test_assert_frame_equal_lazy_with_column_order_lazy_mode(self) -> None:
        """Test that _assert_frame_equal_lazy handles column order differences in lazy mode."""
        df_original, df_reordered = self._get_test_dataframes()

        # Should not raise an error in lazy mode (default)
        balance.testutil._assert_frame_equal_lazy(df_original, df_reordered)

    def test_assert_frame_equal_lazy_with_column_order_strict_mode(self) -> None:
        """Test that _assert_frame_equal_lazy raises AssertionError for column order differences in strict mode."""
        df_original, df_reordered = self._get_test_dataframes()

        # Should raise an error in strict mode (lazy=False)
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_frame_equal_lazy,
            df_original,
            df_reordered,
            False,
        )

    def test_assert_index_equal_lazy_with_different_values(self) -> None:
        """Test that _assert_index_equal_lazy raises AssertionError when indices have different values."""
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_index_equal_lazy,
            pd.Index([1, 2, 3]),
            pd.Index([1, 2, 4]),
        )

    def test_assert_index_equal_lazy_with_order_lazy_mode(self) -> None:
        """Test that _assert_index_equal_lazy handles order differences in lazy mode."""
        index_original, index_reordered = self._get_test_indices()

        # Should not raise an error in lazy mode (default)
        balance.testutil._assert_index_equal_lazy(index_original, index_reordered)

    def test_assert_index_equal_lazy_with_order_strict_mode(self) -> None:
        """Test that _assert_index_equal_lazy raises AssertionError for order differences in strict mode."""
        index_original, index_reordered = self._get_test_indices()

        # Should raise an error in strict mode (lazy=False)
        self.assertRaises(
            AssertionError,
            balance.testutil._assert_index_equal_lazy,
            index_original,
            index_reordered,
            False,
        )


class TestTestUtil_BalanceTestCase_Equal(
    balance.testutil.BalanceTestCase,
):
    """Test cases for the BalanceTestCase assertEqual method with different data types."""

    def test_assertEqual_with_basic_types(self) -> None:
        """Test assertEqual method with basic Python types (int, str)."""
        # Test successful equality
        self.assertEqual(1, 1)
        self.assertEqual("a", "a")

        # Test failure cases
        self.assertRaises(AssertionError, self.assertEqual, 1, 2)
        self.assertRaises(AssertionError, self.assertEqual, "a", "b")

    def test_assertEqual_with_numpy_arrays(self) -> None:
        """Test assertEqual method with numpy arrays."""
        # Test successful equality
        self.assertEqual(np.array((1, 2)), np.array((1, 2)))

        # Test failure case
        self.assertRaises(
            AssertionError, self.assertEqual, np.array((1, 2)), np.array((2, 1))
        )

    def test_assertEqual_with_dataframes_strict_mode(self) -> None:
        """Test assertEqual method with DataFrames in strict mode (default behavior)."""
        df_original = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)})
        df_reordered = pd.DataFrame(
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")  # pyre-ignore[6]
        )

        # Should raise error by default (strict mode)
        self.assertRaises(AssertionError, self.assertEqual, df_original, df_reordered)

        # Should raise error when explicitly set to strict mode
        self.assertRaises(
            AssertionError, self.assertEqual, df_original, df_reordered, lazy=False
        )

    def test_assertEqual_with_dataframes_lazy_mode(self) -> None:
        """Test assertEqual method with DataFrames in lazy mode."""
        df_original = pd.DataFrame({"a": (1, 2, 3), "b": (4, 5, 6)})
        df_reordered = pd.DataFrame(
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")  # pyre-ignore[6]
        )

        # Should not raise error in lazy mode
        self.assertEqual(df_original, df_reordered, lazy=True)

    def test_assertEqual_with_pandas_series(self) -> None:
        """Test assertEqual method with pandas Series."""
        # Test successful equality
        self.assertEqual(pd.Series([1, 2]), pd.Series([1, 2]))

        # Test failure case
        self.assertRaises(
            AssertionError, self.assertEqual, pd.Series([1, 2]), pd.Series([2, 1])
        )

    def test_assertEqual_with_pandas_index_strict_mode(self) -> None:
        """Test assertEqual method with pandas Index in strict mode."""
        # Test successful equality
        self.assertEqual(pd.Index((1, 2)), pd.Index((1, 2)))

        # Test failure cases in strict mode
        self.assertRaises(
            AssertionError, self.assertEqual, pd.Index((1, 2)), pd.Index((2, 1))
        )
        self.assertRaises(
            AssertionError,
            self.assertEqual,
            pd.Index((1, 2)),
            pd.Index((2, 1)),
            lazy=False,
        )

    def test_assertEqual_with_pandas_index_lazy_mode(self) -> None:
        """Test assertEqual method with pandas Index in lazy mode."""
        # Should not raise error in lazy mode despite different order
        self.assertEqual(pd.Index((1, 2)), pd.Index((2, 1)), lazy=True)


class TestTestUtil_BalanceTestCase_Warns(
    balance.testutil.BalanceTestCase,
):
    """Test cases for the BalanceTestCase warning assertion methods."""

    def test_assertIfWarns_with_warning(self) -> None:
        """Test assertIfWarns method when a warning is produced."""
        self.assertIfWarns(lambda: logging.getLogger(__package__).warning("test"))

    def test_assertNotWarns_without_warning(self) -> None:
        """Test assertNotWarns method when no warning is produced."""
        self.assertNotWarns(lambda: "x")

    def test_assertWarnsRegexp_with_matching_pattern(self) -> None:
        """Test assertWarnsRegexp method when warning matches the regex pattern."""
        self.assertWarnsRegexp(
            "abc", lambda: logging.getLogger(__package__).warning("abcde")
        )

    def test_assertWarnsRegexp_with_non_matching_pattern(self) -> None:
        """Test assertWarnsRegexp method when warning does not match the regex pattern."""
        self.assertRaises(
            AssertionError,
            self.assertWarnsRegexp,
            "abcdef",
            lambda: logging.getLogger(__package__).warning("abcde"),
        )

    def test_assertNotWarnsRegexp_with_non_matching_pattern(self) -> None:
        """Test assertNotWarnsRegexp method when warning does not match the regex pattern."""
        self.assertNotWarnsRegexp(
            "abcdef", lambda: logging.getLogger(__package__).warning("abcde")
        )


class TestTestUtil_BalanceTestCase_Print(
    balance.testutil.BalanceTestCase,
):
    """Test cases for the BalanceTestCase print assertion methods."""

    def test_assertPrints_with_stdout_output(self) -> None:
        """Test assertPrints method when output is printed to stdout."""
        self.assertPrints(lambda: print("x"))

    def test_assertNotPrints_without_output(self) -> None:
        """Test assertNotPrints method when no output is produced."""
        self.assertNotPrints(lambda: "x")

    def test_assertPrintsRegexp_with_matching_pattern_stdout(self) -> None:
        """Test assertPrintsRegexp method when stdout output matches the regex pattern."""
        self.assertPrintsRegexp("abc", lambda: print("abcde"))

    def test_assertPrintsRegexp_with_non_matching_pattern(self) -> None:
        """Test assertPrintsRegexp method when output does not match the regex pattern."""
        self.assertRaises(
            AssertionError, self.assertPrintsRegexp, "abcdef", lambda: print("abcde")
        )

    def test_assertPrintsRegexp_with_stderr_output(self) -> None:
        """Test assertPrintsRegexp method when output is printed to stderr."""
        # NOTE: assertPrintsRegexp() doesn't necessarily work with logging.warning(),
        # as logging handlers can change (e.g. in PyTest)
        # assertPrintsRegexp() works with both stdout and stderr output
        self.assertPrintsRegexp("abc", lambda: print("abcde", file=sys.stderr))


class TestNoneThrows(
    balance.testutil.BalanceTestCase,
):
    """Test cases for the _verify_value_type utility function."""

    def test__verify_value_type_returns_non_none_value(self) -> None:
        """Test that _verify_value_type returns the value when it is not None."""
        value = "test_value"
        result = balance.testutil._verify_value_type(value)
        self.assertEqual(result, "test_value")

    def test__verify_value_type_raises_value_error_on_none(self) -> None:
        """Test that _verify_value_type raises ValueError when value is None."""
        with self.assertRaises(ValueError) as context:
            balance.testutil._verify_value_type(None)
        self.assertIn("Unexpected None value", str(context.exception))

    def test__verify_value_type_with_correct_type_single_type(self) -> None:
        """Test that _verify_value_type accepts value with correct single type."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = balance.testutil._verify_value_type(df, pd.DataFrame)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result, df)

    def test__verify_value_type_with_incorrect_type_single_type(self) -> None:
        """Test that _verify_value_type raises TypeError when value has incorrect single type."""
        value = "not a dataframe"
        with self.assertRaises(TypeError) as context:
            balance.testutil._verify_value_type(value, pd.DataFrame)
        self.assertIn("Expected type", str(context.exception))
        self.assertIn("DataFrame", str(context.exception))
        self.assertIn("str", str(context.exception))

    def test__verify_value_type_with_correct_type_tuple_of_types(self) -> None:
        """Test that _verify_value_type accepts value when it matches one of multiple types."""
        # Test with string (first type in tuple)
        str_value = "test"
        # pyre-fixme[6]: Tuple types work at runtime but overloads don't support union narrowing
        result_str = balance.testutil._verify_value_type(str_value, (str, int))
        self.assertEqual(result_str, "test")

        # Test with int (second type in tuple)
        int_value = 42
        # pyre-fixme[6]: Tuple types work at runtime but overloads don't support union narrowing
        result_int = balance.testutil._verify_value_type(int_value, (str, int))
        self.assertEqual(result_int, 42)

    def test__verify_value_type_with_incorrect_type_tuple_of_types(self) -> None:
        """Test that _verify_value_type raises TypeError when value doesn't match any type in tuple."""
        value = 3.14  # float, not in (str, int)
        with self.assertRaises(TypeError) as context:
            # pyre-fixme[6]: Tuple types work at runtime but overloads don't support union narrowing
            balance.testutil._verify_value_type(value, (str, int))
        self.assertIn("Expected type", str(context.exception))
        self.assertIn("float", str(context.exception))

    def test__verify_value_type_without_type_check(self) -> None:
        """Test that _verify_value_type works without type checking (backward compatibility)."""
        # Test with various types without type checking
        self.assertEqual(balance.testutil._verify_value_type("test"), "test")
        self.assertEqual(balance.testutil._verify_value_type(42), 42)
        self.assertEqual(balance.testutil._verify_value_type([1, 2, 3]), [1, 2, 3])

    def test__verify_value_type_with_none_and_type_check(self) -> None:
        """Test that _verify_value_type raises ValueError for None even when type checking is enabled."""
        # ValueError should be raised before TypeError check
        with self.assertRaises(ValueError) as context:
            balance.testutil._verify_value_type(None, pd.DataFrame)
        self.assertIn("Unexpected None value", str(context.exception))

    def test__verify_value_type_with_numpy_array(self) -> None:
        """Test that _verify_value_type works with numpy arrays."""
        arr = np.array([1, 2, 3])
        result = balance.testutil._verify_value_type(arr, np.ndarray)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)  # pyre-ignore[6]

    def test__verify_value_type_with_pandas_series(self) -> None:
        """Test that _verify_value_type works with pandas Series."""
        series = pd.Series([1, 2, 3])
        result = balance.testutil._verify_value_type(series, pd.Series)
        self.assertIsInstance(result, pd.Series)
        pd.testing.assert_series_equal(result, series)
