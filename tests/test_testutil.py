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
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")
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
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")
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
            {"a": (1, 2, 3), "b": (4, 5, 6)}, columns=("b", "a")
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
