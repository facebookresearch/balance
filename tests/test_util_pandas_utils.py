# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from copy import deepcopy

import numpy as np
import pandas as pd

import balance.testutil
from balance import util as balance_util
from balance.sample_class import Sample
from balance.util import _coerce_scalar


class TestUtilPandasUtils(balance.testutil.BalanceTestCase):
    def test__coerce_scalar(self) -> None:
        """Ensure scalar coercion handles edge cases without raising errors."""

        self.assertTrue(np.isnan(_coerce_scalar(None)))
        self.assertTrue(np.isnan(_coerce_scalar([1, 2, 3])))
        self.assertEqual(_coerce_scalar(5), 5.0)
        self.assertEqual(_coerce_scalar(np.float64(2.5)), 2.5)
        self.assertTrue(np.isnan(_coerce_scalar("not-a-number")))
        self.assertEqual(_coerce_scalar(True), 1.0)
        self.assertEqual(_coerce_scalar("7.125"), 7.125)
        self.assertTrue(np.isnan(_coerce_scalar(complex(1, 2))))
        self.assertTrue(np.isnan(_coerce_scalar(())))
        self.assertTrue(np.isnan(_coerce_scalar(np.array([4.5]))))

    def test__is_categorical_dtype_object(self) -> None:
        series = pd.Series(["a", "b", "c"])
        result = balance_util._is_categorical_dtype(series)
        self.assertTrue(result)

    def test__is_categorical_dtype_numeric(self) -> None:
        int_series = pd.Series([1, 2, 3])
        float_series = pd.Series([4.0, 5.0, 6.0])

        int_result = balance_util._is_categorical_dtype(int_series)
        float_result = balance_util._is_categorical_dtype(float_series)

        self.assertFalse(int_result)
        self.assertFalse(float_result)

    def test__is_categorical_dtype_with_category_dtype(self) -> None:
        series = pd.Series(pd.Categorical(["a", "b", "c"]))
        result = balance_util._is_categorical_dtype(series)
        self.assertTrue(result)

    def test__is_categorical_dtype_with_string_dtype(self) -> None:
        series = pd.Series(["a", "b", "c"], dtype="string")
        result = balance_util._is_categorical_dtype(series)
        self.assertTrue(result)

    def test__is_categorical_dtype_bool(self) -> None:
        series = pd.Series([True, False, True])
        result = balance_util._is_categorical_dtype(series)
        self.assertFalse(result)

    def test_truncate_text(self) -> None:
        self.assertEqual(balance_util._truncate_text("a" * 6, length=5), "a" * 5 + "...")
        self.assertEqual(balance_util._truncate_text("a" * 4, length=5), "a" * 4)
        self.assertEqual(balance_util._truncate_text("a" * 5, length=5), "a" * 5)

    def test__dict_intersect(self) -> None:
        d1 = {"a": 1, "b": 2}
        d2 = {"c": 3, "b": 2222}
        self.assertEqual(balance_util._dict_intersect(d1, d2), {"b": 2})

    def test__astype_in_df_from_dtypes(self) -> None:
        df = pd.DataFrame({"id": ("1", "2"), "a": (1.0, 2.0), "weight": (1.0, 2.0)})
        df_orig = pd.DataFrame(
            {"id": (1, 2), "a": (1, 2), "forest": ("tree", "banana")}
        )

        self.assertEqual(
            df.dtypes.to_dict(),
            {"id": np.dtype("O"), "a": np.dtype("float64"), "weight": np.dtype("float64")},
        )
        self.assertEqual(
            df_orig.dtypes.to_dict(),
            {"id": np.dtype("int64"), "a": np.dtype("int64"), "forest": np.dtype("O")},
        )

        df_fixed = balance_util._astype_in_df_from_dtypes(df, df_orig.dtypes)
        self.assertEqual(
            df_fixed.dtypes.to_dict(),
            {"id": np.dtype("int64"), "a": np.dtype("int64"), "weight": np.dtype("float64")},
        )

    def test__are_dtypes_equal(self) -> None:
        df1 = pd.DataFrame({"int": np.arange(5), "flt": np.random.randn(5)})
        df2 = pd.DataFrame({"flt": np.random.randn(5), "int": np.random.randn(5)})
        df11 = pd.DataFrame(
            {"int": np.arange(5), "flt": np.random.randn(5), "miao": np.random.randn(5)}
        )

        self.assertTrue(
            balance_util._are_dtypes_equal(df1.dtypes, df1.dtypes)["is_equal"]
        )
        self.assertFalse(
            balance_util._are_dtypes_equal(df1.dtypes, df2.dtypes)["is_equal"]
        )
        self.assertFalse(
            balance_util._are_dtypes_equal(df11.dtypes, df2.dtypes)["is_equal"]
        )

    def test__warn_of_df_dtypes_change(self) -> None:
        df = pd.DataFrame({"int": np.arange(5), "flt": np.random.randn(5)})
        new_df = deepcopy(df)
        new_df.int = new_df.int.astype(float)
        new_df.flt = new_df.flt.astype(int)

        self.assertWarnsRegexp(
            "The dtypes of new_df were changed from the original dtypes of the input df, here are the differences - ",
            balance_util._warn_of_df_dtypes_change,
            df.dtypes,
            new_df.dtypes,
        )

    def test__make_df_column_names_unique(self) -> None:
        data = {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "A2": [7, 8, 9],
            "C": [10, 11, 12],
        }

        df1 = pd.DataFrame(data)
        df1.columns = ["A", "B", "A", "A"]

        self.assertEqual(
            balance_util._make_df_column_names_unique(df1).to_dict(),
            {
                "A": {0: 1, 1: 2, 2: 3},
                "B": {0: 4, 1: 5, 2: 6},
                "A_1": {0: 7, 1: 8, 2: 9},
                "A_2": {0: 10, 1: 11, 2: 12},
            },
        )

    def test__safe_replace_and_infer(self) -> None:
        """Test safe replacement and dtype inference to avoid pandas deprecation warnings."""
        series_with_inf = pd.Series([1.0, np.inf, 2.0, -np.inf, 3.0])
        result = balance_util._safe_replace_and_infer(series_with_inf)
        expected = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        pd.testing.assert_series_equal(result, expected)

        df_with_inf = pd.DataFrame({"a": [1.0, np.inf, 2.0], "b": [-np.inf, 3.0, 4.0]})
        result = balance_util._safe_replace_and_infer(df_with_inf)
        expected = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan, 3.0, 4.0]})
        pd.testing.assert_frame_equal(result, expected)

        series_test = pd.Series([1, 2, 3, 4])
        result = balance_util._safe_replace_and_infer(series_test, to_replace=2, value=99)
        expected = pd.Series([1, 99, 3, 4])
        pd.testing.assert_series_equal(result, expected)

        series_obj = pd.Series(["a", "b", "c"], dtype="object")
        result = balance_util._safe_replace_and_infer(series_obj, to_replace="b", value="x")
        expected = pd.Series(["a", "x", "c"], dtype="object")
        pd.testing.assert_series_equal(result, expected)

    def test__safe_fillna_and_infer(self) -> None:
        """Test safe NA filling and dtype inference to avoid pandas deprecation warnings."""
        series_with_nan = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = balance_util._safe_fillna_and_infer(series_with_nan, value=0)
        expected = pd.Series([1.0, 0.0, 2.0, 0.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

        df_with_nan = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan, 3.0, 4.0]})
        result = balance_util._safe_fillna_and_infer(df_with_nan, value=-1)
        expected = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [-1.0, 3.0, 4.0]})
        pd.testing.assert_frame_equal(result, expected)

        series_str = pd.Series(["a", None, "c"])
        result = balance_util._safe_fillna_and_infer(series_str, value="_NA")
        expected = pd.Series(["a", "_NA", "c"])
        pd.testing.assert_series_equal(result, expected)

        series_test = pd.Series([1, None, 3])
        result = balance_util._safe_fillna_and_infer(series_test)
        expected = pd.Series([1.0, np.nan, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test__safe_groupby_apply(self) -> None:
        """Test safe groupby apply operations that handle include_groups parameter."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C"],
                "value": [1, 2, 3, 4, 5],
                "other": [10, 20, 30, 40, 50],
            }
        )

        result = balance_util._safe_groupby_apply(
            df, "group", lambda x: x["value"].sum()
        )
        expected = pd.Series([3, 7, 5], index=pd.Index(["A", "B", "C"], name="group"))
        pd.testing.assert_series_equal(result, expected)

        df_multi = pd.DataFrame(
            {
                "group1": ["A", "A", "B", "B"],
                "group2": ["X", "Y", "X", "Y"],
                "value": [1, 2, 3, 4],
            }
        )
        result = balance_util._safe_groupby_apply(
            df_multi, ["group1", "group2"], lambda x: x["value"].mean()
        )
        expected = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=pd.MultiIndex.from_tuples(
                [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")],
                names=["group1", "group2"],
            ),
        )
        pd.testing.assert_series_equal(result, expected)

        result = balance_util._safe_groupby_apply(df, "group", lambda x: len(x))
        expected = pd.Series([2, 2, 1], index=pd.Index(["A", "B", "C"], name="group"))
        pd.testing.assert_series_equal(result, expected)

    def test__safe_show_legend(self) -> None:
        """Test safe legend display that only shows legends when there are labeled artists."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="line1")
        ax.plot([1, 2, 3], [3, 2, 1], label="line2")

        balance_util._safe_show_legend(ax)

        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 2)

        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.plot([1, 2, 3], [3, 2, 1])

        balance_util._safe_show_legend(ax)

        legend = ax.get_legend()
        self.assertIsNone(legend)

        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="labeled")
        ax.plot([1, 2, 3], [3, 2, 1])

        balance_util._safe_show_legend(ax)

        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 1)
        self.assertEqual(legend.get_texts()[0].get_text(), "labeled")

        plt.close(fig)

    def test__safe_divide_with_zero_handling(self) -> None:
        """Test safe division with proper numpy error state management."""
        result = balance_util._safe_divide_with_zero_handling(10, 2)
        self.assertEqual(result, 5.0)

        numerator = np.array([1, 2, 3, 4])
        denominator = np.array([1, 0, 3, 2])
        result = balance_util._safe_divide_with_zero_handling(numerator, denominator)
        expected = np.array([1.0, np.inf, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

        num_series = pd.Series([10, 20, 30])
        den_series = pd.Series([2, 0, 5])
        result = balance_util._safe_divide_with_zero_handling(num_series, den_series)
        expected = pd.Series([5.0, np.inf, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test__process_series_for_missing_mask(self) -> None:
        """Test _process_series_for_missing_mask with various input scenarios."""
        test_cases = [
            (
                pd.Series([1.0, 2.0, np.inf, -np.inf, np.nan, 5.0]),
                pd.Series([False, False, True, True, True, False]),
                "Series with inf, -inf, nan, and regular values",
            ),
            (
                pd.Series([1.0, 2.0, 3.0, 4.0]),
                pd.Series([False, False, False, False]),
                "Series with no missing values",
            ),
            (
                pd.Series([np.inf, -np.inf, np.nan]),
                pd.Series([True, True, True]),
                "Series with only missing values",
            ),
        ]

        for input_series, expected_mask, description in test_cases:
            with self.subTest(description=description):
                result = balance_util._process_series_for_missing_mask(input_series)
                pd.testing.assert_series_equal(result, expected_mask)

    def test__pd_convert_all_types(self) -> None:
        """Test _pd_convert_all_types with various conversion scenarios."""
        df1 = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3], dtype=pd.Int64Dtype()),
                "b": pd.array([4.0, 5.0, 6.0], dtype=np.float64),
            }
        )
        result1 = balance_util._pd_convert_all_types(df1, "Int64", "float64")
        self.assertEqual(result1["a"].dtype, np.float64)
        self.assertEqual(result1["b"].dtype, np.float64)

        df2 = pd.DataFrame(
            {
                "z": pd.array([1, 2], dtype=pd.Int64Dtype()),
                "a": pd.array([3, 4], dtype=np.float64),
                "m": pd.array([5, 6], dtype=pd.Int64Dtype()),
            }
        )
        result2 = balance_util._pd_convert_all_types(df2, "Int64", "float64")
        self.assertEqual(result2["z"].dtype, np.float64)
        self.assertEqual(result2["a"].dtype, np.float64)
        self.assertEqual(result2["m"].dtype, np.float64)
        self.assertEqual(list(result2.columns), ["z", "a", "m"])

    def test__truncate_text(self) -> None:
        """Test _truncate_text with various string lengths."""
        test_cases = [
            ("Hello", 10, "Hello", "Short string not truncated"),
            (
                "This is a very long string that needs truncation",
                10,
                "This is a ...",
                "Long string truncated with ellipsis",
            ),
            ("1234567890", 10, "1234567890", "Exact length string not truncated"),
        ]

        for input_text, length, expected_result, description in test_cases:
            with self.subTest(description=description):
                result = balance_util._truncate_text(input_text, length)
                self.assertEqual(result, expected_result)
                if len(input_text) > length:
                    self.assertEqual(len(result), length + 3)

    def test_TruncationFormatter(self) -> None:
        """Test TruncationFormatter with long and short log messages."""
        formatter = balance_util.TruncationFormatter("%(message)s")
        max_message_length = 2000
        ellipsis_length = 3

        test_cases = [
            (
                "x" * (max_message_length + 1000),
                True,
                "Long message gets truncated",
            ),
            (
                "This is a short message",
                False,
                "Short message remains unchanged",
            ),
        ]

        for message, should_truncate, description in test_cases:
            with self.subTest(description=description):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
                result = formatter.format(record)

                if should_truncate:
                    self.assertEqual(len(result), max_message_length + ellipsis_length)
                    self.assertTrue(result.endswith("..."))
                else:
                    self.assertEqual(result, message)


class TestSampleHighCardinalityWarnings(balance.testutil.BalanceTestCase):
    """Tests for high-cardinality feature warnings during adjustment."""

    def test_warns_for_high_cardinality_features_with_nas(self) -> None:
        """Adjust should warn when high-cardinality features with NAs lead to equal weights."""
        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
                "id": range(11),
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
                "id": range(11),
            }
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_warns_for_high_cardinality_features_with_nas_when_dropping(
        self,
    ) -> None:
        """The high-cardinality NA warning should surface even when NAs are dropped."""
        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            {"identifier": unique_values + [np.nan], "id": range(11)}
        )
        target_df = pd.DataFrame(
            {"identifier": unique_values + [np.nan], "id": range(11)}
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(
                target, variables=["identifier"], num_lambdas=1, na_action="drop"
            )

        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_warns_for_high_cardinality_categoricals_with_nas(self) -> None:
        """Categorical dtype columns with high cardinality and NAs should be flagged."""
        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(
                    [f"user_{i}" for i in range(9)] + [np.nan], dtype="category"
                ),
                "id": range(10),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=9" in log
                for log in logs.output
            )
        )

    def test_does_not_flag_low_cardinality_categoricals_with_nas(self) -> None:
        """Low-cardinality categoricals with NAs should not be reported as a cause."""
        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(["a", "a", "b", np.nan], dtype="category"),
                "id": range(4),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            result = sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertTrue(np.allclose(result.weight_column, np.ones(len(sample_df))))
        self.assertFalse(
            any("High-cardinality features detected" in log for log in logs.output)
        )

    def test_warns_for_high_cardinality_features_without_nas(self) -> None:
        """High-cardinality categoricals should be reported even without missing values."""
        identifiers = [f"user_{i}" for i in range(8)]
        sample_df = pd.DataFrame(
            {
                "identifier": identifiers,
                "signal": np.concatenate((np.zeros(4), np.ones(4))),
                "id": range(8),
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": identifiers[::-1],
                "signal": np.concatenate((np.ones(4), np.zeros(4))),
                "id": range(8),
            }
        )

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["identifier", "signal"], num_lambdas=1)

        self.assertTrue(
            any(
                "High-cardinality features detected" in log
                and "identifier (unique=8" in log
                for log in logs.output
            )
        )

    def test_does_not_warn_for_high_cardinality_numeric_features(self) -> None:
        """High-cardinality numeric features (e.g., IDs) should NOT be reported."""
        identifiers = np.arange(12)
        sample_df = pd.DataFrame({"identifier": identifiers, "id": range(12)})
        target_df = pd.DataFrame({"identifier": identifiers, "id": range(12)})

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["identifier"], num_lambdas=1)

        self.assertFalse(
            any("High-cardinality features detected" in log for log in logs.output),
            "Numeric features should not be flagged as high-cardinality.",
        )

    def test_high_cardinality_warning_sorts_by_cardinality(self) -> None:
        """Warnings should list columns from highest to lowest cardinality."""
        sample_df = pd.DataFrame(
            {
                "higher": [f"user_{i}" for i in range(7)],
                "high": [f"alias_{i}" for i in range(6)] + ["alias_0"],
                "id": range(7),
            }
        )
        target_df = sample_df.copy()

        sample = Sample.from_frame(sample_df)
        target = Sample.from_frame(target_df)

        with self.assertLogs("balance", level="WARNING") as logs:
            sample.adjust(target, variables=["higher", "high"], num_lambdas=1)

        warning_logs = [
            log for log in logs.output if "High-cardinality features detected" in log
        ]
        self.assertTrue(warning_logs)
        self.assertTrue(
            warning_logs[0].find("higher (unique")
            < warning_logs[0].find(", high (unique"),
            "Expected higher-cardinality column to appear first in warning.",
        )
