# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import numpy.testing
import pandas as pd

# TODO: remove the use of balance_util in most cases, and just import the functions to be tested directly
from balance import util as balance_util
from balance.sample_class import Sample
from balance.util import _verify_value_type
from balance.utils.input_validation import (
    _extract_series_and_weights,
    _isinstance_sample,
)


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test__check_weighting_methods_input(self) -> None:
        """Test input validation for weighting methods.

        Validates that _check_weighting_methods_input properly validates:
        - DataFrame type requirements
        - Series type requirements
        - Length matching between DataFrame and weights
        - Index matching between DataFrame and weights
        """
        self.assertRaisesRegex(
            TypeError,
            "sample_df must be a pandas DataFrame",
            balance_util._check_weighting_methods_input,
            df=1,
            weights=pd.Series(1),
            object_name="sample",
        )
        self.assertRaisesRegex(
            TypeError,
            "sample_weights must be a pandas Series",
            balance_util._check_weighting_methods_input,
            df=pd.DataFrame({"a": (1, 2)}),
            weights=1,
            object_name="sample",
        )
        self.assertRaisesRegex(
            ValueError,
            "sample_weights must be the same length as sample_df",
            balance_util._check_weighting_methods_input,
            df=pd.DataFrame({"a": [1, 2]}),
            weights=pd.Series([1]),
            object_name="sample",
        )
        self.assertRaisesRegex(
            ValueError,
            "sample_df index must be the same as sample_weights index",
            balance_util._check_weighting_methods_input,
            df=pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
            weights=pd.Series([1, 1], index=[3, 4]),
            object_name="sample",
        )

    def test_guess_id_column(self) -> None:
        """Test automatic identification of ID columns in DataFrames.

        Tests the guess_id_column function's ability to:
        - Automatically identify 'id' columns when present
        - Use custom candidate id column names
        - Use explicitly provided column names
        - Handle cases where no ID column exists
        - Handle ambiguous candidate columns
        - Raise appropriate errors for invalid column names
        """
        # test when id column is presented
        df = pd.DataFrame(
            {
                "a": (0, 1, 2),
                "id": (1, 2, 3),
            }
        )
        self.assertEqual(balance_util.guess_id_column(df), "id")
        self.assertWarnsRegexp(
            "Guessed id column name id for the data",
            balance_util.guess_id_column,
            df,
        )

        # test when custom candidate columns are provided
        df = pd.DataFrame(
            {
                "user_id": (1, 2, 3),
                "a": (0, 1, 2),
            }
        )
        self.assertEqual(
            balance_util.guess_id_column(
                df,
                possible_id_columns=["user_id", "id"],
            ),
            "user_id",
        )
        self.assertWarnsRegexp(
            "Guessed id column name user_id for the data",
            balance_util.guess_id_column,
            df,
            possible_id_columns="user_id",
        )
        self.assertEqual(
            balance_util.guess_id_column(
                df,
                possible_id_columns=["user_id", "user_id"],
            ),
            "user_id",
        )

        with self.assertRaisesRegex(
            ValueError,
            "possible_id_columns cannot contain empty values",
        ):
            balance_util.guess_id_column(
                df,
                possible_id_columns=["user_id", ""],
            )

        with self.assertRaisesRegex(
            TypeError,
            "possible_id_columns must contain only string column names",
        ):
            balance_util.guess_id_column(
                df,
                possible_id_columns=["user_id", 1],
            )

        # test when column_name is passed
        df = pd.DataFrame(
            {
                "a": (0, 1, 2),
                "b": (1, 2, 3),
            }
        )
        self.assertEqual(balance_util.guess_id_column(df, column_name="b"), "b")
        with self.assertRaisesRegex(
            ValueError,
            "Dataframe does not have column*",
        ):
            balance_util.guess_id_column(df, column_name="c")

        # test when no id column is passed and no id column in dataframe
        with self.assertRaisesRegex(
            ValueError,
            "Cannot guess id column name for this DataFrame. None of the possible_id_columns candidates",
        ):
            balance_util.guess_id_column(df)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot guess id column name for this DataFrame. Please provide a value in id_column or possible_id_columns",
        ):
            balance_util.guess_id_column(df, possible_id_columns=[])

        df = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "user_id": (4, 5, 6),
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "Multiple candidate id columns found in the DataFrame",
        ):
            balance_util.guess_id_column(
                df,
                possible_id_columns=["id", "user_id"],
            )

    def test__isinstance_sample(self) -> None:
        """Test type checking for Sample objects.

        Validates that _isinstance_sample correctly distinguishes between:
        - Regular pandas DataFrames
        - Sample objects created from DataFrames
        """
        s_df = pd.DataFrame(
            {
                "a": (0, 1, 2),
                "b": (0, None, 2),
                "c": ("a", "b", "a"),
                "id": (1, 2, 3),
            }
        )
        s = Sample.from_frame(s_df)
        self.assertFalse(_isinstance_sample(s_df))
        self.assertTrue(_isinstance_sample(s))

    def test_isarraylike(self) -> None:
        self.assertFalse(balance_util._is_arraylike(""))
        self.assertFalse(balance_util._is_arraylike("test"))
        self.assertTrue(balance_util._is_arraylike(()))
        self.assertTrue(balance_util._is_arraylike([]))
        self.assertTrue(balance_util._is_arraylike([1, 2]))
        self.assertTrue(balance_util._is_arraylike(range(10)))
        self.assertTrue(balance_util._is_arraylike(np.array([1, 2, "a"])))
        self.assertTrue(balance_util._is_arraylike(pd.Series((1, 2, 3))))

    def test_rm_mutual_nas_basic_functionality(self) -> None:
        """Test basic functionality of rm_mutual_nas with simple arrays."""
        from balance.util import rm_mutual_nas

        # Test with lists containing None values
        result = rm_mutual_nas([1, 2, 3], [2, 3, None])
        self.assertEqual(result, [[1, 2], [2.0, 3.0]])

    def test_rm_mutual_nas_single_arrays(self) -> None:
        """Test rm_mutual_nas with single arrays of various types."""
        from balance.util import rm_mutual_nas

        # Test with numpy arrays without NA values
        d = np.array((0, 1, 2))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), d)

        # Test with multiple arrays without NA values
        d2 = np.array((5, 6, 7))
        numpy.testing.assert_array_equal(rm_mutual_nas(d, d2), (d, d2))

        # Test with None argument
        r = rm_mutual_nas(d, d2, None)
        for i, j in zip(r, (d, d2, None)):
            numpy.testing.assert_array_equal(i, j)

    def test_rm_mutual_nas_with_na_values(self) -> None:
        """Test rm_mutual_nas behavior with various NA values."""
        from balance.util import rm_mutual_nas

        d = np.array((0, 1, 2))
        d2 = np.array((5, 6, 7))
        d4 = np.array((np.nan, 9, -np.inf))
        expected = [np.array((1,)), np.array((6,)), np.array((9,))]

        # Test with NA values
        result = rm_mutual_nas(d, d2, d4)
        for i, j in zip(result, expected):
            numpy.testing.assert_array_equal(i, j)

        # Test with None argument included
        result = rm_mutual_nas(d, d2, d4, None)
        for i, j in zip(result, expected):
            numpy.testing.assert_array_equal(i, j)

        # Test with string arrays
        d5 = np.array(("a", "b", "c"))
        numpy.testing.assert_array_equal(rm_mutual_nas(d5), d5)

        # Test mixed string and numeric with NA
        expected_mixed = [np.array((9,)), np.array(("b",))]
        result = rm_mutual_nas(d4, d5)
        for i, j in zip(result, expected_mixed):
            numpy.testing.assert_array_equal(i, j)

    def test_rm_mutual_nas_single_array_with_na(self) -> None:
        """Test rm_mutual_nas with single arrays containing NA values."""
        from balance.util import rm_mutual_nas

        # Test with numpy array containing None
        d = np.array((0, 1, 2, None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))

        # Test with numpy array containing NaN
        d = np.array((0, 1, 2, np.nan))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))

        # Test with string array containing None
        d = np.array(("a", "b", None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), ("a", "b"))

        # Test with mixed type array containing None
        d = np.array(("a", 1, None))
        # NOTE: In the next test we must define that `dtype=object`
        # since this dtype is preserved from d. Otherwise, using np.array(("a", 1)) will have
        # a dtype of '<U1'
        numpy.testing.assert_array_equal(
            rm_mutual_nas(d), np.array(("a", 1), dtype=object)
        )
        self.assertTrue(isinstance(rm_mutual_nas(d), np.ndarray))

        # Test with tuple containing None
        d = (0, 1, 2, None)
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))
        d = ("a", "b", None)
        numpy.testing.assert_array_equal(rm_mutual_nas(d), ("a", "b"))
        d = ("a", 1, None)
        numpy.testing.assert_array_equal(rm_mutual_nas(d), ("a", 1))
        self.assertTrue(isinstance(rm_mutual_nas(d), tuple))

    def _create_test_arrays(
        self,
    ) -> tuple[
        pd.core.arrays.base.ExtensionArray,
        pd.core.arrays.base.ExtensionArray,
        pd.core.arrays.base.ExtensionArray,
        pd.core.arrays.base.ExtensionArray,
        pd.core.arrays.base.ExtensionArray,
        np.ndarray,
        np.ndarray,
        list[int | float | str],
    ]:
        """Helper method to create test arrays for rm_mutual_nas testing.

        Returns:
            tuple: (x1, x2, x3, x4, x5, x6, x7, x8) - Various array types with different dtypes and NA values
        """
        x1 = pd.array([1, 2, None, np.nan, pd.NA, 3])
        x2 = pd.array([1.1, 2, 3, None, np.nan, pd.NA])
        x3 = pd.array([1.1, 2, 3, 4, 5, 6])
        x4 = pd.array(["1.1", 2, 3, None, np.nan, pd.NA])
        x5 = pd.array(["1.1", "2", "3", None, np.nan, pd.NA], dtype="string")
        x6 = np.array([1, 2, 3.3, 4, 5, 6])
        x7 = np.array([1, 2, 3.3, 4, "5", "6"])
        x8 = [1, 2, 3.3, 4, "5", "6"]

        return x1, x2, x3, x4, x5, x6, x7, x8

    def test_rm_mutual_nas_pandas_arrays(self) -> None:
        """Test rm_mutual_nas with various pandas array types."""
        from balance.util import rm_mutual_nas

        # Get test arrays from helper method
        x1, x2, x3, x4, x5, x6, x7, x8 = self._create_test_arrays()

        # Test that values are correctly filtered
        expected_values = [
            [1, 2],
            [1.1, 2],
            [1.1, 2.0],
            ["1.1", 2],
            ["1.1", "2"],
            [1.0, 2.0],
            ["1", "2"],
            [1, 2],
        ]
        self.assertEqual(
            [list(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)],
            expected_values,
        )

    def test_rm_mutual_nas_type_preservation(self) -> None:
        """Test that rm_mutual_nas preserves input types."""
        from balance.util import rm_mutual_nas

        # Get test arrays from helper method
        x1, x2, x3, x4, x5, x6, x7, x8 = self._create_test_arrays()

        # Test that types are preserved after NA removal
        input_types = [type(x) for x in (x1, x2, x3, x4, x5, x6, x7, x8)]
        result_types = [type(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)]
        self.assertEqual(result_types, input_types)

        # Test specific type preservation
        # x4 is a mixed-type array which in pandas becomes the same type as x2 (both contain floats)
        numpy_array_type = type(x4)
        expected_types = [
            pd.core.arrays.integer.IntegerArray,
            numpy_array_type,
            pd.core.arrays.string_.StringArray,
            np.ndarray,
            np.ndarray,
            list,
        ]
        result_types = [type(x) for x in rm_mutual_nas(x1, x4, x5, x6, x7, x8)]
        self.assertEqual(result_types, expected_types)

        # Test pandas version-specific FloatingArray types
        # NOTE: pd.FloatingArray were only added in pandas version 1.2.0.
        # Before that, they were called PandasArray. For details, see:
        # https://pandas.pydata.org/docs/dev/reference/api/pandas.arrays.FloatingArray.html
        # pyre-ignore[16]: Module `pandas` has no attribute `__version__`.
        if pd.__version__ < "1.2.0":
            expected_floating_types = [
                numpy_array_type,
                numpy_array_type,
            ]
        else:
            expected_floating_types = [
                pd.core.arrays.floating.FloatingArray,
                pd.core.arrays.floating.FloatingArray,
            ]
        self.assertEqual(
            [type(x) for x in rm_mutual_nas(x2, x3)], expected_floating_types
        )

    def test_rm_mutual_nas_dtype_preservation(self) -> None:
        """Test that rm_mutual_nas preserves dtypes for numpy and pandas arrays."""
        from balance.util import rm_mutual_nas

        # Get test arrays from helper method (only need first 7 for this test)
        x1, x2, x3, x4, x5, x6, x7, _ = self._create_test_arrays()

        # Test that dtypes are preserved
        input_dtypes = [x.dtype for x in (x1, x2, x3, x4, x5, x6, x7)]
        result_dtypes = [x.dtype for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7)]
        self.assertEqual(result_dtypes, input_dtypes)

    def test_rm_mutual_nas_pandas_series_index_preservation(self) -> None:
        """Test that rm_mutual_nas preserves pandas Series indexes."""
        from balance.util import rm_mutual_nas

        x1 = pd.Series([1, 2, 3, 4])
        x2 = pd.Series([np.nan, 2, 3, 4])
        x3 = np.array([1, 2, 3, 4])

        # Test that index is preserved when no values are removed
        result_0 = rm_mutual_nas(x1, x3)[0]
        assert isinstance(result_0, pd.Series)
        self.assertEqual(result_0.to_dict(), {0: 1, 1: 2, 2: 3, 3: 4})

        # Test with sorted series
        sorted_x1 = x1.sort_values(ascending=False)
        result_1 = rm_mutual_nas(sorted_x1, x3)[0]
        assert isinstance(result_1, pd.Series)
        self.assertEqual(result_1.to_dict(), {3: 4, 2: 3, 1: 2, 0: 1})

        # Test that index is preserved when NA values are removed
        result_2 = rm_mutual_nas(x1, x2)[0]
        assert isinstance(result_2, pd.Series)
        self.assertEqual(result_2.to_dict(), {1: 2, 2: 3, 3: 4})

        # Test with sorted series and NA removal
        result_3 = rm_mutual_nas(sorted_x1, x2)[0]
        assert isinstance(result_3, pd.Series)
        self.assertEqual(result_3.to_dict(), {3: 4, 2: 3, 1: 2})

    def test_rm_mutual_nas_error_handling(self) -> None:
        """Test that rm_mutual_nas properly handles error conditions."""
        from balance.util import rm_mutual_nas

        d = np.array((0, 1, 2))

        # Test arrays of different lengths
        d3a = np.array((5, 6, 7, 8))
        self.assertRaisesRegex(
            ValueError,
            "All arrays must be of same length",
            rm_mutual_nas,
            d,
            d3a,
        )

        # Test non-arraylike arguments
        d3b = "a"
        self.assertRaisesRegex(
            ValueError,
            "All arguments must be arraylike",
            rm_mutual_nas,
            d3b,
            d3b,
        )

        # Test mixed arraylike and non-arraylike arguments
        self.assertRaises(ValueError, rm_mutual_nas, d, "a")

    def test_choose_variables(self) -> None:
        from balance.util import choose_variables

        # For one dataframe
        self.assertEqual(
            choose_variables(pd.DataFrame({"a": [1], "b": [2]})),
            ["a", "b"],
        )
        # For two dataframes
        # Not providing variables
        self.assertEqual(
            choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}),
                pd.DataFrame({"c": [1], "b": [2]}),
                variables=None,
            ),
            ["b"],
        )
        self.assertEqual(
            choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"c": [1], "b": [2]})
            ),
            ["b"],
        )

        self.assertWarnsRegexp(
            "Ignoring variables not present in all Samples",
            choose_variables,
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"c": [1], "d": [2]}),
        )
        self.assertWarnsRegexp(
            "Sample and target have no variables in common",
            choose_variables,
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"c": [1], "d": [2]}),
        )

        self.assertEqual(
            choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"c": [1], "d": [2]})
            ),
            [],
        )

        with self.assertRaisesRegex(
            ValueError, "requested variables are not in all Samples"
        ):
            choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}),
                pd.DataFrame({"c": [1], "b": [2]}),
                variables=["a"],
            )

        #  Three dataframes
        self.assertEqual(
            choose_variables(
                pd.DataFrame({"a": [1], "b": [2], "c": [2]}),
                pd.DataFrame({"c": [1], "b": [2]}),
                pd.DataFrame({"a": [1], "b": [2]}),
            ),
            ["b"],
        )

        df1 = pd.DataFrame({"a": [1], "b": [2], "c": [2]})
        df2 = pd.DataFrame({"c": [1], "b": [2]})

        with self.assertRaisesRegex(
            ValueError, "requested variables are not in all Samples: {'a'}"
        ):
            choose_variables(df1, df2, variables=["a", "b", "c"])

        # Control order
        df1 = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4], "C": [5, 6], "E": [1, 1], "F": [1, 1]}
        )
        df2 = pd.DataFrame(
            {"C": [7, 8], "J": [9, 10], "B": [11, 12], "K": [1, 1], "A": [1, 1]}
        )

        self.assertEqual(
            choose_variables(df1, df2),
            ["A", "B", "C"],
        )
        self.assertEqual(
            choose_variables(df1, df2, df_for_var_order=1),
            ["C", "B", "A"],
        )
        self.assertEqual(
            choose_variables(df1, df2, variables=["B", "A"]),
            ["B", "A"],
        )

    def test__true_false_str_to_bool(self) -> None:
        self.assertFalse(balance_util._true_false_str_to_bool("falsE"))
        self.assertTrue(balance_util._true_false_str_to_bool("TrUe"))
        with self.assertRaisesRegex(
            ValueError,
            "Banana is not an accepted value, please pass either 'True' or 'False'*",
        ):
            balance_util._true_false_str_to_bool("Banana")

    def test__verify_value_type(self) -> None:
        """Test _verify_value_type with various inputs including error cases."""
        # Test successful cases
        success_cases = [
            # (value, expected_type, description)
            (
                pd.DataFrame({"a": [1, 2, 3]}),
                pd.DataFrame,
                "DataFrame with correct type",
            ),
            ("test", str, "String with correct type"),
            (123, int, "Integer with correct type"),
            ([1, 2, 3], list, "List with correct type"),
            ("test", (str, int), "String with tuple of types"),
            (42, (str, int), "Integer with tuple of types"),
            ("test", None, "String without type check"),
            (123, None, "Integer without type check"),
            ([1, 2, 3], None, "List without type check"),
        ]

        for value, expected_type, description in success_cases:
            with self.subTest(description=description):
                # pyre-ignore[6]: Testing runtime behavior with various types
                result = _verify_value_type(value, expected_type)
                self.assertEqual(result, value)

        # Test error cases
        error_cases = [
            # (value, expected_type, expected_exception, description)
            (None, None, ValueError, "None value raises ValueError"),
            (None, str, ValueError, "None with type check raises ValueError"),
            ("not an int", int, TypeError, "Wrong type raises TypeError"),
            (
                [1, 2, 3],
                (str, int),
                TypeError,
                "List with tuple types raises TypeError",
            ),
        ]

        for value, expected_type, expected_exception, description in error_cases:
            with self.subTest(description=description):
                with self.assertRaises(expected_exception):
                    # pyre-ignore[6]: Testing runtime behavior with various types
                    _verify_value_type(value, expected_type)

    def test__float_or_none(self) -> None:
        """Test _float_or_none with various inputs."""
        test_cases = [
            # (input_value, expected_result, description)
            (None, None, "None input returns None"),
            ("None", None, "String 'None' returns None"),
            (3.14, 3.14, "Float returns same float"),
            (42, 42.0, "Int returns float"),
            ("3.14", 3.14, "Numeric string returns float"),
        ]

        for input_value, expected_result, description in test_cases:
            with self.subTest(description=description):
                result = balance_util._float_or_none(input_value)
                self.assertEqual(result, expected_result)
                if expected_result is not None:
                    self.assertIsInstance(result, float)

    def test_find_items_index_in_list(self) -> None:
        """Test find_items_index_in_list with various input scenarios."""
        test_cases_numeric = [
            ([1, 2, 3, 4, 5, 6, 7], [2, 7], [1, 6], "Numeric list with found items"),
            ([1, 2, 3, 4, 5, 6, 7], [1000], [], "Numeric list with missing items"),
            ([10, 20, 30, 40], [20, 100, 40, 200], [1, 3], "Mixed found and missing"),
        ]

        test_cases_string = [
            (
                ["a", "b", "c"],
                ["c", "c", "a"],
                [2, 2, 0],
                "String list with duplicates",
            ),
        ]

        test_cases_edge = [
            ([1, 2, 3], [], [], "Empty items list"),
        ]

        all_test_cases = test_cases_numeric + test_cases_string + test_cases_edge

        for a_list, items, expected, description in all_test_cases:
            with self.subTest(description=description):
                result = balance_util.find_items_index_in_list(a_list, items)
                self.assertEqual(result, expected)
                if result:  # Check type only if result is not empty
                    self.assertIsInstance(result[0], int)

    def test_get_items_from_list_via_indices(self) -> None:
        """Test get_items_from_list_via_indices with various input scenarios."""
        test_cases = [
            # (a_list, indices, expected, description)
            (["a", "b", "c", "d"], [2, 0], ["c", "a"], "String valid indices"),
            ([10, 20, 30, 40, 50], [0, 2, 4], [10, 30, 50], "Numeric list"),
            ([1, 2, 3], [], [], "Empty indices"),
            (["x", "y", "z"], [1], ["y"], "Single index"),
            (["a", "b", "c"], [0, 0, 2, 0], ["a", "a", "c", "a"], "Duplicate indices"),
        ]

        for a_list, indices, expected, description in test_cases:
            with self.subTest(description=description):
                result = balance_util.get_items_from_list_via_indices(a_list, indices)
                self.assertEqual(result, expected)

        # Test IndexError case separately
        with self.assertRaises(IndexError):
            balance_util.get_items_from_list_via_indices(["a", "b", "c"], [100])


class TestExtractSeriesAndWeights(balance.testutil.BalanceTestCase):
    """Test _extract_series_and_weights function (lines 107-114)."""

    def test_weights_length_mismatch(self) -> None:
        """Test that ValueError is raised when weights length doesn't match series (line 108)."""
        series = pd.Series([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0])  # Wrong length

        with self.assertRaisesRegex(ValueError, "Weights must match"):
            _extract_series_and_weights(series, weights, "test")

    def test_empty_filtered_series(self) -> None:
        """Test that ValueError is raised when filtered series is empty (line 113)."""
        # Series with all null values - after filtering, series is empty
        series = pd.Series([None, None, None])
        weights = np.array([1.0, 1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "must contain at least one non-null"):
            _extract_series_and_weights(series, weights, "test_label")

    def test_successful_extraction(self) -> None:
        """Test successful extraction of series and weights."""
        series = pd.Series([1.0, None, 3.0])
        weights = np.array([1.0, 2.0, 3.0])

        result_series, result_weights = _extract_series_and_weights(
            series, weights, "test"
        )

        self.assertEqual(list(result_series), [1.0, 3.0])
        self.assertEqual(list(result_weights), [1.0, 3.0])


class TestChooseVariablesEmptySet(balance.testutil.BalanceTestCase):
    """Test choose_variables with empty variables set (line 350)."""

    def test_empty_variables_set_treated_as_none(self) -> None:
        """Test that empty variables set is treated as None (line 350)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        # Empty set should be treated as None and return intersection of columns
        result = balance_util.choose_variables(df1, df2, variables=set())
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_empty_list_treated_as_none(self) -> None:
        """Test that empty list is treated as None."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})

        # Empty list should be treated as None
        result = balance_util.choose_variables(df1, df2, variables=[])
        # Should return intersection (only 'a')
        self.assertEqual(result, ["a"])


class TestIsinstanceSample(balance.testutil.BalanceTestCase):
    """Test _isinstance_sample function behavior."""

    def test_isinstance_sample_returns_false_for_non_sample(self) -> None:
        """Test _isinstance_sample returns False for non-Sample objects."""
        self.assertFalse(_isinstance_sample("not a sample"))
        self.assertFalse(_isinstance_sample(123))
        self.assertFalse(_isinstance_sample([1, 2, 3]))
        self.assertFalse(_isinstance_sample(pd.DataFrame({"a": [1, 2]})))

    def test_isinstance_sample_returns_true_for_sample(self) -> None:
        """Test _isinstance_sample returns True for Sample objects."""
        sample = Sample.from_frame(pd.DataFrame({"id": [1, 2], "a": [3, 4]}))
        self.assertTrue(_isinstance_sample(sample))


class TestCoerceToNumericAndValidate(balance.testutil.BalanceTestCase):
    """Test cases for _coerce_to_numeric_and_validate function."""

    def test_successful_numeric_conversion(self) -> None:
        """Test successful conversion of numeric series."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0, 3.0])

        result_vals, result_weights = _coerce_to_numeric_and_validate(
            series, weights, "test"
        )

        np.testing.assert_array_equal(result_vals, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result_weights, np.array([1.0, 2.0, 3.0]))

    def test_conversion_with_coercible_strings(self) -> None:
        """Test conversion when series contains numeric strings."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series(["1", "2", "3"])
        weights = np.array([1.0, 2.0, 3.0])

        result_vals, result_weights = _coerce_to_numeric_and_validate(
            series, weights, "test"
        )

        np.testing.assert_array_equal(result_vals, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result_weights, np.array([1.0, 2.0, 3.0]))

    def test_partial_conversion_with_some_non_numeric(self) -> None:
        """Test conversion when some values can't be converted to numeric."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        # Mix of convertible and non-convertible values
        series = pd.Series([1.0, "abc", 3.0])
        weights = np.array([1.0, 2.0, 3.0])

        result_vals, result_weights = _coerce_to_numeric_and_validate(
            series, weights, "test"
        )

        # "abc" should be coerced to NaN and dropped
        np.testing.assert_array_equal(result_vals, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(result_weights, np.array([1.0, 3.0]))

    def test_raises_on_all_non_numeric(self) -> None:
        """Test that ValueError is raised when all values fail numeric conversion.

        This is the key test case that verifies the previously unreachable code
        path is now testable. When all values in a series cannot be converted
        to numeric, the function should raise ValueError.
        """
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        # All values are non-numeric strings
        series = pd.Series(["abc", "def", "ghi"])
        weights = np.array([1.0, 2.0, 3.0])

        with self.assertRaisesRegex(
            ValueError, "must contain at least one valid numeric value"
        ):
            _coerce_to_numeric_and_validate(series, weights, "test")

    def test_raises_on_empty_series(self) -> None:
        """Test that ValueError is raised when series is empty."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series([], dtype=float)
        weights = np.array([])

        with self.assertRaisesRegex(
            ValueError, "must contain at least one valid numeric value"
        ):
            _coerce_to_numeric_and_validate(series, weights, "empty series")

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are properly dropped during conversion."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series([1.0, np.nan, 3.0])
        weights = np.array([1.0, 2.0, 3.0])

        result_vals, result_weights = _coerce_to_numeric_and_validate(
            series, weights, "test"
        )

        np.testing.assert_array_equal(result_vals, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(result_weights, np.array([1.0, 3.0]))

    def test_raises_on_all_nan_values(self) -> None:
        """Test that ValueError is raised when all values are NaN after conversion."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series([np.nan, np.nan, np.nan])
        weights = np.array([1.0, 2.0, 3.0])

        with self.assertRaisesRegex(
            ValueError, "must contain at least one valid numeric value"
        ):
            _coerce_to_numeric_and_validate(series, weights, "all_nan")

    def test_preserves_weight_alignment(self) -> None:
        """Test that weights are correctly aligned after dropping invalid values."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        # Series with some non-convertible values at specific positions
        series = pd.Series([1.0, "bad", 3.0, "bad", 5.0], index=[0, 1, 2, 3, 4])
        weights = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result_vals, result_weights = _coerce_to_numeric_and_validate(
            series, weights, "test"
        )

        # Only positions 0, 2, 4 should remain
        np.testing.assert_array_equal(result_vals, np.array([1.0, 3.0, 5.0]))
        np.testing.assert_array_equal(result_weights, np.array([10.0, 30.0, 50.0]))

    def test_error_message_includes_label(self) -> None:
        """Test that error message includes the provided label."""
        from balance.utils.input_validation import _coerce_to_numeric_and_validate

        series = pd.Series(["abc", "def"])
        weights = np.array([1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "my_custom_label"):
            _coerce_to_numeric_and_validate(series, weights, "my_custom_label")
