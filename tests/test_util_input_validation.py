# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import numpy as np
import numpy.testing
import pandas as pd

import balance.testutil
from balance import util as balance_util
from balance.sample_class import Sample
from balance.util import _verify_value_type


class TestUtilInputValidation(balance.testutil.BalanceTestCase):
    def test__check_weighting_methods_input(self) -> None:
        """Test input validation for weighting methods."""
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
        """Test automatic identification of ID columns in DataFrames."""
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

        with self.assertRaisesRegex(
            ValueError,
            "Cannot guess id column name for this DataFrame. Please provide a value in id_column",
        ):
            balance_util.guess_id_column(df)

    def test__isinstance_sample(self) -> None:
        """Test type checking for Sample objects."""
        from balance.util import _isinstance_sample

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

        result = rm_mutual_nas([1, 2, 3], [2, 3, None])
        self.assertEqual(result, [[1, 2], [2.0, 3.0]])

    def test_rm_mutual_nas_single_arrays(self) -> None:
        """Test rm_mutual_nas with single arrays of various types."""
        from balance.util import rm_mutual_nas

        d = np.array((0, 1, 2))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), d)

        d2 = np.array((5, 6, 7))
        numpy.testing.assert_array_equal(rm_mutual_nas(d, d2), (d, d2))

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

        result = rm_mutual_nas(d, d2, d4)
        for i, j in zip(result, expected):
            numpy.testing.assert_array_equal(i, j)

        result = rm_mutual_nas(d, d2, d4, None)
        for i, j in zip(result, expected):
            numpy.testing.assert_array_equal(i, j)

        d5 = np.array(("a", "b", "c"))
        numpy.testing.assert_array_equal(rm_mutual_nas(d5), d5)

        expected_mixed = [np.array((9,)), np.array(("b",))]
        result = rm_mutual_nas(d4, d5)
        for i, j in zip(result, expected_mixed):
            numpy.testing.assert_array_equal(i, j)

    def test_rm_mutual_nas_single_array_with_na(self) -> None:
        """Test rm_mutual_nas with single arrays containing NA values."""
        from balance.util import rm_mutual_nas

        d = np.array((0, 1, 2, None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))

        d = np.array((0, 1, 2, np.nan))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))

        d = np.array(("a", "b", None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), ("a", "b"))

        d = np.array(("a", 1, None))
        numpy.testing.assert_array_equal(
            rm_mutual_nas(d), np.array(("a", 1), dtype=object)
        )
        self.assertTrue(isinstance(rm_mutual_nas(d), np.ndarray))

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
        """Helper method to create test arrays for rm_mutual_nas testing."""
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

        x1, x2, x3, x4, x5, x6, x7, x8 = self._create_test_arrays()

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

        x1, x2, x3, x4, x5, x6, x7, x8 = self._create_test_arrays()

        input_types = [type(x) for x in (x1, x2, x3, x4, x5, x6, x7, x8)]
        result_types = [type(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)]
        self.assertEqual(result_types, input_types)

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

        x1, x2, x3, x4, x5, x6, x7, _ = self._create_test_arrays()

        input_dtypes = [x.dtype for x in (x1, x2, x3, x4, x5, x6, x7)]
        result_dtypes = [x.dtype for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7)]
        self.assertEqual(result_dtypes, input_dtypes)

    def test_rm_mutual_nas_pandas_series_index_preservation(self) -> None:
        """Test that rm_mutual_nas preserves pandas Series indexes."""
        from balance.util import rm_mutual_nas

        x1 = pd.Series([1, 2, 3, 4])
        x2 = pd.Series([np.nan, 2, 3, 4])
        x3 = np.array([1, 2, 3, 4])

        result_0 = rm_mutual_nas(x1, x3)[0]
        assert isinstance(result_0, pd.Series)
        self.assertEqual(result_0.to_dict(), {0: 1, 1: 2, 2: 3, 3: 4})

        sorted_x1 = x1.sort_values(ascending=False)
        result_1 = rm_mutual_nas(sorted_x1, x3)[0]
        assert isinstance(result_1, pd.Series)
        self.assertEqual(result_1.to_dict(), {3: 4, 2: 3, 1: 2, 0: 1})

        result_2 = rm_mutual_nas(x1, x2)[0]
        assert isinstance(result_2, pd.Series)
        self.assertEqual(result_2.to_dict(), {1: 2, 2: 3, 3: 4})

        result_3 = rm_mutual_nas(sorted_x1, x2)[0]
        assert isinstance(result_3, pd.Series)
        self.assertEqual(result_3.to_dict(), {3: 4, 2: 3, 1: 2})

    def test_rm_mutual_nas_error_handling(self) -> None:
        """Test that rm_mutual_nas properly handles error conditions."""
        from balance.util import rm_mutual_nas

        d = np.array((0, 1, 2))

        d3a = np.array((5, 6, 7, 8))
        self.assertRaisesRegex(
            ValueError,
            "All arrays must be of same length",
            rm_mutual_nas,
            d,
            d3a,
        )

        d3b = "a"
        self.assertRaisesRegex(
            ValueError,
            "All arguments must be arraylike",
            rm_mutual_nas,
            d3b,
            d3b,
        )

        self.assertRaises(ValueError, rm_mutual_nas, d, "a")

    def test_choose_variables(self) -> None:
        from balance.util import choose_variables

        self.assertEqual(
            choose_variables(pd.DataFrame({"a": [1], "b": [2]})),
            ["a", "b"],
        )
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

        with self.assertRaises(IndexError):
            balance_util.get_items_from_list_via_indices(["a", "b", "c"], [100])

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
        success_cases = [
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
                result = _verify_value_type(value, expected_type)
                self.assertEqual(result, value)

        error_cases = [
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
                    _verify_value_type(value, expected_type)

    def test__float_or_none(self) -> None:
        """Test _float_or_none with various inputs."""
        test_cases = [
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
