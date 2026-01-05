# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from copy import deepcopy

import balance.testutil

import numpy as np
import numpy.testing
import pandas as pd

# TODO: remove the use of balance_util in most cases, and just import the functions to be tested directly
from balance import util as balance_util
from balance.sample_class import Sample
from balance.util import _coerce_scalar, _verify_value_type

from numpy import dtype

from scipy.sparse import csc_matrix


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test_balance_util_reexports_utils_defs(self) -> None:
        """Ensure balance.util re-exports every helper defined in balance.utils."""
        import importlib
        import inspect
        import typing

        from balance import util as balance_util

        utils_modules = [
            "balance.utils.data_transformation",
            "balance.utils.file_utils",
            "balance.utils.input_validation",
            "balance.utils.logging_utils",
            "balance.utils.model_matrix",
            "balance.utils.pandas_utils",
        ]

        reexported = set(balance_util.__all__)
        missing = []

        for module_name in utils_modules:
            module = importlib.import_module(module_name)
            for name, obj in module.__dict__.items():
                if name == "logger":
                    continue
                if isinstance(obj, typing.TypeVar):
                    continue
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    if obj.__module__ == module_name and name not in reexported:
                        missing.append(f"{module_name}.{name}")
                elif name.isupper() and name not in reexported:
                    missing.append(f"{module_name}.{name}")

        self.assertEqual(
            missing,
            [],
            msg=f"balance.util is missing re-exports: {missing}",
        )

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

    def test__is_categorical_dtype_object(self) -> None:
        """Test _is_categorical_dtype when series dtype is object.

        Verifies that the function returns True when series dtype is an object type (common for string columns).
        """
        # Setup: Create an object dtype series
        series = pd.Series(["a", "b", "c"])

        # Execute: Check if dtype is categorical
        result = balance_util._is_categorical_dtype(series)

        # Assert: Object type should return True
        self.assertTrue(result)

    def test__is_categorical_dtype_numeric(self) -> None:
        """Test _is_categorical_dtype when series dtype is numeric.

        Verifies that the function returns False when series dtype is a numeric type (int or float).
        """
        # Setup: Create numeric dtype series
        int_series = pd.Series([1, 2, 3])
        float_series = pd.Series([4.0, 5.0, 6.0])

        # Execute: Check if dtypes are categorical
        int_result = balance_util._is_categorical_dtype(int_series)
        float_result = balance_util._is_categorical_dtype(float_series)

        # Assert: Numeric types should return False
        self.assertFalse(int_result)
        self.assertFalse(float_result)

    def test__is_categorical_dtype_with_category_dtype(self) -> None:
        """Test _is_categorical_dtype with pandas categorical dtype.

        Verifies that the function correctly identifies pandas categorical dtype as categorical.
        """
        # Setup: Create pandas categorical dtype series
        series = pd.Series(pd.Categorical(["a", "b", "c"]))

        # Execute: Check if dtype is categorical
        result = balance_util._is_categorical_dtype(series)

        # Assert: Categorical dtype should return True
        self.assertTrue(result)

    def test__is_categorical_dtype_with_string_dtype(self) -> None:
        """Test _is_categorical_dtype with pandas string dtype.

        Verifies that the function correctly identifies pandas string dtype (StringDtype) as categorical.
        """
        # Setup: Create pandas string dtype series
        series = pd.Series(["a", "b", "c"], dtype="string")

        # Execute: Check if dtype is categorical
        result = balance_util._is_categorical_dtype(series)

        # Assert: String dtype should return True
        self.assertTrue(result)

    def test__is_categorical_dtype_bool(self) -> None:
        """Test _is_categorical_dtype when dtype is boolean.

        Verifies that the function returns False for boolean dtypes, as they are not considered categorical for this purpose.
        """
        # Setup: Create boolean dtype series
        series = pd.Series([True, False, True])

        # Execute: Check if dtype is categorical
        result = balance_util._is_categorical_dtype(series)

        # Assert: Boolean types are not categorical, should return False
        self.assertFalse(result)

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
        - Use explicitly provided column names
        - Handle cases where no ID column exists
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
            "Cannot guess id column name for this DataFrame. Please provide a value in id_column",
        ):
            balance_util.guess_id_column(df)

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

    def test__isinstance_sample(self) -> None:
        """Test type checking for Sample objects.

        Validates that _isinstance_sample correctly distinguishes between:
        - Regular pandas DataFrames
        - Sample objects created from DataFrames
        """
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

    def test_add_na_indicator(self) -> None:
        """Test addition of NA indicator columns to DataFrames.

        Tests the add_na_indicator function's ability to:
        - Add indicator columns for missing values (None, NaN)
        - Handle different data types (numeric, string, categorical)
        - Replace NA values with specified replacement values
        - Handle edge cases and validation errors
        """
        df = pd.DataFrame({"a": (0, None, 2, np.nan), "b": (None, "b", "", np.nan)})
        e = pd.DataFrame(
            {
                "a": (0, 0, 2.0, 0),
                "b": ("_NA", "b", "", "_NA"),
                "_is_na_a": (False, True, False, True),
                "_is_na_b": (True, False, False, True),
            },
            columns=("a", "b", "_is_na_a", "_is_na_b"),
        )
        r = balance_util.add_na_indicator(df)
        self.assertEqual(r, e)

        # No change if no missing variables
        df = pd.DataFrame(
            {"a": (0, 1, 2), "b": ("a", "b", ""), "c": pd.Categorical(("a", "b", "a"))}
        )
        self.assertEqual(balance_util.add_na_indicator(df), df)

        # Test that it works with categorical variables
        df = pd.DataFrame(
            {
                "c": pd.Categorical(("a", "b", "a", "b")),
                "d": pd.Categorical(("a", "b", None, np.nan)),
            }
        )
        e = pd.DataFrame(
            {
                "c": pd.Categorical(("a", "b", "a", "b")),
                "d": pd.Categorical(
                    ("a", "b", "_NA", "_NA"), categories=("a", "b", "_NA")
                ),
                "_is_na_d": (False, False, True, True),
            },
            columns=("c", "d", "_is_na_d"),
        )
        self.assertEqual(balance_util.add_na_indicator(df), e)

        # test arguments
        df = pd.DataFrame({"a": (0, None, 2, np.nan), "b": (None, "b", "", np.nan)})
        e = pd.DataFrame(
            {
                "a": (0.0, 42.0, 2.0, 42.0),
                "b": ("AAA", "b", "", "AAA"),
                "_is_na_a": (False, True, False, True),
                "_is_na_b": (True, False, False, True),
            },
            columns=("a", "b", "_is_na_a", "_is_na_b"),
        )
        r = balance_util.add_na_indicator(df, replace_val_obj="AAA", replace_val_num=42)
        self.assertEqual(r, e)

        # check exceptions
        d = pd.DataFrame({"a": [0, 1, np.nan, None], "b": ["x", "y", "_NA", None]})
        self.assertRaisesRegex(
            Exception,
            "Can't add NA indicator to columns containing NAs and the value '_NA', ",
            balance_util.add_na_indicator,
            d,
        )
        d = pd.DataFrame({"a": [0, 1, np.nan, None], "_is_na_b": ["x", "y", "z", None]})
        self.assertRaisesRegex(
            Exception,
            "Can't add NA indicator to DataFrame which contains",
            balance_util.add_na_indicator,
            d,
        )

    def test_drop_na_rows(self) -> None:
        """Test removal of rows containing NA values from DataFrames.

        Tests the drop_na_rows function's ability to:
        - Remove rows with NA values from both DataFrame and corresponding weights
        - Maintain proper indexing after row removal
        - Handle edge cases where all rows would be removed
        """
        sample_df = pd.DataFrame(
            {"a": (0, None, 2, np.nan), "b": (None, "b", "c", np.nan)}
        )
        sample_weights = pd.Series([1, 2, 3, 4])
        (
            sample_df,
            sample_weights,
        ) = balance_util.drop_na_rows(sample_df, sample_weights, "sample")
        self.assertEqual(sample_df, pd.DataFrame({"a": (2.0), "b": ("c")}, index=[2]))
        self.assertEqual(sample_weights, pd.Series([3], index=[2]))

        # check exceptions
        sample_df = pd.DataFrame({"a": (None), "b": ("b")}, index=[1])
        sample_weights = pd.Series([1])
        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty",
            balance_util.drop_na_rows,
            sample_df,
            sample_weights,
            "sample",
        )

    def test_formula_generator(self) -> None:
        """Test generation of formula strings from variable specifications.

        Tests the formula_generator function's ability to:
        - Convert single variables to formula strings
        - Convert variable lists to combined formula strings
        - Handle unsupported formula types with appropriate errors
        """
        self.assertEqual(balance_util.formula_generator(["a"]), "a")
        self.assertEqual(balance_util.formula_generator(["a", "b", "c"]), "c + b + a")
        # check exceptions
        self.assertRaisesRegex(
            Exception,
            "This formula type is not supported",
            balance_util.formula_generator,
            ["aa"],
            "interaction",
        )

    def test_dot_expansion(self) -> None:
        self.assertEqual(
            balance_util.dot_expansion(".", ["a", "b", "c", "d"]), "(a+b+c+d)"
        )
        self.assertEqual(
            balance_util.dot_expansion("b:(. - a)", ["a", "b", "c", "d"]),
            "b:((a+b+c+d) - a)",
        )
        self.assertEqual(balance_util.dot_expansion("a*b", ["a", "b", "c", "d"]), "a*b")
        d = {"a": ["a1", "a2", "a1", "a1"]}
        df = pd.DataFrame(data=d)
        self.assertEqual(balance_util.dot_expansion(".", list(df.columns)), "(a)")

        # check exceptions
        self.assertRaisesRegex(
            Exception,
            "Variables should not be empty. Please provide a list of strings.",
            balance_util.dot_expansion,
            ".",
            None,
        )
        self.assertRaisesRegex(
            Exception,
            "Variables should be a list of strings and have to be included.",
            balance_util.dot_expansion,
            ".",
            df,
        )

    def test_process_formula(self) -> None:
        from patsy import EvalFactor, Term  # pyre-ignore[21]

        f1 = balance_util.process_formula("a:(b+aab)", ["a", "b", "aab"])
        self.assertEqual(
            f1.rhs_termlist,
            [
                Term([EvalFactor("a"), EvalFactor("b")]),  # pyre-ignore[16]
                Term([EvalFactor("a"), EvalFactor("aab")]),  # pyre-ignore[16]
            ],
        )

        f2 = balance_util.process_formula("a:(b+aab)", ["a", "b", "aab"], ["a", "b"])
        self.assertEqual(
            f2.rhs_termlist,
            [
                Term(  # pyre-ignore[16]
                    [
                        EvalFactor(  # pyre-ignore[16]
                            "C(a, one_hot_encoding_greater_2)"
                        ),
                        EvalFactor(  # pyre-ignore[16]
                            "C(b, one_hot_encoding_greater_2)"
                        ),
                    ]
                ),
                Term(  # pyre-ignore[16]
                    [
                        EvalFactor(  # pyre-ignore[16]
                            "C(a, one_hot_encoding_greater_2)"
                        ),
                        EvalFactor("aab"),  # pyre-ignore[16]
                    ]
                ),
            ],
        )

        # check exceptions
        self.assertRaisesRegex(
            Exception,
            "Not all factor variables are contained in variables",
            balance_util.process_formula,
            formula="a:(b+aab)",
            variables=["a", "b", "aab"],
            factor_variables="c",
        )

    def test_build_model_matrix(self) -> None:
        df = pd.DataFrame(
            {"a": ["a1", "a2", "a1", "a1"], "b": ["b1", "b2", "b3", "b3"]}
        )
        res = pd.DataFrame(
            {"a[a1]": (1.0, 0.0, 1.0, 1.0), "a[a2]": (0.0, 1.0, 0.0, 0.0)}
        )
        # explicit formula
        x_matrix = balance_util.build_model_matrix(df, "a")
        self.assertEqual(x_matrix["model_matrix"], res)
        self.assertEqual(x_matrix["model_matrix_columns"], res.columns.tolist())

        # formula with dot
        x_matrix = balance_util.build_model_matrix(df, ".")
        res = pd.DataFrame(
            {
                "a[a1]": (1.0, 0.0, 1.0, 1.0),
                "a[a2]": (0.0, 1.0, 0.0, 0.0),
                "b[T.b2]": (0.0, 1.0, 0.0, 0.0),
                "b[T.b3]": (0.0, 0.0, 1.0, 1.0),
            }
        )
        self.assertEqual(x_matrix["model_matrix"], res)
        self.assertEqual(x_matrix["model_matrix_columns"], res.columns.tolist())

        # formula with factor_variables
        x_matrix = balance_util.build_model_matrix(df, ".", factor_variables=["a"])
        res = pd.DataFrame(
            {
                "C(a, one_hot_encoding_greater_2)[a2]": (0.0, 1.0, 0.0, 0.0),
                "b[T.b2]": (0.0, 1.0, 0.0, 0.0),
                "b[T.b3]": (0.0, 0.0, 1.0, 1.0),
            }
        )
        self.assertEqual(x_matrix["model_matrix"], res)
        self.assertEqual(x_matrix["model_matrix_columns"], res.columns.tolist())

        # Sparse output
        x_matrix = balance_util.build_model_matrix(df, "a", return_sparse=True)
        res = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
        self.assertEqual(x_matrix["model_matrix"].toarray(), res)
        self.assertEqual(x_matrix["model_matrix_columns"], ["a[a1]", "a[a2]"])
        self.assertTrue(type(x_matrix["model_matrix"]) is csc_matrix)

        # Check exceptions
        self.assertRaisesRegex(
            Exception,
            "Not all factor variables are contained in df",
            balance_util.build_model_matrix,
            df,
            formula="a",
            factor_variables="c",
        )

        df = pd.DataFrame({"[a]": ["a1", "a2", "a1", "a1"]})
        self.assertRaisesRegex(
            Exception,
            "Variable names cannot contain characters",
            balance_util.build_model_matrix,
            df,
            "a",
        )

        # Int64Dtype input
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        df = df.astype(dtype={"a": "Int64"})
        res = pd.DataFrame({"a": (1.0, 2.0, 3.0, 4.0)})
        # explicit formula
        x_matrix = balance_util.build_model_matrix(df, "a")
        self.assertEqual(x_matrix["model_matrix"], res)
        self.assertEqual(x_matrix["model_matrix_columns"], res.columns.tolist())

    def test_model_matrix(self) -> None:
        s_df = pd.DataFrame(
            {
                "a": (0, 1, 2),
                "b": (0, None, 2),
                "c": ("a", "b", "a"),
                "id": (1, 2, 3),
            }
        )
        s = Sample.from_frame(s_df)

        # Tests on a single sample
        e = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "b": (0.0, 0.0, 2.0),
                "_is_na_b[T.True]": (0.0, 1.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
            }
        )
        r = balance_util.model_matrix(s)
        sample_result_433 = _verify_value_type(r["sample"])
        self.assertEqual(sample_result_433, e, lazy=True)
        self.assertTrue(r["target"] is None)

        # Tests on a single sample dataframe
        e = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "b": (0.0, 0.0, 2.0),
                "_is_na_b[T.True]": (0.0, 1.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
            }
        )
        r = balance_util.model_matrix(s_df[["a", "b", "c"]])
        sample_result_447 = _verify_value_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_result_447.sort_index(axis=1), e, lazy=True)

        # Tests on a single sample with a target
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, 1, 2, None),
                    "d": (0, 2, 2, 1),
                    "c": ("a", "b", "a", "c"),
                    "id": (1, 2, 3, 5),
                }
            )
        )

        r = balance_util.model_matrix(s, t)
        e_s = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "_is_na_a[T.True]": (0.0, 0.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
                "c[c]": (0.0, 0.0, 0.0),
            }
        )
        e_t = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0, 0.0),
                "_is_na_a[T.True]": (0.0, 0.0, 0.0, 1.0),
                "c[a]": (1.0, 0.0, 1.0, 0.0),
                "c[b]": (0.0, 1.0, 0.0, 0.0),
                "c[c]": (0.0, 0.0, 0.0, 1.0),
            }
        )
        sample_result_480 = r["sample"]
        target_result_481 = r["target"]
        sample_result_480 = _verify_value_type(sample_result_480, pd.DataFrame)
        target_result_481 = _verify_value_type(target_result_481, pd.DataFrame)
        self.assertEqual(sample_result_480.sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(target_result_481.sort_index(axis=1), e_t, lazy=True)

        # Test passing DataFrames rather than Samples
        r = balance_util.model_matrix(
            pd.DataFrame({"a": (0, 1, 2), "b": (0, None, 2), "c": ("a", "b", "a")}),
            pd.DataFrame(
                {"a": (0, 1, 2, None), "d": (0, 2, 2, 1), "c": ("a", "b", "a", "c")}
            ),
        )
        sample_result_494 = r["sample"]
        target_result_495 = r["target"]
        sample_result_494 = _verify_value_type(sample_result_494, pd.DataFrame)
        target_result_495 = _verify_value_type(target_result_495, pd.DataFrame)
        self.assertEqual(sample_result_494.sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(target_result_495.sort_index(axis=1), e_t, lazy=True)

        # Check warnings for variables not present in both
        self.assertWarnsRegexp(
            "Ignoring variables not present in all Samples",
            balance_util.model_matrix,
            s,
            t,
        )

        # Test zero rows warning:
        self.assertRaisesRegex(
            AssertionError,
            "sample must have more than zero rows",
            balance_util.model_matrix,
            pd.DataFrame(),
        )

        # Tests on a single DataFrame with bad column names
        s_df_bad_col_names = pd.DataFrame(
            {
                "a * / |a": (0, 1, 2),
                "b  b": (0, None, 2),
                "c._$c": ("a", "b", "a"),
                "id": (1, 2, 3),
            }
        )
        r = balance_util.model_matrix(s_df_bad_col_names)
        exp = ["_is_na_b__b[T.True]", "a______a", "b__b", "c___c[a]", "c___c[b]", "id"]
        self.assertEqual(r["model_matrix_columns_names"], exp)
        exp = {
            "_is_na_b__b[T.True]": {0: 0.0, 1: 1.0, 2: 0.0},
            "a______a": {0: 0.0, 1: 1.0, 2: 2.0},
            "b__b": {0: 0.0, 1: 0.0, 2: 2.0},
            "c___c[a]": {0: 1.0, 1: 0.0, 2: 1.0},
            "c___c[b]": {0: 0.0, 1: 1.0, 2: 0.0},
            "id": {0: 1.0, 1: 2.0, 2: 3.0},
        }
        sample_result_536 = _verify_value_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_result_536.to_dict(), exp)

        # Tests that we can handle multiple columns what would be turned to have the same column name
        s_df_bad_col_names = pd.DataFrame(
            {
                "b1 ": (0, 1, 2),
                "b1*": (0, None, 2000),
                "b1_": (3, 30, 300),
                "b1$": ["a", "b", "c"],
                "id": (1, 2, 3),
            }
        )
        r = balance_util.model_matrix(s_df_bad_col_names)
        exp = [
            "_is_na_b1_[T.True]",
            "b1_",
            "b1__1",
            "b1__2",
            "b1__3[a]",
            "b1__3[b]",
            "b1__3[c]",
            "id",
        ]
        self.assertEqual(r["model_matrix_columns_names"], exp)
        # r["sample"].to_dict()
        exp = {
            "_is_na_b1_[T.True]": {0: 0.0, 1: 1.0, 2: 0.0},
            "b1_": {0: 0.0, 1: 1.0, 2: 2.0},
            "b1__1": {0: 0.0, 1: 0.0, 2: 2000.0},
            "b1__2": {0: 3.0, 1: 30.0, 2: 300.0},
            "b1__3[a]": {0: 1.0, 1: 0.0, 2: 0.0},
            "b1__3[b]": {0: 0.0, 1: 1.0, 2: 0.0},
            "b1__3[c]": {0: 0.0, 1: 0.0, 2: 1.0},
            "id": {0: 1.0, 1: 2.0, 2: 3.0},
        }
        sample_result_571 = _verify_value_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_result_571.to_dict(), exp)

    def test_model_matrix_arguments(self) -> None:
        s_df = pd.DataFrame(
            {
                "a": (0, 1, 2),
                "b": (0, None, 2),
                "c": ("a", "b", "a"),
                "id": (1, 2, 3),
            }
        )
        s = Sample.from_frame(s_df)
        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, 1, 2, None),
                    "d": (0, 2, 2, 1),
                    "c": ("a", "b", "a", "c"),
                    "id": (1, 2, 3, 5),
                }
            )
        )
        # Test variables argument
        r = balance_util.model_matrix(s, variables=["c"])
        e = pd.DataFrame({"c[a]": (1.0, 0.0, 1.0), "c[b]": (0.0, 1.0, 0.0)})
        self.assertEqual(r["sample"], e)
        self.assertTrue(r["target"] is None)

        #  Single covariate which doesn't exist in both should raise error
        self.assertRaisesRegex(
            Exception,
            "requested variables are not in all Samples",
            balance_util.model_matrix,
            s,
            t,
            ["b"],
        )

        # Test add_na argument
        e = pd.DataFrame(
            {"a": (0.0, 2.0), "b": (0.0, 2.0), "c[a]": (1.0, 1.0), "c[b]": (0.0, 0.0)},
            index=(0, 2),
        )
        r = balance_util.model_matrix(s, add_na=False)
        self.assertWarnsRegexp(
            "Dropping all rows with NAs", balance_util.model_matrix, s, add_na=False
        )
        sample_add_na = _verify_value_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_add_na.sort_index(axis=1), e)
        self.assertTrue(r["target"] is None)

        #  Test return_type argument
        r_one = balance_util.model_matrix(s, t, return_type="one")["model_matrix"]
        e_s = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "_is_na_a[T.True]": (0.0, 0.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
                "c[c]": (0.0, 0.0, 0.0),
            }
        )
        e_t = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0, 0.0),
                "_is_na_a[T.True]": (0.0, 0.0, 0.0, 1.0),
                "c[a]": (1.0, 0.0, 1.0, 0.0),
                "c[b]": (0.0, 1.0, 0.0, 0.0),
                "c[c]": (0.0, 0.0, 0.0, 1.0),
            }
        )
        r_one = _verify_value_type(r_one, pd.DataFrame)
        self.assertEqual(r_one.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)

        # Test return_var_type argument
        r_df = balance_util.model_matrix(
            s, t, return_type="one", return_var_type="dataframe"
        )["model_matrix"]
        r_df = _verify_value_type(r_df, pd.DataFrame)
        self.assertEqual(r_df.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)
        r_mat = balance_util.model_matrix(
            s, t, return_type="one", return_var_type="matrix"
        )
        model_matrix_mat = _verify_value_type(r_mat["model_matrix"])
        self.assertEqual(
            model_matrix_mat,
            pd.concat((e_s, e_t))
            .reindex(columns=r_mat["model_matrix_columns_names"])
            .values,
        )
        r_sparse = balance_util.model_matrix(
            s, t, return_type="one", return_var_type="sparse"
        )
        model_matrix_sparse = _verify_value_type(r_sparse["model_matrix"], csc_matrix)
        self.assertEqual(
            model_matrix_sparse.toarray(),
            pd.concat((e_s, e_t))
            .reindex(columns=r_sparse["model_matrix_columns_names"])
            .values,
        )
        self.assertEqual(
            r_sparse["model_matrix_columns_names"],
            ["_is_na_a[T.True]", "a", "c[a]", "c[b]", "c[c]"],
        )
        self.assertTrue(type(model_matrix_sparse) is csc_matrix)

        # Test formula argument
        result_a_plus_b = _verify_value_type(
            balance_util.model_matrix(s, formula="a + b")["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_a_plus_b.sort_index(axis=1),
            pd.DataFrame({"a": (0.0, 1.0, 2.0), "b": (0.0, 0.0, 2.0)}),
        )

        result_b = _verify_value_type(
            balance_util.model_matrix(s, formula="b ")["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_b.sort_index(axis=1),
            pd.DataFrame({"b": (0.0, 0.0, 2.0)}),
        )
        result_a_times_c = _verify_value_type(
            balance_util.model_matrix(s, formula="a * c ")["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_a_times_c.sort_index(axis=1),
            pd.DataFrame(
                {
                    "a": (0.0, 1.0, 2.0),
                    "a:c[T.b]": (0.0, 1.0, 0.0),
                    "c[a]": (1.0, 0.0, 1.0),
                    "c[b]": (0.0, 1.0, 0.0),
                }
            ),
        )
        result_a_b_list = _verify_value_type(
            balance_util.model_matrix(s, formula=["a", "b"])["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_a_b_list.sort_index(axis=1),
            pd.DataFrame({"a": (0.0, 1.0, 2.0), "b": (0.0, 0.0, 2.0)}),
        )

        # Test penalty_factor argument
        self.assertEqual(
            balance_util.model_matrix(s, formula=["a", "b"])["penalty_factor"],
            np.array([1, 1]),
        )
        self.assertEqual(
            balance_util.model_matrix(s, formula=["a", "b"], penalty_factor=[1, 2])[
                "penalty_factor"
            ],
            np.array([1, 2]),
        )
        self.assertEqual(
            balance_util.model_matrix(s, formula="a+b", penalty_factor=[2])[
                "penalty_factor"
            ],
            np.array([2, 2]),
        )
        self.assertRaisesRegex(
            AssertionError,
            "penalty factor and formula must have the same length",
            balance_util.model_matrix,
            s,
            formula="a+b",
            penalty_factor=[1, 2],
        )

        # Test one_hot_encoding argument
        e = pd.DataFrame(
            {
                "C(_is_na_b, one_hot_encoding_greater_2)[True]": (0.0, 1.0, 0.0),
                "C(c, one_hot_encoding_greater_2)[b]": (0.0, 1.0, 0.0),
                "a": (0.0, 1.0, 2.0),
                "b": (0.0, 0.0, 2.0),
            }
        )
        r = balance_util.model_matrix(s, one_hot_encoding=True)
        sample_result_750 = _verify_value_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_result_750.sort_index(axis=1), e, lazy=True)

    def test_qcut(self) -> None:
        d = pd.Series([0, 1, 2, 3, 4])
        self.assertEqual(
            balance_util.qcut(d, 4).astype(str),
            pd.Series(
                [
                    "(-0.001, 1.0]",
                    "(-0.001, 1.0]",
                    "(1.0, 2.0]",
                    "(2.0, 3.0]",
                    "(3.0, 4.0]",
                ]
            ),
        )
        self.assertEqual(balance_util.qcut(d, 6), d)
        self.assertWarnsRegexp(
            "Not quantizing, too few values",
            balance_util.qcut,
            d,
            6,
        )

    def test_quantize(self) -> None:
        d = pd.DataFrame(np.random.rand(1000, 2))
        d = d.rename(columns={i: "ab"[i] for i in range(0, 2)})
        d["c"] = ["x"] * 1000

        r = balance_util.quantize(d, variables=["a"])
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], float))
        self.assertEqual(r["c"][0] == "x")

        r = balance_util.quantize(d)
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], pd.Interval))
        self.assertEqual(r["c"][0] == "x")

        # Test that it does not affect categorical columns
        d["d"] = pd.Categorical(["y"] * 1000)
        r = balance_util.quantize(d)
        self.assertEqual(r["d"][0] == "y")

        # Test on Series input
        r = balance_util.quantize(pd.Series(np.random.uniform(0, 1, 100)), 7)
        self.assertEqual(len(set(r.values)) == 7)

        # Test on numpy array input
        r = balance_util.quantize(np.random.uniform(0, 1, 100), 7)
        self.assertEqual(len(set(r.values)) == 7)

        # Test on single integer input
        r = balance_util.quantize(pd.Series([1]), 1)
        self.assertEqual(len(set(r.values)) == 1)

    def test_quantize_preserves_column_order(self) -> None:
        df = pd.DataFrame(
            {
                "first": np.linspace(0.0, 19.0, 20),
                "second": list("abcdefghijklmnopqrst"),
                "third": np.linspace(100.0, 119.0, 20),
            }
        )

        result = balance_util.quantize(df, q=4, variables=["first", "third"])

        self.assertListEqual(list(result.columns), ["first", "second", "third"])
        self.assertIsInstance(result.loc[0, "first"], pd.Interval)
        self.assertEqual(result.loc[0, "second"], "a")
        self.assertIsInstance(result.loc[0, "third"], pd.Interval)

    def test_quantize_non_numeric_series_raises(self) -> None:
        self.assertRaisesRegex(
            TypeError,
            "series must be numeric",
            balance_util.quantize,
            pd.Series(["x", "y", "z"]),
        )

    def test_row_pairwise_diffs(self) -> None:
        d = pd.DataFrame({"a": (1, 2, 3), "b": (-42, 8, 2)})
        e = pd.DataFrame(
            {"a": (1, 2, 3, 1, 2, 1), "b": (-42, 8, 2, 50, 44, -6)},
            index=(0, 1, 2, "1 - 0", "2 - 0", "2 - 1"),
        )
        self.assertEqual(balance_util.row_pairwise_diffs(d), e)

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

    def test_auto_spread(self) -> None:
        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 1, 2, 2, 4),
            }
        )

        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 2.0, 4.0),
                "key_b_value": (1.0, 2.0, np.nan),
            },
            columns=("id", "key_a_value", "key_b_value"),
        )
        self.assertEqual(expected, balance_util.auto_spread(data))

        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 1, 2, 2, 4),
                "other_value": (2, 2, 4, 4, 6),
            }
        )

        self.assertEqual(
            expected, balance_util.auto_spread(data, features=["key", "value"])
        )

        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 2.0, 4.0),
                "key_b_value": (1.0, 2.0, np.nan),
                "key_a_other_value": (2.0, 4.0, 6.0),
                "key_b_other_value": (2.0, 4.0, np.nan),
            },
            columns=(
                "id",
                "key_a_other_value",
                "key_b_other_value",
                "key_a_value",
                "key_b_value",
            ),
        )
        self.assertEqual(expected, balance_util.auto_spread(data), lazy=True)

        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "a", "c", "d", "a"),
                "value": (1, 1, 2, 4, 1),
            }
        )
        self.assertWarnsRegexp("no unique groupings", balance_util.auto_spread, data)

    def test_auto_spread_multiple_groupings(self) -> None:
        # Multiple possible groupings
        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 3, 2, 4, 1),
            }
        )
        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 4.0, 1.0),
                "key_b_value": (3.0, 2.0, np.nan),
            },
            columns=("id", "key_a_value", "key_b_value"),
        )
        self.assertEqual(expected, balance_util.auto_spread(data))
        self.assertWarnsRegexp("2 possible groupings", balance_util.auto_spread, data)

    def test_auto_aggregate(self) -> None:
        r = balance_util.auto_aggregate(
            pd.DataFrame(
                {"x": [1, 2, 3, 4], "y": [1, 1, 1, np.nan], "id": [1, 1, 2, 3]}
            )
        )
        e = pd.DataFrame({"id": [1, 2, 3], "x": [3, 3, 4], "y": [2, 1, np.nan]})

        self.assertEqual(r, e, lazy=True)

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            pd.DataFrame({"b": ["a", "b", "b"], "id": [1, 1, 2]}),
        )

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            r,
            None,
            "id2",
        )

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            r,
            None,
            aggfunc="not_sum",
        )

    def test_fct_lump_basic_functionality(self) -> None:
        """Test basic functionality of fct_lump for category lumping.

        Tests the fct_lump function's ability to:
        - Preserve categories that meet the threshold
        - Lump categories below the threshold into '_lumped_other'
        - Handle different threshold values
        """
        # Count above the threshold, value preserved
        s = pd.Series(["a"] * 95 + ["b"] * 5)
        self.assertEqual(balance_util.fct_lump(s), s)

        # Move the threshold up
        self.assertEqual(
            balance_util.fct_lump(s, 0.10),
            pd.Series(["a"] * 95 + ["_lumped_other"] * 5),
        )

        # Default threshold, slightly below number of values
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 4)),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

    def test_fct_lump_multiple_categories(self) -> None:
        """Test fct_lump with multiple small categories and edge cases.

        Tests the fct_lump function's ability to:
        - Combine multiple small categories into '_lumped_other'
        - Handle existing '_lumped_other' categories properly
        - Work with categorical data types
        """
        # Multiple categories combined
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 2 + ["c"] * 2)),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

        # Category already called '_lumped_other' is handled
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["_lumped_other"] * 4)),
            pd.Series(["a"] * 96 + ["_lumped_other_lumped_other"] * 4),
        )

        # Categorical series type
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 4, dtype="category")),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

    def _create_wine_test_data(self) -> tuple[Sample, Sample]:
        """Helper method to create synthetic wine dataset for testing.

        Creates synthetic wine data that mimics the structure of the sklearn wine dataset
        but doesn't rely on sklearn's load_wine() function which has compatibility issues
        with newer Python versions.

        Returns:
            tuple: (wine_survey, wine_survey_copy) for categorical and string testing
        """
        # Create synthetic wine data with similar structure to sklearn wine dataset
        np.random.seed(42)  # For reproducible results
        n_samples = 178

        # Create synthetic wine features
        wine_data = {
            "alcohol": np.random.uniform(11.0, 14.8, n_samples),
            "malic_acid": np.random.uniform(0.74, 5.8, n_samples),
            "ash": np.random.uniform(1.36, 3.23, n_samples),
            "alcalinity_of_ash": np.random.uniform(10.6, 30.0, n_samples),
            "magnesium": np.random.uniform(70, 162, n_samples),
            "total_phenols": np.random.uniform(0.98, 3.88, n_samples),
            "flavanoids": np.random.uniform(0.34, 5.08, n_samples),
            "nonflavanoid_phenols": np.random.uniform(0.13, 0.66, n_samples),
            "proanthocyanins": np.random.uniform(0.41, 3.58, n_samples),
            "color_intensity": np.random.uniform(1.28, 13.0, n_samples),
            "hue": np.random.uniform(0.48, 1.71, n_samples),
            "od280_od315_of_diluted_wines": np.random.uniform(1.27, 4.0, n_samples),
            "proline": np.random.uniform(278, 1680, n_samples),
        }

        wine_df = pd.DataFrame(wine_data)
        wine_df["id"] = pd.Series(range(1, len(wine_df) + 1))

        # Create categorical alcohol variable
        wine_df.alcohol = pd.cut(
            wine_df.alcohol, bins=[0, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 100]
        )

        # Create string version for comparison
        wine_df_copy = wine_df.copy(deep=True)
        wine_df_copy.alcohol = wine_df_copy.alcohol.astype("object")

        # Create synthetic target classes (0, 1, 2)
        wine_class = pd.Series(
            np.random.choice([0, 1, 2], size=n_samples, p=[0.33, 0.4, 0.27])
        )

        # Split datasets
        wine_survey = Sample.from_frame(wine_df.loc[wine_class == 0, :])
        wine_pop = Sample.from_frame(wine_df.loc[wine_class != 0, :])
        wine_survey = wine_survey.set_target(wine_pop)

        wine_survey_copy = Sample.from_frame(wine_df_copy.loc[wine_class == 0, :])
        wine_pop_copy = Sample.from_frame(wine_df_copy.loc[wine_class != 0, :])
        wine_survey_copy = wine_survey_copy.set_target(wine_pop_copy)

        return wine_survey, wine_survey_copy

    def test_fct_lump_categorical_vs_string_consistency(self) -> None:
        """Test that fct_lump produces consistent results for categorical vs string variables.

        Tests that fct_lump works identically when applied to:
        - Categorical variables
        - String variables with the same content

        This ensures consistency in model coefficient generation.
        """
        wine_survey, wine_survey_copy = self._create_wine_test_data()

        transformations = {
            "alcohol": lambda x: balance_util.fct_lump(x, prop=0.05),
            "flavanoids": balance_util.quantize,
            "total_phenols": balance_util.quantize,
            "nonflavanoid_phenols": balance_util.quantize,
            "color_intensity": balance_util.quantize,
            "hue": balance_util.quantize,
            "ash": balance_util.quantize,
            "alcalinity_of_ash": balance_util.quantize,
            "malic_acid": balance_util.quantize,
            "magnesium": balance_util.quantize,
        }

        # Generate weights for both categorical and string versions
        output_cat_var = wine_survey.adjust(
            transformations=transformations, method="ipw", max_de=2.5
        )
        output_string_var = wine_survey_copy.adjust(
            transformations=transformations, method="ipw", max_de=2.5
        )

        # Check that model coefficients are identical
        output_cat_var_model = output_cat_var.model()
        output_string_var_model = output_string_var.model()
        output_cat_var_model = _verify_value_type(output_cat_var_model)
        output_string_var_model = _verify_value_type(output_string_var_model)
        self.assertEqual(
            output_cat_var_model["perf"]["coefs"],
            output_string_var_model["perf"]["coefs"],
        )

    def test_fct_lump_by(self) -> None:
        """Test category lumping with grouping by another variable.

        Tests the fct_lump_by function's ability to:
        - Lump categories within groups defined by another variable
        - Handle cases where grouping variable has uniform values
        - Preserve DataFrame indices when combining data
        """
        # test by argument works
        s = pd.Series([1, 1, 1, 2, 3, 1, 2])
        by = pd.Series(["a", "a", "a", "a", "a", "b", "b"])
        self.assertEqual(
            balance_util.fct_lump_by(s, by, 0.5),
            pd.Series([1, 1, 1, "_lumped_other", "_lumped_other", 1, 2]),
        )

        # test case where all values in 'by' are the same
        s = pd.Series([1, 1, 1, 2, 3, 1, 2])
        by = pd.Series(["a", "a", "a", "a", "a", "a", "a"])
        self.assertEqual(
            balance_util.fct_lump_by(s, by, 0.5),
            pd.Series(
                [1, 1, 1, "_lumped_other", "_lumped_other", 1, "_lumped_other"],
            ),
        )

        # test fct_lump_by doesn't affect indices when combining dataframes
        s = pd.DataFrame({"d": [1, 1, 1], "e": ["a1", "a2", "a1"]}, index=(0, 6, 7))
        t = pd.DataFrame(
            {"d": [2, 3, 1, 2], "e": ["a2", "a2", "a1", "a2"]}, index=(0, 1, 2, 3)
        )
        df = pd.concat([s, t])
        r = balance_util.fct_lump_by(df.d, df.e, 0.5)
        e = pd.Series(
            [1, "_lumped_other", 1, 2, "_lumped_other", 1, 2],
            index=(0, 6, 7, 0, 1, 2, 3),
            name="d",
        )
        self.assertEqual(r, e)

    def test_one_hot_encoding_greater_2(self) -> None:
        """Test one-hot encoding for categorical variables with >2 categories.

        Tests the one_hot_encoding_greater_2 function's ability to:
        - Apply one-hot encoding only to variables with more than 2 categories
        - Handle variables with exactly 2 categories differently
        - Work correctly with patsy's dmatrix function
        """
        from balance.util import one_hot_encoding_greater_2  # noqa
        from patsy import (  # pyre-ignore[21]: Import `patsy.dmatrix` is not defined as a type.
            dmatrix,
        )

        d = {
            "a": ["a1", "a2", "a1", "a1"],
            "b": ["b1", "b2", "b3", "b3"],
            "c": ["c1", "c1", "c1", "c1"],
        }
        df = pd.DataFrame(data=d)

        res = dmatrix(  # pyre-ignore[16]: Module `patsy` has no attribute `dmatrix`.
            "C(a, one_hot_encoding_greater_2)", df, return_type="dataframe"
        )
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(a, one_hot_encoding_greater_2)[a2]": [0.0, 1.0, 0.0, 0.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

        res = dmatrix(  # pyre-ignore[16]: Module `patsy` has no attribute `dmatrix`.
            "C(b, one_hot_encoding_greater_2)", df, return_type="dataframe"
        )
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(b, one_hot_encoding_greater_2)[b1]": [1.0, 0.0, 0.0, 0.0],
            "C(b, one_hot_encoding_greater_2)[b2]": [0.0, 1.0, 0.0, 0.0],
            "C(b, one_hot_encoding_greater_2)[b3]": [0.0, 0.0, 1.0, 1.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

        res = dmatrix(  # pyre-ignore[16]: Module `patsy` has no attribute `dmatrix`.
            "C(c, one_hot_encoding_greater_2)", df, return_type="dataframe"
        )
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(c, one_hot_encoding_greater_2)[c1]": [1.0, 1.0, 1.0, 1.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

    def test_truncate_text(self) -> None:
        self.assertEqual(
            balance_util._truncate_text("a" * 6, length=5), "a" * 5 + "..."
        )
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
            {"id": dtype("O"), "a": dtype("float64"), "weight": dtype("float64")},
        )
        self.assertEqual(
            df_orig.dtypes.to_dict(),
            {"id": dtype("int64"), "a": dtype("int64"), "forest": dtype("O")},
        )

        df_fixed = balance_util._astype_in_df_from_dtypes(df, df_orig.dtypes)
        self.assertEqual(
            df_fixed.dtypes.to_dict(),
            {"id": dtype("int64"), "a": dtype("int64"), "weight": dtype("float64")},
        )

    def test__true_false_str_to_bool(self) -> None:
        self.assertFalse(balance_util._true_false_str_to_bool("falsE"))
        self.assertTrue(balance_util._true_false_str_to_bool("TrUe"))
        with self.assertRaisesRegex(
            ValueError,
            "Banana is not an accepted value, please pass either 'True' or 'False'*",
        ):
            balance_util._true_false_str_to_bool("Banana")

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
        # Sample DataFrame with duplicate column names
        data = {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "A2": [7, 8, 9],
            "C": [10, 11, 12],
        }

        df1 = pd.DataFrame(data)
        df1.columns = ["A", "B", "A", "A"]

        # TODO: understand in the future why the names here appear to be consistent while when using the function in
        # `model_matrix` it does not appear to work.
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
        # Test with Series containing infinities
        series_with_inf = pd.Series([1.0, np.inf, 2.0, -np.inf, 3.0])
        result = balance_util._safe_replace_and_infer(series_with_inf)
        expected = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        pd.testing.assert_series_equal(result, expected)

        # Test with DataFrame
        df_with_inf = pd.DataFrame({"a": [1.0, np.inf, 2.0], "b": [-np.inf, 3.0, 4.0]})
        result = balance_util._safe_replace_and_infer(df_with_inf)
        expected = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan, 3.0, 4.0]})
        pd.testing.assert_frame_equal(result, expected)

        # Test with custom replace values
        series_test = pd.Series([1, 2, 3, 4])
        result = balance_util._safe_replace_and_infer(
            series_test, to_replace=2, value=99
        )
        expected = pd.Series([1, 99, 3, 4])
        pd.testing.assert_series_equal(result, expected)

        # Test with object dtype
        series_obj = pd.Series(["a", "b", "c"], dtype="object")
        result = balance_util._safe_replace_and_infer(
            series_obj, to_replace="b", value="x"
        )
        expected = pd.Series(["a", "x", "c"], dtype="object")
        pd.testing.assert_series_equal(result, expected)

    def test__safe_fillna_and_infer(self) -> None:
        """Test safe NA filling and dtype inference to avoid pandas deprecation warnings."""
        # Test with Series containing NaN values
        series_with_nan = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = balance_util._safe_fillna_and_infer(series_with_nan, value=0)
        expected = pd.Series([1.0, 0.0, 2.0, 0.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

        # Test with DataFrame
        df_with_nan = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan, 3.0, 4.0]})
        result = balance_util._safe_fillna_and_infer(df_with_nan, value=-1)
        expected = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [-1.0, 3.0, 4.0]})
        pd.testing.assert_frame_equal(result, expected)

        # Test with string replacement
        series_str = pd.Series(["a", None, "c"])
        result = balance_util._safe_fillna_and_infer(series_str, value="_NA")
        expected = pd.Series(["a", "_NA", "c"])
        pd.testing.assert_series_equal(result, expected)

        # Test with no value provided (default None -> nan)
        series_test = pd.Series([1, None, 3])
        result = balance_util._safe_fillna_and_infer(series_test)
        expected = pd.Series([1.0, np.nan, 3.0])  # Type gets converted to float
        pd.testing.assert_series_equal(result, expected)

    def test__safe_groupby_apply(self) -> None:
        """Test safe groupby apply operations that handle include_groups parameter."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C"],
                "value": [1, 2, 3, 4, 5],
                "other": [10, 20, 30, 40, 50],
            }
        )

        # Test with simple aggregation function
        result = balance_util._safe_groupby_apply(
            df, "group", lambda x: x["value"].sum()
        )
        expected = pd.Series([3, 7, 5], index=pd.Index(["A", "B", "C"], name="group"))
        pd.testing.assert_series_equal(result, expected)

        # Test with multiple grouping columns
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

        # Test with function that accesses the grouping column
        result = balance_util._safe_groupby_apply(df, "group", lambda x: len(x))
        expected = pd.Series([2, 2, 1], index=pd.Index(["A", "B", "C"], name="group"))
        pd.testing.assert_series_equal(result, expected)

    def test__safe_show_legend(self) -> None:
        """Test safe legend display that only shows legends when there are labeled artists."""
        import matplotlib.pyplot as plt

        # Test with labeled artists
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="line1")
        ax.plot([1, 2, 3], [3, 2, 1], label="line2")

        # This should not raise a warning
        balance_util._safe_show_legend(ax)

        # Verify legend was created
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 2)

        plt.close(fig)

        # Test with no labeled artists
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])  # No label
        ax.plot([1, 2, 3], [3, 2, 1])  # No label

        # This should not create a legend or raise warnings
        balance_util._safe_show_legend(ax)

        # Verify no legend was created
        legend = ax.get_legend()
        self.assertIsNone(legend)

        plt.close(fig)

        # Test with mixed labeled and unlabeled artists
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="labeled")
        ax.plot([1, 2, 3], [3, 2, 1])  # No label

        balance_util._safe_show_legend(ax)

        # Verify legend was created with only labeled items
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 1)
        self.assertEqual(legend.get_texts()[0].get_text(), "labeled")

        plt.close(fig)

    def test__safe_divide_with_zero_handling(self) -> None:
        """Test safe division with proper numpy error state management."""
        # Test normal division
        result = balance_util._safe_divide_with_zero_handling(10, 2)
        self.assertEqual(result, 5.0)

        # Test with numpy arrays - the main use case for this function
        numerator = np.array([1, 2, 3, 4])
        denominator = np.array([1, 0, 3, 2])
        result = balance_util._safe_divide_with_zero_handling(numerator, denominator)
        expected = np.array([1.0, np.inf, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

        # Test with pandas Series
        num_series = pd.Series([10, 20, 30])
        den_series = pd.Series([2, 0, 5])
        result = balance_util._safe_divide_with_zero_handling(num_series, den_series)
        expected = pd.Series([5.0, np.inf, 6.0])
        pd.testing.assert_series_equal(result, expected)


class TestSample_high_cardinality_warnings(balance.testutil.BalanceTestCase):
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

        # The main assertion: verify the high-cardinality warning appears
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

    def test__process_series_for_missing_mask(self) -> None:
        """Test _process_series_for_missing_mask with various input scenarios."""
        test_cases = [
            # (input_series, expected_mask, description)
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
        # Test 1: Convert pandas Int64 to float64, keeping other types unchanged
        df1 = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3], dtype=pd.Int64Dtype()),
                "b": pd.array([4.0, 5.0, 6.0], dtype=np.float64),
            }
        )
        result1 = balance_util._pd_convert_all_types(df1, "Int64", "float64")
        self.assertEqual(result1["a"].dtype, np.float64)  # Int64 -> float64
        self.assertEqual(result1["b"].dtype, np.float64)  # float64 unchanged

        # Test 2: Multiple Int64 columns and column order preservation
        df2 = pd.DataFrame(
            {
                "z": pd.array([1, 2], dtype=pd.Int64Dtype()),
                "a": pd.array([3, 4], dtype=np.float64),
                "m": pd.array([5, 6], dtype=pd.Int64Dtype()),
            }
        )
        result2 = balance_util._pd_convert_all_types(df2, "Int64", "float64")
        # Check conversions
        self.assertEqual(result2["z"].dtype, np.float64)
        self.assertEqual(result2["a"].dtype, np.float64)  # Already float64
        self.assertEqual(result2["m"].dtype, np.float64)
        # Check column order preserved
        self.assertEqual(list(result2.columns), ["z", "a", "m"])

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

    def test__truncate_text(self) -> None:
        """Test _truncate_text with various string lengths."""
        test_cases = [
            # (input_text, length, expected_result, description)
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
                    self.assertEqual(len(result), length + 3)  # length + '...'

    def test_TruncationFormatter(self) -> None:
        """Test TruncationFormatter with long and short log messages."""
        formatter = balance_util.TruncationFormatter("%(message)s")
        MAX_MESSAGE_LENGTH = 2000  # TruncationFormatter truncates at 2000 characters
        ELLIPSIS_LENGTH = 3

        test_cases = [
            # (message, should_truncate, description)
            (
                "x" * (MAX_MESSAGE_LENGTH + 1000),
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
                    self.assertEqual(len(result), MAX_MESSAGE_LENGTH + ELLIPSIS_LENGTH)
                    self.assertTrue(result.endswith("..."))
                else:
                    self.assertEqual(result, message)
