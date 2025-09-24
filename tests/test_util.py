# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from copy import deepcopy

import balance.testutil

import numpy as np
import numpy.testing
import pandas as pd

# TODO: remove the use of balance_util in most cases, and just import the functions to be tested directly
from balance import util as balance_util
from balance.sample_class import Sample

from numpy import dtype

from patsy import dmatrix  # pyre-ignore[21]: this module exists
from scipy.sparse import csc_matrix


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test__check_weighting_methods_input(self):
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
            df=pd.DataFrame({"a": [1]}),
            weights="a",
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

    def test_guess_id_column(self):
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

    def test__isinstance_sample(self):
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

    def test_add_na_indicator(self):
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

    def test_drop_na_rows(self):
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

    def test_formula_generator(self):
        """Test generation of formula strings from variable specifications.

        Tests the formula_generator function's ability to:
        - Convert single variables to formula strings
        - Convert variable lists to combined formula strings
        - Handle unsupported formula types with appropriate errors
        """
        self.assertEqual(balance_util.formula_generator("a"), "a")
        self.assertEqual(balance_util.formula_generator(["a", "b", "c"]), "c + b + a")
        # check exceptions
        self.assertRaisesRegex(
            Exception,
            "This formula type is not supported",
            balance_util.formula_generator,
            ["a", "b"],
            "interaction",
        )

    def test_dot_expansion(self):
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

    def test_process_formula(self):
        from patsy import EvalFactor, Term  # pyre-ignore[21]: this module exists

        f1 = balance_util.process_formula("a:(b+aab)", ["a", "b", "aab"])
        self.assertEqual(
            f1.rhs_termlist,
            [
                Term([EvalFactor("a"), EvalFactor("b")]),
                Term([EvalFactor("a"), EvalFactor("aab")]),
            ],
        )

        f2 = balance_util.process_formula("a:(b+aab)", ["a", "b", "aab"], ["a", "b"])
        self.assertEqual(
            f2.rhs_termlist,
            [
                Term(
                    [
                        EvalFactor("C(a, one_hot_encoding_greater_2)"),
                        EvalFactor("C(b, one_hot_encoding_greater_2)"),
                    ]
                ),
                Term(
                    [EvalFactor("C(a, one_hot_encoding_greater_2)"), EvalFactor("aab")]
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

    def test_build_model_matrix(self):
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

    def test_model_matrix(self):
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
        self.assertEqual(r["sample"], e, lazy=True)
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
        self.assertEqual(r["sample"].sort_index(axis=1), e, lazy=True)

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
        self.assertEqual(r["sample"].sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(r["target"].sort_index(axis=1), e_t, lazy=True)

        # Test passing DataFrames rather than Samples
        r = balance_util.model_matrix(
            pd.DataFrame({"a": (0, 1, 2), "b": (0, None, 2), "c": ("a", "b", "a")}),
            pd.DataFrame(
                {"a": (0, 1, 2, None), "d": (0, 2, 2, 1), "c": ("a", "b", "a", "c")}
            ),
        )
        self.assertEqual(r["sample"].sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(r["target"].sort_index(axis=1), e_t, lazy=True)

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
        self.assertEqual(r["sample"].to_dict(), exp)

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
        self.assertEqual(r["sample"].to_dict(), exp)

    def test_model_matrix_arguments(self):
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
        r = balance_util.model_matrix(s, variables="c")
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
            "b",
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
        self.assertEqual(r["sample"].sort_index(axis=1), e)
        self.assertTrue(r["target"] is None)

        #  Test return_type argument
        r = balance_util.model_matrix(s, t, return_type="one")["model_matrix"]
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
        self.assertEqual(r.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)

        # Test return_var_type argument
        r = balance_util.model_matrix(
            s, t, return_type="one", return_var_type="dataframe"
        )["model_matrix"]
        self.assertEqual(r.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)
        r = balance_util.model_matrix(s, t, return_type="one", return_var_type="matrix")
        self.assertEqual(
            r["model_matrix"],
            pd.concat((e_s, e_t))
            .reindex(columns=r["model_matrix_columns_names"])
            .values,
        )
        r = balance_util.model_matrix(s, t, return_type="one", return_var_type="sparse")
        self.assertEqual(
            r["model_matrix"].toarray(),
            pd.concat((e_s, e_t))
            .reindex(columns=r["model_matrix_columns_names"])
            .values,
        )
        self.assertEqual(
            r["model_matrix_columns_names"],
            ["_is_na_a[T.True]", "a", "c[a]", "c[b]", "c[c]"],
        )
        self.assertTrue(type(r["model_matrix"]) is csc_matrix)

        # Test formula argument
        self.assertEqual(
            balance_util.model_matrix(s, formula="a + b")["sample"].sort_index(axis=1),
            pd.DataFrame({"a": (0.0, 1.0, 2.0), "b": (0.0, 0.0, 2.0)}),
        )

        self.assertEqual(
            balance_util.model_matrix(s, formula="b ")["sample"].sort_index(axis=1),
            pd.DataFrame({"b": (0.0, 0.0, 2.0)}),
        )
        self.assertEqual(
            balance_util.model_matrix(s, formula="a * c ")["sample"].sort_index(axis=1),
            pd.DataFrame(
                {
                    "a": (0.0, 1.0, 2.0),
                    "a:c[T.b]": (0.0, 1.0, 0.0),
                    "c[a]": (1.0, 0.0, 1.0),
                    "c[b]": (0.0, 1.0, 0.0),
                }
            ),
        )
        self.assertEqual(
            balance_util.model_matrix(s, formula=["a", "b"])["sample"].sort_index(
                axis=1
            ),
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
        self.assertEqual(r["sample"].sort_index(axis=1), e, lazy=True)

    def test_qcut(self):
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

    def test_quantize(self):
        d = pd.DataFrame(np.random.rand(1000, 2))
        d = d.rename(columns={i: "ab"[i] for i in range(0, 2)})
        d["c"] = ["x"] * 1000

        r = balance_util.quantize(d, variables=("a"))
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], float))
        self.assertTrue(r["c"][0] == "x")

        r = balance_util.quantize(d)
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], pd.Interval))
        self.assertTrue(r["c"][0] == "x")

        # Test that it does not affect categorical columns
        d["d"] = pd.Categorical(["y"] * 1000)
        r = balance_util.quantize(d)
        self.assertTrue(r["d"][0] == "y")

        # Test on Series input
        r = balance_util.quantize(pd.Series(np.random.uniform(0, 1, 100)), 7)
        self.assertTrue(len(set(r.values)) == 7)

        # Test on numpy array input
        r = balance_util.quantize(np.random.uniform(0, 1, 100), 7)
        self.assertTrue(len(set(r.values)) == 7)

        # Test on single integer input
        r = balance_util.quantize(1, 1)
        self.assertTrue(len(set(r.values)) == 1)

    def test_row_pairwise_diffs(self):
        d = pd.DataFrame({"a": (1, 2, 3), "b": (-42, 8, 2)})
        e = pd.DataFrame(
            {"a": (1, 2, 3, 1, 2, 1), "b": (-42, 8, 2, 50, 44, -6)},
            index=(0, 1, 2, "1 - 0", "2 - 0", "2 - 1"),
        )
        self.assertEqual(balance_util.row_pairwise_diffs(d), e)

    def test_isarraylike(self):
        self.assertFalse(balance_util._is_arraylike(""))
        self.assertFalse(balance_util._is_arraylike("test"))
        self.assertTrue(balance_util._is_arraylike(()))
        self.assertTrue(balance_util._is_arraylike([]))
        self.assertTrue(balance_util._is_arraylike([1, 2]))
        self.assertTrue(balance_util._is_arraylike(range(10)))
        self.assertTrue(balance_util._is_arraylike(np.array([1, 2, "a"])))
        self.assertTrue(balance_util._is_arraylike(pd.Series((1, 2, 3))))

    def test_rm_mutual_nas_basic_functionality(self):
        """Test basic functionality of rm_mutual_nas with simple arrays."""
        from balance.util import rm_mutual_nas

        # Test with lists containing None values
        result = rm_mutual_nas([1, 2, 3], [2, 3, None])
        self.assertEqual(result, [[1, 2], [2.0, 3.0]])

    def test_rm_mutual_nas_single_arrays(self):
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

    def test_rm_mutual_nas_with_na_values(self):
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

    def test_rm_mutual_nas_single_array_with_na(self):
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

    def _create_test_arrays(self):
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

    def test_rm_mutual_nas_pandas_arrays(self):
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

    def test_rm_mutual_nas_type_preservation(self):
        """Test that rm_mutual_nas preserves input types."""
        from balance.util import rm_mutual_nas

        # Get test arrays from helper method
        x1, x2, x3, x4, x5, x6, x7, x8 = self._create_test_arrays()

        # Test that types are preserved after NA removal
        input_types = [type(x) for x in (x1, x2, x3, x4, x5, x6, x7, x8)]
        result_types = [type(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)]
        self.assertEqual(result_types, input_types)

        # Test specific type preservation
        # Handle pandas array type compatibility - PandasArray was renamed to NumpyExtensionArray
        if hasattr(pd.core.arrays.numpy_, "NumpyExtensionArray"):
            numpy_array_type = pd.core.arrays.numpy_.NumpyExtensionArray
        else:
            numpy_array_type = pd.core.arrays.numpy_.PandasArray
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

    def test_rm_mutual_nas_dtype_preservation(self):
        """Test that rm_mutual_nas preserves dtypes for numpy and pandas arrays."""
        from balance.util import rm_mutual_nas

        # Get test arrays from helper method (only need first 7 for this test)
        x1, x2, x3, x4, x5, x6, x7, _ = self._create_test_arrays()

        # Test that dtypes are preserved
        input_dtypes = [x.dtype for x in (x1, x2, x3, x4, x5, x6, x7)]
        result_dtypes = [x.dtype for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7)]
        self.assertEqual(result_dtypes, input_dtypes)

    def test_rm_mutual_nas_pandas_series_index_preservation(self):
        """Test that rm_mutual_nas preserves pandas Series indexes."""
        from balance.util import rm_mutual_nas

        x1 = pd.Series([1, 2, 3, 4])
        x2 = pd.Series([np.nan, 2, 3, 4])
        x3 = np.array([1, 2, 3, 4])

        # Test that index is preserved when no values are removed
        result = rm_mutual_nas(x1, x3)[0]
        self.assertEqual(result.to_dict(), {0: 1, 1: 2, 2: 3, 3: 4})

        # Test with sorted series
        sorted_x1 = x1.sort_values(ascending=False)
        result = rm_mutual_nas(sorted_x1, x3)[0]
        self.assertEqual(result.to_dict(), {3: 4, 2: 3, 1: 2, 0: 1})

        # Test that index is preserved when NA values are removed
        result = rm_mutual_nas(x1, x2)[0]
        self.assertEqual(result.to_dict(), {1: 2, 2: 3, 3: 4})

        # Test with sorted series and NA removal
        result = rm_mutual_nas(sorted_x1, x2)[0]
        self.assertEqual(result.to_dict(), {3: 4, 2: 3, 1: 2})

    def test_rm_mutual_nas_error_handling(self):
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

    def test_choose_variables(self):
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
                variables="",
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
                variables="a",
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
            choose_variables(df1, df2, variables=("a", "b", "c"))

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

    def test_auto_spread(self):
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

    def test_auto_spread_multiple_groupings(self):
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

    def test_auto_aggregate(self):
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

    def test_fct_lump_basic_functionality(self):
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

    def test_fct_lump_multiple_categories(self):
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

    def _create_wine_test_data(self):
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

    def test_fct_lump_categorical_vs_string_consistency(self):
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
        self.assertEqual(
            output_cat_var.model()["perf"]["coefs"],
            output_string_var.model()["perf"]["coefs"],
        )

    def test_fct_lump_by(self):
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

    def test_one_hot_encoding_greater_2(self):
        """Test one-hot encoding for categorical variables with >2 categories.

        Tests the one_hot_encoding_greater_2 function's ability to:
        - Apply one-hot encoding only to variables with more than 2 categories
        - Handle variables with exactly 2 categories differently
        - Work correctly with patsy's dmatrix function
        """
        from balance.util import one_hot_encoding_greater_2  # noqa

        d = {
            "a": ["a1", "a2", "a1", "a1"],
            "b": ["b1", "b2", "b3", "b3"],
            "c": ["c1", "c1", "c1", "c1"],
        }
        df = pd.DataFrame(data=d)

        res = dmatrix("C(a, one_hot_encoding_greater_2)", df, return_type="dataframe")
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(a, one_hot_encoding_greater_2)[a2]": [0.0, 1.0, 0.0, 0.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

        res = dmatrix("C(b, one_hot_encoding_greater_2)", df, return_type="dataframe")
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(b, one_hot_encoding_greater_2)[b1]": [1.0, 0.0, 0.0, 0.0],
            "C(b, one_hot_encoding_greater_2)[b2]": [0.0, 1.0, 0.0, 0.0],
            "C(b, one_hot_encoding_greater_2)[b3]": [0.0, 0.0, 1.0, 1.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

        res = dmatrix("C(c, one_hot_encoding_greater_2)", df, return_type="dataframe")
        expected = {
            "Intercept": [1.0, 1.0, 1.0, 1.0],
            "C(c, one_hot_encoding_greater_2)[c1]": [1.0, 1.0, 1.0, 1.0],
        }
        expected = pd.DataFrame(data=expected)
        self.assertEqual(res, expected)

    def test_truncate_text(self):
        self.assertEqual(
            balance_util._truncate_text("a" * 6, length=5), "a" * 5 + "..."
        )
        self.assertEqual(balance_util._truncate_text("a" * 4, length=5), "a" * 4)
        self.assertEqual(balance_util._truncate_text("a" * 5, length=5), "a" * 5)

    def test__dict_intersect(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"c": 3, "b": 2222}
        self.assertEqual(balance_util._dict_intersect(d1, d2), {"b": 2})

    def test__astype_in_df_from_dtypes(self):
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

    def test__true_false_str_to_bool(self):
        self.assertFalse(balance_util._true_false_str_to_bool("falsE"))
        self.assertTrue(balance_util._true_false_str_to_bool("TrUe"))
        with self.assertRaisesRegex(
            ValueError,
            "Banana is not an accepted value, please pass either 'True' or 'False'*",
        ):
            balance_util._true_false_str_to_bool("Banana")

    def test__are_dtypes_equal(self):
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

    def test__warn_of_df_dtypes_change(self):
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

    def test__make_df_column_names_unique(self):
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

    def test__safe_replace_and_infer(self):
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

    def test__safe_fillna_and_infer(self):
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

    def test__safe_groupby_apply(self):
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

    def test__safe_show_legend(self):
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

    def test__safe_divide_with_zero_handling(self):
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
