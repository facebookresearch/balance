# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.util import _assert_type
from balance.utils.model_matrix import (
    build_model_matrix,
    dot_expansion,
    formula_generator,
    model_matrix,
    process_formula,
)
from scipy.sparse import csc_matrix


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test_formula_generator(self) -> None:
        """Test generation of formula strings from variable specifications.

        Tests the formula_generator function's ability to:
        - Convert single variables to formula strings
        - Convert variable lists to combined formula strings
        - Handle unsupported formula types with appropriate errors
        """
        self.assertEqual(formula_generator(["a"]), "a")
        self.assertEqual(formula_generator(["a", "b", "c"]), "c + b + a")
        # check exceptions
        self.assertRaisesRegex(
            ValueError,
            "This formula type is not supported",
            formula_generator,
            ["aa"],
            "interaction",
        )

    def test_dot_expansion(self) -> None:
        self.assertEqual(dot_expansion(".", ["a", "b", "c", "d"]), "(a+b+c+d)")
        self.assertEqual(
            dot_expansion("b:(. - a)", ["a", "b", "c", "d"]),
            "b:((a+b+c+d) - a)",
        )
        self.assertEqual(dot_expansion("a*b", ["a", "b", "c", "d"]), "a*b")
        d = {"a": ["a1", "a2", "a1", "a1"]}
        df = pd.DataFrame(data=d)
        self.assertEqual(dot_expansion(".", list(df.columns)), "(a)")

        # check exceptions
        self.assertRaisesRegex(
            TypeError,
            "Variables should not be empty. Please provide a list of strings.",
            dot_expansion,
            ".",
            None,
        )
        self.assertRaisesRegex(
            TypeError,
            "Variables should be a list of strings and have to be included.",
            dot_expansion,
            ".",
            df,
        )

    def test_process_formula(self) -> None:
        from patsy import EvalFactor, Term  # pyre-ignore[21]

        f1 = process_formula("a:(b+aab)", ["a", "b", "aab"])
        self.assertEqual(
            f1.rhs_termlist,
            [
                Term([EvalFactor("a"), EvalFactor("b")]),  # pyre-ignore[16]
                Term([EvalFactor("a"), EvalFactor("aab")]),  # pyre-ignore[16]
            ],
        )

        f2 = process_formula("a:(b+aab)", ["a", "b", "aab"], ["a", "b"])
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
            ValueError,
            "Not all factor variables are contained in variables",
            process_formula,
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
        x_matrix = build_model_matrix(df, "a")
        self.assertEqual(x_matrix["model_matrix"], res)
        self.assertEqual(x_matrix["model_matrix_columns"], res.columns.tolist())

        # formula with dot
        x_matrix = build_model_matrix(df, ".")
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
        x_matrix = build_model_matrix(df, ".", factor_variables=["a"])
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
        x_matrix = build_model_matrix(df, "a", return_sparse=True)
        res = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
        self.assertEqual(x_matrix["model_matrix"].toarray(), res)
        self.assertEqual(x_matrix["model_matrix_columns"], ["a[a1]", "a[a2]"])
        self.assertIs(type(x_matrix["model_matrix"]), csc_matrix)

        # Check exceptions
        self.assertRaisesRegex(
            ValueError,
            "Not all factor variables are contained in df",
            build_model_matrix,
            df,
            formula="a",
            factor_variables="c",
        )

        df = pd.DataFrame({"[a]": ["a1", "a2", "a1", "a1"]})
        self.assertRaisesRegex(
            ValueError,
            "Variable names cannot contain characters",
            build_model_matrix,
            df,
            "a",
        )

        # Int64Dtype input
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        df = df.astype(dtype={"a": "Int64"})
        res = pd.DataFrame({"a": (1.0, 2.0, 3.0, 4.0)})
        # explicit formula
        x_matrix = build_model_matrix(df, "a")
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
        r = model_matrix(s)
        sample_result_433 = _assert_type(r["sample"])
        self.assertEqual(sample_result_433, e, lazy=True)
        self.assertIsNone(r["target"])

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
        r = model_matrix(s_df[["a", "b", "c"]])
        sample_result_447 = _assert_type(r["sample"], pd.DataFrame)
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

        r = model_matrix(s, t)
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
        sample_result_480 = _assert_type(sample_result_480, pd.DataFrame)
        target_result_481 = _assert_type(target_result_481, pd.DataFrame)
        self.assertEqual(sample_result_480.sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(target_result_481.sort_index(axis=1), e_t, lazy=True)

        # Test passing DataFrames rather than Samples
        r = model_matrix(
            pd.DataFrame({"a": (0, 1, 2), "b": (0, None, 2), "c": ("a", "b", "a")}),
            pd.DataFrame(
                {"a": (0, 1, 2, None), "d": (0, 2, 2, 1), "c": ("a", "b", "a", "c")}
            ),
        )
        sample_result_494 = r["sample"]
        target_result_495 = r["target"]
        sample_result_494 = _assert_type(sample_result_494, pd.DataFrame)
        target_result_495 = _assert_type(target_result_495, pd.DataFrame)
        self.assertEqual(sample_result_494.sort_index(axis=1), e_s, lazy=True)
        self.assertEqual(target_result_495.sort_index(axis=1), e_t, lazy=True)

        # Check warnings for variables not present in both
        self.assertWarnsRegexp(
            "Ignoring variables not present in all Samples",
            model_matrix,
            s,
            t,
        )

        # Test zero rows warning:
        self.assertRaisesRegex(
            AssertionError,
            "sample must have more than zero rows",
            model_matrix,
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
        r = model_matrix(s_df_bad_col_names)
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
        sample_result_536 = _assert_type(r["sample"], pd.DataFrame)
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
        r = model_matrix(s_df_bad_col_names)
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
        sample_result_571 = _assert_type(r["sample"], pd.DataFrame)
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
        r = model_matrix(s, variables=["c"])
        e = pd.DataFrame({"c[a]": (1.0, 0.0, 1.0), "c[b]": (0.0, 1.0, 0.0)})
        self.assertEqual(r["sample"], e)
        self.assertIsNone(r["target"])

        #  Single covariate which doesn't exist in both should raise error
        self.assertRaisesRegex(
            ValueError,
            "requested variables are not in all Samples",
            model_matrix,
            s,
            t,
            ["b"],
        )

        # Test add_na argument
        sample_no_na = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": ["keep", "keep"]}
        )
        target_with_na = pd.DataFrame(
            {"a": [1.0, None], "b": [1.0, 2.0], "c": ["keep", "keep"]}
        )
        add_na_result = model_matrix(sample_no_na, target_with_na, add_na=True)
        add_na_sample = _assert_type(add_na_result["sample"], pd.DataFrame)
        add_na_target = _assert_type(add_na_result["target"], pd.DataFrame)
        self.assertIn("_is_na_a[T.True]", add_na_sample.columns)
        self.assertIn("_is_na_a[T.True]", add_na_target.columns)
        self.assertTrue((add_na_sample["_is_na_a[T.True]"] == 0.0).all())
        self.assertEqual(add_na_target["_is_na_a[T.True]"].tolist(), [0.0, 1.0])

        e = pd.DataFrame(
            {"a": (0.0, 2.0), "b": (0.0, 2.0), "c[a]": (1.0, 1.0), "c[b]": (0.0, 0.0)},
            index=(0, 2),
        )
        r = model_matrix(s, add_na=False)
        self.assertWarnsRegexp(
            "Dropping all rows with NAs", model_matrix, s, add_na=False
        )
        sample_add_na = _assert_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_add_na.sort_index(axis=1), e)
        self.assertIsNone(r["target"])

        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty sample. Consider using add_na=True",
            model_matrix,
            pd.DataFrame({"a": [None], "b": [None], "c": [None]}),
            add_na=False,
        )

        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty target. Consider using add_na=True",
            model_matrix,
            pd.DataFrame({"a": [1.0], "b": [1.0], "c": ["keep"]}),
            pd.DataFrame({"a": [None], "b": [None], "c": [None]}),
            add_na=False,
        )

        cat_df = pd.DataFrame(
            {
                "a": [1.0, None, 2.0],
                "b": [1.0, 2.0, 3.0],
                "c": pd.Categorical(
                    ["keep", "drop", "keep"], categories=["keep", "drop"]
                ),
            }
        )
        cat_result = model_matrix(cat_df, add_na=False)["sample"]
        cat_result = _assert_type(cat_result, pd.DataFrame)
        self.assertIn("c[drop]", cat_result.columns)
        self.assertTrue((cat_result["c[drop]"] == 0.0).all())

        string_df = pd.DataFrame(
            {
                "a": [1.0, None, 2.0],
                "b": [1.0, 2.0, 3.0],
                "c": pd.Series(["keep", "string_only", "keep"], dtype="string"),
            }
        )
        string_result = model_matrix(string_df, add_na=False)["sample"]
        string_result = _assert_type(string_result, pd.DataFrame)
        self.assertIn("c[string_only]", string_result.columns)
        self.assertTrue((string_result["c[string_only]"] == 0.0).all())

        obj_df = pd.DataFrame(
            {
                "a": [1.0, None, 2.0],
                "b": [1.0, 2.0, 3.0],
                "c": ["keep", "in_dropped_row", "keep"],
            }
        )
        obj_result = model_matrix(obj_df.copy(), add_na=False)["sample"]
        obj_result = _assert_type(obj_result, pd.DataFrame)
        self.assertIn("c[in_dropped_row]", obj_result.columns)
        self.assertTrue((obj_result["c[in_dropped_row]"] == 0.0).all())

        target_df = pd.DataFrame(
            {
                "a": [1.0, None],
                "b": [1.0, 2.0],
                "c": ["keep", "target_only"],
            }
        )
        combined = model_matrix(obj_df.copy(), target_df, add_na=False)
        sample_combined = _assert_type(combined["sample"], pd.DataFrame)
        target_combined = _assert_type(combined["target"], pd.DataFrame)
        self.assertIn("c[target_only]", sample_combined.columns)
        self.assertIn("c[target_only]", target_combined.columns)
        self.assertTrue((sample_combined["c[target_only]"] == 0.0).all())
        self.assertTrue((target_combined["c[target_only]"] == 0.0).all())

        #  Test return_type argument
        r_one = model_matrix(s, t, return_type="one")["model_matrix"]
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
        r_one = _assert_type(r_one, pd.DataFrame)
        self.assertEqual(r_one.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)

        # Test return_var_type argument
        r_df = model_matrix(s, t, return_type="one", return_var_type="dataframe")[
            "model_matrix"
        ]
        r_df = _assert_type(r_df, pd.DataFrame)
        self.assertEqual(r_df.sort_index(axis=1), pd.concat((e_s, e_t)), lazy=True)
        r_mat = model_matrix(s, t, return_type="one", return_var_type="matrix")
        model_matrix_mat = _assert_type(r_mat["model_matrix"])
        self.assertEqual(
            model_matrix_mat,
            pd.concat((e_s, e_t))
            .reindex(columns=r_mat["model_matrix_columns_names"])
            .values,
        )
        r_sparse = model_matrix(s, t, return_type="one", return_var_type="sparse")
        model_matrix_sparse = _assert_type(r_sparse["model_matrix"], csc_matrix)
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
        self.assertIs(type(model_matrix_sparse), csc_matrix)

        # Test formula argument
        result_a_plus_b = _assert_type(
            model_matrix(s, formula="a + b")["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_a_plus_b.sort_index(axis=1),
            pd.DataFrame({"a": (0.0, 1.0, 2.0), "b": (0.0, 0.0, 2.0)}),
        )

        result_b = _assert_type(model_matrix(s, formula="b ")["sample"], pd.DataFrame)
        self.assertEqual(
            result_b.sort_index(axis=1),
            pd.DataFrame({"b": (0.0, 0.0, 2.0)}),
        )
        result_a_times_c = _assert_type(
            model_matrix(s, formula="a * c ")["sample"], pd.DataFrame
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
        result_a_b_list = _assert_type(
            model_matrix(s, formula=["a", "b"])["sample"], pd.DataFrame
        )
        self.assertEqual(
            result_a_b_list.sort_index(axis=1),
            pd.DataFrame({"a": (0.0, 1.0, 2.0), "b": (0.0, 0.0, 2.0)}),
        )

        # Test penalty_factor argument
        self.assertEqual(
            model_matrix(s, formula=["a", "b"])["penalty_factor"],
            np.array([1, 1]),
        )
        self.assertEqual(
            model_matrix(s, formula=["a", "b"], penalty_factor=[1, 2])[
                "penalty_factor"
            ],
            np.array([1, 2]),
        )
        self.assertEqual(
            model_matrix(s, formula="a+b", penalty_factor=[2])["penalty_factor"],
            np.array([2, 2]),
        )
        self.assertRaisesRegex(
            AssertionError,
            "penalty factor and formula must have the same length",
            model_matrix,
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
        r = model_matrix(s, one_hot_encoding=True)
        sample_result_750 = _assert_type(r["sample"], pd.DataFrame)
        self.assertEqual(sample_result_750.sort_index(axis=1), e, lazy=True)

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


class TestModelMatrixEdgeCases(balance.testutil.BalanceTestCase):
    """Test edge cases in model_matrix functions."""

    def test_model_matrix_with_bracket_in_variable_names(self) -> None:
        """Test that model_matrix raises error for variable names with brackets (line 306)."""
        sample_df = pd.DataFrame({"var[one]": [1, 2, 3], "b": [4, 5, 6]})
        target_df = pd.DataFrame({"var[one]": [7, 8, 9], "b": [10, 11, 12]})

        with self.assertRaisesRegex(
            ValueError, "Variable names cannot contain characters"
        ):
            model_matrix(sample_df, target_df)

    def test_model_matrix_with_bracket_in_multiple_variables(self) -> None:
        """Test that model_matrix reports all bracket-containing variables."""
        sample_df = pd.DataFrame(
            {
                "var[one]": [1, 2, 3],
                "var]two[": [4, 5, 6],
                "normal": [7, 8, 9],
            }
        )
        target_df = pd.DataFrame(
            {
                "var[one]": [1, 2, 3],
                "var]two[": [4, 5, 6],
                "normal": [7, 8, 9],
            }
        )

        with self.assertRaisesRegex(
            ValueError, "Variable names cannot contain characters.*\\[.*\\]"
        ):
            model_matrix(sample_df, target_df)

    def test_model_matrix_empty_target_after_dropna(self) -> None:
        """Test that model_matrix raises error when target is empty after dropna (line 362)."""
        sample_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Target has all NA values - after dropna, it becomes empty
        target_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "Dropping rows led to empty target"):
            model_matrix(sample_df, target_df, add_na=False)
