# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import balance.testutil

import numpy as np
import numpy.testing
import pandas as pd

from balance import util as balance_util
from balance.sample_class import Sample

from patsy import dmatrix  # pyre-ignore[21]: this module exists
from scipy.sparse import csc_matrix


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test__check_weighting_methods_input(self):
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
        self.assertEqual(balance_util.formula_generator("a"), "a")
        self.assertEqual(balance_util.formula_generator(["a", "b", "c"]), "c + b + a")
        # check exceptions
        self.assertRaisesRegex(
            Exception,
            "This formula type is not supported",
            balance_util.formula_generator,
            ["a", "b"],
            "intercation",
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

    def test_rm_mutual_nas(self):
        from balance.util import rm_mutual_nas

        self.assertEqual(rm_mutual_nas([1, 2, 3], [2, 3, None]), [[1, 2], [2.0, 3.0]])

        d = np.array((0, 1, 2))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), d)

        d2 = np.array((5, 6, 7))
        numpy.testing.assert_array_equal(rm_mutual_nas(d, d2), (d, d2))

        r = rm_mutual_nas(d, d2, None)
        for i, j in zip(r, (d, d2, None)):
            numpy.testing.assert_array_equal(i, j)

        # test exceptions
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

        d4 = np.array((np.nan, 9, -np.inf))
        e = [np.array((1,)), np.array((6,)), np.array((9,))]

        r = rm_mutual_nas(d, d2, d4)
        for i, j in zip(r, e):
            print("A", i, j)
            numpy.testing.assert_array_equal(i, j)

        r = rm_mutual_nas(d, d2, d4, None)
        for i, j in zip(r, e):
            numpy.testing.assert_array_equal(i, j)

        d5 = np.array(("a", "b", "c"))
        numpy.testing.assert_array_equal(rm_mutual_nas(d5), d5)

        e = [np.array((9,)), np.array(("b",))]
        r = rm_mutual_nas(d4, d5)
        for i, j in zip(r, e):
            numpy.testing.assert_array_equal(i, j)

        # Single arraylikes
        d = np.array((0, 1, 2, None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))
        d = np.array((0, 1, 2, np.nan))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), (0, 1, 2))
        d = np.array(("a", "b", None))
        numpy.testing.assert_array_equal(rm_mutual_nas(d), ("a", "b"))
        d = np.array(("a", 1, None))
        # NOTE: In the next test we must define that `dtype=object`
        # since this dtype is preserved from d. Otherwise, using np.array(("a", 1)) will have
        # a dtype of '<U1'
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

        # Should only accept array like or none arguments
        self.assertRaises(ValueError, rm_mutual_nas, d, "a")

        # Make sure we can deal with various np and pd arrays
        x1 = pd.array([1, 2, None, np.nan, pd.NA, 3])
        x2 = pd.array([1.1, 2, 3, None, np.nan, pd.NA])
        x3 = pd.array([1.1, 2, 3, 4, 5, 6])
        x4 = pd.array(["1.1", 2, 3, None, np.nan, pd.NA])
        x5 = pd.array(["1.1", "2", "3", None, np.nan, pd.NA], dtype="string")
        x6 = np.array([1, 2, 3.3, 4, 5, 6])
        x7 = np.array([1, 2, 3.3, 4, "5", "6"])
        x8 = [1, 2, 3.3, 4, "5", "6"]

        # The values we expect to see after using rm_mutual_nas:
        self.assertEqual(
            [list(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)],
            [
                [1, 2],
                [1.1, 2],
                [1.1, 2.0],
                ["1.1", 2],
                ["1.1", "2"],
                [1.0, 2.0],
                ["1", "2"],
                [1, 2],
            ],
        )
        # The types before and after the na removal will remain the same:
        self.assertEqual(
            [type(x) for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7, x8)],
            [type(x) for x in (x1, x2, x3, x4, x5, x6, x7, x8)],
        )
        self.assertEqual(
            [type(x) for x in rm_mutual_nas(x1, x4, x5, x6, x7, x8)],
            [
                pd.core.arrays.integer.IntegerArray,
                pd.core.arrays.numpy_.PandasArray,
                pd.core.arrays.string_.StringArray,
                np.ndarray,
                np.ndarray,
                list,
            ],
        )

        # NOTE: pd.FloatingArray were only added in pandas version 1.2.0.
        # Before that, they were called PandasArray. For details, see:
        # https://pandas.pydata.org/docs/dev/reference/api/pandas.arrays.FloatingArray.html
        if pd.__version__ < "1.2.0":
            e = [
                pd.core.arrays.numpy_.PandasArray,
                pd.core.arrays.numpy_.PandasArray,
            ]
        else:
            e = [
                pd.core.arrays.floating.FloatingArray,
                pd.core.arrays.floating.FloatingArray,
            ]
        self.assertEqual([type(x) for x in rm_mutual_nas(x2, x3)], e)

        # The dtype before and after the na removal will remain the same: (only relevant for np and pd arrays)
        self.assertEqual(
            [x.dtype for x in rm_mutual_nas(x1, x2, x3, x4, x5, x6, x7)],
            [x.dtype for x in (x1, x2, x3, x4, x5, x6, x7)],
        )
        # The dtypes:
        # [Int64Dtype(),
        #  PandasDtype('object'),
        #  PandasDtype('float64'),
        #  PandasDtype('object'),
        #  StringDtype,
        #  dtype('float64'),
        #  dtype('<U32')]

    def test_choose_variables(self):
        # For one dataframe
        self.assertEqual(
            sorted(balance_util.choose_variables(pd.DataFrame({"a": [1], "b": [2]}))),
            ["a", "b"],
        )
        # For two dataframes
        # Not providing variables
        self.assertEqual(
            balance_util.choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}),
                pd.DataFrame({"c": [1], "b": [2]}),
                variables="",
            ),
            ("b",),
        )
        self.assertEqual(
            balance_util.choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"c": [1], "b": [2]})
            ),
            ("b",),
        )
        self.assertWarnsRegexp(
            "Ignoring variables not present in all Samples",
            balance_util.choose_variables,
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"c": [1], "d": [2]}),
        )
        self.assertWarnsRegexp(
            "Sample and target have no variables in common",
            balance_util.choose_variables,
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"c": [1], "d": [2]}),
        )
        self.assertEqual(
            balance_util.choose_variables(
                pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"c": [1], "d": [2]})
            ),
            (),
        )

        self.assertRaisesRegex(
            Exception,
            "requested variables are not in all Samples",
            balance_util.choose_variables,
            pd.DataFrame({"a": [1], "b": [2]}),
            pd.DataFrame({"c": [1], "b": [2]}),
            variables="a",
        )

        #  Three dataframes
        self.assertEqual(
            balance_util.choose_variables(
                pd.DataFrame({"a": [1], "b": [2], "c": [2]}),
                pd.DataFrame({"c": [1], "b": [2]}),
                pd.DataFrame({"a": [1], "b": [2]}),
            ),
            ("b",),
        )

        self.assertRaisesRegex(
            Exception,
            "requested variables are not in all Samples: {'a'}",
            balance_util.choose_variables,
            pd.DataFrame({"a": [1], "b": [2], "c": [2]}),
            pd.DataFrame({"c": [1], "b": [2]}),
            variables=("a", "b", "c"),
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

    def test_fct_lump(self):
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

        # Categorical model matrix is equal to string model matrix
        # Load and process test data
        from sklearn import datasets

        wine_df = pd.DataFrame(datasets.load_wine().data)
        wine_df.columns = datasets.load_wine().feature_names
        wine_df = wine_df.rename(
            columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
        )
        wine_df["id"] = pd.Series(
            range(1, len(wine_df) + 1)
        )  # Create a fake id variable, required by balance
        wine_df.alcohol = pd.cut(
            wine_df.alcohol, bins=[0, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 100]
        )

        # Create a version of the dataset that treats factor variable as character
        wine_df_copy = wine_df.copy(deep=True)
        wine_df_copy.alcohol = wine_df_copy.alcohol.astype("object")

        # Split into "survey" and "population" datasets based on "wine class"
        wine_class = pd.Series(datasets.load_wine().target)
        wine_survey = Sample.from_frame(wine_df.loc[wine_class == 0, :])
        wine_pop = Sample.from_frame(wine_df.loc[wine_class != 0, :])
        wine_survey = wine_survey.set_target(wine_pop)
        wine_survey_copy = Sample.from_frame(wine_df_copy.loc[wine_class == 0, :])
        wine_pop_copy = Sample.from_frame(wine_df_copy.loc[wine_class != 0, :])
        wine_survey_copy = wine_survey_copy.set_target(wine_pop_copy)

        # Generate weights
        output_cat_var = wine_survey.adjust(
            transformations={
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
        )

        output_string_var = wine_survey_copy.adjust(
            transformations={
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
        )

        # Check that model coefficients are identical in categorical and string variable coding
        self.assertEqual(
            output_cat_var.model()["perf"]["coefs"],
            output_string_var.model()["perf"]["coefs"],
        )

    def test_fct_lump_by(self):
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
                name="a",
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
