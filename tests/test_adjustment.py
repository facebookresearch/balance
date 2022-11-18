# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import random

import balance.testutil

import numpy as np
import pandas as pd

from balance import adjustment as balance_adjustment, util as balance_util

from balance.sample_class import Sample
from balance.weighting_methods import (
    cbps as balance_cbps,
    ipw as balance_ipw,
    poststratify as balance_poststratify,
)

EPSILON = 0.00001


sample = Sample.from_frame(
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

target = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": (-42, 8, 2),
            "c": ("x", "y", "z"),
            "id": (1, 2, 3),
            "w": (2, 0.5, 1),
        }
    ),
    id_column="id",
    weight_column="w",
)


class TestAdjustment(
    balance.testutil.BalanceTestCase,
):
    def test_trim_weights(self):
        from balance.adjustment import trim_weights

        # Test no trimming
        # Notice how it changes the dtype of int64 to float64~
        pd.testing.assert_series_equal(
            trim_weights(pd.Series([0, 1, 2])), pd.Series([0.0, 1.0, 2.0])
        )
        self.assertEqual(type(trim_weights(pd.Series([0, 1, 2]))), pd.Series)
        self.assertEqual(trim_weights(pd.Series([0, 1, 2])).dtype, np.float64)

        random.seed(42)
        w = np.random.uniform(0, 1, 10000)
        self.assertEqual(
            trim_weights(
                w,
                weight_trimming_percentile=None,
                weight_trimming_mean_ratio=None,
                keep_sum_of_weights=False,
            ),
            w,
        )

        # Test exceptions
        with self.assertRaisesRegex(
            TypeError, "weights must be np.array or pd.Series, are of type*"
        ):
            trim_weights("Strings don't get trimmed", weight_trimming_mean_ratio=1)
        with self.assertRaisesRegex(ValueError, "Only one"):
            trim_weights(
                np.array([0, 1, 2]),
                1,
                1,
            )

        # Test weight_trimming_mean_ratio
        random.seed(42)
        w = np.random.uniform(0, 1, 10000)
        res = trim_weights(w, weight_trimming_mean_ratio=1)
        self.assertAlmostEqual(np.mean(w), np.mean(res), delta=EPSILON)
        self.assertAlmostEqual(
            np.mean(w) / np.min(w), np.max(res) / np.min(res), delta=EPSILON
        )

        # Test weight_trimming_percentile
        random.seed(42)
        w = np.random.uniform(0, 1, 10000)
        self.assertTrue(
            max(
                trim_weights(
                    w, weight_trimming_percentile=(0, 0.11), keep_sum_of_weights=False
                )
            )
            < 0.9
        )
        self.assertTrue(
            min(
                trim_weights(
                    w, weight_trimming_percentile=(0.11, 0), keep_sum_of_weights=False
                )
            )
            > 0.1
        )
        e = trim_weights(w, weight_trimming_percentile=(0.11, 0.11))
        self.assertTrue(min(e) > 0.1)
        self.assertTrue(max(e) < 0.9)

    def test_default_transformations(self):
        # For multiple dataframes
        input = (
            pd.DataFrame({"a": (1, 2), "b": ("a", "b")}),
            pd.DataFrame({"c": (1, 2), "d": ("a", "b")}),
        )
        r = balance_adjustment.default_transformations(input)
        self.assertEqual(
            r,
            {
                "a": balance_util.quantize,
                "b": balance_util.fct_lump,
                "c": balance_util.quantize,
                "d": balance_util.fct_lump,
            },
        )

        # For one dataframe
        input = pd.DataFrame({"a": (1, 2), "b": ("a", "b")})
        r = balance_adjustment.default_transformations([input])
        self.assertEqual(
            r,
            {
                "a": balance_util.quantize,
                "b": balance_util.fct_lump,
            },
        )

        # For boolean and Int64 input
        input = pd.DataFrame({"a": (1, 2), "b": (True, False)})
        input = input.astype(
            dtype={
                "a": "Int64",
                "b": "boolean",
            }
        )
        r = balance_adjustment.default_transformations([input])
        self.assertEqual(
            r,
            {
                "a": balance_util.quantize,
                "b": balance_util.fct_lump,
            },
        )

    def test_default_transformations_pd_int64(self):
        nullable_int = pd.DataFrame({"a": pd.array((1, 2), dtype="Int64")})

        numpy_int = nullable_int.astype(np.int64)

        test = balance_adjustment.default_transformations([nullable_int])
        truth = balance_adjustment.default_transformations([numpy_int])

        self.assertEqual(test, truth)

    def test_apply_transformations(self):
        s = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        t = pd.DataFrame({"d": [4, 5, 6, 7], "e": [1, 2, 3, 4]})

        transformations = {"d": lambda x: x * 2, "f": lambda x: x.d + 1}
        r = balance_adjustment.apply_transformations((s, t), transformations)

        e = (
            pd.DataFrame({"d": [2, 4, 6], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "f": [5, 6, 7, 8]}),
        )

        self.assertEqual(r[0], e[0], lazy=True)
        self.assertEqual(r[1], e[1], lazy=True)

        # No transformations or additions
        self.assertEqual(balance_adjustment.apply_transformations((s, t), None), (s, t))

        # Only transformations
        r = balance_adjustment.apply_transformations((s, t), {"d": lambda x: x * 2})
        e = (pd.DataFrame({"d": [2, 4, 6]}), pd.DataFrame({"d": [8, 10, 12, 14]}))
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Only additions
        r = balance_adjustment.apply_transformations((s, t), {"f": lambda x: x.d + 1})
        e = (pd.DataFrame({"f": [2, 3, 4]}), pd.DataFrame({"f": [5, 6, 7, 8]}))
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Warns about dropping variable
        self.assertWarnsRegexp(
            r"Dropping the variables: \['e'\]",
            balance_adjustment.apply_transformations,
            (s, t),
            transformations,
        )

        # Does not drop
        r = balance_adjustment.apply_transformations(
            (s, t), transformations, drop=False
        )
        e = (
            pd.DataFrame({"d": [2, 4, 6], "e": [1, 2, 3], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "e": [1, 2, 3, 4], "f": [5, 6, 7, 8]}),
        )
        self.assertEqual(r[0], e[0], lazy=True)
        self.assertEqual(r[1], e[1], lazy=True)

        # Works on three dfs
        q = pd.DataFrame({"d": [8, 9], "g": [1, 2]})
        r = balance_adjustment.apply_transformations((s, t, q), transformations)
        e = (
            pd.DataFrame({"d": [2, 4, 6], "f": [2, 3, 4]}),
            pd.DataFrame({"d": [8, 10, 12, 14], "f": [5, 6, 7, 8]}),
            pd.DataFrame({"d": [16, 18], "f": [9, 10]}),
        )
        self.assertEqual(r[0], e[0], lazy=True)
        self.assertEqual(r[1], e[1], lazy=True)
        self.assertEqual(r[2], e[2], lazy=True)

        # Test that functions are computed over all dfs passed, not each individually
        transformations = {"d": lambda x: x / max(x)}
        r = balance_adjustment.apply_transformations((s, t), transformations)
        e = (
            pd.DataFrame({"d": [2 / 14, 4 / 14, 6 / 14]}),
            pd.DataFrame({"d": [8 / 14, 10 / 14, 12 / 14, 14 / 14]}),
        )
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Transformation of a column which does not exist in one of the dataframes
        s = pd.DataFrame({"d": [1, 2, 3], "e": [1, 2, 3]})
        t = pd.DataFrame({"d": [4, 5, 6, 7]})
        transformations = {"e": lambda x: x * 2}
        r = balance_adjustment.apply_transformations((s, t), transformations)
        e = (
            pd.DataFrame({"e": [2.0, 4.0, 6.0]}),
            pd.DataFrame({"e": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Additon of a column based on one which does not exist in one of the dataframes
        transformations = {"f": lambda x: x.e * 2}
        r = balance_adjustment.apply_transformations((s, t), transformations)
        e = (
            pd.DataFrame({"f": [2.0, 4.0, 6.0]}),
            pd.DataFrame({"f": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Column which does not exist in one of the dataframes
        # and is also specified
        s = pd.DataFrame({"d": [1, 2, 3], "e": [0, 0, 0]})
        t = pd.DataFrame({"d": [4, 5, 6, 7]})
        transformations = {"e": lambda x: x + 1}
        r = balance_adjustment.apply_transformations((s, t), transformations)
        e = (
            pd.DataFrame({"e": [1.0, 1.0, 1.0]}),
            pd.DataFrame({"e": [np.nan, np.nan, np.nan, np.nan]}),
        )
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Test that indices are ignored in splitting dfs
        s = pd.DataFrame({"d": [1, 2, 3]}, index=(5, 6, 7))
        t = pd.DataFrame({"d": [4, 5, 6, 7]}, index=(0, 1, 2, 3))
        transformations = {"d": lambda x: x}
        r = balance_adjustment.apply_transformations((s, t), transformations)
        e = (s, t)
        self.assertEqual(r[0], e[0])
        self.assertEqual(r[1], e[1])

        # Test indices are handeled okay (this example reuired reset_index of all_data)
        s = pd.DataFrame({"a": (0, 0, 0, 0, 0, 0, 0, 0)})
        t = pd.DataFrame({"a": (1, 1, 1, 1)})
        r = balance_adjustment.apply_transformations((s, t), "default")
        e = (
            pd.DataFrame({"a": ["(-0.001, 0.7]"] * 8}),
            pd.DataFrame({"a": ["(0.7, 1.0]"] * 4}),
        )
        self.assertEqual(r[0].astype(str), e[0])
        self.assertEqual(r[1].astype(str), e[1])

        #  Test default transformations
        s = pd.DataFrame({"d": range(0, 100), "e": ["a"] * 96 + ["b"] * 4})
        t = pd.DataFrame({"d": range(0, 100), "e": ["a"] * 96 + ["b"] * 4})
        r_s, r_t = balance_adjustment.apply_transformations((s, t), "default")

        self.assertEqual(r_s["d"].drop_duplicates().values.shape[0], 10)
        self.assertEqual(r_t["d"].drop_duplicates().values.shape[0], 10)

        self.assertEqual(r_s["e"].drop_duplicates().values, ("a", "_lumped_other"))
        self.assertEqual(r_t["e"].drop_duplicates().values, ("a", "_lumped_other"))

    def test_invalid_input_to_apply_transformations(self):
        # Test non-existent transformation
        self.assertRaisesRegex(
            NotImplementedError,
            "Unknown transformations",
            balance_adjustment.apply_transformations,
            (sample.df,),
            "foobar",
        )

        # Test non-dataframe input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' must contain DataFrames",
            balance_adjustment.apply_transformations,
            (sample,),
            "foobar",
        )

        # Test non-tuple input
        self.assertRaisesRegex(
            AssertionError,
            "'dfs' argument must be a tuple of DataFrames",
            balance_adjustment.apply_transformations,
            sample.df,
            "foobar",
        )

    def test__find_adjustment_method(self):
        self.assertTrue(
            balance_adjustment._find_adjustment_method("ipw") is balance_ipw.ipw
        )
        self.assertTrue(
            balance_adjustment._find_adjustment_method("cbps") is balance_cbps.cbps
        )
        self.assertTrue(
            balance_adjustment._find_adjustment_method("poststratify")
            is balance_poststratify.poststratify
        )
        with self.assertRaisesRegex(ValueError, "Unknown adjustment method*"):
            balance_adjustment._find_adjustment_method("some_other_value")
