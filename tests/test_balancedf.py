# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile

from copy import deepcopy

import balance.testutil
import IPython.display

import numpy as np
import pandas as pd

from balance.balancedf_class import (  # noqa
    BalanceCovarsDF,  # noqa
    BalanceOutcomesDF,  # noqa
    BalanceWeightsDF,  # noqa
)
from balance.sample_class import Sample

# TODO: verify all the objects below are used
s1 = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3, 1),
            "b": (-42, 8, 2, -42),
            "o": (7, 8, 9, 10),
            "c": ("x", "y", "z", "v"),
            "id": (1, 2, 3, 4),
            "w": (0.5, 2, 1, 1),
        }
    ),
    id_column="id",
    weight_column="w",
    outcome_columns="o",
)

s2 = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": (4, 6, 8),
            "id": (1, 2, 3),
            "w": (0.5, 1, 2),
            "c": ("x", "y", "z"),
        }
    ),
    id_column="id",
    weight_column="w",
)

s3 = s1.set_target(s2)
s3_null = s3.adjust(method="null")

s3_null_madeup_weights = deepcopy(s3_null)
s3_null_madeup_weights.set_weights((1, 2, 3, 1))

s4 = Sample.from_frame(
    pd.DataFrame(
        {"a": (0, None, 2), "b": (0, None, 2), "c": ("a", "b", "c"), "id": (1, 2, 3)}
    ),
    outcome_columns=("b", "c"),
)

o = s1.outcomes()


s_o = Sample.from_frame(
    pd.DataFrame({"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}),
    id_column="id",
    outcome_columns=("o1", "o2"),
)

t_o = Sample.from_frame(
    pd.DataFrame(
        {
            "o1": (7, 8, 9, 10, 11, 12, 13, 14),
            "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
            "id": (1, 2, 3, 4, 5, 6, 7, 8),
        }
    ),
    id_column="id",
    outcome_columns=("o1", "o2"),
)
s_o2 = s_o.set_target(t_o)

c = s1.covars()

w = s1.weights()


class TestBalanceOutcomesDF(balance.testutil.BalanceTestCase):
    def test_Sample_outcomes(self):
        self.assertTrue(isinstance(s4.outcomes(), BalanceOutcomesDF))
        self.assertEqual(
            s4.outcomes().df, pd.DataFrame({"b": (0, None, 2), "c": ("a", "b", "c")})
        )

        # Test with multicharacter string name
        s = Sample.from_frame(
            pd.DataFrame({"aardvark": (0, None, 2), "id": (1, 2, 3)}),
            outcome_columns="aardvark",
        )
        self.assertEqual(s.outcomes().df, pd.DataFrame({"aardvark": (0, None, 2)}))

        # Null outcomes
        self.assertTrue(s2.outcomes() is None)

    def test_BalanceOutcomesDF_df(self):
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceOutcomesDF.df, property))
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            o.df()
        # Here is how we can call it as a function:
        self.assertEqual(BalanceOutcomesDF.df.fget(o), o.df)

        # Check values are as expected:
        # NOTE that values changed from integer to float
        self.assertEqual(o.df, pd.DataFrame({"o": (7.0, 8.0, 9.0, 10.0)}))

    def test__get_df_and_weights(self):
        from balance.balancedf_class import BalanceDF

        df, w = BalanceDF._get_df_and_weights(o)
        # Check types
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(type(w), np.ndarray)
        # Check values
        self.assertEqual(df.to_dict(), {"o": {0: 7.0, 1: 8.0, 2: 9.0, 3: 10.0}})
        self.assertEqual(w, np.array([0.5, 2.0, 1.0, 1.0]))

    def test_BalanceOutcomesDF_names(self):
        self.assertEqual(o.names(), ["o"])

    def test_BalanceOutcomesDF__sample(self):
        self.assertTrue(o._sample is s1)

    def test_BalanceOutcomesDF_weights(self):
        pd.testing.assert_series_equal(o._weights, pd.Series((0.5, 2, 1, 1)))

    def test_BalanceOutcomesDF_relative_response_rates(self):
        self.assertEqual(
            s_o.outcomes().relative_response_rates(),
            pd.DataFrame({"o1": [100.0, 4], "o2": [75.0, 3]}, index=["%", "n"]),
            lazy=True,
        )

        self.assertEqual(s_o.outcomes().relative_response_rates(target=True), None)

        # compared with a larget target
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(True, per_column=True)
            .round(3)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 50.0}},
        )

        df_target = pd.DataFrame(
            {
                "o1": (7, 8, 9, 10, 11, 12, 13, 14),
                "o2": (7, 8, 9, np.nan, np.nan, 12, 13, 14),
            }
        )
        # Relative to per column:
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(target=df_target, per_column=True)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 50.0}},
        )
        # Checking that if we force per_column=True
        # On a df_target that is not the same column structure as s_o2.outcomes()
        # It will lead to a ValueError
        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns*"
        ):
            s_o2.outcomes().relative_response_rates(
                df_target.iloc[:, 0:1], per_column=True
            )

        # Relative to all notnull rows in outcome
        self.assertEqual(
            s_o2.outcomes().relative_response_rates(target=True).round(3).to_dict(),
            {"o1": {"n": 4.0, "%": 66.667}, "o2": {"n": 3.0, "%": 50.0}},
        )
        self.assertEqual(
            s_o2.outcomes()
            .relative_response_rates(
                target=df_target, per_column=False  # This is the default.
            )
            .round(3)
            .to_dict(),
            s_o2.outcomes().relative_response_rates(target=True).round(3).to_dict(),
        )
        # This will also work with different shape of columns (exactly what we need for .summary())
        self.assertEqual(
            s_o.outcomes()
            .relative_response_rates(df_target.iloc[:, 0:1], per_column=False)
            .round(3)
            .to_dict(),
            {"o1": {"n": 4.0, "%": 50.0}, "o2": {"n": 3.0, "%": 37.5}},
        )

    def test_BalanceOutcomesDF_target_response_rates(self):
        self.assertEqual(
            s_o2.outcomes().target_response_rates(),
            pd.DataFrame({"o1": {"n": 8.0, "%": 100.0}, "o2": {"n": 6.0, "%": 75.0}}),
            lazy=True,
        )

    def test_BalanceOutcomesDF_summary(self):
        def _remove_whitespace_and_newlines(s):
            return " ".join(s.split())

        e_str = """\
            2 outcomes: ['o1' 'o2']
            Mean outcomes:
                    _is_na_o2[False]  _is_na_o2[True]   o1   o2
            source
            self                0.75             0.25  8.5  6.0

            Response rates (relative to number of respondents in sample):
                o1    o2
            n    4.0   3.0
            %  100.0  75.0
            """
        self.assertEqual(
            _remove_whitespace_and_newlines(s_o.outcomes().summary()),
            _remove_whitespace_and_newlines(e_str),
        )

        e_str = """\
            2 outcomes: ['o1' 'o2']
            Mean outcomes:
                    _is_na_o2[False]  _is_na_o2[True]    o1     o2
            source
            self                0.75             0.25   8.5  6.000
            target              0.75             0.25  10.5  7.875

            Response rates (relative to number of respondents in sample):
                o1    o2
            n    4.0   3.0
            %  100.0  75.0
            Response rates (relative to notnull rows in the target):
                    o1    o2
            n   4.000000   3.0
            %  66.666667  50.0
            Response rates (in the target):
                o1    o2
            n    8.0   6.0
            %  100.0  75.0
            """
        self.assertEqual(
            _remove_whitespace_and_newlines(s_o2.outcomes().summary()),
            _remove_whitespace_and_newlines(e_str),
        )


class TestBalanceCovarsDF(balance.testutil.BalanceTestCase):
    def test_BalanceCovarsDF_df(self):
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceCovarsDF.df, property))
        self.assertEqual(BalanceOutcomesDF.df.fget(c), c.df)
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            c.df()

        # NOTE: while the original datatype had integers, the stored df has only floats:
        self.assertEqual(
            c.df,
            pd.DataFrame(
                {
                    "a": (1.0, 2.0, 3.0, 1.0),
                    "b": (-42.0, 8.0, 2.0, -42.0),
                    "c": ("x", "y", "z", "v"),
                }
            ),
        )

    def test_BalanceCovarsDF_names(self):
        self.assertEqual(c.names(), ["a", "b", "c"])
        self.assertEqual(type(c.names()), list)

    def test_BalanceCovarsDF__sample(self):
        self.assertTrue(c._sample is s1)

    def test_BalanceCovarsDF_weights(self):
        pd.testing.assert_series_equal(
            c._weights, pd.Series(np.array([0.5, 2.0, 1.0, 1.0]))
        )


class TestBalanceWeightsDF(balance.testutil.BalanceTestCase):
    def test_BalanceWeightsDF_df(self):
        # Verify that the @property decorator works properly.
        self.assertTrue(isinstance(BalanceWeightsDF.df, property))
        self.assertEqual(BalanceWeightsDF.df.fget(w), w.df)
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            w.df()

        # Check values are as expected:
        self.assertEqual(w.df, pd.DataFrame({"w": (0.5, 2, 1, 1)}))

    def test_BalanceWeightsDF_names(self):
        self.assertEqual(w.names(), ["w"])

    def test_BalanceWeightsDF__sample(self):
        self.assertTrue(c._sample is s1)

    def test_BalanceWeightsDF_weights(self):
        self.assertTrue(w._weights is None)

    def test_BalanceWeightsDF_design_effect(self):

        s = Sample.from_frame(
            pd.DataFrame({"w": (1, 2, 4), "id": (1, 2, 3)}),
            id_column="id",
            weight_column="w",
        )
        self.assertTrue(s.weights().design_effect(), 7 / 3)

    def test_BalanceWeightsDF_trim(self):
        s = Sample.from_frame(
            pd.DataFrame({"w": np.random.uniform(0, 1, 10000), "id": range(0, 10000)}),
            id_column="id",
            weight_column="w",
        )
        s.weights().trim(percentile=(0, 0.11), keep_sum_of_weights=False)
        print(s.weights().df)
        print(max(s.weights().df.iloc[:, 0]))
        self.assertTrue(max(s.weights().df.iloc[:, 0]) < 0.9)


class TestBalanceDF__BalanceDF_child_from_linked_samples(
    balance.testutil.BalanceTestCase
):
    def test__BalanceDF_child_from_linked_samples_keys(self):
        self.assertEqual(
            list(s1.covars()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.covars()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.covars()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

        self.assertEqual(
            list(s1.weights()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.weights()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.weights()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

        self.assertEqual(
            list(s1.outcomes()._BalanceDF_child_from_linked_samples().keys()), ["self"]
        )
        self.assertEqual(
            list(s3.outcomes()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target"],
        )
        self.assertEqual(
            list(s3_null.outcomes()._BalanceDF_child_from_linked_samples().keys()),
            ["self", "target", "unadjusted"],
        )

    def test__BalanceDF_child_from_linked_samples_values(self):
        # We can get a calss using .__class__
        self.assertEqual(s1.covars().__class__, balance.balancedf_class.BalanceCovarsDF)

        # We get a different number of classes based on the number of linked items:
        the_dict = s1.covars()._BalanceDF_child_from_linked_samples()
        exp = [balance.balancedf_class.BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3.covars()._BalanceDF_child_from_linked_samples()
        exp = 2 * [balance.balancedf_class.BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
        exp = 3 * [balance.balancedf_class.BalanceCovarsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        # This also works for things other than BalanceCovarsDF:
        the_dict = s3_null.weights()._BalanceDF_child_from_linked_samples()
        exp = 3 * [balance.balancedf_class.BalanceWeightsDF]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        # Notice that with something like outcomes, we might get a None in return!
        the_dict = s3_null.outcomes()._BalanceDF_child_from_linked_samples()
        exp = [
            balance.balancedf_class.BalanceOutcomesDF,
            type(None),
            balance.balancedf_class.BalanceOutcomesDF,
        ]
        self.assertEqual([v.__class__ for (k, v) in the_dict.items()], exp)

        # Verify DataFrame values makes sense:
        # for covars
        the_dict = s3_null.covars()._BalanceDF_child_from_linked_samples()
        exp = [
            {
                "a": {0: 1, 1: 2, 2: 3, 3: 1},
                "b": {0: -42, 1: 8, 2: 2, 3: -42},
                "c": {0: "x", 1: "y", 2: "z", 3: "v"},
            },
            {
                "a": {0: 1, 1: 2, 2: 3},
                "b": {0: 4, 1: 6, 2: 8},
                "c": {0: "x", 1: "y", 2: "z"},
            },
            {
                "a": {0: 1, 1: 2, 2: 3, 3: 1},
                "b": {0: -42, 1: 8, 2: 2, 3: -42},
                "c": {0: "x", 1: "y", 2: "z", 3: "v"},
            },
        ]
        self.assertEqual([v.df.to_dict() for (k, v) in the_dict.items()], exp)

        # for outcomes
        the_dict = s3_null.outcomes()._BalanceDF_child_from_linked_samples()
        exp = [{"o": {0: 7, 1: 8, 2: 9, 3: 10}}, {"o": {0: 7, 1: 8, 2: 9, 3: 10}}]
        # need to exclude None v:
        self.assertEqual(
            [v.df.to_dict() for (k, v) in the_dict.items() if v is not None], exp
        )

        # for weights
        the_dict = s3_null.weights()._BalanceDF_child_from_linked_samples()
        exp = [
            {"w": {0: 0.5, 1: 2.0, 2: 1.0, 3: 1.0}},
            {"w": {0: 0.5, 1: 1.0, 2: 2.0}},
            {"w": {0: 0.5, 1: 2.0, 2: 1.0, 3: 1.0}},
        ]
        self.assertEqual([v.df.to_dict() for (k, v) in the_dict.items()], exp)


class TestBalanceDF__call_on_linked(balance.testutil.BalanceTestCase):
    def test_BalanceDF__call_on_linked(self):
        self.assertEqual(
            s1.weights()._call_on_linked("mean").values[0][0], (0.5 + 2 + 1 + 1) / 4
        )

        self.assertEqual(
            s1.weights()._call_on_linked("mean"),
            s3.weights()._call_on_linked("mean", exclude="target"),
        )

        self.assertEqual(
            # it's tricky to compare nan values, so using fillna
            s3.covars()._call_on_linked("mean").fillna(0).round(3).to_dict(),
            {
                "a": {"self": 1.889, "target": 2.429},
                "b": {"self": -10.0, "target": 6.857},
                "c[v]": {"self": 0.222, "target": 0},
                "c[x]": {"self": 0.111, "target": 0.143},
                "c[y]": {"self": 0.444, "target": 0.286},
                "c[z]": {"self": 0.222, "target": 0.571},
            },
        )

        self.assertEqual(
            # it's tricky to compare nan values, so using fillna
            # checking also on std, and on a larger object (with both self, target and unadjusted)
            s3_null.covars()._call_on_linked("std").fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.886, "target": 0.964, "unadjusted": 0.886},
                "b": {"self": 27.355, "target": 1.927, "unadjusted": 27.355},
                "c[v]": {"self": 0.5, "target": 0.0, "unadjusted": 0.5},
                "c[x]": {"self": 0.378, "target": 0.463, "unadjusted": 0.378},
                "c[y]": {"self": 0.598, "target": 0.598, "unadjusted": 0.598},
                "c[z]": {"self": 0.5, "target": 0.655, "unadjusted": 0.5},
            },
        )

        # verify exclude works:
        self.assertEqual(
            s3.covars()._call_on_linked("mean", exclude=("self")).round(3).to_dict(),
            {
                "a": {"target": 2.429},
                "b": {"target": 6.857},
                "c[x]": {"target": 0.143},
                "c[y]": {"target": 0.286},
                "c[z]": {"target": 0.571},
            },
        )
        self.assertEqual(
            s3.covars()._call_on_linked("mean", exclude=("target")).round(3).to_dict(),
            {
                "a": {"self": 1.889},
                "b": {"self": -10.0},
                "c[v]": {"self": 0.222},
                "c[x]": {"self": 0.111},
                "c[y]": {"self": 0.444},
                "c[z]": {"self": 0.222},
            },
        )

        # Verify we can also access df (i.e.: an attribute)
        self.assertEqual(
            s3.covars()._call_on_linked("df").round(3).to_dict(),
            {
                "a": {"self": 1, "target": 3},
                "b": {"self": -42, "target": 8},
                "c": {"self": "v", "target": "z"},
            },
        )


class TestBalanceDF__descriptive_stats(balance.testutil.BalanceTestCase):
    def test_BalanceDF__descriptive_stats(self):
        self.assertEqual(
            s1.weights()._descriptive_stats("mean", weighted=False).values[0][0], 1.125
        )
        self.assertAlmostEqual(
            s1.weights()._descriptive_stats("std", weighted=False).values[0][0],
            0.62915286,
        )
        # Not that you would ever really want the weighted weights
        self.assertEqual(
            s1.weights()._descriptive_stats("mean", weighted=True).values[0][0], 1.125
        )
        self.assertAlmostEqual(
            s1.weights()._descriptive_stats("std", weighted=True).values[0][0],
            0.62915286,
        )

        self.assertAlmostEqual(
            s1.covars()._descriptive_stats("mean", weighted=True)["a"][0], 1.88888888
        )

        # Test numeric_only and weighted flags
        r = s1.covars()._descriptive_stats("mean", weighted=False, numeric_only=True)
        e = pd.DataFrame({"a": [(1 + 2 + 3 + 1) / 4], "b": [(-42 + 8 + 2 - 42) / 4]})
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=False, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 + 2 + 3 + 1) / 4],
                "b": [(-42 + 8 + 2 - 42) / 4],
                "c[v]": [0.25],
                "c[x]": [0.25],
                "c[y]": [0.25],
                "c[z]": [0.25],
            }
        )
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=True)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 * 0.5 + 2 * 2 + 3 + 1) / 4.5],
                "b": [(-42 * 0.5 + 8 * 2 + 2 - 42) / 4.5],
            }
        )
        self.assertEqual(r, e)

        r = (
            s1.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "a": [(1 * 0.5 + 2 * 2 + 3 + 1) / 4.5],
                "b": [(-42 * 0.5 + 8 * 2 + 2 - 42) / 4.5],
                "c[v]": [1 / 4.5],
                "c[x]": [0.5 / 4.5],
                "c[y]": [2 / 4.5],
                "c[z]": [1 / 4.5],
            }
        )
        self.assertEqual(r, e)

        # Test with missing values and weights
        s_ds = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "b": (-42, 8, 2, np.nan),
                    "c": ("a", "b", "c", "a"),
                    "id": (1, 2, 3, 4),
                    "w": (np.nan, 2, 1, 1),  # np.nan makes it to a float64 dtype
                }
            ),
            id_column="id",
            weight_column="w",
        )

        r = (
            s_ds.covars()
            ._descriptive_stats("mean", weighted=True, numeric_only=False)
            .sort_index(axis=1)
        )
        e = pd.DataFrame(
            {
                "_is_na_b[T.True]": [(1 * 1) / (2 + 1 + 1)],
                "a": [(2 * 2 + 3 + 1) / (2 + 1 + 1)],
                "b": [(8 * 2 + 2 * 1) / (2 + 1 + 1)],
                "c[a]": [(1 * 1) / (2 + 1 + 1)],
                "c[b]": [(1 * 2) / (2 + 1 + 1)],
                "c[c]": [(1 * 1) / (2 + 1 + 1)],
            }
        )
        self.assertEqual(r, e)

    def test_Balance_df_summary_stats_numeric_only(self):
        #  Test that the asmd, std, and mean methods pass the `numeric_only`
        #  argument to _descriptive_stats

        # Default is numeric_only=False
        e_all = pd.Index(("a", "b", "c[x]", "c[y]", "c[z]", "c[v]"))
        e_numeric_only = pd.Index(("a", "b"))
        self.assertEqual(s1.covars().mean().columns, e_all, lazy=True)
        self.assertEqual(s1.covars().mean(numeric_only=True).columns, e_numeric_only)
        self.assertEqual(s1.covars().mean(numeric_only=False).columns, e_all, lazy=True)

        self.assertEqual(s1.covars().std().columns, e_all, lazy=True)
        self.assertEqual(s1.covars().std(numeric_only=True).columns, e_numeric_only)
        self.assertEqual(s1.covars().std(numeric_only=False).columns, e_all, lazy=True)


class TestBalanceDF_mean(balance.testutil.BalanceTestCase):
    def test_BalanceDF_mean(self):
        self.assertEqual(
            s1.weights().mean(),
            pd.DataFrame({"w": [1.125], "source": "self"}).set_index("source"),
        )

        self.assertEqual(
            s3_null.covars().mean().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 1.889, "target": 2.429, "unadjusted": 1.889},
                "b": {"self": -10.0, "target": 6.857, "unadjusted": -10.0},
                "c[v]": {"self": 0.222, "target": 0.0, "unadjusted": 0.222},
                "c[x]": {"self": 0.111, "target": 0.143, "unadjusted": 0.111},
                "c[y]": {"self": 0.444, "target": 0.286, "unadjusted": 0.444},
                "c[z]": {"self": 0.222, "target": 0.571, "unadjusted": 0.222},
            },
        )


class TestBalanceDF_std(balance.testutil.BalanceTestCase):
    def test_BalanceDF_std(self):
        self.assertEqual(
            s1.weights().std(),
            pd.DataFrame({"w": [0.6291529], "source": "self"}).set_index("source"),
        )

        self.assertEqual(
            s3_null.covars().std().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.886, "target": 0.964, "unadjusted": 0.886},
                "b": {"self": 27.355, "target": 1.927, "unadjusted": 27.355},
                "c[v]": {"self": 0.5, "target": 0.0, "unadjusted": 0.5},
                "c[x]": {"self": 0.378, "target": 0.463, "unadjusted": 0.378},
                "c[y]": {"self": 0.598, "target": 0.598, "unadjusted": 0.598},
                "c[z]": {"self": 0.5, "target": 0.655, "unadjusted": 0.5},
            },
        )


class TestBalanceDF_asmd(balance.testutil.BalanceTestCase):
    def test_BalanceDF_asmd(self):
        # Test with BalanceDF
        from balance.balancedf_class import BalanceDF

        r = BalanceDF._asmd_BalanceDF(
            Sample.from_frame(
                pd.DataFrame(
                    {"id": (1, 2), "a": (1, 2), "b": (-1, 12), "weight": (1, 2)}
                )
            ).covars(),
            Sample.from_frame(
                pd.DataFrame(
                    {"id": (1, 2), "a": (3, 4), "b": (0, 42), "weight": (1, 2)}
                )
            ).covars(),
        ).sort_index()
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        with self.assertRaisesRegex(ValueError, "has no target set"):
            s1.weights().asmd()

        with self.assertRaisesRegex(ValueError, "has no target set"):
            s3_null.outcomes().asmd()

        self.assertEqual(
            s3.covars().asmd().loc["self"],
            pd.Series(
                (
                    0.560055,
                    8.746742,
                    np.nan,
                    0.068579,
                    0.265606,
                    0.533422,
                    (
                        0.560055
                        + 8.746742
                        + 0.068579 * 0.25
                        + 0.265606 * 0.25
                        + 0.533422 * 0.25
                    )
                    / 3,
                ),
                index=("a", "b", "c[v]", "c[x]", "c[y]", "c[z]", "mean(asmd)"),
                name="self",
            ),
        )

        self.assertEqual(
            s3_null.covars().asmd().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.56, "unadjusted": 0.56, "unadjusted - self": 0.0},
                "b": {"self": 8.747, "unadjusted": 8.747, "unadjusted - self": 0.0},
                "c[v]": {"self": 0.0, "unadjusted": 0.0, "unadjusted - self": 0.0},
                "c[x]": {"self": 0.069, "unadjusted": 0.069, "unadjusted - self": 0.0},
                "c[y]": {"self": 0.266, "unadjusted": 0.266, "unadjusted - self": 0.0},
                "c[z]": {"self": 0.533, "unadjusted": 0.533, "unadjusted - self": 0.0},
                "mean(asmd)": {
                    "self": 3.175,
                    "unadjusted": 3.175,
                    "unadjusted - self": 0.0,
                },
            },
        )
        # also check that on_linked_samples = False works:
        self.assertEqual(
            s3_null.covars().asmd(on_linked_samples=False).fillna(0).round(3).to_dict(),
            {
                "a": {"covars": 0.56},
                "b": {"covars": 8.747},
                "c[v]": {"covars": 0.0},
                "c[x]": {"covars": 0.069},
                "c[y]": {"covars": 0.266},
                "c[z]": {"covars": 0.533},
                "mean(asmd)": {"covars": 3.175},
            },
        )

        self.assertEqual(
            s3_null_madeup_weights.covars().asmd().fillna(0).round(3).to_dict(),
            {
                "a": {"self": 0.296, "unadjusted": 0.56, "unadjusted - self": 0.264},
                "b": {"self": 8.154, "unadjusted": 8.747, "unadjusted - self": 0.593},
                "c[v]": {"self": 0.0, "unadjusted": 0.0, "unadjusted - self": 0.0},
                "c[x]": {"self": 0.0, "unadjusted": 0.069, "unadjusted - self": 0.069},
                "c[y]": {"self": 0.0, "unadjusted": 0.266, "unadjusted - self": 0.266},
                "c[z]": {
                    "self": 0.218,
                    "unadjusted": 0.533,
                    "unadjusted - self": 0.315,
                },
                "mean(asmd)": {
                    "self": 2.835,
                    "unadjusted": 3.175,
                    "unadjusted - self": 0.34,
                },
            },
        )

    def test_BalanceDF_asmd_improvement(self):
        with self.assertRaisesRegex(
            ValueError, "has no unadjusted set or unadjusted has no covars"
        ):
            s3.covars().asmd_improvement()

        s3_unadjusted = deepcopy(s3)
        s3_unadjusted.set_weights((1, 1, 1, 1))
        s3_2 = s3.set_unadjusted(s3_unadjusted)
        self.assertEqual(s3_2.covars().asmd_improvement(), 0.3224900694460681)

        s1_with_unadjusted = deepcopy(s1)
        s1_with_unadjusted = s1.set_unadjusted(s3_unadjusted)
        with self.assertRaisesRegex(
            ValueError, "has no target set or target has no covars"
        ):
            s1_with_unadjusted.covars().asmd_improvement()

        self.assertEqual(s3_null.covars().asmd_improvement().round(3), 0)
        self.assertEqual(
            s3_null_madeup_weights.covars().asmd_improvement().round(3), 0.107
        )

        asmd_df = s3_null_madeup_weights.covars().asmd()
        exp = round(
            (asmd_df["mean(asmd)"][1] - asmd_df["mean(asmd)"][0])
            / asmd_df["mean(asmd)"][1],
            3,
        )
        self.assertEqual(exp, 0.107)
        self.assertEqual(
            s3_null_madeup_weights.covars().asmd_improvement().round(3), exp
        )

    def test_BalanceDF_asmd_aggregate_by_main_covar(self):
        # TODO: re-use this example across tests
        # TODO: bugfix - adjust failes with apply_transform when inputing a df with categorical column :(

        # Prepare dummy data
        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 3))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abc"[i] for i in range(0, 3)})
        # make 'a' a categorical column in d
        # d = d.assign(a=lambda x: pd.cut(x.a,[0,.25,.5,.75,1]))
        d["a"] = pd.cut(d["a"], [0, 0.25, 0.5, 0.75, 1]).astype(str)
        # make b "interesting" (so that the model would have something to do)
        d["b"] = np.sqrt(d["b"])
        s = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(1000, 3))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abc"[i] for i in range(0, 3)})
        # make 'a' a categorical column in d
        # d = d.assign(a=lambda x: pd.cut(x.a,[0,.25,.5,.75,1]))
        d["a"] = pd.cut(d["a"], [0, 0.25, 0.5, 0.75, 1]).astype(str)
        t = Sample.from_frame(d)

        st = s.set_target(t)

        # Fit IPW
        a = st.adjust()

        # Check ASMD
        tmp_asmd_default = a.covars().asmd()
        tmp_asmd_main_covar = a.covars().asmd(aggregate_by_main_covar=True)

        outcome_default = tmp_asmd_default.round(2).to_dict()
        outcome_main_covar = tmp_asmd_main_covar.round(2).to_dict()

        expected_default = {
            "a[(0.0, 0.25]]": {
                "self": 0.04,
                "unadjusted": 0.09,
                "unadjusted - self": 0.05,
            },
            "a[(0.25, 0.5]]": {
                "self": 0.0,
                "unadjusted": 0.06,
                "unadjusted - self": 0.06,
            },
            "a[(0.5, 0.75]]": {
                "self": 0.0,
                "unadjusted": 0.01,
                "unadjusted - self": 0.01,
            },
            "a[(0.75, 1.0]]": {
                "self": 0.04,
                "unadjusted": 0.02,
                "unadjusted - self": -0.02,
            },
            "c": {"self": 0.02, "unadjusted": 0.03, "unadjusted - self": 0.01},
            "b": {"self": 0.14, "unadjusted": 0.6, "unadjusted - self": 0.46},
            "mean(asmd)": {"self": 0.06, "unadjusted": 0.23, "unadjusted - self": 0.17},
        }
        expected_main_covar = {
            "a": {"self": 0.02, "unadjusted": 0.05, "unadjusted - self": 0.02},
            "b": {"self": 0.14, "unadjusted": 0.6, "unadjusted - self": 0.46},
            "c": {"self": 0.02, "unadjusted": 0.03, "unadjusted - self": 0.01},
            "mean(asmd)": {"self": 0.06, "unadjusted": 0.23, "unadjusted - self": 0.17},
        }

        self.assertEqual(outcome_default, expected_default)
        self.assertEqual(outcome_main_covar, expected_main_covar)


class TestBalanceDF_to_download(balance.testutil.BalanceTestCase):
    def test_BalanceDF_to_download(self):
        r = s1.covars().to_download()
        self.assertIsInstance(r, IPython.display.FileLink)


class TestBalanceDF_to_csv(balance.testutil.BalanceTestCase):
    def test_BalanceDF_to_csv(self):
        with tempfile.NamedTemporaryFile() as tf:
            s1.weights().to_csv(path_or_buf=tf.name)
            r = tf.read()
            e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
            self.assertEqual(r, e)

    def test_BalanceDF_to_csv_first_default_argument_is_path(self):
        with tempfile.NamedTemporaryFile() as tf:
            s1.weights().to_csv(tf.name)
            r = tf.read()
            e = b"id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n"
            self.assertEqual(r, e)

    def test_BalanceDF_to_csv_output_with_no_path(self):
        with tempfile.NamedTemporaryFile():
            out = s1.weights().to_csv()
        self.assertEqual(out, "id,w\n1,0.5\n2,2.0\n3,1.0\n4,1.0\n")

    def test_BalanceDF_to_csv_output_with_path(self):
        with tempfile.NamedTemporaryFile() as tf:
            out = s1.weights().to_csv(path_or_buf=tf.name)
        self.assertEqual(out, None)


class TestBalanceDF__df_with_ids(balance.testutil.BalanceTestCase):
    def test_BalanceDF__df_with_ids(self):
        # Test it has an id column:
        self.assertTrue("id" in s1.weights()._df_with_ids().columns)
        self.assertTrue("id" in s1.covars()._df_with_ids().columns)
        self.assertTrue("id" in s_o.outcomes()._df_with_ids().columns)

        # Test it has df columns:
        self.assertTrue("w" in s1.weights()._df_with_ids().columns)
        self.assertEqual(
            ["id", "a", "b", "c"], s1.covars()._df_with_ids().columns.tolist()
        )
        self.assertEqual((4, 4), s1.covars()._df_with_ids().shape)


class TestBalanceDF_summary(balance.testutil.BalanceTestCase):
    def testBalanceDF_summary(self):
        covar_means = pd.DataFrame(
            {
                "a": [(0.5 * 1 + 2 * 2 + 3 * 1 + 1 * 1) / (0.5 + 2 + 1 + 1)],
                "b": [(-42 * 0.5 + 8 * 2 + 2 * 1 + -42 * 1) / (0.5 + 2 + 1 + 1)],
                "c[x]": [(1 * 0.5) / (0.5 + 2 + 1 + 1)],
                "c[y]": [(1 * 2) / (0.5 + 2 + 1 + 1)],
                "c[z]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
                "c[v]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
            }
        )
        self.assertEqual(
            s1.covars().summary(),
            covar_means.assign(source="self").set_index("source"),
            lazy=True,
        )

        s3_2 = s1.adjust(s2, method="null")
        e = pd.concat((covar_means,) * 2, sort=True).assign(
            source=("self", "unadjusted")
        )
        e = pd.concat(
            (
                e,
                pd.DataFrame(
                    {
                        "a": [(1 * 0.5 + 2 * 1 + 3 * 2) / (0.5 + 1 + 2)],
                        "b": [(4 * 0.5 + 6 * 1 + 8 * 2) / (0.5 + 1 + 2)],
                        "c[x]": [(1 * 0.5) / (0.5 + 1 + 2)],
                        "c[y]": [(1 * 1) / (0.5 + 1 + 2)],
                        "c[z]": [(1 * 2) / (0.5 + 1 + 2)],
                        "source": "target",
                    }
                ),
            ),
            sort=True,
        )
        e = e.set_index("source")
        self.assertEqual(s3_2.covars().summary().sort_index(axis=1), e)

        self.assertEqual(
            s3_2.covars().summary(on_linked_samples=False), covar_means, lazy=True
        )


class TestBalanceDF__str__(balance.testutil.BalanceTestCase):
    def testBalanceDF__str__(self):
        self.assertTrue(s1.outcomes().df.__str__() in s1.outcomes().__str__())

    def test_BalanceOutcomesDF___str__(self):
        # NOTE how the type is float even though the original input was integer.
        self.assertTrue(
            pd.DataFrame({"o": (7.0, 8.0, 9.0, 10.0)}).__str__() in o.__str__()
        )


class TestBalanceDF__repr__(balance.testutil.BalanceTestCase):
    def test_BalanceWeightsDF___repr__(self):
        repr = w.__repr__()
        self.assertTrue("weights from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceCovarsDF___repr__(self):
        repr = c.__repr__()
        self.assertTrue("covars from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)

    def test_BalanceOutcomesDF___repr__(self):
        repr = o.__repr__()
        self.assertTrue("outcomes from" in repr)
        self.assertTrue(object.__repr__(s1) in repr)


class TestBalanceDF(balance.testutil.BalanceTestCase):
    def testBalanceDF_model_matrix(self):
        self.assertEqual(
            s1.covars().model_matrix().sort_index(axis=1).columns.values,
            ("a", "b", "c[v]", "c[x]", "c[y]", "c[z]"),
        )
        self.assertEqual(
            s1.covars().model_matrix().to_dict(),
            {
                "a": {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.0},
                "b": {0: -42.0, 1: 8.0, 2: 2.0, 3: -42.0},
                "c[v]": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "c[x]": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "c[y]": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "c[z]": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            },
        )

    def test_check_if_not_BalanceDF(self):
        from balance.balancedf_class import BalanceDF

        with self.assertRaisesRegex(ValueError, "number must be balancedf_class"):
            BalanceDF._check_if_not_BalanceDF(5, "number")

        self.assertTrue(BalanceDF._check_if_not_BalanceDF(s3.covars()) is None)
