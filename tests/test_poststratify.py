# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import balance.testutil

import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.weighting_methods.poststratify import poststratify


class Testpoststratify(
    balance.testutil.BalanceTestCase,
):
    def test_poststratify(self):
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weights"]
        self.assertEqual(result, t_weights.astype("float64"))

        # same example when dataframe of elements are all related to weights of one
        s = pd.DataFrame(
            {
                "a": (0, 0, 0, 0, 1, 1, 0, 1),
                "c": ("a", "a", "a", "a", "a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weights"]
        self.assertEqual(result, pd.Series((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 8.0)))

        # same example with normalized weights
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([1 / 2, 1 / 4, 1 / 8, 1 / 8])
        result = poststratify(
            sample_df=s, sample_weights=s_weights, target_df=t, target_weights=t_weights
        )["weights"]
        self.assertEqual(result, t_weights.astype("float64"))

        # test through adjustment
        # TODO: test the previous example through adjustment as well
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
        result = sample.adjust(target, method="poststratify", transformations=None)
        expected = pd.Series(
            (
                (2 / 1.5 * 0.5),
                (0.5 / 2 * 2),
                (1 / 1 * 1),
                (2 / 1.5 * 1),
            )
        )
        self.assertEqual(expected, result.weights().df.iloc[:, 0].values)

    def test_poststratify_variables_arg(self):
        s = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 2, 3])
        t = s
        t_weights = pd.Series([4, 2, 2, 8])
        result = poststratify(
            sample_df=s,
            sample_weights=s_weights,
            target_df=t,
            target_weights=t_weights,
            variables=["a"],
        )["weights"]
        self.assertEqual(result, pd.Series([4.0, 4.0, 2.0, 6.0]))

    def test_poststratify_transformations(self):
        # for numeric
        size = 10000
        s = pd.DataFrame({"age": np.random.uniform(0, 1, size)})
        tmp = int(size * 0.2)
        t = pd.DataFrame(
            {
                "age": np.concatenate(
                    (
                        np.random.uniform(0, 0.4, tmp),
                        np.random.uniform(0.4, 1, size - tmp),
                    )
                )
            }
        )
        result = poststratify(
            sample_df=s,
            sample_weights=pd.Series([1] * size),
            target_df=t,
            target_weights=pd.Series([1] * size),
        )["weights"]

        # age>0.4 has 4 times as many people than age <0.4 in the target
        # Check that the weights come out as 0.2 and 0.8
        eps = 0.05
        self.assertTrue(abs(result[s.age < 0.4].sum() / size - 0.2) < eps)
        self.assertTrue(abs(result[s.age >= 0.4].sum() / size - 0.8) < eps)

        # for strings
        size = 10000
        s = pd.DataFrame(
            {"x": np.random.choice(("a", "b", "c"), size, p=(0.95, 0.035, 0.015))}
        )
        t = pd.DataFrame(
            {"x": np.random.choice(("a", "b", "c"), size, p=(0.95, 0.015, 0.035))}
        )
        result = poststratify(
            sample_df=s,
            sample_weights=pd.Series([1] * size),
            target_df=t,
            target_weights=pd.Series([1] * size),
        )["weights"]

        # Output weights should ignore the difference between values 'b' and 'c'
        # since these are combined in default transformations (into '_lumped_other').
        # Hence their frequency would be as in sample
        eps = 0.05
        self.assertTrue(abs(result[s.x == "a"].sum() / size - 0.95) < eps)
        self.assertTrue(abs(result[s.x == "b"].sum() / size - 0.035) < eps)
        self.assertTrue(abs(result[s.x == "c"].sum() / size - 0.015) < eps)

    def test_poststratify_exceptions(self):
        # column with name weight
        s = pd.DataFrame(
            {
                "weight": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        s_weights = pd.Series([4, 2, 1, 1])
        t = pd.DataFrame(
            {
                "a": (0, 1, 0, 1),
                "c": ("a", "a", "b", "b"),
            },
        )
        t_weights = pd.Series([4, 2, 2, 8])
        with self.assertRaisesRegex(
            ValueError,
            "weight can't be a name of a column in sample or target when applying poststratify",
        ):
            poststratify(s, s_weights, t, t_weights)
        with self.assertRaisesRegex(
            ValueError,
            "weight can't be a name of a column in sample or target when applying poststratify",
        ):
            poststratify(t, t_weights, s, s_weights)

        # not all sample cells are in target
        s = pd.DataFrame(
            {
                "a": ("x", "y"),
                "b": (0, 1),
            },
        )
        s_weights = pd.Series([1] * 2)
        t = pd.DataFrame(
            {
                "a": ("x", "x", "y"),
                "b": (0, 1, 0),
            },
        )
        t_weights = pd.Series([2] * 3)
        with self.assertRaisesRegex(
            ValueError, "all combinations of cells in sample_df must be in target_df"
        ):
            poststratify(s, s_weights, t, t_weights)
