# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile

from copy import deepcopy

import balance.testutil
import IPython.display

import numpy as np
import pandas as pd

from balance.sample_class import Sample

# TODO: move s3 and other definitions of sample outside of classes (from example from TestSamplePrivateAPI)

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

s4 = Sample.from_frame(
    pd.DataFrame(
        {"a": (0, None, 2), "b": (0, None, 2), "c": ("a", "b", "c"), "id": (1, 2, 3)}
    ),
    outcome_columns=("b", "c"),
)


class TestSample(
    balance.testutil.BalanceTestCase,
):
    def test_constructor_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            s1 = Sample()
            print(s1)

    def test_Sample__str__(self):
        self.assertTrue("4 observations x 3 variables" in s1.__str__())
        self.assertTrue("outcome_columns: o" in s1.__str__())
        self.assertTrue("weight_column: w" in s1.__str__())

        self.assertTrue("outcome_columns: None" in s2.__str__())
        self.assertTrue("weight_column: w" in s2.__str__())

        s3 = s1.set_target(s2)
        self.assertTrue("Sample object with target set" in s3.__str__())
        self.assertTrue("target:" in s3.__str__())
        self.assertTrue("3 common variables" in s3.__str__())

        s4 = s3.adjust(method="null")
        self.assertTrue(
            "Adjusted balance Sample object with target set using" in s4.__str__()
        )

    def test_Sample__str__multiple_outcomes(self):
        s1 = Sample.from_frame(
            pd.DataFrame(
                {"a": (1, 2, 3), "b": (4, 6, 8), "id": (1, 2, 3), "w": (0.5, 1, 2)}
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=("a", "b"),
        )
        self.assertTrue("outcome_columns: a,b" in s1.__str__())

    def test_Sample_from_frame(self):
        # test id_column
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertWarnsRegexp(
            "Guessed id column name id for the data", Sample.from_frame, df
        )
        # TODO: add tests for the two other warnings:
        # self.assertWarnsRegexp("Casting id column to string", Sample.from_frame, df)
        # self.assertWarnsRegexp("No weights passed, setting all weights to 1", Sample.from_frame, df)
        # Using the above would fail since the warnings are sent sequentially and using self.assertWarnsRegexp
        # only catches the first warning.
        self.assertEqual(
            Sample.from_frame(df).id_column, pd.Series((1, 2), name="id").astype(str)
        )
        df = pd.DataFrame({"b": (1, 2), "a": (1, 2)})
        self.assertEqual(
            Sample.from_frame(df, id_column="b").id_column,
            pd.Series((1, 2), name="b").astype(str),
        )
        with self.assertRaisesRegex(
            ValueError,
            "Cannot guess id column name for this DataFrame. Please provide a value in id_column",
        ):
            Sample.from_frame(df)
        with self.assertRaisesRegex(
            ValueError,
            "Dataframe does not have column*",
        ):
            Sample.from_frame(df, id_column="c")

        # test exception if values in id are null
        df = pd.DataFrame({"id": (1, None), "a": (1, 2)})
        with self.assertRaisesRegex(
            ValueError,
            "Null values are not allowed in the id_column",
        ):
            Sample.from_frame(df)

        # test check_id_uniqueness argument
        df = pd.DataFrame({"id": (1, 2, 2)})
        with self.assertRaisesRegex(
            ValueError,
            "Values in the id_column must be unique",
        ):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 2)})
        self.assertEqual(
            Sample.from_frame(df, check_id_uniqueness=False).df.id,
            pd.Series(("1", "2", "2"), name="id"),
        )

        # Test weights_column
        df = pd.DataFrame({"id": (1, 2), "weight": (1, 2)})
        self.assertWarnsRegexp("Guessing weight", Sample.from_frame, df)
        # NOTE how weight that was integer was changed into floats.
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((1.0, 2.0), name="weight")
        )

        df = pd.DataFrame({"id": (1, 2)})
        self.assertWarnsRegexp("No weights passed", Sample.from_frame, df)
        # NOTE that the default weights are integers, not floats
        # TODO: decide if it's o.k. to keep the default weights be 1s, or change the default to floats
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((1, 1), name="weight")
        )

        # Assert error for null values
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (None, 3, 1.1)})
        with self.assertRaisesRegex(ValueError, "Null values are not allowed"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (None, None, None)})
        with self.assertRaisesRegex(ValueError, "Null values are not allowed"):
            Sample.from_frame(df)

        # Assert error for non-numeric weights
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, "b", 2.1)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, "5", 2.1)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, 2.1, True)})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            Sample.from_frame(df)

        # Assert error for negative values
        df = pd.DataFrame({"id": (1, 2, 3), "weight": (1, -2, 2.1)})
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            Sample.from_frame(df)

        # Zero is legitimate
        df = pd.DataFrame({"id": (1, 2), "weight": (0, 2), "weight_column": "weight"})
        self.assertEqual(
            Sample.from_frame(df).weight_column, pd.Series((0.0, 2.0), name="weight")
        )

        # Test type conversion
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.a.dtype.type, np.int64)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float64)
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int32)
        self.assertEqual(df.a.dtype.type, np.int32)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float32)
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int16)
        self.assertEqual(df.a.dtype.type, np.int16)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float16)
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)}, dtype=np.int8)
        self.assertEqual(df.a.dtype.type, np.int8)
        self.assertEqual(Sample.from_frame(df).df.a.dtype.type, np.float16)
        # TODO: add tests for other types of conversions

        # Test use_deepcopy
        # after we invoked Sample.from_frame with use_deepcopy=False, we expect the dtype of id to be np.object_
        #   in BOTH the df inside sample and the ORIGINAL df:
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.id.dtype.type, np.int64)
        self.assertEqual(
            Sample.from_frame(df, use_deepcopy=False).df.id.dtype.type, np.object_
        )
        self.assertEqual(df.id.dtype.type, np.object_)
        # after we invoked Sample.from_frame with use_deepcopy=True (default), we expect the dtype of id to be object_
        #   in the df inside sample, but id in the ORIGINAL df to remain int64:
        df = pd.DataFrame({"id": (1, 2), "a": (1, 2)})
        self.assertEqual(df.id.dtype.type, np.int64)
        self.assertEqual(Sample.from_frame(df).df.id.dtype.type, np.object_)
        self.assertEqual(df.id.dtype.type, np.int64)

    def test_Sample_adjust(self):
        from balance.weighting_methods.adjust_null import adjust_null

        s3 = s1.set_target(s2).adjust(method="null")
        self.assertTrue(s3.is_adjusted())

        s3 = s1.set_target(s2).adjust(method=adjust_null)
        self.assertTrue(s3.is_adjusted())

        # test exception
        with self.assertRaisesRegex(
            ValueError,
            "Method should be one of existing weighting methods",
        ):
            s1.set_target(s2).adjust(method=None)


class TestSample_base_and_adjust_methods(
    balance.testutil.BalanceTestCase,
):
    def test_Sample_df(self):
        # NOTE how integers were changed into floats.
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0, 1.0),
                "b": (-42.0, 8.0, 2.0, -42.0),
                "o": (7.0, 8.0, 9.0, 10.0),
                "c": ("x", "y", "z", "v"),
                "id": ("1", "2", "3", "4"),
                "w": (0.5, 2, 1, 1),
            },
            columns=("id", "a", "b", "c", "o", "w"),
        )
        # Verify we get the expected output:
        self.assertEqual(s1.df, e)

        # Check that @property works:
        self.assertTrue(isinstance(Sample.df, property))
        self.assertEqual(Sample.df.fget(s1), s1.df)
        # We can no longer call .df() as if it was a function:
        with self.assertRaisesRegex(TypeError, "'DataFrame' object is not callable"):
            s1.df()

        # NOTE how integers were changed into floats.
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0),
                "b": (4.0, 6.0, 8.0),
                "id": ("1", "2", "3"),
                "w": (0.5, 1, 2),
                "c": ("x", "y", "z"),
            },
            columns=("id", "a", "b", "c", "w"),
        )
        self.assertEqual(s2.df, e)

    def test_Sample_outcomes(self):
        # NOTE how integers were changed into floats.
        # TODO: consider removing this test, since it's already tested in test_balancedf.py
        e = pd.DataFrame(
            {
                "o": (7.0, 8.0, 9.0, 10.0),
            },
            columns=["o"],
        )
        self.assertEqual(s1.outcomes().df, e)

    def test_Sample_weights(self):
        e = pd.DataFrame(
            {
                "w": (0.5, 2, 1, 1),
            },
            columns=["w"],
        )
        self.assertEqual(s1.weights().df, e)

    # TODO: consider removing this test, since it's already tested in test_balancedf.py
    def test_Sample_covars(self):
        # NOTE how integers were changed into floats.
        e = pd.DataFrame(
            {
                "a": (1.0, 2.0, 3.0, 1.0),
                "b": (-42.0, 8.0, 2.0, -42.0),
                "c": ("x", "y", "z", "v"),
            }
        )
        self.assertEqual(s1.covars().df, e)

    def test_Sample_model(self):
        np.random.seed(112358)
        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        s = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(10000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        a = s.adjust(t, max_de=None, method="null")
        m = a.model()
        self.assertEqual(m["method"], "null_adjustment")
        a = s.adjust(t, max_de=None)
        m = a.model()
        self.assertEqual(m["method"], "ipw")
        # Just test the structure of ipw output
        self.assertTrue("perf" in m.keys())
        self.assertTrue("fit" in m.keys())
        self.assertTrue("coefs" in m["perf"].keys())

    def test_Sample_model_matrix(self):
        #  Main tests for model_matrix are in test_util.py
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (0, 1, 2),
                    "b": (0, None, 2),
                    "c": ("a", "b", "a"),
                    "id": (1, 2, 3),
                }
            ),
            id_column="id",
        )
        e = pd.DataFrame(
            {
                "a": (0.0, 1.0, 2.0),
                "b": (0.0, 0.0, 2.0),
                "_is_na_b[T.True]": (0.0, 1.0, 0.0),
                "c[a]": (1.0, 0.0, 1.0),
                "c[b]": (0.0, 1.0, 0.0),
            }
        )
        r = s.model_matrix()
        self.assertEqual(r, e, lazy=True)

    def test_Sample_set_weights(self):
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": (1, 2, 3, 1),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                }
            ),
            id_column="id",
            weight_column="w",
        )
        # NOTE that if using set_weights with integers, the weights remain integers
        s.set_weights(pd.Series([1, 2, 3, 4]))
        self.assertEqual(s.weight_column, pd.Series([1, 2, 3, 4], name="w"))
        s.set_weights(pd.Series([1, 2, 3, 4], index=(1, 2, 5, 6)))
        self.assertEqual(
            s.weight_column, pd.Series([np.nan, 1.0, 2.0, np.nan], name="w")
        )
        # test warning
        self.assertWarnsRegexp(
            """Note that not all Sample units will be assigned weights""",
            Sample.set_weights,
            s,
            pd.Series([1, 2, 3, 4], index=(1, 2, 5, 6)),
        )
        # no warning
        self.assertNotWarns(
            Sample.set_weights,
            s,
            pd.Series([1, 2, 3, 4], index=(0, 1, 2, 3)),
        )

    def test_Sample_set_unadjusted(self):
        s5 = s1.set_unadjusted(s2)
        self.assertTrue(s5._links["unadjusted"] is s2)
        # test exceptions when there is no a second sample
        with self.assertRaisesRegex(
            TypeError,
            "set_unadjusted must be called with second_sample argument of type Sample",
        ):
            s1.set_unadjusted("Not a Sample object")

    def test_Sample_is_adjusted(self):
        self.assertFalse(s1.is_adjusted())
        # TODO: move definitions of s3 outside of function
        s3 = s1.set_target(s2)
        self.assertFalse(s3.is_adjusted())
        # TODO: move definitions of s3 outside of function
        s3_adjusted = s3.adjust(method="null")
        self.assertTrue(s3_adjusted.is_adjusted())

    def test_Sample_set_target(self):
        s5 = s1.set_target(s2)
        self.assertTrue(s5._links["target"] is s2)
        # test exceptions when the provided object is not a second sample
        with self.assertRaisesRegex(
            ValueError,
            "A target, a Sample object, must be specified",
        ):
            s1.set_target("Not a Sample object")

    def test_Sample_has_target(self):
        self.assertFalse(s1.has_target())
        self.assertTrue(s1.set_target(s2).has_target())


class TestSample_metrics_methods(
    balance.testutil.BalanceTestCase,
):
    def test_Sample_covar_means(self):
        # TODO: take definition of s3_null outside of function
        s3_null = s1.adjust(s2, method="null")
        e = pd.DataFrame(
            {
                "a": [(0.5 * 1 + 2 * 2 + 3 * 1 + 1 * 1) / (0.5 + 2 + 1 + 1)],
                "b": [(-42 * 0.5 + 8 * 2 + 2 * 1 + -42 * 1) / (0.5 + 2 + 1 + 1)],
                "c[x]": [(1 * 0.5) / (0.5 + 2 + 1 + 1)],
                "c[y]": [(1 * 2) / (0.5 + 2 + 1 + 1)],
                "c[z]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
                "c[v]": [(1 * 1) / (0.5 + 2 + 1 + 1)],
            }
        ).transpose()
        e = pd.concat((e,) * 2, axis=1, sort=True)
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
                        "c[v]": np.nan,
                    }
                ).transpose(),
            ),
            axis=1,
            sort=True,
        )
        e.columns = pd.Series(("unadjusted", "adjusted", "target"), name="source")
        self.assertEqual(s3_null.covar_means(), e)

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1.covar_means()

    def test_Sample_design_effect(self):
        self.assertEqual(s1.design_effect().round(3), 1.235)
        self.assertEqual(s4.design_effect(), 1.0)

    def test_Sample_design_effect_prop(self):
        # TODO: take definition of s3_null outside of function
        s3_null = s1.adjust(s2, method="null")
        self.assertEqual(s3_null.design_effect_prop(), 0.0)

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1.design_effect_prop()

    def test_Sample_outcome_sd_prop(self):
        # TODO: take definition of s3_null outside of function
        s3_null = s1.adjust(s2, method="null")
        self.assertEqual(s3_null.outcome_sd_prop(), pd.Series((0.0), index=["o"]))
        # test with two outcomes
        # TODO: take definition of s1_two_outcomes outside of function
        s1_two_outcomes = Sample.from_frame(
            pd.DataFrame(
                {
                    "o1": (7, 8, 9, 10),
                    "o2": (7, 8, 9, 11),
                    "c": ("x", "y", "z", "y"),
                    "id": (1, 2, 3, 4),
                    "w": (0.5, 2, 1, 1),
                },
            ),
            id_column="id",
            weight_column="w",
            outcome_columns=["o1", "o2"],
        )
        s3_null = s1_two_outcomes.adjust(s2, method="null")
        self.assertEqual(
            s3_null.outcome_sd_prop(), pd.Series((0.0, 0.0), index=["o1", "o2"])
        )

        # test exceptions when there is no adjusted
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have outcome columns specified",
        ):
            s2.adjust(s2, method="null").outcome_sd_prop()

    def test_outcome_variance_ratio(self):
        from balance.stats_and_plots.weighted_stats import weighted_var

        # Testing it also works with outcomes
        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(1000, 11))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghijk"[i] for i in range(0, 11)})
        d["b"] = np.sqrt(d["b"])

        a_with_outcome = Sample.from_frame(d, outcome_columns=["k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, max_de=1.5)

        # verifying this does what we expect it does:
        self.assertEqual(
            round(a_with_outcome_adjusted.outcome_variance_ratio()[0], 5),
            round(
                (
                    weighted_var(
                        a_with_outcome_adjusted.outcomes().df,
                        a_with_outcome_adjusted.weights().df["weight"],
                    )
                    / weighted_var(
                        a_with_outcome_adjusted._links["unadjusted"].outcomes().df,
                        a_with_outcome_adjusted._links["unadjusted"]
                        .weights()
                        .df["weight"],
                    )
                )[0],
                5,
            ),
        )

        self.assertEqual(
            round(a_with_outcome_adjusted.outcome_variance_ratio()[0], 5), 0.97724
        )

        # two outcomes, with no adjustment (var ratio should be 1)
        a_with_outcome = Sample.from_frame(d, outcome_columns=["j", "k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, method="null")
        self.assertEqual(
            a_with_outcome_adjusted.outcome_variance_ratio(),
            pd.Series([1.0, 1.0], index=["j", "k"]),
        )

    def test_Sample_weights_summary(self):
        self.assertEqual(
            s1.weights().summary().round(2).to_dict(),
            {
                "var": {
                    0: "design_effect",
                    1: "effective_sample_proportion",
                    2: "effective_sample_size",
                    3: "sum",
                    4: "describe_count",
                    5: "describe_mean",
                    6: "describe_std",
                    7: "describe_min",
                    8: "describe_25%",
                    9: "describe_50%",
                    10: "describe_75%",
                    11: "describe_max",
                    12: "prop(w < 0.1)",
                    13: "prop(w < 0.2)",
                    14: "prop(w < 0.333)",
                    15: "prop(w < 0.5)",
                    16: "prop(w < 1)",
                    17: "prop(w >= 1)",
                    18: "prop(w >= 2)",
                    19: "prop(w >= 3)",
                    20: "prop(w >= 5)",
                    21: "prop(w >= 10)",
                    22: "nonparametric_skew",
                    23: "weighted_median_breakdown_point",
                },
                "val": {
                    0: 1.23,
                    1: 0.81,
                    2: 3.24,
                    3: 4.5,
                    4: 4.0,
                    5: 1.0,
                    6: 0.56,
                    7: 0.44,
                    8: 0.78,
                    9: 0.89,
                    10: 1.11,
                    11: 1.78,
                    12: 0.0,
                    13: 0.0,
                    14: 0.0,
                    15: 0.25,
                    16: 0.75,
                    17: 0.25,
                    18: 0.0,
                    19: 0.0,
                    20: 0.0,
                    21: 0.0,
                    22: 0.2,
                    23: 0.25,
                },
            },
        )

    def test_Sample_summary(self):
        s1_summ = s1.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD" not in s1_summ)

        s3 = s1.set_target(s2)
        s3_summ = s3.summary()
        self.assertTrue("Model performance" not in s1_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("design effect" not in s3_summ)

        s3 = s3.set_unadjusted(s1)
        s3_summ = s3.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("Covar ASMD (6 variables)" in s3_summ)
        self.assertTrue("->" in s3_summ)
        self.assertTrue("design effect" in s3_summ)

        s3 = s1.set_target(s2).adjust(method="null")
        s3_summ = s3.summary()
        self.assertTrue("Covar ASMD reduction: 0.0%" in s3_summ)
        self.assertTrue("design effect" in s3_summ)

    def test_Sample_invalid_outcomes(self):
        with self.assertRaisesRegex(
            ValueError,
            r"outcome columns \['o'\] not in df columns \['a', 'id', 'weight'\]",
        ):
            Sample.from_frame(
                pd.DataFrame({"a": (1, 2, 3, 1), "id": (1, 2, 3, 4)}),
                outcome_columns="o",
            )

    def test_Sample_diagnostics(self):
        import numpy as np
        import pandas as pd

        # TODO (p2): move the objects created here outside of this function and possible make this simpler.
        from balance.sample_class import Sample

        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        s = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        a = s.adjust(t)
        a_diagnostics = a.diagnostics()
        # print(a_diagnostics)

        self.assertEqual(a_diagnostics.shape, (203, 3))
        self.assertEqual(a_diagnostics.columns.to_list(), ["metric", "val", "var"])
        self.assertEqual(
            a_diagnostics[a_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["ipw"]),
        )

        output = a_diagnostics.groupby("metric").size().to_dict()
        expected = {
            "adjustment_failure": 1,
            "adjustment_method": 1,
            "covar_asmd_adjusted": 11,
            "covar_asmd_improvement": 11,
            "covar_asmd_unadjusted": 11,
            "covar_main_asmd_adjusted": 11,
            "covar_main_asmd_improvement": 11,
            "covar_main_asmd_unadjusted": 11,
            "ipw_model_glance": 2,
            "ipw_multi_class": 1,
            "ipw_penalty": 1,
            "ipw_solver": 1,
            "model_coef": 92,
            "model_glance": 10,
            "size": 4,
            "weights_diagnostics": 24,
        }
        self.assertEqual(output, expected)

        b = s.adjust(t, method="cbps")
        b_diagnostics = b.diagnostics()
        # print(b_diagnostics)

        self.assertEqual(b_diagnostics.shape, (196, 3))
        self.assertEqual(b_diagnostics.columns.to_list(), ["metric", "val", "var"])
        self.assertEqual(
            b_diagnostics[b_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["cbps"]),
        )

        output = b_diagnostics.groupby("metric").size().to_dict()
        expected = {
            "adjustment_failure": 1,
            "balance_optimize_result": 2,
            "gmm_optimize_result_bal_init": 2,
            "gmm_optimize_result_glm_init": 2,
            "rescale_initial_result": 2,
            "beta_optimal": 92,
            "covar_asmd_adjusted": 11,
            "covar_asmd_improvement": 11,
            "covar_asmd_unadjusted": 11,
            "covar_main_asmd_adjusted": 11,
            "covar_main_asmd_improvement": 11,
            "covar_main_asmd_unadjusted": 11,
            "adjustment_method": 1,
            "size": 4,
            "weights_diagnostics": 24,
        }
        self.assertEqual(output, expected)

        c = s.adjust(t, method="null")
        c_diagnostics = c.diagnostics()
        self.assertEqual(c_diagnostics.shape, (96, 3))
        self.assertEqual(c_diagnostics.columns.to_list(), ["metric", "val", "var"])
        self.assertEqual(
            c_diagnostics[c_diagnostics["metric"] == "adjustment_method"]["var"].values,
            np.array(["null_adjustment"]),
        )

    def test_Sample_keep_only_some_rows_columns(self):
        import numpy as np
        import pandas as pd

        # TODO (p2): move the objects created here outside of this function and possible make this simpler.
        from balance.sample_class import Sample

        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        d["b"] = np.sqrt(d["b"])
        s = Sample.from_frame(d)

        d = pd.DataFrame(np.random.rand(1000, 10))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        t = Sample.from_frame(d)

        a = s.adjust(t, max_de=1.5)

        self.maxDiff = None
        # if both rows_to_keep = None, columns_to_keep = None - then keep_only_some_rows_columns returns the same object
        self.assertTrue(
            a is a.keep_only_some_rows_columns(rows_to_keep=None, columns_to_keep=None)
        )

        # let's remove some columns and rows:
        a2 = a.keep_only_some_rows_columns(
            rows_to_keep=None, columns_to_keep=["b", "c"]
        )

        # Making sure asmd works
        output_orig = a.covars().asmd().round(2).to_dict()
        output_new = a2.covars().asmd().round(2).to_dict()
        expected_orig = {
            "j": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.02},
            "i": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "h": {"self": 0.02, "unadjusted": 0.09, "unadjusted - self": 0.06},
            "g": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "f": {"self": 0.01, "unadjusted": 0.03, "unadjusted - self": 0.01},
            "e": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.01},
            "d": {"self": 0.03, "unadjusted": 0.12, "unadjusted - self": 0.09},
            "c": {"self": 0.04, "unadjusted": 0.05, "unadjusted - self": 0.02},
            "b": {"self": 0.18, "unadjusted": 0.55, "unadjusted - self": 0.37},
            "a": {"self": 0.01, "unadjusted": 0.0, "unadjusted - self": -0.0},
            "mean(asmd)": {"self": 0.03, "unadjusted": 0.09, "unadjusted - self": 0.06},
        }
        expected_new = {
            "c": {"self": 0.04, "unadjusted": 0.05, "unadjusted - self": 0.02},
            "b": {"self": 0.18, "unadjusted": 0.55, "unadjusted - self": 0.37},
            "mean(asmd)": {"self": 0.11, "unadjusted": 0.3, "unadjusted - self": 0.20},
        }

        self.assertEqual(output_orig, expected_orig)
        self.assertEqual(output_new, expected_new)

        # Making sure diagnostics works, and also seeing we got change in
        # what we expect
        a_diag = a.diagnostics()
        a2_diag = a2.diagnostics()
        a_diag_tbl = a_diag.groupby("metric").size().to_dict()
        a2_diag_tbl = a2_diag.groupby("metric").size().to_dict()

        # The mean weight should be 1 (since we normalize for the sum of weights to be equal to len(weights))
        ss = a_diag.eval("(metric == 'weights_diagnostics') & (var == 'describe_mean')")
        self.assertEqual(round(float(a_diag[ss].val), 4), 1.000)

        # keeping only columns 'b' and 'c' leads to have only 3 asmd instead of 11:
        self.assertEqual(a_diag_tbl["covar_main_asmd_adjusted"], 11)
        self.assertEqual(a2_diag_tbl["covar_main_asmd_adjusted"], 3)

        # now we get only 2 covars counted instead of 10:
        ss_condition = "(metric == 'size') & (var == 'sample_covars')"
        ss = a_diag.eval(ss_condition)
        ss2 = a2_diag.eval(ss_condition)
        self.assertEqual(int(a_diag[ss].val), 10)
        self.assertEqual(int(a2_diag[ss2].val), 2)

        # And the mean asmd is different
        ss_condition = "(metric == 'covar_main_asmd_adjusted') & (var == 'mean(asmd)')"
        ss = a_diag.eval(ss_condition)
        ss2 = a2_diag.eval(ss_condition)
        self.assertEqual(round(float(a_diag[ss].val), 4), 0.0329)
        self.assertEqual(round(float(a2_diag[ss2].val), 3), 0.109)

        # Also checking filtering using rows_to_keep:
        a3 = a.keep_only_some_rows_columns(
            rows_to_keep="a>0.5", columns_to_keep=["b", "c"]
        )

        # Making sure the weights are of the same length as the df
        self.assertEqual(a3.df.shape[0], a3.weights().df.shape[0])

        # Making sure asmd works - we can see it's different then for a2
        output_new = a3.covars().asmd().round(2).to_dict()
        expected_new = {
            "c": {"self": 0.05, "unadjusted": 0.07, "unadjusted - self": 0.02},
            "b": {"self": 0.25, "unadjusted": 0.61, "unadjusted - self": 0.36},
            "mean(asmd)": {"self": 0.15, "unadjusted": 0.34, "unadjusted - self": 0.19},
        }
        self.assertEqual(output_new, expected_new)

        a3_diag = a3.diagnostics()
        a3_diag_tbl = a3_diag.groupby("metric").size().to_dict()

        # The structure of the diagnostics table is the same with and without
        # the filtering. So when comparing a3 to a2, we should get the same results:
        # i.e.: a2_diag_tbl == a3_diag_tbl # True
        self.assertEqual(a2_diag_tbl, a3_diag_tbl)
        # However, the number of samples is different!
        ss_condition = "(metric == 'size') & (var == 'sample_obs')"
        self.assertEqual(int(a_diag[a_diag.eval(ss_condition)].val), 1000)
        self.assertEqual(int(a2_diag[a2_diag.eval(ss_condition)].val), 1000)
        self.assertEqual(int(a3_diag[a3_diag.eval(ss_condition)].val), 508)
        # also in the target
        ss_condition = "(metric == 'size') & (var == 'target_obs')"
        self.assertEqual(int(a_diag[a_diag.eval(ss_condition)].val), 1000)
        self.assertEqual(int(a2_diag[a2_diag.eval(ss_condition)].val), 1000)
        self.assertEqual(
            int(a3_diag[a3_diag.eval(ss_condition)].val), 516
        )  # since a<0.5 is different for target!
        # also in the weights
        ss = a_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'describe_count')"
        )
        self.assertEqual(int(a_diag[ss].val), 1000)
        ss = a3_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'describe_count')"
        )
        self.assertEqual(int(a3_diag[ss].val), 508)
        # Notice also that the calculated values from the weights are different
        ss = a_diag.eval("(metric == 'weights_diagnostics') & (var == 'design_effect')")
        self.assertEqual(round(float(a_diag[ss].val), 4), 1.4679)
        ss = a3_diag.eval(
            "(metric == 'weights_diagnostics') & (var == 'design_effect')"
        )
        self.assertEqual(round(float(a3_diag[ss].val), 4), 1.4325)

        # Testing it also works with outcomes
        np.random.seed(112358)

        d = pd.DataFrame(np.random.rand(1000, 11))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghijk"[i] for i in range(0, 11)})
        d["b"] = np.sqrt(d["b"])
        a_with_outcome = Sample.from_frame(d, outcome_columns=["k"])
        a_with_outcome_adjusted = a_with_outcome.adjust(t, max_de=1.5)

        # We can also filter using an outcome variable (although this would NOT filter on target)
        # a proper logger warning is issued
        self.assertEqual(
            a_with_outcome_adjusted.keep_only_some_rows_columns(
                rows_to_keep="k>0.5"
            ).df.shape,
            (481, 13),
        )

        a_with_outcome_adjusted2 = a_with_outcome_adjusted.keep_only_some_rows_columns(
            rows_to_keep="b>0.5", columns_to_keep=["b", "c"]
        )

        self.assertEqual(
            a_with_outcome_adjusted2.outcomes().mean().round(3).to_dict(),
            {"k": {"self": 0.492, "unadjusted": 0.494}},
        )

        # TODO (p2): possibly add checks for columns_to_keep = None while doing something with rows_to_keep

        # test if only some columns exists
        self.assertWarnsRegexp(
            "Note that not all columns_to_keep are in Sample",
            s1.keep_only_some_rows_columns,
            columns_to_keep=["g", "a"],
        )
        self.assertEqual(
            s1.keep_only_some_rows_columns(
                columns_to_keep=["g", "a"]
            )._df.columns.tolist(),
            ["a"],
        )


class TestSample_to_download(balance.testutil.BalanceTestCase):
    def test_Sample_to_download(self):
        r = s1.to_download()
        self.assertIsInstance(r, IPython.display.FileLink)

    def test_Sample_to_csv(self):
        with tempfile.NamedTemporaryFile() as tf:
            s1.to_csv(path_or_buf=tf.name)
            r = tf.read()
            e = (
                b"id,a,b,c,o,w\n1,1,-42,x,7,0.5\n"
                b"2,2,8,y,8,2.0\n3,3,2,z,9,1.0\n4,1,-42,v,10,1.0\n"
            )
            self.assertTrue(r, e)


class TestSamplePrivateAPI(balance.testutil.BalanceTestCase):
    def test__links(self):
        self.assertTrue(len(s1._links.keys()) == 0)

        s3 = s1.set_target(s2)
        self.assertTrue(s3._links["target"] is s2)
        self.assertTrue(s3.has_target())

        s3_adjusted = s3.adjust(method="null")
        self.assertTrue(s3_adjusted._links["target"] is s2)
        self.assertTrue(s3_adjusted._links["unadjusted"] is s3)
        self.assertTrue(s3_adjusted.has_target())

    def test__special_columns_names(self):
        self.assertEqual(
            sorted(s4._special_columns_names()), ["b", "c", "id", "weight"]
        )

    # NOTE how integers were changed into floats.
    def test__special_columns(self):
        # NOTE how integers in weight were changed into floats.
        self.assertEqual(
            s4._special_columns(),
            pd.DataFrame(
                {
                    "id": ("1", "2", "3"),
                    # Weights were filled automatically to be integers of 1s:
                    "weight": (1, 1, 1),
                    "b": (0.0, None, 2.0),
                    "c": ("a", "b", "c"),
                }
            ),
        )

    def test__covar_columns_names(self):
        self.assertEqual(sorted(s1._covar_columns_names()), ["a", "b", "c"])

    def test__covar_columns(self):
        # NOTE how integers were changed into floats.
        self.assertEqual(
            s1._covar_columns(),
            pd.DataFrame(
                {
                    "a": (1.0, 2.0, 3.0, 1.0),
                    "b": (-42.0, 8.0, 2.0, -42.0),
                    "c": ("x", "y", "z", "v"),
                }
            ),
        )

    def test_Sample__check_if_adjusted(self):
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s1._check_if_adjusted()
        # TODO: move definitions of s3 outside of function
        s3 = s1.set_target(s2)
        with self.assertRaisesRegex(
            ValueError,
            "This is not an adjusted Sample. Use sample.adjust to adjust the sample to target",
        ):
            s3._check_if_adjusted()
        # TODO: move definitions of s3 outside of function
        s3_adjusted = s3.adjust(method="null")
        self.assertTrue(
            s3_adjusted._check_if_adjusted() is None
        )  # Does not raise an error

    def test_Sample__no_target_error(self):
        # test exception when the is no target
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have a target set. Use sample.set_target to add target",
        ):
            s1._no_target_error()
        s3 = s1.set_target(s2)
        s3._no_target_error()  # Should not raise an error

    def test_Sample__check_outcomes_exists(self):
        with self.assertRaisesRegex(
            ValueError,
            "This Sample does not have outcome columns specified",
        ):
            s2._check_outcomes_exists()
        self.assertTrue(s1._check_outcomes_exists() is None)  # Does not raise an error


class TestSample_NA_behavior(balance.testutil.BalanceTestCase):
    def test_can_handle_various_NAs(self):
        # Testing if we can handle NA values from pandas

        def get_sample_to_adjust(df, standardize_types=True):
            s1 = Sample.from_frame(df, standardize_types=standardize_types)
            s2 = deepcopy(s1)
            s2.set_weights(np.ones(100))
            return s1.set_target(s2)

        np.random.seed(123)
        df = pd.DataFrame(
            {
                "a": np.random.uniform(size=100),
                "c": np.random.choice(
                    ["a", "b", "c", "d"],
                    size=100,
                    replace=True,
                    p=[0.01, 0.04, 0.5, 0.45],
                ),
                "id": range(100),
                "weight": np.random.uniform(size=100) + 0.5,
            }
        )

        # This works fine
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # This should raise a TypeError:
        with self.assertRaisesRegex(
            TypeError,
            "boolean value of NA is ambiguous",
        ):
            smpl_to_adj = get_sample_to_adjust(df)
            # smpl_to_adj._df.iloc[0, 0] = pd.NA
            smpl_to_adj._df.iloc[0, 1] = pd.NA
            # This will raise the error:
            smpl_to_adj.adjust(method="ipw")

        # This should raise a TypeError:
        with self.assertRaisesRegex(
            Exception,
            "series must be numeric",
        ):
            # Adding NA to a numeric column turns it into an object.
            # This raises an error in util.quantize
            smpl_to_adj = get_sample_to_adjust(df)
            smpl_to_adj._df.iloc[0, 0] = pd.NA
            # smpl_to_adj._df.iloc[0, 1] = pd.NA
            # This will raise the error:
            smpl_to_adj.adjust(method="ipw")

        # This works fine
        df.iloc[0, 0] = np.nan
        df.iloc[0, 1] = np.nan
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # This also works fine (thanks to standardize_types=True)
        df.iloc[0, 0] = pd.NA
        df.iloc[0, 1] = pd.NA
        smpl_to_adj = get_sample_to_adjust(df)
        self.assertIsInstance(smpl_to_adj.adjust(method="ipw"), Sample)

        # Turning standardize_types to False should raise a TypeError (since we have pd.NA):
        with self.assertRaisesRegex(
            TypeError,
            "boolean value of NA is ambiguous",
        ):
            # df.iloc[0, 0] = pd.NA
            df.iloc[0, 1] = pd.NA
            smpl_to_adj = get_sample_to_adjust(df, standardize_types=False)
            smpl_to_adj.adjust(method="ipw")
