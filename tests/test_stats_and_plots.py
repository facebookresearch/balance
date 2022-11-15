# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import balance.testutil

import numpy as np
import pandas as pd


class TestBalance_weights_stats(
    balance.testutil.BalanceTestCase,
):
    def test__check_weights_are_valid(self):
        from balance.stats_and_plots.weights_stats import _check_weights_are_valid

        w = [np.inf, 1, 2, 1.0, 0]
        self.assertEqual(_check_weights_are_valid(w), None)
        self.assertEqual(_check_weights_are_valid(np.array(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.Series(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.DataFrame(w)), None)
        self.assertEqual(_check_weights_are_valid(pd.DataFrame({"a": w})), None)
        self.assertEqual(
            _check_weights_are_valid(pd.DataFrame({"a": w, "b": [str(x) for x in w]})),
            None,
        )  # checking only the first column

        with self.assertRaisesRegex(TypeError, "weights \\(w\\) must be a number*"):
            _check_weights_are_valid([str(x) for x in w])

        with self.assertRaisesRegex(TypeError, "weights \\(w\\) must be a number*"):
            w = ["a", "b"]
            _check_weights_are_valid(w)

        with self.assertRaisesRegex(
            ValueError, "weights \\(w\\) must all be non-negative values."
        ):
            w = [-1, 0, 1]
            _check_weights_are_valid(w)

    def test_design_effect(self):
        from balance.stats_and_plots.weights_stats import design_effect

        self.assertEqual(design_effect(pd.Series((1, 1, 1, 1))), 1)
        self.assertEqual(
            design_effect(pd.Series((0, 1, 2, 3))),
            1.555_555_555_555_555_6,
        )
        self.assertEqual(type(design_effect(pd.Series((0, 1, 2, 3)))), np.float64)

    def test_nonparametric_skew(self):
        from balance.stats_and_plots.weights_stats import nonparametric_skew

        self.assertEqual(nonparametric_skew(pd.Series((1, 1, 1, 1))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1, 2, 3, 4))), 0)
        self.assertEqual(nonparametric_skew(pd.Series((1, 1, 1, 2))), 0.5)

    def test_prop_above_and_below(self):
        from balance.stats_and_plots.weights_stats import prop_above_and_below

        self.assertEqual(
            prop_above_and_below(pd.Series((1, 1, 1, 1))).astype(int).to_list(),
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        )
        self.assertEqual(
            prop_above_and_below(pd.Series((1, 2, 3, 4))).to_list(),
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(
            prop_above_and_below(
                pd.Series((1, 2, 3, 4)), below=(0.1, 0.5), above=(2, 3)
            ).to_list(),
            [0.0, 0.25, 0.0, 0.0],
        )
        self.assertEqual(
            prop_above_and_below(
                pd.Series((1, 2, 3, 4)), below=(0.1, 0.5), above=(2, 3)
            ).index.to_list(),
            ["prop(w < 0.1)", "prop(w < 0.5)", "prop(w >= 2)", "prop(w >= 3)"],
        )

        self.assertEqual(
            prop_above_and_below(pd.Series((1, 2, 3, 4)), above=None, below=None), None
        )

        # test return_as_series = False
        self.assertEqual(
            {
                k: v.to_list()
                for k, v in prop_above_and_below(
                    pd.Series((1, 2, 3, 4)), return_as_series=False
                ).items()
            },
            {"below": [0.0, 0.0, 0.0, 0.25, 0.5], "above": [0.5, 0.0, 0.0, 0.0, 0.0]},
        )

    def test_weighted_median_breakdown_point(self):
        import pandas as pd
        from balance.stats_and_plots.weights_stats import (
            weighted_median_breakdown_point,
        )

        self.assertEqual(weighted_median_breakdown_point(pd.Series((1, 1, 1, 1))), 0.5)
        self.assertEqual(weighted_median_breakdown_point(pd.Series((2, 2, 2, 2))), 0.5)
        self.assertEqual(
            weighted_median_breakdown_point(pd.Series((1, 1, 1, 10))), 0.25
        )
        self.assertEqual(
            weighted_median_breakdown_point(pd.Series((1, 1, 1, 1, 10))), 0.2
        )


class TestBalance_weighted_stats(
    balance.testutil.BalanceTestCase,
):
    def test__prepare_weighted_stat_args(self):
        from balance.stats_and_plots.weighted_stats import _prepare_weighted_stat_args

        v, w = [-1, 0, 1, np.inf], [np.inf, 1, 2, 1.0]
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)

        v, w = pd.Series([-1, 0, 1, np.inf]), pd.Series([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # check the values are the same:
        self.assertEqual(v.to_list(), v2[0].to_list())
        self.assertEqual(w.to_list(), w2.to_list())
        # Do we have any inf or nan?
        self.assertTrue(any(np.isinf(v2[0])))
        self.assertTrue(any(np.isinf(w2)))
        self.assertTrue(not any(np.isnan(v2[0])))
        self.assertTrue(not any(np.isnan(w2)))

        # Checking how it works when inf_rm = True
        v2, w2 = _prepare_weighted_stat_args(v, w, True)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # check the values are the NOT same (because the Inf was turned to nan):
        self.assertNotEqual(v.to_list(), v2[0].to_list())
        self.assertNotEqual(w.to_list(), w2.to_list())
        # Do we have any inf or nan?
        self.assertTrue(not any(np.isinf(v2[0])))
        self.assertTrue(not any(np.isinf(w2)))
        self.assertTrue(any(np.isnan(v2[0])))
        self.assertTrue(any(np.isnan(w2)))

        # Check that it catches wrong input types
        with self.assertRaises(TypeError):
            v, w = pd.Series([1, 2]), np.matrix([1, 2]).T
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(TypeError):
            v, w = pd.Series([1, 2]), (1, 2)
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(TypeError):
            v, w = (1, 2), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(ValueError):
            v, w = pd.Series([1, 2, 3]), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)
        with self.assertRaises(TypeError):
            v, w = pd.Series(["a", "b"]), pd.Series([1, 2])
            v2, w2 = _prepare_weighted_stat_args(v, w)

        # check other input types for v and w
        # np.array
        v, w = np.array([-1, 0, 1, np.inf]), np.array([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # pd.DataFrame
        v, w = pd.DataFrame([-1, 0, 1, np.inf]), np.array([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        # np.matrix
        v, w = np.matrix([-1, 0, 1, np.inf]).T, np.array([np.inf, 1, 2, 1.0])
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        # assert the new types:
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)

        # Dealing with w=None
        v, w = pd.Series([-1, 0, 1, np.inf]), None
        v2, w2 = _prepare_weighted_stat_args(v, w, False)
        self.assertEqual(type(v2), pd.DataFrame)
        self.assertEqual(type(w2), pd.Series)
        self.assertEqual(w2, pd.Series([1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(len(w2) == 4)
        # Verify defaults:
        self.assertEqual(
            _prepare_weighted_stat_args(v, None, False)[0],
            _prepare_weighted_stat_args(v)[0],
        )

        with self.assertRaises(ValueError):
            v, w = pd.Series([-1, 0, 1, np.inf]), pd.Series([np.inf, 1, -2, 1.0])
            v2, w2 = _prepare_weighted_stat_args(v, w)

    def test_weighted_mean(self):
        from balance.stats_and_plots.weighted_stats import weighted_mean

        # No weights
        self.assertEqual(weighted_mean(pd.Series([-1, 0, 1, 2])), pd.Series(0.5))
        self.assertEqual(weighted_mean(pd.Series([-1, None, 1, 2])), pd.Series(0.5))

        # No weights, with one inf value
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.inf, 1, 2])),
            pd.Series(np.inf),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.inf, 1, 2]), inf_rm=True),
            pd.Series(0.5),
        )

        # Inf value in weights
        self.assertTrue(
            all(
                np.isnan(
                    weighted_mean(
                        pd.Series([-1, 2, 1, 2]), w=pd.Series([1, np.inf, 1, 1])
                    )
                )
            )
        )
        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2]), w=pd.Series([1, np.inf, 1, 1]), inf_rm=True
            ),
            pd.Series(2 / 3),
        )

        # With weights
        self.assertEqual(
            weighted_mean(pd.Series([-1, 2, 1, 2]), w=pd.Series([1, 2, 3, 4])),
            pd.Series(1.4),
        )

        self.assertEqual(
            weighted_mean(pd.Series([-1, 2, 1, 2]), w=pd.Series([1, None, 1, 1])),
            pd.Series(2 / 3),
        )

        # Notice that while None values are ignored, their weights will still be used in the denominator
        # Hence, None in values acts like 0
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 0, 1, 1])),
            pd.Series(2 / 3),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, np.nan, 1, 2]), w=pd.Series([1, 0, 1, 1])),
            pd.Series(2 / 3),
        )
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 1, 1, 1])),
            pd.Series(1 / 2),
        )
        # None in the values is just like 0
        self.assertEqual(
            weighted_mean(pd.Series([-1, None, 1, 2]), w=pd.Series([1, 1, 1, 1])),
            weighted_mean(pd.Series([-1, 0, 1, 2]), w=pd.Series([1, 1, 1, 1])),
        )

        # pd.DataFrame v
        d = pd.DataFrame([(-1, 2, 1, 2), (1, 2, 3, 4)]).transpose()
        pd.testing.assert_series_equal(weighted_mean(d), pd.Series((1, 2.5)))
        pd.testing.assert_series_equal(
            weighted_mean(d, w=pd.Series((1, 2, 3, 4))),
            pd.Series((1.4, 3)),
        )

        # np.matrix v
        d = np.matrix([(-1, 2, 1, 2), (1, 2, 3, 4)]).transpose()
        pd.testing.assert_series_equal(weighted_mean(d), pd.Series((1, 2.5)))
        pd.testing.assert_series_equal(
            weighted_mean(d, w=pd.Series((1, 2, 3, 4))),
            pd.Series((1.4, 3)),
        )

        #  Test input validation
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            weighted_mean(pd.Series(["a", "b"]))
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            weighted_mean(pd.DataFrame({"a": [1, 2], "b": ["a", "b"]}))

        with self.assertRaisesRegex(ValueError, "must have same number of rows"):
            weighted_mean(pd.Series([1, 2]), pd.Series([1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "must have same number of rows"):
            weighted_mean(pd.DataFrame({"a": [1, 2]}), pd.Series([1]))

        # Make sure that index is ignored
        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2], index=(0, 1, 2, 3)),
                w=pd.Series([1, 2, 3, 4], index=(5, 6, 7, 8)),
                inf_rm=True,
            ),
            pd.Series((-1 * 1 + 2 * 2 + 1 * 3 + 2 * 4) / (1 + 2 + 3 + 4)),
        )

        self.assertEqual(
            weighted_mean(
                pd.Series([-1, 2, 1, 2], index=(0, 1, 2, 3)),
                w=pd.Series([1, 2, 3, 4], index=(0, 6, 7, 8)),
                inf_rm=True,
            ),
            pd.Series((-1 * 1 + 2 * 2 + 1 * 3 + 2 * 4) / (1 + 2 + 3 + 4)),
        )

    def test_weighted_var(self):
        from balance.stats_and_plots.weighted_stats import weighted_var

        #  > var(c(1, 2, 3, 4))
        #  [1] 1.66667
        self.assertEqual(weighted_var(pd.Series((1, 2, 3, 4))), pd.Series(5 / 3))

        #  > SDMTools::wt.var(c(1, 2), c(1, 2))
        #  [1] 0.5
        self.assertEqual(
            weighted_var(pd.Series((1, 2)), pd.Series((1, 2))), pd.Series(0.5)
        )

    def test_weighted_sd(self):
        from balance.stats_and_plots.weighted_stats import weighted_sd

        #  > sd(c(1, 2, 3, 4))
        #  [1] 1.290994
        self.assertEqual(weighted_sd(pd.Series((1, 2, 3, 4))), pd.Series(1.290_994))

        #  > SDMTools::wt.sd(c(1, 2), c(1, 2))
        #  [1] 0.7071068
        self.assertEqual(
            weighted_sd(pd.Series((1, 2)), pd.Series((1, 2))),
            pd.Series(0.707_106_8),
        )

        x = [1, 2, 3, 4]
        x2 = pd.Series(x)
        manual_std = np.sqrt(np.sum((x2 - x2.mean()) ** 2) / (len(x) - 1))
        self.assertEqual(round(weighted_sd(x)[0], 5), round(manual_std, 5))

    def test_weighted_quantile(self):
        from balance.stats_and_plots.weighted_stats import weighted_quantile

        self.assertEqual(
            weighted_quantile(np.arange(1, 100, 1), 0.5).values,
            np.array(((50,),)),
        )

        # In R: reldist::wtd.quantile(c(1, 2, 3), q=c(0.5, 0.75), weight=c(1, 1, 2))
        self.assertEqual(
            weighted_quantile(np.array([1, 2, 3]), (0.5, 0.75)).values,
            np.array(((2,), (3,))),
        )

        self.assertEqual(
            weighted_quantile(np.array([1, 2, 3]), 0.5, np.array([1, 1, 2])).values,
            np.percentile([1, 2, 3, 3], 50),
        )

        # verify it indeed works with pd.DataFrame input
        # no weights
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
            ).values,
            np.array([[1.5, 1.0], [2.5, 1.0]]),
        )
        # with weights
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 1, 0, 0]),
            ).values,
            np.array([[1.0, 1.0], [1.5, 1.0]]),
        )
        self.assertEqual(
            weighted_quantile(
                pd.DataFrame([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 100, 1, 1]),
            ).values,
            np.array([[2.0, 1.0], [2.0, 1.0]]),
        )

        # verify it indeed works with np.matrix input
        # no weights
        self.assertEqual(
            weighted_quantile(
                np.matrix([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
            ).values,
            np.array([[1.5, 1.0], [2.5, 1.0]]),
        )
        # with weights
        self.assertEqual(
            weighted_quantile(
                np.matrix([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 1, 0, 0]),
            ).values,
            np.array([[1.0, 1.0], [1.5, 1.0]]),
        )
        self.assertEqual(
            weighted_quantile(
                np.matrix([[1, 2, 3, 4], [1, 1, 1, 1]]).transpose(),
                [0.25, 0.5],
                w=pd.Series([1, 100, 1, 1]),
            ).values,
            np.array([[2.0, 1.0], [2.0, 1.0]]),
        )

    def test_descriptive_stats(self):
        from balance.stats_and_plots.weighted_stats import (
            descriptive_stats,
            weighted_mean,
            weighted_sd,
        )
        from statsmodels.stats.weightstats import DescrStatsW

        x = pd.Series([-1, 0, 1, 2])
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), stat="mean"),
            weighted_mean(x).to_frame().T,
        )
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), stat="std"), weighted_sd(x).to_frame().T
        )

        # with weights
        x, w = pd.Series([-1, 2, 1, 2]), pd.Series([1, 2, 3, 4])
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), w, stat="mean"),
            weighted_mean(x, w).to_frame().T,
        )
        self.assertEqual(
            descriptive_stats(pd.DataFrame(x), w, stat="std"),
            weighted_sd(x, w).to_frame().T,
        )
        # show that with/without weights gives different results
        self.assertNotEqual(
            descriptive_stats(pd.DataFrame(x), stat="mean").iloc[0, 0],
            descriptive_stats(pd.DataFrame(x), w, stat="mean").iloc[0, 0],
        )
        self.assertNotEqual(
            descriptive_stats(pd.DataFrame(x), stat="std").iloc[0, 0],
            descriptive_stats(pd.DataFrame(x), w, stat="std").iloc[0, 0],
        )
        # shows that descriptive_stats can calculate std_mean and that it's smaller than std (as expected.)
        self.assertTrue(
            descriptive_stats(pd.DataFrame(x), w, stat="std").iloc[0, 0]
            > descriptive_stats(pd.DataFrame(x), w, stat="std_mean").iloc[0, 0]
        )

        x = [1, 2, 3, 4]
        x2 = pd.Series(x)
        tmp_sd = np.sqrt(np.sum((x2 - x2.mean()) ** 2) / (len(x) - 1))
        tmp_se = tmp_sd / np.sqrt(len(x))
        self.assertEqual(
            round(descriptive_stats(pd.DataFrame(x), stat="std_mean").iloc[0, 0], 5),
            round(tmp_se, 5),
        )

        # verify DescrStatsW can deal with None weights
        self.assertEqual(
            DescrStatsW([1, 2, 3], weights=[1, 1, 1]).mean,
            DescrStatsW([1, 2, 3], weights=None).mean,
        )
        self.assertEqual(
            DescrStatsW([1, 2, 3], weights=None).mean,
            2.0,
        )


class TestBalance_weighted_comparisons_stats(
    balance.testutil.BalanceTestCase,
):
    def test_outcome_variance_ratio(self):
        from balance.stats_and_plots.weighted_comparisons_stats import (
            outcome_variance_ratio,
        )

        np.random.seed(876324)

        d = pd.DataFrame(np.random.rand(1000, 11))
        d["id"] = range(0, d.shape[0])
        d = d.rename(columns={i: "abcdefghijk"[i] for i in range(0, 11)})

        self.assertEqual(
            outcome_variance_ratio(d[["j", "k"]], d[["j", "k"]]),
            pd.Series([1.0, 1.0], index=["j", "k"]),
        )

    def test__weights_per_covars_names(self):
        # toy example
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _weights_per_covars_names,
        )

        asmd_df = pd.DataFrame(
            {
                "age": 0.5,
                "education[T.high_school]": 1,
                "education[T. bachelor]": 1,
                "education[T. masters]": 1,
                "education[T. phd]": 1,
            },
            index=("self",),
        )

        outcome = _weights_per_covars_names(asmd_df.columns.values.tolist()).to_dict()
        # We expect a df with 2 columns: weight and main_covar_names.
        expected = {
            "weight": {
                "age": 1.0,
                "education[T.high_school]": 0.25,
                "education[T. bachelor]": 0.25,
                "education[T. masters]": 0.25,
                "education[T. phd]": 0.25,
            },
            "main_covar_names": {
                "age": "age",
                "education[T.high_school]": "education",
                "education[T. bachelor]": "education",
                "education[T. masters]": "education",
                "education[T. phd]": "education",
            },
        }

        self.assertEqual(outcome, expected)

    def test_asmd(self):
        from balance.stats_and_plots.weighted_comparisons_stats import asmd

        with self.assertRaisesRegex(ValueError, "sample_df must be pd.DataFrame, is*"):
            # Using wild card since it will return:
            # "sample_df must be pd.DataFrame, is* <class 'pandas.core.series.Series'>"
            asmd(
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
            )

        with self.assertRaisesRegex(ValueError, "target_df must be pd.DataFrame, is*"):
            asmd(
                pd.DataFrame({"a": (0, 1, 2, 3)}),
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
                pd.Series((0, 1, 2, 3)),
            )

        # If column names are different, it will only calculate asmd for
        # the overlapping columns. The rest will be np.nan.
        # The mean(asmd) will be calculated while treating the nan values as 0s.
        self.assertEqual(
            np.isnan(
                asmd(
                    pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
                    pd.DataFrame({"a": (3, 4), "c": (5, 6)}),
                )
            ).tolist(),
            [False, True, True, False],
        )

        with self.assertRaisesRegex(ValueError, "std_type must be in*"):
            asmd(
                pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
                pd.DataFrame({"a": (3, 4), "c": (5, 6)}),
                std_type="magic variance type that doesn't exist",
            )

        # TODO: (p2) add comparison to the following numbers
        # Benchmark for the numbers:
        # ---------------------------
        #  Test data from R cobalt package:
        #  cobalt::bal.tab(
        #    data.frame(a=c(1, 2, 3, 4), b=c(-1, 12, 0, 42)),
        #    treat=c(1, 1, 0, 0),
        #    weights=c(1, 2, 1, 2),
        #    s.d.denom='control',
        #    method='weighting'
        #  )
        # Output:
        # Balance Measures
        #      Type Diff.Adj
        # a Contin.  -2.8284
        # b Contin.  -0.6847

        # Effective sample sizes
        #            Control Treated
        # Unadjusted     2.      2.
        # Adjusted       1.8     1.8

        # show the default "target" calculation is working as expected
        a1 = pd.Series((1, 2))
        b1 = pd.Series((-1, 1))
        a2 = pd.Series((3, 4))
        b2 = pd.Series((-2, 2))
        w1 = pd.Series((1, 1))
        w2 = w1
        r = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
            w1,
            w2,
        ).to_dict()
        exp_a = np.abs(a1.mean() - a2.mean()) / a2.std()
        exp_b = np.abs(b1.mean() - b2.mean()) / b2.std()
        self.assertEqual((r["a"], r["b"]), (exp_a, exp_b))
        # show that the default is weights equal 1.
        r_no_weights = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
        ).to_dict()
        self.assertEqual(r, r_no_weights)
        # demonstrate that weights effect the outcome (use 0 weights.)
        a1 = pd.Series((1, 2, 100))
        b1 = pd.Series((-1, 1, 100))
        a2 = pd.Series((3, 4, 100))
        b2 = pd.Series((-2, 2, 100))
        w1 = pd.Series((1, 1, 0))
        w2 = w1
        r_with_0_3rd_weight = asmd(
            pd.DataFrame({"a": a1, "b": b1}),
            pd.DataFrame({"a": a2, "b": b2}),
            w1,
            w2,
        ).to_dict()
        self.assertEqual(r, r_with_0_3rd_weight)

        # Not passing weights is the same as weights of all 1s
        r = asmd(
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)
        self.assertEqual(type(r), pd.Series)

        r = asmd(
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )
        e_asmd = pd.Series(
            (2.828_427_1, 0.684_658_9, (2.828_427_1 + 0.684_658_9) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        # Test different std_types
        args = (
            pd.DataFrame({"a": (1, 2), "b": (-1, 12)}),
            pd.DataFrame({"a": (3, 4), "b": (0, 42)}),
            pd.Series((1, 2)),
            pd.Series((1, 2)),
        )

        r = asmd(*args, std_type="target")
        self.assertEqual(r, e_asmd)

        # TODO: this should be called : test consistency of asmd
        r = asmd(*args, std_type="sample")
        e_asmd_sample = pd.Series(
            (
                2.828_427_124_746_189_4,
                2.211_975_059_096_379,
                (2.828_427_124_746_189_4 + 2.211_975_059_096_379) / 2,
            ),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd_sample)

        r = asmd(*args, std_type="pooled")
        e_asmd_pooled = pd.Series(
            (2.828_427_1, 0.924_959_4, (2.828_427_1 + 0.924_959_4) / 2),
            index=("a", "b", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd_pooled)

        #  Test with categoricals
        #  Categorical variable has missing value in one df
        r = asmd(
            pd.DataFrame({"c": ("x", "y", "x", "x"), "a": (5, 6, 7, 8)}),
            pd.DataFrame({"c": ("x", "y", "x", "z"), "a": (1, 2, 3, 4)}),
            pd.Series((1, 2, 3, 4)),
            pd.Series((1, 1, 1, 1)),
        )
        e_asmd = pd.Series(
            (
                3.485_685_011_586_675_3,
                0.519_615_242_270_663_2,
                0.099_999_999_999_999_98,
                np.nan,
                (
                    3.485_685_011_586_675_3
                    + 0.519_615_242_270_663_2 * (1 / 3)
                    + 0.099_999_999_999_999_98 * (1 / 3)
                    + 0 * (1 / 3)
                )
                / (2),
            ),
            index=("a", "c[x]", "c[y]", "c[z]", "mean(asmd)"),
        )
        self.assertEqual(r, e_asmd)

        # Check that using aggregate_by_main_covar works
        a1 = pd.Series((1, 2))
        b1_A = pd.Series((1, 3))
        b1_B = pd.Series((-1, -3))
        a2 = pd.Series((3, 4))
        b2_A = pd.Series((2, 3))
        b2_B = pd.Series((-2, -3))
        w1 = pd.Series((1, 1))
        w2 = w1

        r1 = asmd(
            pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
            pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
            w1,
            w2,
        ).to_list()

        r2 = asmd(
            pd.DataFrame({"a": a1, "b[A]": b1_A, "b[B]": b1_B}),
            pd.DataFrame({"a": a2, "b[A]": b2_A, "b[B]": b2_B}),
            w1,
            w2,
            "target",
            True,
        ).to_list()

        self.assertTrue(
            all((np.round(r1, 5)) == np.array([2.82843, 0.70711, 0.70711, 1.76777]))
        )
        self.assertTrue(all((np.round(r2, 5)) == np.array([2.82843, 0.70711, 1.76777])))

    def test__aggregate_asmd_by_main_covar(self):
        from balance.stats_and_plots.weighted_comparisons_stats import (
            _aggregate_asmd_by_main_covar,
        )

        # toy example
        asmd_series = pd.Series(
            {
                "age": 0.5,
                "education[T.high_school]": 1,
                "education[T. bachelor]": 2,
                "education[T. masters]": 3,
                "education[T. phd]": 4,
            }
        )

        outcome = _aggregate_asmd_by_main_covar(asmd_series).to_dict()
        expected = {"age": 0.5, "education": 2.5}

        self.assertEqual(outcome, expected)

    def test_asmd_improvement(self):
        from balance.stats_and_plots.weighted_comparisons_stats import asmd_improvement

        r = asmd_improvement(
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.Series((1, 1, 1, 1)),
            pd.Series((2, 2, 2, 2)),
            pd.Series((1, 1, 1, 1)),
        )
        self.assertEqual(r, 0)
        self.assertEqual(type(r), np.float64)

        r = asmd_improvement(
            pd.DataFrame({"a": (5, 6, 7, 8)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.DataFrame({"a": (1, 2, 3, 4)}),
            pd.Series((1, 1, 1, 1)),
            pd.Series((1, 1, 1, 1)),
            pd.Series((1, 1, 1, 1)),
        )
        self.assertEqual(r, 1)
        self.assertEqual(type(r), np.float64)

        r = asmd_improvement(
            pd.DataFrame({"a": (3, 4)}),
            pd.DataFrame({"a": (2, 3)}),
            pd.DataFrame({"a": (1, 2)}),
            pd.Series((1, 1)),
            pd.Series((1, 1)),
            pd.Series((1, 1)),
        )
        self.assertEqual(r, 0.5)
        self.assertEqual(type(r), np.float64)


class TestBalance_general_stats(
    balance.testutil.BalanceTestCase,
):
    def test_relative_response_rates(self):
        from balance.stats_and_plots.general_stats import relative_response_rates

        df = pd.DataFrame(
            {"o1": (7, 8, 9, 10), "o2": (7, 8, 9, np.nan), "id": (1, 2, 3, 4)}
        )

        self.assertEqual(
            relative_response_rates(df).to_dict(),
            {
                "o1": {"n": 4.0, "%": 100.0},
                "o2": {"n": 3.0, "%": 75.0},
                "id": {"n": 4.0, "%": 100.0},
            },
        )

        df_target = pd.concat([df, df])
        self.assertEqual(
            relative_response_rates(df, df_target).to_dict(),
            {
                "o1": {"n": 4.0, "%": 50.0},
                "o2": {"n": 3.0, "%": 50.0},
                "id": {"n": 4.0, "%": 50.0},
            },
        )

        # verify behavior when per_column is set to False
        self.assertEqual(
            relative_response_rates(df, df_target, per_column=False).round(3).to_dict(),
            {
                "o1": {"n": 4.0, "%": 66.667},
                "o2": {"n": 3.0, "%": 50.0},
                "id": {"n": 4.0, "%": 66.667},
            },
        )

        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns*"
        ):
            relative_response_rates(df, df_target.iloc[:, 0:1], per_column=True)
