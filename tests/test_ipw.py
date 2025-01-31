# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import absolute_import, division, print_function, unicode_literals

import balance.testutil

import numpy as np
import pandas as pd

from balance.sample_class import Sample
from balance.weighting_methods import ipw as balance_ipw


class Testipw(
    balance.testutil.BalanceTestCase,
):
    def test_ipw_weights_order(self):
        sample = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 1)})
        target = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 9)})

        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            transformations=None,
            max_de=1.5,
        )

        w = result["weight"].values

        self.assertEqual(w[0], w[8])
        self.assertTrue(w[0] < w[1])
        self.assertTrue(w[0] < w[7])

    def test_ipw_sample_indicator(self):
        # TODO: verify this behavior of balance is sensible

        # error message
        # TODO: is this test working?
        s = pd.DataFrame({"a": np.random.uniform(0, 1, 1000), "id": range(0, 1000)})
        t = pd.DataFrame({"a": np.random.uniform(0.5, 1.5, 1000), "id": range(0, 1000)})
        s_w = pd.Series(np.array((1,) * 1000))
        t_w = pd.Series(np.array((1,) * 1000))
        self.assertRaises(
            Exception, "same number of rows", balance_ipw.ipw, s, s_w, t, t_w
        )

        t = pd.DataFrame({"a": np.random.uniform(0.5, 1.5, 999), "id": range(0, 999)})
        # Doesn't raise an error
        balance_ipw.ipw(
            sample_df=s,
            sample_weights=s_w,
            target_df=t,
            target_weights=pd.Series(np.array((1,) * 999)),
        )

    def test_ipw_bad_adjustment_warnings(self):
        #  Test the check for identical weights
        #  Test the check for no large coefficients
        #  Test the check for model accuracy
        n = 100
        sample = Sample.from_frame(
            df=pd.DataFrame(
                {"a": np.random.normal(0, 1, n).reshape((n,)), "id": range(0, n)}
            ),
            id_column="id",
        )
        sample = sample.set_target(sample)
        self.assertWarnsRegexp(
            "All weights are identical. The estimates will not be adjusted",
            sample.adjust,
            method="ipw",
            balance_classes=False,
        )
        self.assertWarnsRegexp(
            (
                "The propensity model has low fraction null deviance explained "
                ".*. Results may not be accurate"
            ),
            sample.adjust,
            method="ipw",
            balance_classes=False,
        )

    def test_ipw_na_drop(self):
        s = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.concatenate(
                        (np.array([np.nan, np.nan]), np.arange(3, 100))
                    ),
                    "id": np.arange(1, 100),
                }
            ),
            id_column="id",
        )

        t = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.concatenate((np.array([np.nan]), np.arange(2, 100))),
                    "id": np.arange(1, 100),
                }
            ),
            id_column="id",
        )

        self.assertWarnsRegexp(
            "Dropped 2/99 rows of sample",
            s.adjust,
            t,
            na_action="drop",
            transformations=None,
        )

    def test_weights_from_link(self):
        link = np.array((1, 2, 3)).reshape(3, 1)
        target_weights = (1, 2)
        r = balance_ipw.weights_from_link(link, False, (1, 1, 1), target_weights)
        e = np.array((1 / np.exp(1), 1 / np.exp(2), 1 / np.exp(3)))
        e = e * np.sum(target_weights) / np.sum(e)
        self.assertEqual(r, e)
        self.assertEqual(r.shape, (3,))

        # balance_classes does nothing if classes have same sum weights
        r = balance_ipw.weights_from_link(link, True, (1, 1, 1), (1, 2))
        self.assertEqual(r, e)

        # balance_classes uses link+log(odda)
        target_weights = (1, 2, 3)
        r = balance_ipw.weights_from_link(
            link, True, (1, 1, 1), target_weights, keep_sum_of_weights=False
        )
        e = np.array(
            (
                1 / np.exp(1 + np.log(1 / 2)),
                1 / np.exp(2 + np.log(1 / 2)),
                1 / np.exp(3 + np.log(1 / 2)),
            )
        )
        e = e * np.sum(target_weights) / np.sum(e)
        self.assertEqual(r, e)

        # sample_weights doubles
        target_weights = (1, 2)
        r = balance_ipw.weights_from_link(link, False, (2, 2, 2), target_weights)
        e = np.array((2 / np.exp(1), 2 / np.exp(2), 2 / np.exp(3)))
        e = e * np.sum(target_weights) / np.sum(e)
        self.assertEqual(r, e)

        # trimming works
        r = balance_ipw.weights_from_link(
            np.random.uniform(0, 1, 10000),
            False,
            (1,) * 10000,
            (1),
            weight_trimming_percentile=(0, 0.11),
            keep_sum_of_weights=False,  # this parameter is passed to trim_weights
        )
        self.assertTrue(r.max() < 0.9)

    def test_ipw_input_assertions(self):
        s_w = np.array((1))
        self.assertRaisesRegex(
            TypeError,
            "must be a pandas DataFrame",
            balance_ipw.ipw,
            s_w,
            s_w,
            s_w,
            s_w,
        )

        s = pd.DataFrame({"a": (1, 2), "id": (1, 2)})
        s_w = pd.Series((1,))
        self.assertRaisesRegex(
            Exception,
            "must be the same length",
            balance_ipw.ipw,
            s,
            s_w,
            s,
            s_w,
        )

    def test_ipw_dropna_empty(self):
        s = pd.DataFrame({"a": (1, None), "b": (np.nan, 2), "id": (1, 2)})
        s_w = pd.Series((1, 2))
        self.assertRaisesRegex(
            Exception,
            "Dropping rows led to empty",
            balance_ipw.ipw,
            s,
            s_w,
            s,
            s_w,
            na_action="drop",
        )

    def test_ipw_formula(self):
        sample = pd.DataFrame(
            {
                "a": (1, 2, 3, 4, 5, 6, 7, 9, 1),
                "b": (1, 2, 3, 4, 5, 6, 7, 9, 1),
                "c": (1, 2, 3, 4, 5, 6, 7, 9, 1),
            }
        )
        target = pd.DataFrame(
            {
                "a": (1, 2, 3, 4, 5, 6, 7, 9, 9),
                "b": (1, 2, 3, 4, 5, 6, 7, 9, 1),
                "c": (1, 2, 3, 4, 5, 6, 7, 9, 1),
            }
        )

        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            formula="a : b + c",
            transformations=None,
        )
        self.assertEqual(result["model"]["X_matrix_columns"], ["a:b", "c"])

    def test_ipw_consistency_with_default_arguments(self):
        # This test is meant to check the consistency of the ipw function with the default arguments,
        # Note that this test rely on all balance functions that are part of ipw:
        # choose_variables, apply_transformations, model_matrix, choose_regularization,
        # weights_from_link, cv_glmnet_performance
        # Therefore a failure in this test may indicate a failure in one
        # of these functions as well as a failure of the ipw function

        np.random.seed(2021)
        n_sample = 1000
        n_target = 2000

        sample_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 10, size=n_sample), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_sample, 4)), columns=range(1, 5)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_sample, 5)
                    ),
                    columns=range(5, 10),
                ),
            ],
            axis=1,
        )
        sample_df = sample_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        target_df = pd.concat(
            [
                pd.DataFrame(np.random.uniform(8, 18, size=n_target), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(n_target, 4)), columns=range(1, 5)
                ),
                pd.DataFrame(
                    np.random.choice(
                        ["level1", "level2", "level3"], size=(n_target, 5)
                    ),
                    columns=range(5, 10),
                ),
            ],
            axis=1,
        )
        target_df = target_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        sample_weights = pd.Series(np.random.uniform(0, 1, size=n_sample))
        target_weights = pd.Series(np.random.uniform(0, 1, size=n_target))

        res = balance_ipw.ipw(
            sample_df, sample_weights, target_df, target_weights, max_de=1.5
        )

        # Compare output weights (examples and distribution)
        self.maxDiff = None
        self.assertEqual(round(res["weight"][15], 4), 0.4575)
        self.assertEqual(round(res["weight"][995], 4), 0.4059)
        self.assertEqual(
            np.around(res["weight"].describe().values, 4),
            np.array([1000, 1.0167, 0.7159, 0.0003, 0.4292, 0.8928, 1.4316, 2.5720]),
        )

        # Compare properties of output model
        self.assertEqual(
            np.around(res["model"]["perf"]["prop_dev_explained"], 5), 0.27296
        )
        self.assertEqual(np.around(res["model"]["lambda"], 5), 0.52831)
        self.assertEqual(res["model"]["regularisation_perf"]["best"]["trim"], 2.5)
