# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from __future__ import absolute_import, division, print_function, unicode_literals

import balance.testutil

import numpy as np
import pandas as pd
import scipy

from balance.sample_class import Sample
from balance.stats_and_plots.weights_stats import design_effect
from balance.weighting_methods import cbps as balance_cbps


class Testcbps(
    balance.testutil.BalanceTestCase,
):
    def test_cbps_from_adjust_function(self):
        sample = Sample.from_frame(
            pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9, 1), "id": range(0, 10)})
        )
        target = Sample.from_frame(
            pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9, 9), "id": range(0, 10)})
        )
        sample = sample.set_target(target)
        result_adjust = sample.adjust(method="cbps", transformations=None)

        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)})
        target_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9, 9)})
        sample_weights = pd.Series((1,) * 10)
        target_weights = pd.Series((1,) * 10)
        result_cbps = balance_cbps.cbps(
            sample_df, sample_weights, target_df, target_weights, transformations=None
        )
        self.assertEqual(
            result_adjust.df["weight"], result_cbps["weights"].rename("weight")
        )

    def test_logit_truncated(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        result = balance_cbps.logit_truncated(X, beta)
        self.assertEqual(
            np.around(result, 6), np.array([0.993307, 0.99999, 1.00000000e-05])
        )

        # test truncation_value
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        result = balance_cbps.logit_truncated(X, beta, truncation_value=0.1)
        self.assertEqual(np.around(result, 6), np.array([0.9, 0.9, 0.1]))

    def test_compute_pseudo_weights_from_logit_probs(self):
        probs = np.array([0.1, 0.6, 0.2])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        result = balance_cbps.compute_pseudo_weights_from_logit_probs(
            probs, design_weights, in_pop
        )
        self.assertEqual(np.around(result, 1), np.array([3.0, -4.5, 3.0]))

    # Testing consistency result of bal_loss
    def test_bal_loss(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        XtXinv = np.linalg.inv(np.matmul(X.T, X))
        result = balance_cbps.bal_loss(beta, X, design_weights, in_pop, XtXinv)
        self.assertEqual(round(result, 2), 39999200004.99)

    # Testing consistency result of gmm_function
    def test_gmm_function(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        result = balance_cbps.gmm_function(beta, X, design_weights, in_pop)
        self.assertEqual(round(result["loss"], 2), 91665.75)

        # with given invV
        X = np.array([[1, 2], [4, 5], [0, -100]])
        beta = np.array([1, 0.5])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        invV = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 0, 0], [0, 0, 0, 1]])
        result = balance_cbps.gmm_function(beta, X, design_weights, in_pop, invV)
        self.assertEqual(round(result["loss"], 4), 45967903.9923)
        self.assertEqual(result["invV"], invV)

    # Testing consistency result of gmm_loss
    def test_gmm_loss(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        result = balance_cbps.gmm_loss(beta, X, design_weights, in_pop)
        self.assertEqual(round(result, 2), 91665.75)

        # with given invV
        X = np.array([[1, 2], [4, 5], [0, -100]])
        beta = np.array([1, 0.5])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        invV = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 0, 0], [0, 0, 0, 1]])
        result = balance_cbps.gmm_loss(beta, X, design_weights, in_pop, invV)
        self.assertEqual(round(result, 4), 45967903.9923)

    # Testing consistency result of alpha_function
    def test_alpha_function(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        alpha = 1
        result = balance_cbps.alpha_function(alpha, beta, X, design_weights, in_pop)
        self.assertEqual(result, balance_cbps.gmm_loss(beta, X, design_weights, in_pop))

        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        alpha = 0.5
        result_smaller_alpha = balance_cbps.alpha_function(
            alpha, beta, X, design_weights, in_pop
        )
        self.assertEqual(round(result_smaller_alpha, 4), 25345.0987)

        # smaller alpha gives smaller loss
        self.assertTrue(result_smaller_alpha <= result)

    # Testing consistency result of compute_deff_from_beta function
    def test_compute_deff_from_beta(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([0.0, 0, 1.0])
        result = balance_cbps.compute_deff_from_beta(X, beta, design_weights, in_pop)
        self.assertEqual(round(result, 6), 1.999258)

    def test__standardize_model_matrix(self):
        # numpy array as input
        mat = np.array([[1, 2], [3, 4]])
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0, -1.0], [1.0, 1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["a", "b"])
        self.assertEqual(res["model_matrix_mean"], np.array([2.0, 3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0, 1.0]))
        # check when column is constant
        mat = np.array([[1, 2], [1, 4]])
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0], [1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["b"])
        self.assertEqual(res["model_matrix_mean"], np.array([3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0]))

        # pandas dataframe as input
        mat = pd.DataFrame({"a": (1, 3), "b": (2, 4)}).values
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0, -1.0], [1.0, 1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["a", "b"])
        self.assertEqual(res["model_matrix_mean"], np.array([2.0, 3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0, 1.0]))
        # check when column is constant
        mat = pd.DataFrame({"a": (1, 1), "b": (2, 4)}).values
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0], [1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["b"])
        self.assertEqual(res["model_matrix_mean"], np.array([3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0]))

        # check warnings
        self.assertWarnsRegexp(
            "The following variables have only one level, and are omitted:",
            balance_cbps._standardize_model_matrix,
            mat,
            ["a", "b"],
        )

    def test__reverse_svd_and_centralization(self):
        np.random.seed(10)
        m, n = 4, 3
        X_matrix = np.random.randn(m, n)
        X_matrix_std = np.std(X_matrix, axis=0)
        X_matrix_mean = np.mean(X_matrix, axis=0)
        X_matrix_new = (X_matrix - X_matrix_mean) / X_matrix_std
        X_matrix_new = np.c_[
            np.ones(X_matrix_new.shape[0]), X_matrix_new
        ]  # Add intercept
        U, s, Vh = scipy.linalg.svd(X_matrix_new, full_matrices=False)
        beta = np.array([-5, 1, -2, 3])
        self.assertEqual(
            np.around(
                np.matmul(
                    np.c_[np.ones(X_matrix.shape[0]), X_matrix],
                    balance_cbps._reverse_svd_and_centralization(
                        beta, U, s, Vh, X_matrix_mean, X_matrix_std
                    ),
                ),
                7,
            ),
            np.around(np.matmul(U, beta), 7),
        )

    # Test consistency result of cbps
    def test_cbps_consistency_with_default_arguments(self):
        # This test is meant to check the consistency of the cbps function with the default arguments,
        # Note that this test rely on all balance functions that are part of cbps:
        # choose_variables, apply_transformations, model_matrix, trim_weights
        # Therefore a failure in this test may indicate a failure in one
        # of these functions as well as a failure of the cbps function
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
                pd.DataFrame(
                    np.concatenate(
                        (
                            np.random.uniform(0, 8, size=int(n_target / 2)),
                            np.random.uniform(8, 10, size=int(n_target / 2)),
                        )
                    ),
                    columns=[0],
                ),
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

        res = balance_cbps.cbps(sample_df, sample_weights, target_df, target_weights)

        # Compare output weights (examples and distribution)
        # TODO: The results are not 100% reproducible due to rounding issues in SVD that produce slightly different U:
        # http://numpy-discussion.10968.n7.nabble.com/strange-behavior-of-numpy-random-multivariate-normal-ticket-1842-td31547.html
        # This results in slightly different optimizations solutions (that might have some randomness in them too).
        # self.assertEqual(round(res["weights"][4],4), 4.3932)
        # self.assertEqual(round(res["weights"][997],4), 0.7617)
        # self.assertEqual(np.around(res["weights"].describe().values,4),
        #                np.array([1.0000e+03, 1.0167e+00, 1.1340e+00, 3.0000e-04,
        #                          3.3410e-01, 6.8400e-01, 1.2317e+00, 7.4006e+00]))
        self.assertTrue(
            res["weights"][995] < res["weights"][999]
        )  # these are obs with different a value

    # Test cbps constraints
    def test_cbps_constraints(self):
        sample_df = pd.DataFrame({"a": [-20] + [1] * 13 + [10] * 1})
        sample_weights = pd.Series((1,) * 15)
        target_df = pd.DataFrame({"a": [10] * 10 + [11] * 5})
        target_weights = pd.Series((1,) * 15)

        unconconstrained_result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=None,
            weight_trimming_mean_ratio=None,
        )

        # Ensure that example df would produce DE > 1.5 if unconstrained
        self.assertTrue(design_effect(unconconstrained_result["weights"]) > 1.5)

        # Same data but now with constraint produces desired design effect - for cbps_method = "over"
        de_constrained_result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=1.5,
            weight_trimming_mean_ratio=None,
        )
        self.assertTrue(
            round(design_effect(de_constrained_result["weights"]), 5) <= 1.5
        )
        # Same data but now with constraint produces desired design effect - for cbps_method = "exact"
        de_constrained_result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=1.5,
            weight_trimming_mean_ratio=None,
            cbps_method="exact",
        )
        self.assertTrue(
            round(design_effect(de_constrained_result["weights"]), 5) <= 1.5
        )

    def test_cbps_weights_order(self):
        sample = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 1)})
        target = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 9)})

        result = balance_cbps.cbps(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            transformations=None,
        )

        w = result["weights"].values
        self.assertEqual(round(w[0], 10), round(w[8], 10))
        self.assertTrue(w[0] < w[1])
        self.assertTrue(w[0] < w[7])

    def test_cbps_all_weight_identical(self):
        #  Test the check for identical weights
        np.random.seed(1)
        n = 1000
        sample_df = pd.DataFrame({"a": np.random.normal(0, 1, n).reshape((n,))})
        sample_weights = pd.Series((1,) * n)
        target_df = sample_df
        target_weights = sample_weights
        result = balance_cbps.cbps(
            sample_df, sample_weights, target_df, target_weights, transformations=None
        )
        self.assertTrue(np.var(result["weights"]) < 1e-10)

        sample = Sample.from_frame(
            df=pd.DataFrame(
                {"a": np.random.normal(0, 1, n).reshape((n,)), "id": range(0, n)}
            ),
            id_column="id",
        )
        sample = sample.set_target(sample)
        self.assertWarnsRegexp(
            "All weights are identical",
            sample.adjust,
            method="cbps",
            transformations=None,
        )

    def test_cbps_na_drop(self):
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
            method="cbps",
            na_action="drop",
            transformations=None,
        )

    def test_cbps_input_assertions(self):
        s_w = np.array((1))
        self.assertRaisesRegex(
            TypeError,
            "must be a pandas DataFrame",
            balance_cbps.cbps,
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
            balance_cbps.cbps,
            s,
            s_w,
            s,
            s_w,
        )

    def test_cbps_dropna_empty(self):
        s = pd.DataFrame({"a": (1, None), "b": (np.nan, 2), "id": (1, 2)})
        s_w = pd.Series((1, 2))
        self.assertRaisesRegex(
            Exception,
            "Dropping rows led to empty",
            balance_cbps.cbps,
            s,
            s_w,
            s,
            s_w,
            na_action="drop",
        )

    def test_cbps_formula(self):
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
        result = balance_cbps.cbps(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            formula="a : b + c",
            transformations=None,
        )
        self.assertEqual(result["model"]["X_matrix_columns"], ["Intercept", "a:b", "c"])

    def test_cbps_warning_for_variable_with_one_level(self):
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [-20] + [1] * 13 + [10] * 1,
                    "b": [1] * 15,
                    "id": np.arange(0, 15),
                }
            ),
            id_column="id",
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": [10] * 10 + [11] * 5,
                    "b": [1] * 15,
                    "id": np.arange(0, 15),
                }
            ),
            id_column="id",
        )
        sample = sample.set_target(target)
        self.assertWarnsRegexp(
            "The following variables have only one level",
            sample.adjust,
            method="cbps",
            transformations=None,
        )
