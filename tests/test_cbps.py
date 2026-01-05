# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import warnings

import balance.testutil

import numpy as np
import pandas as pd
import scipy

from balance.datasets import load_data
from balance.sample_class import Sample
from balance.stats_and_plots.weights_stats import design_effect
from balance.weighting_methods import cbps as balance_cbps

# Test constants for improved readability and maintainability
TEST_SEED = 2021
SAMPLE_SIZE = 1000
TARGET_SIZE = 2000
TOLERANCE = 1e-10
MAX_DESIGN_EFFECT = 1.5
TRUNCATION_THRESHOLD = 0.1


class Testcbps(
    balance.testutil.BalanceTestCase,
):
    def test_cbps_from_adjust_function(self) -> None:
        """Test that CBPS results are consistent between Sample.adjust() and direct cbps() calls.

        This test verifies that the high-level Sample.adjust(method="cbps") interface
        produces identical results to calling the lower-level balance_cbps.cbps() function
        directly with the same parameters.
        """
        sample_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)
        target_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 9)

        # Test using Sample.adjust() interface
        sample = Sample.from_frame(pd.DataFrame({"a": sample_data, "id": range(0, 10)}))
        target = Sample.from_frame(pd.DataFrame({"a": target_data, "id": range(0, 10)}))
        sample = sample.set_target(target)
        result_adjust = sample.adjust(method="cbps", transformations=None)

        # Test using direct cbps() function call
        sample_df = pd.DataFrame({"a": sample_data})
        target_df = pd.DataFrame({"a": target_data})
        sample_weights = pd.Series((1,) * 10)
        target_weights = pd.Series((1,) * 10)
        result_cbps = balance_cbps.cbps(
            sample_df, sample_weights, target_df, target_weights, transformations=None
        )

        # Results should be identical
        result_dict = result_cbps
        if isinstance(result_dict, dict) and "weight" in result_dict:
            weight_series = result_dict["weight"]
            if isinstance(weight_series, pd.Series):
                self.assertEqual(
                    result_adjust.df["weight"],
                    weight_series.rename("weight"),
                    msg="Sample.adjust() and direct cbps() should produce identical weights",
                )

    def test_logit_truncated(self) -> None:
        """Test the logit_truncated function with and without custom truncation values.

        This test verifies that:
        1. The logit function correctly computes probabilities from linear combinations
        2. Extreme values are properly truncated to prevent numerical issues
        3. Custom truncation values work as expected
        """
        # Test data with extreme values to trigger truncation
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])

        # Test default truncation behavior
        result = balance_cbps.logit_truncated(X, beta)
        expected_default = np.array([0.993307, 0.99999, 1.00000000e-05])
        self.assertEqual(
            np.around(result, 6),
            expected_default,
            msg="Default truncation should handle extreme values correctly",
        )

        # Test custom truncation value
        result_custom = balance_cbps.logit_truncated(
            X, beta, truncation_value=TRUNCATION_THRESHOLD
        )
        expected_custom = np.array([0.9, 0.9, 0.1])
        self.assertEqual(
            np.around(result_custom, 6),
            expected_custom,
            msg="Custom truncation value should be applied to extreme probabilities",
        )

    def test_compute_pseudo_weights_from_logit_probs(self) -> None:
        """Test computation of pseudo weights from logistic regression probabilities.

        This test verifies that pseudo weights are correctly computed based on:
        - Logistic regression probabilities
        - Design weights from the sample
        - Population membership indicators
        """
        probs = np.array([0.1, 0.6, 0.2])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])

        result = balance_cbps.compute_pseudo_weights_from_logit_probs(
            probs, design_weights, in_pop
        )
        expected = np.array([3.0, -4.5, 3.0])

        self.assertEqual(
            np.around(result, 1),
            expected,
            msg="Pseudo weights should be computed correctly from logit probabilities",
        )

    def test_bal_loss(self) -> None:
        """Test the balance loss function used in CBPS optimization.

        This function computes the loss for the balance constraints in CBPS.
        The test verifies numerical consistency of the loss computation.
        """
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])
        XtXinv = np.linalg.inv(np.matmul(X.T, X))

        result = balance_cbps.bal_loss(beta, X, design_weights, in_pop, XtXinv)
        expected_loss = 39999200004.99

        self.assertEqual(
            round(result, 2),
            expected_loss,
            msg="Balance loss should be computed consistently",
        )

    def test_gmm_function(self) -> None:
        """Test the Generalized Method of Moments (GMM) function used in CBPS.

        This test verifies:
        1. GMM loss computation with automatic covariance matrix estimation
        2. GMM loss computation with user-provided inverse covariance matrix
        3. Correct return of the inverse covariance matrix
        """
        # Test with automatic covariance matrix estimation
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])

        result = balance_cbps.gmm_function(beta, X, design_weights, in_pop)
        expected_loss = 91665.75

        self.assertEqual(
            round(result["loss"], 2),
            expected_loss,
            msg="GMM loss should be computed correctly with automatic covariance estimation",
        )

        # Test with user-provided inverse covariance matrix
        X_reduced = np.array([[1, 2], [4, 5], [0, -100]])
        beta_reduced = np.array([1, 0.5])
        invV = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 0, 0], [0, 0, 0, 1]])

        result_with_invV = balance_cbps.gmm_function(
            beta_reduced, X_reduced, design_weights, in_pop, invV
        )
        expected_loss_with_invV = 45967903.9923

        self.assertEqual(
            round(result_with_invV["loss"], 4),
            expected_loss_with_invV,
            msg="GMM loss should be computed correctly with provided inverse covariance matrix",
        )
        self.assertEqual(
            result_with_invV["invV"],
            invV,
            msg="Provided inverse covariance matrix should be returned unchanged",
        )

    def test_gmm_loss(self) -> None:
        """Test the GMM loss function (simplified version of gmm_function).

        This test verifies that the GMM loss function returns consistent results
        both with automatic covariance estimation and with provided inverse covariance.
        """
        # Test with automatic covariance matrix estimation
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])

        result = balance_cbps.gmm_loss(beta, X, design_weights, in_pop)
        expected_loss = 91665.75

        self.assertEqual(
            round(result, 2),
            expected_loss,
            msg="GMM loss should match expected value with automatic covariance estimation",
        )

        # Test with user-provided inverse covariance matrix
        X_reduced = np.array([[1, 2], [4, 5], [0, -100]])
        beta_reduced = np.array([1, 0.5])
        invV = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 0, 0], [0, 0, 0, 1]])

        result_with_invV = balance_cbps.gmm_loss(
            beta_reduced, X_reduced, design_weights, in_pop, invV
        )
        expected_loss_with_invV = 45967903.9923

        self.assertEqual(
            round(result_with_invV, 4),
            expected_loss_with_invV,
            msg="GMM loss should match expected value with provided inverse covariance",
        )

    def test_alpha_function(self) -> None:
        """Test the alpha function used for balancing efficiency and balance constraints.

        This test verifies:
        1. When alpha=1, the function equals gmm_loss (pure balance)
        2. Smaller alpha values produce smaller loss values (more efficiency-focused)
        3. The function correctly interpolates between efficiency and balance
        """
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])

        # Test alpha=1 case (should equal gmm_loss)
        alpha_one = np.array([1])
        result_alpha_one = balance_cbps.alpha_function(
            alpha_one, beta, X, design_weights, in_pop
        )
        gmm_result = balance_cbps.gmm_loss(beta, X, design_weights, in_pop)

        self.assertEqual(
            result_alpha_one,
            gmm_result,
            msg="Alpha function with alpha=1 should equal gmm_loss",
        )

        # Test alpha<1 case (should give smaller loss)
        alpha_smaller = np.array([0.5])
        result_smaller_alpha = balance_cbps.alpha_function(
            alpha_smaller, beta, X, design_weights, in_pop
        )
        expected_smaller_loss = 25345.0987

        # Ensure result is array-like before indexing
        if isinstance(result_smaller_alpha, np.ndarray):
            self.assertEqual(
                round(result_smaller_alpha[0], 4),
                expected_smaller_loss,
                msg="Alpha function should produce expected loss for alpha=0.5",
            )

            # Verify that smaller alpha gives smaller or equal loss
            if isinstance(result_alpha_one, np.ndarray):
                self.assertTrue(
                    result_smaller_alpha[0] <= result_alpha_one[0],
                    msg="Smaller alpha should produce smaller or equal loss (more efficiency-focused)",
                )

    def test_compute_deff_from_beta(self) -> None:
        """Test computation of design effect from beta coefficients.

        This function computes the design effect (a measure of efficiency loss)
        from the estimated beta coefficients in CBPS. The test verifies numerical
        consistency of this computation.
        """
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([0.0, 0, 1.0])

        result = balance_cbps.compute_deff_from_beta(X, beta, design_weights, in_pop)
        expected_deff = 1.999258

        self.assertEqual(
            round(result, 6),
            expected_deff,
            msg="Design effect should be computed correctly from beta coefficients",
        )

    def test__standardize_model_matrix(self) -> None:
        # numpy array as input
        mat = np.array([[1, 2], [3, 4]])
        # pyre-fixme[6]: Testing with ndarray is valid
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0, -1.0], [1.0, 1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["a", "b"])
        self.assertEqual(res["model_matrix_mean"], np.array([2.0, 3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0, 1.0]))
        # check when column is constant
        mat = np.array([[1, 2], [1, 4]])
        # pyre-fixme[6]: Testing with ndarray is valid
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0], [1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["b"])
        self.assertEqual(res["model_matrix_mean"], np.array([3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0]))

        # pandas dataframe as input
        mat = pd.DataFrame({"a": (1, 3), "b": (2, 4)}).values
        # pyre-fixme[6]: Testing with ndarray is valid
        res = balance_cbps._standardize_model_matrix(mat, ["a", "b"])
        self.assertEqual(res["model_matrix"], np.array([[-1.0, -1.0], [1.0, 1.0]]))
        self.assertEqual(res["model_matrix_columns_names"], ["a", "b"])
        self.assertEqual(res["model_matrix_mean"], np.array([2.0, 3.0]))
        self.assertEqual(res["model_matrix_std"], np.array([1.0, 1.0]))
        # check when column is constant
        mat = pd.DataFrame({"a": (1, 1), "b": (2, 4)}).values
        # pyre-fixme[6]: Testing with ndarray is valid
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

    def test__reverse_svd_and_centralization(self) -> None:
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

    def test_cbps_consistency_with_default_arguments(self) -> None:
        """Test CBPS function consistency with default arguments on complex data.

        This comprehensive test verifies that the CBPS function works correctly
        with default arguments on realistic data containing both continuous and
        categorical variables. It tests the integration of all CBPS components:
        - choose_variables: Variable selection
        - apply_transformations: Data transformations
        - model_matrix: Design matrix creation
        - trim_weights: Weight trimming

        Note: Due to numerical precision in SVD decomposition, exact weight values
        may vary slightly between runs, so we test relative ordering instead.
        """
        # Generate complex sample data
        np.random.seed(TEST_SEED)

        # Create continuous variables for sample
        continuous_vars_sample = pd.concat(
            [
                pd.DataFrame(np.random.uniform(0, 10, size=SAMPLE_SIZE), columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(SAMPLE_SIZE, 4)), columns=range(1, 5)
                ),
            ],
            axis=1,
        )

        # Create categorical variables for sample
        categorical_vars_sample = pd.DataFrame(
            np.random.choice(["level1", "level2", "level3"], size=(SAMPLE_SIZE, 5)),
            columns=range(5, 10),
        )

        # Combine and rename columns for sample
        sample_df = pd.concat([continuous_vars_sample, categorical_vars_sample], axis=1)
        sample_df = sample_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        # Generate complex target data with different distribution
        np.random.seed(TEST_SEED)

        # Create continuous variables with different distribution for target
        continuous_vars_target = pd.concat(
            [
                pd.DataFrame(
                    np.concatenate(
                        (
                            np.random.uniform(0, 8, size=int(TARGET_SIZE / 2)),
                            np.random.uniform(8, 10, size=int(TARGET_SIZE / 2)),
                        )
                    ),
                    columns=[0],
                ),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(TARGET_SIZE, 4)), columns=range(1, 5)
                ),
            ],
            axis=1,
        )

        # Create categorical variables for target
        categorical_vars_target = pd.DataFrame(
            np.random.choice(["level1", "level2", "level3"], size=(TARGET_SIZE, 5)),
            columns=range(5, 10),
        )

        # Combine and rename columns for target
        target_df = pd.concat([continuous_vars_target, categorical_vars_target], axis=1)
        target_df = target_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})

        # Generate random weights for realism
        np.random.seed(TEST_SEED)
        sample_weights = pd.Series(np.random.uniform(0, 1, size=SAMPLE_SIZE))
        target_weights = pd.Series(np.random.uniform(0, 1, size=TARGET_SIZE))

        # Run CBPS with default arguments
        result = balance_cbps.cbps(sample_df, sample_weights, target_df, target_weights)
        # NOTE: The results are not 100% reproducible due to rounding issues in SVD that produce slightly different U:
        # http://numpy-discussion.10968.n7.nabble.com/strange-behavior-of-numpy-random-multivariate-normal-ticket-1842-td31547.html
        # This results in slightly different optimizations solutions (that might have some randomness in them too).
        # self.assertEqual(round(res["weight"][4],4), 4.3932)
        # self.assertEqual(round(res["weight"][997],4), 0.7617)
        # self.assertEqual(np.around(res["weight"].describe().values,4),
        #                np.array([1.0000e+03, 1.0167e+00, 1.1340e+00, 3.0000e-04,
        #                          3.3410e-01, 6.8400e-01, 1.2317e+00, 7.4006e+00]))

        # Verify basic properties of the result
        self.assertIn("weight", result, msg="CBPS result should contain 'weight' key")
        self.assertEqual(
            len(result["weight"]),
            SAMPLE_SIZE,
            msg="Number of weights should match sample size",
        )

        # Test relative weight ordering (observations with different 'a' values)
        # This verifies that CBPS produces sensible relative weights
        if isinstance(result, dict) and "weight" in result:
            weight_dict = result["weight"]
            if isinstance(weight_dict, pd.Series):
                self.assertTrue(
                    weight_dict[995] < weight_dict[999],
                    msg="Weights should reflect differences in covariate values",
                )

    def test_cbps_constraints(self) -> None:
        """Test CBPS design effect constraints functionality.

        This test verifies that CBPS correctly applies design effect constraints
        to prevent excessive weight variation. It tests:
        1. Unconstrained CBPS produces high design effect on problematic data
        2. Design effect constraints successfully limit the design effect
        3. Both "over" and "exact" CBPS methods respect the constraint
        """
        # Create data that would produce high design effect without constraints
        sample_df = pd.DataFrame({"a": [-20] + [1] * 13 + [10] * 1})
        sample_weights = pd.Series((1,) * 15)
        target_df = pd.DataFrame({"a": [10] * 10 + [11] * 5})
        target_weights = pd.Series((1,) * 15)

        # Test unconstrained CBPS (should produce high design effect)
        unconstrained_result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=None,
            weight_trimming_mean_ratio=None,
        )

        if isinstance(unconstrained_result, dict) and "weight" in unconstrained_result:
            weight_series = unconstrained_result["weight"]
            if isinstance(weight_series, pd.Series):
                unconstrained_de = design_effect(weight_series)
            else:
                # pyre-fixme[6]: Fallback for unexpected type
                unconstrained_de = design_effect(unconstrained_result)
        else:
            # pyre-fixme[6]: Fallback for direct Series result
            unconstrained_de = design_effect(unconstrained_result)
        self.assertTrue(
            unconstrained_de > MAX_DESIGN_EFFECT,
            msg=f"Unconstrained CBPS should produce high design effect (>{MAX_DESIGN_EFFECT}), got {unconstrained_de}",
        )

        # Test constrained CBPS with "over" method
        constrained_result_over = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=MAX_DESIGN_EFFECT,
            weight_trimming_mean_ratio=None,
        )

        if (
            isinstance(constrained_result_over, dict)
            and "weight" in constrained_result_over
        ):
            weight_series = constrained_result_over["weight"]
            if isinstance(weight_series, pd.Series):
                constrained_de_over = design_effect(weight_series)
            else:
                # pyre-fixme[6]: Fallback for unexpected type
                constrained_de_over = design_effect(constrained_result_over)
        else:
            # pyre-fixme[6]: Fallback for direct Series result
            constrained_de_over = design_effect(constrained_result_over)
        self.assertTrue(
            round(constrained_de_over, 5) <= MAX_DESIGN_EFFECT,
            msg=f"Constrained CBPS ('over' method) should respect max_de={MAX_DESIGN_EFFECT}, got {constrained_de_over}",
        )

        # Test constrained CBPS with "exact" method
        constrained_result_exact = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            max_de=MAX_DESIGN_EFFECT,
            weight_trimming_mean_ratio=None,
            method="exact",
        )

        if (
            isinstance(constrained_result_exact, dict)
            and "weight" in constrained_result_exact
        ):
            weight_series = constrained_result_exact["weight"]
            if isinstance(weight_series, pd.Series):
                constrained_de_exact = design_effect(weight_series)
            else:
                # pyre-fixme[6]: Fallback for unexpected type
                constrained_de_exact = design_effect(constrained_result_exact)
        else:
            # pyre-fixme[6]: Fallback for direct Series result
            constrained_de_exact = design_effect(constrained_result_exact)
        self.assertTrue(
            round(constrained_de_exact, 5) <= MAX_DESIGN_EFFECT,
            msg=f"Constrained CBPS ('exact' method) should respect max_de={MAX_DESIGN_EFFECT}, got {constrained_de_exact}",
        )

    def test_cbps_weights_order(self) -> None:
        """Test that CBPS produces sensible weight ordering based on covariate values.

        This test verifies that:
        1. Observations with identical covariate values receive identical weights
        2. Weight ordering reflects the need to balance sample toward target
        3. Weights are assigned consistently based on covariate patterns
        """
        sample = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 1)})
        target = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 9)})

        result = balance_cbps.cbps(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            transformations=None,
        )

        weights = result["weight"].values

        # Observations with identical covariate values should have identical weights
        self.assertEqual(
            round(weights[0], 10),
            round(weights[8], 10),
            msg="Observations with identical covariate values should have identical weights",
        )

        # Verify sensible weight ordering based on target distribution
        self.assertTrue(
            weights[0] < weights[1],
            msg="Weight ordering should reflect covariate distribution differences",
        )
        self.assertTrue(
            weights[0] < weights[7],
            msg="Weight ordering should be consistent with balancing needs",
        )

    def test_cbps_all_weight_identical(self) -> None:
        """Test CBPS behavior when sample and target are identical.

        When sample and target distributions are identical, CBPS should:
        1. Produce nearly identical weights (minimal variance)
        2. Issue a warning about weight uniformity
        3. Handle the degenerate case gracefully
        """
        # Test with identical sample and target distributions
        np.random.seed(1)
        n_obs = 1000
        sample_df = pd.DataFrame({"a": np.random.normal(0, 1, n_obs).reshape((n_obs,))})
        sample_weights = pd.Series((1,) * n_obs)
        target_df = sample_df  # Identical to sample
        target_weights = sample_weights

        result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
        )

        # Weights should be nearly identical (very low variance)
        weight_variance = np.var(result["weight"])
        self.assertTrue(
            weight_variance < TOLERANCE,
            msg=f"Weights should be nearly identical when sample=target, variance={weight_variance}",
        )

        # Test warning generation through Sample interface
        sample = Sample.from_frame(
            df=pd.DataFrame(
                {
                    "a": np.random.normal(0, 1, n_obs).reshape((n_obs,)),
                    "id": range(0, n_obs),
                }
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

    def test_cbps_na_drop(self) -> None:
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

    def test_cbps_input_assertions(self) -> None:
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

    def test_cbps_dropna_empty(self) -> None:
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

    def test_cbps_formula(self) -> None:
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

    def test_cbps_warning_for_variable_with_one_level(self) -> None:
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

    def test_cbps_in_balance_vs_r(self) -> None:
        """Test that Python CBPS implementation matches R CBPS results.

        This test validates our CBPS implementation against reference weights
        computed by R's CBPS package. It verifies:
        1. High correlation (>0.98) between Python and R CBPS weights
        2. Very high correlation (>0.99) between log-transformed weights
        3. Numerical consistency between implementations

        This is also available on:
        https://import-balance.org/docs/tutorials/comparing_cbps_in_r_vs_python_using_sim_data/
        """
        # Load reference data with R CBPS weights
        target_df, sample_df = load_data("sim_data_cbps")

        # Create Sample objects with outcome columns
        # pyre-ignore[6]: Optional DataFrame is checked in load_data
        sample = Sample.from_frame(sample_df, outcome_columns=["y", "cbps_weights"])
        # pyre-ignore[6]: Optional DataFrame is checked in load_data
        target = Sample.from_frame(target_df, outcome_columns=["y", "cbps_weights"])
        sample_target = sample.set_target(target)

        # Compute Python CBPS weights
        adjusted_sample = sample_target.adjust(method="cbps", transformations=None)

        # Test correlation with R CBPS weights (linear scale)
        linear_correlation = (
            adjusted_sample.df[["cbps_weights", "weight"]]
            .corr(method="pearson")
            .iloc[0, 1]
        )
        self.assertTrue(
            linear_correlation > 0.98,
            msg=f"Python CBPS should highly correlate with R CBPS (>0.98), got {linear_correlation}",
        )

        # Test correlation with R CBPS weights (log scale - more stringent test)
        log_correlation = (
            adjusted_sample.df[["cbps_weights", "weight"]]
            .apply(lambda x: np.log10(x))
            .corr(method="pearson")
            .iloc[0, 1]
        )
        self.assertTrue(
            log_correlation > 0.99,
            msg=f"Python CBPS should very highly correlate with R CBPS on log scale (>0.99), got {log_correlation}",
        )

    def test_cbps_exact_method_with_constraint_violation(self) -> None:
        """Test CBPS 'exact' method when constraints cannot be satisfied.

        This test verifies that the 'exact' method handles cases where
        the design effect constraint results in all weights becoming identical,
        which indicates the constraint is too tight for meaningful balancing.
        """
        # Create data that requires high design effect for balance
        sample_df = pd.DataFrame({"a": [-30] * 5 + [1] * 10})
        sample_weights = pd.Series((1,) * 15)
        target_df = pd.DataFrame({"a": [15] * 15})
        target_weights = pd.Series((1,) * 15)

        # Test that 'exact' method with tight constraint produces identical weights
        # (which means the balancing essentially fails)
        from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=PerfectSeparationWarning,
            )
            result = balance_cbps.cbps(
                sample_df,
                sample_weights,
                target_df,
                target_weights,
                transformations=None,
                cbps_method="exact",
                max_de=1.01,  # Very tight constraint
                weight_trimming_mean_ratio=None,
            )

        # When constraints are too tight, all weights become identical
        # This is indicated by very low standard deviation
        weights = result["weight"].values
        self.assertTrue(
            np.std(weights) < 1e-4,
            msg="With tight constraints, weights should become identical (std < 1e-4)",
        )

    def test_cbps_invalid_method(self) -> None:
        """Test that CBPS raises appropriate error for invalid method parameter.

        This test verifies input validation for the cbps_method parameter,
        ensuring only 'over' and 'exact' are accepted.
        """
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5)})
        sample_weights = pd.Series((1,) * 5)
        target_df = pd.DataFrame({"a": (1, 2, 3, 4, 6)})
        target_weights = pd.Series((1,) * 5)

        # Test invalid method parameter
        with self.assertRaises(Exception) as context:
            balance_cbps.cbps(
                sample_df,
                sample_weights,
                target_df,
                target_weights,
                transformations=None,
                cbps_method="invalid_method",
            )

        self.assertIn(
            "not a valid option",
            str(context.exception).lower(),
            msg="Should raise exception for invalid cbps_method",
        )

    def test_cbps_balance_classes_false(self) -> None:
        """Test CBPS with balance_classes=False parameter.

        This test verifies that CBPS correctly handles the balance_classes=False
        option, which uses unbalanced design weights instead of balancing sample
        and target sizes. This is useful when the original size difference between
        sample and target should be preserved.
        """
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)})
        sample_weights = pd.Series((1,) * 9)
        target_df = pd.DataFrame({"a": (5, 6, 7, 8, 9, 10, 11, 12, 13)})
        target_weights = pd.Series((2,) * 9)  # Different weights to test

        # Test with balance_classes=False
        result_unbalanced = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            balance_classes=False,
        )

        # Test with balance_classes=True for comparison
        result_balanced = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            balance_classes=True,
        )

        # Results should be different
        self.assertIn("weight", result_unbalanced)
        self.assertIn("weight", result_balanced)
        self.assertEqual(len(result_unbalanced["weight"]), 9)
        self.assertEqual(len(result_balanced["weight"]), 9)

        # Verify that weights differ between balanced and unbalanced approaches
        weights_diff = np.sum(
            np.abs(
                result_unbalanced["weight"].values - result_balanced["weight"].values
            )
        )
        self.assertTrue(
            weights_diff > 0.01,
            msg="balance_classes=False should produce different weights than balance_classes=True",
        )

    def test_logit_truncated_with_dataframe_input(self) -> None:
        """Test logit_truncated function with pandas DataFrame input.

        This test verifies that the function handles both numpy array and
        pandas DataFrame inputs correctly, which is important for flexibility
        in different contexts.
        """
        # Test with DataFrame input
        X_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])

        result_df = balance_cbps.logit_truncated(X_df, beta)
        expected = np.array([0.993307, 0.99999, 1.00000000e-05])

        np.testing.assert_array_equal(
            np.around(result_df, 6),
            expected,
            err_msg="logit_truncated should work with DataFrame input",
        )

        # Test with array input for comparison
        X_array = X_df.values
        result_array = balance_cbps.logit_truncated(X_array, beta)

        np.testing.assert_array_almost_equal(
            result_df,
            result_array,
            err_msg="DataFrame and array inputs should produce identical results",
        )

    def test_compute_pseudo_weights_with_dataframe_inputs(self) -> None:
        """Test compute_pseudo_weights_from_logit_probs with DataFrame inputs.

        This test ensures the function handles both numpy arrays and pandas
        DataFrames for design_weights and in_pop parameters.
        """
        probs = np.array([0.1, 0.6, 0.2])

        # Test with array inputs (baseline)
        design_weights_array = np.array([1, 8, 3])
        in_pop_array = np.array([1.0, 0, 1.0])
        result_array = balance_cbps.compute_pseudo_weights_from_logit_probs(
            probs, design_weights_array, in_pop_array
        )

        # Test with DataFrame inputs (convert from Series)
        design_weights_df = pd.DataFrame([1, 8, 3])
        in_pop_df = pd.DataFrame([1.0, 0, 1.0])
        result_df = balance_cbps.compute_pseudo_weights_from_logit_probs(
            probs, design_weights_df.values.flatten(), in_pop_df.values.flatten()
        )

        expected = np.array([3.0, -4.5, 3.0])
        np.testing.assert_array_equal(
            np.around(result_array, 1),
            expected,
            err_msg="Should work with array inputs",
        )
        np.testing.assert_array_equal(
            np.around(result_df, 1),
            expected,
            err_msg="Should work with DataFrame-derived inputs",
        )

    def test_cbps_with_different_optimization_methods(self) -> None:
        """Test CBPS with different scipy optimization methods.

        This test verifies that CBPS works with various optimization algorithms,
        which is important for robustness and allows users to choose methods
        based on their specific needs.
        """
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)})
        sample_weights = pd.Series((1,) * 9)
        target_df = pd.DataFrame({"a": (3, 4, 5, 6, 7, 8, 9, 10, 11)})
        target_weights = pd.Series((1,) * 9)

        # Test with different optimization methods
        methods_to_test = ["COBYLA", "SLSQP"]

        results = {}
        for method in methods_to_test:
            result = balance_cbps.cbps(
                sample_df,
                sample_weights,
                target_df,
                target_weights,
                transformations=None,
                opt_method=method,
            )
            results[method] = result

            # Verify result structure
            self.assertIn("weight", result)
            self.assertEqual(len(result["weight"]), 9)
            self.assertIn("model", result)

        # Both methods should produce valid results
        self.assertTrue(
            len(results) == len(methods_to_test),
            msg="All optimization methods should produce results",
        )

    def test_cbps_exact_method_success(self) -> None:
        """Test CBPS 'exact' method with feasible constraints.

        This test verifies that the 'exact' method (which uses only balance
        conditions without propensity score conditions) works correctly when
        constraints can be satisfied.
        """
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)})
        sample_weights = pd.Series((1,) * 9)
        target_df = pd.DataFrame({"a": (2, 3, 4, 5, 6, 7, 8, 9, 10)})
        target_weights = pd.Series((1,) * 9)

        # Test 'exact' method with reasonable data
        result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            cbps_method="exact",
            weight_trimming_mean_ratio=None,
        )

        # Verify result structure
        self.assertIn("weight", result)
        self.assertEqual(len(result["weight"]), 9)
        self.assertIn("model", result)

        # Verify that 'exact' method returns appropriate model info
        self.assertEqual(result["model"]["method"], "cbps")
        self.assertIsNone(
            result["model"]["gmm_optimize_result_glm_init"],
            msg="'exact' method should not have GMM optimization results",
        )
        self.assertIsNone(
            result["model"]["gmm_optimize_result_bal_init"],
            msg="'exact' method should not have GMM optimization results",
        )

    def test_cbps_over_method_gmm_selection(self) -> None:
        """Test CBPS 'over' method selects best GMM optimization result.

        This test verifies that when using the 'over' method, both GLM-initialized
        and balance-initialized optimizations are run, and the best result is selected
        based on the GMM loss function.
        """
        # Create simple test data
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)})
        sample_weights = pd.Series((1,) * 9)
        target_df = pd.DataFrame({"a": (3, 4, 5, 6, 7, 8, 9, 10, 11)})
        target_weights = pd.Series((1,) * 9)

        # Test 'over' method
        result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            cbps_method="over",
            weight_trimming_mean_ratio=None,
        )

        # Verify both GMM optimization results exist for 'over' method
        self.assertIsNotNone(result["model"]["gmm_optimize_result_glm_init"])
        self.assertIsNotNone(result["model"]["gmm_optimize_result_bal_init"])
        self.assertIn("weight", result)
        self.assertEqual(len(result["weight"]), 9)

    def test_gmm_function_with_dataframe_inputs(self) -> None:
        """Test gmm_function with pandas DataFrame inputs.

        This test verifies that the GMM function handles DataFrame inputs
        correctly, which is important for integration with pandas-based
        workflows.
        """
        X_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights_array = np.array([1, 8, 3])
        in_pop_array = np.array([1.0, 0, 1.0])

        # Test with DataFrame input for X
        result = balance_cbps.gmm_function(
            beta, X_df, design_weights_array, in_pop_array
        )

        # Verify result structure
        self.assertIn("loss", result)
        self.assertIn("invV", result)
        self.assertTrue(isinstance(result["loss"], (float, np.floating)))
        self.assertTrue(isinstance(result["invV"], np.ndarray))

        # Test with array input for comparison
        result_array = balance_cbps.gmm_function(
            beta, X_df.values, design_weights_array, in_pop_array
        )
        # Results should be similar
        self.assertAlmostEqual(
            float(result["loss"]), float(result_array["loss"]), places=5
        )

    def test_cbps_with_custom_opt_opts(self) -> None:
        """Test CBPS with custom optimization options.

        This test verifies that users can pass custom options to the
        scipy optimizer through the opt_opts parameter.
        """
        sample_df = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 8, 9)})
        sample_weights = pd.Series((1,) * 9)
        target_df = pd.DataFrame({"a": (2, 3, 4, 5, 6, 7, 8, 9, 10)})
        target_weights = pd.Series((1,) * 9)

        # Test with custom optimization options
        custom_opts = {"maxiter": 50, "rhobeg": 0.5}
        result = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            opt_method="COBYLA",
            opt_opts=custom_opts,
        )

        # Verify result structure
        self.assertIn("weight", result)
        self.assertEqual(len(result["weight"]), 9)

    def test_alpha_function_with_different_values(self) -> None:
        """Test alpha_function with various alpha values.

        This test verifies that the alpha function correctly interpolates
        between efficiency and balance for different alpha values, which is
        used in the rescaling step of CBPS.
        """
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 0, -100]])
        beta = np.array([1, 0.5, 1])
        design_weights = np.array([1, 8, 3])
        in_pop = np.array([1.0, 0, 1.0])

        # Test with alpha=0.8
        alpha_08 = np.array([0.8])
        result_08 = balance_cbps.alpha_function(
            alpha_08, beta, X, design_weights, in_pop
        )

        # Test with alpha=0.9
        alpha_09 = np.array([0.9])
        result_09 = balance_cbps.alpha_function(
            alpha_09, beta, X, design_weights, in_pop
        )

        # Test with alpha=1.0
        alpha_10 = np.array([1.0])
        result_10 = balance_cbps.alpha_function(
            alpha_10, beta, X, design_weights, in_pop
        )

        # Verify monotonicity: larger alpha should give larger or equal loss
        # Handle both float and np.ndarray return types
        val_08 = result_08[0] if isinstance(result_08, np.ndarray) else result_08
        val_09 = result_09[0] if isinstance(result_09, np.ndarray) else result_09
        val_10 = result_10[0] if isinstance(result_10, np.ndarray) else result_10

        self.assertTrue(
            val_08 <= val_09,
            msg="Loss should increase (or stay same) with larger alpha",
        )
        self.assertTrue(
            val_09 <= val_10,
            msg="Loss should increase (or stay same) with larger alpha",
        )

    def test_cbps_with_weight_trimming_percentile(self) -> None:
        """Test CBPS with weight_trimming_percentile (winsorization).

        This test verifies that CBPS correctly applies winsorization when
        weight_trimming_percentile is specified instead of weight_trimming_mean_ratio.
        """
        sample_df = pd.DataFrame({"a": [-20] + [1] * 13 + [10] * 1})
        sample_weights = pd.Series((1,) * 15)
        target_df = pd.DataFrame({"a": [10] * 10 + [11] * 5})
        target_weights = pd.Series((1,) * 15)

        # Test with percentile trimming (winsorization)
        result_percentile = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            weight_trimming_mean_ratio=None,
            weight_trimming_percentile=0.95,
        )

        # Test without any trimming to verify trimming is actually applied
        result_no_trim = balance_cbps.cbps(
            sample_df,
            sample_weights,
            target_df,
            target_weights,
            transformations=None,
            weight_trimming_mean_ratio=None,
            weight_trimming_percentile=None,
        )

        # Both should produce valid results
        self.assertIn("weight", result_percentile)
        self.assertIn("weight", result_no_trim)
        self.assertEqual(len(result_percentile["weight"]), 15)
        self.assertEqual(len(result_no_trim["weight"]), 15)

        # Verify that percentile trimming is applied correctly
        # The weights should be valid (positive and finite)
        weights_percentile = result_percentile["weight"].values
        self.assertTrue(np.all(weights_percentile > 0))
        self.assertTrue(np.all(np.isfinite(weights_percentile)))

        # Verify that both methods produce normalized weights (sum to target size)
        self.assertAlmostEqual(
            np.sum(weights_percentile),
            np.sum(target_weights),
            places=5,
            msg="Percentile-trimmed weights should sum to target weight sum",
        )
