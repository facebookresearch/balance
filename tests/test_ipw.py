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

import balance.testutil

import numpy as np
import pandas as pd

from balance.sample_class import Sample
from balance.weighting_methods import ipw as balance_ipw
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class TestIPW(
    balance.testutil.BalanceTestCase,
):
    """Test suite for Inverse Propensity Weighting (IPW) functionality."""

    def test_ipw_weights_order(self) -> None:
        """Test that IPW assigns correct relative weight ordering.

        Tests that identical values in sample and target receive equal weights,
        and that underrepresented values receive higher weights than overrepresented ones.
        """
        # Create sample with duplicate values at positions 0 and 8
        sample = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 1)})
        # Target has extra '9' value, making it overrepresented
        target = pd.DataFrame({"a": (1, 2, 3, 4, 5, 6, 7, 9, 9)})

        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            transformations=None,
            max_de=1.5,
        )

        weights = result["weight"].values

        # Identical values should have identical weights
        self.assertEqual(weights[0], weights[8])
        # Underrepresented values should have higher weights than overrepresented ones
        self.assertTrue(weights[0] < weights[1])
        self.assertTrue(weights[0] < weights[7])

    def test_ipw_different_sample_sizes(self) -> None:
        """Test IPW behavior with different sample and target sizes.

        Verifies that IPW can handle datasets of different sizes and produces
        reasonable results. The original test was checking edge cases around
        sample size validation.
        """
        # Create sample and target with different sizes and distributions
        sample_size = 1000
        target_size_different = 999

        sample_df = pd.DataFrame(
            {"a": np.random.uniform(0, 1, sample_size), "id": range(0, sample_size)}
        )

        # Target with different size and shifted distribution
        target_df_different_size = pd.DataFrame(
            {
                "a": np.random.uniform(0.5, 1.5, target_size_different),
                "id": range(0, target_size_different),
            }
        )

        sample_weights = pd.Series(np.ones(sample_size))
        target_weights_different = pd.Series(np.ones(target_size_different))

        # This should work without raising an exception
        result = balance_ipw.ipw(
            sample_df=sample_df,
            sample_weights=sample_weights,
            target_df=target_df_different_size,
            target_weights=target_weights_different,
        )

        # Verify that we get reasonable weights
        self.assertIsNotNone(result)
        self.assertIn("weight", result)
        self.assertEqual(len(result["weight"]), sample_size)

        # Test with equal sizes but different distributions - should also work
        target_df_same_size = pd.DataFrame(
            {
                "a": np.random.uniform(0.5, 1.5, sample_size),
                "id": range(0, sample_size),
            }
        )
        target_weights_same = pd.Series(np.ones(sample_size))

        result_same_size = balance_ipw.ipw(
            sample_df=sample_df,
            sample_weights=sample_weights,
            target_df=target_df_same_size,
            target_weights=target_weights_same,
        )

        # Should also work and produce weights
        self.assertIsNotNone(result_same_size)
        self.assertIn("weight", result_same_size)

    def test_ipw_adjustment_warnings(self) -> None:
        """Test that IPW generates appropriate warnings for problematic adjustments.

        Tests warning generation for:
        1. Identical weights (no adjustment needed)
        2. Low model accuracy (poor propensity model fit)
        """
        sample_size = 100

        # Create sample with identical target (should produce identical weights)
        sample = Sample.from_frame(
            df=pd.DataFrame(
                {
                    "a": np.random.normal(0, 1, sample_size).reshape((sample_size,)),
                    "id": range(0, sample_size),
                }
            ),
            id_column="id",
        )
        sample = sample.set_target(sample)

        # Should warn about identical weights when sample equals target
        self.assertWarnsRegexp(
            "All weights are identical. The estimates will not be adjusted",
            sample.adjust,
            method="ipw",
            balance_classes=False,
        )

        # Should warn about low model accuracy when distributions are too similar
        self.assertWarnsRegexp(
            (
                "The propensity model has low fraction null deviance explained "
                ".*. Results may not be accurate"
            ),
            sample.adjust,
            method="ipw",
            balance_classes=False,
        )

    def test_ipw_warns_for_high_cardinality_features_with_nas(self) -> None:
        """IPW should warn when high-cardinality features with NAs lead to equal weights."""

        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": unique_values + [np.nan],
            }
        )

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            result = balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier"],
                num_lambdas=1,
            )

        self.assertTrue(np.allclose(result["weight"], np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_ipw_warns_for_high_cardinality_features_with_nas_when_dropping(
        self,
    ) -> None:
        """The high-cardinality NA warning should surface even when NAs are dropped."""

        unique_values = [f"user_{i}" for i in range(10)]
        sample_df = pd.DataFrame({"identifier": unique_values + [np.nan]})
        target_df = pd.DataFrame({"identifier": unique_values + [np.nan]})

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            result = balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier"],
                num_lambdas=1,
                na_action="drop",
            )

        self.assertTrue(np.allclose(result["weight"], np.ones(len(sample_df) - 1)))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=10" in log
                for log in logs.output
            )
        )

    def test_ipw_warns_for_high_cardinality_categoricals_with_nas(self) -> None:
        """Categorical dtype columns with high cardinality and NAs should be flagged."""

        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(
                    [f"user_{i}" for i in range(9)] + [np.nan], dtype="category"
                ),
            }
        )
        target_df = sample_df.copy()

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            result = balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier"],
                num_lambdas=1,
            )

        self.assertTrue(np.allclose(result["weight"], np.ones(len(sample_df))))
        self.assertTrue(
            any(
                "High-cardinality features detected" in log and "unique=9" in log
                for log in logs.output
            )
        )

    def test_ipw_does_not_flag_low_cardinality_categoricals_with_nas(self) -> None:
        """Low-cardinality categoricals with NAs should not be reported as a cause."""

        sample_df = pd.DataFrame(
            {
                "identifier": pd.Series(["a", "a", "b", np.nan], dtype="category"),
            }
        )
        target_df = sample_df.copy()

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            result = balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier"],
                num_lambdas=1,
            )

        self.assertTrue(np.allclose(result["weight"], np.ones(len(sample_df))))
        self.assertFalse(
            any("High-cardinality features detected" in log for log in logs.output)
        )

    def test_ipw_warns_for_high_cardinality_features_without_nas(self) -> None:
        """High-cardinality categoricals should be reported even without missing values."""

        identifiers = [f"user_{i}" for i in range(8)]
        sample_df = pd.DataFrame(
            {
                "identifier": identifiers,
                "signal": np.concatenate((np.zeros(4), np.ones(4))),
            }
        )
        target_df = pd.DataFrame(
            {
                "identifier": identifiers[::-1],
                "signal": np.concatenate((np.ones(4), np.zeros(4))),
            }
        )

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier", "signal"],
                num_lambdas=1,
            )

        self.assertTrue(
            any(
                "High-cardinality features detected" in log
                and "identifier (unique=8" in log
                for log in logs.output
            )
        )

    def test_ipw_warns_for_high_cardinality_numeric_features(self) -> None:
        """High-cardinality numeric features (e.g., IDs) should be reported."""

        identifiers = np.arange(12)
        sample_df = pd.DataFrame({"identifier": identifiers})
        target_df = pd.DataFrame({"identifier": identifiers})

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["identifier"],
                num_lambdas=1,
            )

        self.assertTrue(
            any(
                "High-cardinality features detected" in log
                and "identifier (unique=12" in log
                for log in logs.output
            ),
            "Expected high-cardinality numeric feature to be reported.",
        )

    def test_ipw_high_cardinality_warning_sorts_by_cardinality(self) -> None:
        """Warnings should list columns from highest to lowest cardinality."""

        sample_df = pd.DataFrame(
            {
                "higher": [f"user_{i}" for i in range(7)],
                "high": [f"alias_{i}" for i in range(6)] + ["alias_0"],
            }
        )
        target_df = sample_df.copy()

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            balance_ipw.ipw(
                sample_df=sample_df,
                sample_weights=pd.Series(np.ones(len(sample_df))),
                target_df=target_df,
                target_weights=pd.Series(np.ones(len(target_df))),
                variables=["higher", "high"],
                num_lambdas=1,
            )

        warning_logs = [
            log for log in logs.output if "High-cardinality features detected" in log
        ]
        self.assertTrue(warning_logs)
        self.assertTrue(
            warning_logs[0].find("higher (unique")
            < warning_logs[0].find(", high (unique"),
            "Expected higher-cardinality column to appear first in warning.",
        )

    def test_ipw_na_drop_behavior(self) -> None:
        """Test that IPW correctly handles and warns about dropping NA values.

        Verifies that when na_action="drop" is specified, IPW properly drops
        rows with missing values and warns about the number of dropped rows.
        """
        # Create sample with 2 NaN values at the beginning
        sample = Sample.from_frame(
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

        # Create target with 1 NaN value at the beginning
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "a": np.concatenate((np.array([np.nan]), np.arange(2, 100))),
                    "id": np.arange(1, 100),
                }
            ),
            id_column="id",
        )

        # Should warn about dropping 2 out of 99 sample rows due to NaN values
        self.assertWarnsRegexp(
            "Dropped 2/99 rows of sample",
            sample.adjust,
            target,
            na_action="drop",
            transformations=None,
        )

    def test_ipw_allows_custom_logistic_regression_model(self) -> None:
        """Users can override LogisticRegression configuration by passing an estimator."""

        sample = pd.DataFrame({"a": (0, 1, 1, 0), "b": (1, 2, 3, 4)})
        target = pd.DataFrame({"a": (1, 0, 0, 1), "b": (4, 3, 2, 1)})

        configured_lr = LogisticRegression(solver="lbfgs", max_iter=500)
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series((1,) * len(sample)),
            target_df=target,
            target_weights=pd.Series((1,) * len(target)),
            model=configured_lr,
            max_de=None,
            num_lambdas=1,
        )

        fit = result["model"]["fit"]
        self.assertIsInstance(fit, LogisticRegression)
        self.assertEqual(fit.solver, "lbfgs")
        self.assertEqual(fit.max_iter, 500)

    def test_ipw_supports_custom_sklearn_model(self) -> None:
        """Custom sklearn models (e.g., RandomForest) can drive propensity scores."""

        rng = np.random.RandomState(1234)
        sample = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 80),
                "b": rng.binomial(1, 0.4, 80),
            }
        )
        target = pd.DataFrame(
            {
                "a": rng.normal(0.5, 1.2, 120),
                "b": rng.binomial(1, 0.6, 120),
            }
        )

        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=7)
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series(np.ones(len(sample))),
            target_df=target,
            target_weights=pd.Series(np.ones(len(target))),
            model=rf,
            transformations=None,
            num_lambdas=1,
            max_de=1.5,
        )

        self.assertIsInstance(result["model"]["fit"], RandomForestClassifier)
        self.assertEqual(len(result["weight"]), len(sample))
        self.assertTrue(np.isnan(result["model"]["lambda"]))
        self.assertTrue(result["model"]["perf"]["coefs"].empty)
        prop_dev = result["model"]["perf"]["prop_dev_explained"]
        self.assertGreaterEqual(prop_dev, 0.0)
        self.assertLessEqual(prop_dev, 1.0)
        self.assertIsNotNone(result["model"]["regularisation_perf"])

    def test_ipw_supports_custom_model_parameter(self) -> None:
        """The ``model`` parameter accepts sklearn classifiers directly."""

        rng = np.random.RandomState(11)
        sample = pd.DataFrame({"a": rng.normal(size=25), "b": rng.binomial(1, 0.3, 25)})
        target = pd.DataFrame({"a": rng.normal(size=40), "b": rng.binomial(1, 0.6, 40)})

        classifier = RandomForestClassifier(
            n_estimators=10, max_depth=2, random_state=4
        )
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series(np.ones(len(sample))),
            target_df=target,
            target_weights=pd.Series(np.ones(len(target))),
            model=classifier,
            transformations=None,
            num_lambdas=1,
            max_de=1.5,
        )

        self.assertIsInstance(result["model"]["fit"], RandomForestClassifier)
        self.assertTrue(np.isnan(result["model"]["lambda"]))
        self.assertEqual(len(result["weight"]), len(sample))

    def test_ipw_supports_dense_only_estimators(self) -> None:
        """Estimators that require dense matrices (e.g., GaussianNB) are supported."""

        rng = np.random.RandomState(42)
        sample = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
        target = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})

        model = GaussianNB()
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series(np.ones(len(sample))),
            target_df=target,
            target_weights=pd.Series(np.ones(len(target))),
            model=model,
            transformations=None,
            num_lambdas=1,
            max_de=1.5,
        )

        weights = result["weight"].to_numpy()
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertIsInstance(result["model"]["fit"], GaussianNB)

    def test_ipw_extreme_probabilities_yield_finite_weights(self) -> None:
        """Models producing 0/1 probabilities result in finite stabilized weights."""

        rng = np.random.RandomState(0)
        sample = pd.DataFrame({"a": rng.binomial(1, 0.5, 50)})
        target = pd.DataFrame({"a": rng.binomial(1, 0.5, 75)})

        tree = DecisionTreeClassifier(random_state=0)
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series(np.ones(len(sample))),
            target_df=target,
            target_weights=pd.Series(np.ones(len(target))),
            model=tree,
            transformations=None,
            num_lambdas=1,
            max_de=1.5,
        )

        weights = result["weight"].to_numpy()
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_ipw_requires_predict_proba_for_custom_model(self) -> None:
        """Custom sklearn models without predict_proba are rejected."""

        sample = pd.DataFrame({"a": (0, 1, 1, 0)})
        target = pd.DataFrame({"a": (1, 0, 0, 1)})

        with self.assertRaisesRegex(ValueError, "predict_proba"):
            balance_ipw.ipw(
                sample_df=sample,
                sample_weights=pd.Series((1,) * len(sample)),
                target_df=target,
                target_weights=pd.Series((1,) * len(target)),
                model=LinearSVC(),
                transformations=None,
                num_lambdas=1,
            )

    def test_ipw_rejects_custom_models_with_single_proba_column(self) -> None:
        """Custom models must return probability estimates for both classes."""

        class SingleColumnRF(RandomForestClassifier):
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                full = super().predict_proba(X)
                return full[:, :1]

        rng = np.random.RandomState(1)
        sample = pd.DataFrame({"a": rng.normal(size=40)})
        target = pd.DataFrame({"a": rng.normal(size=55)})

        with self.assertRaisesRegex(
            ValueError, "probability estimates for both classes"
        ):
            balance_ipw.ipw(
                sample_df=sample,
                sample_weights=pd.Series(np.ones(len(sample))),
                target_df=target,
                target_weights=pd.Series(np.ones(len(target))),
                model=SingleColumnRF(n_estimators=5, random_state=2),
                transformations=None,
                num_lambdas=1,
            )

    def test_ipw_rejects_custom_models_without_binary_classes(self) -> None:
        """Custom models must be trained on labels containing both 0 and 1."""

        class InvalidClassLabelModel(LogisticRegression):
            def fit(self, X, y, sample_weight=None):  # type: ignore[override]
                super().fit(X, y + 2, sample_weight=sample_weight)
                return self

        rng = np.random.RandomState(12)
        sample = pd.DataFrame({"a": rng.normal(size=30)})
        target = pd.DataFrame({"a": rng.normal(size=40)})

        with self.assertRaisesRegex(ValueError, "must be trained on the binary labels"):
            balance_ipw.ipw(
                sample_df=sample,
                sample_weights=pd.Series(np.ones(len(sample))),
                target_df=target,
                target_weights=pd.Series(np.ones(len(target))),
                model=InvalidClassLabelModel(max_iter=50),
                transformations=None,
                num_lambdas=1,
            )

    def test_ipw_warns_when_penalty_factor_with_custom_model(self) -> None:
        """Providing penalty_factor with custom models emits a warning."""

        rng = np.random.RandomState(5)
        sample = pd.DataFrame({"a": rng.normal(size=30)})
        target = pd.DataFrame({"a": rng.normal(size=45)})

        with self.assertLogs(balance_ipw.logger, level="WARNING") as logs:
            balance_ipw.ipw(
                sample_df=sample,
                sample_weights=pd.Series(np.ones(len(sample))),
                target_df=target,
                target_weights=pd.Series(np.ones(len(target))),
                model=RandomForestClassifier(n_estimators=5, random_state=7),
                penalty_factor=[1.0],
                transformations=None,
                num_lambdas=1,
                max_de=1.5,
            )

        self.assertTrue(
            any("penalty_factor is ignored" in message for message in logs.output)
        )

    def test_ipw_rejects_unknown_model_identifier(self) -> None:
        """Non-supported model identifiers raise NotImplementedError."""

        sample = pd.DataFrame({"a": (0, 1)})
        target = pd.DataFrame({"a": (1, 0)})

        with self.assertRaises(NotImplementedError):
            balance_ipw.ipw(
                sample_df=sample,
                sample_weights=pd.Series((1,) * len(sample)),
                target_df=target,
                target_weights=pd.Series((1,) * len(target)),
                model="unsupported-model",
                transformations=None,
                num_lambdas=1,
            )

    def test_model_coefs_handles_linear_and_non_linear_estimators(self) -> None:
        """model_coefs returns coefficients for linear models and empty series otherwise."""

        X = pd.DataFrame({"a": (0, 1, 0, 1), "b": (1, 1, 0, 0)})
        y = np.array((1, 1, 0, 0))
        lr = LogisticRegression().fit(X, y)
        coefs = balance_ipw.model_coefs(lr, feature_names=list(X.columns))["coefs"]
        self.assertListEqual(list(coefs.index), ["intercept", "a", "b"])
        self.assertAlmostEqual(coefs.loc["intercept"], lr.intercept_[0])
        self.assertAlmostEqual(coefs.loc["a"], lr.coef_[0][0])

        rf = RandomForestClassifier(n_estimators=5, random_state=3).fit(X, y)
        rf_coefs = balance_ipw.model_coefs(rf, feature_names=list(X.columns))["coefs"]
        self.assertTrue(rf_coefs.empty)

    def test_weights_from_link_function(self) -> None:
        """Test the weights_from_link function with various scenarios.

        Tests the core weight calculation function that converts logistic
        regression link values to IPW weights, including:
        1. Basic weight calculation
        2. Class balancing behavior
        3. Sample weight scaling
        4. Weight trimming functionality
        """
        # Test basic weight calculation
        link_values = np.array((1, 2, 3)).reshape(3, 1)
        target_weights = (1, 2)

        result_weights = balance_ipw.weights_from_link(
            link_values, False, pd.Series((1, 1, 1)), pd.Series(target_weights)
        )
        expected_weights = np.array((1 / np.exp(1), 1 / np.exp(2), 1 / np.exp(3)))
        expected_weights = (
            expected_weights * np.sum(target_weights) / np.sum(expected_weights)
        )

        self.assertEqual(result_weights, expected_weights)
        self.assertEqual(result_weights.shape, (3,))

        # Test that balance_classes does nothing when classes have same sum weights
        result_balanced = balance_ipw.weights_from_link(
            link_values, True, pd.Series((1, 1, 1)), pd.Series((1, 2))
        )
        self.assertEqual(result_balanced, expected_weights)

        # Test balance_classes with different class weights
        target_weights_unbalanced = (1, 2, 3)
        result_class_balanced = balance_ipw.weights_from_link(
            link_values,
            True,
            pd.Series((1, 1, 1)),
            pd.Series(target_weights_unbalanced),
            keep_sum_of_weights=False,
        )
        # Expected calculation includes log odds adjustment
        expected_class_balanced = np.array(
            (
                1 / np.exp(1 + np.log(1 / 2)),
                1 / np.exp(2 + np.log(1 / 2)),
                1 / np.exp(3 + np.log(1 / 2)),
            )
        )
        expected_class_balanced = (
            expected_class_balanced
            * np.sum(target_weights_unbalanced)
            / np.sum(expected_class_balanced)
        )
        self.assertEqual(result_class_balanced, expected_class_balanced)

        # Test sample weight scaling
        target_weights_scaled = (1, 2)
        result_scaled = balance_ipw.weights_from_link(
            link_values, False, pd.Series((2, 2, 2)), pd.Series(target_weights_scaled)
        )
        expected_scaled = np.array((2 / np.exp(1), 2 / np.exp(2), 2 / np.exp(3)))
        expected_scaled = (
            expected_scaled * np.sum(target_weights_scaled) / np.sum(expected_scaled)
        )
        self.assertEqual(result_scaled, expected_scaled)

        # Test weight trimming functionality
        large_sample_size = 10000
        result_trimmed = balance_ipw.weights_from_link(
            np.random.uniform(0, 1, large_sample_size),
            False,
            pd.Series((1,) * large_sample_size),
            pd.Series((1,)),
            weight_trimming_percentile=0.11,
            keep_sum_of_weights=False,
        )
        # Verify that trimming limits extreme weights
        self.assertTrue(result_trimmed.max() < 0.9)

    def test_ipw_input_validation(self) -> None:
        """Test that IPW properly validates input parameters and raises appropriate errors.

        Tests validation for:
        1. DataFrame type requirements
        2. Length matching between DataFrames and weight Series
        """
        invalid_input = np.array((1,))

        # Should raise TypeError when non-DataFrame is passed
        self.assertRaisesRegex(
            TypeError,
            "must be a pandas DataFrame",
            balance_ipw.ipw,
            invalid_input,
            invalid_input,
            invalid_input,
            invalid_input,
        )

        # Should raise Exception when DataFrame and weights have different lengths
        sample_df = pd.DataFrame({"a": (1, 2), "id": (1, 2)})
        mismatched_weights = pd.Series((1,))  # Length 1 vs DataFrame length 2

        self.assertRaisesRegex(
            Exception,
            "must be the same length",
            balance_ipw.ipw,
            sample_df,
            mismatched_weights,
            sample_df,
            mismatched_weights,
        )

    def test_ipw_dropna_empty_result(self) -> None:
        """Test IPW error handling when dropping NA values results in empty DataFrame.

        When all rows contain NaN values and na_action="drop" is specified,
        IPW should raise an exception indicating that dropping rows led to an empty result.
        """
        # Create DataFrame where all rows have at least one NaN value
        sample_with_all_na = pd.DataFrame(
            {"a": (1, None), "b": (np.nan, 2), "id": (1, 2)}
        )
        sample_weights = pd.Series((1, 2))

        # Should raise exception when dropping NaN rows results in empty DataFrame
        self.assertRaisesRegex(
            Exception,
            "Dropping rows led to empty",
            balance_ipw.ipw,
            sample_with_all_na,
            sample_weights,
            sample_with_all_na,
            sample_weights,
            na_action="drop",
        )

    def test_ipw_custom_formula(self) -> None:
        """Test IPW with custom formula specification.

        Verifies that IPW correctly interprets and applies custom formulas
        for propensity score modeling, including interaction terms.
        """
        # Create sample and target DataFrames with multiple features
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

        # Test IPW with interaction term formula "a : b + c"
        # Disable transformations to get simple column names for testing
        result = balance_ipw.ipw(
            sample_df=sample,
            sample_weights=pd.Series((1,) * 9),
            target_df=target,
            target_weights=pd.Series((1,) * 9),
            formula="a : b + c",
            transformations=None,
        )

        # Verify that the formula was correctly parsed into model matrix columns
        expected_columns = ["a:b", "c"]
        self.assertEqual(result["model"]["X_matrix_columns"], expected_columns)

    def _build_mixed_dataframe(self, size: int, is_sample: bool = True) -> pd.DataFrame:
        """Build DataFrame with continuous and categorical features.

        Args:
            size: Number of rows to generate
            is_sample: If True, uses sample distribution; else uses target distribution

        Returns:
            pd.DataFrame: Mixed data type DataFrame with labeled columns
        """
        if is_sample:
            # Sample distribution: continuous feature from 0-10
            continuous_data = np.random.uniform(0, 10, size=size)
        else:
            # Target distribution: continuous feature from 8-18 (shifted)
            continuous_data = np.random.uniform(8, 18, size=size)

        # Combine continuous, numerical, and categorical features
        mixed_df = pd.concat(
            [
                pd.DataFrame(continuous_data, columns=[0]),
                pd.DataFrame(
                    np.random.uniform(0, 1, size=(size, 4)), columns=range(1, 5)
                ),
                pd.DataFrame(
                    np.random.choice(["level1", "level2", "level3"], size=(size, 5)),
                    columns=range(5, 10),
                ),
            ],
            axis=1,
        )

        # Rename columns to be more readable
        mixed_df = mixed_df.rename(columns={i: "abcdefghij"[i] for i in range(0, 10)})
        return mixed_df

    def test_ipw_consistency_with_default_arguments(self) -> None:
        """Test IPW consistency and reproducibility with default parameters.

        This comprehensive test verifies that IPW produces consistent results
        when run with default arguments. It tests the entire IPW pipeline including:
        - Variable selection (choose_variables)
        - Data transformations (apply_transformations)
        - Model matrix creation (model_matrix)
        - Regularization selection (choose_regularization)
        - Weight calculation (weights_from_link)
        - Cross-validation performance (cv_glmnet_performance)

        A failure in this test may indicate issues in any of these components.
        """
        # Create consistent test datasets
        np.random.seed(2021)  # Fixed seed for reproducible results
        sample_size = 1000
        target_size = 2000

        # Create sample DataFrame with mixed data types
        sample_df: pd.DataFrame = self._build_mixed_dataframe(
            sample_size, is_sample=True
        )
        # Create target DataFrame with different distribution
        target_df: pd.DataFrame = self._build_mixed_dataframe(
            target_size, is_sample=False
        )

        # Generate random weights
        sample_weights = pd.Series(np.random.uniform(0, 1, size=sample_size))
        target_weights = pd.Series(np.random.uniform(0, 1, size=target_size))

        # Run IPW with default arguments
        result = balance_ipw.ipw(
            sample_df, sample_weights, target_df, target_weights, max_de=1.5
        )

        # Verify weight distribution consistency
        weights = result["weight"]

        # Check specific weight values for reproducibility
        # Note: Using assertAlmostEqual to handle floating point precision differences in Python 3.12
        self.maxDiff = None
        self.assertAlmostEqual(round(weights[15], 4), 0.4575, places=3)
        self.assertAlmostEqual(round(weights[995], 4), 0.4059, places=3)

        # Check overall weight distribution statistics
        # Note: Using assertAlmostEqual to handle floating point precision differences in Python 3.12
        expected_stats = np.array(
            [1000, 1.0167, 0.7159, 0.0003, 0.4292, 0.8928, 1.4316, 2.5720]
        )
        actual_stats = np.around(weights.describe().values, 4)
        np.testing.assert_allclose(actual_stats, expected_stats, rtol=1e-3, atol=1e-3)

        # Verify model performance metrics
        model = result["model"]

        # Check propensity model performance
        prop_dev_explained = np.around(model["perf"]["prop_dev_explained"], 5)
        self.assertAlmostEqual(prop_dev_explained, 0.27296, places=4)

        # Check regularization parameter
        lambda_value = np.around(model["lambda"], 5)
        self.assertAlmostEqual(lambda_value, 0.52831, places=4)

        # Check regularization performance metrics
        best_trim = model["regularisation_perf"]["best"]["trim"]
        self.assertEqual(best_trim, 2.5)

    def test_compute_deviance_without_labels(self) -> None:
        """Test _compute_deviance computes 2 * log_loss correctly without labels parameter.

        Verifies that the helper function correctly computes deviance as 2 * log_loss
        when no labels parameter is provided.
        """
        # Setup: Create test data with known values
        y = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        model_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Execute: Compute deviance using the helper function
        result = balance_ipw._compute_deviance(y, pred, model_weights)

        # Assert: Verify result equals 2 * log_loss
        expected = 2 * log_loss(y, pred, sample_weight=model_weights)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertIsInstance(result, float)

    def test_compute_deviance_with_labels(self) -> None:
        """Test _compute_deviance computes 2 * log_loss correctly with labels parameter.

        Verifies that the helper function correctly computes deviance when
        explicit labels are provided.
        """
        # Setup: Create test data with known values
        y = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        model_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        labels = [0, 1]

        # Execute: Compute deviance with labels parameter
        result = balance_ipw._compute_deviance(y, pred, model_weights, labels=labels)

        # Assert: Verify result equals 2 * log_loss with labels
        expected = 2 * log_loss(y, pred, sample_weight=model_weights, labels=labels)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertIsInstance(result, float)

    def test_compute_deviance_with_different_weights(self) -> None:
        """Test _compute_deviance handles non-uniform sample weights correctly.

        Verifies that the deviance calculation properly incorporates
        different sample weights when computing log loss.
        """
        # Setup: Create test data with varying weights
        y = np.array([1, 0, 1, 0, 1])
        pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        model_weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0])

        # Execute: Compute deviance with non-uniform weights
        result = balance_ipw._compute_deviance(y, pred, model_weights)

        # Assert: Verify weighted deviance is computed correctly
        expected = 2 * log_loss(y, pred, sample_weight=model_weights)
        self.assertAlmostEqual(result, expected, places=10)

    def test_compute_proportion_deviance_basic(self) -> None:
        """Test _compute_proportion_deviance computes (1 - dev/null_dev) correctly.

        Verifies basic computation of proportion of deviance explained
        using the formula (1 - dev/null_dev).
        """
        # Setup: Define deviance and null deviance values
        dev = 50.0
        null_dev = 100.0

        # Execute: Compute proportion deviance
        result = balance_ipw._compute_proportion_deviance(dev, null_dev)

        # Assert: Verify result equals 1 - dev/null_dev
        expected = 1 - (50.0 / 100.0)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertEqual(result, 0.5)

    def test_compute_proportion_deviance_perfect_fit(self) -> None:
        """Test _compute_proportion_deviance when dev equals 0 (perfect fit).

        Verifies that when deviance is 0 (perfect model fit),
        the proportion of deviance explained is 1.0.
        """
        # Setup: Perfect fit scenario (dev = 0)
        dev = 0.0
        null_dev = 100.0

        # Execute: Compute proportion deviance for perfect fit
        result = balance_ipw._compute_proportion_deviance(dev, null_dev)

        # Assert: Perfect fit should yield proportion of 1.0
        self.assertEqual(result, 1.0)

    def test_compute_proportion_deviance_no_improvement(self) -> None:
        """Test _compute_proportion_deviance when dev equals null_dev (no improvement).

        Verifies that when model deviance equals null deviance (no improvement
        over null model), the proportion of deviance explained is 0.0.
        """
        # Setup: No improvement scenario (dev = null_dev)
        dev = 100.0
        null_dev = 100.0

        # Execute: Compute proportion deviance for no improvement
        result = balance_ipw._compute_proportion_deviance(dev, null_dev)

        # Assert: No improvement should yield proportion of 0.0
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_compute_proportion_deviance_partial_improvement(self) -> None:
        """Test _compute_proportion_deviance with various partial improvement levels.

        Verifies correct computation of proportion deviance for different
        levels of model improvement over the null model.
        """
        # Setup: Test different improvement levels
        test_cases = [
            (75.0, 100.0, 0.25),  # 25% improvement
            (25.0, 100.0, 0.75),  # 75% improvement
            (90.0, 100.0, 0.10),  # 10% improvement
        ]

        for dev, null_dev, expected in test_cases:
            # Execute: Compute proportion deviance
            result = balance_ipw._compute_proportion_deviance(dev, null_dev)

            # Assert: Verify expected proportion
            self.assertAlmostEqual(result, expected, places=10)
