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


class TestIPW(
    balance.testutil.BalanceTestCase,
):
    """Test suite for Inverse Propensity Weighting (IPW) functionality."""

    def test_ipw_weights_order(self):
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

    def test_ipw_different_sample_sizes(self):
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

    def test_ipw_adjustment_warnings(self):
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

    def test_ipw_na_drop_behavior(self):
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

    def test_weights_from_link_function(self):
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
            link_values, False, (1, 1, 1), target_weights
        )
        expected_weights = np.array((1 / np.exp(1), 1 / np.exp(2), 1 / np.exp(3)))
        expected_weights = (
            expected_weights * np.sum(target_weights) / np.sum(expected_weights)
        )

        self.assertEqual(result_weights, expected_weights)
        self.assertEqual(result_weights.shape, (3,))

        # Test that balance_classes does nothing when classes have same sum weights
        result_balanced = balance_ipw.weights_from_link(
            link_values, True, (1, 1, 1), (1, 2)
        )
        self.assertEqual(result_balanced, expected_weights)

        # Test balance_classes with different class weights
        target_weights_unbalanced = (1, 2, 3)
        result_class_balanced = balance_ipw.weights_from_link(
            link_values,
            True,
            (1, 1, 1),
            target_weights_unbalanced,
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
            link_values, False, (2, 2, 2), target_weights_scaled
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
            (1,) * large_sample_size,
            (1,),
            weight_trimming_percentile=(0, 0.11),
            keep_sum_of_weights=False,
        )
        # Verify that trimming limits extreme weights
        self.assertTrue(result_trimmed.max() < 0.9)

    def test_ipw_input_validation(self):
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

    def test_ipw_dropna_empty_result(self):
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

    def test_ipw_custom_formula(self):
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

    def _build_mixed_dataframe(self, size, is_sample=True):
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

    def test_ipw_consistency_with_default_arguments(self):
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
        sample_df = self._build_mixed_dataframe(sample_size, is_sample=True)
        # Create target DataFrame with different distribution
        target_df = self._build_mixed_dataframe(target_size, is_sample=False)

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
        self.maxDiff = None
        self.assertEqual(round(weights[15], 4), 0.4575)
        self.assertEqual(round(weights[995], 4), 0.4059)

        # Check overall weight distribution statistics
        expected_stats = np.array(
            [1000, 1.0167, 0.7159, 0.0003, 0.4292, 0.8928, 1.4316, 2.5720]
        )
        actual_stats = np.around(weights.describe().values, 4)
        self.assertEqual(actual_stats, expected_stats)

        # Verify model performance metrics
        model = result["model"]

        # Check propensity model performance
        prop_dev_explained = np.around(model["perf"]["prop_dev_explained"], 5)
        self.assertEqual(prop_dev_explained, 0.27296)

        # Check regularization parameter
        lambda_value = np.around(model["lambda"], 5)
        self.assertEqual(lambda_value, 0.52831)

        # Check regularization performance metrics
        best_trim = model["regularisation_perf"]["best"]["trim"]
        self.assertEqual(best_trim, 2.5)
