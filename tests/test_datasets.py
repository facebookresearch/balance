# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd
from balance.datasets import load_cbps_data, load_data, load_sim_data
from balance.util import _verify_value_type


class TestDatasets(
    balance.testutil.BalanceTestCase,
):
    def test_load_data(self) -> None:
        target_df, sample_df = load_data()

        sample_df = _verify_value_type(sample_df)
        target_df = _verify_value_type(target_df)

        self.assertEqual(sample_df.shape, (1000, 5))
        self.assertEqual(target_df.shape, (10000, 5))

        self.assertEqual(
            target_df.columns.to_numpy().tolist(),
            ["id", "gender", "age_group", "income", "happiness"],
        )
        self.assertEqual(
            sample_df.columns.to_numpy().tolist(),
            ["id", "gender", "age_group", "income", "happiness"],
        )

        o = sample_df.head().round(2).to_dict()
        e = {
            "id": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
            "gender": {0: "Male", 1: "Female", 2: "Male", 3: np.nan, 4: np.nan},
            "age_group": {0: "25-34", 1: "18-24", 2: "18-24", 3: "18-24", 4: "18-24"},
            "income": {0: 6.43, 1: 9.94, 2: 2.67, 3: 10.55, 4: 2.69},
            "happiness": {0: 26.04, 1: 66.89, 2: 37.09, 3: 49.39, 4: 72.3},
        }
        self.assertEqual(o.__str__(), e.__str__())
        # NOTE: using .__str__() since doing o==e will give False

        o = target_df.head().round(2).to_dict()
        e = {
            "id": {0: "100000", 1: "100001", 2: "100002", 3: "100003", 4: "100004"},
            "gender": {0: "Male", 1: "Male", 2: "Male", 3: np.nan, 4: np.nan},
            "age_group": {0: "45+", 1: "45+", 2: "35-44", 3: "45+", 4: "25-34"},
            "income": {0: 10.18, 1: 6.04, 2: 5.23, 3: 5.75, 4: 4.84},
            "happiness": {0: 61.71, 1: 79.12, 2: 44.21, 3: 83.99, 4: 49.34},
        }
        self.assertEqual(o.__str__(), e.__str__())

    def test_load_data_cbps(self) -> None:
        target_df, sample_df = load_data("sim_data_cbps")

        sample_df = _verify_value_type(sample_df)
        target_df = _verify_value_type(target_df)

        self.assertEqual(sample_df.shape, (246, 7))
        self.assertEqual(target_df.shape, (254, 7))

        self.assertEqual(
            target_df.columns.to_numpy().tolist(),
            ["X1", "X2", "X3", "X4", "cbps_weights", "y", "id"],
        )
        self.assertEqual(
            sample_df.columns.to_numpy().tolist(),
            target_df.columns.to_numpy().tolist(),
        )

        o = sample_df.head().round(2).to_dict()
        e = {
            "X1": {0: 1.07, 2: 0.69, 4: 0.5, 5: 1.52, 6: 1.03},
            "X2": {0: 10.32, 2: 10.65, 4: 9.59, 5: 10.03, 6: 9.79},
            "X3": {0: 0.21, 2: 0.22, 4: 0.23, 5: 0.33, 6: 0.22},
            "X4": {0: 463.28, 2: 424.29, 4: 472.85, 5: 438.38, 6: 436.39},
            "cbps_weights": {0: 0.01, 2: 0.01, 4: 0.01, 5: 0.0, 6: 0.0},
            "y": {0: 227.53, 2: 196.89, 4: 191.3, 5: 280.45, 6: 227.07},
            "id": {0: 1, 2: 3, 4: 5, 5: 6, 6: 7},
        }
        self.assertEqual(o.__str__(), e.__str__())
        # NOTE: using .__str__() since doing o==e will give False

        o = target_df.head().round(2).to_dict()
        e = {
            "X1": {1: 0.72, 3: 0.35, 11: 0.69, 12: 0.78, 13: 0.82},
            "X2": {1: 9.91, 3: 9.91, 11: 10.73, 12: 9.56, 13: 9.8},
            "X3": {1: 0.19, 3: 0.1, 11: 0.21, 12: 0.18, 13: 0.21},
            "X4": {1: 383.76, 3: 399.37, 11: 398.31, 12: 370.18, 13: 434.45},
            "cbps_weights": {1: 0.0, 3: 0.0, 11: 0.0, 12: 0.0, 13: 0.0},
            "y": {1: 199.82, 3: 174.69, 11: 189.58, 12: 208.18, 13: 214.28},
            "id": {1: 2, 3: 4, 11: 12, 12: 13, 13: 14},
        }
        self.assertEqual(o.__str__(), e.__str__())

    # Comprehensive tests for load_sim_data function
    def test_load_sim_data_structure_and_types(self) -> None:
        """Test that load_sim_data returns DataFrames with correct structure and types."""
        target_df, sample_df = load_sim_data(version="01")

        # Verify we got DataFrames
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        self.assertIsInstance(target_df, pd.DataFrame)
        self.assertIsInstance(sample_df, pd.DataFrame)

        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)

        # Check dimensions
        self.assertEqual(target_df.shape, (10000, 5))
        self.assertEqual(sample_df.shape, (1000, 5))

        # Check column names
        expected_columns = ["id", "gender", "age_group", "income", "happiness"]
        self.assertEqual(target_df.columns.tolist(), expected_columns)
        self.assertEqual(sample_df.columns.tolist(), expected_columns)

        # Check column types for both dataframes
        for df in [target_df, sample_df]:
            self.assertTrue(pd.api.types.is_string_dtype(df["id"]))
            self.assertTrue(pd.api.types.is_string_dtype(df["gender"]))
            self.assertTrue(pd.api.types.is_string_dtype(df["age_group"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(df["income"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(df["happiness"]))

    def test_load_sim_data_reproducibility(self) -> None:
        """Test that load_sim_data returns identical results on multiple calls."""
        target_df1, sample_df1 = load_sim_data(version="01")
        target_df2, sample_df2 = load_sim_data(version="01")

        target_df1 = _verify_value_type(target_df1)
        sample_df1 = _verify_value_type(sample_df1)
        target_df2 = _verify_value_type(target_df2)
        sample_df2 = _verify_value_type(sample_df2)

        # Check that the DataFrames are identical
        pd.testing.assert_frame_equal(target_df1, target_df2)
        pd.testing.assert_frame_equal(sample_df1, sample_df2)

    def test_load_sim_data_data_validity(self) -> None:
        """Test that load_sim_data returns data with valid values and expected properties."""
        target_df, sample_df = load_sim_data(version="01")

        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)

        # Test gender values and missing data
        expected_gender_values = {"Male", "Female"}
        self.assertEqual(
            set(target_df["gender"].dropna().unique()), expected_gender_values
        )
        self.assertEqual(
            set(sample_df["gender"].dropna().unique()), expected_gender_values
        )

        # Test that some reasonable proportion of gender values are missing (not exact counts)
        target_missing_prop = target_df["gender"].isna().sum() / len(target_df)
        sample_missing_prop = sample_df["gender"].isna().sum() / len(sample_df)
        self.assertGreater(target_missing_prop, 0.05)  # At least 5% missing
        self.assertLess(target_missing_prop, 0.15)  # Less than 15% missing
        self.assertGreater(sample_missing_prop, 0.05)  # At least 5% missing
        self.assertLess(sample_missing_prop, 0.15)  # Less than 15% missing

        # Test age group values
        expected_age_groups = {"18-24", "25-34", "35-44", "45+"}
        self.assertEqual(
            set(target_df["age_group"].dropna().unique()), expected_age_groups
        )
        self.assertEqual(
            set(sample_df["age_group"].dropna().unique()), expected_age_groups
        )

        # Test value ranges
        # Happiness should be between 0 and 100
        self.assertTrue((target_df["happiness"] >= 0).all())
        self.assertTrue((target_df["happiness"] <= 100).all())
        self.assertTrue((sample_df["happiness"] >= 0).all())
        self.assertTrue((sample_df["happiness"] <= 100).all())

        # Income should be non-negative (squared normal distribution)
        self.assertTrue((target_df["income"] >= 0).all())
        self.assertTrue((sample_df["income"] >= 0).all())

        # Test ID uniqueness and format
        self.assertEqual(len(target_df["id"]), len(target_df["id"].unique()))
        self.assertEqual(len(sample_df["id"]), len(sample_df["id"].unique()))

        # Test that IDs are strings and have reasonable lengths
        self.assertTrue(all(isinstance(id_val, str) for id_val in target_df["id"]))
        self.assertTrue(all(isinstance(id_val, str) for id_val in sample_df["id"]))

    def test_load_sim_data_invalid_versions(self) -> None:
        """Test that invalid versions return (None, None)."""
        invalid_versions = ["invalid", "", "02", "2", "v1", None]

        for version in invalid_versions:
            if version is None:
                # Skip None as it's not a valid string argument
                continue
            target_df, sample_df = load_sim_data(version=version)
            self.assertIsNone(target_df, f"Expected None for version '{version}'")
            self.assertIsNone(sample_df, f"Expected None for version '{version}'")

    # Comprehensive tests for load_cbps_data function
    def test_load_cbps_data_structure_and_types(self) -> None:
        """Test that load_cbps_data returns DataFrames with correct structure and types."""
        target_df, sample_df = load_cbps_data()

        # Verify we got DataFrames
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        self.assertIsInstance(target_df, pd.DataFrame)
        self.assertIsInstance(sample_df, pd.DataFrame)

        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)

        # Check dimensions
        self.assertEqual(target_df.shape, (254, 7))
        self.assertEqual(sample_df.shape, (246, 7))

        # Check column names
        expected_columns = ["X1", "X2", "X3", "X4", "cbps_weights", "y", "id"]
        self.assertEqual(target_df.columns.tolist(), expected_columns)
        self.assertEqual(sample_df.columns.tolist(), expected_columns)

        # Check column types - all should be numeric
        for col in ["X1", "X2", "X3", "X4", "cbps_weights", "y", "id"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(target_df[col]))
            self.assertTrue(pd.api.types.is_numeric_dtype(sample_df[col]))

    def test_load_cbps_data_reproducibility(self) -> None:
        """Test that load_cbps_data returns identical results on multiple calls."""
        target_df1, sample_df1 = load_cbps_data()
        target_df2, sample_df2 = load_cbps_data()

        target_df1 = _verify_value_type(target_df1)
        sample_df1 = _verify_value_type(sample_df1)
        target_df2 = _verify_value_type(target_df2)
        sample_df2 = _verify_value_type(sample_df2)

        # Check that the DataFrames are identical
        pd.testing.assert_frame_equal(target_df1, target_df2)
        pd.testing.assert_frame_equal(sample_df1, sample_df2)

    def test_load_cbps_data_data_validity(self) -> None:
        """Test that load_cbps_data returns data with valid values and properties."""
        target_df, sample_df = load_cbps_data()

        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)

        # No missing values should be present
        self.assertEqual(target_df.isna().sum().sum(), 0)
        self.assertEqual(sample_df.isna().sum().sum(), 0)

        # All IDs should be unique within each DataFrame
        self.assertEqual(len(target_df["id"]), len(target_df["id"].unique()))
        self.assertEqual(len(sample_df["id"]), len(sample_df["id"].unique()))

        # CBPS weights should be non-negative
        self.assertTrue((target_df["cbps_weights"] >= 0).all())
        self.assertTrue((sample_df["cbps_weights"] >= 0).all())

        # Verify that dataframes have substantial number of rows (not testing exact count)
        self.assertGreater(
            len(target_df), 100, "Target DataFrame should have substantial data"
        )
        self.assertGreater(
            len(sample_df), 100, "Sample DataFrame should have substantial data"
        )

    # Comprehensive tests for load_data wrapper function
    def test_load_data_routing(self) -> None:
        """Test that load_data correctly routes to appropriate data loading functions."""
        # Test default (should return sim_data_01)
        target_df, sample_df = load_data()
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)
        self.assertEqual(target_df.shape, (10000, 5))
        self.assertEqual(sample_df.shape, (1000, 5))

        # Test explicit sim_data_01
        target_df, sample_df = load_data(source="sim_data_01")
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)
        self.assertEqual(target_df.shape, (10000, 5))
        self.assertEqual(sample_df.shape, (1000, 5))

        # Test sim_data_cbps
        target_df, sample_df = load_data(source="sim_data_cbps")
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        target_df = _verify_value_type(target_df)
        sample_df = _verify_value_type(sample_df)
        self.assertEqual(target_df.shape, (254, 7))
        self.assertEqual(sample_df.shape, (246, 7))

    def test_load_data_consistency_with_load_sim_data(self) -> None:
        """Test that load_data returns same results as load_sim_data."""
        target_df1, sample_df1 = load_data(source="sim_data_01")
        target_df2, sample_df2 = load_sim_data(version="01")

        target_df1 = _verify_value_type(target_df1)
        sample_df1 = _verify_value_type(sample_df1)
        target_df2 = _verify_value_type(target_df2)
        sample_df2 = _verify_value_type(sample_df2)

        # Should be identical
        pd.testing.assert_frame_equal(target_df1, target_df2)
        pd.testing.assert_frame_equal(sample_df1, sample_df2)

    def test_load_data_consistency_with_load_cbps_data(self) -> None:
        """Test that load_data returns same results as load_cbps_data."""
        target_df1, sample_df1 = load_data(source="sim_data_cbps")
        target_df2, sample_df2 = load_cbps_data()

        target_df1 = _verify_value_type(target_df1)
        sample_df1 = _verify_value_type(sample_df1)
        target_df2 = _verify_value_type(target_df2)
        sample_df2 = _verify_value_type(sample_df2)

        # Should be identical
        pd.testing.assert_frame_equal(target_df1, target_df2)
        pd.testing.assert_frame_equal(sample_df1, sample_df2)

    def test_load_data_invalid_source(self) -> None:
        """Test that load_data returns (None, None) for invalid source."""
        invalid_sources = ["invalid", "sim_data_02", "", "cbps", "sim_01"]

        for source in invalid_sources:
            target_df, sample_df = load_data(source=source)  # type: ignore
            self.assertIsNone(target_df, f"Expected None for source '{source}'")
            self.assertIsNone(sample_df, f"Expected None for source '{source}'")
