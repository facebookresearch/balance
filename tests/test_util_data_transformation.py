# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd

# TODO: remove the use of balance_util in most cases, and just import the functions to be tested directly
from balance import util as balance_util
from balance.sample_class import Sample
from balance.util import _verify_value_type


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test_add_na_indicator(self) -> None:
        """Test addition of NA indicator columns to DataFrames.

        Tests the add_na_indicator function's ability to:
        - Add indicator columns for missing values (None, NaN)
        - Handle different data types (numeric, string, categorical)
        - Replace NA values with specified replacement values
        - Handle edge cases and validation errors
        """
        df = pd.DataFrame({"a": (0, None, 2, np.nan), "b": (None, "b", "", np.nan)})
        e = pd.DataFrame(
            {
                "a": (0, 0, 2.0, 0),
                "b": ("_NA", "b", "", "_NA"),
                "_is_na_a": (False, True, False, True),
                "_is_na_b": (True, False, False, True),
            },
            columns=("a", "b", "_is_na_a", "_is_na_b"),
        )
        r = balance_util.add_na_indicator(df)
        self.assertEqual(r, e)

        # No change if no missing variables
        df = pd.DataFrame(
            {"a": (0, 1, 2), "b": ("a", "b", ""), "c": pd.Categorical(("a", "b", "a"))}
        )
        self.assertEqual(balance_util.add_na_indicator(df), df)

        # Test that it works with categorical variables
        df = pd.DataFrame(
            {
                "c": pd.Categorical(("a", "b", "a", "b")),
                "d": pd.Categorical(("a", "b", None, np.nan)),
            }
        )
        e = pd.DataFrame(
            {
                "c": pd.Categorical(("a", "b", "a", "b")),
                "d": pd.Categorical(
                    ("a", "b", "_NA", "_NA"), categories=("a", "b", "_NA")
                ),
                "_is_na_d": (False, False, True, True),
            },
            columns=("c", "d", "_is_na_d"),
        )
        self.assertEqual(balance_util.add_na_indicator(df), e)

        # test arguments
        df = pd.DataFrame({"a": (0, None, 2, np.nan), "b": (None, "b", "", np.nan)})
        e = pd.DataFrame(
            {
                "a": (0.0, 42.0, 2.0, 42.0),
                "b": ("AAA", "b", "", "AAA"),
                "_is_na_a": (False, True, False, True),
                "_is_na_b": (True, False, False, True),
            },
            columns=("a", "b", "_is_na_a", "_is_na_b"),
        )
        r = balance_util.add_na_indicator(df, replace_val_obj="AAA", replace_val_num=42)
        self.assertEqual(r, e)

        # check exceptions
        d = pd.DataFrame({"a": [0, 1, np.nan, None], "b": ["x", "y", "_NA", None]})
        self.assertRaisesRegex(
            Exception,
            "Can't add NA indicator to columns containing NAs and the value '_NA', ",
            balance_util.add_na_indicator,
            d,
        )
        d = pd.DataFrame({"a": [0, 1, np.nan, None], "_is_na_b": ["x", "y", "z", None]})
        self.assertRaisesRegex(
            Exception,
            "Can't add NA indicator to DataFrame which contains",
            balance_util.add_na_indicator,
            d,
        )

    def test_drop_na_rows(self) -> None:
        """Test removal of rows containing NA values from DataFrames.

        Tests the drop_na_rows function's ability to:
        - Remove rows with NA values from both DataFrame and corresponding weights
        - Maintain proper indexing after row removal
        - Handle edge cases where all rows would be removed
        """
        sample_df = pd.DataFrame(
            {"a": (0, None, 2, np.nan), "b": (None, "b", "c", np.nan)}
        )
        sample_weights = pd.Series([1, 2, 3, 4])
        (
            sample_df,
            sample_weights,
        ) = balance_util.drop_na_rows(sample_df, sample_weights, "sample")
        self.assertEqual(sample_df, pd.DataFrame({"a": (2.0), "b": ("c")}, index=[2]))
        self.assertEqual(sample_weights, pd.Series([3], index=[2]))

        # check exceptions
        sample_df = pd.DataFrame({"a": (None), "b": ("b")}, index=[1])
        sample_weights = pd.Series([1])
        self.assertRaisesRegex(
            ValueError,
            "Dropping rows led to empty",
            balance_util.drop_na_rows,
            sample_df,
            sample_weights,
            "sample",
        )

    def test_qcut(self) -> None:
        d = pd.Series([0, 1, 2, 3, 4])
        self.assertEqual(
            balance_util.qcut(d, 4).astype(str),
            pd.Series(
                [
                    "(-0.001, 1.0]",
                    "(-0.001, 1.0]",
                    "(1.0, 2.0]",
                    "(2.0, 3.0]",
                    "(3.0, 4.0]",
                ]
            ),
        )
        self.assertEqual(balance_util.qcut(d, 6), d)
        self.assertWarnsRegexp(
            "Not quantizing, too few values",
            balance_util.qcut,
            d,
            6,
        )

    def test_quantize(self) -> None:
        d = pd.DataFrame(np.random.rand(1000, 2))
        d = d.rename(columns={i: "ab"[i] for i in range(0, 2)})
        d["c"] = ["x"] * 1000

        r = balance_util.quantize(d, variables=["a"])
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], float))
        self.assertEqual(r["c"][0], "x")

        r = balance_util.quantize(d)
        self.assertTrue(isinstance(r["a"][0], pd.Interval))
        self.assertTrue(isinstance(r["b"][0], pd.Interval))
        self.assertEqual(r["c"][0], "x")

        # Test that it does not affect categorical columns
        d["d"] = pd.Categorical(["y"] * 1000)
        r = balance_util.quantize(d)
        self.assertEqual(r["d"][0], "y")

        # Test on Series input
        r = balance_util.quantize(pd.Series(np.random.uniform(0, 1, 100)), 7)
        self.assertEqual(len(set(r.values)), 7)

        # Test on numpy array input
        r = balance_util.quantize(np.random.uniform(0, 1, 100), 7)
        self.assertEqual(len(set(r.values)), 7)

        # Test on single integer input
        r = balance_util.quantize(pd.Series([1]), 1)
        self.assertEqual(len(set(r.values)), 1)

    def test_quantize_preserves_column_order(self) -> None:
        df = pd.DataFrame(
            {
                "first": np.linspace(0.0, 19.0, 20),
                "second": list("abcdefghijklmnopqrst"),
                "third": np.linspace(100.0, 119.0, 20),
            }
        )

        result = balance_util.quantize(df, q=4, variables=["first", "third"])

        self.assertListEqual(list(result.columns), ["first", "second", "third"])
        self.assertIsInstance(result.loc[0, "first"], pd.Interval)
        self.assertEqual(result.loc[0, "second"], "a")
        self.assertIsInstance(result.loc[0, "third"], pd.Interval)

    def test_quantize_non_numeric_series_raises(self) -> None:
        self.assertRaisesRegex(
            TypeError,
            "series must be numeric",
            balance_util.quantize,
            pd.Series(["x", "y", "z"]),
        )

    def test_row_pairwise_diffs(self) -> None:
        d = pd.DataFrame({"a": (1, 2, 3), "b": (-42, 8, 2)})
        e = pd.DataFrame(
            {"a": (1, 2, 3, 1, 2, 1), "b": (-42, 8, 2, 50, 44, -6)},
            index=(0, 1, 2, "1 - 0", "2 - 0", "2 - 1"),
        )
        self.assertEqual(balance_util.row_pairwise_diffs(d), e)

    def test_auto_spread(self) -> None:
        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 1, 2, 2, 4),
            }
        )

        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 2.0, 4.0),
                "key_b_value": (1.0, 2.0, np.nan),
            },
            columns=("id", "key_a_value", "key_b_value"),
        )
        self.assertEqual(expected, balance_util.auto_spread(data))

        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 1, 2, 2, 4),
                "other_value": (2, 2, 4, 4, 6),
            }
        )

        self.assertEqual(
            expected, balance_util.auto_spread(data, features=["key", "value"])
        )

        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 2.0, 4.0),
                "key_b_value": (1.0, 2.0, np.nan),
                "key_a_other_value": (2.0, 4.0, 6.0),
                "key_b_other_value": (2.0, 4.0, np.nan),
            },
            columns=(
                "id",
                "key_a_other_value",
                "key_b_other_value",
                "key_a_value",
                "key_b_value",
            ),
        )
        self.assertEqual(expected, balance_util.auto_spread(data), lazy=True)

        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "a", "c", "d", "a"),
                "value": (1, 1, 2, 4, 1),
            }
        )
        self.assertWarnsRegexp("no unique groupings", balance_util.auto_spread, data)

    def test_auto_spread_multiple_groupings(self) -> None:
        # Multiple possible groupings
        data = pd.DataFrame(
            {
                "id": (1, 1, 2, 2, 3),
                "key": ("a", "b", "b", "a", "a"),
                "value": (1, 3, 2, 4, 1),
            }
        )
        expected = pd.DataFrame(
            {
                "id": (1, 2, 3),
                "key_a_value": (1.0, 4.0, 1.0),
                "key_b_value": (3.0, 2.0, np.nan),
            },
            columns=("id", "key_a_value", "key_b_value"),
        )
        self.assertEqual(expected, balance_util.auto_spread(data))
        self.assertWarnsRegexp("2 possible groupings", balance_util.auto_spread, data)

    def test_auto_aggregate(self) -> None:
        r = balance_util.auto_aggregate(
            pd.DataFrame(
                {"x": [1, 2, 3, 4], "y": [1, 1, 1, np.nan], "id": [1, 1, 2, 3]}
            )
        )
        e = pd.DataFrame({"id": [1, 2, 3], "x": [3, 3, 4], "y": [2, 1, np.nan]})

        self.assertEqual(r, e, lazy=True)

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            pd.DataFrame({"b": ["a", "b", "b"], "id": [1, 1, 2]}),
        )

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            r,
            None,
            "id2",
        )

        self.assertRaises(
            ValueError,
            balance_util.auto_aggregate,
            r,
            None,
            aggfunc="not_sum",
        )

    def test_auto_aggregate_features_deprecation(self) -> None:
        """Test that auto_aggregate warns when features parameter is used.

        Tests that the deprecated 'features' parameter triggers a
        DeprecationWarning when it is not None.
        This covers line 264 in data_transformation.py.
        """
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4], "y": [1, 1, 1, np.nan], "id": [1, 1, 2, 3]}
        )

        # Test that passing features parameter triggers deprecation warning
        with self.assertWarns(DeprecationWarning):
            balance_util.auto_aggregate(df, features=["x", "y"])

    def test_fct_lump_basic_functionality(self) -> None:
        """Test basic functionality of fct_lump for category lumping.

        Tests the fct_lump function's ability to:
        - Preserve categories that meet the threshold
        - Lump categories below the threshold into '_lumped_other'
        - Handle different threshold values
        """
        # Count above the threshold, value preserved
        s = pd.Series(["a"] * 95 + ["b"] * 5)
        self.assertEqual(balance_util.fct_lump(s), s)

        # Move the threshold up
        self.assertEqual(
            balance_util.fct_lump(s, 0.10),
            pd.Series(["a"] * 95 + ["_lumped_other"] * 5),
        )

        # Default threshold, slightly below number of values
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 4)),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

    def test_fct_lump_multiple_categories(self) -> None:
        """Test fct_lump with multiple small categories and edge cases.

        Tests the fct_lump function's ability to:
        - Combine multiple small categories into '_lumped_other'
        - Handle existing '_lumped_other' categories properly
        - Work with categorical data types
        """
        # Multiple categories combined
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 2 + ["c"] * 2)),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

        # Category already called '_lumped_other' is handled
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["_lumped_other"] * 4)),
            pd.Series(["a"] * 96 + ["_lumped_other_lumped_other"] * 4),
        )

        # Categorical series type
        self.assertEqual(
            balance_util.fct_lump(pd.Series(["a"] * 96 + ["b"] * 4, dtype="category")),
            pd.Series(["a"] * 96 + ["_lumped_other"] * 4),
        )

    def _create_wine_test_data(self) -> tuple[Sample, Sample]:
        """Helper method to create synthetic wine dataset for testing.

        Creates synthetic wine data that mimics the structure of the sklearn wine dataset
        but doesn't rely on sklearn's load_wine() function which has compatibility issues
        with newer Python versions.

        Returns:
            tuple: (wine_survey, wine_survey_copy) for categorical and string testing
        """
        # Create synthetic wine data with similar structure to sklearn wine dataset
        np.random.seed(42)  # For reproducible results
        n_samples = 178

        # Create synthetic wine features
        wine_data = {
            "alcohol": np.random.uniform(11.0, 14.8, n_samples),
            "malic_acid": np.random.uniform(0.74, 5.8, n_samples),
            "ash": np.random.uniform(1.36, 3.23, n_samples),
            "alcalinity_of_ash": np.random.uniform(10.6, 30.0, n_samples),
            "magnesium": np.random.uniform(70, 162, n_samples),
            "total_phenols": np.random.uniform(0.98, 3.88, n_samples),
            "flavanoids": np.random.uniform(0.34, 5.08, n_samples),
            "nonflavanoid_phenols": np.random.uniform(0.13, 0.66, n_samples),
            "proanthocyanins": np.random.uniform(0.41, 3.58, n_samples),
            "color_intensity": np.random.uniform(1.28, 13.0, n_samples),
            "hue": np.random.uniform(0.48, 1.71, n_samples),
            "od280_od315_of_diluted_wines": np.random.uniform(1.27, 4.0, n_samples),
            "proline": np.random.uniform(278, 1680, n_samples),
        }

        wine_df = pd.DataFrame(wine_data)
        wine_df["id"] = pd.Series(range(1, len(wine_df) + 1))

        # Create categorical alcohol variable
        wine_df.alcohol = pd.cut(
            wine_df.alcohol, bins=[0, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 100]
        )

        # Create string version for comparison
        wine_df_copy = wine_df.copy(deep=True)
        wine_df_copy.alcohol = wine_df_copy.alcohol.astype("object")

        # Create synthetic target classes (0, 1, 2)
        wine_class = pd.Series(
            np.random.choice([0, 1, 2], size=n_samples, p=[0.33, 0.4, 0.27])
        )

        # Split datasets
        wine_survey = Sample.from_frame(wine_df.loc[wine_class == 0, :])
        wine_pop = Sample.from_frame(wine_df.loc[wine_class != 0, :])
        wine_survey = wine_survey.set_target(wine_pop)

        wine_survey_copy = Sample.from_frame(wine_df_copy.loc[wine_class == 0, :])
        wine_pop_copy = Sample.from_frame(wine_df_copy.loc[wine_class != 0, :])
        wine_survey_copy = wine_survey_copy.set_target(wine_pop_copy)

        return wine_survey, wine_survey_copy

    def test_fct_lump_categorical_vs_string_consistency(self) -> None:
        """Test that fct_lump produces consistent results for categorical vs string variables.

        Tests that fct_lump works identically when applied to:
        - Categorical variables
        - String variables with the same content

        This ensures consistency in model coefficient generation.
        """
        wine_survey, wine_survey_copy = self._create_wine_test_data()

        transformations = {
            "alcohol": lambda x: balance_util.fct_lump(x, prop=0.05),
            "flavanoids": balance_util.quantize,
            "total_phenols": balance_util.quantize,
            "nonflavanoid_phenols": balance_util.quantize,
            "color_intensity": balance_util.quantize,
            "hue": balance_util.quantize,
            "ash": balance_util.quantize,
            "alcalinity_of_ash": balance_util.quantize,
            "malic_acid": balance_util.quantize,
            "magnesium": balance_util.quantize,
        }

        # Generate weights for both categorical and string versions
        output_cat_var = wine_survey.adjust(
            transformations=transformations, method="ipw", max_de=2.5
        )
        output_string_var = wine_survey_copy.adjust(
            transformations=transformations, method="ipw", max_de=2.5
        )

        # Check that model coefficients are identical
        output_cat_var_model = output_cat_var.model()
        output_string_var_model = output_string_var.model()
        output_cat_var_model = _verify_value_type(output_cat_var_model)
        output_string_var_model = _verify_value_type(output_string_var_model)
        self.assertEqual(
            output_cat_var_model["perf"]["coefs"],
            output_string_var_model["perf"]["coefs"],
        )

    def test_fct_lump_by(self) -> None:
        """Test category lumping with grouping by another variable.

        Tests the fct_lump_by function's ability to:
        - Lump categories within groups defined by another variable
        - Handle cases where grouping variable has uniform values
        - Preserve DataFrame indices when combining data
        """
        # test by argument works
        s = pd.Series([1, 1, 1, 2, 3, 1, 2])
        by = pd.Series(["a", "a", "a", "a", "a", "b", "b"])
        self.assertEqual(
            balance_util.fct_lump_by(s, by, 0.5),
            pd.Series([1, 1, 1, "_lumped_other", "_lumped_other", 1, 2]),
        )

        # test case where all values in 'by' are the same
        s = pd.Series([1, 1, 1, 2, 3, 1, 2])
        by = pd.Series(["a", "a", "a", "a", "a", "a", "a"])
        self.assertEqual(
            balance_util.fct_lump_by(s, by, 0.5),
            pd.Series(
                [1, 1, 1, "_lumped_other", "_lumped_other", 1, "_lumped_other"],
            ),
        )

        # test fct_lump_by doesn't affect indices when combining dataframes
        s = pd.DataFrame({"d": [1, 1, 1], "e": ["a1", "a2", "a1"]}, index=(0, 6, 7))
        t = pd.DataFrame(
            {"d": [2, 3, 1, 2], "e": ["a2", "a2", "a1", "a2"]}, index=(0, 1, 2, 3)
        )
        df = pd.concat([s, t])
        r = balance_util.fct_lump_by(df.d, df.e, 0.5)
        e = pd.Series(
            [1, "_lumped_other", 1, 2, "_lumped_other", 1, 2],
            index=(0, 6, 7, 0, 1, 2, 3),
            name="d",
        )
        self.assertEqual(r, e)
