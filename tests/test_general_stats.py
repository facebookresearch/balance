# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import balance.testutil
import numpy as np
import pandas as pd
from balance.stats_and_plots.general_stats import relative_response_rates


class TestGeneralStats(balance.testutil.BalanceTestCase):
    """Test suite for general_stats module functions.

    This test class validates the relative_response_rates function including:
    - Basic response rate calculations
    - Comparisons with target DataFrames
    - Edge cases and error conditions
    - Examples from the function docstring
    """

    def test_relative_response_rates_basic(self) -> None:
        """Test basic functionality without target DataFrame."""
        # Create test DataFrame with some null values
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, np.nan, 4, 5],
                "col3": [1, np.nan, np.nan, 4, 5],
            }
        )

        result = relative_response_rates(df)

        # Check structure
        self.assertEqual(list(result.index), ["n", "%"])
        self.assertEqual(list(result.columns), ["col1", "col2", "col3"])

        # Check values
        self.assertEqual(result.loc["n", "col1"], 5)
        self.assertEqual(result.loc["n", "col2"], 4)
        self.assertEqual(result.loc["n", "col3"], 3)

        self.assertEqual(result.loc["%", "col1"], 100.0)
        self.assertEqual(result.loc["%", "col2"], 80.0)
        self.assertEqual(result.loc["%", "col3"], 60.0)

    def test_relative_response_rates_with_target_per_column(self) -> None:
        """Test with target DataFrame using per-column comparison."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": [1, 2, np.nan, 4],
            }
        )

        df_target = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6, 7, 8],
                "col2": [1, 2, 3, 4, 5, 6, np.nan, np.nan],
            }
        )

        result = relative_response_rates(df, df_target, per_column=True)

        # Check values
        self.assertEqual(result.loc["n", "col1"], 4)
        self.assertEqual(result.loc["n", "col2"], 3)

        # col1: 4 out of 8 non-null in target = 50%
        self.assertEqual(result.loc["%", "col1"], 50.0)
        # col2: 3 out of 6 non-null in target = 50%
        self.assertEqual(result.loc["%", "col2"], 50.0)

    def test_relative_response_rates_with_target_global(self) -> None:
        """Test with target DataFrame using global comparison."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": [1, 2, 3, np.nan],
            }
        )

        df_target = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6, 7, 8],
                "col2": [1, 2, 3, 4, 5, 6, 7, np.nan],
            }
        )

        result = relative_response_rates(df, df_target, per_column=False)

        # Target has 7 complete rows (row 8 has null in col2)
        # col1: 4 out of 7 complete rows
        self.assertAlmostEqual(result.loc["%", "col1"], 4 / 7 * 100, places=5)
        # col2: 3 out of 7 complete rows
        self.assertAlmostEqual(result.loc["%", "col2"], 3 / 7 * 100, places=5)

    def test_relative_response_rates_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = relative_response_rates(df)

        # Should return empty DataFrame with correct structure
        self.assertEqual(list(result.index), ["n", "%"])
        self.assertEqual(len(result.columns), 0)

    def test_relative_response_rates_all_null(self) -> None:
        """Test with DataFrame containing all null values."""
        df = pd.DataFrame(
            {
                "col1": [np.nan, np.nan, np.nan],
                "col2": [np.nan, np.nan, np.nan],
            }
        )

        result = relative_response_rates(df)

        # All counts should be 0
        self.assertEqual(result.loc["n", "col1"], 0)
        self.assertEqual(result.loc["n", "col2"], 0)
        self.assertEqual(result.loc["%", "col1"], 0.0)
        self.assertEqual(result.loc["%", "col2"], 0.0)

    def test_relative_response_rates_mismatched_columns(self) -> None:
        """Test error when df and df_target have different columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df_target = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        with self.assertRaisesRegex(
            ValueError, "df and df_target must have the exact same columns"
        ):
            relative_response_rates(df, df_target, per_column=True)

    def test_relative_response_rates_invalid_comparison(self) -> None:
        """Test error when df has more non-null values than df_target."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1, 2, 3, 4, 5],
            }
        )

        df_target = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, np.nan],
                "col2": [1, 2, 3, np.nan],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "The number of \\(notnull\\) rows in df MUST be smaller or equal",
        ):
            relative_response_rates(df, df_target, per_column=True)

    def test_relative_response_rates_with_mixed_types(self) -> None:
        """Test with DataFrame containing mixed data types with different null patterns."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5],
                "string": ["a", None, "c", None, "e"],
                "boolean": [True, False, True, False, np.nan],
                "mixed": [1, "b", 3.5, None, None],
            }
        )

        result = relative_response_rates(df)

        # Check counts - each column has different number of non-nulls
        self.assertEqual(result.loc["n", "numeric"], 4)  # 4 out of 5
        self.assertEqual(result.loc["n", "string"], 3)  # 3 out of 5
        self.assertEqual(result.loc["n", "boolean"], 4)  # 4 out of 5
        self.assertEqual(result.loc["n", "mixed"], 3)  # 3 out of 5

        # Check percentages
        self.assertEqual(result.loc["%", "numeric"], 80.0)
        self.assertEqual(result.loc["%", "string"], 60.0)
        self.assertEqual(result.loc["%", "boolean"], 80.0)
        self.assertEqual(result.loc["%", "mixed"], 60.0)

    def test_relative_response_rates_single_column(self) -> None:
        """Test with single column DataFrame."""
        df = pd.DataFrame({"single": [1, 2, np.nan, 4, 5]})

        result = relative_response_rates(df)

        self.assertEqual(result.loc["n", "single"], 4)
        self.assertEqual(result.loc["%", "single"], 80.0)
