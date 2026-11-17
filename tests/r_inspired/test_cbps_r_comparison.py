# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CORRELATION ONLY: balance CBPS against archived R CBPS reference weights.

The complete original CSV, R capture script and provenance are test-only in
fixtures/cbps/. The recorded script uses the CBPS::CBPS help-page simulation
and records CBPS 0.23 / R 3.6.1 with an ATT=1 fit. The current help page
uses ATT=0 and credits the simulation to Kang and Shafer (2007); see the bank
README for this configuration distinction. No reference values are regenerated.
The correlation guards (>0.98 linear, >0.99 log10) are unchanged; they do not
assert exact equality between the Python and R fitted weights.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import balance.testutil
import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.util import _assert_type


def load_cbps_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the archived test-only CSV as (target, sample), keeping its indexes."""
    path = Path(__file__).parent / "fixtures" / "cbps" / "sim_data_cbps.csv"
    df_all = pd.read_csv(path)
    target_df = df_all[df_all.treat == 1].drop(["treat"], axis=1)
    sample_df = df_all[df_all.treat == 0].drop(["treat"], axis=1)
    return target_df, sample_df


class CbpsRComparisonTest(
    balance.testutil.BalanceTestCase,
):
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
        target_df, sample_df = load_cbps_reference()

        # Create Sample objects with outcome columns
        sample = Sample.from_frame(sample_df, outcome_columns=["y", "cbps_weights"])
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

    def test_load_data_cbps(self) -> None:
        target_df, sample_df = load_cbps_reference()

        sample_df = _assert_type(sample_df)
        target_df = _assert_type(target_df)

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

    def test_load_cbps_data_structure_and_types(self) -> None:
        """Test that load_cbps_data returns DataFrames with correct structure and types."""
        target_df, sample_df = load_cbps_reference()

        # Verify we got DataFrames
        self.assertIsNotNone(target_df)
        self.assertIsNotNone(sample_df)
        self.assertIsInstance(target_df, pd.DataFrame)
        self.assertIsInstance(sample_df, pd.DataFrame)

        target_df = _assert_type(target_df)
        sample_df = _assert_type(sample_df)

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
        target_df1, sample_df1 = load_cbps_reference()
        target_df2, sample_df2 = load_cbps_reference()

        target_df1 = _assert_type(target_df1)
        sample_df1 = _assert_type(sample_df1)
        target_df2 = _assert_type(target_df2)
        sample_df2 = _assert_type(sample_df2)

        # Check that the DataFrames are identical
        pd.testing.assert_frame_equal(target_df1, target_df2)
        pd.testing.assert_frame_equal(sample_df1, sample_df2)

    def test_load_cbps_data_data_validity(self) -> None:
        """Test that load_cbps_data returns data with valid values and properties."""
        target_df, sample_df = load_cbps_reference()

        target_df = _assert_type(target_df)
        sample_df = _assert_type(sample_df)

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

    def test_archived_cbps_bank_is_not_regenerated_by_current_r_recipes(self) -> None:
        import_path = Path(__file__).parent / "generators" / "regenerate_r_fixtures.py"
        module = runpy.run_path(str(import_path))
        fixtures = Path(__file__).parent / "fixtures"
        with self.assertRaisesRegex(ValueError, "cbps is an archived capture"):
            module["discover_groups"](fixtures, ["cbps"])
        groups = module["discover_groups"](fixtures)
        self.assertNotIn("cbps", [group.name for group in groups])

    def test_local_loader_preserves_all_original_rows(self) -> None:
        path = Path(__file__).parent / "fixtures" / "cbps" / "sim_data_cbps.csv"
        raw = pd.read_csv(path)
        target, sample = load_cbps_reference()
        reconstructed = pd.concat(
            [target.assign(treat=1), sample.assign(treat=0)]
        ).sort_index()
        pd.testing.assert_frame_equal(reconstructed[raw.columns], raw)
