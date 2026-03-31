# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import numpy as np
import pandas as pd
from balance.sample_class import Sample
from balance.sample_frame import SampleFrame
from balance.testutil import BalanceTestCase


class TestSampleFrame(BalanceTestCase):
    def test_direct_init_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            SampleFrame()

    def test_create_basic(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2, 3], "x1": [10, 20, 30], "x2": [4, 5, 6], "w": [1.0, 1.0, 1.0]}
        )
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x1", "x2"],
            weight_columns=["w"],
        )
        self.assertEqual(len(sf.df), 3)
        self.assertListEqual(sf._column_roles["covars"], ["x1", "x2"])
        self.assertListEqual(sf._column_roles["weights"], ["w"])
        self.assertListEqual(sf._column_roles["outcomes"], [])
        self.assertEqual(sf._active_weight_column, "w")

    def test_df_covars(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "a": [10, 20], "b": [30, 40], "w": [1.0, 2.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["a", "b"], weight_columns=["w"]
        )
        covars = sf.df_covars
        self.assertListEqual(list(covars.columns), ["a", "b"])
        self.assertEqual(covars.shape, (2, 2))

    def test_df_weights(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2], "x": [10, 20], "w1": [1.0, 2.0], "w2": [3.0, 4.0]}
        )
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w1", "w2"]
        )
        # Active weight is the first one
        self.assertListEqual(list(sf.df_weights.columns), ["w1"])

    def test_df_outcomes_none(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertIsNone(sf.df_outcomes)

    def test_df_outcomes_present(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0], "y": [5, 6]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        self.assertIsNotNone(sf.df_outcomes)
        self.assertListEqual(list(sf.df_outcomes.columns), ["y"])

    def test_df_misc_none(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertIsNone(sf.df_misc)

    def test_df_misc_present(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0], "region": ["US", "UK"]}
        )
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            misc_columns=["region"],
        )
        self.assertIsNotNone(sf.df_misc)
        self.assertListEqual(list(sf.df_misc.columns), ["region"])

    def test_ignored_columns_none(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertIsNone(sf.ignored_columns())

    def test_ignored_columns_present(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0], "region": ["US", "UK"]}
        )
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            misc_columns=["region"],
        )
        result = sf.ignored_columns()
        self.assertIsNotNone(result)
        self.assertListEqual(list(result.columns), ["region"])
        self.assertListEqual(list(result["region"]), ["US", "UK"])

    def test_ignored_columns_matches_df_misc(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0], "region": ["US", "UK"]}
        )
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            misc_columns=["region"],
        )
        pd.testing.assert_frame_equal(sf.ignored_columns(), sf.df_misc)

    def test_id_column(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30], "w": [1.0, 1.0, 1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertIsInstance(sf.id_column, pd.Series)
        self.assertListEqual(list(sf.id_column), [1, 2, 3])

    def test_df_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        df_copy = sf.df
        df_copy["x"] = [999, 999]
        # Internal state should be unchanged
        self.assertListEqual(list(sf.df["x"]), [10, 20])

    def test_create_copies_input_df(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10, 20], "w": [1.0, 1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        # Mutate the original DataFrame
        df["x"] = [999, 999]
        # Internal state should be unchanged
        self.assertListEqual(list(sf.df_covars["x"]), [10, 20])

    def test_repr(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2], "a": [10, 20], "b": [30, 40], "w": [1.0, 2.0], "y": [5, 6]}
        )
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["a", "b"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        r = repr(sf)
        self.assertIn("2 observations", r)
        self.assertIn("2 covariates", r)
        self.assertIn("a,b", r)
        self.assertIn("id_column: id", r)
        self.assertIn("outcome_columns: y", r)


class TestSampleFrameMutableViewSafety(BalanceTestCase):
    """Verify that DataFrame/Series properties return copies, not views."""

    def _make_sf(self) -> SampleFrame:
        df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "a": [10.0, 20.0, 30.0],
                "b": [40.0, 50.0, 60.0],
                "weight": [1.0, 2.0, 1.5],
                "y": [5.0, 6.0, 7.0],
                "p_y": [0.3, 0.5, 0.7],
                "region": ["US", "UK", "CA"],
            }
        )
        return SampleFrame.from_frame(
            df,
            outcome_columns=["y"],
            predicted_outcome_columns=["p_y"],
            misc_columns=["region"],
        )

    def test_df_covars_returns_copy(self) -> None:
        sf = self._make_sf()
        covars = sf.df_covars
        covars["a"] = [999.0, 999.0, 999.0]
        self.assertEqual(list(sf._df["a"]), [10.0, 20.0, 30.0])

    def test_df_weights_returns_copy(self) -> None:
        sf = self._make_sf()
        weights = sf.df_weights
        weights["weight"] = [999.0, 999.0, 999.0]
        self.assertEqual(list(sf._df["weight"]), [1.0, 2.0, 1.5])

    def test_df_outcomes_returns_copy(self) -> None:
        sf = self._make_sf()
        outcomes = sf.df_outcomes
        self.assertIsNotNone(outcomes)
        outcomes["y"] = [999.0, 999.0, 999.0]
        self.assertEqual(list(sf._df["y"]), [5.0, 6.0, 7.0])

    def test_df_misc_returns_copy(self) -> None:
        sf = self._make_sf()
        misc = sf.df_misc
        self.assertIsNotNone(misc)
        misc["region"] = ["XX", "XX", "XX"]
        self.assertEqual(list(sf._df["region"]), ["US", "UK", "CA"])

    def test_id_column_returns_copy(self) -> None:
        sf = self._make_sf()
        ids = sf.id_column
        ids.iloc[0] = "MUTATED"
        self.assertEqual(sf._df[sf._id_column_name].iloc[0], "1")


class TestSampleFrameColumnRoleProperties(BalanceTestCase):
    def test_covars_columns(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "a": [10, 20], "b": [30, 40], "w": [1.0, 2.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["a", "b"], weight_columns=["w"]
        )
        self.assertEqual(sf.covars_columns, ["a", "b"])

    def test_covars_columns_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        result = sf.covars_columns
        result.append("injected")
        self.assertEqual(sf.covars_columns, ["x"])

    def test_weight_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w1": [1.0], "w2": [2.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w1", "w2"]
        )
        self.assertEqual(sf.weight_columns, ["w1", "w2"])

    def test_weight_columns_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        result = sf.weight_columns
        result.append("injected")
        self.assertEqual(sf.weight_columns, ["w"])

    def test_outcome_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "y": [5]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        self.assertEqual(sf.outcome_columns, ["y"])

    def test_outcome_columns_empty(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(sf.outcome_columns, [])

    def test_outcome_columns_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "y": [5]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            outcome_columns=["y"],
        )
        result = sf.outcome_columns
        result.append("injected")
        self.assertEqual(sf.outcome_columns, ["y"])

    def test_predicted_outcome_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "p_y": [0.5]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            predicted_outcome_columns=["p_y"],
        )
        self.assertEqual(sf.predicted_outcome_columns, ["p_y"])

    def test_predicted_outcome_columns_empty(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(sf.predicted_outcome_columns, [])

    def test_predicted_outcome_columns_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "p_y": [0.5]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            predicted_outcome_columns=["p_y"],
        )
        result = sf.predicted_outcome_columns
        result.append("injected")
        self.assertEqual(sf.predicted_outcome_columns, ["p_y"])

    def test_misc_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "region": ["US"]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            misc_columns=["region"],
        )
        self.assertEqual(sf.misc_columns, ["region"])

    def test_misc_columns_empty(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(sf.misc_columns, [])

    def test_misc_columns_returns_copy(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0], "region": ["US"]})
        sf = SampleFrame._create(
            df=df,
            id_column="id",
            covars_columns=["x"],
            weight_columns=["w"],
            misc_columns=["region"],
        )
        result = sf.misc_columns
        result.append("injected")
        self.assertEqual(sf.misc_columns, ["region"])

    def test_active_weight_column(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10], "w": [1.0]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(sf.active_weight_column, "w")

    def test_active_weight_column_none(self) -> None:
        df = pd.DataFrame({"id": [1], "x": [10]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=[]
        )
        self.assertIsNone(sf.active_weight_column)

    def test_id_column_name(self) -> None:
        df = pd.DataFrame({"my_id": ["a", "b"], "x": [10, 20], "w": [1.0, 1.0]})
        sf = SampleFrame._create(
            df=df, id_column="my_id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(sf.id_column_name, "my_id")

    def test_id_column_name_from_frame(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0]})
        sf = SampleFrame.from_frame(df)
        self.assertEqual(sf.id_column_name, "id")


class TestSampleFrameFromFrame(BalanceTestCase):
    def test_from_frame_auto_detect_id(self) -> None:
        df = pd.DataFrame(
            {"id": [1, 2, 3], "x": [10, 20, 30], "weight": [1.0, 1.0, 1.0]}
        )
        sf = SampleFrame.from_frame(df)
        self.assertEqual(list(sf.id_column), ["1", "2", "3"])

    def test_from_frame_explicit_id(self) -> None:
        df = pd.DataFrame({"my_id": ["a", "b"], "x": [10, 20], "weight": [1.0, 1.0]})
        sf = SampleFrame.from_frame(df, id_column="my_id")
        self.assertEqual(list(sf.id_column), ["a", "b"])

    def test_from_frame_auto_detect_weight(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.5, 2.5]})
        sf = SampleFrame.from_frame(df)
        self.assertAlmostEqual(sf.df_weights.iloc[0, 0], 1.5, places=5)
        self.assertAlmostEqual(sf.df_weights.iloc[1, 0], 2.5, places=5)

    def test_from_frame_auto_detect_weights_plural(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weights": [1.5, 2.5]})
        sf = SampleFrame.from_frame(df)
        self.assertEqual(list(sf.df_weights.columns), ["weights"])

    def test_from_frame_default_weight(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20]})
        sf = SampleFrame.from_frame(df)
        self.assertAlmostEqual(sf.df_weights.iloc[0, 0], 1.0, places=5)
        self.assertAlmostEqual(sf.df_weights.iloc[1, 0], 1.0, places=5)
        self.assertEqual(list(sf.df_weights.columns), ["weight"])

    def test_from_frame_default_weight_no_standardize(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20]})
        sf = SampleFrame.from_frame(df, standardize_types=False)
        self.assertEqual(sf.df_weights.iloc[0, 0], 1)

    def test_from_frame_standardize_types(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x": pd.array([10, 20, 30], dtype="Int64"),
                "weight": [1.0, 1.0, 1.0],
            }
        )
        sf = SampleFrame.from_frame(df)
        # Int64 should be converted to float64
        self.assertEqual(sf.df_covars["x"].dtype, np.float64)

    def test_from_frame_pd_na_to_np_nan(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "x": pd.array([10, pd.NA, 30], dtype="Int64"),
                "weight": [1.0, 1.0, 1.0],
            }
        )
        sf = SampleFrame.from_frame(df)
        # pd.NA should become np.nan
        self.assertTrue(np.isnan(sf.df_covars["x"].iloc[1]))

    def test_from_frame_null_id_raises(self) -> None:
        df = pd.DataFrame(
            {"id": [1, None, 3], "x": [10, 20, 30], "weight": [1.0, 1.0, 1.0]}
        )
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_negative_weight_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, -1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_null_weight_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, None]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_non_numeric_weight_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": ["a", "b"]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_duplicate_id_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "1"], "x": [10, 20], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_duplicate_id_allowed_when_unchecked(self) -> None:
        df = pd.DataFrame({"id": ["1", "1"], "x": [10, 20], "weight": [1.0, 1.0]})
        sf = SampleFrame.from_frame(df, check_id_uniqueness=False)
        self.assertEqual(len(sf.df), 2)

    def test_from_frame_covars_by_exclusion(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "a": [10, 20],
                "b": [30, 40],
                "weight": [1.0, 2.0],
                "y": [5, 6],
            }
        )
        sf = SampleFrame.from_frame(df, outcome_columns=["y"])
        self.assertListEqual(list(sf.df_covars.columns), ["a", "b"])

    def test_from_frame_explicit_covars(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "a": [10, 20],
                "b": [30, 40],
                "c": [50, 60],
                "weight": [1.0, 2.0],
            }
        )
        sf = SampleFrame.from_frame(df, covars_columns=["a", "b"])
        self.assertListEqual(list(sf.df_covars.columns), ["a", "b"])

    def test_from_frame_misc_columns(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "a": [10, 20],
                "region": ["US", "UK"],
                "weight": [1.0, 2.0],
            }
        )
        sf = SampleFrame.from_frame(df, misc_columns=["region"])
        self.assertIsNotNone(sf.df_misc)
        self.assertListEqual(list(sf.df_misc.columns), ["region"])
        # region should NOT be in covars
        self.assertNotIn("region", list(sf.df_covars.columns))

    def test_from_frame_missing_outcome_column_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df, outcome_columns=["nonexistent"])

    def test_from_frame_missing_predicted_column_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df, predicted_outcome_columns=["nonexistent"])

    def test_from_frame_missing_misc_column_raises(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df, misc_columns=["nonexistent"])

    def test_from_frame_no_id_column_raises(self) -> None:
        df = pd.DataFrame({"x": [10, 20], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df)

    def test_from_frame_id_column_candidates(self) -> None:
        df = pd.DataFrame(
            {"respondent_id": ["a", "b"], "x": [10, 20], "weight": [1.0, 1.0]}
        )
        sf = SampleFrame.from_frame(df, id_column_candidates=["respondent_id"])
        self.assertEqual(list(sf.id_column), ["a", "b"])

    def test_from_frame_use_deepcopy_false(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0]})
        sf = SampleFrame.from_frame(df, use_deepcopy=False)
        # _create still copies, but the pre-processing doesn't deep-copy
        self.assertEqual(len(sf.df), 2)

    def test_from_frame_string_outcome(self) -> None:
        """Test that outcome_columns accepts a single string."""
        df = pd.DataFrame(
            {"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0], "y": [5, 6]}
        )
        sf = SampleFrame.from_frame(df, outcome_columns="y")
        self.assertIsNotNone(sf.df_outcomes)
        self.assertListEqual(list(sf.df_outcomes.columns), ["y"])

    def test_from_frame_equivalence_with_sample(self) -> None:
        """KEY TEST: SampleFrame.from_frame() and Sample.from_frame() produce
        equivalent data on the same input."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "gender": ["M", "F", "M", "F", "M"],
                "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
                "weight": [1.0, 1.5, 0.8, 1.2, 1.1],
                "converted": [1, 0, 1, 0, 1],
            }
        )
        sample = Sample.from_frame(df, outcome_columns=["converted"])
        sf = SampleFrame.from_frame(df, outcome_columns=["converted"])

        # Covariates should match
        pd.testing.assert_frame_equal(sample.covars().df, sf.df_covars)

        # Weights should match
        pd.testing.assert_series_equal(
            sample.weight_column, sf.df_weights.iloc[:, 0], check_names=False
        )

        # IDs should match (both should be strings)
        pd.testing.assert_series_equal(
            sample.id_column, sf.id_column, check_names=False
        )

        # Outcomes should match
        pd.testing.assert_frame_equal(sample._outcome_columns, sf.df_outcomes)

    def test_from_frame_zero_weight_allowed(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "x": [10, 20], "weight": [0.0, 1.0]})
        sf = SampleFrame.from_frame(df)
        self.assertAlmostEqual(sf.df_weights.iloc[0, 0], 0.0, places=5)

    def test_from_frame_overlapping_outcome_and_misc_raises(self) -> None:
        """A column in both outcome_columns and misc_columns raises ValueError."""
        df = pd.DataFrame(
            {"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0], "y": [5, 6]}
        )
        with self.assertRaisesRegex(ValueError, "appear in both 'outcomes' and 'misc'"):
            SampleFrame.from_frame(df, outcome_columns=["y"], misc_columns=["y"])

    def test_from_frame_overlapping_covar_and_outcome_raises(self) -> None:
        """A column in both explicit covars_columns and outcome_columns raises ValueError."""
        df = pd.DataFrame(
            {"id": ["1", "2"], "x": [10, 20], "weight": [1.0, 1.0], "y": [5, 6]}
        )
        with self.assertRaisesRegex(
            ValueError, "appear in both 'covars' and 'outcomes'"
        ):
            SampleFrame.from_frame(df, covars_columns=["x", "y"], outcome_columns=["y"])


class TestSampleFrameDunderMethods(BalanceTestCase):
    def _make_sf(self) -> SampleFrame:
        df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30], "w": [1.0, 1.0, 1.0]})
        return SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )

    def test_len_returns_row_count(self) -> None:
        sf = self._make_sf()
        self.assertEqual(len(sf), 3)

    def test_len_empty_frame(self) -> None:
        df = pd.DataFrame({"id": [], "x": [], "w": []})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=["w"]
        )
        self.assertEqual(len(sf), 0)

    def test_deepcopy_produces_independent_copy(self) -> None:
        import copy

        sf = self._make_sf()
        sf2 = copy.deepcopy(sf)

        # Same values
        self.assertEqual(len(sf2), len(sf))
        self.assertListEqual(list(sf2._df["x"]), list(sf._df["x"]))

        # Independent data — mutating sf2's underlying df doesn't affect sf
        sf2._df.loc[0, "x"] = 999
        self.assertEqual(sf._df.loc[0, "x"], 10)

    def test_deepcopy_metadata_is_independent(self) -> None:
        import copy

        sf = self._make_sf()
        sf._column_roles["covars"].append("extra")
        sf2 = copy.deepcopy(sf)

        # Mutating sf2's column_roles doesn't affect sf
        sf2._column_roles["covars"].append("another")
        self.assertNotIn("another", sf._column_roles["covars"])


class TestSampleFrameEdgeCases(BalanceTestCase):
    def test_from_frame_zero_rows(self) -> None:
        df = pd.DataFrame(
            {
                "id": pd.Series([], dtype="int64"),
                "x": pd.Series([], dtype="float64"),
                "weight": pd.Series([], dtype="float64"),
            }
        )
        sf = SampleFrame.from_frame(df, id_column="id", weight_column="weight")
        self.assertEqual(len(sf), 0)
        self.assertListEqual(list(sf.df_covars.columns), ["x"])
        self.assertEqual(sf.df_covars.shape[0], 0)

    def test_df_weights_empty_weight_columns(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
        sf = SampleFrame._create(
            df=df, id_column="id", covars_columns=["x"], weight_columns=[]
        )
        w = sf.df_weights
        self.assertIsInstance(w, pd.DataFrame)
        self.assertEqual(w.shape[1], 0)

    def test_from_frame_predicted_column_missing_from_df(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0], "weight": [1.0, 1.0]})
        with self.assertRaises(ValueError):
            SampleFrame.from_frame(df, predicted_outcome_columns=["y_hat"])

    def test_skip_copy_optimization_preserves_isolation(self) -> None:
        # from_frame() passes _skip_copy=True to _create() since it already deep-copied.
        # Verify the resulting SampleFrame is still independent of the original DataFrame.
        df = pd.DataFrame(
            {"id": [1, 2, 3], "x": [10.0, 20.0, 30.0], "weight": [1.0, 1.0, 1.0]}
        )
        sf = SampleFrame.from_frame(df, id_column="id", weight_column="weight")
        # Mutate original df — sf must not be affected
        df.loc[0, "x"] = 999.0
        self.assertAlmostEqual(sf._df.loc[0, "x"], 10.0)
