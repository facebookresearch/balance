# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample
from balance.util import _assert_type


def _make_sample() -> Sample:
    return Sample.from_frame(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [2.0, 4.0, 8.0, 10.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        outcome_columns="y",
    )


class SampleActiveWeightTest(unittest.TestCase):
    def test_switch_uses_same_weight_for_all_accessors(self) -> None:
        sample = _make_sample()
        sample.add_weight_column("w2", pd.Series([1.0, 2.0, 3.0, 4.0]))
        sample.set_active_weight("w2")

        self.assertEqual(sample.weight_column, "w2")
        self.assertEqual(
            _assert_type(sample.weight_series).tolist(), [1.0, 2.0, 3.0, 4.0]
        )
        self.assertEqual(sample.df_weights["w2"].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sample.responders.weight_column, "w2")

    def test_update_changes_selected_weight_only(self) -> None:
        sample = _make_sample()
        sample.add_weight_column("w2", pd.Series([1.0, 2.0, 3.0, 4.0]))
        sample.set_active_weight("w2")
        sample.set_weights(pd.Series([4.0, 3.0, 2.0, 1.0]))

        self.assertEqual(
            _assert_type(sample.weight_series).tolist(), [4.0, 3.0, 2.0, 1.0]
        )
        self.assertEqual(sample._df["weight"].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_rename_active_weight_keeps_access_and_updates_working(self) -> None:
        sample = _make_sample()
        sample.rename_weight_column("weight", "renamed")
        sample.set_weights(2.0)

        self.assertEqual(sample.weight_column, "renamed")
        self.assertEqual(_assert_type(sample.weight_series).name, "renamed")
        self.assertEqual(
            _assert_type(sample.weight_series).tolist(), [2.0, 2.0, 2.0, 2.0]
        )
        self.assertNotIn("weight", sample.df.columns)


def _make_target() -> Sample:
    return Sample.from_frame(
        pd.DataFrame(
            {
                "id": ["t1", "t2", "t3", "t4"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "weight": [6.0, 1.0, 1.0, 2.0],
            }
        )
    )


class InplaceFitBaselineTest(unittest.TestCase):
    def test_fit_preserves_a_separate_unadjusted_link(self) -> None:
        sample = _make_sample().set_target(_make_target())
        sample.fit(method="poststratify", variables=["x"], transformations=None)
        baseline = _assert_type(sample._links)["unadjusted"]

        self.assertIsNot(baseline, sample)
        self.assertEqual(baseline.weight_series.tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            _assert_type(sample.weight_series).tolist(), [6.0, 1.0, 1.0, 2.0]
        )

    def test_conversion_preserves_prefit_weights(self) -> None:
        sample = _make_sample().set_target(_make_target())
        sample.fit(method="poststratify", variables=["x"], transformations=None)
        converted = sample.to_balance_frame()

        self.assertEqual(
            converted.df_responders_unadjusted["weight"].tolist(), [1.0] * 4
        )
        self.assertEqual(
            converted.df_responders["weight"].tolist(), [6.0, 1.0, 1.0, 2.0]
        )


class FilteredSampleRolesTest(unittest.TestCase):
    def test_removed_prediction_role_is_not_left_on_facade(self) -> None:
        sample = _make_sample()
        sample.add_outcomes_hat_column("y_hat", pd.Series([1.0, 2.0, 3.0, 4.0]))
        filtered = sample.keep_only_some_rows_columns(columns_to_keep=["x"])

        self.assertEqual(filtered.outcomes_hat_columns, [])
        self.assertIsNone(filtered.df_outcomes_hat)
        self.assertEqual(sample.outcomes_hat_columns, ["y_hat"])


class FrameConversionRolesTest(unittest.TestCase):
    def test_conversion_preserves_the_fitted_outcome_model(self) -> None:
        sample = _make_sample().fit_outcome_model()
        converted = sample.to_sample_frame()

        self.assertIsNotNone(converted.outcome_model)
        self.assertIs(
            converted.outcome_model["fit"]["y"],
            _assert_type(sample.outcome_model)["fit"]["y"],
        )

    def test_sample_frame_target_survives_conversion(self) -> None:
        target = _make_target().to_sample_frame()
        sample = _make_sample().set_target(target, inplace=False)
        converted = sample.to_balance_frame()

        self.assertIsNot(converted.target, target)
        pd.testing.assert_frame_equal(_assert_type(converted.target).df, target.df)

    def test_conversion_uses_fresh_canonical_target_predictions(self) -> None:
        target = _make_target()
        target.add_outcomes_hat_column("y_hat", pd.Series([999.0] * 4))
        sample = _make_sample().set_target(target).fit_outcome_model()
        sample.predict_outcomes(on="target")
        expected = _assert_type(_assert_type(sample.target).df_outcomes_hat)
        converted = sample.to_balance_frame()

        self.assertIsNotNone(converted.outcome_model)
        pd.testing.assert_frame_equal(
            _assert_type(_assert_type(converted.target).df_outcomes_hat), expected
        )
        self.assertEqual(
            _assert_type(target.df_outcomes_hat)["y_hat"].tolist(), [999.0] * 4
        )

    def test_sample_frame_conversion_preserves_all_weight_roles(self) -> None:
        sample = (
            _make_sample()
            .set_target(_make_target())
            .adjust(method="poststratify", variables=["x"], transformations=None)
        )
        converted = sample.to_sample_frame()

        self.assertEqual(converted.covar_columns, ["x"])
        self.assertEqual(converted.weight_columns_all, sample.weight_columns_all)
        self.assertEqual(sample._covar_columns_names(), ["x"])
        self.assertEqual(
            converted.weight_metadata("weight_adjusted_1")["method"], "poststratify"
        )
