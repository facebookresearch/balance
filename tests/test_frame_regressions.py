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
