# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.balancedf_class import BalanceDFCovars
from balance.util import _assert_type


class NamedSurveyWeightsTest(unittest.TestCase):
    def test_named_survey_weights_produce_weighted_mean_nine_not_five(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]})
        weights = pd.Series([1.0, 9.0], name="survey_weight")

        covars = BalanceDFCovars.from_frame(values, weights=weights)

        self.assertEqual(covars.mean(on_linked_samples=False)["x"].iloc[0], 9.0)
        self.assertEqual(_assert_type(covars._weights).tolist(), [1.0, 9.0])
        self.assertEqual(covars.df.columns.tolist(), ["index", "x"])

    def test_extra_matching_weight_labels_align_before_length_validation(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]}, index=pd.Index([8, 9]))
        weights = pd.Series([100.0, 9.0, 1.0], index=[10, 9, 8], name="survey_weight")

        covars = BalanceDFCovars.from_frame(values, weights=weights)

        self.assertEqual(covars.mean(on_linked_samples=False)["x"].iloc[0], 9.0)
        self.assertEqual(weights.index.tolist(), [10, 9, 8])

    def test_extra_weights_with_missing_data_labels_remain_invalid(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]}, index=pd.Index([8, 9]))
        weights = pd.Series([1.0, 4.0, 5.0], index=[8, 10, 11])

        with self.assertRaisesRegex(ValueError, "weights.*length"):
            BalanceDFCovars.from_frame(values, weights=weights)

    def test_equal_length_labels_missing_a_row_are_rejected(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]}, index=pd.Index([8, 9]))
        weights = pd.Series([9.0, 1.0], index=[9, 10], name="survey_weight")
        with self.assertRaisesRegex(ValueError, "weights index.*DataFrame"):
            BalanceDFCovars.from_frame(values, weights=weights)

    def test_default_weights_stay_positional_when_frame_index_is_reversed(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]}, index=pd.Index([1, 0]))
        weights = pd.Series([1.0, 9.0], name="weight")
        covars = BalanceDFCovars.from_frame(values, weights=weights)
        self.assertEqual(covars.mean(on_linked_samples=False)["x"].iloc[0], 9.0)

    def test_duplicate_indices_keep_equal_length_positional_weights(self) -> None:
        for data_index, weight_index in (
            ([8, 8], None),
            ([8, 9], [8, 8]),
            ([8, 8], [8, 8]),
        ):
            with self.subTest(data_index=data_index, weight_index=weight_index):
                values = pd.DataFrame({"x": [0.0, 10.0]}, index=pd.Index(data_index))
                weights = pd.Series(
                    [1.0, 9.0], index=weight_index, name="survey_weight"
                )
                covars = BalanceDFCovars.from_frame(values, weights=weights)
                self.assertEqual(covars.mean(on_linked_samples=False)["x"].iloc[0], 9.0)

    def test_default_positional_weights_must_match_frame_length(self) -> None:
        values = pd.DataFrame({"x": [0.0, 10.0]})
        weights = pd.Series([1.0, 9.0, 5.0])

        with self.assertRaisesRegex(
            ValueError, r"weights length \(3\) must match DataFrame length \(2\)"
        ):
            BalanceDFCovars.from_frame(values, weights=weights)
