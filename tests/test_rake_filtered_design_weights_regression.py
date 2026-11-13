# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.balance_frame import BalanceFrame
from balance.sample_frame import SampleFrame


class RakeFilteredDesignWeightsRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        sample = pd.DataFrame(
            {
                "id": ["s0", "s1", "s2"],
                "a": ["A", None, "B"],
                "b": ["X", "Y", "Y"],
                "weight": [10.0, 20.0, 30.0],
            },
            index=pd.Index([0, 0, 1]),
        )
        target = pd.DataFrame(
            {
                "id": ["t0", "t1"],
                "a": ["A", "B"],
                "b": ["X", "Y"],
                "weight": [10.0, 30.0],
            }
        )
        self.frame = BalanceFrame(
            sample=SampleFrame.from_frame(sample, standardize_types=False),
            target=SampleFrame.from_frame(target, standardize_types=False),
        )
        self.expected = pd.Series([10.0, np.nan, 30.0], index=[0, 0, 1], name="weight")

    def test_filtered_design_weights_replay_at_exact_duplicate_row_positions(
        self,
    ) -> None:
        fitted = self.frame.fit(
            method="rake", transformations=None, na_action="drop", inplace=False
        )
        model = fitted.model
        assert model is not None
        self.assertEqual(model["method"], "rake")
        pd.testing.assert_series_equal(
            model["training_sample_weights"],
            pd.Series([10.0, 30.0], index=[0, 1], name="weight"),
        )
        pd.testing.assert_series_equal(fitted.weights().df["weight"], self.expected)
        pd.testing.assert_series_equal(fitted.predict_weights(), self.expected)

    def test_filtered_design_weights_reject_nonfinite_stored_values(self) -> None:
        fitted = self.frame.fit(
            method="rake", transformations=None, na_action="drop", inplace=False
        )
        model = fitted.model
        assert model is not None
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                model["training_sample_weights"] = pd.Series(
                    [10.0, invalid], index=[0, 1], name="weight"
                )
                with self.assertRaisesRegex(ValueError, "design weights.*finite"):
                    fitted.predict_weights()

    def test_filtered_design_weights_reject_incompatible_stored_row_labels(
        self,
    ) -> None:
        fitted = self.frame.fit(
            method="rake", transformations=None, na_action="drop", inplace=False
        )
        model = fitted.model
        assert model is not None
        model["training_sample_weights"] = pd.Series(
            [10.0, 30.0], index=[10, 11], name="weight"
        )
        with self.assertRaisesRegex(ValueError, "design weights.*non-missing"):
            fitted.predict_weights()

    def test_full_scoring_design_weights_are_filtered_only_once(self) -> None:
        fitted = self.frame.fit(
            method="rake", transformations=None, na_action="drop", inplace=False
        )
        with self.assertLogs("balance", level="WARNING"):
            predicted = fitted.predict_weights(data=self.frame)
        pd.testing.assert_series_equal(predicted, self.expected)
        pd.testing.assert_series_equal(
            self.frame.weights().df["weight"],
            pd.Series([10.0, 20.0, 30.0], index=[0, 0, 1], name="weight"),
        )
