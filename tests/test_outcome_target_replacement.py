# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from balance import Sample
from balance.util import _assert_type
from sklearn.linear_model import LinearRegression


class OutcomeTargetReplacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [0.0, 1.0], "y": [2.0, 4.0]}),
            outcome_columns="y",
        )
        self.old_target = Sample.from_frame(
            pd.DataFrame({"id": ["c", "d"], "x": [0.0, 1.0], "weight": [3.0, 1.0]})
        )
        self.new_target = Sample.from_frame(
            pd.DataFrame({"id": ["e", "f"], "x": [0.25, 0.75]})
        )
        self.adjusted = self.sample.set_target(self.old_target).adjust(
            method="poststratify", transformations=None
        )

    def test_new_outcome_target_preserves_weights_and_warns(self) -> None:
        previous_model = self.adjusted.model
        previous_history = self.adjusted.adjustment_history
        previous_weights = self.adjusted.df_weights.copy()
        previous_metadata = deepcopy(self.adjusted.responders._weight_metadata)

        with self.assertLogs("balance", level="WARNING") as warnings:
            result = self.adjusted.fit_outcome_model(
                target=self.new_target, model=LinearRegression(), inplace=True
            )

        self.assertIs(result, self.adjusted)
        self.assertEqual(result.df_weights["weight"].tolist(), [3.0, 1.0])
        pd.testing.assert_frame_equal(result.df_weights, previous_weights)
        self.assertIs(result.model, previous_model)
        self.assertEqual(result.adjustment_history, previous_history)
        self.assertEqual(result.responders._weight_metadata, previous_metadata)
        self.assertTrue(result.is_adjusted)
        self.assertIsNotNone(result.outcome_model)
        self.assertIn("previous target", " ".join(warnings.output))
        self.assertIn("preserved", " ".join(warnings.output))
        self.assertEqual(_assert_type(result.df_target)["id"].tolist(), ["e", "f"])
        self.assertEqual(result.df_responders_unadjusted["weight"].tolist(), [1.0, 1.0])
        baseline = _assert_type(result._links)["unadjusted"]
        self.assertEqual(baseline.df_target["id"].tolist(), ["c", "d"])
        np.testing.assert_allclose(
            result.predict_outcomes(on="target", populate=False)["y_hat"], [2.5, 3.5]
        )
        self.assertEqual(self.old_target.df["weight"].tolist(), [3.0, 1.0])

    def test_sampleframe_target_and_copy_keep_weighting_state(self) -> None:
        for target in (self.new_target, self.new_target.to_sample_frame()):
            with self.subTest(target_type=type(target).__name__):
                with self.assertLogs("balance", level="WARNING"):
                    result = self.adjusted.fit_outcome_model(
                        target=target, model=LinearRegression(), inplace=False
                    )
                self.assertIsNot(result, self.adjusted)
                pd.testing.assert_frame_equal(
                    result.df_weights, self.adjusted.df_weights
                )
                self.assertEqual(result.model, self.adjusted.model)
                self.assertEqual(
                    result.adjustment_history, self.adjusted.adjustment_history
                )
                self.assertTrue(result.is_adjusted)
                self.assertEqual(
                    _assert_type(result.df_target)["id"].tolist(), ["e", "f"]
                )
                self.assertEqual(
                    _assert_type(self.adjusted.df_target)["id"].tolist(), ["c", "d"]
                )
                self.assertIsNone(self.adjusted.outcome_model)
                self.assertIsNotNone(result.outcome_model)

    def test_sampleframe_target_inplace_keeps_existing_weights(self) -> None:
        with self.assertLogs("balance", level="WARNING"):
            result = self.adjusted.fit_outcome_model(
                target=self.new_target.to_sample_frame(),
                model=LinearRegression(),
                inplace=True,
            )
        self.assertIs(result, self.adjusted)
        self.assertEqual(result.df_weights["weight"].tolist(), [3.0, 1.0])
        self.assertTrue(result.is_adjusted)
        self.assertEqual(len(result.adjustment_history), 1)

    def test_unchanged_or_omitted_target_does_not_warn_or_reset_weights(self) -> None:
        for target in (None, self.old_target, deepcopy(self.old_target)):
            with self.subTest(target=target):
                with self.assertNoLogs("balance", level="WARNING"):
                    result = self.adjusted.fit_outcome_model(
                        target=target, model=LinearRegression(), inplace=False
                    )
                self.assertEqual(result.df_weights["weight"].tolist(), [3.0, 1.0])
                self.assertTrue(result.is_adjusted)
                self.assertIsNotNone(result.outcome_model)

    def test_first_target_does_not_report_a_target_change(self) -> None:
        with self.assertNoLogs("balance", level="WARNING"):
            result = self.sample.fit_outcome_model(
                target=self.new_target, model=LinearRegression(), inplace=False
            )
        self.assertFalse(result.is_adjusted)
        self.assertEqual(result.df_weights["weight"].tolist(), [1.0, 1.0])
        self.assertEqual(_assert_type(result.df_target)["id"].tolist(), ["e", "f"])
        self.assertIsNone(self.sample.df_target)

    def test_new_shared_target_is_not_modified_by_fit_or_prediction(self) -> None:
        self.new_target.add_outcomes_hat_column("y_hat", pd.Series([99.0, 99.0]))
        with self.assertLogs("balance", level="WARNING"):
            result = self.adjusted.fit_outcome_model(
                target=self.new_target, model=LinearRegression(), inplace=False
            )
        self.assertEqual(
            _assert_type(self.new_target.df_outcomes_hat)["y_hat"].tolist(),
            [99.0, 99.0],
        )
        np.testing.assert_allclose(
            result.predict_outcomes(on="target")["y_hat"], [2.5, 3.5]
        )
        self.assertEqual(
            _assert_type(self.new_target.df_outcomes_hat)["y_hat"].tolist(),
            [99.0, 99.0],
        )
        self.assertEqual(result.df_weights["weight"].tolist(), [3.0, 1.0])
        self.assertEqual(
            _assert_type(result._links)["target"].df["id"].tolist(), ["e", "f"]
        )

    def test_weighted_outcome_fit_uses_preserved_adjusted_weights(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {"id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 10.0]}
            ),
            outcome_columns="y",
        )
        old_target = Sample.from_frame(
            pd.DataFrame(
                {"id": ["d", "e", "f"], "x": [0.0, 1.0, 2.0], "weight": [1.0, 1.0, 8.0]}
            )
        )
        adjusted = sample.set_target(old_target).adjust(
            method="poststratify", transformations=None
        )
        with self.assertLogs("balance", level="WARNING"):
            result = adjusted.fit_outcome_model(
                target=self.new_target,
                model=LinearRegression(),
                weighted=True,
                use_model_matrix=False,
                inplace=False,
            )
        reference = LinearRegression().fit(
            sample.df_covars,
            _assert_type(sample.df_outcomes)["y"],
            sample_weight=[1.0, 1.0, 8.0],
        )
        np.testing.assert_allclose(
            result.predict_outcomes(on="sample", populate=False)["y_hat"],
            reference.predict(sample.df_covars),
        )
        self.assertEqual(result.df_weights["weight"].tolist(), [1.0, 1.0, 8.0])

    def test_failed_fit_does_not_replace_target_or_weighting_state(self) -> None:
        original = self.adjusted.df.copy()
        original_target = _assert_type(self.adjusted.df_target).copy()
        model = self.adjusted.model
        history = self.adjusted.adjustment_history
        with self.assertRaises(ValueError):
            self.adjusted.fit_outcome_model(
                target=self.new_target,
                outcome_columns="missing",
                model=LinearRegression(),
            )
        pd.testing.assert_frame_equal(self.adjusted.df, original)
        pd.testing.assert_frame_equal(
            _assert_type(self.adjusted.df_target), original_target
        )
        self.assertIs(self.adjusted.model, model)
        self.assertEqual(self.adjusted.adjustment_history, history)
        self.assertIsNone(self.adjusted.outcome_model)
