# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from balance import Sample
from balance.balance_frame import BalanceFrame
from balance.outcome_models import fit_outcome_model
from balance.sample_frame import SampleFrame
from balance.testutil import _SKLEARN_1_4_AVAILABLE
from balance.utils.input_validation import _assert_type
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def _sample() -> Sample:
    return Sample.from_frame(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [2.0, 4.0, 8.0, 10.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        outcome_columns=["y"],
    )


def _target() -> Sample:
    return Sample.from_frame(
        pd.DataFrame(
            {
                "id": ["t1", "t2", "t3", "t4"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "weight": [6.0, 1.0, 1.0, 2.0],
            }
        )
    )


def _interval(frame: BalanceFrame) -> pd.DataFrame:
    return _assert_type(frame.outcomes_hat()).mean_with_ci(
        n_bootstrap=5, random_seed=11, round_ndigits=8
    )


class OutcomeFitProvenanceTest(unittest.TestCase):
    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_bootstrap_preserves_variable_subset(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d"],
                    "x": [0.0] * 4,
                    "z": [0.0, 1.0, 0.0, 1.0],
                    "y": [2.0, 12.0, 2.0, 12.0],
                }
            ),
            outcome_columns=["y"],
        )
        target = Sample.from_frame(pd.DataFrame({"id": ["t"], "x": [0.0], "z": [1.0]}))
        frame = sample.set_target(target).fit_outcome_model(
            variables=["x"],
            transformations=None,
            use_model_matrix=False,
            model=HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                min_samples_leaf=1,
                max_iter=2,
                learning_rate=1.0,
                early_stopping=False,
                random_state=0,
            ),
        )
        frame.predict_outcomes(on="target")
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 7.0
        )
        interval = _assert_type(frame.outcomes_hat()).mean_with_ci(
            n_bootstrap=5, random_seed=11, round_ndigits=8
        )
        self.assertAlmostEqual(interval.loc["y_hat", "target"], 7.0)

    def test_bootstrap_preserves_fit_weights_after_adjustment(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d"],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "y": [2.0, 4.0, 8.0, 10.0],
                    "weight": [1.0] * 4,
                }
            ),
            outcome_columns=["y"],
        )
        target = Sample.from_frame(
            pd.DataFrame(
                {
                    "id": ["t1", "t2", "t3", "t4"],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "weight": [6.0, 1.0, 1.0, 2.0],
                }
            )
        )
        frame = sample.set_target(target).fit_outcome_model(
            model=LinearRegression(fit_intercept=True),
            weighted=True,
            transformations=None,
        )
        frame.predict_outcomes(on="target")
        expected = _assert_type(frame.outcomes_hat()).mean_with_ci(
            n_bootstrap=5, random_seed=11, round_ndigits=8
        )
        self.assertAlmostEqual(expected.loc["y_hat", "target"], 4.32)
        adjusted = frame.adjust(
            method="poststratify", variables=["x"], transformations=None
        )
        np.testing.assert_allclose(adjusted.weight_series, [6.0, 1.0, 1.0, 2.0])
        actual = _assert_type(adjusted.outcomes_hat()).mean_with_ci(
            n_bootstrap=5, random_seed=11, round_ndigits=8
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_bootstrap_transferred_model_uses_training_not_holdout_outcomes(
        self,
    ) -> None:
        train = (
            _sample().set_target(_target()).fit_outcome_model(model=LinearRegression())
        )
        train.predict_outcomes(on="target")
        expected = _interval(train)
        holdout = (
            Sample.from_frame(
                pd.DataFrame(
                    {"id": ["h1", "h2"], "x": [50.0, 100.0], "y": [999.0, 888.0]}
                ),
                outcome_columns=["y"],
            )
            .set_target(_target())
            .set_fitted_outcome_model(train)
        )
        holdout.predict_outcomes(on="target")
        pd.testing.assert_frame_equal(_interval(holdout), expected)
        self.assertIsNot(
            _assert_type(holdout.outcome_model)["training_inputs"],
            _assert_type(train.outcome_model)["training_inputs"],
        )
        _assert_type(holdout.outcome_model)["training_inputs"]["outcomes"].iloc[
            0, 0
        ] = -999
        self.assertEqual(
            _assert_type(train.outcome_model)["training_inputs"]["outcomes"].iloc[0, 0],
            2.0,
        )

    def test_bootstrap_transferred_model_needs_no_holdout_outcomes(self) -> None:
        train = (
            _sample().set_target(_target()).fit_outcome_model(model=LinearRegression())
        )
        train.predict_outcomes(on="target")
        expected = _interval(train)
        holdout = (
            Sample.from_frame(pd.DataFrame({"id": ["h1", "h2"], "x": [50.0, 100.0]}))
            .set_target(_target())
            .set_fitted_outcome_model(train)
        )
        holdout.predict_outcomes(on="target")
        pd.testing.assert_frame_equal(_interval(holdout), expected)

    def test_training_snapshot_preserves_complete_rows_weights_and_copy_isolation(
        self,
    ) -> None:
        responder = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d"],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "z": [4.0, 5.0, 6.0, 7.0],
                    "y": [2.0, np.nan, 8.0, 10.0],
                    "weight": [1.0, 2.0, 3.0, 4.0],
                },
                index=pd.Index([40, 10, 30, 20]),
            ),
            outcome_columns=["y"],
        ).fit_outcome_model(variables=["x"], weighted=True, model=LinearRegression())
        model = responder.outcome_model
        self.assertIsNotNone(model)
        training = model["training_inputs"]
        self.assertEqual(training["covars"].columns.tolist(), ["x"])
        self.assertEqual(training["covars"].index.tolist(), [40, 30, 20])
        self.assertEqual(training["outcomes"].index.tolist(), [40, 30, 20])
        self.assertEqual(training["sample_weight"].tolist(), [1.0, 3.0, 4.0])
        self.assertEqual(training["sample_weight"].index.tolist(), [40, 30, 20])
        copied = deepcopy(responder)
        restored = pickle.loads(pickle.dumps(responder))
        self.assertIs(_assert_type(copied.outcome_model)["fit"]["y"], model["fit"]["y"])
        _assert_type(copied.outcome_model)["training_inputs"]["covars"].iloc[
            0, 0
        ] = -100
        self.assertEqual(training["covars"].iloc[0, 0], 0.0)
        pd.testing.assert_frame_equal(
            _assert_type(restored.outcome_model)["training_inputs"]["covars"],
            training["covars"],
        )
        pd.testing.assert_series_equal(
            _assert_type(restored.outcome_model)["training_inputs"]["sample_weight"],
            training["sample_weight"],
        )

    def test_fit_snapshot_does_not_alias_inputs(self) -> None:
        covars = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0, 10.0]})
        weights = pd.Series([1.0, 2.0, 3.0, 4.0])
        model = fit_outcome_model(
            covars, outcomes, sample_weight=weights, model=LinearRegression()
        )
        covars.iloc[0, 0] = 100
        outcomes.iloc[0, 0] = 100
        weights.iloc[0] = 100
        self.assertEqual(model["training_inputs"]["covars"].iloc[0, 0], 0.0)
        self.assertEqual(model["training_inputs"]["outcomes"].iloc[0, 0], 2.0)
        self.assertEqual(model["training_inputs"]["sample_weight"].iloc[0], 1.0)

    def test_legacy_model_without_training_provenance_rejects_bootstrap(self) -> None:
        frame = (
            _sample().set_target(_target()).fit_outcome_model(model=LinearRegression())
        )
        frame.predict_outcomes(on="target")
        _assert_type(frame.outcome_model).pop("training_inputs", None)
        with self.assertRaisesRegex(ValueError, "(?i)training.*refit|refit.*training"):
            _interval(frame)
