# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from balance import Sample
from balance.balance_frame import BalanceFrame
from balance.outcome_models import (
    bootstrap_outcome_estimate,
    fit_outcome_model,
    predict_outcome,
)
from balance.sample_frame import SampleFrame
from balance.testutil import _SKLEARN_1_4_AVAILABLE
from balance.utils.input_validation import _assert_type
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


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


class OutcomePredictionPreservationTest(unittest.TestCase):
    def _populated(self) -> Sample:
        frame = (
            _sample().set_target(_target()).fit_outcome_model(model=LinearRegression())
        )
        frame.predict_outcomes(on="both")
        return frame

    def _assert_refresh_warning(self, output: list[str]) -> None:
        warning = " ".join(output)
        self.assertIn("Existing target predictions", warning)
        self.assertIn("y_hat", warning)
        self.assertIn("preserved", warning)
        self.assertIn("stale", warning)
        self.assertIn("frame.predict_outcomes(on='target')", warning)

    def _assert_preserved_then_refresh(self, frame: BalanceFrame) -> None:
        self.assertIsNotNone(_assert_type(frame.target).df_outcomes_hat)
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 4.32
        )
        predictions = frame.predict_outcomes(on="target")
        target = _assert_type(frame.target, SampleFrame)
        expected = np.average(predictions["y_hat"], weights=target.weight_series)
        self.assertAlmostEqual(expected, 3.2142857142857144)
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], expected
        )

    def test_refit_preserves_target_predictions_and_warns_until_refresh(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {"id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "y": [2.0, 4.0, 6.0]}
            ),
            outcome_columns=["y"],
        )
        target = Sample.from_frame(pd.DataFrame({"id": ["t1", "t2"], "x": [1.0, 2.0]}))
        frame = sample.set_target(target).fit_outcome_model(
            model=LinearRegression(fit_intercept=True), transformations=None
        )
        np.testing.assert_allclose(
            frame.predict_outcomes(on="target")["y_hat"], [4.0, 6.0]
        )
        shared_target = _assert_type(frame.target)
        with self.assertLogs("balance", level="WARNING") as logs:
            frame.fit_outcome_model(
                model=LinearRegression(fit_intercept=False), transformations=None
            )
        self._assert_refresh_warning(logs.output)
        self.assertIs(frame.target, shared_target)
        np.testing.assert_allclose(
            _assert_type(shared_target.df_outcomes_hat)["y_hat"], [4.0, 6.0]
        )
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 5.0
        )
        np.testing.assert_allclose(
            frame.predict_outcomes(on="target")["y_hat"], [3.2, 6.4]
        )
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 4.8
        )
        np.testing.assert_allclose(
            _assert_type(shared_target.df_outcomes_hat)["y_hat"], [4.0, 6.0]
        )

    def test_transfer_preserves_target_predictions_and_warns_until_refresh(
        self,
    ) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {"id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "y": [2.0, 4.0, 6.0]}
            ),
            outcome_columns=["y"],
        )
        target = Sample.from_frame(pd.DataFrame({"id": ["t1", "t2"], "x": [1.0, 2.0]}))
        frame = sample.set_target(target).fit_outcome_model(
            model=LinearRegression(), transformations=None
        )
        np.testing.assert_allclose(
            frame.predict_outcomes(on="target")["y_hat"], [4.0, 6.0]
        )
        replacement = sample.fit_outcome_model(
            model=LinearRegression(fit_intercept=False), inplace=False
        )
        with self.assertLogs("balance", level="WARNING") as logs:
            frame.set_fitted_outcome_model(replacement)
        self._assert_refresh_warning(logs.output)
        np.testing.assert_allclose(
            _assert_type(_assert_type(frame.target).df_outcomes_hat)["y_hat"],
            [4.0, 6.0],
        )
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 5.0
        )
        np.testing.assert_allclose(
            frame.predict_outcomes(on="target")["y_hat"], [3.2, 6.4]
        )
        self.assertAlmostEqual(
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"], 4.8
        )

    def test_refit_copy_preserves_predictions_and_leaves_original_unchanged(
        self,
    ) -> None:
        original = self._populated()
        before = _assert_type(original.target).df.copy(deep=True)
        sample_before = _assert_type(original.df_outcomes_hat).copy(deep=True)
        with self.assertLogs("balance", level="WARNING") as logs:
            fitted = original.fit_outcome_model(
                model=LinearRegression(fit_intercept=False), inplace=False
            )
        self._assert_refresh_warning(logs.output)
        pd.testing.assert_frame_equal(fitted.df_outcomes_hat, sample_before)
        self._assert_preserved_then_refresh(fitted)
        pd.testing.assert_frame_equal(_assert_type(original.target).df, before)

    def test_transfer_copy_preserves_predictions_and_leaves_original_unchanged(
        self,
    ) -> None:
        original = self._populated()
        before = _assert_type(original.target).df.copy(deep=True)
        sample_before = _assert_type(original.df_outcomes_hat).copy(deep=True)
        train = _sample().fit_outcome_model(model=LinearRegression(fit_intercept=False))
        with self.assertLogs("balance", level="WARNING") as logs:
            fitted = original.set_fitted_outcome_model(train, inplace=False)
        self._assert_refresh_warning(logs.output)
        pd.testing.assert_frame_equal(fitted.df_outcomes_hat, sample_before)
        self._assert_preserved_then_refresh(fitted)
        pd.testing.assert_frame_equal(_assert_type(original.target).df, before)

    def test_refit_does_not_mutate_shared_populated_target(self) -> None:
        first = self._populated()
        shared = _assert_type(first.target, SampleFrame)
        second = BalanceFrame(sample=_sample()._sf_sample, target=shared)
        before = shared.df.copy(deep=True)
        with self.assertLogs("balance", level="WARNING") as logs:
            first.fit_outcome_model(model=LinearRegression(fit_intercept=False))
        self._assert_refresh_warning(logs.output)
        self.assertIs(first.target, shared)
        self._assert_preserved_then_refresh(first)
        self.assertIsNot(first.target, shared)
        pd.testing.assert_frame_equal(_assert_type(second.target).df, before)

    def test_transfer_does_not_mutate_shared_populated_target(self) -> None:
        first = self._populated()
        shared = _assert_type(first.target, SampleFrame)
        second = BalanceFrame(sample=_sample()._sf_sample, target=shared)
        before = shared.df.copy(deep=True)
        train = _sample().fit_outcome_model(model=LinearRegression(fit_intercept=False))
        with self.assertLogs("balance", level="WARNING") as logs:
            first.set_fitted_outcome_model(train)
        self._assert_refresh_warning(logs.output)
        self.assertIs(first.target, shared)
        self._assert_preserved_then_refresh(first)
        self.assertIsNot(first.target, shared)
        pd.testing.assert_frame_equal(_assert_type(second.target).df, before)

    def test_manual_predictions_and_metadata_are_preserved(self) -> None:
        frame = self._populated()
        target = _assert_type(frame.target)
        target.add_outcomes_hat_column(
            "manual_hat", pd.Series(np.arange(len(target.df)), index=target.df.index)
        )
        target_before = target.df.copy(deep=True)
        roles = list(target.outcomes_hat_columns)
        metadata = dict(target._prediction_metadata)
        sample_before = _assert_type(frame.df_outcomes_hat).copy(deep=True)
        for action in ("refit", "transfer"):
            with self.subTest(action=action):
                with self.assertLogs("balance", level="WARNING") as logs:
                    if action == "refit":
                        frame.fit_outcome_model(
                            model=LinearRegression(fit_intercept=False)
                        )
                    else:
                        frame.set_fitted_outcome_model(
                            _sample().fit_outcome_model(model=LinearRegression())
                        )
                self._assert_refresh_warning(logs.output)
                self.assertIn("manual_hat", " ".join(logs.output))
                pd.testing.assert_frame_equal(target.df, target_before)
                self.assertEqual(target.outcomes_hat_columns, roles)
                self.assertEqual(target._prediction_metadata, metadata)
                pd.testing.assert_frame_equal(frame.df_outcomes_hat, sample_before)

    def test_no_warning_when_no_predictions_exist(self) -> None:
        frame = _sample().set_target(_target())
        train = _sample().fit_outcome_model(model=LinearRegression())
        with patch("balance.balance_frame.logger.warning") as warning:
            frame.fit_outcome_model(model=LinearRegression())
            frame.set_fitted_outcome_model(train)
        warning.assert_not_called()
        self.assertIsNone(_assert_type(frame.target).df_outcomes_hat)

    def test_failed_refit_preserves_predictions_and_does_not_warn_of_a_new_model(
        self,
    ) -> None:
        frame = self._populated()
        model_before = frame.outcome_model
        before = _assert_type(frame.target).df.copy(deep=True)
        sample_before = _assert_type(frame.df_outcomes_hat).copy(deep=True)
        with patch("balance.balance_frame.logger.warning") as warning:
            with self.assertRaises(ValueError):
                frame.fit_outcome_model(transformations={"x": "quantize"})
        warning.assert_not_called()
        self.assertIs(frame.outcome_model, model_before)
        pd.testing.assert_frame_equal(_assert_type(frame.target).df, before)
        pd.testing.assert_frame_equal(frame.df_outcomes_hat, sample_before)


class MixedOutcomeLearnersTest(unittest.TestCase):
    def _data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        covars = pd.DataFrame({"cat": ["a", "b", "a", "b"], "x": [0.0, 1.0, 2.0, 3.0]})
        outcomes = pd.DataFrame(
            {"reg": [2.0, 4.0, 6.0, 8.0], "binary": [0.0, 1.0, 0.0, 1.0]}
        )
        return covars, outcomes

    def _learners(self, by_type: bool = False) -> dict[str, Any]:
        reg = HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            max_iter=2,
            learning_rate=1.0,
            min_samples_leaf=1,
            early_stopping=False,
            random_state=0,
        )
        clf = LogisticRegression(
            solver="liblinear", C=1.0, max_iter=100, random_state=0
        )
        return (
            {"_continuous": reg, "_discrete": clf}
            if by_type
            else {"reg": reg, "binary": clf}
        )

    def _assert_orders(
        self, *, by_type: bool = False, mode: bool | str = "auto"
    ) -> None:
        covars, outcomes = self._data()
        holdout = covars.iloc[[3, 0]]
        reference = fit_outcome_model(
            covars,
            outcomes,
            model=self._learners(by_type),
            transformations=None,
            use_model_matrix=True,
        )
        expected = predict_outcome(reference, holdout)
        for order in (["reg", "binary"], ["binary", "reg"]):
            with self.subTest(order=order):
                model = fit_outcome_model(
                    covars,
                    outcomes[order],
                    model=self._learners(by_type),
                    transformations=None,
                    use_model_matrix=mode,
                )
                self.assertTrue(model["use_model_matrix"])
                predictions = predict_outcome(model, holdout)
                restored = pickle.loads(pickle.dumps(model))
                replayed = predict_outcome(restored, holdout)
                self.assertEqual(set(predictions), {"reg", "binary"})
                np.testing.assert_allclose(predictions["reg"], [8.0, 2.0], atol=1e-12)
                for column in predictions:
                    np.testing.assert_allclose(
                        predictions[column], expected[column], atol=1e-12
                    )
                    np.testing.assert_array_equal(replayed[column], predictions[column])

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_column_map_both_orders_pickle_and_replay(self) -> None:
        self._assert_orders()

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_type_map_both_orders_pickle_and_replay(self) -> None:
        self._assert_orders(by_type=True)

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_explicit_one_hot_both_orders_pickle_and_replay(self) -> None:
        self._assert_orders(mode=True)

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_explicit_native_mixed_categorical_has_actionable_error(self) -> None:
        covars, outcomes = self._data()
        for order in (["reg", "binary"], ["binary", "reg"]):
            with (
                self.subTest(order=order),
                self.assertRaisesRegex(ValueError, "every learner.*use_model_matrix"),
            ):
                fit_outcome_model(
                    covars,
                    outcomes[order],
                    model=self._learners(),
                    transformations=None,
                    use_model_matrix=False,
                )


class OutcomePreprocessingReplayTest(unittest.TestCase):
    def _data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        covars = pd.DataFrame(
            {
                "grp": pd.Categorical(
                    ["a", "b", "c", None] * 3, categories=["c", "a", "b"], ordered=True
                ),
                "x": [0.0, 1.0, 2.0, np.nan] * 3,
            }
        )
        outcomes = pd.DataFrame({"y": np.arange(12, dtype=float) + 2.5})
        return covars, outcomes

    def test_linear_partition_and_pickle_replay_preserve_frozen_state(self) -> None:
        covars = pd.DataFrame({"grp": ["a", "b", "c"]})
        outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0]})
        model = fit_outcome_model(
            covars,
            outcomes,
            model=LinearRegression(fit_intercept=True),
            transformations=None,
            use_model_matrix=True,
            formula="C(grp, Sum)",
        )
        np.testing.assert_allclose(predict_outcome(model, covars)["y"], [2.0, 4.0, 8.0])
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(
            predict_outcome(restored, pd.DataFrame({"grp": ["b", "a"]}))["y"],
            [4.0, 2.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            predict_outcome(restored, pd.DataFrame({"grp": ["b", "c"]}))["y"],
            [4.0, 8.0],
            atol=1e-12,
        )
        self.assertEqual(restored["preprocessing_state"], model["preprocessing_state"])

    def test_one_hot_replay_rejects_unseen_categories(self) -> None:
        covars, outcomes = self._data()
        model = fit_outcome_model(covars, outcomes, model=LinearRegression())
        with self.assertRaisesRegex(ValueError, "unseen categories"):
            predict_outcome(model, pd.DataFrame({"grp": ["novel"], "x": [1.0]}))

    def test_stateful_formula_replay_rejects_fitted_transform(self) -> None:
        covars = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0, 10.0]})
        model = fit_outcome_model(
            covars, outcomes, model=LinearRegression(), formula="center(x)"
        )
        self.assertIsInstance(model.get("preprocessing_state"), dict)
        self.assertEqual(
            model["preprocessing_state"]["unsupported_stateful_formulas"], ["center(x)"]
        )
        with self.assertRaisesRegex(ValueError, "stateful"):
            predict_outcome(model, pd.DataFrame({"x": [10.0, 20.0]}))

    def test_frame_copy_and_transfer_isolate_frozen_state_share_estimators(
        self,
    ) -> None:
        covars, outcomes = self._data()
        data = covars.assign(id=[str(i) for i in range(len(covars))], y=outcomes["y"])
        train = SampleFrame.from_frame(data, outcome_columns=["y"]).fit_outcome_model(
            model=LinearRegression()
        )
        model = _assert_type(train.outcome_model)
        self.assertIsInstance(model.get("preprocessing_state"), dict)
        copied = deepcopy(train)
        copied_model = _assert_type(copied.outcome_model)
        self.assertIs(copied_model["fit"]["y"], model["fit"]["y"])
        self.assertIsNot(
            copied_model["preprocessing_state"], model["preprocessing_state"]
        )
        scored = BalanceFrame(
            sample=SampleFrame.from_frame(data, outcome_columns=["y"])
        ).set_fitted_outcome_model(train)
        scored_model = _assert_type(scored.outcome_model)
        self.assertIs(scored_model["fit"]["y"], model["fit"]["y"])
        self.assertIsNot(
            scored_model["preprocessing_state"], model["preprocessing_state"]
        )
        scored_model["preprocessing_state"]["variables"].append("foreign")
        self.assertNotIn("foreign", model["preprocessing_state"]["variables"])

    def test_bootstrap_keeps_rare_training_category_domain(self) -> None:
        covars = pd.DataFrame({"grp": ["a", "a", "a", "b"]})
        outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0, 10.0]})
        responder = SampleFrame.from_frame(
            covars.assign(id=["a", "b", "c", "d"], y=outcomes["y"]),
            outcome_columns=["y"],
        )
        target = SampleFrame.from_frame(
            pd.DataFrame({"id": ["t1", "t2"], "grp": ["a", "b"]})
        )
        frame = BalanceFrame(sample=responder, target=target).fit_outcome_model(
            model=LinearRegression()
        )
        frame.predict_outcomes(on="target")
        rng = np.random.default_rng(11)
        self.assertTrue(any(3 not in rng.integers(0, 4, size=4) for _ in range(20)))
        result = _assert_type(frame.outcomes_hat()).mean_with_ci(
            n_bootstrap=20, random_seed=11, round_ndigits=8
        )
        self.assertTrue(np.isfinite(result.loc["y_hat", "target_ci"]).all())
        self.assertAlmostEqual(
            result.loc["y_hat", "target"],
            _assert_type(frame.outcomes_hat()).mean().loc["target", "y_hat"],
        )

    def test_numeric_object_formula_bootstrap_preserves_value_types(self) -> None:
        covars = pd.DataFrame({"grp": pd.Series([1] * 10 + [2] * 10, dtype=object)})
        outcomes = pd.DataFrame({"y": [2.0] * 10 + [10.0] * 10})
        target = pd.DataFrame({"grp": pd.Series([1, 2], dtype=object)})
        rng = np.random.default_rng(11)
        for _ in range(5):
            rows = rng.integers(0, 20, size=20)
            self.assertTrue((rows < 10).any() and (rows >= 10).any())
        result = bootstrap_outcome_estimate(
            covars,
            outcomes,
            None,
            target,
            pd.Series([1.0, 9.0]),
            fit_kwargs={"model": LinearRegression(), "formula": "I(grp == 1)"},
            n_bootstrap=5,
            random_seed=11,
        )
        for value in ("estimate", "ci_low", "ci_high"):
            self.assertAlmostEqual(result["y"][value], 9.2)

    @pytest.mark.requires_sklearn_1_4
    @unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, "requires sklearn >= 1.4")
    def test_native_partition_and_pickle_replay_preserve_frozen_state(self) -> None:
        covars, outcomes = self._data()
        model = fit_outcome_model(
            covars,
            outcomes,
            model=HistGradientBoostingRegressor(
                categorical_features="from_dtype",
                min_samples_leaf=2,
                max_iter=5,
                random_state=0,
            ),
        )
        self.assertIsInstance(model.get("preprocessing_state"), dict)
        full = predict_outcome(model, covars)["y"]
        restored = pickle.loads(pickle.dumps(model))
        for rows in ([1, 5, 9], [0, 4], [3], [2, 6, 10]):
            with self.subTest(rows=rows):
                np.testing.assert_array_equal(
                    predict_outcome(restored, covars.iloc[rows])["y"], full[rows]
                )
