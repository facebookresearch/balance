# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from balance.balance_frame import BalanceFrame
from balance.outcome_models import fit_outcome_model, predict_outcome
from balance.sample_frame import SampleFrame
from balance.util import _assert_type
from balance.weighting_methods.cbps import cbps
from balance.weighting_methods.ipw import ipw
from sklearn.linear_model import LinearRegression, LogisticRegression

_FORMULA = "I(x ** (1/2))"


class FormulaDropConsumerRegressionTest(unittest.TestCase):
    def _inputs(
        self, *, duplicate_index: bool = False, raw_missing: bool = True
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        index = pd.Index(["dup"] * 4) if duplicate_index else pd.Index([8, 3, 9, 1])
        first = np.nan if raw_missing else -4.0
        sample = pd.DataFrame({"x": [first, -1.0, 4.0, 9.0]}, index=index)
        target = pd.DataFrame({"x": [first, 4.0, -1.0, 9.0]}, index=index)
        return (
            sample,
            pd.Series([100.0, 200.0, 3.0, 5.0], index=index, name="weight"),
            target,
            pd.Series([1000.0, 7.0, 2000.0, 11.0], index=index, name="weight"),
        )

    def _frame(
        self,
        sample: pd.DataFrame,
        sample_weights: pd.Series,
        target: pd.DataFrame,
        target_weights: pd.Series,
    ) -> BalanceFrame:
        sample_data = sample.copy()
        sample_data["weight"] = sample_weights.to_numpy()
        sample_data["id"] = [f"s{i}" for i in range(len(sample))]
        target_data = target.copy()
        target_data["weight"] = target_weights.to_numpy()
        target_data["id"] = [f"t{i}" for i in range(len(target))]
        return BalanceFrame(
            sample=SampleFrame.from_frame(sample_data),
            target=SampleFrame.from_frame(target_data),
        )

    def test_ipw_raw_and_formula_drops_match_filtered_control(self) -> None:
        for duplicate_index in (False, True):
            with self.subTest(duplicate_index=duplicate_index):
                sample, sw, target, tw = self._inputs(duplicate_index=duplicate_index)
                kwargs = {
                    "formula": _FORMULA,
                    "transformations": None,
                    "na_action": "drop",
                    "num_lambdas": 1,
                }
                result = ipw(sample, sw, target, tw, **kwargs)
                control = ipw(
                    sample.iloc[[2, 3]],
                    sw.iloc[[2, 3]],
                    target.iloc[[1, 3]],
                    tw.iloc[[1, 3]],
                    **kwargs,
                )
                weights = result["weight"]
                self.assertEqual(weights.index.tolist(), sample.index.tolist())
                self.assertEqual(weights.isna().tolist(), [True, True, False, False])
                np.testing.assert_allclose(
                    weights.iloc[[2, 3]], control["weight"], rtol=1e-10
                )
                self.assertAlmostEqual(float(weights.sum()), 18.0)

    def test_cbps_raw_and_formula_drops_match_filtered_control(self) -> None:
        sample, sw, target, tw = self._inputs(duplicate_index=True)
        kwargs = {"formula": _FORMULA, "transformations": None, "na_action": "drop"}
        result = cbps(sample, sw, target, tw, **kwargs)
        control = cbps(
            sample.iloc[[2, 3]],
            sw.iloc[[2, 3]],
            target.iloc[[1, 3]],
            tw.iloc[[1, 3]],
            **kwargs,
        )
        weights = _assert_type(result["weight"], pd.Series)
        control_weights = _assert_type(control["weight"], pd.Series)
        self.assertEqual(weights.isna().tolist(), [True, True, False, False])
        self.assertEqual(weights.index.tolist(), sample.index.tolist())
        np.testing.assert_allclose(weights.iloc[[2, 3]], control_weights, rtol=1e-8)
        self.assertAlmostEqual(float(weights.sum()), 18.0)

    def test_ipw_fit_replay_keeps_compact_matrices_and_exact_output_positions(
        self,
    ) -> None:
        sample, sw, target, tw = self._inputs(duplicate_index=True, raw_missing=False)
        fitted = self._frame(sample, sw, target, tw).fit(
            method="ipw", formula=_FORMULA, transformations=None, num_lambdas=1
        )
        sample_matrix, target_matrix = fitted.design_matrix()
        self.assertEqual(sample_matrix.shape, (2, 1))
        self.assertEqual(target_matrix.shape, (2, 1))
        self.assertEqual(sample_matrix.index.tolist(), ["dup", "dup"])
        self.assertTrue(np.isfinite(sample_matrix.to_numpy()).all())
        self.assertTrue(np.isfinite(target_matrix.to_numpy()).all())
        sample_proba, target_proba = fitted.predict_proba()
        self.assertEqual(sample_proba.isna().tolist(), [True, True, False, False])
        self.assertEqual(target_proba.isna().tolist(), [True, False, True, False])
        weights = _assert_type(fitted.weight_series, pd.Series)
        pd.testing.assert_series_equal(fitted.predict_weights(), weights)
        self.assertAlmostEqual(float(weights.sum()), 18.0)
        model = fitted.model
        self.assertIsNotNone(model)
        cached = model["model_matrix_sample"]
        fitted.design_matrix()
        self.assertIs(model["model_matrix_sample"], cached)
        np.testing.assert_allclose(
            fitted.responders.df["weight_adjusted_1"],
            fitted.weight_series,
            equal_nan=True,
        )

    def test_transfer_scores_each_retained_population_once(self) -> None:
        sample, sw, target, tw = self._inputs(raw_missing=False)
        fitted = self._frame(sample, sw, target, tw).fit(
            method="ipw", formula=_FORMULA, transformations=None, num_lambdas=1
        )
        estimator = _assert_type(fitted.model)["fit"]
        with patch.object(
            estimator, "predict_proba", wraps=estimator.predict_proba
        ) as predict:
            scored = self._frame(sample, sw, target, tw).set_fitted_model(
                fitted, inplace=False
            )
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(
            scored.predict_proba(on="sample").isna().tolist(),
            [True, True, False, False],
        )
        np.testing.assert_allclose(
            scored.predict_weights(), scored.weight_series, equal_nan=True
        )

    def test_ipw_custom_classifier_uses_retained_labels_and_weights(self) -> None:
        sample, sw, target, tw = self._inputs(raw_missing=False)
        kwargs = {
            "formula": _FORMULA,
            "transformations": None,
            "model": LogisticRegression(),
            "num_lambdas": 1,
        }
        result = ipw(sample, sw, target, tw, **kwargs)
        control = ipw(
            sample.iloc[[2, 3]],
            sw.iloc[[2, 3]],
            target.iloc[[1, 3]],
            tw.iloc[[1, 3]],
            **kwargs,
        )
        np.testing.assert_allclose(result["weight"].iloc[[2, 3]], control["weight"])
        self.assertEqual(result["weight"].isna().tolist(), [True, True, False, False])

    def test_ipw_transfer_scattering_and_weight_reproduction(self) -> None:
        sample, sw, target, tw = self._inputs(duplicate_index=True, raw_missing=False)
        fitted = self._frame(sample, sw, target, tw).fit(
            method="ipw", formula=_FORMULA, transformations=None, num_lambdas=1
        )
        sample = sample.assign(x=[-1.0, 4.0, -4.0, 9.0])
        target = target.assign(x=[4.0, -1.0, 9.0, -4.0])
        holdout = self._frame(sample, sw, target, tw)
        control = self._frame(
            sample.iloc[[1, 3]],
            sw.iloc[[1, 3]],
            target.iloc[[0, 2]],
            tw.iloc[[0, 2]],
        )
        predicted = fitted.predict_weights(data=holdout)
        expected = fitted.predict_weights(data=control)
        self.assertEqual(predicted.isna().tolist(), [True, False, True, False])
        np.testing.assert_allclose(predicted.iloc[[1, 3]], expected)
        self.assertAlmostEqual(float(predicted.sum()), 3000.0)
        proba = fitted.predict_proba(on="sample", data=holdout)
        self.assertEqual(proba.isna().tolist(), [True, False, True, False])
        self.assertEqual(fitted.design_matrix(on="sample", data=holdout).shape, (2, 1))
        scored = holdout.set_fitted_model(fitted, inplace=False)
        pd.testing.assert_series_equal(scored.weight_series, predicted)
        pd.testing.assert_series_equal(scored.predict_weights(), predicted)
        pd.testing.assert_series_equal(scored.predict_proba(on="sample"), proba)

    def test_cbps_replay_normalizes_retained_target_mass(self) -> None:
        sample, sw, target, tw = self._inputs(duplicate_index=True, raw_missing=False)
        fitted = self._frame(sample, sw, target, tw).fit(
            method="cbps", formula=_FORMULA, transformations=None
        )
        replay = fitted.predict_weights()
        np.testing.assert_allclose(
            replay, fitted.weight_series, equal_nan=True, rtol=1e-8
        )
        self.assertAlmostEqual(float(replay.sum()), 18.0)
        transfer = fitted.predict_weights(data=self._frame(sample, sw, target, tw))
        np.testing.assert_allclose(transfer, replay, equal_nan=True, rtol=1e-8)
        self.assertEqual(transfer.isna().tolist(), [True, True, False, False])

    def test_outcome_fit_aligns_labels_scaler_weights_and_output_positions(
        self,
    ) -> None:
        index = pd.Index(["dup", "dup", "dup"])
        covars = pd.DataFrame({"x": [-1.0, 4.0, 9.0]}, index=index)
        outcomes = pd.DataFrame({"y": [999.0, 10.0, 20.0]}, index=index)
        weights = pd.Series([200.0, 3.0, 5.0], index=index)
        kwargs = {
            "model": LinearRegression(),
            "formula": _FORMULA,
            "use_model_matrix": True,
        }
        fitted = fit_outcome_model(covars, outcomes, sample_weight=weights, **kwargs)
        control = fit_outcome_model(
            covars.iloc[[1, 2]],
            outcomes.iloc[[1, 2]],
            sample_weight=weights.iloc[[1, 2]],
            **kwargs,
        )
        self.assertEqual(fitted["perf"]["y"]["n"], 2)
        np.testing.assert_allclose(fitted["fit"]["y"].coef_, control["fit"]["y"].coef_)
        np.testing.assert_allclose(
            fitted["fit_scaler"].mean_, control["fit_scaler"].mean_
        )
        pd.testing.assert_frame_equal(fitted["training_inputs"]["covars"], covars)
        pd.testing.assert_frame_equal(fitted["training_inputs"]["outcomes"], outcomes)
        pd.testing.assert_series_equal(
            fitted["training_inputs"]["sample_weight"], weights
        )
        predictions = predict_outcome(fitted, covars)["y"]
        self.assertEqual(np.isnan(predictions).tolist(), [True, False, False])
        np.testing.assert_allclose(predictions[[1, 2]], [10.0, 20.0])

    def test_all_dropped_outcome_scoring_returns_missing_outputs(self) -> None:
        covars = pd.DataFrame({"x": [4.0, 9.0]})
        fitted = fit_outcome_model(
            covars,
            pd.DataFrame({"y": [10.0, 20.0]}),
            model=LinearRegression(),
            formula=_FORMULA,
            use_model_matrix=True,
        )
        predictions = predict_outcome(fitted, pd.DataFrame({"x": [-1.0, -4.0]}))["y"]
        self.assertEqual(np.isnan(predictions).tolist(), [True, True])

    def test_all_dropped_weighting_scoring_returns_missing_outputs(self) -> None:
        sample, sw, target, tw = self._inputs(raw_missing=False)
        dropped = self._frame(
            sample.assign(x=[-1.0] * 4), sw, target.assign(x=[-1.0] * 4), tw
        )
        for method in ("ipw", "cbps"):
            with self.subTest(method=method):
                fitted = self._frame(sample, sw, target, tw).fit(
                    method=method, formula=_FORMULA, transformations=None, num_lambdas=1
                )
                self.assertEqual(
                    fitted.predict_weights(data=dropped).isna().tolist(), [True] * 4
                )
                if method == "ipw":
                    self.assertEqual(
                        fitted.predict_proba(on="sample", data=dropped).isna().tolist(),
                        [True] * 4,
                    )
                    self.assertEqual(
                        fitted.design_matrix(on="sample", data=dropped).shape, (0, 1)
                    )

    def test_formula_dropping_positive_weight_rows_leaves_no_fit_mass(self) -> None:
        sample = pd.DataFrame({"x": [4.0, -1.0]})
        sample_weights = pd.Series([0.0, 1.0])
        target = pd.DataFrame({"x": [4.0, 9.0]})
        target_weights = pd.Series([1.0, 1.0])
        for method in (ipw, cbps):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(
                    ValueError, "positive finite retained weight sums"
                ):
                    method(
                        sample,
                        sample_weights,
                        target,
                        target_weights,
                        formula=_FORMULA,
                        transformations=None,
                    )
                with self.assertRaisesRegex(
                    ValueError, "positive finite retained weight sums"
                ):
                    method(
                        target,
                        target_weights,
                        sample,
                        sample_weights,
                        formula=_FORMULA,
                        transformations=None,
                    )

    def test_empty_fit_and_lost_outcome_class_are_actionable_failures(self) -> None:
        invalid = pd.DataFrame({"x": [-1.0, -4.0]})
        weights = pd.Series([1.0, 1.0])
        valid = pd.DataFrame({"x": [4.0, 9.0]})
        for method in (ipw, cbps):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "empty sample or target"):
                    method(
                        invalid,
                        weights,
                        valid,
                        weights,
                        formula=_FORMULA,
                        transformations=None,
                    )
        with self.assertRaisesRegex(ValueError, "no rows"):
            fit_outcome_model(
                invalid,
                pd.DataFrame({"y": [1.0, 2.0]}),
                model=LinearRegression(),
                formula=_FORMULA,
                use_model_matrix=True,
            )
        with self.assertRaisesRegex(ValueError, "single observed class"):
            fit_outcome_model(
                pd.DataFrame({"x": [-1.0, 4.0, 9.0]}),
                pd.DataFrame({"y": [0, 1, 1]}),
                model=LogisticRegression(),
                formula=_FORMULA,
                use_model_matrix=True,
            )
