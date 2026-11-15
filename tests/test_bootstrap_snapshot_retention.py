# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from balance.outcome_models import (
    bootstrap_outcome_estimate,
    fit_outcome_model,
    predict_outcome,
)
from sklearn.linear_model import LinearRegression


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    covars = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    outcomes = pd.DataFrame({"y": [2.0, 4.0, 8.0, 10.0]})
    weights = pd.Series([1.0, 2.0, 3.0, 4.0], name="design_weight")
    target = pd.DataFrame({"x": [0.0, 2.0]})
    target_weights = pd.Series([1.0, 3.0])
    return covars, outcomes, weights, target, target_weights


class BootstrapSnapshotRetentionTest(unittest.TestCase):
    def test_temporary_fits_omit_snapshots_without_changing_interval(self) -> None:
        covars, outcomes, weights, target, target_weights = _inputs()
        fitted = fit_outcome_model(
            covars, outcomes, sample_weight=weights, model=LinearRegression()
        )
        snapshots = deepcopy(fitted["training_inputs"])
        retained: list[bool] = []

        def record_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            model = fit_outcome_model(*args, **kwargs)
            retained.append("training_inputs" in model)
            return model

        for supplied_model in (None, fitted):
            with self.subTest(supplied_model=supplied_model is not None):
                retained.clear()
                fit_kwargs: dict[str, Any] = {
                    "model": LinearRegression(),
                    "_store_training_inputs": True,
                }
                with patch(
                    "balance.outcome_models.outcome_model.fit_outcome_model",
                    side_effect=record_fit,
                ):
                    result = bootstrap_outcome_estimate(
                        covars,
                        outcomes,
                        weights,
                        target,
                        target_weights,
                        fit_kwargs=fit_kwargs,
                        fitted_model=supplied_model,
                        n_bootstrap=5,
                        random_seed=11,
                    )
                self.assertTrue(retained)
                self.assertFalse(any(retained))
                self.assertAlmostEqual(result["y"]["estimate"], 6.0)
                self.assertAlmostEqual(result["y"]["ci_low"], 5.669943019943021)
                self.assertAlmostEqual(result["y"]["ci_high"], 6.95)
                self.assertTrue(fit_kwargs["_store_training_inputs"])
        pd.testing.assert_frame_equal(
            fitted["training_inputs"]["covars"], snapshots["covars"]
        )
        pd.testing.assert_frame_equal(
            fitted["training_inputs"]["outcomes"], snapshots["outcomes"]
        )
        pd.testing.assert_series_equal(
            fitted["training_inputs"]["sample_weight"], snapshots["sample_weight"]
        )

    def test_internal_snapshot_free_fit_preserves_predictions(self) -> None:
        covars, outcomes, weights, target, _ = _inputs()
        retained = fit_outcome_model(
            covars, outcomes, sample_weight=weights, model=LinearRegression()
        )
        transient = fit_outcome_model(
            covars,
            outcomes,
            sample_weight=weights,
            model=LinearRegression(),
            _store_training_inputs=False,
        )
        self.assertIn("training_inputs", retained)
        self.assertNotIn("training_inputs", transient)
        np.testing.assert_array_equal(
            predict_outcome(retained, target)["y"],
            predict_outcome(transient, target)["y"],
        )
        covars.iloc[0, 0] = -999.0
        outcomes.iloc[0, 0] = -999.0
        weights.iloc[0] = 999.0
        self.assertEqual(retained["training_inputs"]["covars"].iloc[0, 0], 0.0)
        self.assertEqual(retained["training_inputs"]["outcomes"].iloc[0, 0], 2.0)
        self.assertEqual(retained["training_inputs"]["sample_weight"].iloc[0], 1.0)

    def test_bootstrap_matches_snapshot_retaining_reference_refits(self) -> None:
        covars, outcomes, weights, target, target_weights = _inputs()
        fitted = fit_outcome_model(
            covars, outcomes, sample_weight=weights, model=LinearRegression()
        )
        point = np.average(predict_outcome(fitted, target)["y"], weights=target_weights)
        rng = np.random.default_rng(11)
        reference = []
        replay_kwargs: dict[str, Any] = {}
        preprocessing_state = fitted.get("preprocessing_state")
        if preprocessing_state is not None:
            replay_kwargs["preprocessing_state"] = preprocessing_state
        for _ in range(5):
            rows = rng.integers(0, len(covars), size=len(covars))
            model = fit_outcome_model(
                covars.iloc[rows].reset_index(drop=True),
                outcomes.iloc[rows].reset_index(drop=True),
                sample_weight=weights.iloc[rows].reset_index(drop=True),
                model=LinearRegression(),
                **replay_kwargs,
            )
            self.assertIn("training_inputs", model)
            reference.append(
                np.average(predict_outcome(model, target)["y"], weights=target_weights)
            )
        result = bootstrap_outcome_estimate(
            covars,
            outcomes,
            weights,
            target,
            target_weights,
            fit_kwargs={"model": LinearRegression()},
            fitted_model=fitted,
            n_bootstrap=5,
            random_seed=11,
        )
        self.assertAlmostEqual(result["y"]["estimate"], point)
        self.assertAlmostEqual(result["y"]["ci_low"], np.percentile(reference, 2.5))
        self.assertAlmostEqual(result["y"]["ci_high"], np.percentile(reference, 97.5))
