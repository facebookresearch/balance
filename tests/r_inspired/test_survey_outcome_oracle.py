# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Prediction and AIPW oracles on survey 4.1-1's actual api data.

Source GPL-2 | GPL-3, copyright 2002-2014 Thomas Lumley except as stated;
source data references, hashes and notices: fixtures/outcomes/README.md / manifest.json.
R lm and svyglm generated every prediction. R calibrate with per-unit variance
set to design weight produces the OLS-slope GREG estimator; svymean on that
calibrated design supplies the independent matching AIPW point, including a
nonzero augmentation. Design-based uncertainty is deliberately not compared.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from balance import Sample
from balance.outcome_models import fit_outcome_model, predict_outcome
from balance.util import _assert_type
from sklearn.linear_model import LinearRegression


# Scientific-stack stubs omit this supported runtime parser literal.
_ROUND_TRIP: Any = "round_trip"


class SurveyOutcomeOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parent / "fixtures" / "outcomes"
        self.sample = pd.read_csv(
            root / "survey_api_outcome_sample.csv", float_precision=_ROUND_TRIP
        )
        self.target = pd.read_csv(
            root / "survey_api_outcome_target.csv", float_precision=_ROUND_TRIP
        )
        self.expected = pd.read_csv(
            root / "survey_api_outcome_expected.csv", float_precision=_ROUND_TRIP
        ).iloc[0]
        self.columns = ["api99", "meals", "ell"]

    def test_all_weighted_and_unweighted_predictions_match_r_models(self) -> None:
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                fitted = fit_outcome_model(
                    self.sample[self.columns],
                    self.sample[["api00"]],
                    sample_weight=self.sample.weight if weighted else None,
                    model=LinearRegression(),
                    transformations=None,
                )
                for frame in (self.sample, self.target):
                    expected = (
                        frame.expected_pred_weighted
                        if weighted
                        else frame.expected_pred_unweighted
                    )
                    actual = predict_outcome(fitted, frame[self.columns])["api00"]
                    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-12)

    def _frame(self) -> Sample:
        sample = Sample.from_frame(
            self.sample[["row_id"] + self.columns + ["api00", "weight"]],
            id_column="row_id",
            weight_column="weight",
            outcome_columns=["api00"],
        )
        target = Sample.from_frame(
            self.target[["row_id"] + self.columns], id_column="row_id"
        )
        return sample.set_target(target).adjust(method="null")

    def test_frame_prediction_transfer_and_target_mean_match_r_svyglm(self) -> None:
        fitted = self._frame()
        fitted.fit_outcome_model(
            model=LinearRegression(), transformations=None, weighted=True
        )
        fitted.predict_outcomes(on="both")
        sample_predictions = _assert_type(fitted.df_outcomes_hat, pd.DataFrame)[
            "api00_hat"
        ]
        target = _assert_type(fitted.target)
        target_predictions = _assert_type(target.df_outcomes_hat, pd.DataFrame)[
            "api00_hat"
        ]
        np.testing.assert_allclose(
            sample_predictions,
            self.sample.expected_pred_weighted,
            atol=2e-8,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            target_predictions,
            self.target.expected_pred_weighted,
            atol=2e-8,
            rtol=1e-12,
        )
        self.assertAlmostEqual(
            float(target_predictions.mean()), self.expected.mu_om_weighted, places=8
        )

    def test_aipw_matches_independent_survey_greg_with_nonzero_augmentation(
        self,
    ) -> None:
        fitted = self._frame()
        fitted.fit_outcome_model(
            model=LinearRegression(), transformations=None, weighted=False
        )
        fitted.predict_outcomes(on="both")
        actual = float(fitted.aipw()["api00"])
        self.assertAlmostEqual(actual, self.expected.mu_dr_greg, places=8)
        self.assertGreater(abs(self.expected.augmentation), 1.0)
        self.assertNotAlmostEqual(actual, self.expected.mu_om_unweighted, places=5)
        self.assertAlmostEqual(
            float(np.average(self.sample.api00, weights=self.sample.weight)),
            self.expected.mu_ipw,
            places=8,
        )
