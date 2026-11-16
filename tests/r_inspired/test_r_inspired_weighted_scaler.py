# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Weighted preprocessing on every stored glmnet BinomialExample value.

glmnet 4.1-4, third-party/R/glmnet/data/BinomialExample.rda, source GPL-2.
The fixture captures independently executed R weighted population moments.
Exact hashes, authors, runtime/configuration and source
notices are in fixtures/weighted_scaler/glmnet_binomial_ipw.json and README.md.
Constant-support/replay assertions are additional balance contract guards on
those same actual values; they are not a claim about glmnet dropping constants.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from balance.utils.model_matrix import build_design_matrix


class RInspiredWeightedScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(
            (
                Path(__file__).parent
                / "fixtures"
                / "weighted_scaler"
                / "glmnet_binomial_ipw.json"
            ).read_text()
        )
        self.values = pd.DataFrame(
            self.fixture["x"],
            columns=self.fixture["columns"],
            index=pd.Index(range(1, 101)),
        )
        self.y = np.asarray(self.fixture["y"])
        self.sample = self.values.loc[self.y == 1]
        self.target = self.values.loc[self.y == 0]
        self.order = np.r_[np.flatnonzero(self.y == 1), np.flatnonzero(self.y == 0)]

    def test_scaler_uses_weighted_population_moments_without_centering(self) -> None:
        for balanced in (False, True):
            with self.subTest(balanced=balanced):
                weights = np.where(self.y == 1, 1.0, 56 / 44 if balanced else 1.0)
                result = build_design_matrix(
                    self.sample, self.target, scaler_weights=weights[self.order]
                )
                reference = self.fixture["scalers"][str(balanced).upper()]
                np.testing.assert_allclose(
                    result["fit_scaler"].mean_, reference["mean"], atol=1e-12, rtol=0
                )
                np.testing.assert_allclose(
                    result["fit_scaler"].scale_, reference["scale"], atol=1e-12, rtol=0
                )
                expected = self.values.to_numpy()[self.order] / np.asarray(
                    reference["scale"]
                )
                np.testing.assert_allclose(
                    result["combined_matrix"].toarray(), expected, atol=1e-12, rtol=0
                )
                self.assertEqual(result["columns"], self.fixture["columns"])
                self.assertEqual(result["sample_n"], 56)
                np.testing.assert_array_equal(result["sample_rows"], np.arange(56))
                np.testing.assert_array_equal(result["target_rows"], np.arange(44))

    def test_zero_weights_make_a_varying_column_weighted_constant(self) -> None:
        weights = np.zeros(100)
        weights[0] = 1.0
        result = build_design_matrix(self.sample, self.target, scaler_weights=weights)
        scaler = result["fit_scaler"]
        np.testing.assert_allclose(
            scaler.mean_, self.sample.iloc[0], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(scaler.var_, np.zeros(30), atol=1e-12, rtol=0)
        np.testing.assert_allclose(scaler.scale_, np.ones(30), atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            result["combined_matrix"].toarray(),
            self.values.to_numpy()[self.order],
            atol=1e-12,
            rtol=0,
        )

    def test_scoring_keeps_fit_time_scale_despite_changed_distribution(self) -> None:
        fit = build_design_matrix(self.sample, self.target, scaler_weights=np.ones(100))
        scoring = self.values.iloc[[99, 25, 4]]
        result = build_design_matrix(
            scoring,
            scoring.iloc[:0],
            fit_scaler=fit["fit_scaler"],
            project_to_columns=fit["columns"],
            preprocessing_state=fit["preprocessing_state"],
        )
        reference = np.asarray(self.fixture["scalers"]["FALSE"]["scale"])
        np.testing.assert_allclose(
            result["combined_matrix"].toarray(),
            scoring.to_numpy() / reference,
            atol=1e-12,
            rtol=0,
        )
        self.assertEqual(result["columns"], self.fixture["columns"])
        np.testing.assert_array_equal(result["sample_rows"], [0, 1, 2])
        np.testing.assert_allclose(
            fit["fit_scaler"].scale_, reference, atol=1e-12, rtol=0
        )
