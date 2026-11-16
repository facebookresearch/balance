# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Replicate executed glmnet ridge fits on its exact stored BinomialExample.

glmnet 4.1-4, third-party/R/glmnet/data/BinomialExample.rda, declared GPL-2.
All 100x30 doubles/labels are reused, not regenerated. Source hash, authors,
R recipe, runtime/build compatibility and measured tolerances: fixtures/ipw/.
Matching objective: alpha=0, unpenalized intercept, weighted population scaling,
C=1/(sum(model_weights)*lambda). Output weights are inverse membership odds,
normalized to target total 44, not causal ATE inverse-probability weights.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from balance.weighting_methods.ipw import ipw


class GlmnetIPWOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(
            (
                Path(__file__).parent / "fixtures" / "ipw" / "glmnet_binomial_ipw.json"
            ).read_text()
        )
        self.values = pd.DataFrame(
            self.fixture["x"],
            columns=self.fixture["columns"],
            index=pd.Index(range(1, 101), name="R row"),
        )
        self.y = np.asarray(self.fixture["y"])
        self.sample = self.values.loc[self.y == 1]
        self.target = self.values.loc[self.y == 0]
        self.sw = pd.Series(1.0, index=self.sample.index)
        self.tw = pd.Series(1.0, index=self.target.index)

    def _fit(self, balanced: bool, lambdas: int) -> dict[str, Any]:
        return ipw(
            self.sample,
            self.sw,
            self.target,
            self.tw,
            transformations=None,
            lambda_max=1.0,
            lambda_min=1.0 if lambdas == 1 else 0.1,
            num_lambdas=lambdas,
            balance_classes=balanced,
            max_de=None,
            weight_trimming_mean_ratio=None,
            store_fit_metadata=True,
        )

    def _assert_vectors(
        self,
        result: dict[str, Any],
        reference: dict[str, Any],
        probability_atol: float,
        weight_atol: float,
        weight_rtol: float = 0,
    ) -> None:
        probabilities = np.empty(100)
        probabilities[self.y == 1] = result["model"]["sample_probability"]
        probabilities[self.y == 0] = result["model"]["target_probability"]
        np.testing.assert_allclose(
            probabilities, reference["probability"], atol=probability_atol, rtol=0
        )
        np.testing.assert_allclose(
            result["weight"],
            reference["sample_weight"],
            atol=weight_atol,
            rtol=weight_rtol,
        )
        pd.testing.assert_index_equal(result["weight"].index, self.sample.index)
        self.assertAlmostEqual(float(result["weight"].sum()), 44.0, places=12)
        self.assertEqual(result["model"]["X_matrix_columns"], self.fixture["columns"])

    def test_all_fitted_probabilities_and_weights_match_glmnet(self) -> None:
        for balanced in (False, True):
            with self.subTest(balanced=balanced):
                reference = self.fixture["cases"][
                    f"lambda_1_balanced_{str(balanced).upper()}"
                ]
                result = self._fit(balanced, lambdas=1)
                self._assert_vectors(
                    result, reference, probability_atol=2e-4, weight_atol=5e-4
                )
                self.assertAlmostEqual(
                    float(result["model"]["fit"].C),
                    1 / (112 if balanced else 100),
                    places=14,
                )

    def test_cross_validated_selected_model_is_the_full_data_glmnet_fit(self) -> None:
        for balanced in (False, True):
            with self.subTest(balanced=balanced):
                result = self._fit(balanced, lambdas=2)
                self.assertAlmostEqual(float(result["model"]["lambda"]), 0.1)
                reference = self.fixture["cases"][
                    f"lambda_0.1_balanced_{str(balanced).upper()}"
                ]
                # Measured warm-start solver tolerance; a last-CV-fold fit differs
                # by about .18-.20 in probability and fails this full-vector check.
                self._assert_vectors(
                    result,
                    reference,
                    probability_atol=5e-4,
                    weight_atol=1e-5,
                    weight_rtol=2e-3,
                )
