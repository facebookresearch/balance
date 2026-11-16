# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from balance.weighting_methods.ipw import calc_dev
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Behavioral inspiration: rsample 1.3.2, tests/testthat/test-vfold.R:184-219.
# Source license: MIT + file LICENSE; YEAR: 2025;
# COPYRIGHT HOLDER: rsample authors. https://rsample.tidymodels.org/
# Inspected at third-party/R/rsample, fbsource revision
# 5535fa9ed74ed389900dd9d9858c15b44cb3f4ac.
# Original Python classifier/row-ID fixture. The R example uses warpbreaks;
# no warpbreaks measurements, upstream fixture values, or R code are copied.


class _RecordingFoldClassifier(ClassifierMixin):
    def __init__(self, varied: bool) -> None:
        self.varied = varied
        self.fit_rows: list[list[int]] = []
        self.fit_weights: list[list[float]] = []
        self.fit_labels: list[list[int]] = []
        self.scored_rows: list[list[int]] = []

    def fit(
        self, X: csr_matrix, y: np.ndarray, sample_weight: np.ndarray
    ) -> _RecordingFoldClassifier:
        self.fit_rows.append(X.toarray()[:, 0].astype(int).tolist())
        self.fit_weights.append(sample_weight.tolist())
        self.fit_labels.append(y.tolist())
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        rows = X.toarray()[:, 0].astype(int)
        self.scored_rows.append(rows.tolist())
        probability = (int(rows[0]) // 4 + 1) / 12 if self.varied else 0.5
        return np.column_stack(
            (np.full(len(rows), 1 - probability), np.full(len(rows), probability))
        )


class RInspiredCrossValidationTest(unittest.TestCase):
    def test_each_fit_and_prediction_use_their_supplied_fold_rows_and_weights(
        self,
    ) -> None:
        rows = np.arange(40)
        labels = np.array([0, 1] * 20)
        weights = rows.astype(float) + 1
        folds = np.repeat(np.arange(10), 4)
        classifier = _RecordingFoldClassifier(varied=False)
        mean, se = calc_dev(
            csr_matrix(rows.reshape(-1, 1)), labels, classifier, weights, folds
        )
        self.assertEqual(len(classifier.fit_rows), 10)
        self.assertEqual(len(classifier.scored_rows), 10)
        for fold in range(10):
            with self.subTest(fold=fold):
                train = rows[folds != fold]
                holdout = rows[folds == fold]
                self.assertEqual(classifier.fit_rows[fold], train.tolist())
                self.assertEqual(classifier.scored_rows[fold], holdout.tolist())
                self.assertEqual(
                    classifier.fit_weights[fold], weights[folds != fold].tolist()
                )
                self.assertEqual(
                    classifier.fit_labels[fold], labels[folds != fold].tolist()
                )
                self.assertFalse(
                    set(classifier.fit_rows[fold]) & set(classifier.scored_rows[fold])
                )
        self.assertAlmostEqual(float(mean), 2 * np.log(2))
        self.assertAlmostEqual(float(se), 0.0)

    def test_weighted_holdout_losses_determine_both_cv_mean_and_standard_error(
        self,
    ) -> None:
        rows = np.arange(40)
        folds = np.repeat(np.arange(10), 4)
        classifier = _RecordingFoldClassifier(varied=True)
        mean, se = calc_dev(
            csr_matrix(rows.reshape(-1, 1)),
            np.array([0, 1] * 20),
            classifier,
            rows.astype(float) + 1,
            folds,
        )
        # In fold i, class-0 weights sum to 8i+4 and class-1 weights to 8i+6.
        expected = []
        for fold in range(10):
            probability = (fold + 1) / 12
            loss = (
                -2
                * (
                    (8 * fold + 4) * np.log(1 - probability)
                    + (8 * fold + 6) * np.log(probability)
                )
                / (16 * fold + 10)
            )
            expected.append(loss)
        self.assertAlmostEqual(float(mean), float(np.mean(expected)))
        self.assertAlmostEqual(float(se), float(np.std(expected, ddof=1) / np.sqrt(10)))
        self.assertEqual(len(classifier.scored_rows), 10)


class GlmnetCrossValidationOracleTest(unittest.TestCase):
    """Executed glmnet 4.1-4 folds on all stored BinomialExample values.

    Source GPL-2; data hash, authors, generator and R4.6 compatibility metadata
    are in fixtures/cv/glmnet_binomial_ipw.json. These are real fitted R models,
    not the partition-spy guard above. Global scaling and fixed full-data C
    require lambda_fold=lambda_full*S_full/S_train, not ordinary cv.glmnet.
    """

    def test_calc_dev_matches_independently_fitted_glmnet_folds(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).parent / "fixtures" / "cv" / "glmnet_binomial_ipw.json"
            ).read_text()
        )
        values = np.asarray(fixture["x"])
        y = np.asarray(fixture["y"])
        order = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
        folds = np.asarray(fixture["foldids_sample_first"])
        for balanced in (False, True):
            with self.subTest(balanced=balanced):
                weights = np.where(y == 1, 1.0, 56 / 44 if balanced else 1.0)
                scaler = StandardScaler(with_mean=False).fit(
                    values, sample_weight=weights
                )
                matrix = scaler.transform(values)[order]
                model = LogisticRegression(
                    C=1 / weights.sum(),
                    solver="lbfgs",
                    tol=1e-4,
                    max_iter=5000,
                    warm_start=True,
                )
                model.fit(matrix, y[order], sample_weight=weights[order])
                mean, se = calc_dev(
                    csr_matrix(matrix), y[order], model, weights[order], folds
                )
                reference = fixture["cases"][
                    f"lambda_1_balanced_{str(balanced).upper()}"
                ]
                np.testing.assert_allclose(
                    [mean, se],
                    [reference["cv_mean"], reference["cv_se"]],
                    atol=5e-5,
                    rtol=0,
                )
