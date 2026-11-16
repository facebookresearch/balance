# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical R statistics on exact upstream fixtures, not new synthetic values.

matrixStats 0.62.0 rowWeightedMeans.R, weightedMean_subset.R and weightedVar.R;
source Artistic-2.0, author/holder Henrik Bengtsson. All original mean assertions
ran before capture. Reliability variance comes from stats::cov.wt, not the
incompatible matrixStats frequency variance. Quantiles are survey 4.1-1
svyquantile(qrule='hf2') on its actual apiclus1 data / newquantile.R probabilities.
survey declares GPL-2 | GPL-3, copyright 2002-2014 Thomas Lumley, except as stated.
Exact values, source hashes, runtime and license notices: fixtures/weighted_statistics/.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_stats import (
    weighted_mean,
    weighted_quantile,
    weighted_var,
)


# Scientific-stack stubs omit this supported runtime parser literal.
_ROUND_TRIP: Any = "round_trip"


class RInspiredWeightedStatisticsTest(unittest.TestCase):
    def test_reliability_variance_is_not_frequency_replication_variance(self) -> None:
        root = Path(__file__).parent / "fixtures" / "weighted_statistics"
        data = pd.read_csv(root / "matrixstats_variance_inputs.csv")
        expected = pd.read_csv(root / "matrixstats_variance_expected.csv").iloc[0]
        self.assertAlmostEqual(
            float(weighted_mean(data.value, data.weight).iloc[0]), expected["mean"]
        )
        self.assertAlmostEqual(
            float(weighted_var(data.value, data.weight).iloc[0]), expected.variance
        )
        replicated = np.repeat(
            data.value.to_numpy(), data.weight.to_numpy().astype(int)
        )
        self.assertAlmostEqual(
            float(np.var(replicated, ddof=1)), expected.frequency_variance
        )
        self.assertNotAlmostEqual(
            float(weighted_var(data.value, data.weight).iloc[0]),
            expected.frequency_variance,
        )

    def test_statistics_are_invariant_to_weight_scale_and_joint_permutation(
        self,
    ) -> None:
        root = Path(__file__).parent / "fixtures" / "weighted_statistics"
        data = pd.read_csv(root / "matrixstats_variance_inputs.csv")
        expected = pd.read_csv(root / "matrixstats_variance_expected.csv").iloc[0]
        for statistic, reference in (
            (weighted_mean, expected["mean"]),
            (weighted_var, expected.variance),
        ):
            for scale in (1e-6, 1.0, 1e6):
                with self.subTest(statistic=statistic.__name__, scale=scale):
                    actual = statistic(
                        data.value.iloc[[2, 0, 1]],
                        (data.weight * scale).iloc[[2, 0, 1]],
                    )
                    self.assertAlmostEqual(float(actual.iloc[0]), reference)

    def test_repeated_position_selection_repeats_values_and_weights(self) -> None:
        root = Path(__file__).parent / "fixtures" / "weighted_statistics"
        data = pd.read_csv(root / "matrixstats_subset_inputs.csv")
        expected = pd.read_csv(root / "matrixstats_subset_expected.csv").iloc[0, 0]
        # R's c(3,0,0,5,1,5,5): zero selects nothing, other indices are 1-based.
        positions = [2, 4, 0, 4, 4]
        values = pd.Series(data.value.to_numpy(), index=[10, 10, 30, 40, 50, 60])
        weights = pd.Series(data.weight.to_numpy(), index=[90, 91, 92, 93, 94, 95])
        self.assertAlmostEqual(
            float(
                weighted_mean(values.iloc[positions], weights.iloc[positions]).iloc[0]
            ),
            expected,
        )

    def test_actual_apiclus1_quantiles_match_survey_hf2_under_weight_scaling(
        self,
    ) -> None:
        root = Path(__file__).parent / "fixtures" / "weighted_statistics"
        data = pd.read_csv(root / "survey_quantile_inputs.csv")
        expected = pd.read_csv(root / "survey_quantile_expected.csv")
        for scale in (1.0, 100.0):
            with self.subTest(scale=scale):
                actual = weighted_quantile(
                    data.value, expected.probability, data.weight * scale
                )
                np.testing.assert_allclose(
                    actual.iloc[:, 0], expected.expected, atol=1e-12, rtol=0
                )
                self.assertEqual(actual.index.tolist(), expected.probability.tolist())

    def test_columnwise_missingness_and_zero_weights_match_matrixstats(self) -> None:
        root = Path(__file__).parent / "fixtures" / "weighted_statistics"
        values = pd.read_csv(
            root / "matrixstats_values.csv", float_precision=_ROUND_TRIP
        )
        missing = pd.read_csv(
            root / "matrixstats_missing_values.csv", float_precision=_ROUND_TRIP
        )
        expected = pd.read_csv(
            root / "matrixstats_expected.csv", float_precision=_ROUND_TRIP
        )
        for frame, weights, column in (
            (values, [1, 2, 3, 4], "weighted_mean"),
            (values, [1, 1, 0, 1], "zero_weight_mean"),
            (missing, [1, 2, 3, 4], "missing_mean"),
        ):
            with self.subTest(reference=column):
                actual = weighted_mean(frame, weights)
                self.assertEqual(actual.index.tolist(), expected.column.tolist())
                np.testing.assert_allclose(actual, expected[column], atol=1e-12, rtol=0)
