# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_stats import cvmd


class CVMDProbabilityMeasureRegressionTest(unittest.TestCase):
    def test_missing_values_cannot_supply_positive_numeric_mass(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
        for side in ("sample", "target"):
            with self.subTest(side=side), self.assertRaises(ValueError):
                cvmd(
                    frame,
                    frame,
                    sample_weights=[0.0, 0.0, 10.0] if side == "sample" else None,
                    target_weights=[0.0, 0.0, 10.0] if side == "target" else None,
                )

    def test_rejects_invalid_weight_mass_on_either_side(self) -> None:
        frames = (
            pd.DataFrame({"x": [2.0, 3.0, 4.0]}),
            pd.DataFrame({"x": pd.Categorical(["a", "b", "c"])}),
        )
        invalid_weights = (
            [0.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
            [np.inf, 1.0, 1.0],
            [-1.0, 2.0, 2.0],
            [1e308, 1e308, 1e308],
        )
        for frame in frames:
            for invalid in invalid_weights:
                for side in ("sample", "target"):
                    with self.subTest(
                        dtype=str(frame["x"].dtype), weights=invalid, side=side
                    ):
                        with self.assertRaises(ValueError):
                            cvmd(
                                frame,
                                frame,
                                sample_weights=invalid if side == "sample" else None,
                                target_weights=invalid if side == "target" else None,
                            )

    def test_positive_weight_mass_keeps_zero_weight_rows_valid(self) -> None:
        for frame in (
            pd.DataFrame({"x": [2.0, 3.0, 4.0]}),
            pd.DataFrame({"x": pd.Categorical(["a", "b", "c"])}),
        ):
            result = cvmd(frame, frame, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
            self.assertEqual(result["x"], 0.0)
            self.assertEqual(result["mean(cvmd)"], 0.0)

    def test_all_missing_values_raise(self) -> None:
        for frame in (
            pd.DataFrame({"x": [np.nan, np.nan, np.nan]}),
            pd.DataFrame({"x": pd.Categorical([None, None, None])}),
        ):
            target = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
            with self.assertRaises(ValueError):
                cvmd(frame, target)
            with self.assertRaises(ValueError):
                cvmd(target, frame)

    def test_disjoint_categories_have_nonzero_distance(self) -> None:
        result = cvmd(pd.DataFrame({"x": ["a", "a"]}), pd.DataFrame({"x": ["b", "b"]}))
        self.assertEqual(result["x"], 0.5)
        self.assertEqual(result["mean(cvmd)"], 0.5)

    def test_independent_weight_scaling_preserves_distance(self) -> None:
        sample = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        target = pd.DataFrame({"x": [3.0, 4.0, 5.0]})
        baseline = cvmd(sample, target, [1, 1, 1], [1, 1, 1])["x"]
        self.assertAlmostEqual(
            cvmd(sample, target, [10, 10, 10], [1, 1, 1])["x"], baseline
        )
        self.assertAlmostEqual(
            cvmd(sample, target, [1, 1, 1], [10, 10, 10])["x"], baseline
        )


class NumericCVMDOracleRegressionTest(unittest.TestCase):
    def test_numeric_cvmd_uses_equal_mixture_not_raw_weight_totals(self) -> None:
        sample = pd.DataFrame({"x": [2.0, 2.0]})
        target = pd.DataFrame({"x": [2.0, 4.0]})
        result = cvmd(sample, target, [10.0, 10.0], [1.0, 1.0])
        # CDF gaps at [2,4] are [1/2,0]; equal-mixture masses are [3/4,1/4].
        # Thus distance=(1/2)^2*(3/4)=3/16, independent of the two totals.
        self.assertEqual(result["x"], 3 / 16)
        self.assertEqual(result["mean(cvmd)"], 3 / 16)
