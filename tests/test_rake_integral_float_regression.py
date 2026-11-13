# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance import BalanceFrame, SampleFrame
from balance.weighting_methods.rake import rake


class IntegralFloatRakeLevelsTest(unittest.TestCase):
    def test_integer_sample_matches_float_target_explicit_weights(self) -> None:
        sample = pd.DataFrame({"k": [1, 2, 1, 2], "g": ["a", "a", "b", "b"]})
        target = pd.DataFrame({"k": [1.0, 1.0, 1.0, 2.0], "g": ["a", "b", "b", "b"]})
        result = rake(
            sample,
            pd.Series([1.0] * 4),
            target,
            pd.Series([1.0] * 4),
            transformations=None,
            na_action="drop",
        )
        np.testing.assert_allclose(result["weight"], [0.75, 0.25, 2.25, 0.75])

    def test_integer_fit_scores_equivalent_float_rows_publicly(self) -> None:
        sample = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": [1, 2, 3, 4], "k": [1, 2, 1, 2], "g": ["a", "a", "b", "b"]}
            ),
            standardize_types=False,
        )
        target = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": [5, 6, 7, 8], "k": [1, 1, 1, 2], "g": ["a", "b", "b", "b"]}
            ),
            standardize_types=False,
        )
        fitted = BalanceFrame(sample=sample, target=target).fit(
            method="rake",
            transformations=None,
            na_action="drop",
        )
        self.assertEqual(str(sample.df_covars["k"].dtype), "int64")
        scoring = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "k": [1.0, 2.0, 1.0, 2.0],
                    "g": ["a", "a", "b", "b"],
                }
            ),
            standardize_types=False,
        )
        predicted = fitted.predict_weights(
            data=BalanceFrame(sample=scoring, target=target)
        )
        np.testing.assert_allclose(predicted, [0.75, 0.25, 2.25, 0.75])
