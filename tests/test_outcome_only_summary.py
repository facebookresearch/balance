# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample
from sklearn.linear_model import LinearRegression


class OutcomeOnlySummaryTest(unittest.TestCase):
    def test_outcome_fit_without_weight_adjustment_still_has_a_summary(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [0.0, 1.0], "y": [2.0, 4.0]}),
            outcome_columns="y",
        )
        target = Sample.from_frame(pd.DataFrame({"id": ["c", "d"], "x": [0.0, 1.0]}))
        sample = sample.set_target(target)
        sample.fit_outcome_model(model=LinearRegression())

        summary = sample.summary()

        self.assertIn("g-computation", summary)
        self.assertIn("AIPW", summary)
        self.assertIn("unavailable", summary)
        self.assertIn("adjust()-calibrated", summary)
