# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance import BalanceFrame, SampleFrame


class NumericMissingPoststratifyTest(unittest.TestCase):
    def test_default_numeric_bins_with_missing_have_explicit_weights(self) -> None:
        # 21 levels make each filled-object level rarer than the 5% lump threshold.
        ages = [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            np.nan,
        ]
        sample = SampleFrame.from_frame(pd.DataFrame({"id": range(21), "age": ages}))
        target = SampleFrame.from_frame(
            pd.DataFrame(
                {"id": range(21, 42), "age": ages, "weight": [3.0] * 10 + [1.0] * 11}
            )
        )
        fitted = BalanceFrame(sample=sample, target=target).fit(
            method="poststratify",
            transformations="default",
            na_action="add_indicator",
        )
        expected = [3.0] * 10 + [1.0] * 11
        np.testing.assert_allclose(fitted.weight_series, expected)
        np.testing.assert_allclose(fitted.predict_weights(), expected)
