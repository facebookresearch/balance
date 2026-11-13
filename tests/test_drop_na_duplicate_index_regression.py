# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.utils.data_transformation import drop_na_rows


class DropNADuplicateIndexRegressionTest(unittest.TestCase):
    def test_filters_values_and_weights_positionally(self) -> None:
        frame = pd.DataFrame({"x": [1.0, None, 2.0]}, index=pd.Index([0, 0, 1]))
        weights = pd.Series([10.0, 20.0, 30.0], index=frame.index)
        values, kept_weights = drop_na_rows(frame, weights)
        self.assertEqual(values["x"].tolist(), [1.0, 2.0])
        self.assertEqual(kept_weights.tolist(), [10.0, 30.0])
        self.assertTrue(values.index.equals(kept_weights.index))
