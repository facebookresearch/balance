# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.utils.data_transformation import add_na_indicator


class NAIndicatorInputRegressionTest(unittest.TestCase):
    def test_preserves_input_missingness(self) -> None:
        frame = pd.DataFrame({"x": [1.0, None]})
        original = frame.copy(deep=True)
        result = add_na_indicator(frame)
        self.assertEqual(result["_is_na_x"].tolist(), [False, True])
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(add_na_indicator(frame), result)
