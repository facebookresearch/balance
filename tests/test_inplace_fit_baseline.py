# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample


class InplaceFitBaselineTest(unittest.TestCase):
    def test_filtering_after_inplace_fit_keeps_the_original_weights(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [0.0, 1.0], "weight": [1.0, 1.0]})
        )
        target = Sample.from_frame(
            pd.DataFrame({"id": ["c", "d"], "x": [0.0, 1.0], "weight": [3.0, 1.0]})
        )

        sample = sample.set_target(target)
        sample.fit(method="poststratify", transformations=None, inplace=True)
        filtered = sample.keep_only_some_rows_columns(columns_to_keep=["x"])

        self.assertEqual(filtered.df_responders["weight"].tolist(), [3.0, 1.0])
        self.assertEqual(
            filtered.df_responders_unadjusted["weight"].tolist(), [1.0, 1.0]
        )
