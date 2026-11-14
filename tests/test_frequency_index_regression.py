# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.stats_and_plots.weighted_stats import relative_frequency_table


class FrequencyIndexRegressionTest(unittest.TestCase):
    def test_unit_weights_follow_filtered_index(self) -> None:
        values = pd.Series(["a", "b"], name="group", index=[10, 20])
        self.assertEqual(relative_frequency_table(values)["prop"].tolist(), [0.5, 0.5])
