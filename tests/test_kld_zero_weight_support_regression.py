# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.stats_and_plots.weighted_comparisons_stats import kld


class KLDZeroWeightSupportRegressionTest(unittest.TestCase):
    def test_zero_weight_extreme_does_not_change_distance(self) -> None:
        sample = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        target = pd.DataFrame({"x": [3.0, 4.0, 5.0]})
        expected = kld(sample, target)["x"]
        extended = pd.DataFrame({"x": [2.0, 3.0, 4.0, -1000.0]})
        self.assertAlmostEqual(kld(extended, target, [1, 1, 1, 0])["x"], expected)
