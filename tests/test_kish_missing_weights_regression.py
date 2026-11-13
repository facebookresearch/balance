# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
from balance.stats_and_plots.weights_stats import kish_deff_stats, kish_ess


class KishMissingWeightsRegressionTest(unittest.TestCase):
    def test_missing_weights_add_no_information(self) -> None:
        weights = [1.0, 1.0, np.nan]
        self.assertEqual(kish_ess(weights), 2.0)
        bundle = kish_deff_stats(weights)
        self.assertEqual(bundle.ess, 2.0)
        self.assertEqual(bundle.deff, 1.0)
        self.assertEqual(bundle.essp, 1.0)
