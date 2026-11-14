# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_stats import asmd, asmd_improvement


class ConstantReferenceASMDRegressionTest(unittest.TestCase):
    def test_undefined_infinite_improvement_does_not_warn(self) -> None:
        sample = pd.DataFrame({"x": [1.0, 1.0]})
        target = pd.DataFrame({"x": [2.0, 2.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertTrue(np.isnan(asmd_improvement(sample, sample, target)))

    def test_nonzero_difference_is_not_perfect_balance(self) -> None:
        result = asmd(pd.DataFrame({"x": [1.0, 1.0]}), pd.DataFrame({"x": [2.0, 2.0]}))
        self.assertTrue(np.isinf(result["x"]))
        self.assertTrue(np.isinf(result["mean(asmd)"]))

    def test_equal_constants_have_zero_deviation(self) -> None:
        frame = pd.DataFrame({"x": [2.0, 2.0]})
        self.assertEqual(asmd(frame, frame)["x"], 0.0)
