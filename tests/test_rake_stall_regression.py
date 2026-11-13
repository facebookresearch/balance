# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.weighting_methods.rake import rake


class StalledRakeFitTest(unittest.TestCase):
    def test_unattainable_margins_warn_and_do_not_converge(self) -> None:
        sample = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        target = pd.DataFrame({"a": ["x"] * 5 + ["y"] * 5, "b": ["p"] * 8 + ["q"] * 2})
        with self.assertLogs("balance", level="WARNING"):
            result = rake(
                sample,
                pd.Series([1.0, 1.0]),
                target,
                pd.Series([1.0] * 10),
                transformations=None,
                convergence_rate=0.0005,
                max_iteration=20,
                rate_tolerance=1e-8,
            )
        self.assertEqual(result["model"]["converged"], 0)
        self.assertAlmostEqual(result["model"]["iterations"]["conv"].iloc[-1], 0.6)
        self.assertLess(len(result["model"]["iterations"]), 20)
        weights = result["weight"].to_numpy()
        self.assertAlmostEqual(weights[0] / weights.sum(), 0.8)
