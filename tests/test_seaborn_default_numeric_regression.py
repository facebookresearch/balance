# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from balance.stats_and_plots.weighted_comparisons_plots import plot_dist
from balance.util import _assert_type
from matplotlib.axes import Axes


class SeabornDefaultNumericRegressionTest(unittest.TestCase):
    def test_default_single_numeric_frame_uses_histogram(self) -> None:
        try:
            result = plot_dist(
                [{"df": pd.DataFrame({"x": np.arange(20.0)}), "weight": None}],
                library="seaborn",
                return_axes=True,
            )
            axes = _assert_type(result, list)
            self.assertEqual(len(axes), 1)
            self.assertGreater(len(_assert_type(axes[0], Axes).patches), 0)
        finally:
            plt.close("all")
