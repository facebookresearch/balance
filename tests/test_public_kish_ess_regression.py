# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance import Sample


class PublicKishESSRegressionTest(unittest.TestCase):
    def test_missing_weights_do_not_inflate_public_ess(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "w": [1.0] * 3}),
            id_column="id",
            weight_column="w",
            standardize_types=False,
        )
        sample.set_weights(pd.Series([1.0, 1.0, np.nan], index=sample.df.index))
        summary = sample.weights().summary().set_index("var")["val"]
        self.assertEqual(summary["effective_sample_size"], 2.0)
        self.assertEqual(sample._design_effect_diagnostics(), (1.0, 2.0, 1.0))
        self.assertEqual(sample._design_effect_diagnostics(n_rows=3), (1.0, 2.0, 1.0))

    def test_complete_weights_keep_existing_diagnostics(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "w": [1.0] * 3}),
            id_column="id",
            weight_column="w",
            standardize_types=False,
        )
        summary = sample.weights().summary().set_index("var")["val"]
        self.assertEqual(summary["effective_sample_size"], 3.0)
        self.assertEqual(sample._design_effect_diagnostics(), (1.0, 3.0, 1.0))
        self.assertEqual(sample._design_effect_diagnostics(n_rows=2), (1.0, 2.0, 1.0))
