# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance.weighting_methods.poststratify import poststratify


class ObservedCategoricalPoststratifyTest(unittest.TestCase):
    def test_categorical_sample_cell_missing_from_target_raises(self) -> None:
        levels = pd.CategoricalDtype(["a", "b"])
        sample = pd.DataFrame({"g": pd.Series(["a", "b"], dtype=levels)})
        target = pd.DataFrame({"g": pd.Series(["a", "a"], dtype=levels)})
        with self.assertRaisesRegex(ValueError, "all combinations of cells"):
            poststratify(
                sample,
                pd.Series([1.0, 1.0]),
                target,
                pd.Series([1.0, 1.0]),
                transformations=None,
                na_action="drop",
                strict_matching=True,
            )
