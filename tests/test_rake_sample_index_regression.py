# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.weighting_methods.rake import rake


class RakeSampleIndexTest(unittest.TestCase):
    def test_named_index_keeps_explicit_weights_and_row_labels(self) -> None:
        index = pd.Index([10, 11, 12, 13], name="respondent")
        sample = pd.DataFrame(
            {"a": ["x", "y", "x", "y"], "b": ["p", "p", "q", "q"]}, index=index
        )
        target = pd.DataFrame({"a": ["x", "x", "x", "y"], "b": ["p", "q", "q", "q"]})
        weights = rake(
            sample,
            pd.Series(1.0, index=index),
            target,
            pd.Series([1.0] * 4),
            transformations=None,
        )["weight"]
        pd.testing.assert_index_equal(weights.index, index)
        np.testing.assert_allclose(weights, [0.75, 0.25, 2.25, 0.75])

    def test_index_covariate_does_not_replace_row_labels(self) -> None:
        index = pd.Index([10, 11, 12, 13])
        sample = pd.DataFrame(
            {"a": ["x", "y", "x", "y"], "index": ["p", "p", "q", "q"]}, index=index
        )
        target = pd.DataFrame(
            {"a": ["x", "x", "x", "y"], "index": ["p", "q", "q", "q"]}
        )
        weights = rake(
            sample,
            pd.Series(1.0, index=index),
            target,
            pd.Series([1.0] * 4),
            transformations=None,
        )["weight"]
        pd.testing.assert_index_equal(weights.index, index)
        np.testing.assert_allclose(weights, [0.75, 0.25, 2.25, 0.75])
