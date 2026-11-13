# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from balance.balance_frame import BalanceFrame
from balance.sample_frame import SampleFrame
from balance.util import _assert_type


class RakeDuplicateIndexReplayTest(unittest.TestCase):
    def test_fitting_and_replay_keep_the_dropped_duplicate_index_row_unweighted(
        self,
    ) -> None:
        sample = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["s1", "s2", "s3"],
                    "a": ["A", None, "B"],
                    "b": ["X", "X", "Y"],
                    "weight": [1.0, 9.0, 1.0],
                },
                index=pd.Index([0, 0, 1], name="respondent"),
            ),
            standardize_types=False,
        )
        target = SampleFrame.from_frame(
            pd.DataFrame(
                {
                    "id": ["t1", "t2"],
                    "a": ["A", "B"],
                    "b": ["X", "Y"],
                    "weight": [1.0, 1.0],
                }
            ),
            standardize_types=False,
        )
        fitted = BalanceFrame(sample=sample, target=target).fit(
            method="rake",
            transformations=None,
            na_action="drop",
            store_fit_metadata=True,
            inplace=False,
        )

        np.testing.assert_equal(
            _assert_type(fitted.weight_series).to_numpy(), [1.0, np.nan, 1.0]
        )
        predicted = fitted.predict_weights()

        np.testing.assert_equal(predicted.to_numpy(), [1.0, np.nan, 1.0])
        pd.testing.assert_index_equal(
            predicted.index, pd.Index([0, 0, 1], name="respondent")
        )
