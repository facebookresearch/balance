# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest

import pandas as pd
from balance import Sample
from balance.util import _assert_type


class SamplePickleConstructionTest(unittest.TestCase):
    def test_pickle_round_trip_reconstructs_without_public_constructor(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0], "weight": [1.0, 3.0]})
        )
        # @lint-ignore PYTHONPICKLEISBAD - Trusted synthetic Sample created in this test.
        restored = pickle.loads(pickle.dumps(sample))

        pd.testing.assert_frame_equal(restored.df, sample.df)
        self.assertEqual(_assert_type(restored.weight_series).tolist(), [1.0, 3.0])

    def test_direct_construction_remains_rejected(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "from_frame"):
            Sample()
