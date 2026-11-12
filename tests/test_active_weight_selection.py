# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample


class ActiveWeightSelectionTest(unittest.TestCase):
    def test_selecting_survey_weights_changes_the_weights_used_by_views(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [0.0, 10.0], "weight": [1.0, 1.0]})
        )
        sample.add_weight_column("survey_weight", pd.Series([1.0, 9.0]))

        sample.set_active_weight("survey_weight")

        self.assertEqual(
            sample.df_weights.to_dict("list"), {"survey_weight": [1.0, 9.0]}
        )
        self.assertEqual(
            sample.covars().mean(on_linked_samples=False)["x"].iloc[0], 9.0
        )
