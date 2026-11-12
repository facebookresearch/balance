# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample


class WeightRolesRoundTripTest(unittest.TestCase):
    def test_frame_round_trip_keeps_both_weight_columns_and_the_selected_one(
        self,
    ) -> None:
        sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "x": [0.0, 10.0], "weight": [1.0, 1.0]})
        )
        target = Sample.from_frame(pd.DataFrame({"id": ["c", "d"], "x": [0.0, 10.0]}))
        sample = sample.set_target(target)
        sample.add_weight_column("survey_weight", pd.Series([1.0, 9.0]))
        sample.set_active_weight("survey_weight")

        restored = sample.to_balance_frame().to_sample()

        self.assertEqual(restored.weight_columns_all, ["weight", "survey_weight"])
        self.assertEqual(restored.weight_column, "survey_weight")
        self.assertEqual(
            restored.df_weights.to_dict("list"), {"survey_weight": [1.0, 9.0]}
        )
        self.assertEqual(restored.covar_columns, ["x"])
