# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import balance.testutil
import numpy as np


class TestDatasets(
    balance.testutil.BalanceTestCase,
):
    def test_load_data(self):
        from balance.datasets import load_data

        target_df, sample_df = load_data()

        self.assertEqual(sample_df.shape, (1000, 5))
        self.assertEqual(target_df.shape, (10000, 4))

        self.assertEqual(
            target_df.columns.to_numpy().tolist(),
            ["id", "gender", "age_group", "income"],
        )
        self.assertEqual(
            sample_df.columns.to_numpy().tolist(),
            ["id", "gender", "age_group", "income", "happiness"],
        )

        o = sample_df.head().round(2).to_dict()
        e = {
            "id": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
            "gender": {0: "Female", 1: "Male", 2: "Male", 3: np.nan, 4: np.nan},
            "age_group": {0: "25-34", 1: "45+", 2: "35-44", 3: "18-24", 4: "35-44"},
            "income": {0: 1.04, 1: 0.21, 2: 2.32, 3: 0.09, 4: 17.16},
            "happiness": {0: 55.98, 1: 58.65, 2: 42.29, 3: 49.21, 4: 49.33},
        }
        self.assertEqual(o.__str__(), e.__str__())
        # NOTE: using .__str__() since doing o==e will give False

        o = target_df.head().round(2).to_dict()
        e = {
            "id": {0: "100000", 1: "100001", 2: "100002", 3: "100003", 4: "100004"},
            "gender": {0: "Male", 1: "Male", 2: "Male", 3: np.nan, 4: np.nan},
            "age_group": {0: "45+", 1: "45+", 2: "35-44", 3: "45+", 4: "25-34"},
            "income": {0: 10.18, 1: 6.04, 2: 5.23, 3: 5.75, 4: 4.84},
        }
        self.assertEqual(o.__str__(), e.__str__())
