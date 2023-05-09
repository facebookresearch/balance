# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import balance.testutil
import numpy as np

from balance.datasets import load_data


class TestDatasets(
    balance.testutil.BalanceTestCase,
):
    def test_load_data(self):
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
            "age_group": {0: "35-44", 1: "35-44", 2: "25-34", 3: "18-24", 4: "18-24"},
            "income": {0: 5.04, 1: 15.69, 2: 3.87, 3: 8.92, 4: 3.56},
            "happiness": {0: 58.82, 1: 34.92, 2: 40.76, 3: 39.2, 4: 43.91},
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

    def test_load_data_cbps(self):

        target_df, sample_df = load_data("sim_data_cbps")

        self.assertEqual(sample_df.shape, (246, 7))
        self.assertEqual(target_df.shape, (254, 7))

        self.assertEqual(
            target_df.columns.to_numpy().tolist(),
            ["X1", "X2", "X3", "X4", "cbps_weights", "y", "id"],
        )
        self.assertEqual(
            sample_df.columns.to_numpy().tolist(),
            target_df.columns.to_numpy().tolist(),
        )

        o = sample_df.head().round(2).to_dict()
        e = {
            "X1": {0: 1.07, 2: 0.69, 4: 0.5, 5: 1.52, 6: 1.03},
            "X2": {0: 10.32, 2: 10.65, 4: 9.59, 5: 10.03, 6: 9.79},
            "X3": {0: 0.21, 2: 0.22, 4: 0.23, 5: 0.33, 6: 0.22},
            "X4": {0: 463.28, 2: 424.29, 4: 472.85, 5: 438.38, 6: 436.39},
            "cbps_weights": {0: 0.01, 2: 0.01, 4: 0.01, 5: 0.0, 6: 0.0},
            "y": {0: 227.53, 2: 196.89, 4: 191.3, 5: 280.45, 6: 227.07},
            "id": {0: 1, 2: 3, 4: 5, 5: 6, 6: 7},
        }
        self.assertEqual(o.__str__(), e.__str__())
        # NOTE: using .__str__() since doing o==e will give False

        o = target_df.head().round(2).to_dict()
        e = {
            "X1": {1: 0.72, 3: 0.35, 11: 0.69, 12: 0.78, 13: 0.82},
            "X2": {1: 9.91, 3: 9.91, 11: 10.73, 12: 9.56, 13: 9.8},
            "X3": {1: 0.19, 3: 0.1, 11: 0.21, 12: 0.18, 13: 0.21},
            "X4": {1: 383.76, 3: 399.37, 11: 398.31, 12: 370.18, 13: 434.45},
            "cbps_weights": {1: 0.0, 3: 0.0, 11: 0.0, 12: 0.0, 13: 0.0},
            "y": {1: 199.82, 3: 174.69, 11: 189.58, 12: 208.18, 13: 214.28},
            "id": {1: 2, 3: 4, 11: 12, 12: 13, 13: 14},
        }
        self.assertEqual(o.__str__(), e.__str__())
