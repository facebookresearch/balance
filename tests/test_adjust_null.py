# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import balance.testutil
import pandas as pd

from balance.sample_class import Sample
from balance.weighting_methods import adjust_null as balance_adjust_null


sample: Sample = Sample.from_frame(
    df=pd.DataFrame(
        {
            "a": (1, 2, 3, 1),
            "b": (-42, 8, 2, -42),
            "o": (7, 8, 9, 10),
            "c": ("x", "y", "z", "x"),
            "id": (1, 2, 3, 4),
            "w": (0.5, 2, 1, 1),
        }
    ),
    id_column="id",
    weight_column="w",
    outcome_columns="o",
)

target: Sample = Sample.from_frame(
    pd.DataFrame(
        {
            "a": (1, 2, 3),
            "b": (-42, 8, 2),
            "c": ("x", "y", "z"),
            "id": (1, 2, 3),
            "w": (2, 0.5, 1),
        }
    ),
    id_column="id",
    weight_column="w",
)


class TestAdjustNull(
    balance.testutil.BalanceTestCase,
):
    def test_check_adjust_null_generates_null_model_dict(self) -> None:
        res = balance_adjust_null.adjust_null(
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.Series([4, 5, 6]),
            pd.DataFrame({"a": [7, 8, 9]}),
            pd.Series([10, 11, 12]),
        )
        self.assertEqual(res["weight"], pd.Series([4, 5, 6]))
        self.assertEqual(res["model"]["method"], "null_adjustment")

        result = sample.adjust(target, method="null")
        self.assertEqual(sample.weights().df, result.weights().df)
