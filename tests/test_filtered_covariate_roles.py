# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import pandas as pd
from balance import Sample


class FilteredCovariateRolesTest(unittest.TestCase):
    def test_removed_income_is_not_still_advertised_as_a_covariate(self) -> None:
        sample = Sample.from_frame(
            pd.DataFrame(
                {"id": ["a", "b"], "age": [20.0, 30.0], "income": [10.0, 90.0]}
            )
        )

        filtered = sample.keep_only_some_rows_columns(columns_to_keep=["age"])

        self.assertEqual(filtered.covar_columns, ["age"])
        self.assertEqual(filtered.df_covars.columns.tolist(), ["age"])
        self.assertEqual(sample.covar_columns, ["age", "income"])
