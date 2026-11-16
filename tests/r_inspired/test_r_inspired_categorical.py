# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact upstream categorical fixtures, with matching definitions kept explicit.

forcats 0.5.1 test-lump.R:65-73 supplies the NA values and asserted levels;
MIT + file LICENSE, YEAR:2020, COPYRIGHT HOLDER:forcats authors.
hardhat 1.1.0 test-forge-formula.R:312-346 supplies factor data/unused levels;
MIT + file LICENSE, YEAR:2019, COPYRIGHT HOLDER:Davis Vaughan.
stats::model.matrix with fitted xlev produced the frozen numerical matrix.
Source paths/hashes and notices are under fixtures/categorical/. Python harness is new.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from balance.util import _assert_type
from balance.utils.data_transformation import fct_lump
from balance.utils.model_matrix import model_matrix


class RInspiredCategoricalContractTest(unittest.TestCase):
    def test_lumping_counts_missing_rows_in_the_denominator(self) -> None:
        # Exact forcats fixture; normalize only its different Other label.
        values = pd.Series(["a", "a", "b", "c"] + [None] * 7, dtype=object)
        result = fct_lump(values.copy(), prop=0.10)
        self.assertEqual(
            sorted(result.dropna().replace("_lumped_other", "Other").unique()),
            ["Other", "a"],
        )
        self.assertEqual(
            result.iloc[:4].tolist(), ["a", "a", "_lumped_other", "_lumped_other"]
        )
        self.assertEqual(int(result.isna().sum()), 7)

    def test_lumping_keeps_categories_exactly_at_the_threshold(self) -> None:
        # Exact forcats 'do not change label' fixture at prop=.1. The just-above
        # case is a balance contract guard, not a matching forcats result.
        values = pd.Series(["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"])
        self.assertEqual(fct_lump(values.copy(), prop=0.1).tolist(), values.tolist())
        result = fct_lump(values.copy(), prop=0.100001)
        self.assertEqual(
            result.tolist(), ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["_lumped_other"]
        )

    def test_unused_novel_metadata_does_not_expand_frozen_matrix(self) -> None:
        root = Path(__file__).parent / "fixtures" / "categorical"
        training = pd.read_csv(root / "hardhat_factor_train.csv")
        scoring = pd.read_csv(root / "hardhat_factor_test.csv")
        training["f"] = pd.Categorical(training.f, categories=list("abcd"))
        scoring["f"] = pd.Categorical(scoring.f, categories=list("abcde"))
        scoring.index = pd.Index([40, 10, 30, 20], name="respondent")
        fitted = model_matrix(training, formula="f", one_hot_encoding=True)
        replay = model_matrix(
            scoring,
            formula="f",
            one_hot_encoding=True,
            preprocessing_state=_assert_type(fitted["preprocessing_state"], dict),
        )
        expected = pd.read_csv(root / "hardhat_factor_expected.csv")
        expected.columns = [
            f"C(f, one_hot_encoding_greater_2)[{name[1:]}]" for name in expected.columns
        ]
        expected.index = scoring.index
        self.assertEqual(
            replay["model_matrix_columns_names"], expected.columns.tolist()
        )
        self.assertEqual(
            fitted["model_matrix_columns_names"], expected.columns.tolist()
        )
        pd.testing.assert_frame_equal(
            _assert_type(replay["sample"], pd.DataFrame), expected.astype(float)
        )
