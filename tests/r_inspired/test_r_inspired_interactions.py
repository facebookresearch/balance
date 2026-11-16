# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""R interaction oracles on the exact recipes 0.2.0 training/scoring values.

Source: third-party/R/recipes/tests/testthat/test_interact.R:4-95.
The expected matrices were executed with stats::model.matrix, the oracle used
by recipes' own assertion. recipes: MIT + file LICENSE, copyright 2021 recipes
authors. The lexical fixture is hardhat 1.1.0 test-mold-formula.R:211-219;
MIT + file LICENSE, copyright 2019 Davis Vaughan. Notices/hashes: fixtures/interactions/.
Only names/column order are normalized; numeric fixture values are unchanged.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from balance.util import _assert_type
from balance.utils.model_matrix import model_matrix


class RInspiredInteractionMatrixTest(unittest.TestCase):
    def test_replay_uses_scoring_values_for_second_and_third_order_products(
        self,
    ) -> None:
        root = Path(__file__).parent / "fixtures" / "interactions"
        training = pd.read_csv(root / "recipes_interaction_train.csv")
        scoring = pd.read_csv(root / "recipes_interaction_test.csv")
        scoring.index = pd.Index(scoring.x1, name="scoring_x1")
        formula = "x1+x2+x3+x4+x5+x1:x2+x3:x4:x5"
        fitted = model_matrix(training, formula=formula)
        replay = model_matrix(
            scoring,
            formula=formula,
            preprocessing_state=_assert_type(fitted["preprocessing_state"], dict),
        )
        expected = pd.read_csv(
            root / "recipes_interaction_numeric_expected.csv"
        ).astype(float)
        expected.index = scoring.index
        expected = expected.sort_index(axis=1)
        pd.testing.assert_frame_equal(
            _assert_type(replay["sample"], pd.DataFrame), expected
        )
        self.assertEqual(
            replay["model_matrix_columns_names"], expected.columns.tolist()
        )
        self.assertEqual(
            fitted["model_matrix_columns_names"], expected.columns.tolist()
        )
        self.assertEqual(float(expected.iloc[0]["x1:x2"]), 462.0)
        self.assertEqual(float(expected.iloc[0]["x3:x4:x5"]), 13248.0)

    def test_categorical_prefix_does_not_capture_numeric_expression(self) -> None:
        # Exact hardhat input values; the square expression is an added lexical
        # guard, not a claim that hardhat exposes balance's formula API.
        frame = pd.DataFrame({"y": [1, 2], "x": ["a", "b"], "x2": [2, 3]})
        output = model_matrix(frame, formula="x + I(x2**2)", one_hot_encoding=True)
        expected = pd.DataFrame(
            {
                "C(x, one_hot_encoding_greater_2)[b]": [0.0, 1.0],
                "I(x2 ** 2)": [4.0, 9.0],
            }
        )
        pd.testing.assert_frame_equal(
            _assert_type(output["sample"], pd.DataFrame), expected
        )

    def test_dummy_interactions_follow_new_category_order_and_values(self) -> None:
        root = Path(__file__).parent / "fixtures" / "interactions"
        training = pd.read_csv(root / "recipes_interaction_train.csv")
        raw_scoring = pd.read_csv(root / "recipes_interaction_test.csv")
        expected = pd.read_csv(root / "recipes_interaction_dummy_expected.csv").astype(
            float
        )
        training = training[["x1", "x2", "x3", "x4", "x5"]].assign(
            zb=(training.z == "b").astype(float), zc=(training.z == "c").astype(float)
        )
        # R-produced treatment dummies avoid comparing incompatible contrasts:
        # recipes drops the reference level; balance's >2-level coding does not.
        scoring = raw_scoring[["x1", "x2", "x3", "x4", "x5"]].assign(
            zb=(raw_scoring.z == "b").astype(float),
            zc=(raw_scoring.z == "c").astype(float),
        )
        formula = "x1+x2+x3+x4+x5+zb+zc+x1:zb+x1:zc"
        fitted = model_matrix(training, formula=formula)
        result = model_matrix(
            scoring,
            formula=formula,
            preprocessing_state=_assert_type(fitted["preprocessing_state"], dict),
        )
        expected = expected.sort_index(axis=1)
        pd.testing.assert_frame_equal(
            _assert_type(result["sample"], pd.DataFrame), expected
        )
        self.assertEqual(
            result["model_matrix_columns_names"], expected.columns.tolist()
        )
