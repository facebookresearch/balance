# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pandas as pd
from balance import Sample
from balance.interop import diff_diff as balance_diff_diff
from balance.interop.diff_diff import to_survey_design

try:
    import diff_diff

    HAS_DIFF_DIFF = True
except ImportError:
    HAS_DIFF_DIFF = False


class ReplicateDesignValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = Sample.from_frame(
            pd.DataFrame({"id": ["a", "b"], "w": [1.0, 1.0], "r1": [0.0, 2.0]}),
            weight_column="w",
        )
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(patch("balance.interop.diff_diff._IMPORT_ERROR", None))
        patches.enter_context(
            patch("balance.interop.diff_diff._dd", ModuleType("diff_diff"))
        )

    def test_replicate_weights_rejects_non_column_name_lists(self) -> None:
        for columns in ("r1", 3, ["r1", 3]):
            with self.subTest(columns=columns):
                with self.assertRaisesRegex(
                    ValueError, "replicate_weights must be a list"
                ):
                    to_survey_design(
                        self.sample, design_columns={"replicate_weights": columns}
                    )

    def test_replicate_weights_rejects_an_absent_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Replicate weight columns.*missing"):
            to_survey_design(
                self.sample, design_columns={"replicate_weights": ["r1", "missing"]}
            )


class ReplicateDesignPatchCleanupTest(unittest.TestCase):
    def test_optional_dependency_patches_are_restored_after_setup_failure(self) -> None:
        import_error = balance_diff_diff._IMPORT_ERROR
        dependency = balance_diff_diff._dd

        class SetupFailure(ReplicateDesignValidationTest):
            def setUp(self) -> None:
                super().setUp()
                raise RuntimeError("later setup failed")

        case = SetupFailure("test_replicate_weights_rejects_an_absent_column")
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("later setup failed", result.errors[0][1])
        self.assertIs(balance_diff_diff._IMPORT_ERROR, import_error)
        self.assertIs(balance_diff_diff._dd, dependency)


@unittest.skipUnless(HAS_DIFF_DIFF, "requires diff-diff")
class ReplicateDesignRegressionTest(unittest.TestCase):
    def test_accepts_upstream_list_and_boolean_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "w": [1.0] * 4,
                "r1": [0.0, 0.0, 2.0, 2.0],
                "r2": [2.0, 2.0, 0.0, 0.0],
            }
        )
        sample = Sample.from_frame(
            frame, id_column="id", weight_column="w", standardize_types=False
        )
        for method in ("JK1", "JKn"):
            columns: dict[str, Any] = {
                "replicate_weights": ["r1", "r2"],
                "replicate_method": method,
                "combined_weights": False,
            }
            if method == "JKn":
                columns["replicate_strata"] = [1, 1]
            with self.subTest(method=method):
                design = to_survey_design(sample, design_columns=columns)
                self.assertEqual(design.weight_type, "pweight")
                self.assertFalse(design.combined_weights)
                self.assertEqual(design.resolve(sample.df).n_replicates, 2)
                direct = diff_diff.SurveyDesign(weights="w", **columns)
                self.assertEqual(direct.resolve(sample.df).n_replicates, 2)
