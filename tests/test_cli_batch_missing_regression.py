# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from balance.cli import BalanceCLI, make_parser


class CLIMissingBatchRegressionTest(unittest.TestCase):
    def test_missing_batch_key_is_explicit_error(self) -> None:
        frame = pd.DataFrame(
            {
                "id": range(8),
                "x": [0.0, 1.0] * 4,
                "weight": [1.0] * 8,
                "is_respondent": [1, 1, 0, 0] * 2,
                "batch": ["a"] * 4 + [None] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame.to_csv(path / "input.csv", index=False)
            args = make_parser().parse_args(
                [
                    "--input_file",
                    str(path / "input.csv"),
                    "--output_file",
                    str(path / "weights.csv"),
                    "--diagnostics_output_file",
                    str(path / "diagnostics.csv"),
                    "--covariate_columns",
                    "x",
                    "--method",
                    "null",
                    "--batch_columns",
                    "batch",
                ]
            )
            cli = BalanceCLI(args)
            cli.update_attributes_for_main_used_by_adjust()
            with self.assertRaisesRegex(ValueError, "batch.*missing|missing.*batch"):
                cli.main()
            self.assertFalse((path / "weights.csv").exists())
            self.assertFalse((path / "diagnostics.csv").exists())
