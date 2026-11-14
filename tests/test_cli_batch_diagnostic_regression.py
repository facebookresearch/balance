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


class CLIBatchDiagnosticRegressionTest(unittest.TestCase):
    def test_diagnostics_retain_batch_provenance(self) -> None:
        for batch_column in ("batch", "metric"):
            with self.subTest(
                batch_column=batch_column
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                frame = pd.DataFrame(
                    {
                        "id": range(8),
                        "x": [0.0, 1.0] * 4,
                        "weight": [1.0] * 8,
                        "is_respondent": [1, 1, 0, 0] * 2,
                        batch_column: ["a"] * 4 + ["b"] * 4,
                    }
                )
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
                        batch_column,
                    ]
                )
                cli = BalanceCLI(args)
                cli.update_attributes_for_main_used_by_adjust()
                cli.main()
                diagnostics = pd.read_csv(path / "diagnostics.csv")
                provenance = "batch_metric" if batch_column == "metric" else "batch"
                self.assertIn(provenance, diagnostics.columns)
                self.assertTrue({"metric", "val", "var"}.issubset(diagnostics.columns))
                failures = diagnostics[diagnostics["metric"] == "adjustment_failure"]
                self.assertEqual(failures[provenance].tolist(), ["a", "b"])
                self.assertTrue((path / "weights.csv").is_file())
