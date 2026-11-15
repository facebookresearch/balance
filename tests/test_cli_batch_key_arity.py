# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from balance.cli import BalanceCLI, make_parser


def _input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(16),
            "x": [0.0, 1.0] * 8,
            "weight": [1.0] * 16,
            "is_respondent": [1, 1, 0, 0] * 4,
            "batch": ["a"] * 8 + ["b"] * 8,
            "region": (["north"] * 4 + ["south"] * 4) * 2,
        }
    )


def _cli(input_path: Path, output_path: Path, diagnostics_path: Path) -> BalanceCLI:
    args = make_parser().parse_args(
        [
            "--input_file",
            str(input_path),
            "--output_file",
            str(output_path),
            "--diagnostics_output_file",
            str(diagnostics_path),
            "--covariate_columns",
            "x",
            "--method",
            "null",
            "--batch_columns",
            "batch,region",
        ]
    )
    cli = BalanceCLI(args)
    cli.update_attributes_for_main_used_by_adjust()
    return cli


class CLIBatchKeyArityTest(unittest.TestCase):
    def test_public_two_column_cli_preserves_complete_batch_keys(self) -> None:
        frame = _input()
        keys = list(frame.groupby(["batch", "region"]).groups)
        self.assertTrue(all(isinstance(key, tuple) and len(key) == 2 for key in keys))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame.to_csv(path / "input.csv", index=False)
            cli = _cli(
                path / "input.csv", path / "weights.csv", path / "diagnostics.csv"
            )
            cli.main()
            diagnostics = pd.read_csv(path / "diagnostics.csv")
        actual = diagnostics.loc[
            diagnostics["metric"] == "adjustment_failure", ["batch", "region"]
        ]
        self.assertEqual(
            {("a", "north"), ("a", "south"), ("b", "north"), ("b", "south")},
            set(actual.itertuples(index=False, name=None)),
        )

    def test_malformed_external_group_key_fails_before_output(self) -> None:
        frame = _input()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cli = _cli(
                path / "unused.csv",
                path / "unused_weights.csv",
                path / "unused_diagnostics.csv",
            )
            # Inject a malformed group key to exercise the defensive arity check;
            # leave batch processing and the validation under test unchanged.
            with (
                patch.object(cli, "load_and_check_input", return_value=frame),
                patch.object(frame, "groupby", return_value=[("a", frame)]),
                patch.object(cli, "write_outputs") as output,
            ):
                with self.assertRaisesRegex(
                    ValueError, "Batch key arity.*expected 2.*got 1"
                ):
                    cli.main()
            output.assert_not_called()
            self.assertFalse((path / "unused_weights.csv").exists())
            self.assertFalse((path / "unused_diagnostics.csv").exists())
