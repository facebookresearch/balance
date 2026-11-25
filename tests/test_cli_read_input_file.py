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


class CLIReadInputFileTest(unittest.TestCase):
    def test_read_input_file_matches_pandas_read_csv(self) -> None:
        csv = "id,x,country\n1,1.5,US\n2,NA,NA\n3,,ZA\n4,null,N/A\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(csv)
            args = make_parser().parse_args(
                [
                    "--input_file",
                    str(path),
                    "--output_file",
                    str(Path(directory) / "weights.csv"),
                    "--covariate_columns",
                    "x",
                ]
            )
            cli = BalanceCLI(args)
            expected = pd.read_csv(path)

            pd.testing.assert_frame_equal(cli.read_input_file(), expected)
            pd.testing.assert_frame_equal(cli.load_and_check_input(), expected)
            self.assertEqual(
                expected["country"].isna().tolist(), [False, True, False, True]
            )
