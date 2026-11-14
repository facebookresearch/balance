# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from balance.csv_utils import to_csv_with_defaults


class CSVDestinationRegressionTest(unittest.TestCase):
    def test_append_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            to_csv_with_defaults(pd.DataFrame({"x": [1]}), path)
            to_csv_with_defaults(pd.DataFrame({"x": [2]}), path, mode="a", header=False)
            self.assertEqual(path.read_text(), "x\n1\n2\n")

    def test_compression_inference_uses_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv.gz"
            to_csv_with_defaults(pd.DataFrame({"x": [1]}), path)
            with gzip.open(path, "rt") as stream:
                self.assertEqual(stream.read(), "x\n1\n")
