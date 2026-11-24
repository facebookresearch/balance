# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from balance.cli import BalanceCLI, make_parser, PANDAS_DEFAULT_NA_VALUES
from pandas._libs.parsers import STR_NA_VALUES


def _cli(input_file: Path, *extra_args: str) -> BalanceCLI:
    args = make_parser().parse_args(
        [
            "--input_file",
            str(input_file),
            "--output_file",
            str(input_file.with_name("weights.csv")),
            "--diagnostics_output_file",
            str(input_file.with_name("diagnostics.csv")),
            "--covariate_columns",
            "x",
            *extra_args,
        ]
    )
    return BalanceCLI(args)


def _values(series: pd.Series) -> list[object]:
    return series.where(series.notna(), None).tolist()


def _write_country_batches(input_file: Path, countries: list[str | None]) -> None:
    pd.DataFrame(
        {
            "id": range(4 * len(countries)),
            "x": [0.0, 1.0] * 2 * len(countries),
            "weight": [1.0] * 4 * len(countries),
            "is_respondent": [1, 1, 0, 0] * len(countries),
            "country": [country for country in countries for _ in range(4)],
        }
    ).to_csv(input_file, index=False)


class CLINaValuesTest(unittest.TestCase):
    def test_default_na_values_match_pandas(self) -> None:
        self.assertEqual(PANDAS_DEFAULT_NA_VALUES, frozenset(STR_NA_VALUES))

    def test_na_values_apply_only_to_selected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("id,x,country\n1,1.5,US\n2,NA,NA\n3,,ZA\n4,null,\n")
            cli = _cli(
                path, "--na_values", "NULL,null", "--na_values_columns", "country"
            )
            loaded = cli.read_input_file()

        self.assertEqual(_values(loaded["country"]), ["US", "NA", "ZA", None])
        self.assertEqual(loaded["x"].dtype, "float64")
        self.assertEqual(loaded["x"].isna().tolist(), [False, True, True, True])

    def test_na_values_apply_to_every_column_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("id,x,answer\n1,1.5,None\n2,NA,NA\n3,,Yes\n")
            loaded = _cli(path, "--na_values", "NA").read_input_file()

        self.assertEqual(_values(loaded["answer"]), ["None", None, "Yes"])
        self.assertEqual(loaded["x"].dtype, "float64")
        self.assertEqual(loaded["x"].isna().tolist(), [False, True, True])

    def test_na_values_columns_must_be_input_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("id,x,country\n1,1.5,NA\n")
            cli = _cli(path, "--na_values", "NULL", "--na_values_columns", "countryy")
            with self.assertRaisesRegex(
                ValueError, r"not in the input file: \['countryy'\]"
            ):
                cli.read_input_file()

    def test_na_values_columns_requires_na_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("id,x,country\n1,1.5,NA\n")
            cli = _cli(path, "--na_values_columns", "country")
            with self.assertRaisesRegex(
                ValueError, "--na_values_columns requires --na_values"
            ):
                cli.read_input_file()

    def test_na_batch_key_is_missing_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            _write_country_batches(path, ["US", "NA", "ZA"])
            cli = _cli(path, "--method", "null", "--batch_columns", "country")
            cli.update_attributes_for_main_used_by_adjust()
            with self.assertRaisesRegex(
                ValueError, r"batch columns contain missing values: \['country'\]"
            ):
                cli.main()

    def test_na_batch_key_is_kept_with_na_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            _write_country_batches(path, ["US", "NA", "ZA"])
            cli = _cli(
                path,
                "--method",
                "null",
                "--batch_columns",
                "country",
                "--na_values",
                "NULL,null",
                "--na_values_columns",
                "country",
            )
            cli.update_attributes_for_main_used_by_adjust()
            cli.main()
            weights = pd.read_csv(path.with_name("weights.csv"))
            diagnostics = pd.read_csv(
                path.with_name("diagnostics.csv"), keep_default_na=False
            )

        self.assertCountEqual(weights["id"].tolist(), [0, 1, 4, 5, 8, 9])
        self.assertIn("NA", set(diagnostics["country"]))

    def test_empty_batch_key_still_raises_with_na_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            _write_country_batches(path, ["US", None])
            cli = _cli(
                path,
                "--method",
                "null",
                "--batch_columns",
                "country",
                "--na_values",
                "NULL,null",
                "--na_values_columns",
                "country",
            )
            cli.update_attributes_for_main_used_by_adjust()
            with self.assertRaisesRegex(
                ValueError, r"batch columns contain missing values: \['country'\]"
            ):
                cli.main()
